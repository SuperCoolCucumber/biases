from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
from typing import Any, Literal
from uuid import uuid4

from biases.parser_integrity import (
    ParserIntegrityError,
    migrate_record_to_current_parser,
)
from biases.position_bias import (
    JUDGE_OUTPUT_PARSER_VERSION,
    _record_to_uncertainty_row,
)
from biases.schemas import RunRecord
from biases.silent_bias_runner import _clean_summary_row, _cued_summary_row
from biases.stage_planning import (
    CleanPairSummary,
    PlannedCondition,
    generate_stage_b_conditions,
)


StageName = Literal["stage_a", "stage_b"]
RAW_FILENAMES: Mapping[StageName, str] = {
    "stage_a": "silent_bias_stage_a_run_records.jsonl",
    "stage_b": "silent_bias_stage_b_run_records.jsonl",
}
SCORE_FILENAMES: Mapping[StageName, str] = {
    "stage_a": "silent_bias_stage_a_uncertainty_scores.jsonl",
    "stage_b": "silent_bias_stage_b_uncertainty_scores.jsonl",
}
PAIR_FILENAMES: Mapping[StageName, str] = {
    "stage_a": "silent_bias_stage_a_pair_summary.jsonl",
    "stage_b": "silent_bias_stage_b_pair_summary.jsonl",
}
SUMMARY_FILENAMES: Mapping[StageName, str] = {
    "stage_a": "silent_bias_stage_a_summary.json",
    "stage_b": "silent_bias_stage_b_summary.json",
}
PLANNING_FILENAMES: Mapping[StageName, str] = {
    "stage_a": "silent_bias_stage_a_planning_issues.json",
    "stage_b": "silent_bias_stage_b_planning_issues.json",
}
PROTECTED_FIELDS = (
    "record_id",
    "pair_key",
    "condition_group_id",
    "ordering_twin_key",
    "spec_hash",
    "input_file_hash",
    "clean_record_id",
)


@dataclass(frozen=True, slots=True)
class JsonlCheckpoint:
    rows: tuple[dict[str, Any], ...]
    dropped_incomplete_tail: bool


@dataclass(frozen=True, slots=True)
class MigratedStage:
    raw_rows: tuple[dict[str, Any], ...]
    score_rows: tuple[dict[str, Any], ...]
    pair_rows: tuple[dict[str, Any], ...]
    summary: dict[str, Any]
    dropped_incomplete_tail: bool


def _read_jsonl_checkpoint(path: Path) -> JsonlCheckpoint:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = path.read_bytes()
    lines = payload.splitlines(keepends=True)
    rows: list[dict[str, Any]] = []
    dropped_tail = False
    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            is_incomplete_tail = (
                line_number == len(lines)
                and not raw_line.endswith((b"\n", b"\r"))
            )
            if is_incomplete_tail:
                dropped_tail = True
                break
            raise ValueError(
                f"invalid JSONL at {path}:{line_number}"
            ) from exc
        if not isinstance(row, dict):
            raise ValueError(f"expected a JSON object at {path}:{line_number}")
        rows.append(row)
    return JsonlCheckpoint(
        rows=tuple(rows),
        dropped_incomplete_tail=dropped_tail,
    )


def _read_json_object_optional(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON at {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object at {path}")
    return payload


def _record_sort_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    condition = row.get("condition")
    variant = (
        condition.get("variant_id") if isinstance(condition, Mapping) else None
    )
    return (
        str(row.get("pair_key") or ""),
        str(variant or ""),
        str(row.get("record_id") or ""),
    )


def _preserved_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    condition = row.get("condition")
    condition = condition if isinstance(condition, Mapping) else {}
    return {
        "record_id": row.get("record_id"),
        "pair_key": row.get("pair_key"),
        "condition_group_id": row.get("condition_group_id"),
        "ordering_twin_key": row.get("ordering_twin_key"),
        "spec_hash": row.get("spec_hash"),
        "input_file_hash": row.get("input_file_hash"),
        "clean_record_id": condition.get("clean_record_id"),
    }


def _scheduler_value(
    summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    field: str,
    stage: StageName,
) -> int | None:
    observed: set[object] = set()
    for row in rows:
        metadata = row.get("metadata")
        if not isinstance(metadata, Mapping) or field not in metadata:
            continue
        value = metadata.get(field)
        try:
            observed.add(value)
        except TypeError as exc:
            raise ValueError(f"{stage} raw {field} is not scalar") from exc
    if field in summary:
        value = summary.get(field)
        if observed and observed != {value}:
            raise ValueError(
                f"{stage} raw {field} values {sorted(map(str, observed))!r} "
                f"disagree with stage summary value {value!r}"
            )
    elif len(observed) == 1:
        value = next(iter(observed))
    elif not observed:
        value = None
    else:
        raise ValueError(
            f"{stage} has inconsistent legacy {field} values: "
            f"{sorted(map(str, observed))!r}"
        )
    if value is not None and (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 1
    ):
        raise ValueError(f"{stage} {field} must be a positive integer or null")
    return value


def _migrate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    stage: StageName,
    max_num_batched_tokens: int | None,
    max_num_seqs: int | None,
) -> tuple[dict[str, Any], ...]:
    migrated: list[dict[str, Any]] = []
    seen_record_ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        record_id = str(row.get("record_id") or "")
        if not record_id:
            raise ValueError(f"{stage} row {index} has no record_id")
        if record_id in seen_record_ids:
            raise ValueError(f"{stage} has duplicate record_id {record_id!r}")
        seen_record_ids.add(record_id)
        before = _preserved_fields(row)
        metadata = row.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError(f"{stage} record {record_id!r} has no metadata object")
        candidate = dict(row)
        candidate["metadata"] = {
            **metadata,
            "max_num_batched_tokens": max_num_batched_tokens,
            "max_num_seqs": max_num_seqs,
        }
        try:
            migrated_row = migrate_record_to_current_parser(
                candidate,
                require_stored_verdict_match=True,
            )
        except ParserIntegrityError as exc:
            raise ValueError(
                f"{stage} record {record_id!r} cannot be migrated: {exc}"
            ) from exc
        if _preserved_fields(migrated_row) != before:
            raise AssertionError(
                f"{stage} record {record_id!r} changed protected provenance "
                "or linkage fields"
            )
        migrated.append(migrated_row)
    migrated.sort(key=_record_sort_key)
    return tuple(migrated)


def _rematerialize(
    raw_rows: Sequence[Mapping[str, Any]],
    *,
    stage: StageName,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    records = tuple(RunRecord.model_validate(row) for row in raw_rows)
    score_rows = tuple(
        {
            **_record_to_uncertainty_row(record),
            "verbalized_parse_status": record.metadata.get(
                "verbalized_parse_status"
            ),
        }
        for record in records
    )
    pair_rows = tuple(
        _clean_summary_row(record)
        if stage == "stage_a"
        else _cued_summary_row(record)
        for record in records
    )
    return score_rows, pair_rows


def _uniform_value(
    rows: Sequence[Mapping[str, Any]],
    *,
    path: tuple[str, ...],
) -> Any:
    values: list[Any] = []
    for row in rows:
        current: Any = row
        for component in path:
            if not isinstance(current, Mapping):
                current = None
                break
            current = current.get(component)
        values.append(current)
    first = values[0] if values else None
    return first if all(value == first for value in values) else None


def _inferred_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    stage: StageName,
) -> dict[str, Any]:
    identities = {
        metadata.get("pair_identity_key")
        for row in rows
        if isinstance((metadata := row.get("metadata")), Mapping)
        and metadata.get("pair_identity_key") is not None
    }
    methods = [
        spec.get("uncertainty_methods")
        for row in rows
        if isinstance((spec := row.get("spec")), Mapping)
    ]
    return {
        "stage": "A" if stage == "stage_a" else "B",
        "model_name": _uniform_value(rows, path=("spec", "model_name")),
        "model_revision": _uniform_value(
            rows,
            path=("spec", "model_revision"),
        ),
        "dataset_split": _uniform_value(rows, path=("spec", "dataset_split")),
        "input_file_hash": _uniform_value(rows, path=("input_file_hash",)),
        "source_pairs": len(identities),
        "records_written": len(rows),
        "consistency_schedule": _uniform_value(
            rows,
            path=("spec", "consistency_schedule"),
        ),
        "sampling_temperature": _uniform_value(
            rows,
            path=("spec", "temperature"),
        ),
        "include_verbalized_confidence": any(
            isinstance(value, list) and "verbalized_confidence" in value
            for value in methods
        ),
        "migration_reconstructed_summary": True,
    }


def _migrated_summary(
    source: Mapping[str, Any] | None,
    *,
    stage: StageName,
    target_dir: Path,
    record_count: int,
    max_num_batched_tokens: int | None,
    max_num_seqs: int | None,
    raw_rows: Sequence[Mapping[str, Any]],
    dropped_incomplete_tail: bool,
) -> dict[str, Any]:
    migrated = (
        dict(source)
        if source is not None
        else _inferred_summary(raw_rows, stage=stage)
    )
    migrated["judge_output_parser_version"] = JUDGE_OUTPUT_PARSER_VERSION
    migrated["records_written"] = record_count
    migrated["max_num_batched_tokens"] = max_num_batched_tokens
    migrated["max_num_seqs"] = max_num_seqs
    migrated["parser_migration_dropped_incomplete_tail"] = (
        dropped_incomplete_tail
    )
    parse_status_counts: dict[str, int] = {}
    for row in raw_rows:
        metadata = row.get("metadata")
        status = (
            metadata.get("verbalized_parse_status")
            if isinstance(metadata, Mapping)
            else None
        )
        label = str(status or "missing_status")
        parse_status_counts[label] = parse_status_counts.get(label, 0) + 1
    migrated["verbalized_parse_status_counts"] = dict(
        sorted(parse_status_counts.items())
    )
    migrated["raw_records_path"] = str(target_dir / RAW_FILENAMES[stage])
    migrated["uncertainty_scores_path"] = str(
        target_dir / SCORE_FILENAMES[stage]
    )
    migrated["pair_summary_path"] = str(target_dir / PAIR_FILENAMES[stage])
    migrated["planning_issues_path"] = str(
        target_dir / PLANNING_FILENAMES[stage]
    )
    if stage == "stage_b":
        migrated["stage_a_summary_path"] = str(
            target_dir / PAIR_FILENAMES["stage_a"]
        )
    return migrated


def _expected_stage_b_conditions(
    stage_a_rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], PlannedCondition]:
    summaries = [
        CleanPairSummary.from_mapping(
            _clean_summary_row(RunRecord.model_validate(row))
        )
        for row in stage_a_rows
    ]
    plan = generate_stage_b_conditions(summaries)
    fatal = [issue for issue in plan.issues if issue.code != "clean_and_human_tie"]
    if fatal:
        codes = sorted({issue.code for issue in fatal})
        raise ValueError(
            f"Stage A cannot semantically anchor Stage B: {codes!r}"
        )
    return {
        (condition.pair_key, condition.condition.variant_id): condition
        for condition in plan.conditions
    }


def _assert_stage_b_semantics(
    stage_a_rows: Sequence[Mapping[str, Any]],
    stage_b_rows: Sequence[Mapping[str, Any]],
) -> None:
    expected = _expected_stage_b_conditions(stage_a_rows)
    seen: set[tuple[str, str]] = set()
    for row in stage_b_rows:
        condition = row.get("condition")
        metadata = row.get("metadata")
        if not isinstance(condition, Mapping) or not isinstance(metadata, Mapping):
            raise ValueError("Stage B row has no condition/metadata object")
        key = (str(row.get("pair_key")), str(condition.get("variant_id")))
        if key in seen:
            raise ValueError(f"Stage B has duplicate condition key {key!r}")
        seen.add(key)
        planned = expected.get(key)
        if planned is None:
            raise ValueError(
                f"Stage B condition {key!r} has no semantic Stage A partner"
            )
        expected_links = {
            "pair_key": planned.pair_key,
            "condition_group_id": planned.condition_group_id,
            "ordering_twin_key": planned.ordering_twin_key,
            "input_file_hash": planned.input_file_hash,
        }
        for field, expected_value in expected_links.items():
            if row.get(field) != expected_value:
                raise ValueError(
                    f"Stage B record {row.get('record_id')!r} has {field}="
                    f"{row.get(field)!r}; expected {expected_value!r}"
                )
        if metadata.get("pair_identity_key") != planned.pair_identity_key:
            raise ValueError(
                f"Stage B record {row.get('record_id')!r} links to the wrong "
                "pair identity"
            )
        spec = row.get("spec")
        if (
            not isinstance(spec, Mapping)
            or spec.get("model_name") != planned.model_name
        ):
            raise ValueError(
                f"Stage B record {row.get('record_id')!r} links to the wrong "
                "clean-model provenance"
            )
        expected_condition = planned.condition.model_dump(mode="json")
        for field, expected_value in expected_condition.items():
            if condition.get(field) != expected_value:
                raise ValueError(
                    f"Stage B record {row.get('record_id')!r} has semantic "
                    f"condition mismatch in {field}"
                )


def _build_migration(
    source_dir: Path,
    *,
    target_dir: Path,
) -> dict[StageName, MigratedStage]:
    stage_a_path = source_dir / RAW_FILENAMES["stage_a"]
    stage_a_checkpoint = _read_jsonl_checkpoint(stage_a_path)
    if not stage_a_checkpoint.rows:
        raise ValueError("Stage A checkpoint has no complete raw records")

    checkpoints: dict[StageName, JsonlCheckpoint] = {
        "stage_a": stage_a_checkpoint
    }
    stage_b_path = source_dir / RAW_FILENAMES["stage_b"]
    if stage_b_path.is_file():
        stage_b_checkpoint = _read_jsonl_checkpoint(stage_b_path)
        if stage_b_checkpoint.rows:
            checkpoints["stage_b"] = stage_b_checkpoint

    migrated: dict[StageName, MigratedStage] = {}
    for stage, checkpoint in checkpoints.items():
        source_summary = _read_json_object_optional(
            source_dir / SUMMARY_FILENAMES[stage]
        )
        summary_for_scheduler = source_summary or {}
        max_num_batched_tokens = _scheduler_value(
            summary_for_scheduler,
            checkpoint.rows,
            field="max_num_batched_tokens",
            stage=stage,
        )
        max_num_seqs = _scheduler_value(
            summary_for_scheduler,
            checkpoint.rows,
            field="max_num_seqs",
            stage=stage,
        )
        raw_rows = _migrate_rows(
            checkpoint.rows,
            stage=stage,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
        )
        score_rows, pair_rows = _rematerialize(raw_rows, stage=stage)
        summary = _migrated_summary(
            source_summary,
            stage=stage,
            target_dir=target_dir,
            record_count=len(raw_rows),
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            raw_rows=raw_rows,
            dropped_incomplete_tail=checkpoint.dropped_incomplete_tail,
        )
        migrated[stage] = MigratedStage(
            raw_rows=raw_rows,
            score_rows=score_rows,
            pair_rows=pair_rows,
            summary=summary,
            dropped_incomplete_tail=checkpoint.dropped_incomplete_tail,
        )

    if "stage_b" in migrated:
        _assert_stage_b_semantics(
            migrated["stage_a"].raw_rows,
            migrated["stage_b"].raw_rows,
        )
    return migrated


def _jsonl_payload(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(
        f"{json.dumps(row, sort_keys=True, ensure_ascii=True)}\n"
        for row in rows
    ).encode("utf-8")


def _json_payload(value: Mapping[str, Any]) -> bytes:
    return (
        f"{json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True)}\n"
    ).encode("utf-8")


def _target_payloads(
    migrated: Mapping[StageName, MigratedStage],
    *,
    source_dir: Path,
    target_dir: Path,
    in_place: bool,
) -> dict[Path, bytes]:
    payloads: dict[Path, bytes] = {}
    for stage, result in migrated.items():
        payloads[target_dir / RAW_FILENAMES[stage]] = _jsonl_payload(
            result.raw_rows
        )
        payloads[target_dir / SCORE_FILENAMES[stage]] = _jsonl_payload(
            result.score_rows
        )
        payloads[target_dir / PAIR_FILENAMES[stage]] = _jsonl_payload(
            result.pair_rows
        )
        payloads[target_dir / SUMMARY_FILENAMES[stage]] = _json_payload(
            result.summary
        )
        source_planning = source_dir / PLANNING_FILENAMES[stage]
        target_planning = target_dir / PLANNING_FILENAMES[stage]
        if not in_place or not target_planning.exists():
            payloads[target_planning] = (
                source_planning.read_bytes()
                if source_planning.is_file()
                else b"[]\n"
            )
    return payloads


def _validate_backup_suffix(suffix: str) -> None:
    if not suffix or "/" in suffix or "\\" in suffix:
        raise ValueError("backup_suffix must be a non-empty filename suffix")


def _backup_path(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.name}{suffix}")


def _replace_file(source: Path, target: Path) -> None:
    os.replace(source, target)


def _install_new_file(source: Path, target: Path) -> None:
    os.link(source, target)


def _stage_payloads(payloads: Mapping[Path, bytes]) -> dict[Path, Path]:
    staged: dict[Path, Path] = {}
    try:
        for target, payload in payloads.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(
                f".{target.name}.parser-migration-{uuid4().hex}.tmp"
            )
            with temporary.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            staged[target] = temporary
    except BaseException:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)
        raise
    return staged


def _create_backups(
    targets: Sequence[Path],
    *,
    backup_suffix: str,
) -> dict[Path, Path]:
    existing_targets = [path for path in targets if path.exists()]
    backups = {
        path: _backup_path(path, backup_suffix) for path in existing_targets
    }
    collisions = [backup for backup in backups.values() if backup.exists()]
    if collisions:
        raise FileExistsError(
            f"refusing to overwrite existing backups: {collisions!r}"
        )
    created: list[Path] = []
    try:
        for source, backup in backups.items():
            shutil.copy2(source, backup)
            created.append(backup)
    except BaseException:
        for backup in created:
            backup.unlink(missing_ok=True)
        raise
    return backups


def _restore_from_backups(
    backups: Mapping[Path, Path],
    *,
    replaced_existing: Sequence[Path],
    installed_new: Sequence[Path],
) -> list[str]:
    errors: list[str] = []
    for target in installed_new:
        try:
            target.unlink(missing_ok=True)
        except OSError as exc:
            errors.append(f"remove {target}: {exc}")
    for target in replaced_existing:
        backup = backups[target]
        restore = target.with_name(
            f".{target.name}.parser-restore-{uuid4().hex}.tmp"
        )
        try:
            shutil.copy2(backup, restore)
            _replace_file(restore, target)
        except OSError as exc:
            errors.append(f"restore {target}: {exc}")
        finally:
            restore.unlink(missing_ok=True)
    if not errors:
        for backup in backups.values():
            backup.unlink(missing_ok=True)
    return errors


def _commit_transaction(
    payloads: Mapping[Path, bytes],
    *,
    in_place: bool,
    backup_suffix: str,
) -> list[Path]:
    targets = list(payloads)
    preexisting = {path: path.exists() for path in targets}
    if not in_place:
        collisions = [path for path, exists in preexisting.items() if exists]
        if collisions:
            raise FileExistsError(
                f"refusing to overwrite destination artifacts: {collisions!r}"
            )
    staged = _stage_payloads(payloads)
    backups: dict[Path, Path] = {}
    try:
        if in_place:
            backups = _create_backups(
                targets,
                backup_suffix=backup_suffix,
            )
        installed_new: list[Path] = []
        replaced_existing: list[Path] = []
        try:
            for target in targets:
                temporary = staged[target]
                if preexisting[target]:
                    _replace_file(temporary, target)
                    replaced_existing.append(target)
                else:
                    _install_new_file(temporary, target)
                    installed_new.append(target)
        except BaseException as exc:
            rollback_errors = _restore_from_backups(
                backups,
                replaced_existing=replaced_existing,
                installed_new=installed_new,
            )
            if rollback_errors:
                raise RuntimeError(
                    "migration commit failed and rollback was incomplete; "
                    f"backups were retained: {rollback_errors!r}"
                ) from exc
            raise
        return list(backups.values())
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)


def _resolved_report_path(report_path: Path | None) -> Path | None:
    return report_path.resolve() if report_path is not None else None


def _check_report_path(
    report_path: Path | None,
    *,
    payload_paths: Sequence[Path],
    backup_paths: Sequence[Path],
) -> None:
    if report_path is None:
        return
    protected = {path.resolve() for path in (*payload_paths, *backup_paths)}
    if report_path in protected:
        raise ValueError(
            "report_path must not collide with migrated artifacts or backups"
        )
    if report_path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing report_path {report_path}"
        )


def migrate_artifact_directory(
    *,
    source_dir: Path,
    destination_dir: Path | None = None,
    in_place: bool = False,
    dry_run: bool = False,
    backup_suffix: str = ".pre-strict-v2.bak",
    report_path: Path | None = None,
) -> dict[str, Any]:
    selected_modes = sum((destination_dir is not None, in_place, dry_run))
    if selected_modes != 1:
        raise ValueError(
            "choose exactly one of destination_dir, in_place, or dry_run"
        )
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)
    _validate_backup_suffix(backup_suffix)
    resolved_source = source_dir.resolve()
    target_dir = (
        resolved_source
        if destination_dir is None
        else destination_dir.resolve()
    )
    if destination_dir is not None and target_dir == resolved_source:
        raise ValueError("destination_dir must differ from source_dir")

    migrated = _build_migration(resolved_source, target_dir=target_dir)
    artifact_payloads = _target_payloads(
        migrated,
        source_dir=resolved_source,
        target_dir=target_dir,
        in_place=in_place,
    )
    predicted_backups = (
        [
            _backup_path(path, backup_suffix)
            for path in artifact_payloads
            if path.exists()
        ]
        if in_place
        else []
    )
    resolved_report = _resolved_report_path(report_path)
    _check_report_path(
        resolved_report,
        payload_paths=list(artifact_payloads),
        backup_paths=predicted_backups,
    )

    report: dict[str, Any] = {
        "passed": True,
        "mode": (
            "dry_run"
            if dry_run
            else "in_place"
            if in_place
            else "destination"
        ),
        "source_dir": str(resolved_source),
        "target_dir": None if dry_run else str(target_dir),
        "judge_output_parser_version": JUDGE_OUTPUT_PARSER_VERSION,
        "protected_fields_preserved": list(PROTECTED_FIELDS),
        "stages_migrated": list(migrated),
        "records": {
            stage: len(result.raw_rows)
            for stage, result in migrated.items()
        },
        "dropped_incomplete_tail": {
            stage: result.dropped_incomplete_tail
            for stage, result in migrated.items()
        },
        "verbalized_parse_status_counts": {
            stage: result.summary["verbalized_parse_status_counts"]
            for stage, result in migrated.items()
        },
        "files_written": (
            []
            if dry_run
            else [str(path) for path in sorted(artifact_payloads)]
        ),
        "backup_files": (
            [str(path) for path in sorted(predicted_backups)]
            if in_place
            else []
        ),
        "report_path": str(resolved_report) if resolved_report else None,
    }

    payloads_to_commit: dict[Path, bytes] = {}
    if not dry_run:
        payloads_to_commit.update(artifact_payloads)
    if resolved_report is not None:
        payloads_to_commit[resolved_report] = _json_payload(report)
    if payloads_to_commit:
        backups = _commit_transaction(
            payloads_to_commit,
            in_place=in_place,
            backup_suffix=backup_suffix,
        )
        actual_artifact_backups = [
            path
            for path in backups
            if resolved_report is None
            or path != _backup_path(resolved_report, backup_suffix)
        ]
        if in_place and set(actual_artifact_backups) != set(predicted_backups):
            raise AssertionError("committed backup set differed from preflight")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reparse stored Silent Bias outputs with the current strict parser "
            "and rematerialize derived artifacts without GPU inference."
        )
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--destination-dir", type=Path)
    mode.add_argument("--in-place", action="store_true")
    parser.add_argument(
        "--backup-suffix",
        default=".pre-strict-v2.bak",
        help="Suffix for mandatory in-place backups.",
    )
    parser.add_argument("--report-path", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = migrate_artifact_directory(
            source_dir=args.source_dir,
            destination_dir=args.destination_dir,
            in_place=args.in_place,
            dry_run=args.dry_run,
            backup_suffix=args.backup_suffix,
            report_path=args.report_path,
        )
    except (OSError, ParserIntegrityError, RuntimeError, TypeError, ValueError) as exc:
        report = {
            "passed": False,
            "error": str(exc),
            "judge_output_parser_version": JUDGE_OUTPUT_PARSER_VERSION,
        }
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
