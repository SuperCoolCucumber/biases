from __future__ import annotations

import copy
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from biases.pairing import (
    canonical_sha256,
    make_condition_group_id,
    make_ordering_twin_key,
    make_pair_identity_key,
    make_pair_key,
    normalize_ordering,
)
from biases.schemas import PairOrdering
from biases.social_cue_prompts import parse_variant_id


LINKAGE_STATUS_KEY = "linkage_migration_status"


@dataclass(frozen=True)
class MigrationReport:
    source_path: str
    destination_path: str
    total_rows: int
    resolved_rows: int
    unresolved_rows: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _nonempty(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _infer_ordering(row: Mapping[str, Any]) -> PairOrdering | None:
    condition = row.get("condition")
    condition = condition if isinstance(condition, Mapping) else {}
    metadata = row.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}

    explicit = condition.get("ordering") or metadata.get("ordering")
    if explicit is not None:
        try:
            return normalize_ordering(str(explicit))
        except ValueError:
            return None

    variant_id = _nonempty(condition.get("variant_id") or metadata.get("variant_id"))
    if variant_id is not None:
        try:
            return parse_variant_id(variant_id).ordering
        except ValueError:
            if variant_id in {"original", "position_control"}:
                return PairOrdering.AB
            if variant_id in {"swapped", "position_swapped"}:
                return PairOrdering.BA

    example_id = _nonempty(row.get("example_id"))
    if example_id is not None:
        if example_id.endswith(":original"):
            return PairOrdering.AB
        if example_id.endswith(":swapped"):
            return PairOrdering.BA
    return None


def _infer_condition_components(
    row: Mapping[str, Any],
) -> tuple[str | None, str | None, int | None]:
    condition = row.get("condition")
    condition = condition if isinstance(condition, Mapping) else {}
    metadata = row.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    spec = row.get("spec")
    spec = spec if isinstance(spec, Mapping) else {}

    variant_id = _nonempty(condition.get("variant_id") or metadata.get("variant_id"))
    if variant_id is not None:
        try:
            parsed = parse_variant_id(variant_id)
            return (
                parsed.family,
                parsed.direction.value if parsed.direction is not None else None,
                parsed.dose,
            )
        except ValueError:
            pass

    family = _nonempty(condition.get("bias_type") or spec.get("bias_name"))
    if variant_id in {"original", "swapped", "position_control", "position_swapped"}:
        family = "position"

    raw_direction = _nonempty(condition.get("cue_congruency"))
    direction = (
        raw_direction
        if raw_direction in {"congruent", "incongruent"}
        else None
    )
    raw_dose = condition.get("dose")
    if raw_dose is None:
        raw_dose = metadata.get("dose")
    try:
        dose = None if raw_dose is None else int(raw_dose)
    except (TypeError, ValueError):
        dose = None
    return family, direction, dose


def migrate_run_record_linkage(
    row: Mapping[str, Any],
    *,
    input_file_hash: str | None = None,
) -> dict[str, Any]:
    """Return a migrated copy of one serialized RunRecord.

    Linkage is derived only when the source row contains enough information.
    Missing or conflicting values produce an explicit unresolved status rather
    than a guessed key.
    """

    migrated = copy.deepcopy(dict(row))
    metadata_value = migrated.get("metadata")
    metadata: dict[str, Any] = (
        copy.deepcopy(dict(metadata_value))
        if isinstance(metadata_value, Mapping)
        else {}
    )
    migrated["metadata"] = metadata

    spec_value = migrated.get("spec")
    spec = dict(spec_value) if isinstance(spec_value, Mapping) else {}
    if spec and migrated.get("spec_hash") is None:
        migrated["spec_hash"] = canonical_sha256(spec)

    existing_input_hash = _nonempty(migrated.get("input_file_hash"))
    supplied_input_hash = _nonempty(input_file_hash)
    conflicts: list[str] = []
    if (
        existing_input_hash is not None
        and supplied_input_hash is not None
        and existing_input_hash != supplied_input_hash
    ):
        conflicts.append("input_file_hash")
    resolved_input_hash = existing_input_hash or supplied_input_hash
    migrated["input_file_hash"] = resolved_input_hash

    pair_identity_key = _nonempty(metadata.get("pair_identity_key"))
    ordering = _infer_ordering(migrated)
    model_name = _nonempty(spec.get("model_name"))
    dataset_name = _nonempty(spec.get("dataset_name"))
    source_row_index = metadata.get("source_row_index")
    question_id = migrated.get("question_id")
    turn = metadata.get("turn")

    missing: list[str] = []
    if resolved_input_hash is None:
        missing.append("input_file_hash")
    if pair_identity_key is None:
        if dataset_name is None:
            missing.append("dataset_name")
        if source_row_index is None:
            missing.append("source_row_index")
        if question_id is None:
            missing.append("question_id")
        if not missing:
            pair_identity_key = make_pair_identity_key(
                dataset_name=dataset_name or "",
                input_file_hash=resolved_input_hash or "",
                source_row_index=source_row_index,
                question_id=question_id,
                turn=turn,
            )
            metadata["pair_identity_key"] = pair_identity_key

    if model_name is None:
        missing.append("model_name")
    if ordering is None:
        missing.append("ordering")

    family, direction, dose = _infer_condition_components(migrated)
    if family is None:
        missing.append("condition_family")

    if conflicts:
        metadata[LINKAGE_STATUS_KEY] = (
            "unresolved_conflicting_fields:" + ",".join(sorted(conflicts))
        )
        return migrated
    if missing or pair_identity_key is None or model_name is None or ordering is None:
        metadata[LINKAGE_STATUS_KEY] = (
            "unresolved_missing_fields:" + ",".join(sorted(set(missing)))
        )
        migrated.setdefault("pair_key", None)
        migrated.setdefault("condition_group_id", None)
        migrated.setdefault("ordering_twin_key", None)
        return migrated

    expected_pair_key = make_pair_key(
        pair_identity_key=pair_identity_key,
        model_name=model_name,
        ordering=ordering,
    )
    expected_twin_key = make_ordering_twin_key(
        pair_identity_key=pair_identity_key,
        model_name=model_name,
        ordering=ordering,
    )
    expected_group_id = make_condition_group_id(
        pair_identity_key=pair_identity_key,
        model_name=model_name,
        family=family or "",
        direction=direction,
        dose=dose,
    )
    expected = {
        "pair_key": expected_pair_key,
        "condition_group_id": expected_group_id,
        "ordering_twin_key": expected_twin_key,
    }
    linkage_conflicts = [
        key
        for key, value in expected.items()
        if migrated.get(key) is not None and migrated.get(key) != value
    ]
    if linkage_conflicts:
        metadata[LINKAGE_STATUS_KEY] = (
            "unresolved_conflicting_fields:" + ",".join(sorted(linkage_conflicts))
        )
        return migrated

    migrated.update(expected)
    condition_value = migrated.get("condition")
    if isinstance(condition_value, Mapping):
        condition = copy.deepcopy(dict(condition_value))
        condition.setdefault("ordering", ordering.value)
        migrated["condition"] = condition
    metadata[LINKAGE_STATUS_KEY] = "resolved"
    return migrated


def migrate_jsonl(
    *,
    source_path: Path,
    destination_path: Path,
    input_file_hash: str | None = None,
    overwrite: bool = False,
) -> MigrationReport:
    source = source_path.resolve()
    destination = destination_path.resolve()
    if source == destination:
        raise ValueError("Migration must write to a different destination path")
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Destination already exists: {destination}. Pass overwrite=True explicitly."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    resolved = 0
    temporary_path: Path | None = None
    try:
        with source.open(encoding="utf-8") as source_handle:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as destination_handle:
                temporary_path = Path(destination_handle.name)
                for line_number, line in enumerate(source_handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Invalid JSON on line {line_number} of {source}"
                        ) from exc
                    migrated = migrate_run_record_linkage(
                        row,
                        input_file_hash=input_file_hash,
                    )
                    status = str(
                        migrated.get("metadata", {}).get(LINKAGE_STATUS_KEY, "")
                    )
                    total += 1
                    resolved += int(status == "resolved")
                    destination_handle.write(json.dumps(migrated, ensure_ascii=True))
                    destination_handle.write("\n")
        if temporary_path is None:
            raise RuntimeError("Migration did not create a temporary output file")
        temporary_path.replace(destination)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise

    return MigrationReport(
        source_path=str(source),
        destination_path=str(destination),
        total_rows=total,
        resolved_rows=resolved,
        unresolved_rows=total - resolved,
    )


__all__ = [
    "LINKAGE_STATUS_KEY",
    "MigrationReport",
    "migrate_jsonl",
    "migrate_run_record_linkage",
]
