from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from biases.models import get_model_profile
from biases.pairing import file_sha256, make_pair_identity_key
from biases.parser_integrity import ParserIntegrityError, derive_parser_fields
from biases.position_bias import (
    CONSTRAINED_LOGPROBS_MODE,
    JUDGE_OUTPUT_PARSER_VERSION,
    load_position_pairs,
    verbalized_parse_status,
)
from biases.schemas import JudgeExample, PairOrdering, RunRecord, VerdictLabel
from biases.silent_bias_runner import (
    _clean_summary_row,
    _cued_summary_row,
    consistency_runs_for_condition,
)
from biases.social_cue_prompts import (
    AUTHORITY_DOSES,
    BANDWAGON_DOSES,
    format_variant_id,
)
from biases.stage_planning import (
    CleanPairSummary,
    PlannedCondition,
    StageAPairInput,
    generate_stage_a_conditions,
    generate_stage_b_conditions,
)
from biases.utils import stable_hash


StageName = Literal["stage_a", "stage_b"]
REQUIRED_PROBABILITY_LABELS = frozenset(("A", "B", "tie"))
RAW_FILENAMES: Mapping[StageName, str] = {
    "stage_a": "silent_bias_stage_a_run_records.jsonl",
    "stage_b": "silent_bias_stage_b_run_records.jsonl",
}
SCORE_FILENAMES: Mapping[StageName, str] = {
    "stage_a": "silent_bias_stage_a_uncertainty_scores.jsonl",
    "stage_b": "silent_bias_stage_b_uncertainty_scores.jsonl",
}
PAIR_SUMMARY_FILENAMES: Mapping[StageName, str] = {
    "stage_a": "silent_bias_stage_a_pair_summary.jsonl",
    "stage_b": "silent_bias_stage_b_pair_summary.jsonl",
}
STAGE_SUMMARY_FILENAMES: Mapping[StageName, str] = {
    "stage_a": "silent_bias_stage_a_summary.json",
    "stage_b": "silent_bias_stage_b_summary.json",
}


@dataclass(frozen=True, slots=True)
class SourcePair:
    pair_input: StageAPairInput
    pair_identity_key: str
    examples_by_ordering: Mapping[str, JudgeExample]


@dataclass(frozen=True, slots=True)
class ArtifactResult:
    report: dict[str, Any]
    grids: Mapping[StageName, frozenset[tuple[str, str, str]]]


class IssueCollector:
    def __init__(self, *, max_reported: int) -> None:
        if max_reported < 1:
            raise ValueError("max_reported must be at least 1")
        self._max_reported = max_reported
        self._issues: list[dict[str, Any]] = []
        self._counts: Counter[str] = Counter()

    def add(
        self,
        code: str,
        message: str,
        *,
        artifact_dir: Path | None = None,
        stage: StageName | None = None,
        record_id: object | None = None,
    ) -> None:
        self._counts[code] += 1
        if len(self._issues) >= self._max_reported:
            return
        issue: dict[str, Any] = {"code": code, "message": message}
        if artifact_dir is not None:
            issue["artifact_dir"] = str(artifact_dir)
        if stage is not None:
            issue["stage"] = stage
        if record_id is not None:
            issue["record_id"] = str(record_id)
        self._issues.append(issue)

    @property
    def error_count(self) -> int:
        return sum(self._counts.values())

    def summary(self) -> dict[str, Any]:
        return {
            "error_count": self.error_count,
            "error_counts_by_code": dict(sorted(self._counts.items())),
            "errors": self._issues,
            "errors_truncated": self.error_count > len(self._issues),
        }


def _read_jsonl(
    path: Path,
    *,
    collector: IssueCollector,
    artifact_dir: Path,
    stage: StageName,
) -> list[dict[str, Any]]:
    if not path.is_file():
        collector.add(
            "missing_artifact_file",
            f"required artifact file is missing: {path.name}",
            artifact_dir=artifact_dir,
            stage=stage,
        )
        return []

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                collector.add(
                    "invalid_jsonl",
                    f"{path.name}:{line_number}: {exc.msg}",
                    artifact_dir=artifact_dir,
                    stage=stage,
                )
                continue
            if not isinstance(row, dict):
                collector.add(
                    "invalid_jsonl_row",
                    f"{path.name}:{line_number} is not a JSON object",
                    artifact_dir=artifact_dir,
                    stage=stage,
                )
                continue
            rows.append(row)
    return rows


def _read_json_object(
    path: Path,
    *,
    collector: IssueCollector,
    artifact_dir: Path,
    stage: StageName,
) -> dict[str, Any]:
    if not path.is_file():
        collector.add(
            "missing_artifact_file",
            f"required artifact file is missing: {path.name}",
            artifact_dir=artifact_dir,
            stage=stage,
        )
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        collector.add(
            "invalid_json",
            f"{path.name}: {exc.msg}",
            artifact_dir=artifact_dir,
            stage=stage,
        )
        return {}
    if not isinstance(payload, dict):
        collector.add(
            "invalid_json",
            f"{path.name} is not a JSON object",
            artifact_dir=artifact_dir,
            stage=stage,
        )
        return {}
    return payload


def _source_pairs(
    source_csv: Path,
    *,
    input_file_hash: str,
    limit: int | None,
) -> tuple[SourcePair, ...]:
    pairs = load_position_pairs(csv_path=source_csv, limit=limit)
    source_pairs: list[SourcePair] = []
    for pair in pairs:
        original = pair.original
        source_row_index = original.metadata["source_row_index"]
        question_id = (
            original.metadata.get("question_cluster_id") or original.question_id
        )
        pair_input = StageAPairInput(
            dataset_name=source_csv.name,
            input_file_hash=input_file_hash,
            source_row_index=source_row_index,
            question_id=question_id,
            model_name="",
            human_winner=original.human_winner or VerdictLabel.TIE,
            turn=original.metadata.get("turn"),
            response_a_id=original.candidates["A"].response_id,
            response_b_id=original.candidates["B"].response_id,
        )
        pair_identity_key = make_pair_identity_key(
            dataset_name=pair_input.dataset_name,
            input_file_hash=pair_input.input_file_hash,
            source_row_index=pair_input.source_row_index,
            question_id=pair_input.question_id,
            turn=pair_input.turn,
            response_a_id=pair_input.response_a_id,
            response_b_id=pair_input.response_b_id,
        )
        source_pairs.append(
            SourcePair(
                pair_input=pair_input,
                pair_identity_key=pair_identity_key,
                examples_by_ordering={
                    PairOrdering.AB.value: original,
                    PairOrdering.BA.value: pair.swapped,
                },
            )
        )
    return tuple(source_pairs)


def _model_name(rows_by_stage: Mapping[StageName, Sequence[Mapping[str, Any]]]) -> str | None:
    names = {
        str(spec["model_name"])
        for rows in rows_by_stage.values()
        for row in rows
        if isinstance((spec := row.get("spec")), Mapping)
        and spec.get("model_name") not in (None, "")
    }
    return next(iter(names)) if len(names) == 1 else None


def _stage_a_expectations(
    source_pairs: Sequence[SourcePair],
    *,
    model_name: str,
) -> tuple[
    dict[tuple[str, str], PlannedCondition],
    dict[tuple[str, str], JudgeExample],
]:
    inputs = [
        StageAPairInput(
            dataset_name=source.pair_input.dataset_name,
            input_file_hash=source.pair_input.input_file_hash,
            source_row_index=source.pair_input.source_row_index,
            question_id=source.pair_input.question_id,
            model_name=model_name,
            human_winner=source.pair_input.human_winner,
            turn=source.pair_input.turn,
            response_a_id=source.pair_input.response_a_id,
            response_b_id=source.pair_input.response_b_id,
        )
        for source in source_pairs
    ]
    plan = generate_stage_a_conditions(inputs)
    if plan.issues:
        raise ValueError(
            f"source CSV produced {len(plan.issues)} Stage A planning issue(s)"
        )
    source_by_identity = {
        source.pair_identity_key: source
        for source in source_pairs
    }
    expected: dict[tuple[str, str], PlannedCondition] = {}
    examples: dict[tuple[str, str], JudgeExample] = {}
    for planned in plan.conditions:
        key = (planned.pair_key, planned.condition.variant_id)
        expected[key] = planned
        ordering = str(planned.condition.ordering)
        examples[key] = source_by_identity[
            planned.pair_identity_key
        ].examples_by_ordering[ordering]
    return expected, examples


def _condition_key(row: Mapping[str, Any], *, raw: bool) -> tuple[str, str] | None:
    pair_key = row.get("pair_key")
    if raw:
        condition = row.get("condition")
        variant_id = (
            condition.get("variant_id") if isinstance(condition, Mapping) else None
        )
    else:
        variant_id = row.get("variant_id")
    if not isinstance(pair_key, str) or not pair_key:
        return None
    if not isinstance(variant_id, str) or not variant_id:
        return None
    return pair_key, variant_id


def _unique_index(
    rows: Sequence[Mapping[str, Any]],
    *,
    field_name: str,
    collector: IssueCollector,
    artifact_dir: Path,
    stage: StageName,
    code: str,
) -> dict[str, Mapping[str, Any]]:
    index: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        value = row.get(field_name)
        if not isinstance(value, str) or not value:
            collector.add(
                f"missing_{field_name}",
                f"row has no non-empty {field_name}",
                artifact_dir=artifact_dir,
                stage=stage,
            )
            continue
        if value in index:
            collector.add(
                code,
                f"duplicate {field_name} {value!r}",
                artifact_dir=artifact_dir,
                stage=stage,
                record_id=value if field_name == "record_id" else row.get("record_id"),
            )
            continue
        index[value] = row
    return index


def _condition_index(
    rows: Sequence[Mapping[str, Any]],
    *,
    raw: bool,
    collector: IssueCollector,
    artifact_dir: Path,
    stage: StageName,
) -> dict[tuple[str, str], Mapping[str, Any]]:
    index: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = _condition_key(row, raw=raw)
        if key is None:
            collector.add(
                "missing_condition_key",
                "row has no complete pair_key/variant_id condition key",
                artifact_dir=artifact_dir,
                stage=stage,
                record_id=row.get("record_id"),
            )
            continue
        if key in index:
            collector.add(
                "duplicate_condition_key",
                f"duplicate condition key {key!r}",
                artifact_dir=artifact_dir,
                stage=stage,
                record_id=row.get("record_id"),
            )
            continue
        index[key] = row
    return index


def _validate_global_uniqueness(
    rows_by_stage: Mapping[StageName, Sequence[Mapping[str, Any]]],
    *,
    raw: bool,
    collector: IssueCollector,
    artifact_dir: Path,
) -> None:
    record_locations: dict[str, StageName] = {}
    condition_locations: dict[tuple[str, str], StageName] = {}
    for stage, rows in rows_by_stage.items():
        for row in rows:
            record_id = row.get("record_id")
            if isinstance(record_id, str) and record_id:
                previous_stage = record_locations.get(record_id)
                if previous_stage is not None:
                    collector.add(
                        "duplicate_record_id_across_stages",
                        f"record_id also occurs in {previous_stage}",
                        artifact_dir=artifact_dir,
                        stage=stage,
                        record_id=record_id,
                    )
                else:
                    record_locations[record_id] = stage
            condition_key = _condition_key(row, raw=raw)
            if condition_key is None:
                continue
            previous_stage = condition_locations.get(condition_key)
            if previous_stage is not None:
                collector.add(
                    "duplicate_condition_key_across_stages",
                    f"condition key also occurs in {previous_stage}: {condition_key!r}",
                    artifact_dir=artifact_dir,
                    stage=stage,
                    record_id=record_id,
                )
            else:
                condition_locations[condition_key] = stage


def _report_grid_difference(
    *,
    actual: set[tuple[str, str]],
    expected: set[tuple[str, str]],
    collector: IssueCollector,
    artifact_dir: Path,
    stage: StageName,
) -> None:
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        collector.add(
            "missing_grid_cells",
            f"{len(missing)} expected cells are missing; sample={missing[:3]!r}",
            artifact_dir=artifact_dir,
            stage=stage,
        )
    if extra:
        collector.add(
            "unexpected_grid_cells",
            f"{len(extra)} unexpected cells are present; sample={extra[:3]!r}",
            artifact_dir=artifact_dir,
            stage=stage,
        )


def _check_equal(
    actual: object,
    expected: object,
    *,
    code: str,
    field: str,
    collector: IssueCollector,
    artifact_dir: Path,
    stage: StageName,
    record_id: object | None,
) -> None:
    if actual != expected:
        collector.add(
            code,
            f"{field} is {actual!r}; expected {expected!r}",
            artifact_dir=artifact_dir,
            stage=stage,
            record_id=record_id,
        )


def _validate_probabilities(
    probabilities: Mapping[str, Any] | None,
    *,
    collector: IssueCollector,
    artifact_dir: Path,
    stage: StageName,
    record_id: object | None,
) -> None:
    if not isinstance(probabilities, Mapping):
        collector.add(
            "invalid_label_probabilities",
            "label probabilities are missing or are not an object",
            artifact_dir=artifact_dir,
            stage=stage,
            record_id=record_id,
        )
        return
    if set(probabilities) != REQUIRED_PROBABILITY_LABELS:
        collector.add(
            "invalid_label_probabilities",
            "label probabilities must have exactly A/B/tie support",
            artifact_dir=artifact_dir,
            stage=stage,
            record_id=record_id,
        )
        return
    values = list(probabilities.values())
    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
        for value in values
    ):
        collector.add(
            "invalid_label_probabilities",
            "label probabilities must be finite numbers in [0, 1]",
            artifact_dir=artifact_dir,
            stage=stage,
            record_id=record_id,
        )
        return
    if not math.isclose(
        sum(float(value) for value in values),
        1.0,
        rel_tol=1e-7,
        abs_tol=1e-7,
    ):
        collector.add(
            "invalid_label_probabilities",
            "label probabilities do not sum to one",
            artifact_dir=artifact_dir,
            stage=stage,
            record_id=record_id,
        )


def _score_probabilities(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "A": row.get("label_prob_A"),
        "B": row.get("label_prob_B"),
        "tie": row.get("label_prob_tie"),
    }


def _values_match(actual: object, expected: object) -> bool:
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        if isinstance(actual, bool) or isinstance(expected, bool):
            return actual == expected
        return math.isclose(
            float(actual),
            float(expected),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    if isinstance(actual, Mapping) and isinstance(expected, Mapping):
        return set(actual) == set(expected) and all(
            _values_match(actual[key], expected[key]) for key in expected
        )
    return actual == expected


def _validate_parser_derived_fields(
    row: Mapping[str, Any],
    *,
    collector: IssueCollector,
    artifact_dir: Path,
    stage: StageName,
) -> None:
    record_id = row.get("record_id")
    spec = row.get("spec")
    metadata = row.get("metadata")
    uncertainty = row.get("uncertainty")
    spec = spec if isinstance(spec, Mapping) else {}
    metadata = metadata if isinstance(metadata, Mapping) else {}
    uncertainty = uncertainty if isinstance(uncertainty, Mapping) else {}
    if metadata.get("judge_output_parser_version") != JUDGE_OUTPUT_PARSER_VERSION:
        collector.add(
            "stale_parser_version",
            f"raw parser version is "
            f"{metadata.get('judge_output_parser_version')!r}; expected "
            f"{JUDGE_OUTPUT_PARSER_VERSION!r}",
            artifact_dir=artifact_dir,
            stage=stage,
            record_id=record_id,
        )
    methods = spec.get("uncertainty_methods")
    expected_parse_status = verbalized_parse_status(
        uncertainty_methods=methods if isinstance(methods, list) else [],
        raw_output=metadata.get("verbalized_raw_output"),
    )
    actual_parse_status = metadata.get("verbalized_parse_status")
    if (
        metadata.get("judge_output_parser_version")
        == JUDGE_OUTPUT_PARSER_VERSION
        and actual_parse_status != expected_parse_status
    ):
        collector.add(
            "stored_verbalized_mismatch",
            f"metadata.verbalized_parse_status is {actual_parse_status!r}; "
            f"expected {expected_parse_status!r}",
            artifact_dir=artifact_dir,
            stage=stage,
            record_id=record_id,
        )
    try:
        derived = derive_parser_fields(row)
    except ParserIntegrityError as exc:
        collector.add(
            "strict_parser_mismatch",
            str(exc),
            artifact_dir=artifact_dir,
            stage=stage,
            record_id=record_id,
        )
        return

    if row.get("verdict") != derived.verdict.value:
        collector.add(
            "stored_verdict_mismatch",
            f"stored verdict is {row.get('verdict')!r}; strict verdict is "
            f"{derived.verdict.value!r}",
            artifact_dir=artifact_dir,
            stage=stage,
            record_id=record_id,
        )
    expected_fields = {
        "logit": derived.logit.model_dump(mode="json"),
        "verbalized": derived.verbalized.model_dump(mode="json"),
        "consistency": (
            derived.consistency.model_dump(mode="json")
            if derived.consistency is not None
            else None
        ),
    }
    for field, expected in expected_fields.items():
        actual = uncertainty.get(field)
        if not _values_match(actual, expected):
            collector.add(
                "derived_uncertainty_mismatch",
                f"uncertainty.{field} does not match values recomputed from "
                "stored parser primitives",
                artifact_dir=artifact_dir,
                stage=stage,
                record_id=record_id,
            )
    expected_verbalized_verdict = (
        derived.verbalized.verdict.value
        if derived.verbalized.verdict is not None
        else None
    )
    if metadata.get("verbalized_verdict") != expected_verbalized_verdict:
        collector.add(
            "stored_verbalized_mismatch",
            f"metadata.verbalized_verdict is "
            f"{metadata.get('verbalized_verdict')!r}; expected "
            f"{expected_verbalized_verdict!r}",
            artifact_dir=artifact_dir,
            stage=stage,
            record_id=record_id,
        )
def _expected_flat_projection(record: Mapping[str, Any]) -> dict[str, Any]:
    spec = record.get("spec")
    condition = record.get("condition")
    uncertainty = record.get("uncertainty")
    metadata = record.get("metadata")
    spec = spec if isinstance(spec, Mapping) else {}
    condition = condition if isinstance(condition, Mapping) else {}
    uncertainty = uncertainty if isinstance(uncertainty, Mapping) else {}
    metadata = metadata if isinstance(metadata, Mapping) else {}
    logit = uncertainty.get("logit")
    verbalized = uncertainty.get("verbalized")
    consistency = uncertainty.get("consistency")
    logit = logit if isinstance(logit, Mapping) else {}
    verbalized = verbalized if isinstance(verbalized, Mapping) else {}
    consistency = consistency if isinstance(consistency, Mapping) else {}
    probabilities = record.get("raw_prompt_logprobs")
    probabilities = probabilities if isinstance(probabilities, Mapping) else {}
    return {
        "record_id": record.get("record_id"),
        "model_name": spec.get("model_name"),
        "dataset_name": spec.get("dataset_name"),
        "dataset_split": spec.get("dataset_split"),
        "bias_name": spec.get("bias_name"),
        "example_id": record.get("example_id"),
        "question_id": record.get("question_id"),
        "pair_id": metadata.get("pair_id"),
        "source_row_index": metadata.get("source_row_index"),
        "pair_identity_key": metadata.get("pair_identity_key"),
        "pair_key": record.get("pair_key"),
        "condition_group_id": record.get("condition_group_id"),
        "ordering_twin_key": record.get("ordering_twin_key"),
        "spec_hash": record.get("spec_hash"),
        "input_file_hash": record.get("input_file_hash"),
        "routing_split": metadata.get("routing_split"),
        "turn": metadata.get("turn"),
        "selected_turn": metadata.get("selected_turn"),
        "conversation_extraction_mode": metadata.get(
            "conversation_extraction_mode"
        ),
        "variant_id": condition.get("variant_id"),
        "ordering": condition.get("ordering"),
        "dose": condition.get("dose"),
        "cue_congruency": condition.get("cue_congruency"),
        "direction_relative_human": condition.get("direction_relative_human"),
        "cue_target": condition.get("cue_target"),
        "clean_tie": condition.get("clean_tie"),
        "clean_record_id": condition.get("clean_record_id"),
        "human_winner": metadata.get("human_winner"),
        "verdict": record.get("verdict"),
        "underlying_response_id": metadata.get("underlying_response_id"),
        "label_prob_A": probabilities.get("A"),
        "label_prob_B": probabilities.get("B"),
        "label_prob_tie": probabilities.get("tie"),
        "entropy": logit.get("entropy"),
        "normalized_entropy": logit.get("normalized_entropy"),
        "msp": logit.get("msp"),
        "margin": logit.get("margin"),
        "verbalized_confidence": verbalized.get("confidence"),
        "verbalized_uncertainty": verbalized.get("uncertainty"),
        "verbalized_verdict": (
            verbalized.get("verdict") or metadata.get("verbalized_verdict")
        ),
        "verbalized_parse_status": metadata.get("verbalized_parse_status"),
        "consistency_agreement_rate": consistency.get("agreement_rate"),
        "consistency_vote_entropy": consistency.get("vote_entropy"),
        "consistency_unique_verdict_count": consistency.get(
            "unique_verdict_count"
        ),
        "consistency_flip_rate": consistency.get("flip_rate"),
        "consistency_verdict_counts": consistency.get("verdict_counts"),
        "consistency_majority_verdict": consistency.get("majority_verdict"),
        "decision_token_index": metadata.get("decision_token_index"),
        "decision_token_labels": metadata.get("decision_token_labels"),
        "judge_output_parser_version": metadata.get(
            "judge_output_parser_version"
        ),
        "logprobs_mode": spec.get("logprobs_mode"),
        "max_num_batched_tokens": metadata.get("max_num_batched_tokens"),
        "max_num_seqs": metadata.get("max_num_seqs"),
    }


def _validate_flat_scores(
    raw_rows: Sequence[Mapping[str, Any]],
    score_rows: Sequence[Mapping[str, Any]],
    *,
    collector: IssueCollector,
    artifact_dir: Path,
    stage: StageName,
) -> None:
    raw_by_id = _unique_index(
        raw_rows,
        field_name="record_id",
        collector=collector,
        artifact_dir=artifact_dir,
        stage=stage,
        code="duplicate_record_id",
    )
    score_by_id = _unique_index(
        score_rows,
        field_name="record_id",
        collector=collector,
        artifact_dir=artifact_dir,
        stage=stage,
        code="duplicate_score_record_id",
    )
    _condition_index(
        score_rows,
        raw=False,
        collector=collector,
        artifact_dir=artifact_dir,
        stage=stage,
    )
    missing = sorted(set(raw_by_id) - set(score_by_id))
    extra = sorted(set(score_by_id) - set(raw_by_id))
    if missing:
        collector.add(
            "missing_flat_score_rows",
            f"{len(missing)} raw records have no flat score row; sample={missing[:3]!r}",
            artifact_dir=artifact_dir,
            stage=stage,
        )
    if extra:
        collector.add(
            "unexpected_flat_score_rows",
            f"{len(extra)} flat score rows have no raw record; sample={extra[:3]!r}",
            artifact_dir=artifact_dir,
            stage=stage,
        )

    for record_id in sorted(set(raw_by_id) & set(score_by_id)):
        raw = raw_by_id[record_id]
        score = score_by_id[record_id]
        if score.get("judge_output_parser_version") != JUDGE_OUTPUT_PARSER_VERSION:
            collector.add(
                "stale_parser_version",
                f"flat parser version is "
                f"{score.get('judge_output_parser_version')!r}; expected "
                f"{JUDGE_OUTPUT_PARSER_VERSION!r}",
                artifact_dir=artifact_dir,
                stage=stage,
                record_id=record_id,
            )
        projection = _expected_flat_projection(raw)
        for field, expected in projection.items():
            if score.get(field) != expected:
                collector.add(
                    "flat_score_mismatch",
                    f"flat field {field} is {score.get(field)!r}; expected {expected!r}",
                    artifact_dir=artifact_dir,
                    stage=stage,
                    record_id=record_id,
                )
        _validate_probabilities(
            _score_probabilities(score),
            collector=collector,
            artifact_dir=artifact_dir,
            stage=stage,
            record_id=record_id,
        )


def _validate_pair_summaries(
    raw_rows: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
    *,
    collector: IssueCollector,
    artifact_dir: Path,
    stage: StageName,
) -> None:
    raw_by_id = {
        str(row["record_id"]): row
        for row in raw_rows
        if isinstance(row.get("record_id"), str) and row.get("record_id")
    }
    pairs_by_id = _unique_index(
        pair_rows,
        field_name="record_id",
        collector=collector,
        artifact_dir=artifact_dir,
        stage=stage,
        code="duplicate_pair_summary_record_id",
    )
    missing = sorted(set(raw_by_id) - set(pairs_by_id))
    extra = sorted(set(pairs_by_id) - set(raw_by_id))
    if missing:
        collector.add(
            "missing_pair_summary_rows",
            f"{len(missing)} raw records have no pair summary; sample={missing[:3]!r}",
            artifact_dir=artifact_dir,
            stage=stage,
        )
    if extra:
        collector.add(
            "unexpected_pair_summary_rows",
            f"{len(extra)} pair summaries have no raw record; sample={extra[:3]!r}",
            artifact_dir=artifact_dir,
            stage=stage,
        )
    for record_id in sorted(set(raw_by_id) & set(pairs_by_id)):
        pair_row = pairs_by_id[record_id]
        if (
            pair_row.get("judge_output_parser_version")
            != JUDGE_OUTPUT_PARSER_VERSION
        ):
            collector.add(
                "stale_parser_version",
                f"pair-summary parser version is "
                f"{pair_row.get('judge_output_parser_version')!r}; expected "
                f"{JUDGE_OUTPUT_PARSER_VERSION!r}",
                artifact_dir=artifact_dir,
                stage=stage,
                record_id=record_id,
            )
        try:
            record = RunRecord.model_validate(raw_by_id[record_id])
        except ValidationError:
            continue
        expected = (
            _clean_summary_row(record)
            if stage == "stage_a"
            else _cued_summary_row(record)
        )
        for field, expected_value in expected.items():
            if not _values_match(pair_row.get(field), expected_value):
                collector.add(
                    "pair_summary_mismatch",
                    f"pair-summary field {field} is {pair_row.get(field)!r}; "
                    f"expected {expected_value!r}",
                    artifact_dir=artifact_dir,
                    stage=stage,
                    record_id=record_id,
                )


def _validate_stage_summary(
    summary: Mapping[str, Any],
    *,
    expected_records: int,
    collector: IssueCollector,
    artifact_dir: Path,
    stage: StageName,
) -> None:
    if summary.get("judge_output_parser_version") != JUDGE_OUTPUT_PARSER_VERSION:
        collector.add(
            "stale_parser_version",
            f"stage-summary parser version is "
            f"{summary.get('judge_output_parser_version')!r}; expected "
            f"{JUDGE_OUTPUT_PARSER_VERSION!r}",
            artifact_dir=artifact_dir,
            stage=stage,
        )
    if summary.get("records_written") != expected_records:
        collector.add(
            "stage_summary_count_mismatch",
            f"stage summary records_written is {summary.get('records_written')!r}; "
            f"expected {expected_records}",
            artifact_dir=artifact_dir,
            stage=stage,
        )


def _validate_scheduler_provenance(
    raw_rows: Sequence[Mapping[str, Any]],
    score_rows: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    *,
    collector: IssueCollector,
    artifact_dir: Path,
    stage: StageName,
) -> None:
    for field in ("max_num_batched_tokens", "max_num_seqs"):
        if field not in summary:
            collector.add(
                "missing_scheduler_provenance",
                f"stage summary has no {field}",
                artifact_dir=artifact_dir,
                stage=stage,
            )
        expected = summary.get(field)
        if expected is not None and (
            not isinstance(expected, int)
            or isinstance(expected, bool)
            or expected < 1
        ):
            collector.add(
                "invalid_scheduler_provenance",
                f"stage summary {field} must be a positive integer or null",
                artifact_dir=artifact_dir,
                stage=stage,
            )
        row_groups = (
            ("raw metadata", raw_rows, True),
            ("flat row", score_rows, False),
            ("pair summary", pair_rows, False),
        )
        for group_name, rows, nested in row_groups:
            for row in rows:
                container = row.get("metadata") if nested else row
                container = container if isinstance(container, Mapping) else {}
                if field not in container or container.get(field) != expected:
                    collector.add(
                        "scheduler_provenance_mismatch",
                        f"{group_name} {field} is {container.get(field)!r}; "
                        f"expected {expected!r}",
                        artifact_dir=artifact_dir,
                        stage=stage,
                        record_id=row.get("record_id"),
                    )


def _validate_logprobs_mode_provenance(
    raw_rows: Sequence[Mapping[str, Any]],
    score_rows: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    *,
    collector: IssueCollector,
    artifact_dir: Path,
    stage: StageName,
) -> None:
    expected = CONSTRAINED_LOGPROBS_MODE
    if summary.get("logprobs_mode") != expected:
        collector.add(
            "logprobs_mode_mismatch",
            f"stage summary logprobs_mode is "
            f"{summary.get('logprobs_mode')!r}; expected {expected!r}",
            artifact_dir=artifact_dir,
            stage=stage,
        )

    for row in raw_rows:
        record_id = row.get("record_id")
        spec = row.get("spec")
        metadata = row.get("metadata")
        spec = spec if isinstance(spec, Mapping) else {}
        metadata = metadata if isinstance(metadata, Mapping) else {}
        for location, actual in (
            ("raw spec", spec.get("logprobs_mode")),
            ("raw metadata", metadata.get("logprobs_mode")),
        ):
            if actual != expected:
                collector.add(
                    "logprobs_mode_mismatch",
                    f"{location} logprobs_mode is {actual!r}; "
                    f"expected {expected!r}",
                    artifact_dir=artifact_dir,
                    stage=stage,
                    record_id=record_id,
                )

    for location, rows in (
        ("flat row", score_rows),
        ("pair summary", pair_rows),
    ):
        for row in rows:
            if row.get("logprobs_mode") != expected:
                collector.add(
                    "logprobs_mode_mismatch",
                    f"{location} logprobs_mode is "
                    f"{row.get('logprobs_mode')!r}; expected {expected!r}",
                    artifact_dir=artifact_dir,
                    stage=stage,
                    record_id=row.get("record_id"),
                )


def _validate_verbalized_status_summary(
    raw_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    *,
    collector: IssueCollector,
    artifact_dir: Path,
    stage: StageName,
) -> None:
    expected: Counter[str] = Counter()
    for row in raw_rows:
        try:
            status = derive_parser_fields(row).verbalized_parse_status
        except ParserIntegrityError:
            continue
        expected[status] += 1
    actual = summary.get("verbalized_parse_status_counts")
    if actual != dict(sorted(expected.items())):
        collector.add(
            "verbalized_missingness_summary_mismatch",
            f"verbalized_parse_status_counts is {actual!r}; expected "
            f"{dict(sorted(expected.items()))!r}",
            artifact_dir=artifact_dir,
            stage=stage,
        )


def _validate_verbalized_availability(
    raw_rows: Sequence[Mapping[str, Any]],
    *,
    minimum: float,
    collector: IssueCollector,
    artifact_dir: Path,
    stage: StageName,
) -> dict[str, int | float | None]:
    requested = 0
    parsed = 0
    for row in raw_rows:
        spec = row.get("spec")
        methods = (
            spec.get("uncertainty_methods")
            if isinstance(spec, Mapping)
            else None
        )
        if not isinstance(methods, list) or "verbalized_confidence" not in methods:
            continue
        requested += 1
        try:
            status = derive_parser_fields(row).verbalized_parse_status
        except ParserIntegrityError:
            continue
        parsed += int(status == "parsed")

    availability = parsed / requested if requested else None
    if (
        availability is not None
        and availability + 1e-12 < minimum
    ):
        collector.add(
            "verbalized_availability_below_minimum",
            f"parsed {parsed}/{requested} requested verbalized-confidence "
            f"channels ({availability:.6f}); required at least {minimum:.6f}",
            artifact_dir=artifact_dir,
            stage=stage,
        )
    return {
        "requested": requested,
        "parsed": parsed,
        "availability": availability,
        "minimum_required": minimum,
    }


def _condition_fields(condition: PlannedCondition) -> dict[str, Any]:
    return condition.condition.model_dump(mode="json")


def _validate_source_metadata(
    row: Mapping[str, Any],
    *,
    expected_example: JudgeExample,
    pair_identity_key: str,
    collector: IssueCollector,
    artifact_dir: Path,
    stage: StageName,
) -> None:
    metadata = row.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    record_id = row.get("record_id")
    expected_metadata = expected_example.metadata
    expected = {
        "example_id": expected_example.example_id,
        "question_id": str(
            expected_metadata.get("question_cluster_id")
            or expected_example.question_id
        ),
        "pair_id": expected_metadata.get("pair_id"),
        "source_row_index": expected_metadata.get("source_row_index"),
        "routing_split": expected_metadata.get("routing_split"),
        "turn": expected_metadata.get("turn"),
        "selected_turn": expected_metadata.get("selected_turn"),
        "conversation_extraction_mode": expected_metadata.get(
            "conversation_extraction_mode"
        ),
        "human_winner": expected_example.human_winner,
        "pair_identity_key": pair_identity_key,
    }
    actual = {
        "example_id": row.get("example_id"),
        "question_id": row.get("question_id"),
        **{field: metadata.get(field) for field in expected if field not in {"example_id", "question_id"}},
    }
    for field, expected_value in expected.items():
        _check_equal(
            actual.get(field),
            expected_value,
            code="source_metadata_mismatch",
            field=field,
            collector=collector,
            artifact_dir=artifact_dir,
            stage=stage,
            record_id=record_id,
        )


def _validate_spec_and_consistency(
    row: Mapping[str, Any],
    *,
    expected_condition: PlannedCondition,
    source_csv: Path,
    input_file_hash: str,
    model_name: str,
    model_revision: str | None,
    dataset_split: str | None,
    consistency_runs: int,
    consistency_schedule: Literal["all", "extremes"],
    sampling_temperature: float,
    collector: IssueCollector,
    artifact_dir: Path,
    stage: StageName,
) -> None:
    record_id = row.get("record_id")
    spec = row.get("spec")
    metadata = row.get("metadata")
    uncertainty = row.get("uncertainty")
    if not isinstance(spec, Mapping):
        collector.add(
            "missing_spec",
            "raw record has no spec object",
            artifact_dir=artifact_dir,
            stage=stage,
            record_id=record_id,
        )
        return
    metadata = metadata if isinstance(metadata, Mapping) else {}
    uncertainty = uncertainty if isinstance(uncertainty, Mapping) else {}
    expected_run_count = consistency_runs_for_condition(
        expected_condition.condition,
        consistency_runs=consistency_runs,
        consistency_schedule=consistency_schedule,
    )
    expected_spec_fields: dict[str, Any] = {
        "dataset_name": source_csv.name,
        "model_name": model_name,
        "model_revision": model_revision,
        "backend_name": "vllm",
        "bias_name": str(expected_condition.condition.bias_type),
        "output_mode": "choice_only",
        "consistency_runs": expected_run_count,
        "temperature": sampling_temperature,
        "consistency_schedule": consistency_schedule,
    }
    if dataset_split is not None:
        expected_spec_fields["dataset_split"] = dataset_split
    for field, expected in expected_spec_fields.items():
        _check_equal(
            spec.get(field),
            expected,
            code="spec_provenance_mismatch",
            field=f"spec.{field}",
            collector=collector,
            artifact_dir=artifact_dir,
            stage=stage,
            record_id=record_id,
        )
    _check_equal(
        row.get("input_file_hash"),
        input_file_hash,
        code="input_hash_mismatch",
        field="input_file_hash",
        collector=collector,
        artifact_dir=artifact_dir,
        stage=stage,
        record_id=record_id,
    )
    _check_equal(
        row.get("spec_hash"),
        stable_hash(dict(spec)),
        code="spec_hash_mismatch",
        field="spec_hash",
        collector=collector,
        artifact_dir=artifact_dir,
        stage=stage,
        record_id=record_id,
    )
    _check_equal(
        metadata.get("stage"),
        stage,
        code="stage_metadata_mismatch",
        field="metadata.stage",
        collector=collector,
        artifact_dir=artifact_dir,
        stage=stage,
        record_id=record_id,
    )
    _check_equal(
        metadata.get("consistency_runs_actual"),
        expected_run_count,
        code="consistency_schedule_mismatch",
        field="metadata.consistency_runs_actual",
        collector=collector,
        artifact_dir=artifact_dir,
        stage=stage,
        record_id=record_id,
    )

    methods_value = spec.get("uncertainty_methods")
    methods = (
        {str(value) for value in methods_value}
        if isinstance(methods_value, list)
        else set()
    )
    if "logit" not in methods:
        collector.add(
            "uncertainty_method_mismatch",
            "spec.uncertainty_methods does not contain logit",
            artifact_dir=artifact_dir,
            stage=stage,
            record_id=record_id,
        )
    consistency = uncertainty.get("consistency")
    consistency_methods = {"consistency", "consistency_entropy"}
    if expected_run_count == 0:
        if consistency is not None or methods & consistency_methods:
            collector.add(
                "consistency_schedule_mismatch",
                "consistency output is present outside the configured schedule",
                artifact_dir=artifact_dir,
                stage=stage,
                record_id=record_id,
            )
        return

    if not consistency_methods.issubset(methods):
        collector.add(
            "uncertainty_method_mismatch",
            "scheduled consistency row is missing consistency methods",
            artifact_dir=artifact_dir,
            stage=stage,
            record_id=record_id,
        )
    if not isinstance(consistency, Mapping):
        collector.add(
            "missing_consistency_metrics",
            "scheduled consistency row has no consistency metrics",
            artifact_dir=artifact_dir,
            stage=stage,
            record_id=record_id,
        )
        return
    _check_equal(
        consistency.get("run_count"),
        expected_run_count,
        code="consistency_run_count_mismatch",
        field="uncertainty.consistency.run_count",
        collector=collector,
        artifact_dir=artifact_dir,
        stage=stage,
        record_id=record_id,
    )
    verdict_counts = consistency.get("verdict_counts")
    if not isinstance(verdict_counts, Mapping) or not all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in verdict_counts.values()
    ):
        collector.add(
            "invalid_consistency_counts",
            "consistency verdict_counts must contain nonnegative integers",
            artifact_dir=artifact_dir,
            stage=stage,
            record_id=record_id,
        )
    elif sum(verdict_counts.values()) != expected_run_count:
        collector.add(
            "invalid_consistency_counts",
            "consistency verdict_counts do not sum to run_count",
            artifact_dir=artifact_dir,
            stage=stage,
            record_id=record_id,
        )


def _expected_record_id(row: Mapping[str, Any]) -> str | None:
    spec = row.get("spec")
    condition = row.get("condition")
    if not isinstance(spec, Mapping) or not isinstance(condition, Mapping):
        return None
    identity = {
        "example_id": row.get("example_id"),
        "model_name": spec.get("model_name"),
        "variant_id": condition.get("variant_id"),
        "seed": row.get("seed"),
        "prompt_hash": row.get("prompt_hash"),
        "pair_key": row.get("pair_key"),
    }
    return stable_hash(identity)


def _validate_record(
    row: Mapping[str, Any],
    *,
    planned: PlannedCondition,
    expected_example: JudgeExample,
    expected_model_revision: str | None,
    source_csv: Path,
    input_file_hash: str,
    dataset_split: str | None,
    consistency_runs: int,
    consistency_schedule: Literal["all", "extremes"],
    sampling_temperature: float,
    collector: IssueCollector,
    artifact_dir: Path,
    stage: StageName,
) -> None:
    record_id = row.get("record_id")
    try:
        RunRecord.model_validate(row)
    except ValidationError as exc:
        collector.add(
            "invalid_run_record_schema",
            f"RunRecord validation failed: {exc.errors(include_url=False)[:2]!r}",
            artifact_dir=artifact_dir,
            stage=stage,
            record_id=record_id,
        )
    _validate_parser_derived_fields(
        row,
        collector=collector,
        artifact_dir=artifact_dir,
        stage=stage,
    )

    expected_links = {
        "pair_key": planned.pair_key,
        "condition_group_id": planned.condition_group_id,
        "ordering_twin_key": planned.ordering_twin_key,
    }
    for field, expected in expected_links.items():
        _check_equal(
            row.get(field),
            expected,
            code="pair_linkage_mismatch",
            field=field,
            collector=collector,
            artifact_dir=artifact_dir,
            stage=stage,
            record_id=record_id,
        )
    condition = row.get("condition")
    condition = condition if isinstance(condition, Mapping) else {}
    for field, expected in _condition_fields(planned).items():
        _check_equal(
            condition.get(field),
            expected,
            code="condition_definition_mismatch",
            field=f"condition.{field}",
            collector=collector,
            artifact_dir=artifact_dir,
            stage=stage,
            record_id=record_id,
        )
    metadata = row.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    _check_equal(
        metadata.get("template_variant_id"),
        planned.condition.variant_id,
        code="condition_definition_mismatch",
        field="metadata.template_variant_id",
        collector=collector,
        artifact_dir=artifact_dir,
        stage=stage,
        record_id=record_id,
    )
    _validate_source_metadata(
        row,
        expected_example=expected_example,
        pair_identity_key=planned.pair_identity_key,
        collector=collector,
        artifact_dir=artifact_dir,
        stage=stage,
    )
    _validate_spec_and_consistency(
        row,
        expected_condition=planned,
        source_csv=source_csv,
        input_file_hash=input_file_hash,
        model_name=planned.model_name,
        model_revision=expected_model_revision,
        dataset_split=dataset_split,
        consistency_runs=consistency_runs,
        consistency_schedule=consistency_schedule,
        sampling_temperature=sampling_temperature,
        collector=collector,
        artifact_dir=artifact_dir,
        stage=stage,
    )
    _validate_probabilities(
        row.get("raw_prompt_logprobs")
        if isinstance(row.get("raw_prompt_logprobs"), Mapping)
        else None,
        collector=collector,
        artifact_dir=artifact_dir,
        stage=stage,
        record_id=record_id,
    )
    _check_equal(
        record_id,
        _expected_record_id(row),
        code="record_id_mismatch",
        field="record_id",
        collector=collector,
        artifact_dir=artifact_dir,
        stage=stage,
        record_id=record_id,
    )


def _stage_b_grid(
    stage_a_expected: Mapping[tuple[str, str], PlannedCondition],
) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    family_doses = (
        ("bandwagon", BANDWAGON_DOSES),
        ("authority", AUTHORITY_DOSES),
    )
    for planned in stage_a_expected.values():
        ordering = str(planned.condition.ordering)
        for direction in ("congruent", "incongruent"):
            for family, doses in family_doses:
                for dose in doses:
                    variant = format_variant_id(
                        family=family,
                        direction=direction,
                        dose=dose,
                        ordering=ordering,
                    )
                    keys.add((planned.pair_key, variant))
    return keys


def _stage_b_expectations(
    stage_a_expected: Mapping[tuple[str, str], PlannedCondition],
    stage_a_rows: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    collector: IssueCollector,
    artifact_dir: Path,
) -> dict[tuple[str, str], PlannedCondition]:
    clean_summaries: list[CleanPairSummary] = []
    for key, planned in stage_a_expected.items():
        row = stage_a_rows.get(key)
        if row is None:
            continue
        verdict = row.get("verdict")
        if verdict not in {"A", "B", "tie"}:
            collector.add(
                "invalid_clean_verdict",
                f"clean verdict is {verdict!r}; expected A, B, or tie",
                artifact_dir=artifact_dir,
                stage="stage_a",
                record_id=row.get("record_id"),
            )
            continue
        expected_human = planned.condition.metadata.get("human_winner")
        clean_summaries.append(
            CleanPairSummary(
                pair_identity_key=planned.pair_identity_key,
                pair_key=planned.pair_key,
                ordering=str(planned.condition.ordering),
                ordering_twin_key=planned.ordering_twin_key,
                model_name=planned.model_name,
                input_file_hash=planned.input_file_hash,
                clean_record_id=str(row.get("record_id") or ""),
                clean_verdict=str(verdict),
                human_winner=str(expected_human),
            )
        )
    plan = generate_stage_b_conditions(clean_summaries)
    fatal = [issue for issue in plan.issues if issue.code != "clean_and_human_tie"]
    if fatal:
        collector.add(
            "stage_b_planning_failed",
            f"validated clean rows produced {len(fatal)} fatal planning issue(s)",
            artifact_dir=artifact_dir,
            stage="stage_b",
        )
    return {
        (planned.pair_key, planned.condition.variant_id): planned
        for planned in plan.conditions
    }


def _expected_example_for_planned(
    planned: PlannedCondition,
    *,
    source_by_identity: Mapping[str, SourcePair],
) -> JudgeExample:
    return source_by_identity[
        planned.pair_identity_key
    ].examples_by_ordering[str(planned.condition.ordering)]


def _infer_revision(
    rows_by_stage: Mapping[StageName, Sequence[Mapping[str, Any]]],
    *,
    model_name: str | None,
    collector: IssueCollector,
    artifact_dir: Path,
) -> str | None:
    revisions = {
        spec.get("model_revision")
        for rows in rows_by_stage.values()
        for row in rows
        if isinstance((spec := row.get("spec")), Mapping)
    }
    if len(revisions) != 1:
        collector.add(
            "model_revision_mismatch",
            f"raw records contain multiple model revisions: {sorted(map(str, revisions))!r}",
            artifact_dir=artifact_dir,
        )
        return None
    revision = next(iter(revisions), None)
    if revision in (None, ""):
        collector.add(
            "missing_model_revision",
            "raw records do not identify an immutable model revision",
            artifact_dir=artifact_dir,
        )
        return None
    if model_name is not None:
        try:
            registered = get_model_profile(model_name)
        except KeyError:
            registered = None
        if (
            registered is not None
            and registered.revision is not None
            and revision != registered.revision
        ):
            collector.add(
                "model_revision_mismatch",
                f"artifact revision {revision!r} differs from registry revision "
                f"{registered.revision!r}",
                artifact_dir=artifact_dir,
            )
    return str(revision)


def _infer_dataset_split(
    rows_by_stage: Mapping[StageName, Sequence[Mapping[str, Any]]],
    *,
    requested: str | None,
    collector: IssueCollector,
    artifact_dir: Path,
) -> str | None:
    observed = {
        str(spec["dataset_split"])
        for rows in rows_by_stage.values()
        for row in rows
        if isinstance((spec := row.get("spec")), Mapping)
        and spec.get("dataset_split") not in (None, "")
    }
    if len(observed) != 1:
        collector.add(
            "dataset_split_mismatch",
            f"expected one dataset split per directory; observed {sorted(observed)!r}",
            artifact_dir=artifact_dir,
        )
        return requested
    inferred = next(iter(observed))
    if requested is not None and requested != inferred:
        collector.add(
            "dataset_split_mismatch",
            f"artifact split is {inferred!r}; requested {requested!r}",
            artifact_dir=artifact_dir,
        )
        return requested
    return inferred


def _normalized_grid(
    rows: Iterable[Mapping[str, Any]],
) -> frozenset[tuple[str, str, str]]:
    cells: set[tuple[str, str, str]] = set()
    for row in rows:
        metadata = row.get("metadata")
        condition = row.get("condition")
        if not isinstance(metadata, Mapping) or not isinstance(condition, Mapping):
            continue
        identity = metadata.get("pair_identity_key")
        ordering = condition.get("ordering")
        variant = condition.get("variant_id")
        if all(isinstance(value, str) and value for value in (identity, ordering, variant)):
            cells.add((str(identity), str(ordering), str(variant)))
    return frozenset(cells)


def _validate_artifact_dir(
    artifact_dir: Path,
    *,
    source_csv: Path,
    input_file_hash: str,
    source_pairs: Sequence[SourcePair],
    consistency_runs: int,
    consistency_schedule: Literal["all", "extremes"],
    sampling_temperature: float,
    dataset_split: str | None,
    min_verbalized_availability: float,
    collector: IssueCollector,
) -> ArtifactResult:
    raw_rows = {
        stage: _read_jsonl(
            artifact_dir / filename,
            collector=collector,
            artifact_dir=artifact_dir,
            stage=stage,
        )
        for stage, filename in RAW_FILENAMES.items()
    }
    score_rows = {
        stage: _read_jsonl(
            artifact_dir / SCORE_FILENAMES[stage],
            collector=collector,
            artifact_dir=artifact_dir,
            stage=stage,
        )
        for stage in ("stage_a", "stage_b")
    }
    pair_rows = {
        stage: _read_jsonl(
            artifact_dir / PAIR_SUMMARY_FILENAMES[stage],
            collector=collector,
            artifact_dir=artifact_dir,
            stage=stage,
        )
        for stage in ("stage_a", "stage_b")
    }
    stage_summaries = {
        stage: _read_json_object(
            artifact_dir / STAGE_SUMMARY_FILENAMES[stage],
            collector=collector,
            artifact_dir=artifact_dir,
            stage=stage,
        )
        for stage in ("stage_a", "stage_b")
    }
    _validate_global_uniqueness(
        raw_rows,
        raw=True,
        collector=collector,
        artifact_dir=artifact_dir,
    )
    _validate_global_uniqueness(
        score_rows,
        raw=False,
        collector=collector,
        artifact_dir=artifact_dir,
    )
    observed_models = {
        str(spec["model_name"])
        for rows in raw_rows.values()
        for row in rows
        if isinstance((spec := row.get("spec")), Mapping)
        and spec.get("model_name") not in (None, "")
    }
    model_name = _model_name(raw_rows)
    if model_name is None:
        collector.add(
            "model_name_mismatch",
            f"expected one model per directory; observed {sorted(observed_models)!r}",
            artifact_dir=artifact_dir,
        )
        model_name = sorted(observed_models)[0] if observed_models else ""
    revision = _infer_revision(
        raw_rows,
        model_name=model_name or None,
        collector=collector,
        artifact_dir=artifact_dir,
    )
    effective_dataset_split = _infer_dataset_split(
        raw_rows,
        requested=dataset_split,
        collector=collector,
        artifact_dir=artifact_dir,
    )
    verbalized_availability = {
        stage: _validate_verbalized_availability(
            raw_rows[stage],
            minimum=min_verbalized_availability,
            collector=collector,
            artifact_dir=artifact_dir,
            stage=stage,
        )
        for stage in ("stage_a", "stage_b")
    }
    if not model_name:
        for stage in ("stage_a", "stage_b"):
            _validate_flat_scores(
                raw_rows[stage],
                score_rows[stage],
                collector=collector,
                artifact_dir=artifact_dir,
                stage=stage,
            )
            _validate_pair_summaries(
                raw_rows[stage],
                pair_rows[stage],
                collector=collector,
                artifact_dir=artifact_dir,
                stage=stage,
            )
            _validate_stage_summary(
                stage_summaries[stage],
                expected_records=len(raw_rows[stage]),
                collector=collector,
                artifact_dir=artifact_dir,
                stage=stage,
            )
            _validate_scheduler_provenance(
                raw_rows[stage],
                score_rows[stage],
                pair_rows[stage],
                stage_summaries[stage],
                collector=collector,
                artifact_dir=artifact_dir,
                stage=stage,
            )
            _validate_logprobs_mode_provenance(
                raw_rows[stage],
                score_rows[stage],
                pair_rows[stage],
                stage_summaries[stage],
                collector=collector,
                artifact_dir=artifact_dir,
                stage=stage,
            )
            _validate_verbalized_status_summary(
                raw_rows[stage],
                stage_summaries[stage],
                collector=collector,
                artifact_dir=artifact_dir,
                stage=stage,
            )
        return ArtifactResult(
            report={
                "artifact_dir": str(artifact_dir),
                "model_name": None,
                "model_revision": revision,
                "dataset_split": effective_dataset_split,
                "verbalized_availability": verbalized_availability,
                "counts": {
                    "source_pairs": len(source_pairs),
                    "stage_a_expected": 2 * len(source_pairs),
                    "stage_a_raw": len(raw_rows["stage_a"]),
                    "stage_a_flat": len(score_rows["stage_a"]),
                    "stage_a_pair_summary": len(pair_rows["stage_a"]),
                    "stage_b_expected": 32 * len(source_pairs),
                    "stage_b_raw": len(raw_rows["stage_b"]),
                    "stage_b_flat": len(score_rows["stage_b"]),
                    "stage_b_pair_summary": len(pair_rows["stage_b"]),
                },
            },
            grids={
                stage: _normalized_grid(raw_rows[stage])
                for stage in ("stage_a", "stage_b")
            },
        )

    expected_a, examples_a = _stage_a_expectations(
        source_pairs,
        model_name=model_name,
    )
    stage_a_index = _condition_index(
        raw_rows["stage_a"],
        raw=True,
        collector=collector,
        artifact_dir=artifact_dir,
        stage="stage_a",
    )
    _report_grid_difference(
        actual=set(stage_a_index),
        expected=set(expected_a),
        collector=collector,
        artifact_dir=artifact_dir,
        stage="stage_a",
    )
    if len(raw_rows["stage_a"]) != len(expected_a):
        collector.add(
            "record_count_mismatch",
            f"Stage A has {len(raw_rows['stage_a'])} raw records; "
            f"expected {len(expected_a)}",
            artifact_dir=artifact_dir,
            stage="stage_a",
        )
    for key in sorted(set(expected_a) & set(stage_a_index)):
        _validate_record(
            stage_a_index[key],
            planned=expected_a[key],
            expected_example=examples_a[key],
            expected_model_revision=revision,
            source_csv=source_csv,
            input_file_hash=input_file_hash,
            dataset_split=effective_dataset_split,
            consistency_runs=consistency_runs,
            consistency_schedule=consistency_schedule,
            sampling_temperature=sampling_temperature,
            collector=collector,
            artifact_dir=artifact_dir,
            stage="stage_a",
        )

    expected_b_grid = _stage_b_grid(expected_a)
    expected_b = _stage_b_expectations(
        expected_a,
        stage_a_index,
        collector=collector,
        artifact_dir=artifact_dir,
    )
    stage_b_index = _condition_index(
        raw_rows["stage_b"],
        raw=True,
        collector=collector,
        artifact_dir=artifact_dir,
        stage="stage_b",
    )
    _report_grid_difference(
        actual=set(stage_b_index),
        expected=expected_b_grid,
        collector=collector,
        artifact_dir=artifact_dir,
        stage="stage_b",
    )
    if len(raw_rows["stage_b"]) != len(expected_b_grid):
        collector.add(
            "record_count_mismatch",
            f"Stage B has {len(raw_rows['stage_b'])} raw records; "
            f"expected {len(expected_b_grid)}",
            artifact_dir=artifact_dir,
            stage="stage_b",
        )
    identity_counts = Counter(
        str(metadata.get("pair_identity_key"))
        for row in raw_rows["stage_b"]
        if isinstance((metadata := row.get("metadata")), Mapping)
    )
    bad_identity_counts = {
        identity: count
        for identity, count in identity_counts.items()
        if count != 32
    }
    expected_identities = {
        planned.pair_identity_key for planned in expected_a.values()
    }
    for missing_identity in expected_identities - set(identity_counts):
        bad_identity_counts[missing_identity] = 0
    if bad_identity_counts:
        collector.add(
            "stage_b_pair_grid_mismatch",
            f"{len(bad_identity_counts)} source pair(s) do not have exactly 32 "
            f"Stage B records; sample={list(sorted(bad_identity_counts.items()))[:3]!r}",
            artifact_dir=artifact_dir,
            stage="stage_b",
        )

    source_by_identity = {
        source.pair_identity_key: source
        for source in source_pairs
    }
    for key in sorted(set(expected_b) & set(stage_b_index)):
        planned = expected_b[key]
        row = stage_b_index[key]
        _validate_record(
            row,
            planned=planned,
            expected_example=_expected_example_for_planned(
                planned,
                source_by_identity=source_by_identity,
            ),
            expected_model_revision=revision,
            source_csv=source_csv,
            input_file_hash=input_file_hash,
            dataset_split=effective_dataset_split,
            consistency_runs=consistency_runs,
            consistency_schedule=consistency_schedule,
            sampling_temperature=sampling_temperature,
            collector=collector,
            artifact_dir=artifact_dir,
            stage="stage_b",
        )
        clean_record_id = planned.condition.clean_record_id
        actual_clean_id = (
            row.get("condition", {}).get("clean_record_id")
            if isinstance(row.get("condition"), Mapping)
            else None
        )
        if actual_clean_id != clean_record_id:
            collector.add(
                "stage_b_clean_link_mismatch",
                f"clean_record_id is {actual_clean_id!r}; expected {clean_record_id!r}",
                artifact_dir=artifact_dir,
                stage="stage_b",
                record_id=row.get("record_id"),
            )

    for stage in ("stage_a", "stage_b"):
        _validate_flat_scores(
            raw_rows[stage],
            score_rows[stage],
            collector=collector,
            artifact_dir=artifact_dir,
            stage=stage,
        )
        if len(score_rows[stage]) != len(raw_rows[stage]):
            collector.add(
                "flat_score_count_mismatch",
                f"{stage} has {len(score_rows[stage])} flat rows and "
                f"{len(raw_rows[stage])} raw rows",
                artifact_dir=artifact_dir,
                stage=stage,
            )
        _validate_pair_summaries(
            raw_rows[stage],
            pair_rows[stage],
            collector=collector,
            artifact_dir=artifact_dir,
            stage=stage,
        )
        _validate_stage_summary(
            stage_summaries[stage],
            expected_records=len(raw_rows[stage]),
            collector=collector,
            artifact_dir=artifact_dir,
            stage=stage,
        )
        _validate_scheduler_provenance(
            raw_rows[stage],
            score_rows[stage],
            pair_rows[stage],
            stage_summaries[stage],
            collector=collector,
            artifact_dir=artifact_dir,
            stage=stage,
        )
        _validate_logprobs_mode_provenance(
            raw_rows[stage],
            score_rows[stage],
            pair_rows[stage],
            stage_summaries[stage],
            collector=collector,
            artifact_dir=artifact_dir,
            stage=stage,
        )
        _validate_verbalized_status_summary(
            raw_rows[stage],
            stage_summaries[stage],
            collector=collector,
            artifact_dir=artifact_dir,
            stage=stage,
        )

    directory_report = {
        "artifact_dir": str(artifact_dir),
        "model_name": model_name or None,
        "model_revision": revision,
        "dataset_split": effective_dataset_split,
        "verbalized_availability": verbalized_availability,
        "counts": {
            "source_pairs": len(source_pairs),
            "stage_a_expected": len(expected_a),
            "stage_a_raw": len(raw_rows["stage_a"]),
            "stage_a_flat": len(score_rows["stage_a"]),
            "stage_a_pair_summary": len(pair_rows["stage_a"]),
            "stage_b_expected": len(expected_b_grid),
            "stage_b_raw": len(raw_rows["stage_b"]),
            "stage_b_flat": len(score_rows["stage_b"]),
            "stage_b_pair_summary": len(pair_rows["stage_b"]),
        },
    }
    return ArtifactResult(
        report=directory_report,
        grids={
            stage: _normalized_grid(raw_rows[stage])
            for stage in ("stage_a", "stage_b")
        },
    )


def validate_artifact_directories(
    *,
    source_csv: Path,
    artifact_dirs: Sequence[Path],
    consistency_runs: int,
    consistency_schedule: Literal["all", "extremes"],
    sampling_temperature: float = 0.7,
    dataset_split: str | None = None,
    limit: int | None = None,
    max_reported_errors: int = 50,
    min_verbalized_availability: float = 0.99,
) -> dict[str, Any]:
    if not source_csv.is_file():
        raise FileNotFoundError(source_csv)
    if not artifact_dirs:
        raise ValueError("at least one artifact directory is required")
    if consistency_runs < 0:
        raise ValueError("consistency_runs must be nonnegative")
    if consistency_schedule not in {"all", "extremes"}:
        raise ValueError("consistency_schedule must be 'all' or 'extremes'")
    if not math.isfinite(sampling_temperature) or sampling_temperature < 0:
        raise ValueError("sampling_temperature must be finite and nonnegative")
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")
    if (
        not math.isfinite(min_verbalized_availability)
        or not 0.0 <= min_verbalized_availability <= 1.0
    ):
        raise ValueError(
            "min_verbalized_availability must be finite and in [0, 1]"
        )

    collector = IssueCollector(max_reported=max_reported_errors)
    source_hash = file_sha256(source_csv)
    source_pairs = _source_pairs(
        source_csv,
        input_file_hash=source_hash,
        limit=limit,
    )
    results = [
        _validate_artifact_dir(
            directory,
            source_csv=source_csv,
            input_file_hash=source_hash,
            source_pairs=source_pairs,
            consistency_runs=consistency_runs,
            consistency_schedule=consistency_schedule,
            sampling_temperature=sampling_temperature,
            dataset_split=dataset_split,
            min_verbalized_availability=min_verbalized_availability,
            collector=collector,
        )
        for directory in artifact_dirs
    ]

    if len(results) > 1:
        reference = results[0]
        reference_dir = artifact_dirs[0]
        for directory, result in zip(artifact_dirs[1:], results[1:], strict=True):
            for stage in ("stage_a", "stage_b"):
                if result.grids[stage] == reference.grids[stage]:
                    continue
                missing = reference.grids[stage] - result.grids[stage]
                extra = result.grids[stage] - reference.grids[stage]
                collector.add(
                    "cross_model_grid_mismatch",
                    f"grid differs from {reference_dir}: missing={len(missing)}, "
                    f"extra={len(extra)}",
                    artifact_dir=directory,
                    stage=stage,
                )

    issue_summary = collector.summary()
    return {
        "passed": collector.error_count == 0,
        "source": {
            "csv": str(source_csv),
            "input_file_hash": source_hash,
            "usable_pairs": len(source_pairs),
            "limit": limit,
        },
        "design": {
            "stage_a_records_per_pair": 2,
            "stage_b_records_per_pair": 32,
            "consistency_runs": consistency_runs,
            "consistency_schedule": consistency_schedule,
            "sampling_temperature": sampling_temperature,
            "dataset_split": dataset_split,
            "judge_output_parser_version": JUDGE_OUTPUT_PARSER_VERSION,
            "min_verbalized_availability": min_verbalized_availability,
        },
        "artifacts": [result.report for result in results],
        **issue_summary,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate Silent Bias Stage A/B raw and flat artifacts against the "
            "source CSV and experimental grid."
        )
    )
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        action="append",
        required=True,
        dest="artifact_dirs",
        help="Model artifact directory; repeat to check cross-model grid identity.",
    )
    parser.add_argument("--consistency-runs", type=int, required=True)
    parser.add_argument(
        "--consistency-schedule",
        choices=("all", "extremes"),
        required=True,
    )
    parser.add_argument("--sampling-temperature", type=float, default=0.7)
    parser.add_argument(
        "--dataset-split",
        help="Optional expected ExperimentSpec.dataset_split (for example pilot or full).",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-reported-errors", type=int, default=50)
    parser.add_argument(
        "--min-verbalized-availability",
        type=float,
        default=0.99,
        help=(
            "Minimum parsed/requested verbalized-confidence fraction required "
            "per stage (default: 0.99)."
        ),
    )
    parser.add_argument("--report-path", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = validate_artifact_directories(
            source_csv=args.source_csv,
            artifact_dirs=args.artifact_dirs,
            consistency_runs=args.consistency_runs,
            consistency_schedule=args.consistency_schedule,
            sampling_temperature=args.sampling_temperature,
            dataset_split=args.dataset_split,
            limit=args.limit,
            max_reported_errors=args.max_reported_errors,
            min_verbalized_availability=args.min_verbalized_availability,
        )
    except (FileNotFoundError, TypeError, ValueError) as exc:
        report = {
            "passed": False,
            "error_count": 1,
            "error_counts_by_code": {"validation_setup_failed": 1},
            "errors": [
                {
                    "code": "validation_setup_failed",
                    "message": str(exc),
                }
            ],
            "errors_truncated": False,
        }
    serialized = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True)
    print(serialized)
    if args.report_path is not None:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(f"{serialized}\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
