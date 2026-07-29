from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any, Literal

from biases.position_bias import (
    JUDGE_OUTPUT_PARSER_VERSION,
    VLLMJudge,
    _compute_consistency,
    parse_verbalized_output as parse_strict_verbalized_output,
    verbalized_parse_status as classify_verbalized_parse_status,
)
from biases.schemas import (
    ConsistencyMetrics,
    LogitMetrics,
    RunRecord,
    VerbalizedMetrics,
    VerdictLabel,
)


_LABELS = ("A", "B", "tie")
_VERDICTS = {
    "A": VerdictLabel.A,
    "B": VerdictLabel.B,
    "tie": VerdictLabel.TIE,
}


class ParserIntegrityError(ValueError):
    """A stored judge output cannot satisfy the current strict parser contract."""


@dataclass(frozen=True, slots=True)
class DerivedParserFields:
    verdict: VerdictLabel
    logit: LogitMetrics
    verbalized: VerbalizedMetrics
    verbalized_parse_status: Literal[
        "parsed",
        "missing",
        "unparseable",
        "not_requested",
    ]
    consistency: ConsistencyMetrics | None


def validated_label_probabilities(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(_LABELS):
        raise ParserIntegrityError(
            "raw_prompt_logprobs must have exactly A/B/tie support"
        )
    probabilities: dict[str, float] = {}
    for label in _LABELS:
        raw_probability = value[label]
        if (
            not isinstance(raw_probability, (int, float))
            or isinstance(raw_probability, bool)
            or not math.isfinite(float(raw_probability))
            or not 0.0 <= float(raw_probability) <= 1.0
        ):
            raise ParserIntegrityError(
                "raw_prompt_logprobs must contain finite probabilities in [0, 1]"
            )
        probabilities[label] = float(raw_probability)
    if not math.isclose(
        sum(probabilities.values()),
        1.0,
        rel_tol=1e-7,
        abs_tol=1e-7,
    ):
        raise ParserIntegrityError("raw_prompt_logprobs do not sum to one")
    return probabilities


def parse_deterministic_output(
    raw_output: object,
    probabilities: Mapping[str, float],
) -> VerdictLabel:
    if not isinstance(raw_output, str):
        raise ParserIntegrityError("raw_output must be a string")
    parsed = VLLMJudge._parse_verdict_text(raw_output)
    if parsed is None:
        raise ParserIntegrityError(
            f"raw_output is unparseable or ambiguous under "
            f"{JUDGE_OUTPUT_PARSER_VERSION}: {raw_output!r}"
        )
    probability_label = max(_LABELS, key=probabilities.__getitem__)
    probability_verdict = _VERDICTS[probability_label]
    if parsed != probability_verdict:
        raise ParserIntegrityError(
            f"strict raw verdict {parsed.value!r} disagrees with aggregated "
            f"probability MAP {probability_verdict.value!r}"
        )
    return parsed


def parse_verbalized_output(raw_output: object) -> tuple[VerdictLabel, float]:
    if not isinstance(raw_output, str):
        raise ParserIntegrityError("verbalized_raw_output must be a string")
    verdict, confidence = parse_strict_verbalized_output(raw_output)
    if verdict is None or confidence is None:
        raise ParserIntegrityError(
            f"verbalized_raw_output is unparseable or ambiguous under "
            f"{JUDGE_OUTPUT_PARSER_VERSION}: {raw_output!r}"
        )
    return verdict, confidence


def _verdicts_from_counts(
    value: object,
    *,
    expected_run_count: int,
) -> list[VerdictLabel]:
    if not isinstance(value, Mapping):
        raise ParserIntegrityError("consistency verdict_counts must be an object")
    if not set(value).issubset(_VERDICTS):
        raise ParserIntegrityError(
            "consistency verdict_counts may only contain A/B/tie"
        )
    verdicts: list[VerdictLabel] = []
    for label in _LABELS:
        count = value.get(label, 0)
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ParserIntegrityError(
                "consistency verdict_counts must contain nonnegative integers"
            )
        verdicts.extend([_VERDICTS[label]] * count)
    if len(verdicts) != expected_run_count:
        raise ParserIntegrityError(
            "consistency verdict_counts do not sum to spec.consistency_runs"
        )
    return verdicts


def recompute_consistency(
    value: object,
    *,
    expected_run_count: int,
    anchor: VerdictLabel,
) -> ConsistencyMetrics | None:
    if expected_run_count == 0:
        if value is not None:
            raise ParserIntegrityError(
                "consistency metrics are present when spec.consistency_runs is zero"
            )
        return None
    if not isinstance(value, Mapping):
        raise ParserIntegrityError("scheduled consistency metrics are missing")
    run_count = value.get("run_count")
    if run_count != expected_run_count:
        raise ParserIntegrityError(
            f"consistency run_count is {run_count!r}; "
            f"expected {expected_run_count}"
        )
    verdicts = _verdicts_from_counts(
        value.get("verdict_counts"),
        expected_run_count=expected_run_count,
    )
    return _compute_consistency(verdicts, anchor=anchor)


def derive_parser_fields(row: Mapping[str, Any]) -> DerivedParserFields:
    probabilities = validated_label_probabilities(row.get("raw_prompt_logprobs"))
    verdict = parse_deterministic_output(row.get("raw_output"), probabilities)
    logit = LogitMetrics.from_probs(probabilities)

    spec = row.get("spec")
    metadata = row.get("metadata")
    uncertainty = row.get("uncertainty")
    if not isinstance(spec, Mapping):
        raise ParserIntegrityError("record has no spec object")
    if not isinstance(metadata, Mapping):
        raise ParserIntegrityError("record has no metadata object")
    if not isinstance(uncertainty, Mapping):
        raise ParserIntegrityError("record has no uncertainty object")

    methods_value = spec.get("uncertainty_methods")
    methods = (
        {str(method) for method in methods_value}
        if isinstance(methods_value, list)
        else set()
    )
    verbalized_raw_output = metadata.get("verbalized_raw_output")
    verbalized_parse_status = classify_verbalized_parse_status(
        uncertainty_methods=sorted(methods),
        raw_output=verbalized_raw_output,
    )
    if verbalized_parse_status == "parsed":
        verbalized_verdict, confidence = parse_verbalized_output(
            verbalized_raw_output
        )
        verbalized = VerbalizedMetrics.from_confidence(
            confidence,
            verdict=verbalized_verdict,
        )
    else:
        verbalized = VerbalizedMetrics()

    run_count = spec.get("consistency_runs")
    if not isinstance(run_count, int) or isinstance(run_count, bool) or run_count < 0:
        raise ParserIntegrityError(
            "spec.consistency_runs must be a nonnegative integer"
        )
    consistency = recompute_consistency(
        uncertainty.get("consistency"),
        expected_run_count=run_count,
        anchor=verdict,
    )
    return DerivedParserFields(
        verdict=verdict,
        logit=logit,
        verbalized=verbalized,
        verbalized_parse_status=verbalized_parse_status,
        consistency=consistency,
    )


def migrate_record_to_current_parser(
    row: Mapping[str, Any],
    *,
    require_stored_verdict_match: bool = True,
) -> dict[str, Any]:
    derived = derive_parser_fields(row)
    if require_stored_verdict_match and row.get("verdict") != derived.verdict.value:
        raise ParserIntegrityError(
            f"stored verdict {row.get('verdict')!r} disagrees with strict "
            f"verdict {derived.verdict.value!r}"
        )

    migrated = deepcopy(dict(row))
    migrated["verdict"] = derived.verdict.value
    uncertainty = migrated.setdefault("uncertainty", {})
    if not isinstance(uncertainty, dict):
        raise ParserIntegrityError("record uncertainty must be an object")
    uncertainty["logit"] = derived.logit.model_dump(mode="json")
    uncertainty["verbalized"] = derived.verbalized.model_dump(mode="json")
    uncertainty["consistency"] = (
        derived.consistency.model_dump(mode="json")
        if derived.consistency is not None
        else None
    )
    metadata = migrated.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        raise ParserIntegrityError("record metadata must be an object")
    metadata["judge_output_parser_version"] = JUDGE_OUTPUT_PARSER_VERSION
    metadata["verbalized_verdict"] = (
        derived.verbalized.verdict.value
        if derived.verbalized.verdict is not None
        else None
    )
    metadata["verbalized_parse_status"] = derived.verbalized_parse_status
    RunRecord.model_validate(migrated)
    return migrated


__all__ = [
    "DerivedParserFields",
    "ParserIntegrityError",
    "derive_parser_fields",
    "migrate_record_to_current_parser",
    "parse_deterministic_output",
    "parse_verbalized_output",
    "recompute_consistency",
    "validated_label_probabilities",
]
