from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from biases.analysis.records import (
    ConditionRecord,
    PairedCondition,
    pair_clean_and_cued,
)
from biases.analysis.resampling import cluster_resamples, percentile
from biases.dataset_splits import ROUTING_SPLITS, routing_assignment_sha256
from biases.schemas import CueReferenceKind
from biases.social_cue_prompts import AUTHORITY_DOSES, BANDWAGON_DOSES


BIAS_FAMILIES: tuple[str, ...] = ("authority", "bandwagon")
BIAS_DIRECTIONS: tuple[str, ...] = ("congruent", "incongruent")
PRIMARY_REFERENCE_KIND = CueReferenceKind.MODEL_CLEAN_VERDICT.value
FALLBACK_REFERENCE_KINDS = frozenset(
    {
        CueReferenceKind.HUMAN_LABEL_FALLBACK.value,
        CueReferenceKind.DETERMINISTIC_FALLBACK.value,
    }
)
VALID_REFERENCE_KINDS = frozenset({PRIMARY_REFERENCE_KIND, *FALLBACK_REFERENCE_KINDS})
GroupKey = tuple[str, str, str, str, float]


@dataclass(frozen=True, slots=True)
class PredictorSpec:
    """One uncertainty predictor, evaluated without any composite score."""

    name: str
    higher_is_more_confident: bool = True

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("predictor name must not be empty")

    def orient(self, score: float) -> float:
        return score if self.higher_is_more_confident else -score


@dataclass(frozen=True, slots=True)
class QuestionSplit:
    calibration_question_ids: tuple[str, ...]
    test_question_ids: tuple[str, ...]
    raw_assignment_sha256: str
    eligible_assignment_sha256: str
    raw_question_count: int
    eligible_question_count: int

    def assignment(self, question_id: str) -> str:
        if question_id in self.calibration_question_ids:
            return "calibration"
        if question_id in self.test_question_ids:
            return "test"
        raise ValueError(f"question {question_id!r} is absent from the frozen split")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ThresholdRule:
    predictor: str
    target_risk: float
    higher_is_more_confident: bool
    operator: str
    threshold: float | None
    oriented_threshold: float | None
    calibration_population_n: int
    calibration_available_n: int
    accepted: int
    errors: int
    coverage_among_available: float
    coverage_among_population: float
    risk: float | None


@dataclass(frozen=True, slots=True)
class MatchedScore:
    pair: PairedCondition
    clean_score: float
    cued_score: float

    @property
    def question_id(self) -> str:
        return self.pair.clean.question_id


@dataclass(frozen=True, slots=True)
class MetricInterval:
    estimate: float | None
    low: float | None
    high: float | None
    finite_resamples: int
    n_resamples: int


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_shared_question_universe(
    clean_records: Sequence[ConditionRecord],
) -> tuple[str, ...]:
    """Require every model/order cell to expose the same question universe."""

    by_cell: dict[tuple[str, str], set[str]] = defaultdict(set)
    for record in clean_records:
        if not record.model_name:
            raise ValueError(f"clean record {record.record_id!r} has no model name")
        if not record.ordering:
            raise ValueError(f"clean record {record.record_id!r} has no ordering")
        by_cell[(record.model_name, record.ordering)].add(record.question_id)
    if not by_cell:
        raise ValueError("no clean records")
    reference_cell = min(by_cell)
    reference = by_cell[reference_cell]
    for cell, observed in sorted(by_cell.items()):
        if observed != reference:
            missing = sorted(reference - observed)
            extra = sorted(observed - reference)
            raise ValueError(
                "model/order cells do not share one question universe: "
                f"{cell!r} missing={missing!r} extra={extra!r}"
            )
    return tuple(sorted(reference))


def validate_single_model_identity(
    clean_records: Sequence[ConditionRecord],
    cued_records: Sequence[ConditionRecord] = (),
    *,
    expected_model_name: str | None = None,
    expected_model_revision: str | None = None,
) -> tuple[str, str]:
    """Require one model and one immutable Hugging Face commit per analysis."""

    if expected_model_name is not None and not expected_model_name.strip():
        raise ValueError("expected model name must not be blank")
    if expected_model_revision is not None and not re.fullmatch(
        r"[0-9a-f]{40}", expected_model_revision
    ):
        raise ValueError(
            "expected model revision must be a lowercase 40-hex commit"
        )
    records = (*clean_records, *cued_records)
    if not records:
        raise ValueError("no records")
    identities: set[tuple[str, str]] = set()
    for record in records:
        model_name = record.model_name.strip()
        model_revision = str(record.model_revision or "").strip()
        if not model_name:
            raise ValueError(f"record {record.record_id!r} has no model name")
        if not re.fullmatch(r"[0-9a-f]{40}", model_revision):
            raise ValueError(
                f"record {record.record_id!r} does not pin a lowercase 40-hex "
                "model revision"
            )
        identities.add((model_name, model_revision))
    if len(identities) != 1:
        raise ValueError(
            "analysis records span multiple model identities: "
            + repr(sorted(identities))
        )
    identity = next(iter(identities))
    if expected_model_name is not None and identity[0] != expected_model_name:
        raise ValueError(
            "analysis model name mismatch: "
            f"expected {expected_model_name!r}, observed {identity[0]!r}"
        )
    if (
        expected_model_revision is not None
        and identity[1] != expected_model_revision
    ):
        raise ValueError(
            "analysis model revision mismatch: "
            f"expected {expected_model_revision!r}, observed {identity[1]!r}"
        )
    return identity


def _normalize_question_assignments(
    assignments: Mapping[str, str],
    *,
    name: str,
) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for question_id, value in assignments.items():
        normalized_question_id = str(question_id).strip()
        normalized_split = str(value).strip().lower()
        if not normalized_question_id:
            raise ValueError(f"{name} contains an empty question ID")
        if normalized_split not in ROUTING_SPLITS:
            raise ValueError(
                f"{name} question {normalized_question_id!r} has invalid "
                f"routing_split {value!r}"
            )
        previous = normalized.setdefault(normalized_question_id, normalized_split)
        if previous != normalized_split:
            raise ValueError(
                f"{name} question {normalized_question_id!r} occurs in both routes"
            )
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


def _assignment_sha256(assignment: Mapping[str, str]) -> str:
    return routing_assignment_sha256(
        pd.DataFrame(
            [
                {
                    "question_id": question_id,
                    "routing_split": assignment[question_id],
                }
                for question_id in sorted(assignment)
            ]
        )
    )


def question_split_from_records(
    clean_records: Sequence[ConditionRecord],
    *,
    expected_raw_assignment_sha256: str | None = None,
    expected_eligible_assignment_sha256: str | None = None,
    frozen_raw_question_assignments: Mapping[str, str] | None = None,
    frozen_eligible_question_assignments: Mapping[str, str] | None = None,
) -> QuestionSplit:
    """Validate and consume the frozen routing assignment in clean records."""

    for name, expected in (
        ("expected raw routing assignment SHA-256", expected_raw_assignment_sha256),
        (
            "expected eligible routing assignment SHA-256",
            expected_eligible_assignment_sha256,
        ),
    ):
        if expected is not None and not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"{name} must be exactly 64 lowercase hex digits")
    validate_shared_question_universe(clean_records)
    assignment: dict[str, str] = {}
    for record in clean_records:
        routing_split = str(record.routing_split or "").strip().lower()
        if routing_split not in ROUTING_SPLITS:
            raise ValueError(
                f"record {record.record_id!r} has invalid routing_split "
                f"{record.routing_split!r}"
            )
        previous = assignment.setdefault(record.question_id, routing_split)
        if previous != routing_split:
            raise ValueError(
                f"question {record.question_id!r} occurs in both routing splits"
            )
    if len(assignment) < 2:
        raise ValueError("at least two routed questions are required")
    calibration = tuple(
        sorted(question_id for question_id, value in assignment.items() if value == "calibration")
    )
    test = tuple(
        sorted(question_id for question_id, value in assignment.items() if value == "test")
    )
    if not calibration or not test:
        raise ValueError("routing assignment must contain calibration and test questions")
    observed_eligible_sha256 = _assignment_sha256(assignment)
    if (frozen_raw_question_assignments is None) != (
        frozen_eligible_question_assignments is None
    ):
        raise ValueError(
            "raw and eligible frozen question assignments must be supplied together"
        )
    if frozen_raw_question_assignments is None:
        raw_assignment = dict(assignment)
        eligible_assignment = dict(assignment)
    else:
        assert frozen_eligible_question_assignments is not None
        raw_assignment = _normalize_question_assignments(
            frozen_raw_question_assignments,
            name="frozen raw routing",
        )
        eligible_assignment = _normalize_question_assignments(
            frozen_eligible_question_assignments,
            name="frozen eligible routing",
        )
        missing_from_raw = sorted(set(eligible_assignment) - set(raw_assignment))
        raw_mismatches = sorted(
            question_id
            for question_id, routing_split in eligible_assignment.items()
            if raw_assignment.get(question_id) != routing_split
        )
        if missing_from_raw or raw_mismatches:
            raise ValueError(
                "eligible frozen routing is not an exact subset of raw routing: "
                f"missing={missing_from_raw[:10]!r}, "
                f"mismatched={raw_mismatches[:10]!r}"
            )
        missing_from_clean = sorted(set(eligible_assignment) - set(assignment))
        unexpected_clean = sorted(set(assignment) - set(eligible_assignment))
        clean_mismatches = sorted(
            question_id
            for question_id, routing_split in assignment.items()
            if eligible_assignment.get(question_id) != routing_split
        )
        if missing_from_clean or unexpected_clean or clean_mismatches:
            raise ValueError(
                "clean-record routing does not exactly match the frozen eligible "
                f"routing: missing={missing_from_clean[:10]!r}, "
                f"unexpected={unexpected_clean[:10]!r}, "
                f"mismatched={clean_mismatches[:10]!r}"
            )
    raw_sha256 = _assignment_sha256(raw_assignment)
    eligible_sha256 = _assignment_sha256(eligible_assignment)
    if eligible_sha256 != observed_eligible_sha256:
        raise ValueError(
            "clean-record eligible routing SHA-256 differs from the frozen "
            "eligible routing"
        )
    if (
        expected_raw_assignment_sha256 is not None
        and raw_sha256 != expected_raw_assignment_sha256
    ):
        raise ValueError(
            "raw routing assignment SHA-256 mismatch: "
            f"expected {expected_raw_assignment_sha256}, observed {raw_sha256}"
        )
    if (
        expected_eligible_assignment_sha256 is not None
        and eligible_sha256 != expected_eligible_assignment_sha256
    ):
        raise ValueError(
            "eligible routing assignment SHA-256 mismatch: "
            f"expected {expected_eligible_assignment_sha256}, "
            f"observed {eligible_sha256}"
        )
    return QuestionSplit(
        calibration_question_ids=calibration,
        test_question_ids=test,
        raw_assignment_sha256=raw_sha256,
        eligible_assignment_sha256=eligible_sha256,
        raw_question_count=len(raw_assignment),
        eligible_question_count=len(eligible_assignment),
    )


def _finite_score(value: Any, *, predictor: str, record_id: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{predictor} score for {record_id!r} is boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{predictor} score for {record_id!r} is not numeric"
        ) from exc
    if not math.isfinite(result):
        raise ValueError(f"{predictor} score for {record_id!r} is not finite")
    return result


def record_score(
    scores_by_record: Mapping[str, Mapping[str, Any]],
    record_id: str,
    predictor: str,
) -> float | None:
    raw = scores_by_record.get(record_id, {}).get(predictor)
    return _finite_score(raw, predictor=predictor, record_id=record_id)


def is_correct(record: ConditionRecord) -> bool:
    if record.human_winner is None:
        raise ValueError(f"record {record.record_id!r} has no human winner")
    return record.verdict == record.human_winner


def _validate_cue_reference_contract(
    clean: ConditionRecord,
    cued: ConditionRecord,
) -> str:
    """Validate the provenance and estimand role of one Stage-B reference."""

    reference_kind = str(cued.reference_kind or "").strip()
    if reference_kind not in VALID_REFERENCE_KINDS:
        raise ValueError(
            f"cued record {cued.record_id!r} must declare one of the exact "
            f"reference_kind values {sorted(VALID_REFERENCE_KINDS)!r}; "
            f"observed {reference_kind or None!r}"
        )
    if reference_kind == PRIMARY_REFERENCE_KIND:
        if clean.clean_tie or cued.clean_tie or clean.verdict not in {"A", "B"}:
            raise ValueError(
                f"cued record {cued.record_id!r} labels a clean tie as "
                f"{PRIMARY_REFERENCE_KIND!r}"
            )
        return reference_kind

    if not clean.clean_tie or not cued.clean_tie or clean.verdict != "tie":
        raise ValueError(
            f"fallback-referenced cued record {cued.record_id!r} is not a "
            "clean-tie robustness row and would enter the primary target-bias "
            "cohort"
        )
    if (
        reference_kind == CueReferenceKind.HUMAN_LABEL_FALLBACK.value
        and clean.human_winner not in {"A", "B"}
    ):
        raise ValueError(
            f"cued record {cued.record_id!r} uses human_label_fallback without "
            "a binary human label"
        )
    if (
        reference_kind == CueReferenceKind.DETERMINISTIC_FALLBACK.value
        and clean.human_winner != "tie"
    ):
        raise ValueError(
            f"cued record {cued.record_id!r} uses deterministic_fallback "
            "outside the clean-and-human-tie robustness stratum"
        )
    return reference_kind


def exact_test_pairs(
    clean_records: Sequence[ConditionRecord],
    cued_records: Sequence[ConditionRecord],
    split: QuestionSplit,
    *,
    exclude_clean_ties: bool = True,
) -> tuple[PairedCondition, ...]:
    """Return strict clean/cued matches from test questions only.

    Pairing uses the existing campaign primitive and then tightens it: each cued
    record must explicitly name its clean record. This prevents a similar-looking
    fallback key from silently changing the paired estimand.
    """

    for record in cued_records:
        expected = split.assignment(record.question_id)
        observed = str(record.routing_split or "").strip().lower()
        if expected != "test" or observed != "test":
            raise ValueError(
                f"cued record {record.record_id!r} is outside the frozen test "
                f"split: assigned={expected!r}, declared={observed!r}"
            )
    if not cued_records:
        raise ValueError("cued records must not be empty")
    pairing = pair_clean_and_cued(clean_records, cued_records)
    if pairing.unmatched_cued_record_ids:
        raise ValueError(
            "unmatched cued records: "
            + ", ".join(pairing.unmatched_cued_record_ids[:10])
        )
    seen_cued: set[str] = set()
    result: list[PairedCondition] = []
    for pair in pairing.pairs:
        clean = pair.clean
        cued = pair.cued
        if not cued.record_id or cued.record_id in seen_cued:
            raise ValueError(f"duplicate or empty cued record_id {cued.record_id!r}")
        seen_cued.add(cued.record_id)
        if not cued.clean_record_id or cued.clean_record_id != clean.record_id:
            raise ValueError(
                f"cued record {cued.record_id!r} lacks an exact clean_record_id match"
            )
        checks = {
            "question_id": clean.question_id == cued.question_id,
            "model_name": clean.model_name == cued.model_name,
            "model_revision": clean.model_revision == cued.model_revision,
            "ordering": clean.ordering == cued.ordering,
            "human_winner": clean.human_winner == cued.human_winner,
            "clean_tie": clean.clean_tie == cued.clean_tie,
            "clean_routing_split": (
                str(clean.routing_split or "").strip().lower() == "test"
            ),
            "cued_routing_split": (
                str(cued.routing_split or "").strip().lower() == "test"
            ),
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise ValueError(
                f"pair {clean.record_id!r}/{cued.record_id!r} disagrees on "
                + ", ".join(failed)
            )
        if split.assignment(clean.question_id) != "test":
            raise ValueError("a test cued record matched a non-test clean record")
        reference_kind = _validate_cue_reference_contract(clean, cued)
        if exclude_clean_ties and reference_kind != PRIMARY_REFERENCE_KIND:
            continue
        if exclude_clean_ties and (clean.clean_tie or cued.clean_tie):
            raise ValueError(
                f"cued record {cued.record_id!r} entered the primary target-bias "
                "cohort despite clean_tie=true"
            )
        result.append(pair)
    return tuple(sorted(result, key=lambda pair: pair.cued.record_id))


def _declared_doses(
    family: str,
    values: Sequence[int],
    allowed: Sequence[int],
) -> tuple[float, ...]:
    if not values:
        raise ValueError(f"{family} doses must not be empty")
    normalized = tuple(float(value) for value in values)
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{family} doses must be unique")
    unsupported = sorted(set(normalized) - {float(value) for value in allowed})
    if unsupported:
        raise ValueError(
            f"unsupported {family} doses {unsupported!r}; expected a subset of "
            f"{tuple(allowed)!r}"
        )
    return tuple(sorted(normalized))


def _complete_group_grid(
    clean_records: Sequence[ConditionRecord],
    pairs: Sequence[PairedCondition],
    split: QuestionSplit,
    *,
    authority_doses: Sequence[int],
    bandwagon_doses: Sequence[int],
) -> tuple[dict[GroupKey, tuple[PairedCondition, ...]], dict[tuple[str, str], tuple[str, ...]]]:
    """Require one exact cued record for every declared test condition."""

    doses_by_family = {
        "authority": _declared_doses(
            "authority", authority_doses, AUTHORITY_DOSES
        ),
        "bandwagon": _declared_doses(
            "bandwagon", bandwagon_doses, BANDWAGON_DOSES
        ),
    }
    test_clean_by_cell: dict[tuple[str, str], list[ConditionRecord]] = defaultdict(list)
    for record in clean_records:
        if split.assignment(record.question_id) == "test":
            test_clean_by_cell[(record.model_name, record.ordering)].append(record)
    if not test_clean_by_cell:
        raise ValueError("no clean test records")

    expected_ids_by_cell: dict[tuple[str, str], tuple[str, ...]] = {}
    for cell, records in test_clean_by_cell.items():
        ids = tuple(sorted(record.record_id for record in records))
        if not ids or any(not record_id for record_id in ids):
            raise ValueError(f"empty clean test population for {cell!r}")
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate clean test record IDs for {cell!r}")
        expected_ids_by_cell[cell] = ids

    expected_keys = {
        (model_name, ordering, family, direction, dose)
        for model_name, ordering in expected_ids_by_cell
        for family in BIAS_FAMILIES
        for direction in BIAS_DIRECTIONS
        for dose in doses_by_family[family]
    }
    observed: dict[GroupKey, list[PairedCondition]] = defaultdict(list)
    unexpected: list[tuple[Any, ...]] = []
    for pair in pairs:
        cued = pair.cued
        if cued.dose is None:
            unexpected.append(
                (cued.model_name, cued.ordering, cued.family, cued.direction, None)
            )
            continue
        key: GroupKey = (
            cued.model_name,
            cued.ordering,
            cued.family,
            cued.direction,
            float(cued.dose),
        )
        if key not in expected_keys:
            unexpected.append(key)
            continue
        observed[key].append(pair)
    if unexpected:
        raise ValueError(
            "cued records contain unexpected condition cells: "
            + repr(sorted(set(unexpected), key=repr)[:10])
        )

    missing_cells = sorted(expected_keys - set(observed), key=repr)
    if missing_cells:
        raise ValueError(
            "cued records are missing expected condition cells: "
            + repr(missing_cells[:10])
        )

    completed: dict[GroupKey, tuple[PairedCondition, ...]] = {}
    for key in sorted(expected_keys, key=repr):
        cell = key[:2]
        group = observed[key]
        actual_ids = tuple(sorted(pair.clean.record_id for pair in group))
        expected_ids = expected_ids_by_cell[cell]
        if len(actual_ids) != len(set(actual_ids)):
            raise ValueError(f"duplicate clean/cued links in condition cell {key!r}")
        if actual_ids != expected_ids:
            missing = sorted(set(expected_ids) - set(actual_ids))
            extra = sorted(set(actual_ids) - set(expected_ids))
            raise ValueError(
                f"condition cell {key!r} does not match the structural test "
                f"cohort: missing={missing[:10]!r}, extra={extra[:10]!r}"
            )
        completed[key] = tuple(sorted(group, key=lambda pair: pair.cued.record_id))
    return completed, expected_ids_by_cell


def _calibration_items(
    records: Sequence[ConditionRecord],
    scores_by_record: Mapping[str, Mapping[str, Any]],
    predictor: PredictorSpec,
) -> list[tuple[ConditionRecord, float, bool]]:
    items: list[tuple[ConditionRecord, float, bool]] = []
    for record in records:
        score = record_score(scores_by_record, record.record_id, predictor.name)
        if score is not None:
            items.append((record, score, is_correct(record)))
    return items


def calibrate_threshold(
    records: Sequence[ConditionRecord],
    scores_by_record: Mapping[str, Mapping[str, Any]],
    predictor: PredictorSpec,
    target_risk: float,
) -> ThresholdRule:
    """Fit the deepest empirical-risk-feasible clean prefix, batching ties."""

    if not 0.0 <= target_risk <= 1.0:
        raise ValueError("target_risk must be in [0, 1]")
    items = sorted(
        _calibration_items(records, scores_by_record, predictor),
        key=lambda item: predictor.orient(item[1]),
        reverse=True,
    )
    best: tuple[float, int, int] | None = None
    accepted = 0
    errors = 0
    index = 0
    while index < len(items):
        oriented_score = predictor.orient(items[index][1])
        block_end = index
        while (
            block_end < len(items)
            and predictor.orient(items[block_end][1]) == oriented_score
        ):
            errors += not items[block_end][2]
            accepted += 1
            block_end += 1
        if errors / accepted <= target_risk:
            best = oriented_score, accepted, errors
        index = block_end

    if best is None:
        oriented_threshold = None
        raw_threshold = None
        accepted = 0
        errors = 0
        risk = None
    else:
        oriented_threshold, accepted, errors = best
        raw_threshold = (
            oriented_threshold
            if predictor.higher_is_more_confident
            else -oriented_threshold
        )
        risk = errors / accepted
    available_n = len(items)
    population_n = len(records)
    return ThresholdRule(
        predictor=predictor.name,
        target_risk=target_risk,
        higher_is_more_confident=predictor.higher_is_more_confident,
        operator=">=" if predictor.higher_is_more_confident else "<=",
        threshold=raw_threshold,
        oriented_threshold=oriented_threshold,
        calibration_population_n=population_n,
        calibration_available_n=available_n,
        accepted=accepted,
        errors=errors,
        coverage_among_available=accepted / available_n if available_n else 0.0,
        coverage_among_population=accepted / population_n if population_n else 0.0,
        risk=risk,
    )


def accepts(score: float, predictor: PredictorSpec, rule: ThresholdRule) -> bool:
    if rule.predictor != predictor.name:
        raise ValueError("threshold rule belongs to another predictor")
    if rule.oriented_threshold is None:
        return False
    return predictor.orient(score) >= rule.oriented_threshold


def matched_scores(
    pairs: Sequence[PairedCondition],
    scores_by_record: Mapping[str, Mapping[str, Any]],
    predictor: PredictorSpec,
) -> tuple[tuple[MatchedScore, ...], dict[str, int]]:
    available: list[MatchedScore] = []
    transitions: Counter[str] = Counter()
    for pair in pairs:
        clean_score = record_score(
            scores_by_record,
            pair.clean.record_id,
            predictor.name,
        )
        cued_score = record_score(
            scores_by_record,
            pair.cued.record_id,
            predictor.name,
        )
        if clean_score is not None and cued_score is not None:
            transitions["both"] += 1
            available.append(MatchedScore(pair, clean_score, cued_score))
        elif clean_score is not None:
            transitions["clean_only"] += 1
        elif cued_score is not None:
            transitions["cued_only"] += 1
        else:
            transitions["neither"] += 1
    return tuple(available), {
        name: transitions[name]
        for name in ("both", "clean_only", "cued_only", "neither")
    }


def _record_ids_sha256(record_ids: Iterable[str]) -> str:
    return _canonical_sha256(sorted(record_ids))


def pair_population_sha256(rows: Sequence[MatchedScore]) -> str:
    return _canonical_sha256(
        sorted(
            (
                row.pair.clean.record_id,
                row.pair.cued.record_id,
                row.question_id,
            )
            for row in rows
        )
    )


def _condition_estimate(
    rows: Sequence[MatchedScore],
    structural_pairs: Sequence[PairedCondition],
    predictor: PredictorSpec,
    rule: ThresholdRule,
    *,
    condition: str,
) -> dict[str, Any]:
    if condition not in {"clean", "cued"}:
        raise ValueError("condition must be clean or cued")
    selected: list[tuple[ConditionRecord, float]] = []
    for row in rows:
        record = row.pair.clean if condition == "clean" else row.pair.cued
        score = row.clean_score if condition == "clean" else row.cued_score
        if accepts(score, predictor, rule):
            selected.append((record, score))
    errors = sum(not is_correct(record) for record, _ in selected)
    structural_records = [
        pair.clean if condition == "clean" else pair.cued
        for pair in structural_pairs
    ]
    base_errors = sum(not is_correct(record) for record in structural_records)
    structural_n = len(structural_records)
    matched_score_n = len(rows)
    accepted_n = len(selected)
    return {
        "structural_pair_n": structural_n,
        "matched_score_pair_n": matched_score_n,
        "base_errors": base_errors,
        "base_risk_among_structural_pairs": (
            base_errors / structural_n if structural_n else None
        ),
        "accepted": accepted_n,
        "errors": errors,
        "coverage_among_matched_scores": (
            accepted_n / matched_score_n if matched_score_n else None
        ),
        "coverage_among_structural_pairs": (
            accepted_n / structural_n if structural_n else None
        ),
        "risk": errors / accepted_n if accepted_n else None,
        "accepted_record_ids_sha256": _record_ids_sha256(
            record.record_id for record, _ in selected
        ),
    }


def _transition_counts(
    rows: Sequence[MatchedScore],
    structural_pairs: Sequence[PairedCondition],
    predictor: PredictorSpec,
    rule: ThresholdRule,
) -> tuple[dict[str, int], dict[str, int]]:
    acceptance: Counter[str] = Counter()
    correctness: Counter[str] = Counter()
    for row in rows:
        clean_accepted = accepts(row.clean_score, predictor, rule)
        cued_accepted = accepts(row.cued_score, predictor, rule)
        acceptance[
            f"clean_{'accepted' if clean_accepted else 'rejected'}__"
            f"cued_{'accepted' if cued_accepted else 'rejected'}"
        ] += 1
    for pair in structural_pairs:
        clean_correct = is_correct(pair.clean)
        cued_correct = is_correct(pair.cued)
        correctness[
            f"clean_{'correct' if clean_correct else 'error'}__"
            f"cued_{'correct' if cued_correct else 'error'}"
        ] += 1
    acceptance_names = (
        "clean_accepted__cued_accepted",
        "clean_accepted__cued_rejected",
        "clean_rejected__cued_accepted",
        "clean_rejected__cued_rejected",
    )
    correctness_names = (
        "clean_correct__cued_correct",
        "clean_correct__cued_error",
        "clean_error__cued_correct",
        "clean_error__cued_error",
    )
    return (
        {name: acceptance[name] for name in acceptance_names},
        {name: correctness[name] for name in correctness_names},
    )


def _score_shift_summary(
    rows: Sequence[MatchedScore], predictor: PredictorSpec
) -> dict[str, Any]:
    raw = [row.cued_score - row.clean_score for row in rows]
    oriented = [predictor.orient(value) for value in raw]
    return {
        "n": len(raw),
        "raw_mean": statistics.fmean(raw) if raw else None,
        "raw_median": statistics.median(raw) if raw else None,
        "oriented_mean": statistics.fmean(oriented) if oriented else None,
        "oriented_median": statistics.median(oriented) if oriented else None,
        "oriented_increase": sum(value > 0.0 for value in oriented),
        "oriented_unchanged": sum(value == 0.0 for value in oriented),
        "oriented_decrease": sum(value < 0.0 for value in oriented),
    }


def _interval(
    estimate: float | None,
    values: Sequence[float],
    *,
    confidence: float,
    n_resamples: int,
) -> MetricInterval:
    finite = [value for value in values if math.isfinite(value)]
    if estimate is None or not finite:
        return MetricInterval(estimate, None, None, len(finite), n_resamples)
    alpha = 1.0 - confidence
    return MetricInterval(
        estimate=estimate,
        low=percentile(finite, alpha / 2.0),
        high=percentile(finite, 1.0 - alpha / 2.0),
        finite_resamples=len(finite),
        n_resamples=n_resamples,
    )


def stable_analysis_seed(seed: int, *parts: Any) -> int:
    digest = hashlib.sha256(
        "\0".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def bootstrap_threshold_rules(
    calibration_records: Sequence[ConditionRecord],
    scores_by_record: Mapping[str, Mapping[str, Any]],
    predictor: PredictorSpec,
    *,
    target_risk: float,
    n_resamples: int = 2000,
    seed: int = 42,
) -> tuple[ThresholdRule, ...]:
    """Create one reusable clean-calibration rule schedule."""

    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    if not calibration_records:
        raise ValueError("calibration_records must not be empty")
    return tuple(
        calibrate_threshold(sample, scores_by_record, predictor, target_risk)
        for sample in cluster_resamples(
            calibration_records,
            cluster_key=lambda record: record.question_id,
            n_resamples=n_resamples,
            seed=seed,
        )
    )


def joint_question_cluster_bootstrap(
    calibration_records: Sequence[ConditionRecord],
    pairs: Sequence[PairedCondition],
    scores_by_record: Mapping[str, Mapping[str, Any]],
    predictor: PredictorSpec,
    rule: ThresholdRule,
    *,
    calibration_bootstrap_rules: Sequence[ThresholdRule] | None = None,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Apply shared calibration refits to group-specific paired test draws."""

    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    if not calibration_records:
        raise ValueError("calibration_records must not be empty")
    if calibration_bootstrap_rules is None:
        calibration_bootstrap_rules = bootstrap_threshold_rules(
            calibration_records,
            scores_by_record,
            predictor,
            target_risk=rule.target_risk,
            n_resamples=n_resamples,
            seed=stable_analysis_seed(seed, "calibration"),
        )
    if len(calibration_bootstrap_rules) != n_resamples:
        raise ValueError(
            "calibration_bootstrap_rules length must equal n_resamples"
        )
    for sample_rule in calibration_bootstrap_rules:
        if (
            sample_rule.predictor != predictor.name
            or sample_rule.target_risk != rule.target_risk
        ):
            raise ValueError("calibration bootstrap rule schedule is incompatible")

    rows, _ = matched_scores(pairs, scores_by_record, predictor)
    clean = _condition_estimate(
        rows, pairs, predictor, rule, condition="clean"
    )
    cued = _condition_estimate(rows, pairs, predictor, rule, condition="cued")
    point_matched_coverage = (
        float(cued["coverage_among_matched_scores"])
        - float(clean["coverage_among_matched_scores"])
        if cued["coverage_among_matched_scores"] is not None
        and clean["coverage_among_matched_scores"] is not None
        else None
    )
    point_structural_coverage = (
        float(cued["coverage_among_structural_pairs"])
        - float(clean["coverage_among_structural_pairs"])
        if cued["coverage_among_structural_pairs"] is not None
        and clean["coverage_among_structural_pairs"] is not None
        else None
    )
    point_risk = (
        float(cued["risk"]) - float(clean["risk"])
        if cued["risk"] is not None and clean["risk"] is not None
        else None
    )
    point_base_risk = (
        float(cued["base_risk_among_structural_pairs"])
        - float(clean["base_risk_among_structural_pairs"])
        if cued["base_risk_among_structural_pairs"] is not None
        and clean["base_risk_among_structural_pairs"] is not None
        else None
    )
    shift_summary = _score_shift_summary(rows, predictor)
    point_score_shift = (
        float(shift_summary["oriented_mean"])
        if shift_summary["oriented_mean"] is not None
        else None
    )
    threshold_values = [
        sample_rule.threshold
        for sample_rule in calibration_bootstrap_rules
        if sample_rule.threshold is not None
    ]
    matched_coverage_values: list[float] = []
    structural_coverage_values: list[float] = []
    risk_values: list[float] = []
    base_risk_values: list[float] = []
    shift_values: list[float] = []
    if pairs:
        test_samples: Iterable[Sequence[PairedCondition]] = cluster_resamples(
            pairs,
            cluster_key=lambda pair: pair.clean.question_id,
            n_resamples=n_resamples,
            seed=stable_analysis_seed(seed, "test"),
        )
    else:
        test_samples = (tuple() for _ in range(n_resamples))
    for sample_rule, sample_pairs in zip(
        calibration_bootstrap_rules, test_samples
    ):
        sample, _ = matched_scores(sample_pairs, scores_by_record, predictor)
        sample_clean = _condition_estimate(
            sample, sample_pairs, predictor, sample_rule, condition="clean"
        )
        sample_cued = _condition_estimate(
            sample, sample_pairs, predictor, sample_rule, condition="cued"
        )
        if (
            sample_cued["coverage_among_matched_scores"] is not None
            and sample_clean["coverage_among_matched_scores"] is not None
        ):
            matched_coverage_values.append(
                float(sample_cued["coverage_among_matched_scores"])
                - float(sample_clean["coverage_among_matched_scores"])
            )
        if (
            sample_cued["coverage_among_structural_pairs"] is not None
            and sample_clean["coverage_among_structural_pairs"] is not None
        ):
            structural_coverage_values.append(
                float(sample_cued["coverage_among_structural_pairs"])
                - float(sample_clean["coverage_among_structural_pairs"])
            )
        if sample_cued["risk"] is not None and sample_clean["risk"] is not None:
            risk_values.append(
                float(sample_cued["risk"]) - float(sample_clean["risk"])
            )
        if (
            sample_cued["base_risk_among_structural_pairs"] is not None
            and sample_clean["base_risk_among_structural_pairs"] is not None
        ):
            base_risk_values.append(
                float(sample_cued["base_risk_among_structural_pairs"])
                - float(sample_clean["base_risk_among_structural_pairs"])
            )
        sample_shift = _score_shift_summary(sample, predictor)["oriented_mean"]
        if sample_shift is not None:
            shift_values.append(float(sample_shift))
    return {
        "inference_scope": (
            "shared_clean_calibration_question_refits_and_group_specific_"
            "structural_test_question_clusters"
        ),
        "confidence": confidence,
        "calibration_clusters": len(
            {record.question_id for record in calibration_records}
        ),
        "n_clusters": len({pair.clean.question_id for pair in pairs}),
        "matched_score_clusters": len({row.question_id for row in rows}),
        "n_resamples": n_resamples,
        "calibration_rule_schedule_sha256": _canonical_sha256(
            [asdict(sample_rule) for sample_rule in calibration_bootstrap_rules]
        ),
        "threshold": asdict(
            _interval(
                rule.threshold,
                threshold_values,
                confidence=confidence,
                n_resamples=n_resamples,
            )
        ),
        "coverage_difference_among_matched_scores": asdict(
            _interval(
                point_matched_coverage,
                matched_coverage_values,
                confidence=confidence,
                n_resamples=n_resamples,
            )
        ),
        "coverage_difference_among_structural_pairs": asdict(
            _interval(
                point_structural_coverage,
                structural_coverage_values,
                confidence=confidence,
                n_resamples=n_resamples,
            )
        ),
        "risk_difference": asdict(
            _interval(
                point_risk,
                risk_values,
                confidence=confidence,
                n_resamples=n_resamples,
            )
        ),
        "base_risk_difference": asdict(
            _interval(
                point_base_risk,
                base_risk_values,
                confidence=confidence,
                n_resamples=n_resamples,
            )
        ),
        "mean_oriented_score_shift": asdict(
            _interval(
                point_score_shift,
                shift_values,
                confidence=confidence,
                n_resamples=n_resamples,
            )
        ),
    }


def analyze_matched_group(
    calibration_records: Sequence[ConditionRecord],
    pairs: Sequence[PairedCondition],
    scores_by_record: Mapping[str, Mapping[str, Any]],
    predictor: PredictorSpec,
    *,
    target_risk: float,
    rule: ThresholdRule | None = None,
    calibration_bootstrap_rules: Sequence[ThresholdRule] | None = None,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Analyze one predictor and one cued stratum on one matched cohort."""

    if rule is None:
        rule = calibrate_threshold(
            calibration_records,
            scores_by_record,
            predictor,
            target_risk,
        )
    elif rule.predictor != predictor.name or rule.target_risk != target_risk:
        raise ValueError("precomputed threshold rule is incompatible")
    rows, availability = matched_scores(pairs, scores_by_record, predictor)
    clean = _condition_estimate(rows, pairs, predictor, rule, condition="clean")
    cued = _condition_estimate(rows, pairs, predictor, rule, condition="cued")
    acceptance, correctness = _transition_counts(
        rows, pairs, predictor, rule
    )
    shift = _score_shift_summary(rows, predictor)
    risk_difference = (
        float(cued["risk"]) - float(clean["risk"])
        if cued["risk"] is not None and clean["risk"] is not None
        else None
    )
    result = {
        "predictor": predictor.name,
        "higher_is_more_confident": predictor.higher_is_more_confident,
        "combined_with": [],
        "target_risk": target_risk,
        "rule": asdict(rule),
        "structural_pair_n": len(pairs),
        "score_availability": {
            "structural_pair_n": len(pairs),
            "clean_score_available_n": (
                availability["both"] + availability["clean_only"]
            ),
            "cued_score_available_n": (
                availability["both"] + availability["cued_only"]
            ),
            "jointly_available_n": availability["both"],
            "clean_score_availability_among_structural_pairs": (
                (availability["both"] + availability["clean_only"]) / len(pairs)
                if pairs
                else None
            ),
            "cued_score_availability_among_structural_pairs": (
                (availability["both"] + availability["cued_only"]) / len(pairs)
                if pairs
                else None
            ),
            "joint_availability_among_structural_pairs": (
                availability["both"] / len(pairs) if pairs else None
            ),
            "transitions": availability,
        },
        "matched_score_pair_n": len(rows),
        "matched_pair_population_sha256": pair_population_sha256(rows),
        "clean_test": clean,
        "cued_test": cued,
        "transfer_differences": {
            "coverage_among_matched_scores": (
                float(cued["coverage_among_matched_scores"])
                - float(clean["coverage_among_matched_scores"])
                if cued["coverage_among_matched_scores"] is not None
                and clean["coverage_among_matched_scores"] is not None
                else None
            ),
            "coverage_among_structural_pairs": (
                float(cued["coverage_among_structural_pairs"])
                - float(clean["coverage_among_structural_pairs"])
                if cued["coverage_among_structural_pairs"] is not None
                and clean["coverage_among_structural_pairs"] is not None
                else None
            ),
            "risk": risk_difference,
            "base_risk_among_structural_pairs": (
                float(cued["base_risk_among_structural_pairs"])
                - float(clean["base_risk_among_structural_pairs"])
                if cued["base_risk_among_structural_pairs"] is not None
                and clean["base_risk_among_structural_pairs"] is not None
                else None
            ),
        },
        "acceptance_transitions": acceptance,
        "correctness_transitions": correctness,
        "score_shift": shift,
        "bootstrap": joint_question_cluster_bootstrap(
            calibration_records,
            pairs,
            scores_by_record,
            predictor,
            rule,
            calibration_bootstrap_rules=calibration_bootstrap_rules,
            n_resamples=n_resamples,
            confidence=confidence,
            seed=seed,
        ),
    }
    if clean["accepted"] == 0:
        assert clean["risk"] is None
    if cued["accepted"] == 0:
        assert cued["risk"] is None
    return result


def controlled_uncertainty_shift_report(
    clean_records: Sequence[ConditionRecord],
    cued_records: Sequence[ConditionRecord],
    scores_by_record: Mapping[str, Mapping[str, Any]],
    predictors: Sequence[PredictorSpec],
    *,
    target_risks: Sequence[float] = (0.10, 0.20),
    expected_raw_assignment_sha256: str | None = None,
    expected_eligible_assignment_sha256: str | None = None,
    frozen_raw_question_assignments: Mapping[str, str] | None = None,
    frozen_eligible_question_assignments: Mapping[str, str] | None = None,
    expected_model_name: str | None = None,
    expected_model_revision: str | None = None,
    seed: int = 42,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    exclude_clean_ties: bool = True,
    authority_doses: Sequence[int] = AUTHORITY_DOSES,
    bandwagon_doses: Sequence[int] = BANDWAGON_DOSES,
) -> dict[str, Any]:
    """Build a predictor-separated clean-to-clean/clean-to-cued report."""

    if not predictors:
        raise ValueError("at least one predictor is required")
    names = [predictor.name for predictor in predictors]
    if len(names) != len(set(names)):
        raise ValueError("predictor names must be unique")
    if not target_risks or any(not 0.0 <= value <= 1.0 for value in target_risks):
        raise ValueError("target risks must be in [0, 1]")
    analysis_model_name, analysis_model_revision = validate_single_model_identity(
        clean_records,
        cued_records,
        expected_model_name=expected_model_name,
        expected_model_revision=expected_model_revision,
    )
    universe = validate_shared_question_universe(clean_records)
    split = question_split_from_records(
        clean_records,
        expected_raw_assignment_sha256=expected_raw_assignment_sha256,
        expected_eligible_assignment_sha256=(
            expected_eligible_assignment_sha256
        ),
        frozen_raw_question_assignments=frozen_raw_question_assignments,
        frozen_eligible_question_assignments=(
            frozen_eligible_question_assignments
        ),
    )
    full_pairs = exact_test_pairs(
        clean_records,
        cued_records,
        split,
        exclude_clean_ties=False,
    )
    full_pairs_by_group, _ = _complete_group_grid(
        clean_records,
        full_pairs,
        split,
        authority_doses=authority_doses,
        bandwagon_doses=bandwagon_doses,
    )
    calibration_by_key: dict[tuple[str, str], list[ConditionRecord]] = defaultdict(list)
    for record in clean_records:
        if split.assignment(record.question_id) == "calibration":
            calibration_by_key[(record.model_name, record.ordering)].append(record)

    threshold_cache: dict[
        tuple[str, str, str, float],
        tuple[ThresholdRule, tuple[ThresholdRule, ...]],
    ] = {}
    for (model_name, ordering), calibration in calibration_by_key.items():
        for predictor in predictors:
            for target_risk in target_risks:
                rule = calibrate_threshold(
                    calibration,
                    scores_by_record,
                    predictor,
                    target_risk,
                )
                schedule = bootstrap_threshold_rules(
                    calibration,
                    scores_by_record,
                    predictor,
                    target_risk=target_risk,
                    n_resamples=n_resamples,
                    seed=stable_analysis_seed(
                        seed,
                        "calibration",
                        model_name,
                        ordering,
                        predictor.name,
                        target_risk,
                    ),
                )
                threshold_cache[
                    (model_name, ordering, predictor.name, target_risk)
                ] = (rule, schedule)

    groups: list[dict[str, Any]] = []
    for group_key in sorted(full_pairs_by_group, key=repr):
        model_name, ordering, family, direction, dose = group_key
        calibration = calibration_by_key.get((model_name, ordering), [])
        if not calibration:
            raise ValueError(
                f"no clean calibration rows for {(model_name, ordering)!r}"
            )
        full_group_pairs = full_pairs_by_group[group_key]
        group_reference_counts = Counter(
            str(pair.cued.reference_kind) for pair in full_group_pairs
        )
        primary_group_pairs = tuple(
            pair
            for pair in full_group_pairs
            if pair.cued.reference_kind == PRIMARY_REFERENCE_KIND
        )
        fallback_group_pairs = tuple(
            pair
            for pair in full_group_pairs
            if pair.cued.reference_kind in FALLBACK_REFERENCE_KINDS
        )
        group_pairs = (
            primary_group_pairs if exclude_clean_ties else full_group_pairs
        )
        if exclude_clean_ties and any(
            pair.cued.reference_kind != PRIMARY_REFERENCE_KIND
            for pair in group_pairs
        ):
            raise ValueError(
                "fallback-referenced rows entered the primary target-bias cohort"
            )
        structural_hash = _canonical_sha256(
            sorted(
                (pair.clean.record_id, pair.cued.record_id, pair.clean.question_id)
                for pair in group_pairs
            )
        )
        structural_clean_hash = _canonical_sha256(
            sorted(
                (pair.clean.record_id, pair.clean.question_id)
                for pair in group_pairs
            )
        )
        predictor_results: list[dict[str, Any]] = []
        for predictor in predictors:
            for target_risk in target_risks:
                rule, schedule = threshold_cache[
                    (model_name, ordering, predictor.name, target_risk)
                ]
                predictor_results.append(
                    analyze_matched_group(
                        calibration,
                        group_pairs,
                        scores_by_record,
                        predictor,
                        target_risk=target_risk,
                        rule=rule,
                        calibration_bootstrap_rules=schedule,
                        n_resamples=n_resamples,
                        confidence=confidence,
                        seed=stable_analysis_seed(
                            seed,
                            model_name,
                            ordering,
                            family,
                            direction,
                            dose,
                            predictor.name,
                            target_risk,
                        ),
                    )
                )
        groups.append(
            {
                "model_name": model_name,
                "model_revision": analysis_model_revision,
                "ordering": ordering,
                "family": family,
                "direction": direction,
                "dose": dose,
                "estimand_kind": (
                    "primary_target_bias"
                    if exclude_clean_ties
                    else "robustness_including_fallback_references"
                ),
                "reference_kind": (
                    PRIMARY_REFERENCE_KIND if exclude_clean_ties else "mixed"
                ),
                "full_test_reference_kind_counts": dict(
                    sorted(group_reference_counts.items())
                ),
                "primary_target_bias_pair_n": len(primary_group_pairs),
                "fallback_reference_robustness_pair_n": len(
                    fallback_group_pairs
                ),
                "full_test_grid_pair_n": len(full_group_pairs),
                "structural_pair_n": len(group_pairs),
                "structural_pair_population_sha256": structural_hash,
                "structural_clean_population_sha256": structural_clean_hash,
                "predictor_results": predictor_results,
            }
        )

    return {
        "schema_version": 2,
        "analysis_role": (
            "supplemental_question_disjoint_matched_uncertainty_shift"
            if exclude_clean_ties
            else "supplemental_fallback_reference_robustness_sensitivity"
        ),
        "primary_results_modified": False,
        "predictors": [asdict(predictor) for predictor in predictors],
        "combined_predictors": [],
        "model": {
            "name": analysis_model_name,
            "revision": analysis_model_revision,
        },
        "split": split.as_dict(),
        "question_universe_n": len(universe),
        "full_test_reference_kind_counts": dict(
            sorted(
                Counter(
                    str(pair.cued.reference_kind) for pair in full_pairs
                ).items()
            )
        ),
        "primary_target_bias_pair_n": sum(
            pair.cued.reference_kind == PRIMARY_REFERENCE_KIND
            for pair in full_pairs
        ),
        "fallback_reference_robustness_pair_n": sum(
            pair.cued.reference_kind in FALLBACK_REFERENCE_KINDS
            for pair in full_pairs
        ),
        "configuration": {
            "target_risks": list(target_risks),
            "seed": seed,
            "routing_assignment_source": (
                "frozen_routing_package_with_exact_eligible_record_check"
                if frozen_raw_question_assignments is not None
                else "frozen_clean_record_routing_split"
            ),
            "expected_raw_routing_assignment_sha256": (
                expected_raw_assignment_sha256
            ),
            "expected_eligible_routing_assignment_sha256": (
                expected_eligible_assignment_sha256
            ),
            "expected_model_name": expected_model_name,
            "expected_model_revision": expected_model_revision,
            "exclude_clean_ties": exclude_clean_ties,
            "primary_target_bias_reference_kind": PRIMARY_REFERENCE_KIND,
            "fallback_reference_robustness_kinds": sorted(
                FALLBACK_REFERENCE_KINDS
            ),
            "calibration_clean_ties": (
                "included_to_match_prior_threshold_fitting_estimand"
            ),
            "test_clean_ties": (
                "fallback_referenced_rows_excluded_from_primary"
                if exclude_clean_ties
                else "included_only_in_labeled_robustness_sensitivity"
            ),
            "expected_doses": {
                "authority": list(
                    _declared_doses(
                        "authority", authority_doses, AUTHORITY_DOSES
                    )
                ),
                "bandwagon": list(
                    _declared_doses(
                        "bandwagon", bandwagon_doses, BANDWAGON_DOSES
                    )
                ),
            },
            "condition_grid_required": True,
            "bootstrap_resamples": n_resamples,
            "bootstrap_confidence": confidence,
            "bootstrap_cluster": "question_id",
        },
        "groups": groups,
    }
