#!/usr/bin/env python3
"""Audit repeatability predictors without changing strict-v3 primary results.

The audit reads immutable Stage-A and Stage-B score JSONL files. It evaluates
each predictor separately and writes one JSON report; it never modifies source
artifacts or combines predictors.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from biases.analysis.records import normalize_label
from biases.analysis.repeatability import (
    anchor_reproducibility,
    degree_matrix_agreement,
    frequency_semantic_entropy_confidence,
    ordering_repeatability_scores,
)


STAGE_A_FILENAME = "silent_bias_stage_a_uncertainty_scores.jsonl"
STAGE_B_FILENAME = "silent_bias_stage_b_uncertainty_scores.jsonl"
MODEL_MARKER_FILENAME = "campaign_model_complete.json"
PREDICTORS = (
    "msp",
    "anchor_reproducibility",
    "frequency_semantic_entropy_confidence",
    "degree_matrix_agreement",
    "order_vote_js_similarity",
    "order_vote_tv_similarity",
    "order_vote_expected_agreement",
)
LM_SEMANTIC_ENTROPY_PREDICTOR = "frequency_semantic_entropy_confidence"
LM_DEGREE_MATRIX_PREDICTOR = "degree_matrix_agreement"


@dataclass(slots=True)
class AuditRow:
    record_id: str
    question_id: str
    model_name: str
    pair_identity_key: str
    condition_group_id: str
    ordering: str
    routing_split: str
    family: str
    direction: str
    dose: float | None
    clean_tie: bool
    cue_target: str | None
    human_winner: str | None
    verdict: str
    verdict_counts: dict[str, int] | None
    scores: dict[str, float | None] = field(default_factory=dict)


def finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _nested(row: Mapping[str, Any], *path: str) -> Any:
    value: Any = row
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _verdict_counts(row: Mapping[str, Any]) -> dict[str, int] | None:
    raw = row.get("consistency_verdict_counts")
    if raw is None:
        raw = _nested(row, "uncertainty", "consistency", "verdict_counts")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("consistency verdict counts must be an object")
    counts: dict[str, int] = {}
    for key, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("consistency verdict counts must be integers")
        counts[str(key)] = value
    return counts


def audit_row_from_mapping(row: Mapping[str, Any]) -> AuditRow:
    verdict = normalize_label(row.get("verdict"))
    if verdict is None:
        raise ValueError(f"record {row.get('record_id')!r} has no valid verdict")
    ordering = str(row.get("ordering") or "").strip().lower()
    if ordering not in {"ab", "ba"}:
        raise ValueError(f"record {row.get('record_id')!r} has invalid ordering")
    family = str(row.get("bias_name") or row.get("family") or "").strip().lower()
    counts = _verdict_counts(row)
    anchor_score = (
        anchor_reproducibility(verdict, counts) if counts is not None else None
    )
    semantic_entropy_score = (
        frequency_semantic_entropy_confidence(counts)
        if counts is not None
        else None
    )
    degree_matrix_score = (
        degree_matrix_agreement(counts) if counts is not None else None
    )
    stored_flip_rate = row.get("consistency_flip_rate")
    if stored_flip_rate is None:
        stored_flip_rate = _nested(row, "uncertainty", "consistency", "flip_rate")
    flip_rate = finite_float(stored_flip_rate)
    if anchor_score is not None and flip_rate is not None:
        if not math.isclose(anchor_score, 1.0 - flip_rate, abs_tol=1e-12):
            raise ValueError(
                f"record {row.get('record_id')!r} has inconsistent flip_rate"
            )
    if stored_flip_rate is not None and (
        flip_rate is None or not 0.0 <= flip_rate <= 1.0
    ):
        raise ValueError("consistency flip rate must be finite and in [0, 1]")
    msp = finite_float(row.get("msp"))
    if row.get("msp") is not None and (msp is None or not 0.0 <= msp <= 1.0):
        raise ValueError("MSP must be finite and in [0, 1]")
    human_winner = normalize_label(row.get("human_winner"))
    if human_winner is None:
        raise ValueError("human_winner must be a valid A/B/tie label")
    cue_target = normalize_label(row.get("cue_target"))
    if row.get("cue_target") is not None and cue_target is None:
        raise ValueError("cue_target must be a valid A/B/tie label")
    clean_tie_value = row.get("clean_tie")
    clean_tie = (
        (
            clean_tie_value.strip().lower() in {"1", "true", "yes"}
            if isinstance(clean_tie_value, str)
            else bool(clean_tie_value)
        )
        if clean_tie_value is not None
        else family == "clean" and verdict == "tie"
    )
    parsed = AuditRow(
        record_id=str(row.get("record_id") or ""),
        question_id=str(row.get("question_id") or ""),
        model_name=str(row.get("model_name") or ""),
        pair_identity_key=str(row.get("pair_identity_key") or ""),
        condition_group_id=str(row.get("condition_group_id") or ""),
        ordering=ordering,
        routing_split=str(row.get("routing_split") or ""),
        family=family,
        direction=str(row.get("cue_congruency") or row.get("direction") or "")
        .strip()
        .lower(),
        dose=finite_float(row.get("dose")),
        clean_tie=clean_tie,
        cue_target=cue_target,
        human_winner=human_winner,
        verdict=verdict,
        verdict_counts=counts,
        scores={
            "msp": msp,
            "anchor_reproducibility": anchor_score,
            "frequency_semantic_entropy_confidence": semantic_entropy_score,
            "degree_matrix_agreement": degree_matrix_score,
            "order_vote_js_similarity": None,
            "order_vote_tv_similarity": None,
            "order_vote_expected_agreement": None,
        },
    )
    for name in (
        "record_id",
        "question_id",
        "model_name",
        "pair_identity_key",
        "condition_group_id",
    ):
        if not getattr(parsed, name):
            raise ValueError(f"record is missing required {name}")
    return parsed


def read_jsonl(
    path: Path,
    *,
    predicate: Callable[[Mapping[str, Any]], bool] | None = None,
) -> list[AuditRow]:
    rows: list[AuditRow] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = json.loads(line)
            if not isinstance(raw, Mapping):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            if predicate is not None and not predicate(raw):
                continue
            try:
                rows.append(audit_row_from_mapping(raw))
            except Exception as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return rows


def canonical_label(label: str | None, ordering: str) -> str | None:
    normalized = normalize_label(label)
    if normalized is None or ordering == "ab" or normalized == "tie":
        return normalized
    if ordering != "ba":
        raise ValueError("ordering must be 'ab' or 'ba'")
    return "B" if normalized == "A" else "A"


def attach_order_vote_scores(rows: Iterable[AuditRow]) -> dict[str, int]:
    groups: dict[tuple[Any, ...], list[AuditRow]] = defaultdict(list)
    for row in rows:
        groups[
            (
                row.pair_identity_key,
                row.condition_group_id,
                row.family,
                row.direction,
                row.dose,
                row.routing_split,
            )
        ].append(row)
    complete = malformed = missing_counts = human_disagreements = 0
    cue_target_disagreements = unequal_repeat_counts = 0
    for group in groups.values():
        by_order = {row.ordering: row for row in group}
        if len(group) != 2 or set(by_order) != {"ab", "ba"}:
            malformed += 1
            continue
        left, right = by_order["ab"], by_order["ba"]
        if left.model_name != right.model_name:
            malformed += 1
            continue
        if canonical_label(left.human_winner, "ab") != canonical_label(
            right.human_winner,
            "ba",
        ):
            human_disagreements += 1
            continue
        if canonical_label(left.cue_target, "ab") != canonical_label(
            right.cue_target,
            "ba",
        ):
            cue_target_disagreements += 1
            continue
        if left.verdict_counts is None or right.verdict_counts is None:
            missing_counts += 1
            continue
        if sum(left.verdict_counts.values()) != sum(right.verdict_counts.values()):
            unequal_repeat_counts += 1
            continue
        scores = ordering_repeatability_scores(
            left.verdict_counts,
            right.verdict_counts,
        )
        for row in (left, right):
            row.scores["order_vote_js_similarity"] = scores.js_similarity
            row.scores[
                "order_vote_tv_similarity"
            ] = scores.total_variation_similarity
            row.scores[
                "order_vote_expected_agreement"
            ] = scores.independent_draw_agreement
        complete += 1
    return {
        "groups": len(groups),
        "complete_pairs": complete,
        "malformed_pairs": malformed,
        "pairs_missing_counts": missing_counts,
        "canonical_human_disagreements": human_disagreements,
        "canonical_cue_target_disagreements": cue_target_disagreements,
        "unequal_repeat_count_pairs": unequal_repeat_counts,
    }


def valid_items(
    rows: Iterable[AuditRow],
    predictor: str,
) -> list[tuple[AuditRow, float, bool]]:
    items: list[tuple[AuditRow, float, bool]] = []
    for row in rows:
        score = finite_float(row.scores.get(predictor))
        if score is None or row.human_winner is None:
            continue
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"{predictor} score must be in [0, 1]")
        items.append((row, score, row.verdict == row.human_winner))
    return items


def expected_calibration_error(
    items: list[tuple[AuditRow, float, bool]],
    *,
    n_bins: int = 10,
) -> float | None:
    if not items:
        return None
    buckets: list[list[tuple[AuditRow, float, bool]]] = [
        [] for _ in range(n_bins)
    ]
    for item in items:
        buckets[min(n_bins - 1, int(item[1] * n_bins))].append(item)
    return sum(
        len(bucket)
        * abs(
            sum(item[1] for item in bucket) / len(bucket)
            - sum(item[2] for item in bucket) / len(bucket)
        )
        for bucket in buckets
        if bucket
    ) / len(items)


def ranking_metrics(
    items: list[tuple[AuditRow, float, bool]],
) -> dict[str, float | None]:
    if not items:
        return {"aurc": None, "correctness_auroc": None}
    ranked = sorted(items, key=lambda item: item[1], reverse=True)
    total = len(ranked)
    aurc = 0.0
    previous_coverage = 0.0
    accepted = errors = cursor = 0
    while cursor < total:
        threshold = ranked[cursor][1]
        while cursor < total and ranked[cursor][1] == threshold:
            accepted += 1
            errors += int(not ranked[cursor][2])
            cursor += 1
        coverage = accepted / total
        risk = errors / accepted
        # A right-continuous step integral is the empirical AURC. In
        # particular, a constant-score predictor has AURC equal to its base
        # error rate; interpolating from an artificial (0, 0) point would
        # unfairly reward predictors with a few large tied score blocks.
        aurc += (coverage - previous_coverage) * risk
        previous_coverage = coverage
    groups: dict[float, list[bool]] = defaultdict(list)
    for _, score, correct in items:
        groups[score].append(correct)
    errors_below = 0
    concordance = 0.0
    n_correct = sum(correct for _, _, correct in items)
    n_error = total - n_correct
    for score in sorted(groups):
        group = groups[score]
        correct_n = sum(group)
        error_n = len(group) - correct_n
        concordance += correct_n * (errors_below + 0.5 * error_n)
        errors_below += error_n
    return {
        "aurc": aurc,
        "correctness_auroc": (
            concordance / (n_correct * n_error) if n_correct and n_error else None
        ),
    }


def record_id_sha256(
    items: Iterable[tuple[AuditRow, float, bool]],
) -> str:
    """Hash a selected record-id multiset without exposing bulky ID lists."""

    digest = hashlib.sha256()
    for record_id in sorted(item[0].record_id for item in items):
        digest.update(record_id.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def metric_summary(rows: list[AuditRow], predictor: str) -> dict[str, Any]:
    items = valid_items(rows, predictor)
    ranking = ranking_metrics(items)
    return {
        "total": len(rows),
        "n": len(items),
        "availability": len(items) / len(rows) if rows else None,
        "accuracy": sum(correct for _, _, correct in items) / len(items)
        if items
        else None,
        "raw_score_ece_10bin": expected_calibration_error(items),
        "raw_score_brier": (
            sum((score - float(correct)) ** 2 for _, score, correct in items)
            / len(items)
            if items
            else None
        ),
        "aurc": ranking["aurc"],
        "correctness_auroc": ranking["correctness_auroc"],
        "unique_score_levels": len({score for _, score, _ in items}),
    }


def threshold_rule(
    rows: list[AuditRow],
    predictor: str,
    target_risk: float,
) -> dict[str, Any]:
    items = sorted(valid_items(rows, predictor), key=lambda item: item[1], reverse=True)
    accepted = errors = cursor = 0
    feasible: list[dict[str, Any]] = []
    while cursor < len(items):
        threshold = items[cursor][1]
        while cursor < len(items) and items[cursor][1] == threshold:
            accepted += 1
            errors += int(not items[cursor][2])
            cursor += 1
        risk = errors / accepted
        if risk <= target_risk:
            feasible.append(
                {
                    "threshold": threshold,
                    "calibration_n": len(items),
                    "accepted": accepted,
                    "coverage": accepted / len(items),
                    "risk": risk,
                    "accepted_record_ids_sha256": record_id_sha256(
                        items[:cursor]
                    ),
                }
            )
    if not feasible:
        return {
            "threshold": None,
            "calibration_n": len(items),
            "accepted": 0,
            "coverage": 0.0 if items else None,
            "risk": None,
            "accepted_record_ids_sha256": record_id_sha256([]),
        }
    return max(feasible, key=lambda row: (row["coverage"], -row["threshold"]))


def threshold_transfer(
    rows: list[AuditRow],
    predictor: str,
    rule: Mapping[str, Any],
) -> dict[str, Any]:
    items = valid_items(rows, predictor)
    threshold = finite_float(rule.get("threshold"))
    accepted = (
        [item for item in items if item[1] >= threshold]
        if threshold is not None
        else []
    )
    errors = sum(not item[2] for item in accepted)
    return {
        "n": len(items),
        "accepted": len(accepted),
        "errors": errors,
        "coverage": len(accepted) / len(items) if items else None,
        "risk": (
            errors / len(accepted)
            if accepted
            else None
        ),
        "accepted_record_ids_sha256": record_id_sha256(accepted),
    }


def question_disjoint_threshold_transfer(
    rows: list[AuditRow],
    predictor: str,
    rules_by_fold: Mapping[int, Mapping[str, Any]],
    fold_by_question: Mapping[str, int],
) -> dict[str, Any]:
    """Apply to each row the rule trained without that row's question fold."""
    items = valid_items(rows, predictor)
    accepted: list[tuple[AuditRow, float, bool]] = []
    for item in items:
        question_id = item[0].question_id
        if question_id not in fold_by_question:
            raise ValueError(
                f"primary question {question_id!r} is missing a fold assignment"
            )
        fold = fold_by_question[question_id]
        if fold not in rules_by_fold:
            raise ValueError(f"no threshold rule for fold {fold}")
        threshold = finite_float(rules_by_fold[fold].get("threshold"))
        if threshold is not None and item[1] >= threshold:
            accepted.append(item)
    errors = sum(not item[2] for item in accepted)
    return {
        "n": len(items),
        "accepted": len(accepted),
        "errors": errors,
        "coverage": len(accepted) / len(items) if items else None,
        "risk": errors / len(accepted) if accepted else None,
        "accepted_record_ids_sha256": record_id_sha256(accepted),
        "finite_fold_rules": sum(
            finite_float(rule.get("threshold")) is not None
            for rule in rules_by_fold.values()
        ),
        "fold_thresholds": {
            str(fold): finite_float(rule.get("threshold"))
            for fold, rule in sorted(rules_by_fold.items())
        },
    }


def assign_question_folds(
    question_ids: Iterable[str],
    *,
    n_folds: int,
    seed: int,
) -> dict[str, int]:
    values = sorted(set(question_ids), key=lambda value: (len(value), value))
    random.Random(seed).shuffle(values)
    return {value: index % n_folds for index, value in enumerate(values)}


def fit_isotonic(
    items: list[tuple[AuditRow, float, bool]],
) -> list[tuple[float, float, float]]:
    grouped: list[list[float]] = []
    for _, score, correct in sorted(items, key=lambda item: item[1]):
        if grouped and grouped[-1][1] == score:
            grouped[-1][3] += 1.0
            grouped[-1][4] += float(correct)
        else:
            grouped.append([score, score, score, 1.0, float(correct)])
    blocks: list[list[float]] = []
    for group in grouped:
        blocks.append(group)
        while len(blocks) >= 2:
            left, right = blocks[-2], blocks[-1]
            if left[4] / left[3] <= right[4] / right[3]:
                break
            blocks[-2:] = [
                [
                    left[0],
                    right[1],
                    right[2],
                    left[3] + right[3],
                    left[4] + right[4],
                ]
            ]
    return [(block[0], block[1], block[4] / block[3]) for block in blocks]


def isotonic_predict(
    blocks: list[tuple[float, float, float]],
    score: float,
) -> float:
    if not blocks:
        raise ValueError("isotonic model has no blocks")
    for _, high, value in blocks:
        if score <= high:
            return value
    return blocks[-1][2]


def median(values: Iterable[float | None]) -> float | None:
    present = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return statistics.median(present) if present else None


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for predictor in PREDICTORS:
        selected = [row for row in rows if row["predictor"] == predictor]
        result[predictor] = {
            "groups": len(selected),
            "median_availability": median(row["availability"] for row in selected),
            "median_raw_score_ece_10bin": median(
                row["raw_score_ece_10bin"] for row in selected
            ),
            "median_raw_score_brier": median(
                row["raw_score_brier"] for row in selected
            ),
            "median_aurc": median(row["aurc"] for row in selected),
            "median_correctness_auroc": median(
                row["correctness_auroc"] for row in selected
            ),
            "min_unique_score_levels": min(
                (row["unique_score_levels"] for row in selected), default=0
            ),
            "max_unique_score_levels": max(
                (row["unique_score_levels"] for row in selected), default=0
            ),
        }
    return result


def aggregate_rules(
    rows: list[dict[str, Any]],
    *,
    targets: tuple[float, ...],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for target in targets:
        result[str(target)] = {}
        for predictor in PREDICTORS:
            selected = [
                row
                for row in rows
                if row["predictor"] == predictor
                and row["target_risk"] == target
            ]
            finite = [row for row in selected if row["threshold"] is not None]
            result[str(target)][predictor] = {
                "rules": len(selected),
                "finite_rules": len(finite),
                "median_calibration_coverage": median(
                    row["coverage"] for row in selected
                ),
                "median_calibration_risk_finite": median(
                    row["risk"] for row in finite
                ),
            }
    return result


def aggregate_cells(
    rows: list[dict[str, Any]],
    *,
    targets: tuple[float, ...],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for target in targets:
        result[str(target)] = {}
        for predictor in PREDICTORS:
            selected = [
                row
                for row in rows
                if row["predictor"] == predictor and row["target_risk"] == target
            ]
            nonzero = [row for row in selected if row["accepted"] > 0]
            accepted_total = sum(row["accepted"] for row in selected)
            error_total = sum(row["errors"] for row in selected)
            result[str(target)][predictor] = {
                "cells": len(selected),
                "evaluated_n": sum(row["n"] for row in selected),
                "nonzero_coverage_cells": len(nonzero),
                "zero_coverage_cells": len(selected) - len(nonzero),
                "cells_risk_at_or_below_target": sum(
                    row["risk"] is not None and row["risk"] <= target
                    for row in selected
                ),
                "accepted": accepted_total,
                "errors": error_total,
                "pooled_risk_when_nonzero": (
                    error_total / accepted_total if accepted_total else None
                ),
                "median_coverage_all_cells": median(
                    row["coverage"] for row in selected
                ),
                "median_coverage_nonzero_cells": median(
                    row["coverage"] for row in nonzero
                ),
                "median_risk_nonzero_cells": median(row["risk"] for row in nonzero),
            }
    return result


def _is_primary_stage_b(row: Mapping[str, Any]) -> bool:
    return (
        str(row.get("routing_split") or "") == "test"
        and str(row.get("cue_congruency") or row.get("direction") or "").lower()
        == "incongruent"
    )


def _optional_csv_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _float_equal(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-15)


def _audit_value_equal(left: Any, right: Any) -> bool:
    if isinstance(left, float) or isinstance(right, float):
        try:
            return _float_equal(
                None if left is None else float(left),
                None if right is None else float(right),
            )
        except (TypeError, ValueError):
            return False
    return left == right


def validate_lm_row_score_equivalence(rows: Iterable[AuditRow]) -> dict[str, Any]:
    """Require the two K=4 exact-label scores to induce identical tie groups."""

    observed: Counter[tuple[float, float]] = Counter()
    row_count = 0
    for row in rows:
        semantic_entropy = finite_float(
            row.scores.get(LM_SEMANTIC_ENTROPY_PREDICTOR)
        )
        degree_matrix = finite_float(row.scores.get(LM_DEGREE_MATRIX_PREDICTOR))
        if semantic_entropy is None or degree_matrix is None:
            if semantic_entropy is not degree_matrix:
                raise ValueError("LM-Polygraph predictor availability differs")
            continue
        observed[(semantic_entropy, degree_matrix)] += 1
        row_count += 1
    ordered = sorted(observed)
    if len(ordered) > 4:
        raise ValueError(
            "four exact-label repeats should yield at most four score partitions"
        )
    if any(
        left[1] >= right[1]
        for left, right in zip(ordered, ordered[1:], strict=False)
    ):
        raise ValueError(
            "Semantic Entropy and Degree Matrix do not induce identical rankings"
        )
    return {
        "passed": True,
        "rows": row_count,
        "observed_tie_groups": len(ordered),
        "score_mapping": [
            {
                LM_SEMANTIC_ENTROPY_PREDICTOR: semantic_entropy,
                LM_DEGREE_MATRIX_PREDICTOR: degree_matrix,
                "rows": observed[(semantic_entropy, degree_matrix)],
            }
            for semantic_entropy, degree_matrix in reversed(ordered)
        ],
    }


def validate_lm_result_equivalence(
    rows: list[dict[str, Any]],
    *,
    label: str,
    key_fields: tuple[str, ...],
    value_fields: tuple[str, ...],
) -> dict[str, Any]:
    """Compare selection/ranking outputs while excluding raw score scale."""

    selected = {
        predictor: [row for row in rows if row["predictor"] == predictor]
        for predictor in (
            LM_SEMANTIC_ENTROPY_PREDICTOR,
            LM_DEGREE_MATRIX_PREDICTOR,
        )
    }
    indexed: dict[str, dict[tuple[Any, ...], dict[str, Any]]] = {}
    for predictor, predictor_rows in selected.items():
        by_key = {
            tuple(row[field] for field in key_fields): row
            for row in predictor_rows
        }
        if len(by_key) != len(predictor_rows):
            raise ValueError(
                f"duplicate {predictor} equivalence keys in {label}"
            )
        indexed[predictor] = by_key
    left = indexed[LM_SEMANTIC_ENTROPY_PREDICTOR]
    right = indexed[LM_DEGREE_MATRIX_PREDICTOR]
    mismatches: list[str] = []
    if set(left) != set(right):
        mismatches.append("key sets differ")
    for key in sorted(set(left) & set(right), key=repr):
        for field in value_fields:
            if not _audit_value_equal(left[key].get(field), right[key].get(field)):
                mismatches.append(
                    f"{key!r} {field}: {left[key].get(field)!r} != "
                    f"{right[key].get(field)!r}"
                )
    if mismatches:
        raise ValueError(
            f"LM-Polygraph equivalence regression failed for {label}: "
            + "; ".join(mismatches[:5])
        )
    return {"label": label, "passed": True, "groups": len(left)}


def validate_msp_primary_regression(
    primary_cells: list[dict[str, Any]],
    oracle_path: Path,
) -> dict[str, Any]:
    observed_rows = [
        row
        for row in primary_cells
        if row["predictor"] == "msp"
        and math.isclose(row["target_risk"], 0.10)
    ]
    observed = {
        (row["model_name"], row["ordering"], row["family"]): row
        for row in observed_rows
    }
    with oracle_path.open(newline="") as handle:
        oracle_rows = [
            row
            for row in csv.DictReader(handle)
            if str(row.get("primary") or "").strip().lower() == "true"
        ]
    expected = {
        (row["model_name"], row["ordering"], row["family"]): row
        for row in oracle_rows
    }
    mismatches: list[str] = []
    if len(observed_rows) != len(observed):
        mismatches.append("duplicate observed MSP primary keys")
    if len(oracle_rows) != len(expected):
        mismatches.append("duplicate oracle MSP primary keys")
    if set(observed) != set(expected):
        mismatches.append(
            "primary key sets differ: "
            f"observed_only={sorted(set(observed) - set(expected))!r}; "
            f"oracle_only={sorted(set(expected) - set(observed))!r}"
        )
    for key in sorted(set(observed) & set(expected)):
        actual = observed[key]
        oracle = expected[key]
        integer_fields = {
            "calibration_n": (actual["rule_calibration_n"], oracle["calibration_n"]),
            "test_n": (actual["n"], oracle["test_n"]),
            "test_accepted": (actual["accepted"], oracle["test_accepted"]),
        }
        for name, (actual_value, expected_value) in integer_fields.items():
            if int(actual_value) != int(expected_value):
                mismatches.append(
                    f"{key!r} {name}: {actual_value!r} != {expected_value!r}"
                )
        float_fields = {
            "dose": (
                actual["dose"],
                _optional_csv_float(oracle["dose"]),
            ),
            "threshold": (
                actual["rule_threshold"],
                _optional_csv_float(oracle["threshold"]),
            ),
            "calibration_coverage": (
                actual["rule_calibration_coverage"],
                _optional_csv_float(oracle["calibration_coverage"]),
            ),
            "calibration_risk": (
                actual["rule_calibration_risk"],
                _optional_csv_float(oracle["calibration_risk"]),
            ),
            "test_coverage": (
                actual["coverage"],
                _optional_csv_float(oracle["test_coverage"]),
            ),
            "test_realized_risk": (
                actual["risk"],
                _optional_csv_float(oracle["test_realized_risk"]),
            ),
        }
        for name, (actual_value, expected_value) in float_fields.items():
            if not _float_equal(actual_value, expected_value):
                mismatches.append(
                    f"{key!r} {name}: {actual_value!r} != {expected_value!r}"
                )
    result = {
        "performed": True,
        "passed": not mismatches,
        "oracle_sha256": file_sha256(oracle_path),
        "expected_cells": len(expected),
        "observed_cells": len(observed),
        "zero_coverage_cells": sum(
            row["accepted"] == 0 for row in observed.values()
        ),
        "accepted": sum(row["accepted"] for row in observed.values()),
        "errors": sum(row["errors"] for row in observed.values()),
        "mismatches": mismatches,
    }
    if mismatches:
        raise ValueError(
            "MSP regression against the published strict-v3 primary failed: "
            + "; ".join(mismatches[:5])
        )
    return result


def validate_published_input_hashes(
    input_hashes: Mapping[str, Mapping[str, str]],
    provenance_path: Path,
) -> dict[str, Any]:
    provenance = json.loads(provenance_path.read_text())
    expected_stage_a = set(provenance["stage_a_input_hashes"])
    expected_stage_b = set(provenance["stage_b_input_hashes"])
    observed_stage_a = {
        hashes[STAGE_A_FILENAME] for hashes in input_hashes.values()
    }
    observed_stage_b = {
        hashes[STAGE_B_FILENAME] for hashes in input_hashes.values()
    }
    passed = (
        observed_stage_a == expected_stage_a
        and observed_stage_b == expected_stage_b
    )
    result = {
        "performed": True,
        "passed": passed,
        "provenance_sha256": file_sha256(provenance_path),
        "analysis_version": provenance.get("analysis_version"),
        "spec_hash": provenance.get("spec_hash"),
        "stage_a_hashes_match": observed_stage_a == expected_stage_a,
        "stage_b_hashes_match": observed_stage_b == expected_stage_b,
    }
    if not passed:
        raise ValueError("raw score hashes do not match published strict-v3 provenance")
    return result


def discover_model_directories(campaign_root: Path) -> tuple[Path, ...]:
    result = tuple(
        sorted(
            path
            for path in campaign_root.iterdir()
            if path.is_dir()
            and (path / STAGE_A_FILENAME).is_file()
            and (path / STAGE_B_FILENAME).is_file()
            and (path / MODEL_MARKER_FILENAME).is_file()
        )
    )
    if not result:
        raise ValueError(f"no model score directories found under {campaign_root}")
    return result


def validate_output_path(
    campaign_root: Path,
    output: Path,
    *,
    additional_immutable_roots: Iterable[Path] = (),
) -> None:
    immutable_roots = (campaign_root, *additional_immutable_roots)
    if any(
        output.resolve().is_relative_to(root.resolve()) for root in immutable_roots
    ):
        raise ValueError("output must be outside every immutable input root")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")


def run_audit(
    campaign_root: Path,
    *,
    targets: tuple[float, ...] = (0.10, 0.20),
    n_folds: int = 5,
    seed: int = 42,
    published_provenance: Path | None = None,
    published_msp_oracle: Path | None = None,
) -> dict[str, Any]:
    calibration_metrics: list[dict[str, Any]] = []
    primary_metrics: list[dict[str, Any]] = []
    rules: list[dict[str, Any]] = []
    clean_test_transfers: list[dict[str, Any]] = []
    primary_cells: list[dict[str, Any]] = []
    oof_clean: list[dict[str, Any]] = []
    question_disjoint_primary_cells: list[dict[str, Any]] = []
    oof_isotonic: list[dict[str, Any]] = []
    lm_rows: list[AuditRow] = []
    structure: dict[str, Any] = {}
    input_hashes: dict[str, Any] = {}

    for model_dir in discover_model_directories(campaign_root):
        stage_a_path = model_dir / STAGE_A_FILENAME
        stage_b_path = model_dir / STAGE_B_FILENAME
        marker_path = model_dir / MODEL_MARKER_FILENAME
        marker = json.loads(marker_path.read_text())
        if not isinstance(marker, Mapping):
            raise ValueError(f"{marker_path}: marker must be an object")
        model_name = str(marker["model_name"])
        if model_name in structure:
            raise ValueError(f"duplicate model marker for {model_name!r}")
        stage_a = read_jsonl(stage_a_path)
        stage_b_candidates = read_jsonl(
            stage_b_path,
            predicate=_is_primary_stage_b,
        )
        observed_models = {
            row.model_name for row in [*stage_a, *stage_b_candidates]
        }
        if observed_models != {model_name}:
            raise ValueError(
                f"{model_dir}: row model names {observed_models!r} do not "
                f"match marker model {model_name!r}"
            )
        max_doses = {
            family: max(
                float(row.dose)
                for row in stage_b_candidates
                if row.family == family and row.dose is not None
            )
            for family in ("authority", "bandwagon")
        }
        primary_with_ties = [
            row
            for row in stage_b_candidates
            if row.family in max_doses and row.dose == max_doses[row.family]
        ]
        lm_rows.extend(stage_a)
        lm_rows.extend(primary_with_ties)
        stage_a_pairing = attach_order_vote_scores(stage_a)
        primary_pairing = attach_order_vote_scores(primary_with_ties)
        for label, diagnostics in (
            ("Stage A", stage_a_pairing),
            ("primary Stage B", primary_pairing),
        ):
            invalid = (
                diagnostics["malformed_pairs"]
                + diagnostics["pairs_missing_counts"]
                + diagnostics["canonical_human_disagreements"]
                + diagnostics["unequal_repeat_count_pairs"]
            )
            if invalid:
                raise ValueError(
                    f"{model_name} {label} order pairing failed: "
                    f"{diagnostics!r}"
                )
        primary = [row for row in primary_with_ties if not row.clean_tie]
        fold_by_question = assign_question_folds(
            (row.question_id for row in stage_a),
            n_folds=n_folds,
            seed=seed,
        )
        run_counts = Counter(
            sum(row.verdict_counts.values())
            for row in [*stage_a, *primary_with_ties]
            if row.verdict_counts is not None
        )
        structure[model_name] = {
            "model_directory": model_dir.name,
            "stage_a_rows": len(stage_a),
            "primary_test_rows_with_ties": len(primary_with_ties),
            "primary_test_rows_without_clean_ties": len(primary),
            "max_doses": max_doses,
            "consistency_run_counts": dict(sorted(run_counts.items())),
            "stage_a_order_pairing": stage_a_pairing,
            "primary_order_pairing": primary_pairing,
        }
        input_hashes[model_name] = {
            STAGE_A_FILENAME: file_sha256(stage_a_path),
            STAGE_B_FILENAME: file_sha256(stage_b_path),
            MODEL_MARKER_FILENAME: file_sha256(marker_path),
        }

        for ordering in ("ab", "ba"):
            calibration = [
                row
                for row in stage_a
                if row.routing_split == "calibration" and row.ordering == ordering
            ]
            clean_test = [
                row
                for row in stage_a
                if row.routing_split == "test" and row.ordering == ordering
            ]
            for predictor in PREDICTORS:
                summary = metric_summary(calibration, predictor)
                summary.update(
                    model_name=model_name,
                    ordering=ordering,
                    predictor=predictor,
                )
                calibration_metrics.append(summary)

                fold_rules: dict[tuple[float, int], dict[str, Any]] = {}
                isotonic_predictions: list[tuple[AuditRow, float, bool]] = []
                for fold in range(n_folds):
                    train = [
                        row
                        for row in calibration
                        if fold_by_question[row.question_id] != fold
                    ]
                    heldout = [
                        row
                        for row in calibration
                        if fold_by_question[row.question_id] == fold
                    ]
                    train_items = valid_items(train, predictor)
                    heldout_items = valid_items(heldout, predictor)
                    blocks = fit_isotonic(train_items)
                    if blocks:
                        isotonic_predictions.extend(
                            (row, isotonic_predict(blocks, score), correct)
                            for row, score, correct in heldout_items
                        )
                    for target in targets:
                        fold_rules[(target, fold)] = threshold_rule(
                            train,
                            predictor,
                            target,
                        )
                oof_isotonic.append(
                    {
                        "model_name": model_name,
                        "ordering": ordering,
                        "predictor": predictor,
                        "n": len(isotonic_predictions),
                        "ece_10bin": expected_calibration_error(isotonic_predictions),
                        "brier": (
                            sum(
                                (score - float(correct)) ** 2
                                for _, score, correct in isotonic_predictions
                            )
                            / len(isotonic_predictions)
                            if isotonic_predictions
                            else None
                        ),
                    }
                )

                for target in targets:
                    rule = threshold_rule(calibration, predictor, target)
                    rules.append(
                        {
                            "model_name": model_name,
                            "ordering": ordering,
                            "predictor": predictor,
                            "target_risk": target,
                            **rule,
                        }
                    )
                    clean_test_transfers.append(
                        {
                            "model_name": model_name,
                            "ordering": ordering,
                            "family": "clean",
                            "predictor": predictor,
                            "target_risk": target,
                            **threshold_transfer(
                                clean_test,
                                predictor,
                                rule,
                            ),
                        }
                    )
                    heldout_items = valid_items(calibration, predictor)
                    heldout_accepted = [
                        item
                        for item in heldout_items
                        if (
                            fold_rules[
                                (target, fold_by_question[item[0].question_id])
                            ]["threshold"]
                            is not None
                            and item[1]
                            >= fold_rules[
                                (target, fold_by_question[item[0].question_id])
                            ]["threshold"]
                        )
                    ]
                    oof_clean.append(
                        {
                            "model_name": model_name,
                            "ordering": ordering,
                            "predictor": predictor,
                            "target_risk": target,
                            "n": len(heldout_items),
                            "accepted": len(heldout_accepted),
                            "errors": sum(not item[2] for item in heldout_accepted),
                            "coverage": (
                                len(heldout_accepted) / len(heldout_items)
                                if heldout_items
                                else None
                            ),
                            "risk": (
                                sum(not item[2] for item in heldout_accepted)
                                / len(heldout_accepted)
                                if heldout_accepted
                                else None
                            ),
                            "accepted_record_ids_sha256": record_id_sha256(
                                heldout_accepted
                            ),
                            "finite_fold_rules": sum(
                                fold_rules[(target, fold)]["threshold"] is not None
                                for fold in range(n_folds)
                            ),
                            "fold_thresholds": [
                                fold_rules[(target, fold)]["threshold"]
                                for fold in range(n_folds)
                            ],
                        }
                    )

                    for family in ("authority", "bandwagon"):
                        test = [
                            row
                            for row in primary
                            if row.ordering == ordering and row.family == family
                        ]
                        if target == targets[0]:
                            test_metrics = metric_summary(test, predictor)
                            test_metrics.update(
                                model_name=model_name,
                                ordering=ordering,
                                family=family,
                                direction="incongruent",
                                dose=max_doses[family],
                                predictor=predictor,
                            )
                            primary_metrics.append(test_metrics)
                        transfer = threshold_transfer(test, predictor, rule)
                        primary_cells.append(
                            {
                                "model_name": model_name,
                                "ordering": ordering,
                                "family": family,
                                "direction": "incongruent",
                                "dose": max_doses[family],
                                "clean_tie": False,
                                "routing_split": "test",
                                "predictor": predictor,
                                "target_risk": target,
                                "rule_threshold": rule["threshold"],
                                "rule_calibration_n": rule["calibration_n"],
                                "rule_calibration_coverage": rule["coverage"],
                                "rule_calibration_risk": rule["risk"],
                                **transfer,
                            }
                        )
                        question_disjoint_primary_cells.append(
                            {
                                "model_name": model_name,
                                "ordering": ordering,
                                "family": family,
                                "direction": "incongruent",
                                "dose": max_doses[family],
                                "clean_tie": False,
                                "routing_split": "test",
                                "predictor": predictor,
                                "target_risk": target,
                                **question_disjoint_threshold_transfer(
                                    test,
                                    predictor,
                                    {
                                        fold: fold_rules[(target, fold)]
                                        for fold in range(n_folds)
                                    },
                                    fold_by_question,
                                ),
                            }
                        )

    observed_consistency_runs = {
        int(run_count)
        for model in structure.values()
        for run_count in model["consistency_run_counts"]
    }
    if len(observed_consistency_runs) != 1:
        raise ValueError(
            "expected one consistency repeat count across the audited rows; "
            f"observed {sorted(observed_consistency_runs)!r}"
        )
    if observed_consistency_runs != {4}:
        raise ValueError(
            "the LM-Polygraph score-equivalence gate requires exactly four "
            f"repeats per row; observed {sorted(observed_consistency_runs)!r}"
        )
    lm_equivalence_regression = {
        "passed": True,
        "row_scores": validate_lm_row_score_equivalence(lm_rows),
        "result_sections": [
            validate_lm_result_equivalence(
                calibration_metrics,
                label="calibration ranking",
                key_fields=("model_name", "ordering"),
                value_fields=(
                    "total",
                    "n",
                    "availability",
                    "accuracy",
                    "aurc",
                    "correctness_auroc",
                    "unique_score_levels",
                ),
            ),
            validate_lm_result_equivalence(
                primary_metrics,
                label="primary ranking",
                key_fields=("model_name", "ordering", "family", "dose"),
                value_fields=(
                    "total",
                    "n",
                    "availability",
                    "accuracy",
                    "aurc",
                    "correctness_auroc",
                    "unique_score_levels",
                ),
            ),
            validate_lm_result_equivalence(
                rules,
                label="full clean rules",
                key_fields=("model_name", "ordering", "target_risk"),
                value_fields=(
                    "calibration_n",
                    "accepted",
                    "coverage",
                    "risk",
                    "accepted_record_ids_sha256",
                ),
            ),
            validate_lm_result_equivalence(
                clean_test_transfers,
                label="clean-test transfer",
                key_fields=("model_name", "ordering", "target_risk"),
                value_fields=(
                    "n",
                    "accepted",
                    "errors",
                    "coverage",
                    "risk",
                    "accepted_record_ids_sha256",
                ),
            ),
            validate_lm_result_equivalence(
                primary_cells,
                label="primary transfer",
                key_fields=(
                    "model_name",
                    "ordering",
                    "family",
                    "dose",
                    "target_risk",
                ),
                value_fields=(
                    "rule_calibration_n",
                    "rule_calibration_coverage",
                    "rule_calibration_risk",
                    "n",
                    "accepted",
                    "errors",
                    "coverage",
                    "risk",
                    "accepted_record_ids_sha256",
                ),
            ),
            validate_lm_result_equivalence(
                oof_clean,
                label="question-disjoint clean transfer",
                key_fields=("model_name", "ordering", "target_risk"),
                value_fields=(
                    "n",
                    "accepted",
                    "errors",
                    "coverage",
                    "risk",
                    "finite_fold_rules",
                    "accepted_record_ids_sha256",
                ),
            ),
            validate_lm_result_equivalence(
                question_disjoint_primary_cells,
                label="question-disjoint primary transfer",
                key_fields=(
                    "model_name",
                    "ordering",
                    "family",
                    "dose",
                    "target_risk",
                ),
                value_fields=(
                    "n",
                    "accepted",
                    "errors",
                    "coverage",
                    "risk",
                    "finite_fold_rules",
                    "accepted_record_ids_sha256",
                ),
            ),
            validate_lm_result_equivalence(
                oof_isotonic,
                label="question-disjoint isotonic calibration",
                key_fields=("model_name", "ordering"),
                value_fields=("n", "ece_10bin", "brier"),
            ),
        ],
    }
    provenance_regression = (
        validate_published_input_hashes(input_hashes, published_provenance)
        if published_provenance is not None
        else {"performed": False, "passed": None}
    )
    msp_regression = (
        validate_msp_primary_regression(primary_cells, published_msp_oracle)
        if published_msp_oracle is not None
        else {"performed": False, "passed": None}
    )
    if published_msp_oracle is not None and (
        msp_regression["observed_cells"],
        msp_regression["zero_coverage_cells"],
        msp_regression["accepted"],
        msp_regression["errors"],
    ) != (16, 10, 141, 8):
        raise ValueError(
            "published MSP summary regression failed; expected "
            "16 cells, 10 zero-coverage cells, 141 accepted, and 8 errors"
        )
    published_regressions_passed = bool(
        provenance_regression.get("performed")
        and provenance_regression.get("passed")
        and msp_regression.get("performed")
        and msp_regression.get("passed")
    )

    return {
        "schema_version": 1,
        "analysis": (
            "standalone exploratory repeatability audit; "
            + (
                "strict-v3 MSP primary and published zero-coverage result "
                "regression-verified unchanged"
                if published_regressions_passed
                else "published strict-v3 preservation was not regression-verified"
            )
        ),
        "predictors": PREDICTORS,
        "combined_predictors": [],
        "primary_status": (
            "exploratory only; "
            + (
                "MSP remains the strict-v3 primary and its published "
                "zero-coverage outcome is regression-verified preserved"
                if published_regressions_passed
                else "supply both published oracles before making preservation claims"
            )
        ),
        "published_regressions_passed": published_regressions_passed,
        "predictor_semantics": {
            "msp": "deterministic constrained-label MSP baseline",
            "anchor_reproducibility": (
                "fraction of same-order stochastic A/B/tie consistency samples "
                "equal to the temperature-zero deterministic anchor; this is "
                "cross-decoding anchor reproduction, not identical-configuration "
                "rerun agreement"
            ),
            "frequency_semantic_entropy_confidence": (
                "one minus frequency Semantic Entropy, normalized by log(3), "
                "over exact same-order A/B/tie repeat classes"
            ),
            "degree_matrix_agreement": (
                "sum of squared exact same-order A/B/tie repeat frequencies; "
                "the confidence complement of categorical Degree-Matrix "
                "disagreement"
            ),
            "order_vote_js_similarity": (
                "one minus normalized JSD between AB and canonicalized BA repeat "
                "vote distributions"
            ),
            "order_vote_tv_similarity": (
                "one minus total-variation distance between AB and canonicalized "
                "BA repeat vote distributions"
            ),
            "order_vote_expected_agreement": (
                "probability that independent AB and canonicalized BA repeat votes agree"
            ),
        },
        "tie_policy": "A/B/tie are three distinct classes; BA swaps A/B and preserves tie",
        "order_pair_policy": (
            "AB/BA rows must share pair identity, condition group, family, "
            "direction, dose, split, model, canonical human label, canonical "
            "cue target, and repeat count; canonical cue-target mismatches are "
            "reported as unavailable rather than scored"
        ),
        "score_direction": "higher means more confident or repeatable",
        "lm_polygraph_mapping": {
            "repository": "https://github.com/IINemo/lm-polygraph",
            "commit": "98dd675cc43e0f5da654c29940872ea913aea2bf",
            "frequency_semantic_entropy": (
                "literal A/B/tie labels are treated as exact semantic classes; "
                "reported score is 1 - H(p)/log(3)"
            ),
            "degree_matrix": (
                "exact-label categorical specialization; reported score is "
                "1 - (1 - sum_c p_c^2) = sum_c p_c^2"
            ),
            "new_model_inference": False,
        },
        "lm_polygraph_equivalence_regression": lm_equivalence_regression,
        "targets": targets,
        "question_disjoint_folds": n_folds,
        "seed": seed,
        "observed_consistency_runs_per_row": next(
            iter(observed_consistency_runs)
        ),
        "implementation_sha256": {
            Path(__file__).name: file_sha256(Path(__file__)),
            Path(anchor_reproducibility.__code__.co_filename).name: file_sha256(
                Path(anchor_reproducibility.__code__.co_filename)
            ),
            Path(normalize_label.__code__.co_filename).name: file_sha256(
                Path(normalize_label.__code__.co_filename)
            ),
        },
        "structure": structure,
        "input_sha256": input_hashes,
        "published_input_regression": provenance_regression,
        "published_msp_primary_regression": msp_regression,
        "calibration_aggregate": aggregate_metrics(calibration_metrics),
        "primary_ranking_aggregate": aggregate_metrics(primary_metrics),
        "full_rule_aggregate": aggregate_rules(rules, targets=targets),
        "clean_test_transfer_aggregate": aggregate_cells(
            clean_test_transfers,
            targets=targets,
        ),
        "full_primary_cell_aggregate": aggregate_cells(
            primary_cells,
            targets=targets,
        ),
        "question_disjoint_primary_transfer_aggregate": aggregate_cells(
            question_disjoint_primary_cells,
            targets=targets,
        ),
        "question_disjoint_clean_aggregate": {
            str(target): {
                predictor: {
                    "groups": len(
                        [
                            row
                            for row in oof_clean
                            if row["target_risk"] == target
                            and row["predictor"] == predictor
                        ]
                    ),
                    "accepted": sum(
                        row["accepted"]
                        for row in oof_clean
                        if row["target_risk"] == target
                        and row["predictor"] == predictor
                    ),
                    "median_coverage": median(
                        row["coverage"]
                        for row in oof_clean
                        if row["target_risk"] == target
                        and row["predictor"] == predictor
                    ),
                    "median_risk": median(
                        row["risk"]
                        for row in oof_clean
                        if row["target_risk"] == target
                        and row["predictor"] == predictor
                    ),
                    "finite_fold_rules": sum(
                        row["finite_fold_rules"]
                        for row in oof_clean
                        if row["target_risk"] == target
                        and row["predictor"] == predictor
                    ),
                }
                for predictor in PREDICTORS
            }
            for target in targets
        },
        "question_disjoint_isotonic_aggregate": {
            predictor: {
                "groups": len(
                    [row for row in oof_isotonic if row["predictor"] == predictor]
                ),
                "median_ece_10bin": median(
                    row["ece_10bin"]
                    for row in oof_isotonic
                    if row["predictor"] == predictor
                ),
                "median_brier": median(
                    row["brier"]
                    for row in oof_isotonic
                    if row["predictor"] == predictor
                ),
            }
            for predictor in PREDICTORS
        },
        "calibration_by_model_order": calibration_metrics,
        "primary_ranking_by_cell": primary_metrics,
        "rules": rules,
        "clean_test_transfers": clean_test_transfers,
        "primary_cells": primary_cells,
        "question_disjoint_clean": oof_clean,
        "question_disjoint_primary_cells": question_disjoint_primary_cells,
        "question_disjoint_isotonic": oof_isotonic,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-root",
        type=Path,
        required=True,
        help="Directory containing one immutable score directory per model.",
    )
    parser.add_argument(
        "--target-risk",
        type=float,
        action="append",
        dest="target_risks",
        help="Empirical clean risk target; repeat for multiple targets.",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--published-provenance",
        type=Path,
        help=(
            "Optional published strict-v3 provenance.json. When supplied, "
            "the eight raw score hashes must match before a report is emitted."
        ),
    )
    parser.add_argument(
        "--published-msp-oracle",
        type=Path,
        help=(
            "Optional published strict-v3 rq2_threshold_transfer.csv. When "
            "supplied, all 16 primary MSP cells must reproduce exactly."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path. Defaults to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = tuple(args.target_risks or (0.10, 0.20))
    if not targets or any(not 0.0 <= target <= 1.0 for target in targets):
        raise ValueError("target risks must be in [0, 1]")
    if args.folds < 2:
        raise ValueError("folds must be at least two")
    if args.output is not None:
        immutable_roots = {
            path.resolve().parent
            for path in (args.published_provenance, args.published_msp_oracle)
            if path is not None
        }
        validate_output_path(
            args.campaign_root,
            args.output,
            additional_immutable_roots=immutable_roots,
        )
    report = run_audit(
        args.campaign_root,
        targets=targets,
        n_folds=args.folds,
        seed=args.seed,
        published_provenance=args.published_provenance,
        published_msp_oracle=args.published_msp_oracle,
    )
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        with args.output.open("x") as handle:
            handle.write(payload)


if __name__ == "__main__":
    main()
