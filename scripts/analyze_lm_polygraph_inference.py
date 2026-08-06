#!/usr/bin/env python3
"""Evaluate separately collected LM-Polygraph scores on strict-v3 populations.

This is a read-only, exploratory analysis.  It joins immutable strict-v3 score
rows to new-inference collector rows by ``record_id`` and evaluates MSP,
P(True), Mean Token Entropy, and Self-Certainty separately.  It never combines
predictors and never rewrites either input tree.

The collector contract is one directory per model containing:

* ``lm_polygraph_inference_scores.jsonl``
* ``lm_polygraph_inference_complete.json``

Collector rows contain the fields documented in ``collector_score_from_mapping``.
Mean Token Entropy and the pinned LM-Polygraph SelfCertainty output are
arbitrary finite uncertainty scores, not probabilities.  They are negated only
to establish the common higher-is-more-confident ordering used by selective
prediction.  Probability calibration is learned from clean rows with
question-disjoint out-of-fold isotonic regression.
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
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

STAGE_A_FILENAME = "silent_bias_stage_a_uncertainty_scores.jsonl"
STAGE_B_FILENAME = "silent_bias_stage_b_uncertainty_scores.jsonl"
MODEL_MARKER_FILENAME = "campaign_model_complete.json"
COLLECTOR_FILENAME = "lm_polygraph_inference_scores.jsonl"
COLLECTOR_MARKER_FILENAME = "lm_polygraph_inference_complete.json"
LM_POLYGRAPH_COMMIT = "98dd675cc43e0f5da654c29940872ea913aea2bf"
PINNED_MAX_DOSES: Mapping[str, float] = {
    "authority": 4.0,
    "bandwagon": 95.0,
}
PUBLISHED_MSP_PRIMARY_INVARIANTS: Mapping[str, int] = {
    "cells": 16,
    "zero_coverage_cells": 10,
    "accepted": 141,
    "errors": 8,
}
PREDICTORS = (
    "msp",
    "p_true",
    "mean_token_entropy",
    "self_certainty",
)

PRIMARY_ESTIMAND = "cross_engine_frozen_vllm_correctness_all_rows"
SAME_ENGINE_ESTIMAND = "same_engine_hf_replay_correctness_all_rows"
MAP_AGREE_ESTIMAND = "map_agree_frozen_vllm_correctness_sensitivity"
ESTIMANDS = (
    PRIMARY_ESTIMAND,
    SAME_ENGINE_ESTIMAND,
    MAP_AGREE_ESTIMAND,
)

ESTIMAND_DEFINITIONS: Mapping[str, Mapping[str, Any]] = {
    PRIMARY_ESTIMAND: {
        "role": "primary_cross_engine_transfer",
        "population": "all joined rows",
        "outcome": "frozen strict-v3 vLLM verdict equals human winner",
        "interpretation": (
            "HF-computed uncertainty predicts correctness of the frozen vLLM "
            "decision. This is the primary cross-engine transfer estimand."
        ),
        "selection_bias_caveat": None,
    },
    SAME_ENGINE_ESTIMAND: {
        "role": "supplemental_same_engine",
        "population": "all joined rows",
        "outcome": "HF restricted-simplex MAP verdict equals human winner",
        "interpretation": (
            "HF-computed uncertainty predicts correctness of the HF replay "
            "decision. This isolates same-engine association from cross-engine drift."
        ),
        "selection_bias_caveat": None,
        "predictor_specific_caveat": (
            "The published MSP is not evaluated because it is the frozen vLLM "
            "MSP rather than HF replay MSP. P(True) is not evaluated because its "
            "meta-prompt verifies the frozen vLLM verdict, not a disagreeing HF "
            "MAP verdict. Mean Token Entropy and Self-Certainty remain applicable."
        ),
    },
    MAP_AGREE_ESTIMAND: {
        "role": "sensitivity_only",
        "population": (
            "only rows whose HF restricted-simplex MAP verdict equals the frozen "
            "strict-v3 vLLM verdict"
        ),
        "outcome": "frozen strict-v3 vLLM verdict equals human winner",
        "interpretation": (
            "Sensitivity analysis restricted to rows with identical HF and frozen "
            "vLLM MAP decisions."
        ),
        "selection_bias_caveat": (
            "Conditioning on MAP agreement can remove low-margin, high-uncertainty "
            "rows and therefore inflate apparent ranking, calibration, or selective-"
            "prediction performance. It must not replace either all-row estimand."
        ),
    },
}


METHOD_MAPPING: Mapping[str, Mapping[str, Any]] = {
    "msp": {
        "source": "immutable strict-v3 constrained A/B/tie probabilities",
        "raw_field": "msp",
        "raw_direction": "higher_is_more_confident",
        "confidence_orientation": "identity",
        "native_probability": True,
        "caveat": (
            "MSP remains the published strict-v3 primary. Its native binary "
            "correctness calibration diagnostic is distinct from the published "
            "three-class probability calibration analysis."
        ),
    },
    "p_true": {
        "source": "LM-Polygraph PTrue",
        "raw_field": "p_true_log_probability",
        "raw_direction": "higher_is_more_confident",
        "confidence_orientation": "identity(log P(True))",
        "native_probability": True,
        "caveat": (
            "P(True) is the probability of the literal True token under the "
            "pinned self-verification prompt; it is evaluated as a correctness "
            "proxy and is not assumed calibrated a priori."
        ),
    },
    "mean_token_entropy": {
        "source": "LM-Polygraph MeanTokenEntropy specialized to one decision token",
        "raw_field": "mean_token_entropy",
        "raw_direction": "lower_is_more_confident",
        "confidence_orientation": "negative raw entropy",
        "native_probability": False,
        "caveat": (
            "Raw entropy is an unbounded statistic in nats. It is never divided "
            "by log(vocabulary size), clipped, or treated as a correctness probability."
        ),
    },
    "self_certainty": {
        "source": "LM-Polygraph SelfCertainty pinned estimator output",
        "raw_field": "self_certainty",
        "raw_direction": "lower_is_more_confident",
        "confidence_orientation": (
            "negative pinned output = KL(Uniform(vocabulary) || next-token distribution)"
        ),
        "native_probability": False,
        "caveat": (
            "The collector field is the pinned estimator output -KL, so it is "
            "negated for confidence ordering. Neither -KL nor KL is a correctness "
            "probability; only clean-trained isotonic predictions are calibrated."
        ),
    },
}


@dataclass(slots=True)
class AuditRow:
    """Minimal immutable-campaign row needed by this standalone analysis."""

    record_id: str
    question_id: str
    model_name: str
    ordering: str
    routing_split: str
    family: str
    direction: str
    dose: float | None
    clean_tie: bool
    human_winner: str
    verdict: str
    scores: dict[str, float | None] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CollectorScore:
    record_id: str
    model_name: str
    p_true_log_probability: float | None
    p_true_probability: float | None
    p_true_uncertainty: float | None
    mean_token_entropy: float | None
    self_certainty: float | None
    collector_msp: float | None
    verdict: str
    hf_restricted_label_probabilities: dict[str, float]
    hf_restricted_map_verdict: str
    hf_restricted_map_matches_stored: bool
    hf_restricted_msp: float
    hf_restricted_verdict_probability: float
    hf_source_probability_max_abs_difference: float


@dataclass(slots=True)
class AnalysisRow:
    campaign: AuditRow
    collector: CollectorScore
    raw_scores: dict[str, float | None]
    confidence_scores: dict[str, float | None]
    native_probabilities: dict[str, float | None]

    @property
    def record_id(self) -> str:
        return self.campaign.record_id

    @property
    def question_id(self) -> str:
        return self.campaign.question_id

    @property
    def correct(self) -> bool:
        """Frozen-vLLM correctness retained as the backward-compatible default."""

        return self.campaign.verdict == self.campaign.human_winner

    @property
    def hf_replay_correct(self) -> bool:
        return (
            self.collector.hf_restricted_map_verdict
            == self.campaign.human_winner
        )

    @property
    def hf_map_agrees_with_frozen(self) -> bool:
        return self.collector.hf_restricted_map_matches_stored


ScoredItem = tuple[AnalysisRow, float, bool]
ProbabilityItem = tuple[AnalysisRow, float, bool]


def normalize_label(value: Any) -> str | None:
    """Normalize the frozen strict-v3 A/B/tie label vocabulary."""

    if value is None:
        return None
    raw = getattr(value, "value", value)
    lowered = str(raw).strip().lower()
    if lowered in {"a", "answer_a", "model_a", "response_a"}:
        return "A"
    if lowered in {"b", "answer_b", "model_b", "response_b"}:
        return "B"
    if lowered in {"t", "tie", "c", "equal"}:
        return "tie"
    return None


def finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
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


def audit_row_from_mapping(row: Mapping[str, Any]) -> AuditRow:
    record_id = str(row.get("record_id") or "")
    question_id = str(row.get("question_id") or "")
    model_name = str(row.get("model_name") or "")
    if not record_id or not question_id or not model_name:
        raise ValueError("campaign row requires record_id, question_id, and model_name")
    ordering = str(row.get("ordering") or "").strip().lower()
    if ordering not in {"ab", "ba"}:
        raise ValueError("ordering must be ab or ba")
    routing_split = str(row.get("routing_split") or "")
    family = str(row.get("bias_name") or row.get("family") or "").strip().lower()
    direction = str(
        row.get("cue_congruency") or row.get("direction") or ""
    ).strip().lower()
    dose = finite_float(row.get("dose"))
    if row.get("dose") is not None and dose is None:
        raise ValueError("dose must be finite or null")
    verdict = normalize_label(row.get("verdict"))
    human_winner = normalize_label(row.get("human_winner"))
    if verdict is None or human_winner is None:
        raise ValueError("verdict and human_winner must be A, B, or tie")
    msp = finite_float(row.get("msp"))
    if msp is None or not 0.0 <= msp <= 1.0:
        raise ValueError("MSP must be finite and in [0, 1]")
    clean_tie_raw = row.get("clean_tie")
    if isinstance(clean_tie_raw, str):
        clean_tie = clean_tie_raw.strip().lower() in {"1", "true", "yes"}
    elif clean_tie_raw is None:
        clean_tie = family == "clean" and verdict == "tie"
    else:
        clean_tie = bool(clean_tie_raw)
    return AuditRow(
        record_id=record_id,
        question_id=question_id,
        model_name=model_name,
        ordering=ordering,
        routing_split=routing_split,
        family=family,
        direction=direction,
        dose=dose,
        clean_tie=clean_tie,
        human_winner=human_winner,
        verdict=verdict,
        scores={"msp": msp},
    )


def _optional_finite(row: Mapping[str, Any], field: str) -> float | None:
    if field not in row or row[field] is None:
        return None
    value = finite_float(row[field])
    if value is None:
        raise ValueError(f"{field} must be finite or null")
    return value


def _required_finite(row: Mapping[str, Any], field: str) -> float:
    value = _optional_finite(row, field)
    if value is None:
        raise ValueError(f"{field} must be present and finite")
    return value


def _nested(row: Mapping[str, Any], *keys: str) -> Any:
    value: Any = row
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def collector_score_from_mapping(row: Mapping[str, Any]) -> CollectorScore:
    """Parse and cross-check one canonical new-inference collector row.

    Canonical scalar fields are ``p_true_log_probability``,
    ``p_true_probability``, ``p_true_uncertainty``, ``mean_token_entropy``, and
    ``self_certainty``.  ``self_certainty`` is specifically LM-Polygraph's
    returned uncertainty value ``-KL`` rather than the positive KL quantity.
    All method fields may be null only to represent an explicitly unavailable
    score; non-finite numeric values are rejected.
    """

    record_id = str(row.get("record_id") or "")
    model_name = str(row.get("model_name") or "")
    if not record_id or not model_name:
        raise ValueError("collector row requires record_id and model_name")

    log_probability = _optional_finite(row, "p_true_log_probability")
    probability = _optional_finite(row, "p_true_probability")
    uncertainty = _optional_finite(row, "p_true_uncertainty")
    entropy = _optional_finite(row, "mean_token_entropy")
    self_certainty = _optional_finite(row, "self_certainty")
    collector_msp = _optional_finite(row, "msp")

    if probability is not None and not 0.0 <= probability <= 1.0:
        raise ValueError("p_true_probability must be in [0, 1]")
    if log_probability is not None and log_probability > 1e-7:
        raise ValueError("p_true_log_probability cannot exceed zero")
    if uncertainty is not None and uncertainty < -1e-7:
        raise ValueError("p_true_uncertainty cannot be negative")
    if entropy is not None and entropy < -1e-7:
        raise ValueError("mean_token_entropy cannot be negative")
    if self_certainty is not None and self_certainty > 1e-6:
        raise ValueError("pinned SelfCertainty output (-KL) cannot be positive")
    if collector_msp is not None and not 0.0 <= collector_msp <= 1.0:
        raise ValueError("collector MSP must be in [0, 1]")

    present_p_true = sum(
        value is not None for value in (log_probability, probability, uncertainty)
    )
    if present_p_true not in {0, 3}:
        raise ValueError(
            "P(True) representations must be either all finite or all null"
        )
    if log_probability is not None:
        assert probability is not None and uncertainty is not None
        if not math.isclose(
            uncertainty,
            -log_probability,
            rel_tol=1e-7,
            abs_tol=1e-7,
        ):
            raise ValueError("p_true_uncertainty != -p_true_log_probability")
        expected_probability = math.exp(log_probability)
        if not math.isclose(
            probability,
            expected_probability,
            rel_tol=1e-7,
            abs_tol=1e-12,
        ):
            raise ValueError("p_true_probability != exp(p_true_log_probability)")

    verdict = normalize_label(row.get("verdict"))
    if verdict is None:
        raise ValueError("collector verdict must be present and be A, B, or tie")

    hf_map_verdict = normalize_label(row.get("hf_restricted_map_verdict"))
    if hf_map_verdict is None:
        raise ValueError(
            "hf_restricted_map_verdict must be present and be A, B, or tie"
        )
    probability_raw = row.get("hf_restricted_label_probabilities")
    if not isinstance(probability_raw, Mapping):
        raise ValueError("hf_restricted_label_probabilities must be an object")
    if set(probability_raw) != {"A", "B", "tie"}:
        raise ValueError(
            "hf_restricted_label_probabilities must have exactly A, B, and tie"
        )
    hf_probabilities: dict[str, float] = {}
    for label in ("A", "B", "tie"):
        value = finite_float(probability_raw[label])
        if value is None or not 0.0 <= value <= 1.0:
            raise ValueError(
                f"hf_restricted_label_probabilities[{label!r}] must be finite in [0, 1]"
            )
        hf_probabilities[label] = value
    if not math.isclose(
        sum(hf_probabilities.values()),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError("hf_restricted_label_probabilities must sum to one")

    hf_msp = _required_finite(row, "hf_restricted_msp")
    hf_verdict_probability = _required_finite(
        row,
        "hf_restricted_verdict_probability",
    )
    hf_source_difference = _required_finite(
        row,
        "hf_source_probability_max_abs_difference",
    )
    for field_name, value in (
        ("hf_restricted_msp", hf_msp),
        ("hf_restricted_verdict_probability", hf_verdict_probability),
        ("hf_source_probability_max_abs_difference", hf_source_difference),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{field_name} must be in [0, 1]")
    expected_msp = max(hf_probabilities.values())
    if not math.isclose(hf_msp, expected_msp, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("hf_restricted_msp does not match the HF simplex maximum")
    expected_hf_map_verdict = max(
        ("A", "B", "tie"),
        key=lambda label: hf_probabilities[label],
    )
    if hf_map_verdict != expected_hf_map_verdict:
        raise ValueError(
            "hf_restricted_map_verdict does not use the pinned A/B/tie argmax rule"
        )
    if not math.isclose(
        hf_verdict_probability,
        hf_probabilities[verdict],
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "hf_restricted_verdict_probability does not match the frozen verdict"
        )
    hf_map_matches = row.get("hf_restricted_map_matches_stored")
    if not isinstance(hf_map_matches, bool):
        raise ValueError("hf_restricted_map_matches_stored must be boolean")
    expected_hf_map_matches = hf_map_verdict == verdict
    if hf_map_matches is not expected_hf_map_matches:
        raise ValueError(
            "hf_restricted_map_matches_stored does not match the two verdicts"
        )

    return CollectorScore(
        record_id=record_id,
        model_name=model_name,
        p_true_log_probability=log_probability,
        p_true_probability=probability,
        p_true_uncertainty=uncertainty,
        mean_token_entropy=entropy,
        self_certainty=self_certainty,
        collector_msp=collector_msp,
        verdict=verdict,
        hf_restricted_label_probabilities=hf_probabilities,
        hf_restricted_map_verdict=hf_map_verdict,
        hf_restricted_map_matches_stored=hf_map_matches,
        hf_restricted_msp=hf_msp,
        hf_restricted_verdict_probability=hf_verdict_probability,
        hf_source_probability_max_abs_difference=hf_source_difference,
    )


def make_analysis_row(campaign: AuditRow, collector: CollectorScore) -> AnalysisRow:
    if campaign.record_id != collector.record_id:
        raise ValueError("record_id mismatch while joining collector row")
    if campaign.model_name != collector.model_name:
        raise ValueError(
            f"record {campaign.record_id!r}: collector model does not match campaign"
        )
    campaign_msp = finite_float(campaign.scores.get("msp"))
    if campaign_msp is None:
        raise ValueError(f"record {campaign.record_id!r}: campaign MSP is unavailable")
    if collector.collector_msp is not None and not math.isclose(
        campaign_msp,
        collector.collector_msp,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"record {campaign.record_id!r}: collector MSP drift")
    if collector.verdict != campaign.verdict:
        raise ValueError(f"record {campaign.record_id!r}: collector verdict drift")

    raw_scores = {
        "msp": campaign_msp,
        "p_true": collector.p_true_log_probability,
        "mean_token_entropy": collector.mean_token_entropy,
        "self_certainty": collector.self_certainty,
    }
    confidence_scores = {
        "msp": campaign_msp,
        "p_true": collector.p_true_log_probability,
        "mean_token_entropy": (
            -collector.mean_token_entropy
            if collector.mean_token_entropy is not None
            else None
        ),
        "self_certainty": (
            -collector.self_certainty
            if collector.self_certainty is not None
            else None
        ),
    }
    return AnalysisRow(
        campaign=campaign,
        collector=collector,
        raw_scores=raw_scores,
        confidence_scores=confidence_scores,
        native_probabilities={
            "msp": campaign_msp,
            "p_true": collector.p_true_probability,
            "mean_token_entropy": None,
            "self_certainty": None,
        },
    )


def _validate_estimand(estimand: str) -> None:
    if estimand not in ESTIMANDS:
        raise ValueError(f"unknown estimand {estimand!r}")


def predictor_applicable_to_estimand(predictor: str, estimand: str) -> bool:
    if predictor not in PREDICTORS:
        raise ValueError(f"unknown predictor {predictor!r}")
    _validate_estimand(estimand)
    return not (
        estimand == SAME_ENGINE_ESTIMAND
        and predictor in {"msp", "p_true"}
    )


def predictor_estimand_caveat(predictor: str, estimand: str) -> str | None:
    if predictor_applicable_to_estimand(predictor, estimand):
        return None
    if predictor == "msp":
        return (
            "Not applicable: the published MSP is the frozen vLLM restricted-"
            "simplex MSP, not the HF replay MSP. It is not relabeled as a same-"
            "engine predictor."
        )
    return (
        "Not applicable: this P(True) score verifies the frozen vLLM verdict, "
        "not the HF replay MAP verdict on disagreement rows."
    )


def eligible_rows(
    rows: Iterable[AnalysisRow],
    estimand: str = PRIMARY_ESTIMAND,
) -> list[AnalysisRow]:
    """Return the explicitly defined population for one estimand."""

    _validate_estimand(estimand)
    materialized = list(rows)
    if estimand == MAP_AGREE_ESTIMAND:
        return [row for row in materialized if row.hf_map_agrees_with_frozen]
    return materialized


def estimand_correct(row: AnalysisRow, estimand: str = PRIMARY_ESTIMAND) -> bool:
    """Resolve the binary correctness target without changing predictor scores."""

    _validate_estimand(estimand)
    if estimand == SAME_ENGINE_ESTIMAND:
        return row.hf_replay_correct
    return row.correct


def valid_items(
    rows: Iterable[AnalysisRow],
    predictor: str,
    estimand: str = PRIMARY_ESTIMAND,
) -> list[ScoredItem]:
    if predictor not in PREDICTORS:
        raise ValueError(f"unknown predictor {predictor!r}")
    if not predictor_applicable_to_estimand(predictor, estimand):
        return []
    items: list[ScoredItem] = []
    for row in eligible_rows(rows, estimand):
        score = finite_float(row.confidence_scores.get(predictor))
        if score is not None:
            items.append((row, score, estimand_correct(row, estimand)))
    return items


def native_probability_items(
    rows: Iterable[AnalysisRow],
    predictor: str,
    estimand: str = PRIMARY_ESTIMAND,
) -> list[ProbabilityItem]:
    if not predictor_applicable_to_estimand(predictor, estimand):
        return []
    items: list[ProbabilityItem] = []
    for row in eligible_rows(rows, estimand):
        probability = finite_float(row.native_probabilities.get(predictor))
        if probability is None:
            continue
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"{predictor} native probability is outside [0, 1]")
        items.append((row, probability, estimand_correct(row, estimand)))
    return items


def expected_calibration_error(
    items: list[ProbabilityItem],
    *,
    n_bins: int = 10,
) -> float | None:
    if not items:
        return None
    buckets: list[list[ProbabilityItem]] = [[] for _ in range(n_bins)]
    for item in items:
        probability = item[1]
        if not 0.0 <= probability <= 1.0:
            raise ValueError("ECE inputs must be probabilities in [0, 1]")
        buckets[min(n_bins - 1, int(probability * n_bins))].append(item)
    return sum(
        len(bucket)
        * abs(
            sum(item[1] for item in bucket) / len(bucket)
            - sum(item[2] for item in bucket) / len(bucket)
        )
        for bucket in buckets
        if bucket
    ) / len(items)


def brier_score(items: list[ProbabilityItem]) -> float | None:
    if not items:
        return None
    return sum((probability - float(correct)) ** 2 for _, probability, correct in items) / len(items)


def ranking_metrics(items: list[ScoredItem]) -> dict[str, float | None]:
    """Tie-aware AUROC and right-continuous empirical AURC."""

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
        aurc += (coverage - previous_coverage) * (errors / accepted)
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


def record_id_sha256(items: Iterable[ScoredItem]) -> str:
    digest = hashlib.sha256()
    for record_id in sorted(item[0].record_id for item in items):
        digest.update(record_id.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def record_ids_sha256(rows: Iterable[AnalysisRow]) -> str:
    digest = hashlib.sha256()
    for record_id in sorted(row.record_id for row in rows):
        digest.update(record_id.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _score_distribution(values: Iterable[float | None]) -> dict[str, Any]:
    present = sorted(value for value in values if value is not None)
    return {
        "n": len(present),
        "min": present[0] if present else None,
        "median": statistics.median(present) if present else None,
        "max": present[-1] if present else None,
        "unique_levels": len(set(present)),
    }


def metric_summary(
    rows: list[AnalysisRow],
    predictor: str,
    estimand: str = PRIMARY_ESTIMAND,
) -> dict[str, Any]:
    population_total = len(rows)
    selected_rows = eligible_rows(rows, estimand)
    applicable = predictor_applicable_to_estimand(predictor, estimand)
    score_rows = selected_rows if applicable else []
    items = valid_items(selected_rows, predictor, estimand)
    native_items = native_probability_items(selected_rows, predictor, estimand)
    ranking = ranking_metrics(items)
    return {
        "estimand": estimand,
        "predictor_applicable": applicable,
        "predictor_estimand_caveat": predictor_estimand_caveat(
            predictor,
            estimand,
        ),
        "population_total": population_total,
        "excluded_by_estimand": population_total - len(selected_rows),
        "total": len(selected_rows),
        "n": len(items),
        "availability": (
            len(items) / len(selected_rows)
            if applicable and selected_rows
            else None
        ),
        "accuracy": sum(correct for _, _, correct in items) / len(items) if items else None,
        "raw_score": _score_distribution(
            row.raw_scores[predictor] for row in score_rows
        ),
        "confidence_oriented_score": _score_distribution(
            row.confidence_scores[predictor] for row in score_rows
        ),
        "aurc": ranking["aurc"],
        "correctness_auroc": ranking["correctness_auroc"],
        "native_probability_n": len(native_items),
        "native_probability_ece_10bin": expected_calibration_error(native_items),
        "native_probability_brier": brier_score(native_items),
        "native_probability_applicable": bool(
            METHOD_MAPPING[predictor]["native_probability"]
        )
        and applicable,
        "available_record_ids_sha256": record_id_sha256(items),
    }


def threshold_rule(
    rows: list[AnalysisRow],
    predictor: str,
    target_risk: float,
    estimand: str = PRIMARY_ESTIMAND,
) -> dict[str, Any]:
    population_n = len(rows)
    eligible_n = len(eligible_rows(rows, estimand))
    items = sorted(
        valid_items(rows, predictor, estimand),
        key=lambda item: item[1],
        reverse=True,
    )
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
                    "predictor_applicable": predictor_applicable_to_estimand(
                        predictor,
                        estimand,
                    ),
                    "predictor_estimand_caveat": predictor_estimand_caveat(
                        predictor,
                        estimand,
                    ),
                    "threshold_scale": "confidence_oriented_raw_score",
                    "calibration_population_n": population_n,
                    "calibration_excluded_by_estimand": population_n - eligible_n,
                    "calibration_n": len(items),
                    "accepted": accepted,
                    "coverage": accepted / len(items),
                    "risk": risk,
                    "accepted_record_ids_sha256": record_id_sha256(items[:cursor]),
                }
            )
    if not feasible:
        return {
            "threshold": None,
            "predictor_applicable": predictor_applicable_to_estimand(
                predictor,
                estimand,
            ),
            "predictor_estimand_caveat": predictor_estimand_caveat(
                predictor,
                estimand,
            ),
            "threshold_scale": "confidence_oriented_raw_score",
            "calibration_population_n": population_n,
            "calibration_excluded_by_estimand": population_n - eligible_n,
            "calibration_n": len(items),
            "accepted": 0,
            "coverage": 0.0 if items else None,
            "risk": None,
            "accepted_record_ids_sha256": record_id_sha256([]),
        }
    return max(feasible, key=lambda row: (row["coverage"], -row["threshold"]))


def threshold_transfer(
    rows: list[AnalysisRow],
    predictor: str,
    rule: Mapping[str, Any],
    estimand: str = PRIMARY_ESTIMAND,
) -> dict[str, Any]:
    items = valid_items(rows, predictor, estimand)
    threshold = finite_float(rule.get("threshold"))
    accepted = [item for item in items if item[1] >= threshold] if threshold is not None else []
    errors = sum(not item[2] for item in accepted)
    return {
        "predictor_applicable": predictor_applicable_to_estimand(
            predictor,
            estimand,
        ),
        "predictor_estimand_caveat": predictor_estimand_caveat(
            predictor,
            estimand,
        ),
        "population_n": len(rows),
        "excluded_by_estimand": len(rows) - len(eligible_rows(rows, estimand)),
        "n": len(items),
        "accepted": len(accepted),
        "errors": errors,
        "coverage": len(accepted) / len(items) if items else None,
        "risk": errors / len(accepted) if accepted else None,
        "accepted_record_ids_sha256": record_id_sha256(accepted),
    }


def assign_question_folds(
    question_ids: Iterable[str],
    *,
    n_folds: int,
    seed: int,
) -> dict[str, int]:
    values = sorted(set(question_ids), key=lambda value: (len(value), value))
    if len(values) < n_folds:
        raise ValueError("fewer unique questions than requested folds")
    random.Random(seed).shuffle(values)
    return {value: index % n_folds for index, value in enumerate(values)}


def fit_isotonic(items: list[ScoredItem]) -> list[tuple[float, float, float]]:
    """Fit increasing P(correct) versus arbitrary confidence score via PAVA."""

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
    for _, high, probability in blocks:
        if score <= high:
            return probability
    return blocks[-1][2]


def _read_campaign_jsonl(
    path: Path,
    *,
    predicate: Any = None,
    require_explicit_clean_tie: bool = False,
) -> list[AuditRow]:
    rows: list[AuditRow] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, Mapping):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            if predicate is not None and not predicate(raw):
                continue
            if require_explicit_clean_tie and "clean_tie" not in raw:
                raise ValueError(f"{path}:{line_number}: clean_tie must be explicit")
            try:
                rows.append(audit_row_from_mapping(raw))
            except Exception as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return rows


def _is_incongruent_test(row: Mapping[str, Any]) -> bool:
    return (
        str(row.get("routing_split") or "") == "test"
        and str(row.get("cue_congruency") or row.get("direction") or "").lower()
        == "incongruent"
        and str(row.get("bias_name") or row.get("family") or "").lower()
        in PINNED_MAX_DOSES
    )


def _read_collector_jsonl(path: Path) -> dict[str, CollectorScore]:
    rows: dict[str, CollectorScore] = {}
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, Mapping):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            try:
                parsed = collector_score_from_mapping(raw)
            except Exception as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if parsed.record_id in rows:
                raise ValueError(f"{path}:{line_number}: duplicate record_id")
            rows[parsed.record_id] = parsed
    if not rows:
        raise ValueError(f"{path}: no collector rows")
    return rows


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
        raise ValueError(f"no complete model directories under {campaign_root}")
    return result


def _marker_commit(marker: Mapping[str, Any]) -> str | None:
    for value in (
        marker.get("lm_polygraph_commit"),
        _nested(marker, "provenance", "lm_polygraph_commit"),
        _nested(marker, "software", "lm_polygraph_commit"),
    ):
        if value:
            return str(value)
    return None


def _marker_scores_sha(marker: Mapping[str, Any]) -> str | None:
    for value in (
        marker.get("scores_sha256"),
        marker.get("score_file_sha256"),
        marker.get("output_sha256"),
        _nested(marker, "artifacts", COLLECTOR_FILENAME, "sha256"),
        _nested(marker, "files", COLLECTOR_FILENAME, "sha256"),
    ):
        if value:
            return str(value)
    return None


def discover_collector_directories(collector_root: Path) -> dict[str, Path]:
    markers = sorted(collector_root.rglob(COLLECTOR_MARKER_FILENAME))
    if not markers:
        raise ValueError(f"no collector completion markers under {collector_root}")
    result: dict[str, Path] = {}
    for marker_path in markers:
        directory = marker_path.parent
        scores_path = directory / COLLECTOR_FILENAME
        if not scores_path.is_file():
            raise ValueError(f"{directory}: completion marker has no score JSONL")
        marker = json.loads(marker_path.read_text())
        if not isinstance(marker, Mapping):
            raise ValueError(f"{marker_path}: marker must be an object")
        status = str(marker.get("status") or "").strip().lower()
        if status not in {"complete", "completed", "success", "succeeded"}:
            raise ValueError(f"{marker_path}: collector status is not complete")
        if bool(marker.get("smoke_only")):
            raise ValueError(f"{marker_path}: smoke-only collector output is not auditable")
        model_name = str(marker.get("model_name") or "")
        if not model_name:
            raise ValueError(f"{marker_path}: marker has no model_name")
        commit = _marker_commit(marker)
        if commit != LM_POLYGRAPH_COMMIT:
            raise ValueError(
                f"{marker_path}: expected LM-Polygraph commit {LM_POLYGRAPH_COMMIT}"
            )
        actual_sha = file_sha256(scores_path)
        recorded_sha = _marker_scores_sha(marker)
        if recorded_sha is not None and recorded_sha != actual_sha:
            raise ValueError(f"{marker_path}: score-file SHA-256 mismatch")
        if model_name in result:
            raise ValueError(f"duplicate collector marker for model {model_name!r}")
        result[model_name] = directory
    return result


def _validate_unique_campaign_rows(rows: Iterable[AuditRow], *, label: str) -> dict[str, AuditRow]:
    result: dict[str, AuditRow] = {}
    for row in rows:
        if row.record_id in result:
            raise ValueError(f"duplicate campaign record_id in {label}: {row.record_id}")
        result[row.record_id] = row
    return result


def _join_expected_rows(
    campaign_rows: list[AuditRow],
    collector_rows: Mapping[str, CollectorScore],
) -> list[AnalysisRow]:
    campaign_by_id = _validate_unique_campaign_rows(campaign_rows, label="analysis set")
    expected_ids = set(campaign_by_id)
    collector_ids = set(collector_rows)
    missing = sorted(expected_ids - collector_ids)
    extra = sorted(collector_ids - expected_ids)
    if missing or extra:
        raise ValueError(
            "collector record_id set differs from required Stage-A + primary-candidate set: "
            f"missing={missing[:5]!r} ({len(missing)}); "
            f"extra={extra[:5]!r} ({len(extra)})"
        )
    return [
        make_analysis_row(campaign_by_id[record_id], collector_rows[record_id])
        for record_id in sorted(expected_ids)
    ]


def _collector_input_summary(
    directory: Path,
    rows: Mapping[str, CollectorScore],
) -> dict[str, Any]:
    marker_path = directory / COLLECTOR_MARKER_FILENAME
    scores_path = directory / COLLECTOR_FILENAME
    marker = json.loads(marker_path.read_text())
    recorded_count = marker.get("record_count")
    if recorded_count is not None and int(recorded_count) != len(rows):
        raise ValueError(f"{marker_path}: record_count does not match score file")
    actual_scores_sha = file_sha256(scores_path)
    recorded_scores_sha = _marker_scores_sha(marker)
    return {
        "directory_name": directory.name,
        "score_filename": COLLECTOR_FILENAME,
        "marker_filename": COLLECTOR_MARKER_FILENAME,
        "score_sha256": actual_scores_sha,
        "marker_sha256": file_sha256(marker_path),
        "record_count": len(rows),
        "marker_record_count": recorded_count,
        "marker_status": marker.get("status"),
        "marker_schema_version": marker.get("schema_version"),
        "marker_kind": marker.get("kind"),
        "smoke_only": marker.get("smoke_only"),
        "model_revision": marker.get("model_revision"),
        "tokenizer_revision": marker.get("tokenizer_revision"),
        "lm_polygraph_commit": _marker_commit(marker),
        "collector_spec_hash": marker.get("collector_spec_hash"),
        "record_id_digest": marker.get("record_id_digest"),
        "selection_manifest": marker.get("selection_manifest"),
        "selection_manifest_sha256": marker.get("selection_manifest_sha256"),
        "scientific_gates": marker.get("scientific_gates"),
        "recorded_score_sha256": recorded_scores_sha,
        "recorded_score_sha256_verified": (
            recorded_scores_sha == actual_scores_sha
            if recorded_scores_sha is not None
            else None
        ),
    }


def _median(values: Iterable[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return statistics.median(present) if present else None


def _aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for predictor in PREDICTORS:
        selected = [row for row in rows if row["predictor"] == predictor]
        applicable = [
            row for row in selected if row.get("predictor_applicable", True)
        ]
        result[predictor] = {
            "groups": len(selected),
            "applicable_groups": len(applicable),
            "nonapplicable_groups": len(selected) - len(applicable),
            "median_availability": _median(
                row["availability"] for row in applicable
            ),
            "median_aurc": _median(row["aurc"] for row in applicable),
            "median_correctness_auroc": _median(
                row["correctness_auroc"] for row in applicable
            ),
            "median_native_probability_ece_10bin": _median(
                row["native_probability_ece_10bin"] for row in applicable
            ),
            "median_native_probability_brier": _median(
                row["native_probability_brier"] for row in applicable
            ),
        }
    return result


def _aggregate_cells(
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
                if row["target_risk"] == target and row["predictor"] == predictor
            ]
            applicable = [
                row for row in selected if row.get("predictor_applicable", True)
            ]
            nonzero = [row for row in applicable if row["accepted"] > 0]
            accepted = sum(row["accepted"] for row in applicable)
            errors = sum(row["errors"] for row in applicable)
            result[str(target)][predictor] = {
                "cells": len(selected),
                "applicable_cells": len(applicable),
                "nonapplicable_cells": len(selected) - len(applicable),
                "evaluated_n": sum(row["n"] for row in applicable),
                "zero_coverage_cells": len(applicable) - len(nonzero),
                "nonzero_coverage_cells": len(nonzero),
                "accepted": accepted,
                "errors": errors,
                "pooled_risk_when_nonzero": errors / accepted if accepted else None,
                "median_coverage": _median(row["coverage"] for row in applicable),
                "median_risk_nonzero": _median(row["risk"] for row in nonzero),
            }
    return result


def _optional_csv_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _float_equal(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-15)


def validate_published_input_hashes(
    input_hashes: Mapping[str, Mapping[str, str]],
    provenance_path: Path,
) -> dict[str, Any]:
    provenance = json.loads(provenance_path.read_text())
    if not isinstance(provenance, Mapping):
        raise ValueError("published provenance must be an object")
    expected_stage_a = list(provenance["stage_a_input_hashes"])
    expected_stage_b = list(provenance["stage_b_input_hashes"])
    observed_stage_a = [
        hashes[STAGE_A_FILENAME] for hashes in input_hashes.values()
    ]
    observed_stage_b = [
        hashes[STAGE_B_FILENAME] for hashes in input_hashes.values()
    ]
    stage_a_match = Counter(observed_stage_a) == Counter(expected_stage_a)
    stage_b_match = Counter(observed_stage_b) == Counter(expected_stage_b)
    result = {
        "performed": True,
        "passed": stage_a_match and stage_b_match,
        "provenance_sha256": file_sha256(provenance_path),
        "analysis_version": provenance.get("analysis_version"),
        "spec_hash": provenance.get("spec_hash"),
        "stage_a_hashes_match": stage_a_match,
        "stage_b_hashes_match": stage_b_match,
    }
    if not result["passed"]:
        raise ValueError("raw score hashes do not match published strict-v3 provenance")
    return result


def validate_msp_primary_regression(
    primary_cells: list[dict[str, Any]],
    oracle_path: Path,
) -> dict[str, Any]:
    observed_rows = [
        row
        for row in primary_cells
        if row["predictor"] == "msp" and math.isclose(row["target_risk"], 0.10)
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
        mismatches.append("MSP primary key sets differ")
    for key in sorted(set(observed) & set(expected)):
        actual = observed[key]
        oracle = expected[key]
        for name, actual_value, expected_value in (
            ("calibration_n", actual["rule_calibration_n"], oracle["calibration_n"]),
            ("test_n", actual["n"], oracle["test_n"]),
            ("test_accepted", actual["accepted"], oracle["test_accepted"]),
        ):
            if int(actual_value) != int(expected_value):
                mismatches.append(
                    f"{key!r} {name}: {actual_value!r} != {expected_value!r}"
                )
        for name, actual_value, expected_value in (
            ("dose", actual["dose"], _optional_csv_float(oracle["dose"])),
            (
                "threshold",
                actual["rule_threshold"],
                _optional_csv_float(oracle["threshold"]),
            ),
            (
                "calibration_coverage",
                actual["rule_calibration_coverage"],
                _optional_csv_float(oracle["calibration_coverage"]),
            ),
            (
                "calibration_risk",
                actual["rule_calibration_risk"],
                _optional_csv_float(oracle["calibration_risk"]),
            ),
            (
                "test_coverage",
                actual["coverage"],
                _optional_csv_float(oracle["test_coverage"]),
            ),
            (
                "test_realized_risk",
                actual["risk"],
                _optional_csv_float(oracle["test_realized_risk"]),
            ),
        ):
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
        "zero_coverage_cells": sum(row["accepted"] == 0 for row in observed.values()),
        "accepted": sum(row["accepted"] for row in observed.values()),
        "errors": sum(row["errors"] for row in observed.values()),
        "mismatches": mismatches,
        "required_invariants": dict(PUBLISHED_MSP_PRIMARY_INVARIANTS),
    }
    observed_invariants = {
        "cells": result["observed_cells"],
        "zero_coverage_cells": result["zero_coverage_cells"],
        "accepted": result["accepted"],
        "errors": result["errors"],
    }
    result["observed_invariants"] = observed_invariants
    for name, expected_value in PUBLISHED_MSP_PRIMARY_INVARIANTS.items():
        if observed_invariants[name] != expected_value:
            mismatches.append(
                f"published MSP invariant {name}: "
                f"{observed_invariants[name]!r} != {expected_value!r}"
            )
    result["passed"] = not mismatches
    if mismatches:
        raise ValueError(
            "MSP regression against published strict-v3 primary failed: "
            + "; ".join(mismatches[:5])
        )
    return result


def validate_output_path(
    campaign_root: Path,
    collector_root: Path,
    output: Path,
    *,
    additional_immutable_roots: Iterable[Path] = (),
) -> None:
    immutable_roots = (campaign_root, collector_root, *additional_immutable_roots)
    if any(output.resolve().is_relative_to(root.resolve()) for root in immutable_roots):
        raise ValueError("output must be outside every immutable input root")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")


def _empty_estimand_results() -> dict[str, list[dict[str, Any]]]:
    return {
        "calibration_metrics": [],
        "primary_metrics": [],
        "rules": [],
        "clean_test_transfers": [],
        "primary_cells": [],
        "oof_clean": [],
        "oof_isotonic": [],
    }


def _finalize_estimand_results(
    estimand: str,
    rows: Mapping[str, list[dict[str, Any]]],
    *,
    targets: tuple[float, ...],
) -> dict[str, Any]:
    oof_isotonic = rows["oof_isotonic"]
    return {
        "definition": ESTIMAND_DEFINITIONS[estimand],
        "calibration_aggregate": _aggregate_metrics(rows["calibration_metrics"]),
        "primary_ranking_aggregate": _aggregate_metrics(rows["primary_metrics"]),
        "clean_test_transfer_aggregate": _aggregate_cells(
            rows["clean_test_transfers"],
            targets=targets,
        ),
        "full_primary_cell_aggregate": _aggregate_cells(
            rows["primary_cells"],
            targets=targets,
        ),
        "question_disjoint_isotonic_aggregate": {
            predictor: {
                "groups": len(
                    [row for row in oof_isotonic if row["predictor"] == predictor]
                ),
                "applicable_groups": len(
                    [
                        row
                        for row in oof_isotonic
                        if row["predictor"] == predictor
                        and row.get("predictor_applicable", True)
                    ]
                ),
                "median_ece_10bin": _median(
                    row["ece_10bin"]
                    for row in oof_isotonic
                    if row["predictor"] == predictor
                    and row.get("predictor_applicable", True)
                ),
                "median_brier": _median(
                    row["brier"]
                    for row in oof_isotonic
                    if row["predictor"] == predictor
                    and row.get("predictor_applicable", True)
                ),
            }
            for predictor in PREDICTORS
        },
        "calibration_by_model_order": rows["calibration_metrics"],
        "primary_ranking_by_cell": rows["primary_metrics"],
        "rules": rows["rules"],
        "clean_test_transfers": rows["clean_test_transfers"],
        "primary_cells": rows["primary_cells"],
        "question_disjoint_clean": rows["oof_clean"],
        "question_disjoint_isotonic": oof_isotonic,
    }


def run_audit(
    campaign_root: Path,
    collector_root: Path,
    *,
    published_provenance: Path,
    published_msp_oracle: Path,
    targets: tuple[float, ...] = (0.10, 0.20),
    n_folds: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    if not targets or any(not 0.0 <= target <= 1.0 for target in targets):
        raise ValueError("target risks must be in [0, 1]")
    if n_folds < 2:
        raise ValueError("n_folds must be at least two")

    collector_dirs = discover_collector_directories(collector_root)
    model_dirs = discover_model_directories(campaign_root)
    results_by_estimand = {
        estimand: _empty_estimand_results() for estimand in ESTIMANDS
    }
    structure: dict[str, Any] = {}
    input_hashes: dict[str, Any] = {}
    collector_inputs: dict[str, Any] = {}

    campaign_models: set[str] = set()
    for model_dir in model_dirs:
        stage_a_path = model_dir / STAGE_A_FILENAME
        stage_b_path = model_dir / STAGE_B_FILENAME
        model_marker_path = model_dir / MODEL_MARKER_FILENAME
        marker = json.loads(model_marker_path.read_text())
        if not isinstance(marker, Mapping):
            raise ValueError(f"{model_marker_path}: marker must be an object")
        model_name = str(marker.get("model_name") or "")
        if not model_name or model_name in campaign_models:
            raise ValueError("campaign model markers must have unique model_name")
        campaign_models.add(model_name)
        if model_name not in collector_dirs:
            raise ValueError(f"missing collector output for model {model_name!r}")

        stage_a = _read_campaign_jsonl(stage_a_path)
        if any(row.family != "clean" for row in stage_a):
            raise ValueError(f"{model_name}: Stage A contains non-clean rows")
        if any(row.routing_split not in {"calibration", "test"} for row in stage_a):
            raise ValueError(f"{model_name}: Stage A has an unexpected routing split")
        stage_b_candidates = _read_campaign_jsonl(
            stage_b_path,
            predicate=_is_incongruent_test,
            require_explicit_clean_tie=True,
        )
        observed_models = {row.model_name for row in [*stage_a, *stage_b_candidates]}
        if observed_models != {model_name}:
            raise ValueError(f"{model_name}: score-row model names do not match marker")
        observed_max_doses = {
            family: max(
                float(row.dose)
                for row in stage_b_candidates
                if row.family == family and row.dose is not None
            )
            for family in PINNED_MAX_DOSES
        }
        if observed_max_doses != dict(PINNED_MAX_DOSES):
            raise ValueError(
                f"{model_name}: expected pinned maximum doses {dict(PINNED_MAX_DOSES)!r}; "
                f"observed {observed_max_doses!r}"
            )
        primary_with_ties = [
            row
            for row in stage_b_candidates
            if row.dose == PINNED_MAX_DOSES[row.family]
        ]
        primary = [row for row in primary_with_ties if not row.clean_tie]

        collector_dir = collector_dirs[model_name]
        collector_rows = _read_collector_jsonl(collector_dir / COLLECTOR_FILENAME)
        joined = _join_expected_rows([*stage_a, *primary_with_ties], collector_rows)
        joined_by_id = {row.record_id: row for row in joined}
        joined_stage_a = [joined_by_id[row.record_id] for row in stage_a]
        joined_primary_with_ties = [
            joined_by_id[row.record_id] for row in primary_with_ties
        ]
        joined_primary = [
            joined_by_id[row.record_id] for row in primary
        ]

        fold_by_question = assign_question_folds(
            (row.question_id for row in stage_a),
            n_folds=n_folds,
            seed=seed,
        )
        input_hashes[model_name] = {
            STAGE_A_FILENAME: file_sha256(stage_a_path),
            STAGE_B_FILENAME: file_sha256(stage_b_path),
            MODEL_MARKER_FILENAME: file_sha256(model_marker_path),
        }
        collector_inputs[model_name] = _collector_input_summary(
            collector_dir,
            collector_rows,
        )
        availability = {
            predictor: sum(
                row.confidence_scores[predictor] is not None for row in joined
            )
            for predictor in PREDICTORS
        }
        structure[model_name] = {
            "model_directory": model_dir.name,
            "stage_a_rows": len(stage_a),
            "stage_a_calibration_rows": sum(
                row.routing_split == "calibration" for row in stage_a
            ),
            "stage_a_test_rows": sum(row.routing_split == "test" for row in stage_a),
            "primary_candidate_rows_with_clean_ties": len(primary_with_ties),
            "primary_candidate_clean_tie_rows": len(primary_with_ties) - len(primary),
            "primary_evaluation_rows": len(primary),
            "max_doses": observed_max_doses,
            "required_join_rows": len(joined),
            "required_record_ids_sha256": record_ids_sha256(joined),
            "method_available_rows": availability,
            "hf_replay_agreement": {
                "joined_rows": len(joined),
                "map_agree_rows": sum(
                    row.hf_map_agrees_with_frozen for row in joined
                ),
                "map_disagree_rows": sum(
                    not row.hf_map_agrees_with_frozen for row in joined
                ),
                "map_agreement_rate": (
                    sum(row.hf_map_agrees_with_frozen for row in joined) / len(joined)
                    if joined
                    else None
                ),
                "stage_a_map_agree_rows": sum(
                    row.hf_map_agrees_with_frozen for row in joined_stage_a
                ),
                "primary_map_agree_rows": sum(
                    row.hf_map_agrees_with_frozen for row in joined_primary
                ),
                "source_probability_max_abs_difference": _score_distribution(
                    row.collector.hf_source_probability_max_abs_difference
                    for row in joined
                ),
            },
        }

        for ordering in ("ab", "ba"):
            calibration = [
                row
                for row in joined_stage_a
                if row.campaign.routing_split == "calibration"
                and row.campaign.ordering == ordering
            ]
            clean_test = [
                row
                for row in joined_stage_a
                if row.campaign.routing_split == "test"
                and row.campaign.ordering == ordering
            ]
            for estimand in ESTIMANDS:
                bucket = results_by_estimand[estimand]
                for predictor in PREDICTORS:
                    calibration_summary = metric_summary(
                        calibration,
                        predictor,
                        estimand,
                    )
                    calibration_summary.update(
                        model_name=model_name,
                        ordering=ordering,
                        predictor=predictor,
                    )
                    bucket["calibration_metrics"].append(calibration_summary)

                    fold_rules: dict[tuple[float, int], dict[str, Any]] = {}
                    isotonic_predictions: list[ProbabilityItem] = []
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
                        train_items = valid_items(train, predictor, estimand)
                        heldout_items = valid_items(heldout, predictor, estimand)
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
                                estimand,
                            )
                    bucket["oof_isotonic"].append(
                        {
                            "model_name": model_name,
                            "ordering": ordering,
                            "predictor": predictor,
                            "predictor_applicable": (
                                predictor_applicable_to_estimand(
                                    predictor,
                                    estimand,
                                )
                            ),
                            "predictor_estimand_caveat": (
                                predictor_estimand_caveat(predictor, estimand)
                            ),
                            "n": len(isotonic_predictions),
                            "ece_10bin": expected_calibration_error(
                                isotonic_predictions
                            ),
                            "brier": brier_score(isotonic_predictions),
                            "probability_source": (
                                f"question-disjoint {n_folds}-fold clean-only "
                                "isotonic regression"
                            ),
                        }
                    )

                    for target in targets:
                        rule = threshold_rule(
                            calibration,
                            predictor,
                            target,
                            estimand,
                        )
                        bucket["rules"].append(
                            {
                                "model_name": model_name,
                                "ordering": ordering,
                                "predictor": predictor,
                                "predictor_applicable": (
                                    predictor_applicable_to_estimand(
                                        predictor,
                                        estimand,
                                    )
                                ),
                                "predictor_estimand_caveat": (
                                    predictor_estimand_caveat(predictor, estimand)
                                ),
                                "target_risk": target,
                                **rule,
                            }
                        )
                        bucket["clean_test_transfers"].append(
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
                                    estimand,
                                ),
                            }
                        )
                        heldout_items = valid_items(
                            calibration,
                            predictor,
                            estimand,
                        )
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
                        heldout_errors = sum(not item[2] for item in heldout_accepted)
                        eligible_calibration_n = len(
                            eligible_rows(calibration, estimand)
                        )
                        bucket["oof_clean"].append(
                            {
                                "model_name": model_name,
                                "ordering": ordering,
                                "predictor": predictor,
                                "predictor_applicable": (
                                    predictor_applicable_to_estimand(
                                        predictor,
                                        estimand,
                                    )
                                ),
                                "predictor_estimand_caveat": (
                                    predictor_estimand_caveat(predictor, estimand)
                                ),
                                "target_risk": target,
                                "population_n": len(calibration),
                                "excluded_by_estimand": (
                                    len(calibration) - eligible_calibration_n
                                ),
                                "n": len(heldout_items),
                                "accepted": len(heldout_accepted),
                                "errors": heldout_errors,
                                "coverage": (
                                    len(heldout_accepted) / len(heldout_items)
                                    if heldout_items
                                    else None
                                ),
                                "risk": (
                                    heldout_errors / len(heldout_accepted)
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

                        for family in PINNED_MAX_DOSES:
                            test = [
                                row
                                for row in joined_primary
                                if row.campaign.ordering == ordering
                                and row.campaign.family == family
                            ]
                            if target == targets[0]:
                                summary = metric_summary(test, predictor, estimand)
                                summary.update(
                                    model_name=model_name,
                                    ordering=ordering,
                                    family=family,
                                    direction="incongruent",
                                    dose=PINNED_MAX_DOSES[family],
                                    predictor=predictor,
                                )
                                bucket["primary_metrics"].append(summary)
                            transfer = threshold_transfer(
                                test,
                                predictor,
                                rule,
                                estimand,
                            )
                            bucket["primary_cells"].append(
                                {
                                    "model_name": model_name,
                                    "ordering": ordering,
                                    "family": family,
                                    "direction": "incongruent",
                                    "dose": PINNED_MAX_DOSES[family],
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
    extra_collector_models = sorted(set(collector_dirs) - campaign_models)
    if extra_collector_models:
        raise ValueError(f"unexpected collector models: {extra_collector_models!r}")

    provenance_regression = validate_published_input_hashes(
        input_hashes,
        published_provenance,
    )
    primary_rows = results_by_estimand[PRIMARY_ESTIMAND]
    msp_regression = validate_msp_primary_regression(
        primary_rows["primary_cells"],
        published_msp_oracle,
    )
    if not provenance_regression["passed"] or not msp_regression["passed"]:
        raise ValueError("published strict-v3 preservation regressions did not pass")

    finalized_estimands = {
        estimand: _finalize_estimand_results(
            estimand,
            results_by_estimand[estimand],
            targets=targets,
        )
        for estimand in ESTIMANDS
    }
    primary_report = finalized_estimands[PRIMARY_ESTIMAND]

    return {
        "schema_version": 2,
        "analysis_version": "silent_bias_lm_polygraph_new_inference_v2",
        "analysis": (
            "standalone exploratory new-inference uncertainty audit; published "
            "strict-v3 MSP primary and zero-coverage outcome regression-verified unchanged"
        ),
        "primary_status": (
            "exploratory only; MSP remains the strict-v3 primary and every new "
            "predictor is reported separately"
        ),
        "predictors": PREDICTORS,
        "combined_predictors": [],
        "primary_estimand": PRIMARY_ESTIMAND,
        "estimand_definitions": ESTIMAND_DEFINITIONS,
        "score_direction": "higher confidence-oriented score means more confident",
        "method_mapping": METHOD_MAPPING,
        "caveats": [
            "No predictor is selected, tuned, or combined based on primary-test outcomes.",
            "Mean Token Entropy and Self-Certainty raw scores are not probabilities.",
            "The same-engine HF-replay estimand reports Mean Token Entropy and "
            "Self-Certainty only; frozen-vLLM MSP and verdict-conditioned P(True) "
            "are explicitly non-applicable there.",
            "Native-probability ECE/Brier is diagnostic only; comparable correctness "
            "calibration is the question-disjoint clean-only isotonic result.",
            "Zero accepted rows imply zero coverage and undefined realized risk, not zero risk.",
            "AURC is the right-continuous empirical integral over complete score-tie blocks.",
            ESTIMAND_DEFINITIONS[MAP_AGREE_ESTIMAND]["selection_bias_caveat"],
        ],
        "lm_polygraph": {
            "repository": "https://github.com/IINemo/lm-polygraph",
            "commit": LM_POLYGRAPH_COMMIT,
            "new_model_inference": True,
            "estimator_source_sha256": {
                "estimators/p_true.py": "3147241fa4a5138ed6632f32210c6c97c789e28ad0d64a0391745c8e1400d79a",
                "estimators/token_entropy.py": "224b36270aa037e1064cf613f62040aab4eef9145efbb233b71158884422870f",
                "estimators/self_certainty.py": "5f3f629afca269df7fadd801a3c4c0012ee56e28cdfde411a0815a56ba380f95",
                "stat_calculators/entropy.py": "97b6074bd17263e1eb9f755c1fb071dc6351f173d055ec08d1f1c9287ed1cbac",
                "stat_calculators/prompt.py": "178c6eb201852879bd9a65bda7bf0f1be9759d5838f6ecdad5264edc95692a77",
            },
        },
        "targets": targets,
        "question_disjoint_folds": n_folds,
        "seed": seed,
        "pinned_primary_max_doses": dict(PINNED_MAX_DOSES),
        "implementation_sha256": {
            Path(__file__).name: file_sha256(Path(__file__)),
        },
        "structure": structure,
        "campaign_input_sha256": input_hashes,
        "collector_inputs": collector_inputs,
        "published_input_regression": provenance_regression,
        "published_msp_primary_regression": msp_regression,
        "estimands": finalized_estimands,
        # Backward-compatible aliases remain bound only to the frozen-vLLM
        # all-row primary. This is also the sole input to the MSP oracle gate.
        "calibration_aggregate": primary_report["calibration_aggregate"],
        "primary_ranking_aggregate": primary_report["primary_ranking_aggregate"],
        "clean_test_transfer_aggregate": primary_report[
            "clean_test_transfer_aggregate"
        ],
        "full_primary_cell_aggregate": primary_report[
            "full_primary_cell_aggregate"
        ],
        "question_disjoint_isotonic_aggregate": primary_report[
            "question_disjoint_isotonic_aggregate"
        ],
        "calibration_by_model_order": primary_report[
            "calibration_by_model_order"
        ],
        "primary_ranking_by_cell": primary_report["primary_ranking_by_cell"],
        "rules": primary_report["rules"],
        "clean_test_transfers": primary_report["clean_test_transfers"],
        "primary_cells": primary_report["primary_cells"],
        "question_disjoint_clean": primary_report["question_disjoint_clean"],
        "question_disjoint_isotonic": primary_report[
            "question_disjoint_isotonic"
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--collector-root", type=Path, required=True)
    parser.add_argument("--published-provenance", type=Path, required=True)
    parser.add_argument("--published-msp-oracle", type=Path, required=True)
    parser.add_argument(
        "--target-risk",
        type=float,
        action="append",
        dest="target_risks",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = tuple(args.target_risks or (0.10, 0.20))
    if args.output is not None:
        validate_output_path(
            args.campaign_root,
            args.collector_root,
            args.output,
            additional_immutable_roots=(
                args.published_provenance.parent,
                args.published_msp_oracle.parent,
            ),
        )
    report = run_audit(
        args.campaign_root,
        args.collector_root,
        published_provenance=args.published_provenance,
        published_msp_oracle=args.published_msp_oracle,
        targets=targets,
        n_folds=args.folds,
        seed=args.seed,
    )
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        with args.output.open("x") as handle:
            handle.write(payload)


if __name__ == "__main__":
    main()
