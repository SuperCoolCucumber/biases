from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from biases.analysis.dose_response import (
    NORMALIZED_FOUR_LEVEL_DOSES,
    SOCIAL_DOSE_LADDERS,
    TrendObservation,
    clustered_monotonic_trend_test,
    dose_observations_from_shifts,
    fit_dose_response_with_cluster_bootstrap,
    normalized_social_dose,
)
from biases.analysis.modeling import (
    MIXED_EFFECTS_FORMULA,
    UNCERTAINTY_GEE_FORMULA,
    OptionalAnalysisDependencyError,
    cluster_bootstrap_uncertainty_gee_slopes,
    fit_flip_mixed_logit,
    fit_uncertainty_gee,
)
from biases.analysis.provenance import canonical_json, file_sha256, input_hashes, spec_sha256
from biases.analysis.records import ConditionRecord, pair_clean_and_cued, record_from_mapping
from biases.analysis.resampling import (
    cluster_percentile_interval,
    cluster_sign_flip_p_value,
    percentile,
)
from biases.analysis.rq1 import (
    PairedShift,
    compute_paired_shifts,
    low_dose_susceptibility_auc_with_cluster_bootstrap,
    shift_metric_value,
)
from biases.analysis.selective import (
    CONFIDENCE_CHANNELS,
    ClusterBootstrapDraws,
    ScoredPrediction,
    bootstrap_threshold_rules,
    calibration_summary,
    cluster_bootstrap_draws,
    clean_calibrated_threshold_transfer_with_cluster_bootstrap,
    confidence_value,
    paired_channel_flip,
    paired_correctness_mcnemar,
    prediction_from_record,
    risk_coverage_curve,
    swap_average_records,
)
from biases.analysis.statistics import holm_adjust


ANALYSIS_VERSION = "silent-bias-p4-v6"
HEADLINE_ROUTING_SPLIT = "test"
OUTPUT_NAMES = (
    "paired_shifts.csv",
    "rq1_silent_shift.csv",
    "rq1_susceptibility.csv",
    "rq2_calibration.csv",
    "rq2_reliability.csv",
    "rq2_risk_coverage.csv",
    "rq2_threshold_transfer.csv",
    "rq2_mcnemar.csv",
    "rq3_dose_response.csv",
    "rq3_uncertainty_trend.csv",
    "rq3_uncertainty_by_dose.csv",
    "rq3_modeling.csv",
)


def read_jsonl(paths: Sequence[Path]) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"{path}:{line_number} is not a JSON object")
                rows.append(payload)
    return tuple(rows)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return canonical_json(value)
    return value


def write_tidy_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    provenance_columns: Mapping[str, str],
) -> None:
    augmented = [
        {**dict(row), **provenance_columns}
        for row in rows
    ]
    fields = sorted({key for row in augmented for key in row}) or sorted(provenance_columns)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in augmented:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _group(
    values: Iterable[Any],
    key: Any,
) -> dict[tuple[Any, ...], list[Any]]:
    grouped: dict[tuple[Any, ...], list[Any]] = defaultdict(list)
    for value in values:
        grouped[key(value)].append(value)
    return dict(grouped)


def _select_routing_split(
    shifts: Sequence[PairedShift],
    *,
    routing_split: str,
) -> tuple[PairedShift, ...]:
    if routing_split not in {"calibration", "test"}:
        raise ValueError(
            "routing_split must be either 'calibration' or 'test'"
        )
    invalid = sorted(
        {
            str(shift.routing_split)
            for shift in shifts
            if shift.routing_split not in {"calibration", "test"}
        }
    )
    if invalid:
        raise ValueError(
            "all paired shifts must declare routing_split as calibration "
            f"or test; observed invalid values: {invalid}"
        )
    return tuple(
        shift for shift in shifts if shift.routing_split == routing_split
    )


def _condition_key(record: ConditionRecord | ScoredPrediction) -> tuple[Any, ...]:
    return (
        record.model_name,
        record.ordering,
        record.family,
        record.direction,
        record.dose,
        record.variant_id,
        record.clean_tie,
        record.routing_split,
    )


def _condition_columns(key: tuple[Any, ...]) -> dict[str, Any]:
    return dict(
        zip(
            (
                "model_name",
                "ordering",
                "family",
                "direction",
                "dose",
                "variant_id",
                "clean_tie",
                "routing_split",
            ),
            key,
            strict=True,
        )
    )


def summarize_silent_shift(
    shifts: Sequence[PairedShift],
    *,
    routing_split: str,
    n_resamples: int,
    seed: int,
) -> list[dict[str, Any]]:
    metrics = (
        "signed_cue_mass",
        "delta_entropy",
        "delta_normalized_entropy",
        "delta_msp",
        "delta_margin",
        "delta_verbalized_confidence",
        "delta_consistency_entropy",
        "js_divergence",
    )
    grouped = _group(
        (
            shift
            for shift in _select_routing_split(
                shifts,
                routing_split=routing_split,
            )
            if not shift.flip
        ),
        lambda shift: (
            shift.model_name,
            shift.family,
            shift.direction,
            shift.dose,
            shift.clean_tie,
        ),
    )
    rows: list[dict[str, Any]] = []
    for key in sorted(grouped, key=repr):
        group = grouped[key]
        for metric in metrics:
            valid = [shift for shift in group if shift_metric_value(shift, metric) is not None]
            if not valid:
                continue
            interval = cluster_percentile_interval(
                valid,
                cluster_key=lambda shift: shift.question_id,
                statistic=lambda sample, name=metric: sum(
                    float(shift_metric_value(item, name)) for item in sample
                )
                / len(sample),
                n_resamples=n_resamples,
                seed=seed,
            )
            primary = (
                metric == "signed_cue_mass"
                and key[2] == "incongruent"
                and not key[4]
            )
            p_value = (
                cluster_sign_flip_p_value(
                    valid,
                    cluster_key=lambda shift: shift.question_id,
                    value=lambda shift: float(shift.signed_cue_mass),
                    n_permutations=n_resamples,
                    seed=seed,
                )
                if primary
                else None
            )
            rows.append(
                {
                    "model_name": key[0],
                    "routing_split": routing_split,
                    "family": key[1],
                    "direction": key[2],
                    "dose": key[3],
                    "clean_tie": key[4],
                    "non_flipped_only": True,
                    "metric": metric,
                    "n": len(valid),
                    "n_questions": interval.n_clusters,
                    "estimate": interval.estimate,
                    "ci_low": interval.low,
                    "ci_high": interval.high,
                    "confidence": interval.confidence,
                    "primary": primary,
                    "p_value_one_sided": p_value,
                    "decision_rule": "ci_low > 0" if metric == "signed_cue_mass" else "",
                }
            )
    primary_rows = [
        row for row in rows
        if row["primary"] and row["p_value_one_sided"] is not None
    ]
    adjusted = holm_adjust(
        [float(row["p_value_one_sided"]) for row in primary_rows]
    )
    for row, adjusted_value in zip(primary_rows, adjusted, strict=True):
        row["p_value_holm"] = adjusted_value
    return rows


def summarize_susceptibility(
    shifts: Sequence[PairedShift],
    *,
    routing_split: str,
    n_resamples: int,
    seed: int,
) -> list[dict[str, Any]]:
    metrics = (
        ("signed_cue_mass", "entropy"),
        ("delta_entropy", "entropy"),
        ("delta_msp", "msp"),
        ("delta_margin", "margin"),
        ("delta_consistency_entropy", "consistency_entropy"),
        ("js_divergence", "entropy"),
    )
    grouped = _group(
        (
            shift
            for shift in _select_routing_split(
                shifts,
                routing_split=routing_split,
            )
            if shift.direction == "incongruent" and not shift.clean_tie
        ),
        lambda shift: (shift.model_name, shift.family, shift.direction),
    )
    rows: list[dict[str, Any]] = []
    for key in sorted(grouped, key=repr):
        group = grouped[key]
        for shift_metric, baseline in metrics:
            result = low_dose_susceptibility_auc_with_cluster_bootstrap(
                group,
                shift_metric=shift_metric,
                baseline_channel=baseline,
                n_resamples=n_resamples,
                seed=seed,
            )
            rows.append(
                {
                    **asdict(result.estimate),
                    "routing_split": routing_split,
                    "clean_tie": False,
                    **{
                        field: value
                        for field, value in asdict(result).items()
                        if field != "estimate"
                    },
                    "primary": shift_metric == "signed_cue_mass",
                    "decision_rule": (
                        "auc_difference_ci_low > 0"
                        if shift_metric == "signed_cue_mass"
                        else ""
                    ),
                }
            )
    return rows


def calibration_outputs(
    predictions: Sequence[ScoredPrediction],
    *,
    n_bins: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    reliability: list[dict[str, Any]] = []
    for key, group in sorted(_group(predictions, _condition_key).items(), key=lambda item: repr(item[0])):
        condition = _condition_columns(key)
        for confidence_channel in CONFIDENCE_CHANNELS:
            result = calibration_summary(
                group,
                n_bins=n_bins,
                confidence_channel=confidence_channel,
            )
            if result.n == 0:
                continue
            summaries.append(
                {
                    **condition,
                    "confidence_channel": confidence_channel,
                    "n": result.n,
                    "total_n": len(group),
                    "missing_n": len(group) - result.n,
                    "availability_rate": result.n / len(group),
                    "brier_n": result.brier_n,
                    "ece": result.ece,
                    "brier": result.brier,
                    "accuracy": result.accuracy,
                    "n_bins": n_bins,
                    "tie_policy": "strict_three_class",
                }
            )
            reliability.extend(
                {
                    **condition,
                    "confidence_channel": confidence_channel,
                    **asdict(bin_),
                    "tie_policy": "strict_three_class",
                }
                for bin_ in result.bins
            )
    return summaries, reliability


def risk_coverage_outputs(predictions: Sequence[ScoredPrediction]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, group in sorted(_group(predictions, _condition_key).items(), key=lambda item: repr(item[0])):
        condition = _condition_columns(key)
        for confidence_channel in CONFIDENCE_CHANNELS:
            result = risk_coverage_curve(
                group,
                confidence_channel=confidence_channel,
            )
            if result.n == 0:
                continue
            rows.extend(
                {
                    **condition,
                    **asdict(point),
                    "aurc": result.aurc,
                    "confidence_channel": confidence_channel,
                }
                for point in result.points
            )
    return rows


def threshold_transfer_outputs(
    clean_predictions: Sequence[ScoredPrediction],
    biased_predictions: Sequence[ScoredPrediction],
    *,
    target_risks: Sequence[float],
    aggregation: str,
    n_resamples: int,
    seed: int,
    confidence_channels: Sequence[str] = CONFIDENCE_CHANNELS,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    test_bootstrap_cache: dict[
        tuple[tuple[str, ...], int, int],
        ClusterBootstrapDraws,
    ] = {}

    def reusable_test_bootstrap(
        predictions: Sequence[ScoredPrediction],
    ) -> ClusterBootstrapDraws:
        cluster_keys = tuple(
            sorted({item.question_id for item in predictions}, key=repr)
        )
        cache_key = (cluster_keys, n_resamples, seed + 1)
        cached = test_bootstrap_cache.get(cache_key)
        if cached is None:
            cached = cluster_bootstrap_draws(
                predictions,
                n_resamples=n_resamples,
                seed=seed + 1,
            )
            test_bootstrap_cache[cache_key] = cached
        return cached

    clean_by_model = _group(
        clean_predictions,
        lambda item: (item.model_name, item.ordering),
    )
    biased_groups = _group(biased_predictions, _condition_key)
    for model_key in sorted(clean_by_model, key=repr):
        calibration = [
            item for item in clean_by_model[model_key] if item.routing_split == "calibration"
        ]
        clean_test = [
            item for item in clean_by_model[model_key] if item.routing_split == "test"
        ]
        for confidence_channel in confidence_channels:
            calibration_available = any(
                confidence_value(item, confidence_channel) is not None
                and item.human_winner is not None
                for item in calibration
            )
            if not calibration_available:
                continue
            for target_risk in target_risks:
                threshold_bootstrap = bootstrap_threshold_rules(
                    calibration,
                    target_risk=target_risk,
                    confidence_channel=confidence_channel,
                    n_resamples=n_resamples,
                    seed=seed,
                )
                clean_channel_available = any(
                    confidence_value(item, confidence_channel) is not None
                    and item.human_winner is not None
                    for item in clean_test
                )
                if clean_channel_available:
                    clean_transfer = clean_calibrated_threshold_transfer_with_cluster_bootstrap(
                        calibration,
                        clean_test,
                        target_risk=target_risk,
                        confidence_channel=confidence_channel,
                        n_resamples=n_resamples,
                        seed=seed,
                        threshold_bootstrap=threshold_bootstrap,
                        test_bootstrap=reusable_test_bootstrap(clean_test),
                    )
                    rows.append(
                        {
                            "model_name": model_key[0],
                            "ordering": model_key[1],
                            "family": "clean",
                            "direction": "clean",
                            "dose": None,
                            "variant_id": "clean",
                            "clean_tie": "all",
                            "routing_split": "test",
                            "aggregation": aggregation,
                            **asdict(clean_transfer.rule),
                            **{
                                f"test_{field}": value
                                for field, value in asdict(clean_transfer.estimate).items()
                                if field
                                not in {
                                    "confidence_channel",
                                    "target_risk",
                                    "threshold",
                                }
                            },
                            **{
                                field: value
                                for field, value in asdict(clean_transfer).items()
                                if field not in {"rule", "estimate"}
                            },
                        }
                    )
                for key in sorted(biased_groups, key=repr):
                    if (
                        key[0] != model_key[0]
                        or key[1] != model_key[1]
                        or key[-1] != "test"
                    ):
                        continue
                    biased_group = biased_groups[key]
                    if not any(
                        confidence_value(item, confidence_channel) is not None
                        and item.human_winner is not None
                        for item in biased_group
                    ):
                        continue
                    transfer = clean_calibrated_threshold_transfer_with_cluster_bootstrap(
                        calibration,
                        biased_group,
                        target_risk=target_risk,
                        confidence_channel=confidence_channel,
                        n_resamples=n_resamples,
                        seed=seed,
                        threshold_bootstrap=threshold_bootstrap,
                        test_bootstrap=reusable_test_bootstrap(biased_group),
                    )
                    rows.append(
                        {
                            **_condition_columns(key),
                            "aggregation": aggregation,
                            **asdict(transfer.rule),
                            **{
                                f"test_{field}": value
                                for field, value in asdict(transfer.estimate).items()
                                if field
                                not in {
                                    "confidence_channel",
                                    "target_risk",
                                    "threshold",
                                }
                            },
                            **{
                                field: value
                                for field, value in asdict(transfer).items()
                                if field not in {"rule", "estimate"}
                            },
                        }
                    )
    candidates = [
        row
        for row in rows
        if (
            row.get("aggregation") == "single_ordering"
            and row.get("family") in {"authority", "bandwagon"}
            and row.get("direction") == "incongruent"
            and row.get("clean_tie") is False
            and row.get("routing_split") == "test"
            and math.isclose(float(row.get("target_risk", math.nan)), 0.10)
            and row.get("confidence_channel") == "msp"
            and row.get("dose") is not None
        )
    ]
    primary_groups = _group(
        candidates,
        lambda row: (
            row["model_name"],
            row["ordering"],
            row["family"],
        ),
    )
    primary_rows: list[dict[str, Any]] = []
    for group in primary_groups.values():
        highest_dose = max(float(row["dose"]) for row in group)
        for row in group:
            if float(row["dose"]) == highest_dose:
                row["primary"] = True
                row["decision_rule"] = "risk_inflation_vs_target_ci_low > 0"
                primary_rows.append(row)
    for row in rows:
        row.setdefault("primary", False)
    tested_rows = [
        row
        for row in primary_rows
        if row.get("risk_inflation_vs_target_p_value_one_sided") is not None
    ]
    adjusted = holm_adjust(
        [
            float(row["risk_inflation_vs_target_p_value_one_sided"])
            for row in tested_rows
        ]
    )
    for row, adjusted_value in zip(tested_rows, adjusted, strict=True):
        row["p_value_holm"] = adjusted_value
    return rows


def mcnemar_outputs(shifts: Sequence[PairedShift]) -> list[dict[str, Any]]:
    grouped = _group(
        shifts,
        lambda shift: (
            shift.model_name,
            shift.ordering,
            shift.family,
            shift.direction,
            shift.dose,
            shift.clean_tie,
            shift.routing_split,
        ),
    )
    rows: list[dict[str, Any]] = []
    for key in sorted(grouped, key=repr):
        group = grouped[key]
        result = paired_correctness_mcnemar(
            [shift.clean_verdict for shift in group],
            [shift.cued_verdict for shift in group],
            [shift.human_winner for shift in group],
        )
        rows.append(
            {
                "model_name": key[0],
                "ordering": key[1],
                "family": key[2],
                "direction": key[3],
                "dose": key[4],
                "clean_tie": key[5],
                "routing_split": key[6],
                **asdict(result),
                "test": "exact_two_sided_mcnemar_clean_vs_cued_correctness",
                "tie_policy": "strict_three_class",
                "primary": (
                    key[3] == "incongruent"
                    and not key[5]
                    and key[6] == "test"
                ),
            }
        )
    adjusted = holm_adjust([float(row["p_value"]) for row in rows])
    for row, adjusted_value in zip(rows, adjusted, strict=True):
        row["p_value_holm"] = adjusted_value
    return rows


def dose_response_outputs(
    shifts: Sequence[PairedShift],
    *,
    routing_split: str,
    n_resamples: int,
    seed: int,
) -> list[dict[str, Any]]:
    grouped = _group(
        _select_routing_split(
            shifts,
            routing_split=routing_split,
        ),
        lambda shift: (
            shift.model_name,
            shift.family,
            shift.direction,
            shift.clean_tie,
        ),
    )
    rows: list[dict[str, Any]] = []
    for key in sorted(grouped, key=repr):
        observations = dose_observations_from_shifts(grouped[key])
        if len({observation.dose for observation in observations}) < 2:
            continue
        result = fit_dose_response_with_cluster_bootstrap(
            observations,
            n_resamples=n_resamples,
            seed=seed,
        )
        rows.append(
            {
                "model_name": key[0],
                "routing_split": routing_split,
                "family": key[1],
                "direction": key[2],
                "clean_tie": key[3],
                **asdict(result),
                "primary": key[2] == "incongruent" and not key[3],
                "decision_rule": (
                    "slope_ci_low > 0"
                    if key[2] == "incongruent" and not key[3]
                    else ""
                ),
                "dose_units": "annotator_percent" if key[1] == "bandwagon" else "ordinal",
            }
        )
    primary_rows = [
        row
        for row in rows
        if row["primary"] and row.get("slope_p_value_one_sided") is not None
    ]
    adjusted = holm_adjust(
        [float(row["slope_p_value_one_sided"]) for row in primary_rows]
    )
    for row, adjusted_value in zip(primary_rows, adjusted, strict=True):
        row["p_value_holm"] = adjusted_value
    return rows


def _gee_slope_cluster_bootstrap(
    observations: Sequence[TrendObservation],
    *,
    n_resamples: int,
    seed: int,
    workers: int = 1,
    confidence: float = 0.95,
) -> tuple[float | None, float | None, int]:
    """Bootstrap the GEE slope over questions, re-keying duplicate draws."""

    if n_resamples < 1:
        return None, None, 0
    draw_slopes = cluster_bootstrap_uncertainty_gee_slopes(
        [
            {
                "question_id": observation.question_id,
                "normalized_dose": observation.dose,
                "uncertainty": observation.value,
            }
            for observation in observations
        ],
        n_resamples=n_resamples,
        seed=seed,
        workers=workers,
    )
    slopes = [slope for slope in draw_slopes if slope is not None]
    if not slopes:
        return None, None, 0
    alpha = 1.0 - confidence
    return (
        percentile(slopes, alpha / 2.0),
        percentile(slopes, 1.0 - alpha / 2.0),
        len(slopes),
    )


def uncertainty_trend_outputs(
    shifts: Sequence[PairedShift],
    *,
    routing_split: str,
    n_permutations: int,
    n_resamples: int = 0,
    seed: int,
    gee_bootstrap_workers: int = 1,
) -> list[dict[str, Any]]:
    if gee_bootstrap_workers < 1:
        raise ValueError("gee_bootstrap_workers must be a positive integer")
    metrics = (
        "cued_entropy",
        "delta_entropy",
        "cued_consistency_entropy",
        "delta_consistency_entropy",
    )
    grouped = _group(
        (
            shift
            for shift in _select_routing_split(
                shifts,
                routing_split=routing_split,
            )
            if shift.family in SOCIAL_DOSE_LADDERS and shift.dose is not None
        ),
        lambda shift: (shift.model_name, shift.family, shift.direction, shift.clean_tie),
    )
    rows: list[dict[str, Any]] = []
    for key in sorted(grouped, key=repr):
        group = grouped[key]
        first_flip: dict[tuple[str, str, str, str, str], float] = {}
        for shift in group:
            if shift.flip and shift.dose is not None:
                first_flip[shift.susceptibility_key] = min(
                    first_flip.get(shift.susceptibility_key, math.inf),
                    shift.dose,
                )
        subsets = {
            "non_flipped_at_current_dose": [
                shift for shift in group if not shift.flip
            ],
            "pre_first_flip": [
                shift
                for shift in group
                if (
                    shift.dose is not None
                    and not shift.flip
                    and shift.dose
                    < first_flip.get(shift.susceptibility_key, math.inf)
                )
            ],
        }
        for stable_set, stable in subsets.items():
            for metric in metrics:
                observations: list[TrendObservation] = []
                normalization_error: ValueError | None = None
                for shift in stable:
                    value = getattr(shift, metric)
                    if shift.dose is None or value is None:
                        continue
                    try:
                        normalized_dose = normalized_social_dose(
                            shift.family,
                            float(shift.dose),
                        )
                    except ValueError as exc:
                        normalization_error = exc
                        break
                    observations.append(
                        TrendObservation(
                            question_id=shift.question_id,
                            dose=normalized_dose,
                            value=float(value),
                        )
                    )
                common = {
                    "model_name": key[0],
                    "routing_split": routing_split,
                    "family": key[1],
                    "direction": key[2],
                    "clean_tie": key[3],
                    "metric": metric,
                    "stable_set": stable_set,
                    "alternative": "monotonic_increase",
                    "dose_scale": "canonical_four_level_normalized_0_1",
                    "normalized_dose_levels": NORMALIZED_FOUR_LEVEL_DOSES,
                    "raw_dose_ladder": SOCIAL_DOSE_LADDERS[key[1]],
                }
                primary = (
                    key[2] == "incongruent"
                    and not key[3]
                    and metric == "cued_entropy"
                    and stable_set == "pre_first_flip"
                )
                gee_inputs = [
                    {
                        "question_id": observation.question_id,
                        "normalized_dose": observation.dose,
                        "uncertainty": observation.value,
                    }
                    for observation in observations
                ]
                try:
                    if normalization_error is not None:
                        raise normalization_error
                    gee_result = fit_uncertainty_gee(gee_inputs)
                except (
                    OptionalAnalysisDependencyError,
                    ValueError,
                    RuntimeError,
                ) as exc:
                    rows.append(
                        {
                            **common,
                            "estimator": "gaussian_gee_exchangeable",
                            "test": "question_clustered_gaussian_gee",
                            "formula": UNCERTAINTY_GEE_FORMULA,
                            "group_column": "question_id",
                            "status": "unavailable",
                            "message": str(exc),
                            "n": len(observations),
                            "n_clusters": len(
                                {
                                    observation.question_id
                                    for observation in observations
                                }
                            ),
                            "intercept": None,
                            "slope": None,
                            "statistic": None,
                            "slope_standard_error": None,
                            "slope_z_value": None,
                            "slope_p_value_one_sided": None,
                            "slope_ci_low": None,
                            "slope_ci_high": None,
                            "bootstrap_resamples_requested": (
                                n_resamples if primary else 0
                            ),
                            "bootstrap_resamples_successful": 0,
                            "p_value": None,
                            "converged": None,
                            "primary": primary,
                            "sensitivity_analysis": False,
                            "decision_rule": (
                                "slope_ci_low > 0 and p_value_holm < 0.05"
                                if primary
                                else ""
                            ),
                        }
                    )
                else:
                    slope_ci_low: float | None = None
                    slope_ci_high: float | None = None
                    bootstrap_successful = 0
                    if primary:
                        (
                            slope_ci_low,
                            slope_ci_high,
                            bootstrap_successful,
                        ) = _gee_slope_cluster_bootstrap(
                            observations,
                            n_resamples=n_resamples,
                            seed=seed,
                            workers=gee_bootstrap_workers,
                        )
                    rows.append(
                        {
                            **common,
                            **asdict(gee_result),
                            "estimator": "gaussian_gee_exchangeable",
                            "test": "question_clustered_gaussian_gee",
                            "status": "ok",
                            "statistic": gee_result.slope,
                            "p_value": gee_result.slope_p_value_one_sided,
                            "slope_ci_low": slope_ci_low,
                            "slope_ci_high": slope_ci_high,
                            "bootstrap_resamples_requested": (
                                n_resamples if primary else 0
                            ),
                            "bootstrap_resamples_successful": bootstrap_successful,
                            "primary": primary,
                            "sensitivity_analysis": False,
                            "decision_rule": (
                                "slope_ci_low > 0 and p_value_holm < 0.05"
                                if primary
                                else ""
                            ),
                        }
                    )
                if normalization_error is not None:
                    continue
                if len({observation.dose for observation in observations}) < 2:
                    continue
                try:
                    result = clustered_monotonic_trend_test(
                        observations,
                        n_permutations=n_permutations,
                        seed=seed,
                    )
                except ValueError:
                    continue
                rows.append(
                    {
                        **common,
                        **asdict(result),
                        "estimator": "cluster_permutation_slope",
                        "test": "within_question_cluster_permutation_slope",
                        "formula": "uncertainty ~ normalized_dose",
                        "status": "ok",
                        "slope": result.statistic,
                        "slope_standard_error": None,
                        "slope_z_value": None,
                        "slope_p_value_one_sided": result.p_value,
                        "slope_ci_low": None,
                        "slope_ci_high": None,
                        "bootstrap_resamples_requested": 0,
                        "bootstrap_resamples_successful": 0,
                        "primary": False,
                        "sensitivity_analysis": True,
                        "decision_rule": "",
                    }
                )
    primary_rows = [
        row
        for row in rows
        if (
            row["primary"]
            and row["estimator"] == "gaussian_gee_exchangeable"
            and row.get("slope_p_value_one_sided") is not None
            and math.isfinite(float(row["slope_p_value_one_sided"]))
        )
    ]
    adjusted = holm_adjust(
        [float(row["slope_p_value_one_sided"]) for row in primary_rows]
    )
    for row, adjusted_value in zip(primary_rows, adjusted, strict=True):
        row["p_value_holm"] = adjusted_value
    return rows


def uncertainty_by_dose_outputs(
    shifts: Sequence[PairedShift],
    *,
    routing_split: str,
    n_resamples: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Primary pre-first-flip entropy means with question-bootstrap intervals."""

    eligible = [
        shift
        for shift in _select_routing_split(
            shifts,
            routing_split=routing_split,
        )
        if (
            shift.family in SOCIAL_DOSE_LADDERS
            and shift.direction == "incongruent"
            and not shift.clean_tie
            and shift.dose is not None
            and shift.cued_entropy is not None
        )
    ]
    first_flip: dict[tuple[str, str, str, str, str], float] = {}
    for shift in eligible:
        if shift.flip and shift.dose is not None:
            first_flip[shift.susceptibility_key] = min(
                first_flip.get(shift.susceptibility_key, math.inf),
                shift.dose,
            )
    stable = [
        shift
        for shift in eligible
        if (
            not shift.flip
            and shift.dose is not None
            and shift.dose
            < first_flip.get(shift.susceptibility_key, math.inf)
        )
    ]
    grouped = _group(
        stable,
        lambda shift: (
            shift.model_name,
            shift.family,
            float(shift.dose),
        ),
    )
    rows: list[dict[str, Any]] = []
    for key in sorted(grouped, key=repr):
        group = grouped[key]
        interval = cluster_percentile_interval(
            group,
            cluster_key=lambda shift: shift.question_id,
            statistic=lambda records: sum(
                float(shift.cued_entropy)
                for shift in records
                if shift.cued_entropy is not None
            )
            / len(records),
            n_resamples=n_resamples,
            seed=seed,
        )
        rows.append(
            {
                "model_name": key[0],
                "routing_split": routing_split,
                "family": key[1],
                "direction": "incongruent",
                "clean_tie": False,
                "dose": key[2],
                "normalized_dose": normalized_social_dose(key[1], key[2]),
                "metric": "cued_entropy",
                "stable_set": "pre_first_flip",
                "estimate": interval.estimate,
                "ci_low": interval.low,
                "ci_high": interval.high,
                "confidence": interval.confidence,
                "n": len(group),
                "n_clusters": interval.n_clusters,
                "n_resamples": interval.n_resamples,
                "primary": True,
            }
        )
    return rows


def modeling_outputs(
    shifts: Sequence[PairedShift],
    *,
    routing_split: str,
) -> list[dict[str, Any]]:
    selected = _select_routing_split(
        shifts,
        routing_split=routing_split,
    )
    by_model = _group(
        (shift for shift in selected if not shift.clean_tie),
        lambda shift: (shift.model_name,),
    )
    rows: list[dict[str, Any]] = []
    for model_key in sorted(by_model, key=repr):
        metadata = {
            "dose_variable": "dose",
            "dose_scale": "canonical_four_level_normalized_0_1",
            "normalized_dose_levels": NORMALIZED_FOUR_LEVEL_DOSES,
            "raw_dose_ladders": SOCIAL_DOSE_LADDERS,
        }
        try:
            inputs = [
                {
                    "question_id": shift.question_id,
                    "flip": shift.flip,
                    "dose": normalized_social_dose(
                        shift.family,
                        float(shift.dose),
                    ),
                    "family": shift.family,
                    "congruence": shift.direction,
                }
                for shift in by_model[model_key]
                if (
                    shift.dose is not None
                    and shift.family in SOCIAL_DOSE_LADDERS
                )
            ]
            result = fit_flip_mixed_logit(inputs)
        except (OptionalAnalysisDependencyError, ValueError, RuntimeError) as exc:
            rows.append(
                {
                    "model_name": model_key[0],
                    "routing_split": routing_split,
                    "clean_tie": False,
                    "formula": MIXED_EFFECTS_FORMULA,
                    **metadata,
                    "status": "unavailable",
                    "message": str(exc),
                }
            )
            continue
        model_rows = [
            {
                "model_name": model_key[0],
                "routing_split": routing_split,
                "clean_tie": False,
                "formula": result.formula,
                "group_column": result.group_column,
                "n": result.n,
                "status": "ok",
                "model_type": "binomial_random_intercept",
                "fit_method": result.fit_method,
                "converged": result.converged,
                "warnings": result.warnings,
                **metadata,
                **asdict(coefficient),
            }
            for coefficient in result.coefficients
        ]
        adjusted = holm_adjust([float(row["p_value"]) for row in model_rows])
        for row, adjusted_value in zip(model_rows, adjusted, strict=True):
            row["p_value_holm"] = adjusted_value
        rows.extend(model_rows)
    return rows


def _attach_flip(
    records: Sequence[ConditionRecord],
    shifts: Sequence[PairedShift],
    clean_records: Sequence[ConditionRecord],
) -> list[ScoredPrediction]:
    shift_by_record = {shift.cued_record_id: shift for shift in shifts}
    clean_by_id = {record.record_id: record for record in clean_records}
    predictions: list[ScoredPrediction] = []
    for record in records:
        shift = shift_by_record.get(record.record_id)
        base_prediction = prediction_from_record(record)
        clean_prediction = (
            prediction_from_record(clean_by_id[shift.clean_record_id])
            if shift is not None and shift.clean_record_id in clean_by_id
            else None
        )
        prediction = replace(
            base_prediction,
            flip=shift.flip if shift is not None else None,
            consistency_flip=(
                paired_channel_flip(
                    clean_prediction,
                    base_prediction,
                    "consistency_agreement",
                )
                if clean_prediction is not None
                else None
            ),
            verbalized_flip=(
                paired_channel_flip(
                    clean_prediction,
                    base_prediction,
                    "verbalized_confidence",
                )
                if clean_prediction is not None
                else None
            ),
        )
        if prediction.human_winner is None and shift is not None:
            prediction = replace(prediction, human_winner=shift.human_winner)
        if shift is not None:
            prediction = replace(prediction, clean_tie=shift.clean_tie)
        predictions.append(prediction)
    return predictions


def _annotate_cued_records(
    records: Sequence[ConditionRecord],
    shifts: Sequence[PairedShift],
) -> tuple[ConditionRecord, ...]:
    shift_by_record = {shift.cued_record_id: shift for shift in shifts}
    annotated: list[ConditionRecord] = []
    for record in records:
        shift = shift_by_record.get(record.record_id)
        if shift is None:
            annotated.append(record)
            continue
        annotated.append(
            replace(
                record,
                clean_tie=shift.clean_tie,
                human_winner=record.human_winner or shift.human_winner,
            )
        )
    return tuple(annotated)


def _attach_swap_flip(
    clean: Sequence[ScoredPrediction],
    biased: Sequence[ScoredPrediction],
) -> tuple[ScoredPrediction, ...]:
    clean_map = {
        (prediction.model_name, prediction.pair_key): prediction
        for prediction in clean
    }
    result: list[ScoredPrediction] = []
    for prediction in biased:
        clean_prediction = clean_map.get((prediction.model_name, prediction.pair_key))
        flip = (
            None
            if clean_prediction is None
            else prediction.verdict != clean_prediction.verdict
        )
        result.append(replace(prediction, flip=flip))
    return tuple(result)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute the Silent Bias RQ1-RQ3 analysis package from Stage A/B flat JSONL.",
    )
    parser.add_argument("--stage-a", type=Path, nargs="+", required=True)
    parser.add_argument("--stage-b", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument(
        "--gee-bootstrap-workers",
        type=_positive_int,
        default=1,
        help=(
            "worker processes for RQ3 GEE bootstrap refits; operational only "
            "and excluded from the scientific spec hash"
        ),
    )
    parser.add_argument("--trend-permutations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ece-bins", type=int, default=10)
    parser.add_argument("--target-risk", type=float, nargs="+", default=[0.10, 0.20])
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    stage_a_rows = read_jsonl(args.stage_a)
    stage_b_rows = read_jsonl(args.stage_b)
    clean_records = tuple(record_from_mapping(row) for row in stage_a_rows)
    cued_records = tuple(record_from_mapping(row) for row in stage_b_rows)
    pairing = pair_clean_and_cued(clean_records, cued_records)
    if pairing.unmatched_cued_record_ids:
        preview = ", ".join(pairing.unmatched_cued_record_ids[:5])
        raise ValueError(
            f"{len(pairing.unmatched_cued_record_ids)} cued records lack a clean partner: {preview}"
        )
    shifts = compute_paired_shifts(pairing.pairs)

    spec = {
        "analysis_version": ANALYSIS_VERSION,
        "bootstrap_resamples": args.bootstrap_resamples,
        "trend_permutations": args.trend_permutations,
        "seed": args.seed,
        "ece_bins": args.ece_bins,
        "target_risks": list(args.target_risk),
        "confidence_channels": list(CONFIDENCE_CHANNELS),
        "confidence_verdict_policy": {
            "msp": "deterministic_logit_verdict",
            "consistency_agreement": "consistency_majority_verdict",
            "verbalized_confidence": "verbalized_pass_verdict",
        },
        "missing_channel_verdict_policy": "exclude_without_fallback",
        "accepted_flip_policy": "matching_channel_clean_vs_cued_verdict",
        "tie_policy": "strict_three_class",
        "routing_split_policy": {
            "calibration": "threshold_selection_only",
            "headline_estimation": HEADLINE_ROUTING_SPLIT,
        },
        "mixed_model_population": {
            "routing_split": HEADLINE_ROUTING_SPLIT,
            "clean_tie": False,
        },
        "formula": MIXED_EFFECTS_FORMULA,
    }
    spec_hash = spec_sha256(spec)
    stage_a_hashes = input_hashes(args.stage_a)
    stage_b_hashes = input_hashes(args.stage_b)
    provenance_columns = {
        "spec_hash": spec_hash,
        "input_hashes": canonical_json(
            {
                "stage_a": sorted(stage_a_hashes.values()),
                "stage_b": sorted(stage_b_hashes.values()),
            }
        ),
    }

    annotated_cued_records = _annotate_cued_records(cued_records, shifts)
    clean_predictions = [
        prediction_from_record(
            record,
            flip=False,
            consistency_flip=(
                False if record.consistency_majority_verdict is not None else None
            ),
            verbalized_flip=False if record.verbalized_verdict is not None else None,
        )
        for record in clean_records
    ]
    biased_predictions = _attach_flip(
        annotated_cued_records,
        shifts,
        clean_records,
    )
    all_predictions = [*clean_predictions, *biased_predictions]
    rq2_calibration, rq2_reliability = calibration_outputs(
        all_predictions,
        n_bins=args.ece_bins,
    )

    swap_clean = swap_average_records(clean_records)
    swap_biased = _attach_swap_flip(
        swap_clean,
        swap_average_records(annotated_cued_records),
    )
    transfer_rows = threshold_transfer_outputs(
        clean_predictions,
        biased_predictions,
        target_risks=args.target_risk,
        aggregation="single_ordering",
        n_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    if swap_clean and swap_biased:
        transfer_rows.extend(
            threshold_transfer_outputs(
                swap_clean,
                swap_biased,
                target_risks=args.target_risk,
                aggregation="swap_average",
                n_resamples=args.bootstrap_resamples,
                seed=args.seed,
                confidence_channels=("msp",),
            )
        )

    outputs: dict[str, list[dict[str, Any]]] = {
        "paired_shifts.csv": [shift.to_dict() for shift in shifts],
        "rq1_silent_shift.csv": summarize_silent_shift(
            shifts,
            routing_split=HEADLINE_ROUTING_SPLIT,
            n_resamples=args.bootstrap_resamples,
            seed=args.seed,
        ),
        "rq1_susceptibility.csv": summarize_susceptibility(
            shifts,
            routing_split=HEADLINE_ROUTING_SPLIT,
            n_resamples=args.bootstrap_resamples,
            seed=args.seed,
        ),
        "rq2_calibration.csv": rq2_calibration,
        "rq2_reliability.csv": rq2_reliability,
        "rq2_risk_coverage.csv": risk_coverage_outputs(all_predictions),
        "rq2_threshold_transfer.csv": transfer_rows,
        "rq2_mcnemar.csv": mcnemar_outputs(shifts),
        "rq3_dose_response.csv": dose_response_outputs(
            shifts,
            routing_split=HEADLINE_ROUTING_SPLIT,
            n_resamples=args.bootstrap_resamples,
            seed=args.seed,
        ),
        "rq3_uncertainty_trend.csv": uncertainty_trend_outputs(
            shifts,
            routing_split=HEADLINE_ROUTING_SPLIT,
            n_permutations=args.trend_permutations,
            n_resamples=args.bootstrap_resamples,
            seed=args.seed,
            gee_bootstrap_workers=args.gee_bootstrap_workers,
        ),
        "rq3_uncertainty_by_dose.csv": uncertainty_by_dose_outputs(
            shifts,
            routing_split=HEADLINE_ROUTING_SPLIT,
            n_resamples=args.bootstrap_resamples,
            seed=args.seed,
        ),
        "rq3_modeling.csv": modeling_outputs(
            shifts,
            routing_split=HEADLINE_ROUTING_SPLIT,
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_NAMES:
        write_tidy_csv(
            args.output_dir / name,
            outputs[name],
            provenance_columns=provenance_columns,
        )

    provenance = {
        "analysis_version": ANALYSIS_VERSION,
        "spec": spec,
        "spec_hash": spec_hash,
        "stage_a_input_hashes": sorted(stage_a_hashes.values()),
        "stage_b_input_hashes": sorted(stage_b_hashes.values()),
    }
    provenance_path = args.output_dir / "provenance.json"
    provenance_path.write_text(canonical_json(provenance) + "\n", encoding="utf-8")
    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "spec_hash": spec_hash,
        "pairing": {
            "paired": len(pairing.pairs),
            "unmatched_cued": len(pairing.unmatched_cued_record_ids),
            "unused_clean": len(pairing.unused_clean_record_ids),
        },
        "outputs": {
            name: file_sha256(args.output_dir / name)
            for name in (*OUTPUT_NAMES, "provenance.json")
        },
    }
    (args.output_dir / "analysis_manifest.json").write_text(
        canonical_json(manifest) + "\n",
        encoding="utf-8",
    )

    print(f"Paired records: {len(pairing.pairs)}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
