from __future__ import annotations

import pytest

from biases.analysis.dose_response import (
    DoseObservation,
    TrendObservation,
    clustered_monotonic_trend_test,
    fit_dose_response_with_cluster_bootstrap,
    page_monotonic_trend_test,
)
from biases.analysis.modeling import MIXED_EFFECTS_FORMULA, fit_uncertainty_gee
from biases.analysis.records import record_from_mapping
from biases.analysis.selective import (
    ScoredPrediction,
    calibrate_threshold_at_target_risk,
    calibration_summary,
    clean_calibrated_threshold_transfer_with_cluster_bootstrap,
    evaluate_threshold_transfer,
    paired_correctness_mcnemar,
    risk_coverage_curve,
    swap_average_pair,
    swap_average_records,
)
from biases.analysis.statistics import holm_adjust


def _prediction(
    record_id: str,
    confidence: float,
    *,
    verdict: str,
    human: str,
    probabilities: tuple[float, float, float],
    flip: bool | None = None,
) -> ScoredPrediction:
    return ScoredPrediction(
        record_id=record_id,
        question_id=record_id,
        pair_key=record_id,
        ordering="ab",
        model_name="judge",
        routing_split="test",
        family="authority",
        direction="incongruent",
        dose=4,
        variant_id="authority_incongruent_4_ab",
        verdict=verdict,
        human_winner=human,
        probabilities=probabilities,
        confidence=confidence,
        flip=flip,
    )


def test_multiclass_calibration_and_tie_grouped_risk_coverage() -> None:
    predictions = [
        _prediction("p1", 0.8, verdict="A", human="A", probabilities=(0.8, 0.1, 0.1)),
        _prediction("p2", 0.6, verdict="A", human="B", probabilities=(0.6, 0.3, 0.1)),
    ]
    calibration = calibration_summary(predictions, n_bins=2)
    curve = risk_coverage_curve(predictions)

    assert calibration.brier == pytest.approx((0.06 + 0.86) / 2)
    assert calibration.ece == pytest.approx(0.2)
    assert calibration.accuracy == pytest.approx(0.5)
    assert curve.points[1].coverage == pytest.approx(0.5)
    assert curve.points[1].risk == pytest.approx(0.0)
    assert curve.points[-1].risk == pytest.approx(0.5)
    assert curve.aurc == pytest.approx(0.125)


def test_clean_threshold_transfer_counts_confident_flips() -> None:
    clean = [
        _prediction("c1", 0.9, verdict="A", human="A", probabilities=(0.9, 0.05, 0.05)),
        _prediction("c2", 0.8, verdict="A", human="B", probabilities=(0.8, 0.1, 0.1)),
    ]
    rule = calibrate_threshold_at_target_risk(clean, target_risk=0.0)
    biased = [
        _prediction(
            "b1",
            0.95,
            verdict="B",
            human="A",
            probabilities=(0.03, 0.95, 0.02),
            flip=True,
        ),
        _prediction(
            "b2",
            0.4,
            verdict="A",
            human="A",
            probabilities=(0.4, 0.35, 0.25),
            flip=False,
        ),
    ]
    transfer = evaluate_threshold_transfer(biased, rule)

    assert rule.threshold == pytest.approx(0.9)
    assert rule.calibration_coverage == pytest.approx(0.5)
    assert transfer.coverage == pytest.approx(0.5)
    assert transfer.realized_risk == pytest.approx(1.0)
    assert transfer.accepted_flip_fraction == pytest.approx(1.0)
    bootstrap = clean_calibrated_threshold_transfer_with_cluster_bootstrap(
        clean,
        biased,
        target_risk=0.0,
        n_resamples=100,
        seed=42,
    )
    assert bootstrap.risk_inflation_vs_target_ci_low is not None
    assert bootstrap.risk_inflation_vs_target_ci_high is not None
    assert bootstrap.risk_inflation_vs_target_p_value_one_sided is not None
    assert bootstrap.accepted_flip_fraction_ci_low is not None
    assert bootstrap.accepted_flip_fraction_ci_high is not None
    assert bootstrap.n_calibration_clusters == 2
    assert bootstrap.n_test_clusters == 2


def test_exact_mcnemar_uses_paired_clean_and_cued_correctness() -> None:
    result = paired_correctness_mcnemar(
        ["A", "A", "A"],
        ["B", "B", "B"],
        ["A", "A", "A"],
    )
    assert result.n == 3
    assert result.b_clean_correct_cued_wrong == 3
    assert result.c_clean_wrong_cued_correct == 0
    assert result.p_value == pytest.approx(0.25)


def test_swap_average_maps_ba_labels_back_to_canonical_order() -> None:
    common = {
        "question_id": "q1",
        "pair_key": "q1",
        "condition_group_id": "q1-group",
        "model_name": "judge",
        "bias_name": "clean",
        "variant_id": "clean",
        "human_winner": "A",
    }
    ab = record_from_mapping(
        {
            **common,
            "record_id": "ab",
            "example_id": "q1:ab",
            "ordering": "ab",
            "verdict": "A",
            "label_prob_A": 0.7,
            "label_prob_B": 0.2,
            "label_prob_tie": 0.1,
        }
    )
    ba = record_from_mapping(
        {
            **common,
            "record_id": "ba",
            "example_id": "q1:ba",
            "ordering": "ba",
            "human_winner": "B",
            "verdict": "B",
            "label_prob_A": 0.1,
            "label_prob_B": 0.8,
            "label_prob_tie": 0.1,
        }
    )
    averaged = swap_average_pair(ab, ba)
    assert averaged.verdict == "A"
    assert averaged.ordering == "swap_average"
    assert averaged.pair_key == "q1"
    assert averaged.probabilities == pytest.approx((0.75, 0.15, 0.10))
    assert averaged.human_winner == "A"


def test_swap_average_keeps_duplicate_judgments_and_flags_any_clean_tie() -> None:
    records = []
    for identity, ab_clean_tie, ba_clean_tie in (
        ("judgment-1", False, True),
        ("judgment-2", False, False),
    ):
        common = {
            "question_id": "shared-question",
            "pair_key": "legacy-colliding-key",
            "pair_identity_key": identity,
            "model_name": "judge",
            "bias_name": "authority",
            "direction": "incongruent",
            "dose": 1,
            "human_winner": "A",
        }
        records.append(
            record_from_mapping(
                {
                    **common,
                    "record_id": f"{identity}-ab",
                    "example_id": "shared-question:ab",
                    "ordering": "ab",
                    "variant_id": "authority_incongruent_1_ab",
                    "cue_target": "B",
                    "verdict": "A",
                    "clean_tie": ab_clean_tie,
                    "label_prob_A": 0.7,
                    "label_prob_B": 0.2,
                    "label_prob_tie": 0.1,
                }
            )
        )
        records.append(
            record_from_mapping(
                {
                    **common,
                    "record_id": f"{identity}-ba",
                    "example_id": "shared-question:ba",
                    "ordering": "ba",
                    "variant_id": "authority_incongruent_1_ba",
                    "cue_target": "A",
                    "human_winner": "B",
                    "verdict": "B",
                    "clean_tie": ba_clean_tie,
                    "label_prob_A": 0.1,
                    "label_prob_B": 0.8,
                    "label_prob_tie": 0.1,
                }
            )
        )

    averaged = swap_average_records(records)
    assert len(averaged) == 2
    by_identity = {prediction.pair_key: prediction for prediction in averaged}
    assert set(by_identity) == {"judgment-1", "judgment-2"}
    assert by_identity["judgment-1"].clean_tie is True
    assert by_identity["judgment-2"].clean_tie is False


def test_dose_fit_trend_and_holm_are_deterministic() -> None:
    observations = [
        DoseObservation(question_id=f"q{index}", dose=dose, event=dose >= 3)
        for index in range(1, 5)
        for dose in (1.0, 2.0, 3.0, 4.0)
    ]
    fit = fit_dose_response_with_cluster_bootstrap(
        observations,
        n_resamples=50,
        seed=7,
    )
    assert fit.slope > 0.0
    assert fit.slope_ci_low is not None
    assert fit.slope_p_value_one_sided is not None
    assert fit.slope_p_value_one_sided < 0.05
    assert fit.p25_dose is not None
    assert fit.dose_min == 1.0
    assert fit.dose_max == 4.0
    assert fit.p25_range_status == "within_tested_range"

    trend = page_monotonic_trend_test(
        [
            TrendObservation(f"q{question}", dose, dose + question / 100)
            for question in range(1, 6)
            for dose in (1.0, 2.0, 3.0, 4.0)
        ],
        n_permutations=500,
        seed=3,
    )
    assert trend.statistic > 0
    assert trend.p_value < 0.1
    incomplete_trend = clustered_monotonic_trend_test(
        [
            TrendObservation(f"q{question}", dose, dose + question / 100)
            for question in range(1, 9)
            for dose in ((1.0, 2.0) if question % 2 else (1.0, 3.0, 4.0))
        ],
        n_permutations=500,
        seed=42,
    )
    assert incomplete_trend.statistic > 0.0
    assert incomplete_trend.p_value < 0.1
    assert holm_adjust((0.01, 0.04, 0.03)) == pytest.approx((0.03, 0.06, 0.06))
    assert (
        MIXED_EFFECTS_FORMULA
        == "flip ~ dose * family * congruence + (1 | question)"
    )


def test_degenerate_uncertainty_gee_fails_explicitly() -> None:
    pytest.importorskip("statsmodels")
    rows = [
        {
            "question_id": f"q{question}",
            "normalized_dose": dose,
            "uncertainty": 0.5,
        }
        for question in range(4)
        for dose in (0.0, 1.0)
    ]
    with pytest.raises(
        (RuntimeError, ValueError),
        match="non-finite|degenerate|converge|deviance",
    ):
        fit_uncertainty_gee(rows)
