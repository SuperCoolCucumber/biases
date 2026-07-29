from __future__ import annotations

import math

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
from biases.analysis.resampling import cluster_resamples, percentile
from biases.analysis.selective import (
    RiskCoveragePoint,
    RiskCoverageResult,
    ScoredPrediction,
    bootstrap_threshold_rules,
    calibrate_threshold_at_target_risk,
    calibration_summary,
    cluster_bootstrap_draws,
    clean_calibrated_threshold_transfer_with_cluster_bootstrap,
    confidence_value,
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
    human: str | None,
    probabilities: tuple[float, float, float],
    flip: bool | None = None,
    question_id: str | None = None,
) -> ScoredPrediction:
    return ScoredPrediction(
        record_id=record_id,
        question_id=question_id or record_id,
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


def _brute_force_risk_coverage(
    predictions: list[ScoredPrediction],
    *,
    confidence_channel: str = "msp",
) -> RiskCoverageResult:
    valid = [
        (prediction, confidence)
        for prediction in predictions
        if prediction.human_winner is not None
        and (confidence := confidence_value(prediction, confidence_channel))
        is not None
    ]
    if not valid:
        return RiskCoverageResult(n=0, aurc=None, points=())
    thresholds = sorted({confidence for _, confidence in valid}, reverse=True)
    points = [
        RiskCoveragePoint(
            threshold=math.inf,
            coverage=0.0,
            risk=0.0,
            accepted=0,
            total=len(valid),
        )
    ]
    for threshold in thresholds:
        accepted = [
            prediction
            for prediction, confidence in valid
            if confidence >= threshold
        ]
        errors = sum(
            prediction.verdict != prediction.human_winner
            for prediction in accepted
        )
        points.append(
            RiskCoveragePoint(
                threshold=threshold,
                coverage=len(accepted) / len(valid),
                risk=errors / len(accepted),
                accepted=len(accepted),
                total=len(valid),
            )
        )
    area = sum(
        (right.coverage - left.coverage) * (left.risk + right.risk) / 2.0
        for left, right in zip(points, points[1:])
    )
    return RiskCoverageResult(
        n=len(valid),
        aurc=area,
        points=tuple(points),
    )


def test_sorted_risk_coverage_is_exactly_equivalent_to_brute_force() -> None:
    predictions = [
        _prediction(
            "p1",
            0.9,
            verdict="A",
            human="A",
            probabilities=(0.9, 0.05, 0.05),
        ),
        _prediction(
            "p2",
            0.9,
            verdict="B",
            human="A",
            probabilities=(0.05, 0.9, 0.05),
        ),
        _prediction(
            "p3",
            0.7,
            verdict="B",
            human="B",
            probabilities=(0.1, 0.7, 0.2),
        ),
        _prediction(
            "p4",
            0.4,
            verdict="A",
            human="B",
            probabilities=(0.4, 0.35, 0.25),
        ),
        _prediction(
            "missing-human",
            0.99,
            verdict="A",
            human=None,
            probabilities=(0.99, 0.005, 0.005),
        ),
        _prediction(
            "non-finite-confidence",
            math.nan,
            verdict="A",
            human="A",
            probabilities=(0.8, 0.1, 0.1),
        ),
    ]

    expected = _brute_force_risk_coverage(predictions)
    actual = risk_coverage_curve(predictions)

    assert actual == expected
    rule = calibrate_threshold_at_target_risk(predictions, target_risk=0.5)
    feasible = [
        point
        for point in expected.points
        if point.accepted > 0 and point.risk <= 0.5
    ]
    selected = max(feasible, key=lambda point: (point.coverage, -point.threshold))
    assert rule.threshold == selected.threshold
    assert rule.calibration_coverage == selected.coverage
    assert rule.calibration_risk == selected.risk


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


def test_reused_bootstrap_threshold_rules_preserve_transfer_results() -> None:
    clean = [
        _prediction(
            f"c{index}",
            confidence,
            verdict=verdict,
            human="A",
            probabilities=(confidence, 1.0 - confidence, 0.0),
        )
        for index, (confidence, verdict) in enumerate(
            ((0.95, "A"), (0.8, "B"), (0.7, "A"), (0.6, "B")),
            start=1,
        )
    ]
    biased = [
        _prediction(
            f"b{index}",
            confidence,
            verdict=verdict,
            human="A",
            probabilities=(confidence, 1.0 - confidence, 0.0),
            flip=verdict == "B",
        )
        for index, (confidence, verdict) in enumerate(
            ((0.9, "B"), (0.75, "A"), (0.5, "B")),
            start=1,
        )
    ]
    precomputed = bootstrap_threshold_rules(
        clean,
        target_risk=0.25,
        n_resamples=50,
        seed=7,
    )

    direct = clean_calibrated_threshold_transfer_with_cluster_bootstrap(
        clean,
        biased,
        target_risk=0.25,
        n_resamples=50,
        seed=7,
    )
    reused = clean_calibrated_threshold_transfer_with_cluster_bootstrap(
        clean,
        biased,
        target_risk=0.25,
        n_resamples=50,
        seed=7,
        threshold_bootstrap=precomputed,
    )

    assert reused == direct


def test_reused_threshold_bootstrap_rejects_different_calibration_data() -> None:
    clean = [
        _prediction(
            "clean",
            0.9,
            verdict="A",
            human="A",
            probabilities=(0.9, 0.1, 0.0),
        ),
    ]
    other_clean = [
        _prediction(
            "other-clean",
            0.9,
            verdict="A",
            human="A",
            probabilities=(0.9, 0.1, 0.0),
        ),
    ]
    biased = [
        _prediction(
            "biased",
            0.8,
            verdict="B",
            human="A",
            probabilities=(0.2, 0.8, 0.0),
            flip=True,
        ),
    ]
    cached = bootstrap_threshold_rules(
        clean,
        target_risk=0.1,
        n_resamples=10,
        seed=5,
    )

    with pytest.raises(ValueError, match="calibration data"):
        clean_calibrated_threshold_transfer_with_cluster_bootstrap(
            other_clean,
            biased,
            target_risk=0.1,
            n_resamples=10,
            seed=5,
            threshold_bootstrap=cached,
        )


def test_sufficient_statistic_bootstrap_matches_materialized_cluster_samples() -> None:
    clean = [
        _prediction(
            f"c{index}",
            confidence,
            verdict=verdict,
            human="A",
            probabilities=(confidence, 1.0 - confidence, 0.0),
            question_id=f"clean-q{(index + 1) // 2}",
        )
        for index, (confidence, verdict) in enumerate(
            ((0.95, "A"), (0.8, "B"), (0.7, "A"), (0.6, "B")),
            start=1,
        )
    ]
    biased = [
        _prediction(
            f"b{index}",
            confidence,
            verdict=verdict,
            human="A",
            probabilities=(confidence, 1.0 - confidence, 0.0),
            flip=flip,
            question_id=question_id,
        )
        for index, (confidence, verdict, flip, question_id) in enumerate(
            (
                (0.9, "B", True, "test-q1"),
                (0.75, "A", False, "test-q1"),
                (0.5, "B", True, "test-q2"),
                (0.4, "A", False, "test-q3"),
            ),
            start=1,
        )
    ]
    n_resamples = 80
    seed = 13
    calibrated = bootstrap_threshold_rules(
        clean,
        target_risk=0.25,
        n_resamples=n_resamples,
        seed=seed,
    )
    reference: dict[str, list[float]] = {
        "realized_risk": [],
        "risk_inflation_vs_target": [],
        "risk_inflation_vs_clean_calibration": [],
        "accepted_flip_fraction": [],
    }
    test_samples = cluster_resamples(
        biased,
        cluster_key=lambda prediction: prediction.question_id,
        n_resamples=n_resamples,
        seed=seed + 1,
    )
    for rule, sample in zip(
        calibrated.sampled_rules,
        test_samples,
        strict=True,
    ):
        transfer = evaluate_threshold_transfer(sample, rule)
        for field in reference:
            value = getattr(transfer, field)
            if value is not None and math.isfinite(value):
                reference[field].append(float(value))

    actual = clean_calibrated_threshold_transfer_with_cluster_bootstrap(
        clean,
        biased,
        target_risk=0.25,
        n_resamples=n_resamples,
        seed=seed,
        threshold_bootstrap=calibrated,
        test_bootstrap=cluster_bootstrap_draws(
            biased,
            n_resamples=n_resamples,
            seed=seed + 1,
        ),
    )

    assert actual.realized_risk_ci_low == pytest.approx(
        percentile(reference["realized_risk"], 0.025)
    )
    assert actual.realized_risk_ci_high == pytest.approx(
        percentile(reference["realized_risk"], 0.975)
    )
    assert actual.risk_inflation_vs_target_ci_low == pytest.approx(
        percentile(reference["risk_inflation_vs_target"], 0.025)
    )
    assert actual.risk_inflation_vs_target_ci_high == pytest.approx(
        percentile(reference["risk_inflation_vs_target"], 0.975)
    )
    assert actual.risk_inflation_vs_clean_calibration_ci_low == pytest.approx(
        percentile(reference["risk_inflation_vs_clean_calibration"], 0.025)
    )
    assert actual.risk_inflation_vs_clean_calibration_ci_high == pytest.approx(
        percentile(reference["risk_inflation_vs_clean_calibration"], 0.975)
    )
    assert actual.accepted_flip_fraction_ci_low == pytest.approx(
        percentile(reference["accepted_flip_fraction"], 0.025)
    )
    assert actual.accepted_flip_fraction_ci_high == pytest.approx(
        percentile(reference["accepted_flip_fraction"], 0.975)
    )
    expected_p = (
        1
        + sum(
            value <= 0.0
            for value in reference["risk_inflation_vs_target"]
        )
    ) / (len(reference["risk_inflation_vs_target"]) + 1)
    assert actual.risk_inflation_vs_target_p_value_one_sided == pytest.approx(
        expected_p
    )


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
