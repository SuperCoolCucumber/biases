from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from unittest.mock import patch

import pytest

from biases.analysis.dose_response import normalized_social_dose
from biases.analysis.modeling import (
    MIXED_EFFECTS_FORMULA,
    UNCERTAINTY_GEE_FORMULA,
    OptionalAnalysisDependencyError,
    UncertaintyGEEResult,
)
from biases.analysis.records import pair_clean_and_cued, record_from_mapping
from biases.analysis.rq1 import PairedShift, compute_paired_shifts
from biases.analysis.selective import ScoredPrediction, bootstrap_threshold_rules
from scripts.analyze_silent_bias import (
    calibration_outputs,
    dose_response_outputs,
    modeling_outputs,
    risk_coverage_outputs,
    summarize_silent_shift,
    summarize_susceptibility,
    threshold_transfer_outputs,
    uncertainty_by_dose_outputs,
    uncertainty_trend_outputs,
)


def _prediction(
    record_id: str,
    *,
    split: str,
    family: str,
    direction: str,
    dose: float | None,
    variant_id: str,
    verdict: str = "A",
    human: str = "A",
    clean_tie: bool = False,
    flip: bool = False,
) -> ScoredPrediction:
    return ScoredPrediction(
        record_id=record_id,
        question_id=record_id,
        pair_key=record_id,
        ordering="ab",
        model_name="judge",
        routing_split=split,
        family=family,
        direction=direction,
        dose=dose,
        variant_id=variant_id,
        verdict=verdict,
        human_winner=human,
        probabilities=(0.8, 0.1, 0.1),
        confidence=0.8,
        flip=flip,
        clean_tie=clean_tie,
        msp=0.8,
        consistency_agreement=0.75,
        consistency_majority_verdict=verdict,
        verbalized_confidence=0.7,
        verbalized_verdict=verdict,
        consistency_flip=flip,
        verbalized_flip=flip,
    )


def test_rq2_outputs_split_clean_ties_and_emit_all_confidence_channels() -> None:
    clean = [
        _prediction(
            f"cal-{index}",
            split="calibration",
            family="clean",
            direction="clean",
            dose=None,
            variant_id="clean",
        )
        for index in range(4)
    ] + [
        _prediction(
            f"clean-test-{index}",
            split="test",
            family="clean",
            direction="clean",
            dose=None,
            variant_id="clean",
        )
        for index in range(4)
    ]
    biased = [
        _prediction(
            f"low-{index}",
            split="test",
            family="authority",
            direction="incongruent",
            dose=1,
            variant_id="authority_incongruent_1_ab",
        )
        for index in range(2)
    ] + [
        _prediction(
            f"high-{index}",
            split="test",
            family="authority",
            direction="incongruent",
            dose=4,
            variant_id="authority_incongruent_4_ab",
            verdict="B",
            flip=True,
        )
        for index in range(2)
    ] + [
        _prediction(
            "high-clean-tie",
            split="test",
            family="authority",
            direction="incongruent",
            dose=4,
            variant_id="authority_incongruent_4_ab",
            verdict="tie",
            human="tie",
            clean_tie=True,
        )
    ]

    calibration, _ = calibration_outputs([*clean, *biased], n_bins=5)
    risk = risk_coverage_outputs([*clean, *biased])
    high_calibration = [
        row
        for row in calibration
        if row["variant_id"] == "authority_incongruent_4_ab"
    ]
    high_risk = [
        row
        for row in risk
        if row["variant_id"] == "authority_incongruent_4_ab"
    ]
    assert {row["confidence_channel"] for row in high_calibration} == {
        "msp",
        "consistency_agreement",
        "verbalized_confidence",
    }
    assert {row["clean_tie"] for row in high_calibration} == {False, True}
    assert {
        (row["clean_tie"], row["n"])
        for row in high_calibration
    } == {(False, 2), (True, 1)}
    assert {row["clean_tie"] for row in high_risk} == {False, True}

    transfer = threshold_transfer_outputs(
        clean,
        biased,
        target_risks=(0.10,),
        aggregation="single_ordering",
        n_resamples=50,
        seed=42,
    )
    assert {
        row["confidence_channel"]
        for row in transfer
        if row["family"] == "authority"
    } == {
        "msp",
        "consistency_agreement",
        "verbalized_confidence",
    }
    assert all(
        row["clean_tie"] == "all"
        for row in transfer
        if row["family"] == "clean"
    )
    primary = [row for row in transfer if row["primary"]]
    assert len(primary) == 1
    assert primary[0]["confidence_channel"] == "msp"
    assert all(row["dose"] == 4 for row in primary)
    assert all(row["clean_tie"] is False for row in primary)
    assert all(
        row["risk_inflation_vs_target_p_value_one_sided"] is not None
        for row in primary
    )
    assert all(row.get("p_value_holm") is not None for row in primary)


def test_threshold_transfer_reuses_clean_bootstrap_per_analysis_cell() -> None:
    clean = [
        _prediction(
            f"cal-{index}",
            split="calibration",
            family="clean",
            direction="clean",
            dose=None,
            variant_id="clean",
        )
        for index in range(4)
    ]
    biased = [
        _prediction(
            f"biased-{dose}-{index}",
            split="test",
            family="authority",
            direction="incongruent",
            dose=dose,
            variant_id=f"authority_incongruent_{dose}_ab",
        )
        for dose in (1, 4)
        for index in range(2)
    ]

    with patch(
        "scripts.analyze_silent_bias.bootstrap_threshold_rules",
        wraps=bootstrap_threshold_rules,
    ) as bootstrap_builder:
        threshold_transfer_outputs(
            clean,
            biased,
            target_risks=(0.10,),
            aggregation="single_ordering",
            n_resamples=10,
            seed=42,
        )

    assert bootstrap_builder.call_count == 3


def _flat_row(
    record_id: str,
    pair_key: str,
    *,
    family: str,
    dose: float | None,
    verdict: str,
    entropy: float = 0.3,
) -> dict[str, object]:
    direction = "clean" if dose is None else "incongruent"
    return {
        "record_id": record_id,
        "example_id": pair_key,
        "question_id": pair_key,
        "pair_key": pair_key,
        "ordering": "ab",
        "model_name": "judge",
        "routing_split": "test",
        "bias_name": family,
        "direction": direction,
        "dose": dose,
        "variant_id": (
            "clean"
            if dose is None
            else f"{family}_incongruent_{int(dose)}_ab"
        ),
        "cue_target": None if dose is None else "B",
        "human_winner": "A",
        "verdict": verdict,
        "label_prob_A": 0.8 if verdict == "A" else 0.1,
        "label_prob_B": 0.1 if verdict == "A" else 0.8,
        "label_prob_tie": 0.1,
        "entropy": entropy,
        "msp": 0.8,
        "margin": 0.7,
    }


def _rq3_shifts() -> tuple[PairedShift, ...]:
    clean = []
    cued = []
    for family, doses in (
        ("authority", (1.0, 2.0, 3.0, 4.0)),
        ("bandwagon", (55.0, 70.0, 85.0, 95.0)),
    ):
        for index in range(6):
            pair_key = f"{family}-q{index}"
            clean.append(
                record_from_mapping(
                    _flat_row(
                        f"clean-{pair_key}",
                        pair_key,
                        family="clean",
                        dose=None,
                        verdict="A",
                    )
                )
            )
            for dose_index, dose in enumerate(doses):
                cued.append(
                    record_from_mapping(
                        _flat_row(
                            f"cued-{pair_key}-{dose_index}",
                            pair_key,
                            family=family,
                            dose=dose,
                            verdict=(
                                "B"
                                if dose == doses[-1] and index < 3
                                else "A"
                            ),
                            entropy=0.35 + dose_index * 0.05,
                        )
                    )
                )
    return compute_paired_shifts(pair_clean_and_cued(clean, cued).pairs)


def _calibration_copies(
    shifts: Sequence[PairedShift],
) -> tuple[PairedShift, ...]:
    return tuple(
        replace(
            shift,
            clean_record_id=f"cal-{shift.clean_record_id}",
            cued_record_id=f"cal-{shift.cued_record_id}",
            example_id=f"cal-{shift.example_id}",
            question_id=f"cal-{shift.question_id}",
            pair_key=f"cal-{shift.pair_key}",
            condition_group_id=(
                None
                if shift.condition_group_id is None
                else f"cal-{shift.condition_group_id}"
            ),
            routing_split="calibration",
            signed_cue_mass=(
                None
                if shift.signed_cue_mass is None
                else -shift.signed_cue_mass
            ),
        )
        for shift in shifts
    )


def test_rq1_and_rq3_headlines_use_only_the_requested_test_split() -> None:
    test_shifts = _rq3_shifts()
    mixed_shifts = (*test_shifts, *_calibration_copies(test_shifts))
    test_tie = replace(
        test_shifts[0],
        clean_record_id=f"tie-{test_shifts[0].clean_record_id}",
        cued_record_id=f"tie-{test_shifts[0].cued_record_id}",
        example_id=f"tie-{test_shifts[0].example_id}",
        question_id=f"tie-{test_shifts[0].question_id}",
        pair_key=f"tie-{test_shifts[0].pair_key}",
        condition_group_id="tie-group",
        clean_tie=True,
    )

    assert summarize_silent_shift(
        mixed_shifts,
        routing_split="test",
        n_resamples=20,
        seed=42,
    ) == summarize_silent_shift(
        test_shifts,
        routing_split="test",
        n_resamples=20,
        seed=42,
    )
    assert summarize_susceptibility(
        mixed_shifts,
        routing_split="test",
        n_resamples=20,
        seed=42,
    ) == summarize_susceptibility(
        test_shifts,
        routing_split="test",
        n_resamples=20,
        seed=42,
    )
    assert dose_response_outputs(
        mixed_shifts,
        routing_split="test",
        n_resamples=20,
        seed=42,
    ) == dose_response_outputs(
        test_shifts,
        routing_split="test",
        n_resamples=20,
        seed=42,
    )
    assert uncertainty_by_dose_outputs(
        mixed_shifts,
        routing_split="test",
        n_resamples=20,
        seed=42,
    ) == uncertainty_by_dose_outputs(
        test_shifts,
        routing_split="test",
        n_resamples=20,
        seed=42,
    )
    with patch(
        "scripts.analyze_silent_bias.fit_uncertainty_gee",
        side_effect=OptionalAnalysisDependencyError(
            "statsmodels unavailable in fixture"
        ),
    ):
        assert uncertainty_trend_outputs(
            mixed_shifts,
            routing_split="test",
            n_permutations=20,
            seed=42,
        ) == uncertainty_trend_outputs(
            test_shifts,
            routing_split="test",
            n_permutations=20,
            seed=42,
        )

    captured: list[dict[str, object]] = []

    def unavailable(rows):
        captured.extend(rows)
        raise OptionalAnalysisDependencyError("statsmodels unavailable in fixture")

    with patch(
        "scripts.analyze_silent_bias.fit_flip_mixed_logit",
        side_effect=unavailable,
    ):
        output = modeling_outputs(
            (*mixed_shifts, test_tie),
            routing_split="test",
        )
    assert output[0]["routing_split"] == "test"
    assert output[0]["clean_tie"] is False
    assert len(captured) == len(test_shifts)
    assert all(row["question_id"] != test_tie.question_id for row in captured)
    assert {row["question_id"] for row in captured}.isdisjoint(
        {
            shift.question_id
            for shift in _calibration_copies(test_shifts)
        }
    )


def test_headline_split_selection_rejects_unknown_input_splits() -> None:
    shifts = _rq3_shifts()
    invalid = (replace(shifts[0], routing_split=None), *shifts[1:])
    with pytest.raises(ValueError, match="invalid values"):
        dose_response_outputs(
            invalid,
            routing_split="test",
            n_resamples=10,
            seed=42,
        )


def test_rq3_primary_slope_tail_p_values_receive_holm_correction() -> None:
    shifts = _rq3_shifts()
    rows = dose_response_outputs(
        shifts,
        routing_split="test",
        n_resamples=50,
        seed=42,
    )
    primary = [row for row in rows if row["primary"]]
    assert len(primary) == 2
    assert all(row["slope_p_value_one_sided"] is not None for row in primary)
    assert all(row.get("p_value_holm") is not None for row in primary)


def test_canonical_social_dose_ladders_share_normalized_four_levels() -> None:
    expected = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)
    authority = tuple(
        normalized_social_dose("authority", dose)
        for dose in (1.0, 2.0, 3.0, 4.0)
    )
    bandwagon = tuple(
        normalized_social_dose("bandwagon", dose)
        for dose in (55.0, 70.0, 85.0, 95.0)
    )
    assert authority == expected
    assert bandwagon == expected


def test_mixed_model_uses_normalized_dose_and_reports_scale_metadata() -> None:
    captured: list[dict[str, object]] = []

    def unavailable(rows):
        captured.extend(rows)
        raise OptionalAnalysisDependencyError("statsmodels unavailable in fixture")

    with patch(
        "scripts.analyze_silent_bias.fit_flip_mixed_logit",
        side_effect=unavailable,
    ):
        output = modeling_outputs(
            _rq3_shifts(),
            routing_split="test",
        )

    assert len(output) == 1
    assert output[0]["formula"] == MIXED_EFFECTS_FORMULA
    assert output[0]["dose_scale"] == "canonical_four_level_normalized_0_1"
    assert output[0]["normalized_dose_levels"] == (
        0.0,
        1.0 / 3.0,
        2.0 / 3.0,
        1.0,
    )
    assert output[0]["raw_dose_ladders"] == {
        "authority": (1.0, 2.0, 3.0, 4.0),
        "bandwagon": (55.0, 70.0, 85.0, 95.0),
    }
    for family in ("authority", "bandwagon"):
        assert {
            row["dose"] for row in captured if row["family"] == family
        } == {0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0}


def test_uncertainty_trends_emit_gee_rows_and_permutation_sensitivity() -> None:
    with patch(
        "scripts.analyze_silent_bias.fit_uncertainty_gee",
        side_effect=OptionalAnalysisDependencyError(
            "statsmodels unavailable in fixture"
        ),
    ):
        rows = uncertainty_trend_outputs(
            _rq3_shifts(),
            routing_split="test",
            n_permutations=20,
            seed=42,
        )

    gee_rows = [
        row for row in rows if row["estimator"] == "gaussian_gee_exchangeable"
    ]
    sensitivity_rows = [
        row for row in rows if row["estimator"] == "cluster_permutation_slope"
    ]
    assert gee_rows
    assert {
        row["stable_set"] for row in gee_rows
    } == {"pre_first_flip", "non_flipped_at_current_dose"}
    assert all(row["formula"] == UNCERTAINTY_GEE_FORMULA for row in gee_rows)
    assert all(row["status"] == "unavailable" for row in gee_rows)
    assert all(
        row["dose_scale"] == "canonical_four_level_normalized_0_1"
        for row in gee_rows
    )
    assert sensitivity_rows
    assert all(row["status"] == "ok" for row in sensitivity_rows)
    assert all(row["sensitivity_analysis"] for row in sensitivity_rows)
    assert len([row for row in gee_rows if row["primary"]]) == 2


def test_primary_uncertainty_trends_carry_cluster_bootstrap_intervals() -> None:
    def fake_gee(rows):
        if not rows:
            raise ValueError("no rows")
        x = [float(row["normalized_dose"]) for row in rows]
        y = [float(row["uncertainty"]) for row in rows]
        mean_x = sum(x) / len(x)
        mean_y = sum(y) / len(y)
        denominator = sum((value - mean_x) ** 2 for value in x)
        slope = 0.0 if denominator == 0.0 else (
            sum(
                (x_value - mean_x) * (y_value - mean_y)
                for x_value, y_value in zip(x, y, strict=True)
            )
            / denominator
        )
        return UncertaintyGEEResult(
            formula=UNCERTAINTY_GEE_FORMULA,
            group_column="question_id",
            n=len(rows),
            n_clusters=len({row["question_id"] for row in rows}),
            intercept=mean_y - slope * mean_x,
            slope=slope,
            slope_standard_error=0.01,
            slope_z_value=slope / 0.01,
            slope_p_value_one_sided=0.01,
            converged=True,
        )

    with (
        patch(
            "scripts.analyze_silent_bias.fit_uncertainty_gee",
            side_effect=fake_gee,
        ),
        patch(
            "biases.analysis.modeling.fit_uncertainty_gee",
            side_effect=fake_gee,
        ),
    ):
        rows = uncertainty_trend_outputs(
            _rq3_shifts(),
            routing_split="test",
            n_permutations=20,
            n_resamples=20,
            seed=42,
        )

    primary = [row for row in rows if row["primary"]]
    assert len(primary) == 2
    assert all(row["slope_ci_low"] is not None for row in primary)
    assert all(row["slope_ci_high"] is not None for row in primary)
    assert all(row["bootstrap_resamples_successful"] == 20 for row in primary)


def test_primary_uncertainty_by_dose_rows_have_clustered_intervals() -> None:
    rows = uncertainty_by_dose_outputs(
        _rq3_shifts(),
        routing_split="test",
        n_resamples=20,
        seed=42,
    )
    assert rows
    assert {row["family"] for row in rows} == {"authority", "bandwagon"}
    assert all(row["metric"] == "cued_entropy" for row in rows)
    assert all(row["stable_set"] == "pre_first_flip" for row in rows)
    assert all(row["ci_low"] <= row["ci_high"] for row in rows)
