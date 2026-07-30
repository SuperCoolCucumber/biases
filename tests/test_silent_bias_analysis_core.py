from __future__ import annotations

import math
import random

import pytest

from biases.analysis.records import pair_clean_and_cued, record_from_mapping
from biases.analysis.resampling import (
    cluster_percentile_interval,
    cluster_sign_flip_p_value,
)
from biases.analysis.rq1 import (
    compute_paired_shifts,
    jensen_shannon_divergence,
    low_dose_susceptibility_auc,
    roc_auc,
)


def _row(
    record_id: str,
    pair_key: str,
    *,
    verdict: str = "A",
    probs: tuple[float, float, float] = (0.8, 0.1, 0.1),
    entropy: float = 0.5,
    dose: float | None = None,
    family: str = "clean",
    direction: str = "clean",
    cue_target: str | None = None,
) -> dict[str, object]:
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
            "clean" if dose is None else f"{family}_{direction}_{int(dose)}_ab"
        ),
        "cue_target": cue_target,
        "human_winner": "A",
        "verdict": verdict,
        "label_prob_A": probs[0],
        "label_prob_B": probs[1],
        "label_prob_tie": probs[2],
        "entropy": entropy,
        "normalized_entropy": entropy / math.log2(3),
        "msp": max(probs),
        "margin": sorted(probs, reverse=True)[0] - sorted(probs, reverse=True)[1],
        "verbalized_confidence": 0.75,
        "consistency_vote_entropy": 0.2,
    }


def test_pairing_and_paired_shift_include_mass_js_flip_and_error() -> None:
    clean = record_from_mapping(_row("clean", "q1"))
    cued = record_from_mapping(
        _row(
            "cued",
            "q1",
            verdict="B",
            probs=(0.25, 0.65, 0.10),
            entropy=0.9,
            dose=55,
            family="bandwagon",
            direction="incongruent",
            cue_target="B",
        )
    )
    pairing = pair_clean_and_cued([clean], [cued])
    shift = compute_paired_shifts(pairing.pairs)[0]

    assert pairing.unmatched_cued_record_ids == ()
    assert shift.delta_entropy == pytest.approx(0.4)
    assert shift.signed_cue_mass == pytest.approx(0.55)
    assert shift.js_divergence is not None and 0.0 < shift.js_divergence < 1.0
    assert shift.flip is True
    assert shift.error is True


def test_analysis_records_preserve_secondary_channel_verdicts() -> None:
    flat = record_from_mapping(
        {
            **_row("flat", "q-flat"),
            "verbalized_verdict": "B",
            "consistency_majority_verdict": "tie",
        }
    )
    nested = record_from_mapping(
        {
            **_row("nested", "q-nested"),
            "metadata": {"verbalized_verdict": "tie"},
            "uncertainty": {
                "consistency": {"majority_verdict": "B"},
            },
        }
    )

    assert flat.verbalized_verdict == "B"
    assert flat.consistency_majority_verdict == "tie"
    assert nested.verbalized_verdict == "tie"
    assert nested.consistency_majority_verdict == "B"


def test_jensen_shannon_is_symmetric_and_zero_for_equal_distributions() -> None:
    first = (0.7, 0.2, 0.1)
    second = (0.2, 0.7, 0.1)
    assert jensen_shannon_divergence(first, first) == pytest.approx(0.0)
    assert jensen_shannon_divergence(first, second) == pytest.approx(
        jensen_shannon_divergence(second, first)
    )


def test_roc_auc_preserves_half_credit_for_ties() -> None:
    assert roc_auc([1, 1, 0, 0], [0.9, 0.5, 0.5, 0.1]) == 0.875


def test_roc_auc_all_equal_scores_are_chance() -> None:
    assert roc_auc([1, 0, 1, 0], [0.25, 0.25, 0.25, 0.25]) == 0.5


def test_roc_auc_preserves_nan_and_infinite_pairwise_semantics() -> None:
    labels = [1, 1, 1, 0, 0, 0]
    scores = [math.nan, math.inf, -math.inf, math.nan, math.inf, -math.inf]

    assert roc_auc(labels, scores) == 2.0 / 9.0


@pytest.mark.parametrize("labels", ([1, 1], [0, 0], []))
def test_roc_auc_returns_none_when_a_class_is_missing(labels: list[int]) -> None:
    assert roc_auc(labels, [0.5] * len(labels)) is None


def test_roc_auc_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        roc_auc([1, 0], [0.5])


def test_roc_auc_matches_pairwise_reference_on_random_tied_scores() -> None:
    rng = random.Random(20260730)
    labels = [rng.randrange(2) for _ in range(400)]
    scores = [rng.randrange(25) / 10.0 for _ in labels]
    positives = [score for label, score in zip(labels, scores, strict=True) if label]
    negatives = [score for label, score in zip(labels, scores, strict=True) if not label]
    wins = sum(
        float(positive > negative) + 0.5 * float(positive == negative)
        for positive in positives
        for negative in negatives
    )
    expected = wins / (len(positives) * len(negatives))

    assert roc_auc(labels, scores) == expected


def test_question_cluster_percentile_bootstrap_is_deterministic() -> None:
    rows = (("q1", 0.0), ("q1", 2.0), ("q2", 4.0), ("q3", 8.0))
    kwargs = {
        "cluster_key": lambda row: row[0],
        "statistic": lambda sample: sum(row[1] for row in sample) / len(sample),
        "n_resamples": 100,
        "seed": 19,
    }
    first = cluster_percentile_interval(rows, **kwargs)
    second = cluster_percentile_interval(rows, **kwargs)
    assert first == second
    assert first.estimate == pytest.approx(3.5)
    assert first.n_clusters == 3


def test_cluster_sign_flip_test_is_one_sided_and_question_clustered() -> None:
    rows = tuple((f"q{index}", 0.2 + index / 100) for index in range(8))
    p_value = cluster_sign_flip_p_value(
        rows,
        cluster_key=lambda row: row[0],
        value=lambda row: row[1],
        n_permutations=500,
        seed=42,
    )
    assert p_value < 0.05


def test_low_dose_shift_auc_is_compared_with_clean_uncertainty() -> None:
    clean_records = []
    cued_records = []
    for index, (low_mass, clean_entropy, high_flip) in enumerate(
        ((0.01, 0.9, False), (0.35, 0.2, True), (0.70, 0.1, True)),
        start=1,
    ):
        pair_key = f"q{index}"
        clean_records.append(
            record_from_mapping(
                _row(f"clean-{index}", pair_key, entropy=clean_entropy)
            )
        )
        cued_records.extend(
            [
                record_from_mapping(
                    _row(
                        f"low-{index}",
                        pair_key,
                        probs=(0.8 - low_mass, 0.1 + low_mass, 0.1),
                        dose=1,
                        family="authority",
                        direction="incongruent",
                        cue_target="B",
                    )
                ),
                record_from_mapping(
                    _row(
                        f"high-{index}",
                        pair_key,
                        verdict="B" if high_flip else "A",
                        probs=(0.1, 0.8, 0.1) if high_flip else (0.8, 0.1, 0.1),
                        dose=4,
                        family="authority",
                        direction="incongruent",
                        cue_target="B",
                    )
                ),
            ]
        )
    shifts = compute_paired_shifts(
        pair_clean_and_cued(clean_records, cued_records).pairs
    )
    result = low_dose_susceptibility_auc(shifts)
    assert result.n == 3
    assert result.positives == 2
    assert result.shift_auc == pytest.approx(1.0)
    assert result.clean_baseline_auc == pytest.approx(0.0)
    assert result.auc_difference == pytest.approx(1.0)
