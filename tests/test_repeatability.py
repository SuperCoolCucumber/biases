from __future__ import annotations

import math

import pytest

from biases.analysis.repeatability import (
    anchor_reproducibility,
    degree_matrix_agreement,
    frequency_semantic_entropy_confidence,
    independent_draw_agreement,
    jensen_shannon_similarity,
    ordering_repeatability_scores,
    total_variation_similarity,
    vote_distribution,
)


def test_anchor_reproducibility_is_tie_aware_anchor_agreement() -> None:
    counts = {"A": 2, "B": 1, "tie": 1}

    assert anchor_reproducibility("A", counts) == pytest.approx(0.5)
    assert anchor_reproducibility("B", counts) == pytest.approx(0.25)
    assert anchor_reproducibility("tie", counts) == pytest.approx(0.25)


def test_anchor_reproducibility_does_not_follow_the_repeat_majority() -> None:
    counts = {"A": 1, "B": 3, "tie": 0}

    assert anchor_reproducibility("A", counts) == pytest.approx(0.25)


def test_lm_polygraph_adaptations_are_one_for_unanimous_repeats() -> None:
    counts = {"A": 4, "B": 0, "tie": 0}

    assert frequency_semantic_entropy_confidence(counts) == pytest.approx(1.0)
    assert degree_matrix_agreement(counts) == pytest.approx(1.0)


def test_lm_polygraph_adaptations_match_exact_three_class_formulas() -> None:
    counts = {"A": 2, "B": 1, "tie": 1}

    expected_entropy_confidence = 1.0 - 1.5 / math.log2(3.0)
    assert frequency_semantic_entropy_confidence(counts) == pytest.approx(
        expected_entropy_confidence
    )
    assert degree_matrix_agreement(counts) == pytest.approx(0.375)


def test_lm_polygraph_adaptations_have_same_k4_partition_ranking() -> None:
    partitions = [
        {"A": 4, "B": 0, "tie": 0},
        {"A": 3, "B": 1, "tie": 0},
        {"A": 2, "B": 2, "tie": 0},
        {"A": 2, "B": 1, "tie": 1},
    ]

    entropy_scores = [
        frequency_semantic_entropy_confidence(counts) for counts in partitions
    ]
    degree_scores = [degree_matrix_agreement(counts) for counts in partitions]

    assert entropy_scores == sorted(entropy_scores, reverse=True)
    assert degree_scores == sorted(degree_scores, reverse=True)
    assert len(set(entropy_scores)) == len(partitions)
    assert len(set(degree_scores)) == len(partitions)


@pytest.mark.parametrize(
    ("counts", "entropy_confidence", "degree_agreement"),
    [
        ({"A": 4, "B": 0, "tie": 0}, 1.0, 1.0),
        ({"A": 3, "B": 1, "tie": 0}, 0.4881404928570852, 0.625),
        ({"A": 2, "B": 2, "tie": 0}, 0.3690702464285425, 0.5),
        ({"A": 2, "B": 1, "tie": 1}, 0.0536053696428138, 0.375),
    ],
)
def test_lm_polygraph_k4_score_table(
    counts: dict[str, int],
    entropy_confidence: float,
    degree_agreement: float,
) -> None:
    assert frequency_semantic_entropy_confidence(counts) == pytest.approx(
        entropy_confidence
    )
    assert degree_matrix_agreement(counts) == pytest.approx(degree_agreement)


def test_lm_polygraph_adaptations_are_label_permutation_invariant() -> None:
    first = {"A": 3, "B": 1, "tie": 0}
    permuted = {"A": 0, "B": 3, "tie": 1}

    assert frequency_semantic_entropy_confidence(first) == pytest.approx(
        frequency_semantic_entropy_confidence(permuted)
    )
    assert degree_matrix_agreement(first) == pytest.approx(
        degree_matrix_agreement(permuted)
    )


def test_ba_vote_distribution_is_canonicalized_without_changing_ties() -> None:
    assert vote_distribution(
        {"A": 1, "B": 2, "tie": 1}, ordering="ba"
    ) == pytest.approx((0.5, 0.25, 0.25))


def test_ordering_scores_equal_one_for_canonical_matches() -> None:
    scores = ordering_repeatability_scores(
        {"A": 3, "B": 0, "tie": 1},
        {"A": 0, "B": 3, "tie": 1},
    )

    assert scores.js_similarity == pytest.approx(1.0)
    assert scores.total_variation_similarity == pytest.approx(1.0)
    assert scores.independent_draw_agreement == pytest.approx(0.625)


def test_distribution_scores_cover_distinct_disagreement_notions() -> None:
    left = (1.0, 0.0, 0.0)
    right = (0.0, 1.0, 0.0)

    assert jensen_shannon_similarity(left, right) == pytest.approx(0.0)
    assert total_variation_similarity(left, right) == pytest.approx(0.0)
    assert independent_draw_agreement(left, right) == pytest.approx(0.0)

    diffuse = (0.5, 0.5, 0.0)
    assert jensen_shannon_similarity(diffuse, diffuse) == pytest.approx(1.0)
    assert total_variation_similarity(diffuse, diffuse) == pytest.approx(1.0)
    assert independent_draw_agreement(diffuse, diffuse) == pytest.approx(0.5)


@pytest.mark.parametrize(
    "counts",
    [
        {},
        {"A": 0, "B": 0, "tie": 0},
        {"A": -1},
        {"A": 1.5},
        {"unknown": 1},
    ],
)
def test_invalid_vote_counts_are_rejected(counts: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        vote_distribution(counts)  # type: ignore[arg-type]


def test_js_similarity_is_finite_with_zero_probability_entries() -> None:
    score = jensen_shannon_similarity((0.0, 1.0, 0.0), (0.0, 0.5, 0.5))
    assert math.isfinite(score)
    assert 0.0 <= score <= 1.0


def test_distribution_scores_normalize_positive_mass() -> None:
    assert jensen_shannon_similarity((2.0, 2.0, 0.0), (1.0, 1.0, 0.0)) \
        == pytest.approx(1.0)
    assert total_variation_similarity((2.0, 0.0, 0.0), (1.0, 0.0, 0.0)) \
        == pytest.approx(1.0)
    assert independent_draw_agreement((2.0, 0.0, 0.0), (3.0, 0.0, 0.0)) \
        == pytest.approx(1.0)


@pytest.mark.parametrize(
    "values",
    [
        (),
        (1.0, 0.0),
        (0.0, 0.0, 0.0),
        (-1.0, 1.0, 1.0),
        (math.nan, 1.0, 0.0),
    ],
)
def test_invalid_vote_distributions_are_rejected(
    values: tuple[float, ...],
) -> None:
    with pytest.raises(ValueError):
        jensen_shannon_similarity(values, (1.0, 0.0, 0.0))
