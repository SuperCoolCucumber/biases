from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from biases.analysis.records import LABELS, normalize_label


VoteDistribution = tuple[float, float, float]


def _validated_counts(verdict_counts: Mapping[str, int]) -> tuple[int, int, int]:
    counts = {label: 0 for label in LABELS}
    for raw_label, raw_count in verdict_counts.items():
        label = normalize_label(raw_label)
        if label is None:
            raise ValueError(f"unknown verdict label: {raw_label!r}")
        if isinstance(raw_count, bool) or not isinstance(raw_count, int):
            raise ValueError("verdict counts must be integers")
        if raw_count < 0:
            raise ValueError("verdict counts must be non-negative")
        counts[label] += raw_count
    result = tuple(counts[label] for label in LABELS)
    if sum(result) < 1:
        raise ValueError("verdict counts must contain at least one vote")
    return result  # type: ignore[return-value]


def vote_distribution(
    verdict_counts: Mapping[str, int],
    *,
    ordering: str = "ab",
) -> VoteDistribution:
    """Return a tie-aware A/B/tie vote distribution in canonical AB order."""

    counts = _validated_counts(verdict_counts)
    normalized_ordering = ordering.strip().lower()
    if normalized_ordering not in {"ab", "ba"}:
        raise ValueError("ordering must be 'ab' or 'ba'")
    if normalized_ordering == "ba":
        counts = (counts[1], counts[0], counts[2])
    total = sum(counts)
    return tuple(count / total for count in counts)  # type: ignore[return-value]


def anchor_reproducibility(
    anchor: str,
    verdict_counts: Mapping[str, int],
) -> float:
    """Fraction of sampled verdicts equal to the same-order deterministic anchor."""

    normalized_anchor = normalize_label(anchor)
    if normalized_anchor is None:
        raise ValueError(f"unknown deterministic anchor: {anchor!r}")
    counts = _validated_counts(verdict_counts)
    return counts[LABELS.index(normalized_anchor)] / sum(counts)


def frequency_semantic_entropy_confidence(
    verdict_counts: Mapping[str, int],
) -> float:
    """Return normalized confidence from entropy of exact A/B/tie classes."""

    distribution = vote_distribution(verdict_counts)
    entropy = -sum(
        probability * math.log(probability)
        for probability in distribution
        if probability > 0.0
    )
    return min(1.0, max(0.0, 1.0 - entropy / math.log(len(LABELS))))


def degree_matrix_agreement(verdict_counts: Mapping[str, int]) -> float:
    """Return the agreement complement of exact-label Degree-Matrix uncertainty."""

    distribution = vote_distribution(verdict_counts)
    return sum(probability**2 for probability in distribution)


def _validated_distribution(values: Sequence[float]) -> VoteDistribution:
    if len(values) != len(LABELS):
        raise ValueError("vote distributions must contain A, B, and tie")
    numeric = tuple(float(value) for value in values)
    if any(not math.isfinite(value) or value < 0.0 for value in numeric):
        raise ValueError("vote-distribution values must be finite and non-negative")
    total = sum(numeric)
    if total <= 0.0:
        raise ValueError("vote distributions must have positive total mass")
    return tuple(value / total for value in numeric)  # type: ignore[return-value]


def jensen_shannon_similarity(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    """Return one minus base-2-normalized Jensen--Shannon divergence."""

    left_values = _validated_distribution(left)
    right_values = _validated_distribution(right)

    def kl(first: VoteDistribution, second: VoteDistribution) -> float:
        return sum(
            a * math.log(a / b)
            for a, b in zip(first, second, strict=True)
            if a > 0.0
        )

    midpoint = tuple(
        (a + b) / 2.0
        for a, b in zip(left_values, right_values, strict=True)
    )
    divergence = 0.5 * kl(left_values, midpoint) + 0.5 * kl(
        right_values,
        midpoint,
    )
    return min(1.0, max(0.0, 1.0 - divergence / math.log(2.0)))


def total_variation_similarity(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    """Return one minus total-variation distance between vote distributions."""

    left_values = _validated_distribution(left)
    right_values = _validated_distribution(right)
    distance = 0.5 * sum(
        abs(a - b)
        for a, b in zip(left_values, right_values, strict=True)
    )
    return min(1.0, max(0.0, 1.0 - distance))


def independent_draw_agreement(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    """Probability that independent AB and canonicalized BA votes agree."""

    left_values = _validated_distribution(left)
    right_values = _validated_distribution(right)
    return min(
        1.0,
        max(
            0.0,
            sum(
                a * b
                for a, b in zip(left_values, right_values, strict=True)
            ),
        ),
    )


@dataclass(frozen=True, slots=True)
class OrderingRepeatabilityScores:
    js_similarity: float
    total_variation_similarity: float
    independent_draw_agreement: float


def ordering_repeatability_scores(
    ab_verdict_counts: Mapping[str, int],
    ba_verdict_counts: Mapping[str, int],
) -> OrderingRepeatabilityScores:
    """Compare AB and BA repeat distributions after canonical label remapping."""

    ab = vote_distribution(ab_verdict_counts, ordering="ab")
    ba = vote_distribution(ba_verdict_counts, ordering="ba")
    return OrderingRepeatabilityScores(
        js_similarity=jensen_shannon_similarity(ab, ba),
        total_variation_similarity=total_variation_similarity(ab, ba),
        independent_draw_agreement=independent_draw_agreement(ab, ba),
    )


__all__ = [
    "OrderingRepeatabilityScores",
    "VoteDistribution",
    "anchor_reproducibility",
    "degree_matrix_agreement",
    "frequency_semantic_entropy_confidence",
    "independent_draw_agreement",
    "jensen_shannon_similarity",
    "ordering_repeatability_scores",
    "total_variation_similarity",
    "vote_distribution",
]
