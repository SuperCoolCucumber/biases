from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import NormalDist


NORMAL = NormalDist()


@dataclass(frozen=True)
class ConfidenceInterval:
    estimate: float
    low: float
    high: float
    confidence: float


@dataclass(frozen=True)
class McNemarResult:
    b: int
    c: int
    statistic: int
    p_value: float


@dataclass(frozen=True)
class MannWhitneyResult:
    u: float
    p_value: float
    rank_biserial: float
    n_x: int
    n_y: int


@dataclass(frozen=True)
class DeLongResult:
    auc_1: float
    auc_2: float
    z: float
    p_value: float
    variance: float


def _binomial_pmf(k: int, n: int, p: float) -> float:
    return math.comb(n, k) * (p**k) * ((1.0 - p) ** (n - k))


def mcnemar_exact(b: int, c: int) -> McNemarResult:
    """Exact two-sided McNemar test for paired binary disagreements.

    b and c are the discordant cells: method 1 correct / method 2 wrong and
    method 1 wrong / method 2 correct, respectively.
    """

    if b < 0 or c < 0:
        raise ValueError("b and c must be non-negative")

    n = b + c
    statistic = min(b, c)
    if n == 0:
        return McNemarResult(b=b, c=c, statistic=statistic, p_value=1.0)

    tail = sum(_binomial_pmf(k, n, 0.5) for k in range(statistic + 1))
    return McNemarResult(b=b, c=c, statistic=statistic, p_value=min(1.0, 2.0 * tail))


def wilson_ci(
    successes: int,
    total: int,
    *,
    confidence: float = 0.95,
) -> ConfidenceInterval:
    if total < 0 or successes < 0 or successes > total:
        raise ValueError("successes must be between 0 and total")
    if total == 0:
        return ConfidenceInterval(estimate=math.nan, low=math.nan, high=math.nan, confidence=confidence)

    alpha = 1.0 - confidence
    z = NORMAL.inv_cdf(1.0 - alpha / 2.0)
    phat = successes / total
    denom = 1.0 + z * z / total
    center = (phat + z * z / (2.0 * total)) / denom
    half_width = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * total)) / total) / denom
    return ConfidenceInterval(
        estimate=phat,
        low=max(0.0, center - half_width),
        high=min(1.0, center + half_width),
        confidence=confidence,
    )


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot compute statistic on empty data")
    return sum(values) / len(values)


def bootstrap_bca_ci(
    data: Sequence[float],
    statistic: Callable[[Sequence[float]], float] = _mean,
    *,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
) -> ConfidenceInterval:
    """Bias-corrected and accelerated bootstrap CI for one-sample statistics."""

    values = [float(value) for value in data]
    if not values:
        raise ValueError("data must not be empty")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")

    estimate = float(statistic(values))
    rng = random.Random(seed)
    n = len(values)
    boot = [
        float(statistic([values[rng.randrange(n)] for _ in range(n)]))
        for _ in range(n_resamples)
    ]
    boot.sort()

    if all(sample == estimate for sample in boot):
        return ConfidenceInterval(estimate=estimate, low=estimate, high=estimate, confidence=confidence)

    prop_less = sum(sample < estimate for sample in boot) / n_resamples
    prop_less = min(max(prop_less, 1.0 / (2.0 * n_resamples)), 1.0 - 1.0 / (2.0 * n_resamples))
    z0 = NORMAL.inv_cdf(prop_less)

    jack = []
    if n > 1:
        for i in range(n):
            jack.append(float(statistic(values[:i] + values[i + 1 :])))
    jack_mean = _mean(jack) if jack else estimate
    numerator = sum((jack_mean - sample) ** 3 for sample in jack)
    denominator = 6.0 * (sum((jack_mean - sample) ** 2 for sample in jack) ** 1.5)
    acceleration = numerator / denominator if denominator else 0.0

    alpha = 1.0 - confidence

    def adjusted_quantile(raw_alpha: float) -> float:
        z_alpha = NORMAL.inv_cdf(raw_alpha)
        denom = 1.0 - acceleration * (z0 + z_alpha)
        if denom == 0:
            return raw_alpha
        return NORMAL.cdf(z0 + (z0 + z_alpha) / denom)

    low_q = min(max(adjusted_quantile(alpha / 2.0), 0.0), 1.0)
    high_q = min(max(adjusted_quantile(1.0 - alpha / 2.0), 0.0), 1.0)

    return ConfidenceInterval(
        estimate=estimate,
        low=_quantile_sorted(boot, low_q),
        high=_quantile_sorted(boot, high_q),
        confidence=confidence,
    )


def _quantile_sorted(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    if q <= 0:
        return float(values[0])
    if q >= 1:
        return float(values[-1])
    position = q * (len(values) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(values[low])
    weight = position - low
    return float(values[low] * (1.0 - weight) + values[high] * weight)


def _average_ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = rank
        i = j
    return ranks


def mann_whitney_u(
    x: Sequence[float],
    y: Sequence[float],
) -> MannWhitneyResult:
    """Two-sided Mann-Whitney U test with normal approximation and tie correction."""

    x_values = [float(value) for value in x]
    y_values = [float(value) for value in y]
    n_x = len(x_values)
    n_y = len(y_values)
    if n_x == 0 or n_y == 0:
        raise ValueError("both samples must be non-empty")

    combined = x_values + y_values
    ranks = _average_ranks(combined)
    rank_sum_x = sum(ranks[:n_x])
    u_x = rank_sum_x - n_x * (n_x + 1) / 2.0
    mean_u = n_x * n_y / 2.0

    counts: dict[float, int] = {}
    for value in combined:
        counts[value] = counts.get(value, 0) + 1
    n = n_x + n_y
    tie_term = sum(count**3 - count for count in counts.values())
    variance = n_x * n_y / 12.0 * ((n + 1) - tie_term / (n * (n - 1))) if n > 1 else 0.0
    if variance <= 0:
        p_value = 1.0
    else:
        correction = 0.5 if u_x > mean_u else -0.5 if u_x < mean_u else 0.0
        z = (u_x - mean_u - correction) / math.sqrt(variance)
        p_value = 2.0 * (1.0 - NORMAL.cdf(abs(z)))

    rank_biserial = 2.0 * u_x / (n_x * n_y) - 1.0
    return MannWhitneyResult(
        u=u_x,
        p_value=max(0.0, min(1.0, p_value)),
        rank_biserial=rank_biserial,
        n_x=n_x,
        n_y=n_y,
    )


def roc_auc(labels: Sequence[bool | int], scores: Sequence[float]) -> float:
    y_true = [bool(label) for label in labels]
    y_score = [float(score) for score in scores]
    if len(y_true) != len(y_score):
        raise ValueError("labels and scores must have the same length")
    positives = [score for label, score in zip(y_true, y_score, strict=True) if label]
    negatives = [score for label, score in zip(y_true, y_score, strict=True) if not label]
    if not positives or not negatives:
        raise ValueError("AUC requires at least one positive and one negative")

    ranks = _average_ranks(positives + negatives)
    rank_sum_pos = sum(ranks[: len(positives)])
    return (rank_sum_pos - len(positives) * (len(positives) + 1) / 2.0) / (
        len(positives) * len(negatives)
    )


def delong_test(
    labels: Sequence[bool | int],
    scores_1: Sequence[float],
    scores_2: Sequence[float],
) -> DeLongResult:
    """Paired DeLong test for two correlated ROC AUCs."""

    y_true = [bool(label) for label in labels]
    s1 = [float(score) for score in scores_1]
    s2 = [float(score) for score in scores_2]
    if not (len(y_true) == len(s1) == len(s2)):
        raise ValueError("labels and score arrays must have the same length")

    positives = [i for i, label in enumerate(y_true) if label]
    negatives = [i for i, label in enumerate(y_true) if not label]
    if not positives or not negatives:
        raise ValueError("DeLong test requires positive and negative examples")

    auc_1, v01_1, v10_1 = _delong_components(s1, positives, negatives)
    auc_2, v01_2, v10_2 = _delong_components(s2, positives, negatives)

    m = len(positives)
    n = len(negatives)
    sx = _covariance_2(v01_1, v01_2)
    sy = _covariance_2(v10_1, v10_2)
    var = (sx[0][0] + sx[1][1] - 2.0 * sx[0][1]) / m
    var += (sy[0][0] + sy[1][1] - 2.0 * sy[0][1]) / n

    if var <= 0:
        z = 0.0
        p_value = 1.0
    else:
        z = (auc_1 - auc_2) / math.sqrt(var)
        p_value = 2.0 * (1.0 - NORMAL.cdf(abs(z)))

    return DeLongResult(
        auc_1=auc_1,
        auc_2=auc_2,
        z=z,
        p_value=max(0.0, min(1.0, p_value)),
        variance=max(0.0, var),
    )


def _delong_components(
    scores: Sequence[float],
    positives: Sequence[int],
    negatives: Sequence[int],
) -> tuple[float, list[float], list[float]]:
    pos_scores = [scores[i] for i in positives]
    neg_scores = [scores[i] for i in negatives]
    m = len(pos_scores)
    n = len(neg_scores)
    all_scores = pos_scores + neg_scores
    tx = _average_ranks(pos_scores)
    ty = _average_ranks(neg_scores)
    tz = _average_ranks(all_scores)
    auc = sum(tz[:m]) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = [(tz[i] - tx[i]) / n for i in range(m)]
    v10 = [1.0 - (tz[m + j] - ty[j]) / m for j in range(n)]
    return auc, v01, v10


def _covariance_2(x: Sequence[float], y: Sequence[float]) -> list[list[float]]:
    if len(x) != len(y):
        raise ValueError("covariance inputs must have the same length")
    if len(x) <= 1:
        return [[0.0, 0.0], [0.0, 0.0]]
    mean_x = _mean(x)
    mean_y = _mean(y)
    cov_xx = sum((value - mean_x) ** 2 for value in x) / (len(x) - 1)
    cov_yy = sum((value - mean_y) ** 2 for value in y) / (len(y) - 1)
    cov_xy = sum((x_i - mean_x) * (y_i - mean_y) for x_i, y_i in zip(x, y, strict=True)) / (
        len(x) - 1
    )
    return [[cov_xx, cov_xy], [cov_xy, cov_yy]]


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """Return BH-FDR adjusted p-values in the original order."""

    if not p_values:
        return []
    indexed = sorted(enumerate(float(p) for p in p_values), key=lambda item: item[1])
    m = len(indexed)
    adjusted_sorted = [0.0] * m
    running_min = 1.0
    for rank_from_end, (original_index, p_value) in enumerate(reversed(indexed), start=1):
        rank = m - rank_from_end + 1
        adjusted = min(running_min, p_value * m / rank)
        running_min = adjusted
        adjusted_sorted[rank - 1] = min(1.0, max(0.0, adjusted))

    adjusted_original = [0.0] * m
    for (original_index, _), adjusted in zip(indexed, adjusted_sorted, strict=True):
        adjusted_original[original_index] = adjusted
    return adjusted_original
