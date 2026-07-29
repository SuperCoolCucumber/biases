from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from biases.analysis.resampling import cluster_resamples, percentile
from biases.analysis.rq1 import PairedShift


SOCIAL_DOSE_LADDERS: dict[str, tuple[float, float, float, float]] = {
    "authority": (1.0, 2.0, 3.0, 4.0),
    "bandwagon": (55.0, 70.0, 85.0, 95.0),
}
NORMALIZED_FOUR_LEVEL_DOSES = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)


def normalized_social_dose(family: str, dose: float) -> float:
    ladder = SOCIAL_DOSE_LADDERS.get(family.lower())
    if ladder is None:
        raise ValueError(f"no canonical four-level dose ladder for family {family!r}")
    numeric = float(dose)
    for raw_dose, normalized in zip(
        ladder,
        NORMALIZED_FOUR_LEVEL_DOSES,
        strict=True,
    ):
        if math.isclose(numeric, raw_dose):
            return normalized
    raise ValueError(f"dose {dose!r} is not in the canonical {family} ladder {ladder}")


@dataclass(frozen=True, slots=True)
class DoseObservation:
    question_id: str
    dose: float
    event: bool


@dataclass(frozen=True, slots=True)
class LogisticFit:
    n: int
    events: int
    intercept: float
    slope: float
    p25_dose: float | None
    converged: bool
    iterations: int


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        inverse = math.exp(-min(value, 700.0))
        return 1.0 / (1.0 + inverse)
    exponent = math.exp(max(value, -700.0))
    return exponent / (1.0 + exponent)


def fit_logistic_dose(
    observations: Sequence[DoseObservation],
    *,
    ridge: float = 1e-8,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
) -> LogisticFit:
    if len(observations) < 2:
        raise ValueError("at least two observations are required")
    if len({observation.dose for observation in observations}) < 2:
        raise ValueError("at least two distinct dose values are required")
    intercept = 0.0
    slope = 0.0
    converged = False
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        gradient_0 = 0.0
        gradient_1 = 0.0
        hessian_00 = ridge
        hessian_01 = 0.0
        hessian_11 = ridge
        for observation in observations:
            probability = _sigmoid(intercept + slope * observation.dose)
            residual = float(observation.event) - probability
            weight = max(probability * (1.0 - probability), 1e-12)
            gradient_0 += residual
            gradient_1 += residual * observation.dose
            hessian_00 += weight
            hessian_01 += weight * observation.dose
            hessian_11 += weight * observation.dose * observation.dose
        gradient_0 -= ridge * intercept
        gradient_1 -= ridge * slope
        determinant = hessian_00 * hessian_11 - hessian_01 * hessian_01
        if abs(determinant) < 1e-18:
            break
        step_0 = (hessian_11 * gradient_0 - hessian_01 * gradient_1) / determinant
        step_1 = (-hessian_01 * gradient_0 + hessian_00 * gradient_1) / determinant
        intercept += step_0
        slope += step_1
        if max(abs(step_0), abs(step_1)) < tolerance:
            converged = True
            break
    target_logit = math.log(0.25 / 0.75)
    p25 = (target_logit - intercept) / slope if abs(slope) > 1e-12 else None
    return LogisticFit(
        n=len(observations),
        events=sum(observation.event for observation in observations),
        intercept=intercept,
        slope=slope,
        p25_dose=p25 if p25 is None or math.isfinite(p25) else None,
        converged=converged,
        iterations=iteration,
    )


@dataclass(frozen=True, slots=True)
class DoseResponseResult:
    n: int
    events: int
    intercept: float
    slope: float
    slope_ci_low: float | None
    slope_ci_high: float | None
    slope_p_value_one_sided: float | None
    p25_dose: float | None
    p25_ci_low: float | None
    p25_ci_high: float | None
    dose_min: float
    dose_max: float
    p25_range_status: Literal[
        "below_tested_range",
        "within_tested_range",
        "above_tested_range",
        "unavailable",
    ]
    converged: bool
    n_clusters: int
    n_resamples: int


def fit_dose_response_with_cluster_bootstrap(
    observations: Sequence[DoseObservation],
    *,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
) -> DoseResponseResult:
    fit = fit_logistic_dose(observations)
    dose_min = min(observation.dose for observation in observations)
    dose_max = max(observation.dose for observation in observations)
    slopes: list[float] = []
    thresholds: list[float] = []
    for sample in cluster_resamples(
        observations,
        cluster_key=lambda observation: observation.question_id,
        n_resamples=n_resamples,
        seed=seed,
    ):
        try:
            sampled_fit = fit_logistic_dose(sample)
        except ValueError:
            continue
        if math.isfinite(sampled_fit.slope):
            slopes.append(sampled_fit.slope)
        if sampled_fit.p25_dose is not None and math.isfinite(sampled_fit.p25_dose):
            thresholds.append(sampled_fit.p25_dose)
    alpha = 1.0 - confidence
    if fit.p25_dose is None:
        p25_range_status = "unavailable"
    elif fit.p25_dose < dose_min:
        p25_range_status = "below_tested_range"
    elif fit.p25_dose > dose_max:
        p25_range_status = "above_tested_range"
    else:
        p25_range_status = "within_tested_range"
    return DoseResponseResult(
        n=fit.n,
        events=fit.events,
        intercept=fit.intercept,
        slope=fit.slope,
        slope_ci_low=percentile(slopes, alpha / 2.0) if slopes else None,
        slope_ci_high=percentile(slopes, 1.0 - alpha / 2.0) if slopes else None,
        slope_p_value_one_sided=(
            (1 + sum(slope <= 0.0 for slope in slopes)) / (len(slopes) + 1)
            if slopes
            else None
        ),
        p25_dose=fit.p25_dose,
        p25_ci_low=percentile(thresholds, alpha / 2.0) if thresholds else None,
        p25_ci_high=percentile(thresholds, 1.0 - alpha / 2.0) if thresholds else None,
        dose_min=dose_min,
        dose_max=dose_max,
        p25_range_status=p25_range_status,
        converged=fit.converged,
        n_clusters=len({observation.question_id for observation in observations}),
        n_resamples=n_resamples,
    )


def dose_observations_from_shifts(shifts: Sequence[PairedShift]) -> tuple[DoseObservation, ...]:
    return tuple(
        DoseObservation(
            question_id=shift.question_id,
            dose=shift.dose,
            event=shift.flip,
        )
        for shift in shifts
        if shift.dose is not None
    )


@dataclass(frozen=True, slots=True)
class TrendObservation:
    question_id: str
    dose: float
    value: float


@dataclass(frozen=True, slots=True)
class MonotonicTrendResult:
    statistic: float
    p_value: float
    n_clusters: int
    n_doses: int
    doses: tuple[float, ...]
    n_permutations: int


def _average_ranks(values: Sequence[float]) -> tuple[float, ...]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(indexed):
        end = cursor + 1
        while end < len(indexed) and indexed[end][1] == indexed[cursor][1]:
            end += 1
        rank = (cursor + 1 + end) / 2.0
        for position in range(cursor, end):
            ranks[indexed[position][0]] = rank
        cursor = end
    return tuple(ranks)


def page_monotonic_trend_test(
    observations: Sequence[TrendObservation],
    *,
    n_permutations: int = 10_000,
    seed: int = 0,
) -> MonotonicTrendResult:
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")
    grouped: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for observation in observations:
        grouped[observation.question_id][observation.dose].append(observation.value)
    doses = tuple(sorted({observation.dose for observation in observations}))
    if len(doses) < 2:
        raise ValueError("at least two distinct doses are required")
    blocks = [
        tuple(sum(by_dose[dose]) / len(by_dose[dose]) for dose in doses)
        for _, by_dose in sorted(grouped.items())
        if all(dose in by_dose for dose in doses)
    ]
    if not blocks:
        raise ValueError("no question has a complete dose profile")

    def statistic(block_values: Sequence[Sequence[float]]) -> float:
        return sum(
            (dose_index + 1) * rank
            for block in block_values
            for dose_index, rank in enumerate(_average_ranks(block))
        )

    observed = statistic(blocks)
    rng = random.Random(seed)
    exceedances = 0
    for _ in range(n_permutations):
        permuted: list[tuple[float, ...]] = []
        for block in blocks:
            shuffled = list(block)
            rng.shuffle(shuffled)
            permuted.append(tuple(shuffled))
        exceedances += statistic(permuted) >= observed
    return MonotonicTrendResult(
        statistic=observed,
        p_value=(exceedances + 1) / (n_permutations + 1),
        n_clusters=len(blocks),
        n_doses=len(doses),
        doses=doses,
        n_permutations=n_permutations,
    )


def clustered_monotonic_trend_test(
    observations: Sequence[TrendObservation],
    *,
    n_permutations: int = 10_000,
    seed: int = 0,
) -> MonotonicTrendResult:
    """One-sided within-question permutation test for an increasing slope.

    Unlike Page's complete-block test, this supports pre-first-flip and
    current-dose non-flip sets whose observed dose ladders are incomplete.
    """

    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")
    grouped: dict[str, list[TrendObservation]] = defaultdict(list)
    for observation in observations:
        grouped[observation.question_id].append(observation)
    blocks = [
        tuple(sorted(block, key=lambda observation: observation.dose))
        for _, block in sorted(grouped.items())
        if len({observation.dose for observation in block}) >= 2
    ]
    if not blocks:
        raise ValueError("no question has observations at two or more doses")

    def slope(block_values: Sequence[Sequence[TrendObservation]]) -> float:
        numerator = 0.0
        denominator = 0.0
        for block in block_values:
            mean_dose = sum(item.dose for item in block) / len(block)
            mean_value = sum(item.value for item in block) / len(block)
            numerator += sum(
                (item.dose - mean_dose) * (item.value - mean_value)
                for item in block
            )
            denominator += sum((item.dose - mean_dose) ** 2 for item in block)
        if denominator <= 0.0:
            raise ValueError("within-question dose variance is zero")
        return numerator / denominator

    observed = slope(blocks)
    rng = random.Random(seed)
    exceedances = 0
    for _ in range(n_permutations):
        permuted: list[tuple[TrendObservation, ...]] = []
        for block in blocks:
            values = [item.value for item in block]
            rng.shuffle(values)
            permuted.append(
                tuple(
                    TrendObservation(item.question_id, item.dose, value)
                    for item, value in zip(block, values, strict=True)
                )
            )
        exceedances += slope(permuted) >= observed
    doses = tuple(sorted({item.dose for block in blocks for item in block}))
    return MonotonicTrendResult(
        statistic=observed,
        p_value=(exceedances + 1) / (n_permutations + 1),
        n_clusters=len(blocks),
        n_doses=len(doses),
        doses=doses,
        n_permutations=n_permutations,
    )
