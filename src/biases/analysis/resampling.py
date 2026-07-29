from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Callable, Hashable, Iterator, Sequence
from dataclasses import dataclass
from typing import TypeVar


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class PercentileInterval:
    estimate: float
    low: float
    high: float
    confidence: float
    n_clusters: int
    n_resamples: int


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def cluster_resamples(
    records: Sequence[T],
    *,
    cluster_key: Callable[[T], Hashable],
    n_resamples: int = 2000,
    seed: int = 0,
) -> Iterator[tuple[T, ...]]:
    if not records:
        raise ValueError("records must not be empty")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    grouped: dict[Hashable, list[T]] = defaultdict(list)
    for record in records:
        grouped[cluster_key(record)].append(record)
    keys = sorted(grouped, key=repr)
    rng = random.Random(seed)
    for _ in range(n_resamples):
        sampled: list[T] = []
        for _ in keys:
            sampled.extend(grouped[keys[rng.randrange(len(keys))]])
        yield tuple(sampled)


def cluster_percentile_interval(
    records: Sequence[T],
    *,
    cluster_key: Callable[[T], Hashable],
    statistic: Callable[[Sequence[T]], float],
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
) -> PercentileInterval:
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    estimate = float(statistic(records))
    samples = cluster_resamples(
        records,
        cluster_key=cluster_key,
        n_resamples=n_resamples,
        seed=seed,
    )
    estimates = [float(statistic(sample)) for sample in samples]
    finite = [value for value in estimates if math.isfinite(value)]
    if not finite:
        raise ValueError("bootstrap statistic was non-finite for every resample")
    alpha = 1.0 - confidence
    return PercentileInterval(
        estimate=estimate,
        low=percentile(finite, alpha / 2.0),
        high=percentile(finite, 1.0 - alpha / 2.0),
        confidence=confidence,
        n_clusters=len({cluster_key(record) for record in records}),
        n_resamples=n_resamples,
    )


def cluster_sign_flip_p_value(
    records: Sequence[T],
    *,
    cluster_key: Callable[[T], Hashable],
    value: Callable[[T], float],
    n_permutations: int = 2000,
    seed: int = 0,
) -> float:
    """One-sided cluster sign-flip test for a positive mean.

    Every observation in a sampled question cluster receives the same random
    sign, preserving dependence between turns and orderings.
    """

    if not records:
        raise ValueError("records must not be empty")
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")
    grouped: dict[Hashable, list[float]] = defaultdict(list)
    for record in records:
        grouped[cluster_key(record)].append(float(value(record)))
    keys = sorted(grouped, key=repr)
    observed = sum(sum(grouped[key]) for key in keys) / sum(
        len(grouped[key]) for key in keys
    )
    rng = random.Random(seed)
    exceedances = 0
    for _ in range(n_permutations):
        null_sum = sum(
            (1.0 if rng.random() < 0.5 else -1.0) * sum(grouped[key])
            for key in keys
        )
        null_mean = null_sum / sum(len(grouped[key]) for key in keys)
        exceedances += null_mean >= observed
    return (exceedances + 1) / (n_permutations + 1)
