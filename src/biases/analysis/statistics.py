from __future__ import annotations

import math
from collections.abc import Sequence


def holm_adjust(p_values: Sequence[float]) -> tuple[float, ...]:
    if not p_values:
        return ()
    values = [float(value) for value in p_values]
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("p-values must be finite and in [0, 1]")
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    adjusted_sorted: list[float] = []
    running_max = 0.0
    count = len(values)
    for rank, (_, p_value) in enumerate(ordered):
        adjusted = min(1.0, (count - rank) * p_value)
        running_max = max(running_max, adjusted)
        adjusted_sorted.append(running_max)
    result = [0.0] * count
    for (original_index, _), adjusted in zip(ordered, adjusted_sorted, strict=True):
        result[original_index] = adjusted
    return tuple(result)
