from __future__ import annotations

from pathlib import Path
import random

import pytest

from biases.analysis.modeling import (
    cluster_bootstrap_uncertainty_gee_slopes,
    fit_uncertainty_gee,
)
from scripts.analyze_silent_bias import parse_args


def _nondegenerate_gee_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    doses = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)
    for question_index in range(6):
        for dose_index, dose in enumerate(doses):
            rows.append(
                {
                    "question_id": f"q-{question_index}",
                    "normalized_dose": dose,
                    "uncertainty": (
                        0.15
                        + question_index * 0.013
                        + (0.18 + question_index * 0.007) * dose
                        + ((question_index + dose_index) % 3) * 0.004
                    ),
                }
            )
    return rows


def _legacy_serial_bootstrap_slopes(
    rows: list[dict[str, object]],
    *,
    n_resamples: int,
    seed: int,
) -> tuple[float, ...]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["question_id"]), []).append(row)
    keys = sorted(grouped, key=repr)
    rng = random.Random(seed)
    slopes: list[float] = []
    for _ in range(n_resamples):
        sampled: list[dict[str, object]] = []
        for draw_index in range(len(keys)):
            sampled_key = keys[rng.randrange(len(keys))]
            sampled.extend(
                {
                    **row,
                    "question_id": f"bootstrap-{draw_index}",
                }
                for row in grouped[sampled_key]
            )
        slopes.append(fit_uncertainty_gee(sampled).slope)
    return tuple(slopes)


def test_parallel_gee_bootstrap_matches_serial_draw_for_draw() -> None:
    pytest.importorskip("statsmodels")
    rows = _nondegenerate_gee_rows()
    expected = _legacy_serial_bootstrap_slopes(
        rows,
        n_resamples=12,
        seed=31415,
    )

    serial = cluster_bootstrap_uncertainty_gee_slopes(
        rows,
        n_resamples=12,
        seed=31415,
        workers=1,
    )
    parallel = cluster_bootstrap_uncertainty_gee_slopes(
        rows,
        n_resamples=12,
        seed=31415,
        workers=2,
    )

    assert serial == expected
    assert parallel == expected
    assert len(serial) == 12
    assert all(slope is not None for slope in serial)


def test_analysis_cli_rejects_nonpositive_gee_bootstrap_workers(
    tmp_path: Path,
) -> None:
    required = [
        "--stage-a",
        str(tmp_path / "stage-a.jsonl"),
        "--stage-b",
        str(tmp_path / "stage-b.jsonl"),
        "--output-dir",
        str(tmp_path / "analysis"),
    ]

    with pytest.raises(SystemExit) as exc_info:
        parse_args([*required, "--gee-bootstrap-workers", "0"])

    assert exc_info.value.code == 2
