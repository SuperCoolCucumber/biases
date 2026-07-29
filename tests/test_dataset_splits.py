from __future__ import annotations

import pandas as pd
import pytest

from biases.dataset_splits import assign_routing_split


def test_routing_split_is_deterministic_and_stratified() -> None:
    frame = pd.DataFrame(
        {
            "row_id": list(range(12)),
            "winner": ["A"] * 4 + ["B"] * 4 + ["tie"] * 4,
        }
    )

    first = assign_routing_split(
        frame,
        calibration_fraction=0.5,
        seed=42,
    )
    second = assign_routing_split(
        frame,
        calibration_fraction=0.5,
        seed=42,
    )

    pd.testing.assert_frame_equal(first, second)
    counts = first.groupby(["winner", "routing_split"]).size().to_dict()
    assert counts == {
        ("A", "calibration"): 2,
        ("A", "test"): 2,
        ("B", "calibration"): 2,
        ("B", "test"): 2,
        ("tie", "calibration"): 2,
        ("tie", "test"): 2,
    }


def test_routing_split_validates_inputs() -> None:
    with pytest.raises(ValueError, match="calibration_fraction"):
        assign_routing_split(
            pd.DataFrame({"winner": ["A", "B"]}),
            calibration_fraction=1.0,
            seed=42,
        )
    with pytest.raises(ValueError, match="winner"):
        assign_routing_split(
            pd.DataFrame({"label": ["A", "B"]}),
            calibration_fraction=0.5,
            seed=42,
        )
