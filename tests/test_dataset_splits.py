from __future__ import annotations

import pandas as pd
import pytest

from biases.dataset_splits import (
    assign_question_disjoint_routing_split,
    assign_routing_split,
    routing_assignment_sha256,
    routing_manifest,
)


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


def test_question_disjoint_split_is_deterministic_and_keeps_groups_together() -> None:
    frame = pd.DataFrame(
        {
            "row_id": list(range(16)),
            "question_id": [
                question_id
                for question_id in ("q1", "q2", "q3", "q4")
                for _ in range(4)
            ],
            "turn": [1, 2, 1, 2] * 4,
            "winner": ["A", "B", "tie", "A"] * 4,
        }
    )

    first = assign_question_disjoint_routing_split(
        frame,
        calibration_fraction=0.5,
        seed=42,
    )
    second = assign_question_disjoint_routing_split(
        frame,
        calibration_fraction=0.5,
        seed=42,
    )

    pd.testing.assert_frame_equal(first, second)
    assert first["row_id"].tolist() == frame["row_id"].tolist()
    split_counts = (
        first[["question_id", "routing_split"]]
        .drop_duplicates()["routing_split"]
        .value_counts()
        .to_dict()
    )
    assert split_counts == {"calibration": 2, "test": 2}
    assert first.groupby("question_id")["routing_split"].nunique().eq(1).all()


def test_question_disjoint_split_is_stable_under_row_reordering() -> None:
    frame = pd.DataFrame(
        {
            "question_id": [f"q{index}" for index in range(10) for _ in range(2)],
            "winner": ["A", "B"] * 10,
        }
    )
    reordered = frame.sample(frac=1, random_state=7)

    first = assign_question_disjoint_routing_split(
        frame,
        calibration_fraction=0.5,
        seed=19,
    )
    second = assign_question_disjoint_routing_split(
        reordered,
        calibration_fraction=0.5,
        seed=19,
    )

    first_assignment = (
        first[["question_id", "routing_split"]]
        .drop_duplicates()
        .set_index("question_id")["routing_split"]
        .to_dict()
    )
    second_assignment = (
        second[["question_id", "routing_split"]]
        .drop_duplicates()
        .set_index("question_id")["routing_split"]
        .to_dict()
    )
    assert first_assignment == second_assignment
    assert routing_assignment_sha256(first) == routing_assignment_sha256(second)


@pytest.mark.parametrize(
    ("frame", "message"),
    (
        (pd.DataFrame({"winner": ["A", "B"]}), "question_id"),
        (
            pd.DataFrame({"question_id": ["q1", None]}),
            "missing values",
        ),
        (
            pd.DataFrame({"question_id": ["q1", " "]}),
            "blank values",
        ),
        (
            pd.DataFrame({"question_id": ["q1", "q1"]}),
            "at least two unique questions",
        ),
    ),
)
def test_question_disjoint_split_validates_groups(
    frame: pd.DataFrame,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        assign_question_disjoint_routing_split(
            frame,
            calibration_fraction=0.5,
            seed=42,
        )


def test_routing_assignment_digest_rejects_question_overlap() -> None:
    frame = pd.DataFrame(
        {
            "question_id": ["q1", "q1", "q2"],
            "routing_split": ["calibration", "test", "test"],
        }
    )

    with pytest.raises(ValueError, match="both routing splits"):
        routing_assignment_sha256(frame)


def test_routing_manifest_records_counts_and_assignment_digest() -> None:
    frame = assign_question_disjoint_routing_split(
        pd.DataFrame(
            {
                "question_id": ["q1", "q1", "q2", "q3", "q3", "q4"],
                "winner": ["A", "B", "A", "tie", "B", "A"],
            }
        ),
        calibration_fraction=0.5,
        seed=42,
    )

    manifest = routing_manifest(
        frame,
        routing_unit="question",
        seed=42,
        calibration_fraction=0.5,
    )

    assert manifest["routing_unit"] == "question"
    assert manifest["seed"] == 42
    assert manifest["calibration_fraction"] == 0.5
    assert manifest["row_counts"]["total"] == 6
    assert manifest["question_counts"] == {
        "total": 4,
        "calibration": 2,
        "test": 2,
        "overlap": 0,
    }
    assert (
        manifest["routing_assignment_sha256"]
        == routing_assignment_sha256(frame)
    )


def test_row_routing_manifest_records_question_overlap() -> None:
    frame = assign_routing_split(
        pd.DataFrame(
            {
                "question_id": ["q1", "q2", "q2", "q1"],
                "winner": ["A", "B", "A", "B"],
            }
        ),
        calibration_fraction=0.5,
        seed=42,
    )

    manifest = routing_manifest(
        frame,
        routing_unit="row",
        seed=42,
        calibration_fraction=0.5,
    )

    assert manifest["routing_unit"] == "row"
    assert manifest["row_counts"] == {
        "total": 4,
        "calibration": 2,
        "test": 2,
    }
    assert manifest["question_counts"]["total"] == 2
    assert manifest["question_counts"]["overlap"] == 2
    assert manifest["routing_assignment_sha256"] == routing_assignment_sha256(
        frame,
        routing_unit="row",
    )
