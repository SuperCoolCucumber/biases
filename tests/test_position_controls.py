from __future__ import annotations

from pathlib import Path

from biases.position_bias import load_position_pairs
from biases.position_controls import build_identical_answer_example, build_label_prior_example
from biases.schemas import VerdictLabel


def test_identical_answer_control_duplicates_human_winner(tmp_path: Path) -> None:
    csv_path = tmp_path / "pairs.csv"
    csv_path.write_text(
        "\n".join(
            [
                "example_id,prompt,response_a,response_b,winner",
                "q1,Choose.,Good answer,Bad answer,A",
            ]
        ),
        encoding="utf-8",
    )
    pair = load_position_pairs(csv_path)[0]

    example = build_identical_answer_example(pair, source_side="human_winner")

    assert example.candidates["A"].response == "Good answer"
    assert example.candidates["B"].response == "Good answer"
    assert example.human_winner == VerdictLabel.TIE
    assert example.metadata["response_id_by_label"]["A"] == example.metadata["response_id_by_label"]["B"]


def test_identical_answer_control_can_duplicate_b_side(tmp_path: Path) -> None:
    csv_path = tmp_path / "pairs.csv"
    csv_path.write_text(
        "\n".join(
            [
                "example_id,prompt,response_a,response_b,winner",
                "q1,Choose.,Good answer,Bad answer,A",
            ]
        ),
        encoding="utf-8",
    )
    pair = load_position_pairs(csv_path)[0]

    example = build_identical_answer_example(pair, source_side="B")

    assert example.candidates["A"].response == "Bad answer"
    assert example.candidates["B"].response == "Bad answer"


def test_label_prior_example_has_identical_placeholder_answers() -> None:
    example = build_label_prior_example()

    assert example.candidates["A"].response == example.candidates["B"].response
    assert example.metadata["response_id_by_label"]["A"] == example.metadata["response_id_by_label"]["B"]
