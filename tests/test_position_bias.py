from __future__ import annotations

from pathlib import Path

from biases.position_bias import load_position_pairs
from biases.position_prompts import build_position_prompt_package
from biases.schemas import OutputMode, VerdictLabel


def test_load_position_pairs_creates_original_and_swapped_examples(tmp_path: Path) -> None:
    csv_path = tmp_path / "pairs.csv"
    csv_path.write_text(
        "\n".join(
            [
                "example_id,prompt,response_a,response_b,winner,model_a,model_b",
                "q1,Explain regularization.,Answer from A,Answer from B,B,model-a,model-b",
            ]
        ),
        encoding="utf-8",
    )

    pairs = load_position_pairs(csv_path)

    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.original.candidates["A"].response == "Answer from A"
    assert pair.original.candidates["B"].response == "Answer from B"
    assert pair.swapped.candidates["A"].response == "Answer from B"
    assert pair.swapped.candidates["B"].response == "Answer from A"
    assert pair.original.human_winner == VerdictLabel.B
    assert pair.swapped.human_winner == VerdictLabel.A
    assert pair.original.metadata["response_id_by_label"]["A"] == "q1:response_a"
    assert pair.swapped.metadata["response_id_by_label"]["A"] == "q1:response_b"


def test_position_prompt_package_contains_both_answers(tmp_path: Path) -> None:
    csv_path = tmp_path / "pairs.csv"
    csv_path.write_text(
        "\n".join(
            [
                "example_id,prompt,response_a,response_b,winner",
                "q2,Why does dropout help?,First answer,Second answer,A",
            ]
        ),
        encoding="utf-8",
    )

    pair = load_position_pairs(csv_path)[0]
    prompt = build_position_prompt_package(pair.original, output_mode=OutputMode.CHOICE_ONLY)

    assert "Why does dropout help?" in prompt.prompt_text
    assert "Answer A:\nFirst answer" in prompt.prompt_text
    assert "Answer B:\nSecond answer" in prompt.prompt_text
    assert list(prompt.allowed_labels) == [VerdictLabel.A, VerdictLabel.B, VerdictLabel.TIE] or list(
        prompt.allowed_labels
    ) == ["A", "B", "tie"]
