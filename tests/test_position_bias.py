from __future__ import annotations

from pathlib import Path

from biases.position_bias import QwenJudge, load_position_pairs
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


def test_qwen3_prompt_prefills_empty_thinking_block() -> None:
    judge = QwenJudge.__new__(QwenJudge)
    judge.model_name = "Qwen/Qwen3-32B"

    prompt = "<|im_start|>system\nsys<|im_end|>\n<|im_start|>user\nu<|im_end|>\n<|im_start|>assistant\n"

    prepared = judge._prepare_prompt(prompt)

    assert prepared.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_non_qwen3_prompt_is_unchanged() -> None:
    judge = QwenJudge.__new__(QwenJudge)
    judge.model_name = "Qwen/Qwen2.5-14B-Instruct"
    prompt = "<|im_start|>assistant\n"

    assert judge._prepare_prompt(prompt) == prompt


class _Logprob:
    def __init__(self, decoded_token: str, logprob: float) -> None:
        self.decoded_token = decoded_token
        self.logprob = logprob


def test_label_probs_are_aggregated_from_decision_token_ids() -> None:
    judge = QwenJudge.__new__(QwenJudge)
    judge.decision_token_id_to_label = {
        10: "A",
        11: "A",
        20: "B",
        30: "tie",
    }

    probs = judge._extract_label_probs(
        {
            10: _Logprob("A", -0.2),
            11: _Logprob(" A", -1.2),
            20: _Logprob("B", -2.0),
            30: _Logprob("T", -3.0),
        }
    )

    assert set(probs) == {"A", "B", "tie"}
    assert probs["A"] > probs["B"] > probs["tie"]
    assert abs(sum(probs.values()) - 1.0) < 1e-9


def test_parse_confidence_accepts_expected_two_line_format() -> None:
    assert QwenJudge._parse_confidence("A\nConfidence: 87") == 87.0
    assert QwenJudge._parse_confidence("B\nconfidence = 100") == 100.0
