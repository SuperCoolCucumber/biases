from __future__ import annotations

import csv
from pathlib import Path

import pytest

from biases.position_bias import (
    QwenJudge,
    _extract_conversation_pair,
    _parse_conversation,
    load_position_pairs,
)
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


@pytest.mark.parametrize(
    ("raw_conversation", "expected"),
    [
        (
            repr(
                [
                    {"role": "user", "content": "Question?"},
                    {"role": "assistant", "content": "Answer."},
                ]
            ),
            [
                {"role": "user", "content": "Question?"},
                {"role": "assistant", "content": "Answer."},
            ],
        ),
        (
            repr(
                {
                    "messages": [
                        {"speaker": "user", "text": "Wrapped question?"},
                    ]
                }
            ),
            [{"role": "user", "content": "Wrapped question?"}],
        ),
    ],
)
def test_parse_conversation_accepts_safe_python_literal_structures(
    raw_conversation: str,
    expected: list[dict[str, str]],
) -> None:
    assert _parse_conversation(raw_conversation) == expected


@pytest.mark.parametrize(
    "raw_conversation",
    (
        "not valid structured data",
        "'literal scalar'",
        "42",
        "__import__('os').system('false')",
    ),
)
def test_parse_conversation_keeps_unsupported_literals_as_plain_text(
    raw_conversation: str,
) -> None:
    assert _parse_conversation(raw_conversation) == [
        {"role": "user", "content": raw_conversation}
    ]


def test_load_position_pairs_extracts_python_repr_conversations(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "pairs.csv"
    conversation_a = [
        {"role": "user", "content": "Explain the result."},
        {"role": "assistant", "content": "Answer from A"},
    ]
    conversation_b = [
        {"role": "user", "content": "Explain the result."},
        {"role": "assistant", "content": "Answer from B"},
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "question_id",
                "model_a",
                "model_b",
                "winner",
                "turn",
                "conversation_a",
                "conversation_b",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "question_id": "q-python-repr",
                "model_a": "model-a",
                "model_b": "model-b",
                "winner": "model_a",
                "turn": "1",
                "conversation_a": repr(conversation_a),
                "conversation_b": repr(conversation_b),
            }
        )

    pair = load_position_pairs(csv_path)[0]

    assert pair.original.prompt_messages == [
        {"role": "user", "content": "Explain the result."}
    ]
    assert pair.original.candidates["A"].response == "Answer from A"
    assert pair.original.candidates["B"].response == "Answer from B"
    assert pair.swapped.candidates["A"].response == "Answer from B"
    assert pair.swapped.candidates["B"].response == "Answer from A"


def _write_two_turn_pair(path: Path, *, turn: str) -> None:
    shared_questions = ("Initial question?", "Follow-up question?")
    conversation_a = [
        {"role": "user", "content": shared_questions[0]},
        {"role": "assistant", "content": "A first answer."},
        {"role": "user", "content": shared_questions[1]},
        {"role": "assistant", "content": "A second answer."},
    ]
    conversation_b = [
        {"role": "user", "content": shared_questions[0]},
        {"role": "assistant", "content": "B first answer."},
        {"role": "user", "content": shared_questions[1]},
        {"role": "assistant", "content": "B second answer."},
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "question_id",
                "winner",
                "turn",
                "conversation_a",
                "conversation_b",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "question_id": "q-two-turn",
                "winner": "model_a",
                "turn": turn,
                "conversation_a": repr(conversation_a),
                "conversation_b": repr(conversation_b),
            }
        )


def test_mtbench_turn_one_selects_first_question_and_first_answers(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "turn-one.csv"
    _write_two_turn_pair(csv_path, turn="1")

    pair = load_position_pairs(csv_path)[0]

    assert pair.original.prompt_messages == [
        {"role": "user", "content": "Initial question?"},
    ]
    assert pair.original.candidates["A"].response == "A first answer."
    assert pair.original.candidates["B"].response == "B first answer."
    assert "second" not in pair.original.candidates["A"].response.lower()
    assert pair.original.metadata["selected_turn"] == 1
    assert pair.original.metadata["conversation_extraction_mode"] == "mtbench_turn_1"


def test_mtbench_turn_two_preserves_context_and_marks_target(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "turn-two.csv"
    _write_two_turn_pair(csv_path, turn="2")

    pair = load_position_pairs(csv_path)[0]
    prompt = build_position_prompt_package(
        pair.original,
        output_mode=OutputMode.CHOICE_ONLY,
    ).prompt_text

    assert pair.original.prompt_messages == [
        {
            "role": "user",
            "content": "Turn 1 user question (context):\nInitial question?",
        },
        {
            "role": "user",
            "content": (
                "Turn 2 user question (evaluate the response to this question):\n"
                "Follow-up question?"
            ),
        },
    ]
    assert pair.original.candidates["A"].response == (
        "Turn 1 assistant response (context):\nA first answer.\n\n"
        "Turn 2 assistant response (evaluate this response):\nA second answer."
    )
    assert pair.original.candidates["B"].response == (
        "Turn 1 assistant response (context):\nB first answer.\n\n"
        "Turn 2 assistant response (evaluate this response):\nB second answer."
    )
    assert "Follow-up question?" in prompt
    assert "A first answer." in prompt
    assert "A second answer." in prompt
    assert "evaluate this response" in prompt
    assert pair.original.metadata["selected_turn"] == 2
    assert pair.original.metadata["conversation_extraction_mode"] == "mtbench_turn_2"


def test_mtbench_turn_two_rejects_an_empty_target_answer() -> None:
    conversation_a = [
        {"role": "user", "content": "First question?"},
        {"role": "assistant", "content": "First answer."},
        {"role": "user", "content": "Second question?"},
        {"role": "assistant", "content": ""},
    ]
    conversation_b = [
        {"role": "user", "content": "First question?"},
        {"role": "assistant", "content": "Other first answer."},
        {"role": "user", "content": "Second question?"},
        {"role": "assistant", "content": "Other second answer."},
    ]

    extraction = _extract_conversation_pair(
        conversation_a,
        conversation_b,
        turn="2",
    )

    assert extraction.mode == "mtbench_turn_2"
    assert extraction.response_a == ""
    assert extraction.response_b


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
