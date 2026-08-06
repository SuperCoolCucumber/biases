from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from biases import position_bias
from biases.position_bias import (
    QwenJudge,
    _extract_conversation_pair,
    _parse_conversation,
    load_position_pairs,
    load_position_pairs_with_eligibility,
    parse_verbalized_output,
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


def test_pair_loader_reports_deterministic_row_eligibility(tmp_path: Path) -> None:
    csv_path = tmp_path / "eligibility.csv"
    rows = [
        ("q-valid-cal", "A", "B", "model_a", "calibration"),
        ("q-valid-test", "A", "B", "model_b", "test"),
        ("q-missing-winner", "A", "B", "", "calibration"),
        ("q-invalid-winner", "A", "B", "neither", "test"),
        ("q-missing-a", "", "B", "model_b", "calibration"),
        ("q-missing-b", "A", "", "model_a", "test"),
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "question_id",
                "prompt",
                "response_a",
                "response_b",
                "winner",
                "routing_split",
            )
        )
        for question_id, response_a, response_b, winner, split in rows:
            writer.writerow(
                (question_id, "Question?", response_a, response_b, winner, split)
            )

    pairs, audit = load_position_pairs_with_eligibility(csv_path)
    repeated_pairs, repeated_audit = load_position_pairs_with_eligibility(csv_path)

    assert [pair.pair_id for pair in pairs] == ["q-valid-cal", "q-valid-test"]
    assert [pair.pair_id for pair in load_position_pairs(csv_path)] == [
        pair.pair_id for pair in pairs
    ]
    assert [pair.pair_id for pair in repeated_pairs] == [
        pair.pair_id for pair in pairs
    ]
    assert audit.raw_row_count == 6
    assert audit.to_dict()["eligibility_contract"] == "position_pair_loader_v1"
    assert audit.eligible_pair_count == 2
    assert audit.skipped_row_count == 4
    assert audit.skipped_reason_counts == {
        "invalid_winner": 1,
        "missing_response_a": 1,
        "missing_response_b": 1,
        "missing_winner": 1,
    }
    assert audit.routing_counts == {
        "raw_rows": {"calibration": 3, "test": 3},
        "eligible_pairs": {"calibration": 1, "test": 1},
        "skipped_rows": {"calibration": 2, "test": 2},
    }
    assert [row["skip_reasons"] for row in audit.skipped_rows] == [
        ["missing_winner"],
        ["invalid_winner"],
        ["missing_response_a"],
        ["missing_response_b"],
    ]
    assert len(audit.eligibility_sha256) == 64
    assert repeated_audit.eligibility_sha256 == audit.eligibility_sha256


@pytest.mark.parametrize(
    "header",
    (
        "question_id,prompt,response_a,winner",
        "question_id,prompt,response_b,winner",
        "question_id,conversation_a,winner",
        "question_id,conversation_b,winner",
    ),
)
def test_pair_loader_rejects_incomplete_input_schemas(
    tmp_path: Path,
    header: str,
) -> None:
    csv_path = tmp_path / "incomplete.csv"
    csv_path.write_text(f"{header}\nq1,value,value,model_a\n", encoding="utf-8")

    with pytest.raises(KeyError, match="prompt/response_a/response_b"):
        load_position_pairs_with_eligibility(csv_path)


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


def test_decision_token_map_rejects_cross_label_id_collisions() -> None:
    class _Profile:
        verdict_token_texts = {
            "A": ("A",),
            "B": ("B",),
            "tie": ("T",),
        }

    class _Tokenizer:
        last_text = ""

        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            assert add_special_tokens is False
            self.last_text = text
            return [{"A": 10, "B": 10, "T": 30}[text]]

        def decode(
            self,
            token_ids: list[int],
            *,
            skip_special_tokens: bool,
            clean_up_tokenization_spaces: bool,
        ) -> str:
            assert skip_special_tokens is False
            assert clean_up_tokenization_spaces is False
            assert token_ids == [{"A": 10, "B": 10, "T": 30}[self.last_text]]
            return self.last_text

    judge = QwenJudge.__new__(QwenJudge)
    judge.profile = _Profile()
    judge.tokenizer = _Tokenizer()

    with pytest.raises(RuntimeError, match="maps to both 'A' and 'B'"):
        judge._build_decision_label_token_maps()


def test_decision_token_map_rejects_nonliteral_singleton_decoding() -> None:
    class _Profile:
        verdict_token_texts = {
            "A": ("A",),
            "B": ("B",),
            "tie": ("T",),
        }

    class _Tokenizer:
        @staticmethod
        def encode(text: str, *, add_special_tokens: bool) -> list[int]:
            assert add_special_tokens is False
            return [{"A": 10, "B": 20, "T": 30}[text]]

        @staticmethod
        def decode(
            token_ids: list[int],
            *,
            skip_special_tokens: bool,
            clean_up_tokenization_spaces: bool,
        ) -> str:
            assert skip_special_tokens is False
            assert clean_up_tokenization_spaces is False
            return {10: " A", 20: "B", 30: "T"}[token_ids[0]]

    judge = QwenJudge.__new__(QwenJudge)
    judge.profile = _Profile()
    judge.tokenizer = _Tokenizer()

    with pytest.raises(
        RuntimeError,
        match="Decision surface 'A'.*decoded as ' A'",
    ):
        judge._build_decision_label_token_maps()


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


def test_label_probs_ignore_unregistered_decoded_label_tokens() -> None:
    judge = QwenJudge.__new__(QwenJudge)
    judge.decision_token_id_to_label = {
        10: "A",
        20: "B",
        30: "tie",
    }

    baseline = judge._extract_label_probs(
        {
            10: _Logprob("A", -0.2),
            20: _Logprob("B", -1.0),
            30: _Logprob("T", -2.0),
        }
    )
    with_unregistered_tokens = judge._extract_label_probs(
        {
            10: _Logprob("A", -0.2),
            20: _Logprob("B", -1.0),
            30: _Logprob("T", -2.0),
            40: _Logprob("A", 10.0),
            50: _Logprob(" B", 9.0),
            60: _Logprob("T", 8.0),
        }
    )

    assert with_unregistered_tokens == baseline


def test_label_probs_reject_missing_registered_token_ids() -> None:
    judge = QwenJudge.__new__(QwenJudge)
    judge.decision_token_id_to_label = {
        10: "A",
        20: "B",
        30: "tie",
    }

    with pytest.raises(
        RuntimeError,
        match="missing token IDs: \\[20\\]",
    ):
        judge._extract_label_probs(
            {
                10: _Logprob("A", -0.2),
                30: _Logprob("T", -2.0),
                40: _Logprob("B", 10.0),
            }
        )


def test_deterministic_verdict_rejects_split_surface_token_mass_mismatch() -> None:
    judge = QwenJudge.__new__(QwenJudge)
    judge.decision_token_id_to_label = {
        10: "A",
        11: "A",
        20: "B",
        30: "tie",
    }
    probabilities = judge._extract_label_probs(
        {
            10: _Logprob("A", math.log(0.30)),
            11: _Logprob(" A", math.log(0.30)),
            20: _Logprob("B", math.log(0.39)),
            30: _Logprob("T", math.log(0.01)),
        }
    )
    assert probabilities["A"] > probabilities["B"]

    with pytest.raises(
        RuntimeError,
        match="token verdict 'B'.*label-probability MAP 'A'",
    ):
        judge._resolve_constrained_verdict(
            raw_text="B",
            probabilities=probabilities,
            sampling_temperature=0.0,
        )


def test_constrained_verdict_returns_map_only_for_deterministic_calls() -> None:
    probabilities = {"A": 0.6, "B": 0.35, "tie": 0.05}

    deterministic = QwenJudge._resolve_constrained_verdict(
        raw_text="A",
        probabilities=probabilities,
        sampling_temperature=0.0,
    )
    sampled = QwenJudge._resolve_constrained_verdict(
        raw_text="B",
        probabilities=probabilities,
        sampling_temperature=0.7,
    )

    assert deterministic == VerdictLabel.A
    assert sampled == VerdictLabel.B


@pytest.mark.parametrize("sampling_temperature", (0.0, 0.7))
def test_constrained_verdict_rejects_unparseable_emitted_text(
    sampling_temperature: float,
) -> None:
    with pytest.raises(ValueError, match="not an unambiguous verdict"):
        QwenJudge._resolve_constrained_verdict(
            raw_text="Answer A is probably best",
            probabilities={"A": 0.6, "B": 0.35, "tie": 0.05},
            sampling_temperature=sampling_temperature,
        )


@pytest.mark.parametrize(
    ("text", "expected"),
    (
        ("A", VerdictLabel.A),
        ("b\nConfidence: 80", VerdictLabel.B),
        ("T", VerdictLabel.TIE),
        ("tie\n\n85", VerdictLabel.TIE),
        ("Verdict: A", VerdictLabel.A),
        ("Answer: B", VerdictLabel.B),
        ("Response: T", VerdictLabel.TIE),
        ("Choice = tie", VerdictLabel.TIE),
        ("Label: [A]", VerdictLabel.A),
        ("[[B]]", VerdictLabel.B),
        ("[T]", VerdictLabel.TIE),
    ),
)
def test_parse_verdict_accepts_only_supported_explicit_forms(
    text: str,
    expected: VerdictLabel,
) -> None:
    assert QwenJudge._parse_verdict_text(text) == expected


@pytest.mark.parametrize(
    "text",
    (
        "",
        "Answer A is stronger.",
        "Because B is more complete.",
        "This is a tie.",
        "A response might be correct.",
        "Verdict: A or B",
        "I choose [[A]].",
        "[[A]] and [[B]]",
        "Confidence: 80",
    ),
)
def test_parse_verdict_rejects_prose_and_ambiguous_forms(text: str) -> None:
    assert QwenJudge._parse_verdict_text(text) is None


@pytest.mark.parametrize(
    ("text", "expected_verdict", "expected_confidence"),
    (
        ("A\nConfidence: 95", VerdictLabel.A, 95.0),
        ("[[B]]\n72.5", VerdictLabel.B, 72.5),
        (
            "Verdict: T\nConfidence = 80\nA short rationale follows.",
            VerdictLabel.TIE,
            80.0,
        ),
        ("A 95", VerdictLabel.A, 95.0),
        ("T Confidence: 80", VerdictLabel.TIE, 80.0),
        ("[B] 67.5%", VerdictLabel.B, 67.5),
    ),
)
def test_joint_verbalized_parser_accepts_only_atomic_pairs(
    text: str,
    expected_verdict: VerdictLabel,
    expected_confidence: float,
) -> None:
    assert parse_verbalized_output(text) == (
        expected_verdict,
        expected_confidence,
    )


@pytest.mark.parametrize(
    ("text", "expected_verdict", "expected_confidence"),
    (
        ("Line 1: A\nLine 2: 0", VerdictLabel.A, 0.0),
        (
            "Line 1: B\nLine 2: Confidence: 72.5",
            VerdictLabel.B,
            72.5,
        ),
        ("1. T\n2. 100", VerdictLabel.TIE, 100.0),
        ("1: A\n2: 81", VerdictLabel.A, 81.0),
        ("1: T\n2: 0", VerdictLabel.TIE, 0.0),
        ("1) B\n2) 64.5", VerdictLabel.B, 64.5),
        ("1) A\n2) 100", VerdictLabel.A, 100.0),
        ("A, 99", VerdictLabel.A, 99.0),
        ("B, 99.5", VerdictLabel.B, 99.5),
    ),
)
def test_joint_verbalized_parser_accepts_exact_hermes_forms(
    text: str,
    expected_verdict: VerdictLabel,
    expected_confidence: float,
) -> None:
    assert parse_verbalized_output(text) == (
        expected_verdict,
        expected_confidence,
    )


@pytest.mark.parametrize(
    "text",
    (
        "1\n100",
        "Line 1: A\nLine 2:",
        "1. B\n2.",
        "T,",
        "Line 1: A\nLine 2: 80\nExplanation follows.",
        "1. B\n2. 75\nExplanation follows.",
        "1: A\n2: 80\nExplanation follows.",
        "1) B\n2) 75\nExplanation follows.",
        "1: A\n2) 80",
        "1) A\n2: 80",
        "1:A\n2: 80",
        "1: a\n2: 80",
        "1: [A]\n2: 80",
        "1: A\n2: Confidence: 80",
        "1: A\n2: 80%",
        "T, 60 because the answers are similar",
        "Line 1: A\nLine 2: 100.1",
        "1. B\n2. -1",
        "1: 1\n2: 80",
        "1) A\n2) 100.1",
        "T, 90%",
    ),
)
def test_joint_verbalized_parser_rejects_non_atomic_hermes_forms(
    text: str,
) -> None:
    assert parse_verbalized_output(text) == (None, None)


@pytest.mark.parametrize(
    "text",
    (
        "Confidence: 95",
        "A\n70% of annotators preferred answer B.",
        "Answer A: it scored 95 on the rubric.",
        "A\nThe rubric has 7 criteria.\n95",
        "A\nReasoning line\nConfidence: 95",
        "A 95 because it is clearer",
        "A\nConfidence: 85\nB",
        "A\nConfidence: 85\nConfidence: 90",
    ),
)
def test_joint_verbalized_parser_rejects_partial_prose_or_conflicts(
    text: str,
) -> None:
    assert parse_verbalized_output(text) == (None, None)


def test_verbalized_generation_paths_use_the_joint_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _SamplingParams:
        def __init__(self, **_kwargs: object) -> None:
            pass

    class _Completion:
        def __init__(self, text: str) -> None:
            self.text = text

    class _RequestOutput:
        def __init__(self, text: str) -> None:
            self.outputs = [_Completion(text)]

    class _Model:
        responses: list[str] = []

        def generate(
            self,
            prompts: list[str],
            _sampling_params: object,
            *,
            use_tqdm: bool,
        ) -> list[_RequestOutput]:
            assert use_tqdm is False
            assert len(prompts) == len(self.responses)
            return [_RequestOutput(text) for text in self.responses]

    class _Profile:
        stop_token_texts: tuple[str, ...] = ()

        @staticmethod
        def prepare_legacy_prompt(prompt: str) -> str:
            return prompt

    monkeypatch.setattr(position_bias, "SamplingParams", _SamplingParams)
    judge = QwenJudge.__new__(QwenJudge)
    judge.model = _Model()
    judge.profile = _Profile()

    judge.model.responses = ["A\n70% of annotators preferred answer B."]
    assert judge.verbalize_confidence("prompt") == (
        None,
        "A\n70% of annotators preferred answer B.",
        None,
    )

    judge.model.responses = ["1: B\n2: 88", "Answer A: 95"]
    assert judge.verbalize_confidence_batch(["one", "two"]) == [
        (VerdictLabel.B, "1: B\n2: 88", 88.0),
        (None, "Answer A: 95", None),
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    (
        ("A\nConfidence: 87", 87.0),
        ("B\nconfidence = 100", 100.0),
        ("A\n\n85", 85.0),
        ("[[B]]\n72.5", 72.5),
        ("Verdict: T\n0", 0.0),
        ("A\nConfidence: 99.5%", 99.5),
        ("A\nConfidence: 85\nextra explanation", 85.0),
    ),
)
def test_parse_confidence_accepts_explicit_or_bare_post_verdict_values(
    text: str,
    expected: float,
) -> None:
    assert QwenJudge._parse_confidence(text) == expected


@pytest.mark.parametrize(
    "text",
    (
        "Confidence: 80",
        "Answer A had score 87",
        "A\nThere were 2 candidates and I am 85 percent confident",
        "A\nReasoning\n85",
        "A\nReasoning\nConfidence: 85",
        "A\n101",
        "A\n-1",
        "A\n85 points",
        "A\n85\n90",
        "A\nConfidence: 101",
        "A\nConfidence: -1",
        "A\nconfidence is 85",
        "A\nConfidence: 85\nConfidence: 85",
        "A\nConfidence: 85\nConfidence: 90",
    ),
)
def test_parse_confidence_rejects_arbitrary_or_ambiguous_numbers(text: str) -> None:
    assert QwenJudge._parse_confidence(text) is None
