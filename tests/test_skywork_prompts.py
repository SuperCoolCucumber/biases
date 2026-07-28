from __future__ import annotations

from biases.schemas import Candidate, JudgeExample, VerdictLabel
from biases.skywork_prompts import build_skywork_pairwise_prompt


def test_skywork_prompt_contains_pairwise_fields() -> None:
    example = JudgeExample(
        example_id="x",
        question_id="x",
        prompt_messages=[{"role": "user", "content": "Question?"}],
        candidates={
            "A": Candidate(label=VerdictLabel.A, response="Answer A"),
            "B": Candidate(label=VerdictLabel.B, response="Answer B"),
        },
    )

    prompt = build_skywork_pairwise_prompt(example)

    assert "Assistant A" in prompt.prompt_text
    assert "Assistant B" in prompt.prompt_text
    assert "Return exactly one label" in prompt.prompt_text
