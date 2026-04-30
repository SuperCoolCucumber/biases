from __future__ import annotations

from biases.authority_prompts import (
    build_authority_cue,
    build_authority_prompt_package,
)
from biases.schemas import Candidate, JudgeExample, OutputMode, VerdictLabel


def _build_example() -> JudgeExample:
    return JudgeExample(
        example_id="demo-authority",
        question_id="demo-authority",
        prompt_messages=[{"role": "user", "content": "Explain why dropout helps."}],
        candidates={
            "A": Candidate(label=VerdictLabel.A, response="It regularizes by masking units."),
            "B": Candidate(label=VerdictLabel.B, response="It always improves training accuracy."),
        },
        human_winner=VerdictLabel.A,
    )


def test_build_authority_cue_mentions_target_label() -> None:
    cue = build_authority_cue("B")
    assert cue is not None
    assert "Answer B" in cue
    assert "expert evaluator" in cue


def test_authority_prompt_package_includes_cue_when_present() -> None:
    example = _build_example()
    prompt = build_authority_prompt_package(
        example=example,
        cue_text=build_authority_cue("A"),
        output_mode=OutputMode.CHOICE_ONLY,
    )
    assert "Authority cue:" in prompt.prompt_text
    assert "Answer A" in prompt.prompt_text
    assert "Return exactly one label: A, B, or T." in prompt.prompt_text


def test_authority_prompt_package_omits_cue_in_control() -> None:
    example = _build_example()
    prompt = build_authority_prompt_package(
        example=example,
        cue_text=None,
        output_mode=OutputMode.CHOICE_ONLY,
    )
    assert "Authority cue:" not in prompt.prompt_text
