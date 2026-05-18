from __future__ import annotations

from biases.bandwagon_prompts import (
    build_bandwagon_cue,
    build_bandwagon_prompt_package,
)
from biases.schemas import Candidate, JudgeExample, OutputMode, VerdictLabel


def _build_example() -> JudgeExample:
    return JudgeExample(
        example_id="demo-bandwagon",
        question_id="demo-bandwagon",
        prompt_messages=[{"role": "user", "content": "Explain why dropout helps."}],
        candidates={
            "A": Candidate(label=VerdictLabel.A, response="It regularizes by masking units."),
            "B": Candidate(label=VerdictLabel.B, response="It always improves training accuracy."),
        },
        human_winner=VerdictLabel.A,
    )


def test_build_bandwagon_cue_mentions_target_label() -> None:
    cue = build_bandwagon_cue("B")
    assert cue is not None
    assert "Answer B" in cue
    assert "85% of users preferred" in cue


def test_bandwagon_prompt_package_includes_cue_when_present() -> None:
    example = _build_example()
    prompt = build_bandwagon_prompt_package(
        example=example,
        cue_text=build_bandwagon_cue("A"),
        output_mode=OutputMode.CHOICE_ONLY,
    )
    assert "Majority cue:" in prompt.prompt_text
    assert "Answer A" in prompt.prompt_text
    assert "Return exactly one label: A, B, or T." in prompt.prompt_text


def test_bandwagon_prompt_package_omits_cue_in_control() -> None:
    example = _build_example()
    prompt = build_bandwagon_prompt_package(
        example=example,
        cue_text=None,
        output_mode=OutputMode.CHOICE_ONLY,
    )
    assert "Majority cue:" not in prompt.prompt_text
