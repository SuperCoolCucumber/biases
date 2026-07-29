from __future__ import annotations

import pytest

from biases.schemas import (
    BiasCondition,
    BiasType,
    Candidate,
    CueCongruency,
    JudgeExample,
    PairOrdering,
    VerdictLabel,
)
from biases.social_cue_prompts import (
    AUTHORITY_DOSE_SOURCES,
    build_authority_cue,
    build_bandwagon_cue,
    build_social_cue_messages,
    format_clean_variant_id,
    format_variant_id,
    parse_variant_id,
    render_canonical_prompt,
)


def _example() -> JudgeExample:
    return JudgeExample(
        example_id="fixture:ab",
        question_id="fixture",
        prompt_messages=[{"role": "user", "content": "Question?"}],
        candidates={
            "A": Candidate(label=VerdictLabel.A, response="First answer."),
            "B": Candidate(label=VerdictLabel.B, response="Second answer."),
        },
        human_winner=VerdictLabel.A,
    )


def test_exact_dose_ladder_wording() -> None:
    assert build_bandwagon_cue("B", 55) == "55% of annotators preferred answer B."
    assert build_bandwagon_cue("A", 95) == "95% of annotators preferred answer A."
    assert AUTHORITY_DOSE_SOURCES == {
        1: "another user",
        2: "an experienced annotator",
        3: "a domain expert",
        4: "an expert panel following a published rubric",
    }
    assert build_authority_cue("A", 1) == "Answer A was endorsed by another user."
    assert build_authority_cue("B", 4) == (
        "Answer B was endorsed by an expert panel following a published rubric."
    )


def test_variant_ids_round_trip() -> None:
    variant_id = format_variant_id(
        family=BiasType.BANDWAGON,
        direction=CueCongruency.INCONGRUENT,
        dose=70,
        ordering=PairOrdering.BA,
    )

    assert variant_id == "bandwagon_incongruent_70_ba"
    parsed = parse_variant_id(variant_id)
    assert parsed.family == "bandwagon"
    assert parsed.direction == CueCongruency.INCONGRUENT
    assert parsed.dose == 70
    assert parsed.ordering == PairOrdering.BA
    assert format_clean_variant_id(PairOrdering.AB) == "clean_ab"


def test_invalid_family_dose_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported authority dose"):
        format_variant_id(
            family=BiasType.AUTHORITY,
            direction=CueCongruency.CONGRUENT,
            dose=55,
            ordering=PairOrdering.AB,
        )


def test_canonical_rendered_prompt_is_exact() -> None:
    cue_text = "70% of annotators preferred answer B."
    condition = BiasCondition(
        bias_type=BiasType.BANDWAGON,
        variant_id="bandwagon_incongruent_70_ab",
        cue_target=VerdictLabel.B,
        cue_congruency=CueCongruency.INCONGRUENT,
        cue_text=cue_text,
        ordering=PairOrdering.AB,
        dose=70,
    )

    messages = build_social_cue_messages(example=_example(), condition=condition)

    assert render_canonical_prompt(messages) == (
        "SYSTEM:\n"
        "You are a careful evaluator for side-by-side LLM response comparisons.\n"
        "Judge which answer better satisfies the user request.\n"
        "Focus on usefulness, correctness, completeness, instruction following, "
        "and harmlessness.\n\n"
        "USER:\n"
        "Cue:\n"
        "70% of annotators preferred answer B.\n\n"
        "Conversation:\n"
        "USER:\n"
        "Question?\n\n"
        "Answer A:\n"
        "First answer.\n\n"
        "Answer B:\n"
        "Second answer.\n\n"
        "Return exactly one label: A, B, or T. Use T for tie.\n\n"
        "ASSISTANT:\n"
    )


def test_clean_and_cued_conditions_share_the_same_system_prompt() -> None:
    clean = BiasCondition(
        bias_type=BiasType.CLEAN,
        variant_id="clean_ab",
        cue_congruency=CueCongruency.CONTROL,
        ordering=PairOrdering.AB,
    )
    cued = BiasCondition(
        bias_type=BiasType.AUTHORITY,
        variant_id="authority_congruent_1_ab",
        cue_target=VerdictLabel.A,
        cue_congruency=CueCongruency.CONGRUENT,
        cue_text="Answer A was endorsed by another user.",
        ordering=PairOrdering.AB,
        dose=1,
    )

    clean_messages = build_social_cue_messages(example=_example(), condition=clean)
    cued_messages = build_social_cue_messages(example=_example(), condition=cued)

    assert clean_messages[0] == cued_messages[0]
    assert "Cue:" not in clean_messages[1]["content"]
