from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from biases.pairing import normalize_ordering, normalize_verdict
from biases.schemas import (
    BiasCondition,
    BiasType,
    CueCongruency,
    JudgeExample,
    OutputMode,
    PairOrdering,
    PromptPackage,
    VerdictLabel,
)
from biases.utils import stable_hash


BANDWAGON_DOSES: tuple[int, ...] = (55, 70, 85, 95)
AUTHORITY_DOSE_SOURCES: dict[int, str] = {
    1: "another user",
    2: "an experienced annotator",
    3: "a domain expert",
    4: "an expert panel following a published rubric",
}
AUTHORITY_DOSES: tuple[int, ...] = tuple(AUTHORITY_DOSE_SOURCES)

SOCIAL_CUE_TEMPLATE_VERSION = "silent_bias_social_cues_v1"
SOCIAL_CUE_SYSTEM_PROMPT = (
    "You are a careful evaluator for side-by-side LLM response comparisons.\n"
    "Judge which answer better satisfies the user request.\n"
    "Focus on usefulness, correctness, completeness, instruction following, "
    "and harmlessness."
)

_VARIANT_PATTERN = re.compile(
    r"^(authority|bandwagon)_(congruent|incongruent)_(\d+)_(ab|ba)$"
)

PromptRenderer = Callable[[list[dict[str, str]]], str]


@dataclass(frozen=True)
class ParsedVariantId:
    family: str
    direction: CueCongruency | None
    dose: int | None
    ordering: PairOrdering


def _enum_value(value: str | Enum) -> str:
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


def _normalize_family(family: BiasType | str) -> str:
    normalized = _enum_value(family).strip().lower()
    if normalized not in {BiasType.AUTHORITY.value, BiasType.BANDWAGON.value}:
        raise ValueError("family must be authority or bandwagon")
    return normalized


def _normalize_direction(direction: CueCongruency | str) -> CueCongruency:
    if isinstance(direction, CueCongruency):
        normalized = direction
    else:
        try:
            normalized = CueCongruency(str(direction).strip().lower())
        except ValueError as exc:
            raise ValueError("direction must be congruent or incongruent") from exc
    if normalized not in {CueCongruency.CONGRUENT, CueCongruency.INCONGRUENT}:
        raise ValueError("direction must be congruent or incongruent")
    return normalized


def format_clean_variant_id(ordering: PairOrdering | str) -> str:
    return f"clean_{normalize_ordering(ordering).value}"


def format_variant_id(
    *,
    family: BiasType | str,
    direction: CueCongruency | str,
    dose: int,
    ordering: PairOrdering | str,
) -> str:
    normalized_family = _normalize_family(family)
    normalized_direction = _normalize_direction(direction)
    validate_dose(normalized_family, dose)
    return (
        f"{normalized_family}_{normalized_direction.value}_{dose}_"
        f"{normalize_ordering(ordering).value}"
    )


def parse_variant_id(variant_id: str) -> ParsedVariantId:
    clean_prefix = "clean_"
    if variant_id.startswith(clean_prefix):
        return ParsedVariantId(
            family=BiasType.CLEAN.value,
            direction=None,
            dose=None,
            ordering=normalize_ordering(variant_id.removeprefix(clean_prefix)),
        )

    match = _VARIANT_PATTERN.fullmatch(variant_id)
    if match is None:
        raise ValueError(f"Unsupported Silent Bias variant ID {variant_id!r}")
    family, direction, raw_dose, ordering = match.groups()
    dose = int(raw_dose)
    validate_dose(family, dose)
    return ParsedVariantId(
        family=family,
        direction=_normalize_direction(direction),
        dose=dose,
        ordering=normalize_ordering(ordering),
    )


def validate_dose(family: BiasType | str, dose: int) -> None:
    normalized_family = _normalize_family(family)
    allowed = BANDWAGON_DOSES if normalized_family == BiasType.BANDWAGON.value else AUTHORITY_DOSES
    if dose not in allowed:
        raise ValueError(
            f"Unsupported {normalized_family} dose {dose}; expected one of {allowed}"
        )


def build_bandwagon_cue(target: VerdictLabel | str, dose: int) -> str:
    validate_dose(BiasType.BANDWAGON, dose)
    label = normalize_verdict(target)
    if label not in {VerdictLabel.A, VerdictLabel.B}:
        raise ValueError("bandwagon cue target must be A or B")
    return f"{dose}% of annotators preferred answer {label.value}."


def build_authority_cue(target: VerdictLabel | str, dose: int) -> str:
    validate_dose(BiasType.AUTHORITY, dose)
    label = normalize_verdict(target)
    if label not in {VerdictLabel.A, VerdictLabel.B}:
        raise ValueError("authority cue target must be A or B")
    return f"Answer {label.value} was endorsed by {AUTHORITY_DOSE_SOURCES[dose]}."


def build_social_cue(
    *,
    family: BiasType | str,
    target: VerdictLabel | str,
    dose: int,
) -> str:
    normalized_family = _normalize_family(family)
    if normalized_family == BiasType.BANDWAGON.value:
        return build_bandwagon_cue(target, dose)
    return build_authority_cue(target, dose)


def render_source_messages(messages: list[dict[str, str]]) -> str:
    return "\n\n".join(
        f"{message.get('role', 'user').strip().upper()}:\n"
        f"{message.get('content', '').strip()}"
        for message in messages
    )


def build_social_cue_messages(
    *,
    example: JudgeExample,
    condition: BiasCondition,
    output_mode: OutputMode = OutputMode.CHOICE_ONLY,
) -> list[dict[str, str]]:
    family = _enum_value(condition.bias_type).lower()
    cue_text: str | None
    if family == BiasType.CLEAN.value:
        cue_text = None
    else:
        if condition.cue_target is None or condition.dose is None:
            raise ValueError("cued conditions require cue_target and dose")
        cue_text = build_social_cue(
            family=family,
            target=condition.cue_target,
            dose=condition.dose,
        )
        if condition.cue_text is not None and condition.cue_text != cue_text:
            raise ValueError("condition.cue_text does not match the centralized dose template")

    if output_mode == OutputMode.CHOICE_WITH_CONFIDENCE:
        output_instruction = (
            "Return exactly two lines.\n"
            "Line 1: one label only: A, B, or T\n"
            "Line 2: Confidence: <integer from 0 to 100>"
        )
    else:
        output_instruction = "Return exactly one label: A, B, or T. Use T for tie."

    cue_block = f"Cue:\n{cue_text}\n\n" if cue_text is not None else ""
    user_content = (
        f"{cue_block}"
        f"Conversation:\n{render_source_messages(example.prompt_messages)}\n\n"
        f"Answer A:\n{example.candidates['A'].response.strip()}\n\n"
        f"Answer B:\n{example.candidates['B'].response.strip()}\n\n"
        f"{output_instruction}"
    )
    return [
        {"role": "system", "content": SOCIAL_CUE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def render_canonical_prompt(messages: list[dict[str, str]]) -> str:
    rendered = render_source_messages(messages)
    return f"{rendered}\n\nASSISTANT:\n"


def build_social_cue_prompt_package(
    *,
    example: JudgeExample,
    condition: BiasCondition,
    output_mode: OutputMode = OutputMode.CHOICE_ONLY,
    renderer: PromptRenderer = render_canonical_prompt,
) -> PromptPackage:
    family = _enum_value(condition.bias_type).lower()
    resolved_cue_text = condition.cue_text
    if (
        family != BiasType.CLEAN.value
        and condition.cue_target is not None
        and condition.dose is not None
    ):
        resolved_cue_text = build_social_cue(
            family=family,
            target=condition.cue_target,
            dose=condition.dose,
        )
    messages = build_social_cue_messages(
        example=example,
        condition=condition,
        output_mode=output_mode,
    )
    prompt_text = renderer(messages)
    return PromptPackage(
        prompt_text=prompt_text,
        output_mode=output_mode,
        allowed_labels=[VerdictLabel.A, VerdictLabel.B, VerdictLabel.TIE],
        prompt_hash=stable_hash(
            {
                "template_version": SOCIAL_CUE_TEMPLATE_VERSION,
                "variant_id": condition.variant_id,
                "output_mode": output_mode,
                "prompt_text": prompt_text,
            }
        ),
        metadata={
            "prompt_family": SOCIAL_CUE_TEMPLATE_VERSION,
            "cue_text": resolved_cue_text,
        },
    )


__all__ = [
    "AUTHORITY_DOSES",
    "AUTHORITY_DOSE_SOURCES",
    "BANDWAGON_DOSES",
    "ParsedVariantId",
    "SOCIAL_CUE_SYSTEM_PROMPT",
    "SOCIAL_CUE_TEMPLATE_VERSION",
    "build_authority_cue",
    "build_bandwagon_cue",
    "build_social_cue",
    "build_social_cue_messages",
    "build_social_cue_prompt_package",
    "format_clean_variant_id",
    "format_variant_id",
    "parse_variant_id",
    "render_canonical_prompt",
    "render_source_messages",
    "validate_dose",
]
