from __future__ import annotations

from biases.schemas import JudgeExample, OutputMode, PromptPackage, VerdictLabel
from biases.utils import stable_hash


POSITION_JUDGE_SYSTEM_PROMPT = """
You are a careful evaluator for side-by-side LLM response comparisons.
Judge which answer better satisfies the user request.
Ignore answer position and focus on usefulness, correctness, completeness,
instruction following, and harmlessness.
""".strip()


def render_prompt_messages(messages: list[dict[str, str]]) -> str:
    rendered: list[str] = []
    for message in messages:
        role = message.get("role", "user").strip().upper()
        content = message.get("content", "").strip()
        rendered.append(f"{role}:\n{content}")
    return "\n\n".join(rendered)


def build_position_prompt_text(
    example: JudgeExample,
    output_mode: OutputMode = OutputMode.CHOICE_ONLY,
) -> str:
    answer_a = example.candidates["A"].response.strip()
    answer_b = example.candidates["B"].response.strip()
    conversation = render_prompt_messages(example.prompt_messages)

    if output_mode == OutputMode.CHOICE_WITH_CONFIDENCE:
        instructions = (
            "Return exactly two lines.\n"
            "Line 1: one label only: A, B, or T\n"
            "Line 2: Confidence: <integer from 0 to 100>"
        )
    else:
        instructions = "Return exactly one label: A, B, or T. Use T for tie."

    user_block = (
        f"Conversation:\n{conversation}\n\n"
        f"Answer A:\n{answer_a}\n\n"
        f"Answer B:\n{answer_b}\n\n"
        f"{instructions}"
    )
    return (
        f"<|im_start|>system\n{POSITION_JUDGE_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_block}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def build_position_prompt_package(
    example: JudgeExample,
    output_mode: OutputMode = OutputMode.CHOICE_ONLY,
) -> PromptPackage:
    prompt_text = build_position_prompt_text(example=example, output_mode=output_mode)
    return PromptPackage(
        prompt_text=prompt_text,
        output_mode=output_mode,
        allowed_labels=[VerdictLabel.A, VerdictLabel.B, VerdictLabel.TIE],
        prompt_hash=stable_hash(
            {
                "example_id": example.example_id,
                "output_mode": output_mode,
                "prompt_text": prompt_text,
            }
        ),
        metadata={"prompt_family": "position_bias_v1"},
    )
