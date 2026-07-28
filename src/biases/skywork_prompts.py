from __future__ import annotations

from biases.schemas import JudgeExample, OutputMode, PromptPackage, VerdictLabel
from biases.utils import stable_hash


SKYWORK_SYSTEM_MESSAGE = (
    "You are a fair pairwise response evaluator. Compare the two assistant "
    "responses for helpfulness, correctness, completeness, instruction "
    "following, and safety. Return only A, B, or T."
)


def build_skywork_pairwise_prompt(
    example: JudgeExample,
    *,
    output_mode: OutputMode = OutputMode.CHOICE_ONLY,
) -> PromptPackage:
    conversation = "\n".join(
        f"{message.get('role', 'user')}: {message.get('content', '')}"
        for message in example.prompt_messages
    )
    prompt_text = (
        f"<s>[INST] <<SYS>>\n{SKYWORK_SYSTEM_MESSAGE}\n<</SYS>>\n\n"
        f"[User Conversation]\n{conversation}\n\n"
        f"[Assistant A]\n{example.candidates['A'].response}\n\n"
        f"[Assistant B]\n{example.candidates['B'].response}\n\n"
        "Which assistant response is better? Return exactly one label: A, B, or T. [/INST]"
    )
    return PromptPackage(
        prompt_text=prompt_text,
        output_mode=output_mode,
        allowed_labels=[VerdictLabel.A, VerdictLabel.B, VerdictLabel.TIE],
        prompt_hash=stable_hash(
            {
                "template": "skywork_pairwise_v1",
                "example_id": example.example_id,
                "output_mode": output_mode,
                "prompt_text": prompt_text,
            }
        ),
        metadata={"template": "skywork_pairwise_v1"},
    )
