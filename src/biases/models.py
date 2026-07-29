from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence


class ChatTemplatePolicy(str, Enum):
    TOKENIZER = "tokenizer"


DEFAULT_VERDICT_TOKEN_TEXTS: Mapping[str, tuple[str, ...]] = {
    "A": ("A", " A"),
    "B": ("B", " B"),
    "tie": ("T", " T"),
}


@dataclass(frozen=True, slots=True)
class ModelProfile:
    registry_name: str
    hf_model_name: str
    family: str
    revision: str | None = None
    chat_template: ChatTemplatePolicy = ChatTemplatePolicy.TOKENIZER
    verdict_token_texts: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: DEFAULT_VERDICT_TOKEN_TEXTS
    )
    assistant_prefill: str = ""
    stop_token_texts: tuple[str, ...] = ()
    supports_system_role: bool = True
    supports_text_prompt_transport: bool = True

    def normalize_messages(
        self,
        messages: Sequence[Mapping[str, str]],
    ) -> list[dict[str, str]]:
        normalized = [
            {
                "role": str(message.get("role", "user")),
                "content": str(message.get("content", "")),
            }
            for message in messages
        ]
        if self.supports_system_role:
            return normalized

        system_parts = [
            message["content"] for message in normalized if message["role"] == "system"
        ]
        non_system = [
            message for message in normalized if message["role"] != "system"
        ]
        if not system_parts:
            return non_system

        system_text = "\n\n".join(system_parts)
        if non_system and non_system[0]["role"] == "user":
            first = dict(non_system[0])
            first["content"] = f"{system_text}\n\n{first['content']}"
            return [first, *non_system[1:]]
        return [{"role": "user", "content": system_text}, *non_system]

    def render_prompt(
        self,
        tokenizer: Any,
        messages: Sequence[Mapping[str, str]],
    ) -> str:
        if self.chat_template != ChatTemplatePolicy.TOKENIZER:
            raise ValueError(f"Unsupported chat-template policy: {self.chat_template}")
        if not hasattr(tokenizer, "apply_chat_template"):
            raise RuntimeError(
                f"Tokenizer for {self.registry_name!r} has no apply_chat_template method."
            )
        rendered = tokenizer.apply_chat_template(
            self.normalize_messages(messages),
            tokenize=False,
            add_generation_prompt=True,
        )
        if not isinstance(rendered, str):
            raise TypeError("Tokenizer chat template did not return text.")
        return f"{rendered}{self.assistant_prefill}"

    def prepare_legacy_prompt(self, prompt_text: str) -> str:
        if not self.assistant_prefill or prompt_text.endswith(self.assistant_prefill):
            return prompt_text
        if self.family.startswith("qwen3") and prompt_text.endswith(
            "<|im_start|>assistant\n"
        ):
            return f"{prompt_text}{self.assistant_prefill}"
        return prompt_text


MODEL_REGISTRY: Mapping[str, ModelProfile] = {
    "qwen2.5-14b": ModelProfile(
        registry_name="qwen2.5-14b",
        hf_model_name="Qwen/Qwen2.5-14B-Instruct",
        family="qwen2.5",
        stop_token_texts=("<|im_end|>",),
    ),
    "qwen3-4b": ModelProfile(
        registry_name="qwen3-4b",
        hf_model_name="Qwen/Qwen3-4B",
        family="qwen3",
        revision="1cfa9a7208912126459214e8b04321603b3df60c",
        assistant_prefill="<think>\n\n</think>\n\n",
        stop_token_texts=("<|im_end|>",),
    ),
    "qwen3-14b": ModelProfile(
        registry_name="qwen3-14b",
        hf_model_name="Qwen/Qwen3-14B",
        family="qwen3",
        revision="40c069824f4251a91eefaf281ebe4c544efd3e18",
        assistant_prefill="<think>\n\n</think>\n\n",
        stop_token_texts=("<|im_end|>",),
    ),
    "qwen3-32b": ModelProfile(
        registry_name="qwen3-32b",
        hf_model_name="Qwen/Qwen3-32B",
        family="qwen3",
        assistant_prefill="<think>\n\n</think>\n\n",
        stop_token_texts=("<|im_end|>",),
    ),
    "qwen3.5-4b": ModelProfile(
        registry_name="qwen3.5-4b",
        hf_model_name="Qwen/Qwen3.5-4B",
        family="qwen3.5",
        assistant_prefill="<think>\n\n</think>\n\n",
        stop_token_texts=("<|im_end|>",),
    ),
    "qwen3.5-9b": ModelProfile(
        registry_name="qwen3.5-9b",
        hf_model_name="Qwen/Qwen3.5-9B",
        family="qwen3.5",
        assistant_prefill="<think>\n\n</think>\n\n",
        stop_token_texts=("<|im_end|>",),
    ),
    "qwen3.5-27b": ModelProfile(
        registry_name="qwen3.5-27b",
        hf_model_name="Qwen/Qwen3.5-27B",
        family="qwen3.5",
        assistant_prefill="<think>\n\n</think>\n\n",
        stop_token_texts=("<|im_end|>",),
    ),
    "gemma2-9b-it": ModelProfile(
        registry_name="gemma2-9b-it",
        hf_model_name="google/gemma-2-9b-it",
        family="gemma2",
        stop_token_texts=("<end_of_turn>",),
        supports_system_role=False,
    ),
    "gemma2-27b-it": ModelProfile(
        registry_name="gemma2-27b-it",
        hf_model_name="google/gemma-2-27b-it",
        family="gemma2",
        stop_token_texts=("<end_of_turn>",),
        supports_system_role=False,
    ),
    "gemma3-12b-it": ModelProfile(
        registry_name="gemma3-12b-it",
        hf_model_name="google/gemma-3-12b-it",
        family="gemma3",
        stop_token_texts=("<end_of_turn>",),
        supports_system_role=False,
    ),
    "mistral-7b-instruct-v0.3": ModelProfile(
        registry_name="mistral-7b-instruct-v0.3",
        hf_model_name="mistralai/Mistral-7B-Instruct-v0.3",
        family="mistral",
        revision="c170c708c41dac9275d15a8fff4eca08d52bab71",
        stop_token_texts=("</s>",),
        supports_text_prompt_transport=False,
    ),
    "skywork-critic-8b": ModelProfile(
        registry_name="skywork-critic-8b",
        hf_model_name="Skywork/Skywork-Critic-Llama-3.1-8B",
        family="llama3",
        revision="825f34599593c0145be91644be233d5c634b2380",
        stop_token_texts=("<|eot_id|>",),
    ),
    "hermes3-llama3.1-8b": ModelProfile(
        registry_name="hermes3-llama3.1-8b",
        hf_model_name="NousResearch/Hermes-3-Llama-3.1-8B",
        family="llama3",
        revision="896ea440e5a9e6070e3d8a2774daf2b481ab425b",
        stop_token_texts=("<|im_end|>",),
    ),
    "olmo2-7b-instruct": ModelProfile(
        registry_name="olmo2-7b-instruct",
        hf_model_name="allenai/OLMo-2-1124-7B-Instruct",
        family="olmo2",
        revision="470b1fba1ae01581f270116362ee4aa1b97f4c84",
        stop_token_texts=("<|endoftext|>",),
    ),
    "olmo3-7b-instruct": ModelProfile(
        registry_name="olmo3-7b-instruct",
        hf_model_name="allenai/Olmo-3-7B-Instruct",
        family="olmo3",
        revision="6e5971d9eba42665f5bd5a0fcf047f299ce1dccc",
        chat_template=ChatTemplatePolicy.TOKENIZER,
        stop_token_texts=("<|im_end|>", "<|endoftext|>"),
        supports_system_role=True,
        supports_text_prompt_transport=True,
    ),
    "phi4-14b": ModelProfile(
        registry_name="phi4-14b",
        hf_model_name="microsoft/phi-4",
        family="phi4",
        revision="2db69c1c3e91a05d2c64a3185acfbaf36f744e25",
        stop_token_texts=("<|im_end|>", "<|endoftext|>"),
    ),
}


def get_model_profile(model_name: str) -> ModelProfile:
    direct = MODEL_REGISTRY.get(model_name)
    if direct is not None:
        return direct

    normalized = model_name.casefold()
    for profile in MODEL_REGISTRY.values():
        if profile.hf_model_name.casefold() == normalized:
            return profile
    raise KeyError(
        f"Unknown model {model_name!r}. Register its chat template, verdict tokens, "
        "prefill, and stop tokens in biases.models before running it."
    )


def available_model_names() -> tuple[str, ...]:
    return tuple(MODEL_REGISTRY)
