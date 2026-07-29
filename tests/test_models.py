from __future__ import annotations

import pytest

from biases.models import available_model_names, get_model_profile


class _Tokenizer:
    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        self.messages = messages
        return "rendered:"


def test_qwen3_prefill_is_model_specific() -> None:
    qwen3 = get_model_profile("Qwen/Qwen3-14B")
    qwen25 = get_model_profile("Qwen/Qwen2.5-14B-Instruct")

    assert qwen3.prepare_legacy_prompt("<|im_start|>assistant\n").endswith(
        "<think>\n\n</think>\n\n"
    )
    assert (
        qwen25.prepare_legacy_prompt("<|im_start|>assistant\n")
        == "<|im_start|>assistant\n"
    )


def test_gemma_folds_system_text_into_user_turn() -> None:
    tokenizer = _Tokenizer()
    profile = get_model_profile("gemma2-9b-it")

    rendered = profile.render_prompt(
        tokenizer,
        [
            {"role": "system", "content": "Judge carefully."},
            {"role": "user", "content": "Choose A or B."},
        ],
    )

    assert rendered == "rendered:"
    assert tokenizer.messages == [
        {
            "role": "user",
            "content": "Judge carefully.\n\nChoose A or B.",
        }
    ]


def test_registry_rejects_unknown_model() -> None:
    with pytest.raises(KeyError, match="Register its chat template"):
        get_model_profile("organization/unregistered-model")


def test_required_paper_model_families_are_registered() -> None:
    names = set(available_model_names())
    assert {"qwen3-4b", "qwen3-14b", "gemma2-9b-it"}.issubset(names)
    assert "mistral-7b-instruct-v0.3" in names


def test_existing_slurm_model_names_remain_registered() -> None:
    model_names = {
        "Qwen/Qwen3.5-4B",
        "Qwen/Qwen3.5-9B",
        "Qwen/Qwen3.5-27B",
        "Qwen/Qwen3-14B",
        "Qwen/Qwen3-32B",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "google/gemma-2-27b-it",
        "Skywork/Skywork-Critic-Llama-3.1-8B",
    }
    assert {get_model_profile(name).hf_model_name for name in model_names} == model_names
