from __future__ import annotations

import pytest

from biases import position_bias
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
    assert {
        "qwen3-4b",
        "qwen3-14b",
        "phi4-14b",
        "hermes3-llama3.1-8b",
    }.issubset(names)
    assert "olmo2-7b-instruct" in names
    assert "gemma2-9b-it" in names
    assert "mistral-7b-instruct-v0.3" in names


def test_full_run_models_pin_hugging_face_revisions() -> None:
    model_names = (
        "qwen3-4b",
        "qwen3-14b",
        "mistral-7b-instruct-v0.3",
        "skywork-critic-8b",
        "hermes3-llama3.1-8b",
        "olmo2-7b-instruct",
        "phi4-14b",
    )
    for model_name in model_names:
        revision = get_model_profile(model_name).revision
        assert revision is not None
        assert len(revision) == 40


def test_olmo2_profile_uses_the_pinned_text_prompt_contract() -> None:
    profile = get_model_profile("olmo2-7b-instruct")

    assert profile.hf_model_name == "allenai/OLMo-2-1124-7B-Instruct"
    assert profile.family == "olmo2"
    assert profile.revision == "470b1fba1ae01581f270116362ee4aa1b97f4c84"
    assert profile.assistant_prefill == ""
    assert profile.stop_token_texts == ("<|endoftext|>",)
    assert profile.supports_system_role is True
    assert profile.supports_text_prompt_transport is True


def test_phi4_profile_uses_the_pinned_text_prompt_contract() -> None:
    profile = get_model_profile("phi4-14b")

    assert profile.hf_model_name == "microsoft/phi-4"
    assert profile.family == "phi4"
    assert profile.revision == "2db69c1c3e91a05d2c64a3185acfbaf36f744e25"
    assert profile.stop_token_texts == ("<|im_end|>", "<|endoftext|>")
    assert profile.supports_system_role is True
    assert profile.supports_text_prompt_transport is True


def test_vllm_judge_passes_pinned_model_and_tokenizer_revisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _DecisionTokenizer:
        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            assert add_special_tokens is False
            return [ord(text.strip())]

    class _FakeLLM:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def get_tokenizer(self) -> _DecisionTokenizer:
            return _DecisionTokenizer()

    monkeypatch.setattr(position_bias, "LLM", _FakeLLM)
    monkeypatch.setattr(position_bias, "SamplingParams", object())

    profile = get_model_profile("qwen3-14b")
    position_bias.VLLMJudge("qwen3-14b")

    assert captured["revision"] == profile.revision
    assert captured["tokenizer_revision"] == profile.revision
    assert captured["logprobs_mode"] == "processed_logprobs"
    assert "max_num_batched_tokens" not in captured
    assert "max_num_seqs" not in captured

    captured.clear()
    judge = position_bias.VLLMJudge(
        "qwen3-14b",
        max_num_batched_tokens=32768,
        max_num_seqs=128,
    )

    assert captured["max_num_batched_tokens"] == 32768
    assert captured["max_num_seqs"] == 128
    assert captured["logprobs_mode"] == "processed_logprobs"
    assert judge.logprobs_mode == "processed_logprobs"
    assert judge.max_num_batched_tokens == 32768
    assert judge.max_num_seqs == 128


def test_mistral_fails_closed_until_token_prompt_transport_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeLLM:
        def __init__(self, **kwargs: object) -> None:
            raise AssertionError(f"engine must not start: {kwargs!r}")

    monkeypatch.setattr(position_bias, "LLM", _FakeLLM)
    monkeypatch.setattr(position_bias, "SamplingParams", object())

    with pytest.raises(RuntimeError, match="requires a token-ID prompt adapter"):
        position_bias.VLLMJudge("mistral-7b-instruct-v0.3")


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
        "NousResearch/Hermes-3-Llama-3.1-8B",
        "allenai/OLMo-2-1124-7B-Instruct",
        "microsoft/phi-4",
    }
    assert {get_model_profile(name).hf_model_name for name in model_names} == model_names
