from __future__ import annotations

import importlib.util
import sys
import types
from argparse import Namespace
from pathlib import Path

import pytest

from biases.models import get_model_profile


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "validate_model_tokenizer_contract.py"
)
SPEC = importlib.util.spec_from_file_location(
    "validate_model_tokenizer_contract",
    SCRIPT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
LLAMA_PROFILE = get_model_profile("llama3.3-70b-instruct")
assert LLAMA_PROFILE.revision is not None


class _Config:
    model_type = "llama"
    _commit_hash = LLAMA_PROFILE.revision


class _Tokenizer:
    eos_token_id = 4
    init_kwargs = {"_commit_hash": LLAMA_PROFILE.revision}

    _surface_ids = {
        "A": [1],
        "B": [2],
        "T": [3],
        "<|eot_id|>": [4],
        "rendered": [10, 11],
    }
    _decoded = {
        1: "A",
        2: "B",
        3: "T",
        4: "<|eot_id|>",
    }

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return list(self._surface_ids[text])

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return self._decoded[token_ids[0]]

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str | list[int]:
        assert messages
        assert add_generation_prompt is True
        return [10, 11] if tokenize else "rendered"


def test_validate_loaded_llama_contract_records_exact_token_transport() -> None:
    profile = get_model_profile("llama3.3-70b-instruct")

    payload = MODULE.validate_loaded_tokenizer_contract(
        profile,
        _Tokenizer(),
        _Config(),
    )

    assert payload["passed"] is True
    assert payload["resolved_verdict_token_ids"] == {
        "A": [1],
        "B": [2],
        "tie": [3],
    }
    assert payload["resolved_stop_token_ids"] == {"<|eot_id|>": 4}
    assert payload["chat_transport_match"] is True
    assert payload["rendered_prompt_token_count"] == 2
    assert payload["schema_version"] == 1
    assert payload["trust_remote_code"] is False
    assert set(payload["source_hashes"]) == {
        "models_source_sha256",
        "tokenizer_probe_sha256",
    }


def test_validate_loaded_contract_rejects_multitoken_verdict_surface() -> None:
    class _InvalidTokenizer(_Tokenizer):
        _surface_ids = {**_Tokenizer._surface_ids, "A": [1, 5]}

    profile = get_model_profile("llama3.3-70b-instruct")

    with pytest.raises(ValueError, match="exactly one token"):
        MODULE.validate_loaded_tokenizer_contract(
            profile,
            _InvalidTokenizer(),
            _Config(),
        )


def test_validate_loaded_contract_records_matching_resolved_revisions() -> None:
    profile = get_model_profile("llama3.3-70b-instruct")

    class _PinnedConfig(_Config):
        _commit_hash = profile.revision

    class _PinnedTokenizer(_Tokenizer):
        init_kwargs = {"_commit_hash": profile.revision}

    payload = MODULE.validate_loaded_tokenizer_contract(
        profile,
        _PinnedTokenizer(),
        _PinnedConfig(),
    )

    assert payload["resolved_config_commit_hash"] == profile.revision
    assert payload["resolved_tokenizer_commit_hash"] == profile.revision


def test_validate_loaded_contract_rejects_mismatched_resolved_revision() -> None:
    profile = get_model_profile("llama3.3-70b-instruct")

    class _WrongConfig(_Config):
        _commit_hash = "0" * 40

    with pytest.raises(ValueError, match="does not match the pinned model revision"):
        MODULE.validate_loaded_tokenizer_contract(
            profile,
            _Tokenizer(),
            _WrongConfig(),
        )


def _assert_missing_resolved_revision(
    *,
    model_name: str,
    component: str,
) -> None:
    profile = get_model_profile(model_name)
    assert profile.revision is not None

    class _PinnedConfig(_Config):
        _commit_hash = profile.revision

    class _PinnedTokenizer(_Tokenizer):
        init_kwargs = {"_commit_hash": profile.revision}

    if component == "config":
        class _MissingConfig(_PinnedConfig):
            _commit_hash = None

        config = _MissingConfig()
        tokenizer = _PinnedTokenizer()
    else:
        class _MissingTokenizer(_PinnedTokenizer):
            init_kwargs: dict[str, str] = {}

        config = _PinnedConfig()
        tokenizer = _MissingTokenizer()

    with pytest.raises(
        ValueError,
        match=rf"Resolved {component} revision is absent.*model revision.*is pinned",
    ):
        MODULE.validate_loaded_tokenizer_contract(profile, tokenizer, config)


def test_qwen32b_contract_rejects_absent_resolved_config_revision() -> None:
    _assert_missing_resolved_revision(
        model_name="qwen2.5-32b",
        component="config",
    )


def test_qwen32b_contract_rejects_absent_resolved_tokenizer_revision() -> None:
    _assert_missing_resolved_revision(
        model_name="qwen2.5-32b",
        component="tokenizer",
    )


def test_llama70b_contract_rejects_absent_resolved_config_revision() -> None:
    _assert_missing_resolved_revision(
        model_name="llama3.3-70b-instruct",
        component="config",
    )


def test_llama70b_contract_rejects_absent_resolved_tokenizer_revision() -> None:
    _assert_missing_resolved_revision(
        model_name="llama3.3-70b-instruct",
        component="tokenizer",
    )


def test_sanitize_exception_text_redacts_urls_and_auth_material() -> None:
    raw = (
        "download https://huggingface.co/model?X-Amz-Signature=secret&token=oops "
        "Authorization: Bearer hf_abcdefghijklmnopqrstuvwxyz123456 "
        "access_token=anothersecret"
    )

    sanitized = MODULE.sanitize_exception_text(RuntimeError(raw))

    assert "secret" not in sanitized
    assert "oops" not in sanitized
    assert "abcdefghijklmnopqrstuvwxyz" not in sanitized
    assert "anothersecret" not in sanitized
    assert "https://huggingface.co/model" in sanitized
    assert "<redacted>" in sanitized


def test_run_probe_honors_model_specific_remote_code_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []

    class _AutoConfig:
        @staticmethod
        def from_pretrained(
            model_name: str,
            **kwargs: object,
        ) -> _Config:
            assert model_name == "meta-llama/Llama-3.3-70B-Instruct"
            calls.append(dict(kwargs))
            return _Config()

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(
            model_name: str,
            **kwargs: object,
        ) -> _Tokenizer:
            assert model_name == "meta-llama/Llama-3.3-70B-Instruct"
            calls.append(dict(kwargs))
            return _Tokenizer()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoConfig=_AutoConfig,
            AutoTokenizer=_AutoTokenizer,
        ),
    )
    monkeypatch.setattr(MODULE.importlib.metadata, "version", lambda _: "test")

    payload = MODULE.run_probe(
        Namespace(
            model_name="llama3.3-70b-instruct",
            cache_dir=tmp_path,
            allow_download=False,
            require_authentication=True,
        )
    )

    assert payload["passed"] is True
    assert len(calls) == 2
    assert all(call["trust_remote_code"] is False for call in calls)
    assert all(call["token"] is True for call in calls)
