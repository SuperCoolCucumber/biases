from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from biases import models as models_module
from biases.models import ModelProfile, get_model_profile
from biases.safe_diagnostics import sanitize_exception_text


SCHEMA_VERSION = 1

PROBE_MESSAGES: tuple[Mapping[str, str], ...] = (
    {
        "role": "system",
        "content": "You are an impartial pairwise evaluator.",
    },
    {
        "role": "user",
        "content": "Return only A, B, or T.",
    },
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hashes() -> dict[str, str]:
    models_path_value = getattr(models_module, "__file__", None)
    if models_path_value is None:
        raise RuntimeError("Could not resolve the biases.models source path")
    return {
        "tokenizer_probe_sha256": _sha256_file(Path(__file__).resolve()),
        "models_source_sha256": _sha256_file(Path(models_path_value).resolve()),
    }


def _resolved_commit_hash(value: Any) -> str | None:
    direct = getattr(value, "_commit_hash", None)
    if isinstance(direct, str) and direct:
        return direct
    init_kwargs = getattr(value, "init_kwargs", None)
    if isinstance(init_kwargs, Mapping):
        candidate = init_kwargs.get("_commit_hash")
        if isinstance(candidate, str) and candidate:
            return candidate
    return None


def _validate_resolved_revision(
    *,
    component: str,
    resolved_revision: str | None,
    expected_revision: str | None,
) -> None:
    if expected_revision is not None and resolved_revision is None:
        raise ValueError(
            f"Resolved {component} revision is absent, but model revision "
            f"{expected_revision!r} is pinned"
        )
    if (
        resolved_revision is not None
        and expected_revision is not None
        and resolved_revision != expected_revision
    ):
        raise ValueError(
            f"Resolved {component} revision {resolved_revision!r} does not match "
            f"the pinned model revision {expected_revision!r}"
        )


def _sha256_ints(values: Sequence[int]) -> str:
    payload = json.dumps(
        [int(value) for value in values],
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return _sha256_text(payload)


def _single_round_trip_token_id(tokenizer: Any, surface: str) -> int:
    token_ids = tokenizer.encode(surface, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(
            f"Surface {surface!r} must encode to exactly one token; got {token_ids!r}"
        )
    token_id = int(token_ids[0])
    decoded = tokenizer.decode(
        [token_id],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if decoded != surface:
        raise ValueError(
            f"Surface {surface!r} did not round-trip exactly; decoded {decoded!r}"
        )
    return token_id


def validate_loaded_tokenizer_contract(
    profile: ModelProfile,
    tokenizer: Any,
    config: Any,
) -> dict[str, Any]:
    config_commit_hash = _resolved_commit_hash(config)
    tokenizer_commit_hash = _resolved_commit_hash(tokenizer)
    _validate_resolved_revision(
        component="config",
        resolved_revision=config_commit_hash,
        expected_revision=profile.revision,
    )
    _validate_resolved_revision(
        component="tokenizer",
        resolved_revision=tokenizer_commit_hash,
        expected_revision=profile.revision,
    )

    verdict_token_ids: dict[str, list[int]] = {}
    token_id_to_label: dict[int, str] = {}
    for label in ("A", "B", "tie"):
        surfaces = tuple(profile.verdict_token_texts[label])
        ids: list[int] = []
        for surface in surfaces:
            token_id = _single_round_trip_token_id(tokenizer, surface)
            previous_label = token_id_to_label.get(token_id)
            if previous_label is not None and previous_label != label:
                raise ValueError(
                    f"Token ID {token_id} maps to both {previous_label!r} and {label!r}"
                )
            if token_id not in ids:
                ids.append(token_id)
                token_id_to_label[token_id] = label
        if not ids:
            raise ValueError(f"No valid verdict token IDs resolved for {label!r}")
        verdict_token_ids[label] = ids

    stop_token_ids = {
        surface: _single_round_trip_token_id(tokenizer, surface)
        for surface in profile.stop_token_texts
    }
    normalized_messages = profile.normalize_messages(PROBE_MESSAGES)
    rendered_prompt = profile.render_prompt(tokenizer, PROBE_MESSAGES)
    transport_ids = [
        int(token_id)
        for token_id in tokenizer.encode(
            rendered_prompt,
            add_special_tokens=False,
        )
    ]
    template_ids = [
        int(token_id)
        for token_id in tokenizer.apply_chat_template(
            normalized_messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    ]
    if profile.assistant_prefill:
        template_ids.extend(
            int(token_id)
            for token_id in tokenizer.encode(
                profile.assistant_prefill,
                add_special_tokens=False,
            )
        )
    if transport_ids != template_ids:
        raise ValueError(
            "Rendered text transport does not re-encode to the tokenizer chat-template IDs"
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "passed": True,
        "status": "complete",
        "model_registry_name": profile.registry_name,
        "model_name": profile.hf_model_name,
        "model_revision": profile.revision,
        "resolved_config_commit_hash": config_commit_hash,
        "resolved_tokenizer_commit_hash": tokenizer_commit_hash,
        "trust_remote_code": profile.trust_remote_code,
        "model_type": getattr(config, "model_type", None),
        "config_class": f"{type(config).__module__}.{type(config).__qualname__}",
        "tokenizer_class": (
            f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}"
        ),
        "verdict_token_texts": {
            label: list(profile.verdict_token_texts[label])
            for label in ("A", "B", "tie")
        },
        "resolved_verdict_token_ids": verdict_token_ids,
        "resolved_stop_token_ids": stop_token_ids,
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "chat_transport_match": True,
        "rendered_prompt_sha256": _sha256_text(rendered_prompt),
        "rendered_prompt_token_count": len(transport_ids),
        "rendered_prompt_token_ids_sha256": _sha256_ints(transport_ids),
        "issues": [],
        "source_hashes": _source_hashes(),
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoConfig, AutoTokenizer

    profile = get_model_profile(args.model_name)
    token_argument: bool | None = True if args.require_authentication else None
    common_kwargs = {
        "revision": profile.revision,
        "cache_dir": args.cache_dir,
        "local_files_only": not args.allow_download,
        "trust_remote_code": profile.trust_remote_code,
        "token": token_argument,
    }
    config = AutoConfig.from_pretrained(profile.hf_model_name, **common_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(profile.hf_model_name, **common_kwargs)
    result = validate_loaded_tokenizer_contract(profile, tokenizer, config)
    result["runtime"] = {
        "transformers_version": importlib.metadata.version("transformers"),
        "huggingface_hub_version": importlib.metadata.version("huggingface-hub"),
        "allow_download": bool(args.allow_download),
        "authentication_required": bool(args.require_authentication),
    }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate authenticated model access, the pinned tokenizer revision, "
            "literal verdict tokens, stop tokens, and chat-template text transport."
        )
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--require-authentication", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = run_probe(args)
    except Exception as error:  # preserve a compact immutable failure receipt
        profile = get_model_profile(args.model_name)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "passed": False,
            "status": "failed",
            "model_registry_name": profile.registry_name,
            "model_name": profile.hf_model_name,
            "model_revision": profile.revision,
            "trust_remote_code": profile.trust_remote_code,
            "error_type": type(error).__name__,
            "error": sanitize_exception_text(error),
            "issues": ["tokenizer contract preflight failed"],
            "source_hashes": _source_hashes(),
        }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
