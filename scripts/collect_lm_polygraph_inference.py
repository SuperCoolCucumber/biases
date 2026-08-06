#!/usr/bin/env python3
"""Collect frozen-prompt LM-Polygraph inference scores for one campaign model."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import subprocess
import sys
import uuid
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from biases.analysis.lm_polygraph_inference import (
    COMPLETE_FILE_NAME,
    CROSS_BACKEND_REPLAY_REASON,
    CROSS_BACKEND_REPLAY_ROLE,
    FROZEN_MAX_MODEL_LEN,
    LM_POLYGRAPH_COMMIT,
    PREFLIGHT_COMPLETE_FILE_NAME,
    SCORE_FILE_NAME,
    SELECTION_FILE_NAME,
    SOURCE_PROBABILITY_TOLERANCE,
    canonical_json_sha256,
    full_vocabulary_metrics,
    make_score_row,
    p_true_metrics,
    prompt_token_length_preflight,
    read_jsonl,
    reconstruct_replay_selection,
    replay_selection_manifest,
    restricted_label_metrics,
    token_ids_sha256,
    validate_existing_score_rows,
    validate_score_rows_against_selection,
    validate_scientific_score_gates,
)
from biases.models import get_model_profile
from biases.pairing import file_sha256


MODEL_MARKER_FILE_NAME = "campaign_model_complete.json"
SOURCE_FILE_NAMES = (
    "silent_bias_stage_a_run_records.jsonl",
    "silent_bias_stage_b_run_records.jsonl",
    "silent_bias_stage_a_pair_summary.jsonl",
    MODEL_MARKER_FILE_NAME,
)
BATCH_FAILURE_FILE_PREFIX = "lm_polygraph_inference_batch_failure"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--campaign-model-dir", type=Path, required=True)
    parser.add_argument("--model-name", required=True, help="Registered campaign model name")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--source-contract",
        type=Path,
        required=True,
        help="Externally pinned immutable campaign/source contract JSON",
    )
    parser.add_argument(
        "--source-bundle",
        type=Path,
        required=True,
        help="Frozen Git bundle whose digest and commit are pinned by the source contract",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--limit-records",
        type=int,
        help=(
            "Smoke-only deterministic prefix. Full prompt/hash preflight still runs "
            "before this limit is applied."
        ),
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Reconstruct and hash-check every prompt without loading the model",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Permit Hugging Face access; default is local-files-only",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default="sdpa",
    )
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.limit_records is not None and args.limit_records < 1:
        parser.error("--limit-records must be at least 1")
    return args


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_write_json_exclusive(path: Path, payload: Any) -> None:
    """Atomically publish JSON while refusing to replace an existing path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_safe_evidence(value: Any) -> Any:
    """Return a loss-aware JSON-safe representation for failure evidence."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if value != value:
            return {"nonfinite_float": "nan"}
        if value == float("inf"):
            return {"nonfinite_float": "+inf"}
        if value == float("-inf"):
            return {"nonfinite_float": "-inf"}
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe_evidence(child)
            for key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe_evidence(child) for child in value]
    try:
        scalar = value.item()
    except (AttributeError, TypeError, ValueError):
        return {"unserializable_repr": repr(value)}
    return _json_safe_evidence(scalar)


def write_batch_failure_evidence(
    *,
    output_dir: Path,
    processed_before_batch: int,
    batch_record_ids: Sequence[str],
    stage: str,
    exception: Exception,
    batch_rows: Sequence[Mapping[str, Any]],
    collector_spec_hash: str,
) -> Path:
    """Write one unique no-clobber failure receipt without masking the error."""

    created_at = datetime.now(UTC)
    batch_digest = canonical_json_sha256(list(batch_record_ids))[:16]
    timestamp = created_at.strftime("%Y%m%dT%H%M%S.%fZ")
    filename = (
        f"{BATCH_FAILURE_FILE_PREFIX}.after-{processed_before_batch:06d}."
        f"{batch_digest}.{timestamp}.{uuid.uuid4().hex}.json"
    )
    path = output_dir / filename
    payload = _json_safe_evidence(
        {
            "schema_version": 2,
            "status": (
                "failed_during_append"
                if stage == "score_append"
                else "failed_before_append"
            ),
            "append_may_be_partial": stage == "score_append",
            "created_at": created_at.isoformat(),
            "failure_stage": stage,
            "exception_type": type(exception).__name__,
            "exception_message": str(exception),
            "processed_before_batch": processed_before_batch,
            "batch_record_ids": list(batch_record_ids),
            "batch_rows": list(batch_rows),
            "collector_spec_hash": collector_spec_hash,
            "scheduler": scheduler_provenance(),
        }
    )
    atomic_write_json_exclusive(path, payload)
    return path


def append_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    payload = "".join(
        f"{json.dumps(row, sort_keys=True, ensure_ascii=True, allow_nan=False)}\n"
        for row in rows
    ).encode("utf-8")
    with path.open("ab+") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() > 0:
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) != b"\n":
                handle.seek(0, os.SEEK_END)
                handle.write(b"\n")
        handle.seek(0, os.SEEK_END)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def batches(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _declared_sha256(value: Any, filename: str) -> str | None:
    """Find a declared file digest across schema-2 marker representations."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            if Path(str(key)).name == filename:
                if isinstance(child, str) and len(child) == 64:
                    return child
                if isinstance(child, Mapping):
                    digest = child.get("sha256") or child.get("hash")
                    if isinstance(digest, str) and len(digest) == 64:
                        return digest
            found = _declared_sha256(child, filename)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _declared_sha256(child, filename)
            if found is not None:
                return found
    return None


def validate_campaign_marker(
    *,
    marker_path: Path,
    model_name: str,
    model_revision: str | None,
    input_file_hash: str,
    source_hashes: Mapping[str, str],
) -> Mapping[str, Any]:
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if not isinstance(marker, Mapping):
        raise TypeError(f"{marker_path}: marker must contain an object")
    checks = {
        "schema_version": marker.get("schema_version") in {2, "2"},
        "status": str(marker.get("status", "")).lower() == "complete",
        "model_name": marker.get("model_name") == model_name,
        "model_revision": marker.get("model_revision") == model_revision,
        "input_file_hash": marker.get("input_file_hash") == input_file_hash,
    }
    mismatches = [name for name, matches in checks.items() if not matches]
    if mismatches:
        raise ValueError(
            f"{marker_path}: frozen marker fields mismatch: {', '.join(mismatches)}"
        )
    required_marker_hashes = SOURCE_FILE_NAMES[:2]
    for filename in required_marker_hashes:
        declared = _declared_sha256(marker, filename)
        if declared is None:
            raise ValueError(f"{marker_path}: no declared SHA-256 for {filename}")
        if declared != source_hashes[filename]:
            raise ValueError(f"{marker_path}: SHA-256 mismatch for {filename}")
    summary_filename = "silent_bias_stage_a_pair_summary.jsonl"
    declared_summary = _declared_sha256(marker, summary_filename)
    if declared_summary is None:
        raise ValueError(f"{marker_path}: no declared SHA-256 for {summary_filename}")
    if declared_summary != source_hashes[summary_filename]:
        raise ValueError(f"{marker_path}: SHA-256 mismatch for {summary_filename}")
    return marker


def _repository_commit() -> str:
    repository_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip()
    if len(commit) != 40:
        raise ValueError(f"Could not resolve a full repository commit: {commit!r}")
    return commit


def validate_stage_generation_provenance(
    *,
    marker: Mapping[str, Any],
    model_contract: Mapping[str, Any],
    contract_schema_version: int,
    legacy_generation_commit: str,
    expected_stage_counts: Mapping[str, int],
) -> dict[str, dict[str, Any]]:
    """Validate exact per-stage generation provenance from a source contract.

    Schema 1 contracts predate mixed preserved/new-generation campaigns and
    therefore use one top-level generation commit for every stage. Schema 2
    contracts must pin each stage's commit, mode, and record count explicitly.
    """

    stage_generation = marker.get("stage_generation")
    if not isinstance(stage_generation, Mapping):
        raise ValueError("campaign marker has no stage-generation provenance")
    if set(stage_generation) != set(expected_stage_counts):
        raise ValueError("campaign marker must pin exactly stage_a and stage_b")

    expected_generation = model_contract.get("stage_generation")
    if contract_schema_version == 2:
        if not isinstance(expected_generation, Mapping):
            raise ValueError(
                "schema-2 model source contract has no stage-generation provenance"
            )
        if set(expected_generation) != set(expected_stage_counts):
            raise ValueError(
                "schema-2 model source contract must pin exactly stage_a and stage_b"
            )
    elif expected_generation is not None:
        raise ValueError(
            "schema-1 model source contract cannot override stage generation"
        )

    validated: dict[str, dict[str, Any]] = {}
    for stage_name, expected_count in expected_stage_counts.items():
        stage = stage_generation.get(stage_name)
        if not isinstance(stage, Mapping):
            raise ValueError(f"campaign marker has no {stage_name} provenance")

        if isinstance(expected_generation, Mapping):
            expected_stage = expected_generation.get(stage_name)
            if not isinstance(expected_stage, Mapping):
                raise ValueError(
                    f"model source contract has no {stage_name} provenance"
                )
            expected_commit = str(expected_stage.get("code_commit") or "")
            expected_mode = str(expected_stage.get("mode") or "")
            try:
                contract_record_count = int(expected_stage.get("records", -1))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"model source contract {stage_name} record count is invalid"
                ) from error
            if len(expected_commit) != 40 or expected_mode not in {
                "preserved",
                "new_generation",
            }:
                raise ValueError(
                    f"model source contract {stage_name} provenance is incomplete"
                )
            if contract_record_count != expected_count:
                raise ValueError(
                    f"model source contract {stage_name} record-count drift"
                )
            if stage.get("code_commit") != expected_commit:
                raise ValueError(
                    f"campaign marker {stage_name} generation commit drift"
                )
            if stage.get("mode") != expected_mode:
                raise ValueError(f"campaign marker {stage_name} generation mode drift")
        else:
            expected_commit = legacy_generation_commit
            expected_mode = str(stage.get("mode") or "")
            if stage.get("code_commit") != expected_commit:
                raise ValueError(
                    f"campaign marker {stage_name} generation commit drift"
                )

        try:
            marker_record_count = int(stage.get("records", -1))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"campaign marker {stage_name} record count is invalid"
            ) from error
        if marker_record_count != expected_count:
            raise ValueError(f"campaign marker {stage_name} record-count drift")
        validated[stage_name] = {
            "code_commit": expected_commit,
            "mode": expected_mode,
            "records": expected_count,
        }
    return validated


def validate_source_contract(
    *,
    contract_path: Path,
    source_bundle_path: Path,
    campaign_model_dir: Path,
    model_registry_name: str,
    model_name: str,
    model_revision: str | None,
    input_file_hash: str,
    source_hashes: Mapping[str, str],
    marker: Mapping[str, Any],
    selection: Any,
) -> dict[str, Any]:
    """Bind this run to an externally pinned campaign and source bundle."""

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if not isinstance(contract, Mapping) or contract.get("schema_version") not in {
        1,
        2,
    }:
        raise ValueError("source contract must be a schema-1 or schema-2 object")
    contract_schema_version = int(contract["schema_version"])
    campaign_tag = str(contract.get("campaign_tag") or "")
    parser_commit = str(contract.get("parser_commit") or "")
    generation_commit = str(contract.get("generation_commit") or "")
    bundle_sha256 = str(contract.get("source_bundle_sha256") or "")
    if not campaign_tag or len(parser_commit) != 40 or len(generation_commit) != 40:
        raise ValueError("source contract is missing its campaign tag or commits")
    if len(bundle_sha256) != 64:
        raise ValueError("source contract has no valid source-bundle SHA-256")
    if contract.get("input_file_sha256") != input_file_hash:
        raise ValueError("dataset SHA-256 differs from the external source contract")
    if not source_bundle_path.is_file() or file_sha256(source_bundle_path) != bundle_sha256:
        raise ValueError("source Git bundle differs from the external source contract")
    if _repository_commit() != parser_commit:
        raise ValueError("executed base repository commit differs from the source contract")

    resolved_model_dir = campaign_model_dir.resolve()
    if resolved_model_dir.parent.name != "full" or resolved_model_dir.parent.parent.name != campaign_tag:
        raise ValueError("campaign directory is outside the source-contract campaign tag")
    models = contract.get("models")
    if not isinstance(models, Mapping) or model_registry_name not in models:
        raise ValueError(f"source contract has no model {model_registry_name!r}")
    model_contract = models[model_registry_name]
    if not isinstance(model_contract, Mapping):
        raise ValueError("model source contract must be an object")
    scalar_checks = {
        "directory_name": model_contract.get("directory_name") == resolved_model_dir.name,
        "model_name": model_contract.get("model_name") == model_name,
        "model_revision": model_contract.get("model_revision") == model_revision,
        "stage_a_records": int(model_contract.get("stage_a_records", -1))
        == selection.stage_a_count,
        "stage_b_records": int(model_contract.get("stage_b_records", -1))
        == selection.full_stage_b_count,
        "primary_stage_b_records": int(
            model_contract.get("primary_stage_b_records", -1)
        )
        == selection.primary_stage_b_count,
        "stage_a_pair_summary_records": int(
            model_contract.get("stage_a_pair_summary_records", -1)
        )
        == selection.stage_a_pair_summary_count,
    }
    mismatches = [name for name, matches in scalar_checks.items() if not matches]
    if mismatches:
        raise ValueError(
            "campaign source counts/identity differ from the external contract: "
            + ", ".join(mismatches)
        )
    expected_files = model_contract.get("files")
    if not isinstance(expected_files, Mapping) or set(expected_files) != set(SOURCE_FILE_NAMES):
        raise ValueError("model source contract must pin exactly the four source files")
    for filename in SOURCE_FILE_NAMES:
        if expected_files.get(filename) != source_hashes[filename]:
            raise ValueError(f"{filename} differs from the external source contract")

    validated_stage_generation = validate_stage_generation_provenance(
        marker=marker,
        model_contract=model_contract,
        contract_schema_version=contract_schema_version,
        legacy_generation_commit=generation_commit,
        expected_stage_counts={
            "stage_a": selection.stage_a_count,
            "stage_b": selection.full_stage_b_count,
        },
    )

    return {
        "schema_version": contract_schema_version,
        "path": str(contract_path),
        "sha256": file_sha256(contract_path),
        "campaign_tag": campaign_tag,
        "parser_commit": parser_commit,
        "generation_commit": generation_commit,
        "stage_generation": validated_stage_generation,
        "source_bundle_path": str(source_bundle_path),
        "source_bundle_sha256": bundle_sha256,
        "model_contract": dict(model_contract),
    }


def resolve_single_token(tokenizer: Any, surface: str) -> int:
    token_ids = tokenizer.encode(surface, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(f"{surface!r} must resolve to exactly one token, got {token_ids!r}")
    token_id = int(token_ids[0])
    decoded = tokenizer.decode(
        [token_id],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if decoded != surface:
        raise ValueError(
            f"{surface!r} does not round-trip through token {token_id}: {decoded!r}"
        )
    return token_id


def resolve_label_token_ids(tokenizer: Any, profile: Any) -> dict[str, int]:
    result: dict[str, int] = {}
    for label in ("A", "B", "tie"):
        surfaces = tuple(profile.verdict_token_texts[label])
        if len(surfaces) != 1:
            raise ValueError(f"{profile.registry_name}: expected one canonical {label} surface")
        result[label] = resolve_single_token(tokenizer, surfaces[0])
    if len(set(result.values())) != 3:
        raise ValueError("A, B, and tie token IDs must be distinct")
    return result


def verify_source_label_contract(
    selection: Any,
    label_token_ids: Mapping[str, int],
) -> None:
    expected = {label: [token_id] for label, token_id in label_token_ids.items()}
    for item in selection.items:
        spec = item.source_row.get("spec")
        actual = spec.get("verdict_token_ids") if isinstance(spec, Mapping) else None
        if actual != expected:
            raise ValueError(
                f"Record {item.record_id!r} uses verdict IDs {actual!r}, expected {expected!r}"
            )


def tokenizer_provenance(tokenizer: Any) -> dict[str, Any]:
    chat_template = str(getattr(tokenizer, "chat_template", "") or "")
    vocabulary = {
        str(token): int(token_id) for token, token_id in tokenizer.get_vocab().items()
    }
    special_tokens = json.loads(
        json.dumps(
            dict(getattr(tokenizer, "special_tokens_map", {})),
            sort_keys=True,
            default=str,
        )
    )
    return {
        "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "vocab_size": int(len(tokenizer)),
        "padding_side": str(tokenizer.padding_side),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "chat_template_sha256": canonical_json_sha256(chat_template),
        "vocabulary_sha256": canonical_json_sha256(vocabulary),
        "special_tokens_map": special_tokens,
        "special_tokens_map_sha256": canonical_json_sha256(special_tokens),
    }


def model_config_provenance(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoConfig

    profile = get_model_profile(args.model_name)
    config = AutoConfig.from_pretrained(
        profile.hf_model_name,
        revision=profile.revision,
        cache_dir=args.cache_dir,
        local_files_only=not args.allow_download,
        trust_remote_code=True,
    )
    payload = config.to_dict()
    return {
        "class": f"{type(config).__module__}.{type(config).__qualname__}",
        "sha256": canonical_json_sha256(payload),
        "model_type": payload.get("model_type"),
        "vocab_size": payload.get("vocab_size"),
    }


def scientific_dependency_hashes() -> dict[str, str]:
    module_names = (
        "biases.models",
        "biases.pairing",
        "biases.schemas",
        "biases.silent_bias_runner",
        "biases.social_cue_prompts",
        "biases.stage_planning",
        "biases.analysis.lm_polygraph_inference",
    )
    result: dict[str, str] = {Path(__file__).name: file_sha256(Path(__file__).resolve())}
    for module_name in module_names:
        module = importlib.import_module(module_name)
        module_path = Path(str(module.__file__)).resolve()
        result[f"{module_name}:{module_path.name}"] = file_sha256(module_path)
    return dict(sorted(result.items()))


def load_tokenizer(args: argparse.Namespace) -> Any:
    from transformers import AutoTokenizer

    profile = get_model_profile(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        profile.hf_model_name,
        revision=profile.revision,
        cache_dir=args.cache_dir,
        local_files_only=not args.allow_download,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither a pad token nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(args: argparse.Namespace) -> Any:
    import torch
    from transformers import AutoModelForCausalLM

    profile = get_model_profile(args.model_name)
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        profile.hf_model_name,
        revision=profile.revision,
        cache_dir=args.cache_dir,
        local_files_only=not args.allow_download,
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
    )
    model.to(args.device)
    model.eval()
    return model


def loaded_model_backend_provenance(
    model: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Hard-check the loaded backend against the requested inference contract."""

    expected_dtype = f"torch.{args.dtype}"
    actual_dtype = str(getattr(model, "dtype", ""))
    if actual_dtype != expected_dtype:
        raise ValueError(
            f"Loaded model dtype {actual_dtype!r} does not match {expected_dtype!r}"
        )
    config = getattr(model, "config", None)
    actual_attention = getattr(config, "_attn_implementation", None)
    if actual_attention != args.attn_implementation:
        raise ValueError(
            "Loaded model attention implementation "
            f"{actual_attention!r} does not match {args.attn_implementation!r}"
        )
    output_embeddings = model.get_output_embeddings()
    output_dtype = str(getattr(output_embeddings.weight, "dtype", ""))
    if output_dtype != expected_dtype:
        raise ValueError(
            f"Output-head dtype {output_dtype!r} does not match {expected_dtype!r}"
        )
    return {
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "model_dtype": actual_dtype,
        "output_head_dtype": output_dtype,
        "config_attention_implementation": actual_attention,
    }


def last_position_logits(
    *,
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    device: str,
) -> tuple[Any, list[int], list[str]]:
    import torch

    individual_ids = [
        tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts
    ]
    if any(not token_ids for token_ids in individual_ids):
        raise ValueError("Every rendered prompt must contain at least one token")
    encoded = tokenizer(
        list(prompts),
        add_special_tokens=False,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )
    encoded = {name: value.to(device) for name, value in encoded.items()}
    with torch.inference_mode():
        outputs = model(
            **encoded,
            use_cache=False,
            return_dict=True,
            logits_to_keep=1,
        )
    logits = outputs.logits
    if logits.ndim != 3 or logits.shape[0] != len(prompts) or logits.shape[1] != 1:
        raise RuntimeError(
            "Model ignored logits_to_keep=1; refusing a full-sequence logits path: "
            f"shape={tuple(logits.shape)}"
        )
    return (
        logits[:, 0, :].float().cpu(),
        [len(ids) for ids in individual_ids],
        [token_ids_sha256(ids) for ids in individual_ids],
    )


def environment_provenance(model: Any | None = None) -> dict[str, Any]:
    import numpy as np
    import torch
    import transformers

    result: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms_enabled": (
            torch.are_deterministic_algorithms_enabled()
        ),
    }
    if torch.cuda.is_available():
        result["cuda_device_count"] = torch.cuda.device_count()
        result["cuda_device_names"] = [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ]
        result["cuda_sdp_backends"] = {
            "flash_enabled": torch.backends.cuda.flash_sdp_enabled(),
            "memory_efficient_enabled": (
                torch.backends.cuda.mem_efficient_sdp_enabled()
            ),
            "math_enabled": torch.backends.cuda.math_sdp_enabled(),
            "cudnn_enabled": torch.backends.cuda.cudnn_sdp_enabled(),
        }
        result["cuda_matmul_allow_tf32"] = torch.backends.cuda.matmul.allow_tf32
        result["cudnn_allow_tf32"] = torch.backends.cudnn.allow_tf32
        result["cudnn_enabled"] = torch.backends.cudnn.enabled
        result["cudnn_version"] = torch.backends.cudnn.version()
        result["cuda_device_properties"] = [
            {
                "name": properties.name,
                "major": properties.major,
                "minor": properties.minor,
                "total_memory": properties.total_memory,
                "multi_processor_count": properties.multi_processor_count,
            }
            for properties in (
                torch.cuda.get_device_properties(index)
                for index in range(torch.cuda.device_count())
            )
        ]
    if model is not None:
        result["model_class"] = f"{type(model).__module__}.{type(model).__qualname__}"
        result["model_dtype"] = str(getattr(model, "dtype", ""))
    return result


def scheduler_provenance() -> dict[str, str | None]:
    return {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "slurm_restart_count": os.environ.get("SLURM_RESTART_COUNT"),
    }


def _prepare_output_dir(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    immutable = args.campaign_model_dir.resolve()
    if output == immutable or output.is_relative_to(immutable):
        raise ValueError("Output directory must be outside the immutable campaign model directory")
    if (args.output_dir / COMPLETE_FILE_NAME).exists():
        raise FileExistsError("A completion marker already exists; refusing to overwrite")
    if args.output_dir.exists() and not args.resume and any(args.output_dir.iterdir()):
        raise FileExistsError(
            "Output directory is non-empty; pass --resume or use a new directory"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _prepare_output_dir(args)
    profile = get_model_profile(args.model_name)
    if profile.revision is None:
        raise ValueError("New-inference collection requires a pinned model revision")

    tokenizer = load_tokenizer(args)
    model_config = model_config_provenance(args)
    label_token_ids = resolve_label_token_ids(tokenizer, profile)
    true_token_id = resolve_single_token(tokenizer, "True")
    false_token_id = resolve_single_token(tokenizer, "False")
    if true_token_id == false_token_id:
        raise ValueError("True and False must resolve to different token IDs")

    selection = reconstruct_replay_selection(
        data_path=args.data,
        campaign_model_dir=args.campaign_model_dir,
        model_registry_name=args.model_name,
        tokenizer=tokenizer,
        require_exact_counts=True,
    )
    verify_source_label_contract(selection, label_token_ids)
    token_length_preflight = prompt_token_length_preflight(
        selection,
        tokenizer=tokenizer,
        max_model_len=FROZEN_MAX_MODEL_LEN,
    )
    if args.limit_records is not None and args.limit_records > len(selection.items):
        raise ValueError("--limit-records exceeds the full replay selection")

    source_hashes = {
        filename: file_sha256(args.campaign_model_dir / filename)
        for filename in SOURCE_FILE_NAMES
    }
    marker = validate_campaign_marker(
        marker_path=args.campaign_model_dir / MODEL_MARKER_FILE_NAME,
        model_name=profile.hf_model_name,
        model_revision=profile.revision,
        input_file_hash=selection.input_file_hash,
        source_hashes=source_hashes,
    )
    source_contract = validate_source_contract(
        contract_path=args.source_contract,
        source_bundle_path=args.source_bundle,
        campaign_model_dir=args.campaign_model_dir,
        model_registry_name=args.model_name,
        model_name=profile.hf_model_name,
        model_revision=profile.revision,
        input_file_hash=selection.input_file_hash,
        source_hashes=source_hashes,
        marker=marker,
        selection=selection,
    )
    collector_source_hashes = scientific_dependency_hashes()
    inference_runtime_fingerprint = environment_provenance()
    collector_spec = {
        "schema_version": 2,
        "methods": {
            "p_true": "-log p(True), exact pinned binary meta-prompt",
            "mean_token_entropy": "-sum_v p(v) log p(v), original decision position",
            "self_certainty": "mean_v log p(v) + log(V), pinned estimator output -KL(U||p)",
        },
        "model_name": profile.hf_model_name,
        "model_revision": profile.revision,
        "label_token_ids": label_token_ids,
        "true_token_id": true_token_id,
        "false_token_id": false_token_id,
        "tokenizer": tokenizer_provenance(tokenizer),
        "model_config": model_config,
        "dtype": args.dtype,
        "device": args.device,
        "batch_size": args.batch_size,
        "attn_implementation": args.attn_implementation,
        "inference_runtime_fingerprint": inference_runtime_fingerprint,
        "local_files_only": not args.allow_download,
        "forward_contract": "AutoModelForCausalLM logits_to_keep=1, use_cache=False",
        "frozen_max_model_len": FROZEN_MAX_MODEL_LEN,
        "source_probability_tolerance": SOURCE_PROBABILITY_TOLERANCE,
        "source_probability_tolerance_enforced": False,
        "restricted_map_agreement_enforced": False,
        "cross_backend_replay_role": CROSS_BACKEND_REPLAY_ROLE,
        "cross_backend_replay_reason": CROSS_BACKEND_REPLAY_REASON,
        "evaluation_estimands": (
            "all-row cross-engine transfer; same-engine HF replay; "
            "map-agree-only sensitivity"
        ),
        "collector_source_hashes": collector_source_hashes,
        "source_contract_sha256": source_contract["sha256"],
        "source_repository_commit": source_contract["parser_commit"],
        "source_bundle_sha256": source_contract["source_bundle_sha256"],
    }
    manifest = replay_selection_manifest(
        selection,
        data_path=args.data,
        campaign_model_dir=args.campaign_model_dir,
        limit_records=args.limit_records,
        source_artifact_hashes=source_hashes,
        collector_spec=collector_spec,
    )
    manifest.update(
        {
            "campaign_marker_sha256": source_hashes[MODEL_MARKER_FILE_NAME],
            "campaign_marker_schema_version": marker.get("schema_version"),
            "campaign_marker_status": marker.get("status"),
            "token_length_preflight": token_length_preflight,
            "source_contract": source_contract,
        }
    )
    manifest_path = args.output_dir / SELECTION_FILE_NAME
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest != manifest:
            raise ValueError("Existing selection manifest differs from the current preflight")
    else:
        atomic_write_json(manifest_path, manifest)

    if args.preflight_only:
        preflight_marker = {
            "schema_version": 1,
            "status": "complete",
            "kind": "silent_bias_lm_polygraph_inference_preflight",
            "created_at": datetime.now(UTC).isoformat(),
            "selection_manifest": SELECTION_FILE_NAME,
            "selection_manifest_sha256": file_sha256(manifest_path),
            "full_preflight_count": len(selection.items),
            "inference_count": manifest["inference_count"],
            "collector_spec_hash": manifest["collector_spec_hash"],
        }
        preflight_path = args.output_dir / PREFLIGHT_COMPLETE_FILE_NAME
        if preflight_path.exists():
            raise FileExistsError(f"{preflight_path} already exists")
        atomic_write_json(preflight_path, preflight_marker)
        print(json.dumps(preflight_marker, sort_keys=True))
        return 0

    selected_items = list(selection.items)
    if args.limit_records is not None:
        selected_items = selected_items[: args.limit_records]
    selected_record_ids = [item.record_id for item in selected_items]
    score_path = args.output_dir / SCORE_FILE_NAME
    existing_rows = read_jsonl(score_path) if score_path.exists() else []
    if existing_rows and not args.resume:
        raise FileExistsError(f"{score_path} exists; pass --resume")
    completed_ids = validate_existing_score_rows(
        existing_rows,
        selected_record_ids=selected_record_ids,
        collector_spec_hash=str(manifest["collector_spec_hash"]),
    )
    pending = [item for item in selected_items if item.record_id not in completed_ids]

    model = load_model(args)
    loaded_backend = loaded_model_backend_provenance(model, args)
    vocabulary_size = int(model.get_output_embeddings().weight.shape[0])
    tokenizer_vocabulary_size = len(tokenizer)
    if vocabulary_size < tokenizer_vocabulary_size:
        raise ValueError(
            f"Model vocabulary {vocabulary_size} is smaller than tokenizer length "
            f"{tokenizer_vocabulary_size}"
        )
    if max(*label_token_ids.values(), true_token_id, false_token_id) >= vocabulary_size:
        raise ValueError("One or more resolved tokens are outside the model vocabulary")
    if existing_rows:
        validate_score_rows_against_selection(
            existing_rows,
            selected_items=selected_items,
            collector_spec_hash=str(manifest["collector_spec_hash"]),
            tokenizer=tokenizer,
            true_token_id=true_token_id,
            false_token_id=false_token_id,
            label_token_ids=label_token_ids,
            vocabulary_size=vocabulary_size,
            tokenizer_vocabulary_size=tokenizer_vocabulary_size,
        )

    processed = len(completed_ids)
    last_reported = processed
    for batch in batches(pending, args.batch_size):
        batch_record_ids = [item.record_id for item in batch]
        rows: list[dict[str, Any]] = []
        failure_stage = "original_prompt_forward"
        try:
            (
                original_logits,
                original_lengths,
                original_token_hashes,
            ) = last_position_logits(
                model=model,
                tokenizer=tokenizer,
                prompts=[item.original_prompt for item in batch],
                device=args.device,
            )
            failure_stage = "p_true_prompt_forward"
            (
                p_true_logits,
                p_true_lengths,
                p_true_token_hashes,
            ) = last_position_logits(
                model=model,
                tokenizer=tokenizer,
                prompts=[item.p_true_prompt for item in batch],
                device=args.device,
            )
            failure_stage = "score_row_construction"
            for index, item in enumerate(batch):
                original_values = original_logits[index].numpy()
                p_true_values = p_true_logits[index].numpy()
                rows.append(
                    make_score_row(
                        item=item,
                        full_vocab=full_vocabulary_metrics(original_values),
                        p_true=p_true_metrics(
                            p_true_values,
                            true_token_id=true_token_id,
                        ),
                        restricted=restricted_label_metrics(
                            original_values,
                            label_token_ids=label_token_ids,
                        ),
                        true_token_id=true_token_id,
                        false_token_id=false_token_id,
                        label_token_ids=label_token_ids,
                        original_token_count=original_lengths[index],
                        p_true_token_count=p_true_lengths[index],
                        original_token_ids_sha256=original_token_hashes[index],
                        p_true_token_ids_sha256=p_true_token_hashes[index],
                        vocabulary_size=vocabulary_size,
                        tokenizer_vocabulary_size=tokenizer_vocabulary_size,
                        collector_spec_hash=str(manifest["collector_spec_hash"]),
                    )
                )
            failure_stage = "structural_batch_validation"
            validate_score_rows_against_selection(
                rows,
                selected_items=batch,
                collector_spec_hash=str(manifest["collector_spec_hash"]),
                tokenizer=tokenizer,
                true_token_id=true_token_id,
                false_token_id=false_token_id,
                label_token_ids=label_token_ids,
                vocabulary_size=vocabulary_size,
                tokenizer_vocabulary_size=tokenizer_vocabulary_size,
            )
            failure_stage = "score_append"
            append_jsonl(score_path, rows)
        except Exception as exc:
            try:
                failure_path = write_batch_failure_evidence(
                    output_dir=args.output_dir,
                    processed_before_batch=processed,
                    batch_record_ids=batch_record_ids,
                    stage=failure_stage,
                    exception=exc,
                    batch_rows=rows,
                    collector_spec_hash=str(manifest["collector_spec_hash"]),
                )
                print(
                    f"preserved batch failure evidence at {failure_path}",
                    file=sys.stderr,
                    flush=True,
                )
            except Exception as receipt_error:
                print(
                    "failed to preserve batch failure evidence: "
                    f"{type(receipt_error).__name__}: {receipt_error}",
                    file=sys.stderr,
                    flush=True,
                )
            raise
        processed += len(rows)
        if processed == len(selected_items) or processed - last_reported >= 100:
            print(f"processed {processed}/{len(selected_items)}", flush=True)
            last_reported = processed

    final_rows = read_jsonl(score_path)
    validate_existing_score_rows(
        final_rows,
        selected_record_ids=selected_record_ids,
        collector_spec_hash=str(manifest["collector_spec_hash"]),
    )
    if len(final_rows) != len(selected_items):
        raise RuntimeError("Score file is incomplete after collection")
    scientific_gates = validate_score_rows_against_selection(
        final_rows,
        selected_items=selected_items,
        collector_spec_hash=str(manifest["collector_spec_hash"]),
        tokenizer=tokenizer,
        true_token_id=true_token_id,
        false_token_id=false_token_id,
        label_token_ids=label_token_ids,
        vocabulary_size=vocabulary_size,
        tokenizer_vocabulary_size=tokenizer_vocabulary_size,
    )
    completion_environment = environment_provenance(model)
    completion_runtime_fingerprint = {
        key: value
        for key, value in completion_environment.items()
        if key not in {"model_class", "model_dtype"}
    }
    if completion_runtime_fingerprint != inference_runtime_fingerprint:
        raise RuntimeError(
            "Inference runtime fingerprint changed between preflight and completion"
        )
    completion = {
        "schema_version": 1,
        "status": "complete",
        "kind": "silent_bias_lm_polygraph_inference",
        "created_at": datetime.now(UTC).isoformat(),
        "model_name": profile.hf_model_name,
        "model_revision": profile.revision,
        "input_file_hash": selection.input_file_hash,
        "smoke_only": args.limit_records is not None,
        "record_count": len(final_rows),
        "full_preflight_count": len(selection.items),
        "scores_file": SCORE_FILE_NAME,
        "scores_sha256": file_sha256(score_path),
        "selection_manifest": SELECTION_FILE_NAME,
        "selection_manifest_sha256": file_sha256(manifest_path),
        "record_id_digest": canonical_json_sha256(selected_record_ids),
        "collector_spec_hash": manifest["collector_spec_hash"],
        "lm_polygraph_commit": LM_POLYGRAPH_COMMIT,
        "source_contract_sha256": source_contract["sha256"],
        "source_repository_commit": source_contract["parser_commit"],
        "source_bundle_sha256": source_contract["source_bundle_sha256"],
        "collector_source_hashes": collector_source_hashes,
        "model_config": model_config,
        "loaded_model_backend": loaded_backend,
        "model_output_vocabulary_size": vocabulary_size,
        "tokenizer_vocabulary_size": tokenizer_vocabulary_size,
        "padded_vocabulary_size_delta": vocabulary_size
        - tokenizer_vocabulary_size,
        "scientific_gates": scientific_gates,
        "scheduler": scheduler_provenance(),
        "environment": completion_environment,
    }
    atomic_write_json(args.output_dir / COMPLETE_FILE_NAME, completion)
    print(json.dumps(completion, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
