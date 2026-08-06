#!/usr/bin/env python3
"""Inference-free full-grid preflight for controlled uncertainty shift.

The preflight consumes one frozen, question-routed source CSV, one immutable
routing manifest, one model registry entry, and one runtime mapping.  It uses
the same Stage A/Stage B planners and social-cue prompt builder as inference,
but never constructs a model backend or generates a token.

One invocation validates exactly one model.  This keeps model-specific prompt
lengths, token contracts, runtime settings, and eventual estimands separate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

from biases.dataset_splits import routing_manifest
from biases.models import ModelProfile, get_model_profile
from biases.pairing import (
    file_sha256,
    make_pair_identity_key,
    normalize_ordering,
    normalize_verdict,
)
from biases.position_bias import (
    CONSTRAINED_LOGPROBS_MODE,
    JUDGE_OUTPUT_PARSER_VERSION,
    POSITION_PAIR_ELIGIBILITY_CONTRACT,
    VERBALIZED_OUTPUT_PARSER_VERSION,
    PositionPair,
    load_position_pairs_with_eligibility,
)
from biases.safe_diagnostics import sanitize_exception_text
from biases.schemas import JudgeExample, OutputMode, PairOrdering, VerdictLabel
from biases.social_cue_prompts import (
    AUTHORITY_DOSES,
    BANDWAGON_DOSES,
    build_social_cue,
    build_social_cue_messages,
    build_social_cue_prompt_package,
)
from biases.stage_planning import (
    CleanPairSummary,
    PlannedCondition,
    StageAPairInput,
    clean_summaries_from_rows,
    generate_stage_a_conditions,
    generate_stage_b_conditions,
)


DEFAULT_MAX_MODEL_LEN = 4096
DEFAULT_GENERATION_HEADROOM = 24
ROUTING_SCHEMA_VERSION = 2
ROUTING_ARTIFACT_TYPE = "frozen_question_disjoint_routing_package"
RUNTIME_REQUIRED_FIELDS: tuple[str, ...] = (
    "model_registry_name",
    "model_hf_name",
    "model_revision",
    "tensor_parallel_size",
    "max_model_len",
    "gpu_memory_utilization",
    "dtype",
    "batch_size",
    "max_num_batched_tokens",
    "max_num_seqs",
    "enforce_eager",
    "disable_custom_all_reduce",
    "seed",
    "sampling_temperature",
    "consistency_runs",
    "consistency_schedule",
    "include_verbalized_confidence",
    "engine_versions",
    "runtime_sha256",
)
RUNNER_RUNTIME_FIELDS: tuple[str, ...] = RUNTIME_REQUIRED_FIELDS
ENGINE_VERSION_FIELDS: tuple[str, ...] = (
    "python",
    "torch",
    "transformers",
    "vllm",
)
EXPECTED_VERDICT_SURFACES: Mapping[str, tuple[str, ...]] = {
    "A": ("A",),
    "B": ("B",),
    "tie": ("T",),
}


@dataclass(frozen=True, slots=True)
class PlannedPrompt:
    stage: str
    routing_split: str
    planned: PlannedCondition
    example: JudgeExample
    target_realization: str | None = None


@dataclass(frozen=True, slots=True)
class PromptAudit:
    key: str
    stage: str
    routing_split: str
    family: str
    direction: str
    dose: int | None
    ordering: str
    output_mode: str
    target_realization: str | None
    prompt_sha256: str
    planner_prompt_hash: str
    token_ids_sha256: str
    input_tokens: int


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def value_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def enum_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def read_json_mapping(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain one JSON object: {path}")
    return value


def read_jsonl_mappings(path: Path, *, name: str) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{name} has invalid JSON on line {line_number}: {path}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"{name} line {line_number} must contain one JSON object"
                )
            rows.append(value)
    if not rows:
        raise ValueError(f"{name} must contain at least one JSON object: {path}")
    return tuple(rows)


def _required_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"routing manifest {name} must be a JSON object")
    return value


def _required_sha256(value: Any, *, name: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return text


def _read_csv_cells(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"CSV is empty: {path}") from exc
        rows = [list(row) for row in reader]
    if not header or any(not column for column in header):
        raise ValueError(f"CSV must have a non-empty header: {path}")
    duplicates = sorted({column for column in header if header.count(column) > 1})
    if duplicates:
        raise ValueError(f"CSV has duplicate columns {duplicates}: {path}")
    for row_index, row in enumerate(rows):
        if len(row) != len(header):
            raise ValueError(
                "CSV has a malformed row: "
                f"path={path} row_index={row_index} "
                f"fields={len(row)} expected={len(header)}"
            )
    return header, rows


def _rows_without_routing_sha256(path: Path) -> str:
    header, rows = _read_csv_cells(path)
    retained = [
        index for index, column in enumerate(header) if column != "routing_split"
    ]
    return value_sha256(
        {
            "columns": [header[index] for index in retained],
            "rows": [[row[index] for index in retained] for row in rows],
        }
    )


def _resolve_package_output(
    manifest_path: Path,
    outputs: Mapping[str, Any],
    name: str,
) -> Path:
    record = _required_mapping(outputs.get(name), name=f"outputs.{name}")
    raw_path = Path(str(record.get("path") or ""))
    if (
        raw_path.is_absolute()
        or len(raw_path.parts) != 1
        or raw_path.name in {"", ".", ".."}
    ):
        raise ValueError(
            f"routing manifest outputs.{name}.path must be one filename"
        )
    return (manifest_path.parent / raw_path).resolve()


def _positive_int(runtime: Mapping[str, Any], field: str) -> int:
    value = runtime[field]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"runtime {field} must be a positive integer")
    return value


def validate_runtime_contract(runtime: Mapping[str, Any]) -> dict[str, Any]:
    missing = [field for field in RUNTIME_REQUIRED_FIELDS if field not in runtime]
    if missing:
        raise ValueError(f"runtime mapping is missing fields: {', '.join(missing)}")
    unknown = sorted(set(runtime) - set(RUNTIME_REQUIRED_FIELDS))
    if unknown:
        raise ValueError(
            "runtime mapping contains unsupported fields: " + ", ".join(unknown)
        )

    normalized = dict(runtime)
    for field in ("model_registry_name", "model_hf_name"):
        value = runtime[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"runtime {field} must be a non-empty string")
        normalized[field] = value.strip()
    model_revision = runtime["model_revision"]
    if model_revision is not None and (
        not isinstance(model_revision, str) or not model_revision.strip()
    ):
        raise ValueError("runtime model_revision must be null or a non-empty string")
    normalized["model_revision"] = (
        model_revision.strip() if isinstance(model_revision, str) else None
    )
    for field in (
        "tensor_parallel_size",
        "max_model_len",
        "batch_size",
        "max_num_batched_tokens",
        "max_num_seqs",
    ):
        normalized[field] = _positive_int(runtime, field)
    for field in ("enforce_eager", "disable_custom_all_reduce"):
        if not isinstance(runtime[field], bool):
            raise ValueError(f"runtime {field} must be boolean")
    if not isinstance(runtime["include_verbalized_confidence"], bool):
        raise ValueError("runtime include_verbalized_confidence must be boolean")

    try:
        gpu_memory_utilization = float(runtime["gpu_memory_utilization"])
        sampling_temperature = float(runtime["sampling_temperature"])
    except (TypeError, ValueError) as exc:
        raise ValueError("runtime floating-point controls are invalid") from exc
    if not math.isfinite(gpu_memory_utilization) or not (
        0.0 < gpu_memory_utilization <= 1.0
    ):
        raise ValueError("runtime gpu_memory_utilization must be finite and in (0, 1]")
    if not math.isfinite(sampling_temperature) or sampling_temperature < 0.0:
        raise ValueError("runtime sampling_temperature must be finite and nonnegative")
    normalized["gpu_memory_utilization"] = gpu_memory_utilization
    normalized["sampling_temperature"] = sampling_temperature

    seed = runtime["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed != 0:
        raise ValueError("runtime seed must equal the frozen production seed 0")
    consistency_runs = runtime["consistency_runs"]
    if (
        isinstance(consistency_runs, bool)
        or not isinstance(consistency_runs, int)
        or consistency_runs < 0
    ):
        raise ValueError("runtime consistency_runs must be a nonnegative integer")
    if runtime["consistency_schedule"] not in {"all", "extremes"}:
        raise ValueError("runtime consistency_schedule must be 'all' or 'extremes'")
    dtype = runtime["dtype"]
    if not isinstance(dtype, str) or not dtype.strip():
        raise ValueError("runtime dtype must be a non-empty string")
    normalized["dtype"] = dtype.strip()

    raw_engine_versions = runtime["engine_versions"]
    if not isinstance(raw_engine_versions, Mapping):
        raise ValueError("runtime engine_versions must be a mapping")
    if set(raw_engine_versions) != set(ENGINE_VERSION_FIELDS):
        raise ValueError(
            "runtime engine_versions must contain exactly: "
            + ", ".join(ENGINE_VERSION_FIELDS)
        )
    engine_versions: dict[str, str | None] = {}
    for field in ENGINE_VERSION_FIELDS:
        value = raw_engine_versions[field]
        if value is not None and (
            not isinstance(value, str) or not value.strip()
        ):
            raise ValueError(
                f"runtime engine_versions.{field} must be null or a non-empty string"
            )
        engine_versions[field] = value.strip() if isinstance(value, str) else None
    normalized["engine_versions"] = engine_versions

    digest = runtime["runtime_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError("runtime runtime_sha256 must be a lowercase SHA-256")
    digest_input = {
        field: normalized[field]
        for field in RUNTIME_REQUIRED_FIELDS
        if field != "runtime_sha256"
    }
    expected_digest = value_sha256(digest_input)
    if digest != expected_digest:
        raise ValueError(
            "runtime runtime_sha256 does not match the controlled runtime mapping"
        )
    normalized["runtime_sha256"] = digest
    return normalized


def observed_engine_versions() -> dict[str, str | None]:
    """Return the execution versions that affect prompt/runtime semantics."""

    def distribution_version(distribution: str) -> str | None:
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            return None

    return {
        "python": platform.python_version(),
        "torch": distribution_version("torch"),
        "transformers": distribution_version("transformers"),
        "vllm": distribution_version("vllm"),
    }


def validate_runtime_model_contract(
    runtime: Mapping[str, Any],
    *,
    model_name: str,
    require_active_engine: bool,
) -> ModelProfile:
    """Bind a validated runtime to one registry profile before model access."""

    profile = get_model_profile(model_name)
    if not profile.revision:
        raise ValueError(
            "controlled uncertainty shift requires a pinned model revision"
        )
    expected_identity = {
        "model_registry_name": profile.registry_name,
        "model_hf_name": profile.hf_model_name,
        "model_revision": profile.revision,
    }
    observed_identity = {
        field: runtime[field] for field in expected_identity
    }
    if observed_identity != expected_identity:
        raise ValueError(
            "runtime model identity does not match the selected registry profile: "
            f"observed={observed_identity!r} expected={expected_identity!r}"
        )
    if require_active_engine:
        observed_versions = observed_engine_versions()
        if runtime["engine_versions"] != observed_versions:
            raise ValueError(
                "runtime engine_versions do not match the active execution "
                f"environment: observed={observed_versions!r} "
                f"expected={runtime['engine_versions']!r}"
            )
    return profile


def _runner_runtime(runtime: Mapping[str, Any]) -> dict[str, Any]:
    return {field: runtime[field] for field in RUNNER_RUNTIME_FIELDS}


def _resolved_tokenizer_commit(tokenizer: Any) -> str | None:
    for value in (
        getattr(tokenizer, "_commit_hash", None),
        getattr(tokenizer, "commit_hash", None),
    ):
        if isinstance(value, str) and value:
            return value
    init_kwargs = getattr(tokenizer, "init_kwargs", None)
    if isinstance(init_kwargs, Mapping):
        for key in ("_commit_hash", "commit_hash", "revision"):
            value = init_kwargs.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def _single_round_trip_token_id(tokenizer: Any, surface: str) -> int:
    token_ids = tokenizer.encode(surface, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(
            f"verdict surface {surface!r} must encode to one token; got {token_ids!r}"
        )
    token_id = int(token_ids[0])
    decoded = tokenizer.decode(
        [token_id],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if decoded != surface:
        raise ValueError(
            f"verdict surface {surface!r} did not round-trip exactly; got {decoded!r}"
        )
    return token_id


def validate_verdict_contract(
    profile: ModelProfile,
    tokenizer: Any,
) -> dict[str, Any]:
    observed = {
        label: tuple(profile.verdict_token_texts.get(label, ()))
        for label in EXPECTED_VERDICT_SURFACES
    }
    if (
        set(profile.verdict_token_texts) != set(EXPECTED_VERDICT_SURFACES)
        or observed != dict(EXPECTED_VERDICT_SURFACES)
    ):
        raise ValueError(
            "controlled uncertainty shift requires the exact literal singleton "
            "verdict surfaces A, B, and T"
        )

    resolved: dict[str, list[int]] = {}
    reverse: dict[int, str] = {}
    for label, surfaces in EXPECTED_VERDICT_SURFACES.items():
        ids = [_single_round_trip_token_id(tokenizer, surface) for surface in surfaces]
        for token_id in ids:
            previous = reverse.setdefault(token_id, label)
            if previous != label:
                raise ValueError(
                    f"verdict token ID {token_id} maps to both {previous!r} and {label!r}"
                )
        resolved[label] = ids
    if len(reverse) != 3:
        raise ValueError("A, B, and T must resolve to three distinct token IDs")

    stop_ids = {
        surface: [
            int(token_id)
            for token_id in tokenizer.encode(surface, add_special_tokens=False)
        ]
        for surface in profile.stop_token_texts
    }
    return {
        "verdict_token_texts": {
            label: list(surfaces)
            for label, surfaces in EXPECTED_VERDICT_SURFACES.items()
        },
        "verdict_token_ids": resolved,
        "stop_token_ids": stop_ids,
    }


def validate_routing_contract(
    source_frame: pd.DataFrame,
    source_csv: Path,
    source_sha256: str,
    frozen_manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    expected_calibration_questions: int | None,
    expected_test_questions: int | None,
) -> dict[str, Any]:
    if frozen_manifest.get("schema_version") != ROUTING_SCHEMA_VERSION:
        raise ValueError("routing manifest must use schema version 2")
    if frozen_manifest.get("artifact_type") != ROUTING_ARTIFACT_TYPE:
        raise ValueError(
            "routing manifest artifact type is not the frozen routing package"
        )
    if frozen_manifest.get("routing_unit") != "question":
        raise ValueError("routing manifest must freeze routing_unit='question'")
    try:
        seed = int(frozen_manifest["seed"])
        calibration_fraction = float(frozen_manifest["calibration_fraction"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "routing manifest must record seed and calibration_fraction"
        ) from exc

    observed = routing_manifest(
        source_frame,
        routing_unit="question",
        seed=seed,
        calibration_fraction=calibration_fraction,
    )
    for field in (
        "routing_assignment_sha256",
        "row_counts",
        "question_counts",
    ):
        if observed[field] != frozen_manifest.get(field):
            raise ValueError(f"routing manifest {field} does not match the source CSV")

    question_counts = observed["question_counts"]
    if question_counts["overlap"] != 0:
        raise ValueError("calibration and test questions overlap")
    if question_counts["calibration"] < 1 or question_counts["test"] < 1:
        raise ValueError("both calibration and test question splits are required")
    if (
        expected_calibration_questions is not None
        and question_counts["calibration"] != expected_calibration_questions
    ):
        raise ValueError(
            "unexpected calibration question count: "
            f"{question_counts['calibration']}"
        )
    if (
        expected_test_questions is not None
        and question_counts["test"] != expected_test_questions
    ):
        raise ValueError(
            f"unexpected test question count: {question_counts['test']}"
        )

    outputs = _required_mapping(frozen_manifest.get("outputs"), name="outputs")
    output_hashes = _required_mapping(
        frozen_manifest.get("output_sha256"), name="output_sha256"
    )
    expected_output_names = {"full", "calibration", "test"}
    if set(outputs) != expected_output_names or set(output_hashes) != (
        expected_output_names
    ):
        raise ValueError(
            "routing manifest must pin exactly full/calibration/test outputs"
        )
    output_paths = {
        name: _resolve_package_output(manifest_path, outputs, name)
        for name in sorted(expected_output_names)
    }
    if output_paths["full"] != source_csv.resolve():
        raise ValueError("source CSV is not the routing manifest's full output")
    for name, output_path in output_paths.items():
        if not output_path.is_file():
            raise ValueError(f"routing package output {name} is missing")
        expected_hash = _required_sha256(
            output_hashes.get(name),
            name=f"routing manifest output_sha256.{name}",
        )
        if file_sha256(output_path) != expected_hash:
            raise ValueError(f"routing package output {name} hash does not match")
        record = _required_mapping(outputs[name], name=f"outputs.{name}")
        expected_rows = observed["row_counts"][
            "total" if name == "full" else name
        ]
        if record.get("rows") != expected_rows:
            raise ValueError(f"routing manifest outputs.{name}.rows does not match")
    if output_hashes["full"] != source_sha256:
        raise ValueError("routing manifest full-output SHA-256 does not match source CSV")

    full_text = pd.read_csv(source_csv, dtype=str, keep_default_na=False)
    calibration_text = pd.read_csv(
        output_paths["calibration"], dtype=str, keep_default_na=False
    )
    test_text = pd.read_csv(output_paths["test"], dtype=str, keep_default_na=False)
    if list(full_text.columns) != list(calibration_text.columns) or list(
        full_text.columns
    ) != list(test_text.columns):
        raise ValueError("routing package output columns do not match")
    expected_calibration = full_text.loc[
        full_text["routing_split"] == "calibration"
    ].reset_index(drop=True)
    expected_test = full_text.loc[
        full_text["routing_split"] == "test"
    ].reset_index(drop=True)
    if not calibration_text.equals(expected_calibration) or not test_text.equals(
        expected_test
    ):
        raise ValueError("routing split outputs do not partition the full CSV")

    source = _required_mapping(frozen_manifest.get("source"), name="source")
    original_path = Path(str(source.get("path") or "")).resolve()
    if not original_path.is_file():
        raise ValueError("routing manifest original source is missing")
    original_sha = _required_sha256(
        source.get("sha256"), name="routing manifest source.sha256"
    )
    if file_sha256(original_path) != original_sha:
        raise ValueError("routing manifest original source SHA-256 does not match")
    lineage = _required_mapping(
        source.get("dataset_lineage"), name="source.dataset_lineage"
    )
    if not lineage:
        raise ValueError("routing manifest dataset lineage must not be empty")
    original_header, _ = _read_csv_cells(original_path)
    if source.get("columns") != original_header:
        raise ValueError("routing manifest source columns do not match original")

    preservation = _required_mapping(
        frozen_manifest.get("content_preservation"), name="content_preservation"
    )
    expected_preserved_columns = [
        column for column in original_header if column != "routing_split"
    ]
    if preservation.get("preserved_columns") != expected_preserved_columns:
        raise ValueError("routing manifest preserved columns do not match original")
    if preservation.get("recomputed_columns") != ["routing_split"]:
        raise ValueError("routing manifest must recompute only routing_split")
    if preservation.get("row_order_preserved") is not True:
        raise ValueError("routing manifest must assert preserved row order")
    source_content_hash = _required_sha256(
        source.get("rows_without_routing_sha256"),
        name="routing manifest source.rows_without_routing_sha256",
    )
    preserved_content_hash = _required_sha256(
        preservation.get("rows_without_routing_sha256"),
        name=(
            "routing manifest "
            "content_preservation.rows_without_routing_sha256"
        ),
    )
    if source_content_hash != preserved_content_hash:
        raise ValueError("routing manifest content-preservation hashes disagree")
    if _rows_without_routing_sha256(original_path) != source_content_hash:
        raise ValueError("routing manifest source content hash does not match")
    if _rows_without_routing_sha256(source_csv) != preserved_content_hash:
        raise ValueError("routed full CSV changed source content or row order")

    eligibility = _required_mapping(
        frozen_manifest.get("eligibility"), name="eligibility"
    )
    if eligibility.get("schema_version") != 1:
        raise ValueError("routing manifest eligibility must use schema version 1")
    if (
        eligibility.get("eligibility_contract")
        != POSITION_PAIR_ELIGIBILITY_CONTRACT
    ):
        raise ValueError("routing manifest eligibility contract does not match")
    _required_sha256(
        eligibility.get("eligibility_sha256"),
        name="routing manifest eligibility.eligibility_sha256",
    )
    _required_mapping(frozen_manifest.get("counts"), name="counts")
    return observed


def _validate_stage_a_summary_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    stage_a_prompts: Sequence[PlannedPrompt],
    model_revision: str,
    runtime: Mapping[str, Any],
    verdict_contract: Mapping[str, Any],
) -> tuple[CleanPairSummary, ...]:
    expected = {item.planned.pair_key: item for item in stage_a_prompts}
    if len(expected) != len(stage_a_prompts):
        raise AssertionError("Stage A prompt plan has duplicate pair keys")
    if len(rows) != len(expected):
        raise ValueError(
            "Stage A summary is incomplete: "
            f"expected={len(expected)} observed={len(rows)}"
        )

    seen_pair_keys: set[str] = set()
    seen_record_ids: set[str] = set()
    spec_hashes: set[str] = set()
    runner_runtime = _runner_runtime(runtime)
    required_fields = (
        "record_id",
        "clean_record_id",
        "pair_identity_key",
        "pair_key",
        "condition_group_id",
        "ordering_twin_key",
        "ordering",
        "model_name",
        "model_revision",
        "input_file_hash",
        "spec_hash",
        "question_id",
        "source_row_index",
        "routing_split",
        "judge_output_parser_version",
        "verbalized_output_parser_version",
        "logprobs_mode",
        "verdict_token_texts",
        "verdict_token_ids",
        "max_num_batched_tokens",
        "max_num_seqs",
        "inference_runtime",
        "human_winner",
        "clean_verdict",
        "verdict",
        "clean_tie",
    )
    for row_index, row in enumerate(rows):
        missing = [field for field in required_fields if field not in row]
        if missing:
            raise ValueError(
                "Stage A summary row is missing fields: "
                f"row_index={row_index} fields={','.join(missing)}"
            )
        pair_key = str(row["pair_key"])
        if pair_key in seen_pair_keys:
            raise ValueError(f"Stage A summary has duplicate pair_key {pair_key!r}")
        seen_pair_keys.add(pair_key)
        item = expected.get(pair_key)
        if item is None:
            raise ValueError(f"Stage A summary has unexpected pair_key {pair_key!r}")

        record_id = str(row["record_id"] or "")
        if not record_id or str(row["clean_record_id"] or "") != record_id:
            raise ValueError(
                f"Stage A summary {pair_key!r} has invalid clean record linkage"
            )
        if record_id in seen_record_ids:
            raise ValueError(f"Stage A summary has duplicate record_id {record_id!r}")
        seen_record_ids.add(record_id)
        spec_hash = str(row["spec_hash"] or "")
        if not spec_hash:
            raise ValueError(f"Stage A summary {pair_key!r} has blank spec_hash")
        spec_hashes.add(spec_hash)

        planned = item.planned
        condition = planned.condition
        expected_ordering = normalize_ordering(condition.ordering or "")
        expected_human = normalize_verdict(condition.metadata["human_winner"])
        expected_question = str(item.example.question_id)
        expected_source_index = str(condition.metadata["source_row_index"])
        checks = {
            "pair_identity_key": (
                str(row["pair_identity_key"]),
                planned.pair_identity_key,
            ),
            "condition_group_id": (
                str(row["condition_group_id"]),
                planned.condition_group_id,
            ),
            "ordering_twin_key": (
                str(row["ordering_twin_key"]),
                planned.ordering_twin_key,
            ),
            "model_name": (str(row["model_name"]), planned.model_name),
            "model_revision": (str(row["model_revision"]), model_revision),
            "input_file_hash": (
                str(row["input_file_hash"]),
                planned.input_file_hash,
            ),
            "question_id": (str(row["question_id"]), expected_question),
            "source_row_index": (
                str(row["source_row_index"]),
                expected_source_index,
            ),
            "routing_split": (
                str(row["routing_split"]).strip().lower(),
                item.routing_split,
            ),
            "judge_output_parser_version": (
                str(row["judge_output_parser_version"]),
                JUDGE_OUTPUT_PARSER_VERSION,
            ),
            "verbalized_output_parser_version": (
                str(row["verbalized_output_parser_version"]),
                VERBALIZED_OUTPUT_PARSER_VERSION,
            ),
            "logprobs_mode": (
                str(row["logprobs_mode"]),
                CONSTRAINED_LOGPROBS_MODE,
            ),
        }
        for field, (observed, expected_value) in checks.items():
            if observed != expected_value:
                raise ValueError(
                    f"Stage A summary {pair_key!r} {field} does not match: "
                    f"observed={observed!r} expected={expected_value!r}"
                )
        if normalize_ordering(row["ordering"]) != expected_ordering:
            raise ValueError(
                f"Stage A summary {pair_key!r} ordering does not match"
            )
        if normalize_verdict(row["human_winner"]) != expected_human:
            raise ValueError(
                f"Stage A summary {pair_key!r} human_winner does not match"
            )
        clean_verdict = normalize_verdict(row["clean_verdict"])
        if normalize_verdict(row["verdict"]) != clean_verdict:
            raise ValueError(
                f"Stage A summary {pair_key!r} verdict fields disagree"
            )
        if row["clean_tie"] is not (clean_verdict == VerdictLabel.TIE):
            raise ValueError(
                f"Stage A summary {pair_key!r} clean_tie does not match verdict"
            )
        if row["verdict_token_texts"] != verdict_contract["verdict_token_texts"]:
            raise ValueError(
                f"Stage A summary {pair_key!r} verdict token texts do not match"
            )
        if row["verdict_token_ids"] != verdict_contract["verdict_token_ids"]:
            raise ValueError(
                f"Stage A summary {pair_key!r} verdict token IDs do not match"
            )
        if row["max_num_batched_tokens"] != runtime["max_num_batched_tokens"]:
            raise ValueError(
                f"Stage A summary {pair_key!r} max_num_batched_tokens does not match"
            )
        if row["max_num_seqs"] != runtime["max_num_seqs"]:
            raise ValueError(
                f"Stage A summary {pair_key!r} max_num_seqs does not match"
            )
        if row["inference_runtime"] != runner_runtime:
            raise ValueError(
                f"Stage A summary {pair_key!r} inference_runtime does not match"
            )

    if seen_pair_keys != set(expected):
        raise ValueError("Stage A summary does not cover the complete prompt plan")
    if len(spec_hashes) != 1:
        raise ValueError("Stage A summary rows do not share one frozen spec_hash")
    return clean_summaries_from_rows(rows)


def _target_realization(
    planned: PlannedCondition,
    target: VerdictLabel,
) -> PlannedCondition:
    condition = planned.condition
    if condition.dose is None:
        raise AssertionError("Stage B condition is missing its dose")
    metadata = dict(condition.metadata)
    metadata["preflight_target_realization"] = target.value
    realized_condition = condition.model_copy(
        update={
            "cue_target": target.value,
            "cue_text": build_social_cue(
                family=condition.bias_type,
                target=target,
                dose=condition.dose,
            ),
            "metadata": metadata,
        }
    )
    return replace(planned, condition=realized_condition)


def build_stage_plans(
    *,
    source_csv: Path,
    source_sha256: str,
    canonical_model_name: str,
    model_revision: str,
    runtime: Mapping[str, Any],
    verdict_contract: Mapping[str, Any],
    stage_a_summary_rows: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[tuple[PlannedPrompt, ...], tuple[PlannedPrompt, ...], dict[str, Any]]:
    pairs, eligibility = load_position_pairs_with_eligibility(source_csv)
    source_rows = len(pd.read_csv(source_csv))
    if eligibility.raw_row_count != source_rows:
        raise ValueError(
            "source loader did not examine every frozen CSV row: "
            f"examined={eligibility.raw_row_count} source_rows={source_rows}"
        )
    if eligibility.eligible_pair_count != len(pairs):
        raise AssertionError("pair-loader eligible count differs from loaded pairs")

    pair_inputs: list[StageAPairInput] = []
    pairs_by_identity: dict[str, PositionPair] = {}
    for pair in pairs:
        original = pair.original
        pair_input = StageAPairInput(
            dataset_name=source_csv.name,
            input_file_hash=source_sha256,
            source_row_index=original.metadata["source_row_index"],
            question_id=(
                original.metadata.get("question_cluster_id")
                or original.question_id
            ),
            model_name=canonical_model_name,
            human_winner=original.human_winner or VerdictLabel.TIE,
            turn=original.metadata.get("turn"),
            response_a_id=original.candidates["A"].response_id,
            response_b_id=original.candidates["B"].response_id,
        )
        identity = make_pair_identity_key(
            dataset_name=pair_input.dataset_name,
            input_file_hash=pair_input.input_file_hash,
            source_row_index=pair_input.source_row_index,
            question_id=pair_input.question_id,
            turn=pair_input.turn,
            response_a_id=pair_input.response_a_id,
            response_b_id=pair_input.response_b_id,
        )
        if identity in pairs_by_identity:
            raise ValueError(f"duplicate source pair identity {identity}")
        pair_inputs.append(pair_input)
        pairs_by_identity[identity] = pair

    stage_a_plan = generate_stage_a_conditions(pair_inputs)
    if stage_a_plan.issues:
        raise ValueError(
            f"Stage A planning produced {len(stage_a_plan.issues)} issue(s)"
        )
    if len(stage_a_plan.conditions) != 2 * len(pairs):
        raise ValueError("Stage A did not enumerate exactly clean AB and BA")

    stage_a_prompts: list[PlannedPrompt] = []
    clean_summaries: list[CleanPairSummary] = []
    for planned in stage_a_plan.conditions:
        pair = pairs_by_identity[planned.pair_identity_key]
        ordering = normalize_ordering(planned.condition.ordering or "")
        example = pair.original if ordering == PairOrdering.AB else pair.swapped
        routing_split = str(example.metadata.get("routing_split") or "").lower()
        if routing_split not in {"calibration", "test"}:
            raise ValueError(
                f"source pair {planned.pair_identity_key} has invalid routing split "
                f"{routing_split!r}"
            )
        stage_a_prompts.append(
            PlannedPrompt(
                stage="stage_a",
                routing_split=routing_split,
                planned=planned,
                example=example,
            )
        )
        human_winner = str(planned.condition.metadata["human_winner"])
        clean_summaries.append(
            CleanPairSummary(
                pair_identity_key=planned.pair_identity_key,
                pair_key=planned.pair_key,
                ordering=ordering,
                ordering_twin_key=planned.ordering_twin_key,
                model_name=planned.model_name,
                input_file_hash=planned.input_file_hash,
                clean_record_id=f"preflight:{planned.pair_key}",
                # A binary provisional reference or a tie fallback enumerates
                # the same two A/B cue targets across congruency directions.
                clean_verdict=human_winner,
                human_winner=human_winner,
                routing_split=routing_split,
            )
        )

    exact_post_stage_a = stage_a_summary_rows is not None
    if stage_a_summary_rows is not None:
        clean_summaries = list(
            _validate_stage_a_summary_rows(
                stage_a_summary_rows,
                stage_a_prompts=stage_a_prompts,
                model_revision=model_revision,
                runtime=runtime,
                verdict_contract=verdict_contract,
            )
        )

    test_summaries = tuple(
        summary for summary in clean_summaries if summary.routing_split == "test"
    )
    if not test_summaries:
        raise ValueError("no test summaries are available for Stage B")
    eligible_test_pairs = eligibility.routing_counts["eligible_pairs"].get(
        "test",
        0,
    )
    if len(test_summaries) != 2 * eligible_test_pairs:
        raise ValueError(
            "Stage B source summaries are not exactly AB and BA for every "
            "eligible test pair"
        )
    stage_b_plan = generate_stage_b_conditions(test_summaries)
    fatal_issues = [
        issue
        for issue in stage_b_plan.issues
        if issue.code != "clean_and_human_tie"
    ]
    if fatal_issues:
        raise ValueError(
            f"Stage B planning produced {len(fatal_issues)} fatal issue(s)"
        )
    expected_stage_b_conditions = 16 * len(test_summaries)
    if len(stage_b_plan.conditions) != expected_stage_b_conditions:
        raise ValueError(
            "Stage B did not enumerate the complete family/direction/dose grid: "
            f"expected={expected_stage_b_conditions} "
            f"observed={len(stage_b_plan.conditions)}"
        )
    if len(stage_b_plan.conditions) != 32 * eligible_test_pairs:
        raise ValueError(
            "Stage B is not the complete test-only grid: "
            f"expected={32 * eligible_test_pairs} "
            f"observed={len(stage_b_plan.conditions)}"
        )

    stage_b_prompts: list[PlannedPrompt] = []
    for planned in stage_b_plan.conditions:
        pair = pairs_by_identity[planned.pair_identity_key]
        ordering = normalize_ordering(planned.condition.ordering or "")
        example = pair.original if ordering == PairOrdering.AB else pair.swapped
        if exact_post_stage_a:
            actual_target = normalize_verdict(planned.condition.cue_target or "")
            if actual_target not in {VerdictLabel.A, VerdictLabel.B}:
                raise AssertionError("Stage B planner emitted a non-binary cue target")
            stage_b_prompts.append(
                PlannedPrompt(
                    stage="stage_b",
                    routing_split="test",
                    planned=planned,
                    example=example,
                    target_realization=actual_target.value,
                )
            )
        else:
            for target in (VerdictLabel.A, VerdictLabel.B):
                stage_b_prompts.append(
                    PlannedPrompt(
                        stage="stage_b",
                        routing_split="test",
                        planned=_target_realization(planned, target),
                        example=example,
                        target_realization=target.value,
                    )
                )

    eligible_questions: dict[str, str] = {}
    for pair in pairs:
        metadata = pair.original.metadata
        question_id = str(
            metadata.get("question_cluster_id") or pair.original.question_id
        )
        routing_split = str(metadata.get("routing_split") or "").lower()
        previous = eligible_questions.setdefault(question_id, routing_split)
        if previous != routing_split:
            raise ValueError(
                f"eligible question {question_id!r} occurs in both routing splits"
            )
    eligible_question_values = list(eligible_questions.values())

    return (
        tuple(stage_a_prompts),
        tuple(stage_b_prompts),
        {
            "source_pairs": len(pairs),
            "eligibility": eligibility.to_dict(),
            "eligible_calibration_pairs": eligibility.routing_counts[
                "eligible_pairs"
            ].get("calibration", 0),
            "eligible_test_pairs": eligible_test_pairs,
            "eligible_question_counts": {
                "total": len(eligible_question_values),
                "calibration": sum(
                    value == "calibration" for value in eligible_question_values
                ),
                "test": sum(value == "test" for value in eligible_question_values),
                "overlap": 0,
            },
            "stage_b_plan_mode": (
                "exact_post_stage_a"
                if exact_post_stage_a
                else "provisional_structural_pre_stage_a"
            ),
            "stage_b_release_authorized": exact_post_stage_a,
            "exact_post_stage_a_required": not exact_post_stage_a,
            "stage_b_scientific_condition_count": len(stage_b_plan.conditions),
            "stage_b_rendered_target_realization_count": len(stage_b_prompts),
            "stage_b_nonfatal_clean_and_human_tie_issues": sum(
                issue.code == "clean_and_human_tie"
                for issue in stage_b_plan.issues
            ),
        },
    )


def validate_eligibility_contract(
    plan_metadata: Mapping[str, Any],
    frozen_manifest: Mapping[str, Any],
    observed_routing: Mapping[str, Any],
) -> None:
    frozen_eligibility = _required_mapping(
        frozen_manifest.get("eligibility"), name="eligibility"
    )
    observed_eligibility = _required_mapping(
        plan_metadata.get("eligibility"), name="observed eligibility"
    )
    if dict(frozen_eligibility) != dict(observed_eligibility):
        raise ValueError("routing manifest eligibility audit does not match source")
    frozen_counts = _required_mapping(frozen_manifest.get("counts"), name="counts")
    routing_counts = _required_mapping(
        observed_eligibility.get("routing_counts"),
        name="observed eligibility.routing_counts",
    )
    eligible_pair_counts = _required_mapping(
        routing_counts.get("eligible_pairs"),
        name="observed eligibility.routing_counts.eligible_pairs",
    )
    skipped_row_counts = _required_mapping(
        routing_counts.get("skipped_rows"),
        name="observed eligibility.routing_counts.skipped_rows",
    )
    expected_counts = {
        "raw_rows": dict(observed_routing["row_counts"]),
        "raw_questions": dict(observed_routing["question_counts"]),
        "eligible_pairs": {
            "total": int(plan_metadata["source_pairs"]),
            "calibration": int(eligible_pair_counts.get("calibration", 0)),
            "test": int(eligible_pair_counts.get("test", 0)),
        },
        "eligible_questions": dict(plan_metadata["eligible_question_counts"]),
        "skipped_rows": {
            "total": int(observed_eligibility["skipped_row_count"]),
            "calibration": int(skipped_row_counts.get("calibration", 0)),
            "test": int(skipped_row_counts.get("test", 0)),
        },
    }
    if dict(frozen_counts) != expected_counts:
        raise ValueError(
            "routing manifest raw/eligible/skipped counts do not match source"
        )


def _template_token_ids(
    profile: ModelProfile,
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
) -> list[int]:
    normalized = profile.normalize_messages(messages)
    token_ids = tokenizer.apply_chat_template(
        normalized,
        tokenize=True,
        add_generation_prompt=True,
    )
    if not isinstance(token_ids, Sequence) or isinstance(token_ids, (str, bytes)):
        raise TypeError("tokenized chat template must return a sequence of token IDs")
    result = [int(token_id) for token_id in token_ids]
    if profile.assistant_prefill:
        result.extend(
            int(token_id)
            for token_id in tokenizer.encode(
                profile.assistant_prefill,
                add_special_tokens=False,
            )
        )
    return result


def audit_prompt(
    planned_prompt: PlannedPrompt,
    *,
    profile: ModelProfile,
    tokenizer: Any,
    output_mode: OutputMode,
    max_model_len: int,
    generation_headroom: int,
) -> PromptAudit:
    messages = build_social_cue_messages(
        example=planned_prompt.example,
        condition=planned_prompt.planned.condition,
        output_mode=output_mode,
    )
    rendered = profile.render_prompt(tokenizer, messages)
    transport_ids = [
        int(token_id)
        for token_id in tokenizer.encode(rendered, add_special_tokens=False)
    ]
    template_ids = _template_token_ids(profile, tokenizer, messages)
    if transport_ids != template_ids:
        raise ValueError(
            "rendered prompt text did not re-encode to the chat-template token IDs: "
            f"{planned_prompt.planned.condition.variant_id}"
        )
    if len(transport_ids) + generation_headroom > max_model_len:
        raise ValueError(
            "prompt exceeds configured context with generation headroom: "
            f"variant={planned_prompt.planned.condition.variant_id!r} "
            f"input_tokens={len(transport_ids)} headroom={generation_headroom} "
            f"max_model_len={max_model_len}"
        )

    package = build_social_cue_prompt_package(
        example=planned_prompt.example,
        condition=planned_prompt.planned.condition,
        output_mode=output_mode,
        renderer=lambda _: rendered,
    )
    if package.prompt_text != rendered:
        raise AssertionError("prompt package changed the audited rendered text")
    condition = planned_prompt.planned.condition
    family = enum_value(condition.bias_type)
    direction = enum_value(condition.cue_congruency)
    ordering = normalize_ordering(condition.ordering or "").value
    key = "\0".join(
        (
            planned_prompt.stage,
            planned_prompt.planned.pair_key,
            condition.variant_id,
            planned_prompt.target_realization or "actual",
            output_mode.value,
        )
    )
    return PromptAudit(
        key=key,
        stage=planned_prompt.stage,
        routing_split=planned_prompt.routing_split,
        family=family,
        direction=direction,
        dose=condition.dose,
        ordering=ordering,
        output_mode=output_mode.value,
        target_realization=planned_prompt.target_realization,
        prompt_sha256=text_sha256(rendered),
        planner_prompt_hash=package.prompt_hash,
        token_ids_sha256=value_sha256(transport_ids),
        input_tokens=len(transport_ids),
    )


def _count_values(audits: Sequence[PromptAudit], field: str) -> dict[str, int]:
    counts = Counter(str(getattr(audit, field)) for audit in audits)
    return dict(sorted(counts.items()))


def prompt_set_report(audits: Sequence[PromptAudit]) -> dict[str, Any]:
    keys = [audit.key for audit in audits]
    if len(keys) != len(set(keys)):
        raise ValueError("prompt plan contains duplicate condition/output-mode keys")
    entries = [
        {
            "key": audit.key,
            "target_realization": audit.target_realization,
            "prompt_sha256": audit.prompt_sha256,
            "planner_prompt_hash": audit.planner_prompt_hash,
            "token_ids_sha256": audit.token_ids_sha256,
            "input_tokens": audit.input_tokens,
        }
        for audit in sorted(audits, key=lambda item: item.key)
    ]
    longest = max(audits, key=lambda item: (item.input_tokens, item.key))
    hashes_by_stage = {
        stage: value_sha256(
            [entry for entry in entries if entry["key"].startswith(f"{stage}\0")]
        )
        for stage in ("stage_a", "stage_b")
    }
    return {
        "rendered_prompt_count": len(audits),
        "prompt_set_sha256": value_sha256(entries),
        "prompt_set_sha256_by_stage": hashes_by_stage,
        "counts_by_stage": _count_values(audits, "stage"),
        "counts_by_routing_split": _count_values(audits, "routing_split"),
        "counts_by_family": _count_values(audits, "family"),
        "counts_by_direction": _count_values(audits, "direction"),
        "counts_by_dose": _count_values(audits, "dose"),
        "counts_by_ordering": _count_values(audits, "ordering"),
        "counts_by_output_mode": _count_values(audits, "output_mode"),
        "counts_by_target_realization": _count_values(
            audits, "target_realization"
        ),
        "maximum_input_tokens": longest.input_tokens,
        "longest_prompt": {
            "key": longest.key,
            "prompt_sha256": longest.prompt_sha256,
            "input_tokens": longest.input_tokens,
        },
    }


def resolve_output_modes(
    runtime: Mapping[str, Any],
    requested: Sequence[str] | None,
) -> tuple[OutputMode, ...]:
    raw_modes = list(requested or [OutputMode.CHOICE_ONLY.value])
    if requested is None and runtime.get("include_verbalized_confidence") is True:
        raw_modes.append(OutputMode.CHOICE_WITH_CONFIDENCE.value)
    modes: list[OutputMode] = []
    for value in raw_modes:
        mode = OutputMode(value)
        if mode not in modes:
            modes.append(mode)
    return tuple(modes)


def build_preflight_report(
    *,
    source_csv: Path,
    routing_manifest_path: Path,
    runtime_path: Path,
    model_name: str,
    tokenizer: Any,
    max_model_len: int | None = None,
    generation_headroom: int = DEFAULT_GENERATION_HEADROOM,
    expected_calibration_questions: int | None = 40,
    expected_test_questions: int | None = 40,
    output_modes: Sequence[str] | None = None,
    stage_a_summary_path: Path | None = None,
) -> dict[str, Any]:
    if generation_headroom < 1:
        raise ValueError("generation headroom must be positive")
    source_sha = file_sha256(source_csv)
    frozen_routing = read_json_mapping(routing_manifest_path, name="routing manifest")
    runtime = validate_runtime_contract(
        read_json_mapping(runtime_path, name="runtime mapping")
    )
    resolved_max_model_len = (
        int(max_model_len)
        if max_model_len is not None
        else int(runtime["max_model_len"])
    )
    if resolved_max_model_len < 1:
        raise ValueError("max model length must be positive")
    if (
        runtime.get("max_model_len") is not None
        and int(runtime["max_model_len"]) != resolved_max_model_len
    ):
        raise ValueError("runtime max_model_len differs from the configured preflight")

    source_frame = pd.read_csv(source_csv, dtype=str, keep_default_na=False)
    observed_routing = validate_routing_contract(
        source_frame,
        source_csv,
        source_sha,
        frozen_routing,
        manifest_path=routing_manifest_path,
        expected_calibration_questions=expected_calibration_questions,
        expected_test_questions=expected_test_questions,
    )
    profile = validate_runtime_model_contract(
        runtime,
        model_name=model_name,
        require_active_engine=False,
    )
    if not profile.supports_text_prompt_transport:
        raise ValueError("model profile has no validated text prompt transport")
    resolved_tokenizer_commit = _resolved_tokenizer_commit(tokenizer)
    if resolved_tokenizer_commit != profile.revision:
        raise ValueError(
            "tokenizer resolved commit does not match the pinned model revision: "
            f"observed={resolved_tokenizer_commit!r} expected={profile.revision!r}"
        )
    verdict_contract = validate_verdict_contract(profile, tokenizer)

    stage_a_summary_rows = (
        read_jsonl_mappings(stage_a_summary_path, name="Stage A summary")
        if stage_a_summary_path is not None
        else None
    )

    stage_a, stage_b, plan_metadata = build_stage_plans(
        source_csv=source_csv,
        source_sha256=source_sha,
        canonical_model_name=profile.hf_model_name,
        model_revision=profile.revision,
        runtime=runtime,
        verdict_contract=verdict_contract,
        stage_a_summary_rows=stage_a_summary_rows,
    )
    validate_eligibility_contract(plan_metadata, frozen_routing, observed_routing)
    modes = resolve_output_modes(runtime, output_modes)
    audits = tuple(
        audit_prompt(
            planned_prompt,
            profile=profile,
            tokenizer=tokenizer,
            output_mode=mode,
            max_model_len=resolved_max_model_len,
            generation_headroom=generation_headroom,
        )
        for planned_prompt in (*stage_a, *stage_b)
        for mode in modes
    )
    prompt_report = prompt_set_report(audits)
    stage_b_scientific_count = int(
        plan_metadata["stage_b_scientific_condition_count"]
    )
    condition_counts = {
        "stage_a": len(stage_a),
        "stage_b": stage_b_scientific_count,
        "stage_a_calibration": sum(
            item.routing_split == "calibration" for item in stage_a
        ),
        "stage_a_test": sum(item.routing_split == "test" for item in stage_a),
        "stage_b_test": stage_b_scientific_count,
        "stage_b_rendered_target_realizations": len(stage_b),
    }
    condition_plan = [
        {
            "stage": item.stage,
            "routing_split": item.routing_split,
            "pair_key": item.planned.pair_key,
            "condition_group_id": item.planned.condition_group_id,
            "variant_id": item.planned.condition.variant_id,
            "target_realization": item.target_realization,
        }
        for item in sorted(
            (*stage_a, *stage_b),
            key=lambda value: (
                value.stage,
                value.planned.pair_key,
                value.planned.condition.variant_id,
                value.target_realization or "",
            ),
        )
    ]
    model_contract = {
        "registry_name": profile.registry_name,
        "model_name": profile.hf_model_name,
        "revision": profile.revision,
        "family": profile.family,
        "chat_template": profile.chat_template.value,
        "assistant_prefill": profile.assistant_prefill,
        "stop_token_texts": list(profile.stop_token_texts),
        "trust_remote_code": profile.trust_remote_code,
        **verdict_contract,
    }
    return {
        "schema_version": 1,
        "status": "complete",
        "passed": True,
        "inference_performed": False,
        "excluded_methods": ["BPE", "SCOPE"],
        "scope": {
            "model_count": 1,
            "model_estimands_pooled": False,
            "stage_b_routing_split": "test",
            "families": ["authority", "bandwagon"],
            "authority_doses": list(AUTHORITY_DOSES),
            "bandwagon_doses": list(BANDWAGON_DOSES),
            "directions": ["congruent", "incongruent"],
            "orderings": ["ab", "ba"],
            "output_modes": [mode.value for mode in modes],
        },
        "source": {
            "path": str(source_csv),
            "sha256": source_sha,
            "row_count": len(source_frame),
        },
        "routing": {
            "manifest_path": str(routing_manifest_path),
            "manifest_file_sha256": file_sha256(routing_manifest_path),
            "routing_assignment_sha256": observed_routing[
                "routing_assignment_sha256"
            ],
            "row_counts": observed_routing["row_counts"],
            "question_counts": observed_routing["question_counts"],
            "seed": observed_routing["seed"],
            "calibration_fraction": observed_routing["calibration_fraction"],
        },
        "model": {
            **model_contract,
            "model_contract_sha256": value_sha256(model_contract),
            "model_revision_sha256": text_sha256(
                f"{profile.hf_model_name}@{profile.revision}"
            ),
            "tokenizer_class": (
                f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}"
            ),
            "resolved_tokenizer_commit": resolved_tokenizer_commit,
        },
        "runtime": {
            "path": str(runtime_path),
            "file_sha256": file_sha256(runtime_path),
            "inference_runtime": runtime,
            "inference_runtime_sha256": value_sha256(runtime),
            "max_model_len": resolved_max_model_len,
            "generation_headroom_tokens": generation_headroom,
        },
        "plan": {
            **plan_metadata,
            "stage_a_summary": (
                {
                    "mode": "exact_post_stage_a",
                    "path": str(stage_a_summary_path),
                    "file_sha256": file_sha256(stage_a_summary_path),
                    "row_count": len(stage_a_summary_rows or ()),
                }
                if stage_a_summary_path is not None
                else {
                    "mode": "not_supplied",
                    "row_count": 0,
                }
            ),
            "condition_counts": condition_counts,
            "condition_plan_sha256": value_sha256(condition_plan),
            "condition_plan_status": plan_metadata["stage_b_plan_mode"],
            **prompt_report,
            "actual_stage_b_prompt_set_sha256": (
                prompt_report["prompt_set_sha256_by_stage"]["stage_b"]
                if plan_metadata["stage_b_release_authorized"]
                else None
            ),
            "provisional_structural_stage_b_prompt_set_sha256": (
                None
                if plan_metadata["stage_b_release_authorized"]
                else prompt_report["prompt_set_sha256_by_stage"]["stage_b"]
            ),
            "context_headroom_passed": True,
            "text_transport_match_count": len(audits),
        },
        "release_gate": {
            "stage_a_authorized": True,
            "stage_b_authorized": plan_metadata["stage_b_release_authorized"],
            "exact_post_stage_a_required": plan_metadata[
                "exact_post_stage_a_required"
            ],
            "provisional_stage_b_hashes_must_not_be_released": not plan_metadata[
                "stage_b_release_authorized"
            ],
        },
        "issues": [],
    }


def load_tokenizer(args: argparse.Namespace) -> Any:
    from transformers import AutoTokenizer

    profile = get_model_profile(args.model_name)
    token: bool | None = True if args.require_authentication else None
    return AutoTokenizer.from_pretrained(
        profile.hf_model_name,
        revision=profile.revision,
        cache_dir=args.cache_dir,
        local_files_only=not args.allow_download,
        trust_remote_code=profile.trust_remote_code,
        token=token,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--routing-manifest", type=Path, required=True)
    parser.add_argument("--runtime-json", type=Path, required=True)
    parser.add_argument(
        "--stage-a-summary",
        type=Path,
        help=(
            "Complete Stage A clean pair-summary JSONL. When supplied, validates "
            "the exact Stage A/model/runtime/routing contract and hashes the "
            "actual Stage B prompts. Without it, Stage B hashes are structural "
            "and explicitly prohibited from release."
        ),
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--require-authentication", action="store_true")
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument(
        "--generation-headroom-tokens",
        type=int,
        default=DEFAULT_GENERATION_HEADROOM,
    )
    parser.add_argument("--expected-calibration-questions", type=int, default=40)
    parser.add_argument("--expected-test-questions", type=int, default=40)
    parser.add_argument(
        "--output-mode",
        action="append",
        choices=tuple(mode.value for mode in OutputMode),
        help=(
            "Prompt output mode to audit; repeat as needed. If omitted, audits "
            "choice_only and also choice_with_confidence when frozen runtime "
            "requests verbalized confidence."
        ),
    )
    return parser


def write_exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output_path.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_path}")
    inputs = (
        args.source_csv,
        args.routing_manifest,
        args.runtime_json,
        *((args.stage_a_summary,) if args.stage_a_summary is not None else ()),
    )
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing preflight input(s): " + ", ".join(missing))
    if args.output_path.resolve() in {path.resolve() for path in inputs}:
        raise ValueError("output path must differ from every input path")

    try:
        runtime = validate_runtime_contract(
            read_json_mapping(args.runtime_json, name="runtime mapping")
        )
        validate_runtime_model_contract(
            runtime,
            model_name=args.model_name,
            require_active_engine=True,
        )
        tokenizer = load_tokenizer(args)
        payload = build_preflight_report(
            source_csv=args.source_csv,
            routing_manifest_path=args.routing_manifest,
            runtime_path=args.runtime_json,
            model_name=args.model_name,
            tokenizer=tokenizer,
            max_model_len=args.max_model_len,
            generation_headroom=args.generation_headroom_tokens,
            expected_calibration_questions=args.expected_calibration_questions,
            expected_test_questions=args.expected_test_questions,
            output_modes=args.output_mode,
            stage_a_summary_path=args.stage_a_summary,
        )
    except Exception as error:
        payload = {
            "schema_version": 1,
            "status": "failed",
            "passed": False,
            "inference_performed": False,
            "model_registry_name": args.model_name,
            "error_type": type(error).__name__,
            "error": sanitize_exception_text(error),
            "issues": ["controlled uncertainty-shift preflight failed"],
        }
    write_exclusive_json(args.output_path, payload)
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
