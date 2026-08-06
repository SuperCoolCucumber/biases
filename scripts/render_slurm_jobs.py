from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import re
import shlex
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from biases.models import get_model_profile


DEFAULT_ARTIFACT_ROOT_EXPR = "${REPO_DIR}/artifacts"
SILENT_BIAS_TEMPLATE_PATH = (
    Path(__file__).resolve().parents[1]
    / "slurm"
    / "templates"
    / "silent_bias_job.slurm"
)
SILENT_BIAS_RUNTIME_FIELDS = (
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
ENGINE_VERSION_FIELDS = ("python", "torch", "transformers", "vllm")
ROUTING_ARTIFACT_TYPE = "frozen_question_disjoint_routing_package"
STAGE_A_COMMAND = "run-silent-bias-clean"
STAGE_B_COMMAND = "run-silent-bias-cued"
STAGE_A_SUMMARY_FILENAME = "silent_bias_stage_a_pair_summary.jsonl"


@dataclass(frozen=True)
class ModelSpec:
    slug: str
    model_name: str
    gpus: int
    mem: str
    tensor_parallel_size: int
    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.9
    enforce_eager: bool = False
    disable_custom_all_reduce: bool = False
    max_model_len: int = 8192
    batch_size: int = 64
    max_num_batched_tokens: int | None = None
    max_num_seqs: int | None = None


@dataclass(frozen=True)
class FrozenRuntimeContract:
    path: Path
    file_sha256: str
    embedded_sha256: str
    canonical_sha256: str
    values: Mapping[str, Any]


@dataclass(frozen=True)
class FrozenRoutingContract:
    path: Path
    file_sha256: str
    assignment_sha256: str
    data_path: Path
    data_sha256: str
    values: Mapping[str, Any]


@dataclass(frozen=True)
class StageBReleaseContract:
    path: Path
    file_sha256: str
    stage_a_summary_path: Path
    stage_a_summary_sha256: str
    stage_a_summary_row_count: int
    routing_split: str


@dataclass(frozen=True)
class StageAValidationContract:
    path: Path
    file_sha256: str
    stage_a_expected_count: int
    source_pair_count: int


MODEL_SPECS = {
    "qwen35_4b": ModelSpec("qwen35_4b", "Qwen/Qwen3.5-4B", 1, "96G", 1),
    "qwen35_9b": ModelSpec("qwen35_9b", "Qwen/Qwen3.5-9B", 1, "128G", 1),
    "qwen35_27b": ModelSpec("qwen35_27b", "Qwen/Qwen3.5-27B", 2, "220G", 2, gpu_memory_utilization=0.92),
    "qwen3_14b": ModelSpec("qwen3_14b", "Qwen/Qwen3-14B", 1, "160G", 1),
    "qwen3_32b": ModelSpec("qwen3_32b", "Qwen/Qwen3-32B", 2, "240G", 2),
    "qwen3_4b": ModelSpec("qwen3_4b", "Qwen/Qwen3-4B", 1, "64G", 1),
    "qwen25_32b": ModelSpec(
        "qwen25_32b",
        "Qwen/Qwen2.5-32B-Instruct",
        2,
        "240G",
        2,
    ),
    "llama33_70b_instruct": ModelSpec(
        "llama33_70b_instruct",
        "meta-llama/Llama-3.3-70B-Instruct",
        4,
        "240G",
        4,
        gpu_memory_utilization=0.95,
        enforce_eager=True,
        max_model_len=4096,
        batch_size=1,
        max_num_batched_tokens=4096,
        max_num_seqs=1,
    ),
    "mistral7b": ModelSpec("mistral7b", "mistralai/Mistral-7B-Instruct-v0.3", 1, "128G", 1),
    "gemma2_9b": ModelSpec(
        "gemma2_9b",
        "google/gemma-2-9b-it",
        1,
        "96G",
        1,
        enforce_eager=True,
    ),
    "gemma2_27b": ModelSpec(
        "gemma2_27b",
        "google/gemma-2-27b-it",
        2,
        "240G",
        2,
        enforce_eager=True,
    ),
    "gemma3_12b": ModelSpec(
        "gemma3_12b",
        "google/gemma-3-12b-it",
        1,
        "128G",
        1,
        enforce_eager=True,
    ),
    "skywork_critic_8b": ModelSpec(
        "skywork_critic_8b",
        "Skywork/Skywork-Critic-Llama-3.1-8B",
        1,
        "128G",
        1,
    ),
    "hermes3_llama31_8b": ModelSpec(
        "hermes3_llama31_8b",
        "NousResearch/Hermes-3-Llama-3.1-8B",
        1,
        "128G",
        1,
    ),
    "olmo2_7b_instruct": ModelSpec(
        "olmo2_7b_instruct",
        "allenai/OLMo-2-1124-7B-Instruct",
        1,
        "128G",
        1,
    ),
    "olmo3_7b_instruct": ModelSpec(
        "olmo3_7b_instruct",
        "allenai/Olmo-3-7B-Instruct",
        1,
        "128G",
        1,
    ),
    "phi4_14b": ModelSpec(
        "phi4_14b",
        "microsoft/phi-4",
        1,
        "160G",
        1,
    ),
}


BIAS_COMMANDS = {
    "position": "run-position",
    "authority": "run-authority",
    "bandwagon": "run-bandwagon",
}


CONTROL_COMMANDS = {
    "identical": "run-identical-position-control",
    "label_prior": "run-label-prior-control",
}


def _optional_sbatch_line(flag: str, value: str | None) -> str:
    return "" if value is None else f"#SBATCH {flag} {value}\n"


def _optional_scheduler_block(
    *,
    partition: str | None,
    qos: str | None,
    account: str | None,
) -> str:
    lines: list[str] = []
    if partition is not None:
        lines.append(f"#SBATCH --partition={partition}")
    if qos is not None:
        lines.append(f"#SBATCH --qos={qos}")
    if account is not None:
        lines.append(f"#SBATCH --account={account}")
    return "".join(f"{line}\n" for line in lines)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _value_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_mapping(path: Path, *, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain one JSON object: {path}")
    return value


def _jsonl_mapping_row_count(path: Path, *, name: str) -> int:
    row_count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{name} line {line_number} is not valid JSON"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(f"{name} line {line_number} is not an object")
            row_count += 1
    if row_count < 1:
        raise ValueError(f"{name} must contain at least one row")
    return row_count


def _require_sha256(value: object, *, name: str) -> str:
    text = str(value or "")
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return text


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _normalize_runtime_contract(
    runtime: Mapping[str, Any],
    *,
    model: ModelSpec,
) -> dict[str, Any]:
    missing = [field for field in SILENT_BIAS_RUNTIME_FIELDS if field not in runtime]
    if missing:
        raise ValueError(
            "runtime mapping is missing fields: " + ", ".join(missing)
        )
    unknown = sorted(set(runtime) - set(SILENT_BIAS_RUNTIME_FIELDS))
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
    revision = runtime["model_revision"]
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("runtime model_revision must be a non-empty pinned revision")
    normalized["model_revision"] = revision.strip()

    for field in (
        "tensor_parallel_size",
        "max_model_len",
        "batch_size",
        "max_num_batched_tokens",
        "max_num_seqs",
    ):
        normalized[field] = _positive_int(runtime[field], name=f"runtime {field}")
    for field in (
        "enforce_eager",
        "disable_custom_all_reduce",
        "include_verbalized_confidence",
    ):
        if not isinstance(runtime[field], bool):
            raise ValueError(f"runtime {field} must be boolean")

    try:
        utilization = float(runtime["gpu_memory_utilization"])
        temperature = float(runtime["sampling_temperature"])
    except (TypeError, ValueError) as exc:
        raise ValueError("runtime floating-point controls are invalid") from exc
    if not math.isfinite(utilization) or not 0.0 < utilization <= 1.0:
        raise ValueError("runtime gpu_memory_utilization must be finite and in (0, 1]")
    if not math.isfinite(temperature) or temperature < 0.0:
        raise ValueError("runtime sampling_temperature must be finite and nonnegative")
    normalized["gpu_memory_utilization"] = utilization
    normalized["sampling_temperature"] = temperature

    seed = runtime["seed"]
    if seed != 0 or isinstance(seed, bool):
        raise ValueError("Silent Bias production currently requires runtime seed=0")
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

    versions = runtime["engine_versions"]
    if not isinstance(versions, Mapping) or set(versions) != set(ENGINE_VERSION_FIELDS):
        raise ValueError(
            "runtime engine_versions must contain exactly: "
            + ", ".join(ENGINE_VERSION_FIELDS)
        )
    normalized_versions: dict[str, str | None] = {}
    for field in ENGINE_VERSION_FIELDS:
        value = versions[field]
        if value is not None and (
            not isinstance(value, str) or not value.strip()
        ):
            raise ValueError(
                f"runtime engine_versions.{field} must be null or a non-empty string"
            )
        normalized_versions[field] = value.strip() if isinstance(value, str) else None
    normalized["engine_versions"] = normalized_versions

    digest = _require_sha256(runtime["runtime_sha256"], name="runtime runtime_sha256")
    digest_input = {
        field: normalized[field]
        for field in SILENT_BIAS_RUNTIME_FIELDS
        if field != "runtime_sha256"
    }
    if digest != _value_sha256(digest_input):
        raise ValueError("runtime runtime_sha256 does not match its mapping")
    normalized["runtime_sha256"] = digest

    profile = get_model_profile(model.model_name)
    expected_identity = {
        "model_registry_name": profile.registry_name,
        "model_hf_name": profile.hf_model_name,
        "model_revision": profile.revision,
    }
    observed_identity = {field: normalized[field] for field in expected_identity}
    if observed_identity != expected_identity:
        raise ValueError(
            "runtime model identity does not match the selected model profile: "
            f"observed={observed_identity!r} expected={expected_identity!r}"
        )
    return normalized


def load_frozen_runtime_contract(
    path: Path,
    *,
    model: ModelSpec,
) -> FrozenRuntimeContract:
    resolved = path.expanduser().resolve()
    runtime = _normalize_runtime_contract(
        _read_json_mapping(resolved, name="runtime JSON"),
        model=model,
    )
    return FrozenRuntimeContract(
        path=resolved,
        file_sha256=_file_sha256(resolved),
        embedded_sha256=str(runtime["runtime_sha256"]),
        canonical_sha256=_value_sha256(runtime),
        values=runtime,
    )


def load_frozen_routing_contract(
    manifest_path: Path,
    *,
    data_path: Path,
) -> FrozenRoutingContract:
    resolved_manifest = manifest_path.expanduser().resolve()
    resolved_data = data_path.expanduser().resolve()
    if not resolved_data.is_file():
        raise FileNotFoundError(f"routed data CSV does not exist: {resolved_data}")
    manifest = _read_json_mapping(resolved_manifest, name="routing manifest")
    if manifest.get("schema_version") != 2:
        raise ValueError("routing manifest must use schema version 2")
    if manifest.get("artifact_type") != ROUTING_ARTIFACT_TYPE:
        raise ValueError("routing manifest artifact type does not match")
    outputs = manifest.get("outputs")
    output_hashes = manifest.get("output_sha256")
    if not isinstance(outputs, Mapping) or not isinstance(output_hashes, Mapping):
        raise ValueError("routing manifest is missing output bindings")
    full_output = outputs.get("full")
    if not isinstance(full_output, Mapping):
        raise ValueError("routing manifest outputs.full must be an object")
    raw_full_path = Path(str(full_output.get("path") or ""))
    if raw_full_path.is_absolute() or len(raw_full_path.parts) != 1:
        raise ValueError("routing manifest outputs.full.path must be one filename")
    expected_data_path = (resolved_manifest.parent / raw_full_path).resolve()
    if resolved_data != expected_data_path:
        raise ValueError(
            "data path does not match routing manifest outputs.full.path: "
            f"data={resolved_data} expected={expected_data_path}"
        )
    data_sha = _file_sha256(resolved_data)
    expected_data_sha = _require_sha256(
        output_hashes.get("full"),
        name="routing manifest output_sha256.full",
    )
    if data_sha != expected_data_sha:
        raise ValueError("routed data SHA-256 does not match routing manifest")
    assignment_sha = _require_sha256(
        manifest.get("routing_assignment_sha256"),
        name="routing manifest routing_assignment_sha256",
    )
    return FrozenRoutingContract(
        path=resolved_manifest,
        file_sha256=_file_sha256(resolved_manifest),
        assignment_sha256=assignment_sha,
        data_path=resolved_data,
        data_sha256=data_sha,
        values=manifest,
    )


def load_stage_b_release_contract(
    preflight_path: Path,
    *,
    expected_file_sha256: str,
    runtime: FrozenRuntimeContract,
    routing: FrozenRoutingContract,
    stage_a_summary_path: Path,
) -> StageBReleaseContract:
    expected_sha = _require_sha256(
        expected_file_sha256,
        name="Stage B preflight expected SHA-256",
    )
    resolved_preflight = preflight_path.expanduser().resolve()
    observed_sha = _file_sha256(resolved_preflight)
    if observed_sha != expected_sha:
        raise ValueError("Stage B preflight file SHA-256 does not match")
    report = _read_json_mapping(resolved_preflight, name="Stage B preflight report")
    release_gate = report.get("release_gate")
    if (
        report.get("status") != "complete"
        or report.get("passed") is not True
        or not isinstance(release_gate, Mapping)
        or release_gate.get("stage_b_authorized") is not True
    ):
        raise ValueError("Stage B preflight does not authorize release")

    report_runtime = report.get("runtime")
    if not isinstance(report_runtime, Mapping):
        raise ValueError("Stage B preflight has no runtime binding")
    if report_runtime.get("inference_runtime") != runtime.values:
        raise ValueError("Stage B preflight runtime mapping does not match")
    if report_runtime.get("inference_runtime_sha256") != runtime.canonical_sha256:
        raise ValueError("Stage B preflight runtime outer SHA-256 does not match")

    report_model = report.get("model")
    expected_model = {
        "registry_name": runtime.values["model_registry_name"],
        "model_name": runtime.values["model_hf_name"],
        "revision": runtime.values["model_revision"],
    }
    if not isinstance(report_model, Mapping) or any(
        report_model.get(field) != value for field, value in expected_model.items()
    ):
        raise ValueError("Stage B preflight model binding does not match")

    report_source = report.get("source")
    if (
        not isinstance(report_source, Mapping)
        or Path(str(report_source.get("path") or "")).resolve() != routing.data_path
        or report_source.get("sha256") != routing.data_sha256
    ):
        raise ValueError("Stage B preflight source-data binding does not match")
    report_routing = report.get("routing")
    if (
        not isinstance(report_routing, Mapping)
        or Path(str(report_routing.get("manifest_path") or "")).resolve()
        != routing.path
        or report_routing.get("manifest_file_sha256") != routing.file_sha256
        or report_routing.get("routing_assignment_sha256")
        != routing.assignment_sha256
    ):
        raise ValueError("Stage B preflight routing binding does not match")

    scope = report.get("scope")
    routing_split = (
        str(scope.get("stage_b_routing_split") or "")
        if isinstance(scope, Mapping)
        else ""
    )
    if routing_split != "test":
        raise ValueError("controlled Stage B preflight must bind routing split 'test'")

    resolved_summary = stage_a_summary_path.expanduser().resolve()
    if not resolved_summary.is_file():
        raise FileNotFoundError(f"Stage A summary does not exist: {resolved_summary}")
    summary_sha = _file_sha256(resolved_summary)
    summary_row_count = _jsonl_mapping_row_count(
        resolved_summary,
        name="Stage A summary",
    )
    plan = report.get("plan")
    summary_binding = plan.get("stage_a_summary") if isinstance(plan, Mapping) else None
    if (
        not isinstance(summary_binding, Mapping)
        or summary_binding.get("mode") != "exact_post_stage_a"
        or Path(str(summary_binding.get("path") or "")).resolve() != resolved_summary
        or summary_binding.get("file_sha256") != summary_sha
        or summary_binding.get("row_count") != summary_row_count
        or plan.get("actual_stage_b_prompt_set_sha256") is None
        or plan.get("provisional_structural_stage_b_prompt_set_sha256") is not None
    ):
        raise ValueError("Stage B preflight Stage A/prompt binding does not match")
    return StageBReleaseContract(
        path=resolved_preflight,
        file_sha256=observed_sha,
        stage_a_summary_path=resolved_summary,
        stage_a_summary_sha256=summary_sha,
        stage_a_summary_row_count=summary_row_count,
        routing_split=routing_split,
    )


def load_stage_a_validation_contract(
    validation_path: Path,
    *,
    expected_file_sha256: str,
    runtime: FrozenRuntimeContract,
    routing: FrozenRoutingContract,
    stage_a_summary_path: Path,
    dataset_split: str,
) -> StageAValidationContract:
    """Load one passing Stage-A-only artifact-validation release gate."""

    expected_sha = _require_sha256(
        expected_file_sha256,
        name="Stage A validation expected SHA-256",
    )
    resolved_validation = validation_path.expanduser().resolve()
    observed_sha = _file_sha256(resolved_validation)
    if observed_sha != expected_sha:
        raise ValueError("Stage A validation file SHA-256 does not match")
    report = _read_json_mapping(
        resolved_validation,
        name="Stage A artifact-validation report",
    )
    if report.get("passed") is not True or report.get("error_count") != 0:
        raise ValueError("Stage A artifact validation did not pass")

    design = report.get("design")
    if not isinstance(design, Mapping):
        raise ValueError("Stage A artifact validation has no design binding")
    if design.get("validation_scope") != "stage_a":
        raise ValueError("Stage A artifact validation scope is not stage_a")
    if (
        design.get("expected_inference_runtime_sha256")
        != runtime.canonical_sha256
    ):
        raise ValueError("Stage A validation expected runtime SHA does not match")
    if (
        design.get("expected_question_routing_sha256")
        != routing.assignment_sha256
    ):
        raise ValueError("Stage A validation expected routing SHA does not match")
    if (
        design.get("consistency_runs") != runtime.values["consistency_runs"]
        or design.get("consistency_schedule")
        != runtime.values["consistency_schedule"]
        or design.get("sampling_temperature")
        != runtime.values["sampling_temperature"]
        or design.get("dataset_split") != dataset_split
    ):
        raise ValueError("Stage A validation scientific runtime binding does not match")
    question_routing = design.get("question_routing")
    if (
        not isinstance(question_routing, Mapping)
        or question_routing.get("expected_assignment_sha256")
        != routing.assignment_sha256
        or question_routing.get("assignment_sha256")
        != routing.assignment_sha256
        or question_routing.get("raw_question_assignment_sha256")
        != routing.assignment_sha256
    ):
        raise ValueError("Stage A validation observed routing SHA does not match")

    source = report.get("source")
    if (
        not isinstance(source, Mapping)
        or Path(str(source.get("csv") or "")).resolve() != routing.data_path
        or source.get("input_file_hash") != routing.data_sha256
    ):
        raise ValueError("Stage A validation source-data binding does not match")

    artifacts = report.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 1:
        raise ValueError("Stage A validation must contain exactly one model artifact")
    artifact = artifacts[0]
    if not isinstance(artifact, Mapping):
        raise ValueError("Stage A validation artifact is invalid")
    if (
        artifact.get("validation_scope") != "stage_a"
        or artifact.get("model_name") != runtime.values["model_hf_name"]
        or artifact.get("model_revision") != runtime.values["model_revision"]
        or artifact.get("dataset_split") != dataset_split
        or artifact.get("inference_runtime_sha256_by_stage")
        != {"stage_a": runtime.canonical_sha256}
    ):
        raise ValueError("Stage A validation artifact runtime/model binding does not match")

    resolved_summary = stage_a_summary_path.expanduser().resolve()
    summary_row_count = _jsonl_mapping_row_count(
        resolved_summary,
        name="Stage A summary",
    )
    stage_dirs = artifact.get("stage_dirs")
    if (
        not isinstance(stage_dirs, Mapping)
        or Path(str(stage_dirs.get("stage_a") or "")).resolve()
        != resolved_summary.parent
    ):
        raise ValueError("Stage A validation artifact directory does not bind the summary")

    counts = artifact.get("counts")
    if not isinstance(counts, Mapping):
        raise ValueError("Stage A validation has no count binding")
    source_pairs = counts.get("source_pairs")
    stage_a_expected = counts.get("stage_a_expected")
    if (
        isinstance(source_pairs, bool)
        or not isinstance(source_pairs, int)
        or source_pairs < 1
        or isinstance(stage_a_expected, bool)
        or not isinstance(stage_a_expected, int)
        or stage_a_expected != 2 * source_pairs
        or stage_a_expected != summary_row_count
        or any(
            counts.get(field) != stage_a_expected
            for field in (
                "stage_a_raw",
                "stage_a_flat",
                "stage_a_pair_summary",
            )
        )
    ):
        raise ValueError("Stage A validation expected/observed counts do not match")
    return StageAValidationContract(
        path=resolved_validation,
        file_sha256=observed_sha,
        stage_a_expected_count=stage_a_expected,
        source_pair_count=source_pairs,
    )


def _validate_run_group(run_group: str) -> None:
    """Require one explicit, path-safe immutable campaign identifier."""

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_group):
        raise ValueError(
            "run_group must be an explicit path-safe campaign tag containing "
            "only letters, digits, '.', '_', or '-'"
        )


def _write_new_text(path: Path, content: str) -> None:
    """Create a launcher exactly once rather than silently replacing evidence."""

    with path.open("x", encoding="utf-8") as output:
        output.write(content)


def render_job(
    *,
    job_name: str,
    command: str,
    model: ModelSpec,
    output_slug: str,
    data_file: str,
    time: str,
    partition: str | None,
    qos: str | None,
    account: str | None,
    artifact_root: str,
    limit: int | None = None,
) -> str:
    disable_custom_all_reduce = "1" if model.gpus > 1 else "0"
    command_args = []
    if command != "run-label-prior-control":
        command_args.append('  --data-path "${DATA_PATH}"')
    command_args.extend(
        [
            '  --output-dir "${OUTPUT_DIR}"',
            '  --model-name "${MODEL_NAME}"',
            '  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"',
            '  --max-model-len "${MAX_MODEL_LEN}"',
            '  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"',
            '  --dtype "${DTYPE}"',
        ]
    )
    if limit is not None:
        command_args.append(f"  --limit {limit}")
    command_block = " \\\n".join(command_args)
    optional_sbatch = (
        _optional_sbatch_line("-p", partition)
        + _optional_sbatch_line("-q", qos)
        + _optional_sbatch_line("--account=", account).replace("--account= ", "--account=")
    )

    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:{model.gpus}
#SBATCH --mem={model.mem}
#SBATCH --time={time}
{optional_sbatch}#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO_DIR="${{REPO_DIR:-${{SLURM_SUBMIT_DIR:-$(pwd)}}}}"
ARTIFACT_ROOT="${{BIASES_ARTIFACT_ROOT:-{artifact_root}}}"
export BIASES_ARTIFACT_ROOT="${{ARTIFACT_ROOT}}"
DATA_PATH="${{DATA_PATH:-${{ARTIFACT_ROOT}}/data/processed/{data_file}}}"
OUTPUT_DIR="${{OUTPUT_DIR:-${{ARTIFACT_ROOT}}/outputs/{output_slug}_${{SLURM_JOB_ID:-manual}}}}"
MODEL_NAME="${{MODEL_NAME:-{model.model_name}}}"
MAX_MODEL_LEN="${{MAX_MODEL_LEN:-8192}}"
GPU_MEMORY_UTILIZATION="${{GPU_MEMORY_UTILIZATION:-{model.gpu_memory_utilization}}}"
DTYPE="${{DTYPE:-{model.dtype}}}"
TENSOR_PARALLEL_SIZE="${{TENSOR_PARALLEL_SIZE:-{model.tensor_parallel_size}}}"
EXTRA_ARGS="${{EXTRA_ARGS:-}}"
TMPDIR="${{TMPDIR:-/tmp/${{USER:-user}}/${{SLURM_JOB_ID:-manual}}}}"
HF_ENV_FILE="${{BIASES_HF_ENV:-${{ARTIFACT_ROOT}}/secrets/hf.env}}"

if [ -f "${{HF_ENV_FILE}}" ]; then
  set -a
  source "${{HF_ENV_FILE}}"
  set +a
fi

mkdir -p "${{REPO_DIR}}/logs" "${{ARTIFACT_ROOT}}/cache" "${{OUTPUT_DIR}}" "${{TMPDIR}}"
cd "${{REPO_DIR}}"

VENV_PATH="${{VENV_PATH:-${{UV_PROJECT_ENVIRONMENT:-${{REPO_DIR}}/.venv}}}}"
if [ -f "${{VENV_PATH}}/bin/activate" ]; then
  source "${{VENV_PATH}}/bin/activate"
fi

export HF_HOME="${{HF_HOME:-${{ARTIFACT_ROOT}}/cache/huggingface}}"
export HF_HUB_CACHE="${{HF_HUB_CACHE:-${{HF_HOME}}/hub}}"
export HF_DATASETS_CACHE="${{HF_DATASETS_CACHE:-${{HF_HOME}}/datasets}}"
export HF_HUB_DISABLE_XET="${{HF_HUB_DISABLE_XET:-1}}"
export VLLM_DISABLE_CUSTOM_ALL_REDUCE="${{VLLM_DISABLE_CUSTOM_ALL_REDUCE:-{disable_custom_all_reduce}}}"
export BIASES_VLLM_ENFORCE_EAGER="${{BIASES_VLLM_ENFORCE_EAGER:-{int(model.enforce_eager)}}}"
export XDG_CACHE_HOME="${{XDG_CACHE_HOME:-${{ARTIFACT_ROOT}}/cache/xdg}}"
export VLLM_CACHE_ROOT="${{VLLM_CACHE_ROOT:-${{ARTIFACT_ROOT}}/cache/vllm}}"
export TORCH_HOME="${{TORCH_HOME:-${{ARTIFACT_ROOT}}/cache/torch}}"
export TMPDIR
export TRITON_CACHE_DIR="${{TRITON_CACHE_DIR:-${{TMPDIR}}/triton}}"
export UV_CACHE_DIR="${{UV_CACHE_DIR:-${{ARTIFACT_ROOT}}/cache/uv}}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF="${{PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"

mkdir -p "${{HF_HOME}}" "${{HF_HUB_CACHE}}" "${{HF_DATASETS_CACHE}}" "${{XDG_CACHE_HOME}}" "${{VLLM_CACHE_ROOT}}" "${{TORCH_HOME}}" "${{TRITON_CACHE_DIR}}" "${{UV_CACHE_DIR}}"

echo "Job ID: ${{SLURM_JOB_ID:-manual}}"
echo "Node: ${{SLURMD_NODENAME:-$(hostname)}}"
echo "CUDA_VISIBLE_DEVICES: ${{CUDA_VISIBLE_DEVICES:-unset}}"
nvidia-smi || true

python main.py {command} \\
{command_block} \\
  ${{EXTRA_ARGS}}
"""


def render_silent_bias_job(
    *,
    stage: str,
    job_name: str,
    model: ModelSpec,
    time: str,
    cpus_per_task: int,
    gpus: int,
    mem: str,
    partition: str | None,
    qos: str | None,
    account: str | None,
    artifact_root: str,
    run_group: str,
    runtime: FrozenRuntimeContract,
    routing: FrozenRoutingContract,
    python_bin: Path,
    stage_b_release: StageBReleaseContract | None = None,
    stage_a_validation: StageAValidationContract | None = None,
    dataset_split: str = "full",
) -> str:
    """Render one immutable, runtime-bound Stage A or Stage B launcher."""

    normalized_stage = stage.upper()
    if normalized_stage not in {"A", "B"}:
        raise ValueError("stage must be A or B")
    _validate_run_group(run_group)
    if cpus_per_task < 1 or gpus < 1:
        raise ValueError("CPU and GPU counts must be positive")
    values = runtime.values
    tensor_parallel_size = int(values["tensor_parallel_size"])
    if gpus != tensor_parallel_size or model.gpus != tensor_parallel_size:
        raise ValueError(
            "single-node Silent Bias jobs require requested/model GPUs to equal "
            "the frozen tensor_parallel_size"
        )
    if not dataset_split.strip():
        raise ValueError("dataset_split must be non-empty")
    resolved_python = python_bin.expanduser()
    if not resolved_python.is_absolute():
        raise ValueError("python_bin must be an explicit absolute path")
    if normalized_stage == "B" and (
        stage_b_release is None or stage_a_validation is None
    ):
        raise ValueError(
            "Stage B rendering requires an authorized post-Stage-A preflight "
            "and a passing Stage-A-only artifact validation"
        )
    if normalized_stage == "A" and (
        stage_b_release is not None or stage_a_validation is not None
    ):
        raise ValueError("Stage A rendering must not carry Stage B release contracts")

    if stage_b_release is None:
        stage_a_summary_shell = (
            '"${STAGE_A_OUTPUT_DIR}/' + STAGE_A_SUMMARY_FILENAME + '"'
        )
        preflight_path = ""
        preflight_sha = ""
        stage_a_summary_sha = ""
        stage_a_validation_path = ""
        stage_a_validation_sha = ""
        routing_split = "test"
    else:
        assert stage_a_validation is not None
        stage_a_summary_shell = shlex.quote(
            str(stage_b_release.stage_a_summary_path)
        )
        preflight_path = str(stage_b_release.path)
        preflight_sha = stage_b_release.file_sha256
        stage_a_summary_sha = stage_b_release.stage_a_summary_sha256
        stage_a_validation_path = str(stage_a_validation.path)
        stage_a_validation_sha = stage_a_validation.file_sha256
        routing_split = stage_b_release.routing_split

    launch_binding = {
        "schema_version": 1,
        "stage": normalized_stage,
        "runtime": {
            "path": str(runtime.path),
            "file_sha256": runtime.file_sha256,
            "embedded_sha256": runtime.embedded_sha256,
            "canonical_sha256": runtime.canonical_sha256,
        },
        "model": {
            "registry_name": values["model_registry_name"],
            "hf_name": values["model_hf_name"],
            "revision": values["model_revision"],
        },
        "routing": {
            "manifest_path": str(routing.path),
            "manifest_file_sha256": routing.file_sha256,
            "routing_assignment_sha256": routing.assignment_sha256,
            "data_path": str(routing.data_path),
            "data_sha256": routing.data_sha256,
        },
        "tensor_parallel_size": tensor_parallel_size,
        "dataset_split": dataset_split,
        "stage_b": (
            {
                "preflight_path": preflight_path,
                "preflight_file_sha256": preflight_sha,
                "stage_a_summary_path": str(
                    stage_b_release.stage_a_summary_path
                ),
                "stage_a_summary_sha256": stage_a_summary_sha,
                "stage_a_summary_row_count": (
                    stage_b_release.stage_a_summary_row_count
                ),
                "stage_a_validation_path": stage_a_validation_path,
                "stage_a_validation_file_sha256": stage_a_validation_sha,
                "stage_a_expected_count": (
                    stage_a_validation.stage_a_expected_count
                ),
                "source_pair_count": stage_a_validation.source_pair_count,
                "routing_split": routing_split,
            }
            if stage_b_release is not None
            else None
        ),
    }
    launch_binding_b64 = base64.b64encode(
        _canonical_json(launch_binding).encode("utf-8")
    ).decode("ascii")

    template = SILENT_BIAS_TEMPLATE_PATH.read_text(encoding="utf-8")
    replacements = {
        "@@JOB_NAME@@": job_name,
        "@@CPUS_PER_TASK@@": str(cpus_per_task),
        "@@GPUS@@": str(gpus),
        "@@MEM@@": mem,
        "@@TIME@@": time,
        "@@OPTIONAL_SBATCH@@": _optional_scheduler_block(
            partition=partition,
            qos=qos,
            account=account,
        ),
        "@@ARTIFACT_ROOT@@": artifact_root,
        "@@RUN_GROUP@@": run_group,
        "@@STAGE@@": normalized_stage,
        "@@DATA_PATH@@": shlex.quote(str(routing.data_path)),
        "@@DATA_SHA256@@": routing.data_sha256,
        "@@ROUTING_MANIFEST@@": shlex.quote(str(routing.path)),
        "@@ROUTING_MANIFEST_SHA256@@": routing.file_sha256,
        "@@ROUTING_ASSIGNMENT_SHA256@@": routing.assignment_sha256,
        "@@RUNTIME_JSON@@": shlex.quote(str(runtime.path)),
        "@@RUNTIME_FILE_SHA256@@": runtime.file_sha256,
        "@@RUNTIME_EMBEDDED_SHA256@@": runtime.embedded_sha256,
        "@@RUNTIME_CANONICAL_SHA256@@": runtime.canonical_sha256,
        "@@STAGE_B_PREFLIGHT@@": shlex.quote(preflight_path),
        "@@STAGE_B_PREFLIGHT_SHA256@@": preflight_sha,
        "@@STAGE_A_SUMMARY@@": stage_a_summary_shell,
        "@@STAGE_A_SUMMARY_SHA256@@": stage_a_summary_sha,
        "@@STAGE_A_VALIDATION@@": shlex.quote(stage_a_validation_path),
        "@@STAGE_A_VALIDATION_SHA256@@": stage_a_validation_sha,
        "@@LAUNCH_BINDING_B64@@": shlex.quote(launch_binding_b64),
        "@@PYTHON_BIN@@": shlex.quote(str(resolved_python)),
        "@@DATASET_SPLIT@@": dataset_split,
        "@@MODEL_NAME@@": str(values["model_hf_name"]),
        "@@MODEL_REGISTRY_NAME@@": str(values["model_registry_name"]),
        "@@MODEL_REVISION@@": str(values["model_revision"]),
        "@@MODEL_SLUG@@": model.slug,
        "@@STAGE_A_COMMAND@@": STAGE_A_COMMAND,
        "@@STAGE_B_COMMAND@@": STAGE_B_COMMAND,
        "@@CONSISTENCY_RUNS@@": str(values["consistency_runs"]),
        "@@CONSISTENCY_SCHEDULE@@": str(values["consistency_schedule"]),
        "@@SAMPLING_TEMPERATURE@@": str(values["sampling_temperature"]),
        "@@INCLUDE_VERBALIZED@@": (
            "1" if values["include_verbalized_confidence"] else "0"
        ),
        "@@TENSOR_PARALLEL_SIZE@@": str(tensor_parallel_size),
        "@@MAX_MODEL_LEN@@": str(values["max_model_len"]),
        "@@BATCH_SIZE@@": str(values["batch_size"]),
        "@@MAX_NUM_BATCHED_TOKENS@@": str(values["max_num_batched_tokens"]),
        "@@MAX_NUM_SEQS@@": str(values["max_num_seqs"]),
        "@@STAGE_B_ROUTING_SPLIT@@": routing_split,
        "@@GPU_MEMORY_UTILIZATION@@": str(values["gpu_memory_utilization"]),
        "@@DTYPE@@": str(values["dtype"]),
        "@@ENFORCE_EAGER@@": "1" if values["enforce_eager"] else "0",
        "@@DISABLE_CUSTOM_ALL_REDUCE@@": (
            "1" if values["disable_custom_all_reduce"] else "0"
        ),
    }
    rendered = template
    for marker, value in replacements.items():
        rendered = rendered.replace(marker, value)
    if "@@" in rendered:
        raise ValueError("unresolved placeholder in Silent Bias Slurm template")
    return rendered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render portable Slurm jobs.")
    parser.add_argument("--output-dir", type=Path, default=Path("slurm/generated"))
    parser.add_argument(
        "--kind",
        choices=["controls", "phase3", "silent-bias"],
        required=True,
    )
    parser.add_argument("--partition", default=None, help="Optional Slurm partition.")
    parser.add_argument("--qos", default=None, help="Optional Slurm QOS.")
    parser.add_argument("--account", default=None, help="Optional Slurm account.")
    parser.add_argument(
        "--artifact-root",
        default=DEFAULT_ARTIFACT_ROOT_EXPR,
        help="Default BIASES_ARTIFACT_ROOT expression to render into jobs.",
    )
    parser.add_argument(
        "--run-group",
        default=None,
        help=(
            "Required immutable campaign tag for --kind silent-bias. It is "
            "hard-bound into both Stage A and Stage B launchers."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_SPECS),
        default=None,
        help=(
            "Model keys. Immutable silent-bias rendering requires exactly one "
            "model because each runtime JSON is model-specific."
        ),
    )
    parser.add_argument("--data-file", default="mtbench_full.csv")
    parser.add_argument(
        "--data-path",
        type=Path,
        help="Absolute routed-full CSV path required for --kind silent-bias.",
    )
    parser.add_argument(
        "--routing-manifest",
        type=Path,
        help="Frozen schema-2 routing manifest required for silent-bias.",
    )
    parser.add_argument(
        "--runtime-json",
        type=Path,
        help="Frozen inference-runtime JSON required for silent-bias.",
    )
    parser.add_argument(
        "--python-bin",
        type=Path,
        help="Explicit absolute Python executable required for silent-bias.",
    )
    parser.add_argument(
        "--stage",
        choices=("A", "B"),
        help="Render exactly one silent-bias stage per invocation.",
    )
    parser.add_argument(
        "--stage-b-preflight",
        type=Path,
        help="Exact post-Stage-A preflight report required for Stage B.",
    )
    parser.add_argument(
        "--stage-b-preflight-sha256",
        help="Independently pinned SHA-256 for --stage-b-preflight.",
    )
    parser.add_argument(
        "--stage-a-summary",
        type=Path,
        help="Stage A pair-summary path bound by the Stage B preflight.",
    )
    parser.add_argument(
        "--stage-a-validation",
        type=Path,
        help="Passing Stage-A-only artifact-validation report required for Stage B.",
    )
    parser.add_argument(
        "--stage-a-validation-sha256",
        help="Independently pinned SHA-256 for --stage-a-validation.",
    )
    parser.add_argument("--time", default="48:00:00")
    parser.add_argument("--cpus-per-task", type=int, default=16)
    parser.add_argument(
        "--mem",
        default=None,
        help="Override each selected model template's memory request.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[Path] = []
    if args.kind == "controls":
        for model_key in ("qwen35_4b", "qwen35_9b", "qwen35_27b", "qwen3_14b", "qwen3_32b"):
            model = MODEL_SPECS[model_key]
            for control, command in CONTROL_COMMANDS.items():
                path = args.output_dir / f"{control}_{model.slug}.slurm"
                _write_new_text(
                    path,
                    render_job(
                        job_name=f"{control}-{model.slug}",
                        command=command,
                        model=model,
                        output_slug=f"{control}_{model.slug}",
                        data_file="mtbench_full.csv",
                        time="04:00:00",
                        partition=args.partition,
                        qos=args.qos,
                        account=args.account,
                        artifact_root=args.artifact_root,
                        limit=300 if control == "identical" else None,
                    ),
                )
                rendered.append(path)
    elif args.kind == "phase3":
        for model_key in ("mistral7b", "gemma2_27b", "skywork_critic_8b"):
            model = MODEL_SPECS[model_key]
            for bias, command in BIAS_COMMANDS.items():
                path = args.output_dir / f"{bias}_{model.slug}_mtbench_full.slurm"
                _write_new_text(
                    path,
                    render_job(
                        job_name=f"{bias}-{model.slug}",
                        command=command,
                        model=model,
                        output_slug=f"{bias}_{model.slug}_mtbench_full",
                        data_file="mtbench_full.csv",
                        time="48:00:00",
                        partition=args.partition,
                        qos=args.qos,
                        account=args.account,
                        artifact_root=args.artifact_root,
                    ),
                )
                rendered.append(path)
    else:
        required = {
            "--run-group": args.run_group,
            "--stage": args.stage,
            "--runtime-json": args.runtime_json,
            "--routing-manifest": args.routing_manifest,
            "--data-path": args.data_path,
            "--python-bin": args.python_bin,
        }
        missing = [flag for flag, value in required.items() if value is None]
        if missing:
            raise ValueError(
                "--kind silent-bias requires: " + ", ".join(missing)
            )
        if args.models is None or len(args.models) != 1:
            raise ValueError(
                "--kind silent-bias requires exactly one --models value"
            )

        run_group = str(args.run_group)
        stage = str(args.stage)
        _validate_run_group(run_group)
        model = MODEL_SPECS[args.models[0]]
        runtime = load_frozen_runtime_contract(
            args.runtime_json,
            model=model,
        )
        routing = load_frozen_routing_contract(
            args.routing_manifest,
            data_path=args.data_path,
        )

        release: StageBReleaseContract | None = None
        validation: StageAValidationContract | None = None
        stage_b_values = (
            args.stage_b_preflight,
            args.stage_b_preflight_sha256,
            args.stage_a_summary,
            args.stage_a_validation,
            args.stage_a_validation_sha256,
        )
        if stage == "A":
            if any(value is not None for value in stage_b_values):
                raise ValueError(
                    "Stage A rendering rejects Stage B preflight/summary options"
                )
        else:
            if any(value is None for value in stage_b_values):
                raise ValueError(
                    "Stage B rendering requires --stage-b-preflight, "
                    "--stage-b-preflight-sha256, --stage-a-summary, "
                    "--stage-a-validation, and --stage-a-validation-sha256"
                )
            release = load_stage_b_release_contract(
                args.stage_b_preflight,
                expected_file_sha256=args.stage_b_preflight_sha256,
                runtime=runtime,
                routing=routing,
                stage_a_summary_path=args.stage_a_summary,
            )
            validation = load_stage_a_validation_contract(
                args.stage_a_validation,
                expected_file_sha256=args.stage_a_validation_sha256,
                runtime=runtime,
                routing=routing,
                stage_a_summary_path=args.stage_a_summary,
                dataset_split="full",
            )

        path = (
            args.output_dir
            / f"silent_bias_stage_{stage.lower()}_{model.slug}.slurm"
        )
        _write_new_text(
            path,
            render_silent_bias_job(
                stage=stage,
                job_name=f"silent-{stage.lower()}-{model.slug}",
                model=model,
                time=args.time,
                cpus_per_task=args.cpus_per_task,
                gpus=int(runtime.values["tensor_parallel_size"]),
                mem=args.mem or model.mem,
                partition=args.partition,
                qos=args.qos,
                account=args.account,
                artifact_root=args.artifact_root,
                run_group=run_group,
                runtime=runtime,
                routing=routing,
                python_bin=args.python_bin,
                stage_b_release=release,
                stage_a_validation=validation,
            ),
        )
        rendered.append(path)
    print("Rendered", len(rendered), "jobs")
    for path in rendered:
        print(path)


if __name__ == "__main__":
    main()
