#!/bin/bash
#SBATCH --job-name=@@JOB_NAME@@
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=@@CPUS_PER_TASK@@
#SBATCH --gres=gpu:@@GPUS@@
#SBATCH --mem=@@MEM@@
#SBATCH --time=@@TIME@@
@@OPTIONAL_SBATCH@@#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Portable two-stage Silent Bias job template. Every scheduler directive is a
# starting point and must be reviewed for the active infrastructure.
set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
ARTIFACT_ROOT="@@ARTIFACT_ROOT@@"
export BIASES_ARTIFACT_ROOT="${ARTIFACT_ROOT}"

SILENT_BIAS_STAGE="@@STAGE@@"
DATA_PATH=@@DATA_PATH@@
DATA_SHA256="@@DATA_SHA256@@"
ROUTING_MANIFEST=@@ROUTING_MANIFEST@@
ROUTING_MANIFEST_SHA256="@@ROUTING_MANIFEST_SHA256@@"
ROUTING_ASSIGNMENT_SHA256="@@ROUTING_ASSIGNMENT_SHA256@@"
RUNTIME_JSON=@@RUNTIME_JSON@@
RUNTIME_FILE_SHA256="@@RUNTIME_FILE_SHA256@@"
RUNTIME_EMBEDDED_SHA256="@@RUNTIME_EMBEDDED_SHA256@@"
RUNTIME_CANONICAL_SHA256="@@RUNTIME_CANONICAL_SHA256@@"
STAGE_B_PREFLIGHT=@@STAGE_B_PREFLIGHT@@
STAGE_B_PREFLIGHT_SHA256="@@STAGE_B_PREFLIGHT_SHA256@@"
STAGE_A_SUMMARY_SHA256="@@STAGE_A_SUMMARY_SHA256@@"
STAGE_A_VALIDATION=@@STAGE_A_VALIDATION@@
STAGE_A_VALIDATION_SHA256="@@STAGE_A_VALIDATION_SHA256@@"
LAUNCH_BINDING_B64=@@LAUNCH_BINDING_B64@@
DATASET_SPLIT="@@DATASET_SPLIT@@"
MODEL_NAME="@@MODEL_NAME@@"
MODEL_REGISTRY_NAME="@@MODEL_REGISTRY_NAME@@"
MODEL_REVISION="@@MODEL_REVISION@@"
MODEL_SLUG="@@MODEL_SLUG@@"
RUN_GROUP="@@RUN_GROUP@@"
RUN_ROOT="${ARTIFACT_ROOT}/outputs/${RUN_GROUP}/${MODEL_SLUG}"
STAGE_A_OUTPUT_DIR="${RUN_ROOT}/stage_a"
STAGE_B_OUTPUT_DIR="${RUN_ROOT}/stage_b"
STAGE_A_SUMMARY=@@STAGE_A_SUMMARY@@

STAGE_A_COMMAND="@@STAGE_A_COMMAND@@"
STAGE_B_COMMAND="@@STAGE_B_COMMAND@@"
CONSISTENCY_RUNS="@@CONSISTENCY_RUNS@@"
CONSISTENCY_SCHEDULE="@@CONSISTENCY_SCHEDULE@@"
SAMPLING_TEMPERATURE="@@SAMPLING_TEMPERATURE@@"
INCLUDE_VERBALIZED_CONFIDENCE="@@INCLUDE_VERBALIZED@@"

TENSOR_PARALLEL_SIZE="@@TENSOR_PARALLEL_SIZE@@"
MAX_MODEL_LEN="@@MAX_MODEL_LEN@@"
BATCH_SIZE="@@BATCH_SIZE@@"
MAX_NUM_BATCHED_TOKENS="@@MAX_NUM_BATCHED_TOKENS@@"
MAX_NUM_SEQS="@@MAX_NUM_SEQS@@"
STAGE_B_ROUTING_SPLIT="@@STAGE_B_ROUTING_SPLIT@@"
GPU_MEMORY_UTILIZATION="@@GPU_MEMORY_UTILIZATION@@"
DTYPE="@@DTYPE@@"
ENFORCE_EAGER="@@ENFORCE_EAGER@@"
DISABLE_CUSTOM_ALL_REDUCE="@@DISABLE_CUSTOM_ALL_REDUCE@@"
PYTHON_BIN=@@PYTHON_BIN@@

TMPDIR="${TMPDIR:-/tmp/${USER:-user}/${SLURM_JOB_ID:-manual}}"
HF_ENV_FILE="${BIASES_HF_ENV:-${ARTIFACT_ROOT}/secrets/hf.env}"

if [[ -f "${HF_ENV_FILE}" ]]; then
  set -a
  source "${HF_ENV_FILE}"
  set +a
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Pinned Python is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi

case "${SILENT_BIAS_STAGE}" in
  A|a|stage_a)
    OUTPUT_DIR="${STAGE_A_OUTPUT_DIR}"
    ;;
  B|b|stage_b)
    OUTPUT_DIR="${STAGE_B_OUTPUT_DIR}"
    ;;
  *)
    echo "SILENT_BIAS_STAGE must be A or B, got: ${SILENT_BIAS_STAGE}" >&2
    exit 2
    ;;
esac

if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Fresh scientific stage output already exists: ${OUTPUT_DIR}" >&2
  echo "Use a new immutable run_group for recovery; resume is disabled." >&2
  exit 2
fi

mkdir -p \
  "${ARTIFACT_ROOT}/cache" \
  "$(dirname "${OUTPUT_DIR}")" \
  "${TMPDIR}"
mkdir "${OUTPUT_DIR}"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

export HF_HOME="${HF_HOME:-${ARTIFACT_ROOT}/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export BIASES_VLLM_ENFORCE_EAGER="${ENFORCE_EAGER}"
export VLLM_DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${ARTIFACT_ROOT}/cache/xdg}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${ARTIFACT_ROOT}/cache/vllm}"
export TORCH_HOME="${TORCH_HOME:-${ARTIFACT_ROOT}/cache/torch}"
export TMPDIR
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${TMPDIR}/triton}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${ARTIFACT_ROOT}/cache/uv}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p \
  "${HF_HOME}" \
  "${HF_HUB_CACHE}" \
  "${HF_DATASETS_CACHE}" \
  "${XDG_CACHE_HOME}" \
  "${VLLM_CACHE_ROOT}" \
  "${TORCH_HOME}" \
  "${TRITON_CACHE_DIR}" \
  "${UV_CACHE_DIR}"

# This gate intentionally executes before main.py can construct VLLMJudge or
# load model weights. The launcher embeds the expected immutable bindings and
# revalidates every referenced file in the active execution environment.
"${PYTHON_BIN}" - "${LAUNCH_BINDING_B64}" <<'PY'
from __future__ import annotations

import base64
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path


def fail(message: str) -> None:
    raise SystemExit(f"Silent Bias launch gate failed: {message}")


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_mapping(path: Path, name: str) -> dict[str, object]:
    require(path.is_file(), f"{name} is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{name} is not a JSON object")
    return value


def distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


binding = json.loads(base64.b64decode(sys.argv[1], validate=True))
require(binding.get("schema_version") == 1, "launch binding schema mismatch")
stage = binding.get("stage")
require(stage in {"A", "B"}, "invalid launch stage")

runtime_binding = binding["runtime"]
runtime_path = Path(runtime_binding["path"]).resolve()
require(
    file_sha256(runtime_path) == runtime_binding["file_sha256"],
    "runtime file SHA-256 mismatch",
)
runtime = read_mapping(runtime_path, "runtime JSON")
embedded = runtime.get("runtime_sha256")
require(embedded == runtime_binding["embedded_sha256"], "runtime embedded SHA mismatch")
runtime_payload = dict(runtime)
runtime_payload.pop("runtime_sha256", None)
require(
    canonical_sha256(runtime_payload) == embedded,
    "runtime embedded SHA does not match its mapping",
)
require(
    canonical_sha256(runtime) == runtime_binding["canonical_sha256"],
    "runtime canonical SHA mismatch",
)

model_binding = binding["model"]
require(runtime.get("model_registry_name") == model_binding["registry_name"], "runtime registry mismatch")
require(runtime.get("model_hf_name") == model_binding["hf_name"], "runtime model mismatch")
require(runtime.get("model_revision") == model_binding["revision"], "runtime revision mismatch")
from biases.models import get_model_profile

profile = get_model_profile(model_binding["registry_name"])
require(profile.registry_name == model_binding["registry_name"], "active registry mismatch")
require(profile.hf_model_name == model_binding["hf_name"], "active model profile mismatch")
require(profile.revision == model_binding["revision"], "active model revision mismatch")

active_versions = {
    "python": platform.python_version(),
    "torch": distribution_version("torch"),
    "transformers": distribution_version("transformers"),
    "vllm": distribution_version("vllm"),
}
require(runtime.get("engine_versions") == active_versions, "active engine versions mismatch")
import torch

visible_gpus = torch.cuda.device_count()
require(
    visible_gpus == binding["tensor_parallel_size"],
    f"visible GPU count {visible_gpus} != frozen tensor parallel size {binding['tensor_parallel_size']}",
)

routing_binding = binding["routing"]
routing_path = Path(routing_binding["manifest_path"]).resolve()
require(
    file_sha256(routing_path) == routing_binding["manifest_file_sha256"],
    "routing manifest SHA-256 mismatch",
)
routing = read_mapping(routing_path, "routing manifest")
require(routing.get("schema_version") == 2, "routing schema mismatch")
require(
    routing.get("artifact_type") == "frozen_question_disjoint_routing_package",
    "routing artifact type mismatch",
)
require(
    routing.get("routing_assignment_sha256")
    == routing_binding["routing_assignment_sha256"],
    "routing assignment SHA mismatch",
)
data_path = Path(routing_binding["data_path"]).resolve()
require(file_sha256(data_path) == routing_binding["data_sha256"], "routed data SHA mismatch")
full_output = routing["outputs"]["full"]
require(
    (routing_path.parent / full_output["path"]).resolve() == data_path,
    "routing full-output path mismatch",
)
require(
    routing["output_sha256"]["full"] == routing_binding["data_sha256"],
    "routing full-output SHA mismatch",
)

if stage == "B":
    stage_b = binding.get("stage_b")
    require(isinstance(stage_b, dict), "Stage B release binding is absent")
    preflight_path = Path(stage_b["preflight_path"]).resolve()
    require(
        file_sha256(preflight_path) == stage_b["preflight_file_sha256"],
        "Stage B preflight SHA-256 mismatch",
    )
    report = read_mapping(preflight_path, "Stage B preflight")
    gate = report.get("release_gate")
    require(report.get("status") == "complete", "Stage B preflight is not complete")
    require(report.get("passed") is True, "Stage B preflight did not pass")
    require(isinstance(gate, dict) and gate.get("stage_b_authorized") is True, "Stage B is not authorized")
    require(report["runtime"]["inference_runtime"] == runtime, "preflight runtime mapping mismatch")
    require(
        report["runtime"]["inference_runtime_sha256"]
        == runtime_binding["canonical_sha256"],
        "preflight runtime SHA mismatch",
    )
    require(report["model"]["registry_name"] == model_binding["registry_name"], "preflight registry mismatch")
    require(report["model"]["model_name"] == model_binding["hf_name"], "preflight model mismatch")
    require(report["model"]["revision"] == model_binding["revision"], "preflight revision mismatch")
    require(Path(report["source"]["path"]).resolve() == data_path, "preflight source path mismatch")
    require(report["source"]["sha256"] == routing_binding["data_sha256"], "preflight source SHA mismatch")
    require(Path(report["routing"]["manifest_path"]).resolve() == routing_path, "preflight routing path mismatch")
    require(
        report["routing"]["manifest_file_sha256"]
        == routing_binding["manifest_file_sha256"],
        "preflight routing SHA mismatch",
    )
    require(
        report["routing"]["routing_assignment_sha256"]
        == routing_binding["routing_assignment_sha256"],
        "preflight routing assignment mismatch",
    )
    require(
        report["scope"]["stage_b_routing_split"] == stage_b["routing_split"],
        "preflight Stage B routing split mismatch",
    )
    summary_path = Path(stage_b["stage_a_summary_path"]).resolve()
    require(
        file_sha256(summary_path) == stage_b["stage_a_summary_sha256"],
        "Stage A summary SHA-256 mismatch",
    )
    summary_binding = report["plan"]["stage_a_summary"]
    require(summary_binding["mode"] == "exact_post_stage_a", "preflight is not exact post-Stage-A")
    require(Path(summary_binding["path"]).resolve() == summary_path, "preflight Stage A summary path mismatch")
    require(
        summary_binding["file_sha256"] == stage_b["stage_a_summary_sha256"],
        "preflight Stage A summary SHA mismatch",
    )
    require(
        summary_binding["row_count"] == stage_b["stage_a_summary_row_count"],
        "preflight Stage A summary row-count mismatch",
    )
    require(
        report["plan"]["actual_stage_b_prompt_set_sha256"] is not None,
        "preflight lacks the exact Stage B prompt-set SHA",
    )
    require(
        report["plan"]["provisional_structural_stage_b_prompt_set_sha256"] is None,
        "preflight still carries a provisional Stage B prompt-set SHA",
    )

    validation_path = Path(stage_b["stage_a_validation_path"]).resolve()
    require(
        file_sha256(validation_path)
        == stage_b["stage_a_validation_file_sha256"],
        "Stage A validation file SHA-256 mismatch",
    )
    validation = read_mapping(validation_path, "Stage A artifact validation")
    require(
        validation.get("passed") is True and validation.get("error_count") == 0,
        "Stage A artifact validation did not pass",
    )
    validation_design = validation.get("design")
    require(isinstance(validation_design, dict), "Stage A validation design is absent")
    require(
        validation_design.get("validation_scope") == "stage_a",
        "Stage A validation scope mismatch",
    )
    require(
        validation_design.get("expected_inference_runtime_sha256")
        == runtime_binding["canonical_sha256"],
        "Stage A validation expected runtime SHA mismatch",
    )
    require(
        validation_design.get("expected_question_routing_sha256")
        == routing_binding["routing_assignment_sha256"],
        "Stage A validation expected routing SHA mismatch",
    )
    require(
        validation_design.get("consistency_runs") == runtime["consistency_runs"]
        and validation_design.get("consistency_schedule")
        == runtime["consistency_schedule"]
        and validation_design.get("sampling_temperature")
        == runtime["sampling_temperature"]
        and validation_design.get("dataset_split") == binding["dataset_split"],
        "Stage A validation scientific runtime mismatch",
    )
    validation_routing = validation_design.get("question_routing")
    require(isinstance(validation_routing, dict), "Stage A validation routing is absent")
    require(
        validation_routing.get("expected_assignment_sha256")
        == routing_binding["routing_assignment_sha256"]
        and validation_routing.get("assignment_sha256")
        == routing_binding["routing_assignment_sha256"]
        and validation_routing.get("raw_question_assignment_sha256")
        == routing_binding["routing_assignment_sha256"],
        "Stage A validation observed routing SHA mismatch",
    )
    validation_source = validation.get("source")
    require(isinstance(validation_source, dict), "Stage A validation source is absent")
    require(
        Path(validation_source.get("csv", "")).resolve() == data_path
        and validation_source.get("input_file_hash")
        == routing_binding["data_sha256"],
        "Stage A validation source-data mismatch",
    )
    validation_artifacts = validation.get("artifacts")
    require(
        isinstance(validation_artifacts, list) and len(validation_artifacts) == 1,
        "Stage A validation must contain exactly one artifact",
    )
    validation_artifact = validation_artifacts[0]
    require(isinstance(validation_artifact, dict), "Stage A validation artifact is invalid")
    require(
        validation_artifact.get("validation_scope") == "stage_a"
        and validation_artifact.get("model_name") == model_binding["hf_name"]
        and validation_artifact.get("model_revision") == model_binding["revision"]
        and validation_artifact.get("dataset_split") == binding["dataset_split"]
        and validation_artifact.get("inference_runtime_sha256_by_stage")
        == {"stage_a": runtime_binding["canonical_sha256"]},
        "Stage A validation artifact runtime/model mismatch",
    )
    require(
        Path(validation_artifact["stage_dirs"]["stage_a"]).resolve()
        == summary_path.parent,
        "Stage A validation directory does not bind the summary",
    )
    validation_counts = validation_artifact.get("counts")
    require(isinstance(validation_counts, dict), "Stage A validation counts are absent")
    require(
        validation_counts.get("source_pairs") == stage_b["source_pair_count"]
        and validation_counts.get("stage_a_expected")
        == stage_b["stage_a_expected_count"]
        and stage_b["stage_a_expected_count"] == 2 * stage_b["source_pair_count"]
        and validation_counts.get("stage_a_raw")
        == stage_b["stage_a_expected_count"]
        and validation_counts.get("stage_a_flat")
        == stage_b["stage_a_expected_count"]
        and validation_counts.get("stage_a_pair_summary")
        == stage_b["stage_a_expected_count"],
        "Stage A validation expected/observed counts mismatch",
    )
    require(
        stage_b["stage_a_summary_row_count"]
        == stage_b["stage_a_expected_count"],
        "Stage A summary row count does not match validation",
    )

print(json.dumps({"status": "passed", "stage": stage, "visible_gpus": visible_gpus}, sort_keys=True))
PY

COMMON_ARGS=(
  --data-path "${DATA_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --dataset-split "${DATASET_SPLIT}"
  --model-name "${MODEL_NAME}"
  --consistency-runs "${CONSISTENCY_RUNS}"
  --consistency-schedule "${CONSISTENCY_SCHEDULE}"
  --sampling-temperature "${SAMPLING_TEMPERATURE}"
  --batch-size "${BATCH_SIZE}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --dtype "${DTYPE}"
  --no-resume
)

if [[ "${INCLUDE_VERBALIZED_CONFIDENCE}" != "1" ]]; then
  COMMON_ARGS+=(--skip-verbalized-confidence)
fi
if [[ -n "${MAX_NUM_BATCHED_TOKENS}" ]]; then
  COMMON_ARGS+=(--max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}")
fi
if [[ -n "${MAX_NUM_SEQS}" ]]; then
  COMMON_ARGS+=(--max-num-seqs "${MAX_NUM_SEQS}")
fi
if [[ "${ENFORCE_EAGER}" == "1" ]]; then
  COMMON_ARGS+=(--enforce-eager)
else
  COMMON_ARGS+=(--no-enforce-eager)
fi
if [[ "${DISABLE_CUSTOM_ALL_REDUCE}" == "1" ]]; then
  COMMON_ARGS+=(--disable-custom-all-reduce)
else
  COMMON_ARGS+=(--no-disable-custom-all-reduce)
fi

echo "Stage: ${SILENT_BIAS_STAGE}"
echo "Model: ${MODEL_NAME}"
echo "Data: ${DATA_PATH}"
echo "Output: ${OUTPUT_DIR}"

if [[ "${SILENT_BIAS_STAGE}" == "A" || "${SILENT_BIAS_STAGE}" == "a" || "${SILENT_BIAS_STAGE}" == "stage_a" ]]; then
  "${PYTHON_BIN}" main.py "${STAGE_A_COMMAND}" \
    "${COMMON_ARGS[@]}"
else
  "${PYTHON_BIN}" main.py "${STAGE_B_COMMAND}" \
    "${COMMON_ARGS[@]}" \
    --stage-b-routing-split "${STAGE_B_ROUTING_SPLIT}" \
    --stage-a-summary "${STAGE_A_SUMMARY}"
fi
