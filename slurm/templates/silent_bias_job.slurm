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

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
ARTIFACT_ROOT="${BIASES_ARTIFACT_ROOT:-@@ARTIFACT_ROOT@@}"
export BIASES_ARTIFACT_ROOT="${ARTIFACT_ROOT}"

SILENT_BIAS_STAGE="${SILENT_BIAS_STAGE:-@@STAGE@@}"
DATA_PATH="${DATA_PATH:-${ARTIFACT_ROOT}/data/processed/@@DATA_FILE@@}"
DATASET_SPLIT="${DATASET_SPLIT:-full}"
MODEL_NAME="${MODEL_NAME:-@@MODEL_NAME@@}"
MODEL_SLUG="${MODEL_SLUG:-@@MODEL_SLUG@@}"
RUN_GROUP="${RUN_GROUP:-silent_bias_${MODEL_SLUG}}"
RUN_ROOT="${RUN_ROOT:-${ARTIFACT_ROOT}/outputs/${RUN_GROUP}}"
STAGE_A_OUTPUT_DIR="${STAGE_A_OUTPUT_DIR:-${RUN_ROOT}/stage_a}"
STAGE_B_OUTPUT_DIR="${STAGE_B_OUTPUT_DIR:-${RUN_ROOT}/stage_b}"
STAGE_A_SUMMARY="${STAGE_A_SUMMARY:-${STAGE_A_OUTPUT_DIR}/@@STAGE_A_SUMMARY_FILE@@}"

STAGE_A_COMMAND="${STAGE_A_COMMAND:-@@STAGE_A_COMMAND@@}"
STAGE_B_COMMAND="${STAGE_B_COMMAND:-@@STAGE_B_COMMAND@@}"
CONSISTENCY_RUNS="${CONSISTENCY_RUNS:-@@CONSISTENCY_RUNS@@}"
CONSISTENCY_SCHEDULE="${CONSISTENCY_SCHEDULE:-@@CONSISTENCY_SCHEDULE@@}"
SAMPLING_TEMPERATURE="${SAMPLING_TEMPERATURE:-@@SAMPLING_TEMPERATURE@@}"
INCLUDE_VERBALIZED_CONFIDENCE="${INCLUDE_VERBALIZED_CONFIDENCE:-@@INCLUDE_VERBALIZED@@}"
LIMIT="${LIMIT:-@@LIMIT@@}"

TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-@@TENSOR_PARALLEL_SIZE@@}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-@@MAX_MODEL_LEN@@}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-@@GPU_MEMORY_UTILIZATION@@}"
DTYPE="${DTYPE:-@@DTYPE@@}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

TMPDIR="${TMPDIR:-/tmp/${USER:-user}/${SLURM_JOB_ID:-manual}}"
HF_ENV_FILE="${BIASES_HF_ENV:-${ARTIFACT_ROOT}/secrets/hf.env}"

if [[ -f "${HF_ENV_FILE}" ]]; then
  set -a
  source "${HF_ENV_FILE}"
  set +a
fi

if [[ ! -f "${DATA_PATH}" ]]; then
  echo "Dataset not found: ${DATA_PATH}" >&2
  exit 2
fi

case "${SILENT_BIAS_STAGE}" in
  A|a|stage_a)
    OUTPUT_DIR="${OUTPUT_DIR:-${STAGE_A_OUTPUT_DIR}}"
    ;;
  B|b|stage_b)
    OUTPUT_DIR="${OUTPUT_DIR:-${STAGE_B_OUTPUT_DIR}}"
    if [[ ! -f "${STAGE_A_SUMMARY}" ]]; then
      echo "Stage A summary not found: ${STAGE_A_SUMMARY}" >&2
      exit 2
    fi
    ;;
  *)
    echo "SILENT_BIAS_STAGE must be A or B, got: ${SILENT_BIAS_STAGE}" >&2
    exit 2
    ;;
esac

mkdir -p \
  "${ARTIFACT_ROOT}/cache" \
  "${OUTPUT_DIR}" \
  "${TMPDIR}"
cd "${REPO_DIR}"

VENV_PATH="${VENV_PATH:-${UV_PROJECT_ENVIRONMENT:-${REPO_DIR}/.venv}}"
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
fi

export HF_HOME="${HF_HOME:-${ARTIFACT_ROOT}/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export BIASES_VLLM_ENFORCE_EAGER="${BIASES_VLLM_ENFORCE_EAGER:-@@ENFORCE_EAGER@@}"
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

COMMON_ARGS=(
  --data-path "${DATA_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --dataset-split "${DATASET_SPLIT}"
  --model-name "${MODEL_NAME}"
  --consistency-runs "${CONSISTENCY_RUNS}"
  --consistency-schedule "${CONSISTENCY_SCHEDULE}"
  --sampling-temperature "${SAMPLING_TEMPERATURE}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --dtype "${DTYPE}"
)

if [[ -n "${LIMIT}" ]]; then
  COMMON_ARGS+=(--limit "${LIMIT}")
fi
if [[ "${INCLUDE_VERBALIZED_CONFIDENCE}" != "1" ]]; then
  COMMON_ARGS+=(--skip-verbalized-confidence)
fi

EXTRA_ARG_ARRAY=()
if [[ -n "${EXTRA_ARGS}" ]]; then
  read -r -a EXTRA_ARG_ARRAY <<< "${EXTRA_ARGS}"
fi

echo "Stage: ${SILENT_BIAS_STAGE}"
echo "Model: ${MODEL_NAME}"
echo "Data: ${DATA_PATH}"
echo "Output: ${OUTPUT_DIR}"

if [[ "${SILENT_BIAS_STAGE}" == "A" || "${SILENT_BIAS_STAGE}" == "a" || "${SILENT_BIAS_STAGE}" == "stage_a" ]]; then
  "${PYTHON_BIN}" main.py "${STAGE_A_COMMAND}" \
    "${COMMON_ARGS[@]}" \
    "${EXTRA_ARG_ARRAY[@]}"
else
  "${PYTHON_BIN}" main.py "${STAGE_B_COMMAND}" \
    "${COMMON_ARGS[@]}" \
    --stage-a-summary "${STAGE_A_SUMMARY}" \
    "${EXTRA_ARG_ARRAY[@]}"
fi
