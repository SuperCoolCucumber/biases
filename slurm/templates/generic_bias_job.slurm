#!/bin/bash
#SBATCH --job-name=bias-run
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
# Add infrastructure-specific #SBATCH lines here, for example partition,
# account, or QOS.
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
ARTIFACT_ROOT="${BIASES_ARTIFACT_ROOT:-${REPO_DIR}/artifacts}"
export BIASES_ARTIFACT_ROOT="${ARTIFACT_ROOT}"

COMMAND="${COMMAND:-run-position}"
DATA_PATH="${DATA_PATH:-${ARTIFACT_ROOT}/data/processed/mtbench_full.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${ARTIFACT_ROOT}/outputs/${COMMAND}_${SLURM_JOB_ID:-manual}}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-14B}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
DTYPE="${DTYPE:-bfloat16}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
TMPDIR="${TMPDIR:-/tmp/${USER:-user}/${SLURM_JOB_ID:-manual}}"
HF_ENV_FILE="${BIASES_HF_ENV:-${ARTIFACT_ROOT}/secrets/hf.env}"

if [ -f "${HF_ENV_FILE}" ]; then
  set -a
  source "${HF_ENV_FILE}"
  set +a
fi

mkdir -p "${REPO_DIR}/logs" "${ARTIFACT_ROOT}/cache" "${OUTPUT_DIR}" "${TMPDIR}"
cd "${REPO_DIR}"

VENV_PATH="${VENV_PATH:-${UV_PROJECT_ENVIRONMENT:-${REPO_DIR}/.venv}}"
if [ -f "${VENV_PATH}/bin/activate" ]; then
  source "${VENV_PATH}/bin/activate"
fi

export HF_HOME="${HF_HOME:-${ARTIFACT_ROOT}/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
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

echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

python main.py "${COMMAND}" \
  --data-path "${DATA_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --model-name "${MODEL_NAME}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --dtype "${DTYPE}" \
  ${EXTRA_ARGS}
