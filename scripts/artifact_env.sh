#!/usr/bin/env bash

# Source this file before local setup/runs:
#   source scripts/artifact_env.sh
#
# Set BIASES_ARTIFACT_ROOT first on shared infrastructure:
#   export BIASES_ARTIFACT_ROOT=/path/to/biases-artifacts

_biases_user="${USER:-user}"
_biases_job="${SLURM_JOB_ID:-manual}"
export BIASES_ARTIFACT_ROOT="${BIASES_ARTIFACT_ROOT:-$(pwd)/artifacts}"

export HF_HOME="${HF_HOME:-${BIASES_ARTIFACT_ROOT}/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${BIASES_ARTIFACT_ROOT}/cache/xdg}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${BIASES_ARTIFACT_ROOT}/cache/vllm}"
export TORCH_HOME="${TORCH_HOME:-${BIASES_ARTIFACT_ROOT}/cache/torch}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${BIASES_ARTIFACT_ROOT}/cache/uv}"
export TMPDIR="${TMPDIR:-/tmp/${_biases_user}/${_biases_job}}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${TMPDIR}/triton}"

# Uncomment or override this to keep the whole virtualenv outside the repo.
# export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${BIASES_ARTIFACT_ROOT}/venvs/biases}"

mkdir -p \
  "${BIASES_ARTIFACT_ROOT}/cache" \
  "${BIASES_ARTIFACT_ROOT}/data/processed" \
  "${BIASES_ARTIFACT_ROOT}/outputs" \
  "${TMPDIR}" \
  "${TRITON_CACHE_DIR}"
