from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


DEFAULT_ARTIFACT_ROOT_EXPR = "${REPO_DIR}/artifacts"


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


MODEL_SPECS = {
    "qwen35_4b": ModelSpec("qwen35_4b", "Qwen/Qwen3.5-4B", 1, "96G", 1),
    "qwen35_9b": ModelSpec("qwen35_9b", "Qwen/Qwen3.5-9B", 1, "128G", 1),
    "qwen35_27b": ModelSpec("qwen35_27b", "Qwen/Qwen3.5-27B", 2, "220G", 2, gpu_memory_utilization=0.92),
    "qwen3_14b": ModelSpec("qwen3_14b", "Qwen/Qwen3-14B", 1, "160G", 1),
    "qwen3_32b": ModelSpec("qwen3_32b", "Qwen/Qwen3-32B", 2, "240G", 2),
    "mistral7b": ModelSpec("mistral7b", "mistralai/Mistral-7B-Instruct-v0.3", 1, "128G", 1),
    "gemma2_27b": ModelSpec(
        "gemma2_27b",
        "google/gemma-2-27b-it",
        2,
        "240G",
        2,
        enforce_eager=True,
    ),
    "skywork_critic_8b": ModelSpec(
        "skywork_critic_8b",
        "Skywork/Skywork-Critic-Llama-3.1-8B",
        1,
        "128G",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render portable Slurm jobs.")
    parser.add_argument("--output-dir", type=Path, default=Path("slurm/generated"))
    parser.add_argument("--kind", choices=["controls", "phase3"], required=True)
    parser.add_argument("--partition", default=None, help="Optional Slurm partition.")
    parser.add_argument("--qos", default=None, help="Optional Slurm QOS.")
    parser.add_argument("--account", default=None, help="Optional Slurm account.")
    parser.add_argument(
        "--artifact-root",
        default=DEFAULT_ARTIFACT_ROOT_EXPR,
        help="Default BIASES_ARTIFACT_ROOT expression to render into jobs.",
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
                path.write_text(
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
                    encoding="utf-8",
                )
                rendered.append(path)
    else:
        for model_key in ("mistral7b", "gemma2_27b", "skywork_critic_8b"):
            model = MODEL_SPECS[model_key]
            for bias, command in BIAS_COMMANDS.items():
                path = args.output_dir / f"{bias}_{model.slug}_mtbench_full.slurm"
                path.write_text(
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
                    encoding="utf-8",
                )
                rendered.append(path)
    print("Rendered", len(rendered), "jobs")
    for path in rendered:
        print(path)


if __name__ == "__main__":
    main()
