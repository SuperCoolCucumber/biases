from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


DEFAULT_ARTIFACT_ROOT_EXPR = "${REPO_DIR}/artifacts"
SILENT_BIAS_TEMPLATE_PATH = (
    Path(__file__).resolve().parents[1]
    / "slurm"
    / "templates"
    / "silent_bias_job.slurm"
)


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
    "qwen3_4b": ModelSpec("qwen3_4b", "Qwen/Qwen3-4B", 1, "64G", 1),
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
    data_file: str,
    time: str,
    cpus_per_task: int,
    gpus: int,
    mem: str,
    partition: str | None,
    qos: str | None,
    account: str | None,
    artifact_root: str,
    consistency_runs: int,
    consistency_schedule: str,
    sampling_temperature: float,
    include_verbalized_confidence: bool,
    limit: int | None,
    stage_a_command: str,
    stage_b_command: str,
    stage_a_summary_file: str,
    max_model_len: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    dtype: str,
) -> str:
    """Render one portable Stage A or Stage B Silent Bias job."""

    normalized_stage = stage.upper()
    if normalized_stage not in {"A", "B"}:
        raise ValueError("stage must be A or B")
    if consistency_runs < 0:
        raise ValueError("consistency_runs must be non-negative")
    if consistency_schedule not in {"all", "extremes"}:
        raise ValueError("consistency_schedule must be 'all' or 'extremes'")
    if cpus_per_task < 1 or gpus < 1 or tensor_parallel_size < 1:
        raise ValueError("CPU, GPU, and tensor-parallel counts must be positive")

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
        "@@STAGE@@": normalized_stage,
        "@@DATA_FILE@@": data_file,
        "@@MODEL_NAME@@": model.model_name,
        "@@MODEL_SLUG@@": model.slug,
        "@@STAGE_A_COMMAND@@": stage_a_command,
        "@@STAGE_B_COMMAND@@": stage_b_command,
        "@@CONSISTENCY_RUNS@@": str(consistency_runs),
        "@@CONSISTENCY_SCHEDULE@@": consistency_schedule,
        "@@SAMPLING_TEMPERATURE@@": str(sampling_temperature),
        "@@INCLUDE_VERBALIZED@@": (
            "1" if include_verbalized_confidence else "0"
        ),
        "@@LIMIT@@": "" if limit is None else str(limit),
        "@@TENSOR_PARALLEL_SIZE@@": str(tensor_parallel_size),
        "@@MAX_MODEL_LEN@@": str(max_model_len),
        "@@GPU_MEMORY_UTILIZATION@@": str(gpu_memory_utilization),
        "@@DTYPE@@": dtype,
        "@@ENFORCE_EAGER@@": "1" if model.enforce_eager else "0",
        "@@STAGE_A_SUMMARY_FILE@@": stage_a_summary_file,
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
        "--models",
        nargs="+",
        choices=sorted(MODEL_SPECS),
        default=None,
        help=(
            "Model keys for --kind silent-bias. Defaults to qwen3_14b; "
            "repeat values to render multiple model families."
        ),
    )
    parser.add_argument("--data-file", default="mtbench_full.csv")
    parser.add_argument("--time", default="48:00:00")
    parser.add_argument("--cpus-per-task", type=int, default=16)
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="Override each selected model template's GPU count.",
    )
    parser.add_argument(
        "--mem",
        default=None,
        help="Override each selected model template's memory request.",
    )
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="Override each selected model template's tensor-parallel size.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
    )
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--consistency-runs", type=int, default=8)
    parser.add_argument(
        "--consistency-schedule",
        choices=["all", "extremes"],
        default="all",
    )
    parser.add_argument("--sampling-temperature", type=float, default=0.7)
    parser.add_argument("--skip-verbalized-confidence", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--stage-a-command",
        default="run-silent-bias-clean",
    )
    parser.add_argument(
        "--stage-b-command",
        default="run-silent-bias-cued",
    )
    parser.add_argument(
        "--stage-a-summary-file",
        default="silent_bias_stage_a_pair_summary.jsonl",
        help=(
            "Filename written under STAGE_A_OUTPUT_DIR and passed to Stage B. "
            "STAGE_A_SUMMARY can override the full path at submission time."
        ),
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
    elif args.kind == "phase3":
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
    else:
        model_keys = args.models or ["qwen3_14b"]
        for model_key in model_keys:
            model = MODEL_SPECS[model_key]
            for stage in ("A", "B"):
                stage_slug = f"stage_{stage.lower()}"
                path = (
                    args.output_dir
                    / f"silent_bias_{stage_slug}_{model.slug}.slurm"
                )
                path.write_text(
                    render_silent_bias_job(
                        stage=stage,
                        job_name=f"silent-{stage.lower()}-{model.slug}",
                        model=model,
                        data_file=args.data_file,
                        time=args.time,
                        cpus_per_task=args.cpus_per_task,
                        gpus=args.gpus if args.gpus is not None else model.gpus,
                        mem=args.mem or model.mem,
                        partition=args.partition,
                        qos=args.qos,
                        account=args.account,
                        artifact_root=args.artifact_root,
                        consistency_runs=args.consistency_runs,
                        consistency_schedule=args.consistency_schedule,
                        sampling_temperature=args.sampling_temperature,
                        include_verbalized_confidence=(
                            not args.skip_verbalized_confidence
                        ),
                        limit=args.limit,
                        stage_a_command=args.stage_a_command,
                        stage_b_command=args.stage_b_command,
                        stage_a_summary_file=args.stage_a_summary_file,
                        max_model_len=args.max_model_len,
                        tensor_parallel_size=(
                            args.tensor_parallel_size
                            if args.tensor_parallel_size is not None
                            else model.tensor_parallel_size
                        ),
                        gpu_memory_utilization=(
                            args.gpu_memory_utilization
                            if args.gpu_memory_utilization is not None
                            else model.gpu_memory_utilization
                        ),
                        dtype=args.dtype or model.dtype,
                    ),
                    encoding="utf-8",
                )
                rendered.append(path)
    print("Rendered", len(rendered), "jobs")
    for path in rendered:
        print(path)


if __name__ == "__main__":
    main()
