from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from biases.authority_bias import run_authority_experiment
from biases.position_bias import (
    DEFAULT_MAX_MODEL_LEN,
    DEFAULT_MODEL_NAME,
    run_position_experiment,
)
from biases.schemas import (
    BiasCondition,
    BiasType,
    Candidate,
    ExperimentSpec,
    JudgeExample,
    LogitMetrics,
    OutputMode,
    PromptPackage,
    RunRecord,
    UncertaintyBundle,
    VerdictLabel,
)
from biases.utils import stable_hash


def build_demo_record() -> RunRecord:
    example = JudgeExample(
        example_id="demo-q1",
        question_id=1,
        prompt_messages=[{"role": "user", "content": "Explain the bias-variance tradeoff."}],
        candidates={
            "A": Candidate(
                label=VerdictLabel.A,
                response="It is the balance between underfitting and overfitting.",
                model_id="demo-a",
            ),
            "B": Candidate(
                label=VerdictLabel.B,
                response="It is the balance between variance and bias error components.",
                model_id="demo-b",
            ),
        },
        human_winner=VerdictLabel.B,
    )
    condition = BiasCondition(
        bias_type=BiasType.CONTROL,
        variant_id="control",
    )
    prompt_text = "Judge which response is better. Reply with A, B, or tie."
    prompt = PromptPackage(
        prompt_text=prompt_text,
        output_mode=OutputMode.CHOICE_ONLY,
        allowed_labels=[VerdictLabel.A, VerdictLabel.B, VerdictLabel.TIE],
        prompt_hash=stable_hash({"prompt": prompt_text}),
    )
    uncertainty = UncertaintyBundle(
        logit=LogitMetrics.from_probs({"A": 0.18, "B": 0.77, "tie": 0.05}),
    )
    return RunRecord(
        record_id=stable_hash({"example_id": example.example_id, "seed": 0}),
        spec=ExperimentSpec(
            dataset_name="demo",
            dataset_split="train",
            model_name="demo-judge",
            backend_name="manual",
            bias_name=condition.bias_type,
            output_mode=prompt.output_mode,
            uncertainty_methods=["logit"],
            consistency_runs=1,
            temperature=0.0,
        ),
        example_id=example.example_id,
        question_id=str(example.question_id),
        condition=condition,
        seed=0,
        verdict=VerdictLabel.B,
        raw_output="B",
        prompt_hash=prompt.prompt_hash,
        uncertainty=uncertainty,
        raw_prompt_logprobs={"A": 0.18, "B": 0.77, "tie": 0.05},
    )


def _add_common_vllm_args(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument(
        "--data-path",
        default="mtbench_stratified_198.csv",
        help="Path to the CSV file containing pairwise MT-Bench examples.",
    )
    subparser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face model name to run as the judge.",
    )
    subparser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of pairs to evaluate.",
    )
    subparser.add_argument(
        "--consistency-runs",
        type=int,
        default=5,
        help="Number of sampled runs per condition for consistency metrics.",
    )
    subparser.add_argument(
        "--sampling-temperature",
        type=float,
        default=0.7,
        help="Temperature used for sampled consistency runs.",
    )
    subparser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size passed to vLLM.",
    )
    subparser.add_argument(
        "--max-model-len",
        type=int,
        default=DEFAULT_MAX_MODEL_LEN,
        help="Maximum model length passed to vLLM.",
    )
    subparser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="Target GPU memory utilization for vLLM.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="biases",
        description="Utilities for the LLM judge bias and uncertainty project scaffold.",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser(
        "schema-demo",
        help="Print a sample RunRecord JSON payload built from the current schemas.",
    )

    position_parser = subparsers.add_parser(
        "run-position",
        help="Run the position-bias experiment over a CSV dataset.",
    )
    position_parser.add_argument(
        "--output-dir",
        default="outputs/position_bias_qwen2_5_14b_vllm",
        help="Directory where run artifacts will be written.",
    )
    _add_common_vllm_args(position_parser)

    authority_parser = subparsers.add_parser(
        "run-authority",
        help="Run the authority-bias experiment over a CSV dataset.",
    )
    authority_parser.add_argument(
        "--output-dir",
        default="outputs/authority_bias_qwen2_5_14b_vllm",
        help="Directory where run artifacts will be written.",
    )
    _add_common_vllm_args(authority_parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "schema-demo":
        print(json.dumps(build_demo_record().model_dump(mode="json"), indent=2))
        return 0

    if args.command == "run-position":
        summary = run_position_experiment(
            csv_path=Path(args.data_path),
            output_dir=Path(args.output_dir),
            model_name=args.model_name,
            limit=args.limit,
            consistency_runs=args.consistency_runs,
            sampling_temperature=args.sampling_temperature,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "run-authority":
        summary = run_authority_experiment(
            csv_path=Path(args.data_path),
            output_dir=Path(args.output_dir),
            model_name=args.model_name,
            limit=args.limit,
            consistency_runs=args.consistency_runs,
            sampling_temperature=args.sampling_temperature,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        print(json.dumps(summary, indent=2))
        return 0

    parser.print_help()
    return 0
