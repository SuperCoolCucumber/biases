from __future__ import annotations

import argparse
import json
from typing import Sequence

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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "schema-demo":
        print(json.dumps(build_demo_record().model_dump(mode="json"), indent=2))
        return 0

    parser.print_help()
    return 0
