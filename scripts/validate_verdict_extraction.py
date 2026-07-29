from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

from biases.models import get_model_profile
from biases.paths import data_path, output_path
from biases.position_bias import DEFAULT_MAX_MODEL_LEN, VLLMJudge, load_position_pairs
from biases.schemas import (
    BiasCondition,
    BiasType,
    CueCongruency,
    OutputMode,
    PairOrdering,
    VerdictLabel,
)
from biases.social_cue_prompts import (
    build_social_cue_prompt_package,
    format_clean_variant_id,
)


DEFAULT_DATA_PATH = data_path("processed", "mtbench_stratified_198.csv")
DEFAULT_OUTPUT_PATH = output_path("validation", "verdict_extraction.json")


def validate_smoke_results(
    results: Sequence[tuple[VerdictLabel, str, dict[str, float]]],
    *,
    expected_examples: int,
    minimum_parse_rate: float = 0.99,
) -> dict[str, Any]:
    if expected_examples < 1:
        raise ValueError("expected_examples must be at least 1")
    if not 0.0 <= minimum_parse_rate <= 1.0:
        raise ValueError("minimum_parse_rate must be between zero and one")

    valid_probability_rows = 0
    parseable = 0
    issues: list[str] = []
    for index, (verdict, _raw_output, probabilities) in enumerate(results):
        if verdict in {VerdictLabel.A, VerdictLabel.B, VerdictLabel.TIE}:
            parseable += 1
        else:
            issues.append(f"row {index}: invalid verdict {verdict!r}")

        if set(probabilities) != {"A", "B", "tie"}:
            issues.append(f"row {index}: missing A/B/tie probability support")
            continue
        values = list(probabilities.values())
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            issues.append(f"row {index}: probabilities must be finite and nonnegative")
            continue
        if not math.isclose(sum(values), 1.0, rel_tol=1e-7, abs_tol=1e-7):
            issues.append(f"row {index}: probabilities do not sum to one")
            continue
        valid_probability_rows += 1

    parse_rate = parseable / expected_examples
    probability_rate = valid_probability_rows / expected_examples
    passed = (
        len(results) == expected_examples
        and parse_rate >= minimum_parse_rate
        and valid_probability_rows == expected_examples
    )
    if len(results) != expected_examples:
        issues.append(
            f"expected {expected_examples} results but received {len(results)}"
        )
    return {
        "expected_examples": expected_examples,
        "received_examples": len(results),
        "parseable_examples": parseable,
        "parse_rate": parse_rate,
        "valid_probability_examples": valid_probability_rows,
        "valid_probability_rate": probability_rate,
        "minimum_parse_rate": minimum_parse_rate,
        "passed": passed,
        "issues": issues,
    }


def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    profile = get_model_profile(args.model_name)
    pairs = load_position_pairs(args.data_path, limit=args.examples)
    if len(pairs) != args.examples:
        raise ValueError(
            f"Requested {args.examples} smoke examples but loaded {len(pairs)}"
        )
    judge = VLLMJudge(
        model_name=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
    )
    prompts: list[str] = []
    for pair in pairs:
        condition = BiasCondition(
            bias_type=BiasType.CLEAN,
            variant_id=format_clean_variant_id(PairOrdering.AB),
            cue_congruency=CueCongruency.CONTROL,
            ordering=PairOrdering.AB,
        )
        package = build_social_cue_prompt_package(
            example=pair.original,
            condition=condition,
            output_mode=OutputMode.CHOICE_ONLY,
            renderer=judge.render_messages,
        )
        prompts.append(package.prompt_text)

    results = judge.choose_verdict_batch(
        prompts,
        seed=0,
        sampling_temperature=0.0,
    )
    validation = validate_smoke_results(
        results,
        expected_examples=args.examples,
        minimum_parse_rate=args.minimum_parse_rate,
    )
    return {
        "model_registry_name": profile.registry_name,
        "model_name": profile.hf_model_name,
        "data_path": str(args.data_path),
        "resolved_verdict_token_ids": judge.decision_label_token_ids,
        **validation,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate constrained first-verdict-token extraction before a "
            "Silent Bias model enters full runs."
        )
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--examples", type=int, default=20)
    parser.add_argument("--minimum-parse-rate", type=float, default=0.99)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--dtype", default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_validation(args)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
