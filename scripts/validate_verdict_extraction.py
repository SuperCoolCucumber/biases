from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Collection, Mapping, Sequence

from biases.models import get_model_profile
from biases.pairing import file_sha256
from biases.paths import data_path, output_path
from biases.position_bias import (
    CONSTRAINED_LOGPROBS_MODE,
    DEFAULT_MAX_MODEL_LEN,
    JUDGE_OUTPUT_PARSER_VERSION,
    SamplingParams,
    VLLMJudge,
    load_position_pairs,
)
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
DEFAULT_NATIVE_MAX_TOKENS = 16
DEFAULT_MINIMUM_NATIVE_CONTRACT_RATE = 0.99
DEFAULT_NATIVE_SAMPLE_LIMIT = 2


@dataclass(frozen=True, slots=True)
class NativeGeneration:
    text: str
    token_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class NativeVerdict:
    verdict: VerdictLabel | None
    format_category: str


_NATIVE_FORMAT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "double_bracket_label",
        re.compile(r"\[\[\s*(?:A|B|T|TIE)\s*\]\]", re.IGNORECASE),
    ),
    (
        "single_bracket_label",
        re.compile(r"\[\s*(?:A|B|T|TIE)\s*\]", re.IGNORECASE),
    ),
    (
        "explicit_label_phrase",
        re.compile(
            r"(?:verdict|answer|response|choice|label)\s*[:=]\s*.+",
            re.IGNORECASE,
        ),
    ),
    (
        "direct_tie_word",
        re.compile(r"TIE\s*[.!]?", re.IGNORECASE),
    ),
    (
        "direct_label",
        re.compile(r"(?:A|B|T)\s*[.!]?", re.IGNORECASE),
    ),
)


def classify_native_verdict(text: str) -> NativeVerdict:
    """Classify only complete, unambiguous verdicts on the leading line."""

    verdict = VLLMJudge._parse_verdict_text(text)
    if verdict is None:
        return NativeVerdict(verdict=None, format_category="unparseable")
    first_line = next(
        (line.strip() for line in text.splitlines() if line.strip()),
        "",
    )
    for category, pattern in _NATIVE_FORMAT_PATTERNS:
        if pattern.fullmatch(first_line) is not None:
            return NativeVerdict(verdict=verdict, format_category=category)
    raise AssertionError("parsed verdict did not match a supported format category")


def generate_native_verdict_batch(
    judge: VLLMJudge,
    prompts: Sequence[str],
    *,
    seed: int,
    max_tokens: int,
) -> list[NativeGeneration]:
    """Run a greedy generation with no label-token constraint."""

    if max_tokens < 1:
        raise ValueError("native max_tokens must be at least 1")
    if SamplingParams is None:
        raise RuntimeError(
            "Native smoke generation requires vLLM. Install with `uv sync --extra local`."
        )
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        seed=seed,
        stop=list(judge.profile.stop_token_texts) or None,
        skip_special_tokens=True,
    )
    prepared_prompts = [judge._prepare_prompt(prompt) for prompt in prompts]
    outputs = judge.model.generate(
        prepared_prompts,
        sampling_params,
        use_tqdm=False,
    )
    native_results: list[NativeGeneration] = []
    for output in outputs:
        completion = output.outputs[0]
        raw_token_ids = getattr(completion, "token_ids", ()) or ()
        native_results.append(
            NativeGeneration(
                text=str(completion.text),
                token_ids=tuple(int(token_id) for token_id in raw_token_ids),
            )
        )
    return native_results


def _output_fingerprint(
    *,
    row_index: int,
    result: NativeGeneration,
    classification: NativeVerdict,
    first_token_compatible: bool,
) -> dict[str, Any]:
    return {
        "row_index": row_index,
        "format_category": classification.format_category,
        "parsed_verdict": (
            classification.verdict.value
            if classification.verdict is not None
            else None
        ),
        "output_sha256": hashlib.sha256(result.text.encode("utf-8")).hexdigest(),
        "output_character_count": len(result.text),
        "generated_token_count": len(result.token_ids),
        "first_token_id": result.token_ids[0] if result.token_ids else None,
        "first_token_compatible": first_token_compatible,
    }


def validate_native_smoke_results(
    native_results: Sequence[NativeGeneration],
    constrained_results: Sequence[tuple[VerdictLabel, str, dict[str, float]]],
    *,
    expected_examples: int,
    allowed_first_token_ids: Collection[int],
    minimum_contract_rate: float = DEFAULT_MINIMUM_NATIVE_CONTRACT_RATE,
    sample_limit_per_format: int = DEFAULT_NATIVE_SAMPLE_LIMIT,
) -> dict[str, Any]:
    """Validate that natural greedy decoding obeys the first-token contract."""

    if expected_examples < 1:
        raise ValueError("expected_examples must be at least 1")
    if not 0.0 <= minimum_contract_rate <= 1.0:
        raise ValueError("minimum_contract_rate must be between zero and one")
    if sample_limit_per_format < 0:
        raise ValueError("sample_limit_per_format must be nonnegative")
    allowed_ids = {int(token_id) for token_id in allowed_first_token_ids}
    if not allowed_ids:
        raise ValueError("allowed_first_token_ids must not be empty")

    parseable = 0
    first_token_compatible = 0
    verdict_agreement = 0
    contract_examples = 0
    format_counts: Counter[str] = Counter()
    sample_fingerprints: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    output_hashes: list[str] = []

    for index, native_result in enumerate(native_results):
        classification = classify_native_verdict(native_result.text)
        format_counts[classification.format_category] += 1
        is_parseable = classification.verdict is not None
        has_compatible_first_token = bool(
            native_result.token_ids and native_result.token_ids[0] in allowed_ids
        )
        agrees = bool(
            is_parseable
            and index < len(constrained_results)
            and classification.verdict == constrained_results[index][0]
        )
        if is_parseable:
            parseable += 1
        if has_compatible_first_token:
            first_token_compatible += 1
        if agrees:
            verdict_agreement += 1
        if is_parseable and has_compatible_first_token and agrees:
            contract_examples += 1

        fingerprint = _output_fingerprint(
            row_index=index,
            result=native_result,
            classification=classification,
            first_token_compatible=has_compatible_first_token,
        )
        output_hashes.append(fingerprint["output_sha256"])
        samples = sample_fingerprints[classification.format_category]
        if len(samples) < sample_limit_per_format:
            samples.append(fingerprint)

    parse_rate = parseable / expected_examples
    first_token_compatible_rate = first_token_compatible / expected_examples
    verdict_agreement_rate = verdict_agreement / expected_examples
    contract_rate = contract_examples / expected_examples
    passed = (
        len(native_results) == expected_examples
        and len(constrained_results) == expected_examples
        and contract_rate >= minimum_contract_rate
    )
    issues: list[str] = []
    if len(native_results) != expected_examples:
        issues.append(
            f"expected {expected_examples} native results but received "
            f"{len(native_results)}"
        )
    if len(constrained_results) != expected_examples:
        issues.append(
            f"expected {expected_examples} constrained results for comparison "
            f"but received {len(constrained_results)}"
        )
    if contract_rate < minimum_contract_rate:
        issues.append(
            "native first-token contract rate "
            f"{contract_rate:.6f} is below {minimum_contract_rate:.6f}"
        )

    return {
        "expected_examples": expected_examples,
        "received_examples": len(native_results),
        "parseable_examples": parseable,
        "parse_rate": parse_rate,
        "first_token_compatible_examples": first_token_compatible,
        "first_token_compatible_rate": first_token_compatible_rate,
        "verdict_agreement_examples": verdict_agreement,
        "verdict_agreement_rate": verdict_agreement_rate,
        "contract_examples": contract_examples,
        "contract_rate": contract_rate,
        "minimum_contract_rate": minimum_contract_rate,
        "format_counts": dict(sorted(format_counts.items())),
        "native_output_set_hash": hashlib.sha256(
            json.dumps(
                output_hashes,
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "sample_fingerprints_by_format": {
            category: samples
            for category, samples in sorted(sample_fingerprints.items())
        },
        "passed": passed,
        "issues": issues,
    }


def require_native_validation(
    constrained_validation: Mapping[str, Any],
    native_validation: Mapping[str, Any],
) -> dict[str, Any]:
    """Preserve legacy fields while making the native diagnostic mandatory."""

    merged = dict(constrained_validation)
    constrained_passed = bool(constrained_validation.get("passed"))
    native_passed = bool(native_validation.get("passed"))
    merged["constrained_passed"] = constrained_passed
    merged["native_diagnostic"] = dict(native_validation)
    merged["passed"] = constrained_passed and native_passed
    issues = list(constrained_validation.get("issues", ()))
    if not native_passed:
        issues.append("native unconstrained verdict diagnostic failed")
    merged["issues"] = issues
    return merged


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
    map_aligned = 0
    parseable = 0
    issues: list[str] = []
    for index, (verdict, raw_output, probabilities) in enumerate(results):
        parsed_raw_verdict = VLLMJudge._parse_verdict_text(raw_output)
        if parsed_raw_verdict is None:
            issues.append(f"row {index}: raw output is not a supported verdict form")
        elif parsed_raw_verdict != verdict:
            issues.append(
                f"row {index}: raw verdict {parsed_raw_verdict.value!r} "
                f"does not match returned verdict {verdict!r}"
            )
        elif verdict in {VerdictLabel.A, VerdictLabel.B, VerdictLabel.TIE}:
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
        probability_map = {
            "A": VerdictLabel.A,
            "B": VerdictLabel.B,
            "tie": VerdictLabel.TIE,
        }[max(probabilities, key=probabilities.get)]
        if probability_map != verdict:
            issues.append(
                f"row {index}: probability MAP {probability_map.value!r} "
                f"does not match returned verdict {verdict!r}"
            )
        else:
            map_aligned += 1

    parse_rate = parseable / expected_examples
    probability_rate = valid_probability_rows / expected_examples
    map_alignment_rate = map_aligned / expected_examples
    passed = (
        len(results) == expected_examples
        and parse_rate >= minimum_parse_rate
        and valid_probability_rows == expected_examples
        and map_aligned == expected_examples
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
        "map_aligned_examples": map_aligned,
        "map_alignment_rate": map_alignment_rate,
        "minimum_parse_rate": minimum_parse_rate,
        "passed": passed,
        "issues": issues,
    }


def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    profile = get_model_profile(args.model_name)
    max_num_batched_tokens = getattr(args, "max_num_batched_tokens", None)
    max_num_seqs = getattr(args, "max_num_seqs", None)
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
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
    )
    if judge.logprobs_mode != CONSTRAINED_LOGPROBS_MODE:
        raise RuntimeError(
            "Verdict extraction requires constrained processed log probabilities."
        )
    prompts: list[str] = []
    prompt_hashes: list[str] = []
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
        prompt_hashes.append(package.prompt_hash)

    results = judge.choose_verdict_batch(
        prompts,
        seed=0,
        sampling_temperature=0.0,
    )
    constrained_validation = validate_smoke_results(
        results,
        expected_examples=args.examples,
        minimum_parse_rate=args.minimum_parse_rate,
    )
    native_results = generate_native_verdict_batch(
        judge,
        prompts,
        seed=0,
        max_tokens=args.native_max_tokens,
    )
    native_validation = validate_native_smoke_results(
        native_results,
        results,
        expected_examples=args.examples,
        allowed_first_token_ids=judge.decision_allowed_token_ids,
        minimum_contract_rate=args.minimum_native_contract_rate,
    )
    native_validation["generation_config"] = {
        "max_tokens": args.native_max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "stop_token_texts": list(profile.stop_token_texts),
    }
    validation = require_native_validation(
        constrained_validation,
        native_validation,
    )
    return {
        "model_registry_name": profile.registry_name,
        "model_name": profile.hf_model_name,
        "model_revision": profile.revision,
        "data_path": str(args.data_path),
        "input_file_hash": file_sha256(args.data_path),
        "prompt_set_hash": hashlib.sha256(
            json.dumps(
                prompt_hashes,
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "resolved_verdict_token_ids": judge.decision_label_token_ids,
        "judge_output_parser_version": JUDGE_OUTPUT_PARSER_VERSION,
        "logprobs_mode": judge.logprobs_mode,
        "max_num_batched_tokens": max_num_batched_tokens,
        "max_num_seqs": max_num_seqs,
        **validation,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate constrained first-verdict-token extraction and the "
            "matching unconstrained native generation contract before a "
            "Silent Bias model enters full runs."
        )
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--examples", type=int, default=20)
    parser.add_argument("--minimum-parse-rate", type=float, default=0.99)
    parser.add_argument(
        "--minimum-native-contract-rate",
        type=float,
        default=DEFAULT_MINIMUM_NATIVE_CONTRACT_RATE,
    )
    parser.add_argument(
        "--native-max-tokens",
        type=int,
        default=DEFAULT_NATIVE_MAX_TOKENS,
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=None)
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
