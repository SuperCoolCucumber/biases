from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from biases.paths import output_path
from biases.position_bias import (
    DEFAULT_MAX_MODEL_LEN,
    DEFAULT_MODEL_NAME,
    QwenJudge,
    _judge_example_condition,
    load_position_pairs,
)
from biases.position_prompts import build_position_prompt_package
from biases.rewrite_cues import CueVariant, read_cue_variants
from biases.schemas import BiasCondition, BiasType, Candidate, ExperimentSpec, JudgeExample, OutputMode, VerdictLabel
from biases.utils import write_jsonl


def apply_cue_variant(example: JudgeExample, variant: CueVariant) -> JudgeExample:
    target = variant.target_side.upper()
    if target not in {"A", "B"}:
        raise ValueError("variant target_side must be A or B")
    candidates = dict(example.candidates)
    original = candidates[target]
    candidates[target] = Candidate(
        label=original.label,
        response=variant.rewritten_text,
        model_id=original.model_id,
        response_id=original.response_id,
    )
    return JudgeExample(
        example_id=f"{example.question_id}:intrinsic:{variant.variant_id}",
        question_id=example.question_id,
        prompt_messages=example.prompt_messages,
        candidates=candidates,
        human_winner=example.human_winner,
        metadata={
            **example.metadata,
            "locus": "intrinsic",
            "cue_family": variant.cue_family,
            "dose": variant.dose,
            "target_side": target,
            "cue_variant_id": variant.variant_id,
        },
    )


def validate_within_pair_invariance(base: JudgeExample, cued: JudgeExample, target_side: str) -> bool:
    target = target_side.upper()
    other = "B" if target == "A" else "A"
    return (
        base.prompt_messages == cued.prompt_messages
        and base.candidates[other].response == cued.candidates[other].response
        and base.candidates[target].response != cued.candidates[target].response
    )


def run_intrinsic_cue_experiment(
    *,
    csv_path: Path,
    variants_path: Path,
    output_dir: Path = output_path("intrinsic_cue_run"),
    model_name: str = DEFAULT_MODEL_NAME,
    tensor_parallel_size: int = 1,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    gpu_memory_utilization: float = 0.9,
    dtype: str = "auto",
    consistency_runs: int = 5,
    sampling_temperature: float = 0.7,
    limit: int | None = None,
) -> dict[str, Any]:
    pairs = {pair.pair_id: pair for pair in load_position_pairs(csv_path=csv_path, limit=limit)}
    variants = read_cue_variants(variants_path)
    judge = QwenJudge(
        model_name=model_name,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=dtype,
    )
    spec = ExperimentSpec(
        dataset_name=str(csv_path),
        dataset_split="intrinsic",
        model_name=model_name,
        backend_name="vllm",
        bias_name="intrinsic_cue",
        output_mode=OutputMode.CHOICE_ONLY,
        uncertainty_methods=["logit", "consistency"],
        consistency_runs=consistency_runs,
        temperature=sampling_temperature,
    )

    raw_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for variant in variants:
        pair = pairs.get(variant.pair_id)
        if pair is None:
            continue
        base = pair.original
        cued = apply_cue_variant(base, variant)
        invariant = validate_within_pair_invariance(base, cued, variant.target_side)
        condition = BiasCondition(
            bias_type=BiasType.CONTROL,
            variant_id=f"intrinsic_{variant.cue_family}_{variant.dose}",
            metadata={
                "locus": "intrinsic",
                "cue_family": variant.cue_family,
                "dose": variant.dose,
                "target_side": variant.target_side,
                "within_pair_invariant": invariant,
            },
        )
        prompt = build_position_prompt_package(cued, output_mode=OutputMode.CHOICE_ONLY)
        record = _judge_example_condition(
            judge=judge,
            example=cued,
            condition=condition,
            spec=spec,
            choice_prompt=prompt,
            confidence_prompt=None,
            consistency_runs=consistency_runs,
            sampling_temperature=sampling_temperature,
            include_verbalized_confidence=False,
        )
        raw_rows.append(record.model_dump(mode="json"))
        summary_rows.append(
            {
                "record_id": record.record_id,
                "pair_id": variant.pair_id,
                "variant_id": variant.variant_id,
                "locus": "intrinsic",
                "cue_family": variant.cue_family,
                "dose": variant.dose,
                "target_side": variant.target_side,
                "verdict": record.verdict,
                "entropy": record.uncertainty.logit.entropy,
                "msp": record.uncertainty.logit.msp,
                "margin": record.uncertainty.logit.margin,
                "consistency_entropy": (
                    record.uncertainty.consistency.vote_entropy
                    if record.uncertainty.consistency
                    else None
                ),
                "content_preserved": variant.content_preserved,
                "length_ratio": variant.length_ratio,
                "within_pair_invariant": invariant,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "intrinsic_cue_run_records.jsonl"
    summary_path = output_dir / "intrinsic_cue_summary.jsonl"
    write_jsonl(raw_path, raw_rows)
    write_jsonl(summary_path, summary_rows)
    summary = {
        "model_name": model_name,
        "csv_path": str(csv_path),
        "variants_path": str(variants_path),
        "output_dir": str(output_dir),
        "n_records": len(summary_rows),
        "raw_records_path": str(raw_path),
        "summary_path": str(summary_path),
    }
    (output_dir / "intrinsic_cue_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
