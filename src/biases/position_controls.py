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
    _label_to_str,
    load_position_pairs,
)
from biases.position_prompts import build_position_prompt_package
from biases.schemas import (
    BiasCondition,
    BiasType,
    Candidate,
    ExperimentSpec,
    JudgeExample,
    OutputMode,
    VerdictLabel,
)
from biases.utils import stable_hash, write_jsonl


DEFAULT_IDENTICAL_CONTROL_OUTPUT_DIR = output_path("position_identical_answer_control")
DEFAULT_LABEL_PRIOR_OUTPUT_DIR = output_path("position_label_prior_control")


def _source_response(pair: Any, source_side: str) -> tuple[str, str | None]:
    source = source_side.lower()
    if source == "human_winner":
        winner = _label_to_str(pair.original.human_winner)
        source = winner.lower() if winner in {"A", "B"} else "a"
    if source == "b":
        candidate = pair.original.candidates["B"]
        return candidate.response, candidate.response_id
    candidate = pair.original.candidates["A"]
    return candidate.response, candidate.response_id


def build_identical_answer_example(pair: Any, *, source_side: str = "human_winner") -> JudgeExample:
    response, response_id = _source_response(pair, source_side)
    response_id = response_id or f"{pair.pair_id}:identical_{source_side.lower()}"
    return JudgeExample(
        example_id=f"{pair.pair_id}:identical:{source_side}",
        question_id=pair.pair_id,
        prompt_messages=pair.original.prompt_messages,
        candidates={
            "A": Candidate(
                label=VerdictLabel.A,
                response=response,
                response_id=response_id,
            ),
            "B": Candidate(
                label=VerdictLabel.B,
                response=response,
                response_id=response_id,
            ),
        },
        human_winner=VerdictLabel.TIE,
        metadata={
            **pair.original.metadata,
            "variant_id": "identical_answer_control",
            "source_side": source_side,
            "response_id_by_label": {
                "A": response_id,
                "B": response_id,
            },
        },
    )


def _record_summary(record: Any) -> dict[str, Any]:
    probs = record.raw_prompt_logprobs or {}
    return {
        "record_id": record.record_id,
        "pair_id": record.metadata.get("pair_id"),
        "routing_split": record.metadata.get("routing_split"),
        "variant_id": record.condition.variant_id,
        "verdict": _label_to_str(record.verdict),
        "label_prob_A": probs.get("A"),
        "label_prob_B": probs.get("B"),
        "label_prob_tie": probs.get("tie"),
        "entropy": record.uncertainty.logit.entropy,
        "msp": record.uncertainty.logit.msp,
        "margin": record.uncertainty.logit.margin,
        "consistency_entropy": (
            record.uncertainty.consistency.vote_entropy
            if record.uncertainty.consistency
            else None
        ),
    }


def run_identical_answer_control(
    *,
    csv_path: Path,
    output_dir: Path = DEFAULT_IDENTICAL_CONTROL_OUTPUT_DIR,
    model_name: str = DEFAULT_MODEL_NAME,
    limit: int | None = 300,
    source_side: str = "human_winner",
    tensor_parallel_size: int = 1,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    gpu_memory_utilization: float = 0.9,
    dtype: str = "auto",
) -> dict[str, Any]:
    pairs = load_position_pairs(csv_path=csv_path, limit=limit)
    judge = QwenJudge(
        model_name=model_name,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=dtype,
    )
    spec = ExperimentSpec(
        dataset_name=str(csv_path),
        dataset_split="control",
        model_name=model_name,
        backend_name="vllm",
        bias_name="position_identical_answer_control",
        output_mode=OutputMode.CHOICE_ONLY,
        uncertainty_methods=["logit"],
        consistency_runs=1,
        temperature=0.0,
    )
    condition = BiasCondition(
        bias_type=BiasType.CONTROL,
        variant_id="identical_answer_control",
        metadata={"source_side": source_side},
    )

    raw_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for pair in pairs:
        example = build_identical_answer_example(pair, source_side=source_side)
        prompt = build_position_prompt_package(example, output_mode=OutputMode.CHOICE_ONLY)
        record = _judge_example_condition(
            judge=judge,
            example=example,
            condition=condition,
            spec=spec,
            choice_prompt=prompt,
            confidence_prompt=None,
            consistency_runs=1,
            sampling_temperature=0.0,
            include_verbalized_confidence=False,
        )
        raw_rows.append(record.model_dump(mode="json"))
        summary_rows.append(_record_summary(record))

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "identical_answer_control_records.jsonl"
    summary_path = output_dir / "identical_answer_control_summary.jsonl"
    run_summary_path = output_dir / "identical_answer_control_summary.json"
    write_jsonl(raw_path, raw_rows)
    write_jsonl(summary_path, summary_rows)

    verdict_counts: dict[str, int] = {}
    for row in summary_rows:
        verdict_counts[str(row["verdict"])] = verdict_counts.get(str(row["verdict"]), 0) + 1
    run_summary = {
        "model_name": model_name,
        "csv_path": str(csv_path),
        "output_dir": str(output_dir),
        "limit": limit,
        "source_side": source_side,
        "n": len(summary_rows),
        "verdict_counts": verdict_counts,
        "p_verdict_A_non_tie": _non_tie_a_rate(summary_rows),
        "raw_records_path": str(raw_path),
        "summary_path": str(summary_path),
    }
    run_summary_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    return run_summary


def _non_tie_a_rate(rows: list[dict[str, Any]]) -> float | None:
    non_tie = [row for row in rows if row["verdict"] in {"A", "B"}]
    if not non_tie:
        return None
    return sum(1 for row in non_tie if row["verdict"] == "A") / len(non_tie)


def build_label_prior_example() -> JudgeExample:
    response = "Placeholder answer."
    response_id = "label_prior:placeholder"
    return JudgeExample(
        example_id="label_prior:placeholder",
        question_id="label_prior",
        prompt_messages=[{"role": "user", "content": "Placeholder comparison."}],
        candidates={
            "A": Candidate(label=VerdictLabel.A, response=response, response_id=response_id),
            "B": Candidate(label=VerdictLabel.B, response=response, response_id=response_id),
        },
        human_winner=VerdictLabel.TIE,
        metadata={
            "pair_id": "label_prior",
            "variant_id": "label_prior_placeholder",
            "response_id_by_label": {
                "A": response_id,
                "B": response_id,
            },
        },
    )


def run_label_prior_control(
    *,
    output_dir: Path = DEFAULT_LABEL_PRIOR_OUTPUT_DIR,
    model_name: str = DEFAULT_MODEL_NAME,
    tensor_parallel_size: int = 1,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    gpu_memory_utilization: float = 0.9,
    dtype: str = "auto",
) -> dict[str, Any]:
    judge = QwenJudge(
        model_name=model_name,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=dtype,
    )
    example = build_label_prior_example()
    prompt = build_position_prompt_package(example, output_mode=OutputMode.CHOICE_ONLY)
    prompt_text = prompt.prompt_text
    verdict, raw_output, probs = judge.choose_verdict(
        prompt_text=prompt_text,
        seed=0,
        sampling_temperature=0.0,
    )
    row = {
        "record_id": stable_hash({"model_name": model_name, "control": "label_prior"}),
        "model_name": model_name,
        "variant_id": "label_prior_placeholder",
        "verdict": _label_to_str(verdict),
        "raw_output": raw_output,
        "label_prob_A": probs.get("A"),
        "label_prob_B": probs.get("B"),
        "label_prob_tie": probs.get("tie"),
        "prompt_hash": prompt.prompt_hash,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "label_prior_summary.json"
    summary_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row
