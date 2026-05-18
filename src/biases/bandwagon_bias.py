from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from biases.bandwagon_prompts import build_bandwagon_cue, build_bandwagon_prompt_package
from biases.position_bias import (
    DEFAULT_MAX_MODEL_LEN,
    DEFAULT_MODEL_NAME,
    UNCERTAINTY_METHODS,
    QwenJudge,
    _judge_example_condition,
    _label_to_str,
    _record_to_uncertainty_row,
    load_position_pairs,
)
from biases.schemas import (
    BiasCondition,
    BiasType,
    CueCongruency,
    ExperimentSpec,
    OutputMode,
    RunRecord,
    VerdictLabel,
)
from biases.utils import ensure_parent, write_jsonl


def _opposite_label(label: str | None) -> str | None:
    if label == "A":
        return "B"
    if label == "B":
        return "A"
    return None


def _mean_or_none(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    return mean(valid) if valid else None


def run_bandwagon_experiment(
    *,
    csv_path: Path,
    output_dir: Path,
    model_name: str = DEFAULT_MODEL_NAME,
    dataset_split: str = "full",
    limit: int | None = None,
    consistency_runs: int = 5,
    sampling_temperature: float = 0.7,
    tensor_parallel_size: int = 1,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    gpu_memory_utilization: float = 0.9,
    dtype: str = "auto",
    include_verbalized_confidence: bool = True,
) -> dict[str, Any]:
    if consistency_runs < 1:
        raise ValueError("consistency_runs must be at least 1")

    pairs = load_position_pairs(csv_path=csv_path, limit=limit)
    judge = QwenJudge(
        model_name=model_name,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=dtype,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    spec = ExperimentSpec(
        dataset_name=csv_path.name,
        dataset_split=dataset_split,
        model_name=model_name,
        backend_name="vllm",
        bias_name=BiasType.BANDWAGON.value,
        output_mode=OutputMode.CHOICE_ONLY,
        uncertainty_methods=UNCERTAINTY_METHODS
        if include_verbalized_confidence
        else [method for method in UNCERTAINTY_METHODS if method != "verbalized_confidence"],
        consistency_runs=consistency_runs,
        temperature=sampling_temperature,
    )

    raw_rows: list[dict[str, Any]] = []
    uncertainty_rows: list[dict[str, Any]] = []
    pair_summaries: list[dict[str, Any]] = []

    for pair in pairs:
        example = pair.original
        human_winner = _label_to_str(example.human_winner)
        if human_winner not in {"A", "B"}:
            continue

        congruent_target = human_winner
        incongruent_target = _opposite_label(human_winner)

        conditions = [
            {
                "name": "control",
                "condition": BiasCondition(
                    bias_type=BiasType.BANDWAGON,
                    variant_id="control",
                    cue_congruency=CueCongruency.CONTROL,
                    metadata={"pair_id": pair.pair_id},
                ),
                "cue_text": None,
                "cue_target": None,
            },
            {
                "name": "bandwagon_congruent",
                "condition": BiasCondition(
                    bias_type=BiasType.BANDWAGON,
                    variant_id="bandwagon_congruent",
                    cue_target=congruent_target,
                    cue_congruency=CueCongruency.CONGRUENT,
                    cue_text=build_bandwagon_cue(congruent_target),
                    metadata={"pair_id": pair.pair_id},
                ),
                "cue_text": build_bandwagon_cue(congruent_target),
                "cue_target": congruent_target,
            },
            {
                "name": "bandwagon_incongruent",
                "condition": BiasCondition(
                    bias_type=BiasType.BANDWAGON,
                    variant_id="bandwagon_incongruent",
                    cue_target=incongruent_target,
                    cue_congruency=CueCongruency.INCONGRUENT,
                    cue_text=build_bandwagon_cue(incongruent_target),
                    metadata={"pair_id": pair.pair_id},
                ),
                "cue_text": build_bandwagon_cue(incongruent_target),
                "cue_target": incongruent_target,
            },
        ]

        condition_records: dict[str, RunRecord] = {}
        for condition_spec in conditions:
            prompt = build_bandwagon_prompt_package(
                example=example,
                cue_text=condition_spec["cue_text"],
                output_mode=OutputMode.CHOICE_ONLY,
            )
            confidence_prompt = (
                build_bandwagon_prompt_package(
                    example=example,
                    cue_text=condition_spec["cue_text"],
                    output_mode=OutputMode.CHOICE_WITH_CONFIDENCE,
                )
                if include_verbalized_confidence
                else None
            )

            record = _judge_example_condition(
                judge=judge,
                example=example,
                condition=condition_spec["condition"],
                spec=spec,
                choice_prompt=prompt,
                confidence_prompt=confidence_prompt,
                consistency_runs=consistency_runs,
                sampling_temperature=sampling_temperature,
                include_verbalized_confidence=include_verbalized_confidence,
            )
            condition_records[condition_spec["name"]] = record
            raw_rows.append(record.model_dump(mode="json"))
            uncertainty_rows.append(_record_to_uncertainty_row(record))

        control = condition_records["control"]
        congruent = condition_records["bandwagon_congruent"]
        incongruent = condition_records["bandwagon_incongruent"]

        control_entropy = control.uncertainty.logit.entropy
        congruent_entropy = congruent.uncertainty.logit.entropy
        incongruent_entropy = incongruent.uncertainty.logit.entropy

        pair_summaries.append(
            {
                "pair_id": pair.pair_id,
                "routing_split": example.metadata.get("routing_split"),
                "human_winner": human_winner,
                "control_verdict": control.verdict,
                "bandwagon_congruent_verdict": congruent.verdict,
                "bandwagon_incongruent_verdict": incongruent.verdict,
                "control_entropy": control_entropy,
                "bandwagon_congruent_entropy": congruent_entropy,
                "bandwagon_incongruent_entropy": incongruent_entropy,
                "bandwagon_congruent_delta_entropy": (
                    None if control_entropy is None or congruent_entropy is None else congruent_entropy - control_entropy
                ),
                "bandwagon_incongruent_delta_entropy": (
                    None
                    if control_entropy is None or incongruent_entropy is None
                    else incongruent_entropy - control_entropy
                ),
                "control_agreement_rate": (
                    control.uncertainty.consistency.agreement_rate
                    if control.uncertainty.consistency
                    else None
                ),
                "control_consistency_entropy": (
                    control.uncertainty.consistency.vote_entropy
                    if control.uncertainty.consistency
                    else None
                ),
                "control_verbalized_confidence": control.uncertainty.verbalized.confidence,
                "control_verbalized_uncertainty": control.uncertainty.verbalized.uncertainty,
                "bandwagon_congruent_agreement_rate": (
                    congruent.uncertainty.consistency.agreement_rate
                    if congruent.uncertainty.consistency
                    else None
                ),
                "bandwagon_congruent_consistency_entropy": (
                    congruent.uncertainty.consistency.vote_entropy
                    if congruent.uncertainty.consistency
                    else None
                ),
                "bandwagon_congruent_verbalized_confidence": (
                    congruent.uncertainty.verbalized.confidence
                ),
                "bandwagon_congruent_verbalized_uncertainty": (
                    congruent.uncertainty.verbalized.uncertainty
                ),
                "bandwagon_incongruent_agreement_rate": (
                    incongruent.uncertainty.consistency.agreement_rate
                    if incongruent.uncertainty.consistency
                    else None
                ),
                "bandwagon_incongruent_consistency_entropy": (
                    incongruent.uncertainty.consistency.vote_entropy
                    if incongruent.uncertainty.consistency
                    else None
                ),
                "bandwagon_incongruent_verbalized_confidence": (
                    incongruent.uncertainty.verbalized.confidence
                ),
                "bandwagon_incongruent_verbalized_uncertainty": (
                    incongruent.uncertainty.verbalized.uncertainty
                ),
                "bandwagon_congruent_cue_target": congruent_target,
                "bandwagon_incongruent_cue_target": incongruent_target,
                "bandwagon_congruent_cue_follow": congruent.verdict == congruent_target,
                "bandwagon_incongruent_cue_follow": incongruent.verdict == incongruent_target,
                "bandwagon_congruent_shift_from_control": congruent.verdict != control.verdict,
                "bandwagon_incongruent_shift_from_control": incongruent.verdict != control.verdict,
            }
        )

    raw_path = output_dir / "bandwagon_run_records.jsonl"
    pair_path = output_dir / "bandwagon_pair_summary.jsonl"
    uncertainty_path = output_dir / "bandwagon_uncertainty_scores.jsonl"
    write_jsonl(raw_path, raw_rows)
    write_jsonl(pair_path, pair_summaries)
    write_jsonl(uncertainty_path, uncertainty_rows)

    summary = {
        "model_name": model_name,
        "csv_path": str(csv_path),
        "output_dir": str(output_dir),
        "dataset_split": dataset_split,
        "total_pairs_evaluated": len(pair_summaries),
        "consistency_runs": consistency_runs,
        "sampling_temperature": sampling_temperature,
        "include_verbalized_confidence": include_verbalized_confidence,
        "tensor_parallel_size": tensor_parallel_size,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization,
        "dtype": dtype,
        "mean_control_entropy": _mean_or_none([row["control_entropy"] for row in pair_summaries]),
        "mean_bandwagon_congruent_entropy": _mean_or_none(
            [row["bandwagon_congruent_entropy"] for row in pair_summaries]
        ),
        "mean_bandwagon_incongruent_entropy": _mean_or_none(
            [row["bandwagon_incongruent_entropy"] for row in pair_summaries]
        ),
        "mean_bandwagon_congruent_delta_entropy": _mean_or_none(
            [row["bandwagon_congruent_delta_entropy"] for row in pair_summaries]
        ),
        "mean_bandwagon_incongruent_delta_entropy": _mean_or_none(
            [row["bandwagon_incongruent_delta_entropy"] for row in pair_summaries]
        ),
        "bandwagon_congruent_cue_follow_rate": (
            sum(1 for row in pair_summaries if row["bandwagon_congruent_cue_follow"]) / len(pair_summaries)
            if pair_summaries
            else None
        ),
        "bandwagon_incongruent_cue_follow_rate": (
            sum(1 for row in pair_summaries if row["bandwagon_incongruent_cue_follow"]) / len(pair_summaries)
            if pair_summaries
            else None
        ),
        "bandwagon_congruent_shift_rate": (
            sum(1 for row in pair_summaries if row["bandwagon_congruent_shift_from_control"])
            / len(pair_summaries)
            if pair_summaries
            else None
        ),
        "bandwagon_incongruent_shift_rate": (
            sum(1 for row in pair_summaries if row["bandwagon_incongruent_shift_from_control"])
            / len(pair_summaries)
            if pair_summaries
            else None
        ),
        "raw_records_path": str(raw_path),
        "pair_summary_path": str(pair_path),
        "uncertainty_scores_path": str(uncertainty_path),
    }

    summary_path = output_dir / "bandwagon_summary.json"
    ensure_parent(summary_path)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)

    return summary
