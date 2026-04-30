from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from biases.authority_prompts import build_authority_cue, build_authority_prompt_package
from biases.position_bias import (
    DEFAULT_MAX_MODEL_LEN,
    DEFAULT_MODEL_NAME,
    QwenJudge,
    _build_run_record,
    _compute_consistency,
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


def _label_to_str(label: str | VerdictLabel | None) -> str | None:
    if label is None:
        return None
    if isinstance(label, VerdictLabel):
        return label.value
    return str(label)


def _opposite_label(label: str | None) -> str | None:
    if label == "A":
        return "B"
    if label == "B":
        return "A"
    return None


def _mean_or_none(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    return mean(valid) if valid else None


def run_authority_experiment(
    *,
    csv_path: Path,
    output_dir: Path,
    model_name: str = DEFAULT_MODEL_NAME,
    limit: int | None = None,
    consistency_runs: int = 5,
    sampling_temperature: float = 0.7,
    tensor_parallel_size: int = 1,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    gpu_memory_utilization: float = 0.9,
) -> dict[str, Any]:
    if consistency_runs < 1:
        raise ValueError("consistency_runs must be at least 1")

    pairs = load_position_pairs(csv_path=csv_path, limit=limit)
    judge = QwenJudge(
        model_name=model_name,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    spec = ExperimentSpec(
        dataset_name=csv_path.name,
        dataset_split="authority_stratified",
        model_name=model_name,
        backend_name="vllm",
        bias_name=BiasType.AUTHORITY.value,
        output_mode=OutputMode.CHOICE_ONLY,
        uncertainty_methods=["logit", "consistency"],
        consistency_runs=consistency_runs,
        temperature=sampling_temperature,
    )

    raw_rows: list[dict[str, Any]] = []
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
                    bias_type=BiasType.AUTHORITY,
                    variant_id="control",
                    cue_congruency=CueCongruency.CONTROL,
                    metadata={"pair_id": pair.pair_id},
                ),
                "cue_text": None,
                "cue_target": None,
            },
            {
                "name": "authority_congruent",
                "condition": BiasCondition(
                    bias_type=BiasType.AUTHORITY,
                    variant_id="authority_congruent",
                    cue_target=congruent_target,
                    cue_congruency=CueCongruency.CONGRUENT,
                    cue_text=build_authority_cue(congruent_target),
                    metadata={"pair_id": pair.pair_id},
                ),
                "cue_text": build_authority_cue(congruent_target),
                "cue_target": congruent_target,
            },
            {
                "name": "authority_incongruent",
                "condition": BiasCondition(
                    bias_type=BiasType.AUTHORITY,
                    variant_id="authority_incongruent",
                    cue_target=incongruent_target,
                    cue_congruency=CueCongruency.INCONGRUENT,
                    cue_text=build_authority_cue(incongruent_target),
                    metadata={"pair_id": pair.pair_id},
                ),
                "cue_text": build_authority_cue(incongruent_target),
                "cue_target": incongruent_target,
            },
        ]

        condition_records: dict[str, RunRecord] = {}
        for condition_spec in conditions:
            prompt = build_authority_prompt_package(
                example=example,
                cue_text=condition_spec["cue_text"],
                output_mode=OutputMode.CHOICE_ONLY,
            )

            verdict, _, label_probs = judge.choose_verdict(
                prompt_text=prompt.prompt_text,
                seed=0,
                sampling_temperature=0.0,
            )

            consistency_verdicts: list[VerdictLabel] = []
            for run_seed in range(consistency_runs):
                sampled_verdict, _, _ = judge.choose_verdict(
                    prompt_text=prompt.prompt_text,
                    seed=run_seed,
                    sampling_temperature=sampling_temperature,
                )
                consistency_verdicts.append(sampled_verdict)
            consistency = _compute_consistency(consistency_verdicts, anchor=verdict)

            record = _build_run_record(
                example=example,
                condition=condition_spec["condition"],
                spec=spec,
                prompt_text=prompt.prompt_text,
                prompt_hash=prompt.prompt_hash,
                seed=0,
                verdict=verdict,
                label_probs=label_probs,
                consistency=consistency,
            )
            condition_records[condition_spec["name"]] = record
            raw_rows.append(record.model_dump(mode="json"))

        control = condition_records["control"]
        congruent = condition_records["authority_congruent"]
        incongruent = condition_records["authority_incongruent"]

        control_entropy = control.uncertainty.logit.entropy
        congruent_entropy = congruent.uncertainty.logit.entropy
        incongruent_entropy = incongruent.uncertainty.logit.entropy

        pair_summaries.append(
            {
                "pair_id": pair.pair_id,
                "human_winner": human_winner,
                "control_verdict": control.verdict,
                "authority_congruent_verdict": congruent.verdict,
                "authority_incongruent_verdict": incongruent.verdict,
                "control_entropy": control_entropy,
                "authority_congruent_entropy": congruent_entropy,
                "authority_incongruent_entropy": incongruent_entropy,
                "authority_congruent_delta_entropy": (
                    None if control_entropy is None or congruent_entropy is None else congruent_entropy - control_entropy
                ),
                "authority_incongruent_delta_entropy": (
                    None
                    if control_entropy is None or incongruent_entropy is None
                    else incongruent_entropy - control_entropy
                ),
                "control_agreement_rate": (
                    control.uncertainty.consistency.agreement_rate
                    if control.uncertainty.consistency
                    else None
                ),
                "authority_congruent_agreement_rate": (
                    congruent.uncertainty.consistency.agreement_rate
                    if congruent.uncertainty.consistency
                    else None
                ),
                "authority_incongruent_agreement_rate": (
                    incongruent.uncertainty.consistency.agreement_rate
                    if incongruent.uncertainty.consistency
                    else None
                ),
                "authority_congruent_cue_target": congruent_target,
                "authority_incongruent_cue_target": incongruent_target,
                "authority_congruent_cue_follow": congruent.verdict == congruent_target,
                "authority_incongruent_cue_follow": incongruent.verdict == incongruent_target,
                "authority_congruent_shift_from_control": congruent.verdict != control.verdict,
                "authority_incongruent_shift_from_control": incongruent.verdict != control.verdict,
            }
        )

    raw_path = output_dir / "authority_run_records.jsonl"
    pair_path = output_dir / "authority_pair_summary.jsonl"
    write_jsonl(raw_path, raw_rows)
    write_jsonl(pair_path, pair_summaries)

    summary = {
        "model_name": model_name,
        "csv_path": str(csv_path),
        "output_dir": str(output_dir),
        "total_pairs_evaluated": len(pair_summaries),
        "consistency_runs": consistency_runs,
        "sampling_temperature": sampling_temperature,
        "tensor_parallel_size": tensor_parallel_size,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization,
        "mean_control_entropy": _mean_or_none([row["control_entropy"] for row in pair_summaries]),
        "mean_authority_congruent_entropy": _mean_or_none(
            [row["authority_congruent_entropy"] for row in pair_summaries]
        ),
        "mean_authority_incongruent_entropy": _mean_or_none(
            [row["authority_incongruent_entropy"] for row in pair_summaries]
        ),
        "mean_authority_congruent_delta_entropy": _mean_or_none(
            [row["authority_congruent_delta_entropy"] for row in pair_summaries]
        ),
        "mean_authority_incongruent_delta_entropy": _mean_or_none(
            [row["authority_incongruent_delta_entropy"] for row in pair_summaries]
        ),
        "authority_congruent_cue_follow_rate": (
            sum(1 for row in pair_summaries if row["authority_congruent_cue_follow"]) / len(pair_summaries)
            if pair_summaries
            else None
        ),
        "authority_incongruent_cue_follow_rate": (
            sum(1 for row in pair_summaries if row["authority_incongruent_cue_follow"]) / len(pair_summaries)
            if pair_summaries
            else None
        ),
        "authority_congruent_shift_rate": (
            sum(1 for row in pair_summaries if row["authority_congruent_shift_from_control"]) / len(pair_summaries)
            if pair_summaries
            else None
        ),
        "authority_incongruent_shift_rate": (
            sum(1 for row in pair_summaries if row["authority_incongruent_shift_from_control"]) / len(pair_summaries)
            if pair_summaries
            else None
        ),
        "raw_records_path": str(raw_path),
        "pair_summary_path": str(pair_path),
    }

    summary_path = output_dir / "authority_summary.json"
    ensure_parent(summary_path)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)

    return summary
