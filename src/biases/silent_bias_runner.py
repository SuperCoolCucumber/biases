from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from biases.models import get_model_profile
from biases.pairing import file_sha256, make_pair_identity_key, normalize_ordering
from biases.position_bias import (
    CONSTRAINED_LOGPROBS_MODE,
    DEFAULT_MAX_MODEL_LEN,
    JUDGE_OUTPUT_PARSER_VERSION,
    VERBALIZED_OUTPUT_PARSER_VERSION,
    VLLMJudge,
    _build_run_record,
    _compute_consistency,
    _label_to_str,
    _record_to_uncertainty_row,
    load_position_pairs,
    verbalized_parse_status,
)
from biases.schemas import (
    BiasCondition,
    BiasType,
    ExperimentSpec,
    JudgeExample,
    OutputMode,
    PairOrdering,
    PromptPackage,
    RunRecord,
    VerdictLabel,
)
from biases.social_cue_prompts import (
    AUTHORITY_DOSES,
    BANDWAGON_DOSES,
    build_social_cue_prompt_package,
)
from biases.stage_planning import (
    CleanPairSummary,
    PlannedCondition,
    PlanningIssue,
    StageAPairInput,
    clean_summaries_from_rows,
    generate_stage_a_conditions,
    generate_stage_b_conditions,
)
from biases.utils import ensure_parent, stable_hash, write_jsonl


ConsistencySchedule = Literal["all", "extremes"]

UNCERTAINTY_METHODS = (
    "logit",
    "verbalized_confidence",
    "consistency",
    "consistency_entropy",
)


class JudgeBackend(Protocol):
    model_name: str
    logprobs_mode: str
    decision_label_token_ids: Mapping[str, Sequence[int]]

    def render_messages(self, messages: list[dict[str, str]]) -> str: ...

    def choose_verdict_batch(
        self,
        prompt_texts: list[str],
        seed: int,
        sampling_temperature: float,
    ) -> list[tuple[VerdictLabel, str, dict[str, float]]]: ...

    def verbalize_confidence_batch(
        self,
        prompt_texts: list[str],
        seed: int = 0,
        max_tokens: int = 24,
    ) -> list[tuple[VerdictLabel | None, str, float | None]]: ...


@dataclass(frozen=True, slots=True)
class EvaluationItem:
    planned: PlannedCondition
    example: JudgeExample


@dataclass(frozen=True, slots=True)
class RunnerPaths:
    raw_records: Path
    uncertainty_scores: Path
    pair_summary: Path
    summary: Path
    planning_issues: Path


def _stage_paths(output_dir: Path, stage: Literal["stage_a", "stage_b"]) -> RunnerPaths:
    prefix = f"silent_bias_{stage}"
    return RunnerPaths(
        raw_records=output_dir / f"{prefix}_run_records.jsonl",
        uncertainty_scores=output_dir / f"{prefix}_uncertainty_scores.jsonl",
        pair_summary=output_dir / f"{prefix}_pair_summary.jsonl",
        summary=output_dir / f"{prefix}_summary.json",
        planning_issues=output_dir / f"{prefix}_planning_issues.json",
    )


def _read_jsonl(
    path: Path,
    *,
    recover_incomplete_tail: bool = False,
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    file_size = path.stat().st_size
    truncate_offset: int | None = None
    with path.open("rb") as handle:
        line_number = 0
        while True:
            line_offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            line_number += 1
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                is_incomplete_tail = (
                    recover_incomplete_tail
                    and handle.tell() == file_size
                    and not line.endswith(b"\n")
                )
                if is_incomplete_tail:
                    truncate_offset = line_offset
                    break
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise TypeError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    if truncate_offset is not None:
        with path.open("r+b") as handle:
            handle.truncate(truncate_offset)
            handle.flush()
            os.fsync(handle.fileno())
    return rows


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_parent(path)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    write_jsonl(temporary, rows)
    temporary.replace(path)


def _append_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    ensure_parent(path)
    payload = "".join(
        f"{json.dumps(row, ensure_ascii=True)}\n" for row in rows
    ).encode("utf-8")
    with path.open("ab+") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() > 0:
            handle.seek(-1, os.SEEK_END)
            needs_separator = handle.read(1) != b"\n"
            handle.seek(0, os.SEEK_END)
            if needs_separator:
                handle.write(b"\n")
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_write_json(path: Path, payload: Any) -> None:
    ensure_parent(path)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
    temporary.replace(path)


def _condition_key_from_row(row: Mapping[str, Any]) -> tuple[str, str]:
    condition = row.get("condition")
    if not isinstance(condition, Mapping):
        raise ValueError("Run record is missing a condition object")
    return str(row.get("pair_key")), str(condition.get("variant_id"))


def _record_sort_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    pair_key, variant_id = _condition_key_from_row(row)
    return pair_key, variant_id, str(row.get("record_id"))


def _judge_verdict_token_contract(
    judge: JudgeBackend,
) -> tuple[dict[str, list[str]], dict[str, list[int]]]:
    """Return the exact registered surface forms and resolved IDs in stable order."""

    labels = ("A", "B", "tie")
    profile = get_model_profile(judge.model_name)
    raw_texts = profile.verdict_token_texts
    raw_ids = getattr(judge, "decision_label_token_ids", None)
    if set(raw_texts) != set(labels) or not isinstance(raw_ids, Mapping):
        raise ValueError(
            "Silent Bias vLLM judges require a complete A/B/tie verdict-token contract"
        )
    if set(raw_ids) != set(labels):
        raise ValueError(
            "Silent Bias vLLM judges require resolved token IDs for A, B, and tie"
        )

    texts: dict[str, list[str]] = {}
    token_ids: dict[str, list[int]] = {}
    seen_ids: dict[int, str] = {}
    for label in labels:
        label_texts = [str(text) for text in raw_texts[label]]
        label_ids = [int(token_id) for token_id in raw_ids[label]]
        if not label_texts or any(not text.strip() for text in label_texts):
            raise ValueError(f"Verdict label {label!r} has no usable token text")
        if not label_ids or any(token_id < 0 for token_id in label_ids):
            raise ValueError(f"Verdict label {label!r} has no usable token ID")
        if len(label_texts) != len(label_ids):
            raise ValueError(
                f"Verdict label {label!r} must have one resolved token ID "
                "per registered token text"
            )
        if len(set(label_texts)) != len(label_texts):
            raise ValueError(f"Verdict label {label!r} has duplicate token texts")
        if len(set(label_ids)) != len(label_ids):
            raise ValueError(f"Verdict label {label!r} has duplicate token IDs")
        for token_id in label_ids:
            existing = seen_ids.get(token_id)
            if existing is not None and existing != label:
                raise ValueError(
                    f"Verdict token ID {token_id} maps to both {existing!r} and {label!r}"
                )
            seen_ids[token_id] = label
        texts[label] = label_texts
        token_ids[label] = label_ids
    return texts, token_ids


def _validate_resume_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    judge: JudgeBackend,
    items: Sequence[EvaluationItem],
    stage: Literal["stage_a", "stage_b"],
    input_file_hash: str,
    model_name: str,
    model_revision: str | None,
    dataset_name: str,
    dataset_split: str,
    consistency_runs: int,
    consistency_schedule: ConsistencySchedule,
    sampling_temperature: float,
    include_verbalized_confidence: bool,
    max_num_batched_tokens: int | None,
    max_num_seqs: int | None,
) -> None:
    verdict_token_texts, verdict_token_ids = _judge_verdict_token_contract(judge)
    expected = {
        (item.planned.pair_key, item.planned.condition.variant_id): item
        for item in items
    }
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = _condition_key_from_row(row)
        if key in seen:
            raise ValueError(f"resume file contains duplicate condition key {key!r}")
        seen.add(key)
        item = expected.get(key)
        if item is None:
            raise ValueError(
                "resume file contains a condition outside the current run plan: "
                f"{key!r}"
            )

        spec = row.get("spec")
        metadata = row.get("metadata")
        if not isinstance(spec, Mapping) or not isinstance(metadata, Mapping):
            raise ValueError("resume rows must contain spec and metadata objects")
        expected_consistency = consistency_runs_for_condition(
            item.planned.condition,
            consistency_runs=consistency_runs,
            consistency_schedule=consistency_schedule,
        )
        expected_methods = ["logit"]
        if include_verbalized_confidence:
            expected_methods.append("verbalized_confidence")
        if expected_consistency > 0:
            expected_methods.extend(("consistency", "consistency_entropy"))
        expected_spec = ExperimentSpec(
            dataset_name=dataset_name,
            dataset_split=dataset_split,
            model_name=model_name,
            model_revision=model_revision,
            backend_name="vllm",
            bias_name=str(item.planned.condition.bias_type),
            output_mode=OutputMode.CHOICE_ONLY,
            uncertainty_methods=expected_methods,
            consistency_runs=expected_consistency,
            temperature=sampling_temperature,
            consistency_schedule=consistency_schedule,
            logprobs_mode=CONSTRAINED_LOGPROBS_MODE,
            verdict_token_texts=verdict_token_texts,
            verdict_token_ids=verdict_token_ids,
        )
        expected_choice_prompt = _prompt_package(
            judge=judge,
            item=item,
            output_mode=OutputMode.CHOICE_ONLY,
        )
        expected_confidence_prompt = (
            _prompt_package(
                judge=judge,
                item=item,
                output_mode=OutputMode.CHOICE_WITH_CONFIDENCE,
            )
            if include_verbalized_confidence
            else None
        )
        methods = {str(value) for value in spec.get("uncertainty_methods", ())}
        checks = {
            "input_file_hash": row.get("input_file_hash") == input_file_hash,
            "model_name": spec.get("model_name") == model_name,
            "model_revision": spec.get("model_revision") == model_revision,
            "dataset_name": spec.get("dataset_name") == dataset_name,
            "dataset_split": spec.get("dataset_split") == dataset_split,
            "stage": metadata.get("stage") == stage,
            "prompt_hash": row.get("prompt_hash") == expected_choice_prompt.prompt_hash,
            "verbalized_prompt_hash": (
                metadata.get("verbalized_prompt_hash")
                == (
                    expected_confidence_prompt.prompt_hash
                    if expected_confidence_prompt is not None
                    else None
                )
            ),
            "spec_hash": (
                row.get("spec_hash")
                == stable_hash(expected_spec.model_dump(mode="json"))
            ),
            "source_row_index": (
                metadata.get("source_row_index")
                == item.example.metadata.get("source_row_index")
            ),
            "selected_turn": (
                metadata.get("selected_turn")
                == item.example.metadata.get("selected_turn")
            ),
            "conversation_extraction_mode": (
                metadata.get("conversation_extraction_mode")
                == item.example.metadata.get("conversation_extraction_mode")
            ),
            "judge_output_parser_version": (
                metadata.get("judge_output_parser_version")
                == JUDGE_OUTPUT_PARSER_VERSION
            ),
            "verbalized_output_parser_version": (
                metadata.get("verbalized_output_parser_version")
                == VERBALIZED_OUTPUT_PARSER_VERSION
            ),
            "spec_logprobs_mode": (
                spec.get("logprobs_mode") == CONSTRAINED_LOGPROBS_MODE
            ),
            "metadata_logprobs_mode": (
                metadata.get("logprobs_mode") == CONSTRAINED_LOGPROBS_MODE
            ),
            "verdict_token_texts": (
                spec.get("verdict_token_texts") == verdict_token_texts
            ),
            "verdict_token_ids": (
                spec.get("verdict_token_ids") == verdict_token_ids
            ),
            "verbalized_parse_status": (
                metadata.get("verbalized_parse_status")
                == verbalized_parse_status(
                    uncertainty_methods=expected_methods,
                    raw_output=metadata.get("verbalized_raw_output"),
                )
            ),
            "max_num_batched_tokens": (
                "max_num_batched_tokens" in metadata
                and metadata.get("max_num_batched_tokens")
                == max_num_batched_tokens
            ),
            "max_num_seqs": (
                "max_num_seqs" in metadata
                and metadata.get("max_num_seqs") == max_num_seqs
            ),
            "consistency_runs": spec.get("consistency_runs") == expected_consistency,
            "consistency_schedule": (
                spec.get("consistency_schedule") == consistency_schedule
            ),
            "sampling_temperature": spec.get("temperature") == sampling_temperature,
            "verbalized_confidence": (
                ("verbalized_confidence" in methods)
                == include_verbalized_confidence
            ),
        }
        mismatched = [name for name, matches in checks.items() if not matches]
        if mismatched:
            raise ValueError(
                "resume row is incompatible with the current run "
                f"({', '.join(mismatched)}): {key!r}"
            )


def _build_stage_a_inputs_and_examples(
    *,
    csv_path: Path,
    model_name: str,
    input_file_hash: str,
    limit: int | None,
) -> tuple[list[StageAPairInput], dict[str, Any]]:
    pairs = load_position_pairs(csv_path=csv_path, limit=limit)
    pair_inputs: list[StageAPairInput] = []
    pairs_by_identity: dict[str, Any] = {}
    for pair in pairs:
        original = pair.original
        source_row_index = original.metadata["source_row_index"]
        question_id = original.metadata.get("question_cluster_id") or original.question_id
        pair_input = StageAPairInput(
            dataset_name=csv_path.name,
            input_file_hash=input_file_hash,
            source_row_index=source_row_index,
            question_id=question_id,
            model_name=model_name,
            human_winner=original.human_winner or VerdictLabel.TIE,
            turn=original.metadata.get("turn"),
            response_a_id=original.candidates["A"].response_id,
            response_b_id=original.candidates["B"].response_id,
        )
        identity = make_pair_identity_key(
            dataset_name=pair_input.dataset_name,
            input_file_hash=pair_input.input_file_hash,
            source_row_index=pair_input.source_row_index,
            question_id=pair_input.question_id,
            turn=pair_input.turn,
            response_a_id=pair_input.response_a_id,
            response_b_id=pair_input.response_b_id,
        )
        pair_inputs.append(pair_input)
        pairs_by_identity[identity] = pair
    return pair_inputs, pairs_by_identity


def _evaluation_items(
    planned_conditions: Iterable[PlannedCondition],
    pairs_by_identity: Mapping[str, Any],
) -> list[EvaluationItem]:
    items: list[EvaluationItem] = []
    for planned in planned_conditions:
        pair = pairs_by_identity.get(planned.pair_identity_key)
        if pair is None:
            raise KeyError(
                "The planned condition cannot be linked to the current input file: "
                f"{planned.pair_identity_key}"
            )
        ordering = normalize_ordering(planned.condition.ordering or "")
        example = pair.original if ordering == PairOrdering.AB else pair.swapped
        items.append(EvaluationItem(planned=planned, example=example))
    return items


def consistency_runs_for_condition(
    condition: BiasCondition,
    *,
    consistency_runs: int,
    consistency_schedule: ConsistencySchedule,
) -> int:
    if consistency_runs < 0:
        raise ValueError("consistency_runs must be non-negative")
    if consistency_schedule not in {"all", "extremes"}:
        raise ValueError("consistency_schedule must be 'all' or 'extremes'")
    if consistency_schedule == "all" or condition.bias_type == BiasType.CLEAN.value:
        return consistency_runs
    if condition.dose is None:
        return 0
    family = str(condition.bias_type)
    boundary_doses = (
        {min(BANDWAGON_DOSES), max(BANDWAGON_DOSES)}
        if family == BiasType.BANDWAGON.value
        else {min(AUTHORITY_DOSES), max(AUTHORITY_DOSES)}
    )
    return consistency_runs if condition.dose in boundary_doses else 0


def _prompt_package(
    *,
    judge: JudgeBackend,
    item: EvaluationItem,
    output_mode: OutputMode,
) -> PromptPackage:
    return build_social_cue_prompt_package(
        example=item.example,
        condition=item.planned.condition,
        output_mode=output_mode,
        renderer=judge.render_messages,
    )


def _batched(items: Sequence[EvaluationItem], batch_size: int) -> Iterable[list[EvaluationItem]]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])


def _evaluate_batch(
    *,
    judge: JudgeBackend,
    items: list[EvaluationItem],
    csv_path: Path,
    dataset_split: str,
    model_revision: str | None,
    stage: Literal["stage_a", "stage_b"],
    consistency_runs: int,
    consistency_schedule: ConsistencySchedule,
    sampling_temperature: float,
    include_verbalized_confidence: bool,
    max_num_batched_tokens: int | None,
    max_num_seqs: int | None,
) -> list[RunRecord]:
    verdict_token_texts, verdict_token_ids = _judge_verdict_token_contract(judge)
    choice_prompts = [
        _prompt_package(judge=judge, item=item, output_mode=OutputMode.CHOICE_ONLY)
        for item in items
    ]
    choice_results = judge.choose_verdict_batch(
        [prompt.prompt_text for prompt in choice_prompts],
        seed=0,
        sampling_temperature=0.0,
    )
    if len(choice_results) != len(items):
        raise RuntimeError("Judge returned the wrong number of deterministic verdicts")

    confidence_prompts: list[PromptPackage] = []
    confidence_results: list[tuple[VerdictLabel | None, str, float | None]] = []
    if include_verbalized_confidence:
        confidence_prompts = [
            _prompt_package(
                judge=judge,
                item=item,
                output_mode=OutputMode.CHOICE_WITH_CONFIDENCE,
            )
            for item in items
        ]
        confidence_results = judge.verbalize_confidence_batch(
            [prompt.prompt_text for prompt in confidence_prompts],
            seed=0,
        )
        if len(confidence_results) != len(items):
            raise RuntimeError("Judge returned the wrong number of confidence outputs")

    run_counts = [
        consistency_runs_for_condition(
            item.planned.condition,
            consistency_runs=consistency_runs,
            consistency_schedule=consistency_schedule,
        )
        for item in items
    ]
    sampled_verdicts: list[list[VerdictLabel]] = [[] for _ in items]
    for run_seed in range(max(run_counts, default=0)):
        selected_indices = [
            index for index, run_count in enumerate(run_counts) if run_seed < run_count
        ]
        sampled = judge.choose_verdict_batch(
            [choice_prompts[index].prompt_text for index in selected_indices],
            seed=run_seed,
            sampling_temperature=sampling_temperature,
        )
        if len(sampled) != len(selected_indices):
            raise RuntimeError("Judge returned the wrong number of consistency outputs")
        for index, (verdict, _, _) in zip(selected_indices, sampled, strict=True):
            sampled_verdicts[index].append(verdict)

    records: list[RunRecord] = []
    for index, item in enumerate(items):
        verdict, raw_output, label_probs = choice_results[index]
        verbalized_verdict: VerdictLabel | None = None
        verbalized_raw_output: str | None = None
        verbalized_confidence: float | None = None
        confidence_prompt: PromptPackage | None = None
        if include_verbalized_confidence:
            confidence_prompt = confidence_prompts[index]
            (
                verbalized_verdict,
                verbalized_raw_output,
                verbalized_confidence,
            ) = confidence_results[index]

        consistency = (
            _compute_consistency(sampled_verdicts[index], anchor=verdict)
            if sampled_verdicts[index]
            else None
        )
        methods = ["logit"]
        if include_verbalized_confidence:
            methods.append("verbalized_confidence")
        if run_counts[index] > 0:
            methods.extend(("consistency", "consistency_entropy"))
        condition = item.planned.condition
        spec = ExperimentSpec(
            dataset_name=csv_path.name,
            dataset_split=dataset_split,
            model_name=judge.model_name,
            model_revision=model_revision,
            backend_name="vllm",
            bias_name=str(condition.bias_type),
            output_mode=OutputMode.CHOICE_ONLY,
            uncertainty_methods=methods,
            consistency_runs=run_counts[index],
            temperature=sampling_temperature,
            consistency_schedule=consistency_schedule,
            logprobs_mode=CONSTRAINED_LOGPROBS_MODE,
            verdict_token_texts=verdict_token_texts,
            verdict_token_ids=verdict_token_ids,
        )
        spec_hash = stable_hash(spec.model_dump(mode="json"))
        record = _build_run_record(
            example=item.example,
            condition=condition,
            spec=spec,
            prompt_text=choice_prompts[index].prompt_text,
            prompt_hash=choice_prompts[index].prompt_hash,
            seed=0,
            verdict=verdict,
            raw_output=raw_output,
            label_probs=label_probs,
            verbalized_confidence=verbalized_confidence,
            verbalized_verdict=verbalized_verdict,
            verbalized_raw_output=verbalized_raw_output,
            verbalized_prompt_hash=(
                confidence_prompt.prompt_hash if confidence_prompt else None
            ),
            consistency=consistency,
            pair_key=item.planned.pair_key,
            condition_group_id=item.planned.condition_group_id,
            ordering_twin_key=item.planned.ordering_twin_key,
            spec_hash=spec_hash,
            input_file_hash=item.planned.input_file_hash,
        )
        record.metadata.update(condition.metadata)
        record.metadata.update(
            {
                "pair_identity_key": item.planned.pair_identity_key,
                "stage": stage,
                "consistency_runs_actual": run_counts[index],
                "template_variant_id": condition.variant_id,
                "logprobs_mode": CONSTRAINED_LOGPROBS_MODE,
                "max_num_batched_tokens": max_num_batched_tokens,
                "max_num_seqs": max_num_seqs,
            }
        )
        records.append(record)
    return records


def _evaluate_new_items(
    *,
    judge: JudgeBackend,
    items: list[EvaluationItem],
    existing_rows: list[dict[str, Any]],
    csv_path: Path,
    dataset_split: str,
    model_revision: str | None,
    stage: Literal["stage_a", "stage_b"],
    consistency_runs: int,
    consistency_schedule: ConsistencySchedule,
    sampling_temperature: float,
    include_verbalized_confidence: bool,
    max_num_batched_tokens: int | None,
    max_num_seqs: int | None,
    batch_size: int,
    checkpoint_path: Path | None = None,
) -> list[dict[str, Any]]:
    existing_keys = {_condition_key_from_row(row) for row in existing_rows}
    pending = [
        item
        for item in items
        if (item.planned.pair_key, item.planned.condition.variant_id)
        not in existing_keys
    ]
    accumulated_rows = list(existing_rows)
    for batch in _batched(pending, batch_size):
        records = _evaluate_batch(
            judge=judge,
            items=batch,
            csv_path=csv_path,
            dataset_split=dataset_split,
            model_revision=model_revision,
            stage=stage,
            consistency_runs=consistency_runs,
            consistency_schedule=consistency_schedule,
            sampling_temperature=sampling_temperature,
            include_verbalized_confidence=include_verbalized_confidence,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
        )
        batch_rows = [
            record.model_dump(mode="json") for record in records
        ]
        if checkpoint_path is not None:
            _append_jsonl(checkpoint_path, batch_rows)
        accumulated_rows.extend(batch_rows)
    accumulated_rows.sort(key=_record_sort_key)
    return accumulated_rows


def _clean_summary_row(record: RunRecord) -> dict[str, Any]:
    return {
        "record_id": record.record_id,
        "clean_record_id": record.record_id,
        "pair_identity_key": record.metadata.get("pair_identity_key"),
        "pair_key": record.pair_key,
        "condition_group_id": record.condition_group_id,
        "ordering_twin_key": record.ordering_twin_key,
        "ordering": record.condition.ordering,
        "model_name": record.spec.model_name,
        "input_file_hash": record.input_file_hash,
        "spec_hash": record.spec_hash,
        "question_id": record.question_id,
        "example_id": record.example_id,
        "pair_id": record.metadata.get("pair_id"),
        "source_row_index": record.metadata.get("source_row_index"),
        "routing_split": record.metadata.get("routing_split"),
        "turn": record.metadata.get("turn"),
        "selected_turn": record.metadata.get("selected_turn"),
        "conversation_extraction_mode": record.metadata.get(
            "conversation_extraction_mode"
        ),
        "judge_output_parser_version": record.metadata.get(
            "judge_output_parser_version"
        ),
        "verbalized_output_parser_version": record.metadata.get(
            "verbalized_output_parser_version"
        ),
        "logprobs_mode": record.spec.logprobs_mode,
        "verdict_token_texts": record.spec.verdict_token_texts,
        "verdict_token_ids": record.spec.verdict_token_ids,
        "verbalized_parse_status": record.metadata.get(
            "verbalized_parse_status"
        ),
        "max_num_batched_tokens": record.metadata.get("max_num_batched_tokens"),
        "max_num_seqs": record.metadata.get("max_num_seqs"),
        "human_winner": record.metadata.get("human_winner"),
        "clean_verdict": record.verdict,
        "verdict": record.verdict,
        "clean_tie": record.verdict == VerdictLabel.TIE.value,
    }


def _cued_summary_row(record: RunRecord) -> dict[str, Any]:
    clean_verdict = record.metadata.get("clean_verdict")
    human_winner = record.metadata.get("human_winner")
    return {
        "record_id": record.record_id,
        "clean_record_id": record.condition.clean_record_id,
        "pair_identity_key": record.metadata.get("pair_identity_key"),
        "pair_key": record.pair_key,
        "condition_group_id": record.condition_group_id,
        "ordering_twin_key": record.ordering_twin_key,
        "ordering": record.condition.ordering,
        "model_name": record.spec.model_name,
        "input_file_hash": record.input_file_hash,
        "spec_hash": record.spec_hash,
        "question_id": record.question_id,
        "example_id": record.example_id,
        "pair_id": record.metadata.get("pair_id"),
        "source_row_index": record.metadata.get("source_row_index"),
        "routing_split": record.metadata.get("routing_split"),
        "turn": record.metadata.get("turn"),
        "selected_turn": record.metadata.get("selected_turn"),
        "conversation_extraction_mode": record.metadata.get(
            "conversation_extraction_mode"
        ),
        "judge_output_parser_version": record.metadata.get(
            "judge_output_parser_version"
        ),
        "verbalized_output_parser_version": record.metadata.get(
            "verbalized_output_parser_version"
        ),
        "logprobs_mode": record.spec.logprobs_mode,
        "verdict_token_texts": record.spec.verdict_token_texts,
        "verdict_token_ids": record.spec.verdict_token_ids,
        "verbalized_parse_status": record.metadata.get(
            "verbalized_parse_status"
        ),
        "max_num_batched_tokens": record.metadata.get("max_num_batched_tokens"),
        "max_num_seqs": record.metadata.get("max_num_seqs"),
        "human_winner": human_winner,
        "clean_verdict": clean_verdict,
        "verdict": record.verdict,
        "family": record.condition.bias_type,
        "direction": record.condition.cue_congruency,
        "direction_relative_human": record.condition.direction_relative_human,
        "dose": record.condition.dose,
        "cue_target": record.condition.cue_target,
        "clean_tie": record.condition.clean_tie,
        "flip": clean_verdict is not None and record.verdict != clean_verdict,
        "error": human_winner is not None and record.verdict != human_winner,
    }


def _materialize_derived_outputs(
    *,
    raw_rows: list[dict[str, Any]],
    paths: RunnerPaths,
    stage: Literal["stage_a", "stage_b"],
) -> list[RunRecord]:
    ordered_raw_rows = sorted(raw_rows, key=_record_sort_key)
    records = [RunRecord.model_validate(row) for row in ordered_raw_rows]
    uncertainty_rows = [
        {
            **_record_to_uncertainty_row(record),
            "logprobs_mode": record.spec.logprobs_mode,
        }
        for record in records
    ]
    pair_rows = [
        _clean_summary_row(record)
        if stage == "stage_a"
        else _cued_summary_row(record)
        for record in records
    ]
    _atomic_write_jsonl(paths.raw_records, ordered_raw_rows)
    _atomic_write_jsonl(paths.uncertainty_scores, uncertainty_rows)
    _atomic_write_jsonl(paths.pair_summary, pair_rows)
    return records


def _verbalized_parse_status_counts(
    records: Sequence[RunRecord],
) -> dict[str, int]:
    counts = Counter(
        str(record.metadata.get("verbalized_parse_status") or "missing_status")
        for record in records
    )
    return dict(sorted(counts.items()))


def _write_issues(path: Path, issues: Sequence[PlanningIssue]) -> None:
    _atomic_write_json(path, [asdict(issue) for issue in issues])


def _new_vllm_judge(
    *,
    model_name: str,
    tensor_parallel_size: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    dtype: str,
    max_num_batched_tokens: int | None,
    max_num_seqs: int | None,
) -> VLLMJudge:
    return VLLMJudge(
        model_name=model_name,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=dtype,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
    )


def _vllm_scheduler_provenance(
    *,
    max_num_batched_tokens: int | None,
    max_num_seqs: int | None,
) -> dict[str, int | None]:
    return {
        "max_num_batched_tokens": max_num_batched_tokens,
        "max_num_seqs": max_num_seqs,
    }


def _require_processed_logprobs(judge: JudgeBackend) -> None:
    actual = getattr(judge, "logprobs_mode", None)
    if actual != CONSTRAINED_LOGPROBS_MODE:
        raise ValueError(
            "Silent Bias runs require processed constrained log probabilities; "
            f"judge.logprobs_mode is {actual!r}, expected "
            f"{CONSTRAINED_LOGPROBS_MODE!r}"
        )


def _require_stage_a_verdict_token_contract(
    rows: Sequence[Mapping[str, Any]],
    *,
    judge: JudgeBackend,
) -> None:
    expected_texts, expected_ids = _judge_verdict_token_contract(judge)
    if any(
        row.get("verdict_token_texts") != expected_texts
        or row.get("verdict_token_ids") != expected_ids
        for row in rows
    ):
        raise ValueError(
            "Stage A summary uses an incompatible verdict-token contract"
        )


def run_silent_bias_clean(
    *,
    csv_path: Path,
    output_dir: Path,
    model_name: str,
    dataset_split: str = "pilot",
    limit: int | None = None,
    consistency_runs: int = 8,
    sampling_temperature: float = 0.7,
    consistency_schedule: ConsistencySchedule = "all",
    tensor_parallel_size: int = 1,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    gpu_memory_utilization: float = 0.9,
    dtype: str = "auto",
    max_num_batched_tokens: int | None = None,
    max_num_seqs: int | None = None,
    include_verbalized_confidence: bool = True,
    batch_size: int = 64,
    resume: bool = True,
    judge: JudgeBackend | None = None,
) -> dict[str, Any]:
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    if consistency_runs < 0:
        raise ValueError("consistency_runs must be non-negative")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _stage_paths(output_dir, "stage_a")
    if paths.raw_records.exists() and not resume:
        raise FileExistsError(
            f"{paths.raw_records} exists; pass resume=True or use a new output directory"
        )

    profile = get_model_profile(model_name)
    canonical_model_name = profile.hf_model_name
    model_revision = profile.revision
    input_hash = file_sha256(csv_path)
    pair_inputs, pairs_by_identity = _build_stage_a_inputs_and_examples(
        csv_path=csv_path,
        model_name=canonical_model_name,
        input_file_hash=input_hash,
        limit=limit,
    )
    plan = generate_stage_a_conditions(pair_inputs)
    _write_issues(paths.planning_issues, plan.issues)
    if plan.issues:
        raise RuntimeError(
            f"Stage A planning produced {len(plan.issues)} issue(s); see "
            f"{paths.planning_issues}"
        )
    items = _evaluation_items(plan.conditions, pairs_by_identity)
    active_judge = judge or _new_vllm_judge(
        model_name=model_name,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=dtype,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
    )
    if active_judge.model_name != canonical_model_name:
        raise ValueError(
            f"Judge resolved to {active_judge.model_name!r}, expected "
            f"{canonical_model_name!r}"
        )
    _require_processed_logprobs(active_judge)
    verdict_token_texts, verdict_token_ids = _judge_verdict_token_contract(
        active_judge
    )
    existing_rows = (
        _read_jsonl(paths.raw_records, recover_incomplete_tail=True)
        if resume
        else []
    )
    _validate_resume_rows(
        existing_rows,
        judge=active_judge,
        items=items,
        stage="stage_a",
        input_file_hash=input_hash,
        model_name=canonical_model_name,
        model_revision=model_revision,
        dataset_name=csv_path.name,
        dataset_split=dataset_split,
        consistency_runs=consistency_runs,
        consistency_schedule=consistency_schedule,
        sampling_temperature=sampling_temperature,
        include_verbalized_confidence=include_verbalized_confidence,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
    )
    raw_rows = _evaluate_new_items(
        judge=active_judge,
        items=items,
        existing_rows=existing_rows,
        csv_path=csv_path,
        dataset_split=dataset_split,
        model_revision=model_revision,
        stage="stage_a",
        consistency_runs=consistency_runs,
        consistency_schedule=consistency_schedule,
        sampling_temperature=sampling_temperature,
        include_verbalized_confidence=include_verbalized_confidence,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        batch_size=batch_size,
        checkpoint_path=paths.raw_records,
    )
    records = _materialize_derived_outputs(
        raw_rows=raw_rows,
        paths=paths,
        stage="stage_a",
    )
    summary = {
        "stage": "A",
        "model_name": canonical_model_name,
        "model_revision": model_revision,
        "dataset_path": str(csv_path),
        "dataset_split": dataset_split,
        "input_file_hash": input_hash,
        "source_pairs": len(pair_inputs),
        "conditions_planned": len(items),
        "records_written": len(records),
        "consistency_runs": consistency_runs,
        "consistency_schedule": consistency_schedule,
        "sampling_temperature": sampling_temperature,
        "include_verbalized_confidence": include_verbalized_confidence,
        "judge_output_parser_version": JUDGE_OUTPUT_PARSER_VERSION,
        "verbalized_output_parser_version": (
            VERBALIZED_OUTPUT_PARSER_VERSION
        ),
        "logprobs_mode": CONSTRAINED_LOGPROBS_MODE,
        "verdict_token_texts": verdict_token_texts,
        "verdict_token_ids": verdict_token_ids,
        "verbalized_parse_status_counts": _verbalized_parse_status_counts(
            records
        ),
        **_vllm_scheduler_provenance(
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
        ),
        "raw_records_path": str(paths.raw_records),
        "uncertainty_scores_path": str(paths.uncertainty_scores),
        "pair_summary_path": str(paths.pair_summary),
        "planning_issues_path": str(paths.planning_issues),
    }
    _atomic_write_json(paths.summary, summary)
    return summary


def run_silent_bias_cued(
    *,
    csv_path: Path,
    stage_a_summary_path: Path,
    output_dir: Path,
    model_name: str,
    dataset_split: str = "pilot",
    limit: int | None = None,
    consistency_runs: int = 8,
    sampling_temperature: float = 0.7,
    consistency_schedule: ConsistencySchedule = "all",
    tensor_parallel_size: int = 1,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    gpu_memory_utilization: float = 0.9,
    dtype: str = "auto",
    max_num_batched_tokens: int | None = None,
    max_num_seqs: int | None = None,
    include_verbalized_confidence: bool = True,
    batch_size: int = 64,
    resume: bool = True,
    judge: JudgeBackend | None = None,
) -> dict[str, Any]:
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    if not stage_a_summary_path.is_file():
        raise FileNotFoundError(stage_a_summary_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _stage_paths(output_dir, "stage_b")
    if paths.raw_records.exists() and not resume:
        raise FileExistsError(
            f"{paths.raw_records} exists; pass resume=True or use a new output directory"
        )

    profile = get_model_profile(model_name)
    canonical_model_name = profile.hf_model_name
    model_revision = profile.revision
    input_hash = file_sha256(csv_path)
    pair_inputs, pairs_by_identity = _build_stage_a_inputs_and_examples(
        csv_path=csv_path,
        model_name=canonical_model_name,
        input_file_hash=input_hash,
        limit=limit,
    )
    stage_a_plan = generate_stage_a_conditions(pair_inputs)
    if stage_a_plan.issues:
        raise RuntimeError("Current input file cannot produce a valid Stage A linkage plan")
    pair_key_to_identity = {
        planned.pair_key: planned.pair_identity_key
        for planned in stage_a_plan.conditions
    }

    clean_rows = _read_jsonl(stage_a_summary_path)
    model_clean_rows = [
        row for row in clean_rows if row.get("model_name") == canonical_model_name
    ]
    if any(
        row.get("judge_output_parser_version") != JUDGE_OUTPUT_PARSER_VERSION
        for row in model_clean_rows
    ):
        raise ValueError(
            "Stage A summary uses an incompatible judge_output_parser_version"
        )
    if any(
        row.get("verbalized_output_parser_version")
        != VERBALIZED_OUTPUT_PARSER_VERSION
        for row in model_clean_rows
    ):
        raise ValueError(
            "Stage A summary uses an incompatible "
            "verbalized_output_parser_version"
        )
    if any(
        row.get("logprobs_mode") != CONSTRAINED_LOGPROBS_MODE
        for row in model_clean_rows
    ):
        raise ValueError(
            "Stage A summary uses an incompatible logprobs_mode"
        )
    clean_summaries = tuple(
        summary
        for summary in clean_summaries_from_rows(model_clean_rows)
        if summary.model_name == canonical_model_name
    )
    if not clean_summaries:
        raise ValueError(
            f"No clean summaries for model {canonical_model_name!r} were found"
        )
    for summary in clean_summaries:
        expected_identity = pair_key_to_identity.get(summary.pair_key)
        if expected_identity != summary.pair_identity_key:
            raise ValueError(
                "Stage A summary does not match the current dataset/model/input hash"
            )

    plan = generate_stage_b_conditions(clean_summaries)
    _write_issues(paths.planning_issues, plan.issues)
    fatal_issues = [
        issue for issue in plan.issues if issue.code != "clean_and_human_tie"
    ]
    if fatal_issues:
        raise RuntimeError(
            f"Stage B planning produced {len(fatal_issues)} fatal issue(s); see "
            f"{paths.planning_issues}"
        )
    items = _evaluation_items(plan.conditions, pairs_by_identity)
    active_judge = judge or _new_vllm_judge(
        model_name=model_name,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=dtype,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
    )
    if active_judge.model_name != canonical_model_name:
        raise ValueError(
            f"Judge resolved to {active_judge.model_name!r}, expected "
            f"{canonical_model_name!r}"
        )
    _require_processed_logprobs(active_judge)
    verdict_token_texts, verdict_token_ids = _judge_verdict_token_contract(
        active_judge
    )
    _require_stage_a_verdict_token_contract(
        model_clean_rows,
        judge=active_judge,
    )
    existing_rows = (
        _read_jsonl(paths.raw_records, recover_incomplete_tail=True)
        if resume
        else []
    )
    _validate_resume_rows(
        existing_rows,
        judge=active_judge,
        items=items,
        stage="stage_b",
        input_file_hash=input_hash,
        model_name=canonical_model_name,
        model_revision=model_revision,
        dataset_name=csv_path.name,
        dataset_split=dataset_split,
        consistency_runs=consistency_runs,
        consistency_schedule=consistency_schedule,
        sampling_temperature=sampling_temperature,
        include_verbalized_confidence=include_verbalized_confidence,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
    )
    raw_rows = _evaluate_new_items(
        judge=active_judge,
        items=items,
        existing_rows=existing_rows,
        csv_path=csv_path,
        dataset_split=dataset_split,
        model_revision=model_revision,
        stage="stage_b",
        consistency_runs=consistency_runs,
        consistency_schedule=consistency_schedule,
        sampling_temperature=sampling_temperature,
        include_verbalized_confidence=include_verbalized_confidence,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        batch_size=batch_size,
        checkpoint_path=paths.raw_records,
    )
    records = _materialize_derived_outputs(
        raw_rows=raw_rows,
        paths=paths,
        stage="stage_b",
    )
    summary = {
        "stage": "B",
        "model_name": canonical_model_name,
        "model_revision": model_revision,
        "dataset_path": str(csv_path),
        "dataset_split": dataset_split,
        "input_file_hash": input_hash,
        "stage_a_summary_path": str(stage_a_summary_path),
        "source_pairs": len(pair_inputs),
        "conditions_planned": len(items),
        "records_written": len(records),
        "clean_and_human_tie_groups_reported": sum(
            issue.code == "clean_and_human_tie" for issue in plan.issues
        ),
        "consistency_runs": consistency_runs,
        "consistency_schedule": consistency_schedule,
        "sampling_temperature": sampling_temperature,
        "include_verbalized_confidence": include_verbalized_confidence,
        "judge_output_parser_version": JUDGE_OUTPUT_PARSER_VERSION,
        "verbalized_output_parser_version": (
            VERBALIZED_OUTPUT_PARSER_VERSION
        ),
        "logprobs_mode": CONSTRAINED_LOGPROBS_MODE,
        "verdict_token_texts": verdict_token_texts,
        "verdict_token_ids": verdict_token_ids,
        "verbalized_parse_status_counts": _verbalized_parse_status_counts(
            records
        ),
        **_vllm_scheduler_provenance(
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
        ),
        "raw_records_path": str(paths.raw_records),
        "uncertainty_scores_path": str(paths.uncertainty_scores),
        "pair_summary_path": str(paths.pair_summary),
        "planning_issues_path": str(paths.planning_issues),
    }
    _atomic_write_json(paths.summary, summary)
    return summary


__all__ = [
    "ConsistencySchedule",
    "EvaluationItem",
    "JudgeBackend",
    "RunnerPaths",
    "consistency_runs_for_condition",
    "run_silent_bias_clean",
    "run_silent_bias_cued",
]
