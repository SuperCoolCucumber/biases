"""Replay and score frozen Silent Bias prompts with inference-time UE methods.

The estimators in this module follow the definitions used by LM-Polygraph at
commit ``98dd675cc43e0f5da654c29940872ea913aea2bf``.  The implementation is kept
local and deliberately small so that a frozen campaign can be replayed without
installing LM-Polygraph or changing the original judge outputs.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from biases.models import ModelProfile, get_model_profile
from biases.pairing import file_sha256
from biases.schemas import OutputMode
from biases.silent_bias_runner import (
    EvaluationItem,
    _build_stage_a_inputs_and_examples,
    _evaluation_items,
)
from biases.social_cue_prompts import (
    AUTHORITY_DOSES,
    BANDWAGON_DOSES,
    build_social_cue_messages,
    build_social_cue_prompt_package,
    render_source_messages,
)
from biases.stage_planning import (
    PlannedCondition,
    clean_summaries_from_rows,
    generate_stage_a_conditions,
    generate_stage_b_conditions,
)


LM_POLYGRAPH_COMMIT = "98dd675cc43e0f5da654c29940872ea913aea2bf"
LM_POLYGRAPH_SOURCE_HASHES: Mapping[str, str] = {
    "p_true.py": "3147241fa4a5138ed6632f32210c6c97c789e28ad0d64a0391745c8e1400d79a",
    "prompt.py": "178c6eb201852879bd9a65bda7bf0f1be9759d5838f6ecdad5264edc95692a77",
    "token_entropy.py": "224b36270aa037e1064cf613f62040aab4eef9145efbb233b71158884422870f",
    "entropy.py": "97b6074bd17263e1eb9f755c1fb071dc6351f173d055ec08d1f1c9287ed1cbac",
    "self_certainty.py": "5f3f629afca269df7fadd801a3c4c0012ee56e28cdfde411a0815a56ba380f95",
}

P_TRUE_TEMPLATE = (
    "Question: {question}\n"
    " Possible answer:{answer}\n"
    " Is the possible answer:\n"
    " (A) True\n"
    " (B) False\n"
    " The possible answer is:"
)

EXPECTED_STAGE_A_COUNT = 6_674
EXPECTED_FULL_STAGE_B_COUNT = 106_784
EXPECTED_PRIMARY_STAGE_B_COUNT = 6_696
FROZEN_MAX_MODEL_LEN = 4_096
SOURCE_PROBABILITY_TOLERANCE = 0.02
CROSS_BACKEND_REPLAY_ROLE = "diagnostic_only"
CROSS_BACKEND_REPLAY_REASON = (
    "frozen vLLM source backend (source attention kernel and effective arithmetic "
    "were not preserved) versus Transformers UE recomputation; the exact HF "
    "dtype, attention implementation, batch size, and runtime fingerprint are "
    "recorded in the collector specification"
)
SCORE_FILE_NAME = "lm_polygraph_inference_scores.jsonl"
SELECTION_FILE_NAME = "lm_polygraph_inference_selection.json"
COMPLETE_FILE_NAME = "lm_polygraph_inference_complete.json"
PREFLIGHT_COMPLETE_FILE_NAME = "lm_polygraph_inference_preflight_complete.json"


@dataclass(frozen=True, slots=True)
class FullVocabularyMetrics:
    """Full-vocabulary scores at one autoregressive decision position."""

    mean_token_entropy: float
    mean_token_entropy_confidence: float
    self_certainty: float
    self_certainty_confidence: float


@dataclass(frozen=True, slots=True)
class PTrueMetrics:
    p_true_log_probability: float
    p_true_probability: float
    p_true_uncertainty: float


@dataclass(frozen=True, slots=True)
class RestrictedLabelMetrics:
    probabilities: dict[str, float]
    msp: float


@dataclass(frozen=True, slots=True)
class ReplayItem:
    source_stage: Literal["stage_a", "stage_b"]
    source_row: Mapping[str, Any]
    original_prompt: str
    p_true_prompt: str
    verdict_token_text: str

    @property
    def record_id(self) -> str:
        return str(self.source_row["record_id"])


@dataclass(frozen=True, slots=True)
class ReplaySelection:
    model_registry_name: str
    model_name: str
    model_revision: str | None
    input_file_hash: str
    items: tuple[ReplayItem, ...]
    stage_a_count: int
    full_stage_b_count: int
    primary_stage_b_count: int
    stage_a_pair_summary_count: int


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(payload)


def token_ids_sha256(token_ids: Sequence[int]) -> str:
    payload = json.dumps(
        [int(token_id) for token_id in token_ids],
        separators=(",", ":"),
    ).encode("ascii")
    return sha256_bytes(payload)


def p_true_meta_prompt(question: str, answer: str) -> str:
    """Return the exact pinned LM-Polygraph P(True) meta-prompt."""

    normalized_answer = normalize_verdict_token(answer)
    return P_TRUE_TEMPLATE.format(question=question, answer=normalized_answer)


def normalize_verdict_token(verdict: Any) -> str:
    normalized = str(verdict).strip()
    mapping = {
        "a": "A",
        "b": "B",
        "t": "T",
        "tie": "T",
        "equal": "T",
    }
    try:
        return mapping[normalized.casefold()]
    except KeyError as exc:
        raise ValueError(f"Unsupported stored verdict {verdict!r}") from exc


def log_softmax_1d(logits: Sequence[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("logits must be a one-dimensional vector with at least two entries")
    if not np.all(np.isfinite(values)):
        raise ValueError("logits must be finite")
    maximum = float(np.max(values))
    log_normalizer = maximum + math.log(float(np.exp(values - maximum).sum()))
    return values - log_normalizer


def full_vocabulary_metrics(
    logits: Sequence[float] | np.ndarray,
) -> FullVocabularyMetrics:
    """Calculate single-position entropy and pinned SelfCertainty.

    LM-Polygraph's SelfCertainty estimator is the negative divergence
    ``-KL(U || p) = mean(log p) + log(V)``.  Larger values (closer to zero)
    therefore mean *more* uncertainty.  ``self_certainty_confidence`` stores
    the confidence orientation ``KL(U || p)``.
    """

    log_probabilities = log_softmax_1d(logits)
    probabilities = np.exp(log_probabilities)
    entropy = float(-np.sum(probabilities * log_probabilities))
    vocabulary_size = int(log_probabilities.size)
    self_certainty = float(np.mean(log_probabilities) + math.log(vocabulary_size))
    return FullVocabularyMetrics(
        mean_token_entropy=entropy,
        mean_token_entropy_confidence=-entropy,
        self_certainty=self_certainty,
        self_certainty_confidence=-self_certainty,
    )


def p_true_metrics(
    logits: Sequence[float] | np.ndarray,
    *,
    true_token_id: int,
) -> PTrueMetrics:
    log_probabilities = log_softmax_1d(logits)
    if true_token_id < 0 or true_token_id >= log_probabilities.size:
        raise IndexError("true_token_id is outside the model vocabulary")
    log_probability = float(log_probabilities[true_token_id])
    probability = float(math.exp(log_probability))
    return PTrueMetrics(
        p_true_log_probability=log_probability,
        p_true_probability=probability,
        p_true_uncertainty=-log_probability,
    )


def restricted_label_metrics(
    logits: Sequence[float] | np.ndarray,
    *,
    label_token_ids: Mapping[str, int],
) -> RestrictedLabelMetrics:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("logits must be one-dimensional")
    if set(label_token_ids) != {"A", "B", "tie"}:
        raise ValueError("label_token_ids must contain exactly A, B, and tie")
    ids = [int(label_token_ids[label]) for label in ("A", "B", "tie")]
    if len(set(ids)) != 3 or any(token_id < 0 or token_id >= values.size for token_id in ids):
        raise ValueError("A, B, and tie must resolve to distinct in-vocabulary token IDs")
    restricted_log_probabilities = log_softmax_1d(values[ids])
    probabilities = {
        label: float(math.exp(log_probability))
        for label, log_probability in zip(
            ("A", "B", "tie"), restricted_log_probabilities, strict=True
        )
    }
    return RestrictedLabelMetrics(
        probabilities=probabilities,
        msp=max(probabilities.values()),
    )


def restricted_pairwise_logit_gaps(
    probabilities: Mapping[str, float],
) -> dict[str, float | None]:
    """Recover the three A/B/tie logit gaps from a restricted softmax.

    Pairwise logit gaps are invariant to the restricted softmax normalizer and
    expose backend drift that can be hidden by saturated probabilities.  A gap
    is ``None`` when either serialized probability is exactly zero, because the
    finite pre-softmax gap is no longer recoverable from that rounded simplex.
    """

    if set(probabilities) != {"A", "B", "tie"}:
        raise ValueError("restricted probabilities must contain exact A/B/tie support")
    values: dict[str, float | None] = {}
    for label in ("A", "B", "tie"):
        value = float(probabilities[label])
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("restricted probabilities must be finite and non-negative")
        values[label] = math.log(value) if value > 0.0 else None

    def gap(left: str, right: str) -> float | None:
        left_value = values[left]
        right_value = values[right]
        if left_value is None or right_value is None:
            return None
        return left_value - right_value

    return {
        "A_minus_B": gap("A", "B"),
        "A_minus_tie": gap("A", "tie"),
        "B_minus_tie": gap("B", "tie"),
    }


def is_primary_stage_b_row(row: Mapping[str, Any]) -> bool:
    """Select the frozen highest-dose, incongruent test stratum, including ties."""

    condition = row.get("condition")
    metadata = row.get("metadata")
    if not isinstance(condition, Mapping) or not isinstance(metadata, Mapping):
        return False
    family = str(condition.get("bias_type", "")).strip().lower()
    direction = str(condition.get("cue_congruency", "")).strip().lower()
    try:
        dose = int(condition.get("dose"))
    except (TypeError, ValueError):
        return False
    expected_dose = {
        "authority": max(AUTHORITY_DOSES),
        "bandwagon": max(BANDWAGON_DOSES),
    }.get(family)
    return (
        str(metadata.get("routing_split", "")).strip().lower() == "test"
        and direction == "incongruent"
        and expected_dose is not None
        and dose == expected_dose
    )


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise TypeError(f"Expected an object at {path}:{line_number}")
            yield row


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def _condition_key(row: Mapping[str, Any]) -> tuple[str, str]:
    condition = row.get("condition")
    if not isinstance(condition, Mapping):
        raise ValueError("Source row is missing condition metadata")
    return str(row.get("pair_key")), str(condition.get("variant_id"))


def _planned_key(planned: PlannedCondition) -> tuple[str, str]:
    return planned.pair_key, str(planned.condition.variant_id)


def _unique_map(
    values: Iterable[Any],
    *,
    key: Any,
    description: str,
) -> dict[tuple[str, str], Any]:
    result: dict[tuple[str, str], Any] = {}
    for value in values:
        item_key = key(value)
        if item_key in result:
            raise ValueError(f"Duplicate {description} key {item_key!r}")
        result[item_key] = value
    return result


def _source_msp(row: Mapping[str, Any]) -> float:
    uncertainty = row.get("uncertainty")
    if isinstance(uncertainty, Mapping):
        logit = uncertainty.get("logit")
        if isinstance(logit, Mapping) and logit.get("msp") is not None:
            return float(logit["msp"])
    raw = row.get("raw_prompt_logprobs")
    if isinstance(raw, Mapping) and raw:
        return max(float(value) for value in raw.values())
    raise ValueError(f"Record {row.get('record_id')!r} has no raw MSP")


def _validate_source_row(
    row: Mapping[str, Any],
    *,
    profile: ModelProfile,
    input_file_hash: str,
) -> None:
    spec = row.get("spec")
    if not isinstance(spec, Mapping):
        raise ValueError(f"Record {row.get('record_id')!r} has no experiment spec")
    checks = {
        "model_name": spec.get("model_name") == profile.hf_model_name,
        "model_revision": spec.get("model_revision") == profile.revision,
        "input_file_hash": row.get("input_file_hash") == input_file_hash,
        "prompt_hash": bool(str(row.get("prompt_hash", "")).strip()),
        "record_id": bool(str(row.get("record_id", "")).strip()),
    }
    mismatches = [name for name, matches in checks.items() if not matches]
    if mismatches:
        raise ValueError(
            f"Record {row.get('record_id')!r} failed frozen-source checks: "
            + ", ".join(mismatches)
        )
    _source_msp(row)


def _render_replay_item(
    *,
    source_stage: Literal["stage_a", "stage_b"],
    row: Mapping[str, Any],
    item: EvaluationItem,
    profile: ModelProfile,
    tokenizer: Any,
) -> ReplayItem:
    messages = build_social_cue_messages(
        example=item.example,
        condition=item.planned.condition,
        output_mode=OutputMode.CHOICE_ONLY,
    )
    package = build_social_cue_prompt_package(
        example=item.example,
        condition=item.planned.condition,
        output_mode=OutputMode.CHOICE_ONLY,
        renderer=lambda candidate_messages: profile.render_prompt(
            tokenizer, candidate_messages
        ),
    )
    stored_hash = str(row.get("prompt_hash"))
    if package.prompt_hash != stored_hash:
        raise ValueError(
            f"Prompt hash mismatch for record {row.get('record_id')!r}: "
            f"replayed {package.prompt_hash}, stored {stored_hash}"
        )
    verdict_token = normalize_verdict_token(row.get("verdict"))
    question = render_source_messages(messages)
    meta_prompt = p_true_meta_prompt(question, verdict_token)
    p_true_prompt = profile.render_prompt(
        tokenizer,
        [{"role": "user", "content": meta_prompt}],
    )
    return ReplayItem(
        source_stage=source_stage,
        source_row=row,
        original_prompt=package.prompt_text,
        p_true_prompt=p_true_prompt,
        verdict_token_text=verdict_token,
    )


def reconstruct_replay_selection(
    *,
    data_path: Path,
    campaign_model_dir: Path,
    model_registry_name: str,
    tokenizer: Any,
    require_exact_counts: bool = True,
) -> ReplaySelection:
    """Reconstruct and hash-check every frozen prompt before inference.

    The returned order is deterministic: the stored Stage-A row order followed
    by the stored Stage-B row order after the primary-stratum filter.
    """

    profile = get_model_profile(model_registry_name)
    input_file_hash = file_sha256(data_path)
    stage_a_path = campaign_model_dir / "silent_bias_stage_a_run_records.jsonl"
    stage_b_path = campaign_model_dir / "silent_bias_stage_b_run_records.jsonl"
    clean_summary_path = campaign_model_dir / "silent_bias_stage_a_pair_summary.jsonl"
    for path in (stage_a_path, stage_b_path, clean_summary_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    stage_a_rows = read_jsonl(stage_a_path)
    full_stage_b_count = 0
    primary_stage_b_rows: list[dict[str, Any]] = []
    for row in iter_jsonl(stage_b_path):
        full_stage_b_count += 1
        if is_primary_stage_b_row(row):
            primary_stage_b_rows.append(row)
    if require_exact_counts and len(stage_a_rows) != EXPECTED_STAGE_A_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_STAGE_A_COUNT} Stage-A rows, found {len(stage_a_rows)}"
        )
    if require_exact_counts and full_stage_b_count != EXPECTED_FULL_STAGE_B_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FULL_STAGE_B_COUNT} full Stage-B rows, "
            f"found {full_stage_b_count}"
        )
    if require_exact_counts and len(primary_stage_b_rows) != EXPECTED_PRIMARY_STAGE_B_COUNT:
        raise ValueError(
            "Expected "
            f"{EXPECTED_PRIMARY_STAGE_B_COUNT} primary Stage-B rows, "
            f"found {len(primary_stage_b_rows)}"
        )
    for row in (*stage_a_rows, *primary_stage_b_rows):
        _validate_source_row(row, profile=profile, input_file_hash=input_file_hash)

    pair_inputs, pairs_by_identity = _build_stage_a_inputs_and_examples(
        csv_path=data_path,
        model_name=profile.hf_model_name,
        input_file_hash=input_file_hash,
        limit=None,
    )
    stage_a_plan = generate_stage_a_conditions(pair_inputs)
    if stage_a_plan.issues:
        raise ValueError(f"Stage-A replay planning produced {len(stage_a_plan.issues)} issue(s)")
    stage_a_items = _evaluation_items(stage_a_plan.conditions, pairs_by_identity)

    clean_rows = read_jsonl(clean_summary_path)
    if require_exact_counts and len(clean_rows) != EXPECTED_STAGE_A_COUNT:
        raise ValueError(
            "Expected "
            f"{EXPECTED_STAGE_A_COUNT} Stage-A pair-summary rows, found {len(clean_rows)}"
        )
    stage_b_plan = generate_stage_b_conditions(clean_summaries_from_rows(clean_rows))
    fatal_stage_b_issues = [
        issue for issue in stage_b_plan.issues if issue.code != "clean_and_human_tie"
    ]
    if fatal_stage_b_issues:
        raise ValueError(
            f"Stage-B replay planning produced {len(fatal_stage_b_issues)} fatal issue(s)"
        )
    stage_b_items = _evaluation_items(stage_b_plan.conditions, pairs_by_identity)

    planned_a = _unique_map(
        stage_a_items,
        key=lambda value: _planned_key(value.planned),
        description="Stage-A plan",
    )
    planned_b = _unique_map(
        stage_b_items,
        key=lambda value: _planned_key(value.planned),
        description="Stage-B plan",
    )
    source_a = _unique_map(stage_a_rows, key=_condition_key, description="Stage-A source")
    source_b = _unique_map(
        primary_stage_b_rows,
        key=_condition_key,
        description="Stage-B source",
    )
    if set(source_a) != set(planned_a):
        raise ValueError("Stored Stage-A keys differ from the reconstructed Stage-A plan")
    if not set(source_b).issubset(planned_b):
        raise ValueError("Stored primary Stage-B keys are absent from the reconstructed plan")

    replay_items: list[ReplayItem] = []
    for source_stage, rows, planned in (
        ("stage_a", stage_a_rows, planned_a),
        ("stage_b", primary_stage_b_rows, planned_b),
    ):
        for row in rows:
            replay_items.append(
                _render_replay_item(
                    source_stage=source_stage,
                    row=row,
                    item=planned[_condition_key(row)],
                    profile=profile,
                    tokenizer=tokenizer,
                )
            )

    record_ids = [item.record_id for item in replay_items]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("The replay selection contains duplicate record IDs")
    return ReplaySelection(
        model_registry_name=model_registry_name,
        model_name=profile.hf_model_name,
        model_revision=profile.revision,
        input_file_hash=input_file_hash,
        items=tuple(replay_items),
        stage_a_count=len(stage_a_rows),
        full_stage_b_count=full_stage_b_count,
        primary_stage_b_count=len(primary_stage_b_rows),
        stage_a_pair_summary_count=len(clean_rows),
    )


def source_metadata(row: Mapping[str, Any], source_stage: str) -> dict[str, Any]:
    spec = row.get("spec") if isinstance(row.get("spec"), Mapping) else {}
    condition = row.get("condition") if isinstance(row.get("condition"), Mapping) else {}
    metadata = row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {}
    raw_prompt_logprobs = row.get("raw_prompt_logprobs")
    condition_clean_tie = condition.get("clean_tie")
    clean_tie = (
        bool(condition_clean_tie)
        if condition_clean_tie is not None
        else source_stage == "stage_a" and normalize_verdict_token(row.get("verdict")) == "T"
    )
    family = condition.get("bias_type")
    msp = _source_msp(row)
    return {
        "record_id": str(row.get("record_id")),
        "question_id": str(row.get("question_id")),
        "example_id": str(row.get("example_id")),
        "model_name": str(spec.get("model_name")),
        "model_revision": spec.get("model_revision"),
        "source_stage": source_stage,
        "prompt_hash": str(row.get("prompt_hash")),
        "pair_identity_key": metadata.get("pair_identity_key"),
        "pair_key": row.get("pair_key"),
        "condition_group_id": row.get("condition_group_id"),
        "ordering": condition.get("ordering"),
        "routing_split": metadata.get("routing_split"),
        "family": family,
        "bias_type": family,
        "bias_name": family,
        "direction": condition.get("cue_congruency"),
        "cue_congruency": condition.get("cue_congruency"),
        "direction_relative_human": condition.get("direction_relative_human"),
        "dose": condition.get("dose"),
        "cue_target": condition.get("cue_target"),
        "clean_tie": clean_tie,
        "human_winner": metadata.get("human_winner"),
        "verdict": row.get("verdict"),
        "msp": msp,
        "raw_msp": msp,
        "raw_prompt_logprobs": raw_prompt_logprobs,
        "source_spec_hash": row.get("spec_hash"),
        "input_file_hash": row.get("input_file_hash"),
    }


def make_score_row(
    *,
    item: ReplayItem,
    full_vocab: FullVocabularyMetrics,
    p_true: PTrueMetrics,
    restricted: RestrictedLabelMetrics,
    true_token_id: int,
    false_token_id: int,
    label_token_ids: Mapping[str, int],
    original_token_count: int,
    p_true_token_count: int,
    original_token_ids_sha256: str,
    p_true_token_ids_sha256: str,
    vocabulary_size: int,
    tokenizer_vocabulary_size: int,
    collector_spec_hash: str,
) -> dict[str, Any]:
    row = source_metadata(item.source_row, item.source_stage)
    source_probs = item.source_row.get("raw_prompt_logprobs")
    max_difference: float | None = None
    if isinstance(source_probs, Mapping) and source_probs:
        normalized_source = {
            normalize_verdict_token(label): float(value)
            for label, value in source_probs.items()
        }
        translated = {
            "A": restricted.probabilities["A"],
            "B": restricted.probabilities["B"],
            "T": restricted.probabilities["tie"],
        }
        if set(normalized_source) == {"A", "B", "T"}:
            max_difference = max(
                abs(normalized_source[label] - translated[label])
                for label in ("A", "B", "T")
            )
    map_label = max(
        ("A", "B", "tie"),
        key=lambda label: restricted.probabilities[label],
    )
    map_verdict = "T" if map_label == "tie" else map_label
    source_restricted = {
        "A": float(source_probs["A"]),
        "B": float(source_probs["B"]),
        "tie": float(source_probs["tie"]),
    }
    source_gaps = restricted_pairwise_logit_gaps(source_restricted)
    hf_gaps = restricted_pairwise_logit_gaps(restricted.probabilities)
    gap_differences = {
        key: (
            hf_gaps[key] - source_gaps[key]
            if hf_gaps[key] is not None and source_gaps[key] is not None
            else None
        )
        for key in ("A_minus_B", "A_minus_tie", "B_minus_tie")
    }
    available_gap_differences = [
        abs(value) for value in gap_differences.values() if value is not None
    ]
    map_matches_stored = map_verdict == item.verdict_token_text
    row.update(
        {
            "verdict_token_text": item.verdict_token_text,
            "original_prompt_text_sha256": sha256_bytes(
                item.original_prompt.encode("utf-8")
            ),
            "p_true_prompt_text_sha256": sha256_bytes(
                item.p_true_prompt.encode("utf-8")
            ),
            "original_token_ids_sha256": original_token_ids_sha256,
            "p_true_token_ids_sha256": p_true_token_ids_sha256,
            "p_true_log_probability": p_true.p_true_log_probability,
            "p_true_probability": p_true.p_true_probability,
            "p_true_uncertainty": p_true.p_true_uncertainty,
            "p_true_confidence": p_true.p_true_probability,
            "mean_token_entropy": full_vocab.mean_token_entropy,
            "mean_token_entropy_uncertainty": full_vocab.mean_token_entropy,
            "mean_token_entropy_confidence": full_vocab.mean_token_entropy_confidence,
            "self_certainty": full_vocab.self_certainty,
            "self_certainty_uncertainty": full_vocab.self_certainty,
            "self_certainty_confidence": full_vocab.self_certainty_confidence,
            "hf_restricted_label_probabilities": restricted.probabilities,
            "hf_restricted_msp": restricted.msp,
            "hf_restricted_map_verdict": map_verdict,
            "hf_restricted_map_matches_stored": map_matches_stored,
            "hf_restricted_verdict_probability": restricted.probabilities[
                "tie" if item.verdict_token_text == "T" else item.verdict_token_text
            ],
            "hf_source_probability_max_abs_difference": max_difference,
            "hf_source_probability_within_tolerance": (
                max_difference is not None
                and max_difference <= SOURCE_PROBABILITY_TOLERANCE
            ),
            "source_restricted_pairwise_logit_gaps": source_gaps,
            "hf_restricted_pairwise_logit_gaps": hf_gaps,
            "hf_source_pairwise_logit_gap_differences": gap_differences,
            "hf_source_pairwise_logit_gap_max_abs_difference": (
                max(available_gap_differences)
                if available_gap_differences
                else None
            ),
            "hf_source_pairwise_logit_gap_available_count": len(
                available_gap_differences
            ),
            "hf_source_pairwise_logit_gap_complete": (
                len(available_gap_differences) == 3
            ),
            "true_token_id": true_token_id,
            "false_token_id": false_token_id,
            "verdict_token_ids": dict(label_token_ids),
            "original_token_count": original_token_count,
            "p_true_token_count": p_true_token_count,
            "vocabulary_size": vocabulary_size,
            "tokenizer_vocabulary_size": tokenizer_vocabulary_size,
            "padded_vocabulary_size_delta": vocabulary_size
            - tokenizer_vocabulary_size,
            "collector_spec_hash": collector_spec_hash,
        }
    )
    return row


def prompt_token_length_preflight(
    selection: ReplaySelection,
    *,
    tokenizer: Any,
    max_model_len: int = FROZEN_MAX_MODEL_LEN,
) -> dict[str, Any]:
    """Tokenize both prompts for the *full* selection and enforce the frozen limit."""

    if max_model_len < 1:
        raise ValueError("max_model_len must be positive")
    original_lengths: list[int] = []
    p_true_lengths: list[int] = []
    original_token_ids_digest = hashlib.sha256()
    p_true_token_ids_digest = hashlib.sha256()
    overlimit: list[dict[str, Any]] = []
    for item in selection.items:
        original_ids = tokenizer.encode(item.original_prompt, add_special_tokens=False)
        p_true_ids = tokenizer.encode(item.p_true_prompt, add_special_tokens=False)
        original_length = len(original_ids)
        p_true_length = len(p_true_ids)
        if original_length < 1 or p_true_length < 1:
            raise ValueError(f"Record {item.record_id!r} has an empty tokenized prompt")
        original_lengths.append(original_length)
        p_true_lengths.append(p_true_length)
        for digest, token_ids in (
            (original_token_ids_digest, original_ids),
            (p_true_token_ids_digest, p_true_ids),
        ):
            digest.update(item.record_id.encode("utf-8"))
            digest.update(b"\0")
            digest.update(
                json.dumps(
                    [int(token_id) for token_id in token_ids],
                    separators=(",", ":"),
                ).encode("ascii")
            )
            digest.update(b"\n")
        if original_length > max_model_len or p_true_length > max_model_len:
            overlimit.append(
                {
                    "record_id": item.record_id,
                    "original_token_count": original_length,
                    "p_true_token_count": p_true_length,
                }
            )
    result = {
        "checked_record_count": len(selection.items),
        "prompts_per_record": 2,
        "max_model_len": max_model_len,
        "max_original_token_count": max(original_lengths, default=0),
        "max_p_true_token_count": max(p_true_lengths, default=0),
        "overlimit_count": len(overlimit),
        "original_token_ids_sha256": original_token_ids_digest.hexdigest(),
        "p_true_token_ids_sha256": p_true_token_ids_digest.hexdigest(),
        "length_digest": canonical_json_sha256(
            list(zip(original_lengths, p_true_lengths, strict=True))
        ),
    }
    if overlimit:
        first = overlimit[0]
        raise ValueError(
            f"{len(overlimit)} replay records exceed frozen max_model_len={max_model_len}; "
            f"first={first!r}"
        )
    return result


def validate_scientific_score_gates(
    rows: Sequence[Mapping[str, Any]],
    *,
    probability_tolerance: float = SOURCE_PROBABILITY_TOLERANCE,
) -> dict[str, Any]:
    """Summarize cross-backend replay while enforcing only intrinsic hard gates.

    The frozen decisions were produced by vLLM while these UE scores are
    recomputed by Transformers.  The exact source attention kernel and
    effective arithmetic were not preserved.  Exact prompt, token, model, and
    score-algebra checks remain hard requirements elsewhere in this module.
    MAP and probability equality are reported as transfer diagnostics:
    enforcing them would selectively reject unsaturated, high-uncertainty rows.
    """

    if probability_tolerance < 0:
        raise ValueError("probability_tolerance must be non-negative")
    maximum_difference = 0.0
    differences: list[float] = []
    maximum_logit_gap_difference = 0.0
    logit_gap_differences: list[float] = []
    unavailable_logit_gap_diagnostic_ids: list[str] = []
    partial_logit_gap_diagnostic_ids: list[str] = []
    complete_logit_gap_diagnostic_ids: list[str] = []
    total_available_logit_gaps = 0
    map_mismatch_ids: list[str] = []
    probability_exceedance_ids: list[str] = []
    for row in rows:
        record_id = str(row.get("record_id", ""))
        stored = normalize_verdict_token(row.get("verdict"))
        replayed = normalize_verdict_token(row.get("hf_restricted_map_verdict"))
        if stored != replayed:
            map_mismatch_ids.append(record_id)
        difference = row.get("hf_source_probability_max_abs_difference")
        if difference is None:
            raise ValueError(f"Record {record_id!r} has no source/HF probability comparison")
        numeric_difference = float(difference)
        if not math.isfinite(numeric_difference) or numeric_difference < 0:
            raise ValueError(f"Record {record_id!r} has invalid probability difference")
        maximum_difference = max(maximum_difference, numeric_difference)
        differences.append(numeric_difference)
        if numeric_difference > probability_tolerance:
            probability_exceedance_ids.append(record_id)
        logit_gap_difference = row.get(
            "hf_source_pairwise_logit_gap_max_abs_difference"
        )
        available_gap_count = row.get(
            "hf_source_pairwise_logit_gap_available_count"
        )
        if (
            isinstance(available_gap_count, bool)
            or not isinstance(available_gap_count, int)
            or not 0 <= available_gap_count <= 3
        ):
            raise ValueError(
                f"Record {record_id!r} has invalid pairwise-logit-gap availability"
            )
        expected_complete = available_gap_count == 3
        if row.get("hf_source_pairwise_logit_gap_complete") is not expected_complete:
            raise ValueError(
                f"Record {record_id!r} has inconsistent pairwise-logit-gap completeness"
            )
        total_available_logit_gaps += available_gap_count
        if expected_complete:
            complete_logit_gap_diagnostic_ids.append(record_id)
        elif available_gap_count:
            partial_logit_gap_diagnostic_ids.append(record_id)
        if logit_gap_difference is None:
            if available_gap_count != 0:
                raise ValueError(
                    f"Record {record_id!r} has gaps but no maximum gap difference"
                )
            unavailable_logit_gap_diagnostic_ids.append(record_id)
        else:
            if available_gap_count == 0:
                raise ValueError(
                    f"Record {record_id!r} has a gap maximum but zero available gaps"
                )
            numeric_logit_gap_difference = float(logit_gap_difference)
            if (
                not math.isfinite(numeric_logit_gap_difference)
                or numeric_logit_gap_difference < 0
            ):
                raise ValueError(
                    f"Record {record_id!r} has invalid pairwise-logit-gap difference"
                )
            maximum_logit_gap_difference = max(
                maximum_logit_gap_difference,
                numeric_logit_gap_difference,
            )
            logit_gap_differences.append(numeric_logit_gap_difference)
        for key in ("original_token_count", "p_true_token_count"):
            count = int(row.get(key, 0))
            if not 1 <= count <= FROZEN_MAX_MODEL_LEN:
                raise ValueError(
                    f"Record {record_id!r} has invalid {key}={count}; "
                    f"expected 1..{FROZEN_MAX_MODEL_LEN}"
                )
    quantiles = {
        name: (
            float(np.quantile(np.asarray(differences, dtype=np.float64), probability))
            if differences
            else None
        )
        for name, probability in (("p50", 0.50), ("p90", 0.90), ("p95", 0.95), ("p99", 0.99))
    }
    logit_gap_quantiles = {
        name: (
            float(
                np.quantile(
                    np.asarray(logit_gap_differences, dtype=np.float64),
                    probability,
                )
            )
            if logit_gap_differences
            else None
        )
        for name, probability in (
            ("p50", 0.50),
            ("p90", 0.90),
            ("p95", 0.95),
            ("p99", 0.99),
        )
    }
    checked = len(rows)
    map_matches = checked - len(map_mismatch_ids)
    within_tolerance = checked - len(probability_exceedance_ids)
    return {
        "checked_record_count": len(rows),
        "hard_gates_passed": True,
        "cross_backend_replay_role": CROSS_BACKEND_REPLAY_ROLE,
        "cross_backend_replay_reason": CROSS_BACKEND_REPLAY_REASON,
        "restricted_map_agreement_enforced": False,
        "restricted_map_matches_stored": not map_mismatch_ids,
        "restricted_map_match_count": map_matches,
        "restricted_map_mismatch_count": len(map_mismatch_ids),
        "restricted_map_agreement_rate": map_matches / checked if checked else None,
        "restricted_map_mismatch_record_ids_first_20": map_mismatch_ids[:20],
        "restricted_map_mismatch_record_ids_sha256": canonical_json_sha256(
            map_mismatch_ids
        ),
        "probability_tolerance": probability_tolerance,
        "source_probability_tolerance_enforced": False,
        "source_probability_within_tolerance": not probability_exceedance_ids,
        "source_probability_within_tolerance_count": within_tolerance,
        "source_probability_tolerance_exceedance_count": len(
            probability_exceedance_ids
        ),
        "source_probability_within_tolerance_rate": (
            within_tolerance / checked if checked else None
        ),
        "source_probability_tolerance_exceedance_record_ids_first_20": (
            probability_exceedance_ids[:20]
        ),
        "source_probability_tolerance_exceedance_record_ids_sha256": (
            canonical_json_sha256(probability_exceedance_ids)
        ),
        "max_source_probability_abs_difference": maximum_difference,
        "source_probability_abs_difference_quantiles": quantiles,
        "source_pairwise_logit_gap_available_gap_count": (
            total_available_logit_gaps
        ),
        "source_pairwise_logit_gap_total_gap_count": 3 * checked,
        "source_pairwise_logit_gap_availability_rate": (
            total_available_logit_gaps / (3 * checked) if checked else None
        ),
        "source_pairwise_logit_gap_complete_row_count": len(
            complete_logit_gap_diagnostic_ids
        ),
        "source_pairwise_logit_gap_complete_row_rate": (
            len(complete_logit_gap_diagnostic_ids) / checked if checked else None
        ),
        "source_pairwise_logit_gap_partial_row_count": len(
            partial_logit_gap_diagnostic_ids
        ),
        "source_pairwise_logit_gap_partial_record_ids_first_20": (
            partial_logit_gap_diagnostic_ids[:20]
        ),
        "source_pairwise_logit_gap_partial_record_ids_sha256": (
            canonical_json_sha256(partial_logit_gap_diagnostic_ids)
        ),
        "source_pairwise_logit_gap_no_available_row_count": len(
            unavailable_logit_gap_diagnostic_ids
        ),
        "source_pairwise_logit_gap_no_available_record_ids_first_20": (
            unavailable_logit_gap_diagnostic_ids[:20]
        ),
        "source_pairwise_logit_gap_no_available_record_ids_sha256": (
            canonical_json_sha256(unavailable_logit_gap_diagnostic_ids)
        ),
        "max_source_pairwise_logit_gap_abs_difference": (
            maximum_logit_gap_difference if logit_gap_differences else None
        ),
        "source_pairwise_logit_gap_abs_difference_quantiles": (
            logit_gap_quantiles
        ),
        "token_counts_within_frozen_max_model_len": True,
    }


def replay_selection_manifest(
    selection: ReplaySelection,
    *,
    data_path: Path,
    campaign_model_dir: Path,
    limit_records: int | None,
    source_artifact_hashes: Mapping[str, str],
    collector_spec: Mapping[str, Any],
) -> dict[str, Any]:
    selected = selection.items if limit_records is None else selection.items[:limit_records]
    return {
        "schema_version": 1,
        "kind": "silent_bias_lm_polygraph_inference_selection",
        "model_registry_name": selection.model_registry_name,
        "model_name": selection.model_name,
        "model_revision": selection.model_revision,
        "data_path": str(data_path),
        "campaign_model_dir": str(campaign_model_dir),
        "input_file_hash": selection.input_file_hash,
        "full_preflight_count": len(selection.items),
        "stage_a_count": selection.stage_a_count,
        "full_stage_b_count": selection.full_stage_b_count,
        "primary_stage_b_count": selection.primary_stage_b_count,
        "stage_a_pair_summary_count": selection.stage_a_pair_summary_count,
        "limit_records": limit_records,
        "smoke_only": limit_records is not None,
        "inference_count": len(selected),
        "full_record_id_digest": canonical_json_sha256(
            [item.record_id for item in selection.items]
        ),
        "full_prompt_hash_digest": canonical_json_sha256(
            [str(item.source_row["prompt_hash"]) for item in selection.items]
        ),
        "full_p_true_prompt_sha256_digest": canonical_json_sha256(
            [
                sha256_bytes(item.p_true_prompt.encode("utf-8"))
                for item in selection.items
            ]
        ),
        "inference_record_id_digest": canonical_json_sha256(
            [item.record_id for item in selected]
        ),
        "source_artifact_hashes": dict(sorted(source_artifact_hashes.items())),
        "p_true_template": P_TRUE_TEMPLATE,
        "p_true_template_sha256": sha256_bytes(P_TRUE_TEMPLATE.encode("utf-8")),
        "lm_polygraph_commit": LM_POLYGRAPH_COMMIT,
        "lm_polygraph_source_hashes": dict(LM_POLYGRAPH_SOURCE_HASHES),
        "collector_spec": dict(collector_spec),
        "collector_spec_hash": canonical_json_sha256(collector_spec),
    }


def validate_existing_score_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    selected_record_ids: Sequence[str],
    collector_spec_hash: str,
) -> set[str]:
    allowed = set(selected_record_ids)
    seen: set[str] = set()
    for row in rows:
        record_id = str(row.get("record_id", ""))
        if not record_id or record_id in seen:
            raise ValueError("Existing score file has missing or duplicate record IDs")
        if record_id not in allowed:
            raise ValueError(f"Existing score row {record_id!r} is outside this selection")
        if row.get("collector_spec_hash") != collector_spec_hash:
            raise ValueError(f"Existing score row {record_id!r} has a different collector spec")
        seen.add(record_id)
    expected_prefix = list(selected_record_ids[: len(rows)])
    if [str(row.get("record_id")) for row in rows] != expected_prefix:
        raise ValueError("Existing score rows are not the deterministic selection prefix")
    return seen


def _finite_score_value(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    if isinstance(value, bool):
        raise ValueError(f"{key} must be a finite number")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a finite number") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{key} must be a finite number")
    return numeric


def _source_probability_comparison(
    source_row: Mapping[str, Any],
    restricted_probabilities: Mapping[str, float],
) -> float:
    source_probs = source_row.get("raw_prompt_logprobs")
    if not isinstance(source_probs, Mapping):
        raise ValueError("source row has no A/B/tie probabilities")
    normalized_source = {
        normalize_verdict_token(label): float(value)
        for label, value in source_probs.items()
    }
    if set(normalized_source) != {"A", "B", "T"}:
        raise ValueError("source row probabilities do not have exact A/B/tie support")
    translated = {
        "A": float(restricted_probabilities["A"]),
        "B": float(restricted_probabilities["B"]),
        "T": float(restricted_probabilities["tie"]),
    }
    return max(
        abs(normalized_source[label] - translated[label])
        for label in ("A", "B", "T")
    )


def validate_score_rows_against_selection(
    rows: Sequence[Mapping[str, Any]],
    *,
    selected_items: Sequence[ReplayItem],
    collector_spec_hash: str,
    tokenizer: Any,
    true_token_id: int,
    false_token_id: int,
    label_token_ids: Mapping[str, int],
    vocabulary_size: int,
    tokenizer_vocabulary_size: int,
) -> dict[str, Any]:
    """Recompute every resumable/final row invariant from immutable inputs.

    This validator deliberately does not trust derived MAP, probability-drift,
    score-orientation, prompt-hash, token-count, or vocabulary fields already
    present in the JSONL.  It binds each deterministic-prefix row back to its
    reconstructed frozen prompt and source record before a completion marker
    can be written.
    """

    selected_record_ids = [item.record_id for item in selected_items]
    validate_existing_score_rows(
        rows,
        selected_record_ids=selected_record_ids,
        collector_spec_hash=collector_spec_hash,
    )
    if len(rows) > len(selected_items):
        raise ValueError("score rows exceed the deterministic selection")
    expected_label_ids = {
        label: int(label_token_ids[label]) for label in ("A", "B", "tie")
    }
    if set(expected_label_ids) != {"A", "B", "tie"}:
        raise ValueError("label_token_ids must contain exact A/B/tie support")

    for row, item in zip(rows, selected_items[: len(rows)], strict=True):
        record_id = item.record_id
        expected_metadata = source_metadata(item.source_row, item.source_stage)
        for key in (
            "record_id",
            "question_id",
            "example_id",
            "model_name",
            "model_revision",
            "source_stage",
            "prompt_hash",
            "pair_identity_key",
            "pair_key",
            "condition_group_id",
            "ordering",
            "routing_split",
            "family",
            "direction",
            "dose",
            "clean_tie",
            "human_winner",
            "verdict",
            "source_spec_hash",
            "input_file_hash",
            "raw_prompt_logprobs",
        ):
            if row.get(key) != expected_metadata.get(key):
                raise ValueError(f"Record {record_id!r} has mismatched {key}")
        expected_hashes = {
            "original_prompt_text_sha256": sha256_bytes(
                item.original_prompt.encode("utf-8")
            ),
            "p_true_prompt_text_sha256": sha256_bytes(
                item.p_true_prompt.encode("utf-8")
            ),
        }
        for key, expected in expected_hashes.items():
            if row.get(key) != expected:
                raise ValueError(f"Record {record_id!r} has mismatched {key}")
        expected_original_count = len(
            original_token_ids := tokenizer.encode(
                item.original_prompt,
                add_special_tokens=False,
            )
        )
        expected_p_true_count = len(
            p_true_token_ids := tokenizer.encode(
                item.p_true_prompt,
                add_special_tokens=False,
            )
        )
        if int(row.get("original_token_count", 0)) != expected_original_count:
            raise ValueError(f"Record {record_id!r} has mismatched original token count")
        if int(row.get("p_true_token_count", 0)) != expected_p_true_count:
            raise ValueError(f"Record {record_id!r} has mismatched P(True) token count")
        expected_token_hashes = {
            "original_token_ids_sha256": token_ids_sha256(original_token_ids),
            "p_true_token_ids_sha256": token_ids_sha256(p_true_token_ids),
        }
        for key, expected in expected_token_hashes.items():
            if row.get(key) != expected:
                raise ValueError(f"Record {record_id!r} has mismatched {key}")
        if row.get("verdict_token_text") != item.verdict_token_text:
            raise ValueError(f"Record {record_id!r} has mismatched verdict token text")
        if row.get("verdict_token_ids") != expected_label_ids:
            raise ValueError(f"Record {record_id!r} has mismatched verdict token IDs")
        if int(row.get("true_token_id", -1)) != int(true_token_id):
            raise ValueError(f"Record {record_id!r} has mismatched True token ID")
        if int(row.get("false_token_id", -1)) != int(false_token_id):
            raise ValueError(f"Record {record_id!r} has mismatched False token ID")
        if int(row.get("vocabulary_size", -1)) != vocabulary_size:
            raise ValueError(f"Record {record_id!r} has mismatched model vocabulary size")
        if int(row.get("tokenizer_vocabulary_size", -1)) != tokenizer_vocabulary_size:
            raise ValueError(f"Record {record_id!r} has mismatched tokenizer vocabulary size")
        if int(row.get("padded_vocabulary_size_delta", -1)) != (
            vocabulary_size - tokenizer_vocabulary_size
        ):
            raise ValueError(f"Record {record_id!r} has mismatched padded vocabulary delta")

        p_log = _finite_score_value(row, "p_true_log_probability")
        p_probability = _finite_score_value(row, "p_true_probability")
        p_uncertainty = _finite_score_value(row, "p_true_uncertainty")
        if p_log > 1e-12 or not 0.0 <= p_probability <= 1.0:
            raise ValueError(f"Record {record_id!r} has invalid P(True) range")
        if not math.isclose(p_uncertainty, -p_log, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError(f"Record {record_id!r} has inconsistent P(True) uncertainty")
        if not math.isclose(
            p_probability,
            math.exp(p_log),
            rel_tol=1e-10,
            abs_tol=1e-15,
        ):
            raise ValueError(f"Record {record_id!r} has inconsistent P(True) probability")
        if not math.isclose(
            _finite_score_value(row, "p_true_confidence"),
            p_probability,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(f"Record {record_id!r} has inconsistent P(True) confidence")

        entropy = _finite_score_value(row, "mean_token_entropy")
        entropy_uncertainty = _finite_score_value(row, "mean_token_entropy_uncertainty")
        entropy_confidence = _finite_score_value(row, "mean_token_entropy_confidence")
        if not 0.0 <= entropy <= math.log(vocabulary_size) + 1e-8:
            raise ValueError(f"Record {record_id!r} has invalid token entropy")
        if entropy_uncertainty != entropy or entropy_confidence != -entropy:
            raise ValueError(f"Record {record_id!r} has inconsistent entropy orientation")

        self_certainty = _finite_score_value(row, "self_certainty")
        self_uncertainty = _finite_score_value(row, "self_certainty_uncertainty")
        self_confidence = _finite_score_value(row, "self_certainty_confidence")
        if self_certainty > 1e-8:
            raise ValueError(f"Record {record_id!r} has invalid SelfCertainty")
        if self_uncertainty != self_certainty or self_confidence != -self_certainty:
            raise ValueError(f"Record {record_id!r} has inconsistent SelfCertainty orientation")

        restricted_raw = row.get("hf_restricted_label_probabilities")
        if not isinstance(restricted_raw, Mapping) or set(restricted_raw) != {
            "A",
            "B",
            "tie",
        }:
            raise ValueError(f"Record {record_id!r} has invalid restricted probabilities")
        restricted = {
            label: _finite_score_value(restricted_raw, label)
            for label in ("A", "B", "tie")
        }
        if any(not 0.0 <= value <= 1.0 for value in restricted.values()) or not math.isclose(
            sum(restricted.values()), 1.0, rel_tol=0.0, abs_tol=1e-10
        ):
            raise ValueError(f"Record {record_id!r} has invalid restricted probability simplex")
        restricted_msp = max(restricted.values())
        if not math.isclose(
            _finite_score_value(row, "hf_restricted_msp"),
            restricted_msp,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"Record {record_id!r} has inconsistent restricted MSP")
        map_label = max(("A", "B", "tie"), key=lambda label: restricted[label])
        map_verdict = "T" if map_label == "tie" else map_label
        if row.get("hf_restricted_map_verdict") != map_verdict:
            raise ValueError(f"Record {record_id!r} has inconsistent restricted MAP")
        expected_map_match = map_verdict == item.verdict_token_text
        if row.get("hf_restricted_map_matches_stored") is not expected_map_match:
            raise ValueError(f"Record {record_id!r} has forged MAP-agreement diagnostic")
        verdict_key = "tie" if item.verdict_token_text == "T" else item.verdict_token_text
        if not math.isclose(
            _finite_score_value(row, "hf_restricted_verdict_probability"),
            restricted[verdict_key],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"Record {record_id!r} has inconsistent verdict probability")
        recomputed_difference = _source_probability_comparison(
            item.source_row,
            restricted,
        )
        if not math.isclose(
            _finite_score_value(row, "hf_source_probability_max_abs_difference"),
            recomputed_difference,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"Record {record_id!r} has forged source probability drift")
        expected_within_tolerance = (
            recomputed_difference <= SOURCE_PROBABILITY_TOLERANCE
        )
        if (
            row.get("hf_source_probability_within_tolerance")
            is not expected_within_tolerance
        ):
            raise ValueError(
                f"Record {record_id!r} has forged probability-tolerance diagnostic"
            )

        source_raw = item.source_row.get("raw_prompt_logprobs")
        if not isinstance(source_raw, Mapping):
            raise ValueError(f"Record {record_id!r} has no source probabilities")
        source_restricted = {
            "A": float(source_raw["A"]),
            "B": float(source_raw["B"]),
            "tie": float(source_raw["tie"]),
        }
        expected_source_gaps = restricted_pairwise_logit_gaps(source_restricted)
        expected_hf_gaps = restricted_pairwise_logit_gaps(restricted)
        expected_gap_differences = {
            key: (
                expected_hf_gaps[key] - expected_source_gaps[key]
                if (
                    expected_hf_gaps[key] is not None
                    and expected_source_gaps[key] is not None
                )
                else None
            )
            for key in ("A_minus_B", "A_minus_tie", "B_minus_tie")
        }
        for field, expected_values in (
            ("source_restricted_pairwise_logit_gaps", expected_source_gaps),
            ("hf_restricted_pairwise_logit_gaps", expected_hf_gaps),
            (
                "hf_source_pairwise_logit_gap_differences",
                expected_gap_differences,
            ),
        ):
            actual = row.get(field)
            if not isinstance(actual, Mapping) or set(actual) != set(expected_values):
                raise ValueError(f"Record {record_id!r} has invalid {field}")
            for key, expected in expected_values.items():
                if expected is None:
                    if actual.get(key) is not None:
                        raise ValueError(f"Record {record_id!r} has forged {field}")
                    continue
                if not math.isclose(
                    _finite_score_value(actual, key),
                    expected,
                    rel_tol=0.0,
                    abs_tol=1e-10,
                ):
                    raise ValueError(f"Record {record_id!r} has forged {field}")
        available_gap_differences = [
            abs(value)
            for value in expected_gap_differences.values()
            if value is not None
        ]
        expected_gap_max = (
            max(available_gap_differences)
            if available_gap_differences
            else None
        )
        actual_gap_max = row.get(
            "hf_source_pairwise_logit_gap_max_abs_difference"
        )
        if expected_gap_max is None:
            gap_max_matches = actual_gap_max is None
        else:
            gap_max_matches = math.isclose(
                _finite_score_value(
                    row,
                    "hf_source_pairwise_logit_gap_max_abs_difference",
                ),
                expected_gap_max,
                rel_tol=0.0,
                abs_tol=1e-10,
            )
        if not gap_max_matches:
            raise ValueError(f"Record {record_id!r} has forged logit-gap drift")
        expected_gap_count = len(available_gap_differences)
        if row.get("hf_source_pairwise_logit_gap_available_count") != expected_gap_count:
            raise ValueError(f"Record {record_id!r} has forged logit-gap availability")
        if (
            row.get("hf_source_pairwise_logit_gap_complete")
            is not (expected_gap_count == 3)
        ):
            raise ValueError(f"Record {record_id!r} has forged logit-gap completeness")
        if not math.isclose(
            _finite_score_value(row, "msp"),
            _source_msp(item.source_row),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(f"Record {record_id!r} has source MSP drift")

    return validate_scientific_score_gates(rows)


__all__ = [
    "COMPLETE_FILE_NAME",
    "CROSS_BACKEND_REPLAY_REASON",
    "CROSS_BACKEND_REPLAY_ROLE",
    "EXPECTED_PRIMARY_STAGE_B_COUNT",
    "EXPECTED_FULL_STAGE_B_COUNT",
    "EXPECTED_STAGE_A_COUNT",
    "FROZEN_MAX_MODEL_LEN",
    "FullVocabularyMetrics",
    "LM_POLYGRAPH_COMMIT",
    "PREFLIGHT_COMPLETE_FILE_NAME",
    "P_TRUE_TEMPLATE",
    "PTrueMetrics",
    "ReplayItem",
    "ReplaySelection",
    "RestrictedLabelMetrics",
    "SCORE_FILE_NAME",
    "SELECTION_FILE_NAME",
    "SOURCE_PROBABILITY_TOLERANCE",
    "canonical_json_sha256",
    "full_vocabulary_metrics",
    "is_primary_stage_b_row",
    "iter_jsonl",
    "make_score_row",
    "normalize_verdict_token",
    "p_true_meta_prompt",
    "p_true_metrics",
    "prompt_token_length_preflight",
    "read_jsonl",
    "reconstruct_replay_selection",
    "replay_selection_manifest",
    "restricted_label_metrics",
    "restricted_pairwise_logit_gaps",
    "token_ids_sha256",
    "validate_scientific_score_gates",
    "validate_score_rows_against_selection",
    "validate_existing_score_rows",
]
