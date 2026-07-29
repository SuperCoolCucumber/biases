from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


LABELS = ("A", "B", "tie")
_VARIANT_RE = re.compile(
    r"^(?P<family>[a-z]+)_(?P<direction>congruent|incongruent)_(?P<dose>\d+(?:\.\d+)?)"
    r"(?:_(?P<ordering>ab|ba|original|swapped))?$",
    re.IGNORECASE,
)


def normalize_label(value: Any) -> str | None:
    if value is None:
        return None
    raw = getattr(value, "value", value)
    text = str(raw).strip()
    lowered = text.lower()
    if lowered in {"a", "answer_a", "model_a", "response_a"}:
        return "A"
    if lowered in {"b", "answer_b", "model_b", "response_b"}:
        return "B"
    if lowered in {"t", "tie", "c", "equal"}:
        return "tie"
    return None


def opposite_label(label: str) -> str:
    normalized = normalize_label(label)
    if normalized == "A":
        return "B"
    if normalized == "B":
        return "A"
    return "tie"


def _nested(row: Mapping[str, Any], *path: str) -> Any:
    value: Any = row
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _first(row: Mapping[str, Any], *candidates: str | tuple[str, ...]) -> Any:
    for candidate in candidates:
        value = _nested(row, *candidate) if isinstance(candidate, tuple) else row.get(candidate)
        if value is not None:
            return value
    return None


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _ordering(row: Mapping[str, Any], variant_id: str, example_id: str) -> str:
    explicit = _first(row, "ordering", "order", ("metadata", "ordering"), ("metadata", "order"))
    if explicit is not None:
        return str(explicit).strip().lower()
    match = _VARIANT_RE.match(variant_id)
    if match and match.group("ordering"):
        return match.group("ordering").lower()
    lowered = f"{variant_id} {example_id}".lower()
    if any(token in lowered for token in ("_ba", ":ba", "swapped")):
        return "ba"
    if any(token in lowered for token in ("_ab", ":ab", "original")):
        return "ab"
    return "unknown"


def _family_direction_dose(
    row: Mapping[str, Any],
    variant_id: str,
) -> tuple[str, str, float | None]:
    match = _VARIANT_RE.match(variant_id)
    family = _first(
        row,
        "family",
        "bias_family",
        "bias_name",
        ("condition", "bias_type"),
        ("spec", "bias_name"),
    )
    direction = _first(
        row,
        "direction",
        "cue_direction",
        "cue_congruency",
        ("condition", "cue_congruency"),
    )
    dose = _first(row, "dose", "cue_dose", ("condition", "metadata", "dose"), ("metadata", "dose"))
    if match:
        family = family or match.group("family")
        direction = direction or match.group("direction")
        dose = dose if dose is not None else match.group("dose")
    family_text = str(family or ("clean" if variant_id in {"clean", "control"} else "unknown")).lower()
    direction_text = str(
        direction or ("clean" if variant_id in {"clean", "control"} else "none")
    ).lower()
    return family_text, direction_text, _optional_float(dose)


@dataclass(frozen=True, slots=True)
class ConditionRecord:
    record_id: str
    example_id: str
    question_id: str
    pair_key: str
    pair_identity_key: str | None
    condition_group_id: str | None
    clean_record_id: str | None
    ordering: str
    model_name: str
    routing_split: str | None
    family: str
    direction: str
    dose: float | None
    variant_id: str
    cue_target: str | None
    human_winner: str | None
    verdict: str
    clean_tie: bool
    probability_a: float | None
    probability_b: float | None
    probability_tie: float | None
    entropy: float | None
    normalized_entropy: float | None
    msp: float | None
    margin: float | None
    verbalized_confidence: float | None
    consistency_entropy: float | None
    consistency_agreement: float | None

    @property
    def key(self) -> tuple[str, str, str]:
        return self.model_name, self.pair_key, self.ordering

    @property
    def probabilities(self) -> tuple[float, float, float] | None:
        values = (self.probability_a, self.probability_b, self.probability_tie)
        if any(value is None for value in values):
            return None
        numeric = tuple(max(0.0, float(value)) for value in values if value is not None)
        total = sum(numeric)
        if total <= 0.0:
            return None
        return tuple(value / total for value in numeric)  # type: ignore[return-value]

    def probability_for(self, label: str) -> float | None:
        probs = self.probabilities
        normalized = normalize_label(label)
        if probs is None or normalized is None:
            return None
        return probs[LABELS.index(normalized)]


def record_from_mapping(row: Mapping[str, Any]) -> ConditionRecord:
    variant_id = str(
        _first(row, "variant_id", ("condition", "variant_id"), ("metadata", "variant_id")) or "clean"
    )
    example_id = str(_first(row, "example_id", ("example", "example_id")) or "")
    question_id = str(_first(row, "question_id", ("example", "question_id")) or example_id)
    family, direction, dose = _family_direction_dose(row, variant_id)
    metadata_pair = _first(row, "pair_key", ("metadata", "pair_key"), "pair_id", ("metadata", "pair_id"))
    condition_group_id = _first(
        row,
        "condition_group_id",
        ("metadata", "condition_group_id"),
    )
    pair_identity_key = _first(
        row,
        "pair_identity_key",
        ("metadata", "pair_identity_key"),
    )
    pair_key = str(metadata_pair or condition_group_id or example_id)
    verdict = normalize_label(_first(row, "verdict", ("response", "verdict")))
    if verdict is None:
        raise ValueError(f"record {row.get('record_id')!r} has no valid verdict")
    human_winner = normalize_label(
        _first(row, "human_winner", "human_label", ("metadata", "human_winner"))
    )
    cue_target = normalize_label(
        _first(row, "cue_target", ("condition", "cue_target"), ("metadata", "cue_target"))
    )
    consistency_entropy = _first(
        row,
        "consistency_vote_entropy",
        "consistency_entropy",
        ("uncertainty", "consistency", "vote_entropy"),
    )
    consistency_agreement = _first(
        row,
        "consistency_agreement_rate",
        "agreement_rate",
        ("uncertainty", "consistency", "agreement_rate"),
    )
    clean_tie_value = _first(row, "clean_tie", ("metadata", "clean_tie"))
    return ConditionRecord(
        record_id=str(row.get("record_id") or ""),
        example_id=example_id,
        question_id=question_id,
        pair_key=pair_key,
        pair_identity_key=(
            str(pair_identity_key) if pair_identity_key is not None else None
        ),
        condition_group_id=str(condition_group_id) if condition_group_id is not None else None,
        clean_record_id=(
            str(_first(row, "clean_record_id", "clean_partner_record_id", ("metadata", "clean_record_id")))
            if _first(row, "clean_record_id", "clean_partner_record_id", ("metadata", "clean_record_id"))
            is not None
            else None
        ),
        ordering=_ordering(row, variant_id, example_id),
        model_name=str(_first(row, "model_name", ("spec", "model_name")) or ""),
        routing_split=(
            str(_first(row, "routing_split", ("metadata", "routing_split")))
            if _first(row, "routing_split", ("metadata", "routing_split")) is not None
            else None
        ),
        family=family,
        direction=direction,
        dose=dose,
        variant_id=variant_id,
        cue_target=cue_target,
        human_winner=human_winner,
        verdict=verdict,
        clean_tie=(
            _as_bool(clean_tie_value)
            if clean_tie_value is not None
            else family in {"clean", "control"} and verdict == "tie"
        ),
        probability_a=_optional_float(
            _first(row, "label_prob_A", "probability_a", ("raw_prompt_logprobs", "A"))
        ),
        probability_b=_optional_float(
            _first(row, "label_prob_B", "probability_b", ("raw_prompt_logprobs", "B"))
        ),
        probability_tie=_optional_float(
            _first(
                row,
                "label_prob_tie",
                "label_prob_T",
                "probability_tie",
                ("raw_prompt_logprobs", "tie"),
                ("raw_prompt_logprobs", "T"),
            )
        ),
        entropy=_optional_float(_first(row, "entropy", ("uncertainty", "logit", "entropy"))),
        normalized_entropy=_optional_float(
            _first(row, "normalized_entropy", ("uncertainty", "logit", "normalized_entropy"))
        ),
        msp=_optional_float(_first(row, "msp", ("uncertainty", "logit", "msp"))),
        margin=_optional_float(_first(row, "margin", ("uncertainty", "logit", "margin"))),
        verbalized_confidence=_optional_float(
            _first(
                row,
                "verbalized_confidence",
                ("uncertainty", "verbalized", "confidence"),
            )
        ),
        consistency_entropy=_optional_float(consistency_entropy),
        consistency_agreement=_optional_float(consistency_agreement),
    )


@dataclass(frozen=True, slots=True)
class PairedCondition:
    clean: ConditionRecord
    cued: ConditionRecord


@dataclass(frozen=True, slots=True)
class PairingResult:
    pairs: tuple[PairedCondition, ...]
    unmatched_cued_record_ids: tuple[str, ...]
    unused_clean_record_ids: tuple[str, ...]


def pair_clean_and_cued(
    clean_records: Sequence[ConditionRecord],
    cued_records: Sequence[ConditionRecord],
) -> PairingResult:
    by_id: dict[str, ConditionRecord] = {}
    by_key: dict[tuple[str, str, str], ConditionRecord] = {}
    for record in clean_records:
        if record.record_id:
            if record.record_id in by_id:
                raise ValueError(f"duplicate clean record_id: {record.record_id}")
            by_id[record.record_id] = record
        if record.key in by_key:
            raise ValueError(f"duplicate clean pairing key: {record.key}")
        by_key[record.key] = record

    pairs: list[PairedCondition] = []
    matched: set[str] = set()
    unmatched: list[str] = []
    for cued in cued_records:
        clean = by_id.get(cued.clean_record_id or "") or by_key.get(cued.key)
        if clean is None:
            unmatched.append(cued.record_id)
            continue
        pairs.append(PairedCondition(clean=clean, cued=cued))
        matched.add(clean.record_id or repr(clean.key))

    unused = [
        record.record_id
        for record in clean_records
        if (record.record_id or repr(record.key)) not in matched
    ]
    return PairingResult(
        pairs=tuple(pairs),
        unmatched_cued_record_ids=tuple(unmatched),
        unused_clean_record_ids=tuple(unused),
    )
