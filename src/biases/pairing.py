from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from biases.schemas import HumanCueDirection, PairOrdering, VerdictLabel


def _enum_value(value: str | Enum | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


def _required_text(name: str, value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_pair_identity_key(
    *,
    dataset_name: str,
    input_file_hash: str,
    source_row_index: int | str,
    question_id: int | str,
    turn: int | str | None = None,
    response_a_id: str | None = None,
    response_b_id: str | None = None,
) -> str:
    """Build an order-independent identity for one source answer pair.

    ``input_file_hash`` plus ``source_row_index`` prevents collisions when a
    dataset contains repeated judgments for the same question and turn. The
    additional source fields make the identity auditable without weakening
    that uniqueness guarantee.
    """

    payload = {
        "dataset_name": _required_text("dataset_name", dataset_name),
        "input_file_hash": _required_text("input_file_hash", input_file_hash),
        "source_row_index": str(source_row_index),
        "question_id": str(question_id),
        "turn": None if turn is None else str(turn),
        "response_a_id": response_a_id,
        "response_b_id": response_b_id,
    }
    return f"pairbase_{canonical_sha256(payload)}"


def make_pair_key(
    *,
    pair_identity_key: str,
    model_name: str,
    ordering: PairOrdering | str,
) -> str:
    """Build the key shared by clean and cued records in one ordering."""

    normalized_ordering = normalize_ordering(ordering)
    payload = {
        "pair_identity_key": _required_text("pair_identity_key", pair_identity_key),
        "model_name": _required_text("model_name", model_name),
        "ordering": normalized_ordering.value,
    }
    return f"pair_{canonical_sha256(payload)}"


def make_condition_group_id(
    *,
    pair_identity_key: str,
    model_name: str,
    family: str | Enum,
    direction: str | Enum | None = None,
    dose: int | None = None,
) -> str:
    """Build an ID shared by the AB/BA twins of one condition."""

    if dose is not None and dose < 0:
        raise ValueError("dose must be non-negative")
    payload = {
        "pair_identity_key": _required_text("pair_identity_key", pair_identity_key),
        "model_name": _required_text("model_name", model_name),
        "family": _required_text("family", _enum_value(family) or ""),
        "direction": _enum_value(direction),
        "dose": dose,
    }
    return f"condition_{canonical_sha256(payload)}"


def make_ordering_twin_key(
    *,
    pair_identity_key: str,
    model_name: str,
    ordering: PairOrdering | str,
) -> str:
    return make_pair_key(
        pair_identity_key=pair_identity_key,
        model_name=model_name,
        ordering=opposite_ordering(ordering),
    )


def normalize_ordering(ordering: PairOrdering | str) -> PairOrdering:
    if isinstance(ordering, PairOrdering):
        return ordering
    normalized = str(ordering).strip().lower()
    aliases = {
        "ab": PairOrdering.AB,
        "original": PairOrdering.AB,
        "ba": PairOrdering.BA,
        "swapped": PairOrdering.BA,
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported ordering {ordering!r}; expected AB or BA") from exc


def opposite_ordering(ordering: PairOrdering | str) -> PairOrdering:
    normalized = normalize_ordering(ordering)
    return PairOrdering.BA if normalized == PairOrdering.AB else PairOrdering.AB


def opposite_binary_label(label: VerdictLabel | str) -> VerdictLabel:
    normalized = normalize_verdict(label)
    if normalized == VerdictLabel.A:
        return VerdictLabel.B
    if normalized == VerdictLabel.B:
        return VerdictLabel.A
    raise ValueError(f"Expected a non-tie A/B label, got {label!r}")


def swap_display_label(label: VerdictLabel | str) -> VerdictLabel:
    normalized = normalize_verdict(label)
    if normalized == VerdictLabel.A:
        return VerdictLabel.B
    if normalized == VerdictLabel.B:
        return VerdictLabel.A
    return normalized


def normalize_verdict(label: VerdictLabel | str) -> VerdictLabel:
    if isinstance(label, VerdictLabel):
        return label
    normalized = str(label).strip()
    aliases = {
        "A": VerdictLabel.A,
        "a": VerdictLabel.A,
        "B": VerdictLabel.B,
        "b": VerdictLabel.B,
        "T": VerdictLabel.TIE,
        "t": VerdictLabel.TIE,
        "tie": VerdictLabel.TIE,
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported verdict label {label!r}") from exc


def direction_relative_to_human(
    *,
    cue_target: VerdictLabel | str,
    human_winner: VerdictLabel | str,
) -> HumanCueDirection:
    target = normalize_verdict(cue_target)
    human = normalize_verdict(human_winner)
    if human == VerdictLabel.TIE:
        return HumanCueDirection.HUMAN_TIE
    if target == VerdictLabel.TIE:
        return HumanCueDirection.NONE
    if target == human:
        return HumanCueDirection.TOWARD_HUMAN
    return HumanCueDirection.AGAINST_HUMAN


__all__ = [
    "canonical_sha256",
    "direction_relative_to_human",
    "file_sha256",
    "make_condition_group_id",
    "make_ordering_twin_key",
    "make_pair_identity_key",
    "make_pair_key",
    "normalize_ordering",
    "normalize_verdict",
    "opposite_binary_label",
    "opposite_ordering",
    "swap_display_label",
]
