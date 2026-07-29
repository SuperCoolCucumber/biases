from __future__ import annotations

import json
from typing import Any

import pandas as pd


CONVERSATION_COLUMNS: tuple[str, str] = ("conversation_a", "conversation_b")
MTBENCH_HUMAN_REVISION = "f7d2896d2cc5d80f8b55c2bbc722613555233c25"


def canonical_conversation_json(value: Any) -> str:
    """Serialize a structured conversation as deterministic JSON."""

    if not isinstance(value, (list, dict)):
        raise TypeError(
            "conversation values must be list or dict structures before CSV serialization"
        )
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def serialize_conversation_columns(
    frame: pd.DataFrame,
    *,
    columns: tuple[str, ...] = CONVERSATION_COLUMNS,
) -> pd.DataFrame:
    """Return a copy with conversation structures encoded as canonical JSON."""

    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"frame is missing conversation columns: {missing}")

    serialized = frame.copy()
    for column in columns:
        serialized[column] = serialized[column].map(canonical_conversation_json)
    return serialized


__all__ = [
    "CONVERSATION_COLUMNS",
    "MTBENCH_HUMAN_REVISION",
    "canonical_conversation_json",
    "serialize_conversation_columns",
]
