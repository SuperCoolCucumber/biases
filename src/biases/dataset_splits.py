from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd


ROUTING_SPLITS = frozenset({"calibration", "test"})


def _validate_calibration_fraction(calibration_fraction: float) -> None:
    if not 0 < calibration_fraction < 1:
        raise ValueError("calibration_fraction must be between 0 and 1")


def _normalized_group_ids(frame: pd.DataFrame, group_column: str) -> pd.Series:
    if group_column not in frame.columns:
        raise ValueError(f"frame must contain a {group_column} column")

    normalized: list[str] = []
    for row_index, value in frame[group_column].items():
        if pd.isna(value):
            raise ValueError(
                f"{group_column} must not contain missing values; "
                f"row {row_index!r} is missing"
            )
        text = str(value).strip()
        if not text:
            raise ValueError(
                f"{group_column} must not contain blank values; "
                f"row {row_index!r} is blank"
            )
        normalized.append(text)
    return pd.Series(normalized, index=frame.index, dtype="object")


def assign_routing_split(
    frame: pd.DataFrame,
    *,
    calibration_fraction: float,
    seed: int,
) -> pd.DataFrame:
    """Assign the deterministic winner-stratified calibration/test split.

    This is shared by the full and pilot MT-Bench preparation paths so a row
    receives the same routing assignment regardless of which artifact is
    generated.
    """

    _validate_calibration_fraction(calibration_fraction)
    if "winner" not in frame.columns:
        raise ValueError("frame must contain a winner column")

    pieces: list[pd.DataFrame] = []
    for _, group in frame.groupby("winner", dropna=False, group_keys=False):
        shuffled = group.sample(frac=1, random_state=seed).copy()
        calibration_size = round(len(shuffled) * calibration_fraction)
        calibration_size = (
            min(max(calibration_size, 1), len(shuffled) - 1)
            if len(shuffled) > 1
            else len(shuffled)
        )
        shuffled["routing_split"] = "test"
        shuffled.iloc[
            :calibration_size,
            shuffled.columns.get_loc("routing_split"),
        ] = "calibration"
        pieces.append(shuffled)

    if not pieces:
        return frame.assign(routing_split=pd.Series(dtype="object"))
    return (
        pd.concat(pieces, ignore_index=True)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )


def assign_question_disjoint_routing_split(
    frame: pd.DataFrame,
    *,
    calibration_fraction: float,
    seed: int,
    question_column: str = "question_id",
) -> pd.DataFrame:
    """Assign a deterministic calibration/test split by source question.

    Questions are ranked by a seeded SHA-256 digest rather than by input row
    order or a process-local hash. Every row carrying one question identifier
    therefore receives the same assignment, including different turns and
    repeated pairwise judgments. The function intentionally does not stratify
    individual rows by winner because doing so would break question
    disjointness.
    """

    _validate_calibration_fraction(calibration_fraction)
    question_ids = _normalized_group_ids(frame, question_column)
    unique_questions = sorted(set(question_ids))
    if len(unique_questions) < 2:
        raise ValueError(
            "question-disjoint routing requires at least two unique questions"
        )

    def seeded_digest(question_id: str) -> str:
        return hashlib.sha256(
            f"{seed}\0{question_id}".encode("utf-8")
        ).hexdigest()

    ranked_questions = sorted(
        unique_questions,
        key=lambda question_id: (seeded_digest(question_id), question_id),
    )
    calibration_size = round(
        len(ranked_questions) * calibration_fraction
    )
    calibration_size = min(
        max(calibration_size, 1),
        len(ranked_questions) - 1,
    )
    calibration_questions = set(ranked_questions[:calibration_size])

    result = frame.copy()
    result["routing_split"] = question_ids.map(
        lambda question_id: (
            "calibration"
            if question_id in calibration_questions
            else "test"
        )
    )
    return result


def routing_assignment_sha256(
    frame: pd.DataFrame,
    *,
    question_column: str = "question_id",
    routing_unit: str = "question",
) -> str:
    """Hash a canonical question- or row-level routing assignment."""

    question_ids = _normalized_group_ids(frame, question_column)
    if "routing_split" not in frame.columns:
        raise ValueError("frame must contain a routing_split column")

    normalized_splits = frame["routing_split"].astype(str).str.strip().str.lower()
    if not normalized_splits.isin(ROUTING_SPLITS).all():
        raise ValueError(
            "routing_split must contain only 'calibration' or 'test'"
        )
    if routing_unit not in {"row", "question"}:
        raise ValueError("routing_unit must be either 'row' or 'question'")

    if routing_unit == "row":
        canonical = frame.copy()
        canonical[question_column] = question_ids
        canonical["routing_split"] = normalized_splits
        columns = sorted(canonical.columns)
        payload = [
            {
                column: (
                    None
                    if pd.isna(value)
                    else str(value)
                )
                for column, value in zip(columns, values)
            }
            for values in canonical[columns].itertuples(index=False, name=None)
        ]
        payload.sort(
            key=lambda row: json.dumps(
                row,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        return hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

    assignment: dict[str, str] = {}
    for question_id, raw_split in zip(
        question_ids,
        normalized_splits,
    ):
        routing_split = str(raw_split)
        previous = assignment.setdefault(question_id, routing_split)
        if previous != routing_split:
            raise ValueError(
                f"question {question_id!r} occurs in both routing splits"
            )

    payload = [
        {"question_id": question_id, "routing_split": assignment[question_id]}
        for question_id in sorted(assignment)
    ]
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def routing_manifest(
    frame: pd.DataFrame,
    *,
    routing_unit: str,
    seed: int,
    calibration_fraction: float,
    question_column: str = "question_id",
) -> dict[str, Any]:
    """Build deterministic provenance for a routing assignment."""

    _validate_calibration_fraction(calibration_fraction)
    assignment_sha256 = routing_assignment_sha256(
        frame,
        question_column=question_column,
        routing_unit=routing_unit,
    )
    question_ids = _normalized_group_ids(frame, question_column)
    assignment = pd.DataFrame(
        {
            "question_id": question_ids,
            "routing_split": frame["routing_split"].astype(str).str.strip().str.lower(),
        },
        index=frame.index,
    ).drop_duplicates()
    question_counts = assignment["routing_split"].value_counts().to_dict()
    question_overlap = int(
        assignment.groupby("question_id")["routing_split"]
        .nunique()
        .gt(1)
        .sum()
    )
    row_counts = (
        frame["routing_split"]
        .astype(str)
        .str.strip()
        .str.lower()
        .value_counts()
        .to_dict()
    )
    return {
        "schema_version": 1,
        "routing_unit": routing_unit,
        "seed": seed,
        "calibration_fraction": calibration_fraction,
        "row_counts": {
            "total": len(frame),
            "calibration": int(row_counts.get("calibration", 0)),
            "test": int(row_counts.get("test", 0)),
        },
        "question_counts": {
            "total": int(assignment["question_id"].nunique()),
            "calibration": int(question_counts.get("calibration", 0)),
            "test": int(question_counts.get("test", 0)),
            "overlap": question_overlap,
        },
        "routing_assignment_sha256": assignment_sha256,
    }
