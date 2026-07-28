from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


STANDARD_COLUMNS = [
    "example_id",
    "source_dataset",
    "source_id",
    "prompt",
    "response_a",
    "response_b",
    "winner",
    "model_a",
    "model_b",
    "assignment",
]


@dataclass(frozen=True)
class StandardPair:
    example_id: str
    source_dataset: str
    source_id: str
    prompt: str
    response_a: str
    response_b: str
    winner: str
    model_a: str | None = None
    model_b: str | None = None
    assignment: str = "original"

    def to_row(self) -> dict[str, str]:
        return {
            "example_id": self.example_id,
            "source_dataset": self.source_dataset,
            "source_id": self.source_id,
            "prompt": self.prompt,
            "response_a": self.response_a,
            "response_b": self.response_b,
            "winner": self.winner,
            "model_a": self.model_a or "",
            "model_b": self.model_b or "",
            "assignment": self.assignment,
        }


def write_standard_pairs(path: Path, pairs: Iterable[StandardPair]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=STANDARD_COLUMNS)
        writer.writeheader()
        for pair in pairs:
            writer.writerow(pair.to_row())


def randomized_pair(
    *,
    source_dataset: str,
    source_id: str,
    prompt: str,
    chosen: str,
    rejected: str,
    seed: int,
    model_chosen: str | None = None,
    model_rejected: str | None = None,
) -> StandardPair:
    rng = random.Random(f"{seed}:{source_dataset}:{source_id}")
    chosen_goes_to_a = rng.random() < 0.5
    if chosen_goes_to_a:
        return StandardPair(
            example_id=f"{source_dataset}:{source_id}",
            source_dataset=source_dataset,
            source_id=source_id,
            prompt=prompt,
            response_a=chosen,
            response_b=rejected,
            winner="model_a",
            model_a=model_chosen,
            model_b=model_rejected,
            assignment="chosen_to_a",
        )
    return StandardPair(
        example_id=f"{source_dataset}:{source_id}",
        source_dataset=source_dataset,
        source_id=source_id,
        prompt=prompt,
        response_a=rejected,
        response_b=chosen,
        winner="model_b",
        model_a=model_rejected,
        model_b=model_chosen,
        assignment="chosen_to_b",
    )


def adapt_chatbot_arena_rows(rows: Iterable[dict[str, Any]], *, seed: int = 42) -> list[StandardPair]:
    pairs: list[StandardPair] = []
    for index, row in enumerate(rows):
        winner = str(row.get("winner", "")).lower()
        if winner not in {"model_a", "model_b", "tie"}:
            continue
        conversation_a = _conversation_to_prompt_response(row.get("conversation_a"))
        conversation_b = _conversation_to_prompt_response(row.get("conversation_b"))
        prompt = conversation_a[0] or conversation_b[0]
        response_a = conversation_a[1]
        response_b = conversation_b[1]
        if not prompt or not response_a or not response_b:
            continue
        source_id = str(row.get("question_id") or row.get("conversation_id") or index)
        if winner == "tie":
            pairs.append(
                StandardPair(
                    example_id=f"chatbot_arena:{source_id}",
                    source_dataset="chatbot_arena",
                    source_id=source_id,
                    prompt=prompt,
                    response_a=response_a,
                    response_b=response_b,
                    winner="tie",
                    model_a=_optional_str(row.get("model_a")),
                    model_b=_optional_str(row.get("model_b")),
                    assignment="original_tie",
                )
            )
            continue
        chosen = response_a if winner == "model_a" else response_b
        rejected = response_b if winner == "model_a" else response_a
        model_chosen = row.get("model_a") if winner == "model_a" else row.get("model_b")
        model_rejected = row.get("model_b") if winner == "model_a" else row.get("model_a")
        pairs.append(
            randomized_pair(
                source_dataset="chatbot_arena",
                source_id=source_id,
                prompt=prompt,
                chosen=chosen,
                rejected=rejected,
                model_chosen=_optional_str(model_chosen),
                model_rejected=_optional_str(model_rejected),
                seed=seed,
            )
        )
    return pairs


def adapt_rewardbench_rows(rows: Iterable[dict[str, Any]], *, seed: int = 42) -> list[StandardPair]:
    pairs: list[StandardPair] = []
    for index, row in enumerate(rows):
        prompt = _first_present(row, ("prompt", "question", "instruction"))
        chosen = _first_present(row, ("chosen", "response_chosen", "chosen_response"))
        rejected = _first_present(row, ("rejected", "response_rejected", "rejected_response"))
        if not prompt or not chosen or not rejected:
            continue
        source_id = str(row.get("id") or row.get("example_id") or index)
        pairs.append(
            randomized_pair(
                source_dataset="rewardbench",
                source_id=source_id,
                prompt=prompt,
                chosen=chosen,
                rejected=rejected,
                seed=seed,
            )
        )
    return pairs


def adapt_llmbar_rows(rows: Iterable[dict[str, Any]], *, seed: int = 42) -> list[StandardPair]:
    pairs: list[StandardPair] = []
    for index, row in enumerate(rows):
        prompt = _first_present(row, ("input", "prompt", "question", "instruction"))
        output_1 = _first_present(row, ("output_1", "answer_1", "response_1"))
        output_2 = _first_present(row, ("output_2", "answer_2", "response_2"))
        label = str(row.get("label") or row.get("winner") or "").strip().lower()
        if not prompt or not output_1 or not output_2 or label not in {"1", "2", "output_1", "output_2"}:
            continue
        chosen = output_1 if label in {"1", "output_1"} else output_2
        rejected = output_2 if label in {"1", "output_1"} else output_1
        source_id = str(row.get("id") or row.get("example_id") or index)
        pairs.append(
            randomized_pair(
                source_dataset="llmbar",
                source_id=source_id,
                prompt=prompt,
                chosen=chosen,
                rejected=rejected,
                seed=seed,
            )
        )
    return pairs


def adapt_mtbench_rows(rows: Iterable[dict[str, Any]]) -> list[StandardPair]:
    pairs: list[StandardPair] = []
    for index, row in enumerate(rows):
        conversation_a = _conversation_to_prompt_response(row.get("conversation_a"))
        conversation_b = _conversation_to_prompt_response(row.get("conversation_b"))
        prompt = conversation_a[0] or conversation_b[0]
        response_a = conversation_a[1]
        response_b = conversation_b[1]
        winner = str(row.get("winner", "")).lower()
        if winner not in {"model_a", "model_b", "tie"} or not prompt or not response_a or not response_b:
            continue
        source_id = str(row.get("question_id") or index)
        turn = str(row.get("turn") or "")
        if turn:
            source_id = f"{source_id}:turn-{turn}"
        pairs.append(
            StandardPair(
                example_id=f"mtbench:{source_id}",
                source_dataset="mtbench",
                source_id=source_id,
                prompt=prompt,
                response_a=response_a,
                response_b=response_b,
                winner=winner,
                model_a=_optional_str(row.get("model_a")),
                model_b=_optional_str(row.get("model_b")),
                assignment="original",
            )
        )
    return pairs


def _conversation_to_prompt_response(value: Any) -> tuple[str, str]:
    if value is None:
        return "", ""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return value, ""
    if isinstance(value, dict):
        for key in ("messages", "conversation", "turns"):
            if key in value:
                value = value[key]
                break
    if not isinstance(value, list) or not value:
        return "", ""
    prompt = ""
    response = ""
    for item in value:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", item.get("speaker", ""))).lower()
        content = str(item.get("content", item.get("text", ""))).strip()
        if role == "user" and not prompt:
            prompt = content
        if role == "assistant":
            response = content
    if not prompt and value:
        first = value[0]
        if isinstance(first, dict):
            prompt = str(first.get("content", first.get("text", ""))).strip()
    return prompt, response


def _first_present(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
