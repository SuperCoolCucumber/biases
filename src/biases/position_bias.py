from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from biases.models import get_model_profile
from biases.paths import configure_artifact_environment, data_path
from biases.position_prompts import build_position_prompt_package
from biases.schemas import (
    BiasCondition,
    BiasType,
    Candidate,
    ConsistencyMetrics,
    ExperimentSpec,
    JudgeExample,
    LogitMetrics,
    OutputMode,
    PromptPackage,
    RunRecord,
    UncertaintyBundle,
    VerdictLabel,
    VerbalizedMetrics,
)
from biases.utils import ensure_parent, stable_hash, write_jsonl

configure_artifact_environment()

try:
    from vllm import LLM, SamplingParams
except ImportError:  # pragma: no cover
    LLM = None
    SamplingParams = None


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
DEFAULT_DATA_PATH = data_path("processed", "mtbench_full.csv")
DEFAULT_MAX_MODEL_LEN = 8192
JUDGE_OUTPUT_PARSER_VERSION = "strict_v3"
VERBALIZED_OUTPUT_PARSER_VERSION = "strict_v3"
CONSTRAINED_LOGPROBS_MODE = "processed_logprobs"
POSITION_PAIR_ELIGIBILITY_CONTRACT = "position_pair_loader_v1"
VerbalizedParseStatus = Literal[
    "parsed",
    "missing",
    "unparseable",
    "not_requested",
]
UNCERTAINTY_METHODS = [
    "logit",
    "verbalized_confidence",
    "consistency",
    "consistency_entropy",
]


@dataclass(slots=True)
class PositionPair:
    pair_id: str
    original: JudgeExample
    swapped: JudgeExample


@dataclass(frozen=True, slots=True)
class PositionPairEligibilityAudit:
    """Deterministic account of source rows accepted by the pair loader."""

    raw_row_count: int
    eligible_pair_count: int
    skipped_row_count: int
    skipped_reason_counts: dict[str, int]
    routing_counts: dict[str, dict[str, int]]
    eligibility_sha256: str
    skipped_rows: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "eligibility_contract": POSITION_PAIR_ELIGIBILITY_CONTRACT,
            "raw_row_count": self.raw_row_count,
            "eligible_pair_count": self.eligible_pair_count,
            "skipped_row_count": self.skipped_row_count,
            "skipped_reason_counts": dict(self.skipped_reason_counts),
            "routing_counts": {
                name: dict(counts)
                for name, counts in self.routing_counts.items()
            },
            "eligibility_sha256": self.eligibility_sha256,
            "skipped_rows": [dict(row) for row in self.skipped_rows],
        }


@dataclass(frozen=True, slots=True)
class ConversationExtraction:
    prompt_messages: list[dict[str, str]]
    response_a: str
    response_b: str
    mode: str
    selected_turn: int | None


def _canonicalize(name: str) -> str:
    return "".join(char.lower() for char in name if char.isalnum())


def _find_column(fieldnames: list[str], aliases: tuple[str, ...]) -> str:
    normalized = {_canonicalize(name): name for name in fieldnames}
    for alias in aliases:
        matched = normalized.get(_canonicalize(alias))
        if matched is not None:
            return matched
    raise KeyError(f"Could not find any of {aliases!r} in CSV header {fieldnames!r}")


def _parse_prompt_messages(raw_prompt: str) -> list[dict[str, str]]:
    candidate = raw_prompt.strip()
    if not candidate:
        return [{"role": "user", "content": ""}]

    if candidate.startswith("[") or candidate.startswith("{"):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            messages: list[dict[str, str]] = []
            for item in parsed:
                if isinstance(item, dict) and "content" in item:
                    messages.append(
                        {
                            "role": str(item.get("role", "user")),
                            "content": str(item.get("content", "")),
                        }
                    )
            if messages:
                return messages

    return [{"role": "user", "content": candidate}]


def _parse_conversation(raw_text: str) -> list[dict[str, str]]:
    text = raw_text.strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = None

    if isinstance(parsed, dict):
        for key in ("messages", "conversation", "turns"):
            value = parsed.get(key)
            if isinstance(value, list):
                parsed = value
                break

    if isinstance(parsed, list):
        messages: list[dict[str, str]] = []
        for item in parsed:
            if isinstance(item, dict):
                if "content" in item:
                    messages.append(
                        {
                            "role": str(item.get("role", item.get("speaker", "user"))),
                            "content": str(item.get("content", "")),
                        }
                    )
                elif "text" in item:
                    messages.append(
                        {
                            "role": str(item.get("role", item.get("speaker", "user"))),
                            "content": str(item.get("text", "")),
                        }
                    )
            elif isinstance(item, str):
                messages.append({"role": "user", "content": item})
        if messages:
            return messages

    return [{"role": "user", "content": text}]


def _message_signature(message: dict[str, str]) -> tuple[str, str]:
    return (
        str(message.get("role", "user")).strip().lower(),
        str(message.get("content", "")).strip(),
    )


def _shared_prefix_messages(
    conversation_a: list[dict[str, str]],
    conversation_b: list[dict[str, str]],
) -> list[dict[str, str]]:
    shared: list[dict[str, str]] = []
    for message_a, message_b in zip(conversation_a, conversation_b):
        if _message_signature(message_a) != _message_signature(message_b):
            break
        shared.append(
            {
                "role": str(message_a.get("role", "user")),
                "content": str(message_a.get("content", "")),
            }
        )
    return shared


def _extract_final_response(conversation: list[dict[str, str]]) -> str:
    for message in reversed(conversation):
        role = str(message.get("role", "")).strip().lower()
        if role == "assistant":
            return str(message.get("content", "")).strip()
    if conversation:
        return str(conversation[-1].get("content", "")).strip()
    return ""


def _role_contents(
    conversation: list[dict[str, str]],
    role: str,
) -> list[str]:
    return [
        str(message.get("content", "")).strip()
        for message in conversation
        if str(message.get("role", "")).strip().lower() == role
    ]


def _render_two_turn_candidate(first: str, second: str) -> str:
    return (
        f"Turn 1 assistant response (context):\n{first}\n\n"
        f"Turn 2 assistant response (evaluate this response):\n{second}"
    )


def _extract_conversation_pair(
    conversation_a: list[dict[str, str]],
    conversation_b: list[dict[str, str]],
    *,
    turn: str,
) -> ConversationExtraction:
    """Select the MT-Bench target turn without discarding its prior context."""

    selected_turn = int(turn) if turn in {"1", "2"} else None
    if selected_turn is not None:
        users_a = _role_contents(conversation_a, "user")
        users_b = _role_contents(conversation_b, "user")
        assistants_a = _role_contents(conversation_a, "assistant")
        assistants_b = _role_contents(conversation_b, "assistant")
        enough_messages = all(
            len(messages) >= selected_turn
            for messages in (users_a, users_b, assistants_a, assistants_b)
        )
        shared_questions = (
            enough_messages
            and users_a[:selected_turn] == users_b[:selected_turn]
            and all(users_a[:selected_turn])
        )
        if shared_questions and selected_turn == 1:
            return ConversationExtraction(
                prompt_messages=[
                    {"role": "user", "content": users_a[0]},
                ],
                response_a=assistants_a[0],
                response_b=assistants_b[0],
                mode="mtbench_turn_1",
                selected_turn=1,
            )
        if shared_questions and selected_turn == 2:
            response_a = (
                _render_two_turn_candidate(
                    assistants_a[0],
                    assistants_a[1],
                )
                if all(assistants_a[:2])
                else ""
            )
            response_b = (
                _render_two_turn_candidate(
                    assistants_b[0],
                    assistants_b[1],
                )
                if all(assistants_b[:2])
                else ""
            )
            return ConversationExtraction(
                prompt_messages=[
                    {
                        "role": "user",
                        "content": f"Turn 1 user question (context):\n{users_a[0]}",
                    },
                    {
                        "role": "user",
                        "content": (
                            "Turn 2 user question (evaluate the response to this "
                            f"question):\n{users_a[1]}"
                        ),
                    },
                ],
                response_a=response_a,
                response_b=response_b,
                mode="mtbench_turn_2",
                selected_turn=2,
            )

    prompt_messages = _shared_prefix_messages(conversation_a, conversation_b)
    if not prompt_messages and conversation_a:
        prompt_messages = conversation_a[:-1] or conversation_a
    return ConversationExtraction(
        prompt_messages=prompt_messages,
        response_a=_extract_final_response(conversation_a),
        response_b=_extract_final_response(conversation_b),
        mode="legacy_final_response",
        selected_turn=selected_turn,
    )


def _normalize_winner(raw_winner: str) -> VerdictLabel | None:
    winner = raw_winner.strip().lower()
    mapping = {
        "a": VerdictLabel.A,
        "model_a": VerdictLabel.A,
        "response_a": VerdictLabel.A,
        "1": VerdictLabel.A,
        "left": VerdictLabel.A,
        "b": VerdictLabel.B,
        "model_b": VerdictLabel.B,
        "response_b": VerdictLabel.B,
        "2": VerdictLabel.B,
        "right": VerdictLabel.B,
        "tie": VerdictLabel.TIE,
        "equal": VerdictLabel.TIE,
    }
    return mapping.get(winner)


def _label_to_str(label: str | VerdictLabel | None) -> str | None:
    if label is None:
        return None
    if isinstance(label, VerdictLabel):
        return label.value
    return str(label)


def _swap_label(label: VerdictLabel | None) -> VerdictLabel | None:
    if label == VerdictLabel.A:
        return VerdictLabel.B
    if label == VerdictLabel.B:
        return VerdictLabel.A
    return label


def _eligibility_sha256(rows: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        rows,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _routing_bucket(raw_routing_split: str | None) -> str:
    normalized = str(raw_routing_split or "").strip().lower()
    return normalized or "missing"


def load_position_pairs_with_eligibility(
    csv_path: Path,
    limit: int | None = None,
) -> tuple[list[PositionPair], PositionPairEligibilityAudit]:
    """Load eligible pairs and report every examined row's eligibility.

    The filtering contract is intentionally the same as ``load_position_pairs``:
    a row is skipped only when its winner is missing/unsupported or either
    extracted response is empty. With ``limit`` set, counts cover only the rows
    examined while collecting that many eligible pairs.
    """

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []

        id_column = _find_column(fieldnames, ("example_id", "id", "pair_id", "question_id"))
        winner_column = _find_column(
            fieldnames,
            ("winner", "human_winner", "label", "preferred", "preference"),
        )
        turn_column = next(
            (
                field
                for field in fieldnames
                if _canonicalize(field) in {_canonicalize(name) for name in ("turn", "turn_id")}
            ),
            None,
        )
        model_a_column = next(
            (
                field
                for field in fieldnames
                if _canonicalize(field)
                in {_canonicalize(name) for name in ("model_a", "model_a_name", "generator_a")}
            ),
            None,
        )
        model_b_column = next(
            (
                field
                for field in fieldnames
                if _canonicalize(field)
                in {_canonicalize(name) for name in ("model_b", "model_b_name", "generator_b")}
            ),
            None,
        )
        routing_split_column = next(
            (
                field
                for field in fieldnames
                if _canonicalize(field)
                in {_canonicalize(name) for name in ("routing_split", "split", "dataset_split")}
            ),
            None,
        )
        prompt_column = next(
            (
                field
                for field in fieldnames
                if _canonicalize(field)
                in {
                    _canonicalize(name)
                    for name in ("prompt", "question", "instruction", "user_prompt", "conversation")
                }
            ),
            None,
        )
        response_a_column = next(
            (
                field
                for field in fieldnames
                if _canonicalize(field)
                in {
                    _canonicalize(name)
                    for name in ("response_a", "answer_a", "model_a_output", "output_a", "assistant_a")
                }
            ),
            None,
        )
        response_b_column = next(
            (
                field
                for field in fieldnames
                if _canonicalize(field)
                in {
                    _canonicalize(name)
                    for name in ("response_b", "answer_b", "model_b_output", "output_b", "assistant_b")
                }
            ),
            None,
        )
        conversation_a_column = next(
            (
                field
                for field in fieldnames
                if _canonicalize(field)
                in {_canonicalize(name) for name in ("conversation_a", "messages_a", "chat_a")}
            ),
            None,
        )
        conversation_b_column = next(
            (
                field
                for field in fieldnames
                if _canonicalize(field)
                in {_canonicalize(name) for name in ("conversation_b", "messages_b", "chat_b")}
            ),
            None,
        )

        flat_schema_complete = all(
            column is not None
            for column in (prompt_column, response_a_column, response_b_column)
        )
        conversation_schema_complete = all(
            column is not None
            for column in (conversation_a_column, conversation_b_column)
        )
        if not flat_schema_complete and not conversation_schema_complete:
            raise KeyError(
                "CSV must contain either prompt/response_a/response_b columns or "
                "conversation_a/conversation_b columns."
            )

        pairs: list[PositionPair] = []
        eligibility_rows: list[dict[str, Any]] = []
        skipped_rows: list[dict[str, Any]] = []
        skipped_reason_counts: Counter[str] = Counter()
        raw_routing_counts: Counter[str] = Counter()
        eligible_routing_counts: Counter[str] = Counter()
        skipped_routing_counts: Counter[str] = Counter()
        for index, row in enumerate(reader):
            if limit is not None and len(pairs) >= limit:
                break

            base_id = row[id_column].strip() or f"row-{index}"
            turn_value = row[turn_column].strip() if turn_column and row.get(turn_column) else ""
            pair_id = f"{base_id}:turn-{turn_value}" if turn_value else base_id
            routing_split = (
                row[routing_split_column].strip()
                if routing_split_column and row.get(routing_split_column)
                else None
            )
            routing_bucket = _routing_bucket(routing_split)
            raw_routing_counts[routing_bucket] += 1

            if flat_schema_complete:
                assert prompt_column is not None
                assert response_a_column is not None
                assert response_b_column is not None
                prompt_messages = _parse_prompt_messages(row[prompt_column])
                response_a = row[response_a_column].strip()
                response_b = row[response_b_column].strip()
                extraction_mode = "flat_prompt_responses"
                selected_turn = int(turn_value) if turn_value in {"1", "2"} else None
            else:
                assert conversation_a_column is not None
                assert conversation_b_column is not None
                conversation_a = _parse_conversation(row[conversation_a_column])
                conversation_b = _parse_conversation(row[conversation_b_column])
                extraction = _extract_conversation_pair(
                    conversation_a,
                    conversation_b,
                    turn=turn_value,
                )
                prompt_messages = extraction.prompt_messages
                response_a = extraction.response_a
                response_b = extraction.response_b
                extraction_mode = extraction.mode
                selected_turn = extraction.selected_turn

            raw_winner = row[winner_column].strip()
            human_winner = _normalize_winner(raw_winner)
            skip_reasons: list[str] = []
            if not raw_winner:
                skip_reasons.append("missing_winner")
            elif human_winner is None:
                skip_reasons.append("invalid_winner")
            if not response_a:
                skip_reasons.append("missing_response_a")
            if not response_b:
                skip_reasons.append("missing_response_b")

            eligibility_entry = {
                "source_row_index": index,
                "pair_id": pair_id,
                "routing_split": routing_bucket,
                "eligible": not skip_reasons,
                "skip_reasons": list(skip_reasons),
            }
            eligibility_rows.append(eligibility_entry)
            if skip_reasons:
                skipped_reason_counts.update(skip_reasons)
                skipped_routing_counts[routing_bucket] += 1
                skipped_rows.append(
                    {
                        "source_row_index": index,
                        "pair_id": pair_id,
                        "routing_split": routing_bucket,
                        "skip_reasons": list(skip_reasons),
                    }
                )
                continue
            eligible_routing_counts[routing_bucket] += 1

            model_a = row[model_a_column].strip() if model_a_column and row.get(model_a_column) else None
            model_b = row[model_b_column].strip() if model_b_column and row.get(model_b_column) else None

            base_metadata = {
                "pair_id": pair_id,
                "question_cluster_id": base_id,
                "source_csv": str(csv_path),
                "source_row_index": index,
                "turn": turn_value or None,
                "selected_turn": selected_turn,
                "conversation_extraction_mode": extraction_mode,
                "routing_split": routing_split,
            }
            original = JudgeExample(
                example_id=f"{pair_id}:original",
                question_id=pair_id,
                prompt_messages=prompt_messages,
                candidates={
                    "A": Candidate(
                        label=VerdictLabel.A,
                        response=response_a,
                        model_id=model_a,
                        response_id=f"{pair_id}:response_a",
                    ),
                    "B": Candidate(
                        label=VerdictLabel.B,
                        response=response_b,
                        model_id=model_b,
                        response_id=f"{pair_id}:response_b",
                    ),
                },
                human_winner=human_winner,
                metadata={
                    **base_metadata,
                    "variant_id": "original",
                    "response_id_by_label": {
                        "A": f"{pair_id}:response_a",
                        "B": f"{pair_id}:response_b",
                    },
                },
            )
            swapped = JudgeExample(
                example_id=f"{pair_id}:swapped",
                question_id=pair_id,
                prompt_messages=prompt_messages,
                candidates={
                    "A": Candidate(
                        label=VerdictLabel.A,
                        response=response_b,
                        model_id=model_b,
                        response_id=f"{pair_id}:response_b",
                    ),
                    "B": Candidate(
                        label=VerdictLabel.B,
                        response=response_a,
                        model_id=model_a,
                        response_id=f"{pair_id}:response_a",
                    ),
                },
                human_winner=_swap_label(human_winner),
                metadata={
                    **base_metadata,
                    "variant_id": "swapped",
                    "response_id_by_label": {
                        "A": f"{pair_id}:response_b",
                        "B": f"{pair_id}:response_a",
                    },
                },
            )
            pairs.append(PositionPair(pair_id=pair_id, original=original, swapped=swapped))

    raw_row_count = len(eligibility_rows)
    skipped_row_count = len(skipped_rows)
    if raw_row_count != len(pairs) + skipped_row_count:
        raise AssertionError("pair-loader eligibility accounting is inconsistent")
    audit = PositionPairEligibilityAudit(
        raw_row_count=raw_row_count,
        eligible_pair_count=len(pairs),
        skipped_row_count=skipped_row_count,
        skipped_reason_counts=dict(sorted(skipped_reason_counts.items())),
        routing_counts={
            "raw_rows": dict(sorted(raw_routing_counts.items())),
            "eligible_pairs": dict(sorted(eligible_routing_counts.items())),
            "skipped_rows": dict(sorted(skipped_routing_counts.items())),
        },
        eligibility_sha256=_eligibility_sha256(eligibility_rows),
        skipped_rows=tuple(skipped_rows),
    )
    return pairs, audit


def load_position_pairs(csv_path: Path, limit: int | None = None) -> list[PositionPair]:
    pairs, _ = load_position_pairs_with_eligibility(csv_path, limit=limit)
    return pairs


def _softmax_from_log_scores(scores: dict[str, float], temperature: float) -> dict[str, float]:
    adjusted = dict(scores)
    if temperature > 0:
        adjusted = {label: score / temperature for label, score in adjusted.items()}
    max_score = max(adjusted.values())
    unnormalized = {label: math.exp(score - max_score) for label, score in adjusted.items()}
    total = sum(unnormalized.values())
    return {label: value / total for label, value in unnormalized.items()}


def _normalize_probs(probs: dict[str, float]) -> dict[str, float]:
    total = sum(probs.values())
    if total <= 0:
        raise ValueError("Probability mass must be positive")
    return {label: value / total for label, value in probs.items()}


def _parse_verdict_atom(text: str) -> VerdictLabel | None:
    bracketed = re.fullmatch(
        r"(?:\[\[\s*(A|B|T|TIE)\s*\]\]|\[\s*(A|B|T|TIE)\s*\])",
        text.strip(),
        re.IGNORECASE,
    )
    if bracketed is not None:
        normalized = next(
            group.upper() for group in bracketed.groups() if group is not None
        )
    else:
        bare = re.fullmatch(
            r"(A|B|T|TIE)\s*[.!]?",
            text.strip(),
            re.IGNORECASE,
        )
        if bare is None:
            return None
        normalized = bare.group(1).upper()
    return {
        "A": VerdictLabel.A,
        "B": VerdictLabel.B,
        "T": VerdictLabel.TIE,
        "TIE": VerdictLabel.TIE,
    }[normalized]


def _parse_verdict_line(text: str) -> VerdictLabel | None:
    explicit = re.fullmatch(
        r"(?:verdict|answer|response|choice|label)\s*[:=]\s*(.+)",
        text.strip(),
        re.IGNORECASE,
    )
    verdict_atom = explicit.group(1).strip() if explicit is not None else text
    return _parse_verdict_atom(verdict_atom)


def _parse_confidence_line(
    text: str,
    *,
    allow_bare: bool,
) -> float | None:
    explicit = re.fullmatch(
        r"confidence\s*[:=]\s*(\d+(?:\.\d+)?)\s*%?",
        text.strip(),
        re.IGNORECASE,
    )
    bare = (
        re.fullmatch(r"(\d+(?:\.\d+)?)\s*%?", text.strip())
        if allow_bare
        else None
    )
    match = explicit or bare
    if match is None:
        return None
    value = float(match.group(1))
    return value if 0.0 <= value <= 100.0 else None


def _has_trailing_verdict_or_confidence(lines: list[str]) -> bool:
    for line in lines:
        if _parse_verdict_line(line) is not None:
            return True
        if _parse_confidence_line(line, allow_bare=True) is not None:
            return True
        if re.match(r"^confidence\b", line, re.IGNORECASE) is not None:
            return True
    return False


_HERMES_VERBALIZED_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"Line 1: (?P<verdict>[ABT])\n"
        r"Line 2: (?:Confidence: )?"
        r"(?P<confidence>\d+(?:\.\d+)?)"
    ),
    re.compile(
        r"1\. (?P<verdict>[ABT])\n"
        r"2\. (?P<confidence>\d+(?:\.\d+)?)"
    ),
    re.compile(
        r"1: (?P<verdict>[ABT])\n"
        r"2: (?P<confidence>[0-9]+(?:\.[0-9]+)?)"
    ),
    re.compile(
        r"1\) (?P<verdict>[ABT])\n"
        r"2\) (?P<confidence>[0-9]+(?:\.[0-9]+)?)"
    ),
    re.compile(
        r"(?P<verdict>[ABT]), "
        r"(?P<confidence>\d+(?:\.\d+)?)"
    ),
)


def _parse_hermes_verbalized_output(
    text: str,
) -> tuple[VerdictLabel, float] | None:
    """Parse only the complete, observed Hermes verdict-confidence forms."""

    candidate = text.strip().replace("\r\n", "\n")
    for pattern in _HERMES_VERBALIZED_PATTERNS:
        match = pattern.fullmatch(candidate)
        if match is None:
            continue
        confidence = float(match.group("confidence"))
        if not 0.0 <= confidence <= 100.0:
            return None
        verdict = {
            "A": VerdictLabel.A,
            "B": VerdictLabel.B,
            "T": VerdictLabel.TIE,
        }[match.group("verdict")]
        return verdict, confidence
    return None


def parse_verbalized_output(
    text: str,
) -> tuple[VerdictLabel | None, float | None]:
    """Parse one atomic verdict-confidence pair from strict output forms."""

    hermes_result = _parse_hermes_verbalized_output(text)
    if hermes_result is not None:
        return hermes_result

    nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not nonempty_lines:
        return None, None

    if len(nonempty_lines) >= 2:
        verdict = _parse_verdict_line(nonempty_lines[0])
        confidence = _parse_confidence_line(
            nonempty_lines[1],
            allow_bare=True,
        )
        if (
            verdict is not None
            and confidence is not None
            and not _has_trailing_verdict_or_confidence(nonempty_lines[2:])
        ):
            return verdict, confidence

    single_line = re.fullmatch(
        r"(?P<verdict>"
        r"(?:A|B|T|TIE)\s*[.!]?|"
        r"\[\[\s*(?:A|B|T|TIE)\s*\]\]|"
        r"\[\s*(?:A|B|T|TIE)\s*\]"
        r")\s+"
        r"(?P<confidence>"
        r"(?:confidence\s*[:=]\s*)?\d+(?:\.\d+)?\s*%?"
        r")",
        nonempty_lines[0],
        re.IGNORECASE,
    )
    if single_line is None:
        return None, None
    verdict = _parse_verdict_atom(single_line.group("verdict"))
    confidence = _parse_confidence_line(
        single_line.group("confidence"),
        allow_bare=True,
    )
    if verdict is None or confidence is None:
        return None, None
    if _has_trailing_verdict_or_confidence(nonempty_lines[1:]):
        return None, None
    return verdict, confidence


def verbalized_parse_status(
    *,
    uncertainty_methods: list[str],
    raw_output: object,
) -> VerbalizedParseStatus:
    """Classify availability of the optional verbalized-confidence channel."""

    if "verbalized_confidence" not in uncertainty_methods:
        return "not_requested"
    if raw_output is None or raw_output == "":
        return "missing"
    if not isinstance(raw_output, str):
        return "unparseable"
    if not raw_output.strip():
        return "missing"
    verdict, confidence = parse_verbalized_output(raw_output)
    return (
        "parsed"
        if verdict is not None and confidence is not None
        else "unparseable"
    )


class VLLMJudge:
    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        max_model_len: int = DEFAULT_MAX_MODEL_LEN,
        gpu_memory_utilization: float = 0.9,
        dtype: str = "auto",
        max_num_batched_tokens: int | None = None,
        max_num_seqs: int | None = None,
        enforce_eager: bool | None = None,
        disable_custom_all_reduce: bool | None = None,
    ) -> None:
        if LLM is None or SamplingParams is None:
            raise RuntimeError(
                "vLLM execution requires the 'local' extra. Install with `uv sync --extra local`."
            )
        if max_num_batched_tokens is not None and max_num_batched_tokens < 1:
            raise ValueError("max_num_batched_tokens must be positive when provided")
        if max_num_seqs is not None and max_num_seqs < 1:
            raise ValueError("max_num_seqs must be positive when provided")

        self.profile = get_model_profile(model_name)
        if not self.profile.supports_text_prompt_transport:
            raise RuntimeError(
                f"Model {self.profile.registry_name!r} requires a token-ID "
                "prompt adapter; string chat-template transport is not "
                "validated for this tokenizer."
            )
        self.model_name = self.profile.hf_model_name
        self.require_label_logprobs = True
        self.logprobs_mode = CONSTRAINED_LOGPROBS_MODE
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_seqs = max_num_seqs
        # Explicit scientific-run controls take precedence.  The environment
        # fallback is retained for legacy callers that have not yet adopted
        # the frozen runtime contract.
        if disable_custom_all_reduce is None:
            disable_custom_all_reduce = os.environ.get(
                "VLLM_DISABLE_CUSTOM_ALL_REDUCE",
                "",
            ).lower() in {"1", "true", "yes", "on"}
        if enforce_eager is None:
            enforce_eager = os.environ.get(
                "BIASES_VLLM_ENFORCE_EAGER",
                "",
            ).lower() in {"1", "true", "yes", "on"}
        self.disable_custom_all_reduce = bool(disable_custom_all_reduce)
        self.enforce_eager = bool(enforce_eager)
        llm_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "revision": self.profile.revision,
            "tokenizer_revision": self.profile.revision,
            "tensor_parallel_size": tensor_parallel_size,
            "max_model_len": max_model_len,
            "trust_remote_code": self.profile.trust_remote_code,
            "gpu_memory_utilization": gpu_memory_utilization,
            "dtype": dtype,
            "disable_custom_all_reduce": self.disable_custom_all_reduce,
            "enforce_eager": self.enforce_eager,
            "logprobs_mode": self.logprobs_mode,
        }
        if max_num_batched_tokens is not None:
            llm_kwargs["max_num_batched_tokens"] = max_num_batched_tokens
        if max_num_seqs is not None:
            llm_kwargs["max_num_seqs"] = max_num_seqs
        self.model = LLM(**llm_kwargs)
        self.tokenizer = self._get_tokenizer()
        (
            self.decision_label_token_ids,
            self.decision_token_id_to_label,
        ) = self._build_decision_label_token_maps()

    def _prepare_prompt(self, prompt_text: str) -> str:
        profile = getattr(self, "profile", None)
        if profile is None:
            profile = get_model_profile(self.model_name)
        return profile.prepare_legacy_prompt(prompt_text)

    def render_messages(self, messages: list[dict[str, str]]) -> str:
        return self.profile.render_prompt(self.tokenizer, messages)

    def _get_tokenizer(self) -> Any:
        if hasattr(self.model, "get_tokenizer"):
            return self.model.get_tokenizer()
        engine = getattr(self.model, "llm_engine", None)
        tokenizer = getattr(engine, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("Could not access the vLLM tokenizer for label-token constraints.")
        return tokenizer

    def _encode_single_token(self, text: str) -> int | None:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) != 1:
            return None
        token_id = int(token_ids[0])
        decoded = self.tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if decoded != text:
            raise RuntimeError(
                f"Decision surface {text!r} encoded as singleton token ID "
                f"{token_id}, but that token decoded as {decoded!r}."
            )
        return token_id

    def _build_decision_label_token_maps(self) -> tuple[dict[str, list[int]], dict[int, str]]:
        label_texts = self.profile.verdict_token_texts
        label_to_ids: dict[str, list[int]] = {}
        token_id_to_label: dict[int, str] = {}
        for label, texts in label_texts.items():
            token_ids: list[int] = []
            for text in texts:
                token_id = self._encode_single_token(text)
                if token_id is None or token_id in token_ids:
                    continue
                existing_label = token_id_to_label.get(token_id)
                if existing_label is not None and existing_label != label:
                    raise RuntimeError(
                        f"Decision token ID {token_id} maps to both "
                        f"{existing_label!r} and {label!r}."
                    )
                token_ids.append(token_id)
                token_id_to_label[token_id] = label
            if not token_ids:
                raise RuntimeError(
                    f"Could not find a single-token encoding for decision label {label!r}."
                )
            label_to_ids[label] = token_ids
        return label_to_ids, token_id_to_label

    @property
    def decision_allowed_token_ids(self) -> list[int]:
        return sorted(
            {
                token_id
                for token_ids in self.decision_label_token_ids.values()
                for token_id in token_ids
            }
        )

    @staticmethod
    def _logprob_value(candidate: Any) -> float | None:
        if isinstance(candidate, dict):
            value = candidate.get("logprob")
        else:
            value = getattr(candidate, "logprob", None)
        return None if value is None else float(value)

    def _extract_label_probs(self, first_token_logprobs: Any | None) -> dict[str, float]:
        if not first_token_logprobs:
            raise RuntimeError(
                "vLLM did not return processed first-token log probabilities "
                "for the registered decision token IDs."
            )

        registered_logprobs: dict[int, tuple[str, float]] = {}
        for token_id, candidate in first_token_logprobs.items():
            try:
                normalized_token_id = int(token_id)
            except (TypeError, ValueError):
                continue
            label = self.decision_token_id_to_label.get(normalized_token_id)
            if label is None:
                continue
            logprob = self._logprob_value(candidate)
            if logprob is not None and math.isfinite(logprob):
                registered_logprobs[normalized_token_id] = (label, logprob)

        expected_token_ids = set(self.decision_token_id_to_label)
        missing_token_ids = sorted(expected_token_ids - set(registered_logprobs))
        if missing_token_ids:
            raise RuntimeError(
                "vLLM did not return processed first-token log probabilities "
                "for every registered decision token ID; missing token IDs: "
                f"{missing_token_ids!r}."
            )

        label_logprobs = list(registered_logprobs.values())
        max_logprob = max(logprob for _, logprob in label_logprobs)
        weights = {"A": 0.0, "B": 0.0, "tie": 0.0}
        for label, logprob in label_logprobs:
            weights[label] += math.exp(logprob - max_logprob)
        total = sum(weights.values())
        return {label: value / total for label, value in weights.items()}

    @staticmethod
    def _parse_verdict_text(text: str) -> VerdictLabel | None:
        first_line = next(
            (line.strip() for line in text.splitlines() if line.strip()),
            "",
        )
        return _parse_verdict_line(first_line)

    @staticmethod
    def _parse_confidence(text: str) -> float | None:
        return parse_verbalized_output(text)[1]

    @classmethod
    def _resolve_constrained_verdict(
        cls,
        *,
        raw_text: str,
        probabilities: dict[str, float],
        sampling_temperature: float,
    ) -> VerdictLabel:
        raw_verdict = cls._parse_verdict_text(raw_text)
        probability_verdict: VerdictLabel | None = None
        if probabilities:
            winner = max(probabilities, key=probabilities.get)
            probability_verdict = {
                "A": VerdictLabel.A,
                "B": VerdictLabel.B,
                "tie": VerdictLabel.TIE,
            }[winner]

        if raw_verdict is None:
            raise ValueError(
                "Constrained judge output is not an unambiguous verdict under "
                f"{JUDGE_OUTPUT_PARSER_VERSION}: {raw_text!r}"
            )
        if sampling_temperature == 0.0 and probability_verdict is not None:
            if raw_verdict != probability_verdict:
                raise RuntimeError(
                    "Deterministic constrained token verdict "
                    f"{raw_verdict.value!r} disagrees with aggregated "
                    f"label-probability MAP {probability_verdict.value!r}; "
                    "MSP would not describe the stored verdict."
                )
            return probability_verdict
        return raw_verdict

    def choose_verdict(
        self,
        prompt_text: str,
        seed: int,
        sampling_temperature: float,
    ) -> tuple[VerdictLabel, str, dict[str, float]]:
        allowed_token_ids = self.decision_allowed_token_ids
        sampling_params = SamplingParams(
            max_tokens=1,
            temperature=sampling_temperature,
            top_p=1.0,
            seed=seed,
            logprobs=len(allowed_token_ids),
            allowed_token_ids=allowed_token_ids,
            skip_special_tokens=True,
        )
        prompt_text = self._prepare_prompt(prompt_text)
        output = self.model.generate([prompt_text], sampling_params, use_tqdm=False)[0]
        completion = output.outputs[0]
        raw_text = completion.text
        probs = self._extract_label_probs(completion.logprobs[0] if completion.logprobs else None)

        verdict = self._resolve_constrained_verdict(
            raw_text=raw_text,
            probabilities=probs,
            sampling_temperature=sampling_temperature,
        )

        if not probs and getattr(self, "require_label_logprobs", True):
            raise RuntimeError(
                "vLLM did not return constrained first-token log probabilities; "
                "refusing to fabricate a one-hot uncertainty distribution."
            )
        if not probs:
            probs = {
                "A": 1.0 if verdict == VerdictLabel.A else 0.0,
                "B": 1.0 if verdict == VerdictLabel.B else 0.0,
                "tie": 1.0 if verdict == VerdictLabel.TIE else 0.0,
            }
        return verdict, raw_text, _normalize_probs(probs)

    def choose_verdict_batch(
        self,
        prompt_texts: list[str],
        seed: int,
        sampling_temperature: float,
    ) -> list[tuple[VerdictLabel, str, dict[str, float]]]:
        if not prompt_texts:
            return []
        allowed_token_ids = self.decision_allowed_token_ids
        sampling_params = SamplingParams(
            max_tokens=1,
            temperature=sampling_temperature,
            top_p=1.0,
            seed=seed,
            logprobs=len(allowed_token_ids),
            allowed_token_ids=allowed_token_ids,
            skip_special_tokens=True,
        )
        prepared = [self._prepare_prompt(prompt) for prompt in prompt_texts]
        outputs = self.model.generate(prepared, sampling_params, use_tqdm=False)
        results: list[tuple[VerdictLabel, str, dict[str, float]]] = []
        for output in outputs:
            completion = output.outputs[0]
            raw_text = completion.text
            probs = self._extract_label_probs(
                completion.logprobs[0] if completion.logprobs else None
            )
            verdict = self._resolve_constrained_verdict(
                raw_text=raw_text,
                probabilities=probs,
                sampling_temperature=sampling_temperature,
            )
            if not probs:
                raise RuntimeError(
                    "vLLM did not return constrained first-token log probabilities; "
                    "the uncertainty channel is invalid."
                )
            results.append((verdict, raw_text, _normalize_probs(probs)))
        return results

    def verbalize_confidence(
        self,
        prompt_text: str,
        seed: int = 0,
        max_tokens: int = 24,
    ) -> tuple[VerdictLabel | None, str, float | None]:
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
            seed=seed,
            stop=list(self.profile.stop_token_texts) or None,
            skip_special_tokens=True,
        )
        prompt_text = self._prepare_prompt(prompt_text)
        output = self.model.generate([prompt_text], sampling_params, use_tqdm=False)[0]
        raw_text = output.outputs[0].text.strip()
        verdict, confidence = parse_verbalized_output(raw_text)
        return (
            verdict,
            raw_text,
            confidence,
        )

    def verbalize_confidence_batch(
        self,
        prompt_texts: list[str],
        seed: int = 0,
        max_tokens: int = 24,
    ) -> list[tuple[VerdictLabel | None, str, float | None]]:
        if not prompt_texts:
            return []
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
            seed=seed,
            stop=list(self.profile.stop_token_texts) or None,
            skip_special_tokens=True,
        )
        prepared = [self._prepare_prompt(prompt) for prompt in prompt_texts]
        outputs = self.model.generate(prepared, sampling_params, use_tqdm=False)
        results: list[tuple[VerdictLabel | None, str, float | None]] = []
        for output in outputs:
            raw_text = output.outputs[0].text.strip()
            verdict, confidence = parse_verbalized_output(raw_text)
            results.append((verdict, raw_text, confidence))
        return results


QwenJudge = VLLMJudge


def _compute_consistency(verdicts: list[VerdictLabel], anchor: VerdictLabel) -> ConsistencyMetrics:
    if not verdicts:
        raise ValueError("At least one verdict is required for consistency metrics")
    counts = Counter(verdicts)
    total = len(verdicts)
    modal_count = max(counts.values())
    modal_verdicts = {
        verdict for verdict, count in counts.items() if count == modal_count
    }
    majority_verdict = (
        anchor
        if anchor in modal_verdicts
        else min(modal_verdicts, key=lambda verdict: verdict.value)
    )
    agreement = modal_count / total
    vote_entropy = 0.0
    for count in counts.values():
        prob = count / total
        vote_entropy -= prob * math.log2(prob)
    flips = sum(1 for verdict in verdicts if verdict != anchor) / total
    return ConsistencyMetrics(
        run_count=total,
        agreement_rate=agreement,
        vote_entropy=vote_entropy,
        unique_verdict_count=len(counts),
        flip_rate=flips,
        verdict_counts={_label_to_str(verdict) or "unknown": count for verdict, count in counts.items()},
        majority_verdict=majority_verdict,
    )


def _underlying_response_id(example: JudgeExample, verdict: VerdictLabel) -> str | None:
    label = _label_to_str(verdict)
    if label == VerdictLabel.TIE.value:
        return None
    mapping = example.metadata.get("response_id_by_label", {})
    return mapping.get(label)


def _build_run_record(
    *,
    example: JudgeExample,
    condition: BiasCondition,
    spec: ExperimentSpec,
    prompt_text: str,
    prompt_hash: str,
    seed: int,
    verdict: VerdictLabel,
    label_probs: dict[str, float],
    raw_output: str | None = None,
    verbalized_confidence: float | None = None,
    verbalized_verdict: VerdictLabel | None = None,
    verbalized_raw_output: str | None = None,
    verbalized_prompt_hash: str | None = None,
    consistency: ConsistencyMetrics | None = None,
    pair_key: str | None = None,
    condition_group_id: str | None = None,
    ordering_twin_key: str | None = None,
    spec_hash: str | None = None,
    input_file_hash: str | None = None,
) -> RunRecord:
    uncertainty = UncertaintyBundle(
        logit=LogitMetrics.from_probs(label_probs),
        verbalized=VerbalizedMetrics.from_confidence(
            verbalized_confidence,
            verdict=verbalized_verdict,
        ),
        consistency=consistency,
    )
    record_identity = {
        "example_id": example.example_id,
        "model_name": spec.model_name,
        "variant_id": condition.variant_id,
        "seed": seed,
        "prompt_hash": prompt_hash,
    }
    if pair_key is not None:
        record_identity["pair_key"] = pair_key
    return RunRecord(
        record_id=stable_hash(record_identity),
        spec=spec,
        example_id=example.example_id,
        question_id=str(
            example.metadata.get("question_cluster_id") or example.question_id
        ),
        condition=condition,
        seed=seed,
        verdict=verdict,
        raw_output=raw_output if raw_output is not None else verdict.value,
        prompt_hash=prompt_hash,
        uncertainty=uncertainty,
        raw_prompt_logprobs=label_probs,
        pair_key=pair_key,
        condition_group_id=condition_group_id,
        ordering_twin_key=ordering_twin_key,
        spec_hash=spec_hash,
        input_file_hash=input_file_hash,
        metadata={
            "pair_id": example.metadata.get("pair_id"),
            "source_row_index": example.metadata.get("source_row_index"),
            "routing_split": example.metadata.get("routing_split"),
            "turn": example.metadata.get("turn"),
            "selected_turn": example.metadata.get("selected_turn"),
            "conversation_extraction_mode": example.metadata.get(
                "conversation_extraction_mode"
            ),
            "variant_id": condition.variant_id,
            "human_winner": _label_to_str(example.human_winner),
            "ordering": condition.ordering,
            "dose": condition.dose,
            "direction_relative_human": condition.direction_relative_human,
            "clean_tie": condition.clean_tie,
            "clean_record_id": condition.clean_record_id,
            "underlying_response_id": _underlying_response_id(example, verdict),
            "prompt_preview": prompt_text[:200],
            "decision_token_index": 0,
            "decision_token_labels": ["A", "B", "T"],
            "judge_output_parser_version": JUDGE_OUTPUT_PARSER_VERSION,
            "verbalized_output_parser_version": (
                VERBALIZED_OUTPUT_PARSER_VERSION
            ),
            "verbalized_parse_status": verbalized_parse_status(
                uncertainty_methods=spec.uncertainty_methods,
                raw_output=verbalized_raw_output,
            ),
            "verbalized_verdict": _label_to_str(verbalized_verdict),
            "verbalized_raw_output": verbalized_raw_output,
            "verbalized_prompt_hash": verbalized_prompt_hash,
        },
    )


def _judge_example_condition(
    *,
    judge: QwenJudge,
    example: JudgeExample,
    condition: BiasCondition,
    spec: ExperimentSpec,
    choice_prompt: PromptPackage,
    confidence_prompt: PromptPackage | None,
    consistency_runs: int,
    sampling_temperature: float,
    include_verbalized_confidence: bool,
) -> RunRecord:
    verdict, raw_output, label_probs = judge.choose_verdict(
        prompt_text=choice_prompt.prompt_text,
        seed=0,
        sampling_temperature=0.0,
    )

    verbalized_verdict: VerdictLabel | None = None
    verbalized_raw_output: str | None = None
    verbalized_confidence: float | None = None
    if include_verbalized_confidence and confidence_prompt is not None:
        (
            verbalized_verdict,
            verbalized_raw_output,
            verbalized_confidence,
        ) = judge.verbalize_confidence(prompt_text=confidence_prompt.prompt_text, seed=0)

    consistency_verdicts: list[VerdictLabel] = []
    for run_seed in range(consistency_runs):
        sampled_verdict, _, _ = judge.choose_verdict(
            prompt_text=choice_prompt.prompt_text,
            seed=run_seed,
            sampling_temperature=sampling_temperature,
        )
        consistency_verdicts.append(sampled_verdict)
    consistency = (
        _compute_consistency(consistency_verdicts, anchor=verdict)
        if consistency_verdicts
        else None
    )

    return _build_run_record(
        example=example,
        condition=condition,
        spec=spec,
        prompt_text=choice_prompt.prompt_text,
        prompt_hash=choice_prompt.prompt_hash,
        seed=0,
        verdict=verdict,
        raw_output=raw_output,
        label_probs=label_probs,
        verbalized_confidence=verbalized_confidence,
        verbalized_verdict=verbalized_verdict,
        verbalized_raw_output=verbalized_raw_output,
        verbalized_prompt_hash=confidence_prompt.prompt_hash if confidence_prompt else None,
        consistency=consistency,
    )


def _record_to_uncertainty_row(record: RunRecord) -> dict[str, Any]:
    logit = record.uncertainty.logit
    verbalized = record.uncertainty.verbalized
    consistency = record.uncertainty.consistency
    label_probs = record.raw_prompt_logprobs or {}
    return {
        "record_id": record.record_id,
        "model_name": record.spec.model_name,
        "dataset_name": record.spec.dataset_name,
        "dataset_split": record.spec.dataset_split,
        "bias_name": record.spec.bias_name,
        "example_id": record.example_id,
        "question_id": record.question_id,
        "pair_id": record.metadata.get("pair_id"),
        "source_row_index": record.metadata.get("source_row_index"),
        "pair_identity_key": record.metadata.get("pair_identity_key"),
        "pair_key": record.pair_key,
        "condition_group_id": record.condition_group_id,
        "ordering_twin_key": record.ordering_twin_key,
        "spec_hash": record.spec_hash,
        "verdict_token_texts": record.spec.verdict_token_texts,
        "verdict_token_ids": record.spec.verdict_token_ids,
        "input_file_hash": record.input_file_hash,
        "routing_split": record.metadata.get("routing_split"),
        "turn": record.metadata.get("turn"),
        "selected_turn": record.metadata.get("selected_turn"),
        "conversation_extraction_mode": record.metadata.get(
            "conversation_extraction_mode"
        ),
        "variant_id": record.condition.variant_id,
        "ordering": record.condition.ordering,
        "dose": record.condition.dose,
        "cue_congruency": record.condition.cue_congruency,
        "direction_relative_human": record.condition.direction_relative_human,
        "cue_target": record.condition.cue_target,
        "clean_tie": record.condition.clean_tie,
        "clean_record_id": record.condition.clean_record_id,
        "human_winner": record.metadata.get("human_winner"),
        "verdict": record.verdict,
        "underlying_response_id": record.metadata.get("underlying_response_id"),
        "label_prob_A": label_probs.get("A"),
        "label_prob_B": label_probs.get("B"),
        "label_prob_tie": label_probs.get("tie"),
        "entropy": logit.entropy,
        "normalized_entropy": logit.normalized_entropy,
        "msp": logit.msp,
        "margin": logit.margin,
        "verbalized_confidence": verbalized.confidence,
        "verbalized_uncertainty": verbalized.uncertainty,
        "verbalized_verdict": (
            _label_to_str(verbalized.verdict)
            or record.metadata.get("verbalized_verdict")
        ),
        "consistency_agreement_rate": consistency.agreement_rate if consistency else None,
        "consistency_vote_entropy": consistency.vote_entropy if consistency else None,
        "consistency_unique_verdict_count": consistency.unique_verdict_count if consistency else None,
        "consistency_flip_rate": consistency.flip_rate if consistency else None,
        "consistency_verdict_counts": consistency.verdict_counts if consistency else None,
        "consistency_majority_verdict": (
            consistency.majority_verdict if consistency else None
        ),
        "decision_token_index": record.metadata.get("decision_token_index"),
        "decision_token_labels": record.metadata.get("decision_token_labels"),
        "judge_output_parser_version": record.metadata.get(
            "judge_output_parser_version"
        ),
        "verbalized_output_parser_version": record.metadata.get(
            "verbalized_output_parser_version"
        ),
        "verbalized_parse_status": record.metadata.get(
            "verbalized_parse_status"
        ),
        "max_num_batched_tokens": record.metadata.get("max_num_batched_tokens"),
        "max_num_seqs": record.metadata.get("max_num_seqs"),
    }


def run_position_experiment(
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
        bias_name=BiasType.POSITION.value,
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
    flip_count = 0
    usable_pairs = 0

    for pair in pairs:
        pair_records: dict[str, RunRecord] = {}
        for variant_id, example in (("original", pair.original), ("swapped", pair.swapped)):
            prompt = build_position_prompt_package(example=example, output_mode=OutputMode.CHOICE_ONLY)
            confidence_prompt = (
                build_position_prompt_package(
                    example=example,
                    output_mode=OutputMode.CHOICE_WITH_CONFIDENCE,
                )
                if include_verbalized_confidence
                else None
            )
            condition = BiasCondition(
                bias_type=BiasType.POSITION,
                variant_id=variant_id,
                metadata={"pair_id": pair.pair_id},
            )

            record = _judge_example_condition(
                judge=judge,
                example=example,
                condition=condition,
                spec=spec,
                choice_prompt=prompt,
                confidence_prompt=confidence_prompt,
                consistency_runs=consistency_runs,
                sampling_temperature=sampling_temperature,
                include_verbalized_confidence=include_verbalized_confidence,
            )
            pair_records[variant_id] = record
            raw_rows.append(record.model_dump(mode="json"))
            uncertainty_rows.append(_record_to_uncertainty_row(record))

        original_response = pair_records["original"].metadata["underlying_response_id"]
        swapped_response = pair_records["swapped"].metadata["underlying_response_id"]
        flipped = (
            original_response is not None
            and swapped_response is not None
            and original_response != swapped_response
        )
        if original_response is not None and swapped_response is not None:
            usable_pairs += 1
            flip_count += int(flipped)

        pair_summaries.append(
            {
                "pair_id": pair.pair_id,
                "source_row_index": pair.original.metadata.get("source_row_index"),
                "routing_split": pair.original.metadata.get("routing_split"),
                "human_winner": _label_to_str(pair.original.human_winner),
                "original_verdict": pair_records["original"].verdict,
                "swapped_verdict": pair_records["swapped"].verdict,
                "original_response_id": original_response,
                "swapped_response_id": swapped_response,
                "position_flip": flipped,
                "original_entropy": pair_records["original"].uncertainty.logit.entropy,
                "original_msp": pair_records["original"].uncertainty.logit.msp,
                "original_margin": pair_records["original"].uncertainty.logit.margin,
                "original_agreement_rate": (
                    pair_records["original"].uncertainty.consistency.agreement_rate
                    if pair_records["original"].uncertainty.consistency
                    else None
                ),
                "original_consistency_entropy": (
                    pair_records["original"].uncertainty.consistency.vote_entropy
                    if pair_records["original"].uncertainty.consistency
                    else None
                ),
                "original_verbalized_confidence": (
                    pair_records["original"].uncertainty.verbalized.confidence
                ),
                "original_verbalized_uncertainty": (
                    pair_records["original"].uncertainty.verbalized.uncertainty
                ),
                "swapped_entropy": pair_records["swapped"].uncertainty.logit.entropy,
                "swapped_msp": pair_records["swapped"].uncertainty.logit.msp,
                "swapped_margin": pair_records["swapped"].uncertainty.logit.margin,
                "swapped_agreement_rate": (
                    pair_records["swapped"].uncertainty.consistency.agreement_rate
                    if pair_records["swapped"].uncertainty.consistency
                    else None
                ),
                "swapped_consistency_entropy": (
                    pair_records["swapped"].uncertainty.consistency.vote_entropy
                    if pair_records["swapped"].uncertainty.consistency
                    else None
                ),
                "swapped_verbalized_confidence": (
                    pair_records["swapped"].uncertainty.verbalized.confidence
                ),
                "swapped_verbalized_uncertainty": (
                    pair_records["swapped"].uncertainty.verbalized.uncertainty
                ),
            }
        )

    raw_path = output_dir / "position_run_records.jsonl"
    pair_path = output_dir / "position_pair_summary.jsonl"
    uncertainty_path = output_dir / "position_uncertainty_scores.jsonl"
    write_jsonl(raw_path, raw_rows)
    write_jsonl(pair_path, pair_summaries)
    write_jsonl(uncertainty_path, uncertainty_rows)

    summary = {
        "model_name": model_name,
        "csv_path": str(csv_path),
        "output_dir": str(output_dir),
        "dataset_split": dataset_split,
        "total_pairs_loaded": len(pairs),
        "usable_pairs_for_flip": usable_pairs,
        "flip_count": flip_count,
        "flip_rate": (flip_count / usable_pairs) if usable_pairs else None,
        "consistency_runs": consistency_runs,
        "sampling_temperature": sampling_temperature,
        "include_verbalized_confidence": include_verbalized_confidence,
        "tensor_parallel_size": tensor_parallel_size,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization,
        "dtype": dtype,
        "raw_records_path": str(raw_path),
        "pair_summary_path": str(pair_path),
        "uncertainty_scores_path": str(uncertainty_path),
    }

    summary_path = output_dir / "position_summary.json"
    ensure_parent(summary_path)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)

    return summary
