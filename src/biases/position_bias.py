from __future__ import annotations

import csv
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


def load_position_pairs(csv_path: Path, limit: int | None = None) -> list[PositionPair]:
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

        if prompt_column is None and (conversation_a_column is None or conversation_b_column is None):
            raise KeyError(
                "CSV must contain either prompt/response_a/response_b columns or "
                "conversation_a/conversation_b columns."
            )

        pairs: list[PositionPair] = []
        for index, row in enumerate(reader):
            if limit is not None and len(pairs) >= limit:
                break

            base_id = row[id_column].strip() or f"row-{index}"
            turn_value = row[turn_column].strip() if turn_column and row.get(turn_column) else ""
            pair_id = f"{base_id}:turn-{turn_value}" if turn_value else base_id

            if prompt_column is not None and response_a_column is not None and response_b_column is not None:
                prompt_messages = _parse_prompt_messages(row[prompt_column])
                response_a = row[response_a_column].strip()
                response_b = row[response_b_column].strip()
            else:
                conversation_a = _parse_conversation(row[conversation_a_column or ""])
                conversation_b = _parse_conversation(row[conversation_b_column or ""])
                prompt_messages = _shared_prefix_messages(conversation_a, conversation_b)
                if not prompt_messages and conversation_a:
                    prompt_messages = conversation_a[:-1] or conversation_a
                response_a = _extract_final_response(conversation_a)
                response_b = _extract_final_response(conversation_b)

            human_winner = _normalize_winner(row[winner_column])
            if human_winner is None or not response_a or not response_b:
                continue

            model_a = row[model_a_column].strip() if model_a_column and row.get(model_a_column) else None
            model_b = row[model_b_column].strip() if model_b_column and row.get(model_b_column) else None
            routing_split = (
                row[routing_split_column].strip()
                if routing_split_column and row.get(routing_split_column)
                else None
            )

            base_metadata = {
                "pair_id": pair_id,
                "source_csv": str(csv_path),
                "source_row_index": index,
                "turn": turn_value or None,
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


class QwenJudge:
    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        max_model_len: int = DEFAULT_MAX_MODEL_LEN,
        gpu_memory_utilization: float = 0.9,
        dtype: str = "auto",
    ) -> None:
        if LLM is None or SamplingParams is None:
            raise RuntimeError(
                "vLLM execution requires the 'local' extra. Install with `uv sync --extra local`."
            )

        self.model_name = model_name
        disable_custom_all_reduce = os.environ.get("VLLM_DISABLE_CUSTOM_ALL_REDUCE", "").lower()
        disable_custom_all_reduce = disable_custom_all_reduce in {"1", "true", "yes", "on"}
        enforce_eager = os.environ.get("BIASES_VLLM_ENFORCE_EAGER", "").lower()
        enforce_eager = enforce_eager in {"1", "true", "yes", "on"}
        self.model = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            disable_custom_all_reduce=disable_custom_all_reduce,
            enforce_eager=enforce_eager,
        )
        self.tokenizer = self._get_tokenizer()
        (
            self.decision_label_token_ids,
            self.decision_token_id_to_label,
        ) = self._build_decision_label_token_maps()

    def _prepare_prompt(self, prompt_text: str) -> str:
        if "qwen3" not in self.model_name.lower():
            return prompt_text

        assistant_prefix = "<|im_start|>assistant\n"
        non_thinking_prefix = "<think>\n\n</think>\n\n"
        if not prompt_text.endswith(assistant_prefix):
            return prompt_text
        return f"{prompt_text}{non_thinking_prefix}"

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
        return int(token_ids[0])

    def _build_decision_label_token_maps(self) -> tuple[dict[str, list[int]], dict[int, str]]:
        label_texts = {
            "A": ("A", " A"),
            "B": ("B", " B"),
            "tie": ("T", " T"),
        }
        label_to_ids: dict[str, list[int]] = {}
        token_id_to_label: dict[int, str] = {}
        for label, texts in label_texts.items():
            token_ids: list[int] = []
            for text in texts:
                token_id = self._encode_single_token(text)
                if token_id is None or token_id in token_ids:
                    continue
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

    @staticmethod
    def _decoded_token(candidate: Any) -> str:
        if isinstance(candidate, dict):
            return str(candidate.get("decoded_token", ""))
        return str(getattr(candidate, "decoded_token", ""))

    def _extract_label_probs(self, first_token_logprobs: Any | None) -> dict[str, float]:
        if not first_token_logprobs:
            return {}

        label_logprobs: list[tuple[str, float]] = []
        for token_id, candidate in first_token_logprobs.items():
            logprob = self._logprob_value(candidate)
            if logprob is None:
                continue
            try:
                label = self.decision_token_id_to_label.get(int(token_id))
            except (TypeError, ValueError):
                label = None
            if label is None:
                decoded = self._decoded_token(candidate).strip().upper()
                label = {"A": "A", "B": "B", "T": "tie"}.get(decoded)
            if label is not None:
                label_logprobs.append((label, logprob))

        if not label_logprobs:
            return {}

        max_logprob = max(logprob for _, logprob in label_logprobs)
        weights = {"A": 0.0, "B": 0.0, "tie": 0.0}
        for label, logprob in label_logprobs:
            weights[label] += math.exp(logprob - max_logprob)
        total = sum(weights.values())
        return {label: value / total for label, value in weights.items()}

    @staticmethod
    def _parse_verdict_text(text: str) -> VerdictLabel | None:
        normalized = text.strip().upper()
        if normalized.startswith("A"):
            return VerdictLabel.A
        if normalized.startswith("B"):
            return VerdictLabel.B
        if normalized.startswith("T"):
            return VerdictLabel.TIE
        return None

    @staticmethod
    def _parse_confidence(text: str) -> float | None:
        match = re.search(r"confidence\s*[:=]\s*(100(?:\.0+)?|\d{1,2}(?:\.\d+)?)", text, re.I)
        if match is None:
            numbers = re.findall(r"\b(100(?:\.0+)?|\d{1,2}(?:\.\d+)?)\b", text)
            if not numbers:
                return None
            value = float(numbers[-1])
        else:
            value = float(match.group(1))
        return max(0.0, min(100.0, value))

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

        verdict = self._parse_verdict_text(raw_text)
        if verdict is None and probs:
            winner = max(probs, key=probs.get)
            verdict = {"A": VerdictLabel.A, "B": VerdictLabel.B, "tie": VerdictLabel.TIE}[winner]
        if verdict is None:
            raise ValueError(f"Could not parse judge output {completion.text!r}")

        if not probs:
            probs = {
                "A": 1.0 if verdict == VerdictLabel.A else 0.0,
                "B": 1.0 if verdict == VerdictLabel.B else 0.0,
                "tie": 1.0 if verdict == VerdictLabel.TIE else 0.0,
            }
        return verdict, raw_text, _normalize_probs(probs)

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
            stop=["<|im_end|>"],
            skip_special_tokens=True,
        )
        prompt_text = self._prepare_prompt(prompt_text)
        output = self.model.generate([prompt_text], sampling_params, use_tqdm=False)[0]
        raw_text = output.outputs[0].text.strip()
        return (
            self._parse_verdict_text(raw_text),
            raw_text,
            self._parse_confidence(raw_text),
        )


def _compute_consistency(verdicts: list[VerdictLabel], anchor: VerdictLabel) -> ConsistencyMetrics:
    counts = Counter(verdicts)
    total = len(verdicts)
    agreement = max(counts.values()) / total
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
) -> RunRecord:
    uncertainty = UncertaintyBundle(
        logit=LogitMetrics.from_probs(label_probs),
        verbalized=VerbalizedMetrics.from_confidence(verbalized_confidence),
        consistency=consistency,
    )
    return RunRecord(
        record_id=stable_hash(
            {
                "example_id": example.example_id,
                "model_name": spec.model_name,
                "variant_id": condition.variant_id,
                "seed": seed,
                "prompt_hash": prompt_hash,
            }
        ),
        spec=spec,
        example_id=example.example_id,
        question_id=str(example.question_id),
        condition=condition,
        seed=seed,
        verdict=verdict,
        raw_output=raw_output if raw_output is not None else verdict.value,
        prompt_hash=prompt_hash,
        uncertainty=uncertainty,
        raw_prompt_logprobs=label_probs,
        metadata={
            "pair_id": example.metadata.get("pair_id"),
            "routing_split": example.metadata.get("routing_split"),
            "variant_id": condition.variant_id,
            "underlying_response_id": _underlying_response_id(example, verdict),
            "prompt_preview": prompt_text[:200],
            "decision_token_index": 0,
            "decision_token_labels": ["A", "B", "T"],
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
    consistency = _compute_consistency(consistency_verdicts, anchor=verdict)

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
        "routing_split": record.metadata.get("routing_split"),
        "variant_id": record.condition.variant_id,
        "cue_congruency": record.condition.cue_congruency,
        "cue_target": record.condition.cue_target,
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
        "consistency_agreement_rate": consistency.agreement_rate if consistency else None,
        "consistency_vote_entropy": consistency.vote_entropy if consistency else None,
        "consistency_unique_verdict_count": consistency.unique_verdict_count if consistency else None,
        "consistency_flip_rate": consistency.flip_rate if consistency else None,
        "consistency_verdict_counts": consistency.verdict_counts if consistency else None,
        "decision_token_index": record.metadata.get("decision_token_index"),
        "decision_token_labels": record.metadata.get("decision_token_labels"),
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
