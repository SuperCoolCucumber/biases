from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from math import log2
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class BiasType(str, Enum):
    POSITION = "position"
    AUTHORITY = "authority"
    BANDWAGON = "bandwagon"
    DECOY = "decoy"
    CONTROL = "control"


class CueCongruency(str, Enum):
    CONTROL = "control"
    CONGRUENT = "congruent"
    INCONGRUENT = "incongruent"
    NONE = "none"


class VerdictLabel(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    TIE = "tie"


class OutputMode(str, Enum):
    CHOICE_ONLY = "choice_only"
    CHOICE_WITH_CONFIDENCE = "choice_with_confidence"


class Candidate(BaseModel):
    label: VerdictLabel
    response: str
    model_id: str | None = None
    response_id: str | None = None


class JudgeExample(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    example_id: str
    question_id: int | str
    prompt_messages: list[dict[str, str]]
    candidates: dict[str, Candidate]
    human_winner: VerdictLabel | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("candidates")
    @classmethod
    def validate_candidates(cls, value: dict[str, Candidate]) -> dict[str, Candidate]:
        if not value:
            raise ValueError("candidates must not be empty")
        return value


class BiasCondition(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    bias_type: BiasType
    variant_id: str
    cue_target: VerdictLabel | None = None
    cue_congruency: CueCongruency = CueCongruency.NONE
    cue_text: str | None = None
    decoy_anchor: VerdictLabel | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PromptPackage(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    prompt_text: str
    output_mode: OutputMode
    allowed_labels: list[VerdictLabel]
    prompt_hash: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class JudgeRequest(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    example: JudgeExample
    condition: BiasCondition
    prompt: PromptPackage
    model_name: str
    backend_name: str
    temperature: float
    seed: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class JudgeResponse(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    verdict: VerdictLabel
    raw_output: str
    prompt_logprobs: dict[str, float] | None = None
    verbalized_confidence: float | None = None
    rationale: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class LogitMetrics(BaseModel):
    entropy: float | None = None
    msp: float | None = None
    margin: float | None = None
    normalized_entropy: float | None = None

    @classmethod
    def from_probs(cls, probs: dict[str, float]) -> "LogitMetrics":
        if not probs:
            return cls()

        ordered = sorted(probs.values(), reverse=True)
        entropy = -sum(prob * log2(prob) for prob in probs.values() if prob > 0)
        top = ordered[0]
        second = ordered[1] if len(ordered) > 1 else 0.0
        normalized = entropy / log2(len(probs)) if len(probs) > 1 else 0.0
        return cls(
            entropy=entropy,
            msp=top,
            margin=top - second,
            normalized_entropy=normalized,
        )


class VerbalizedMetrics(BaseModel):
    confidence: float | None = None
    uncertainty: float | None = None

    @classmethod
    def from_confidence(cls, confidence: float | None) -> "VerbalizedMetrics":
        if confidence is None:
            return cls()
        normalized_confidence = max(0.0, min(100.0, confidence)) / 100.0
        return cls(
            confidence=normalized_confidence,
            uncertainty=1.0 - normalized_confidence,
        )


class ConsistencyMetrics(BaseModel):
    run_count: int
    agreement_rate: float
    vote_entropy: float
    unique_verdict_count: int
    flip_rate: float


class UncertaintyBundle(BaseModel):
    logit: LogitMetrics = Field(default_factory=LogitMetrics)
    verbalized: VerbalizedMetrics = Field(default_factory=VerbalizedMetrics)
    consistency: ConsistencyMetrics | None = None


class ExperimentSpec(BaseModel):
    dataset_name: str
    dataset_split: str
    model_name: str
    backend_name: str
    bias_name: str
    output_mode: OutputMode
    uncertainty_methods: list[str]
    consistency_runs: int
    temperature: float


class RunRecord(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    record_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    spec: ExperimentSpec
    example_id: str
    question_id: str
    condition: BiasCondition
    seed: int
    verdict: VerdictLabel
    raw_output: str
    prompt_hash: str
    uncertainty: UncertaintyBundle
    raw_prompt_logprobs: dict[str, float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ItemAggregate(BaseModel):
    example_id: str
    question_id: str
    bias_name: str
    model_name: str
    run_count: int
    dominant_verdict: VerdictLabel
    mean_entropy: float | None = None
    mean_msp: float | None = None
    mean_verbalized_uncertainty: float | None = None
    agreement_rate: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
