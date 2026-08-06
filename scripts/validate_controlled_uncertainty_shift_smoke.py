#!/usr/bin/env python3
"""Run a deterministic, safety-focused small-grid GPU preflight.

This is a gate for the controlled uncertainty-shift campaign, not a scientific
analysis.  It enumerates candidates with the production Stage A/Stage B
planners, renders them with the production social-cue prompt builder and model
profile, selects the longest prompt in every required smoke stratum, and then
uses the production :class:`VLLMJudge` for constrained verdict inference.

The selected cued conditions use the human winner as a provisional clean
reference solely to exercise both cue directions before a full Stage A run.
They must never be treated as experimental observations.  SCOPE and BPE are
intentionally absent.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

import biases.dataset_splits as dataset_splits_module
import biases.models as models_module
import biases.pairing as pairing_module
import biases.position_bias as position_bias_module
import biases.safe_diagnostics as safe_diagnostics_module
import biases.schemas as schemas_module
import biases.social_cue_prompts as prompts_module
import biases.stage_planning as planning_module
from biases.dataset_splits import (
    assign_question_disjoint_routing_split,
    routing_manifest,
)
from biases.models import ModelProfile, get_model_profile
from biases.pairing import file_sha256, make_pair_identity_key, normalize_ordering
from biases.position_bias import (
    CONSTRAINED_LOGPROBS_MODE,
    JUDGE_OUTPUT_PARSER_VERSION,
    SamplingParams,
    VLLMJudge,
    load_position_pairs_with_eligibility,
)
from biases.safe_diagnostics import sanitize_exception_text
from biases.schemas import (
    BiasType,
    CueCongruency,
    JudgeExample,
    OutputMode,
    PairOrdering,
    VerdictLabel,
)
from biases.social_cue_prompts import (
    AUTHORITY_DOSES,
    BANDWAGON_DOSES,
    build_social_cue_messages,
    build_social_cue_prompt_package,
)
from biases.stage_planning import (
    CleanPairSummary,
    PlannedCondition,
    StageAPairInput,
    generate_stage_a_conditions,
    generate_stage_b_conditions,
)


SCHEMA_VERSION = 1
ROUTING_SCHEMA_VERSION = 2
ROUTING_ARTIFACT_TYPE = "frozen_question_disjoint_routing_package"
ROUTING_SEED = 42
CALIBRATION_FRACTION = 0.5
EXPECTED_VERDICT_SURFACES: Mapping[str, tuple[str, ...]] = {
    "A": ("A",),
    "B": ("B",),
    "tie": ("T",),
}
REQUIRED_RUNTIME_FIELDS = (
    "model_registry_name",
    "model_hf_name",
    "model_revision",
    "tensor_parallel_size",
    "max_model_len",
    "gpu_memory_utilization",
    "dtype",
    "batch_size",
    "max_num_batched_tokens",
    "max_num_seqs",
    "enforce_eager",
    "disable_custom_all_reduce",
    "seed",
    "sampling_temperature",
    "consistency_runs",
    "consistency_schedule",
    "include_verbalized_confidence",
    "engine_versions",
    "runtime_sha256",
)
ENGINE_VERSION_FIELDS = ("python", "torch", "transformers", "vllm")
NATIVE_MAX_TOKENS = 16
DETERMINISTIC_PASSES = 2
DETERMINISTIC_PROBABILITY_ATOL = 1e-6
CONSISTENCY_RUNS = 4
CONSISTENCY_SCHEDULE = "extremes"
CONSISTENCY_TEMPERATURE = 0.7
MINIMUM_PRODUCTION_SEQUENCES_PER_SECOND = 0.946
RELEASE_TARGET_PRODUCTION_SEQUENCES_PER_SECOND = 1.261


class JudgeBackend(Protocol):
    profile: ModelProfile
    model_name: str
    tokenizer: Any
    logprobs_mode: str
    decision_label_token_ids: Mapping[str, Sequence[int]]
    decision_allowed_token_ids: Sequence[int]

    def render_messages(self, messages: list[dict[str, str]]) -> str: ...

    def choose_verdict_batch(
        self,
        prompt_texts: list[str],
        seed: int,
        sampling_temperature: float,
    ) -> list[tuple[VerdictLabel, str, dict[str, float]]]: ...


@dataclass(frozen=True, slots=True)
class NativeGeneration:
    text: str
    token_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CandidatePrompt:
    stage: str
    routing_split: str
    planned: PlannedCondition
    example: JudgeExample
    prompt_text: str
    prompt_hash: str
    prompt_sha256: str
    token_ids_sha256: str
    input_tokens: int

    @property
    def key(self) -> str:
        return "\0".join(
            (
                self.stage,
                self.planned.pair_key,
                self.planned.condition.variant_id,
            )
        )

    @property
    def ordering(self) -> str:
        return normalize_ordering(self.planned.condition.ordering or "").value

    @property
    def family(self) -> str:
        return str(self.planned.condition.bias_type)

    @property
    def direction(self) -> str:
        return str(self.planned.condition.cue_congruency)


@dataclass(frozen=True, slots=True)
class SmallGridPlan:
    candidates: tuple[CandidatePrompt, ...]
    selected: tuple[CandidatePrompt, ...]
    source_pair_count: int
    source_row_count: int
    test_pair_count: int
    eligibility_audit: Mapping[str, Any]
    eligible_question_counts: Mapping[str, int]
    provisional_tie_issue_count: int

    @property
    def eligibility_sha256(self) -> str:
        return str(self.eligibility_audit["eligibility_sha256"])


NativeGenerator = Callable[
    [JudgeBackend, Sequence[str], int, int],
    Sequence[NativeGeneration],
]


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def value_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def enum_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def read_json_mapping(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain one JSON object")
    return value


def _positive_int(runtime: Mapping[str, Any], field: str) -> int:
    value = runtime[field]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"runtime {field} must be a positive integer")
    return value


def validate_runtime(runtime: Mapping[str, Any]) -> dict[str, Any]:
    missing = [field for field in REQUIRED_RUNTIME_FIELDS if field not in runtime]
    if missing:
        raise ValueError(f"runtime mapping is missing fields: {', '.join(missing)}")
    unknown = sorted(set(runtime) - set(REQUIRED_RUNTIME_FIELDS))
    if unknown:
        raise ValueError(
            "runtime mapping contains unsupported fields: " + ", ".join(unknown)
        )

    tensor_parallel_size = _positive_int(runtime, "tensor_parallel_size")
    max_model_len = _positive_int(runtime, "max_model_len")
    batch_size = _positive_int(runtime, "batch_size")
    max_num_batched_tokens = _positive_int(runtime, "max_num_batched_tokens")
    max_num_seqs = _positive_int(runtime, "max_num_seqs")
    normalized: dict[str, Any] = {}
    for field in ("model_registry_name", "model_hf_name"):
        value = runtime[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"runtime {field} must be a non-empty string")
        normalized[field] = value.strip()
    model_revision = runtime["model_revision"]
    if model_revision is not None and (
        not isinstance(model_revision, str) or not model_revision.strip()
    ):
        raise ValueError("runtime model_revision must be null or a non-empty string")
    normalized["model_revision"] = (
        model_revision.strip() if isinstance(model_revision, str) else None
    )
    try:
        gpu_memory_utilization = float(runtime["gpu_memory_utilization"])
        sampling_temperature = float(runtime["sampling_temperature"])
    except (TypeError, ValueError) as exc:
        raise ValueError("runtime floating-point controls are invalid") from exc
    if not 0.0 < gpu_memory_utilization <= 1.0:
        raise ValueError("runtime gpu_memory_utilization must be in (0, 1]")
    if sampling_temperature != CONSISTENCY_TEMPERATURE:
        raise ValueError(
            "runtime sampling_temperature must match the frozen repeatability "
            f"temperature {CONSISTENCY_TEMPERATURE}"
        )
    seed = runtime["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed != 0:
        raise ValueError("runtime seed must equal the frozen production seed 0")
    dtype = str(runtime["dtype"]).strip()
    if not dtype:
        raise ValueError("runtime dtype must not be blank")
    for field in ("enforce_eager", "disable_custom_all_reduce"):
        if not isinstance(runtime[field], bool):
            raise ValueError(f"runtime {field} must be boolean")
    if runtime["consistency_runs"] != CONSISTENCY_RUNS:
        raise ValueError(
            f"runtime consistency_runs must equal {CONSISTENCY_RUNS}"
        )
    if runtime["consistency_schedule"] != CONSISTENCY_SCHEDULE:
        raise ValueError(
            f"runtime consistency_schedule must equal {CONSISTENCY_SCHEDULE!r}"
        )
    if runtime["include_verbalized_confidence"] is not False:
        raise ValueError(
            "runtime include_verbalized_confidence must be false for this campaign"
        )

    raw_engine_versions = runtime["engine_versions"]
    if not isinstance(raw_engine_versions, Mapping):
        raise ValueError("runtime engine_versions must be a mapping")
    if set(raw_engine_versions) != set(ENGINE_VERSION_FIELDS):
        raise ValueError(
            "runtime engine_versions must contain exactly: "
            + ", ".join(ENGINE_VERSION_FIELDS)
        )
    engine_versions: dict[str, str | None] = {}
    for field in ENGINE_VERSION_FIELDS:
        value = raw_engine_versions[field]
        if value is not None and (
            not isinstance(value, str) or not value.strip()
        ):
            raise ValueError(
                f"runtime engine_versions.{field} must be null or a non-empty string"
            )
        engine_versions[field] = value.strip() if isinstance(value, str) else None

    normalized.update({
        "tensor_parallel_size": tensor_parallel_size,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization,
        "dtype": dtype,
        "batch_size": batch_size,
        "max_num_batched_tokens": max_num_batched_tokens,
        "max_num_seqs": max_num_seqs,
        "enforce_eager": runtime["enforce_eager"],
        "disable_custom_all_reduce": runtime["disable_custom_all_reduce"],
        "seed": seed,
        "sampling_temperature": sampling_temperature,
        "consistency_runs": CONSISTENCY_RUNS,
        "consistency_schedule": CONSISTENCY_SCHEDULE,
        "include_verbalized_confidence": False,
        "engine_versions": engine_versions,
    })
    digest = runtime["runtime_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError("runtime runtime_sha256 must be a lowercase SHA-256")
    digest_input = {
        field: normalized[field]
        for field in REQUIRED_RUNTIME_FIELDS
        if field != "runtime_sha256"
    }
    expected_digest = value_sha256(digest_input)
    if digest != expected_digest:
        raise ValueError(
            "runtime runtime_sha256 does not match the controlled runtime mapping"
        )
    normalized["runtime_sha256"] = digest
    return normalized


def _distribution_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def observed_engine_versions() -> dict[str, str | None]:
    return {
        "python": platform.python_version(),
        "torch": _distribution_version("torch"),
        "transformers": _distribution_version("transformers"),
        "vllm": _distribution_version("vllm"),
    }


def require_engine_runtime(runtime: Mapping[str, Any]) -> None:
    observed = observed_engine_versions()
    if runtime["engine_versions"] != observed:
        raise ValueError(
            "runtime engine_versions do not match the active execution environment"
        )


def validate_runtime_model_contract(
    runtime: Mapping[str, Any],
    *,
    model_name: str,
) -> ModelProfile:
    """Bind a validated runtime to one pinned profile before model access."""

    profile = get_model_profile(model_name)
    if not profile.revision:
        raise ValueError("small-grid smoke requires a pinned model revision")
    expected_identity = {
        "model_registry_name": profile.registry_name,
        "model_hf_name": profile.hf_model_name,
        "model_revision": profile.revision,
    }
    observed_identity = {
        field: runtime[field] for field in expected_identity
    }
    if observed_identity != expected_identity:
        raise ValueError(
            "runtime model identity does not match the selected registry profile: "
            f"observed={observed_identity!r} expected={expected_identity!r}"
        )
    return profile


def _required_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"routing manifest {name} must be a JSON object")
    return value


def _required_sha256(value: Any, *, name: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"routing manifest {name} must be a lowercase SHA-256")
    return text


def _resolve_package_output(
    manifest_path: Path,
    outputs: Mapping[str, Any],
    name: str,
) -> Path:
    record = _required_mapping(outputs.get(name), name=f"outputs.{name}")
    raw_path = Path(str(record.get("path") or ""))
    if raw_path.is_absolute() or len(raw_path.parts) != 1 or raw_path.name in {"", ".", ".."}:
        raise ValueError(f"routing manifest outputs.{name}.path must be one filename")
    return (manifest_path.parent / raw_path).resolve()


def _frame_records(frame: pd.DataFrame, columns: Sequence[str]) -> list[dict[str, str]]:
    return [
        {column: str(value) for column, value in zip(columns, values)}
        for values in frame[list(columns)].itertuples(index=False, name=None)
    ]


def validate_routing_contract(
    frame: pd.DataFrame,
    source_csv: Path,
    source_sha256: str,
    frozen: Mapping[str, Any],
    *,
    manifest_path: Path,
) -> dict[str, Any]:
    if frozen.get("schema_version") != ROUTING_SCHEMA_VERSION:
        raise ValueError("routing manifest must use schema version 2")
    if frozen.get("artifact_type") != ROUTING_ARTIFACT_TYPE:
        raise ValueError("routing manifest artifact type is not the frozen routing package")
    if frozen.get("routing_unit") != "question":
        raise ValueError("routing manifest must freeze routing_unit='question'")
    if frozen.get("seed") != ROUTING_SEED:
        raise ValueError(f"routing manifest must freeze seed={ROUTING_SEED}")
    if frozen.get("calibration_fraction") != CALIBRATION_FRACTION:
        raise ValueError(
            "routing manifest must freeze calibration_fraction="
            f"{CALIBRATION_FRACTION}"
        )
    if "routing_split" not in frame.columns:
        raise ValueError("routed full CSV must contain routing_split")

    expected_frame = assign_question_disjoint_routing_split(
        frame.drop(columns=["routing_split"]),
        seed=ROUTING_SEED,
        calibration_fraction=CALIBRATION_FRACTION,
    )
    observed_splits = frame["routing_split"].astype(str).str.strip().str.lower()
    expected_splits = expected_frame["routing_split"].astype(str).str.strip().str.lower()
    if observed_splits.tolist() != expected_splits.tolist():
        raise ValueError("routed full CSV does not match the deterministic question split")

    observed = routing_manifest(
        frame,
        routing_unit="question",
        seed=ROUTING_SEED,
        calibration_fraction=CALIBRATION_FRACTION,
    )
    for field in (
        "routing_assignment_sha256",
        "row_counts",
        "question_counts",
    ):
        if observed[field] != frozen.get(field):
            raise ValueError(f"routing manifest {field} does not match source")
    if observed["question_counts"]["overlap"] != 0:
        raise ValueError("calibration and test questions overlap")
    if min(
        observed["question_counts"]["calibration"],
        observed["question_counts"]["test"],
    ) < 1:
        raise ValueError("both question-disjoint routing splits are required")

    outputs = _required_mapping(frozen.get("outputs"), name="outputs")
    output_hashes = _required_mapping(
        frozen.get("output_sha256"),
        name="output_sha256",
    )
    if set(outputs) != {"full", "calibration", "test"} or set(output_hashes) != {
        "full",
        "calibration",
        "test",
    }:
        raise ValueError("routing manifest must pin exactly full/calibration/test outputs")
    output_paths = {
        name: _resolve_package_output(manifest_path, outputs, name)
        for name in ("full", "calibration", "test")
    }
    if output_paths["full"] != source_csv.resolve():
        raise ValueError("source CSV is not the manifest's routed full output")
    for name, output_path in output_paths.items():
        if not output_path.is_file():
            raise ValueError(f"routing package output {name} is missing")
        expected_hash = _required_sha256(
            output_hashes.get(name),
            name=f"output_sha256.{name}",
        )
        if file_sha256(output_path) != expected_hash:
            raise ValueError(f"routing package output {name} hash does not match")
        record = _required_mapping(outputs[name], name=f"outputs.{name}")
        expected_rows = observed["row_counts"]["total" if name == "full" else name]
        if record.get("rows") != expected_rows:
            raise ValueError(f"routing manifest outputs.{name}.rows does not match")
    if output_hashes["full"] != source_sha256:
        raise ValueError("routing manifest full CSV hash does not match source")

    full_text = pd.read_csv(source_csv, dtype=str, keep_default_na=False)
    calibration_text = pd.read_csv(
        output_paths["calibration"], dtype=str, keep_default_na=False
    )
    test_text = pd.read_csv(output_paths["test"], dtype=str, keep_default_na=False)
    if list(full_text.columns) != list(calibration_text.columns) or list(
        full_text.columns
    ) != list(test_text.columns):
        raise ValueError("routing package output columns do not match")
    expected_calibration = full_text.loc[
        full_text["routing_split"] == "calibration"
    ].reset_index(drop=True)
    expected_test = full_text.loc[full_text["routing_split"] == "test"].reset_index(
        drop=True
    )
    if not calibration_text.equals(expected_calibration) or not test_text.equals(
        expected_test
    ):
        raise ValueError("routing package split outputs do not partition the full CSV")

    source = _required_mapping(frozen.get("source"), name="source")
    original_path = Path(str(source.get("path") or "")).resolve()
    if not original_path.is_file():
        raise ValueError("routing manifest original source is missing")
    original_sha = _required_sha256(source.get("sha256"), name="source.sha256")
    if file_sha256(original_path) != original_sha:
        raise ValueError("routing manifest original source hash does not match")
    if not _required_mapping(
        source.get("dataset_lineage"), name="source.dataset_lineage"
    ):
        raise ValueError("routing manifest dataset lineage must not be empty")
    original_text = pd.read_csv(original_path, dtype=str, keep_default_na=False)
    declared_columns = source.get("columns")
    if declared_columns != list(original_text.columns):
        raise ValueError("routing manifest source columns do not match original")
    preserved = _required_mapping(
        frozen.get("content_preservation"), name="content_preservation"
    )
    preserved_columns = preserved.get("preserved_columns")
    expected_preserved = [
        column for column in original_text.columns if column != "routing_split"
    ]
    if preserved_columns != expected_preserved:
        raise ValueError("routing manifest preserved columns do not match original")
    if preserved.get("recomputed_columns") != ["routing_split"] or preserved.get(
        "row_order_preserved"
    ) is not True:
        raise ValueError("routing manifest does not freeze routing-only recomputation")
    if not set(expected_preserved).issubset(full_text.columns):
        raise ValueError("routed full CSV dropped original source columns")
    if _frame_records(original_text, expected_preserved) != _frame_records(
        full_text, expected_preserved
    ):
        raise ValueError("routed full CSV changed original content or row order")
    source_rows_hash = _required_sha256(
        source.get("rows_without_routing_sha256"),
        name="source.rows_without_routing_sha256",
    )
    preserved_rows_hash = _required_sha256(
        preserved.get("rows_without_routing_sha256"),
        name="content_preservation.rows_without_routing_sha256",
    )
    if source_rows_hash != preserved_rows_hash:
        raise ValueError("routing manifest content-preservation hashes disagree")
    return observed


def validate_eligibility_contract(
    plan: SmallGridPlan,
    frozen: Mapping[str, Any],
    observed_routing: Mapping[str, Any],
) -> None:
    expected_eligibility = _required_mapping(
        frozen.get("eligibility"), name="eligibility"
    )
    if dict(expected_eligibility) != dict(plan.eligibility_audit):
        raise ValueError("routing manifest eligibility audit does not match source")
    counts = _required_mapping(frozen.get("counts"), name="counts")
    observed_counts = {
        "raw_rows": dict(observed_routing["row_counts"]),
        "raw_questions": dict(observed_routing["question_counts"]),
        "eligible_pairs": {
            "total": plan.source_pair_count,
            "calibration": int(
                plan.eligibility_audit["routing_counts"]["eligible_pairs"].get(
                    "calibration", 0
                )
            ),
            "test": int(
                plan.eligibility_audit["routing_counts"]["eligible_pairs"].get(
                    "test", 0
                )
            ),
        },
        "eligible_questions": dict(plan.eligible_question_counts),
        "skipped_rows": {
            "total": int(plan.eligibility_audit["skipped_row_count"]),
            "calibration": int(
                plan.eligibility_audit["routing_counts"]["skipped_rows"].get(
                    "calibration", 0
                )
            ),
            "test": int(
                plan.eligibility_audit["routing_counts"]["skipped_rows"].get(
                    "test", 0
                )
            ),
        },
    }
    if dict(counts) != observed_counts:
        raise ValueError("routing manifest raw/eligible/skipped counts do not match")


def _transport_token_ids(
    judge: JudgeBackend,
    messages: Sequence[Mapping[str, str]],
    rendered: str,
) -> list[int]:
    transport_ids = [
        int(token_id)
        for token_id in judge.tokenizer.encode(
            rendered,
            add_special_tokens=False,
        )
    ]
    template_ids = judge.tokenizer.apply_chat_template(
        judge.profile.normalize_messages(messages),
        tokenize=True,
        add_generation_prompt=True,
    )
    if not isinstance(template_ids, Sequence) or isinstance(
        template_ids,
        (str, bytes),
    ):
        raise TypeError("chat template tokenization did not return token IDs")
    expected = [int(token_id) for token_id in template_ids]
    if judge.profile.assistant_prefill:
        expected.extend(
            int(token_id)
            for token_id in judge.tokenizer.encode(
                judge.profile.assistant_prefill,
                add_special_tokens=False,
            )
        )
    if transport_ids != expected:
        raise ValueError("rendered text transport differs from template token IDs")
    return transport_ids


def _render_candidate(
    *,
    stage: str,
    routing_split: str,
    planned: PlannedCondition,
    example: JudgeExample,
    judge: JudgeBackend,
    max_model_len: int,
    required_completion_tokens: int,
) -> CandidatePrompt:
    messages = build_social_cue_messages(
        example=example,
        condition=planned.condition,
        output_mode=OutputMode.CHOICE_ONLY,
    )
    rendered = judge.render_messages(messages)
    package = build_social_cue_prompt_package(
        example=example,
        condition=planned.condition,
        output_mode=OutputMode.CHOICE_ONLY,
        renderer=lambda _: rendered,
    )
    token_ids = _transport_token_ids(judge, messages, package.prompt_text)
    if len(token_ids) + required_completion_tokens > max_model_len:
        raise ValueError(
            "candidate prompt leaves insufficient generation headroom: "
            f"input={len(token_ids)} required_completion={required_completion_tokens} "
            f"max_model_len={max_model_len}"
        )
    return CandidatePrompt(
        stage=stage,
        routing_split=routing_split,
        planned=planned,
        example=example,
        prompt_text=package.prompt_text,
        prompt_hash=package.prompt_hash,
        prompt_sha256=text_sha256(package.prompt_text),
        token_ids_sha256=value_sha256(token_ids),
        input_tokens=len(token_ids),
    )


def _longest(candidates: Sequence[CandidatePrompt]) -> CandidatePrompt:
    if not candidates:
        raise ValueError("required small-grid stratum has no candidates")
    return sorted(candidates, key=lambda item: (-item.input_tokens, item.key))[0]


def _family_boundary_doses(family: str) -> tuple[int, int]:
    doses = (
        AUTHORITY_DOSES
        if family == BiasType.AUTHORITY.value
        else BANDWAGON_DOSES
    )
    return min(doses), max(doses)


def select_small_grid(
    candidates: Sequence[CandidatePrompt],
) -> tuple[CandidatePrompt, ...]:
    selected: list[CandidatePrompt] = []
    for ordering in (PairOrdering.AB.value, PairOrdering.BA.value):
        selected.append(
            _longest(
                [
                    item
                    for item in candidates
                    if item.stage == "stage_a" and item.ordering == ordering
                ]
            )
        )

    for family in (BiasType.AUTHORITY.value, BiasType.BANDWAGON.value):
        for direction in (
            CueCongruency.CONGRUENT.value,
            CueCongruency.INCONGRUENT.value,
        ):
            for dose in _family_boundary_doses(family):
                for ordering in (PairOrdering.AB.value, PairOrdering.BA.value):
                    selected.append(
                        _longest(
                            [
                                item
                                for item in candidates
                                if item.stage == "stage_b"
                                and item.family == family
                                and item.direction == direction
                                and item.planned.condition.dose == dose
                                and item.ordering == ordering
                            ]
                        )
                    )
    if len(selected) != 18 or len({item.key for item in selected}) != 18:
        raise AssertionError("small-grid selection must contain 18 unique prompts")
    return tuple(sorted(selected, key=lambda item: item.key))


def build_small_grid_plan(
    *,
    source_csv: Path,
    canonical_model_name: str,
    judge: JudgeBackend,
    max_model_len: int,
    required_completion_tokens: int,
) -> SmallGridPlan:
    source_sha = file_sha256(source_csv)
    frame = pd.read_csv(source_csv)
    pairs, eligibility = load_position_pairs_with_eligibility(source_csv)
    if eligibility.raw_row_count != len(frame):
        raise ValueError("pair loader did not examine every frozen source row")
    if eligibility.eligible_pair_count != len(pairs):
        raise AssertionError("pair-loader eligible count differs from loaded pairs")
    if eligibility.raw_row_count != (
        eligibility.eligible_pair_count + eligibility.skipped_row_count
    ):
        raise AssertionError("pair-loader eligibility accounting is inconsistent")
    if not pairs:
        raise ValueError("frozen source contains no eligible pairs")

    pair_inputs: list[StageAPairInput] = []
    pairs_by_identity: dict[str, Any] = {}
    for pair in pairs:
        original = pair.original
        pair_input = StageAPairInput(
            dataset_name=source_csv.name,
            input_file_hash=source_sha,
            source_row_index=original.metadata["source_row_index"],
            question_id=(
                original.metadata.get("question_cluster_id")
                or original.question_id
            ),
            model_name=canonical_model_name,
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
        if identity in pairs_by_identity:
            raise ValueError("source contains duplicate pair identities")
        pair_inputs.append(pair_input)
        pairs_by_identity[identity] = pair

    stage_a = generate_stage_a_conditions(pair_inputs)
    if stage_a.issues:
        raise ValueError("production Stage A planner reported issues")
    if len(stage_a.conditions) != 2 * len(pairs):
        raise ValueError("production Stage A planner did not emit AB and BA")

    candidates: list[CandidatePrompt] = []
    provisional_summaries: list[CleanPairSummary] = []
    test_pair_identities: set[str] = set()
    for planned in stage_a.conditions:
        pair = pairs_by_identity[planned.pair_identity_key]
        ordering = normalize_ordering(planned.condition.ordering or "")
        example = pair.original if ordering == PairOrdering.AB else pair.swapped
        routing_split = str(example.metadata.get("routing_split") or "").lower()
        if routing_split not in {"calibration", "test"}:
            raise ValueError("every eligible pair must carry a frozen routing split")
        candidates.append(
            _render_candidate(
                stage="stage_a",
                routing_split=routing_split,
                planned=planned,
                example=example,
                judge=judge,
                max_model_len=max_model_len,
                required_completion_tokens=required_completion_tokens,
            )
        )
        if routing_split == "test":
            test_pair_identities.add(planned.pair_identity_key)
            human_winner = str(planned.condition.metadata["human_winner"])
            provisional_summaries.append(
                CleanPairSummary(
                    pair_identity_key=planned.pair_identity_key,
                    pair_key=planned.pair_key,
                    ordering=ordering,
                    ordering_twin_key=planned.ordering_twin_key,
                    model_name=planned.model_name,
                    input_file_hash=planned.input_file_hash,
                    clean_record_id=f"preflight-only:{planned.pair_key}",
                    clean_verdict=human_winner,
                    human_winner=human_winner,
                    routing_split="test",
                )
            )

    stage_b = generate_stage_b_conditions(provisional_summaries)
    fatal_issues = [
        issue for issue in stage_b.issues if issue.code != "clean_and_human_tie"
    ]
    if fatal_issues:
        raise ValueError("production Stage B planner reported fatal issues")
    if len(stage_b.conditions) != 16 * len(provisional_summaries):
        raise ValueError("production Stage B planner did not emit the full grid")
    for planned in stage_b.conditions:
        pair = pairs_by_identity[planned.pair_identity_key]
        ordering = normalize_ordering(planned.condition.ordering or "")
        example = pair.original if ordering == PairOrdering.AB else pair.swapped
        candidates.append(
            _render_candidate(
                stage="stage_b",
                routing_split="test",
                planned=planned,
                example=example,
                judge=judge,
                max_model_len=max_model_len,
                required_completion_tokens=required_completion_tokens,
            )
        )

    eligible_questions: dict[str, set[str]] = {
        "calibration": set(),
        "test": set(),
    }
    for pair in pairs:
        split = str(pair.original.metadata.get("routing_split") or "").lower()
        question_id = str(
            pair.original.metadata.get("question_cluster_id")
            or pair.original.question_id
        )
        if split not in eligible_questions:
            raise ValueError("eligible pair has an invalid routing split")
        eligible_questions[split].add(question_id)
    return SmallGridPlan(
        candidates=tuple(candidates),
        selected=select_small_grid(candidates),
        source_pair_count=len(pairs),
        source_row_count=len(frame),
        test_pair_count=len(test_pair_identities),
        eligibility_audit=eligibility.to_dict(),
        eligible_question_counts={
            "total": len(eligible_questions["calibration"] | eligible_questions["test"]),
            "calibration": len(eligible_questions["calibration"]),
            "test": len(eligible_questions["test"]),
            "overlap": len(eligible_questions["calibration"] & eligible_questions["test"]),
        },
        provisional_tie_issue_count=sum(
            issue.code == "clean_and_human_tie" for issue in stage_b.issues
        ),
    )


def validate_verdict_token_contract(judge: JudgeBackend) -> dict[str, Any]:
    observed_texts = {
        label: tuple(judge.profile.verdict_token_texts.get(label, ()))
        for label in EXPECTED_VERDICT_SURFACES
    }
    if observed_texts != dict(EXPECTED_VERDICT_SURFACES):
        raise ValueError("small-grid smoke requires literal singleton A/B/T tokens")
    resolved = {
        label: [int(token_id) for token_id in judge.decision_label_token_ids[label]]
        for label in EXPECTED_VERDICT_SURFACES
    }
    flattened = [token_id for values in resolved.values() for token_id in values]
    if any(len(values) != 1 for values in resolved.values()):
        raise ValueError("each verdict surface must resolve to exactly one token")
    if len(set(flattened)) != 3:
        raise ValueError("A/B/T must resolve to distinct token IDs")
    for label, surfaces in EXPECTED_VERDICT_SURFACES.items():
        token_id = resolved[label][0]
        encoded = judge.tokenizer.encode(surfaces[0], add_special_tokens=False)
        decoded = judge.tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if [int(value) for value in encoded] != [token_id] or decoded != surfaces[0]:
            raise ValueError("verdict token failed exact encode/decode round trip")
    return {
        "verdict_token_texts": {
            label: list(surfaces)
            for label, surfaces in EXPECTED_VERDICT_SURFACES.items()
        },
        "verdict_token_ids": resolved,
        "distinct_token_ids": True,
        "exact_round_trip": True,
    }


def _result_fingerprints(
    results: Sequence[tuple[VerdictLabel, str, dict[str, float]]],
) -> list[dict[str, Any]]:
    return [
        {
            "row_index": index,
            "verdict": enum_value(verdict),
            "raw_output_sha256": text_sha256(raw_output),
            "raw_output_character_count": len(raw_output),
            "probabilities_sha256": value_sha256(probabilities),
        }
        for index, (verdict, raw_output, probabilities) in enumerate(results)
    ]


def _deterministic_result_hashes(
    results: Sequence[tuple[VerdictLabel, str, dict[str, float]]],
    *,
    tokenizer: Any,
) -> dict[str, Any]:
    verdicts = [enum_value(verdict) for verdict, _, _ in results]
    raw_outputs = [raw_output for _, raw_output, _ in results]
    token_ids = [
        [
            int(token_id)
            for token_id in tokenizer.encode(
                raw_output,
                add_special_tokens=False,
            )
        ]
        for _, raw_output, _ in results
    ]
    probabilities = [probability for _, _, probability in results]
    return {
        "received_examples": len(results),
        "verdicts_sha256": value_sha256(verdicts),
        "raw_outputs_sha256": value_sha256(raw_outputs),
        "raw_output_token_ids_sha256": value_sha256(token_ids),
        "probabilities_sha256": value_sha256(probabilities),
    }


def validate_deterministic_replay_contract(
    first: Sequence[tuple[VerdictLabel, str, dict[str, float]]],
    second: Sequence[tuple[VerdictLabel, str, dict[str, float]]],
    *,
    tokenizer: Any,
    expected: int,
    probability_atol: float = DETERMINISTIC_PROBABILITY_ATOL,
) -> dict[str, Any]:
    """Require exact deterministic decisions/tokens and tightly matching scores."""

    if not math.isfinite(probability_atol) or probability_atol < 0.0:
        raise ValueError("deterministic probability tolerance must be finite/nonnegative")
    first_hashes = _deterministic_result_hashes(first, tokenizer=tokenizer)
    second_hashes = _deterministic_result_hashes(second, tokenizer=tokenizer)
    exact_verdict_match = (
        first_hashes["verdicts_sha256"] == second_hashes["verdicts_sha256"]
    )
    exact_raw_output_match = (
        first_hashes["raw_outputs_sha256"]
        == second_hashes["raw_outputs_sha256"]
    )
    exact_token_match = (
        first_hashes["raw_output_token_ids_sha256"]
        == second_hashes["raw_output_token_ids_sha256"]
    )
    support = {"A", "B", "tie"}
    probabilities_valid = len(first) == expected and len(second) == expected
    maximum_difference = 0.0
    if probabilities_valid:
        for first_row, second_row in zip(first, second):
            first_probabilities = first_row[2]
            second_probabilities = second_row[2]
            if (
                set(first_probabilities) != support
                or set(second_probabilities) != support
            ):
                probabilities_valid = False
                break
            for label in sorted(support):
                first_value = first_probabilities[label]
                second_value = second_probabilities[label]
                if not (
                    isinstance(first_value, (int, float))
                    and not isinstance(first_value, bool)
                    and isinstance(second_value, (int, float))
                    and not isinstance(second_value, bool)
                    and math.isfinite(float(first_value))
                    and math.isfinite(float(second_value))
                ):
                    probabilities_valid = False
                    break
                maximum_difference = max(
                    maximum_difference,
                    abs(float(first_value) - float(second_value)),
                )
            if not probabilities_valid:
                break
    probability_match = (
        probabilities_valid and maximum_difference <= probability_atol
    )
    passed = (
        len(first) == expected
        and len(second) == expected
        and exact_verdict_match
        and exact_raw_output_match
        and exact_token_match
        and probability_match
    )
    return {
        "passed": passed,
        "required_passes": DETERMINISTIC_PASSES,
        "seed": 0,
        "sampling_temperature": 0.0,
        "expected_examples_per_pass": expected,
        "received_examples_by_pass": [len(first), len(second)],
        "exact_verdict_hash_match": exact_verdict_match,
        "exact_raw_output_hash_match": exact_raw_output_match,
        "exact_raw_output_token_hash_match": exact_token_match,
        "probability_absolute_tolerance": probability_atol,
        "probability_support_valid": probabilities_valid,
        "maximum_absolute_probability_difference": (
            maximum_difference if probabilities_valid else None
        ),
        "probabilities_within_tolerance": probability_match,
        "passes": [first_hashes, second_hashes],
    }


def classify_production_throughput(
    sequences_per_second: float | None,
) -> dict[str, Any]:
    """Classify a measured production-equivalent rate as a release gate."""

    observed: float | None
    try:
        observed = (
            None if sequences_per_second is None else float(sequences_per_second)
        )
    except (TypeError, ValueError):
        observed = None
    valid = observed is not None and math.isfinite(observed) and observed > 0.0
    minimum_met = bool(
        valid and observed >= MINIMUM_PRODUCTION_SEQUENCES_PER_SECOND
    )
    release_target_met = bool(
        valid and observed >= RELEASE_TARGET_PRODUCTION_SEQUENCES_PER_SECOND
    )
    if release_target_met:
        status = "release_target_met"
    elif minimum_met:
        status = "minimum_met_release_target_missed"
    elif valid:
        status = "below_minimum"
    else:
        status = "invalid_or_unavailable"
    return {
        "passed": release_target_met,
        "status": status,
        "observed_sequences_per_second": observed if valid else None,
        "minimum_sequences_per_second": MINIMUM_PRODUCTION_SEQUENCES_PER_SECOND,
        "release_target_sequences_per_second": (
            RELEASE_TARGET_PRODUCTION_SEQUENCES_PER_SECOND
        ),
        "minimum_met": minimum_met,
        "release_target_met": release_target_met,
    }


def choose_verdict_in_batches(
    judge: JudgeBackend,
    prompts: Sequence[str],
    *,
    batch_size: int,
    seed: int,
    sampling_temperature: float,
) -> list[tuple[VerdictLabel, str, dict[str, float]]]:
    results: list[tuple[VerdictLabel, str, dict[str, float]]] = []
    for start in range(0, len(prompts), batch_size):
        results.extend(
            judge.choose_verdict_batch(
                list(prompts[start : start + batch_size]),
                seed=seed,
                sampling_temperature=sampling_temperature,
            )
        )
    return results


def validate_constrained_parse_contract(
    results: Sequence[tuple[VerdictLabel, str, dict[str, float]]],
    *,
    expected: int,
) -> dict[str, Any]:
    parsed = 0
    mismatches = 0
    for verdict, raw_output, _ in results:
        raw_verdict = VLLMJudge._parse_verdict_text(raw_output)
        if raw_verdict is not None:
            parsed += 1
        if raw_verdict != verdict:
            mismatches += 1
    passed = len(results) == expected and parsed == expected and mismatches == 0
    return {
        "passed": passed,
        "expected_examples": expected,
        "received_examples": len(results),
        "parseable_examples": parsed,
        "exact_verdict_matches": len(results) - mismatches,
        "parse_rate": parsed / expected,
        "parser_version": JUDGE_OUTPUT_PARSER_VERSION,
    }


def validate_probability_contract(
    results: Sequence[tuple[VerdictLabel, str, dict[str, float]]],
    *,
    expected: int,
    require_map_alignment: bool,
) -> dict[str, Any]:
    exact_support = 0
    normalized = 0
    finite_nonnegative = 0
    map_aligned = 0
    label_map = {
        "A": VerdictLabel.A,
        "B": VerdictLabel.B,
        "tie": VerdictLabel.TIE,
    }
    for verdict, _, probabilities in results:
        if set(probabilities) != set(label_map):
            continue
        exact_support += 1
        values = list(probabilities.values())
        valid_values = all(
            math.isfinite(value) and value >= 0.0 for value in values
        )
        if valid_values:
            finite_nonnegative += 1
        if valid_values and math.isclose(
            sum(values),
            1.0,
            rel_tol=1e-7,
            abs_tol=1e-7,
        ):
            normalized += 1
        if valid_values and label_map[max(probabilities, key=probabilities.get)] == verdict:
            map_aligned += 1
    passed = (
        len(results) == expected
        and exact_support == expected
        and finite_nonnegative == expected
        and normalized == expected
        and (not require_map_alignment or map_aligned == expected)
    )
    return {
        "passed": passed,
        "expected_examples": expected,
        "received_examples": len(results),
        "exact_support_examples": exact_support,
        "finite_nonnegative_examples": finite_nonnegative,
        "normalized_examples": normalized,
        "map_aligned_examples": map_aligned,
        "map_alignment_required": require_map_alignment,
        "logprobs_mode": CONSTRAINED_LOGPROBS_MODE,
    }


def generate_native_vllm(
    judge: JudgeBackend,
    prompts: Sequence[str],
    seed: int,
    max_tokens: int,
) -> Sequence[NativeGeneration]:
    if SamplingParams is None:
        raise RuntimeError("native generation requires vLLM")
    model = getattr(judge, "model", None)
    prepare_prompt = getattr(judge, "_prepare_prompt", None)
    if model is None or not callable(prepare_prompt):
        raise TypeError("native generation requires the production VLLMJudge")
    sampling = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        seed=seed,
        stop=list(judge.profile.stop_token_texts) or None,
        skip_special_tokens=True,
    )
    outputs = model.generate(
        [prepare_prompt(prompt) for prompt in prompts],
        sampling,
        use_tqdm=False,
    )
    generations: list[NativeGeneration] = []
    for output in outputs:
        completion = output.outputs[0]
        token_ids = getattr(completion, "token_ids", ()) or ()
        generations.append(
            NativeGeneration(
                text=str(completion.text),
                token_ids=tuple(int(token_id) for token_id in token_ids),
            )
        )
    return generations


def validate_native_contract(
    native: Sequence[NativeGeneration],
    constrained: Sequence[tuple[VerdictLabel, str, dict[str, float]]],
    *,
    allowed_token_ids: Sequence[int],
    expected: int,
) -> dict[str, Any]:
    allowed = {int(token_id) for token_id in allowed_token_ids}
    parseable = 0
    first_token_exact = 0
    verdict_matches = 0
    complete = 0
    fingerprints: list[dict[str, Any]] = []
    for index, generation in enumerate(native):
        verdict = VLLMJudge._parse_verdict_text(generation.text)
        compatible = bool(generation.token_ids and generation.token_ids[0] in allowed)
        matches = bool(
            verdict is not None
            and index < len(constrained)
            and verdict == constrained[index][0]
        )
        parseable += verdict is not None
        first_token_exact += compatible
        verdict_matches += matches
        complete += bool(verdict is not None and compatible and matches)
        fingerprints.append(
            {
                "row_index": index,
                "output_sha256": text_sha256(generation.text),
                "output_character_count": len(generation.text),
                "generated_token_count": len(generation.token_ids),
                "first_token_id": generation.token_ids[0] if generation.token_ids else None,
                "parsed_verdict": enum_value(verdict) if verdict is not None else None,
                "first_token_compatible": compatible,
            }
        )
    passed = (
        len(native) == expected
        and len(constrained) == expected
        and complete == expected
    )
    return {
        "passed": passed,
        "expected_examples": expected,
        "received_examples": len(native),
        "parseable_examples": parseable,
        "first_token_compatible_examples": first_token_exact,
        "verdict_agreement_examples": verdict_matches,
        "complete_contract_examples": complete,
        "required_contract_rate": 1.0,
        "contract_rate": complete / expected,
        "output_set_sha256": value_sha256(fingerprints),
        "output_fingerprints": fingerprints,
    }


def _candidate_record(item: CandidatePrompt) -> dict[str, Any]:
    condition = item.planned.condition
    return {
        "key": item.key,
        "stage": item.stage,
        "routing_split": item.routing_split,
        "pair_key": item.planned.pair_key,
        "pair_identity_key": item.planned.pair_identity_key,
        "variant_id": condition.variant_id,
        "family": item.family,
        "direction": item.direction,
        "dose": condition.dose,
        "ordering": item.ordering,
        "prompt_hash": item.prompt_hash,
        "prompt_sha256": item.prompt_sha256,
        "token_ids_sha256": item.token_ids_sha256,
        "input_tokens": item.input_tokens,
    }


def _implementation_hashes() -> dict[str, str]:
    modules = {
        "dataset_splits": dataset_splits_module,
        "models": models_module,
        "pairing": pairing_module,
        "position_bias": position_bias_module,
        "safe_diagnostics": safe_diagnostics_module,
        "schemas": schemas_module,
        "social_cue_prompts": prompts_module,
        "stage_planning": planning_module,
    }
    result: dict[str, str] = {}
    for name, module in modules.items():
        module_path = Path(str(module.__file__ or ""))
        if not module_path.is_file():
            raise RuntimeError(f"cannot hash implementation module {name}")
        result[name] = file_sha256(module_path)
    result["smoke_script"] = file_sha256(Path(__file__).resolve())
    return result


def run_small_grid_preflight(
    *,
    source_csv: Path,
    routing_manifest_path: Path,
    runtime_path: Path,
    model_name: str,
    judge: JudgeBackend,
    native_generator: NativeGenerator = generate_native_vllm,
    judge_initialization_seconds: float = 0.0,
    perform_inference: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    source_sha = file_sha256(source_csv)
    routing_file = read_json_mapping(routing_manifest_path, name="routing manifest")
    runtime_file = read_json_mapping(runtime_path, name="runtime mapping")
    runtime = validate_runtime(runtime_file)
    frame = pd.read_csv(source_csv)
    observed_routing = validate_routing_contract(
        frame,
        source_csv,
        source_sha,
        routing_file,
        manifest_path=routing_manifest_path,
    )
    profile = validate_runtime_model_contract(runtime, model_name=model_name)
    if judge.profile != profile or judge.model_name != profile.hf_model_name:
        raise ValueError("judge does not match the requested model profile")
    if judge.logprobs_mode != CONSTRAINED_LOGPROBS_MODE:
        raise ValueError("judge must use constrained processed log probabilities")
    token_contract = validate_verdict_token_contract(judge)

    planning_started = time.perf_counter()
    plan = build_small_grid_plan(
        source_csv=source_csv,
        canonical_model_name=profile.hf_model_name,
        judge=judge,
        max_model_len=runtime["max_model_len"],
        required_completion_tokens=NATIVE_MAX_TOKENS,
    )
    validate_eligibility_contract(plan, routing_file, observed_routing)
    planning_seconds = time.perf_counter() - planning_started
    selected_records = [_candidate_record(item) for item in plan.selected]
    selection_counts = {
        "stage": dict(sorted(Counter(item.stage for item in plan.selected).items())),
        "family": dict(sorted(Counter(item.family for item in plan.selected).items())),
        "direction": dict(sorted(Counter(item.direction for item in plan.selected).items())),
        "ordering": dict(sorted(Counter(item.ordering for item in plan.selected).items())),
        "dose": dict(
            sorted(
                Counter(str(item.planned.condition.dose) for item in plan.selected).items()
            )
        ),
    }
    base_report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "planning_complete" if not perform_inference else "running",
        "passed": not perform_inference,
        "inference_performed": False,
        "scientific_result": False,
        "excluded_methods": ["BPE", "SCOPE"],
        "selection_policy": (
            "longest rendered prompt per required stage/family/direction/"
            "boundary-dose/ordering stratum; lexical key breaks token-length ties"
        ),
        "stage_b_reference_policy": (
            "human winner used as provisional clean reference for infrastructure "
            "smoke only"
        ),
        "source": {
            "path": str(source_csv),
            "sha256": source_sha,
            "row_count": len(frame),
            "eligible_pair_count": plan.source_pair_count,
            "skipped_row_count": int(plan.eligibility_audit["skipped_row_count"]),
            "skipped_reason_counts": dict(
                plan.eligibility_audit["skipped_reason_counts"]
            ),
            "eligibility_sha256": plan.eligibility_sha256,
            "eligibility_audit": dict(plan.eligibility_audit),
        },
        "routing": {
            "manifest_path": str(routing_manifest_path),
            "manifest_file_sha256": file_sha256(routing_manifest_path),
            "routing_assignment_sha256": observed_routing[
                "routing_assignment_sha256"
            ],
            "row_counts": observed_routing["row_counts"],
            "question_counts": observed_routing["question_counts"],
            "seed": observed_routing["seed"],
            "calibration_fraction": observed_routing["calibration_fraction"],
        },
        "model": {
            "registry_name": profile.registry_name,
            "model_name": profile.hf_model_name,
            "revision": profile.revision,
            "family": profile.family,
            "trust_remote_code": profile.trust_remote_code,
            **token_contract,
        },
        "runtime": {
            "path": str(runtime_path),
            "file_sha256": file_sha256(runtime_path),
            "canonical_sha256": value_sha256(runtime),
            "controls": runtime,
            "smoke_only_controls": {
                "native_max_tokens": NATIVE_MAX_TOKENS,
            },
        },
        "implementation_sha256": _implementation_hashes(),
        "plan": {
            "candidate_count": len(plan.candidates),
            "stage_a_candidate_count": sum(
                item.stage == "stage_a" for item in plan.candidates
            ),
            "stage_b_candidate_count": sum(
                item.stage == "stage_b" for item in plan.candidates
            ),
            "test_pair_count": plan.test_pair_count,
            "provisional_tie_issue_count": plan.provisional_tie_issue_count,
            "selected_count": len(plan.selected),
            "selected_counts": selection_counts,
            "selected_prompt_set_sha256": value_sha256(selected_records),
            "selected_prompts": selected_records,
            "maximum_selected_input_tokens": max(
                item.input_tokens for item in plan.selected
            ),
        },
        "validation": {
            "deterministic_passes": {"status": "not_run"},
            "deterministic_replay_contract": {"status": "not_run"},
            "constrained_parse_contract": {"status": "not_run"},
            "probability_contract": {"status": "not_run"},
            "repeatability_contract": {"status": "not_run"},
            "native_verdict_token_contract": {"status": "not_run"},
            "production_throughput_gate": {"status": "not_run"},
        },
        "execution": {
            "phase": "planning_complete",
            "deterministic_required_passes": DETERMINISTIC_PASSES,
            "deterministic_completed_passes": 0,
            "deterministic_received_examples": 0,
            "repeatability_completed_passes": 0,
            "repeatability_received_examples": 0,
            "native_received_examples": 0,
        },
        "timing": {
            "judge_initialization_seconds": judge_initialization_seconds,
            "planning_seconds": planning_seconds,
        },
    }
    model_contract = {
        key: base_report["model"][key]
        for key in (
            "registry_name",
            "model_name",
            "revision",
            "family",
            "trust_remote_code",
            "verdict_token_texts",
            "verdict_token_ids",
        )
    }
    base_report["model"]["contract_sha256"] = value_sha256(model_contract)
    if not perform_inference:
        base_report["timing"]["total_wall_seconds"] = time.perf_counter() - started
        return base_report

    expected = len(plan.selected)
    prompts = [item.prompt_text for item in plan.selected]
    constrained: list[tuple[VerdictLabel, str, dict[str, float]]] = []
    deterministic_results: list[
        list[tuple[VerdictLabel, str, dict[str, float]]]
    ] = []
    deterministic_passes: list[dict[str, Any]] = []
    deterministic_pass_seconds: list[float] = []
    native: list[NativeGeneration] = []
    repeatability_passes: list[dict[str, Any]] = []
    constrained_seconds = 0.0
    repeatability_seconds = 0.0
    native_seconds = 0.0
    inference_started = False
    try:
        base_report["execution"]["phase"] = "deterministic_inference_started"
        inference_started = True
        for pass_index in range(DETERMINISTIC_PASSES):
            pass_started = time.perf_counter()
            deterministic = choose_verdict_in_batches(
                judge,
                prompts,
                batch_size=runtime["batch_size"],
                seed=runtime["seed"],
                sampling_temperature=0.0,
            )
            pass_seconds = time.perf_counter() - pass_started
            deterministic_results.append(deterministic)
            deterministic_pass_seconds.append(pass_seconds)
            pass_parse = validate_constrained_parse_contract(
                deterministic,
                expected=expected,
            )
            pass_probability = validate_probability_contract(
                deterministic,
                expected=expected,
                require_map_alignment=True,
            )
            deterministic_passes.append(
                {
                    "pass_index": pass_index + 1,
                    "seed": runtime["seed"],
                    "sampling_temperature": 0.0,
                    "passed": pass_parse["passed"]
                    and pass_probability["passed"],
                    "received_examples": len(deterministic),
                    "parse_contract": pass_parse,
                    "probability_contract": pass_probability,
                    "result_hashes": _deterministic_result_hashes(
                        deterministic,
                        tokenizer=judge.tokenizer,
                    ),
                    "inference_seconds": pass_seconds,
                    "sequences_per_second": (
                        expected / pass_seconds if pass_seconds else None
                    ),
                }
            )
            base_report["execution"]["deterministic_completed_passes"] = len(
                deterministic_results
            )
            base_report["execution"]["deterministic_received_examples"] += len(
                deterministic
            )
        constrained = deterministic_results[0]
        constrained_seconds = deterministic_pass_seconds[0]
        base_report["execution"]["phase"] = "repeatability_inference_started"
        for run_seed in range(runtime["consistency_runs"]):
            pass_started = time.perf_counter()
            sampled = choose_verdict_in_batches(
                judge,
                prompts,
                batch_size=runtime["batch_size"],
                seed=run_seed,
                sampling_temperature=runtime["sampling_temperature"],
            )
            pass_seconds = time.perf_counter() - pass_started
            repeatability_seconds += pass_seconds
            pass_parse = validate_constrained_parse_contract(
                sampled,
                expected=expected,
            )
            pass_probability = validate_probability_contract(
                sampled,
                expected=expected,
                require_map_alignment=False,
            )
            pass_fingerprints = _result_fingerprints(sampled)
            repeatability_passes.append(
                {
                    "seed": run_seed,
                    "sampling_temperature": runtime["sampling_temperature"],
                    "passed": pass_parse["passed"] and pass_probability["passed"],
                    "received_examples": len(sampled),
                    "parse_contract": pass_parse,
                    "probability_contract": pass_probability,
                    "result_set_sha256": value_sha256(pass_fingerprints),
                    "result_fingerprints": pass_fingerprints,
                    "inference_seconds": pass_seconds,
                    "sequences_per_second": (
                        expected / pass_seconds if pass_seconds else None
                    ),
                }
            )
            base_report["execution"]["repeatability_completed_passes"] = len(
                repeatability_passes
            )
            base_report["execution"]["repeatability_received_examples"] += len(
                sampled
            )
        base_report["execution"]["phase"] = "native_infrastructure_check_started"
        native_started = time.perf_counter()
        native = list(
            native_generator(
                judge,
                prompts,
                runtime["seed"],
                NATIVE_MAX_TOKENS,
            )
        )
        native_seconds = time.perf_counter() - native_started
        base_report["execution"]["native_received_examples"] = len(native)
        base_report["execution"]["phase"] = "validation_started"
        parse_contract = deterministic_passes[0]["parse_contract"]
        probability_contract = deterministic_passes[0]["probability_contract"]
        deterministic_replay_contract = validate_deterministic_replay_contract(
            deterministic_results[0],
            deterministic_results[1],
            tokenizer=judge.tokenizer,
            expected=expected,
        )
        repeatability_contract = {
            "passed": (
                len(repeatability_passes) == runtime["consistency_runs"]
                and all(item["passed"] for item in repeatability_passes)
                and sum(
                    int(item["received_examples"])
                    for item in repeatability_passes
                )
                == expected * runtime["consistency_runs"]
            ),
            "consistency_runs": runtime["consistency_runs"],
            "consistency_schedule": runtime["consistency_schedule"],
            "sampling_temperature": runtime["sampling_temperature"],
            "include_verbalized_confidence": runtime[
                "include_verbalized_confidence"
            ],
            "expected_examples_per_pass": expected,
            "expected_total_examples": expected * runtime["consistency_runs"],
            "received_total_examples": sum(
                int(item["received_examples"]) for item in repeatability_passes
            ),
            "result_set_sha256": value_sha256(
                [
                    {
                        "seed": item["seed"],
                        "result_set_sha256": item["result_set_sha256"],
                    }
                    for item in repeatability_passes
                ]
            ),
            "passes": repeatability_passes,
        }
        native_contract = validate_native_contract(
            native,
            constrained,
            allowed_token_ids=judge.decision_allowed_token_ids,
            expected=expected,
        )
        structural_passed = all(
            contract["passed"]
            for contract in (
                parse_contract,
                probability_contract,
                deterministic_replay_contract,
                repeatability_contract,
                native_contract,
            )
        ) and all(item["passed"] for item in deterministic_passes)
    except Exception as error:
        deterministic_seconds = sum(deterministic_pass_seconds)
        base_report["timing"].update(
            {
                "constrained_inference_seconds": constrained_seconds,
                "deterministic_inference_seconds": deterministic_seconds,
                "repeatability_inference_seconds": repeatability_seconds,
                "native_inference_seconds": native_seconds,
                "total_wall_seconds": time.perf_counter() - started,
            }
        )
        return failure_report(
            error,
            model_name=model_name,
            base_report=base_report,
            inference_performed=inference_started,
        )

    fingerprints = _result_fingerprints(constrained)
    deterministic_seconds = sum(deterministic_pass_seconds)
    deterministic_mean_pass_seconds = (
        deterministic_seconds / len(deterministic_pass_seconds)
    )
    production_inference_seconds = (
        deterministic_mean_pass_seconds + repeatability_seconds
    )
    production_sequence_count = expected * (1 + runtime["consistency_runs"])
    production_sequences_per_second = (
        production_sequence_count / production_inference_seconds
        if production_inference_seconds
        else None
    )
    production_throughput_gate = classify_production_throughput(
        production_sequences_per_second
    )
    passed = structural_passed and production_throughput_gate["passed"]
    total_inference_seconds = (
        deterministic_seconds + repeatability_seconds + native_seconds
    )
    production_benchmark_sequence_count = expected * (
        DETERMINISTIC_PASSES + runtime["consistency_runs"]
    )
    production_benchmark_seconds = deterministic_seconds + repeatability_seconds
    all_smoke_sequence_count = expected * (
        DETERMINISTIC_PASSES + runtime["consistency_runs"] + 1
    )
    base_report.update(
        {
            "status": "complete" if passed else "failed_validation",
            "passed": passed,
            "inference_performed": True,
            "result_set_sha256": value_sha256(fingerprints),
            "result_fingerprints": fingerprints,
            "validation": {
                "deterministic_passes": deterministic_passes,
                "deterministic_replay_contract": deterministic_replay_contract,
                "constrained_parse_contract": parse_contract,
                "probability_contract": probability_contract,
                "repeatability_contract": repeatability_contract,
                "native_verdict_token_contract": native_contract,
                "production_throughput_gate": production_throughput_gate,
            },
        }
    )
    base_report["execution"]["phase"] = "complete"
    base_report["timing"].update(
        {
            "constrained_inference_seconds": constrained_seconds,
            "deterministic_inference_seconds": deterministic_seconds,
            "deterministic_mean_pass_seconds": deterministic_mean_pass_seconds,
            "repeatability_inference_seconds": repeatability_seconds,
            "native_inference_seconds": native_seconds,
            "total_inference_seconds": total_inference_seconds,
            "constrained_examples_per_second": (
                expected / constrained_seconds if constrained_seconds else None
            ),
            "native_examples_per_second": (
                expected / native_seconds if native_seconds else None
            ),
            "repeatability_sequences_per_second": (
                (expected * runtime["consistency_runs"]) / repeatability_seconds
                if repeatability_seconds
                else None
            ),
            "production_sequence_count": production_sequence_count,
            "production_inference_seconds": production_inference_seconds,
            "production_sequences_per_second": production_sequences_per_second,
            "production_benchmark_sequence_count": (
                production_benchmark_sequence_count
            ),
            "production_benchmark_inference_seconds": (
                production_benchmark_seconds
            ),
            "production_benchmark_sequences_per_second": (
                production_benchmark_sequence_count / production_benchmark_seconds
                if production_benchmark_seconds
                else None
            ),
            "native_infrastructure_sequence_count": expected,
            "all_smoke_sequence_count": all_smoke_sequence_count,
            "all_smoke_sequences_per_second": (
                all_smoke_sequence_count / total_inference_seconds
                if total_inference_seconds
                else None
            ),
            "total_wall_seconds": time.perf_counter() - started,
        }
    )
    return base_report


def configure_runtime_environment(runtime: Mapping[str, Any]) -> None:
    os.environ["BIASES_VLLM_ENFORCE_EAGER"] = (
        "1" if runtime["enforce_eager"] else "0"
    )
    os.environ["VLLM_DISABLE_CUSTOM_ALL_REDUCE"] = (
        "1" if runtime["disable_custom_all_reduce"] else "0"
    )


def build_vllm_judge(model_name: str, runtime: Mapping[str, Any]) -> VLLMJudge:
    validate_runtime_model_contract(runtime, model_name=model_name)
    configure_runtime_environment(runtime)
    return VLLMJudge(
        model_name=model_name,
        tensor_parallel_size=runtime["tensor_parallel_size"],
        max_model_len=runtime["max_model_len"],
        gpu_memory_utilization=runtime["gpu_memory_utilization"],
        dtype=runtime["dtype"],
        max_num_batched_tokens=runtime["max_num_batched_tokens"],
        max_num_seqs=runtime["max_num_seqs"],
        enforce_eager=runtime["enforce_eager"],
        disable_custom_all_reduce=runtime["disable_custom_all_reduce"],
    )


def failure_report(
    error: BaseException,
    *,
    model_name: str,
    base_report: Mapping[str, Any] | None = None,
    inference_performed: bool = False,
) -> dict[str, Any]:
    report = dict(base_report or {})
    execution = dict(report.get("execution", {}))
    failure_phase = str(execution.get("phase") or "initialization")
    execution.update(
        {
            "phase": "failed",
            "failure_phase": failure_phase,
        }
    )
    report.update(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "passed": False,
            "inference_performed": inference_performed,
            "scientific_result": False,
            "model_registry_name": model_name,
            "error_type": type(error).__name__,
            "error": sanitize_exception_text(error),
            "issues": ["controlled uncertainty-shift small-grid smoke failed"],
            "execution": execution,
        }
    )
    return report


def write_exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--routing-manifest", type=Path, required=True)
    parser.add_argument("--runtime-json", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inputs = (args.source_csv, args.routing_manifest, args.runtime_json)
    if args.output_path.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_path}")
    if any(not path.is_file() for path in inputs):
        raise FileNotFoundError("one or more required smoke inputs are missing")
    if args.output_path.resolve() in {path.resolve() for path in inputs}:
        raise ValueError("output path must differ from all inputs")
    try:
        runtime = validate_runtime(
            read_json_mapping(args.runtime_json, name="runtime mapping")
        )
        validate_runtime_model_contract(runtime, model_name=args.model_name)
        require_engine_runtime(runtime)
        judge_started = time.perf_counter()
        judge = build_vllm_judge(args.model_name, runtime)
        judge_seconds = time.perf_counter() - judge_started
        report = run_small_grid_preflight(
            source_csv=args.source_csv,
            routing_manifest_path=args.routing_manifest,
            runtime_path=args.runtime_json,
            model_name=args.model_name,
            judge=judge,
            judge_initialization_seconds=judge_seconds,
        )
    except Exception as error:
        report = failure_report(error, model_name=args.model_name)
    write_exclusive_json(args.output_path, report)
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
