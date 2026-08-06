from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from biases.dataset_splits import (
    assign_question_disjoint_routing_split,
    routing_manifest,
)
from biases.models import get_model_profile
from biases.pairing import file_sha256
from biases.position_bias import (
    CONSTRAINED_LOGPROBS_MODE,
    load_position_pairs_with_eligibility,
)
from biases.schemas import VerdictLabel


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "validate_controlled_uncertainty_shift_smoke.py"
)
SPEC = importlib.util.spec_from_file_location(
    "validate_controlled_uncertainty_shift_smoke",
    SCRIPT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class _Tokenizer:
    _special_ids = {"A": [32], "B": [33], "T": [51], "<|im_end|>": [128009]}
    _decoded = {32: "A", 33: "B", 51: "T", 128009: "<|im_end|>"}

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        if text in self._special_ids:
            return list(self._special_ids[text])
        return [1000 + ord(character) for character in text]

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        if len(token_ids) == 1 and token_ids[0] in self._decoded:
            return self._decoded[token_ids[0]]
        return "".join(chr(token_id - 1000) for token_id in token_ids)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str | list[int]:
        assert add_generation_prompt is True
        rendered = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        ) + "<assistant>"
        return self.encode(rendered, add_special_tokens=False) if tokenize else rendered


class _FakeJudge:
    def __init__(self) -> None:
        self.profile = get_model_profile("qwen2.5-32b")
        self.model_name = self.profile.hf_model_name
        self.tokenizer = _Tokenizer()
        self.logprobs_mode = CONSTRAINED_LOGPROBS_MODE
        self.decision_label_token_ids = {"A": [32], "B": [33], "tie": [51]}
        self.decision_allowed_token_ids = [32, 33, 51]

    def render_messages(self, messages: list[dict[str, str]]) -> str:
        return self.profile.render_prompt(self.tokenizer, messages)

    def choose_verdict_batch(
        self,
        prompt_texts: list[str],
        seed: int,
        sampling_temperature: float,
    ) -> list[tuple[VerdictLabel, str, dict[str, float]]]:
        assert seed in {0, 1, 2, 3}
        assert sampling_temperature in {0.0, 0.7}
        return [
            (VerdictLabel.A, "A", {"A": 0.8, "B": 0.1, "tie": 0.1})
            for _ in prompt_texts
        ]


def _runtime_mapping(*, max_model_len: int = 20_000) -> dict[str, object]:
    profile = get_model_profile("qwen2.5-32b")
    runtime: dict[str, object] = {
        "model_registry_name": profile.registry_name,
        "model_hf_name": profile.hf_model_name,
        "model_revision": profile.revision,
        "tensor_parallel_size": 4,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": 0.95,
        "dtype": "bfloat16",
        "batch_size": 4,
        "max_num_batched_tokens": 4096,
        "max_num_seqs": 1,
        "enforce_eager": True,
        "disable_custom_all_reduce": True,
        "seed": 0,
        "sampling_temperature": 0.7,
        "consistency_runs": 4,
        "consistency_schedule": "extremes",
        "include_verbalized_confidence": False,
        "engine_versions": MODULE.observed_engine_versions(),
    }
    runtime["runtime_sha256"] = MODULE.value_sha256(runtime)
    return runtime


def _rows_without_routing_sha256(frame: pd.DataFrame) -> str:
    columns = [column for column in frame.columns if column != "routing_split"]
    payload = [
        {column: str(value) for column, value in zip(columns, values)}
        for values in frame[columns].itertuples(index=False, name=None)
    ]
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _eligible_question_counts(source_path: Path) -> dict[str, int]:
    pairs, _ = load_position_pairs_with_eligibility(source_path)
    by_split: dict[str, set[str]] = {"calibration": set(), "test": set()}
    for pair in pairs:
        split = str(pair.original.metadata["routing_split"])
        question = str(pair.original.metadata["question_cluster_id"])
        by_split[split].add(question)
    return {
        "total": len(by_split["calibration"] | by_split["test"]),
        "calibration": len(by_split["calibration"]),
        "test": len(by_split["test"]),
        "overlap": len(by_split["calibration"] & by_split["test"]),
    }


def _write_fixture(
    tmp_path: Path,
    *,
    tamper_routing: bool = False,
    max_model_len: int = 20_000,
) -> tuple[Path, Path, Path]:
    original_path = tmp_path / "frozen.csv"
    original = pd.DataFrame(
        [
            {
                "question_id": "q-cal-short",
                "prompt": "Calibration?",
                "response_a": "short A",
                "response_b": "short B",
                "winner": "model_a",
                "turn": 1,
            },
            {
                "question_id": "q-cal-long",
                "prompt": "Long calibration?",
                "response_a": "A" * 90,
                "response_b": "B" * 80,
                "winner": "model_b",
                "turn": 1,
            },
            {
                "question_id": "q-test-short",
                "prompt": "Test?",
                "response_a": "short A",
                "response_b": "short B",
                "winner": "model_a",
                "turn": 1,
            },
            {
                "question_id": "q-test-long",
                "prompt": "Long test?",
                "response_a": "A" * 70,
                "response_b": "B" * 65,
                "winner": "model_b",
                "turn": 1,
            },
            {
                "question_id": "q-skipped",
                "prompt": "Skipped row?",
                "response_a": "",
                "response_b": "present B",
                "winner": "model_a",
                "turn": 1,
            },
        ]
    )
    original.to_csv(original_path, index=False)
    frame = assign_question_disjoint_routing_split(
        original,
        seed=42,
        calibration_fraction=0.5,
    )
    if tamper_routing:
        question_splits = frame.groupby("question_id")["routing_split"].first()
        calibration_question = str(
            question_splits[question_splits == "calibration"].index[0]
        )
        test_question = str(question_splits[question_splits == "test"].index[0])
        frame.loc[
            frame["question_id"] == calibration_question, "routing_split"
        ] = "test"
        frame.loc[frame["question_id"] == test_question, "routing_split"] = (
            "calibration"
        )

    source_path = tmp_path / "routed_full.csv"
    calibration_path = tmp_path / "routed_calibration.csv"
    test_path = tmp_path / "routed_test.csv"
    frame.to_csv(source_path, index=False)
    frame.loc[frame["routing_split"] == "calibration"].to_csv(
        calibration_path, index=False
    )
    frame.loc[frame["routing_split"] == "test"].to_csv(test_path, index=False)
    frozen = routing_manifest(
        frame,
        routing_unit="question",
        seed=42,
        calibration_fraction=0.5,
    )
    pairs, eligibility = load_position_pairs_with_eligibility(source_path)
    frozen.update(
        {
            "schema_version": 2,
            "artifact_type": "frozen_question_disjoint_routing_package",
            "source": {
                "path": str(original_path.resolve()),
                "sha256": file_sha256(original_path),
                "dataset_lineage": {
                    "dataset_name": "fixture",
                    "dataset_revision": "fixture-revision",
                },
                "columns": list(original.columns),
                "had_routing_split_column": False,
                "rows_without_routing_sha256": _rows_without_routing_sha256(
                    original
                ),
            },
            "counts": {
                "raw_rows": dict(frozen["row_counts"]),
                "raw_questions": dict(frozen["question_counts"]),
                "eligible_pairs": {
                    "total": len(pairs),
                    "calibration": eligibility.routing_counts[
                        "eligible_pairs"
                    ].get("calibration", 0),
                    "test": eligibility.routing_counts["eligible_pairs"].get(
                        "test", 0
                    ),
                },
                "eligible_questions": _eligible_question_counts(source_path),
                "skipped_rows": {
                    "total": eligibility.skipped_row_count,
                    "calibration": eligibility.routing_counts["skipped_rows"].get(
                        "calibration", 0
                    ),
                    "test": eligibility.routing_counts["skipped_rows"].get(
                        "test", 0
                    ),
                },
            },
            "eligibility": eligibility.to_dict(),
            "content_preservation": {
                "preserved_columns": list(original.columns),
                "recomputed_columns": ["routing_split"],
                "row_order_preserved": True,
                "rows_without_routing_sha256": _rows_without_routing_sha256(
                    original
                ),
            },
            "outputs": {
                "full": {"path": source_path.name, "rows": len(frame)},
                "calibration": {
                    "path": calibration_path.name,
                    "rows": int((frame["routing_split"] == "calibration").sum()),
                },
                "test": {
                    "path": test_path.name,
                    "rows": int((frame["routing_split"] == "test").sum()),
                },
            },
            "output_sha256": {
                "full": file_sha256(source_path),
                "calibration": file_sha256(calibration_path),
                "test": file_sha256(test_path),
            },
        }
    )
    routing_path = tmp_path / "routing.json"
    routing_path.write_text(json.dumps(frozen, sort_keys=True), encoding="utf-8")

    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(
        json.dumps(_runtime_mapping(max_model_len=max_model_len), sort_keys=True),
        encoding="utf-8",
    )
    return source_path, routing_path, runtime_path


def _native_generator(
    judge: _FakeJudge,
    prompts: list[str],
    seed: int,
    max_tokens: int,
) -> list[MODULE.NativeGeneration]:
    assert seed == 0
    assert max_tokens == 16
    return [MODULE.NativeGeneration(text="A", token_ids=(32,)) for _ in prompts]


def test_small_grid_selects_longest_required_strata(tmp_path: Path) -> None:
    source, _, runtime_path = _write_fixture(tmp_path)
    runtime = MODULE.validate_runtime(
        json.loads(runtime_path.read_text(encoding="utf-8"))
    )
    plan = MODULE.build_small_grid_plan(
        source_csv=source,
        canonical_model_name="Qwen/Qwen2.5-32B-Instruct",
        judge=_FakeJudge(),
        max_model_len=runtime["max_model_len"],
        required_completion_tokens=MODULE.NATIVE_MAX_TOKENS,
    )

    assert plan.source_row_count == 5
    assert plan.source_pair_count == 4
    assert plan.eligibility_audit["skipped_row_count"] == 1
    assert plan.eligibility_audit["skipped_reason_counts"] == {
        "missing_response_a": 1
    }
    assert len(plan.selected) == 18
    assert sum(item.stage == "stage_a" for item in plan.selected) == 2
    assert sum(item.stage == "stage_b" for item in plan.selected) == 16
    assert {item.ordering for item in plan.selected} == {"ab", "ba"}
    assert {
        (item.family, item.direction)
        for item in plan.selected
        if item.stage == "stage_b"
    } == {
        ("authority", "congruent"),
        ("authority", "incongruent"),
        ("bandwagon", "congruent"),
        ("bandwagon", "incongruent"),
    }
    assert {
        item.planned.condition.dose
        for item in plan.selected
        if item.family == "authority"
    } == {1, 4}
    assert {
        item.planned.condition.dose
        for item in plan.selected
        if item.family == "bandwagon"
    } == {55, 95}

    for selected in plan.selected:
        if selected.stage == "stage_a":
            stratum = [
                item
                for item in plan.candidates
                if item.stage == "stage_a"
                and item.ordering == selected.ordering
            ]
        else:
            stratum = [
                item
                for item in plan.candidates
                if item.stage == "stage_b"
                and item.family == selected.family
                and item.direction == selected.direction
                and item.planned.condition.dose == selected.planned.condition.dose
                and item.ordering == selected.ordering
            ]
        assert selected.input_tokens == max(item.input_tokens for item in stratum)


def test_fake_inference_reports_contracts_and_provenance_separately(
    tmp_path: Path,
) -> None:
    source, routing_path, runtime_path = _write_fixture(tmp_path)
    report = MODULE.run_small_grid_preflight(
        source_csv=source,
        routing_manifest_path=routing_path,
        runtime_path=runtime_path,
        model_name="qwen2.5-32b",
        judge=_FakeJudge(),
        native_generator=_native_generator,
    )

    assert report["passed"] is True
    assert report["status"] == "complete"
    assert report["inference_performed"] is True
    assert report["scientific_result"] is False
    assert report["excluded_methods"] == ["BPE", "SCOPE"]
    assert report["plan"]["selected_count"] == 18
    assert report["validation"]["constrained_parse_contract"]["passed"] is True
    assert report["validation"]["probability_contract"]["passed"] is True
    deterministic_passes = report["validation"]["deterministic_passes"]
    assert len(deterministic_passes) == 2
    assert [item["seed"] for item in deterministic_passes] == [0, 0]
    assert [item["sampling_temperature"] for item in deterministic_passes] == [
        0.0,
        0.0,
    ]
    replay = report["validation"]["deterministic_replay_contract"]
    assert replay["passed"] is True
    assert replay["exact_verdict_hash_match"] is True
    assert replay["exact_raw_output_token_hash_match"] is True
    assert replay["probability_absolute_tolerance"] == 1e-6
    assert replay["maximum_absolute_probability_difference"] == 0.0
    repeatability = report["validation"]["repeatability_contract"]
    assert repeatability["passed"] is True
    assert repeatability["consistency_runs"] == 4
    assert repeatability["consistency_schedule"] == "extremes"
    assert repeatability["sampling_temperature"] == 0.7
    assert repeatability["include_verbalized_confidence"] is False
    assert len(repeatability["passes"]) == 4
    assert [item["seed"] for item in repeatability["passes"]] == [0, 1, 2, 3]
    assert all(item["received_examples"] == 18 for item in repeatability["passes"])
    native = report["validation"]["native_verdict_token_contract"]
    assert native["passed"] is True
    assert native["contract_rate"] == 1.0
    assert report["model"]["verdict_token_ids"] == {
        "A": [32],
        "B": [33],
        "tie": [51],
    }
    assert report["routing"]["question_counts"]["overlap"] == 0
    assert len(report["source"]["sha256"]) == 64
    assert len(report["routing"]["manifest_file_sha256"]) == 64
    assert len(report["runtime"]["file_sha256"]) == 64
    assert len(report["model"]["contract_sha256"]) == 64
    assert len(report["result_set_sha256"]) == 64
    assert report["source"]["row_count"] == 5
    assert report["source"]["eligible_pair_count"] == 4
    assert report["source"]["skipped_row_count"] == 1
    serialized = json.dumps(report)
    assert "Long calibration?" not in serialized
    assert report["timing"]["total_wall_seconds"] >= 0.0
    assert report["timing"]["production_sequence_count"] == 90
    assert report["timing"]["production_benchmark_sequence_count"] == 108
    assert report["timing"]["native_infrastructure_sequence_count"] == 18
    assert report["timing"]["all_smoke_sequence_count"] == 126
    assert report["timing"]["production_sequences_per_second"] is not None
    throughput = report["validation"]["production_throughput_gate"]
    assert throughput["passed"] is True
    assert throughput["status"] == "release_target_met"


def test_probability_failure_is_separate_from_parse_and_native(
    tmp_path: Path,
) -> None:
    source, routing_path, runtime_path = _write_fixture(tmp_path)

    class _BadProbabilityJudge(_FakeJudge):
        def choose_verdict_batch(
            self,
            prompt_texts: list[str],
            seed: int,
            sampling_temperature: float,
        ) -> list[tuple[VerdictLabel, str, dict[str, float]]]:
            return [
                (VerdictLabel.A, "A", {"A": 0.8, "B": 0.2})
                for _ in prompt_texts
            ]

    report = MODULE.run_small_grid_preflight(
        source_csv=source,
        routing_manifest_path=routing_path,
        runtime_path=runtime_path,
        model_name="qwen2.5-32b",
        judge=_BadProbabilityJudge(),
        native_generator=_native_generator,
    )

    assert report["passed"] is False
    assert report["validation"]["constrained_parse_contract"]["passed"] is True
    assert report["validation"]["probability_contract"]["passed"] is False
    assert report["validation"]["native_verdict_token_contract"]["passed"] is True


def test_sampled_repeatability_does_not_require_probability_map_alignment(
    tmp_path: Path,
) -> None:
    source, routing_path, runtime_path = _write_fixture(tmp_path)

    class _SampledNonMapJudge(_FakeJudge):
        def choose_verdict_batch(
            self,
            prompt_texts: list[str],
            seed: int,
            sampling_temperature: float,
        ) -> list[tuple[VerdictLabel, str, dict[str, float]]]:
            if sampling_temperature == 0.0:
                return super().choose_verdict_batch(
                    prompt_texts,
                    seed,
                    sampling_temperature,
                )
            return [
                (VerdictLabel.B, "B", {"A": 0.8, "B": 0.1, "tie": 0.1})
                for _ in prompt_texts
            ]

    report = MODULE.run_small_grid_preflight(
        source_csv=source,
        routing_manifest_path=routing_path,
        runtime_path=runtime_path,
        model_name="qwen2.5-32b",
        judge=_SampledNonMapJudge(),
        native_generator=_native_generator,
    )

    assert report["passed"] is True
    repeatability = report["validation"]["repeatability_contract"]
    assert repeatability["passed"] is True
    assert all(
        item["probability_contract"]["map_alignment_required"] is False
        for item in repeatability["passes"]
    )
    assert all(
        item["probability_contract"]["map_aligned_examples"] == 0
        for item in repeatability["passes"]
    )


def test_planning_only_seam_skips_all_inference(tmp_path: Path) -> None:
    source, routing_path, runtime_path = _write_fixture(tmp_path)

    class _NoInferenceJudge(_FakeJudge):
        def choose_verdict_batch(self, *args: object, **kwargs: object) -> object:
            raise AssertionError("inference must not run")

    report = MODULE.run_small_grid_preflight(
        source_csv=source,
        routing_manifest_path=routing_path,
        runtime_path=runtime_path,
        model_name="qwen2.5-32b",
        judge=_NoInferenceJudge(),
        perform_inference=False,
    )

    assert report["passed"] is True
    assert report["status"] == "planning_complete"
    assert report["inference_performed"] is False
    assert report["validation"]["constrained_parse_contract"] == {
        "status": "not_run"
    }
    assert report["validation"]["deterministic_replay_contract"] == {
        "status": "not_run"
    }
    assert report["validation"]["production_throughput_gate"] == {
        "status": "not_run"
    }


def test_routing_contract_rejects_a_matching_but_nonseeded_assignment(
    tmp_path: Path,
) -> None:
    source, routing_path, runtime_path = _write_fixture(
        tmp_path,
        tamper_routing=True,
    )
    with pytest.raises(ValueError, match="deterministic question split"):
        MODULE.run_small_grid_preflight(
            source_csv=source,
            routing_manifest_path=routing_path,
            runtime_path=runtime_path,
            model_name="qwen2.5-32b",
            judge=_FakeJudge(),
            perform_inference=False,
        )


def test_native_generation_headroom_is_reserved_exactly(tmp_path: Path) -> None:
    source, _, runtime_path = _write_fixture(tmp_path)
    runtime = MODULE.validate_runtime(
        json.loads(runtime_path.read_text(encoding="utf-8"))
    )
    baseline = MODULE.build_small_grid_plan(
        source_csv=source,
        canonical_model_name="Qwen/Qwen2.5-32B-Instruct",
        judge=_FakeJudge(),
        max_model_len=runtime["max_model_len"],
        required_completion_tokens=MODULE.NATIVE_MAX_TOKENS,
    )
    longest = max(item.input_tokens for item in baseline.candidates)
    exact = MODULE.build_small_grid_plan(
        source_csv=source,
        canonical_model_name="Qwen/Qwen2.5-32B-Instruct",
        judge=_FakeJudge(),
        max_model_len=longest + MODULE.NATIVE_MAX_TOKENS,
        required_completion_tokens=MODULE.NATIVE_MAX_TOKENS,
    )
    assert max(item.input_tokens for item in exact.candidates) == longest
    with pytest.raises(ValueError, match="insufficient generation headroom"):
        MODULE.build_small_grid_plan(
            source_csv=source,
            canonical_model_name="Qwen/Qwen2.5-32B-Instruct",
            judge=_FakeJudge(),
            max_model_len=longest + MODULE.NATIVE_MAX_TOKENS - 1,
            required_completion_tokens=MODULE.NATIVE_MAX_TOKENS,
        )


def test_post_constrained_failure_preserves_phase_and_provenance(
    tmp_path: Path,
) -> None:
    source, routing_path, runtime_path = _write_fixture(tmp_path)

    def failing_native_generator(
        judge: _FakeJudge,
        prompts: list[str],
        seed: int,
        max_tokens: int,
    ) -> list[MODULE.NativeGeneration]:
        raise RuntimeError(
            "native failed token=hf_abcdefghijklmnopqrstuvwxyz0123456789"
        )

    report = MODULE.run_small_grid_preflight(
        source_csv=source,
        routing_manifest_path=routing_path,
        runtime_path=runtime_path,
        model_name="qwen2.5-32b",
        judge=_FakeJudge(),
        native_generator=failing_native_generator,
    )

    assert report["passed"] is False
    assert report["status"] == "failed"
    assert report["inference_performed"] is True
    assert report["execution"]["phase"] == "failed"
    assert report["execution"]["failure_phase"] == (
        "native_infrastructure_check_started"
    )
    assert report["execution"]["deterministic_completed_passes"] == 2
    assert report["execution"]["deterministic_received_examples"] == 36
    assert report["execution"]["repeatability_completed_passes"] == 4
    assert report["execution"]["repeatability_received_examples"] == 72
    assert report["execution"]["native_received_examples"] == 0
    assert report["source"]["eligible_pair_count"] == 4
    assert len(report["model"]["contract_sha256"]) == 64
    assert "hf_abcdefghijklmnopqrstuvwxyz0123456789" not in json.dumps(report)


def test_exclusive_writer_and_failure_redaction(tmp_path: Path) -> None:
    output_path = tmp_path / "smoke.json"
    MODULE.write_exclusive_json(output_path, {"passed": True})
    with pytest.raises(FileExistsError):
        MODULE.write_exclusive_json(output_path, {"passed": False})

    secret = "hf_abcdefghijklmnopqrstuvwxyz0123456789"
    report = MODULE.failure_report(
        RuntimeError(f"download https://example.test/model?token={secret}"),
        model_name="llama3.3-70b-instruct",
    )
    serialized = json.dumps(report)
    assert secret not in serialized
    assert "?token=" not in serialized


def test_runtime_controls_are_strict_and_configure_vllm_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested = _runtime_mapping(max_model_len=4096)
    requested["disable_custom_all_reduce"] = False
    requested["runtime_sha256"] = MODULE.value_sha256(
        {key: value for key, value in requested.items() if key != "runtime_sha256"}
    )
    runtime = MODULE.validate_runtime(requested)
    MODULE.configure_runtime_environment(runtime)
    assert MODULE.os.environ["BIASES_VLLM_ENFORCE_EAGER"] == "1"
    assert MODULE.os.environ["VLLM_DISABLE_CUSTOM_ALL_REDUCE"] == "0"

    captured: dict[str, object] = {}

    def fake_vllm_judge(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(MODULE, "VLLMJudge", fake_vllm_judge)
    MODULE.build_vllm_judge("qwen2.5-32b", runtime)
    assert captured["enforce_eager"] is True
    assert captured["disable_custom_all_reduce"] is False
    assert captured["max_num_batched_tokens"] == 4096
    assert captured["max_num_seqs"] == 1

    with pytest.raises(ValueError, match="gpu_memory_utilization"):
        changed = {**runtime, "gpu_memory_utilization": 1.1}
        changed["runtime_sha256"] = MODULE.value_sha256(
            {key: value for key, value in changed.items() if key != "runtime_sha256"}
        )
        MODULE.validate_runtime(changed)
    with pytest.raises(ValueError, match="unsupported fields"):
        MODULE.validate_runtime({**runtime, "auth_token": "must-not-be-reported"})
    with pytest.raises(ValueError, match="runtime_sha256"):
        MODULE.validate_runtime({**runtime, "batch_size": 8})
    nonzero_seed = {**runtime, "seed": 1}
    nonzero_seed["runtime_sha256"] = MODULE.value_sha256(
        {
            key: value
            for key, value in nonzero_seed.items()
            if key != "runtime_sha256"
        }
    )
    with pytest.raises(ValueError, match="seed 0"):
        MODULE.validate_runtime(nonzero_seed)


@pytest.mark.parametrize(
    ("rate", "status", "passed"),
    (
        (None, "invalid_or_unavailable", False),
        (0.945999, "below_minimum", False),
        (0.946, "minimum_met_release_target_missed", False),
        (1.260999, "minimum_met_release_target_missed", False),
        (1.261, "release_target_met", True),
    ),
)
def test_production_throughput_gate_boundaries(
    rate: float | None,
    status: str,
    passed: bool,
) -> None:
    gate = MODULE.classify_production_throughput(rate)
    assert gate["status"] == status
    assert gate["passed"] is passed
    assert gate["minimum_sequences_per_second"] == 0.946
    assert gate["release_target_sequences_per_second"] == 1.261


def test_real_inference_fails_when_release_throughput_target_is_missed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source, routing_path, runtime_path = _write_fixture(tmp_path)
    monkeypatch.setattr(
        MODULE,
        "classify_production_throughput",
        lambda rate: {
            "passed": False,
            "status": "minimum_met_release_target_missed",
            "observed_sequences_per_second": rate,
            "minimum_sequences_per_second": 0.946,
            "release_target_sequences_per_second": 1.261,
            "minimum_met": True,
            "release_target_met": False,
        },
    )

    report = MODULE.run_small_grid_preflight(
        source_csv=source,
        routing_manifest_path=routing_path,
        runtime_path=runtime_path,
        model_name="qwen2.5-32b",
        judge=_FakeJudge(),
        native_generator=_native_generator,
    )

    assert report["passed"] is False
    assert report["status"] == "failed_validation"
    assert report["validation"]["production_throughput_gate"]["passed"] is False
    assert report["validation"]["deterministic_replay_contract"]["passed"] is True


def test_deterministic_replay_probability_tolerance_is_tight() -> None:
    first = [
        (VerdictLabel.A, "A", {"A": 0.8, "B": 0.1, "tie": 0.1})
    ]
    within = [
        (
            VerdictLabel.A,
            "A",
            {"A": 0.8000005, "B": 0.0999995, "tie": 0.1},
        )
    ]
    outside = [
        (
            VerdictLabel.A,
            "A",
            {"A": 0.800002, "B": 0.099998, "tie": 0.1},
        )
    ]

    assert MODULE.validate_deterministic_replay_contract(
        first,
        within,
        tokenizer=_Tokenizer(),
        expected=1,
    )["passed"] is True
    assert MODULE.validate_deterministic_replay_contract(
        first,
        outside,
        tokenizer=_Tokenizer(),
        expected=1,
    )["passed"] is False


def test_main_rejects_model_identity_before_model_construction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source, routing_path, runtime_path = _write_fixture(tmp_path)
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["model_hf_name"] = "not/the-selected-model"
    runtime["runtime_sha256"] = MODULE.value_sha256(
        {
            key: value
            for key, value in runtime.items()
            if key != "runtime_sha256"
        }
    )
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    model_constructed = False

    def _unexpected(*args: object, **kwargs: object) -> None:
        nonlocal model_constructed
        model_constructed = True
        raise AssertionError("model must not be constructed")

    monkeypatch.setattr(MODULE, "build_vllm_judge", _unexpected)
    output_path = tmp_path / "identity-failed.json"
    exit_code = MODULE.main(
        [
            "--source-csv",
            str(source),
            "--routing-manifest",
            str(routing_path),
            "--runtime-json",
            str(runtime_path),
            "--model-name",
            "qwen2.5-32b",
            "--output-path",
            str(output_path),
        ]
    )

    assert exit_code == 2
    assert model_constructed is False
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert "runtime model identity" in report["error"]
