from __future__ import annotations

import importlib.util
import json
import sys
import types
from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest

from biases.models import get_model_profile
from scripts.prepare_frozen_question_routing import (
    FULL_FILENAME,
    MANIFEST_FILENAME,
    build_routing_package,
)


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "validate_controlled_uncertainty_shift_preflight.py"
)
SPEC = importlib.util.spec_from_file_location(
    "validate_controlled_uncertainty_shift_preflight",
    SCRIPT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _runtime_contract() -> dict[str, object]:
    profile = get_model_profile("qwen2.5-32b")
    runtime: dict[str, object] = {
        "model_registry_name": profile.registry_name,
        "model_hf_name": profile.hf_model_name,
        "model_revision": profile.revision,
        "tensor_parallel_size": 1,
        "max_model_len": 20_000,
        "gpu_memory_utilization": 0.9,
        "dtype": "bfloat16",
        "batch_size": 4,
        "max_num_batched_tokens": 20_000,
        "max_num_seqs": 4,
        "enforce_eager": False,
        "disable_custom_all_reduce": False,
        "seed": 0,
        "sampling_temperature": 0.7,
        "consistency_runs": 8,
        "consistency_schedule": "all",
        "include_verbalized_confidence": False,
        "engine_versions": {
            "python": "3.12.0",
            "torch": "2.10.0",
            "transformers": "4.57.6",
            "vllm": "0.19.1",
        },
    }
    runtime["runtime_sha256"] = MODULE.value_sha256(runtime)
    return runtime


class _Tokenizer:
    init_kwargs = {
        "_commit_hash": "5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd"
    }
    _special_ids = {
        "A": [32],
        "B": [33],
        "T": [51],
        "<|im_end|>": [128009],
    }
    _decoded = {
        32: "A",
        33: "B",
        51: "T",
        128009: "<|im_end|>",
    }

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
        return (
            self.encode(rendered, add_special_tokens=False)
            if tokenize
            else rendered
        )


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    source_path = tmp_path / "source.csv"
    frame = pd.DataFrame(
        [
            {
                "question_id": "q-cal",
                "prompt": "Calibration question?",
                "response_a": "Calibration A.",
                "response_b": "Calibration B.",
                "winner": "model_a",
                "turn": 1,
                "routing_split": "calibration",
            },
            {
                "question_id": "q-test",
                "prompt": "Test question?",
                "response_a": "Test A.",
                "response_b": "Test B.",
                "winner": "model_b",
                "turn": 1,
                "routing_split": "test",
            },
        ]
    )
    frame.to_csv(source_path, index=False)

    routing_dir = tmp_path / "routing"
    build_routing_package(
        source_csv=source_path,
        output_dir=routing_dir,
        dataset_lineage={"dataset": "fixture", "revision": "frozen"},
    )
    routed_path = routing_dir / FULL_FILENAME
    routing_path = routing_dir / MANIFEST_FILENAME

    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(
        json.dumps(_runtime_contract(), sort_keys=True),
        encoding="utf-8",
    )
    return routed_path, routing_path, runtime_path


def _write_fixture_with_ineligible_rows(
    tmp_path: Path,
) -> tuple[Path, Path, Path]:
    source_path = tmp_path / "source-with-skips.csv"
    frame = pd.DataFrame(
        [
            {
                "question_id": "q3",
                "prompt": "Calibration question?",
                "response_a": "Calibration A.",
                "response_b": "Calibration B.",
                "winner": "model_a",
                "turn": 1,
                "routing_split": "calibration",
            },
            {
                "question_id": "q0",
                "prompt": "Test question?",
                "response_a": "Test A.",
                "response_b": "Test B.",
                "winner": "model_b",
                "turn": 1,
                "routing_split": "test",
            },
            {
                "question_id": "q5",
                "prompt": "Skipped calibration question?",
                "response_a": "A.",
                "response_b": "B.",
                "winner": "",
                "turn": 1,
                "routing_split": "calibration",
            },
            {
                "question_id": "q1",
                "prompt": "Skipped test question?",
                "response_a": "A.",
                "response_b": "B.",
                "winner": "neither",
                "turn": 1,
                "routing_split": "test",
            },
            {
                "question_id": "q6",
                "prompt": "Skipped calibration response?",
                "response_a": "",
                "response_b": "B.",
                "winner": "model_b",
                "turn": 1,
                "routing_split": "calibration",
            },
            {
                "question_id": "q2",
                "prompt": "Skipped test response?",
                "response_a": "A.",
                "response_b": "",
                "winner": "model_a",
                "turn": 1,
                "routing_split": "test",
            },
        ]
    )
    frame.to_csv(source_path, index=False)
    routing_dir = tmp_path / "routing-with-skips"
    build_routing_package(
        source_csv=source_path,
        output_dir=routing_dir,
        dataset_lineage={"dataset": "fixture", "revision": "frozen"},
    )
    routed_path = routing_dir / FULL_FILENAME
    routing_path = routing_dir / MANIFEST_FILENAME
    runtime_path = tmp_path / "runtime-with-skips.json"
    runtime_path.write_text(
        json.dumps(_runtime_contract()),
        encoding="utf-8",
    )
    return routed_path, routing_path, runtime_path


def test_full_grid_preflight_enumerates_and_renders_frozen_design(
    tmp_path: Path,
) -> None:
    source_path, routing_path, runtime_path = _write_fixture(tmp_path)

    report = MODULE.build_preflight_report(
        source_csv=source_path,
        routing_manifest_path=routing_path,
        runtime_path=runtime_path,
        model_name="qwen2.5-32b",
        tokenizer=_Tokenizer(),
        expected_calibration_questions=1,
        expected_test_questions=1,
    )

    assert report["passed"] is True
    assert report["inference_performed"] is False
    assert report["excluded_methods"] == ["BPE", "SCOPE"]
    assert report["routing"]["question_counts"] == {
        "total": 2,
        "calibration": 1,
        "test": 1,
        "overlap": 0,
    }
    assert report["model"]["verdict_token_ids"] == {
        "A": [32],
        "B": [33],
        "tie": [51],
    }
    assert report["plan"]["condition_counts"] == {
        "stage_a": 4,
        "stage_b": 32,
        "stage_a_calibration": 2,
        "stage_a_test": 2,
        "stage_b_test": 32,
        "stage_b_rendered_target_realizations": 64,
    }
    assert report["plan"]["rendered_prompt_count"] == 68
    assert report["plan"]["counts_by_family"] == {
        "authority": 32,
        "bandwagon": 32,
        "clean": 4,
    }
    assert report["plan"]["text_transport_match_count"] == 68
    assert report["plan"]["counts_by_target_realization"] == {
        "A": 32,
        "B": 32,
        "None": 4,
    }
    assert report["plan"]["stage_b_plan_mode"] == (
        "provisional_structural_pre_stage_a"
    )
    assert report["plan"]["actual_stage_b_prompt_set_sha256"] is None
    assert len(
        report["plan"]["provisional_structural_stage_b_prompt_set_sha256"]
    ) == 64
    assert report["release_gate"] == {
        "stage_a_authorized": True,
        "stage_b_authorized": False,
        "exact_post_stage_a_required": True,
        "provisional_stage_b_hashes_must_not_be_released": True,
    }
    assert len(report["plan"]["prompt_set_sha256"]) == 64
    assert set(report["plan"]["prompt_set_sha256_by_stage"]) == {
        "stage_a",
        "stage_b",
    }
    assert len(report["plan"]["condition_plan_sha256"]) == 64
    assert len(report["model"]["model_revision_sha256"]) == 64
    assert len(report["runtime"]["inference_runtime_sha256"]) == 64


def test_full_grid_preflight_accounts_for_skips_without_weakening_grid(
    tmp_path: Path,
) -> None:
    source_path, routing_path, runtime_path = _write_fixture_with_ineligible_rows(
        tmp_path
    )

    report = MODULE.build_preflight_report(
        source_csv=source_path,
        routing_manifest_path=routing_path,
        runtime_path=runtime_path,
        model_name="qwen2.5-32b",
        tokenizer=_Tokenizer(),
        expected_calibration_questions=3,
        expected_test_questions=3,
    )

    eligibility = report["plan"]["eligibility"]
    assert eligibility["eligibility_contract"] == "position_pair_loader_v1"
    assert eligibility["raw_row_count"] == 6
    assert eligibility["eligible_pair_count"] == 2
    assert eligibility["skipped_row_count"] == 4
    assert eligibility["skipped_reason_counts"] == {
        "invalid_winner": 1,
        "missing_response_a": 1,
        "missing_response_b": 1,
        "missing_winner": 1,
    }
    assert eligibility["routing_counts"] == {
        "raw_rows": {"calibration": 3, "test": 3},
        "eligible_pairs": {"calibration": 1, "test": 1},
        "skipped_rows": {"calibration": 2, "test": 2},
    }
    assert report["plan"]["eligible_calibration_pairs"] == 1
    assert report["plan"]["eligible_test_pairs"] == 1
    assert report["plan"]["condition_counts"] == {
        "stage_a": 4,
        "stage_b": 32,
        "stage_a_calibration": 2,
        "stage_a_test": 2,
        "stage_b_test": 32,
        "stage_b_rendered_target_realizations": 64,
    }
    assert report["plan"]["rendered_prompt_count"] == 68
    assert len(eligibility["eligibility_sha256"]) == 64


def test_preflight_rejects_prompt_without_generation_headroom(
    tmp_path: Path,
) -> None:
    source_path, routing_path, runtime_path = _write_fixture(tmp_path)

    with pytest.raises(ValueError, match="prompt exceeds configured context"):
        MODULE.build_preflight_report(
            source_csv=source_path,
            routing_manifest_path=routing_path,
            runtime_path=runtime_path,
            model_name="qwen2.5-32b",
            tokenizer=_Tokenizer(),
            max_model_len=20_000,
            generation_headroom=19_900,
            expected_calibration_questions=1,
            expected_test_questions=1,
        )


def test_preflight_rejects_text_transport_token_drift(tmp_path: Path) -> None:
    source_path, routing_path, runtime_path = _write_fixture(tmp_path)

    class _DriftedTokenizer(_Tokenizer):
        def apply_chat_template(
            self,
            messages: list[dict[str, str]],
            *,
            tokenize: bool,
            add_generation_prompt: bool,
        ) -> str | list[int]:
            rendered = super().apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
            )
            return [999] if tokenize else rendered

    with pytest.raises(ValueError, match="did not re-encode"):
        MODULE.build_preflight_report(
            source_csv=source_path,
            routing_manifest_path=routing_path,
            runtime_path=runtime_path,
            model_name="qwen2.5-32b",
            tokenizer=_DriftedTokenizer(),
            expected_calibration_questions=1,
            expected_test_questions=1,
        )


def test_preflight_rejects_incomplete_runtime_contract(tmp_path: Path) -> None:
    source_path, routing_path, runtime_path = _write_fixture(tmp_path)
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime.pop("disable_custom_all_reduce")
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")

    with pytest.raises(ValueError, match="missing fields: disable_custom_all_reduce"):
        MODULE.build_preflight_report(
            source_csv=source_path,
            routing_manifest_path=routing_path,
            runtime_path=runtime_path,
            model_name="qwen2.5-32b",
            tokenizer=_Tokenizer(),
            expected_calibration_questions=1,
            expected_test_questions=1,
        )


def test_preflight_rejects_unpinned_tokenizer_resolution(tmp_path: Path) -> None:
    source_path, routing_path, runtime_path = _write_fixture(tmp_path)

    class _UnpinnedTokenizer(_Tokenizer):
        init_kwargs: dict[str, str] = {}

    with pytest.raises(ValueError, match="tokenizer resolved commit"):
        MODULE.build_preflight_report(
            source_csv=source_path,
            routing_manifest_path=routing_path,
            runtime_path=runtime_path,
            model_name="qwen2.5-32b",
            tokenizer=_UnpinnedTokenizer(),
            expected_calibration_questions=1,
            expected_test_questions=1,
        )


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("schema_version", "schema version 2"),
        ("artifact_type", "artifact type"),
        ("source", "source must be a JSON object"),
        ("content_preservation", "content_preservation must be a JSON object"),
        ("eligibility", "eligibility must be a JSON object"),
    ),
)
def test_preflight_rejects_missing_routing_package_contract(
    tmp_path: Path,
    field: str,
    message: str,
) -> None:
    source_path, routing_path, runtime_path = _write_fixture(tmp_path)
    manifest = json.loads(routing_path.read_text(encoding="utf-8"))
    manifest.pop(field)
    routing_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        MODULE.build_preflight_report(
            source_csv=source_path,
            routing_manifest_path=routing_path,
            runtime_path=runtime_path,
            model_name="qwen2.5-32b",
            tokenizer=_Tokenizer(),
            expected_calibration_questions=1,
            expected_test_questions=1,
        )


def test_exact_stage_a_summary_authorizes_only_exact_stage_b_hashes(
    tmp_path: Path,
) -> None:
    source_path, routing_path, runtime_path = _write_fixture(tmp_path)
    tokenizer = _Tokenizer()
    profile = get_model_profile("qwen2.5-32b")
    assert profile.revision is not None
    runtime = MODULE.validate_runtime_contract(
        json.loads(runtime_path.read_text(encoding="utf-8"))
    )
    verdict_contract = MODULE.validate_verdict_contract(profile, tokenizer)
    stage_a, _, _ = MODULE.build_stage_plans(
        source_csv=source_path,
        source_sha256=MODULE.file_sha256(source_path),
        canonical_model_name=profile.hf_model_name,
        model_revision=profile.revision,
        runtime=runtime,
        verdict_contract=verdict_contract,
    )
    runner_runtime = {
        field: runtime[field] for field in MODULE.RUNNER_RUNTIME_FIELDS
    }
    summary_rows = []
    for index, item in enumerate(stage_a):
        human_winner = item.planned.condition.metadata["human_winner"]
        summary_rows.append(
            {
                "record_id": f"record-{index}",
                "clean_record_id": f"record-{index}",
                "pair_identity_key": item.planned.pair_identity_key,
                "pair_key": item.planned.pair_key,
                "condition_group_id": item.planned.condition_group_id,
                "ordering_twin_key": item.planned.ordering_twin_key,
                "ordering": item.planned.condition.ordering,
                "model_name": profile.hf_model_name,
                "model_revision": profile.revision,
                "input_file_hash": MODULE.file_sha256(source_path),
                "spec_hash": "a" * 64,
                "question_id": item.example.question_id,
                "source_row_index": item.planned.condition.metadata[
                    "source_row_index"
                ],
                "routing_split": item.routing_split,
                "judge_output_parser_version": "strict_v3",
                "verbalized_output_parser_version": "strict_v3",
                "logprobs_mode": "processed_logprobs",
                "verdict_token_texts": verdict_contract["verdict_token_texts"],
                "verdict_token_ids": verdict_contract["verdict_token_ids"],
                "max_num_batched_tokens": runtime["max_num_batched_tokens"],
                "max_num_seqs": runtime["max_num_seqs"],
                "inference_runtime": runner_runtime,
                "human_winner": human_winner,
                "clean_verdict": human_winner,
                "verdict": human_winner,
                "clean_tie": human_winner == "tie",
            }
        )
    summary_path = tmp_path / "stage-a.jsonl"
    summary_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in summary_rows),
        encoding="utf-8",
    )

    report = MODULE.build_preflight_report(
        source_csv=source_path,
        routing_manifest_path=routing_path,
        runtime_path=runtime_path,
        model_name="qwen2.5-32b",
        tokenizer=tokenizer,
        expected_calibration_questions=1,
        expected_test_questions=1,
        stage_a_summary_path=summary_path,
    )

    assert report["plan"]["stage_b_plan_mode"] == "exact_post_stage_a"
    assert report["plan"]["rendered_prompt_count"] == 36
    assert report["plan"]["condition_counts"][
        "stage_b_rendered_target_realizations"
    ] == 32
    assert report["plan"]["provisional_structural_stage_b_prompt_set_sha256"] is None
    assert len(report["plan"]["actual_stage_b_prompt_set_sha256"]) == 64
    assert report["release_gate"]["stage_b_authorized"] is True
    assert report["release_gate"]["exact_post_stage_a_required"] is False


def test_preflight_rejects_noncanonical_verdict_surface() -> None:
    profile = get_model_profile("qwen2.5-14b")

    with pytest.raises(ValueError, match="exact literal singleton"):
        MODULE.validate_verdict_contract(profile, _Tokenizer())


def test_exclusive_writer_refuses_overwrite(tmp_path: Path) -> None:
    output_path = tmp_path / "preflight.json"
    MODULE.write_exclusive_json(output_path, {"passed": True})

    with pytest.raises(FileExistsError):
        MODULE.write_exclusive_json(output_path, {"passed": False})


def test_runtime_contract_rejects_nonzero_seed() -> None:
    runtime = _runtime_contract()
    runtime["seed"] = 1
    runtime["runtime_sha256"] = MODULE.value_sha256(
        {
            key: value
            for key, value in runtime.items()
            if key != "runtime_sha256"
        }
    )
    with pytest.raises(ValueError, match="seed 0"):
        MODULE.validate_runtime_contract(runtime)


def test_main_rejects_model_identity_before_tokenizer_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_path, routing_path, runtime_path = _write_fixture(tmp_path)
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
    tokenizer_accessed = False

    def _unexpected(_: Namespace) -> None:
        nonlocal tokenizer_accessed
        tokenizer_accessed = True
        raise AssertionError("tokenizer must not be accessed")

    monkeypatch.setattr(MODULE, "load_tokenizer", _unexpected)
    output_path = tmp_path / "identity-failed.json"
    exit_code = MODULE.main(
        [
            "--source-csv",
            str(source_path),
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
    assert tokenizer_accessed is False
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert "runtime model identity" in report["error"]


def test_main_rejects_active_engine_drift_before_tokenizer_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_path, routing_path, runtime_path = _write_fixture(tmp_path)
    tokenizer_accessed = False

    def _unexpected(_: Namespace) -> None:
        nonlocal tokenizer_accessed
        tokenizer_accessed = True
        raise AssertionError("tokenizer must not be accessed")

    monkeypatch.setattr(
        MODULE,
        "observed_engine_versions",
        lambda: {
            "python": "different",
            "torch": None,
            "transformers": None,
            "vllm": None,
        },
    )
    monkeypatch.setattr(MODULE, "load_tokenizer", _unexpected)
    output_path = tmp_path / "engine-failed.json"
    exit_code = MODULE.main(
        [
            "--source-csv",
            str(source_path),
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
    assert tokenizer_accessed is False
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert "active execution environment" in report["error"]


def test_load_tokenizer_honors_model_specific_remote_code_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(model_name: str, **kwargs: object) -> _Tokenizer:
            captured["model_name"] = model_name
            captured.update(kwargs)
            return _Tokenizer()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoTokenizer=_AutoTokenizer),
    )

    tokenizer = MODULE.load_tokenizer(
        Namespace(
            model_name="qwen2.5-32b",
            cache_dir=tmp_path,
            allow_download=False,
            require_authentication=False,
        )
    )

    assert isinstance(tokenizer, _Tokenizer)
    assert captured["model_name"] == "Qwen/Qwen2.5-32B-Instruct"
    assert captured["trust_remote_code"] is False


def test_main_sanitizes_tokenizer_failure_before_writing_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_path, routing_path, runtime_path = _write_fixture(tmp_path)
    output_path = tmp_path / "failed.json"
    secret = "hf_abcdefghijklmnopqrstuvwxyz0123456789"

    def _fail(_: Namespace) -> None:
        raise RuntimeError(
            "download failed at "
            f"https://huggingface.co/model?token={secret}&signature=signed-secret "
            f"Authorization: Bearer {secret}"
        )

    monkeypatch.setattr(MODULE, "load_tokenizer", _fail)
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    monkeypatch.setattr(
        MODULE,
        "observed_engine_versions",
        lambda: runtime["engine_versions"],
    )

    exit_code = MODULE.main(
        [
            "--source-csv",
            str(source_path),
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
    serialized = output_path.read_text(encoding="utf-8")
    assert secret not in serialized
    assert "signed-secret" not in serialized
    assert "https://huggingface.co/model" in serialized
    assert "<redacted>" in serialized
