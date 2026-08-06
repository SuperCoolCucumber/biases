from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from biases.models import get_model_profile


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "render_slurm_jobs.py"
)


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "render_slurm_jobs",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


renderer = _load_module()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_runtime(
    tmp_path: Path,
    *,
    model_key: str = "qwen25_32b",
    overrides: dict[str, object] | None = None,
    refresh_embedded_sha: bool = True,
) -> tuple[object, Path]:
    model = renderer.MODEL_SPECS[model_key]
    profile = get_model_profile(model.model_name)
    runtime: dict[str, object] = {
        "model_registry_name": profile.registry_name,
        "model_hf_name": profile.hf_model_name,
        "model_revision": profile.revision,
        "tensor_parallel_size": model.gpus,
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.91,
        "dtype": "bfloat16",
        "batch_size": 7,
        "max_num_batched_tokens": 16384,
        "max_num_seqs": 11,
        "enforce_eager": True,
        "disable_custom_all_reduce": True,
        "seed": 0,
        "sampling_temperature": 0.65,
        "consistency_runs": 5,
        "consistency_schedule": "extremes",
        "include_verbalized_confidence": False,
        "engine_versions": {
            "python": "3.12.11",
            "torch": "2.7.1",
            "transformers": "4.53.0",
            "vllm": "0.10.0",
        },
    }
    runtime["runtime_sha256"] = renderer._value_sha256(runtime)
    runtime.update(overrides or {})
    if overrides and refresh_embedded_sha:
        runtime["runtime_sha256"] = renderer._value_sha256(
            {key: value for key, value in runtime.items() if key != "runtime_sha256"}
        )
    runtime_path = tmp_path / f"{model_key}_runtime.json"
    runtime_path.write_text(json.dumps(runtime, sort_keys=True), encoding="utf-8")
    return model, runtime_path


def _write_routing(tmp_path: Path) -> tuple[Path, Path]:
    routing_dir = tmp_path / "routing"
    routing_dir.mkdir()
    data_path = routing_dir / "routed_full.csv"
    data_path.write_text(
        "question_id,routing_split\nq-cal,calibration\nq-test,test\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 2,
        "artifact_type": renderer.ROUTING_ARTIFACT_TYPE,
        "routing_assignment_sha256": "a" * 64,
        "outputs": {
            "full": {
                "path": data_path.name,
                "row_count": 2,
            }
        },
        "output_sha256": {"full": _file_sha256(data_path)},
    }
    manifest_path = routing_dir / "routing_manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return data_path, manifest_path


def _load_contracts(
    tmp_path: Path,
    *,
    model_key: str = "qwen25_32b",
) -> tuple[object, object, object, Path, Path]:
    model, runtime_path = _write_runtime(tmp_path, model_key=model_key)
    data_path, routing_path = _write_routing(tmp_path)
    runtime = renderer.load_frozen_runtime_contract(runtime_path, model=model)
    routing = renderer.load_frozen_routing_contract(
        routing_path,
        data_path=data_path,
    )
    return model, runtime, routing, data_path, routing_path


def _write_stage_b_release(
    tmp_path: Path,
    *,
    runtime: object,
    routing: object,
    authorized: bool = True,
    overrides: dict[str, object] | None = None,
) -> tuple[object, Path, Path, str]:
    summary_path = tmp_path / "silent_bias_stage_a_pair_summary.jsonl"
    summary_path.write_text(
        "".join(
            f'{{"record_id":"stage-a-{index}"}}\n'
            for index in range(4)
        ),
        encoding="utf-8",
    )
    report: dict[str, object] = {
        "schema_version": 1,
        "status": "complete",
        "passed": True,
        "release_gate": {
            "stage_a_authorized": True,
            "stage_b_authorized": authorized,
            "exact_post_stage_a_required": True,
        },
        "runtime": {
            "path": str(runtime.path),
            "file_sha256": runtime.file_sha256,
            "inference_runtime": dict(runtime.values),
            "inference_runtime_sha256": runtime.canonical_sha256,
        },
        "model": {
            "registry_name": runtime.values["model_registry_name"],
            "model_name": runtime.values["model_hf_name"],
            "revision": runtime.values["model_revision"],
        },
        "source": {
            "path": str(routing.data_path),
            "sha256": routing.data_sha256,
            "row_count": 2,
        },
        "routing": {
            "manifest_path": str(routing.path),
            "manifest_file_sha256": routing.file_sha256,
            "routing_assignment_sha256": routing.assignment_sha256,
        },
        "scope": {"stage_b_routing_split": "test"},
        "plan": {
            "stage_a_summary": {
                "mode": "exact_post_stage_a",
                "path": str(summary_path.resolve()),
                "file_sha256": _file_sha256(summary_path),
                "row_count": 4,
            },
            "actual_stage_b_prompt_set_sha256": "b" * 64,
            "provisional_structural_stage_b_prompt_set_sha256": None,
        },
    }
    report.update(overrides or {})
    preflight_path = tmp_path / "post_stage_a_preflight.json"
    preflight_path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    preflight_sha = _file_sha256(preflight_path)
    release = renderer.load_stage_b_release_contract(
        preflight_path,
        expected_file_sha256=preflight_sha,
        runtime=runtime,
        routing=routing,
        stage_a_summary_path=summary_path,
    )
    return release, preflight_path, summary_path, preflight_sha


def _write_stage_a_validation(
    tmp_path: Path,
    *,
    runtime: object,
    routing: object,
    summary_path: Path,
    passed: bool = True,
    count_overrides: dict[str, int] | None = None,
) -> tuple[object, Path, str]:
    counts = {
        "source_pairs": 2,
        "stage_a_expected": 4,
        "stage_a_raw": 4,
        "stage_a_flat": 4,
        "stage_a_pair_summary": 4,
    }
    counts.update(count_overrides or {})
    report = {
        "passed": passed,
        "source": {
            "csv": str(routing.data_path),
            "input_file_hash": routing.data_sha256,
            "usable_pairs": 2,
            "limit": None,
        },
        "design": {
            "validation_scope": "stage_a",
            "consistency_runs": runtime.values["consistency_runs"],
            "consistency_schedule": runtime.values["consistency_schedule"],
            "sampling_temperature": runtime.values["sampling_temperature"],
            "dataset_split": "full",
            "expected_question_routing_sha256": routing.assignment_sha256,
            "expected_inference_runtime_sha256": runtime.canonical_sha256,
            "question_routing": {
                "expected_assignment_sha256": routing.assignment_sha256,
                "assignment_sha256": routing.assignment_sha256,
                "raw_question_assignment_sha256": routing.assignment_sha256,
            },
        },
        "artifacts": [
            {
                "artifact_dir": str(summary_path.parent.parent),
                "stage_dirs": {"stage_a": str(summary_path.parent)},
                "validation_scope": "stage_a",
                "model_name": runtime.values["model_hf_name"],
                "model_revision": runtime.values["model_revision"],
                "dataset_split": "full",
                "inference_runtime_sha256_by_stage": {
                    "stage_a": runtime.canonical_sha256,
                },
                "counts": counts,
            }
        ],
        "error_count": 0 if passed else 1,
    }
    validation_path = tmp_path / "stage_a_artifact_validation.json"
    validation_path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    validation_sha = _file_sha256(validation_path)
    validation = renderer.load_stage_a_validation_contract(
        validation_path,
        expected_file_sha256=validation_sha,
        runtime=runtime,
        routing=routing,
        stage_a_summary_path=summary_path,
        dataset_split="full",
    )
    return validation, validation_path, validation_sha


def _render(
    *,
    stage: str,
    model: object,
    runtime: object,
    routing: object,
    release: object | None = None,
    validation: object | None = None,
) -> str:
    return renderer.render_silent_bias_job(
        stage=stage,
        job_name=f"silent-{stage.lower()}-test",
        model=model,
        time="01:00:00",
        cpus_per_task=8,
        gpus=runtime.values["tensor_parallel_size"],
        mem="240G",
        partition="gpu",
        qos="normal",
        account="research",
        artifact_root="/artifact/root",
        run_group="controlled_shift_test",
        runtime=runtime,
        routing=routing,
        python_bin=Path("/opt/biases/.venv/bin/python3.12"),
        stage_b_release=release,
        stage_a_validation=validation,
    )


def _assert_bash_syntax(tmp_path: Path, rendered: str) -> None:
    launcher = tmp_path / "launcher.slurm"
    launcher.write_text(rendered, encoding="utf-8")
    subprocess.run(["bash", "-n", str(launcher)], check=True)


def test_all_rendered_model_specs_are_registered() -> None:
    for model in renderer.MODEL_SPECS.values():
        assert get_model_profile(model.model_name).hf_model_name == model.model_name


@pytest.mark.parametrize(
    ("overrides", "refresh_embedded_sha", "message"),
    (
        ({"unfrozen_override": 1}, True, "unsupported fields"),
        ({"runtime_sha256": "0" * 64}, False, "does not match"),
        ({"seed": 17}, True, "requires runtime seed=0"),
        ({"max_num_seqs": None}, True, "positive integer"),
        ({"model_revision": None}, True, "pinned revision"),
        ({"model_registry_name": "qwen3-4b"}, True, "identity"),
    ),
)
def test_runtime_loader_rejects_unfrozen_or_invalid_contracts(
    tmp_path: Path,
    overrides: dict[str, object],
    refresh_embedded_sha: bool,
    message: str,
) -> None:
    model, runtime_path = _write_runtime(
        tmp_path,
        overrides=overrides,
        refresh_embedded_sha=refresh_embedded_sha,
    )

    with pytest.raises(ValueError, match=message):
        renderer.load_frozen_runtime_contract(runtime_path, model=model)


def test_stage_a_is_derived_from_frozen_runtime_and_is_shell_valid(
    tmp_path: Path,
) -> None:
    model, runtime, routing, _, _ = _load_contracts(tmp_path)
    rendered = _render(
        stage="A",
        model=model,
        runtime=runtime,
        routing=routing,
    )

    assert f'RUNTIME_FILE_SHA256="{runtime.file_sha256}"' in rendered
    assert f'RUNTIME_EMBEDDED_SHA256="{runtime.embedded_sha256}"' in rendered
    assert f'RUNTIME_CANONICAL_SHA256="{runtime.canonical_sha256}"' in rendered
    assert f'DATA_SHA256="{routing.data_sha256}"' in rendered
    assert f'ROUTING_MANIFEST_SHA256="{routing.file_sha256}"' in rendered
    assert 'PYTHON_BIN=/opt/biases/.venv/bin/python3.12' in rendered
    assert 'CONSISTENCY_RUNS="5"' in rendered
    assert 'CONSISTENCY_SCHEDULE="extremes"' in rendered
    assert 'SAMPLING_TEMPERATURE="0.65"' in rendered
    assert 'BATCH_SIZE="7"' in rendered
    assert 'MAX_NUM_BATCHED_TOKENS="16384"' in rendered
    assert 'MAX_NUM_SEQS="11"' in rendered
    assert 'INCLUDE_VERBALIZED_CONFIDENCE="0"' in rendered
    assert 'MODEL_REVISION="5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd"' in rendered
    assert 'visible_gpus == binding["tensor_parallel_size"]' in rendered
    assert "active engine versions mismatch" in rendered
    assert "load model weights" in rendered
    assert "VENV_PATH" not in rendered
    assert "EXTRA_ARGS" not in rendered
    assert "--no-resume" in rendered
    assert "@@" not in rendered
    _assert_bash_syntax(tmp_path, rendered)


def test_render_stage_b_requires_loaded_authorized_release(
    tmp_path: Path,
) -> None:
    model, runtime, routing, _, _ = _load_contracts(tmp_path)

    with pytest.raises(ValueError, match="authorized post-Stage-A preflight"):
        _render(
            stage="B",
            model=model,
            runtime=runtime,
            routing=routing,
        )


def test_stage_b_release_loader_rejects_wrong_file_sha(
    tmp_path: Path,
) -> None:
    _, runtime, routing, _, _ = _load_contracts(tmp_path)
    _, preflight_path, summary_path, _ = _write_stage_b_release(
        tmp_path,
        runtime=runtime,
        routing=routing,
    )

    with pytest.raises(ValueError, match="file SHA-256 does not match"):
        renderer.load_stage_b_release_contract(
            preflight_path,
            expected_file_sha256="0" * 64,
            runtime=runtime,
            routing=routing,
            stage_a_summary_path=summary_path,
        )


def test_stage_b_release_loader_requires_explicit_authorization(
    tmp_path: Path,
) -> None:
    _, runtime, routing, _, _ = _load_contracts(tmp_path)
    summary_path = tmp_path / "silent_bias_stage_a_pair_summary.jsonl"
    summary_path.write_text('{"question_id":"q-cal"}\n', encoding="utf-8")
    report = {
        "schema_version": 1,
        "status": "complete",
        "passed": True,
        "release_gate": {"stage_b_authorized": False},
    }
    preflight_path = tmp_path / "unauthorized.json"
    preflight_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="does not authorize"):
        renderer.load_stage_b_release_contract(
            preflight_path,
            expected_file_sha256=_file_sha256(preflight_path),
            runtime=runtime,
            routing=routing,
            stage_a_summary_path=summary_path,
        )


def test_stage_a_validation_loader_rejects_incomplete_observed_counts(
    tmp_path: Path,
) -> None:
    _, runtime, routing, _, _ = _load_contracts(tmp_path)
    _, _, summary_path, _ = _write_stage_b_release(
        tmp_path,
        runtime=runtime,
        routing=routing,
    )
    counts = {
        "source_pairs": 2,
        "stage_a_expected": 4,
        "stage_a_raw": 3,
        "stage_a_flat": 4,
        "stage_a_pair_summary": 4,
    }
    report = {
        "passed": True,
        "error_count": 0,
        "source": {
            "csv": str(routing.data_path),
            "input_file_hash": routing.data_sha256,
        },
        "design": {
            "validation_scope": "stage_a",
            "consistency_runs": runtime.values["consistency_runs"],
            "consistency_schedule": runtime.values["consistency_schedule"],
            "sampling_temperature": runtime.values["sampling_temperature"],
            "dataset_split": "full",
            "expected_question_routing_sha256": routing.assignment_sha256,
            "expected_inference_runtime_sha256": runtime.canonical_sha256,
            "question_routing": {
                "expected_assignment_sha256": routing.assignment_sha256,
                "assignment_sha256": routing.assignment_sha256,
                "raw_question_assignment_sha256": routing.assignment_sha256,
            },
        },
        "artifacts": [
            {
                "stage_dirs": {"stage_a": str(summary_path.parent)},
                "validation_scope": "stage_a",
                "model_name": runtime.values["model_hf_name"],
                "model_revision": runtime.values["model_revision"],
                "dataset_split": "full",
                "inference_runtime_sha256_by_stage": {
                    "stage_a": runtime.canonical_sha256,
                },
                "counts": counts,
            }
        ],
    }
    validation_path = tmp_path / "incomplete_stage_a_validation.json"
    validation_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="expected/observed counts"):
        renderer.load_stage_a_validation_contract(
            validation_path,
            expected_file_sha256=_file_sha256(validation_path),
            runtime=runtime,
            routing=routing,
            stage_a_summary_path=summary_path,
            dataset_split="full",
        )


def test_stage_a_validation_loader_rejects_wrong_file_sha(
    tmp_path: Path,
) -> None:
    _, runtime, routing, _, _ = _load_contracts(tmp_path)
    _, _, summary_path, _ = _write_stage_b_release(
        tmp_path,
        runtime=runtime,
        routing=routing,
    )
    _, validation_path, _ = _write_stage_a_validation(
        tmp_path,
        runtime=runtime,
        routing=routing,
        summary_path=summary_path,
    )

    with pytest.raises(ValueError, match="file SHA-256 does not match"):
        renderer.load_stage_a_validation_contract(
            validation_path,
            expected_file_sha256="0" * 64,
            runtime=runtime,
            routing=routing,
            stage_a_summary_path=summary_path,
            dataset_split="full",
        )


def test_stage_b_pins_post_stage_a_release_and_is_shell_valid(
    tmp_path: Path,
) -> None:
    model, runtime, routing, _, _ = _load_contracts(tmp_path)
    release, preflight_path, summary_path, preflight_sha = _write_stage_b_release(
        tmp_path,
        runtime=runtime,
        routing=routing,
    )
    validation, validation_path, validation_sha = _write_stage_a_validation(
        tmp_path,
        runtime=runtime,
        routing=routing,
        summary_path=summary_path,
    )
    rendered = _render(
        stage="B",
        model=model,
        runtime=runtime,
        routing=routing,
        release=release,
        validation=validation,
    )

    assert f"STAGE_B_PREFLIGHT={preflight_path.resolve()}" in rendered
    assert f'STAGE_B_PREFLIGHT_SHA256="{preflight_sha}"' in rendered
    assert f"STAGE_A_SUMMARY={summary_path.resolve()}" in rendered
    assert f'STAGE_A_SUMMARY_SHA256="{_file_sha256(summary_path)}"' in rendered
    assert f"STAGE_A_VALIDATION={validation_path.resolve()}" in rendered
    assert f'STAGE_A_VALIDATION_SHA256="{validation_sha}"' in rendered
    assert 'STAGE_B_ROUTING_SPLIT="test"' in rendered
    assert 'report.get("passed") is True' in rendered
    assert 'gate.get("stage_b_authorized") is True' in rendered
    assert "preflight runtime mapping mismatch" in rendered
    assert "preflight routing assignment mismatch" in rendered
    assert "Stage A validation expected/observed counts mismatch" in rendered
    assert "Stage A validation expected runtime SHA mismatch" in rendered
    assert '"${PYTHON_BIN}" main.py "${STAGE_B_COMMAND}"' in rendered
    _assert_bash_syntax(tmp_path, rendered)


def test_cli_renders_only_requested_stage_a(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, runtime_path = _write_runtime(tmp_path)
    data_path, routing_path = _write_routing(tmp_path)
    output_dir = tmp_path / "launchers"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            "--kind",
            "silent-bias",
            "--output-dir",
            str(output_dir),
            "--run-group",
            "controlled_shift_cli",
            "--models",
            "qwen25_32b",
            "--stage",
            "A",
            "--runtime-json",
            str(runtime_path),
            "--routing-manifest",
            str(routing_path),
            "--data-path",
            str(data_path),
            "--python-bin",
            "/cluster/venv/bin/python3.12",
        ],
    )

    renderer.main()

    assert (output_dir / "silent_bias_stage_a_qwen25_32b.slurm").is_file()
    assert not (output_dir / "silent_bias_stage_b_qwen25_32b.slurm").exists()


def test_cli_cannot_prerender_stage_b_without_post_stage_a_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, runtime_path = _write_runtime(tmp_path)
    data_path, routing_path = _write_routing(tmp_path)
    output_dir = tmp_path / "launchers"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            "--kind",
            "silent-bias",
            "--output-dir",
            str(output_dir),
            "--run-group",
            "controlled_shift_cli",
            "--models",
            "qwen25_32b",
            "--stage",
            "B",
            "--runtime-json",
            str(runtime_path),
            "--routing-manifest",
            str(routing_path),
            "--data-path",
            str(data_path),
            "--python-bin",
            "/cluster/venv/bin/python3.12",
        ],
    )

    with pytest.raises(ValueError, match="Stage B rendering requires"):
        renderer.main()

    assert not (output_dir / "silent_bias_stage_b_qwen25_32b.slurm").exists()


def test_cli_renders_stage_b_only_after_both_pinned_gates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, runtime_path = _write_runtime(tmp_path)
    data_path, routing_path = _write_routing(tmp_path)
    runtime = renderer.load_frozen_runtime_contract(runtime_path, model=model)
    routing = renderer.load_frozen_routing_contract(
        routing_path,
        data_path=data_path,
    )
    _, preflight_path, summary_path, preflight_sha = _write_stage_b_release(
        tmp_path,
        runtime=runtime,
        routing=routing,
    )
    _, validation_path, validation_sha = _write_stage_a_validation(
        tmp_path,
        runtime=runtime,
        routing=routing,
        summary_path=summary_path,
    )
    output_dir = tmp_path / "stage-b-launcher"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            "--kind",
            "silent-bias",
            "--output-dir",
            str(output_dir),
            "--run-group",
            "controlled_shift_cli",
            "--models",
            "qwen25_32b",
            "--stage",
            "B",
            "--runtime-json",
            str(runtime_path),
            "--routing-manifest",
            str(routing_path),
            "--data-path",
            str(data_path),
            "--python-bin",
            "/cluster/venv/bin/python3.12",
            "--stage-b-preflight",
            str(preflight_path),
            "--stage-b-preflight-sha256",
            preflight_sha,
            "--stage-a-summary",
            str(summary_path),
            "--stage-a-validation",
            str(validation_path),
            "--stage-a-validation-sha256",
            validation_sha,
        ],
    )

    renderer.main()

    launcher = output_dir / "silent_bias_stage_b_qwen25_32b.slurm"
    assert launcher.is_file()
    assert not (output_dir / "silent_bias_stage_a_qwen25_32b.slurm").exists()
    subprocess.run(["bash", "-n", str(launcher)], check=True)


def test_generic_renderer_keeps_existing_environment_override_behavior() -> None:
    rendered = renderer.render_job(
        job_name="generic",
        command="run-position",
        model=renderer.MODEL_SPECS["qwen3_4b"],
        output_slug="generic",
        data_file="mtbench_full.csv",
        time="01:00:00",
        partition=None,
        qos=None,
        account=None,
        artifact_root="${REPO_DIR}/artifacts",
    )

    assert 'DATA_PATH="${DATA_PATH:-${ARTIFACT_ROOT}/data/processed/mtbench_full.csv}"' in rendered
    assert 'MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B}"' in rendered
    assert "EXTRA_ARGS" in rendered
