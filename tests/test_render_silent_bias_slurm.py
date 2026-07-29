from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

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


def test_all_rendered_model_specs_are_registered() -> None:
    for model in renderer.MODEL_SPECS.values():
        assert get_model_profile(model.model_name).hf_model_name == model.model_name


def test_silent_bias_template_uses_current_stage_a_summary_name() -> None:
    model = renderer.MODEL_SPECS["qwen3_4b"]
    rendered = renderer.render_silent_bias_job(
        stage="B",
        job_name="silent-bias-test",
        model=model,
        data_file="mtbench_stratified_198.csv",
        time="01:00:00",
        cpus_per_task=4,
        gpus=1,
        mem="64G",
        partition=None,
        qos=None,
        account=None,
        artifact_root="${REPO_DIR}/artifacts",
        consistency_runs=4,
        consistency_schedule="extremes",
        sampling_temperature=0.7,
        include_verbalized_confidence=True,
        limit=198,
        stage_a_command="run-silent-bias-clean",
        stage_b_command="run-silent-bias-cued",
        stage_a_summary_file="silent_bias_stage_a_pair_summary.jsonl",
        max_model_len=8192,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        dtype="bfloat16",
    )

    assert "silent_bias_stage_a_pair_summary.jsonl" in rendered
    assert "run-silent-bias-cued" in rendered
    assert rendered.count("--consistency-schedule") == 1
    assert "@@" not in rendered


def test_silent_bias_stage_a_receives_consistency_schedule() -> None:
    model = renderer.MODEL_SPECS["qwen3_4b"]
    rendered = renderer.render_silent_bias_job(
        stage="A",
        job_name="silent-bias-test",
        model=model,
        data_file="mtbench_stratified_198.csv",
        time="01:00:00",
        cpus_per_task=4,
        gpus=1,
        mem="64G",
        partition=None,
        qos=None,
        account=None,
        artifact_root="${REPO_DIR}/artifacts",
        consistency_runs=4,
        consistency_schedule="extremes",
        sampling_temperature=0.7,
        include_verbalized_confidence=True,
        limit=198,
        stage_a_command="run-silent-bias-clean",
        stage_b_command="run-silent-bias-cued",
        stage_a_summary_file="silent_bias_stage_a_pair_summary.jsonl",
        max_model_len=8192,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        dtype="bfloat16",
    )

    assert "run-silent-bias-clean" in rendered
    assert rendered.count("--consistency-schedule") == 1
    assert "@@" not in rendered
