from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType

import pytest

from biases.models import get_model_profile
from biases.position_bias import (
    CONSTRAINED_LOGPROBS_MODE,
    JUDGE_OUTPUT_PARSER_VERSION,
    VERBALIZED_OUTPUT_PARSER_VERSION,
)
from biases.schemas import VerdictLabel
from biases.silent_bias_runner import (
    run_silent_bias_clean,
    run_silent_bias_cued,
)
from biases.utils import stable_hash


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "validate_silent_bias_artifacts.py"
)
MIGRATION_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "migrate_silent_bias_parser.py"
)


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "validate_silent_bias_artifacts",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


validation_module = _load_module()


def _load_migration_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "migrate_silent_bias_parser",
        MIGRATION_SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


migration_module = _load_migration_module()


class _FakeJudge:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.logprobs_mode = CONSTRAINED_LOGPROBS_MODE
        profile = get_model_profile(model_name)
        base_ids = {"A": 10, "B": 20, "tie": 30}
        self.decision_label_token_ids = {
            label: [
                base_ids[label] + offset
                for offset, _ in enumerate(profile.verdict_token_texts[label])
            ]
            for label in ("A", "B", "tie")
        }

    def render_messages(self, messages: list[dict[str, str]]) -> str:
        return "\n".join(
            f"{message['role']}:{message['content']}" for message in messages
        )

    def choose_verdict_batch(
        self,
        prompt_texts: list[str],
        seed: int,
        sampling_temperature: float,
    ) -> list[tuple[VerdictLabel, str, dict[str, float]]]:
        del seed, sampling_temperature
        return [
            (
                VerdictLabel.A,
                "A",
                {"A": 0.8, "B": 0.15, "tie": 0.05},
            )
            for _ in prompt_texts
        ]

    def verbalize_confidence_batch(
        self,
        prompt_texts: list[str],
        seed: int = 0,
        max_tokens: int = 24,
    ) -> list[tuple[VerdictLabel | None, str, float | None]]:
        del seed, max_tokens
        return [
            (VerdictLabel.A, "A\nConfidence: 80", 80.0)
            for _ in prompt_texts
        ]


def _write_source(path: Path) -> None:
    path.write_text(
        "\n".join(
            (
                "question_id,prompt,response_a,response_b,winner,turn,routing_split",
                "q1,Which is better?,Good answer,Bad answer,A,1,calibration",
            )
        )
        + "\n",
        encoding="utf-8",
    )


def _run_fixture(
    csv_path: Path,
    output_dir: Path,
    *,
    registry_name: str,
) -> None:
    profile = get_model_profile(registry_name)
    judge = _FakeJudge(profile.hf_model_name)
    stage_a = run_silent_bias_clean(
        csv_path=csv_path,
        output_dir=output_dir,
        model_name=registry_name,
        dataset_split="pilot",
        consistency_runs=2,
        consistency_schedule="extremes",
        include_verbalized_confidence=True,
        judge=judge,
    )
    run_silent_bias_cued(
        csv_path=csv_path,
        stage_a_summary_path=Path(stage_a["pair_summary_path"]),
        output_dir=output_dir,
        model_name=registry_name,
        dataset_split="pilot",
        consistency_runs=2,
        consistency_schedule="extremes",
        include_verbalized_confidence=True,
        judge=judge,
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows),
        encoding="utf-8",
    )


def _validate(
    csv_path: Path,
    *output_dirs: Path,
    min_verbalized_availability: float = 0.99,
) -> dict[str, object]:
    return validation_module.validate_artifact_directories(
        source_csv=csv_path,
        artifact_dirs=list(output_dirs),
        consistency_runs=2,
        consistency_schedule="extremes",
        sampling_temperature=0.7,
        dataset_split="pilot",
        min_verbalized_availability=min_verbalized_availability,
    )


def _degrade_parser_artifacts(output_dir: Path) -> None:
    for stage in ("stage_a", "stage_b"):
        raw_path = output_dir / f"silent_bias_{stage}_run_records.jsonl"
        score_path = (
            output_dir / f"silent_bias_{stage}_uncertainty_scores.jsonl"
        )
        pair_path = output_dir / f"silent_bias_{stage}_pair_summary.jsonl"
        summary_path = output_dir / f"silent_bias_{stage}_summary.json"
        raw_rows = _read_jsonl(raw_path)
        score_rows = _read_jsonl(score_path)
        pair_rows = _read_jsonl(pair_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for row in raw_rows:
            row["metadata"]["judge_output_parser_version"] = "legacy_v1"
            row["metadata"]["verbalized_output_parser_version"] = "legacy_v1"
            row["metadata"].pop("max_num_batched_tokens", None)
            row["metadata"].pop("max_num_seqs", None)
            row["uncertainty"]["logit"]["entropy"] = 999.0
            row["uncertainty"]["verbalized"]["confidence"] = 0.01
            if row["uncertainty"]["consistency"] is not None:
                row["uncertainty"]["consistency"]["agreement_rate"] = 0.01
        for row in score_rows:
            row["judge_output_parser_version"] = "legacy_v1"
            row["verbalized_output_parser_version"] = "legacy_v1"
            row.pop("max_num_batched_tokens", None)
            row.pop("max_num_seqs", None)
            row["entropy"] = 999.0
            row["verbalized_confidence"] = 0.01
            if row["consistency_agreement_rate"] is not None:
                row["consistency_agreement_rate"] = 0.01
        for row in pair_rows:
            row["judge_output_parser_version"] = "legacy_v1"
            row["verbalized_output_parser_version"] = "legacy_v1"
            row.pop("max_num_batched_tokens", None)
            row.pop("max_num_seqs", None)
        summary["judge_output_parser_version"] = "legacy_v1"
        summary["verbalized_output_parser_version"] = "legacy_v1"
        summary.pop("max_num_batched_tokens", None)
        summary.pop("max_num_seqs", None)
        _write_jsonl(raw_path, raw_rows)
        _write_jsonl(score_path, score_rows)
        _write_jsonl(pair_path, pair_rows)
        summary_path.write_text(
            f"{json.dumps(summary)}\n",
            encoding="utf-8",
        )


def _protected_record_fields(output_dir: Path) -> list[dict[str, object]]:
    protected: list[dict[str, object]] = []
    for stage in ("stage_a", "stage_b"):
        rows = _read_jsonl(
            output_dir / f"silent_bias_{stage}_run_records.jsonl"
        )
        for row in rows:
            protected.append(migration_module._preserved_fields(row))
    return protected


def _artifact_bytes(output_dir: Path) -> dict[str, bytes]:
    return {
        path.name: path.read_bytes()
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    }


def test_validator_accepts_complete_stage_grids_and_schedule(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    csv_path = tmp_path / "pilot.csv"
    output_dir = tmp_path / "qwen"
    _write_source(csv_path)
    _run_fixture(csv_path, output_dir, registry_name="qwen3-4b")

    report = _validate(csv_path, output_dir)

    assert report["passed"] is True
    assert report["error_count"] == 0
    counts = report["artifacts"][0]["counts"]
    assert counts == {
        "source_pairs": 1,
        "stage_a_expected": 2,
        "stage_a_raw": 2,
        "stage_a_flat": 2,
        "stage_a_pair_summary": 2,
        "stage_b_expected": 32,
        "stage_b_raw": 32,
        "stage_b_flat": 32,
        "stage_b_pair_summary": 32,
    }

    exit_code = validation_module.main(
        [
            "--source-csv",
            str(csv_path),
            "--artifact-dir",
            str(output_dir),
            "--consistency-runs",
            "2",
            "--consistency-schedule",
            "extremes",
            "--dataset-split",
            "pilot",
        ]
    )
    cli_report = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert cli_report["passed"] is True
    assert cli_report["design"]["min_verbalized_availability"] == 0.99
    parsed_args = validation_module.build_parser().parse_args(
        [
            "--source-csv",
            str(csv_path),
            "--artifact-dir",
            str(output_dir),
            "--consistency-runs",
            "2",
            "--consistency-schedule",
            "extremes",
            "--min-verbalized-availability",
            "0.75",
        ]
    )
    assert parsed_args.min_verbalized_availability == 0.75


def test_validator_requires_processed_logprobs_mode_in_every_artifact_layer(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "pilot.csv"
    output_dir = tmp_path / "qwen"
    _write_source(csv_path)
    _run_fixture(csv_path, output_dir, registry_name="qwen3-4b")

    raw_path = output_dir / "silent_bias_stage_a_run_records.jsonl"
    flat_path = output_dir / "silent_bias_stage_a_uncertainty_scores.jsonl"
    pair_path = output_dir / "silent_bias_stage_a_pair_summary.jsonl"
    summary_path = output_dir / "silent_bias_stage_a_summary.json"
    raw_rows = _read_jsonl(raw_path)
    flat_rows = _read_jsonl(flat_path)
    pair_rows = _read_jsonl(pair_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    raw_rows[0]["spec"].pop("logprobs_mode")
    raw_rows[0]["metadata"].pop("logprobs_mode")
    flat_rows[0].pop("logprobs_mode")
    pair_rows[0].pop("logprobs_mode")
    summary.pop("logprobs_mode")
    _write_jsonl(raw_path, raw_rows)
    _write_jsonl(flat_path, flat_rows)
    _write_jsonl(pair_path, pair_rows)
    summary_path.write_text(
        f"{json.dumps(summary)}\n",
        encoding="utf-8",
    )

    report = _validate(csv_path, output_dir)

    assert report["passed"] is False
    assert report["error_counts_by_code"]["logprobs_mode_mismatch"] == 5


def test_validator_rejects_missing_processed_verdict_token_contract(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "pilot.csv"
    output_dir = tmp_path / "qwen"
    _write_source(csv_path)
    _run_fixture(csv_path, output_dir, registry_name="qwen3-4b")

    for stage in ("stage_a", "stage_b"):
        raw_path = output_dir / f"silent_bias_{stage}_run_records.jsonl"
        flat_path = output_dir / f"silent_bias_{stage}_uncertainty_scores.jsonl"
        pair_path = output_dir / f"silent_bias_{stage}_pair_summary.jsonl"
        summary_path = output_dir / f"silent_bias_{stage}_summary.json"
        raw_rows = _read_jsonl(raw_path)
        flat_rows = _read_jsonl(flat_path)
        pair_rows = _read_jsonl(pair_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for row in raw_rows:
            row["spec"].pop("verdict_token_texts")
            row["spec"].pop("verdict_token_ids")
            row["spec_hash"] = stable_hash(row["spec"])
        for rows in (flat_rows, pair_rows):
            for row in rows:
                row.pop("verdict_token_texts")
                row.pop("verdict_token_ids")
        summary.pop("verdict_token_texts")
        summary.pop("verdict_token_ids")
        _write_jsonl(raw_path, raw_rows)
        _write_jsonl(flat_path, flat_rows)
        _write_jsonl(pair_path, pair_rows)
        summary_path.write_text(f"{json.dumps(summary)}\n", encoding="utf-8")

    report = _validate(csv_path, output_dir)

    assert report["passed"] is False
    assert (
        report["error_counts_by_code"]["verdict_token_contract_mismatch"] >= 2
    )


def test_validator_requires_current_status_and_stage_status_counts(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "pilot.csv"
    base_dir = tmp_path / "base"
    _write_source(csv_path)
    _run_fixture(csv_path, base_dir, registry_name="qwen3-4b")

    missing_record_status = tmp_path / "missing-record-status"
    shutil.copytree(base_dir, missing_record_status)
    raw_path = (
        missing_record_status / "silent_bias_stage_a_run_records.jsonl"
    )
    raw_rows = _read_jsonl(raw_path)
    raw_rows[0]["metadata"].pop("verbalized_parse_status")
    _write_jsonl(raw_path, raw_rows)

    record_report = _validate(csv_path, missing_record_status)

    assert record_report["passed"] is False
    assert (
        record_report["error_counts_by_code"]["stored_verbalized_mismatch"]
        >= 1
    )

    missing_summary_counts = tmp_path / "missing-summary-counts"
    shutil.copytree(base_dir, missing_summary_counts)
    summary_path = (
        missing_summary_counts / "silent_bias_stage_a_summary.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.pop("verbalized_parse_status_counts")
    summary_path.write_text(f"{json.dumps(summary)}\n", encoding="utf-8")

    summary_report = _validate(csv_path, missing_summary_counts)

    assert summary_report["passed"] is False
    assert (
        summary_report["error_counts_by_code"][
            "verbalized_missingness_summary_mismatch"
        ]
        == 1
    )


def test_validator_rejects_invalid_verbalized_availability_threshold(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "pilot.csv"
    output_dir = tmp_path / "qwen"
    _write_source(csv_path)
    _run_fixture(csv_path, output_dir, registry_name="qwen3-4b")

    with pytest.raises(
        ValueError,
        match="min_verbalized_availability",
    ):
        _validate(
            csv_path,
            output_dir,
            min_verbalized_availability=1.01,
        )


def test_validator_rejects_bad_probabilities_and_clean_links(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    csv_path = tmp_path / "pilot.csv"
    output_dir = tmp_path / "qwen"
    _write_source(csv_path)
    _run_fixture(csv_path, output_dir, registry_name="qwen3-4b")

    raw_path = output_dir / "silent_bias_stage_b_run_records.jsonl"
    score_path = output_dir / "silent_bias_stage_b_uncertainty_scores.jsonl"
    raw_rows = _read_jsonl(raw_path)
    score_rows = _read_jsonl(score_path)
    raw_rows[0]["raw_prompt_logprobs"] = {
        "A": 0.8,
        "B": 0.2,
        "tie": 0.2,
    }
    raw_rows[0]["condition"]["clean_record_id"] = "wrong-clean-record"
    score_rows[0]["label_prob_A"] = 0.8
    score_rows[0]["label_prob_B"] = 0.2
    score_rows[0]["label_prob_tie"] = 0.2
    score_rows[0]["clean_record_id"] = "wrong-clean-record"
    _write_jsonl(raw_path, raw_rows)
    _write_jsonl(score_path, score_rows)

    report = _validate(csv_path, output_dir)

    assert report["passed"] is False
    codes = report["error_counts_by_code"]
    assert codes["invalid_label_probabilities"] >= 2
    assert codes["stage_b_clean_link_mismatch"] >= 1

    exit_code = validation_module.main(
        [
            "--source-csv",
            str(csv_path),
            "--artifact-dir",
            str(output_dir),
            "--consistency-runs",
            "2",
            "--consistency-schedule",
            "extremes",
            "--dataset-split",
            "pilot",
        ]
    )
    captured = capsys.readouterr()
    cli_report = json.loads(captured.out)
    assert exit_code == 1
    assert cli_report["passed"] is False


def test_validator_compares_grids_across_model_directories(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "pilot.csv"
    qwen_dir = tmp_path / "qwen"
    mistral_dir = tmp_path / "mistral"
    _write_source(csv_path)
    _run_fixture(csv_path, qwen_dir, registry_name="qwen3-4b")
    _run_fixture(
        csv_path,
        mistral_dir,
        registry_name="mistral-7b-instruct-v0.3",
    )

    incomplete_dir = tmp_path / "mistral-incomplete"
    shutil.copytree(mistral_dir, incomplete_dir)
    for filename in (
        "silent_bias_stage_b_run_records.jsonl",
        "silent_bias_stage_b_uncertainty_scores.jsonl",
    ):
        path = incomplete_dir / filename
        rows = _read_jsonl(path)
        _write_jsonl(path, rows[:-1])

    report = _validate(csv_path, qwen_dir, incomplete_dir)

    assert report["passed"] is False
    assert report["error_counts_by_code"]["cross_model_grid_mismatch"] == 1


def test_validator_recomputes_parser_derived_fields_and_versions(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "pilot.csv"
    base_dir = tmp_path / "base"
    _write_source(csv_path)
    _run_fixture(csv_path, base_dir, registry_name="qwen3-4b")

    raw_mutation_dir = tmp_path / "raw-mutation"
    shutil.copytree(base_dir, raw_mutation_dir)
    raw_path = raw_mutation_dir / "silent_bias_stage_a_run_records.jsonl"
    raw_rows = _read_jsonl(raw_path)
    raw_rows[0]["raw_output"] = "B"
    _write_jsonl(raw_path, raw_rows)
    raw_report = _validate(csv_path, raw_mutation_dir)
    assert raw_report["passed"] is False
    assert raw_report["error_counts_by_code"]["strict_parser_mismatch"] >= 1

    entropy_mutation_dir = tmp_path / "entropy-mutation"
    shutil.copytree(base_dir, entropy_mutation_dir)
    entropy_raw_path = (
        entropy_mutation_dir / "silent_bias_stage_a_run_records.jsonl"
    )
    entropy_score_path = (
        entropy_mutation_dir / "silent_bias_stage_a_uncertainty_scores.jsonl"
    )
    entropy_raw_rows = _read_jsonl(entropy_raw_path)
    entropy_score_rows = _read_jsonl(entropy_score_path)
    entropy_raw_rows[0]["uncertainty"]["logit"]["entropy"] = 999.0
    entropy_score_rows[0]["entropy"] = 999.0
    _write_jsonl(entropy_raw_path, entropy_raw_rows)
    _write_jsonl(entropy_score_path, entropy_score_rows)
    entropy_report = _validate(csv_path, entropy_mutation_dir)
    assert entropy_report["passed"] is False
    assert (
        entropy_report["error_counts_by_code"]["derived_uncertainty_mismatch"]
        >= 1
    )

    stale_dir = tmp_path / "stale-parser"
    shutil.copytree(base_dir, stale_dir)
    stale_raw_path = stale_dir / "silent_bias_stage_a_run_records.jsonl"
    stale_score_path = (
        stale_dir / "silent_bias_stage_a_uncertainty_scores.jsonl"
    )
    stale_pair_path = stale_dir / "silent_bias_stage_a_pair_summary.jsonl"
    stale_summary_path = stale_dir / "silent_bias_stage_a_summary.json"
    stale_raw_rows = _read_jsonl(stale_raw_path)
    stale_score_rows = _read_jsonl(stale_score_path)
    stale_pair_rows = _read_jsonl(stale_pair_path)
    stale_summary = json.loads(stale_summary_path.read_text(encoding="utf-8"))
    stale_raw_rows[0]["metadata"]["judge_output_parser_version"] = "legacy_v1"
    stale_raw_rows[0]["metadata"]["verbalized_output_parser_version"] = (
        "legacy_v1"
    )
    stale_score_rows[0]["judge_output_parser_version"] = "legacy_v1"
    stale_score_rows[0]["verbalized_output_parser_version"] = "legacy_v1"
    stale_pair_rows[0]["judge_output_parser_version"] = "legacy_v1"
    stale_pair_rows[0]["verbalized_output_parser_version"] = "legacy_v1"
    stale_summary["judge_output_parser_version"] = "legacy_v1"
    stale_summary["verbalized_output_parser_version"] = "legacy_v1"
    _write_jsonl(stale_raw_path, stale_raw_rows)
    _write_jsonl(stale_score_path, stale_score_rows)
    _write_jsonl(stale_pair_path, stale_pair_rows)
    stale_summary_path.write_text(
        f"{json.dumps(stale_summary)}\n",
        encoding="utf-8",
    )
    stale_report = _validate(csv_path, stale_dir)
    assert stale_report["passed"] is False
    assert stale_report["error_counts_by_code"]["stale_parser_version"] >= 4
    assert (
        stale_report["error_counts_by_code"][
            "stale_verbalized_parser_version"
        ]
        >= 4
    )


def test_parser_migration_rebuilds_destination_and_preserves_links(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "pilot.csv"
    source_dir = tmp_path / "legacy"
    destination_dir = tmp_path / "strict"
    report_path = tmp_path / "migration-report.json"
    _write_source(csv_path)
    _run_fixture(csv_path, source_dir, registry_name="qwen3-4b")
    stage_a_raw_path = (
        source_dir / "silent_bias_stage_a_run_records.jsonl"
    )
    stage_a_raw_rows = _read_jsonl(stage_a_raw_path)
    recovered_forms = ("1: A\n2: 81", "1) A\n2) 64.5")
    for row, raw_output in zip(
        stage_a_raw_rows,
        recovered_forms,
        strict=True,
    ):
        row["metadata"]["verbalized_raw_output"] = raw_output
    _write_jsonl(stage_a_raw_path, stage_a_raw_rows)
    _degrade_parser_artifacts(source_dir)
    protected_before = _protected_record_fields(source_dir)

    report = migration_module.migrate_artifact_directory(
        source_dir=source_dir,
        destination_dir=destination_dir,
        report_path=report_path,
    )

    assert report["passed"] is True
    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    assert report["records"] == {"stage_a": 2, "stage_b": 32}
    assert report["dropped_incomplete_tail"] == {
        "stage_a": False,
        "stage_b": False,
    }
    assert report["logprobs_mode"] == {
        "stage_a": CONSTRAINED_LOGPROBS_MODE,
        "stage_b": CONSTRAINED_LOGPROBS_MODE,
    }
    assert _protected_record_fields(destination_dir) == protected_before
    assert _validate(csv_path, destination_dir)["passed"] is True
    expected_filenames = {
        f"silent_bias_{stage}_{suffix}.jsonl"
        for stage in ("stage_a", "stage_b")
        for suffix in ("run_records", "uncertainty_scores", "pair_summary")
    } | {
        f"silent_bias_{stage}_{suffix}.json"
        for stage in ("stage_a", "stage_b")
        for suffix in ("summary", "planning_issues")
    }
    assert set(report["source_artifact_sha256"]) == expected_filenames
    assert set(report["rematerialized_artifact_sha256"]) == expected_filenames
    for filename, expected_hash in report["source_artifact_sha256"].items():
        assert (
            hashlib.sha256((source_dir / filename).read_bytes()).hexdigest()
            == expected_hash
        )
    for filename, expected_hash in report[
        "rematerialized_artifact_sha256"
    ].items():
        assert (
            hashlib.sha256((destination_dir / filename).read_bytes()).hexdigest()
            == expected_hash
        )
    source_rows = _read_jsonl(
        source_dir / "silent_bias_stage_a_run_records.jsonl"
    )
    destination_rows = _read_jsonl(
        destination_dir / "silent_bias_stage_a_run_records.jsonl"
    )
    assert (
        source_rows[0]["metadata"]["judge_output_parser_version"]
        == "legacy_v1"
    )
    assert (
        destination_rows[0]["metadata"]["judge_output_parser_version"]
        == JUDGE_OUTPUT_PARSER_VERSION
        == "strict_v3"
    )
    assert (
        destination_rows[0]["metadata"]["verbalized_output_parser_version"]
        == VERBALIZED_OUTPUT_PARSER_VERSION
        == "strict_v3"
    )
    assert destination_rows[0]["metadata"]["max_num_batched_tokens"] is None
    assert destination_rows[0]["metadata"]["max_num_seqs"] is None
    assert destination_rows[0]["uncertainty"]["logit"]["entropy"] != 999.0
    assert (
        destination_rows[0]["uncertainty"]["verbalized"]["confidence"]
        == pytest.approx(0.81)
    )
    assert [
        row["metadata"]["verbalized_raw_output"] for row in source_rows
    ] == list(recovered_forms)
    assert [
        row["metadata"]["verbalized_raw_output"] for row in destination_rows
    ] == list(recovered_forms)
    assert {
        row["metadata"]["verbalized_parse_status"]
        for row in destination_rows
    } == {"parsed"}
    assert [
        row["uncertainty"]["verbalized"]["confidence"]
        for row in destination_rows
    ] == pytest.approx([0.81, 0.645])
    destination_flat_rows = _read_jsonl(
        destination_dir / "silent_bias_stage_a_uncertainty_scores.jsonl"
    )
    destination_pair_rows = _read_jsonl(
        destination_dir / "silent_bias_stage_a_pair_summary.jsonl"
    )
    assert [
        row["verbalized_confidence"] for row in destination_flat_rows
    ] == pytest.approx([0.81, 0.645])
    assert {
        row["verbalized_parse_status"] for row in destination_pair_rows
    } == {"parsed"}
    for stage in ("stage_a", "stage_b"):
        summary = json.loads(
            (
                destination_dir / f"silent_bias_{stage}_summary.json"
            ).read_text(encoding="utf-8")
        )
        assert (
            summary["judge_output_parser_version"]
            == JUDGE_OUTPUT_PARSER_VERSION
        )
        assert (
            summary["verbalized_output_parser_version"]
            == VERBALIZED_OUTPUT_PARSER_VERSION
        )
        assert summary["logprobs_mode"] == CONSTRAINED_LOGPROBS_MODE
        assert summary["max_num_batched_tokens"] is None
        assert summary["max_num_seqs"] is None


def test_validator_rejects_strict_v2_verbalized_provenance(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "pilot.csv"
    stale_dir = tmp_path / "stale-v2"
    _write_source(csv_path)
    _run_fixture(csv_path, stale_dir, registry_name="qwen3-4b")

    for stage in ("stage_a", "stage_b"):
        raw_path = stale_dir / f"silent_bias_{stage}_run_records.jsonl"
        score_path = (
            stale_dir / f"silent_bias_{stage}_uncertainty_scores.jsonl"
        )
        pair_path = stale_dir / f"silent_bias_{stage}_pair_summary.jsonl"
        summary_path = stale_dir / f"silent_bias_{stage}_summary.json"
        raw_rows = _read_jsonl(raw_path)
        score_rows = _read_jsonl(score_path)
        pair_rows = _read_jsonl(pair_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for row in raw_rows:
            row["metadata"]["verbalized_output_parser_version"] = "strict_v2"
        for row in score_rows:
            row["verbalized_output_parser_version"] = "strict_v2"
        for row in pair_rows:
            row["verbalized_output_parser_version"] = "strict_v2"
        summary["verbalized_output_parser_version"] = "strict_v2"
        _write_jsonl(raw_path, raw_rows)
        _write_jsonl(score_path, score_rows)
        _write_jsonl(pair_path, pair_rows)
        summary_path.write_text(
            f"{json.dumps(summary)}\n",
            encoding="utf-8",
        )

    report = _validate(csv_path, stale_dir)

    assert report["passed"] is False
    assert (
        report["error_counts_by_code"]["stale_verbalized_parser_version"]
        >= 8
    )


def test_parser_migration_marks_undeclared_legacy_logprobs_as_raw(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "pilot.csv"
    source_dir = tmp_path / "legacy"
    destination_dir = tmp_path / "migrated"
    _write_source(csv_path)
    _run_fixture(csv_path, source_dir, registry_name="qwen3-4b")

    for stage in ("stage_a", "stage_b"):
        raw_path = source_dir / f"silent_bias_{stage}_run_records.jsonl"
        flat_path = (
            source_dir / f"silent_bias_{stage}_uncertainty_scores.jsonl"
        )
        pair_path = source_dir / f"silent_bias_{stage}_pair_summary.jsonl"
        summary_path = source_dir / f"silent_bias_{stage}_summary.json"
        raw_rows = _read_jsonl(raw_path)
        flat_rows = _read_jsonl(flat_path)
        pair_rows = _read_jsonl(pair_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for row in raw_rows:
            row["spec"].pop("logprobs_mode")
            row["metadata"].pop("logprobs_mode")
            row["spec_hash"] = stable_hash(row["spec"])
        for row in flat_rows:
            row.pop("logprobs_mode")
        for row in pair_rows:
            row.pop("logprobs_mode")
        summary.pop("logprobs_mode")
        _write_jsonl(raw_path, raw_rows)
        _write_jsonl(flat_path, flat_rows)
        _write_jsonl(pair_path, pair_rows)
        summary_path.write_text(
            f"{json.dumps(summary)}\n",
            encoding="utf-8",
        )

    protected_before = _protected_record_fields(source_dir)
    report = migration_module.migrate_artifact_directory(
        source_dir=source_dir,
        destination_dir=destination_dir,
    )

    assert report["logprobs_mode"] == {
        "stage_a": "raw_logprobs",
        "stage_b": "raw_logprobs",
    }
    assert _protected_record_fields(destination_dir) == protected_before
    for stage in ("stage_a", "stage_b"):
        raw_rows = _read_jsonl(
            destination_dir / f"silent_bias_{stage}_run_records.jsonl"
        )
        flat_rows = _read_jsonl(
            destination_dir
            / f"silent_bias_{stage}_uncertainty_scores.jsonl"
        )
        pair_rows = _read_jsonl(
            destination_dir / f"silent_bias_{stage}_pair_summary.jsonl"
        )
        summary = json.loads(
            (
                destination_dir / f"silent_bias_{stage}_summary.json"
            ).read_text(encoding="utf-8")
        )
        assert {
            (
                row["spec"].get("logprobs_mode"),
                row["metadata"]["logprobs_mode"],
            )
            for row in raw_rows
        } == {(None, "raw_logprobs")}
        assert all(
            row["spec_hash"] == stable_hash(row["spec"])
            for row in raw_rows
        )
        assert {row["logprobs_mode"] for row in flat_rows} == {
            "raw_logprobs"
        }
        assert {row["logprobs_mode"] for row in pair_rows} == {
            "raw_logprobs"
        }
        assert summary["logprobs_mode"] == "raw_logprobs"

    validation = _validate(csv_path, destination_dir)
    assert validation["passed"] is False
    assert validation["error_counts_by_code"]["logprobs_mode_mismatch"] > 0


def test_parser_migration_dry_run_and_in_place_backups(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    csv_path = tmp_path / "pilot.csv"
    source_dir = tmp_path / "legacy"
    _write_source(csv_path)
    _run_fixture(csv_path, source_dir, registry_name="qwen3-4b")
    _degrade_parser_artifacts(source_dir)
    raw_path = source_dir / "silent_bias_stage_a_run_records.jsonl"
    before = raw_path.read_bytes()

    dry_report = migration_module.migrate_artifact_directory(
        source_dir=source_dir,
        dry_run=True,
    )

    assert dry_report["passed"] is True
    assert dry_report["files_written"] == []
    assert raw_path.read_bytes() == before
    exit_code = migration_module.main(
        ["--source-dir", str(source_dir), "--dry-run"]
    )
    cli_report = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert cli_report["mode"] == "dry_run"
    assert raw_path.read_bytes() == before

    migrated = migration_module.migrate_artifact_directory(
        source_dir=source_dir,
        in_place=True,
        backup_suffix=".legacy.bak",
    )

    assert migrated["passed"] is True
    assert len(migrated["backup_files"]) == 8
    assert all(Path(path).is_file() for path in migrated["backup_files"])
    assert _validate(csv_path, source_dir)["passed"] is True


def test_parser_migration_fails_before_writes_on_ambiguous_output(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "pilot.csv"
    source_dir = tmp_path / "legacy"
    destination_dir = tmp_path / "strict"
    _write_source(csv_path)
    _run_fixture(csv_path, source_dir, registry_name="qwen3-4b")
    _degrade_parser_artifacts(source_dir)
    raw_path = source_dir / "silent_bias_stage_a_run_records.jsonl"
    rows = _read_jsonl(raw_path)
    rows[0]["raw_output"] = "A or B"
    _write_jsonl(raw_path, rows)

    with pytest.raises(ValueError, match="unparseable or ambiguous"):
        migration_module.migrate_artifact_directory(
            source_dir=source_dir,
            destination_dir=destination_dir,
        )

    assert not destination_dir.exists()


def test_parser_migration_marks_unparseable_verbalized_channel_missing(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "pilot.csv"
    source_dir = tmp_path / "legacy"
    destination_dir = tmp_path / "strict"
    _write_source(csv_path)
    _run_fixture(csv_path, source_dir, registry_name="qwen3-4b")
    _degrade_parser_artifacts(source_dir)
    raw_path = source_dir / "silent_bias_stage_a_run_records.jsonl"
    rows = _read_jsonl(raw_path)
    raw_verbalized = "I considered 70 factors but cannot give confidence."
    rows[0]["metadata"]["verbalized_raw_output"] = raw_verbalized
    _write_jsonl(raw_path, rows)

    report = migration_module.migrate_artifact_directory(
        source_dir=source_dir,
        destination_dir=destination_dir,
    )

    migrated = _read_jsonl(
        destination_dir / "silent_bias_stage_a_run_records.jsonl"
    )
    migrated_by_id = {row["record_id"]: row for row in migrated}
    row = migrated_by_id[rows[0]["record_id"]]
    assert row["metadata"]["verbalized_raw_output"] == raw_verbalized
    assert row["metadata"]["verbalized_parse_status"] == "unparseable"
    assert row["metadata"]["verbalized_verdict"] is None
    assert row["uncertainty"]["verbalized"] == {
        "confidence": None,
        "uncertainty": None,
        "verdict": None,
    }
    assert (
        report["verbalized_parse_status_counts"]["stage_a"]["unparseable"]
        == 1
    )
    assert _validate(
        csv_path,
        destination_dir,
        min_verbalized_availability=0.0,
    )["passed"] is True


def test_validator_treats_unparseable_verbalized_output_as_missing_channel(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "pilot.csv"
    output_dir = tmp_path / "current"
    _write_source(csv_path)
    _run_fixture(csv_path, output_dir, registry_name="qwen3-4b")
    raw_path = output_dir / "silent_bias_stage_a_run_records.jsonl"
    flat_path = output_dir / "silent_bias_stage_a_uncertainty_scores.jsonl"
    pair_path = output_dir / "silent_bias_stage_a_pair_summary.jsonl"
    summary_path = output_dir / "silent_bias_stage_a_summary.json"
    raw_rows = _read_jsonl(raw_path)
    flat_rows = _read_jsonl(flat_path)
    pair_rows = _read_jsonl(pair_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    record_id = raw_rows[0]["record_id"]
    raw_rows[0]["metadata"]["verbalized_raw_output"] = (
        "I considered 70 factors but cannot give confidence."
    )
    raw_rows[0]["metadata"]["verbalized_verdict"] = None
    raw_rows[0]["metadata"]["verbalized_parse_status"] = "unparseable"
    raw_rows[0]["uncertainty"]["verbalized"] = {
        "confidence": None,
        "uncertainty": None,
        "verdict": None,
    }
    flat_row = next(row for row in flat_rows if row["record_id"] == record_id)
    flat_row["verbalized_confidence"] = None
    flat_row["verbalized_uncertainty"] = None
    flat_row["verbalized_verdict"] = None
    flat_row["verbalized_parse_status"] = "unparseable"
    pair_row = next(row for row in pair_rows if row["record_id"] == record_id)
    pair_row["verbalized_parse_status"] = "unparseable"
    summary["verbalized_parse_status_counts"] = {
        "parsed": len(raw_rows) - 1,
        "unparseable": 1,
    }
    _write_jsonl(raw_path, raw_rows)
    _write_jsonl(flat_path, flat_rows)
    _write_jsonl(pair_path, pair_rows)
    summary_path.write_text(
        f"{json.dumps(summary)}\n",
        encoding="utf-8",
    )

    strict_report = _validate(csv_path, output_dir)
    report = _validate(
        csv_path,
        output_dir,
        min_verbalized_availability=0.0,
    )

    assert strict_report["passed"] is False
    assert (
        strict_report["error_counts_by_code"][
            "verbalized_availability_below_minimum"
        ]
        == 1
    )
    assert report["passed"] is True


def test_parser_migration_rejects_semantically_wrong_clean_partner(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "pilot.csv"
    source_dir = tmp_path / "legacy"
    _write_source(csv_path)
    _run_fixture(csv_path, source_dir, registry_name="qwen3-4b")
    _degrade_parser_artifacts(source_dir)
    stage_a_rows = _read_jsonl(
        source_dir / "silent_bias_stage_a_run_records.jsonl"
    )
    stage_b_path = source_dir / "silent_bias_stage_b_run_records.jsonl"
    stage_b_rows = _read_jsonl(stage_b_path)
    correct_clean_id = stage_b_rows[0]["condition"]["clean_record_id"]
    wrong_clean_id = next(
        row["record_id"]
        for row in stage_a_rows
        if row["record_id"] != correct_clean_id
    )
    stage_b_rows[0]["condition"]["clean_record_id"] = wrong_clean_id
    stage_b_rows[0]["metadata"]["clean_record_id"] = wrong_clean_id
    _write_jsonl(stage_b_path, stage_b_rows)

    with pytest.raises(ValueError, match="semantic condition mismatch"):
        migration_module.migrate_artifact_directory(
            source_dir=source_dir,
            destination_dir=tmp_path / "strict",
        )


def test_parser_migration_preserves_unknown_raw_schema_fields(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "pilot.csv"
    source_dir = tmp_path / "legacy"
    destination_dir = tmp_path / "strict"
    _write_source(csv_path)
    _run_fixture(csv_path, source_dir, registry_name="qwen3-4b")
    _degrade_parser_artifacts(source_dir)
    raw_path = source_dir / "silent_bias_stage_a_run_records.jsonl"
    rows = _read_jsonl(raw_path)
    rows[0]["future_top_level"] = {"keep": [1, 2, 3]}
    rows[0]["spec"]["future_spec"] = "keep-spec"
    rows[0]["condition"]["future_condition"] = "keep-condition"
    _write_jsonl(raw_path, rows)

    migration_module.migrate_artifact_directory(
        source_dir=source_dir,
        destination_dir=destination_dir,
    )

    migrated = _read_jsonl(
        destination_dir / "silent_bias_stage_a_run_records.jsonl"
    )
    migrated_by_id = {row["record_id"]: row for row in migrated}
    migrated_row = migrated_by_id[rows[0]["record_id"]]
    assert migrated_row["future_top_level"] == {"keep": [1, 2, 3]}
    assert migrated_row["spec"]["future_spec"] == "keep-spec"
    assert migrated_row["condition"]["future_condition"] == "keep-condition"


def test_parser_migration_supports_stage_a_only_and_partial_stage_b(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "pilot.csv"
    full_dir = tmp_path / "full"
    _write_source(csv_path)
    _run_fixture(csv_path, full_dir, registry_name="qwen3-4b")
    _degrade_parser_artifacts(full_dir)

    stage_a_only = tmp_path / "stage-a-only"
    shutil.copytree(full_dir, stage_a_only)
    for path in stage_a_only.glob("silent_bias_stage_b_*"):
        path.unlink()
    (stage_a_only / "silent_bias_stage_a_summary.json").unlink()
    stage_a_destination = tmp_path / "stage-a-migrated"
    stage_a_report = migration_module.migrate_artifact_directory(
        source_dir=stage_a_only,
        destination_dir=stage_a_destination,
    )
    assert stage_a_report["stages_migrated"] == ["stage_a"]
    assert (
        stage_a_destination / "silent_bias_stage_a_pair_summary.jsonl"
    ).is_file()
    assert not (
        stage_a_destination / "silent_bias_stage_b_run_records.jsonl"
    ).exists()
    reconstructed = json.loads(
        (
            stage_a_destination / "silent_bias_stage_a_summary.json"
        ).read_text(encoding="utf-8")
    )
    assert reconstructed["migration_reconstructed_summary"] is True

    partial_dir = tmp_path / "partial-stage-b"
    shutil.copytree(full_dir, partial_dir)
    partial_raw_path = (
        partial_dir / "silent_bias_stage_b_run_records.jsonl"
    )
    partial_rows = _read_jsonl(partial_raw_path)[:5]
    partial_raw_path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in partial_rows)
        + '{"record_id":"truncated',
        encoding="utf-8",
    )
    for filename in (
        "silent_bias_stage_b_uncertainty_scores.jsonl",
        "silent_bias_stage_b_pair_summary.jsonl",
        "silent_bias_stage_b_summary.json",
    ):
        (partial_dir / filename).unlink()
    partial_destination = tmp_path / "partial-migrated"
    partial_report = migration_module.migrate_artifact_directory(
        source_dir=partial_dir,
        destination_dir=partial_destination,
    )
    assert partial_report["records"]["stage_b"] == 5
    assert partial_report["dropped_incomplete_tail"]["stage_b"] is True
    assert len(
        _read_jsonl(
            partial_destination
            / "silent_bias_stage_b_uncertainty_scores.jsonl"
        )
    ) == 5


def test_parser_migration_destination_transaction_rolls_back_and_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path = tmp_path / "pilot.csv"
    source_dir = tmp_path / "legacy"
    destination_dir = tmp_path / "strict"
    _write_source(csv_path)
    _run_fixture(csv_path, source_dir, registry_name="qwen3-4b")
    _degrade_parser_artifacts(source_dir)
    original_installer = migration_module._install_new_file
    calls = 0

    def _fail_once(source: Path, target: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("simulated destination commit failure")
        original_installer(source, target)

    monkeypatch.setattr(migration_module, "_install_new_file", _fail_once)
    with pytest.raises(OSError, match="simulated"):
        migration_module.migrate_artifact_directory(
            source_dir=source_dir,
            destination_dir=destination_dir,
        )
    assert not any(
        path.is_file() for path in destination_dir.glob("*")
    )

    monkeypatch.setattr(
        migration_module,
        "_install_new_file",
        original_installer,
    )
    retried = migration_module.migrate_artifact_directory(
        source_dir=source_dir,
        destination_dir=destination_dir,
    )
    assert retried["passed"] is True


def test_parser_migration_rejects_unrelated_destination_contents(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "pilot.csv"
    source_dir = tmp_path / "legacy"
    destination_dir = tmp_path / "strict"
    stale_contract = destination_dir / "campaign_execution_contract.json"
    _write_source(csv_path)
    _run_fixture(csv_path, source_dir, registry_name="qwen3-4b")
    _degrade_parser_artifacts(source_dir)
    destination_dir.mkdir()
    stale_contract.write_text('{"parser":"strict_v2"}\n', encoding="utf-8")
    before = stale_contract.read_bytes()

    with pytest.raises(FileExistsError, match="absent or empty"):
        migration_module.migrate_artifact_directory(
            source_dir=source_dir,
            destination_dir=destination_dir,
        )

    assert stale_contract.read_bytes() == before
    assert list(destination_dir.iterdir()) == [stale_contract]


def test_parser_migration_rejects_source_drift_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path = tmp_path / "pilot.csv"
    source_dir = tmp_path / "legacy"
    destination_dir = tmp_path / "strict"
    _write_source(csv_path)
    _run_fixture(csv_path, source_dir, registry_name="qwen3-4b")
    _degrade_parser_artifacts(source_dir)
    original_builder = migration_module._build_migration

    def _mutating_builder(
        source: Path,
        *,
        target_dir: Path,
    ) -> object:
        result = original_builder(source, target_dir=target_dir)
        raw_path = source / "silent_bias_stage_a_run_records.jsonl"
        raw_path.write_bytes(raw_path.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(
        migration_module,
        "_build_migration",
        _mutating_builder,
    )
    with pytest.raises(RuntimeError, match="changed during"):
        migration_module.migrate_artifact_directory(
            source_dir=source_dir,
            destination_dir=destination_dir,
        )
    assert not destination_dir.exists()


def test_parser_migration_rejects_nonderived_record_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path = tmp_path / "pilot.csv"
    source_dir = tmp_path / "legacy"
    destination_dir = tmp_path / "strict"
    _write_source(csv_path)
    _run_fixture(csv_path, source_dir, registry_name="qwen3-4b")
    _degrade_parser_artifacts(source_dir)
    original_migrator = migration_module.migrate_record_to_current_parser

    def _mutating_migrator(
        row: object,
        **kwargs: object,
    ) -> dict[str, object]:
        migrated = original_migrator(row, **kwargs)
        migrated["metadata"]["verbalized_prompt_hash"] = "tampered"
        return migrated

    monkeypatch.setattr(
        migration_module,
        "migrate_record_to_current_parser",
        _mutating_migrator,
    )
    with pytest.raises(AssertionError, match="protected provenance"):
        migration_module.migrate_artifact_directory(
            source_dir=source_dir,
            destination_dir=destination_dir,
        )
    assert not destination_dir.exists()


def test_parser_migration_in_place_transaction_restores_and_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path = tmp_path / "pilot.csv"
    source_dir = tmp_path / "legacy"
    _write_source(csv_path)
    _run_fixture(csv_path, source_dir, registry_name="qwen3-4b")
    _degrade_parser_artifacts(source_dir)
    before = _artifact_bytes(source_dir)
    original_replacer = migration_module._replace_file
    calls = 0

    def _fail_once(source: Path, target: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("simulated in-place commit failure")
        original_replacer(source, target)

    monkeypatch.setattr(migration_module, "_replace_file", _fail_once)
    with pytest.raises(OSError, match="simulated"):
        migration_module.migrate_artifact_directory(
            source_dir=source_dir,
            in_place=True,
            backup_suffix=".retry.bak",
        )
    assert _artifact_bytes(source_dir) == before
    assert not list(source_dir.glob("*.retry.bak"))

    monkeypatch.setattr(migration_module, "_replace_file", original_replacer)
    retried = migration_module.migrate_artifact_directory(
        source_dir=source_dir,
        in_place=True,
        backup_suffix=".retry.bak",
    )
    assert retried["passed"] is True


def test_parser_migration_preflights_planning_and_report_collisions(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "pilot.csv"
    source_dir = tmp_path / "legacy"
    destination_dir = tmp_path / "strict"
    _write_source(csv_path)
    _run_fixture(csv_path, source_dir, registry_name="qwen3-4b")
    _degrade_parser_artifacts(source_dir)
    destination_dir.mkdir()
    planning_collision = (
        destination_dir / "silent_bias_stage_a_planning_issues.json"
    )
    planning_collision.write_text("do not overwrite\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="destination_dir"):
        migration_module.migrate_artifact_directory(
            source_dir=source_dir,
            destination_dir=destination_dir,
        )
    assert planning_collision.read_text(encoding="utf-8") == "do not overwrite\n"
    assert len(list(destination_dir.iterdir())) == 1

    report_collision = (
        source_dir
        / "silent_bias_stage_a_run_records.jsonl.pre-strict-v3.bak"
    )
    with pytest.raises(ValueError, match="report_path"):
        migration_module.migrate_artifact_directory(
            source_dir=source_dir,
            in_place=True,
            report_path=report_collision,
        )
    assert not list(source_dir.glob("*.pre-strict-v3.bak"))


def test_parser_migration_report_failure_cannot_partially_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path = tmp_path / "pilot.csv"
    source_dir = tmp_path / "legacy"
    destination_dir = tmp_path / "strict"
    report_path = tmp_path / "migration-report.json"
    _write_source(csv_path)
    _run_fixture(csv_path, source_dir, registry_name="qwen3-4b")
    _degrade_parser_artifacts(source_dir)
    original_installer = migration_module._install_new_file

    def _fail_report(source: Path, target: Path) -> None:
        if target == report_path:
            raise OSError("simulated report commit failure")
        original_installer(source, target)

    monkeypatch.setattr(migration_module, "_install_new_file", _fail_report)
    with pytest.raises(OSError, match="report commit failure"):
        migration_module.migrate_artifact_directory(
            source_dir=source_dir,
            destination_dir=destination_dir,
            report_path=report_path,
        )

    assert not any(
        path.is_file() for path in destination_dir.glob("*")
    )
    assert not report_path.exists()
