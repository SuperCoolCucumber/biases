from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "make_paper_assets.py"
)


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("make_paper_assets", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


assets = _load_module()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _fixture_rows() -> dict[str, list[dict[str, object]]]:
    provenance = {"spec_hash": "spec-123", "input_hashes": '["input-123"]'}
    return {
        "paired_shifts.csv": [
            {
                **provenance,
                "model_name": "Qwen_Model",
                "family": "authority",
                "direction": "incongruent",
                "dose": 4,
                "clean_tie": False,
                "flip": False,
                "signed_cue_mass": 0.25,
            },
            {
                **provenance,
                "model_name": "Qwen_Model",
                "family": "authority",
                "direction": "incongruent",
                "dose": 1,
                "clean_tie": False,
                "flip": False,
                "signed_cue_mass": 0.10,
            },
        ],
        "rq1_silent_shift.csv": [
            {
                **provenance,
                "model_name": "Qwen_Model",
                "family": "authority",
                "direction": "incongruent",
                "dose": 4,
                "clean_tie": False,
                "metric": "signed_cue_mass",
                "primary": True,
                "n": 20,
                "estimate": 0.25,
                "ci_low": 0.12,
                "ci_high": 0.36,
                "p_value_holm": 0.01,
            },
            {
                **provenance,
                "model_name": "Qwen_Model",
                "family": "authority",
                "direction": "incongruent",
                "dose": 1,
                "clean_tie": False,
                "metric": "signed_cue_mass",
                "primary": True,
                "n": 20,
                "estimate": 0.10,
                "ci_low": 0.02,
                "ci_high": 0.18,
                "p_value_holm": 0.03,
            },
        ],
        "rq1_susceptibility.csv": [
            {
                **provenance,
                "model_name": "Qwen_Model",
                "family": "authority",
                "primary": True,
                "shift_metric": "signed_cue_mass",
                "low_dose": 1,
                "high_dose": 4,
                "n": 20,
                "shift_auc": 0.75,
                "clean_baseline_auc": 0.60,
                "auc_difference": 0.15,
                "auc_difference_ci_low": 0.02,
                "auc_difference_ci_high": 0.28,
            }
        ],
        "rq2_reliability.csv": [
            {
                **provenance,
                "model_name": "Qwen_Model",
                "family": "clean",
                "direction": "clean",
                "dose": "",
                "ordering": "ab",
                "variant_id": "clean_ab",
                "routing_split": "test",
                "confidence_channel": "msp",
                "clean_tie": False,
                "bin_index": 7,
                "mean_confidence": 0.75,
                "accuracy": 0.70,
                "n": 10,
            },
            {
                **provenance,
                "model_name": "Qwen_Model",
                "family": "authority",
                "direction": "incongruent",
                "dose": 4,
                "ordering": "ab",
                "variant_id": "authority_incongruent_4_ab",
                "routing_split": "test",
                "confidence_channel": "msp",
                "clean_tie": False,
                "bin_index": 8,
                "mean_confidence": 0.85,
                "accuracy": 0.60,
                "n": 10,
            },
            {
                **provenance,
                "model_name": "Qwen_Model",
                "family": "authority",
                "direction": "incongruent",
                "dose": 4,
                "ordering": "ab",
                "variant_id": "authority_incongruent_4_ab",
                "routing_split": "test",
                "confidence_channel": "verbalized_confidence",
                "clean_tie": False,
                "bin_index": 8,
                "mean_confidence": 0.99,
                "accuracy": 0.01,
                "n": 999,
            },
            {
                **provenance,
                "model_name": "Qwen_Model",
                "family": "authority",
                "direction": "incongruent",
                "dose": 4,
                "ordering": "ab",
                "variant_id": "authority_incongruent_4_ab",
                "routing_split": "test",
                "confidence_channel": "msp",
                "clean_tie": True,
                "bin_index": 8,
                "mean_confidence": 0.01,
                "accuracy": 0.99,
                "n": 998,
            },
        ],
        "rq2_risk_coverage.csv": [
            {
                **provenance,
                "model_name": "Qwen_Model",
                "family": "clean",
                "direction": "clean",
                "dose": "",
                "ordering": "ab",
                "variant_id": "clean_ab",
                "routing_split": "test",
                "confidence_channel": "msp",
                "clean_tie": False,
                "coverage": 0.5,
                "risk": 0.10,
            },
            {
                **provenance,
                "model_name": "Qwen_Model",
                "family": "authority",
                "direction": "incongruent",
                "dose": 4,
                "ordering": "ab",
                "variant_id": "authority_incongruent_4_ab",
                "routing_split": "test",
                "confidence_channel": "msp",
                "clean_tie": False,
                "coverage": 0.5,
                "risk": 0.30,
            },
            {
                **provenance,
                "model_name": "Qwen_Model",
                "family": "authority",
                "direction": "incongruent",
                "dose": 4,
                "ordering": "ab",
                "variant_id": "authority_incongruent_4_ab",
                "routing_split": "calibration",
                "confidence_channel": "msp",
                "clean_tie": False,
                "coverage": 0.5,
                "risk": 0.99,
            },
        ],
        "rq2_threshold_transfer.csv": [
            {
                **provenance,
                "model_name": "Qwen_Model",
                "family": "authority",
                "direction": "incongruent",
                "dose": 4,
                "ordering": "ab",
                "aggregation": "single_ordering",
                "target_risk": 0.10,
                "confidence_channel": "msp",
                "clean_tie": False,
                "routing_split": "test",
                "primary": True,
                "test_coverage": 0.55,
                "test_realized_risk": 0.30,
                "test_risk_inflation_vs_target": 0.20,
                "risk_inflation_vs_target_ci_low": 0.08,
                "risk_inflation_vs_target_ci_high": 0.32,
                "test_accepted_flip_fraction": 0.80,
                "accepted_flip_fraction_ci_low": 0.66,
                "accepted_flip_fraction_ci_high": 0.91,
                "p_value_holm": 0.02,
            },
            {
                **provenance,
                "model_name": "Qwen_Model",
                "family": "authority",
                "direction": "incongruent",
                "dose": 4,
                "ordering": "ab",
                "aggregation": "single_ordering",
                "target_risk": 0.10,
                "confidence_channel": "verbalized_confidence",
                "clean_tie": False,
                "routing_split": "test",
                "primary": True,
                "test_coverage": 0.99,
                "test_realized_risk": 0.99,
                "test_risk_inflation_vs_target": 0.89,
                "risk_inflation_vs_target_ci_low": 0.80,
                "risk_inflation_vs_target_ci_high": 0.95,
                "test_accepted_flip_fraction": 0.99,
                "accepted_flip_fraction_ci_low": 0.95,
                "accepted_flip_fraction_ci_high": 1.00,
                "p_value_holm": 0.001,
            },
        ],
        "rq3_dose_response.csv": [
            {
                **provenance,
                "model_name": "Qwen_Model",
                "family": "authority",
                "direction": "incongruent",
                "clean_tie": False,
                "primary": True,
                "n": 80,
                "events": 20,
                "intercept": -3.0,
                "slope": 0.9,
                "slope_ci_low": 0.2,
                "slope_ci_high": 1.5,
                "p25_dose": 2.1,
                "p25_ci_low": 1.5,
                "p25_ci_high": 2.8,
                "dose_min": 1.0,
                "dose_max": 4.0,
                "p25_range_status": "within_tested_range",
                "p_value_holm": 0.01,
            }
        ],
        "rq3_uncertainty_trend.csv": [
            {
                **provenance,
                "model_name": "Qwen_Model",
                "family": "authority",
                "direction": "incongruent",
                "clean_tie": False,
                "metric": "cued_entropy",
                "stable_set": "pre_first_flip",
                "primary": True,
                "statistic": 0.08,
                "slope_ci_low": 0.02,
                "slope_ci_high": 0.14,
                "p_value_holm": 0.02,
                "n_clusters": 20,
            }
        ],
        "rq3_uncertainty_by_dose.csv": [
            {
                **provenance,
                "model_name": "Qwen_Model",
                "family": "authority",
                "direction": "incongruent",
                "clean_tie": False,
                "dose": dose,
                "normalized_dose": (dose - 1.0) / 3.0,
                "metric": "cued_entropy",
                "stable_set": "pre_first_flip",
                "estimate": 0.20 + dose * 0.02,
                "ci_low": 0.18 + dose * 0.02,
                "ci_high": 0.22 + dose * 0.02,
                "n": 20,
                "n_clusters": 10,
                "primary": True,
            }
            for dose in (1.0, 2.0, 3.0, 4.0)
        ],
    }


def _write_fixture(directory: Path) -> None:
    directory.mkdir(parents=True)
    for name, rows in _fixture_rows().items():
        _write_csv(directory / name, rows)


def test_table_and_digest_generation_is_byte_identical(tmp_path: Path) -> None:
    analysis_dir = tmp_path / "analysis"
    _write_fixture(analysis_dir)
    output_one = tmp_path / "assets-one"
    output_two = tmp_path / "assets-two"
    report_one = tmp_path / "report-one" / "paper_results.md"
    report_two = tmp_path / "report-two" / "paper_results.md"

    first = assets.generate_paper_assets(
        analysis_dir=analysis_dir,
        output_dir=output_one,
        report_path=report_one,
        include_figures=False,
    )
    second = assets.generate_paper_assets(
        analysis_dir=analysis_dir,
        output_dir=output_two,
        report_path=report_two,
        include_figures=False,
    )

    assert first == second
    assert report_one.read_bytes() == report_two.read_bytes()
    for first_path in sorted(path for path in output_one.rglob("*") if path.is_file()):
        relative = first_path.relative_to(output_one)
        assert first_path.read_bytes() == (output_two / relative).read_bytes()

    table = (output_one / "tables" / "rq1_silent_shift.tex").read_text()
    digest = report_one.read_text()
    manifest_text = (output_one / "paper_assets_manifest.json").read_text()
    assert "\\toprule" in table
    assert "Qwen\\_Model" in table
    assert table.index("1 & 20") < table.index("4 & 20")
    assert "## RQ1 — Silent bias" in digest
    assert "## Evidence scope" in digest
    assert "clean risk guarantee fails" in digest
    assert "[0.660, 0.910]" in digest
    assert "incomplete: adjusted p unavailable" not in digest
    threshold_table = (
        output_one / "tables" / "rq2_threshold_transfer.tex"
    ).read_text()
    assert "[0.080, 0.320]" in threshold_table
    assert "[0.660, 0.910]" in threshold_table
    assert "0.890" not in threshold_table
    assert "generated_at" not in manifest_text
    assert str(tmp_path) not in manifest_text


def test_rq2_headline_filters_do_not_pool_channels_splits_or_ties() -> None:
    rows = _fixture_rows()
    threshold = assets._headline_threshold_rows(rows["rq2_threshold_transfer.csv"])
    assert len(threshold) == 1
    assert threshold[0]["confidence_channel"] == "msp"

    reliability = assets._headline_rq2_figure_rows(rows["rq2_reliability.csv"])
    risk = assets._headline_rq2_figure_rows(rows["rq2_risk_coverage.csv"])
    assert {row["confidence_channel"] for row in reliability} == {"msp"}
    assert {row["routing_split"] for row in risk} == {"test"}
    assert all(row["clean_tie"] is False for row in [*reliability, *risk])


def test_digest_never_claims_failure_when_transferred_threshold_has_zero_coverage() -> None:
    decision, interval = assets._rq2_decision(
        {
            "test_coverage": 0.0,
            "test_risk_inflation_vs_target": "",
            "risk_inflation_vs_target_ci_low": 0.2,
            "risk_inflation_vs_target_ci_high": 0.9,
            "p_value_holm": 0.01,
        }
    )
    assert decision == "unavailable: zero test coverage"
    assert interval == "--"
    assert (
        assets._rq2_adjusted_p(
            {
                "test_coverage": 0.0,
                "test_risk_inflation_vs_target": "",
                "p_value_holm": 0.01,
            }
        )
        == "--"
    )


def test_evidence_scope_counts_source_pairs_from_ordered_pair_keys() -> None:
    rows = [
        {
            "model_name": "model-a",
            "pair_key": pair_key,
            "ordering": ordering,
        }
        for pair_key, ordering in (
            ("pair-1-ab", "ab"),
            ("pair-1-ba", "ba"),
            ("pair-2-ab", "ab"),
            ("pair-2-ba", "ba"),
        )
        for _dose in range(2)
    ]
    assert assets._evidence_unit_summary(rows) == (
        "- Ordered clean pairs: 4 per model across 2 orderings "
        "(2 source pairs per model)."
    )


def test_out_of_range_p25_is_labeled_as_extrapolation() -> None:
    estimate, interval, decision = assets._format_p25(
        {
            "p25_dose": -5.9,
            "p25_ci_low": -8.9,
            "p25_ci_high": -3.9,
            "dose_min": 1.0,
            "dose_max": 4.0,
            "p25_range_status": "below_tested_range",
        }
    )
    assert estimate == "below 1.000"
    assert interval == "extrapolated [-8.900, -3.900]"
    assert decision == "below tested range"


def test_figure_generation_is_byte_identical(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    analysis_dir = tmp_path / "analysis"
    _write_fixture(analysis_dir)
    output_one = tmp_path / "assets-one"
    output_two = tmp_path / "assets-two"

    assets.generate_paper_assets(
        analysis_dir=analysis_dir,
        output_dir=output_one,
        report_path=tmp_path / "report-one.md",
    )
    assets.generate_paper_assets(
        analysis_dir=analysis_dir,
        output_dir=output_two,
        report_path=tmp_path / "report-two.md",
    )

    figure_one = output_one / "figures"
    figure_two = output_two / "figures"
    pdfs = sorted(figure_one.glob("*.pdf"))
    assert len(pdfs) == 5
    for path in pdfs:
        other = figure_two / path.name
        assert path.read_bytes().startswith(b"%PDF")
        assert path.read_bytes() == other.read_bytes()


def test_missing_inputs_fail_by_default(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Missing required analysis CSVs"):
        assets.generate_paper_assets(
            analysis_dir=tmp_path / "empty",
            output_dir=tmp_path / "assets",
            report_path=tmp_path / "paper_results.md",
            include_figures=False,
        )


def test_allow_missing_emits_explicit_unavailable_evidence(tmp_path: Path) -> None:
    output_dir = tmp_path / "assets"
    report_path = tmp_path / "paper_results.md"

    manifest = assets.generate_paper_assets(
        analysis_dir=tmp_path / "empty",
        output_dir=output_dir,
        report_path=report_path,
        include_figures=False,
        allow_missing=True,
    )

    assert sorted(manifest["missing_inputs"]) == sorted(assets.REQUIRED_ANALYSIS_FILES)
    assert "No eligible evidence rows." in report_path.read_text()
    persisted = json.loads(
        (output_dir / "paper_assets_manifest.json").read_text(encoding="utf-8")
    )
    assert persisted == manifest
