from __future__ import annotations

import csv
import json
import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from biases.analysis.package_validation import (
    ANALYSIS_CSV_NAMES,
    ANALYSIS_MANIFEST_OUTPUT_NAMES,
    ANALYSIS_VERSION,
    ASSET_OUTPUT_NAMES,
    ASSET_VERSION,
    MIXED_EFFECTS_FORMULA,
    MIXED_MODEL_TERMS,
    UNCERTAINTY_GEE_FORMULA,
    AnalysisValidationConfig,
    AssetPackage,
    main,
    validate_analysis_package,
)
from biases.analysis.provenance import canonical_json, file_sha256, spec_sha256


MODEL = "Qwen/Qwen3-4B"
RESAMPLES = 20


@dataclass(frozen=True)
class CompletePackage:
    analysis_dir: Path
    stage_a: Path
    stage_b: Path
    assets: tuple[AssetPackage, AssetPackage]
    config: AnalysisValidationConfig


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{canonical_json(payload)}\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    assert rows
    fieldnames = sorted({field for row in rows for field in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _provenance_row(spec_hash: str, input_hashes: str) -> dict[str, object]:
    return {
        "input_hashes": input_hashes,
        "model_name": MODEL,
        "spec_hash": spec_hash,
    }


def _analysis_rows(
    *,
    spec_hash: str,
    input_hashes: str,
) -> dict[str, list[dict[str, object]]]:
    provenance = _provenance_row(spec_hash, input_hashes)
    doses = {
        "authority": (1.0, 2.0, 3.0, 4.0),
        "bandwagon": (55.0, 70.0, 85.0, 95.0),
    }
    paired = [
        {
            **provenance,
            "clean_entropy": 0.20,
            "clean_msp": 0.80,
            "clean_tie": False,
            "clean_verdict": "A",
            "cued_entropy": 0.25,
            "cued_msp": 0.75,
            "cued_record_id": (
                f"{family}-{direction}-{dose:g}-{ordering}"
            ),
            "cued_verdict": "A",
            "direction": direction,
            "dose": dose,
            "family": family,
            "flip": False,
            "js_divergence": 0.01,
            "ordering": ordering,
            "routing_split": "test",
            "signed_cue_mass": 0.05,
            "variant_id": (
                f"{family}_{direction}_{int(dose)}_{ordering}"
            ),
        }
        for family, family_doses in doses.items()
        for direction in ("congruent", "incongruent")
        for dose in family_doses
        for ordering in ("ab", "ba")
    ]
    silent = [
        {
            **provenance,
            "ci_high": 0.15,
            "ci_low": 0.05,
            "clean_tie": False,
            "direction": "incongruent",
            "dose": dose,
            "estimate": 0.10,
            "family": family,
            "metric": "signed_cue_mass",
            "n": 1,
            "n_questions": 1,
            "non_flipped_only": True,
            "p_value_holm": 0.04,
            "p_value_one_sided": 0.02,
            "primary": True,
            "routing_split": "test",
        }
        for family, family_doses in doses.items()
        for dose in family_doses
    ]
    susceptibility = [
        {
            **provenance,
            "auc_difference": 0.10,
            "auc_difference_ci_high": 0.20,
            "auc_difference_ci_low": 0.01,
            "baseline_channel": "entropy",
            "clean_tie": False,
            "clean_baseline_auc": 0.60,
            "clean_baseline_auc_ci_high": 0.70,
            "clean_baseline_auc_ci_low": 0.50,
            "direction": "incongruent",
            "family": family,
            "high_dose": max(family_doses),
            "low_dose": min(family_doses),
            "n": 1,
            "n_clusters": 1,
            "n_resamples": RESAMPLES,
            "positives": 1,
            "primary": True,
            "routing_split": "test",
            "shift_auc": 0.70,
            "shift_auc_ci_high": 0.80,
            "shift_auc_ci_low": 0.60,
            "shift_metric": "signed_cue_mass",
        }
        for family, family_doses in doses.items()
    ]
    conditions: list[dict[str, object]] = []
    for ordering in ("ab", "ba"):
        conditions.append(
            {
                "clean_tie": False,
                "direction": "control",
                "dose": None,
                "family": "clean",
                "ordering": ordering,
                "routing_split": "test",
                "variant_id": f"clean_{ordering}",
            }
        )
        conditions.extend(
            {
                "clean_tie": False,
                "direction": direction,
                "dose": dose,
                "family": family,
                "ordering": ordering,
                "routing_split": "test",
                "variant_id": (
                    f"{family}_{direction}_{int(dose)}_{ordering}"
                ),
            }
            for family, family_doses in doses.items()
            for direction in ("congruent", "incongruent")
            for dose in family_doses
        )
    channel_confidence = {
        "msp": 0.8,
        "consistency_agreement": 0.75,
        "verbalized_confidence": 0.7,
    }
    calibration: list[dict[str, object]] = []
    reliability: list[dict[str, object]] = []
    risk_coverage: list[dict[str, object]] = []
    for condition in conditions:
        for channel, confidence in channel_confidence.items():
            calibration.append(
                {
                    **provenance,
                    **condition,
                    "accuracy": 1.0,
                    "availability_rate": 1.0,
                    "brier": 0.06 if channel == "msp" else None,
                    "brier_n": 1 if channel == "msp" else 0,
                    "confidence_channel": channel,
                    "ece": 1.0 - confidence,
                    "missing_n": 0,
                    "n": 1,
                    "n_bins": 10,
                    "tie_policy": "strict_three_class",
                    "total_n": 1,
                }
            )
            occupied_bin = min(9, int(confidence * 10))
            reliability.extend(
                {
                    **provenance,
                    **condition,
                    "accuracy": 1.0 if index == occupied_bin else None,
                    "bin_index": index,
                    "confidence_channel": channel,
                    "lower": index / 10,
                    "mean_confidence": (
                        confidence if index == occupied_bin else None
                    ),
                    "n": 1 if index == occupied_bin else 0,
                    "tie_policy": "strict_three_class",
                    "upper": (index + 1) / 10,
                }
                for index in range(10)
            )
            risk_coverage.extend(
                (
                    {
                        **provenance,
                        **condition,
                        "accepted": 0,
                        "aurc": 0.0,
                        "confidence_channel": channel,
                        "coverage": 0.0,
                        "risk": 0.0,
                        "threshold": math.inf,
                        "total": 1,
                    },
                    {
                        **provenance,
                        **condition,
                        "accepted": 1,
                        "aurc": 0.0,
                        "confidence_channel": channel,
                        "coverage": 1.0,
                        "risk": 0.0,
                        "threshold": confidence,
                        "total": 1,
                    },
                )
            )
    mcnemar = [
        {
            **provenance,
            **{
                field: value
                for field, value in condition.items()
                if field != "variant_id"
            },
            "b_clean_correct_cued_wrong": 0,
            "c_clean_wrong_cued_correct": 0,
            "clean_correct": 1,
            "cued_correct": 1,
            "n": 1,
            "p_value": 1.0,
            "p_value_holm": 1.0,
            "primary": (
                condition["direction"] == "incongruent"
                and condition["clean_tie"] is False
                and condition["routing_split"] == "test"
            ),
            "statistic": 0,
            "test": "exact_two_sided_mcnemar_clean_vs_cued_correctness",
            "tie_policy": "strict_three_class",
        }
        for condition in conditions
        if condition["family"] != "clean"
    ]
    raw_threshold_p = 1.0 / (RESAMPLES + 1)
    adjusted_threshold_p = 4.0 / (RESAMPLES + 1)
    transfer = [
        {
            **provenance,
            "accepted_flip_fraction_ci_high": 1.0,
            "accepted_flip_fraction_ci_low": 0.9,
            "aggregation": "single_ordering",
            "calibration_coverage": 1.0,
            "calibration_n": 1,
            "calibration_risk": 0.0,
            "clean_tie": False,
            "confidence": 0.95,
            "confidence_channel": "msp",
            "decision_rule": "risk_inflation_vs_target_ci_low > 0",
            "direction": "incongruent",
            "dose": max(family_doses),
            "family": family,
            "n_calibration_clusters": 1,
            "n_resamples": RESAMPLES,
            "n_test_clusters": 1,
            "ordering": ordering,
            "p_value_holm": adjusted_threshold_p,
            "primary": True,
            "realized_risk_ci_high": 1.0,
            "realized_risk_ci_low": 0.9,
            "risk_inflation_vs_clean_calibration_ci_high": 1.0,
            "risk_inflation_vs_clean_calibration_ci_low": 0.9,
            "risk_inflation_vs_target_ci_high": 0.9,
            "risk_inflation_vs_target_ci_low": 0.8,
            "risk_inflation_vs_target_p_value_one_sided": raw_threshold_p,
            "routing_split": "test",
            "target_risk": 0.10,
            "test_accepted": 1,
            "test_accepted_flip_fraction": 1.0,
            "test_accepted_flips": 1,
            "test_coverage": 1.0,
            "test_flips": 1,
            "test_n": 1,
            "test_realized_risk": 1.0,
            "test_risk_inflation_vs_clean_calibration": 1.0,
            "test_risk_inflation_vs_target": 0.9,
            "threshold": 0.75,
            "variant_id": (
                f"{family}_incongruent_{int(max(family_doses))}_{ordering}"
            ),
        }
        for family, family_doses in doses.items()
        for ordering in ("ab", "ba")
    ]
    transfer.extend(
        {
            **provenance,
            "aggregation": "single_ordering",
            "calibration_coverage": 1.0,
            "calibration_n": 1,
            "calibration_risk": 0.0,
            "clean_tie": "all",
            "confidence": 0.95,
            "confidence_channel": "msp",
            "decision_rule": "",
            "direction": "clean",
            "dose": None,
            "family": "clean",
            "n_calibration_clusters": 1,
            "n_resamples": RESAMPLES,
            "n_test_clusters": 1,
            "ordering": ordering,
            "primary": False,
            "routing_split": "test",
            "target_risk": 0.10,
            "test_accepted": 1,
            "test_accepted_flip_fraction": 0.0,
            "test_accepted_flips": 0,
            "test_coverage": 1.0,
            "test_flips": 0,
            "test_n": 1,
            "test_realized_risk": 0.0,
            "test_risk_inflation_vs_clean_calibration": 0.0,
            "test_risk_inflation_vs_target": -0.1,
            "threshold": 0.75,
            "variant_id": "clean",
        }
        for ordering in ("ab", "ba")
    )
    dose_response = [
        {
            **provenance,
            "clean_tie": False,
            "converged": True,
            "direction": "incongruent",
            "events": 1,
            "family": family,
            "intercept": -2.0,
            "n": 4,
            "n_clusters": 1,
            "n_resamples": RESAMPLES,
            "p25_ci_high": max(family_doses),
            "p25_ci_low": min(family_doses),
            "p25_dose": family_doses[1],
            "p_value_holm": 0.04,
            "primary": True,
            "routing_split": "test",
            "slope": 0.50,
            "slope_ci_high": 0.80,
            "slope_ci_low": 0.20,
            "slope_p_value_one_sided": 0.02,
        }
        for family, family_doses in doses.items()
    ]
    trend = [
        {
            **provenance,
            "bootstrap_resamples_requested": RESAMPLES,
            "bootstrap_resamples_successful": RESAMPLES,
            "clean_tie": False,
            "converged": True,
            "direction": "incongruent",
            "estimator": "gaussian_gee_exchangeable",
            "family": family,
            "formula": UNCERTAINTY_GEE_FORMULA,
            "intercept": 0.20,
            "metric": "cued_entropy",
            "n": 4,
            "n_clusters": 1,
            "p_value_holm": 0.04,
            "primary": True,
            "routing_split": "test",
            "sensitivity_analysis": False,
            "slope": 0.10,
            "slope_ci_high": 0.15,
            "slope_ci_low": 0.05,
            "slope_p_value_one_sided": 0.02,
            "slope_standard_error": 0.02,
            "slope_z_value": 5.0,
            "stable_set": "pre_first_flip",
            "statistic": 0.10,
            "status": "ok",
        }
        for family in doses
    ]
    uncertainty_by_dose = [
        {
            **provenance,
            "ci_high": 0.30,
            "ci_low": 0.20,
            "clean_tie": False,
            "direction": "incongruent",
            "dose": dose,
            "estimate": 0.25,
            "family": family,
            "metric": "cued_entropy",
            "n": 1,
            "n_clusters": 1,
            "n_resamples": RESAMPLES,
            "normalized_dose": index / 3.0,
            "primary": True,
            "routing_split": "test",
            "stable_set": "pre_first_flip",
        }
        for family, family_doses in doses.items()
        for index, dose in enumerate(family_doses)
    ]
    modeling = [
        {
            **provenance,
            "clean_tie": False,
            "converged": True,
            "estimate": 0.10,
            "fit_method": "statsmodels.BinomialBayesMixedGLM.fit_vb",
            "formula": MIXED_EFFECTS_FORMULA,
            "model_type": "binomial_random_intercept",
            "n": 32,
            "p_value": 0.02,
            "p_value_holm": 0.04,
            "routing_split": "test",
            "standard_error": 0.02,
            "status": "ok",
            "term": term,
            "z_value": 5.0,
        }
        for term in sorted(MIXED_MODEL_TERMS)
    ]
    return {
        "paired_shifts.csv": paired,
        "rq1_silent_shift.csv": silent,
        "rq1_susceptibility.csv": susceptibility,
        "rq2_calibration.csv": calibration,
        "rq2_reliability.csv": reliability,
        "rq2_risk_coverage.csv": risk_coverage,
        "rq2_threshold_transfer.csv": transfer,
        "rq2_mcnemar.csv": mcnemar,
        "rq3_dose_response.csv": dose_response,
        "rq3_uncertainty_trend.csv": trend,
        "rq3_uncertainty_by_dose.csv": uncertainty_by_dose,
        "rq3_modeling.csv": modeling,
    }


def _refresh_analysis_manifest(package: CompletePackage) -> None:
    provenance_path = package.analysis_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "spec_hash": provenance["spec_hash"],
        "pairing": {
            "paired": package.config.expected_paired_records,
            "unmatched_cued": 0,
            "unused_clean": 0,
        },
        "outputs": {
            name: file_sha256(package.analysis_dir / name)
            for name in ANALYSIS_MANIFEST_OUTPUT_NAMES
        },
    }
    _write_json(package.analysis_dir / "analysis_manifest.json", manifest)


def _asset_manifest(package: CompletePackage, asset: AssetPackage) -> dict[str, object]:
    return {
        "asset_version": ASSET_VERSION,
        "deterministic": {
            "stable_input_and_row_sorting": True,
            "timestamps_embedded": False,
            "fixed_pdf_metadata": True,
        },
        "inputs": {
            name: {
                "available": True,
                "rows": len(_read_csv(package.analysis_dir / name)),
                "sha256": file_sha256(package.analysis_dir / name),
            }
            for name in ANALYSIS_CSV_NAMES
        },
        "missing_inputs": [],
        "outputs": {
            name: file_sha256(
                asset.report_path
                if name == "report/paper_results.md"
                else asset.directory / name
            )
            for name in ASSET_OUTPUT_NAMES
        },
    }


def _refresh_asset_manifests(package: CompletePackage) -> None:
    for asset in package.assets:
        _write_json(
            asset.directory / "paper_assets_manifest.json",
            _asset_manifest(package, asset),
        )


def _make_complete_package(tmp_path: Path) -> CompletePackage:
    analysis_dir = tmp_path / "analysis"
    stage_a = tmp_path / "stage-a.jsonl"
    stage_b = tmp_path / "stage-b.jsonl"
    stage_a.write_text('{"stage":"a"}\n', encoding="utf-8")
    stage_b.write_text('{"stage":"b"}\n', encoding="utf-8")
    stage_a_hashes = [file_sha256(stage_a)]
    stage_b_hashes = [file_sha256(stage_b)]
    spec = {
        "accepted_flip_policy": "matching_channel_clean_vs_cued_verdict",
        "analysis_version": ANALYSIS_VERSION,
        "bootstrap_resamples": RESAMPLES,
        "confidence_channels": [
            "msp",
            "consistency_agreement",
            "verbalized_confidence",
        ],
        "confidence_verdict_policy": {
            "msp": "deterministic_logit_verdict",
            "consistency_agreement": "consistency_majority_verdict",
            "verbalized_confidence": "verbalized_pass_verdict",
        },
        "ece_bins": 10,
        "formula": MIXED_EFFECTS_FORMULA,
        "missing_channel_verdict_policy": "exclude_without_fallback",
        "routing_split_policy": {
            "calibration": "threshold_selection_only",
            "headline_estimation": "test",
        },
        "mixed_model_population": {
            "routing_split": "test",
            "clean_tie": False,
        },
        "seed": 42,
        "target_risks": [0.10, 0.20],
        "tie_policy": "strict_three_class",
        "trend_permutations": 50,
    }
    spec_hash = spec_sha256(spec)
    provenance = {
        "analysis_version": ANALYSIS_VERSION,
        "spec": spec,
        "spec_hash": spec_hash,
        "stage_a_input_hashes": stage_a_hashes,
        "stage_b_input_hashes": stage_b_hashes,
    }
    input_hashes = canonical_json(
        {"stage_a": stage_a_hashes, "stage_b": stage_b_hashes}
    )
    for name, rows in _analysis_rows(
        spec_hash=spec_hash,
        input_hashes=input_hashes,
    ).items():
        _write_csv(analysis_dir / name, rows)
    _write_json(analysis_dir / "provenance.json", provenance)
    config = AnalysisValidationConfig(
        expected_models=(MODEL,),
        source_pairs=1,
        bootstrap_resamples=RESAMPLES,
        trend_permutations=50,
    )
    assets = (
        AssetPackage(tmp_path / "assets-one", tmp_path / "report-one.md"),
        AssetPackage(tmp_path / "assets-two", tmp_path / "report-two.md"),
    )
    package = CompletePackage(
        analysis_dir=analysis_dir,
        stage_a=stage_a,
        stage_b=stage_b,
        assets=assets,
        config=config,
    )
    _refresh_analysis_manifest(package)
    for asset in assets:
        for name in ASSET_OUTPUT_NAMES:
            path = (
                asset.report_path
                if name == "report/paper_results.md"
                else asset.directory / name
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"deterministic:{name}\n".encode())
    _refresh_asset_manifests(package)
    return package


def _validate(package: CompletePackage):
    return validate_analysis_package(
        analysis_dir=package.analysis_dir,
        stage_a_paths=(package.stage_a,),
        stage_b_paths=(package.stage_b,),
        asset_packages=package.assets,
        config=package.config,
    )


def _codes(package: CompletePackage) -> set[str]:
    return {issue.code for issue in _validate(package).errors}


def _replace_csv(
    package: CompletePackage,
    name: str,
    mutate: Callable[[list[dict[str, str]]], None],
) -> None:
    path = package.analysis_dir / name
    rows = _read_csv(path)
    mutate(rows)
    _write_csv(path, rows)
    _refresh_analysis_manifest(package)
    _refresh_asset_manifests(package)


def test_complete_package_passes(tmp_path: Path) -> None:
    package = _make_complete_package(tmp_path)

    report = _validate(package)

    assert report.passed
    assert report.error_count == 0
    assert report.expected_paired_records == 32


def test_fresh_producer_clean_control_direction_is_canonicalized(
    tmp_path: Path,
) -> None:
    package = _make_complete_package(tmp_path)
    calibration = _read_csv(package.analysis_dir / "rq2_calibration.csv")

    assert {
        row["direction"]
        for row in calibration
        if row["family"] == "clean"
    } == {"control"}
    assert _validate(package).passed


def test_primary_unavailable_warns_unless_strict(tmp_path: Path) -> None:
    package = _make_complete_package(tmp_path)

    def invalidate(rows: list[dict[str, str]]) -> None:
        rows[0]["status"] = "unavailable"
        rows[0]["slope"] = "nan"

    _replace_csv(package, "rq3_uncertainty_trend.csv", invalidate)

    report = _validate(package)
    assert report.passed
    assert report.integrity_passed
    assert not report.primary_available
    assert "primary_result_unavailable" in {
        issue.code for issue in report.availability_warnings
    }

    strict_package = replace(
        package,
        config=replace(package.config, require_primary_available=True),
    )
    assert not _validate(strict_package).passed


def test_missing_expected_model_fails(tmp_path: Path) -> None:
    package = _make_complete_package(tmp_path)
    config = AnalysisValidationConfig(
        expected_models=(MODEL, "allenai/Olmo-3-7B-Instruct"),
        source_pairs=1,
        bootstrap_resamples=RESAMPLES,
        trend_permutations=50,
    )
    changed = CompletePackage(
        analysis_dir=package.analysis_dir,
        stage_a=package.stage_a,
        stage_b=package.stage_b,
        assets=package.assets,
        config=config,
    )

    report = _validate(changed)

    assert not report.passed
    assert "csv_model_set_mismatch" in {issue.code for issue in report.errors}


def test_direct_stage_input_hash_drift_fails(tmp_path: Path) -> None:
    package = _make_complete_package(tmp_path)
    package.stage_a.write_text('{"stage":"changed"}\n', encoding="utf-8")

    assert "stage_input_hash_mismatch" in _codes(package)


def test_duplicate_paired_record_fails_even_with_refreshed_manifests(
    tmp_path: Path,
) -> None:
    package = _make_complete_package(tmp_path)

    def duplicate_record(rows: list[dict[str, str]]) -> None:
        rows[1]["cued_record_id"] = rows[0]["cued_record_id"]

    _replace_csv(package, "paired_shifts.csv", duplicate_record)

    assert "duplicate_cued_record_id" in _codes(package)


def test_incomplete_reproduction_assets_fail(tmp_path: Path) -> None:
    package = _make_complete_package(tmp_path)
    missing = package.assets[1].directory / "tables" / "rq1_susceptibility.tex"
    missing.unlink()

    assert "asset_output_missing" in _codes(package)


def test_mixed_effects_unavailable_is_an_availability_warning(
    tmp_path: Path,
) -> None:
    package = _make_complete_package(tmp_path)

    def make_unavailable(rows: list[dict[str, str]]) -> None:
        del rows[1:]
        rows[0]["status"] = "unavailable"
        rows[0]["estimate"] = ""

    _replace_csv(package, "rq3_modeling.csv", make_unavailable)

    report = _validate(package)
    assert report.passed
    assert report.integrity_passed
    assert not report.primary_available
    assert "primary_result_unavailable" in {
        issue.code for issue in report.availability_warnings
    }


def test_zero_accepted_and_flips_are_availability_warnings(
    tmp_path: Path,
) -> None:
    package = _make_complete_package(tmp_path)

    def clear_headline(rows: list[dict[str, str]]) -> None:
        row = rows[0]
        row["test_accepted"] = "0"
        row["test_accepted_flips"] = "0"
        row["test_coverage"] = "0"
        row["test_flips"] = "0"
        for field in (
            "test_realized_risk",
            "test_risk_inflation_vs_clean_calibration",
            "test_risk_inflation_vs_target",
            "test_accepted_flip_fraction",
            "accepted_flip_fraction_ci_low",
            "accepted_flip_fraction_ci_high",
        ):
            row[field] = ""

    _replace_csv(package, "rq2_threshold_transfer.csv", clear_headline)

    report = _validate(package)
    assert report.passed
    assert report.integrity_passed
    assert not report.primary_available
    assert report.availability_warning_count > 0

    strict_package = replace(
        package,
        config=replace(package.config, require_primary_available=True),
    )
    assert not _validate(strict_package).passed


def test_threshold_headline_equation_corruption_fails(tmp_path: Path) -> None:
    package = _make_complete_package(tmp_path)

    def corrupt(rows: list[dict[str, str]]) -> None:
        rows[0]["test_coverage"] = "0.5"

    _replace_csv(package, "rq2_threshold_transfer.csv", corrupt)

    assert "rq2_threshold_equation_mismatch" in _codes(package)


def test_threshold_headline_bound_corruption_fails(tmp_path: Path) -> None:
    package = _make_complete_package(tmp_path)

    def corrupt(rows: list[dict[str, str]]) -> None:
        rows[0]["risk_inflation_vs_target_ci_high"] = "0.95"

    _replace_csv(package, "rq2_threshold_transfer.csv", corrupt)

    assert "rq2_threshold_interval_invalid" in _codes(package)


def test_threshold_clean_control_direction_corruption_fails(
    tmp_path: Path,
) -> None:
    package = _make_complete_package(tmp_path)

    def corrupt(rows: list[dict[str, str]]) -> None:
        clean = next(row for row in rows if row["family"] == "clean")
        clean["direction"] = "incongruent"

    _replace_csv(package, "rq2_threshold_transfer.csv", corrupt)

    codes = _codes(package)
    assert "rq2_threshold_condition_invalid" in codes
    assert "rq2_threshold_clean_control_missing" in codes


def test_threshold_p_value_can_use_fewer_valid_bootstrap_draws(
    tmp_path: Path,
) -> None:
    package = _make_complete_package(tmp_path)

    def use_valid_draw_denominator(rows: list[dict[str, str]]) -> None:
        rows[0]["risk_inflation_vs_target_p_value_one_sided"] = str(1 / 11)

    _replace_csv(
        package,
        "rq2_threshold_transfer.csv",
        use_valid_draw_denominator,
    )

    assert _validate(package).passed


def test_threshold_inflation_interval_identity_corruption_fails(
    tmp_path: Path,
) -> None:
    package = _make_complete_package(tmp_path)

    def corrupt(rows: list[dict[str, str]]) -> None:
        rows[0]["risk_inflation_vs_target_ci_low"] = "0.81"

    _replace_csv(package, "rq2_threshold_transfer.csv", corrupt)

    assert "rq2_threshold_interval_identity_mismatch" in _codes(package)


def test_calibration_arithmetic_corruption_fails(tmp_path: Path) -> None:
    package = _make_complete_package(tmp_path)

    def corrupt(rows: list[dict[str, str]]) -> None:
        rows[0]["missing_n"] = "1"

    _replace_csv(package, "rq2_calibration.csv", corrupt)

    assert "rq2_calibration_equation_mismatch" in _codes(package)


def test_reliability_bin_corruption_fails(tmp_path: Path) -> None:
    package = _make_complete_package(tmp_path)

    def corrupt(rows: list[dict[str, str]]) -> None:
        rows[0]["upper"] = "0.2"

    _replace_csv(package, "rq2_reliability.csv", corrupt)

    assert "rq2_reliability_bin_invalid" in _codes(package)


def test_risk_coverage_arithmetic_corruption_fails(tmp_path: Path) -> None:
    package = _make_complete_package(tmp_path)

    def corrupt(rows: list[dict[str, str]]) -> None:
        rows[1]["coverage"] = "0.5"

    _replace_csv(package, "rq2_risk_coverage.csv", corrupt)

    assert "rq2_risk_coverage_equation_mismatch" in _codes(package)


def test_rq2_routing_split_stratum_omission_fails(tmp_path: Path) -> None:
    package = _make_complete_package(tmp_path)
    variant = "authority_congruent_1_ab"

    def move_stratum(rows: list[dict[str, str]]) -> None:
        for row in rows:
            if row["variant_id"] == variant:
                row["routing_split"] = "calibration"

    for name in (
        "rq2_calibration.csv",
        "rq2_reliability.csv",
        "rq2_risk_coverage.csv",
    ):
        _replace_csv(package, name, move_stratum)

    codes = _codes(package)
    assert "rq2_calibration_stratum_set_mismatch" in codes
    assert "rq2_reliability_stratum_set_mismatch" in codes
    assert "rq2_risk_coverage_stratum_set_mismatch" in codes


def test_rq2_clean_tie_stratum_omission_fails(tmp_path: Path) -> None:
    package = _make_complete_package(tmp_path)
    variant = "bandwagon_congruent_55_ba"

    def move_stratum(rows: list[dict[str, str]]) -> None:
        for row in rows:
            if row["variant_id"] == variant:
                row["clean_tie"] = "True"

    for name in (
        "rq2_calibration.csv",
        "rq2_reliability.csv",
        "rq2_risk_coverage.csv",
    ):
        _replace_csv(package, name, move_stratum)

    assert "rq2_calibration_stratum_set_mismatch" in _codes(package)


def test_mcnemar_arithmetic_corruption_fails(tmp_path: Path) -> None:
    package = _make_complete_package(tmp_path)

    def corrupt(rows: list[dict[str, str]]) -> None:
        rows[0]["clean_correct"] = "0"

    _replace_csv(package, "rq2_mcnemar.csv", corrupt)

    assert "rq2_mcnemar_equation_mismatch" in _codes(package)


def test_missing_primary_test_nontie_mcnemar_cell_fails(
    tmp_path: Path,
) -> None:
    package = _make_complete_package(tmp_path)

    def move_primary_cell(rows: list[dict[str, str]]) -> None:
        primary = next(row for row in rows if row["primary"] == "True")
        primary["routing_split"] = "calibration"
        primary["primary"] = "False"

    _replace_csv(package, "rq2_mcnemar.csv", move_primary_cell)

    assert "rq2_mcnemar_primary_cell_set_mismatch" in _codes(package)


def test_duplicate_mixed_effects_term_fails(tmp_path: Path) -> None:
    package = _make_complete_package(tmp_path)

    def duplicate(rows: list[dict[str, str]]) -> None:
        rows[1]["term"] = rows[0]["term"]

    _replace_csv(package, "rq3_modeling.csv", duplicate)

    assert "modeling_duplicate_term" in _codes(package)


def test_false_preregistered_primary_selector_fails(tmp_path: Path) -> None:
    package = _make_complete_package(tmp_path)

    def clear_primary(rows: list[dict[str, str]]) -> None:
        rows[0]["primary"] = "False"

    _replace_csv(package, "rq3_dose_response.csv", clear_primary)

    assert "primary_selector_mismatch" in _codes(package)


def test_missing_primary_column_fails_schema_validation(tmp_path: Path) -> None:
    package = _make_complete_package(tmp_path)

    def remove_primary(rows: list[dict[str, str]]) -> None:
        for row in rows:
            row.pop("primary")

    _replace_csv(package, "rq3_uncertainty_by_dose.csv", remove_primary)

    assert "csv_header_missing" in _codes(package)


def test_calibration_tagged_rq1_or_rq3_headline_row_fails(
    tmp_path: Path,
) -> None:
    package = _make_complete_package(tmp_path)

    def move_to_calibration(rows: list[dict[str, str]]) -> None:
        rows[0]["routing_split"] = "calibration"

    _replace_csv(
        package,
        "rq1_silent_shift.csv",
        move_to_calibration,
    )

    assert "headline_routing_split_mismatch" in _codes(package)


def test_missing_headline_routing_split_column_fails(tmp_path: Path) -> None:
    package = _make_complete_package(tmp_path)

    def remove_split(rows: list[dict[str, str]]) -> None:
        for row in rows:
            row.pop("routing_split")

    _replace_csv(package, "rq3_dose_response.csv", remove_split)

    assert "csv_header_missing" in _codes(package)


def test_stale_routing_split_policy_in_spec_fails(tmp_path: Path) -> None:
    package = _make_complete_package(tmp_path)
    path = package.analysis_dir / "provenance.json"
    provenance = json.loads(path.read_text(encoding="utf-8"))
    provenance["spec"]["routing_split_policy"][
        "headline_estimation"
    ] = "calibration_and_test"
    _write_json(path, provenance)
    _refresh_analysis_manifest(package)
    _refresh_asset_manifests(package)

    assert "analysis_spec_mismatch" in _codes(package)


def test_mixed_model_n_must_match_test_nontie_paired_population(
    tmp_path: Path,
) -> None:
    package = _make_complete_package(tmp_path)

    def corrupt_n(rows: list[dict[str, str]]) -> None:
        for row in rows:
            row["n"] = "31"

    _replace_csv(package, "rq3_modeling.csv", corrupt_n)

    assert "modeling_count_mismatch" in _codes(package)


def test_mixed_model_n_derives_from_mixed_split_and_tie_population(
    tmp_path: Path,
) -> None:
    package = _make_complete_package(tmp_path)

    def create_excluded_rows(rows: list[dict[str, str]]) -> None:
        rows[0]["routing_split"] = "calibration"
        rows[1]["clean_tie"] = "True"

    def use_expected_n(rows: list[dict[str, str]]) -> None:
        for row in rows:
            row["n"] = "30"

    _replace_csv(package, "paired_shifts.csv", create_excluded_rows)
    _replace_csv(package, "rq3_modeling.csv", use_expected_n)

    assert "modeling_count_mismatch" not in _codes(package)


def test_cli_accepts_registry_alias_and_writes_report(
    tmp_path: Path,
    capsys,
) -> None:
    package = _make_complete_package(tmp_path)
    report_path = tmp_path / "validation.json"

    exit_code = main(
        [
            "--analysis-dir",
            str(package.analysis_dir),
            "--stage-a",
            str(package.stage_a),
            "--stage-b",
            str(package.stage_b),
            "--asset-package",
            str(package.assets[0].directory),
            str(package.assets[0].report_path),
            "--asset-package",
            str(package.assets[1].directory),
            str(package.assets[1].report_path),
            "--expected-model",
            "qwen3-4b",
            "--source-pairs",
            "1",
            "--bootstrap-resamples",
            str(RESAMPLES),
            "--trend-permutations",
            "50",
            "--report-path",
            str(report_path),
        ]
    )

    stdout = json.loads(capsys.readouterr().out)
    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert stdout["passed"]
    assert persisted == stdout
