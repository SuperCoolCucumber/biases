from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from biases.analysis.provenance import canonical_json, file_sha256, spec_sha256
from biases.analysis.statistics import holm_adjust
from biases.models import get_model_profile
from biases.stats import mcnemar_exact


ANALYSIS_VERSION = "silent-bias-p4-v4"
ASSET_VERSION = "silent-bias-paper-assets-v2"
MIXED_EFFECTS_FORMULA = (
    "flip ~ dose * family * congruence + (1 | question)"
)
UNCERTAINTY_GEE_FORMULA = "uncertainty ~ normalized_dose"
CUED_CONDITIONS_PER_PAIR_MODEL = 32
CONFIDENCE_CHANNELS = frozenset(
    {"msp", "consistency_agreement", "verbalized_confidence"}
)

ANALYSIS_CSV_NAMES = (
    "paired_shifts.csv",
    "rq1_silent_shift.csv",
    "rq1_susceptibility.csv",
    "rq2_calibration.csv",
    "rq2_reliability.csv",
    "rq2_risk_coverage.csv",
    "rq2_threshold_transfer.csv",
    "rq2_mcnemar.csv",
    "rq3_dose_response.csv",
    "rq3_uncertainty_trend.csv",
    "rq3_uncertainty_by_dose.csv",
    "rq3_modeling.csv",
)
ANALYSIS_MANIFEST_OUTPUT_NAMES = (*ANALYSIS_CSV_NAMES, "provenance.json")
ASSET_OUTPUT_NAMES = (
    "figures/rq1_silent_shift_distribution.pdf",
    "figures/rq2_reliability_diagrams.pdf",
    "figures/rq2_risk_coverage.pdf",
    "figures/rq3_dose_response.pdf",
    "figures/rq3_uncertainty_dose_response.pdf",
    "report/paper_results.md",
    "tables/rq1_silent_shift.tex",
    "tables/rq1_susceptibility.tex",
    "tables/rq2_threshold_transfer.tex",
    "tables/rq3_dose_response.tex",
    "tables/rq3_uncertainty_by_dose.tex",
    "tables/rq3_uncertainty_trend.tex",
)
SOCIAL_DOSES: Mapping[str, tuple[float, ...]] = {
    "authority": (1.0, 2.0, 3.0, 4.0),
    "bandwagon": (55.0, 70.0, 85.0, 95.0),
}
PRIMARY_CSV_NAMES = frozenset(
    {
        "rq1_silent_shift.csv",
        "rq1_susceptibility.csv",
        "rq2_threshold_transfer.csv",
        "rq3_dose_response.csv",
        "rq3_uncertainty_trend.csv",
        "rq3_uncertainty_by_dose.csv",
    }
)
PRIMARY_COLUMN_CSV_NAMES = PRIMARY_CSV_NAMES | {"rq2_mcnemar.csv"}
SELECTED_ROW_CSV_NAMES = PRIMARY_COLUMN_CSV_NAMES | {"rq3_modeling.csv"}
COMMON_CONDITION_COLUMNS = frozenset(
    {
        "clean_tie",
        "direction",
        "dose",
        "family",
        "model_name",
        "ordering",
        "routing_split",
        "variant_id",
    }
)
CSV_REQUIRED_COLUMNS: Mapping[str, frozenset[str]] = {
    "paired_shifts.csv": frozenset(
        {
            "clean_entropy",
            "clean_msp",
            "clean_tie",
            "clean_verdict",
            "cued_entropy",
            "cued_msp",
            "cued_record_id",
            "cued_verdict",
            "direction",
            "dose",
            "family",
            "flip",
            "js_divergence",
            "model_name",
            "ordering",
            "routing_split",
            "signed_cue_mass",
            "variant_id",
        }
    ),
    "rq1_silent_shift.csv": frozenset(
        {
            "ci_high",
            "ci_low",
            "clean_tie",
            "direction",
            "dose",
            "estimate",
            "family",
            "metric",
            "n",
            "n_questions",
            "non_flipped_only",
            "p_value_one_sided",
            "primary",
        }
    ),
    "rq1_susceptibility.csv": frozenset(
        {
            "auc_difference",
            "auc_difference_ci_high",
            "auc_difference_ci_low",
            "baseline_channel",
            "clean_baseline_auc",
            "clean_baseline_auc_ci_high",
            "clean_baseline_auc_ci_low",
            "direction",
            "family",
            "high_dose",
            "low_dose",
            "n",
            "n_clusters",
            "n_resamples",
            "positives",
            "primary",
            "shift_auc",
            "shift_auc_ci_high",
            "shift_auc_ci_low",
            "shift_metric",
        }
    ),
    "rq2_calibration.csv": COMMON_CONDITION_COLUMNS
    | frozenset(
        {
            "accuracy",
            "availability_rate",
            "brier",
            "brier_n",
            "confidence_channel",
            "ece",
            "missing_n",
            "n",
            "n_bins",
            "tie_policy",
            "total_n",
        }
    ),
    "rq2_reliability.csv": COMMON_CONDITION_COLUMNS
    | frozenset(
        {
            "accuracy",
            "bin_index",
            "confidence_channel",
            "lower",
            "mean_confidence",
            "n",
            "tie_policy",
            "upper",
        }
    ),
    "rq2_risk_coverage.csv": COMMON_CONDITION_COLUMNS
    | frozenset(
        {
            "accepted",
            "aurc",
            "confidence_channel",
            "coverage",
            "risk",
            "threshold",
            "total",
        }
    ),
    "rq2_threshold_transfer.csv": frozenset(
        {
            "accepted_flip_fraction_ci_high",
            "accepted_flip_fraction_ci_low",
            "aggregation",
            "calibration_coverage",
            "calibration_n",
            "calibration_risk",
            "clean_tie",
            "confidence",
            "confidence_channel",
            "decision_rule",
            "direction",
            "dose",
            "family",
            "model_name",
            "n_calibration_clusters",
            "n_resamples",
            "n_test_clusters",
            "ordering",
            "primary",
            "realized_risk_ci_high",
            "realized_risk_ci_low",
            "risk_inflation_vs_clean_calibration_ci_high",
            "risk_inflation_vs_clean_calibration_ci_low",
            "risk_inflation_vs_target_ci_high",
            "risk_inflation_vs_target_ci_low",
            "risk_inflation_vs_target_p_value_one_sided",
            "routing_split",
            "target_risk",
            "test_accepted",
            "test_accepted_flip_fraction",
            "test_accepted_flips",
            "test_coverage",
            "test_flips",
            "test_n",
            "test_realized_risk",
            "test_risk_inflation_vs_clean_calibration",
            "test_risk_inflation_vs_target",
            "threshold",
            "variant_id",
        }
    ),
    "rq2_mcnemar.csv": frozenset(
        {
            "b_clean_correct_cued_wrong",
            "c_clean_wrong_cued_correct",
            "clean_correct",
            "clean_tie",
            "cued_correct",
            "direction",
            "dose",
            "family",
            "model_name",
            "n",
            "ordering",
            "p_value",
            "p_value_holm",
            "primary",
            "routing_split",
            "statistic",
            "test",
            "tie_policy",
        }
    ),
    "rq3_dose_response.csv": frozenset(
        {
            "clean_tie",
            "converged",
            "direction",
            "events",
            "family",
            "intercept",
            "n",
            "n_clusters",
            "n_resamples",
            "p25_ci_high",
            "p25_ci_low",
            "p25_dose",
            "primary",
            "slope",
            "slope_ci_high",
            "slope_ci_low",
            "slope_p_value_one_sided",
        }
    ),
    "rq3_uncertainty_trend.csv": frozenset(
        {
            "bootstrap_resamples_requested",
            "bootstrap_resamples_successful",
            "clean_tie",
            "converged",
            "direction",
            "estimator",
            "family",
            "formula",
            "metric",
            "n",
            "n_clusters",
            "primary",
            "sensitivity_analysis",
            "slope",
            "slope_ci_high",
            "slope_ci_low",
            "slope_p_value_one_sided",
            "stable_set",
            "statistic",
            "status",
        }
    ),
    "rq3_uncertainty_by_dose.csv": frozenset(
        {
            "ci_high",
            "ci_low",
            "clean_tie",
            "direction",
            "dose",
            "estimate",
            "family",
            "metric",
            "n",
            "n_clusters",
            "n_resamples",
            "normalized_dose",
            "primary",
            "stable_set",
        }
    ),
    "rq3_modeling.csv": frozenset({"formula", "model_name", "status"}),
}
MIXED_MODEL_TERMS = frozenset(
    {
        "Intercept",
        "family[T.bandwagon]",
        "congruence[T.incongruent]",
        "family[T.bandwagon]:congruence[T.incongruent]",
        "dose",
        "dose:family[T.bandwagon]",
        "dose:congruence[T.incongruent]",
        "dose:family[T.bandwagon]:congruence[T.incongruent]",
    }
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class AssetPackage:
    directory: Path
    report_path: Path


@dataclass(frozen=True, slots=True)
class AnalysisValidationConfig:
    expected_models: tuple[str, ...]
    source_pairs: int
    bootstrap_resamples: int = 2_000
    trend_permutations: int = 10_000
    seed: int = 42
    ece_bins: int = 10
    target_risks: tuple[float, ...] = (0.10, 0.20)
    required_asset_copies: int = 2
    require_primary_available: bool = False

    def __post_init__(self) -> None:
        if not self.expected_models:
            raise ValueError("expected_models must not be empty")
        if len(set(self.expected_models)) != len(self.expected_models):
            raise ValueError("expected_models must be unique")
        if self.source_pairs < 1:
            raise ValueError("source_pairs must be positive")
        if self.bootstrap_resamples < 1:
            raise ValueError("bootstrap_resamples must be positive")
        if self.trend_permutations < 1:
            raise ValueError("trend_permutations must be positive")
        if self.ece_bins < 1:
            raise ValueError("ece_bins must be positive")
        if not self.target_risks or any(
            not math.isfinite(value) or not 0.0 < value < 1.0
            for value in self.target_risks
        ):
            raise ValueError("target_risks must be finite values in (0, 1)")
        if self.required_asset_copies < 1:
            raise ValueError("required_asset_copies must be positive")

    @property
    def expected_paired_records(self) -> int:
        return (
            self.source_pairs
            * CUED_CONDITIONS_PER_PAIR_MODEL
            * len(self.expected_models)
        )


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    code: str
    message: str
    path: str | None = None
    row: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
        }
        if self.path is not None:
            payload["path"] = self.path
        if self.row is not None:
            payload["row"] = self.row
        return payload


@dataclass(frozen=True, slots=True)
class CsvRow:
    line_number: int
    values: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class CsvInspection:
    name: str
    path: Path
    sha256: str | None
    fieldnames: frozenset[str]
    row_count: int
    models: frozenset[str]
    model_counts: Mapping[str, int]
    selected_rows: tuple[CsvRow, ...]


@dataclass(frozen=True, slots=True)
class AnalysisValidationReport:
    passed: bool
    integrity_passed: bool
    primary_available: bool
    require_primary_available: bool
    analysis_version: str
    expected_models: tuple[str, ...]
    source_pairs: int
    expected_paired_records: int
    csv_row_counts: Mapping[str, int]
    asset_packages_checked: int
    error_count: int
    errors: tuple[ValidationIssue, ...]
    errors_truncated: bool
    availability_warning_count: int
    availability_warnings: tuple[ValidationIssue, ...]
    availability_warnings_truncated: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "integrity_passed": self.integrity_passed,
            "primary_available": self.primary_available,
            "require_primary_available": self.require_primary_available,
            "analysis_version": self.analysis_version,
            "expected_models": list(self.expected_models),
            "source_pairs": self.source_pairs,
            "expected_paired_records": self.expected_paired_records,
            "csv_row_counts": dict(sorted(self.csv_row_counts.items())),
            "asset_packages_checked": self.asset_packages_checked,
            "error_count": self.error_count,
            "errors": [issue.to_dict() for issue in self.errors],
            "errors_truncated": self.errors_truncated,
            "availability_warning_count": self.availability_warning_count,
            "availability_warnings": [
                issue.to_dict() for issue in self.availability_warnings
            ],
            "availability_warnings_truncated": (
                self.availability_warnings_truncated
            ),
        }


class _IssueCollector:
    def __init__(self, *, max_reported: int = 100) -> None:
        self._max_reported = max_reported
        self._count = 0
        self._issues: list[ValidationIssue] = []

    @property
    def count(self) -> int:
        return self._count

    @property
    def issues(self) -> tuple[ValidationIssue, ...]:
        return tuple(self._issues)

    @property
    def truncated(self) -> bool:
        return self._count > len(self._issues)

    def add(
        self,
        code: str,
        message: str,
        *,
        path: Path | str | None = None,
        row: int | None = None,
    ) -> None:
        self._count += 1
        if len(self._issues) >= self._max_reported:
            return
        self._issues.append(
            ValidationIssue(
                code=code,
                message=message,
                path=str(path) if path is not None else None,
                row=row,
            )
        )


def _read_json_object(
    path: Path,
    *,
    collector: _IssueCollector,
) -> dict[str, Any] | None:
    if not path.is_file():
        collector.add("missing_file", "required JSON file is missing", path=path)
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        collector.add("invalid_json", str(exc), path=path)
        return None
    if not isinstance(payload, dict):
        collector.add("invalid_json", "top-level JSON value must be an object", path=path)
        return None
    return payload


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        numeric = int(str(value))
    except (TypeError, ValueError):
        return None
    return numeric


def _as_finite_float(value: Any) -> float | None:
    try:
        numeric = float(str(value))
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().casefold()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    return None


def _is_missing(value: Any) -> bool:
    return value is None or not str(value).strip()


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-12)


def _required_int(
    row: Mapping[str, str],
    field_name: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
    code: str,
    path: Path,
    line_number: int,
    collector: _IssueCollector,
) -> int | None:
    value = _as_int(row.get(field_name))
    if (
        value is None
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        bounds = (
            f"[{minimum}, {maximum}]"
            if maximum is not None
            else f">= {minimum}"
        )
        collector.add(
            code,
            f"{field_name} must be an integer {bounds}",
            path=path,
            row=line_number,
        )
        return None
    return value


def _optional_float(
    row: Mapping[str, str],
    field_name: str,
    *,
    lower: float | None = None,
    upper: float | None = None,
    code: str,
    path: Path,
    line_number: int,
    collector: _IssueCollector,
) -> float | None:
    raw = row.get(field_name)
    if _is_missing(raw):
        return None
    value = _as_finite_float(raw)
    if (
        value is None
        or (lower is not None and value < lower)
        or (upper is not None and value > upper)
    ):
        collector.add(
            code,
            f"{field_name} is non-finite or outside its allowed bounds",
            path=path,
            row=line_number,
        )
        return None
    return value


def _required_float(
    row: Mapping[str, str],
    field_name: str,
    *,
    lower: float | None = None,
    upper: float | None = None,
    code: str,
    path: Path,
    line_number: int,
    collector: _IssueCollector,
) -> float | None:
    value = _optional_float(
        row,
        field_name,
        lower=lower,
        upper=upper,
        code=code,
        path=path,
        line_number=line_number,
        collector=collector,
    )
    if value is None and _is_missing(row.get(field_name)):
        collector.add(
            code,
            f"{field_name} is required",
            path=path,
            row=line_number,
        )
    return value


def _check_equation(
    observed: float | None,
    expected: float,
    *,
    field_name: str,
    code: str,
    path: Path,
    line_number: int | None,
    collector: _IssueCollector,
) -> None:
    if observed is not None and not _close(observed, expected):
        collector.add(
            code,
            f"{field_name}={observed} does not equal {expected}",
            path=path,
            row=line_number,
        )


def _hash_paths(
    paths: Sequence[Path],
    *,
    label: str,
    expected_count: int,
    collector: _IssueCollector,
) -> tuple[str, ...]:
    if len(paths) != expected_count:
        collector.add(
            "stage_input_count_mismatch",
            f"{label} has {len(paths)} files; expected {expected_count}",
        )
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        collector.add(
            "duplicate_stage_input",
            f"{label} contains a repeated input path",
        )
    hashes: list[str] = []
    for path in paths:
        if not path.is_file():
            collector.add("missing_stage_input", f"{label} input is missing", path=path)
            continue
        try:
            hashes.append(file_sha256(path))
        except OSError as exc:
            collector.add("unreadable_stage_input", str(exc), path=path)
    if len(set(hashes)) != len(hashes):
        collector.add(
            "duplicate_stage_input_hash",
            f"{label} inputs do not have unique content hashes",
        )
    return tuple(sorted(hashes))


def _validate_spec(
    provenance: Mapping[str, Any] | None,
    *,
    config: AnalysisValidationConfig,
    stage_a_hashes: tuple[str, ...],
    stage_b_hashes: tuple[str, ...],
    collector: _IssueCollector,
    path: Path,
) -> tuple[str | None, str]:
    expected_input_hashes = canonical_json(
        {"stage_a": list(stage_a_hashes), "stage_b": list(stage_b_hashes)}
    )
    if provenance is None:
        return None, expected_input_hashes
    spec = provenance.get("spec")
    if not isinstance(spec, Mapping):
        collector.add("invalid_provenance", "provenance.spec must be an object", path=path)
        return None, expected_input_hashes
    expected_spec_values: Mapping[str, Any] = {
        "analysis_version": ANALYSIS_VERSION,
        "bootstrap_resamples": config.bootstrap_resamples,
        "trend_permutations": config.trend_permutations,
        "seed": config.seed,
        "ece_bins": config.ece_bins,
        "target_risks": list(config.target_risks),
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
        "missing_channel_verdict_policy": "exclude_without_fallback",
        "accepted_flip_policy": "matching_channel_clean_vs_cued_verdict",
        "tie_policy": "strict_three_class",
        "formula": MIXED_EFFECTS_FORMULA,
    }
    for field, expected in expected_spec_values.items():
        if spec.get(field) != expected:
            collector.add(
                "analysis_spec_mismatch",
                f"spec.{field} is {spec.get(field)!r}; expected {expected!r}",
                path=path,
            )
    computed_spec_hash = spec_sha256(spec)
    if provenance.get("spec_hash") != computed_spec_hash:
        collector.add(
            "spec_hash_mismatch",
            "provenance spec_hash does not match canonical spec bytes",
            path=path,
        )
    if provenance.get("analysis_version") != ANALYSIS_VERSION:
        collector.add(
            "analysis_version_mismatch",
            f"provenance declares {provenance.get('analysis_version')!r}",
            path=path,
        )
    for field, observed, expected in (
        ("stage_a_input_hashes", provenance.get("stage_a_input_hashes"), list(stage_a_hashes)),
        ("stage_b_input_hashes", provenance.get("stage_b_input_hashes"), list(stage_b_hashes)),
    ):
        if observed != expected:
            collector.add(
                "stage_input_hash_mismatch",
                f"{field} does not match the supplied direct inputs",
                path=path,
            )
        if not isinstance(observed, list) or any(
            not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None
            for value in observed
        ):
            collector.add(
                "invalid_stage_input_hash",
                f"{field} must contain only lowercase SHA-256 values",
                path=path,
            )
    return computed_spec_hash, expected_input_hashes


def _inspect_csv(
    path: Path,
    *,
    name: str,
    expected_models: frozenset[str],
    expected_spec_hash: str | None,
    expected_input_hashes: str,
    collector: _IssueCollector,
) -> CsvInspection:
    collect_rows = name in SELECTED_ROW_CSV_NAMES
    selected_rows: list[CsvRow] = []
    models: Counter[str] = Counter()
    row_count = 0
    digest: str | None = None
    fieldnames: frozenset[str] = frozenset()
    if not path.is_file():
        collector.add("missing_file", "required analysis CSV is missing", path=path)
        return CsvInspection(
            name=name,
            path=path,
            sha256=None,
            fieldnames=fieldnames,
            row_count=0,
            models=frozenset(),
            model_counts={},
            selected_rows=(),
        )
    try:
        digest = file_sha256(path)
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = frozenset(reader.fieldnames or ())
            required = {
                "model_name",
                "spec_hash",
                "input_hashes",
                *CSV_REQUIRED_COLUMNS.get(name, ()),
            }
            missing = sorted(required - fieldnames)
            if missing:
                collector.add(
                    "csv_header_missing",
                    f"missing required columns: {missing}",
                    path=path,
                )
            for line_number, row in enumerate(reader, start=2):
                row_count += 1
                model = str(row.get("model_name", ""))
                models[model] += 1
                if expected_spec_hash is not None and row.get("spec_hash") != expected_spec_hash:
                    collector.add(
                        "csv_spec_hash_mismatch",
                        "row spec_hash differs from provenance",
                        path=path,
                        row=line_number,
                    )
                if row.get("input_hashes") != expected_input_hashes:
                    collector.add(
                        "csv_input_hashes_mismatch",
                        "row input_hashes differs from supplied direct inputs",
                        path=path,
                        row=line_number,
                    )
                if collect_rows:
                    selected_rows.append(CsvRow(line_number, dict(row)))
    except (OSError, UnicodeError, csv.Error) as exc:
        collector.add("invalid_csv", str(exc), path=path)
    observed_models = frozenset(model for model in models if model)
    if observed_models != expected_models:
        collector.add(
            "csv_model_set_mismatch",
            f"models are {sorted(observed_models)!r}; expected {sorted(expected_models)!r}",
            path=path,
        )
    if row_count == 0:
        collector.add("empty_analysis_output", "analysis CSV has no data rows", path=path)
    return CsvInspection(
        name=name,
        path=path,
        sha256=digest,
        fieldnames=fieldnames,
        row_count=row_count,
        models=observed_models,
        model_counts=dict(models),
        selected_rows=tuple(selected_rows),
    )


def _validate_analysis_manifest(
    manifest: Mapping[str, Any] | None,
    *,
    analysis_dir: Path,
    inspections: Mapping[str, CsvInspection],
    provenance_hash: str | None,
    spec_hash: str | None,
    config: AnalysisValidationConfig,
    collector: _IssueCollector,
) -> None:
    path = analysis_dir / "analysis_manifest.json"
    if manifest is None:
        return
    if manifest.get("analysis_version") != ANALYSIS_VERSION:
        collector.add(
            "analysis_version_mismatch",
            f"manifest declares {manifest.get('analysis_version')!r}",
            path=path,
        )
    if spec_hash is not None and manifest.get("spec_hash") != spec_hash:
        collector.add(
            "spec_hash_mismatch",
            "manifest spec_hash differs from provenance",
            path=path,
        )
    pairing = manifest.get("pairing")
    if not isinstance(pairing, Mapping):
        collector.add("invalid_pairing_manifest", "pairing must be an object", path=path)
    else:
        expected_pairing = {
            "paired": config.expected_paired_records,
            "unmatched_cued": 0,
            "unused_clean": 0,
        }
        for field, expected in expected_pairing.items():
            if _as_int(pairing.get(field)) != expected:
                collector.add(
                    "pairing_count_mismatch",
                    f"pairing.{field} is {pairing.get(field)!r}; expected {expected}",
                    path=path,
                )
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        collector.add("invalid_analysis_outputs", "outputs must be an object", path=path)
        return
    expected_names = set(ANALYSIS_MANIFEST_OUTPUT_NAMES)
    if set(outputs) != expected_names:
        collector.add(
            "analysis_output_set_mismatch",
            f"output keys differ; expected {sorted(expected_names)!r}",
            path=path,
        )
    actual_hashes = {
        name: inspection.sha256
        for name, inspection in inspections.items()
    }
    actual_hashes["provenance.json"] = provenance_hash
    for name in expected_names:
        if outputs.get(name) != actual_hashes.get(name):
            collector.add(
                "analysis_output_hash_mismatch",
                f"manifest hash for {name} does not match the file",
                path=path,
            )


def _paired_cell_key(row: Mapping[str, str]) -> tuple[str, str, str, float, str] | None:
    dose = _as_finite_float(row.get("dose"))
    if dose is None:
        return None
    return (
        str(row.get("model_name")),
        str(row.get("family")),
        str(row.get("direction")),
        dose,
        str(row.get("ordering")),
    )


def _validate_paired_grid(
    path: Path,
    *,
    config: AnalysisValidationConfig,
    expected_spec_hash: str | None,
    expected_input_hashes: str,
    collector: _IssueCollector,
) -> None:
    if not path.is_file():
        return
    cell_counts: Counter[tuple[str, str, str, float, str]] = Counter()
    model_counts: Counter[str] = Counter()
    seen_record_ids: set[str] = set()
    row_count = 0
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for line_number, row in enumerate(reader, start=2):
                row_count += 1
                model = str(row.get("model_name", ""))
                model_counts[model] += 1
                key = _paired_cell_key(row)
                if key is None:
                    collector.add(
                        "paired_grid_mismatch",
                        "paired row has an invalid grid key",
                        path=path,
                        row=line_number,
                    )
                else:
                    cell_counts[key] += 1
                record_id = str(row.get("cued_record_id", ""))
                if not record_id or record_id in seen_record_ids:
                    collector.add(
                        "duplicate_cued_record_id",
                        f"cued_record_id is missing or duplicated: {record_id!r}",
                        path=path,
                        row=line_number,
                    )
                seen_record_ids.add(record_id)
                if expected_spec_hash is not None and row.get("spec_hash") != expected_spec_hash:
                    collector.add(
                        "csv_spec_hash_mismatch",
                        "row spec_hash differs from provenance",
                        path=path,
                        row=line_number,
                    )
                if row.get("input_hashes") != expected_input_hashes:
                    collector.add(
                        "csv_input_hashes_mismatch",
                        "row input_hashes differs from supplied direct inputs",
                        path=path,
                        row=line_number,
                    )
                flip = _as_bool(row.get("flip"))
                if flip is None or flip != (row.get("clean_verdict") != row.get("cued_verdict")):
                    collector.add(
                        "paired_flip_mismatch",
                        "flip does not match clean versus cued verdicts",
                        path=path,
                        row=line_number,
                    )
                for field in (
                    "clean_entropy",
                    "clean_msp",
                    "cued_entropy",
                    "cued_msp",
                    "signed_cue_mass",
                    "js_divergence",
                ):
                    if _as_finite_float(row.get(field)) is None:
                        collector.add(
                            "paired_nonfinite",
                            f"{field} is missing or non-finite",
                            path=path,
                            row=line_number,
                        )
    except (OSError, UnicodeError, csv.Error) as exc:
        collector.add("invalid_csv", str(exc), path=path)
        return
    if row_count != config.expected_paired_records:
        collector.add(
            "paired_row_count_mismatch",
            f"paired_shifts has {row_count} rows; expected {config.expected_paired_records}",
            path=path,
        )
    expected_per_model = config.source_pairs * CUED_CONDITIONS_PER_PAIR_MODEL
    for model in config.expected_models:
        if model_counts[model] != expected_per_model:
            collector.add(
                "paired_model_count_mismatch",
                f"{model} has {model_counts[model]} paired rows; expected {expected_per_model}",
                path=path,
            )
    expected_cells = {
        (model, family, direction, dose, ordering)
        for model in config.expected_models
        for family, doses in SOCIAL_DOSES.items()
        for direction in ("congruent", "incongruent")
        for dose in doses
        for ordering in ("ab", "ba")
    }
    if set(cell_counts) != expected_cells:
        collector.add(
            "paired_grid_mismatch",
            "paired-shift family/direction/dose/ordering cells are incomplete",
            path=path,
        )
    for key, count in cell_counts.items():
        if key in expected_cells and count != config.source_pairs:
            collector.add(
                "paired_cell_count_mismatch",
                f"cell {key!r} has {count} rows; expected {config.source_pairs}",
                path=path,
            )


def _require_literal(
    row: CsvRow,
    *,
    field: str,
    expected: str,
    path: Path,
    collector: _IssueCollector,
) -> None:
    if str(row.values.get(field, "")) != expected:
        collector.add(
            "primary_semantics_mismatch",
            f"{field} is {row.values.get(field)!r}; expected {expected!r}",
            path=path,
            row=row.line_number,
        )


def _require_bool(
    row: CsvRow,
    *,
    field: str,
    expected: bool,
    path: Path,
    collector: _IssueCollector,
) -> None:
    if _as_bool(row.values.get(field)) is not expected:
        collector.add(
            "primary_semantics_mismatch",
            f"{field} is {row.values.get(field)!r}; expected {expected}",
            path=path,
            row=row.line_number,
        )


def _require_available_bool(
    row: CsvRow,
    *,
    field: str,
    expected: bool,
    path: Path,
    availability: _IssueCollector,
) -> None:
    if _as_bool(row.values.get(field)) is not expected:
        availability.add(
            "primary_result_unavailable",
            f"{field} is {row.values.get(field)!r}; expected {expected}",
            path=path,
            row=row.line_number,
        )


def _require_finite_fields(
    row: CsvRow,
    fields: Sequence[str],
    *,
    path: Path,
    availability: _IssueCollector,
) -> dict[str, float]:
    values: dict[str, float] = {}
    for field in fields:
        value = _as_finite_float(row.values.get(field))
        if value is None:
            availability.add(
                "primary_value_unavailable",
                f"{field} is missing or non-finite",
                path=path,
                row=row.line_number,
            )
        else:
            values[field] = value
    return values


def _require_positive_int_fields(
    row: CsvRow,
    fields: Sequence[str],
    *,
    path: Path,
    collector: _IssueCollector,
    availability: _IssueCollector,
) -> None:
    for field in fields:
        value = _as_int(row.values.get(field))
        if value is None or value < 0:
            collector.add(
                "primary_count_invalid",
                f"{field} must be a non-negative integer",
                path=path,
                row=row.line_number,
            )
        elif value == 0:
            availability.add(
                "primary_count_zero",
                f"{field} is zero, so some primary estimates may be unavailable",
                path=path,
                row=row.line_number,
            )


def _require_probability_fields(
    row: CsvRow,
    fields: Sequence[str],
    *,
    path: Path,
    collector: _IssueCollector,
    availability: _IssueCollector,
) -> None:
    values = _require_finite_fields(
        row,
        fields,
        path=path,
        availability=availability,
    )
    for field, value in values.items():
        if not 0.0 <= value <= 1.0:
            collector.add(
                "primary_range_invalid",
                f"{field}={value} is outside [0, 1]",
                path=path,
                row=row.line_number,
            )


def _require_ordered_interval(
    row: CsvRow,
    low_field: str,
    high_field: str,
    *,
    path: Path,
    collector: _IssueCollector,
) -> None:
    low = _as_finite_float(row.values.get(low_field))
    high = _as_finite_float(row.values.get(high_field))
    if low is not None and high is not None and low > high:
        collector.add(
            "primary_interval_invalid",
            f"{low_field}={low} exceeds {high_field}={high}",
            path=path,
            row=row.line_number,
        )


def _primary_index(
    inspection: CsvInspection,
    *,
    key: Callable[[Mapping[str, str]], tuple[Any, ...] | None],
    expected_keys: set[tuple[Any, ...]],
    collector: _IssueCollector,
    availability: _IssueCollector,
) -> dict[tuple[Any, ...], CsvRow]:
    rows: list[CsvRow] = []
    for row in inspection.selected_rows:
        primary = _as_bool(row.values.get("primary"))
        if primary:
            rows.append(row)
    indexed: dict[tuple[Any, ...], CsvRow] = {}
    for row in rows:
        row_key = key(row.values)
        if row_key is None:
            collector.add(
                "primary_cell_invalid",
                "primary row has an invalid cell key",
                path=inspection.path,
                row=row.line_number,
            )
            continue
        if row_key in indexed:
            collector.add(
                "duplicate_primary_cell",
                f"duplicate primary cell {row_key!r}",
                path=inspection.path,
                row=row.line_number,
            )
        indexed[row_key] = row
    if set(indexed) != expected_keys:
        missing = expected_keys - set(indexed)
        extra = set(indexed) - expected_keys
        if missing:
            availability.add(
                "primary_cell_missing",
                f"{len(missing)} expected primary cells are unavailable",
                path=inspection.path,
            )
        if extra:
            collector.add(
                "primary_cell_set_mismatch",
                f"{len(extra)} unexpected primary cells are present",
                path=inspection.path,
            )
    return indexed


def _dose_key(row: Mapping[str, str]) -> tuple[str, str, float] | None:
    dose = _as_finite_float(row.get("dose"))
    if dose is None:
        return None
    return str(row.get("model_name")), str(row.get("family")), dose


def _family_key(row: Mapping[str, str]) -> tuple[str, str]:
    return str(row.get("model_name")), str(row.get("family"))


def _threshold_key(row: Mapping[str, str]) -> tuple[str, str, str]:
    return (
        str(row.get("model_name")),
        str(row.get("family")),
        str(row.get("ordering")),
    )


def _expected_primary(name: str, row: Mapping[str, str]) -> bool:
    family = str(row.get("family", ""))
    dose = _as_finite_float(row.get("dose"))
    clean_tie = _as_bool(row.get("clean_tie"))
    if name == "rq1_silent_shift.csv":
        return (
            row.get("direction") == "incongruent"
            and clean_tie is False
            and row.get("metric") == "signed_cue_mass"
            and _as_bool(row.get("non_flipped_only")) is True
        )
    if name == "rq1_susceptibility.csv":
        return row.get("shift_metric") == "signed_cue_mass"
    if name == "rq2_threshold_transfer.csv":
        doses = SOCIAL_DOSES.get(family, ())
        return (
            row.get("aggregation") == "single_ordering"
            and family in SOCIAL_DOSES
            and row.get("direction") == "incongruent"
            and clean_tie is False
            and row.get("routing_split") == "test"
            and _as_finite_float(row.get("target_risk")) == 0.10
            and row.get("confidence_channel") == "msp"
            and bool(doses)
            and dose == max(doses)
        )
    if name == "rq2_mcnemar.csv":
        return (
            row.get("direction") == "incongruent"
            and clean_tie is False
            and row.get("routing_split") == "test"
        )
    if name == "rq3_dose_response.csv":
        return row.get("direction") == "incongruent" and clean_tie is False
    if name == "rq3_uncertainty_trend.csv":
        return (
            row.get("direction") == "incongruent"
            and clean_tie is False
            and row.get("metric") == "cued_entropy"
            and row.get("stable_set") == "pre_first_flip"
            and row.get("estimator") == "gaussian_gee_exchangeable"
            and _as_bool(row.get("sensitivity_analysis")) is False
        )
    if name == "rq3_uncertainty_by_dose.csv":
        return True
    raise ValueError(f"unsupported primary-selector CSV: {name}")


def _validate_primary_selectors(
    inspections: Mapping[str, CsvInspection],
    *,
    collector: _IssueCollector,
) -> None:
    for name in PRIMARY_COLUMN_CSV_NAMES:
        inspection = inspections[name]
        if "primary" not in inspection.fieldnames:
            continue
        for row in inspection.selected_rows:
            observed = _as_bool(row.values.get("primary"))
            if observed is None:
                collector.add(
                    "primary_selector_invalid",
                    "primary must be an explicit true/false value",
                    path=inspection.path,
                    row=row.line_number,
                )
                continue
            expected = _expected_primary(name, row.values)
            if observed is not expected:
                collector.add(
                    "primary_selector_mismatch",
                    (
                        f"primary is {observed}; analyzer semantics require "
                        f"{expected}"
                    ),
                    path=inspection.path,
                    row=row.line_number,
                )


def _validate_primary_outputs(
    inspections: Mapping[str, CsvInspection],
    *,
    config: AnalysisValidationConfig,
    collector: _IssueCollector,
    availability: _IssueCollector,
) -> None:
    dose_cells = {
        (model, family, dose)
        for model in config.expected_models
        for family, doses in SOCIAL_DOSES.items()
        for dose in doses
    }
    family_cells = {
        (model, family)
        for model in config.expected_models
        for family in SOCIAL_DOSES
    }

    silent = inspections["rq1_silent_shift.csv"]
    for row in _primary_index(
        silent,
        key=_dose_key,
        expected_keys=dose_cells,
        collector=collector,
        availability=availability,
    ).values():
        _require_literal(row, field="direction", expected="incongruent", path=silent.path, collector=collector)
        _require_literal(row, field="metric", expected="signed_cue_mass", path=silent.path, collector=collector)
        _require_bool(row, field="clean_tie", expected=False, path=silent.path, collector=collector)
        _require_bool(row, field="non_flipped_only", expected=True, path=silent.path, collector=collector)
        _require_finite_fields(
            row,
            ("estimate", "ci_low", "ci_high"),
            path=silent.path,
            availability=availability,
        )
        _require_probability_fields(
            row,
            ("p_value_one_sided", "p_value_holm"),
            path=silent.path,
            collector=collector,
            availability=availability,
        )
        _require_positive_int_fields(
            row,
            ("n", "n_questions"),
            path=silent.path,
            collector=collector,
            availability=availability,
        )
        _require_ordered_interval(row, "ci_low", "ci_high", path=silent.path, collector=collector)

    susceptibility = inspections["rq1_susceptibility.csv"]
    for row in _primary_index(
        susceptibility,
        key=_family_key,
        expected_keys=family_cells,
        collector=collector,
        availability=availability,
    ).values():
        family = str(row.values.get("family"))
        doses = SOCIAL_DOSES.get(family, ())
        _require_literal(row, field="direction", expected="incongruent", path=susceptibility.path, collector=collector)
        _require_literal(row, field="shift_metric", expected="signed_cue_mass", path=susceptibility.path, collector=collector)
        _require_literal(row, field="baseline_channel", expected="entropy", path=susceptibility.path, collector=collector)
        if doses:
            for field, expected in (("low_dose", min(doses)), ("high_dose", max(doses))):
                value = _as_finite_float(row.values.get(field))
                if value != expected:
                    collector.add(
                        "primary_semantics_mismatch",
                        f"{field} is {value}; expected {expected}",
                        path=susceptibility.path,
                        row=row.line_number,
                    )
        _require_probability_fields(
            row,
            (
                "shift_auc",
                "shift_auc_ci_low",
                "shift_auc_ci_high",
                "clean_baseline_auc",
                "clean_baseline_auc_ci_low",
                "clean_baseline_auc_ci_high",
            ),
            path=susceptibility.path,
            collector=collector,
            availability=availability,
        )
        _require_finite_fields(
            row,
            ("auc_difference", "auc_difference_ci_low", "auc_difference_ci_high"),
            path=susceptibility.path,
            availability=availability,
        )
        _require_positive_int_fields(
            row,
            ("n", "n_clusters", "n_resamples", "positives"),
            path=susceptibility.path,
            collector=collector,
            availability=availability,
        )
        if _as_int(row.values.get("n_resamples")) != config.bootstrap_resamples:
            collector.add(
                "primary_resample_mismatch",
                "susceptibility bootstrap count differs from the analysis spec",
                path=susceptibility.path,
                row=row.line_number,
            )
        for low, high in (
            ("shift_auc_ci_low", "shift_auc_ci_high"),
            ("clean_baseline_auc_ci_low", "clean_baseline_auc_ci_high"),
            ("auc_difference_ci_low", "auc_difference_ci_high"),
        ):
            _require_ordered_interval(row, low, high, path=susceptibility.path, collector=collector)

    transfer = inspections["rq2_threshold_transfer.csv"]
    threshold_cells = {
        (model, family, ordering)
        for model in config.expected_models
        for family in SOCIAL_DOSES
        for ordering in ("ab", "ba")
    }
    for row in _primary_index(
        transfer,
        key=_threshold_key,
        expected_keys=threshold_cells,
        collector=collector,
        availability=availability,
    ).values():
        family = str(row.values.get("family"))
        highest_dose = max(SOCIAL_DOSES.get(family, (math.nan,)))
        expected_literals = {
            "aggregation": "single_ordering",
            "confidence_channel": "msp",
            "decision_rule": "risk_inflation_vs_target_ci_low > 0",
            "direction": "incongruent",
            "routing_split": "test",
        }
        for field, expected in expected_literals.items():
            _require_literal(row, field=field, expected=expected, path=transfer.path, collector=collector)
        _require_bool(row, field="clean_tie", expected=False, path=transfer.path, collector=collector)
        if _as_finite_float(row.values.get("dose")) != highest_dose:
            collector.add(
                "primary_semantics_mismatch",
                "RQ2 primary row is not at the highest family dose",
                path=transfer.path,
                row=row.line_number,
            )
        expected_variant = (
            f"{family}_incongruent_{int(highest_dose)}_"
            f"{row.values.get('ordering')}"
        )
        if row.values.get("variant_id") != expected_variant:
            collector.add(
                "primary_semantics_mismatch",
                f"variant_id does not match {expected_variant!r}",
                path=transfer.path,
                row=row.line_number,
            )
        if _as_finite_float(row.values.get("target_risk")) != 0.10:
            collector.add(
                "primary_semantics_mismatch",
                "RQ2 primary row does not use target risk 0.10",
                path=transfer.path,
                row=row.line_number,
            )
        _require_finite_fields(
            row,
            (
                "threshold",
                "calibration_coverage",
                "calibration_risk",
                "test_coverage",
                "test_realized_risk",
                "test_risk_inflation_vs_target",
                "risk_inflation_vs_target_ci_low",
                "risk_inflation_vs_target_ci_high",
                "test_accepted_flip_fraction",
                "accepted_flip_fraction_ci_low",
                "accepted_flip_fraction_ci_high",
            ),
            path=transfer.path,
            availability=availability,
        )
        _require_probability_fields(
            row,
            ("risk_inflation_vs_target_p_value_one_sided", "p_value_holm"),
            path=transfer.path,
            collector=collector,
            availability=availability,
        )
        _require_positive_int_fields(
            row,
            (
                "calibration_n",
                "test_n",
                "n_calibration_clusters",
                "n_test_clusters",
                "n_resamples",
                "test_accepted",
                "test_flips",
            ),
            path=transfer.path,
            collector=collector,
            availability=availability,
        )
        if _as_int(row.values.get("n_resamples")) != config.bootstrap_resamples:
            collector.add(
                "primary_resample_mismatch",
                "threshold-transfer bootstrap count differs from the analysis spec",
                path=transfer.path,
                row=row.line_number,
            )
        for low, high in (
            ("risk_inflation_vs_target_ci_low", "risk_inflation_vs_target_ci_high"),
            ("accepted_flip_fraction_ci_low", "accepted_flip_fraction_ci_high"),
        ):
            _require_ordered_interval(row, low, high, path=transfer.path, collector=collector)

    dose = inspections["rq3_dose_response.csv"]
    for row in _primary_index(
        dose,
        key=_family_key,
        expected_keys=family_cells,
        collector=collector,
        availability=availability,
    ).values():
        _require_literal(row, field="direction", expected="incongruent", path=dose.path, collector=collector)
        _require_bool(row, field="clean_tie", expected=False, path=dose.path, collector=collector)
        _require_available_bool(
            row,
            field="converged",
            expected=True,
            path=dose.path,
            availability=availability,
        )
        _require_finite_fields(
            row,
            (
                "intercept",
                "slope",
                "slope_ci_low",
                "slope_ci_high",
                "p25_dose",
                "p25_ci_low",
                "p25_ci_high",
            ),
            path=dose.path,
            availability=availability,
        )
        _require_probability_fields(
            row,
            ("slope_p_value_one_sided", "p_value_holm"),
            path=dose.path,
            collector=collector,
            availability=availability,
        )
        _require_positive_int_fields(
            row,
            ("n", "n_clusters", "n_resamples", "events"),
            path=dose.path,
            collector=collector,
            availability=availability,
        )
        if _as_int(row.values.get("n_resamples")) != config.bootstrap_resamples:
            collector.add(
                "primary_resample_mismatch",
                "dose-response bootstrap count differs from the analysis spec",
                path=dose.path,
                row=row.line_number,
            )
        for low, high in (
            ("slope_ci_low", "slope_ci_high"),
            ("p25_ci_low", "p25_ci_high"),
        ):
            _require_ordered_interval(row, low, high, path=dose.path, collector=collector)

    trend = inspections["rq3_uncertainty_trend.csv"]
    for row in _primary_index(
        trend,
        key=_family_key,
        expected_keys=family_cells,
        collector=collector,
        availability=availability,
    ).values():
        for field, expected in {
            "direction": "incongruent",
            "metric": "cued_entropy",
            "stable_set": "pre_first_flip",
            "estimator": "gaussian_gee_exchangeable",
            "formula": UNCERTAINTY_GEE_FORMULA,
        }.items():
            _require_literal(row, field=field, expected=expected, path=trend.path, collector=collector)
        _require_bool(row, field="clean_tie", expected=False, path=trend.path, collector=collector)
        _require_bool(row, field="sensitivity_analysis", expected=False, path=trend.path, collector=collector)
        _require_positive_int_fields(
            row,
            (
                "n",
                "n_clusters",
                "bootstrap_resamples_requested",
                "bootstrap_resamples_successful",
            ),
            path=trend.path,
            collector=collector,
            availability=availability,
        )
        if (
            _as_int(row.values.get("bootstrap_resamples_requested"))
            != config.bootstrap_resamples
        ):
            collector.add(
                "primary_resample_mismatch",
                "bootstrap_resamples_requested differs from the analysis spec",
                path=trend.path,
                row=row.line_number,
            )
        successful = _as_int(row.values.get("bootstrap_resamples_successful"))
        if successful is not None and successful > config.bootstrap_resamples:
            collector.add(
                "primary_resample_mismatch",
                "successful bootstrap count exceeds the requested count",
                path=trend.path,
                row=row.line_number,
            )
        elif successful is not None and successful < config.bootstrap_resamples:
            availability.add(
                "primary_bootstrap_incomplete",
                (
                    f"only {successful} of {config.bootstrap_resamples} "
                    "primary bootstrap fits succeeded"
                ),
                path=trend.path,
                row=row.line_number,
            )
        status = str(row.values.get("status", ""))
        if status == "unavailable":
            availability.add(
                "primary_result_unavailable",
                str(row.values.get("message") or "uncertainty trend unavailable"),
                path=trend.path,
                row=row.line_number,
            )
            continue
        if status != "ok":
            collector.add(
                "primary_semantics_mismatch",
                f"status is {status!r}; expected 'ok' or 'unavailable'",
                path=trend.path,
                row=row.line_number,
            )
        _require_available_bool(
            row,
            field="converged",
            expected=True,
            path=trend.path,
            availability=availability,
        )
        _require_finite_fields(
            row,
            (
                "intercept",
                "slope",
                "statistic",
                "slope_standard_error",
                "slope_z_value",
                "slope_ci_low",
                "slope_ci_high",
            ),
            path=trend.path,
            availability=availability,
        )
        _require_probability_fields(
            row,
            ("slope_p_value_one_sided", "p_value_holm"),
            path=trend.path,
            collector=collector,
            availability=availability,
        )
        _require_ordered_interval(
            row,
            "slope_ci_low",
            "slope_ci_high",
            path=trend.path,
            collector=collector,
        )

    by_dose = inspections["rq3_uncertainty_by_dose.csv"]
    for row in _primary_index(
        by_dose,
        key=_dose_key,
        expected_keys=dose_cells,
        collector=collector,
        availability=availability,
    ).values():
        for field, expected in {
            "direction": "incongruent",
            "metric": "cued_entropy",
            "stable_set": "pre_first_flip",
        }.items():
            _require_literal(row, field=field, expected=expected, path=by_dose.path, collector=collector)
        _require_bool(row, field="clean_tie", expected=False, path=by_dose.path, collector=collector)
        _require_finite_fields(
            row,
            ("normalized_dose", "estimate", "ci_low", "ci_high"),
            path=by_dose.path,
            availability=availability,
        )
        _require_positive_int_fields(
            row,
            ("n", "n_clusters", "n_resamples"),
            path=by_dose.path,
            collector=collector,
            availability=availability,
        )
        if _as_int(row.values.get("n_resamples")) != config.bootstrap_resamples:
            collector.add(
                "primary_resample_mismatch",
                "uncertainty-by-dose bootstrap count differs from the analysis spec",
                path=by_dose.path,
                row=row.line_number,
            )
        _require_ordered_interval(row, "ci_low", "ci_high", path=by_dose.path, collector=collector)


ConditionCell = tuple[
    str,
    str,
    str,
    str,
    Optional[float],
    str,
    bool,
    str,
]
CalibrationKey = tuple[
    str,
    str,
    str,
    str,
    Optional[float],
    str,
    bool,
    str,
    str,
]


@dataclass(frozen=True, slots=True)
class _CalibrationCell:
    n: int
    total_n: int
    accuracy: float
    ece: float
    n_bins: int


@dataclass(slots=True)
class _ReliabilityState:
    bin_indices: set[int] = field(default_factory=set)
    n: int = 0
    weighted_accuracy: float = 0.0
    weighted_ece: float = 0.0


@dataclass(slots=True)
class _RiskState:
    total: int
    aurc: float
    line_number: int
    row_count: int = 0
    accepted: int = 0
    coverage: float = 0.0
    risk: float = 0.0
    threshold: float = math.inf
    area: float = 0.0


def _iter_csv_rows(
    path: Path,
    *,
    collector: _IssueCollector,
) -> Iterator[CsvRow]:
    if not path.is_file():
        return
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for line_number, row in enumerate(reader, start=2):
                yield CsvRow(line_number, dict(row))
    except (OSError, UnicodeError, csv.Error) as exc:
        collector.add("invalid_csv", str(exc), path=path)


def _condition_cell(
    row: CsvRow,
    *,
    path: Path,
    collector: _IssueCollector,
    require_variant: bool = True,
    allow_clean: bool = True,
) -> ConditionCell | None:
    values = row.values
    model = str(values.get("model_name", ""))
    ordering = str(values.get("ordering", "")).lower()
    family = str(values.get("family", "")).lower()
    direction = str(values.get("direction", "")).lower()
    routing_split = str(values.get("routing_split", "")).lower()
    clean_tie = _as_bool(values.get("clean_tie"))
    raw_dose = values.get("dose")
    dose = None if _is_missing(raw_dose) else _as_finite_float(raw_dose)
    valid = True
    if ordering not in {"ab", "ba"}:
        valid = False
    if routing_split not in {"calibration", "test"}:
        valid = False
    if clean_tie is None:
        valid = False
    if family == "clean" and allow_clean:
        if (
            direction not in {"clean", "control"}
            or dose is not None
            or (
                require_variant
                and values.get("variant_id") != f"clean_{ordering}"
            )
        ):
            valid = False
        direction = "clean"
    elif family in SOCIAL_DOSES:
        doses = SOCIAL_DOSES[family]
        expected_variant = (
            f"{family}_{direction}_{int(dose)}_{ordering}"
            if dose is not None and dose.is_integer()
            else ""
        )
        if (
            direction not in {"congruent", "incongruent"}
            or dose not in doses
            or (
                require_variant
                and values.get("variant_id") != expected_variant
            )
        ):
            valid = False
    else:
        valid = False
    if not valid:
        collector.add(
            "rq2_condition_invalid",
            "condition columns do not encode a valid Silent Bias cell",
            path=path,
            row=row.line_number,
        )
        return None
    return (
        model,
        ordering,
        family,
        direction,
        dose,
        str(values.get("variant_id", "")) if require_variant else "",
        bool(clean_tie),
        routing_split,
    )


def _calibration_key(
    row: CsvRow,
    *,
    path: Path,
    collector: _IssueCollector,
) -> CalibrationKey | None:
    condition = _condition_cell(row, path=path, collector=collector)
    channel = str(row.values.get("confidence_channel", ""))
    if channel not in CONFIDENCE_CHANNELS:
        collector.add(
            "rq2_confidence_channel_invalid",
            f"unsupported confidence channel {channel!r}",
            path=path,
            row=row.line_number,
        )
        return None
    if condition is None:
        return None
    return (*condition, channel)


def _expected_base_conditions(
    expected_models: Sequence[str],
) -> set[tuple[str, str, str, str, Optional[float]]]:
    clean = {
        (model, ordering, "clean", "clean", None)
        for model in expected_models
        for ordering in ("ab", "ba")
    }
    social = {
        (model, ordering, family, direction, dose)
        for model in expected_models
        for ordering in ("ab", "ba")
        for family, doses in SOCIAL_DOSES.items()
        for direction in ("congruent", "incongruent")
        for dose in doses
    }
    return clean | social


def _base_conditions(keys: Sequence[CalibrationKey]) -> set[
    tuple[str, str, str, str, Optional[float]]
]:
    return {
        (key[0], key[1], key[2], key[3], key[4])
        for key in keys
        if key[-1] == "msp"
    }


def _expected_rq2_conditions_from_paired(
    path: Path,
    *,
    collector: _IssueCollector,
) -> set[ConditionCell]:
    social: set[ConditionCell] = set()
    clean: set[ConditionCell] = set()
    for row in _iter_csv_rows(path, collector=collector):
        condition = _condition_cell(
            row,
            path=path,
            collector=collector,
            require_variant=True,
            allow_clean=False,
        )
        if condition is None:
            continue
        social.add(condition)
        (
            model,
            ordering,
            _family,
            _direction,
            _dose,
            _variant_id,
            clean_tie,
            routing_split,
        ) = condition
        clean.add(
            (
                model,
                ordering,
                "clean",
                "clean",
                None,
                f"clean_{ordering}",
                clean_tie,
                routing_split,
            )
        )
    return social | clean


def _validate_rq2_calibration(
    path: Path,
    *,
    config: AnalysisValidationConfig,
    expected_conditions: set[ConditionCell],
    collector: _IssueCollector,
    availability: _IssueCollector,
) -> dict[CalibrationKey, _CalibrationCell]:
    cells: dict[CalibrationKey, _CalibrationCell] = {}
    total_by_condition: dict[ConditionCell, int] = {}
    for row in _iter_csv_rows(path, collector=collector):
        key = _calibration_key(row, path=path, collector=collector)
        if key is None:
            continue
        if key in cells:
            collector.add(
                "rq2_calibration_duplicate_cell",
                "calibration condition/channel cell is duplicated",
                path=path,
                row=row.line_number,
            )
            continue
        n = _required_int(
            row.values,
            "n",
            minimum=1,
            code="rq2_calibration_count_invalid",
            path=path,
            line_number=row.line_number,
            collector=collector,
        )
        total_n = _required_int(
            row.values,
            "total_n",
            minimum=1,
            code="rq2_calibration_count_invalid",
            path=path,
            line_number=row.line_number,
            collector=collector,
        )
        missing_n = _required_int(
            row.values,
            "missing_n",
            code="rq2_calibration_count_invalid",
            path=path,
            line_number=row.line_number,
            collector=collector,
        )
        brier_n = _required_int(
            row.values,
            "brier_n",
            code="rq2_calibration_count_invalid",
            path=path,
            line_number=row.line_number,
            collector=collector,
        )
        n_bins = _required_int(
            row.values,
            "n_bins",
            minimum=1,
            code="rq2_calibration_count_invalid",
            path=path,
            line_number=row.line_number,
            collector=collector,
        )
        accuracy = _required_float(
            row.values,
            "accuracy",
            lower=0.0,
            upper=1.0,
            code="rq2_calibration_value_invalid",
            path=path,
            line_number=row.line_number,
            collector=collector,
        )
        ece = _required_float(
            row.values,
            "ece",
            lower=0.0,
            upper=1.0,
            code="rq2_calibration_value_invalid",
            path=path,
            line_number=row.line_number,
            collector=collector,
        )
        availability_rate = _required_float(
            row.values,
            "availability_rate",
            lower=0.0,
            upper=1.0,
            code="rq2_calibration_value_invalid",
            path=path,
            line_number=row.line_number,
            collector=collector,
        )
        channel = key[-1]
        brier = _optional_float(
            row.values,
            "brier",
            lower=0.0,
            upper=2.0,
            code="rq2_calibration_value_invalid",
            path=path,
            line_number=row.line_number,
            collector=collector,
        )
        if row.values.get("tie_policy") != "strict_three_class":
            collector.add(
                "rq2_calibration_semantics_invalid",
                "tie_policy must be strict_three_class",
                path=path,
                row=row.line_number,
            )
        if n_bins is not None and n_bins != config.ece_bins:
            collector.add(
                "rq2_calibration_bin_count_mismatch",
                f"n_bins={n_bins}; expected {config.ece_bins}",
                path=path,
                row=row.line_number,
            )
        if n is not None and total_n is not None and missing_n is not None:
            if n > total_n or missing_n != total_n - n:
                collector.add(
                    "rq2_calibration_equation_mismatch",
                    "n, total_n, and missing_n are inconsistent",
                    path=path,
                    row=row.line_number,
                )
            _check_equation(
                availability_rate,
                n / total_n,
                field_name="availability_rate",
                code="rq2_calibration_equation_mismatch",
                path=path,
                line_number=row.line_number,
                collector=collector,
            )
            condition = key[:-1]
            previous_total = total_by_condition.setdefault(condition, total_n)
            if previous_total != total_n:
                collector.add(
                    "rq2_calibration_total_mismatch",
                    "confidence channels disagree on total_n",
                    path=path,
                    row=row.line_number,
                )
        if n is not None and accuracy is not None and not _close(
            accuracy * n,
            round(accuracy * n),
        ):
            collector.add(
                "rq2_calibration_equation_mismatch",
                "accuracy*n is not an integer count",
                path=path,
                row=row.line_number,
            )
        if channel == "msp":
            if n is not None and brier_n != n:
                collector.add(
                    "rq2_calibration_brier_mismatch",
                    "MSP requires brier_n=n",
                    path=path,
                    row=row.line_number,
                )
            if brier is None:
                availability.add(
                    "rq2_calibration_brier_unavailable",
                    "MSP Brier score is unavailable",
                    path=path,
                    row=row.line_number,
                )
        elif brier_n != 0 or not _is_missing(row.values.get("brier")):
            collector.add(
                "rq2_calibration_brier_mismatch",
                "secondary confidence channels require brier_n=0 and blank brier",
                path=path,
                row=row.line_number,
            )
        if (
            n is not None
            and total_n is not None
            and accuracy is not None
            and ece is not None
            and n_bins is not None
        ):
            cells[key] = _CalibrationCell(
                n=n,
                total_n=total_n,
                accuracy=accuracy,
                ece=ece,
                n_bins=n_bins,
            )
    expected = _expected_base_conditions(config.expected_models)
    observed = _base_conditions(tuple(cells))
    if observed != expected:
        collector.add(
            "rq2_calibration_condition_grid_mismatch",
            (
                f"MSP base grid has {len(observed)} cells; expected "
                f"{len(expected)}"
            ),
            path=path,
        )
    observed_strata = {key[:-1] for key in cells if key[-1] == "msp"}
    if observed_strata != expected_conditions:
        collector.add(
            "rq2_calibration_stratum_set_mismatch",
            (
                f"MSP has {len(observed_strata)} condition/tie/split strata; "
                f"expected {len(expected_conditions)} from paired shifts"
            ),
            path=path,
        )
    return cells


def _validate_rq2_reliability(
    path: Path,
    *,
    calibration: Mapping[CalibrationKey, _CalibrationCell],
    expected_conditions: set[ConditionCell],
    config: AnalysisValidationConfig,
    collector: _IssueCollector,
) -> None:
    states: dict[CalibrationKey, _ReliabilityState] = {}
    for row in _iter_csv_rows(path, collector=collector):
        key = _calibration_key(row, path=path, collector=collector)
        if key is None:
            continue
        summary = calibration.get(key)
        if summary is None:
            collector.add(
                "rq2_reliability_orphan_cell",
                "reliability cell has no matching calibration summary",
                path=path,
                row=row.line_number,
            )
            continue
        if row.values.get("tie_policy") != "strict_three_class":
            collector.add(
                "rq2_reliability_semantics_invalid",
                "tie_policy must be strict_three_class",
                path=path,
                row=row.line_number,
            )
        bin_index = _required_int(
            row.values,
            "bin_index",
            maximum=config.ece_bins - 1,
            code="rq2_reliability_bin_invalid",
            path=path,
            line_number=row.line_number,
            collector=collector,
        )
        n = _required_int(
            row.values,
            "n",
            code="rq2_reliability_count_invalid",
            path=path,
            line_number=row.line_number,
            collector=collector,
        )
        lower = _required_float(
            row.values,
            "lower",
            lower=0.0,
            upper=1.0,
            code="rq2_reliability_bin_invalid",
            path=path,
            line_number=row.line_number,
            collector=collector,
        )
        upper = _required_float(
            row.values,
            "upper",
            lower=0.0,
            upper=1.0,
            code="rq2_reliability_bin_invalid",
            path=path,
            line_number=row.line_number,
            collector=collector,
        )
        mean_confidence = _optional_float(
            row.values,
            "mean_confidence",
            lower=0.0,
            upper=1.0,
            code="rq2_reliability_value_invalid",
            path=path,
            line_number=row.line_number,
            collector=collector,
        )
        accuracy = _optional_float(
            row.values,
            "accuracy",
            lower=0.0,
            upper=1.0,
            code="rq2_reliability_value_invalid",
            path=path,
            line_number=row.line_number,
            collector=collector,
        )
        if bin_index is not None:
            expected_lower = bin_index / config.ece_bins
            expected_upper = (bin_index + 1) / config.ece_bins
            if (
                lower is None
                or upper is None
                or not _close(lower, expected_lower)
                or not _close(upper, expected_upper)
            ):
                collector.add(
                    "rq2_reliability_bin_invalid",
                    "bin bounds do not match bin_index/n_bins",
                    path=path,
                    row=row.line_number,
                )
        state = states.setdefault(key, _ReliabilityState())
        if bin_index is not None:
            if bin_index in state.bin_indices:
                collector.add(
                    "rq2_reliability_duplicate_bin",
                    f"bin {bin_index} is duplicated",
                    path=path,
                    row=row.line_number,
                )
            state.bin_indices.add(bin_index)
        if n is None:
            continue
        state.n += n
        if n == 0:
            if not _is_missing(row.values.get("mean_confidence")) or not _is_missing(
                row.values.get("accuracy")
            ):
                collector.add(
                    "rq2_reliability_empty_bin_invalid",
                    "empty bins require blank confidence and accuracy",
                    path=path,
                    row=row.line_number,
                )
            continue
        if mean_confidence is None or accuracy is None:
            collector.add(
                "rq2_reliability_nonempty_bin_invalid",
                "nonempty bins require confidence and accuracy",
                path=path,
                row=row.line_number,
            )
            continue
        if (
            lower is not None
            and upper is not None
            and (
                mean_confidence < lower - 1e-12
                or mean_confidence > upper + 1e-12
                or (
                    bin_index != config.ece_bins - 1
                    and _close(mean_confidence, upper)
                )
            )
        ):
            collector.add(
                "rq2_reliability_mean_outside_bin",
                "mean_confidence is outside its bin",
                path=path,
                row=row.line_number,
            )
        if not _close(accuracy * n, round(accuracy * n)):
            collector.add(
                "rq2_reliability_equation_mismatch",
                "accuracy*n is not an integer count",
                path=path,
                row=row.line_number,
            )
        state.weighted_accuracy += n * accuracy
        state.weighted_ece += n * abs(accuracy - mean_confidence)
    if set(states) != set(calibration):
        collector.add(
            "rq2_reliability_cell_set_mismatch",
            "reliability and calibration condition/channel cells differ",
            path=path,
        )
    observed_strata = {key[:-1] for key in states if key[-1] == "msp"}
    if observed_strata != expected_conditions:
        collector.add(
            "rq2_reliability_stratum_set_mismatch",
            "MSP reliability strata differ from paired shifts",
            path=path,
        )
    expected_bins = set(range(config.ece_bins))
    for key, summary in calibration.items():
        state = states.get(key)
        if state is None:
            continue
        if state.bin_indices != expected_bins:
            collector.add(
                "rq2_reliability_bin_set_mismatch",
                "reliability cell does not contain every configured bin",
                path=path,
            )
        if state.n != summary.n:
            collector.add(
                "rq2_reliability_count_mismatch",
                f"bin counts sum to {state.n}; calibration n={summary.n}",
                path=path,
            )
            continue
        _check_equation(
            summary.accuracy,
            state.weighted_accuracy / summary.n,
            field_name="calibration accuracy",
            code="rq2_reliability_equation_mismatch",
            path=path,
            line_number=None,
            collector=collector,
        )
        _check_equation(
            summary.ece,
            state.weighted_ece / summary.n,
            field_name="calibration ece",
            code="rq2_reliability_equation_mismatch",
            path=path,
            line_number=None,
            collector=collector,
        )


def _risk_threshold(
    row: CsvRow,
    *,
    path: Path,
    collector: _IssueCollector,
) -> float | None:
    raw = str(row.values.get("threshold", "")).strip().lower()
    if raw in {"inf", "+inf", "infinity", "+infinity"}:
        return math.inf
    return _required_float(
        row.values,
        "threshold",
        lower=0.0,
        upper=1.0,
        code="rq2_risk_coverage_threshold_invalid",
        path=path,
        line_number=row.line_number,
        collector=collector,
    )


def _validate_rq2_risk_coverage(
    path: Path,
    *,
    calibration: Mapping[CalibrationKey, _CalibrationCell],
    expected_conditions: set[ConditionCell],
    collector: _IssueCollector,
) -> None:
    states: dict[CalibrationKey, _RiskState] = {}
    for row in _iter_csv_rows(path, collector=collector):
        key = _calibration_key(row, path=path, collector=collector)
        if key is None:
            continue
        summary = calibration.get(key)
        if summary is None:
            collector.add(
                "rq2_risk_coverage_orphan_cell",
                "risk-coverage cell has no matching calibration summary",
                path=path,
                row=row.line_number,
            )
            continue
        total = _required_int(
            row.values,
            "total",
            minimum=1,
            code="rq2_risk_coverage_count_invalid",
            path=path,
            line_number=row.line_number,
            collector=collector,
        )
        accepted = _required_int(
            row.values,
            "accepted",
            code="rq2_risk_coverage_count_invalid",
            path=path,
            line_number=row.line_number,
            collector=collector,
        )
        coverage = _required_float(
            row.values,
            "coverage",
            lower=0.0,
            upper=1.0,
            code="rq2_risk_coverage_value_invalid",
            path=path,
            line_number=row.line_number,
            collector=collector,
        )
        risk = _required_float(
            row.values,
            "risk",
            lower=0.0,
            upper=1.0,
            code="rq2_risk_coverage_value_invalid",
            path=path,
            line_number=row.line_number,
            collector=collector,
        )
        aurc = _required_float(
            row.values,
            "aurc",
            lower=0.0,
            upper=1.0,
            code="rq2_risk_coverage_value_invalid",
            path=path,
            line_number=row.line_number,
            collector=collector,
        )
        threshold = _risk_threshold(row, path=path, collector=collector)
        if None in {total, accepted, coverage, risk, aurc, threshold}:
            continue
        assert total is not None
        assert accepted is not None
        assert coverage is not None
        assert risk is not None
        assert aurc is not None
        assert threshold is not None
        if accepted > total:
            collector.add(
                "rq2_risk_coverage_count_invalid",
                "accepted exceeds total",
                path=path,
                row=row.line_number,
            )
        _check_equation(
            coverage,
            accepted / total,
            field_name="coverage",
            code="rq2_risk_coverage_equation_mismatch",
            path=path,
            line_number=row.line_number,
            collector=collector,
        )
        if accepted > 0 and not _close(risk * accepted, round(risk * accepted)):
            collector.add(
                "rq2_risk_coverage_equation_mismatch",
                "risk*accepted is not an integer error count",
                path=path,
                row=row.line_number,
            )
        state = states.get(key)
        if state is None:
            state = _RiskState(
                total=total,
                aurc=aurc,
                line_number=row.line_number,
            )
            states[key] = state
        elif state.total != total or not _close(state.aurc, aurc):
            collector.add(
                "rq2_risk_coverage_cell_mismatch",
                "total or AURC changes within one curve",
                path=path,
                row=row.line_number,
            )
        if state.row_count == 0:
            if (
                not math.isinf(threshold)
                or threshold < 0.0
                or accepted != 0
                or not _close(coverage, 0.0)
                or not _close(risk, 0.0)
            ):
                collector.add(
                    "rq2_risk_coverage_origin_invalid",
                    "curve must begin at (+inf, coverage=0, risk=0)",
                    path=path,
                    row=row.line_number,
                )
        else:
            if (
                not math.isfinite(threshold)
                or accepted <= state.accepted
                or threshold >= state.threshold
            ):
                collector.add(
                    "rq2_risk_coverage_order_invalid",
                    "curve thresholds must descend as accepted count increases",
                    path=path,
                    row=row.line_number,
                )
            state.area += (
                (coverage - state.coverage) * (state.risk + risk) / 2.0
            )
        state.row_count += 1
        state.accepted = accepted
        state.coverage = coverage
        state.risk = risk
        state.threshold = threshold
    if set(states) != set(calibration):
        collector.add(
            "rq2_risk_coverage_cell_set_mismatch",
            "risk-coverage and calibration condition/channel cells differ",
            path=path,
        )
    observed_strata = {key[:-1] for key in states if key[-1] == "msp"}
    if observed_strata != expected_conditions:
        collector.add(
            "rq2_risk_coverage_stratum_set_mismatch",
            "MSP risk-coverage strata differ from paired shifts",
            path=path,
        )
    for key, summary in calibration.items():
        state = states.get(key)
        if state is None:
            continue
        if state.total != summary.n:
            collector.add(
                "rq2_risk_coverage_count_mismatch",
                f"curve total={state.total}; calibration n={summary.n}",
                path=path,
                row=state.line_number,
            )
        if state.accepted != state.total or not _close(state.coverage, 1.0):
            collector.add(
                "rq2_risk_coverage_endpoint_invalid",
                "curve must end at full coverage",
                path=path,
                row=state.line_number,
            )
        if not _close(state.aurc, state.area):
            collector.add(
                "rq2_risk_coverage_aurc_mismatch",
                f"AURC={state.aurc}; trapezoid area={state.area}",
                path=path,
                row=state.line_number,
            )
        if not _close(state.risk, 1.0 - summary.accuracy):
            collector.add(
                "rq2_risk_coverage_accuracy_mismatch",
                "full-coverage risk does not equal 1-accuracy",
                path=path,
                row=state.line_number,
            )


def _validate_rq2_mcnemar(
    path: Path,
    *,
    config: AnalysisValidationConfig,
    collector: _IssueCollector,
    availability: _IssueCollector,
) -> None:
    seen: set[ConditionCell] = set()
    primary_cells: set[
        tuple[str, str, str, str, Optional[float]]
    ] = set()
    p_values: list[float] = []
    adjusted_values: list[tuple[int, float]] = []
    for row in _iter_csv_rows(path, collector=collector):
        condition = _condition_cell(
            row,
            path=path,
            collector=collector,
            require_variant=False,
            allow_clean=False,
        )
        if condition is None:
            continue
        if condition in seen:
            collector.add(
                "rq2_mcnemar_duplicate_cell",
                "McNemar condition cell is duplicated",
                path=path,
                row=row.line_number,
            )
        seen.add(condition)
        if (
            _as_bool(row.values.get("primary")) is True
            and condition[3] == "incongruent"
            and condition[6] is False
            and condition[7] == "test"
        ):
            primary_cells.add(
                (
                    condition[0],
                    condition[1],
                    condition[2],
                    condition[3],
                    condition[4],
                )
            )
        n = _required_int(
            row.values,
            "n",
            code="rq2_mcnemar_count_invalid",
            path=path,
            line_number=row.line_number,
            collector=collector,
        )
        clean_correct = _required_int(
            row.values,
            "clean_correct",
            code="rq2_mcnemar_count_invalid",
            path=path,
            line_number=row.line_number,
            collector=collector,
        )
        cued_correct = _required_int(
            row.values,
            "cued_correct",
            code="rq2_mcnemar_count_invalid",
            path=path,
            line_number=row.line_number,
            collector=collector,
        )
        b = _required_int(
            row.values,
            "b_clean_correct_cued_wrong",
            code="rq2_mcnemar_count_invalid",
            path=path,
            line_number=row.line_number,
            collector=collector,
        )
        c = _required_int(
            row.values,
            "c_clean_wrong_cued_correct",
            code="rq2_mcnemar_count_invalid",
            path=path,
            line_number=row.line_number,
            collector=collector,
        )
        statistic = _required_int(
            row.values,
            "statistic",
            code="rq2_mcnemar_count_invalid",
            path=path,
            line_number=row.line_number,
            collector=collector,
        )
        p_value = _required_float(
            row.values,
            "p_value",
            lower=0.0,
            upper=1.0,
            code="rq2_mcnemar_value_invalid",
            path=path,
            line_number=row.line_number,
            collector=collector,
        )
        p_value_holm = _required_float(
            row.values,
            "p_value_holm",
            lower=0.0,
            upper=1.0,
            code="rq2_mcnemar_value_invalid",
            path=path,
            line_number=row.line_number,
            collector=collector,
        )
        if (
            row.values.get("test")
            != "exact_two_sided_mcnemar_clean_vs_cued_correctness"
            or row.values.get("tie_policy") != "strict_three_class"
        ):
            collector.add(
                "rq2_mcnemar_semantics_invalid",
                "McNemar test or tie policy is not the analyzer contract",
                path=path,
                row=row.line_number,
            )
        counts = (n, clean_correct, cued_correct, b, c, statistic)
        if all(value is not None for value in counts):
            assert n is not None
            assert clean_correct is not None
            assert cued_correct is not None
            assert b is not None
            assert c is not None
            assert statistic is not None
            if (
                clean_correct > n
                or cued_correct > n
                or b + c > n
                or clean_correct - cued_correct != b - c
                or statistic != min(b, c)
            ):
                collector.add(
                    "rq2_mcnemar_equation_mismatch",
                    "McNemar counts or statistic are internally inconsistent",
                    path=path,
                    row=row.line_number,
                )
            expected_p = mcnemar_exact(b, c).p_value
            _check_equation(
                p_value,
                expected_p,
                field_name="p_value",
                code="rq2_mcnemar_equation_mismatch",
                path=path,
                line_number=row.line_number,
                collector=collector,
            )
            if n == 0 and _as_bool(row.values.get("primary")) is True:
                availability.add(
                    "rq2_mcnemar_primary_unavailable",
                    "primary McNemar cell has no labeled pairs",
                    path=path,
                    row=row.line_number,
                )
        if p_value is not None and p_value_holm is not None:
            p_values.append(p_value)
            adjusted_values.append((row.line_number, p_value_holm))
    if len(p_values) == len(adjusted_values):
        for (line_number, observed), expected in zip(
            adjusted_values,
            holm_adjust(p_values),
            strict=True,
        ):
            _check_equation(
                observed,
                expected,
                field_name="p_value_holm",
                code="rq2_mcnemar_holm_mismatch",
                path=path,
                line_number=line_number,
                collector=collector,
            )
    expected = {
        (model, ordering, family, direction, dose)
        for model in config.expected_models
        for ordering in ("ab", "ba")
        for family, doses in SOCIAL_DOSES.items()
        for direction in ("congruent", "incongruent")
        for dose in doses
    }
    observed = {
        (cell[0], cell[1], cell[2], cell[3], cell[4])
        for cell in seen
    }
    if observed != expected:
        collector.add(
            "rq2_mcnemar_condition_grid_mismatch",
            f"McNemar grid has {len(observed)} cells; expected {len(expected)}",
            path=path,
        )
    expected_primary = {
        (model, ordering, family, "incongruent", dose)
        for model in config.expected_models
        for ordering in ("ab", "ba")
        for family, doses in SOCIAL_DOSES.items()
        for dose in doses
    }
    if primary_cells != expected_primary:
        collector.add(
            "rq2_mcnemar_primary_cell_set_mismatch",
            (
                f"primary test/non-tie grid has {len(primary_cells)} cells; "
                f"expected {len(expected_primary)}"
            ),
            path=path,
        )


def _optional_interval(
    row: CsvRow,
    low_field: str,
    high_field: str,
    *,
    lower: float,
    upper: float,
    code: str,
    path: Path,
    collector: _IssueCollector,
) -> tuple[float, float] | None:
    low_missing = _is_missing(row.values.get(low_field))
    high_missing = _is_missing(row.values.get(high_field))
    if low_missing and high_missing:
        return None
    if low_missing != high_missing:
        collector.add(
            code,
            f"{low_field} and {high_field} must both be present or both blank",
            path=path,
            row=row.line_number,
        )
        return None
    low = _optional_float(
        row.values,
        low_field,
        lower=lower,
        upper=upper,
        code=code,
        path=path,
        line_number=row.line_number,
        collector=collector,
    )
    high = _optional_float(
        row.values,
        high_field,
        lower=lower,
        upper=upper,
        code=code,
        path=path,
        line_number=row.line_number,
        collector=collector,
    )
    if low is None or high is None:
        return None
    if low > high:
        collector.add(
            code,
            f"{low_field} exceeds {high_field}",
            path=path,
            row=row.line_number,
        )
    return low, high


def _threshold_value(
    row: CsvRow,
    *,
    path: Path,
    collector: _IssueCollector,
) -> float | None:
    raw = str(row.values.get("threshold", "")).strip().lower()
    if raw in {"inf", "+inf", "infinity", "+infinity"}:
        return math.inf
    return _required_float(
        row.values,
        "threshold",
        lower=0.0,
        upper=1.0,
        code="rq2_threshold_value_invalid",
        path=path,
        line_number=row.line_number,
        collector=collector,
    )


def _validate_rq2_threshold_conditions(
    inspection: CsvInspection,
    *,
    config: AnalysisValidationConfig,
    collector: _IssueCollector,
) -> None:
    controls: set[tuple[str, str, str, float, str]] = set()
    primary_rules: set[tuple[str, str, str, float, str]] = set()
    for row in inspection.selected_rows:
        values = row.values
        model = str(values.get("model_name", ""))
        ordering = str(values.get("ordering", ""))
        family = str(values.get("family", ""))
        direction = str(values.get("direction", ""))
        aggregation = str(values.get("aggregation", ""))
        channel = str(values.get("confidence_channel", ""))
        target = _as_finite_float(values.get("target_risk"))
        dose = (
            None
            if _is_missing(values.get("dose"))
            else _as_finite_float(values.get("dose"))
        )
        valid = (
            values.get("routing_split") == "test"
            and channel in CONFIDENCE_CHANNELS
            and target is not None
            and any(_close(target, expected) for expected in config.target_risks)
        )
        if aggregation == "single_ordering":
            valid = valid and ordering in {"ab", "ba"}
            social_suffix = f"_{ordering}"
        elif aggregation == "swap_average":
            valid = (
                valid
                and ordering == "swap_average"
                and channel == "msp"
            )
            social_suffix = ""
        else:
            valid = False
            social_suffix = ""
        if family == "clean":
            valid = (
                valid
                and direction == "clean"
                and dose is None
                and values.get("variant_id") == "clean"
                and values.get("clean_tie") == "all"
            )
            if valid and target is not None:
                controls.add(
                    (model, ordering, channel, target, aggregation)
                )
        elif family in SOCIAL_DOSES:
            clean_tie = _as_bool(values.get("clean_tie"))
            expected_variant = (
                f"{family}_{direction}_{int(dose)}{social_suffix}"
                if dose is not None and dose.is_integer()
                else ""
            )
            valid = (
                valid
                and direction in {"congruent", "incongruent"}
                and dose in SOCIAL_DOSES[family]
                and clean_tie is not None
                and values.get("variant_id") == expected_variant
            )
        else:
            valid = False
        if not valid:
            collector.add(
                "rq2_threshold_condition_invalid",
                "threshold-transfer condition/control columns are inconsistent",
                path=inspection.path,
                row=row.line_number,
            )
        if _as_bool(values.get("primary")) is True and target is not None:
            primary_rules.add(
                (model, ordering, channel, target, aggregation)
            )
    missing_controls = primary_rules - controls
    if missing_controls:
        collector.add(
            "rq2_threshold_clean_control_missing",
            (
                f"{len(missing_controls)} primary threshold rules lack their "
                "clean-direction control row"
            ),
            path=inspection.path,
        )


def _validate_rq2_threshold_headlines(
    inspection: CsvInspection,
    *,
    config: AnalysisValidationConfig,
    collector: _IssueCollector,
    availability: _IssueCollector,
) -> None:
    tested_rows: list[tuple[CsvRow, float, float | None]] = []
    for row in inspection.selected_rows:
        if _as_bool(row.values.get("primary")) is not True:
            continue
        target = _required_float(
            row.values,
            "target_risk",
            lower=0.0,
            upper=1.0,
            code="rq2_threshold_value_invalid",
            path=inspection.path,
            line_number=row.line_number,
            collector=collector,
        )
        calibration_n = _required_int(
            row.values,
            "calibration_n",
            minimum=1,
            code="rq2_threshold_count_invalid",
            path=inspection.path,
            line_number=row.line_number,
            collector=collector,
        )
        test_n = _required_int(
            row.values,
            "test_n",
            minimum=1,
            code="rq2_threshold_count_invalid",
            path=inspection.path,
            line_number=row.line_number,
            collector=collector,
        )
        accepted = _required_int(
            row.values,
            "test_accepted",
            code="rq2_threshold_count_invalid",
            path=inspection.path,
            line_number=row.line_number,
            collector=collector,
        )
        flips = _required_int(
            row.values,
            "test_flips",
            code="rq2_threshold_count_invalid",
            path=inspection.path,
            line_number=row.line_number,
            collector=collector,
        )
        accepted_flips = _required_int(
            row.values,
            "test_accepted_flips",
            code="rq2_threshold_count_invalid",
            path=inspection.path,
            line_number=row.line_number,
            collector=collector,
        )
        calibration_coverage = _required_float(
            row.values,
            "calibration_coverage",
            lower=0.0,
            upper=1.0,
            code="rq2_threshold_value_invalid",
            path=inspection.path,
            line_number=row.line_number,
            collector=collector,
        )
        test_coverage = _required_float(
            row.values,
            "test_coverage",
            lower=0.0,
            upper=1.0,
            code="rq2_threshold_value_invalid",
            path=inspection.path,
            line_number=row.line_number,
            collector=collector,
        )
        calibration_risk = _optional_float(
            row.values,
            "calibration_risk",
            lower=0.0,
            upper=1.0,
            code="rq2_threshold_value_invalid",
            path=inspection.path,
            line_number=row.line_number,
            collector=collector,
        )
        threshold = _threshold_value(
            row,
            path=inspection.path,
            collector=collector,
        )
        if (
            _required_float(
                row.values,
                "confidence",
                lower=0.0,
                upper=1.0,
                code="rq2_threshold_value_invalid",
                path=inspection.path,
                line_number=row.line_number,
                collector=collector,
            )
            != 0.95
        ):
            collector.add(
                "rq2_threshold_semantics_invalid",
                "bootstrap confidence must be 0.95",
                path=inspection.path,
                row=row.line_number,
            )
        for field_name, count, total in (
            ("n_calibration_clusters", calibration_n, calibration_n),
            ("n_test_clusters", test_n, test_n),
        ):
            value = _required_int(
                row.values,
                field_name,
                minimum=1,
                code="rq2_threshold_count_invalid",
                path=inspection.path,
                line_number=row.line_number,
                collector=collector,
            )
            if value is not None and total is not None and value > total:
                collector.add(
                    "rq2_threshold_count_invalid",
                    f"{field_name} exceeds its observation count",
                    path=inspection.path,
                    row=row.line_number,
                )
        n_resamples = _required_int(
            row.values,
            "n_resamples",
            minimum=1,
            code="rq2_threshold_count_invalid",
            path=inspection.path,
            line_number=row.line_number,
            collector=collector,
        )
        if n_resamples is not None and n_resamples != config.bootstrap_resamples:
            collector.add(
                "rq2_threshold_resample_mismatch",
                "n_resamples differs from the analysis spec",
                path=inspection.path,
                row=row.line_number,
            )
        if calibration_n is not None and calibration_coverage is not None:
            if not _close(
                calibration_coverage * calibration_n,
                round(calibration_coverage * calibration_n),
            ):
                collector.add(
                    "rq2_threshold_equation_mismatch",
                    "calibration_coverage*calibration_n is not integral",
                    path=inspection.path,
                    row=row.line_number,
                )
            if _close(calibration_coverage, 0.0):
                if threshold != math.inf or calibration_risk is not None:
                    collector.add(
                        "rq2_threshold_equation_mismatch",
                        "zero calibration coverage requires +inf threshold and blank risk",
                        path=inspection.path,
                        row=row.line_number,
                    )
                availability.add(
                    "rq2_threshold_rule_unavailable",
                    "no positive-coverage clean threshold meets the target risk",
                    path=inspection.path,
                    row=row.line_number,
                )
            elif (
                threshold is None
                or not math.isfinite(threshold)
                or calibration_risk is None
                or (target is not None and calibration_risk > target + 1e-12)
            ):
                collector.add(
                    "rq2_threshold_equation_mismatch",
                    "positive calibration coverage requires a feasible finite rule",
                    path=inspection.path,
                    row=row.line_number,
                )
        if (
            test_n is not None
            and accepted is not None
            and test_coverage is not None
        ):
            if accepted > test_n:
                collector.add(
                    "rq2_threshold_count_invalid",
                    "test_accepted exceeds test_n",
                    path=inspection.path,
                    row=row.line_number,
                )
            _check_equation(
                test_coverage,
                accepted / test_n,
                field_name="test_coverage",
                code="rq2_threshold_equation_mismatch",
                path=inspection.path,
                line_number=row.line_number,
                collector=collector,
            )
        if (
            test_n is not None
            and accepted is not None
            and flips is not None
            and accepted_flips is not None
        ):
            if (
                flips > test_n
                or accepted_flips > accepted
                or accepted_flips > flips
            ):
                collector.add(
                    "rq2_threshold_count_invalid",
                    "flip counts exceed their denominators",
                    path=inspection.path,
                    row=row.line_number,
                )
        realized_risk = _optional_float(
            row.values,
            "test_realized_risk",
            lower=0.0,
            upper=1.0,
            code="rq2_threshold_value_invalid",
            path=inspection.path,
            line_number=row.line_number,
            collector=collector,
        )
        inflation_target = _optional_float(
            row.values,
            "test_risk_inflation_vs_target",
            lower=-1.0,
            upper=1.0,
            code="rq2_threshold_value_invalid",
            path=inspection.path,
            line_number=row.line_number,
            collector=collector,
        )
        inflation_clean = _optional_float(
            row.values,
            "test_risk_inflation_vs_clean_calibration",
            lower=-1.0,
            upper=1.0,
            code="rq2_threshold_value_invalid",
            path=inspection.path,
            line_number=row.line_number,
            collector=collector,
        )
        if accepted == 0:
            if any(
                value is not None
                for value in (realized_risk, inflation_target, inflation_clean)
            ):
                collector.add(
                    "rq2_threshold_equation_mismatch",
                    "zero accepted examples require blank point risk and inflations",
                    path=inspection.path,
                    row=row.line_number,
                )
            availability.add(
                "rq2_threshold_primary_unavailable",
                "transferred threshold accepts zero test examples",
                path=inspection.path,
                row=row.line_number,
            )
        elif accepted is not None:
            if realized_risk is None or target is None:
                collector.add(
                    "rq2_threshold_equation_mismatch",
                    "accepted examples require a finite realized risk",
                    path=inspection.path,
                    row=row.line_number,
                )
            else:
                if not _close(
                    realized_risk * accepted,
                    round(realized_risk * accepted),
                ):
                    collector.add(
                        "rq2_threshold_equation_mismatch",
                        "test_realized_risk*test_accepted is not integral",
                        path=inspection.path,
                        row=row.line_number,
                    )
                if inflation_target is None:
                    collector.add(
                        "rq2_threshold_equation_mismatch",
                        "accepted examples require target-risk inflation",
                        path=inspection.path,
                        row=row.line_number,
                    )
                _check_equation(
                    inflation_target,
                    realized_risk - target,
                    field_name="test_risk_inflation_vs_target",
                    code="rq2_threshold_equation_mismatch",
                    path=inspection.path,
                    line_number=row.line_number,
                    collector=collector,
                )
                if calibration_risk is None:
                    if inflation_clean is not None:
                        collector.add(
                            "rq2_threshold_equation_mismatch",
                            "clean-risk inflation requires calibration_risk",
                            path=inspection.path,
                            row=row.line_number,
                        )
                else:
                    if inflation_clean is None:
                        collector.add(
                            "rq2_threshold_equation_mismatch",
                            "accepted examples require clean-risk inflation",
                            path=inspection.path,
                            row=row.line_number,
                        )
                    _check_equation(
                        inflation_clean,
                        realized_risk - calibration_risk,
                        field_name="test_risk_inflation_vs_clean_calibration",
                        code="rq2_threshold_equation_mismatch",
                        path=inspection.path,
                        line_number=row.line_number,
                        collector=collector,
                    )
        flip_fraction = _optional_float(
            row.values,
            "test_accepted_flip_fraction",
            lower=0.0,
            upper=1.0,
            code="rq2_threshold_value_invalid",
            path=inspection.path,
            line_number=row.line_number,
            collector=collector,
        )
        if flips == 0:
            if accepted_flips not in {None, 0} or flip_fraction is not None:
                collector.add(
                    "rq2_threshold_equation_mismatch",
                    "zero flips require zero accepted_flips and blank fraction",
                    path=inspection.path,
                    row=row.line_number,
                )
            availability.add(
                "rq2_threshold_flip_fraction_unavailable",
                "no flips are available for the accepted-flip fraction",
                path=inspection.path,
                row=row.line_number,
            )
        elif flips is not None and accepted_flips is not None:
            if flip_fraction is None:
                collector.add(
                    "rq2_threshold_equation_mismatch",
                    "observed flips require an accepted-flip fraction",
                    path=inspection.path,
                    row=row.line_number,
                )
            _check_equation(
                flip_fraction,
                accepted_flips / flips,
                field_name="test_accepted_flip_fraction",
                code="rq2_threshold_equation_mismatch",
                path=inspection.path,
                line_number=row.line_number,
                collector=collector,
            )
        intervals = {
            "realized": _optional_interval(
                row,
                "realized_risk_ci_low",
                "realized_risk_ci_high",
                lower=0.0,
                upper=1.0,
                code="rq2_threshold_interval_invalid",
                path=inspection.path,
                collector=collector,
            ),
            "target": _optional_interval(
                row,
                "risk_inflation_vs_target_ci_low",
                "risk_inflation_vs_target_ci_high",
                lower=-(target or 0.0),
                upper=1.0 - (target or 0.0),
                code="rq2_threshold_interval_invalid",
                path=inspection.path,
                collector=collector,
            ),
            "clean": _optional_interval(
                row,
                "risk_inflation_vs_clean_calibration_ci_low",
                "risk_inflation_vs_clean_calibration_ci_high",
                lower=-1.0,
                upper=1.0,
                code="rq2_threshold_interval_invalid",
                path=inspection.path,
                collector=collector,
            ),
            "flips": _optional_interval(
                row,
                "accepted_flip_fraction_ci_low",
                "accepted_flip_fraction_ci_high",
                lower=0.0,
                upper=1.0,
                code="rq2_threshold_interval_invalid",
                path=inspection.path,
                collector=collector,
            ),
        }
        if flips == 0 and intervals["flips"] is not None:
            collector.add(
                "rq2_threshold_equation_mismatch",
                "zero flips require blank accepted-flip interval",
                path=inspection.path,
                row=row.line_number,
            )
        if (
            target is not None
            and intervals["realized"] is not None
            and intervals["target"] is not None
        ):
            realized_interval = intervals["realized"]
            target_interval = intervals["target"]
            if (
                not _close(target_interval[0], realized_interval[0] - target)
                or not _close(
                    target_interval[1],
                    realized_interval[1] - target,
                )
            ):
                collector.add(
                    "rq2_threshold_interval_identity_mismatch",
                    (
                        "risk-inflation-vs-target CI must equal the realized-"
                        "risk CI shifted by target_risk"
                    ),
                    path=inspection.path,
                    row=row.line_number,
                )
        raw_p = _optional_float(
            row.values,
            "risk_inflation_vs_target_p_value_one_sided",
            lower=0.0,
            upper=1.0,
            code="rq2_threshold_value_invalid",
            path=inspection.path,
            line_number=row.line_number,
            collector=collector,
        )
        adjusted_p = _optional_float(
            row.values,
            "p_value_holm",
            lower=0.0,
            upper=1.0,
            code="rq2_threshold_value_invalid",
            path=inspection.path,
            line_number=row.line_number,
            collector=collector,
        )
        if (
            raw_p is not None
            and n_resamples is not None
            and not any(
                1
                <= round(raw_p * (valid_draws + 1))
                <= valid_draws + 1
                and _close(
                    raw_p * (valid_draws + 1),
                    round(raw_p * (valid_draws + 1)),
                )
                for valid_draws in range(1, n_resamples + 1)
            )
        ):
            collector.add(
                "rq2_threshold_p_value_invalid",
                (
                    "bootstrap p-value is not attainable from any positive "
                    "number of valid draws up to B"
                ),
                path=inspection.path,
                row=row.line_number,
            )
        if accepted not in {None, 0} and (
            intervals["target"] is None or raw_p is None
        ):
            availability.add(
                "rq2_threshold_inference_unavailable",
                "headline risk-inflation interval or p-value is unavailable",
                path=inspection.path,
                row=row.line_number,
            )
        if flips not in {None, 0} and intervals["flips"] is None:
            availability.add(
                "rq2_threshold_inference_unavailable",
                "headline accepted-flip interval is unavailable",
                path=inspection.path,
                row=row.line_number,
            )
        if raw_p is None:
            if adjusted_p is not None:
                collector.add(
                    "rq2_threshold_holm_mismatch",
                    "adjusted p-value is present without a raw p-value",
                    path=inspection.path,
                    row=row.line_number,
                )
        elif target is not None:
            tested_rows.append((row, raw_p, adjusted_p))
    for (row, _, observed), expected in zip(
        tested_rows,
        holm_adjust([raw for _, raw, _ in tested_rows]),
        strict=True,
    ):
        if observed is None:
            availability.add(
                "rq2_threshold_inference_unavailable",
                "Holm-adjusted headline p-value is unavailable",
                path=inspection.path,
                row=row.line_number,
            )
        else:
            _check_equation(
                observed,
                expected,
                field_name="p_value_holm",
                code="rq2_threshold_holm_mismatch",
                path=inspection.path,
                line_number=row.line_number,
                collector=collector,
            )


def _validate_modeling(
    inspection: CsvInspection,
    *,
    config: AnalysisValidationConfig,
    collector: _IssueCollector,
    availability: _IssueCollector,
) -> None:
    rows_by_model: dict[str, list[CsvRow]] = {
        model: [] for model in config.expected_models
    }
    for row in inspection.selected_rows:
        rows_by_model.setdefault(str(row.values.get("model_name")), []).append(row)
    expected_n = config.source_pairs * CUED_CONDITIONS_PER_PAIR_MODEL
    for model in config.expected_models:
        rows = rows_by_model.get(model, [])
        unavailable_rows = [
            row
            for row in rows
            if str(row.values.get("status", "")) == "unavailable"
        ]
        if unavailable_rows:
            if len(rows) != 1 or len(unavailable_rows) != 1:
                collector.add(
                    "modeling_unavailable_shape_invalid",
                    (
                        f"{model} must have either one unavailable row or "
                        "eight coefficient rows"
                    ),
                    path=inspection.path,
                )
            unavailable_row = unavailable_rows[0]
            _require_literal(
                unavailable_row,
                field="formula",
                expected=MIXED_EFFECTS_FORMULA,
                path=inspection.path,
                collector=collector,
            )
            availability.add(
                "primary_result_unavailable",
                str(
                    unavailable_row.values.get("message")
                    or f"mixed-effects model unavailable for {model}"
                ),
                path=inspection.path,
                row=unavailable_row.line_number,
            )
            continue
        if len(rows) != len(MIXED_MODEL_TERMS):
            collector.add(
                "modeling_row_count_mismatch",
                (
                    f"{model} has {len(rows)} coefficient rows; expected "
                    f"{len(MIXED_MODEL_TERMS)}"
                ),
                path=inspection.path,
            )
        term_counts = Counter(str(row.values.get("term")) for row in rows)
        duplicate_terms = sorted(
            term for term, count in term_counts.items() if count > 1
        )
        if duplicate_terms:
            collector.add(
                "modeling_duplicate_term",
                f"{model} repeats coefficient terms {duplicate_terms!r}",
                path=inspection.path,
            )
        terms = set(term_counts)
        if terms != MIXED_MODEL_TERMS:
            collector.add(
                "modeling_term_set_mismatch",
                f"{model} terms differ from the preregistered eight coefficients",
                path=inspection.path,
            )
        for row in rows:
            for field, expected in {
                "status": "ok",
                "formula": MIXED_EFFECTS_FORMULA,
                "model_type": "binomial_random_intercept",
                "fit_method": "statsmodels.BinomialBayesMixedGLM.fit_vb",
            }.items():
                _require_literal(row, field=field, expected=expected, path=inspection.path, collector=collector)
            _require_available_bool(
                row,
                field="converged",
                expected=True,
                path=inspection.path,
                availability=availability,
            )
            values = _require_finite_fields(
                row,
                ("estimate", "standard_error", "z_value"),
                path=inspection.path,
                availability=availability,
            )
            if (
                "standard_error" in values
                and values["standard_error"] <= 0.0
            ):
                collector.add(
                    "modeling_standard_error_invalid",
                    "standard_error must be positive",
                    path=inspection.path,
                    row=row.line_number,
                )
            _require_probability_fields(
                row,
                ("p_value", "p_value_holm"),
                path=inspection.path,
                collector=collector,
                availability=availability,
            )
            if _as_int(row.values.get("n")) != expected_n:
                collector.add(
                    "modeling_count_mismatch",
                    f"n is {row.values.get('n')!r}; expected {expected_n}",
                    path=inspection.path,
                    row=row.line_number,
                )


def _asset_output_path(package: AssetPackage, name: str) -> Path:
    if name == "report/paper_results.md":
        return package.report_path
    return package.directory / name


def _validate_assets(
    packages: Sequence[AssetPackage],
    *,
    inspections: Mapping[str, CsvInspection],
    config: AnalysisValidationConfig,
    collector: _IssueCollector,
) -> None:
    if len(packages) != config.required_asset_copies:
        collector.add(
            "asset_copy_count_mismatch",
            f"received {len(packages)} asset packages; expected {config.required_asset_copies}",
        )
    resolved_dirs = [package.directory.resolve() for package in packages]
    if len(set(resolved_dirs)) != len(resolved_dirs):
        collector.add("duplicate_asset_package", "asset package directories must be distinct")
    manifests: list[Mapping[str, Any]] = []
    expected_outputs = set(ASSET_OUTPUT_NAMES)
    expected_inputs = set(ANALYSIS_CSV_NAMES)
    for package in packages:
        manifest_path = package.directory / "paper_assets_manifest.json"
        manifest = _read_json_object(manifest_path, collector=collector)
        if manifest is None:
            continue
        manifests.append(manifest)
        if manifest.get("asset_version") != ASSET_VERSION:
            collector.add(
                "asset_version_mismatch",
                f"manifest declares {manifest.get('asset_version')!r}",
                path=manifest_path,
            )
        deterministic = manifest.get("deterministic")
        expected_deterministic = {
            "stable_input_and_row_sorting": True,
            "timestamps_embedded": False,
            "fixed_pdf_metadata": True,
        }
        if deterministic != expected_deterministic:
            collector.add(
                "asset_determinism_mismatch",
                "deterministic asset flags do not match the paper contract",
                path=manifest_path,
            )
        if manifest.get("missing_inputs") != []:
            collector.add(
                "asset_inputs_missing",
                "paper asset manifest reports missing analysis inputs",
                path=manifest_path,
            )
        inputs = manifest.get("inputs")
        if not isinstance(inputs, Mapping) or set(inputs) != expected_inputs:
            collector.add(
                "asset_input_set_mismatch",
                "paper asset input set is incomplete",
                path=manifest_path,
            )
        else:
            for name in ANALYSIS_CSV_NAMES:
                entry = inputs.get(name)
                inspection = inspections[name]
                if not isinstance(entry, Mapping):
                    collector.add(
                        "asset_input_invalid",
                        f"{name} input metadata must be an object",
                        path=manifest_path,
                    )
                    continue
                if (
                    entry.get("available") is not True
                    or _as_int(entry.get("rows")) != inspection.row_count
                    or entry.get("sha256") != inspection.sha256
                ):
                    collector.add(
                        "asset_input_mismatch",
                        f"{name} metadata differs from the analysis CSV",
                        path=manifest_path,
                    )
        outputs = manifest.get("outputs")
        if not isinstance(outputs, Mapping) or set(outputs) != expected_outputs:
            collector.add(
                "asset_output_set_mismatch",
                "paper asset output set is incomplete",
                path=manifest_path,
            )
            continue
        for name in ASSET_OUTPUT_NAMES:
            output_path = _asset_output_path(package, name)
            if not output_path.is_file():
                collector.add(
                    "asset_output_missing",
                    f"missing paper output {name}",
                    path=output_path,
                )
                continue
            try:
                actual_hash = file_sha256(output_path)
            except OSError as exc:
                collector.add("asset_output_unreadable", str(exc), path=output_path)
                continue
            if outputs.get(name) != actual_hash:
                collector.add(
                    "asset_output_hash_mismatch",
                    f"hash mismatch for {name}",
                    path=manifest_path,
                )
    if len(manifests) > 1:
        reference = canonical_json(manifests[0])
        if any(canonical_json(manifest) != reference for manifest in manifests[1:]):
            collector.add(
                "asset_reproduction_mismatch",
                "asset and reproduction manifests are not byte-equivalent in content",
            )


def validate_analysis_package(
    *,
    analysis_dir: Path,
    stage_a_paths: Sequence[Path],
    stage_b_paths: Sequence[Path],
    asset_packages: Sequence[AssetPackage],
    config: AnalysisValidationConfig,
    max_reported_errors: int = 100,
) -> AnalysisValidationReport:
    """Validate a completed Silent Bias analysis and deterministic paper package."""

    collector = _IssueCollector(max_reported=max_reported_errors)
    availability = _IssueCollector(max_reported=max_reported_errors)
    expected_models = frozenset(config.expected_models)
    stage_a_hashes = _hash_paths(
        stage_a_paths,
        label="Stage A",
        expected_count=len(config.expected_models),
        collector=collector,
    )
    stage_b_hashes = _hash_paths(
        stage_b_paths,
        label="Stage B",
        expected_count=len(config.expected_models),
        collector=collector,
    )
    provenance_path = analysis_dir / "provenance.json"
    manifest_path = analysis_dir / "analysis_manifest.json"
    provenance = _read_json_object(provenance_path, collector=collector)
    manifest = _read_json_object(manifest_path, collector=collector)
    spec_hash, input_hash_payload = _validate_spec(
        provenance,
        config=config,
        stage_a_hashes=stage_a_hashes,
        stage_b_hashes=stage_b_hashes,
        collector=collector,
        path=provenance_path,
    )
    inspections = {
        name: _inspect_csv(
            analysis_dir / name,
            name=name,
            expected_models=expected_models,
            expected_spec_hash=spec_hash,
            expected_input_hashes=input_hash_payload,
            collector=collector,
        )
        for name in ANALYSIS_CSV_NAMES
    }
    provenance_hash = (
        file_sha256(provenance_path) if provenance_path.is_file() else None
    )
    _validate_analysis_manifest(
        manifest,
        analysis_dir=analysis_dir,
        inspections=inspections,
        provenance_hash=provenance_hash,
        spec_hash=spec_hash,
        config=config,
        collector=collector,
    )
    _validate_paired_grid(
        analysis_dir / "paired_shifts.csv",
        config=config,
        expected_spec_hash=spec_hash,
        expected_input_hashes=input_hash_payload,
        collector=collector,
    )
    _validate_primary_selectors(inspections, collector=collector)
    _validate_primary_outputs(
        inspections,
        config=config,
        collector=collector,
        availability=availability,
    )
    expected_rq2_conditions = _expected_rq2_conditions_from_paired(
        analysis_dir / "paired_shifts.csv",
        collector=collector,
    )
    calibration_cells = _validate_rq2_calibration(
        analysis_dir / "rq2_calibration.csv",
        config=config,
        expected_conditions=expected_rq2_conditions,
        collector=collector,
        availability=availability,
    )
    _validate_rq2_reliability(
        analysis_dir / "rq2_reliability.csv",
        calibration=calibration_cells,
        expected_conditions=expected_rq2_conditions,
        config=config,
        collector=collector,
    )
    _validate_rq2_risk_coverage(
        analysis_dir / "rq2_risk_coverage.csv",
        calibration=calibration_cells,
        expected_conditions=expected_rq2_conditions,
        collector=collector,
    )
    _validate_rq2_mcnemar(
        analysis_dir / "rq2_mcnemar.csv",
        config=config,
        collector=collector,
        availability=availability,
    )
    _validate_rq2_threshold_conditions(
        inspections["rq2_threshold_transfer.csv"],
        config=config,
        collector=collector,
    )
    _validate_rq2_threshold_headlines(
        inspections["rq2_threshold_transfer.csv"],
        config=config,
        collector=collector,
        availability=availability,
    )
    _validate_modeling(
        inspections["rq3_modeling.csv"],
        config=config,
        collector=collector,
        availability=availability,
    )
    _validate_assets(
        asset_packages,
        inspections=inspections,
        config=config,
        collector=collector,
    )
    integrity_passed = collector.count == 0
    primary_available = availability.count == 0
    return AnalysisValidationReport(
        passed=(
            integrity_passed
            and (primary_available or not config.require_primary_available)
        ),
        integrity_passed=integrity_passed,
        primary_available=primary_available,
        require_primary_available=config.require_primary_available,
        analysis_version=ANALYSIS_VERSION,
        expected_models=config.expected_models,
        source_pairs=config.source_pairs,
        expected_paired_records=config.expected_paired_records,
        csv_row_counts={
            name: inspection.row_count for name, inspection in inspections.items()
        },
        asset_packages_checked=len(asset_packages),
        error_count=collector.count,
        errors=collector.issues,
        errors_truncated=collector.truncated,
        availability_warning_count=availability.count,
        availability_warnings=availability.issues,
        availability_warnings_truncated=availability.truncated,
    )


def _canonical_model_name(value: str) -> str:
    try:
        return get_model_profile(value).hf_model_name
    except KeyError:
        return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed semantic validation for a completed Silent Bias "
            "analysis and deterministic paper-assets package."
        )
    )
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--stage-a", type=Path, nargs="+", required=True)
    parser.add_argument("--stage-b", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--asset-package",
        action="append",
        nargs=2,
        metavar=("ASSET_DIR", "REPORT_PATH"),
        required=True,
        help="Asset directory and its external paper_results.md path; repeat twice.",
    )
    parser.add_argument(
        "--expected-model",
        action="append",
        required=True,
        help="Expected registry alias or canonical model name; repeat per model.",
    )
    parser.add_argument("--source-pairs", type=int, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=2_000)
    parser.add_argument("--trend-permutations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ece-bins", type=int, default=10)
    parser.add_argument("--target-risk", type=float, nargs="+", default=[0.10, 0.20])
    parser.add_argument("--required-asset-copies", type=int, default=2)
    parser.add_argument(
        "--require-primary-available",
        action="store_true",
        help=(
            "Fail when a structurally valid package has unavailable primary "
            "estimates; by default these are reported as warnings."
        ),
    )
    parser.add_argument("--max-reported-errors", type=int, default=100)
    parser.add_argument("--report-path", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = AnalysisValidationConfig(
            expected_models=tuple(
                _canonical_model_name(value) for value in args.expected_model
            ),
            source_pairs=args.source_pairs,
            bootstrap_resamples=args.bootstrap_resamples,
            trend_permutations=args.trend_permutations,
            seed=args.seed,
            ece_bins=args.ece_bins,
            target_risks=tuple(args.target_risk),
            required_asset_copies=args.required_asset_copies,
            require_primary_available=args.require_primary_available,
        )
        report = validate_analysis_package(
            analysis_dir=args.analysis_dir,
            stage_a_paths=args.stage_a,
            stage_b_paths=args.stage_b,
            asset_packages=tuple(
                AssetPackage(Path(directory), Path(report_path))
                for directory, report_path in args.asset_package
            ),
            config=config,
            max_reported_errors=args.max_reported_errors,
        )
        payload = report.to_dict()
    except (OSError, TypeError, ValueError) as exc:
        payload = {
            "passed": False,
            "integrity_passed": False,
            "primary_available": False,
            "require_primary_available": bool(
                getattr(args, "require_primary_available", False)
            ),
            "analysis_version": ANALYSIS_VERSION,
            "error_count": 1,
            "errors": [
                {
                    "code": "validation_setup_failed",
                    "message": str(exc),
                }
            ],
            "errors_truncated": False,
            "availability_warning_count": 0,
            "availability_warnings": [],
            "availability_warnings_truncated": False,
        }
    serialized = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True)
    print(serialized)
    if args.report_path is not None:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(f"{serialized}\n", encoding="utf-8")
    return 0 if payload["passed"] else 1


__all__ = [
    "ANALYSIS_CSV_NAMES",
    "ANALYSIS_VERSION",
    "ASSET_OUTPUT_NAMES",
    "ASSET_VERSION",
    "AnalysisValidationConfig",
    "AnalysisValidationReport",
    "AssetPackage",
    "ValidationIssue",
    "build_parser",
    "main",
    "validate_analysis_package",
]
