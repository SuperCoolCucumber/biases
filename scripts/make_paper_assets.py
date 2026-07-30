from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ASSET_VERSION = "silent-bias-paper-assets-v3"
REQUIRED_ANALYSIS_FILES = (
    "paired_shifts.csv",
    "rq1_silent_shift.csv",
    "rq1_susceptibility.csv",
    "rq2_reliability.csv",
    "rq2_risk_coverage.csv",
    "rq2_threshold_transfer.csv",
    "rq3_dose_response.csv",
    "rq3_uncertainty_trend.csv",
    "rq3_uncertainty_by_dose.csv",
)
OPTIONAL_ANALYSIS_FILES = (
    "rq2_calibration.csv",
    "rq2_mcnemar.csv",
    "rq3_modeling.csv",
)
ALL_ANALYSIS_FILES = REQUIRED_ANALYSIS_FILES + OPTIONAL_ANALYSIS_FILES


class OptionalPaperDependencyError(ImportError):
    pass


@dataclass(frozen=True, slots=True)
class AnalysisInput:
    name: str
    sha256: str | None
    rows: tuple[dict[str, str], ...]
    available: bool


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _read_csv(path: Path) -> tuple[dict[str, str], ...]:
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(dict(row) for row in csv.DictReader(handle))


def load_analysis_inputs(
    analysis_dir: Path,
    *,
    allow_missing: bool = False,
) -> dict[str, AnalysisInput]:
    inputs: dict[str, AnalysisInput] = {}
    missing: list[str] = []
    for name in ALL_ANALYSIS_FILES:
        path = analysis_dir / name
        if not path.is_file():
            if name in REQUIRED_ANALYSIS_FILES:
                missing.append(name)
            inputs[name] = AnalysisInput(
                name=name,
                sha256=None,
                rows=(),
                available=False,
            )
            continue
        inputs[name] = AnalysisInput(
            name=name,
            sha256=_file_sha256(path),
            rows=_read_csv(path),
            available=True,
        )
    if missing and not allow_missing:
        raise FileNotFoundError(
            "Missing required analysis CSVs: " + ", ".join(sorted(missing))
        )
    return inputs


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: Any) -> int | None:
    number = _number(value)
    return None if number is None else int(number)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _false_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return not value
    return str(value).strip().lower() in {"0", "false", "no", "n"}


def _sort_component(value: Any) -> tuple[int, float | str]:
    if value is None or str(value).strip() == "":
        return 2, ""
    number = _number(value)
    if number is not None:
        return 0, number
    return 1, str(value)


def _stable_rows(
    rows: Iterable[Mapping[str, Any]],
    fields: Sequence[str],
) -> list[dict[str, Any]]:
    materialized = [dict(row) for row in rows]
    return sorted(
        materialized,
        key=lambda row: tuple(_sort_component(row.get(field)) for field in fields),
    )


def _format_number(value: Any, digits: int = 3) -> str:
    number = _number(value)
    if number is None:
        return "--"
    if abs(number) >= 1000:
        return f"{number:,.0f}"
    return f"{number:.{digits}f}"


def _format_p(value: Any) -> str:
    number = _number(value)
    if number is None:
        return "--"
    return "<0.001" if number < 0.001 else f"{number:.3f}"


def _format_ci(low: Any, high: Any) -> str:
    if _number(low) is None or _number(high) is None:
        return "--"
    return f"[{_format_number(low)}, {_format_number(high)}]"


def _format_p25(row: Mapping[str, Any]) -> tuple[str, str, str]:
    status = str(row.get("p25_range_status", "")).strip()
    dose_min = _number(row.get("dose_min"))
    dose_max = _number(row.get("dose_max"))
    raw_ci = _format_ci(row.get("p25_ci_low"), row.get("p25_ci_high"))
    if status == "below_tested_range" and dose_min is not None:
        return (
            f"below {_format_number(dose_min)}",
            f"extrapolated {raw_ci}" if raw_ci != "--" else "--",
            "below tested range",
        )
    if status == "above_tested_range" and dose_max is not None:
        return (
            f"above {_format_number(dose_max)}",
            f"extrapolated {raw_ci}" if raw_ci != "--" else "--",
            "above tested range",
        )
    if status == "unavailable":
        return "--", "--", "threshold unavailable"
    return (
        _format_number(row.get("p25_dose")),
        raw_ci,
        "descriptive threshold",
    )


def _rq2_decision(row: Mapping[str, Any]) -> tuple[str, str]:
    point = _number(row.get("test_risk_inflation_vs_target"))
    coverage = _number(row.get("test_coverage"))
    if point is None or coverage == 0.0:
        return "unavailable: zero test coverage", "--"
    ci_low = _number(row.get("risk_inflation_vs_target_ci_low"))
    ci_high = _number(row.get("risk_inflation_vs_target_ci_high"))
    if ci_low is None or ci_high is None:
        return "incomplete: clustered CI unavailable", "--"
    formatted_ci = _format_ci(ci_low, ci_high)
    if ci_low > 0.0:
        p_value = _number(row.get("p_value_holm"))
        if p_value is None:
            return "incomplete: adjusted p unavailable", formatted_ci
        if p_value < 0.05:
            return "clean risk guarantee fails", formatted_ci
        return "inconclusive after Holm correction", formatted_ci
    if ci_high <= 0.0:
        return "clean risk guarantee survives", formatted_ci
    return "inconclusive", formatted_ci


def _rq2_adjusted_p(row: Mapping[str, Any]) -> str:
    if (
        _number(row.get("test_risk_inflation_vs_target")) is None
        or _number(row.get("test_coverage")) == 0.0
    ):
        return "--"
    return _format_p(row.get("p_value_holm"))


def _latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def _latex_table(
    *,
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    alignment: str,
    source_name: str,
    source_hash: str | None,
) -> str:
    lines = [
        "% Auto-generated by scripts/make_paper_assets.py.",
        f"% Source: {source_name}",
        f"% Source SHA256: {source_hash or 'unavailable'}",
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        " & ".join(_latex_escape(header) for header in headers) + r" \\",
        r"\midrule",
    ]
    if rows:
        lines.extend(
            " & ".join(_latex_escape(value) for value in row) + r" \\"
            for row in rows
        )
    else:
        lines.append(
            rf"\multicolumn{{{len(headers)}}}{{c}}{{No eligible rows.}} \\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}", ""))
    return "\n".join(lines)


def _decision_ci_positive(row: Mapping[str, Any], *, p_field: str) -> str:
    ci_low = _number(row.get("ci_low"))
    p_value = _number(row.get(p_field))
    if ci_low is None:
        return "incomplete: CI unavailable"
    if p_value is None:
        return "incomplete: adjusted p unavailable"
    return "supports" if ci_low > 0.0 and p_value < 0.05 else "does not support"


def _rq1_silent_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    primary = [
        row
        for row in rows
        if _truthy(row.get("primary"))
        and row.get("routing_split") == "test"
    ]
    if not primary:
        primary = [
            row
            for row in rows
            if row.get("metric") == "signed_cue_mass"
            and row.get("direction") == "incongruent"
            and not _truthy(row.get("clean_tie"))
            and row.get("routing_split") == "test"
        ]
    return _stable_rows(primary, ("model_name", "family", "dose"))


def _rq1_susceptibility_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    primary = [
        row
        for row in rows
        if _truthy(row.get("primary"))
        and row.get("routing_split") == "test"
    ]
    if not primary:
        primary = [
            row
            for row in rows
            if row.get("shift_metric") == "signed_cue_mass"
            and row.get("routing_split") == "test"
        ]
    return _stable_rows(primary, ("model_name", "family"))


def _headline_threshold_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    eligible = [
        dict(row)
        for row in rows
        if row.get("family") not in {"", "clean", None}
        and row.get("direction") == "incongruent"
        and _number(row.get("target_risk")) == 0.1
        and row.get("confidence_channel") == "msp"
        and _false_flag(row.get("clean_tie"))
        and row.get("routing_split") == "test"
    ]
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        grouped[
            (
                str(row.get("model_name", "")),
                str(row.get("family", "")),
                str(row.get("ordering", "")),
                str(row.get("aggregation", "")),
            )
        ].append(row)
    selected = [
        max(group, key=lambda row: _number(row.get("dose")) or -math.inf)
        for group in grouped.values()
    ]
    return _stable_rows(
        selected,
        ("model_name", "family", "aggregation", "ordering", "dose"),
    )


def _rq3_dose_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    primary = [
        row
        for row in rows
        if _truthy(row.get("primary"))
        and row.get("routing_split") == "test"
    ]
    if not primary:
        primary = [
            row
            for row in rows
            if row.get("direction") == "incongruent"
            and not _truthy(row.get("clean_tie"))
            and row.get("routing_split") == "test"
        ]
    return _stable_rows(primary, ("model_name", "family"))


def _rq3_trend_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    primary = [
        row
        for row in rows
        if _truthy(row.get("primary"))
        and row.get("routing_split") == "test"
    ]
    if not primary:
        primary = [
            row
            for row in rows
            if row.get("direction") == "incongruent"
            and row.get("metric") == "cued_entropy"
            and row.get("stable_set") == "pre_first_flip"
            and not _truthy(row.get("clean_tie"))
            and row.get("routing_split") == "test"
        ]
    return _stable_rows(primary, ("model_name", "family"))


def _rq3_uncertainty_dose_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    primary = [
        row
        for row in rows
        if _truthy(row.get("primary"))
        and row.get("routing_split") == "test"
    ]
    return _stable_rows(
        primary
        or [row for row in rows if row.get("routing_split") == "test"],
        ("model_name", "family", "normalized_dose", "dose"),
    )


def build_latex_tables(
    inputs: Mapping[str, AnalysisInput],
) -> dict[str, str]:
    silent_input = inputs["rq1_silent_shift.csv"]
    silent_rows = _rq1_silent_rows(silent_input.rows)
    susceptibility_input = inputs["rq1_susceptibility.csv"]
    susceptibility_rows = _rq1_susceptibility_rows(susceptibility_input.rows)
    threshold_input = inputs["rq2_threshold_transfer.csv"]
    threshold_rows = _headline_threshold_rows(threshold_input.rows)
    dose_input = inputs["rq3_dose_response.csv"]
    dose_rows = _rq3_dose_rows(dose_input.rows)
    trend_input = inputs["rq3_uncertainty_trend.csv"]
    trend_rows = _rq3_trend_rows(trend_input.rows)
    uncertainty_dose_input = inputs["rq3_uncertainty_by_dose.csv"]
    uncertainty_dose_rows = _rq3_uncertainty_dose_rows(
        uncertainty_dose_input.rows
    )

    return {
        "rq1_silent_shift.tex": _latex_table(
            headers=("Model", "Family", "Dose", "n", "Mean shift", "95% CI", "Holm p"),
            rows=[
                (
                    row.get("model_name", ""),
                    row.get("family", ""),
                    row.get("dose", ""),
                    row.get("n", ""),
                    _format_number(row.get("estimate")),
                    _format_ci(row.get("ci_low"), row.get("ci_high")),
                    _format_p(row.get("p_value_holm")),
                )
                for row in silent_rows
            ],
            alignment="lllrrrr",
            source_name=silent_input.name,
            source_hash=silent_input.sha256,
        ),
        "rq1_susceptibility.tex": _latex_table(
            headers=(
                "Model",
                "Family",
                "n",
                "Shift AUROC",
                "Clean AUROC",
                "Difference",
                "95% CI",
            ),
            rows=[
                (
                    row.get("model_name", ""),
                    row.get("family", ""),
                    row.get("n", ""),
                    _format_number(row.get("shift_auc")),
                    _format_number(row.get("clean_baseline_auc")),
                    _format_number(row.get("auc_difference")),
                    _format_ci(
                        row.get("auc_difference_ci_low"),
                        row.get("auc_difference_ci_high"),
                    ),
                )
                for row in susceptibility_rows
            ],
            alignment="lllrrrr",
            source_name=susceptibility_input.name,
            source_hash=susceptibility_input.sha256,
        ),
        "rq2_threshold_transfer.tex": _latex_table(
            headers=(
                "Model",
                "Family",
                "Method",
                "Order",
                "Dose",
                "Coverage",
                "Risk",
                "Risk inflation",
                "Inflation 95% CI",
                "Accepted flips",
                "Accepted-flip 95% CI",
                "Holm p",
            ),
            rows=[
                (
                    row.get("model_name", ""),
                    row.get("family", ""),
                    row.get("aggregation", ""),
                    row.get("ordering", ""),
                    row.get("dose", ""),
                    _format_number(row.get("test_coverage")),
                    _format_number(row.get("test_realized_risk")),
                    _format_number(row.get("test_risk_inflation_vs_target")),
                    _rq2_decision(row)[1],
                    _format_number(row.get("test_accepted_flip_fraction")),
                    _format_ci(
                        row.get("accepted_flip_fraction_ci_low"),
                        row.get("accepted_flip_fraction_ci_high"),
                    ),
                    _rq2_adjusted_p(row),
                )
                for row in threshold_rows
            ],
            alignment="lllllrrrrrrr",
            source_name=threshold_input.name,
            source_hash=threshold_input.sha256,
        ),
        "rq3_dose_response.tex": _latex_table(
            headers=("Model", "Family", "n", "Events", "Slope", "95% CI", "P25", "P25 95% CI"),
            rows=[
                (
                    row.get("model_name", ""),
                    row.get("family", ""),
                    row.get("n", ""),
                    row.get("events", ""),
                    _format_number(row.get("slope")),
                    _format_ci(row.get("slope_ci_low"), row.get("slope_ci_high")),
                    _format_p25(row)[0],
                    _format_p25(row)[1],
                )
                for row in dose_rows
            ],
            alignment="llrrrrrr",
            source_name=dose_input.name,
            source_hash=dose_input.sha256,
        ),
        "rq3_uncertainty_trend.tex": _latex_table(
            headers=(
                "Model",
                "Family",
                "Metric",
                "Stable set",
                "Slope",
                "95% CI",
                "Holm p",
                "Questions",
            ),
            rows=[
                (
                    row.get("model_name", ""),
                    row.get("family", ""),
                    row.get("metric", ""),
                    row.get("stable_set", ""),
                    _format_number(row.get("statistic")),
                    _format_ci(
                        row.get("slope_ci_low"),
                        row.get("slope_ci_high"),
                    ),
                    _format_p(row.get("p_value_holm")),
                    row.get("n_clusters", ""),
                )
                for row in trend_rows
            ],
            alignment="llllrrrr",
            source_name=trend_input.name,
            source_hash=trend_input.sha256,
        ),
        "rq3_uncertainty_by_dose.tex": _latex_table(
            headers=(
                "Model",
                "Family",
                "Dose",
                "n",
                "Entropy",
                "95% CI",
            ),
            rows=[
                (
                    row.get("model_name", ""),
                    row.get("family", ""),
                    row.get("dose", ""),
                    row.get("n", ""),
                    _format_number(row.get("estimate")),
                    _format_ci(row.get("ci_low"), row.get("ci_high")),
                )
                for row in uncertainty_dose_rows
            ],
            alignment="lllrrr",
            source_name=uncertainty_dose_input.name,
            source_hash=uncertainty_dose_input.sha256,
        ),
    }


def _markdown_escape(value: Any) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ")


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    if rows:
        lines.extend(
            "| " + " | ".join(_markdown_escape(value) for value in row) + " |"
            for row in rows
        )
    else:
        lines.append(
            "| "
            + " | ".join(["No eligible evidence rows."] + ["--"] * (len(headers) - 1))
            + " |"
        )
    return lines


def _source_reference(name: str, input_: AnalysisInput) -> str:
    digest = input_.sha256[:12] if input_.sha256 is not None else "unavailable"
    return f"`$ANALYSIS_DIR/{name}` (`{digest}`)"


def _evidence_unit_summary(rows: Sequence[Mapping[str, Any]]) -> str:
    pair_keys_by_model: dict[str, set[str]] = defaultdict(set)
    orderings_by_model: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        model_name = str(row.get("model_name", "")).strip()
        pair_key = str(row.get("pair_key", "")).strip()
        ordering = str(row.get("ordering", "")).strip().lower()
        if model_name and pair_key:
            pair_keys_by_model[model_name].add(pair_key)
        if model_name and ordering:
            orderings_by_model[model_name].add(ordering)

    model_counts: list[tuple[int, int, int]] = []
    for model_name in sorted(pair_keys_by_model):
        ordered_count = len(pair_keys_by_model[model_name])
        ordering_count = len(orderings_by_model[model_name])
        if ordering_count == 0 or ordered_count % ordering_count != 0:
            return f"- Ordered clean pairs: {ordered_count} for {model_name}."
        model_counts.append(
            (ordered_count, ordering_count, ordered_count // ordering_count)
        )

    if not model_counts:
        return "- Ordered clean pairs: unavailable."
    if len(set(model_counts)) != 1:
        details = ", ".join(
            f"{model_name}={len(pair_keys_by_model[model_name])}"
            for model_name in sorted(pair_keys_by_model)
        )
        return f"- Ordered clean pairs by model: {details}."

    ordered_count, ordering_count, source_count = model_counts[0]
    return (
        f"- Ordered clean pairs: {ordered_count} per model across "
        f"{ordering_count} orderings ({source_count} source pairs per model)."
    )


def build_results_digest(inputs: Mapping[str, AnalysisInput]) -> str:
    paired_input = inputs["paired_shifts.csv"]
    silent_input = inputs["rq1_silent_shift.csv"]
    susceptibility_input = inputs["rq1_susceptibility.csv"]
    threshold_input = inputs["rq2_threshold_transfer.csv"]
    dose_input = inputs["rq3_dose_response.csv"]
    trend_input = inputs["rq3_uncertainty_trend.csv"]

    lines = [
        "# Silent Bias Paper Results",
        "",
        "> Auto-generated by `scripts/make_paper_assets.py`. Do not edit values manually.",
        "",
        "## Evidence status",
        "",
    ]
    lines.extend(
        _markdown_table(
            ("Artifact", "Rows", "SHA-256", "Status"),
            [
                (
                    name,
                    len(inputs[name].rows),
                    inputs[name].sha256 or "--",
                    "available" if inputs[name].available else "missing",
                )
                for name in ALL_ANALYSIS_FILES
            ],
        )
    )
    paired_rows = paired_input.rows
    models = sorted(
        {
            str(row.get("model_name", "")).strip()
            for row in paired_rows
            if str(row.get("model_name", "")).strip()
        }
    )
    questions = {
        str(row.get("question_id", "")).strip()
        for row in paired_rows
        if str(row.get("question_id", "")).strip()
    }
    lines.extend(
        (
            "",
            "## Evidence scope",
            "",
            f"- Models: {', '.join(models) if models else 'unavailable'}.",
            _evidence_unit_summary(paired_rows),
            (
                f"- Question clusters: {len(questions)}."
                if questions
                else "- Question clusters: unavailable."
            ),
            f"- Paired clean–cue comparisons: {len(paired_rows)}"
            + ".",
            "- Headline RQ1 and RQ3 estimates use only routing_split=test; "
            "calibration rows are retained only in paired artifacts and RQ2 "
            "threshold selection.",
            "- This digest describes only the supplied artifacts; claims for unrun "
            "models or a larger dataset remain pending.",
        )
    )

    silent_rows = _rq1_silent_rows(silent_input.rows)
    susceptibility_rows = _rq1_susceptibility_rows(susceptibility_input.rows)
    lines.extend(("", "## RQ1 — Silent bias", ""))
    rq1_claims: list[tuple[Any, ...]] = []
    for row in silent_rows:
        rq1_claims.append(
            (
                "Non-flipped incongruent signed cue-mass shift",
                row.get("model_name", ""),
                row.get("family", ""),
                row.get("dose", ""),
                _format_number(row.get("estimate")),
                _format_ci(row.get("ci_low"), row.get("ci_high")),
                _decision_ci_positive(row, p_field="p_value_holm"),
                _source_reference(silent_input.name, silent_input),
            )
        )
    for row in susceptibility_rows:
        ci_low = _number(row.get("auc_difference_ci_low"))
        if ci_low is None:
            decision = "incomplete: CI unavailable"
        else:
            decision = "supports" if ci_low > 0.0 else "does not support"
        rq1_claims.append(
            (
                "Low-dose shift predicts highest-dose flip beyond clean uncertainty",
                row.get("model_name", ""),
                row.get("family", ""),
                f"{row.get('low_dose', '')}→{row.get('high_dose', '')}",
                _format_number(row.get("auc_difference")),
                _format_ci(
                    row.get("auc_difference_ci_low"),
                    row.get("auc_difference_ci_high"),
                ),
                decision,
                _source_reference(susceptibility_input.name, susceptibility_input),
            )
        )
    lines.extend(
        _markdown_table(
            ("Claim", "Model", "Family", "Dose", "Estimate", "95% CI", "Decision", "Source"),
            rq1_claims,
        )
    )

    threshold_rows = _headline_threshold_rows(threshold_input.rows)
    lines.extend(("", "## RQ2 — Selective evaluation under bias", ""))
    rq2_claims: list[tuple[Any, ...]] = []
    for row in threshold_rows:
        decision, displayed_ci = _rq2_decision(row)
        condition = (
            f"{row.get('family', '')} {row.get('direction', '')} "
            f"dose {row.get('dose', '')}; {row.get('aggregation', '')}; "
            f"{row.get('ordering', '')}"
        )
        rq2_claims.append(
            (
                "Realized-risk inflation at clean-calibrated 10% target",
                row.get("model_name", ""),
                condition,
                _format_number(row.get("test_risk_inflation_vs_target")),
                displayed_ci,
                decision,
                _source_reference(threshold_input.name, threshold_input),
            )
        )
        rq2_claims.append(
            (
                "Test coverage at transferred threshold",
                row.get("model_name", ""),
                condition,
                _format_number(row.get("test_coverage")),
                "--",
                "descriptive",
                _source_reference(threshold_input.name, threshold_input),
            )
        )
        rq2_claims.append(
            (
                "Accepted fraction of incongruent flips",
                row.get("model_name", ""),
                condition,
                _format_number(row.get("test_accepted_flip_fraction")),
                _format_ci(
                    row.get("accepted_flip_fraction_ci_low"),
                    row.get("accepted_flip_fraction_ci_high"),
                ),
                "descriptive",
                _source_reference(threshold_input.name, threshold_input),
            )
        )
    lines.extend(
        _markdown_table(
            ("Claim", "Model", "Condition", "Estimate", "95% CI", "Decision", "Source"),
            rq2_claims,
        )
    )

    dose_rows = _rq3_dose_rows(dose_input.rows)
    trend_rows = _rq3_trend_rows(trend_input.rows)
    lines.extend(("", "## RQ3 — Dose–response", ""))
    rq3_claims: list[tuple[Any, ...]] = []
    for row in dose_rows:
        slope_low = _number(row.get("slope_ci_low"))
        p_value = _number(row.get("p_value_holm"))
        if slope_low is None:
            decision = "incomplete: slope CI unavailable"
        elif p_value is None:
            decision = "incomplete: adjusted p unavailable"
        else:
            decision = (
                "positive dose response"
                if slope_low > 0.0 and p_value < 0.05
                else "not established"
            )
        rq3_claims.append(
            (
                "Flip-probability dose slope",
                row.get("model_name", ""),
                row.get("family", ""),
                _format_number(row.get("slope")),
                _format_ci(row.get("slope_ci_low"), row.get("slope_ci_high")),
                decision,
                _source_reference(dose_input.name, dose_input),
            )
        )
        p25_estimate, p25_ci, p25_decision = _format_p25(row)
        rq3_claims.append(
            (
                "Dose at 25% flip probability",
                row.get("model_name", ""),
                row.get("family", ""),
                p25_estimate,
                p25_ci,
                p25_decision,
                _source_reference(dose_input.name, dose_input),
            )
        )
    for row in trend_rows:
        statistic = _number(row.get("statistic"))
        ci_low = _number(row.get("slope_ci_low"))
        ci_high = _number(row.get("slope_ci_high"))
        p_value = _number(row.get("p_value_holm"))
        if ci_low is None or ci_high is None:
            decision = "incomplete: clustered CI unavailable"
        elif p_value is None:
            decision = "incomplete: adjusted p unavailable"
        elif statistic is not None and ci_low > 0.0 and p_value < 0.05:
            decision = "early warning supported"
        else:
            decision = "early warning not established"
        rq3_claims.append(
            (
                "Pre-first-flip uncertainty trend",
                row.get("model_name", ""),
                row.get("family", ""),
                _format_number(statistic),
                _format_ci(ci_low, ci_high),
                decision,
                _source_reference(trend_input.name, trend_input),
            )
        )
    lines.extend(
        _markdown_table(
            ("Claim", "Model", "Family", "Estimate", "95% CI", "Decision", "Source"),
            rq3_claims,
        )
    )
    lines.extend(
        (
            "",
            "## Interpretation guardrail",
            "",
            "A claim remains marked incomplete whenever its preregistered clustered "
            "confidence interval or multiplicity-adjusted test is absent from the "
            "analysis artifact. The digest never upgrades a point estimate into a "
            "paper claim.",
            "",
        )
    )
    return "\n".join(lines)


def _load_pyplot() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        from matplotlib import pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional runtime
        raise OptionalPaperDependencyError(
            "Figure generation requires matplotlib; install the 'analysis' extra."
        ) from exc
    return plt


def _subplot_grid(plt: Any, count: int, *, panel_height: float = 3.2) -> tuple[Any, list[Any]]:
    panel_count = max(1, count)
    columns = 2 if panel_count > 1 else 1
    rows = math.ceil(panel_count / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(6.4 * columns, panel_height * rows),
        squeeze=False,
    )
    flattened = list(axes.flat)
    for axis in flattened[panel_count:]:
        axis.set_visible(False)
    return figure, flattened[:panel_count]


def _save_pdf(figure: Any, path: Path, *, source_hashes: Sequence[str | None]) -> None:
    source = ",".join(value for value in source_hashes if value is not None)
    figure.savefig(
        path,
        format="pdf",
        bbox_inches="tight",
        metadata={
            "Title": path.stem,
            "Author": "Silent Bias pipeline",
            "Subject": f"Source SHA256: {source}",
            "Keywords": ASSET_VERSION,
            "Creator": "scripts/make_paper_assets.py",
            "Producer": "matplotlib",
            "CreationDate": None,
            "ModDate": None,
        },
    )


def _eligible_paired_shifts(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in rows
        if row.get("direction") == "incongruent"
        and not _truthy(row.get("clean_tie"))
        and not _truthy(row.get("flip"))
        and _number(row.get("signed_cue_mass")) is not None
        and row.get("routing_split") == "test"
    ]


def _plot_silent_shift(
    *,
    plt: Any,
    rows: Sequence[Mapping[str, Any]],
    path: Path,
    source_hash: str | None,
) -> None:
    eligible = _eligible_paired_shifts(rows)
    panels = sorted(
        {(str(row.get("model_name", "")), str(row.get("family", ""))) for row in eligible}
    )
    figure, axes = _subplot_grid(plt, len(panels))
    if not panels:
        axes[0].text(0.5, 0.5, "No eligible non-flipped incongruent rows", ha="center")
        axes[0].set_axis_off()
    for axis, (model, family) in zip(axes[: len(panels)], panels, strict=True):
        panel_rows = [
            row
            for row in eligible
            if row.get("model_name") == model and row.get("family") == family
        ]
        doses = sorted(
            {
                float(value)
                for row in panel_rows
                if (value := _number(row.get("dose"))) is not None
            }
        )
        values = [
            [
                float(row["signed_cue_mass"])
                for row in panel_rows
                if _number(row.get("dose")) == dose
            ]
            for dose in doses
        ]
        if values:
            boxes = axis.boxplot(
                values,
                positions=range(len(doses)),
                widths=0.55,
                patch_artist=True,
                showfliers=False,
                tick_labels=[_format_number(dose, digits=0) for dose in doses],
            )
            for box in boxes["boxes"]:
                box.set_facecolor("#4C78A8")
                box.set_alpha(0.65)
        axis.axhline(0.0, color="#333333", linewidth=0.8, linestyle="--")
        axis.set_title(f"{model} — {family}")
        axis.set_xlabel("Cue dose")
        axis.set_ylabel("Signed mass shift toward cue")
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Silent bias among non-flipped incongruent pairs")
    figure.tight_layout()
    _save_pdf(figure, path, source_hashes=(source_hash,))
    plt.close(figure)


def _condition_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        str(row.get("model_name", "")),
        str(row.get("family", "")),
        str(row.get("direction", "")),
        str(row.get("dose", "")),
        str(row.get("ordering", "")),
        str(row.get("variant_id", "")),
        str(row.get("confidence_channel", "")),
        str(row.get("clean_tie", "")),
        str(row.get("routing_split", "")),
    )


def _headline_rq2_figure_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Select the preregistered MSP/test/non-tie channel before grouping."""

    return [
        dict(row)
        for row in rows
        if row.get("confidence_channel") == "msp"
        and _false_flag(row.get("clean_tie"))
        and row.get("routing_split") == "test"
    ]


def _headline_condition_keys(rows: Sequence[Mapping[str, Any]]) -> set[tuple[str, ...]]:
    keys: set[tuple[str, ...]] = {
        _condition_key(row)
        for row in rows
        if row.get("family") == "clean"
    }
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("direction") != "incongruent" or row.get("family") in {"", "clean", None}:
            continue
        grouped[
            (
                str(row.get("model_name", "")),
                str(row.get("family", "")),
                str(row.get("ordering", "")),
            )
        ].append(row)
    for group in grouped.values():
        maximum = max(_number(row.get("dose")) or -math.inf for row in group)
        keys.update(
            _condition_key(row)
            for row in group
            if _number(row.get("dose")) == maximum
        )
    return keys


def _condition_label(key: Sequence[str]) -> str:
    _, family, direction, dose, ordering, _, _, _, _ = key
    if family == "clean":
        return f"clean {ordering}".strip()
    return f"{family} {direction} {dose} {ordering}".strip()


def _plot_reliability(
    *,
    plt: Any,
    rows: Sequence[Mapping[str, Any]],
    path: Path,
    source_hash: str | None,
) -> None:
    eligible = _headline_rq2_figure_rows(rows)
    selected_keys = _headline_condition_keys(eligible)
    selected = [row for row in eligible if _condition_key(row) in selected_keys]
    models = sorted({str(row.get("model_name", "")) for row in selected})
    figure, axes = _subplot_grid(plt, len(models))
    if not models:
        axes[0].text(0.5, 0.5, "No reliability rows", ha="center")
        axes[0].set_axis_off()
    for axis, model in zip(axes[: len(models)], models, strict=True):
        axis.plot((0, 1), (0, 1), color="#333333", linestyle="--", linewidth=0.8)
        groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in selected:
            if row.get("model_name") == model:
                groups[_condition_key(row)].append(row)
        for key in sorted(groups):
            points = _stable_rows(groups[key], ("bin_index",))
            x = [_number(row.get("mean_confidence")) for row in points]
            y = [_number(row.get("accuracy")) for row in points]
            valid = [
                (float(x_value), float(y_value))
                for x_value, y_value in zip(x, y, strict=True)
                if x_value is not None and y_value is not None
            ]
            if valid:
                axis.plot(
                    [point[0] for point in valid],
                    [point[1] for point in valid],
                    marker="o",
                    markersize=3,
                    linewidth=1,
                    label=_condition_label(key),
                )
        axis.set_title(model)
        axis.set_xlabel("Mean confidence")
        axis.set_ylabel("Empirical accuracy")
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=6)
    figure.suptitle("Reliability: clean vs. highest incongruent dose")
    figure.tight_layout()
    _save_pdf(figure, path, source_hashes=(source_hash,))
    plt.close(figure)


def _plot_risk_coverage(
    *,
    plt: Any,
    rows: Sequence[Mapping[str, Any]],
    path: Path,
    source_hash: str | None,
) -> None:
    eligible = _headline_rq2_figure_rows(rows)
    selected_keys = _headline_condition_keys(eligible)
    selected = [row for row in eligible if _condition_key(row) in selected_keys]
    models = sorted({str(row.get("model_name", "")) for row in selected})
    figure, axes = _subplot_grid(plt, len(models))
    if not models:
        axes[0].text(0.5, 0.5, "No risk–coverage rows", ha="center")
        axes[0].set_axis_off()
    for axis, model in zip(axes[: len(models)], models, strict=True):
        groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in selected:
            if row.get("model_name") == model:
                groups[_condition_key(row)].append(row)
        for key in sorted(groups):
            points = _stable_rows(groups[key], ("coverage",))
            valid = [
                (coverage, risk)
                for row in points
                if (coverage := _number(row.get("coverage"))) is not None
                and (risk := _number(row.get("risk"))) is not None
            ]
            if valid:
                axis.plot(
                    [point[0] for point in valid],
                    [point[1] for point in valid],
                    linewidth=1.2,
                    label=_condition_label(key),
                )
        axis.set_title(model)
        axis.set_xlabel("Coverage")
        axis.set_ylabel("Selective risk")
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(bottom=0.0)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=6)
    figure.suptitle("Risk–coverage: clean vs. highest incongruent dose")
    figure.tight_layout()
    _save_pdf(figure, path, source_hashes=(source_hash,))
    plt.close(figure)


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        inverse = math.exp(-min(value, 700.0))
        return 1.0 / (1.0 + inverse)
    exponent = math.exp(max(value, -700.0))
    return exponent / (1.0 + exponent)


def _plot_dose_response(
    *,
    plt: Any,
    fit_rows: Sequence[Mapping[str, Any]],
    paired_rows: Sequence[Mapping[str, Any]],
    path: Path,
    source_hashes: Sequence[str | None],
) -> None:
    fits = _rq3_dose_rows(fit_rows)
    figure, axes = _subplot_grid(plt, len(fits))
    if not fits:
        axes[0].text(0.5, 0.5, "No primary dose-response rows", ha="center")
        axes[0].set_axis_off()
    for axis, row in zip(axes[: len(fits)], fits, strict=True):
        model = str(row.get("model_name", ""))
        family = str(row.get("family", ""))
        panel_rows = [
            item
            for item in paired_rows
            if item.get("model_name") == model
            and item.get("family") == family
            and item.get("direction") == "incongruent"
            and not _truthy(item.get("clean_tie"))
            and _number(item.get("dose")) is not None
            and item.get("routing_split") == "test"
        ]
        observed_doses = sorted(
            {float(item["dose"]) for item in panel_rows if _number(item.get("dose")) is not None}
        )
        if observed_doses:
            minimum, maximum = min(observed_doses), max(observed_doses)
        elif family == "bandwagon":
            minimum, maximum = 55.0, 95.0
        else:
            minimum, maximum = 1.0, 4.0
        intercept = _number(row.get("intercept"))
        slope = _number(row.get("slope"))
        if intercept is not None and slope is not None:
            x_values = [
                minimum + (maximum - minimum) * index / 100.0
                for index in range(101)
            ]
            axis.plot(
                x_values,
                [_sigmoid(intercept + slope * value) for value in x_values],
                color="#E45756",
                linewidth=1.8,
                label="logistic fit",
            )
        empirical: list[tuple[float, float]] = []
        for dose in observed_doses:
            values = [
                _truthy(item.get("flip"))
                for item in panel_rows
                if _number(item.get("dose")) == dose
            ]
            if values:
                empirical.append((dose, sum(values) / len(values)))
        if empirical:
            axis.scatter(
                [item[0] for item in empirical],
                [item[1] for item in empirical],
                color="#4C78A8",
                s=18,
                zorder=3,
                label="empirical",
            )
        p25 = _number(row.get("p25_dose"))
        p25_low = _number(row.get("p25_ci_low"))
        p25_high = _number(row.get("p25_ci_high"))
        p25_status = str(row.get("p25_range_status", "")).strip()
        if (
            p25_status in {"", "within_tested_range"}
            and p25_low is not None
            and p25_high is not None
        ):
            axis.axvspan(p25_low, p25_high, color="#72B7B2", alpha=0.2, label="P25 95% CI")
        if p25_status in {"", "within_tested_range"} and p25 is not None:
            axis.axvline(p25, color="#72B7B2", linewidth=1, linestyle="--")
        elif p25_status == "below_tested_range":
            axis.text(
                0.02,
                0.04,
                "P25 below tested range",
                transform=axis.transAxes,
                fontsize=7,
            )
        elif p25_status == "above_tested_range":
            axis.text(
                0.02,
                0.04,
                "P25 above tested range",
                transform=axis.transAxes,
                fontsize=7,
            )
        axis.set_title(f"{model} — {family}")
        axis.set_xlabel("Cue dose")
        axis.set_ylabel("Flip probability")
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    figure.suptitle("Psychometric flip dose–response")
    figure.tight_layout()
    _save_pdf(figure, path, source_hashes=source_hashes)
    plt.close(figure)


def _plot_uncertainty_dose_response(
    *,
    plt: Any,
    rows: Sequence[Mapping[str, Any]],
    path: Path,
    source_hash: str | None,
) -> None:
    selected = _rq3_uncertainty_dose_rows(rows)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        groups[
            (str(row.get("model_name", "")), str(row.get("family", "")))
        ].append(row)
    ordered_groups = sorted(groups)
    figure, axes = _subplot_grid(plt, len(ordered_groups))
    if not ordered_groups:
        axes[0].text(0.5, 0.5, "No uncertainty dose rows", ha="center")
        axes[0].set_axis_off()
    for axis, group_key in zip(
        axes[: len(ordered_groups)],
        ordered_groups,
        strict=True,
    ):
        points = _stable_rows(groups[group_key], ("normalized_dose", "dose"))
        valid = [
            (dose, estimate, low, high)
            for row in points
            if (dose := _number(row.get("dose"))) is not None
            and (estimate := _number(row.get("estimate"))) is not None
            and (low := _number(row.get("ci_low"))) is not None
            and (high := _number(row.get("ci_high"))) is not None
        ]
        if valid:
            axis.errorbar(
                [item[0] for item in valid],
                [item[1] for item in valid],
                yerr=[
                    [max(0.0, item[1] - item[2]) for item in valid],
                    [max(0.0, item[3] - item[1]) for item in valid],
                ],
                marker="o",
                color="#4C78A8",
                linewidth=1.4,
                capsize=3,
            )
        axis.set_title(f"{group_key[0]} — {group_key[1]}")
        axis.set_xlabel("Cue dose")
        axis.set_ylabel("Entropy before first flip")
        axis.set_ylim(bottom=0.0)
        axis.grid(alpha=0.2)
    figure.suptitle("Pre-first-flip uncertainty dose–response")
    figure.tight_layout()
    _save_pdf(figure, path, source_hashes=(source_hash,))
    plt.close(figure)


def make_figures(
    inputs: Mapping[str, AnalysisInput],
    figure_dir: Path,
) -> tuple[Path, ...]:
    plt = _load_pyplot()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "pdf.compression": 9,
            "savefig.dpi": 150,
        }
    )
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = (
        figure_dir / "rq1_silent_shift_distribution.pdf",
        figure_dir / "rq2_reliability_diagrams.pdf",
        figure_dir / "rq2_risk_coverage.pdf",
        figure_dir / "rq3_dose_response.pdf",
        figure_dir / "rq3_uncertainty_dose_response.pdf",
    )
    _plot_silent_shift(
        plt=plt,
        rows=inputs["paired_shifts.csv"].rows,
        path=paths[0],
        source_hash=inputs["paired_shifts.csv"].sha256,
    )
    _plot_reliability(
        plt=plt,
        rows=inputs["rq2_reliability.csv"].rows,
        path=paths[1],
        source_hash=inputs["rq2_reliability.csv"].sha256,
    )
    _plot_risk_coverage(
        plt=plt,
        rows=inputs["rq2_risk_coverage.csv"].rows,
        path=paths[2],
        source_hash=inputs["rq2_risk_coverage.csv"].sha256,
    )
    _plot_dose_response(
        plt=plt,
        fit_rows=inputs["rq3_dose_response.csv"].rows,
        paired_rows=inputs["paired_shifts.csv"].rows,
        path=paths[3],
        source_hashes=(
            inputs["rq3_dose_response.csv"].sha256,
            inputs["paired_shifts.csv"].sha256,
        ),
    )
    _plot_uncertainty_dose_response(
        plt=plt,
        rows=inputs["rq3_uncertainty_by_dose.csv"].rows,
        path=paths[4],
        source_hash=inputs["rq3_uncertainty_by_dose.csv"].sha256,
    )
    return paths


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _manifest_output_name(path: Path, *, output_dir: Path) -> str:
    try:
        return path.resolve().relative_to(output_dir.resolve()).as_posix()
    except ValueError:
        return f"report/{path.name}"


def generate_paper_assets(
    *,
    analysis_dir: Path,
    output_dir: Path,
    report_path: Path,
    include_figures: bool = True,
    allow_missing: bool = False,
) -> dict[str, Any]:
    inputs = load_analysis_inputs(analysis_dir, allow_missing=allow_missing)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[Path] = []
    for name, text in sorted(build_latex_tables(inputs).items()):
        path = table_dir / name
        _write_text(path, text)
        output_paths.append(path)

    digest = build_results_digest(inputs)
    _write_text(report_path, digest)
    output_paths.append(report_path)

    if include_figures:
        output_paths.extend(make_figures(inputs, output_dir / "figures"))

    manifest = {
        "asset_version": ASSET_VERSION,
        "deterministic": {
            "stable_input_and_row_sorting": True,
            "timestamps_embedded": False,
            "fixed_pdf_metadata": True,
        },
        "inputs": {
            name: {
                "available": inputs[name].available,
                "rows": len(inputs[name].rows),
                "sha256": inputs[name].sha256,
            }
            for name in ALL_ANALYSIS_FILES
        },
        "missing_inputs": [
            name for name in REQUIRED_ANALYSIS_FILES if not inputs[name].available
        ],
        "outputs": {
            _manifest_output_name(path, output_dir=output_dir): _file_sha256(path)
            for path in sorted(output_paths, key=lambda item: item.name)
        },
    }
    manifest_path = output_dir / "paper_assets_manifest.json"
    _write_text(manifest_path, _canonical_json(manifest) + "\n")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate deterministic Silent Bias paper figures, tables, and results digest."
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        required=True,
        help="Directory containing the conventional RQ1-RQ3 analysis CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Paper-assets directory; defaults to ANALYSIS_DIR/paper_assets.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("reports/paper_results.md"),
        help="Path for the generated claims-to-evidence digest.",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Generate tables, manifest, and digest without importing matplotlib.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Emit explicit unavailable rows instead of failing on missing CSVs.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = args.output_dir or args.analysis_dir / "paper_assets"
    manifest = generate_paper_assets(
        analysis_dir=args.analysis_dir,
        output_dir=output_dir,
        report_path=args.report_path,
        include_figures=not args.skip_figures,
        allow_missing=args.allow_missing,
    )
    print(f"Paper assets: {output_dir}")
    print(f"Results digest: {args.report_path}")
    print(f"Outputs hashed: {len(manifest['outputs'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
