from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

from biases.paths import data_path, output_path
from biases.stats import (
    benjamini_hochberg,
    bootstrap_bca_ci,
    mann_whitney_u,
    mcnemar_exact,
    wilson_ci,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def _run_metadata(pair_path: Path) -> dict[str, Any]:
    summary_path = pair_path.with_name(pair_path.name.replace("_pair_summary.jsonl", "_summary.json"))
    if not summary_path.exists():
        return {"model_name": pair_path.parent.name, "run_dir": str(pair_path.parent)}
    return json.loads(summary_path.read_text(encoding="utf-8")) | {"run_dir": str(pair_path.parent)}


def _latest_pair_summaries(outputs_dir: Path) -> list[Path]:
    candidates = sorted(outputs_dir.glob("*_full_*/*_pair_summary.jsonl"))
    latest: dict[tuple[str, str], Path] = {}
    for path in candidates:
        metadata = _run_metadata(path)
        bias = metadata.get("bias_name") or path.name.split("_pair_summary.jsonl")[0]
        latest[(str(bias), str(metadata.get("model_name")))] = path
    return list(latest.values())


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def _ci_dict(prefix: str, successes: int, total: int) -> dict[str, Any]:
    ci = wilson_ci(successes, total)
    return {
        prefix: ci.estimate,
        f"{prefix}_ci_low": ci.low,
        f"{prefix}_ci_high": ci.high,
    }


def _float_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df:
        return pd.Series([math.nan] * len(df), index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce")


def _mean_ci(values: list[float], prefix: str) -> dict[str, Any]:
    values = [float(value) for value in values if not math.isnan(float(value))]
    if not values:
        return {prefix: None, f"{prefix}_ci_low": None, f"{prefix}_ci_high": None}
    ci = bootstrap_bca_ci(values, n_resamples=1000, seed=11)
    return {prefix: ci.estimate, f"{prefix}_ci_low": ci.low, f"{prefix}_ci_high": ci.high}


def _classify_position_direction(a_to_a: int, b_to_b: int) -> str:
    total = a_to_a + b_to_b
    if total == 0:
        return "none"
    if a_to_a / total >= 0.8:
        return "primacy"
    if b_to_b / total >= 0.8:
        return "recency"
    return "mixed"


def analyze_position(pair_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metadata = _run_metadata(pair_path)
    model_name = metadata.get("model_name")
    df = pd.DataFrame(_read_jsonl(pair_path))
    if df.empty:
        return [], {}

    original_response = df["original_response_id"].notna()
    swapped_response = df["swapped_response_id"].notna()
    usable = df[original_response & swapped_response].copy()
    usable["position_flip"] = usable["position_flip"].map(_as_bool)
    flip_count = int(usable["position_flip"].sum())
    ci = wilson_ci(flip_count, len(usable)) if len(usable) else None

    verdict_pairs = Counter(zip(df["original_verdict"], df["swapped_verdict"], strict=True))
    a_to_a = int(verdict_pairs.get(("A", "A"), 0))
    b_to_b = int(verdict_pairs.get(("B", "B"), 0))
    original_non_tie = df[df["original_verdict"].isin(["A", "B"])]
    original_a = int(original_non_tie["original_verdict"].eq("A").sum())
    original_a_ci = wilson_ci(original_a, len(original_non_tie)) if len(original_non_tie) else None

    stable_entropy = _float_series(usable[~usable["position_flip"]], "original_entropy").dropna().tolist()
    flipped_entropy = _float_series(usable[usable["position_flip"]], "original_entropy").dropna().tolist()
    mw = mann_whitney_u(flipped_entropy, stable_entropy) if flipped_entropy and stable_entropy else None

    rows = [
        {
            "run_dir": str(pair_path.parent),
            "model_name": model_name,
            "bias_name": "position",
            "metric": "flip_rate",
            "n": len(usable),
            "events": flip_count,
            **(_ci_dict("estimate", flip_count, len(usable)) if len(usable) else {}),
            "p_value": None,
            "effect_size": None,
        },
        {
            "run_dir": str(pair_path.parent),
            "model_name": model_name,
            "bias_name": "position",
            "metric": "entropy_flipped_vs_stable_mann_whitney",
            "n": len(usable),
            "events": flip_count,
            "estimate": None,
            "estimate_ci_low": None,
            "estimate_ci_high": None,
            "p_value": mw.p_value if mw else None,
            "effect_size": mw.rank_biserial if mw else None,
        },
    ]

    directionality = {
        "run_dir": str(pair_path.parent),
        "model_name": model_name,
        "n_total": len(df),
        "n_usable": len(usable),
        "n_excluded_tie_either_order": int((~(original_response & swapped_response)).sum()),
        "flip_count": flip_count,
        "flip_rate": ci.estimate if ci else None,
        "flip_rate_ci_low": ci.low if ci else None,
        "flip_rate_ci_high": ci.high if ci else None,
        "original_verdict_a_count": original_a,
        "original_verdict_non_tie_count": len(original_non_tie),
        "p_original_verdict_a": original_a_ci.estimate if original_a_ci else None,
        "p_original_verdict_a_ci_low": original_a_ci.low if original_a_ci else None,
        "p_original_verdict_a_ci_high": original_a_ci.high if original_a_ci else None,
        "flip_a_to_a": a_to_a,
        "flip_b_to_b": b_to_b,
        "direction_class": _classify_position_direction(a_to_a, b_to_b),
        **_mean_ci(stable_entropy, "stable_entropy_mean"),
        **_mean_ci(flipped_entropy, "flipped_entropy_mean"),
        "entropy_rank_biserial": mw.rank_biserial if mw else None,
        "entropy_mann_whitney_p": mw.p_value if mw else None,
    }
    return rows, directionality


def _correct(verdict: Any, human: Any) -> bool:
    return str(verdict) == str(human)


def analyze_cue_bias(pair_path: Path, bias: str) -> list[dict[str, Any]]:
    metadata = _run_metadata(pair_path)
    model_name = metadata.get("model_name")
    df = pd.DataFrame(_read_jsonl(pair_path))
    if df.empty:
        return []

    incongruent_verdict = f"{bias}_incongruent_verdict"
    shift_column = f"{bias}_incongruent_shift_from_control"
    cue_follow_column = f"{bias}_incongruent_cue_follow"
    incongruent_entropy = f"{bias}_incongruent_entropy"

    shifts = df[shift_column].map(_as_bool)
    cue_follow = df[cue_follow_column].map(_as_bool)
    non_tie_control = df["control_verdict"].isin(["A", "B"])
    conditional_cue_follow = cue_follow[non_tie_control]

    control_correct = [
        _correct(verdict, human) for verdict, human in zip(df["control_verdict"], df["human_winner"], strict=True)
    ]
    incongruent_correct = [
        _correct(verdict, human) for verdict, human in zip(df[incongruent_verdict], df["human_winner"], strict=True)
    ]
    b = sum(c and not i for c, i in zip(control_correct, incongruent_correct, strict=True))
    c = sum((not c0) and i for c0, i in zip(control_correct, incongruent_correct, strict=True))
    mc = mcnemar_exact(b, c)

    control_entropy = _float_series(df, "control_entropy").dropna().tolist()
    inc_entropy = _float_series(df, incongruent_entropy).dropna().tolist()
    mw = mann_whitney_u(inc_entropy, control_entropy) if inc_entropy and control_entropy else None
    paired_delta = (
        _float_series(df, incongruent_entropy) - _float_series(df, "control_entropy")
    ).dropna().tolist()

    total = len(df)
    shift_count = int(shifts.sum())
    cue_follow_count = int(cue_follow.sum())
    conditional_cue_follow_count = int(conditional_cue_follow.sum())
    rows = [
        {
            "run_dir": str(pair_path.parent),
            "model_name": model_name,
            "bias_name": bias,
            "metric": "incongruent_shift_rate",
            "n": total,
            "events": shift_count,
            **_ci_dict("estimate", shift_count, total),
            "p_value": None,
            "effect_size": None,
        },
        {
            "run_dir": str(pair_path.parent),
            "model_name": model_name,
            "bias_name": bias,
            "metric": "incongruent_cue_follow_rate",
            "n": total,
            "events": cue_follow_count,
            **_ci_dict("estimate", cue_follow_count, total),
            "p_value": None,
            "effect_size": None,
        },
        {
            "run_dir": str(pair_path.parent),
            "model_name": model_name,
            "bias_name": bias,
            "metric": "incongruent_cue_follow_rate_non_tie_control",
            "n": int(non_tie_control.sum()),
            "events": conditional_cue_follow_count,
            **_ci_dict("estimate", conditional_cue_follow_count, int(non_tie_control.sum())),
            "p_value": None,
            "effect_size": None,
        },
        {
            "run_dir": str(pair_path.parent),
            "model_name": model_name,
            "bias_name": bias,
            "metric": "human_agreement_control_vs_incongruent_mcnemar",
            "n": total,
            "events": b + c,
            "estimate": None,
            "estimate_ci_low": None,
            "estimate_ci_high": None,
            "p_value": mc.p_value,
            "effect_size": None,
            "mcnemar_b": b,
            "mcnemar_c": c,
        },
        {
            "run_dir": str(pair_path.parent),
            "model_name": model_name,
            "bias_name": bias,
            "metric": "entropy_incongruent_vs_control_mann_whitney",
            "n": total,
            "events": shift_count,
            "estimate": None,
            "estimate_ci_low": None,
            "estimate_ci_high": None,
            "p_value": mw.p_value if mw else None,
            "effect_size": mw.rank_biserial if mw else None,
        },
        {
            "run_dir": str(pair_path.parent),
            "model_name": model_name,
            "bias_name": bias,
            "metric": "entropy_delta_incongruent_minus_control",
            "n": len(paired_delta),
            "events": shift_count,
            **_mean_ci(paired_delta, "estimate"),
            "p_value": None,
            "effect_size": None,
        },
    ]
    return rows


def analyze_data_balance(csv_path: Path) -> dict[str, Any]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    winner_counts = Counter(row.get("winner") for row in rows)
    slot_counts = {
        "model_a_assignments": len([row for row in rows if row.get("model_a")]),
        "model_b_assignments": len([row for row in rows if row.get("model_b")]),
    }
    generator_slots: Counter[str] = Counter()
    for row in rows:
        if row.get("model_a"):
            generator_slots[f"{row['model_a']}::A"] += 1
        if row.get("model_b"):
            generator_slots[f"{row['model_b']}::B"] += 1
    return {
        "csv_path": str(csv_path),
        "n_rows": len(rows),
        "winner_counts": dict(winner_counts),
        "winner_rates": {key: value / len(rows) for key, value in winner_counts.items()} if rows else {},
        "slot_counts": slot_counts,
        "unique_generators": len({row.get("model_a") for row in rows} | {row.get("model_b") for row in rows}),
        "top_generator_slot_counts": dict(generator_slots.most_common(30)),
    }


def apply_bh(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = [(index, row["p_value"]) for index, row in enumerate(rows) if row.get("p_value") is not None]
    adjusted = benjamini_hochberg([p_value for _, p_value in indexed])
    for (index, _), adjusted_p in zip(indexed, adjusted, strict=True):
        rows[index]["p_value_bh"] = adjusted_p
    for row in rows:
        row.setdefault("p_value_bh", None)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute statistical summaries for completed bias runs.")
    parser.add_argument("--outputs-dir", type=Path, default=output_path())
    parser.add_argument("--data-path", type=Path, default=data_path("processed", "mtbench_full.csv"))
    parser.add_argument("--output-dir", type=Path, default=output_path("statistical_analysis"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pair_paths = _latest_pair_summaries(args.outputs_dir)
    stats_rows: list[dict[str, Any]] = []
    directionality_rows: list[dict[str, Any]] = []

    for pair_path in pair_paths:
        metadata = _run_metadata(pair_path)
        bias = metadata.get("bias_name") or pair_path.name.split("_pair_summary.jsonl")[0]
        if bias == "position":
            rows, directionality = analyze_position(pair_path)
            stats_rows.extend(rows)
            if directionality:
                directionality_rows.append(directionality)
        elif bias in {"authority", "bandwagon"}:
            stats_rows.extend(analyze_cue_bias(pair_path, str(bias)))

    stats_rows = apply_bh(stats_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(stats_rows).to_csv(args.output_dir / "bias_stats_summary.csv", index=False)
    pd.DataFrame(directionality_rows).to_csv(args.output_dir / "position_directionality.csv", index=False)
    _write_json(args.output_dir / "bias_stats_summary.json", stats_rows)
    _write_json(args.output_dir / "position_directionality.json", directionality_rows)

    if args.data_path.exists():
        balance = analyze_data_balance(args.data_path)
        _write_json(args.output_dir / "data_balance.json", balance)

    print("Analyzed pair summaries:", len(pair_paths))
    print("Saved:", args.output_dir / "bias_stats_summary.csv")
    print("Saved:", args.output_dir / "position_directionality.csv")


if __name__ == "__main__":
    main()
