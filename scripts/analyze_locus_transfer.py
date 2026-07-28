from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from biases.paths import output_path
from biases.stats import roc_auc, wilson_ci


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _threshold_for_top_fraction(scores: pd.Series, top_fraction: float) -> float | None:
    valid = pd.to_numeric(scores, errors="coerce").dropna()
    if valid.empty:
        return None
    return float(valid.quantile(1.0 - top_fraction, interpolation="lower"))


def analyze_transfer(
    df: pd.DataFrame,
    *,
    score_column: str,
    event_column: str,
    top_fraction: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for train_locus in sorted(df["locus"].dropna().unique()):
        train = df[(df["locus"] == train_locus) & (df["routing_split"] == "calibration")]
        threshold = _threshold_for_top_fraction(train[score_column], top_fraction)
        if threshold is None:
            continue
        for test_locus in sorted(df["locus"].dropna().unique()):
            test = df[(df["locus"] == test_locus) & (df["routing_split"] == "test")].copy()
            test[score_column] = pd.to_numeric(test[score_column], errors="coerce")
            test = test.dropna(subset=[score_column, event_column])
            if test.empty:
                continue
            routed = test[score_column] >= threshold
            events = test[event_column].astype(bool)
            routed_events = int((routed & events).sum())
            total_events = int(events.sum())
            recall = wilson_ci(routed_events, total_events) if total_events else None
            auc = roc_auc(events.tolist(), test[score_column].tolist()) if events.nunique() == 2 else None
            rows.append(
                {
                    "train_locus": train_locus,
                    "test_locus": test_locus,
                    "score": score_column,
                    "target_top_fraction": top_fraction,
                    "threshold": threshold,
                    "test_n": len(test),
                    "test_events": total_events,
                    "test_routed_fraction": float(routed.mean()),
                    "event_recall": recall.estimate if recall else None,
                    "event_recall_ci_low": recall.low if recall else None,
                    "event_recall_ci_high": recall.high if recall else None,
                    "auc": auc,
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze cross-locus uncertainty transfer.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=output_path("locus_transfer"))
    parser.add_argument("--score-column", default="entropy")
    parser.add_argument("--event-column", default="shift_from_control")
    parser.add_argument("--top-fraction", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.DataFrame(_read_jsonl(args.input_jsonl))
    rows = analyze_transfer(
        df,
        score_column=args.score_column,
        event_column=args.event_column,
        top_fraction=args.top_fraction,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "locus_transfer.csv", index=False)
    (args.output_dir / "locus_transfer.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("Saved:", args.output_dir / "locus_transfer.csv")


if __name__ == "__main__":
    main()
