from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd


MAX_LABEL_ENTROPY = math.log2(3)
DEFAULT_ROUTING_FRACTIONS = (0.1, 0.25, 0.5)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result):
        return None
    return result


def _pearson(x: pd.Series, y: pd.Series) -> float | None:
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(frame) < 2 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return None
    return float(frame["x"].corr(frame["y"], method="pearson"))


def _spearman(x: pd.Series, y: pd.Series) -> float | None:
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(frame) < 2 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return None
    return float(frame["x"].rank(method="average").corr(frame["y"].rank(method="average")))


def _roc_auc(scores: pd.Series, events: pd.Series) -> float | None:
    frame = pd.DataFrame({"score": scores, "event": events}).dropna()
    frame["event"] = frame["event"].astype(bool)
    positives = int(frame["event"].sum())
    negatives = len(frame) - positives
    if positives == 0 or negatives == 0:
        return None

    ranks = frame["score"].rank(method="average")
    positive_rank_sum = float(ranks[frame["event"]].sum())
    auc = (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
    return float(auc)


def _threshold_for_top_fraction(scores: pd.Series, top_fraction: float) -> float | None:
    valid = scores.dropna()
    if valid.empty:
        return None
    return float(valid.quantile(1.0 - top_fraction, interpolation="lower"))


def _routing_metrics(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    score_column: str,
    event_column: str,
    top_fraction: float,
) -> dict[str, Any]:
    threshold = _threshold_for_top_fraction(train[score_column], top_fraction)
    if threshold is None:
        return {
            "score": score_column,
            "target_top_fraction": top_fraction,
            "threshold": None,
            "test_n": len(test),
            "test_events": int(test[event_column].sum()) if event_column in test else None,
            "test_routed": 0,
            "test_routed_fraction": 0.0,
            "test_event_recall": None,
            "test_precision": None,
        }

    valid_test = test.dropna(subset=[score_column]).copy()
    routed = valid_test[score_column] >= threshold
    events = valid_test[event_column].astype(bool)
    routed_events = int((routed & events).sum())
    total_events = int(events.sum())
    total_routed = int(routed.sum())
    return {
        "score": score_column,
        "target_top_fraction": top_fraction,
        "threshold": threshold,
        "test_n": len(valid_test),
        "test_events": total_events,
        "test_routed": total_routed,
        "test_routed_fraction": total_routed / len(valid_test) if len(valid_test) else None,
        "test_routed_events": routed_events,
        "test_event_recall": routed_events / total_events if total_events else None,
        "test_precision": routed_events / total_routed if total_routed else None,
        "test_auc": _roc_auc(valid_test[score_column], events),
    }


def _add_combined_scores(df: pd.DataFrame, prefix: str) -> None:
    entropy = pd.to_numeric(df[f"{prefix}_entropy"], errors="coerce") / MAX_LABEL_ENTROPY
    verbalized = pd.to_numeric(df[f"{prefix}_verbalized_uncertainty"], errors="coerce")
    consistency = pd.to_numeric(df.get(f"{prefix}_consistency_entropy"), errors="coerce") / MAX_LABEL_ENTROPY
    df[f"{prefix}_normalized_entropy"] = entropy
    df[f"{prefix}_mean_entropy_verbalized"] = pd.concat([entropy, verbalized], axis=1).mean(axis=1)
    df[f"{prefix}_max_entropy_verbalized"] = pd.concat([entropy, verbalized], axis=1).max(axis=1)
    df[f"{prefix}_mean_all_uncertainty"] = pd.concat([entropy, verbalized, consistency], axis=1).mean(axis=1)


def _latest_full_pair_summaries(outputs_dir: Path) -> list[Path]:
    candidates = sorted(outputs_dir.glob("*_full_*/*_pair_summary.jsonl"))
    latest: dict[tuple[str, str], Path] = {}
    for path in candidates:
        summary_path = path.with_name(path.name.replace("_pair_summary.jsonl", "_summary.json"))
        model_name = path.parent.name
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                model_name = summary.get("model_name", model_name)
            except json.JSONDecodeError:
                pass
        bias = path.name.split("_pair_summary.jsonl")[0]
        latest[(bias, model_name)] = path
    return list(latest.values())


def _run_metadata(pair_path: Path) -> dict[str, Any]:
    summary_path = pair_path.with_name(pair_path.name.replace("_pair_summary.jsonl", "_summary.json"))
    if not summary_path.exists():
        return {"model_name": pair_path.parent.name, "run_dir": str(pair_path.parent)}
    return json.loads(summary_path.read_text(encoding="utf-8")) | {"run_dir": str(pair_path.parent)}


def analyze_pair_routing(
    *,
    pair_paths: list[Path],
    routing_fractions: tuple[float, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair_path in pair_paths:
        metadata = _run_metadata(pair_path)
        df = pd.DataFrame(_read_jsonl(pair_path))
        if df.empty or "routing_split" not in df:
            continue

        bias = metadata.get("bias_name") or pair_path.name.split("_pair_summary.jsonl")[0]
        model_name = metadata.get("model_name")
        df["routing_split"] = df["routing_split"].fillna("unknown")

        tasks: list[tuple[str, str, str, list[str]]] = []
        if bias == "position":
            df["event"] = df["position_flip"].astype(bool)
            _add_combined_scores(df, "original")
            tasks.append(
                (
                    "position_flip_from_original",
                    "event",
                    "original",
                    [
                        "original_entropy",
                        "original_verbalized_uncertainty",
                        "original_mean_entropy_verbalized",
                        "original_max_entropy_verbalized",
                        "original_mean_all_uncertainty",
                    ],
                )
            )
        elif bias in {"authority", "bandwagon"}:
            event_column = f"{bias}_incongruent_shift_from_control"
            df["event"] = df[event_column].astype(bool)
            _add_combined_scores(df, "control")
            _add_combined_scores(df, f"{bias}_incongruent")
            tasks.extend(
                [
                    (
                        f"{bias}_shift_from_control_score_control",
                        "event",
                        "control",
                        [
                            "control_entropy",
                            "control_verbalized_uncertainty",
                            "control_mean_entropy_verbalized",
                            "control_max_entropy_verbalized",
                            "control_mean_all_uncertainty",
                        ],
                    ),
                    (
                        f"{bias}_shift_from_control_score_incongruent",
                        "event",
                        f"{bias}_incongruent",
                        [
                            f"{bias}_incongruent_entropy",
                            f"{bias}_incongruent_verbalized_uncertainty",
                            f"{bias}_incongruent_mean_entropy_verbalized",
                            f"{bias}_incongruent_max_entropy_verbalized",
                            f"{bias}_incongruent_mean_all_uncertainty",
                        ],
                    ),
                ]
            )
        else:
            continue

        train = df[df["routing_split"] == "calibration"].copy()
        test = df[df["routing_split"] == "test"].copy()
        if train.empty or test.empty:
            continue

        for task_name, event_column, score_context, score_columns in tasks:
            for score_column in score_columns:
                for fraction in routing_fractions:
                    metrics = _routing_metrics(
                        train=train,
                        test=test,
                        score_column=score_column,
                        event_column=event_column,
                        top_fraction=fraction,
                    )
                    rows.append(
                        {
                            "run_dir": str(pair_path.parent),
                            "model_name": model_name,
                            "bias_name": bias,
                            "routing_task": task_name,
                            "score_context": score_context,
                            **metrics,
                        }
                    )
    return rows


def analyze_correlations(uncertainty_paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in uncertainty_paths:
        df = pd.DataFrame(_read_jsonl(path))
        if df.empty:
            continue
        for variant_id, group in df.groupby("variant_id", dropna=False):
            entropy = pd.to_numeric(group["entropy"], errors="coerce")
            verbalized = pd.to_numeric(group["verbalized_uncertainty"], errors="coerce")
            rows.append(
                {
                    "run_dir": str(path.parent),
                    "model_name": group["model_name"].iloc[0],
                    "bias_name": group["bias_name"].iloc[0],
                    "variant_id": variant_id,
                    "n": int(len(group)),
                    "entropy_mean": _safe_float(entropy.mean()),
                    "verbalized_uncertainty_mean": _safe_float(verbalized.mean()),
                    "verbalized_uncertainty_unique": int(verbalized.nunique(dropna=True)),
                    "pearson_entropy_verbalized_uncertainty": _pearson(entropy, verbalized),
                    "spearman_entropy_verbalized_uncertainty": _spearman(entropy, verbalized),
                    "pearson_entropy_consistency_entropy": _pearson(
                        entropy,
                        pd.to_numeric(group["consistency_vote_entropy"], errors="coerce"),
                    ),
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze uncertainty-based routing on completed full-dataset bias runs.",
    )
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/routing_analysis"))
    parser.add_argument(
        "--routing-fractions",
        type=float,
        nargs="+",
        default=list(DEFAULT_ROUTING_FRACTIONS),
        help="Target top fractions to route, calibrated on the calibration split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pair_paths = _latest_full_pair_summaries(args.outputs_dir)
    uncertainty_paths = sorted(args.outputs_dir.glob("*_full_*/*_uncertainty_scores.jsonl"))
    fractions = tuple(args.routing_fractions)

    routing_rows = analyze_pair_routing(pair_paths=pair_paths, routing_fractions=fractions)
    correlation_rows = analyze_correlations(uncertainty_paths)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    routing_df = pd.DataFrame(routing_rows)
    correlation_df = pd.DataFrame(correlation_rows)
    routing_df.to_csv(args.output_dir / "routing_summary.csv", index=False)
    correlation_df.to_csv(args.output_dir / "uncertainty_correlations.csv", index=False)
    _write_json(args.output_dir / "routing_summary.json", routing_rows)
    _write_json(args.output_dir / "uncertainty_correlations.json", correlation_rows)

    print("Analyzed pair summaries:", len(pair_paths))
    print("Analyzed uncertainty files:", len(uncertainty_paths))
    print("Saved:", args.output_dir / "routing_summary.csv")
    print("Saved:", args.output_dir / "uncertainty_correlations.csv")

    if not routing_df.empty:
        best = (
            routing_df[routing_df["target_top_fraction"].eq(0.5)]
            .sort_values(["routing_task", "model_name", "test_event_recall"], ascending=[True, True, False])
            .groupby(["model_name", "bias_name", "routing_task"], as_index=False)
            .head(1)
        )
        print("\nBest routing scores at target top fraction 0.5:")
        columns = [
            "model_name",
            "bias_name",
            "routing_task",
            "score",
            "test_events",
            "test_routed_fraction",
            "test_event_recall",
            "test_precision",
            "test_auc",
        ]
        print(best[columns].to_string(index=False))

    if not correlation_df.empty:
        print("\nEntropy vs verbalized uncertainty correlations by run/variant:")
        columns = [
            "model_name",
            "bias_name",
            "variant_id",
            "n",
            "verbalized_uncertainty_unique",
            "pearson_entropy_verbalized_uncertainty",
            "spearman_entropy_verbalized_uncertainty",
        ]
        print(correlation_df[columns].to_string(index=False))


if __name__ == "__main__":
    main()
