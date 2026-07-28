from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

from biases.paths import output_path
from biases.stats import bootstrap_bca_ci, wilson_ci


MAX_LABEL_ENTROPY = math.log2(3)
DEFAULT_ROUTING_FRACTIONS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
DEFAULT_ESCALATION_PAIRS = (
    ("Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-27B"),
    ("Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-27B"),
)


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


def _accuracy_ci(correct: pd.Series) -> dict[str, Any]:
    values = [bool(value) for value in correct.dropna().tolist()]
    if not values:
        return {"accuracy": None, "accuracy_ci_low": None, "accuracy_ci_high": None}
    successes = sum(values)
    interval = wilson_ci(successes, len(values))
    return {
        "accuracy": interval.estimate,
        "accuracy_ci_low": interval.low,
        "accuracy_ci_high": interval.high,
    }


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
    if score_column not in train or score_column not in test:
        return {
            "score": score_column,
            "target_top_fraction": top_fraction,
            "threshold": None,
            "test_n": 0,
            "test_events": None,
            "test_routed": 0,
            "test_routed_fraction": 0.0,
            "test_routed_events": 0,
            "test_event_recall": None,
            "test_event_recall_ci_low": None,
            "test_event_recall_ci_high": None,
            "test_precision": None,
            "test_precision_ci_low": None,
            "test_precision_ci_high": None,
            "test_auc": None,
        }
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
        "test_event_recall_ci_low": wilson_ci(routed_events, total_events).low if total_events else None,
        "test_event_recall_ci_high": wilson_ci(routed_events, total_events).high if total_events else None,
        "test_precision": routed_events / total_routed if total_routed else None,
        "test_precision_ci_low": wilson_ci(routed_events, total_routed).low if total_routed else None,
        "test_precision_ci_high": wilson_ci(routed_events, total_routed).high if total_routed else None,
        "test_auc": _roc_auc(valid_test[score_column], events),
    }


def _add_combined_scores(df: pd.DataFrame, prefix: str) -> None:
    entropy = _numeric_column(df, f"{prefix}_entropy") / MAX_LABEL_ENTROPY
    consistency = _numeric_column(df, f"{prefix}_consistency_entropy") / MAX_LABEL_ENTROPY
    msp = _numeric_column(df, f"{prefix}_msp")
    margin = _numeric_column(df, f"{prefix}_margin")
    df[f"{prefix}_normalized_entropy"] = entropy
    df[f"{prefix}_msp_uncertainty"] = 1.0 - msp
    df[f"{prefix}_margin_uncertainty"] = 1.0 - margin
    df[f"{prefix}_normalized_consistency_entropy"] = consistency
    df[f"{prefix}_mean_output_uncertainty"] = pd.concat(
        [entropy, 1.0 - msp, 1.0 - margin, consistency],
        axis=1,
    ).mean(axis=1)


def _numeric_column(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df:
        return pd.Series([math.nan] * len(df), index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce")


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
                        "original_msp_uncertainty",
                        "original_margin_uncertainty",
                        "original_normalized_consistency_entropy",
                        "original_mean_output_uncertainty",
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
                            "control_msp_uncertainty",
                            "control_margin_uncertainty",
                            "control_normalized_consistency_entropy",
                            "control_mean_output_uncertainty",
                        ],
                    ),
                    (
                        f"{bias}_shift_from_control_score_incongruent",
                        "event",
                        f"{bias}_incongruent",
                        [
                            f"{bias}_incongruent_entropy",
                            f"{bias}_incongruent_msp_uncertainty",
                            f"{bias}_incongruent_margin_uncertainty",
                            f"{bias}_incongruent_normalized_consistency_entropy",
                            f"{bias}_incongruent_mean_output_uncertainty",
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


def _is_correct(verdict: Any, human_winner: Any) -> bool:
    return str(verdict) == str(human_winner)


def _position_frames_by_model(pair_paths: list[Path]) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for pair_path in pair_paths:
        metadata = _run_metadata(pair_path)
        bias = metadata.get("bias_name") or pair_path.name.split("_pair_summary.jsonl")[0]
        if bias != "position":
            continue
        df = pd.DataFrame(_read_jsonl(pair_path))
        if df.empty:
            continue
        model_name = str(metadata.get("model_name"))
        df = df.copy()
        df["model_name"] = model_name
        df["join_id"] = range(len(df))
        frames[model_name] = df
    return frames


def _trapezoid_auc(x_values: list[float], y_values: list[float]) -> float | None:
    if len(x_values) < 2:
        return None
    pairs = sorted(zip(x_values, y_values, strict=True))
    total = 0.0
    for (x0, y0), (x1, y1) in zip(pairs, pairs[1:], strict=False):
        total += (x1 - x0) * (y0 + y1) / 2.0
    return total / (pairs[-1][0] - pairs[0][0]) if pairs[-1][0] != pairs[0][0] else None


def analyze_escalation_routing(
    *,
    pair_paths: list[Path],
    routing_fractions: tuple[float, ...],
    model_pairs: tuple[tuple[str, str], ...] = DEFAULT_ESCALATION_PAIRS,
) -> list[dict[str, Any]]:
    frames = _position_frames_by_model(pair_paths)
    rows: list[dict[str, Any]] = []

    for weak_model, strong_model in model_pairs:
        weak = frames.get(weak_model)
        strong = frames.get(strong_model)
        if weak is None or strong is None:
            continue

        merged = weak.merge(
            strong[
                [
                    "join_id",
                    "pair_id",
                    "original_verdict",
                    "human_winner",
                    "routing_split",
                ]
            ].rename(
                columns={
                    "original_verdict": "strong_verdict",
                    "human_winner": "strong_human_winner",
                    "routing_split": "strong_routing_split",
                }
            ),
            on="join_id",
            how="inner",
        )
        merged = merged.rename(columns={"original_verdict": "weak_verdict"})
        merged = merged[merged["human_winner"].eq(merged["strong_human_winner"])].copy()
        merged = merged[merged["routing_split"].eq(merged["strong_routing_split"])].copy()
        if merged.empty:
            continue

        merged["weak_correct"] = [
            _is_correct(verdict, human)
            for verdict, human in zip(merged["weak_verdict"], merged["human_winner"], strict=True)
        ]
        merged["strong_correct"] = [
            _is_correct(verdict, human)
            for verdict, human in zip(merged["strong_verdict"], merged["human_winner"], strict=True)
        ]
        merged["weak_entropy"] = pd.to_numeric(merged["original_entropy"], errors="coerce")

        calibration = merged[merged["routing_split"].eq("calibration")].dropna(subset=["weak_entropy"])
        test = merged[merged["routing_split"].eq("test")].dropna(subset=["weak_entropy"]).copy()
        if calibration.empty or test.empty:
            continue

        all_weak = _accuracy_ci(test["weak_correct"])
        all_strong = _accuracy_ci(test["strong_correct"])
        budget_accuracies: list[float] = []
        budget_values: list[float] = []

        for fraction in routing_fractions:
            threshold = _threshold_for_top_fraction(calibration["weak_entropy"], fraction)
            if threshold is None:
                continue
            routed = test["weak_entropy"] >= threshold
            routed_correct = test["strong_correct"].where(routed, test["weak_correct"])
            routed_values = [bool(value) for value in routed_correct.tolist()]
            routed_ci = wilson_ci(sum(routed_values), len(routed_values))

            random_expected = (
                (1.0 - fraction) * float(test["weak_correct"].mean())
                + fraction * float(test["strong_correct"].mean())
            )
            budget_n = int(round(fraction * len(test)))
            beneficial = int((~test["weak_correct"] & test["strong_correct"]).sum())
            oracle_correct = int(test["weak_correct"].sum()) + min(budget_n, beneficial)
            oracle_accuracy = oracle_correct / len(test)
            boot = bootstrap_bca_ci(
                [1.0 if value else 0.0 for value in routed_values],
                n_resamples=1000,
                seed=17,
            )

            accuracy = routed_ci.estimate
            budget_values.append(fraction)
            budget_accuracies.append(accuracy)
            rows.append(
                {
                    "weak_model": weak_model,
                    "strong_model": strong_model,
                    "score": "weak_original_entropy",
                    "target_top_fraction": fraction,
                    "threshold": threshold,
                    "test_n": len(test),
                    "test_routed": int(routed.sum()),
                    "test_routed_fraction": float(routed.mean()),
                    "accuracy": accuracy,
                    "accuracy_ci_low": routed_ci.low,
                    "accuracy_ci_high": routed_ci.high,
                    "accuracy_bootstrap_bca_low": boot.low,
                    "accuracy_bootstrap_bca_high": boot.high,
                    "all_weak_accuracy": all_weak["accuracy"],
                    "all_weak_accuracy_ci_low": all_weak["accuracy_ci_low"],
                    "all_weak_accuracy_ci_high": all_weak["accuracy_ci_high"],
                    "all_strong_accuracy": all_strong["accuracy"],
                    "all_strong_accuracy_ci_low": all_strong["accuracy_ci_low"],
                    "all_strong_accuracy_ci_high": all_strong["accuracy_ci_high"],
                    "random_expected_accuracy": random_expected,
                    "oracle_accuracy": oracle_accuracy,
                }
            )

        area = _trapezoid_auc(budget_values, budget_accuracies)
        if area is not None:
            for row in rows:
                if row["weak_model"] == weak_model and row["strong_model"] == strong_model:
                    row["cost_accuracy_auc"] = area

    return rows


def analyze_correlations(uncertainty_paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in uncertainty_paths:
        df = pd.DataFrame(_read_jsonl(path))
        if df.empty:
            continue
        for variant_id, group in df.groupby("variant_id", dropna=False):
            entropy = pd.to_numeric(group["entropy"], errors="coerce")
            msp_uncertainty = 1.0 - pd.to_numeric(group["msp"], errors="coerce")
            margin_uncertainty = 1.0 - pd.to_numeric(group["margin"], errors="coerce")
            consistency_entropy = pd.to_numeric(group["consistency_vote_entropy"], errors="coerce")
            rows.append(
                {
                    "run_dir": str(path.parent),
                    "model_name": group["model_name"].iloc[0],
                    "bias_name": group["bias_name"].iloc[0],
                    "variant_id": variant_id,
                    "n": int(len(group)),
                    "entropy_mean": _safe_float(entropy.mean()),
                    "msp_uncertainty_mean": _safe_float(msp_uncertainty.mean()),
                    "margin_uncertainty_mean": _safe_float(margin_uncertainty.mean()),
                    "consistency_entropy_mean": _safe_float(consistency_entropy.mean()),
                    "pearson_entropy_msp_uncertainty": _pearson(entropy, msp_uncertainty),
                    "spearman_entropy_msp_uncertainty": _spearman(entropy, msp_uncertainty),
                    "pearson_entropy_margin_uncertainty": _pearson(entropy, margin_uncertainty),
                    "spearman_entropy_margin_uncertainty": _spearman(entropy, margin_uncertainty),
                    "pearson_entropy_consistency_entropy": _pearson(entropy, consistency_entropy),
                    "spearman_entropy_consistency_entropy": _spearman(entropy, consistency_entropy),
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze uncertainty-based routing on completed full-dataset bias runs.",
    )
    parser.add_argument("--outputs-dir", type=Path, default=output_path())
    parser.add_argument("--output-dir", type=Path, default=output_path("routing_analysis"))
    parser.add_argument(
        "--routing-fractions",
        type=float,
        nargs="+",
        default=list(DEFAULT_ROUTING_FRACTIONS),
        help="Target top fractions to route, calibrated on the calibration split.",
    )
    parser.add_argument(
        "--escalation-pair",
        action="append",
        default=[],
        metavar="WEAK=STRONG",
        help="Weak-to-strong model pair for escalation analysis. Can be repeated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pair_paths = _latest_full_pair_summaries(args.outputs_dir)
    uncertainty_paths = sorted(args.outputs_dir.glob("*_full_*/*_uncertainty_scores.jsonl"))
    fractions = tuple(args.routing_fractions)
    escalation_pairs = tuple(
        tuple(item.split("=", 1)) for item in args.escalation_pair
    ) or DEFAULT_ESCALATION_PAIRS

    routing_rows = analyze_pair_routing(pair_paths=pair_paths, routing_fractions=fractions)
    escalation_rows = analyze_escalation_routing(
        pair_paths=pair_paths,
        routing_fractions=fractions,
        model_pairs=escalation_pairs,  # type: ignore[arg-type]
    )
    correlation_rows = analyze_correlations(uncertainty_paths)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    routing_df = pd.DataFrame(routing_rows)
    escalation_df = pd.DataFrame(escalation_rows)
    correlation_df = pd.DataFrame(correlation_rows)
    routing_df.to_csv(args.output_dir / "routing_summary.csv", index=False)
    escalation_df.to_csv(args.output_dir / "escalation_summary.csv", index=False)
    correlation_df.to_csv(args.output_dir / "uncertainty_correlations.csv", index=False)
    _write_json(args.output_dir / "routing_summary.json", routing_rows)
    _write_json(args.output_dir / "escalation_summary.json", escalation_rows)
    _write_json(args.output_dir / "uncertainty_correlations.json", correlation_rows)

    print("Analyzed pair summaries:", len(pair_paths))
    print("Analyzed uncertainty files:", len(uncertainty_paths))
    print("Saved:", args.output_dir / "routing_summary.csv")
    print("Saved:", args.output_dir / "escalation_summary.csv")
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

    if not escalation_df.empty:
        print("\nWeak-to-strong escalation at target top fraction 0.5:")
        columns = [
            "weak_model",
            "strong_model",
            "test_routed_fraction",
            "accuracy",
            "all_weak_accuracy",
            "all_strong_accuracy",
            "random_expected_accuracy",
            "oracle_accuracy",
        ]
        print(escalation_df[escalation_df["target_top_fraction"].eq(0.5)][columns].to_string(index=False))

    if not correlation_df.empty:
        print("\nOutput-uncertainty correlations by run/variant:")
        columns = [
            "model_name",
            "bias_name",
            "variant_id",
            "n",
            "pearson_entropy_msp_uncertainty",
            "spearman_entropy_msp_uncertainty",
            "pearson_entropy_consistency_entropy",
            "spearman_entropy_consistency_entropy",
        ]
        print(correlation_df[columns].to_string(index=False))


if __name__ == "__main__":
    main()
