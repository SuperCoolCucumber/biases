"""Analyze paired lowest-dose judgments after validating archived inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from biases.analysis.lowest_dose import (
    build_lowest_dose_rows,
    paired_human_congruence_rows,
    summarize_lowest_dose,
)
from biases.analysis.records import ConditionRecord, record_from_mapping
from biases.dataset_splits import (
    assign_question_disjoint_routing_split,
    routing_assignment_sha256,
)
from biases.position_bias import load_position_pairs_with_eligibility


ANALYSIS_VERSION = "lowest-dose-exploratory-v1"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(json_safe(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            safe = json_safe(row)
            writer.writerow({
                key: json.dumps(value, sort_keys=True)
                if isinstance(value, (dict, list)) else value
                for key, value in safe.items()
            })


def reconstruct_source(
    source_path: Path, contract: Mapping[str, Any]
) -> tuple[dict[int, dict[str, str]], dict[str, Any]]:
    """Reconstruct routing in memory without changing the original CSV."""
    observed = file_hash(source_path)
    if observed != contract["source_sha256"]:
        raise ValueError("source dataset hash does not match the input contract")
    with source_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        cells = list(reader)
    question_index = header.index("question_id")
    routed = assign_question_disjoint_routing_split(
        pd.DataFrame({"question_id": [row[question_index] for row in cells]}),
        calibration_fraction=float(contract["calibration_fraction"]),
        seed=int(contract["seed"]),
    )
    if "routing_split" not in header:
        header.append("routing_split")
        cells = [row + [""] for row in cells]
    split_index = header.index("routing_split")
    for row, assignment in zip(cells, routed["routing_split"], strict=True):
        row[split_index] = str(assignment)
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(cells)
    routed_hash = hashlib.sha256(buffer.getvalue().encode()).hexdigest()
    assignment_hash = routing_assignment_sha256(routed, routing_unit="question")
    if routed_hash != contract["routed_sha256"]:
        raise ValueError("reconstructed routed CSV hash does not match the contract")
    if assignment_hash != contract["assignment_sha256"]:
        raise ValueError("question-assignment hash does not match the contract")
    pairs, eligibility = load_position_pairs_with_eligibility(source_path)
    by_index = {
        int(pair.original.metadata["source_row_index"]): {
            **dict(zip(header, cells[int(pair.original.metadata["source_row_index"])])),
            "human_winner": str(pair.original.human_winner),
        }
        for pair in pairs
    }
    counts = Counter(row["routing_split"] for row in by_index.values())
    questions = {
        split: sorted({row["question_id"] for row in by_index.values()
                       if row["routing_split"] == split})
        for split in ("calibration", "test")
    }
    if set(questions["calibration"]) & set(questions["test"]):
        raise ValueError("calibration and test questions overlap")
    return by_index, {
        "source_sha256": observed,
        "routed_sha256": routed_hash,
        "assignment_sha256": assignment_hash,
        "raw_rows": eligibility.raw_row_count,
        "eligible_pairs": len(pairs),
        "skipped_rows": eligibility.skipped_row_count,
        "eligible_split_counts": dict(counts),
        "question_ids": questions,
        "human_label_counts": {
            split: dict(Counter(row["human_winner"] for row in by_index.values()
                                if row["routing_split"] == split))
            for split in ("calibration", "test")
        },
    }


def load_stage(
    paths: Sequence[Path], stage: str, models: Mapping[str, Any],
    source: Mapping[int, Mapping[str, str]], routed_hash: str,
    *, source_indices: dict[str, int] | None = None,
) -> tuple[tuple[ConditionRecord, ...], list[dict[str, Any]]]:
    records: list[ConditionRecord] = []
    receipts: list[dict[str, Any]] = []
    observed_models: set[str] = set()
    record_source_indices: dict[str, int] = {}
    for path in paths:
        observed_hash = file_hash(path)
        model_candidates = [
            model for model, specification in models.items()
            if specification[stage]["sha256"] == observed_hash
        ]
        if len(model_candidates) != 1:
            raise ValueError(f"{stage} input hash is unrecognized or ambiguous: {path}")
        model = model_candidates[0]
        if model in observed_models:
            raise ValueError(f"duplicate {stage} input for {model}")
        observed_models.add(model)
        before = len(records)
        clean_keys: set[tuple[int, str]] = set()
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                record = record_from_mapping(raw)
                if record.model_name != model:
                    raise ValueError(f"unexpected model in {path}:{number}")
                index = raw.get("source_row_index")
                if type(index) is not int or index not in source:
                    raise ValueError(f"missing or ineligible source row in {path}:{number}")
                original = source[index]
                if record.ordering not in {"ab", "ba"}:
                    raise ValueError(f"invalid presentation order in {path}:{number}")
                human = original["human_winner"]
                if record.ordering == "ba" and human in {"A", "B"}:
                    human = "B" if human == "A" else "A"
                if record.human_winner != human:
                    raise ValueError(f"human label/order mismatch in {path}:{number}")
                if record.question_id != original["question_id"]:
                    raise ValueError(f"question identity mismatch in {path}:{number}")
                if record.routing_split != original["routing_split"]:
                    raise ValueError(f"routing mismatch in {path}:{number}")
                if raw.get("input_file_hash") != routed_hash:
                    raise ValueError(f"routed input hash mismatch in {path}:{number}")
                if stage == "stage_a":
                    key = index, record.ordering
                    if record.family != "clean" or key in clean_keys:
                        raise ValueError(f"duplicate or non-clean Stage A row in {path}:{number}")
                    clean_keys.add(key)
                elif record.routing_split != "test":
                    raise ValueError(f"non-test Stage B row in {path}:{number}")
                if not record.record_id or record.record_id in record_source_indices:
                    raise ValueError(f"duplicate or empty record ID in {path}:{number}")
                record_source_indices[record.record_id] = index
                records.append(record)
        count = len(records) - before
        if count != int(models[model][stage]["records"]):
            raise ValueError(f"unexpected record count for {model} {stage}: {count}")
        if stage == "stage_a" and clean_keys != {
            (index, order) for index in source for order in ("ab", "ba")
        }:
            raise ValueError(f"incomplete Stage A source grid for {model}")
        receipts.append({"path": str(path.resolve()), "model_name": model,
                         "stage": stage, "sha256": observed_hash, "records": count})
    if observed_models != set(models):
        raise ValueError(f"{stage} model set differs from the input contract")
    if source_indices is not None:
        if set(source_indices) & set(record_source_indices):
            raise ValueError("source-index output map already contains input record IDs")
        source_indices.update(record_source_indices)
    return tuple(records), receipts


def validate_source_linkage(
    clean: Sequence[ConditionRecord], cued: Sequence[ConditionRecord],
    clean_source_indices: Mapping[str, int], cued_source_indices: Mapping[str, int],
) -> None:
    """Require the same physical CSV row across paired records and order/model twins.

    Questions and human winners are shared by many different answer pairs, so
    agreement on those fields alone cannot establish the source-row match.
    Pair identities and source rows must form one common bijection throughout
    both stages, across both models and presentation orders.
    """
    clean_by_id = {record.record_id: record for record in clean}
    if len(clean_by_id) != len(clean):
        raise ValueError("duplicate clean record ID in source linkage")
    for record in cued:
        partner = clean_by_id.get(record.clean_record_id or "")
        if partner is None:
            raise ValueError(f"missing named clean partner for source linkage: {record.record_id!r}")
        clean_index = clean_source_indices.get(partner.record_id)
        cued_index = cued_source_indices.get(record.record_id)
        if type(clean_index) is not int or type(cued_index) is not int:
            raise ValueError(f"missing source index for named clean/cued pair: {record.record_id!r}")
        if clean_index != cued_index:
            raise ValueError(
                f"source-row mismatch for named clean/cued pair {record.record_id!r}: "
                f"clean={clean_index}, cued={cued_index}"
            )
    identity_to_source: dict[str, int] = {}
    source_to_identity: dict[int, str] = {}
    for records, indices in ((clean, clean_source_indices), (cued, cued_source_indices)):
        for record in records:
            index = indices.get(record.record_id)
            identity = record.pair_identity_key
            if type(index) is not int or not identity:
                raise ValueError(f"missing source index or pair identity: {record.record_id!r}")
            if identity in identity_to_source and identity_to_source[identity] != index:
                raise ValueError(
                    f"source identity mismatch across stages, models or answer orders: {identity!r}"
                )
            if index in source_to_identity and source_to_identity[index] != identity:
                raise ValueError(
                    f"source row {index} has inconsistent pair identities across stages, models or answer orders"
                )
            identity_to_source[identity] = index
            source_to_identity[index] = identity


def short_model(name: str) -> str:
    return name.split("/")[-1].replace("-Instruct", "")


def save_figure(figure: Any, output: Path, stem: str) -> None:
    figure.savefig(output / f"{stem}.png", dpi=180, bbox_inches="tight")
    figure.savefig(output / f"{stem}.pdf", bbox_inches="tight",
                   metadata={"CreationDate": None, "ModDate": None})


def plot_distributions(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = sorted({str(row["model_name"]) for row in rows})
    figure, axes = plt.subplots(len(models) * 2, 2,
                                 figsize=(12, 3.0 * len(models) * 2), squeeze=False)
    metrics = [("delta_msp_pp", "Change in maximum confidence (percentage points)"),
               ("delta_human_probability_pp", "Change in human-winner probability (percentage points)")]
    for index, (model, family) in enumerate(
        (model, family) for model in models for family in ("authority", "bandwagon")
    ):
        subset = [row for row in rows if row["model_name"] == model
                  and row["family"] == family and row["event"] == "decisive_reversal"
                  and row["human_winner"] in {"A", "B"}]
        for column, (metric, label) in enumerate(metrics):
            ax = axes[index, column]
            shown = False
            for transition, caption, color in [
                ("correction", "Correction", "#177A78"),
                ("new_error", "New disagreement", "#BA4A32"),
            ]:
                values = [float(row[metric]) for row in subset
                          if row["alignment_transition"] == transition and row[metric] is not None]
                if values:
                    ax.hist(values, bins=np.linspace(-100, 100, 41), histtype="step",
                            weights=np.full(len(values), 100 / len(values)), linewidth=1.7,
                            label=f"{caption} (n={len(values):,})", color=color)
                    shown = True
            ax.axvline(0, color="#888888", linewidth=0.7)
            ax.set_xlim(-100, 100)
            ax.set_xlabel(label)
            ax.set_ylabel("Reversals per bin (%)")
            ax.set_title(f"{short_model(model)} · {family}", loc="left")
            if shown:
                ax.legend(fontsize=8, frameon=False)
            else:
                ax.text(0.5, 0.5, "No eligible reversals", transform=ax.transAxes, ha="center")
    figure.suptitle("Confidence changes among A↔B reversals at the lowest tested cue strength\n"
                     "Both presentation orders included; distributions are conditional on reversal",
                     fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(figure, output, "reversal_confidence_distributions")
    plt.close(figure)


def plot_transitions(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = sorted({str(row["model_name"]) for row in rows})
    figure, axes = plt.subplots(len(models) * 2, 2,
                                 figsize=(12, 3.3 * len(models) * 2), squeeze=False)
    states = ("agrees", "disagrees", "tie")
    labels = ("Human\nwinner", "Other\nanswer", "Tie")
    for index, (model, family) in enumerate(
        (model, family) for model in models for family in ("authority", "bandwagon")
    ):
        for column, direction in enumerate(("congruent", "incongruent")):
            ax = axes[index, column]
            matrix = np.zeros((3, 3), dtype=int)
            for row in rows:
                if (row["model_name"] != model or row["family"] != family
                        or row["human_direction"] != direction):
                    continue
                after = ("tie" if row["cued_verdict"] == "tie" else
                         "agrees" if row["cued_verdict"] == row["human_winner"] else "disagrees")
                matrix[states.index(str(row["clean_state"]))][states.index(after)] += 1
            denominators = matrix.sum(axis=1, keepdims=True)
            rates = np.divide(100 * matrix, denominators,
                              out=np.full((3, 3), np.nan), where=denominators > 0)
            ax.imshow(np.ma.masked_invalid(rates), vmin=0, vmax=100, cmap="Blues")
            for i in range(3):
                for j in range(3):
                    label = "—" if not denominators[i, 0] else f"{rates[i,j]:.1f}%\n(n={matrix[i,j]:,})"
                    ax.text(j, i, label, ha="center", va="center", fontsize=9,
                            color="white" if rates[i,j] > 55 else "#222222")
            ax.set_xticks(range(3), labels)
            ax.set_yticks(range(3), labels)
            ax.set_xlabel("Cued verdict")
            ax.set_ylabel("Clean verdict")
            ax.set_title(f"{short_model(model)} · {family}\nHuman-{direction} cue", fontsize=10)
    figure.suptitle("Lowest-dose verdict transitions relative to human labels\n"
                     "Percentages within each clean-verdict row; human ties excluded", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(figure, output, "verdict_transitions")
    plt.close(figure)


def plot_contrasts(summary: Mapping[str, Any], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = summary["paired_contrasts"]
    selected = [row for row in table if row["ordering"] == "pooled"
                and row["clean_state"] in {"agrees", "disagrees"}]
    keys = sorted({(row["model_name"], row["family"], row["clean_state"]) for row in selected})
    metrics = [("delta_msp_pp", "Maximum-confidence change"),
               ("delta_human_probability_pp", "Human-winner probability change")]
    figure, axes = plt.subplots(1, 2, figsize=(13, max(4, len(keys) * 0.55)), sharey=True)
    labels = [f"{short_model(model)} / {family}\nInitially {'correct' if state == 'agrees' else 'incorrect'}"
              for model, family, state in keys]
    for ax, (metric, title) in zip(axes, metrics, strict=True):
        for index, key in enumerate(keys):
            matches = [row for row in selected if
                       (row["model_name"], row["family"], row["clean_state"]) == key and row["metric"] == metric]
            if len(matches) != 1:
                raise ValueError(f"non-unique paired contrast for {key}/{metric}")
            row = matches[0]
            if row["estimate"] is None:
                ax.annotate("Unavailable", (0, index), xytext=(4, 0), textcoords="offset points")
                continue
            color = "#177A78" if key[2] == "agrees" else "#BA4A32"
            ax.plot(row["estimate"], index, "o", color=color)
            if row["ci_low"] is not None and row["ci_high"] is not None:
                ax.plot([row["ci_low"], row["ci_high"]], [index, index], color=color, linewidth=1.6)
        ax.axvline(0, color="#888888", linewidth=0.7)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Human-congruent minus human-incongruent (percentage points)")
        ax.grid(axis="x", color="#DDDDDD", linewidth=0.5)
    axes[0].set_yticks(range(len(keys)), labels)
    axes[0].invert_yaxis()
    figure.suptitle("Paired cue contrasts on the same examples, including stable judgments\n"
                     "Pointwise 95% question-cluster bootstrap intervals; both answer orders", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(figure, output, "paired_human_congruence_contrasts")
    plt.close(figure)


def format_estimate(row: Mapping[str, Any], *, multiplier: float = 1.0) -> str:
    if row["estimate"] is None:
        return "Unavailable"
    value = float(row["estimate"]) * multiplier
    if row["ci_low"] is None or row["ci_high"] is None:
        return f"{value:.2f}"
    return f"{value:.2f} [{float(row['ci_low']) * multiplier:.2f}, {float(row['ci_high']) * multiplier:.2f}]"


def summary_row(table: Sequence[Mapping[str, Any]], **keys: Any) -> Mapping[str, Any]:
    matches = [row for row in table if all(row.get(key) == value for key, value in keys.items())]
    if len(matches) != 1:
        raise ValueError(f"expected one summary row, found {len(matches)}: {keys}")
    return matches[0]


def write_report(summary: Mapping[str, Any], source: Mapping[str, Any], output: Path) -> None:
    audit = summary["audit"]
    lines = [
        "# Lowest-dose verdict reversals and confidence shifts", "",
        "Exploratory reanalysis of archived judgments. Authority uses endorsement by another user; "
        "bandwagon uses a claimed 55% majority. These strengths are not quantitatively equivalent.", "",
        f"Validated {source['eligible_split_counts']['test']:,} test source rows across "
        f"{len(source['question_ids']['test'])} questions. Primary comparisons require both a binary "
        "human label and a binary clean verdict. Changes involving ties remain in the companion tables.", "",
        "## Input and probability checks", "",
        "| Model | Lowest-dose cued records | Clean partners | Source rows | Questions | Clean / cued verdict-MAP discrepancies |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    models = sorted(row["model_name"] for row in audit["models"])
    for row in audit["models"]:
        lines.append(f"| {short_model(row['model_name'])} | {row['n_cued_records']:,} | "
                     f"{row['n_clean_records']:,} | {row['n_pairs']:,} | {row['n_questions']} | "
                     f"{row['n_clean_verdict_map_mismatch']} / {row['n_cued_verdict_map_mismatch']} |")
    lines += ["", "## Reversal incidence and conditional confidence changes", "",
        "A reversal means A→B or B→A. Rate denominators include every eligible judgment in the named "
        "initial-state group. Shift estimates condition on those judgments actually reversing. "
        "Numbers in brackets are pointwise 95% question-cluster bootstrap intervals.", "",
        "| Model | Cue | Human direction | Clean agreement | Reversals / eligible | Reversal rate (%) | ΔMSP (pp) among reversals | ΔP(human winner) (pp) among reversals |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for model in models:
        for family in ("authority", "bandwagon"):
            for direction in ("congruent", "incongruent"):
                for state in ("agrees", "disagrees"):
                    key = dict(model_name=model, family=family, ordering="pooled",
                               human_direction=direction, clean_state=state)
                    rate = summary_row(summary["transition_rates"], **key, metric="decisive_reversal_rate")
                    msp = summary_row(summary["flip_shifts"], **key, event="decisive_reversal", metric="delta_msp_pp")
                    human = summary_row(summary["flip_shifts"], **key, event="decisive_reversal", metric="delta_human_probability_pp")
                    lines.append(f"| {short_model(model)} | {family} | {direction} | {state} | "
                                 f"{int(rate['numerator'] or 0):,} / {rate['n']:,} | "
                                 f"{format_estimate(rate, multiplier=100)} | {format_estimate(msp)} | {format_estimate(human)} |")
    lines += ["", "## Paired human-congruence contrasts", "",
        "Each contrast compares both cue directions on the exact same clean judgment, including cases "
        "that do not change verdict. Positive values mean a larger change under a human-congruent cue. "
        "Separate initial-agreement strata prevent reinforcement and opposition from being silently mixed.", "",
        "| Model | Cue | Clean agreement | Paired judgments | ΔMSP contrast (pp) | ΔP(human winner) contrast (pp) |",
        "|---|---|---|---:|---:|---:|",
    ]
    for model in models:
        for family in ("authority", "bandwagon"):
            for state in ("agrees", "disagrees"):
                key = dict(model_name=model, family=family, ordering="pooled", clean_state=state)
                msp = summary_row(summary["paired_contrasts"], **key, metric="delta_msp_pp")
                human = summary_row(summary["paired_contrasts"], **key, metric="delta_human_probability_pp")
                lines.append(f"| {short_model(model)} | {family} | {state} | {msp['n']:,} | "
                             f"{format_estimate(msp)} | {format_estimate(human)} |")
    lines += ["", "## Interpretation and companion results", "",
        "- Maximum confidence refers to the current most probable label. It can remain high after a reversal; "
        "human-winner probability and cue-target probability preserve the direction of movement.",
        "- Corrections and errors mean agreement or disagreement with the human reference label, not independently established objective truth.",
        "- The JSON and CSV summaries retain medians, quartiles, initial/final confidence, increased-confidence "
        "fractions, separate presentation orders, changes toward/away from cues, tie strata, fixed clean-confidence "
        "bins, and equal-question-weight sensitivity estimates.",
        "- The same question bootstrap schedule is used for all cells; repeated answer orders, conditions, "
        "and models are not treated as independent observations.",
        "- Intervals are pointwise exploratory intervals, without familywise multiplicity adjustment. "
        "Selecting observed reversals does not identify the causal effect of a confidence shift on reversal.",
        "- Empty subgroups are unavailable. Bootstrap intervals for zero observed events or a single "
        "contributing question can be degenerate; they do not establish zero population risk or certainty. "
        "Inspect contributing-question counts and non-estimable bootstrap draws in the tables.", "",
        "## Files", "",
        "- `paired_lowest_dose.csv`: every selected clean/cued comparison.",
        "- `paired_human_congruence.csv`: same-example differences between the two human-relative cue directions.",
        "- `transition_rates.csv`, `flip_shifts.csv`, `paired_contrasts.csv`, `confidence_bins.csv`: all strata and uncertainty estimates.",
        "- `analysis.json`, `input_audit.json`, `analysis_complete.json`: complete estimates, validation, and file hashes.",
        "- PNG/PDF figures, when enabled: transitions, reversal confidence distributions, and paired cue contrasts.", "",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-a", type=Path, nargs="+", required=True)
    parser.add_argument("--stage-b", type=Path, nargs="+", required=True)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--input-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("output directory already exists; choose a new result directory")
    contract = json.loads(args.input_contract.read_text())
    source, source_audit = reconstruct_source(args.source_csv, contract["dataset"])
    clean_source_indices: dict[str, int] = {}
    cued_source_indices: dict[str, int] = {}
    clean, clean_receipts = load_stage(args.stage_a, "stage_a", contract["models"],
                                       source, source_audit["routed_sha256"],
                                       source_indices=clean_source_indices)
    cued, cued_receipts = load_stage(args.stage_b, "stage_b", contract["models"],
                                    source, source_audit["routed_sha256"],
                                    source_indices=cued_source_indices)
    validate_source_linkage(clean, cued, clean_source_indices, cued_source_indices)
    rows = build_lowest_dose_rows(
        clean, cued,
        expected_test_pairs_per_model=source_audit["eligible_split_counts"]["test"],
        expected_question_ids=source_audit["question_ids"]["test"],
    )
    rows = tuple({**row, "source_row_index": cued_source_indices[row["record_id"]]}
                 for row in rows)
    summary = summarize_lowest_dose(
        rows, bootstrap_draws=args.bootstrap_resamples, seed=args.seed,
        expected_question_ids=source_audit["question_ids"]["test"],
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(args.output_dir / "paired_lowest_dose.csv", rows)
    identity_sources = {record.pair_identity_key: clean_source_indices[record.record_id]
                        for record in clean}
    paired_directions = tuple(
        {**row, "source_row_index": identity_sources[row["pair_identity_key"]]}
        for row in paired_human_congruence_rows(rows)
    )
    write_csv(args.output_dir / "paired_human_congruence.csv", paired_directions)
    write_json(args.output_dir / "analysis.json", summary)
    for name, table in summary.items():
        if isinstance(table, (list, tuple)) and (not table or isinstance(table[0], Mapping)):
            write_csv(args.output_dir / f"{name}.csv", table)
    write_json(args.output_dir / "input_audit.json", {
        "source": source_audit, "inputs": clean_receipts + cued_receipts,
        "input_contract_sha256": file_hash(args.input_contract),
        "source_linkage": {
            "exact_named_clean_partner_source_rows": True,
            "shared_source_identity_across_models_and_orders": True,
            "n_checked_cued_records": len(cued),
        },
    })
    if not args.no_figures:
        plot_transitions(rows, args.output_dir)
        plot_distributions(rows, args.output_dir)
        plot_contrasts(summary, args.output_dir)
    write_report(summary, source_audit, args.output_dir)
    repository_root = Path(__file__).resolve().parents[1]
    code_paths = [Path(__file__), repository_root / "src/biases/analysis/lowest_dose.py"]
    write_json(args.output_dir / "analysis_complete.json", {
        "status": "complete", "analysis_version": ANALYSIS_VERSION,
        "bootstrap_resamples": args.bootstrap_resamples, "seed": args.seed,
        "paired_lowest_dose_records": len(rows),
        "models": sorted(contract["models"]),
        "analysis_spec_sha256": file_hash(repository_root / "docs/lowest_dose_analysis.md"),
        "code_sha256": {path.name: file_hash(path) for path in code_paths},
        "outputs": {path.name: file_hash(path) for path in sorted(args.output_dir.iterdir()) if path.is_file()},
        "interpretation": "Exploratory pointwise intervals; conditional reversal summaries do not identify causal effects of reversal.",
    })
    print(json.dumps({"status": "complete", "output_dir": str(args.output_dir.resolve()), "records": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
