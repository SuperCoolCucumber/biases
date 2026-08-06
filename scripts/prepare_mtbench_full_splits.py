from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from biases.dataset_splits import (
    assign_question_disjoint_routing_split,
    assign_routing_split,
    routing_assignment_sha256,
    routing_manifest,
)
from biases.mtbench_io import (
    MTBENCH_HUMAN_REVISION,
    serialize_conversation_columns,
)
from biases.pairing import file_sha256
from biases.paths import configure_artifact_environment, data_path

configure_artifact_environment()

from datasets import load_dataset  # noqa: E402


DEFAULT_DATASET_NAME = "lmsys/mt_bench_human_judgments"
DEFAULT_SPLIT = "human"
DEFAULT_SEED = 42
DEFAULT_CALIBRATION_FRACTION = 0.5
DEFAULT_FULL_OUTPUT_PATH = data_path("processed", "mtbench_full.csv")
DEFAULT_CALIBRATION_OUTPUT_PATH = data_path("processed", "mtbench_full_calibration.csv")
DEFAULT_TEST_OUTPUT_PATH = data_path("processed", "mtbench_full_test.csv")

OUTPUT_COLUMNS = [
    "question_id",
    "model_a",
    "model_b",
    "winner",
    "turn",
    "conversation_a",
    "conversation_b",
    "routing_split",
]


_assign_split = assign_routing_split


def build_full_dataset_with_splits(
    *,
    dataset_name: str,
    dataset_revision: str,
    split: str,
    calibration_fraction: float,
    seed: int,
    routing_unit: str = "row",
) -> pd.DataFrame:
    dataset = load_dataset(dataset_name, revision=dataset_revision)
    df = pd.DataFrame(dataset[split])
    missing = [column for column in OUTPUT_COLUMNS if column != "routing_split" and column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")
    if routing_unit == "row":
        routed = assign_routing_split(
            df,
            calibration_fraction=calibration_fraction,
            seed=seed,
        )
    elif routing_unit == "question":
        routed = assign_question_disjoint_routing_split(
            df,
            calibration_fraction=calibration_fraction,
            seed=seed,
        )
    else:
        raise ValueError("routing_unit must be either 'row' or 'question'")
    return routed[OUTPUT_COLUMNS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create full MT-Bench human-judgment CSVs with calibration/test routing splits."
        ),
    )
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument(
        "--dataset-revision",
        default=MTBENCH_HUMAN_REVISION,
    )
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--routing-unit",
        choices=("row", "question"),
        default="row",
        help=(
            "unit assigned to calibration/test; 'row' preserves the legacy "
            "winner-stratified split and 'question' keeps all rows for one "
            "question together"
        ),
    )
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=DEFAULT_CALIBRATION_FRACTION,
    )
    parser.add_argument("--full-output-path", type=Path, default=DEFAULT_FULL_OUTPUT_PATH)
    parser.add_argument(
        "--calibration-output-path",
        type=Path,
        default=DEFAULT_CALIBRATION_OUTPUT_PATH,
    )
    parser.add_argument("--test-output-path", type=Path, default=DEFAULT_TEST_OUTPUT_PATH)
    parser.add_argument(
        "--routing-manifest-path",
        type=Path,
        default=None,
        help=(
            "optional immutable JSON manifest for the routing "
            "assignment; the command refuses to overwrite an existing file"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.routing_manifest_path is not None:
        immutable_targets = (
            args.routing_manifest_path,
            args.full_output_path,
            args.calibration_output_path,
            args.test_output_path,
        )
        existing = [path for path in immutable_targets if path.exists()]
        if existing:
            raise FileExistsError(
                "refusing to overwrite immutable routing outputs: "
                + ", ".join(str(path) for path in existing)
            )
    df = build_full_dataset_with_splits(
        dataset_name=args.dataset_name,
        dataset_revision=args.dataset_revision,
        split=args.split,
        calibration_fraction=args.calibration_fraction,
        seed=args.seed,
        routing_unit=args.routing_unit,
    )
    df = serialize_conversation_columns(df)

    calibration = df[df["routing_split"] == "calibration"].copy()
    test = df[df["routing_split"] == "test"].copy()

    if args.routing_unit == "question":
        question_counts = (
            df[["question_id", "routing_split"]]
            .drop_duplicates()["routing_split"]
            .value_counts()
        )
        print("Question routing distribution:")
        print(question_counts)
        print(
            "Question routing assignment SHA-256: "
            f"{routing_assignment_sha256(df)}"
        )

    for path, frame in (
        (args.full_output_path, df),
        (args.calibration_output_path, calibration),
        (args.test_output_path, test),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        print(f"Saved: {path}")
        print(f"Rows: {len(frame)}")
        print("Winner distribution:")
        print(frame["winner"].value_counts(dropna=False))
        print("Routing split distribution:")
        print(frame["routing_split"].value_counts(dropna=False))
        print()

    if args.routing_manifest_path is not None:
        manifest = routing_manifest(
            df,
            routing_unit=args.routing_unit,
            seed=args.seed,
            calibration_fraction=args.calibration_fraction,
        )
        manifest["source"] = {
            "dataset_name": args.dataset_name,
            "dataset_revision": args.dataset_revision,
            "dataset_split": args.split,
        }
        manifest["output_sha256"] = {
            "full": file_sha256(args.full_output_path),
            "calibration": file_sha256(args.calibration_output_path),
            "test": file_sha256(args.test_output_path),
        }
        args.routing_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        args.routing_manifest_path.write_text(
            json.dumps(
                manifest,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Saved immutable routing manifest: {args.routing_manifest_path}")


if __name__ == "__main__":
    main()
