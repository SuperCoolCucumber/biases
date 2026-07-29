from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from biases.dataset_splits import assign_routing_split
from biases.mtbench_io import (
    MTBENCH_HUMAN_REVISION,
    serialize_conversation_columns,
)
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
) -> pd.DataFrame:
    dataset = load_dataset(dataset_name, revision=dataset_revision)
    df = pd.DataFrame(dataset[split])
    missing = [column for column in OUTPUT_COLUMNS if column != "routing_split" and column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")
    return assign_routing_split(
        df,
        calibration_fraction=calibration_fraction,
        seed=seed,
    )[OUTPUT_COLUMNS]


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = build_full_dataset_with_splits(
        dataset_name=args.dataset_name,
        dataset_revision=args.dataset_revision,
        split=args.split,
        calibration_fraction=args.calibration_fraction,
        seed=args.seed,
    )
    df = serialize_conversation_columns(df)

    calibration = df[df["routing_split"] == "calibration"].copy()
    test = df[df["routing_split"] == "test"].copy()

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


if __name__ == "__main__":
    main()
