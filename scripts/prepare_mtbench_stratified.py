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

DEFAULT_OUTPUT_PATH = data_path("processed", "mtbench_stratified_198.csv")
DEFAULT_DATASET_NAME = "lmsys/mt_bench_human_judgments"
DEFAULT_SPLIT = "human"
DEFAULT_TARGET_SIZE = 200
DEFAULT_SEED = 42
DEFAULT_CALIBRATION_FRACTION = 0.5

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


def build_stratified_sample(
    *,
    dataset_name: str,
    dataset_revision: str,
    split: str,
    target_size: int,
    seed: int,
    calibration_fraction: float = DEFAULT_CALIBRATION_FRACTION,
) -> pd.DataFrame:
    dataset = load_dataset(dataset_name, revision=dataset_revision)
    df = pd.DataFrame(dataset[split])
    return stratified_sample_from_frame(
        df,
        target_size=target_size,
        seed=seed,
        calibration_fraction=calibration_fraction,
    )


def stratified_sample_from_frame(
    frame: pd.DataFrame,
    *,
    target_size: int,
    seed: int,
    calibration_fraction: float = DEFAULT_CALIBRATION_FRACTION,
) -> pd.DataFrame:
    """Sample evenly by winner and inherited calibration/test routing."""

    indexed = frame.reset_index(drop=True).copy()
    indexed["_source_row_index"] = range(len(indexed))
    routed = assign_routing_split(
        indexed,
        calibration_fraction=calibration_fraction,
        seed=seed,
    )
    strata = sorted(
        {
            (str(winner), str(routing_split))
            for winner, routing_split in zip(
                routed["winner"],
                routed["routing_split"],
                strict=True,
            )
        }
    )
    samples_per_stratum = target_size // len(strata)
    if samples_per_stratum < 1:
        raise ValueError("target_size is too small for the winner/routing strata")

    sampled = (
        routed.groupby(
            ["winner", "routing_split"],
            group_keys=False,
        )[routed.columns]
        .apply(
            lambda group: group.sample(
                n=samples_per_stratum,
                random_state=seed,
            )
        )
        .reset_index(drop=True)
    )

    return sampled.sample(frac=1, random_state=seed).reset_index(drop=True)[OUTPUT_COLUMNS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a stratified MT-Bench human-judgment CSV for bias experiments.",
    )
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument(
        "--dataset-revision",
        default=MTBENCH_HUMAN_REVISION,
    )
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--target-size", type=int, default=DEFAULT_TARGET_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=DEFAULT_CALIBRATION_FRACTION,
    )
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample = build_stratified_sample(
        dataset_name=args.dataset_name,
        dataset_revision=args.dataset_revision,
        split=args.split,
        target_size=args.target_size,
        seed=args.seed,
        calibration_fraction=args.calibration_fraction,
    )
    sample = serialize_conversation_columns(sample)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(args.output_path, index=False)

    print("Saved:", args.output_path)
    print("Rows:", len(sample))
    print("Winner distribution:")
    print(sample["winner"].value_counts())
    print("Unique models:", len(set(sample["model_a"]).union(sample["model_b"])))


if __name__ == "__main__":
    main()
