from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset


DEFAULT_OUTPUT_PATH = Path("data/processed/mtbench_stratified_198.csv")
DEFAULT_DATASET_NAME = "lmsys/mt_bench_human_judgments"
DEFAULT_SPLIT = "human"
DEFAULT_TARGET_SIZE = 200
DEFAULT_SEED = 42

OUTPUT_COLUMNS = [
    "question_id",
    "model_a",
    "model_b",
    "winner",
    "turn",
    "conversation_a",
    "conversation_b",
]


def build_stratified_sample(
    *,
    dataset_name: str,
    split: str,
    target_size: int,
    seed: int,
) -> pd.DataFrame:
    dataset = load_dataset(dataset_name)
    df = pd.DataFrame(dataset[split])

    labels = sorted(df["winner"].dropna().unique())
    samples_per_label = target_size // len(labels)
    if samples_per_label < 1:
        raise ValueError("target_size is too small for the number of winner labels")

    sampled = (
        df.groupby("winner", group_keys=False)[df.columns]
        .apply(lambda group: group.sample(n=samples_per_label, random_state=seed))
        .reset_index(drop=True)
    )

    return sampled.sample(frac=1, random_state=seed).reset_index(drop=True)[OUTPUT_COLUMNS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a stratified MT-Bench human-judgment CSV for bias experiments.",
    )
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--target-size", type=int, default=DEFAULT_TARGET_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample = build_stratified_sample(
        dataset_name=args.dataset_name,
        split=args.split,
        target_size=args.target_size,
        seed=args.seed,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(args.output_path, index=False)

    print("Saved:", args.output_path)
    print("Rows:", len(sample))
    print("Winner distribution:")
    print(sample["winner"].value_counts())
    print("Unique models:", len(set(sample["model_a"]).union(sample["model_b"])))


if __name__ == "__main__":
    main()
