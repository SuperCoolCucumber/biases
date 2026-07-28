from __future__ import annotations

import argparse
from pathlib import Path

from biases.dataset_adapters import (
    adapt_chatbot_arena_rows,
    adapt_llmbar_rows,
    adapt_mtbench_rows,
    adapt_rewardbench_rows,
    write_standard_pairs,
)
from biases.paths import configure_artifact_environment, data_path

configure_artifact_environment()

from datasets import load_dataset  # noqa: E402


ADAPTERS = {
    "chatbot_arena": adapt_chatbot_arena_rows,
    "rewardbench": adapt_rewardbench_rows,
    "llmbar": adapt_llmbar_rows,
    "mtbench": adapt_mtbench_rows,
}

DEFAULT_DATASET_NAMES = {
    "chatbot_arena": "lmsys/chatbot_arena_conversations",
    "rewardbench": "allenai/reward-bench",
    "llmbar": "llm-blender/LLMBar",
    "mtbench": "lmsys/mt_bench_human_judgments",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare pairwise human-preference datasets.")
    parser.add_argument("--adapter", choices=sorted(ADAPTERS), required=True)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_name = args.dataset_name or DEFAULT_DATASET_NAMES[args.adapter]
    dataset = load_dataset(dataset_name)
    split = args.split or next(iter(dataset.keys()))
    rows = list(dataset[split])
    adapter = ADAPTERS[args.adapter]
    if args.adapter == "mtbench":
        pairs = adapter(rows)  # type: ignore[misc]
    else:
        pairs = adapter(rows, seed=args.seed)  # type: ignore[misc]

    output = args.output_path or data_path("processed", f"{args.adapter}_{split}.csv")
    write_standard_pairs(output, pairs)
    print("Saved:", output)
    print("Rows:", len(pairs))


if __name__ == "__main__":
    main()
