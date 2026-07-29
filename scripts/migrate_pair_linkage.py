from __future__ import annotations

import argparse
import json
from pathlib import Path

from biases.migrations import migrate_jsonl
from biases.pairing import file_sha256


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Add backward-compatible pair linkage fields to RunRecord JSONL. "
            "The source file is never modified."
        )
    )
    parser.add_argument("source_path", type=Path)
    parser.add_argument("destination_path", type=Path)
    hash_group = parser.add_mutually_exclusive_group()
    hash_group.add_argument(
        "--input-file-hash",
        help="SHA-256 of the dataset file used by the run.",
    )
    hash_group.add_argument(
        "--dataset-file",
        type=Path,
        help="Dataset file to hash for collision-safe pair keys.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace an existing destination; never replaces the source.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_file_hash = args.input_file_hash
    if args.dataset_file is not None:
        input_file_hash = file_sha256(args.dataset_file)

    report = migrate_jsonl(
        source_path=args.source_path,
        destination_path=args.destination_path,
        input_file_hash=input_file_hash,
        overwrite=args.overwrite,
    )
    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
