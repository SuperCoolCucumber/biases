from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from biases.dataset_splits import (
    assign_question_disjoint_routing_split,
    routing_manifest,
)
from biases.position_bias import PositionPair, load_position_pairs_with_eligibility


ROUTING_SEED = 42
CALIBRATION_FRACTION = 0.5
QUESTION_COLUMN = "question_id"
ROUTING_COLUMN = "routing_split"
FULL_FILENAME = "routed_full.csv"
CALIBRATION_FILENAME = "routed_calibration.csv"
TEST_FILENAME = "routed_test.csv"
MANIFEST_FILENAME = "routing_manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _read_csv_cells(path: Path) -> tuple[list[str], list[list[str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"source CSV does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("source CSV is empty") from exc
        rows = [list(row) for row in reader]

    if not header or any(not name for name in header):
        raise ValueError("source CSV must have a non-empty header")
    duplicates = sorted({name for name in header if header.count(name) > 1})
    if duplicates:
        raise ValueError(f"source CSV has duplicate columns: {duplicates}")
    if QUESTION_COLUMN not in header:
        raise ValueError(
            f"source CSV must contain the exact column {QUESTION_COLUMN!r}"
        )
    for row_index, row in enumerate(rows):
        if len(row) != len(header):
            raise ValueError(
                "source CSV has a malformed row: "
                f"row_index={row_index} fields={len(row)} expected={len(header)}"
            )
    if not rows:
        raise ValueError("source CSV must contain at least one data row")
    return header, rows


def _rows_without_routing_sha256(
    header: Sequence[str],
    rows: Sequence[Sequence[str]],
) -> str:
    retained_indices = [
        index for index, column in enumerate(header) if column != ROUTING_COLUMN
    ]
    return _canonical_sha256(
        {
            "columns": [header[index] for index in retained_indices],
            "rows": [
                [row[index] for index in retained_indices]
                for row in rows
            ],
        }
    )


def _route_rows(
    header: Sequence[str],
    rows: Sequence[Sequence[str]],
) -> tuple[list[str], list[list[str]]]:
    question_index = header.index(QUESTION_COLUMN)
    questions = [row[question_index] for row in rows]
    routed = assign_question_disjoint_routing_split(
        pd.DataFrame({QUESTION_COLUMN: questions}),
        calibration_fraction=CALIBRATION_FRACTION,
        seed=ROUTING_SEED,
    )
    assignments = routed[ROUTING_COLUMN].astype(str).tolist()

    output_header = list(header)
    output_rows = [list(row) for row in rows]
    if ROUTING_COLUMN in output_header:
        routing_index = output_header.index(ROUTING_COLUMN)
        for row, assignment in zip(output_rows, assignments, strict=True):
            row[routing_index] = assignment
    else:
        output_header.append(ROUTING_COLUMN)
        for row, assignment in zip(output_rows, assignments, strict=True):
            row.append(assignment)
    return output_header, output_rows


def _write_csv_exclusive(
    path: Path,
    header: Sequence[str],
    rows: Sequence[Sequence[str]],
) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")


def _split_counts(values: Sequence[str]) -> dict[str, int]:
    return {
        "total": len(values),
        "calibration": sum(value == "calibration" for value in values),
        "test": sum(value == "test" for value in values),
    }


def _eligible_question_counts(pairs: Sequence[PositionPair]) -> dict[str, int]:
    assignments: dict[str, str] = {}
    for pair in pairs:
        metadata = pair.original.metadata
        question_id = str(
            metadata.get("question_cluster_id") or pair.original.question_id
        ).strip()
        routing_split = str(metadata.get(ROUTING_COLUMN) or "").strip().lower()
        if routing_split not in {"calibration", "test"}:
            raise ValueError(
                f"eligible pair {pair.pair_id!r} has invalid routing_split "
                f"{routing_split!r}"
            )
        previous = assignments.setdefault(question_id, routing_split)
        if previous != routing_split:
            raise ValueError(
                f"eligible question {question_id!r} occurs in both routing splits"
            )
    values = list(assignments.values())
    return {
        **_split_counts(values),
        "overlap": 0,
    }


def _validate_dataset_lineage(dataset_lineage: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(dataset_lineage)
    if not normalized:
        raise ValueError("dataset lineage must be a non-empty JSON object")
    try:
        json.dumps(
            normalized,
            ensure_ascii=True,
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("dataset lineage must be JSON serializable") from exc
    return normalized


def _validate_written_package(
    *,
    source_header: Sequence[str],
    source_rows: Sequence[Sequence[str]],
    full_path: Path,
    calibration_path: Path,
    test_path: Path,
    manifest: Mapping[str, Any],
) -> None:
    full_header, full_rows = _read_csv_cells(full_path)
    calibration_header, calibration_rows = _read_csv_cells(calibration_path)
    test_header, test_rows = _read_csv_cells(test_path)
    if full_header != calibration_header or full_header != test_header:
        raise ValueError("routed output CSV headers do not match")
    if _rows_without_routing_sha256(source_header, source_rows) != (
        _rows_without_routing_sha256(full_header, full_rows)
    ):
        raise ValueError("full routed CSV changed source content or row order")

    routing_index = full_header.index(ROUTING_COLUMN)
    expected_calibration = [
        row for row in full_rows if row[routing_index] == "calibration"
    ]
    expected_test = [row for row in full_rows if row[routing_index] == "test"]
    if calibration_rows != expected_calibration:
        raise ValueError("calibration CSV does not match the routed full CSV")
    if test_rows != expected_test:
        raise ValueError("test CSV does not match the routed full CSV")

    output_sha256 = manifest.get("output_sha256")
    if not isinstance(output_sha256, Mapping):
        raise ValueError("routing manifest is missing output_sha256")
    observed_hashes = {
        "full": _sha256(full_path),
        "calibration": _sha256(calibration_path),
        "test": _sha256(test_path),
    }
    if observed_hashes != dict(output_sha256):
        raise ValueError("routing manifest output hashes do not match")

    observed = routing_manifest(
        pd.read_csv(full_path, dtype=str, keep_default_na=False),
        routing_unit="question",
        seed=ROUTING_SEED,
        calibration_fraction=CALIBRATION_FRACTION,
    )
    for field in (
        "routing_unit",
        "seed",
        "calibration_fraction",
        "row_counts",
        "question_counts",
        "routing_assignment_sha256",
    ):
        if observed[field] != manifest.get(field):
            raise ValueError(f"routing manifest {field} does not match output")


def _build_staged_package(
    *,
    source_csv: Path,
    staging_dir: Path,
    dataset_lineage: Mapping[str, Any],
) -> dict[str, Any]:
    source_header, source_rows = _read_csv_cells(source_csv)
    routed_header, routed_rows = _route_rows(source_header, source_rows)
    routing_index = routed_header.index(ROUTING_COLUMN)
    calibration_rows = [
        row for row in routed_rows if row[routing_index] == "calibration"
    ]
    test_rows = [row for row in routed_rows if row[routing_index] == "test"]

    full_path = staging_dir / FULL_FILENAME
    calibration_path = staging_dir / CALIBRATION_FILENAME
    test_path = staging_dir / TEST_FILENAME
    _write_csv_exclusive(full_path, routed_header, routed_rows)
    _write_csv_exclusive(calibration_path, routed_header, calibration_rows)
    _write_csv_exclusive(test_path, routed_header, test_rows)

    routed_frame = pd.read_csv(full_path, dtype=str, keep_default_na=False)
    routing = routing_manifest(
        routed_frame,
        routing_unit="question",
        seed=ROUTING_SEED,
        calibration_fraction=CALIBRATION_FRACTION,
    )
    pairs, eligibility = load_position_pairs_with_eligibility(full_path)
    raw_routing = [row[routing_index] for row in routed_rows]
    raw_row_counts = _split_counts(raw_routing)
    if raw_row_counts != routing["row_counts"]:
        raise AssertionError("raw-row routing accounting is inconsistent")
    if eligibility.raw_row_count != raw_row_counts["total"]:
        raise ValueError(
            "pair-loader raw-row count differs from the routed CSV: "
            f"loader={eligibility.raw_row_count} csv={raw_row_counts['total']}"
        )
    if eligibility.raw_row_count != (
        eligibility.eligible_pair_count + eligibility.skipped_row_count
    ):
        raise AssertionError("raw and eligible pair counts were conflated")

    output_sha256 = {
        "full": _sha256(full_path),
        "calibration": _sha256(calibration_path),
        "test": _sha256(test_path),
    }
    manifest: dict[str, Any] = {
        **routing,
        "schema_version": 2,
        "artifact_type": "frozen_question_disjoint_routing_package",
        "source": {
            "path": str(source_csv.resolve()),
            "sha256": _sha256(source_csv),
            "dataset_lineage": dict(dataset_lineage),
            "columns": list(source_header),
            "had_routing_split_column": ROUTING_COLUMN in source_header,
            "rows_without_routing_sha256": _rows_without_routing_sha256(
                source_header,
                source_rows,
            ),
        },
        "counts": {
            "raw_rows": raw_row_counts,
            "raw_questions": dict(routing["question_counts"]),
            "eligible_pairs": {
                "total": eligibility.eligible_pair_count,
                "calibration": int(
                    eligibility.routing_counts["eligible_pairs"].get(
                        "calibration", 0
                    )
                ),
                "test": int(
                    eligibility.routing_counts["eligible_pairs"].get("test", 0)
                ),
            },
            "eligible_questions": _eligible_question_counts(pairs),
            "skipped_rows": {
                "total": eligibility.skipped_row_count,
                "calibration": int(
                    eligibility.routing_counts["skipped_rows"].get(
                        "calibration", 0
                    )
                ),
                "test": int(
                    eligibility.routing_counts["skipped_rows"].get("test", 0)
                ),
            },
        },
        "eligibility": eligibility.to_dict(),
        "content_preservation": {
            "preserved_columns": [
                column for column in source_header if column != ROUTING_COLUMN
            ],
            "recomputed_columns": [ROUTING_COLUMN],
            "row_order_preserved": True,
            "rows_without_routing_sha256": _rows_without_routing_sha256(
                routed_header,
                routed_rows,
            ),
        },
        "outputs": {
            "full": {"path": FULL_FILENAME, "rows": len(routed_rows)},
            "calibration": {
                "path": CALIBRATION_FILENAME,
                "rows": len(calibration_rows),
            },
            "test": {"path": TEST_FILENAME, "rows": len(test_rows)},
        },
        "output_sha256": output_sha256,
    }
    if (
        manifest["source"]["rows_without_routing_sha256"]
        != manifest["content_preservation"]["rows_without_routing_sha256"]
    ):
        raise AssertionError("source content changed while assigning routing")

    _validate_written_package(
        source_header=source_header,
        source_rows=source_rows,
        full_path=full_path,
        calibration_path=calibration_path,
        test_path=test_path,
        manifest=manifest,
    )
    manifest_path = staging_dir / MANIFEST_FILENAME
    _write_json_exclusive(manifest_path, manifest)
    with manifest_path.open("r", encoding="utf-8") as handle:
        if json.load(handle) != manifest:
            raise ValueError("serialized routing manifest failed round-trip validation")
    return manifest


def build_routing_package(
    *,
    source_csv: Path,
    output_dir: Path,
    dataset_lineage: Mapping[str, Any],
    expected_source_sha256: str | None = None,
) -> dict[str, Any]:
    """Build and atomically publish an immutable routing package."""

    source_csv = source_csv.resolve()
    output_dir = output_dir.absolute()
    lineage = _validate_dataset_lineage(dataset_lineage)
    observed_source_sha256 = _sha256(source_csv)
    if expected_source_sha256 is not None:
        expected_source_sha256 = str(expected_source_sha256).strip()
        if len(expected_source_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in expected_source_sha256
        ):
            raise ValueError(
                "expected source SHA-256 must be exactly 64 lowercase hex characters"
            )
        if observed_source_sha256 != expected_source_sha256:
            raise ValueError(
                "source CSV SHA-256 mismatch: "
                f"observed={observed_source_sha256} "
                f"expected={expected_source_sha256}"
            )
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(
            f"refusing to overwrite routing package: {output_dir}"
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir.parent / f".{output_dir.name}.publish.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(
            f"routing package publication is already locked: {lock_path}"
        ) from exc
    os.close(lock_fd)

    staging_dir: Path | None = None
    try:
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError(
                f"refusing to overwrite routing package: {output_dir}"
            )
        staging_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{output_dir.name}.staging-",
                dir=output_dir.parent,
            )
        )
        manifest = _build_staged_package(
            source_csv=source_csv,
            staging_dir=staging_dir,
            dataset_lineage=lineage,
        )
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError(
                f"refusing to overwrite routing package: {output_dir}"
            )
        staging_dir.rename(output_dir)
        staging_dir = None
        return manifest
    finally:
        if staging_dir is not None and staging_dir.exists():
            shutil.rmtree(staging_dir)
        lock_path.unlink(missing_ok=True)


def parse_dataset_lineage(raw: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("--dataset-lineage-json must be valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("--dataset-lineage-json must decode to a JSON object")
    return _validate_dataset_lineage(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Derive an immutable seed-42, 50/50 question-disjoint routing "
            "package from an already frozen source CSV."
        )
    )
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-source-sha256",
        required=True,
        help="exact lowercase SHA-256 required for the frozen source CSV",
    )
    lineage_group = parser.add_mutually_exclusive_group(required=True)
    lineage_group.add_argument(
        "--dataset-lineage-json",
        help=(
            "required JSON object identifying the frozen dataset lineage; "
            "for example '{\"dataset\":\"mt_bench\",\"revision\":\"...\"}'"
        ),
    )
    lineage_group.add_argument(
        "--dataset-lineage-path",
        type=Path,
        help="path to a UTF-8 JSON object identifying the frozen dataset lineage",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_lineage = (
        args.dataset_lineage_json
        if args.dataset_lineage_json is not None
        else args.dataset_lineage_path.read_text(encoding="utf-8")
    )
    manifest = build_routing_package(
        source_csv=args.source_csv,
        output_dir=args.output_dir,
        dataset_lineage=parse_dataset_lineage(raw_lineage),
        expected_source_sha256=args.expected_source_sha256,
    )
    print(f"Published immutable routing package: {args.output_dir.resolve()}")
    print(
        "Raw rows / eligible pairs / skipped rows: "
        f"{manifest['counts']['raw_rows']['total']} / "
        f"{manifest['counts']['eligible_pairs']['total']} / "
        f"{manifest['counts']['skipped_rows']['total']}"
    )
    print(
        "Question routing assignment SHA-256: "
        f"{manifest['routing_assignment_sha256']}"
    )


if __name__ == "__main__":
    main()
