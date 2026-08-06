from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts.prepare_frozen_question_routing import (
    CALIBRATION_FILENAME,
    FULL_FILENAME,
    MANIFEST_FILENAME,
    TEST_FILENAME,
    build_routing_package,
    parse_dataset_lineage,
)


HEADER = [
    "question_id",
    "prompt",
    "response_a",
    "response_b",
    "winner",
    "turn",
    "routing_split",
    "opaque",
]


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        return next(reader), [list(row) for row in reader]


def _fixture_rows() -> list[list[str]]:
    return [
        ["q1", "First?", "A, with comma", "B", "model_a", "1", "old", "x"],
        ["q1", "First turn 2?", "A2", "B2", "unsupported", "2", "old", " y "],
        ["q2", "Second?", "", "B", "model_b", "1", "old", "z"],
        ["q3", "Third?", "A", "B", "tie", "1", "old", "unicode-π"],
        ["q4", "Fourth?", "A", "B", "model_b", "1", "old", "line\nbreak"],
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_builds_atomic_question_disjoint_package_and_separates_counts(
    tmp_path: Path,
) -> None:
    source = tmp_path / "frozen.csv"
    source_rows = _fixture_rows()
    _write_csv(source, HEADER, source_rows)
    source_bytes = source.read_bytes()
    output_dir = tmp_path / "routing-v1"
    lineage = {
        "dataset_name": "lmsys/mt_bench_human_judgments",
        "dataset_revision": "frozen-revision",
        "campaign_source": "controlled-shift-v1",
    }

    manifest = build_routing_package(
        source_csv=source,
        output_dir=output_dir,
        dataset_lineage=lineage,
    )

    assert source.read_bytes() == source_bytes
    assert set(path.name for path in output_dir.iterdir()) == {
        FULL_FILENAME,
        CALIBRATION_FILENAME,
        TEST_FILENAME,
        MANIFEST_FILENAME,
    }
    assert manifest["routing_unit"] == "question"
    assert manifest["seed"] == 42
    assert manifest["calibration_fraction"] == 0.5
    assert manifest["question_counts"] == {
        "total": 4,
        "calibration": 2,
        "test": 2,
        "overlap": 0,
    }
    assert manifest["counts"]["raw_rows"]["total"] == 5
    assert manifest["counts"]["eligible_pairs"]["total"] == 3
    assert manifest["counts"]["skipped_rows"]["total"] == 2
    assert (
        manifest["counts"]["eligible_pairs"]["total"]
        + manifest["counts"]["skipped_rows"]["total"]
        == manifest["counts"]["raw_rows"]["total"]
    )
    for count_name in ("raw_rows", "eligible_pairs", "skipped_rows"):
        counts = manifest["counts"][count_name]
        assert counts["calibration"] + counts["test"] == counts["total"]
    assert manifest["eligibility"]["raw_row_count"] == 5
    assert manifest["eligibility"]["eligible_pair_count"] == 3
    assert manifest["eligibility"]["skipped_reason_counts"] == {
        "invalid_winner": 1,
        "missing_response_a": 1,
    }
    assert manifest["source"]["path"] == str(source.resolve())
    assert manifest["source"]["sha256"] == _sha256(source)
    assert manifest["source"]["dataset_lineage"] == lineage

    full_header, full_rows = _read_csv(output_dir / FULL_FILENAME)
    assert full_header == HEADER
    routing_index = full_header.index("routing_split")
    retained_indices = [
        index for index, name in enumerate(full_header) if name != "routing_split"
    ]
    assert [
        [row[index] for index in retained_indices] for row in full_rows
    ] == [
        [row[index] for index in retained_indices] for row in source_rows
    ]
    assignments: dict[str, set[str]] = {}
    for row in full_rows:
        assignments.setdefault(row[0], set()).add(row[routing_index])
    assert all(len(splits) == 1 for splits in assignments.values())
    assert {next(iter(splits)) for splits in assignments.values()} == {
        "calibration",
        "test",
    }

    _, calibration_rows = _read_csv(output_dir / CALIBRATION_FILENAME)
    _, test_rows = _read_csv(output_dir / TEST_FILENAME)
    assert calibration_rows == [
        row for row in full_rows if row[routing_index] == "calibration"
    ]
    assert test_rows == [
        row for row in full_rows if row[routing_index] == "test"
    ]
    assert manifest["output_sha256"] == {
        "full": _sha256(output_dir / FULL_FILENAME),
        "calibration": _sha256(output_dir / CALIBRATION_FILENAME),
        "test": _sha256(output_dir / TEST_FILENAME),
    }
    assert json.loads((output_dir / MANIFEST_FILENAME).read_text()) == manifest


def test_assignment_and_csv_hashes_are_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "frozen.csv"
    _write_csv(source, HEADER, _fixture_rows())
    lineage = {"dataset": "mtbench", "revision": "abc123"}

    first = build_routing_package(
        source_csv=source,
        output_dir=tmp_path / "first",
        dataset_lineage=lineage,
    )
    second = build_routing_package(
        source_csv=source,
        output_dir=tmp_path / "second",
        dataset_lineage=lineage,
    )

    assert (
        first["routing_assignment_sha256"]
        == second["routing_assignment_sha256"]
    )
    assert first["output_sha256"] == second["output_sha256"]
    assert first == second


def test_adds_routing_column_without_changing_existing_cells(
    tmp_path: Path,
) -> None:
    source = tmp_path / "without-routing.csv"
    source_header = [column for column in HEADER if column != "routing_split"]
    routing_index = HEADER.index("routing_split")
    source_rows = [
        [value for index, value in enumerate(row) if index != routing_index]
        for row in _fixture_rows()
    ]
    _write_csv(source, source_header, source_rows)

    manifest = build_routing_package(
        source_csv=source,
        output_dir=tmp_path / "routing",
        dataset_lineage={"dataset": "mtbench"},
    )

    full_header, full_rows = _read_csv(tmp_path / "routing" / FULL_FILENAME)
    assert full_header == [*source_header, "routing_split"]
    assert [row[:-1] for row in full_rows] == source_rows
    assert manifest["source"]["had_routing_split_column"] is False


def test_refuses_to_overwrite_existing_package(tmp_path: Path) -> None:
    source = tmp_path / "frozen.csv"
    _write_csv(source, HEADER, _fixture_rows())
    output_dir = tmp_path / "routing"
    first = build_routing_package(
        source_csv=source,
        output_dir=output_dir,
        dataset_lineage={"dataset": "mtbench"},
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_routing_package(
            source_csv=source,
            output_dir=output_dir,
            dataset_lineage={"dataset": "mtbench"},
        )

    assert json.loads((output_dir / MANIFEST_FILENAME).read_text()) == first


def test_expected_source_sha256_fails_closed_before_publication(
    tmp_path: Path,
) -> None:
    source = tmp_path / "frozen.csv"
    _write_csv(source, HEADER, _fixture_rows())
    output_dir = tmp_path / "routing"

    with pytest.raises(ValueError, match="source CSV SHA-256 mismatch"):
        build_routing_package(
            source_csv=source,
            output_dir=output_dir,
            dataset_lineage={"dataset": "mtbench"},
            expected_source_sha256="0" * 64,
        )

    assert not output_dir.exists()


def test_expected_source_sha256_rejects_noncanonical_hex(tmp_path: Path) -> None:
    source = tmp_path / "frozen.csv"
    _write_csv(source, HEADER, _fixture_rows())

    with pytest.raises(ValueError, match="64 lowercase hex"):
        build_routing_package(
            source_csv=source,
            output_dir=tmp_path / "routing",
            dataset_lineage={"dataset": "mtbench"},
            expected_source_sha256="A" * 64,
        )


def test_failed_validation_never_publishes_partial_package(
    tmp_path: Path,
) -> None:
    source = tmp_path / "invalid.csv"
    _write_csv(
        source,
        ["question_id", "winner"],
        [["q1", "model_a"], ["q2", "model_b"]],
    )
    output_dir = tmp_path / "routing"

    with pytest.raises(KeyError, match="prompt/response_a/response_b"):
        build_routing_package(
            source_csv=source,
            output_dir=output_dir,
            dataset_lineage={"dataset": "mtbench"},
        )

    assert not output_dir.exists()
    assert not (tmp_path / ".routing.publish.lock").exists()
    assert not list(tmp_path.glob(".routing.staging-*"))


def test_dataset_lineage_must_be_a_nonempty_json_object() -> None:
    assert parse_dataset_lineage('{"dataset":"mtbench"}') == {
        "dataset": "mtbench"
    }
    with pytest.raises(ValueError, match="valid JSON"):
        parse_dataset_lineage("not-json")
    with pytest.raises(ValueError, match="JSON object"):
        parse_dataset_lineage("[]")
    with pytest.raises(ValueError, match="non-empty"):
        parse_dataset_lineage("{}")
