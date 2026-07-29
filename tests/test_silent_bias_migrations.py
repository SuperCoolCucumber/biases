from __future__ import annotations

import json
from pathlib import Path

import pytest

from biases.migrations import (
    LINKAGE_STATUS_KEY,
    migrate_jsonl,
    migrate_run_record_linkage,
)


def _legacy_row() -> dict:
    return {
        "record_id": "legacy-r1",
        "spec": {
            "dataset_name": "fixture.csv",
            "dataset_split": "test",
            "model_name": "fixture-judge",
            "backend_name": "vllm",
            "bias_name": "authority",
            "output_mode": "choice_only",
            "uncertainty_methods": ["logit"],
            "consistency_runs": 4,
            "temperature": 0.7,
        },
        "example_id": "q1:turn-1:original",
        "question_id": "q1:turn-1",
        "condition": {
            "bias_type": "authority",
            "variant_id": "authority_congruent",
            "cue_congruency": "congruent",
        },
        "seed": 0,
        "verdict": "A",
        "raw_output": "A",
        "prompt_hash": "prompt-hash",
        "uncertainty": {},
        "metadata": {
            "pair_id": "q1:turn-1",
            "source_row_index": 9,
            "turn": "1",
            "variant_id": "authority_congruent",
        },
    }


def test_legacy_linkage_migration_is_resolved_and_idempotent() -> None:
    first = migrate_run_record_linkage(
        _legacy_row(),
        input_file_hash="d" * 64,
    )
    second = migrate_run_record_linkage(
        first,
        input_file_hash="d" * 64,
    )

    assert first == second
    assert first["pair_key"].startswith("pair_")
    assert first["condition_group_id"].startswith("condition_")
    assert first["ordering_twin_key"].startswith("pair_")
    assert first["condition"]["ordering"] == "ab"
    assert first["spec_hash"]
    assert first["input_file_hash"] == "d" * 64
    assert first["metadata"][LINKAGE_STATUS_KEY] == "resolved"


def test_missing_source_identity_is_explicitly_unresolved() -> None:
    migrated = migrate_run_record_linkage(_legacy_row())

    assert migrated["pair_key"] is None
    assert migrated["condition_group_id"] is None
    assert migrated["ordering_twin_key"] is None
    assert migrated["metadata"][LINKAGE_STATUS_KEY].startswith(
        "unresolved_missing_fields:"
    )
    assert "input_file_hash" in migrated["metadata"][LINKAGE_STATUS_KEY]


def test_jsonl_migration_never_writes_in_place(tmp_path: Path) -> None:
    source = tmp_path / "records.jsonl"
    destination = tmp_path / "migrated.jsonl"
    source_text = json.dumps(_legacy_row()) + "\n"
    source.write_text(source_text, encoding="utf-8")

    report = migrate_jsonl(
        source_path=source,
        destination_path=destination,
        input_file_hash="e" * 64,
    )

    assert source.read_text(encoding="utf-8") == source_text
    assert report.total_rows == 1
    assert report.resolved_rows == 1
    assert report.unresolved_rows == 0
    migrated = json.loads(destination.read_text(encoding="utf-8"))
    assert migrated["metadata"][LINKAGE_STATUS_KEY] == "resolved"

    with pytest.raises(ValueError, match="different destination"):
        migrate_jsonl(
            source_path=source,
            destination_path=source,
            input_file_hash="e" * 64,
        )
    with pytest.raises(FileExistsError):
        migrate_jsonl(
            source_path=source,
            destination_path=destination,
            input_file_hash="e" * 64,
        )
