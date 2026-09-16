from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.analyze_lowest_dose import (
    file_hash,
    load_stage,
    main,
    reconstruct_source,
    validate_source_linkage,
)


# The seeded question assignment is fixed in this independent fixture contract.
SOURCE_TEXT = (
    "question_id,prompt,response_a,response_b,winner\n"
    'q1,"Question, one?",Answer A1,Answer B1,model_a\n'
    "q2,Question two?,Answer A2,Answer B2,model_b\n"
    "q3,Question three?,Answer A3,Answer B3,model_a\n"
    "q4,Question four?,Answer A4,Answer B4,model_b\n"
)
ROUTED_TEXT = (
    "question_id,prompt,response_a,response_b,winner,routing_split\n"
    'q1,"Question, one?",Answer A1,Answer B1,model_a,calibration\n'
    "q2,Question two?,Answer A2,Answer B2,model_b,calibration\n"
    "q3,Question three?,Answer A3,Answer B3,model_a,test\n"
    "q4,Question four?,Answer A4,Answer B4,model_b,test\n"
)
ASSIGNMENT_TEXT = (
    '[{"question_id":"q1","routing_split":"calibration"},'
    '{"question_id":"q2","routing_split":"calibration"},'
    '{"question_id":"q3","routing_split":"test"},'
    '{"question_id":"q4","routing_split":"test"}]'
)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dataset_contract() -> dict[str, Any]:
    return {
        "source_sha256": digest(SOURCE_TEXT),
        "routed_sha256": digest(ROUTED_TEXT),
        "assignment_sha256": digest(ASSIGNMENT_TEXT),
        "calibration_fraction": 0.5,
        "seed": 19,
    }


def source_rows() -> dict[int, dict[str, str]]:
    return {
        index: {
            "question_id": f"q{index + 1}",
            "human_winner": "A" if index % 2 == 0 else "B",
            "routing_split": "calibration" if index < 2 else "test",
        }
        for index in range(4)
    }


def stage_rows(model: str, stage: str) -> list[dict[str, Any]]:
    records = []
    for index, source in source_rows().items():
        if stage == "stage_b" and source["routing_split"] != "test":
            continue
        for ordering in ("ab", "ba"):
            human = source["human_winner"]
            if ordering == "ba":
                human = "B" if human == "A" else "A"
            clean_id = f"{model}-clean-{index}-{ordering}"
            clean = {
                "record_id": clean_id,
                "example_id": f"row-{index}-{ordering}",
                "source_row_index": index,
                "question_id": source["question_id"],
                "pair_identity_key": f"source-row-{index}",
                "pair_key": f"{model}-row-{index}-{ordering}",
                "ordering": ordering,
                "model_name": model,
                "model_revision": "a" * 40,
                "routing_split": source["routing_split"],
                "bias_name": "clean",
                "variant_id": "clean",
                "cue_congruency": "clean",
                "human_winner": human,
                "verdict": human,
                "label_prob_A": 0.80 if human == "A" else 0.15,
                "label_prob_B": 0.80 if human == "B" else 0.15,
                "label_prob_tie": 0.05,
                "input_file_hash": digest(ROUTED_TEXT),
            }
            if stage == "stage_a":
                records.append(clean)
                continue
            for family, dose in (("authority", 1), ("bandwagon", 55)):
                for target in ("A", "B"):
                    direction = "congruent" if target == human else "incongruent"
                    verdict = target if family == "authority" else human
                    records.append({
                        **clean,
                        "record_id": f"{model}-{family}-{index}-{ordering}-{target}",
                        "clean_record_id": clean_id,
                        "bias_name": family,
                        "dose": dose,
                        "variant_id": f"{family}_{direction}_{dose}_{ordering}",
                        "cue_congruency": direction,
                        "cue_target": target,
                        "reference_kind": "model_clean_verdict",
                        "verdict": verdict,
                        "label_prob_A": 0.80 if verdict == "A" else 0.15,
                        "label_prob_B": 0.80 if verdict == "B" else 0.15,
                    })
    return records


def write_stage(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return {"sha256": file_hash(path), "records": len(rows)}


@pytest.fixture
def campaign(tmp_path: Path) -> dict[str, Any]:
    source = tmp_path / "source.csv"
    source.write_text(SOURCE_TEXT, encoding="utf-8")
    specifications = {}
    paths: dict[str, list[Path]] = {"stage_a": [], "stage_b": []}
    for model in ("fixture-alpha", "fixture-beta"):
        specifications[model] = {}
        for stage in paths:
            path = tmp_path / f"{model}-{stage}.jsonl"
            specifications[model][stage] = write_stage(path, stage_rows(model, stage))
            paths[stage].append(path)
    contract = {"dataset": dataset_contract(), "models": specifications}
    contract_path = tmp_path / "input-contract.json"
    contract_path.write_text(json.dumps(contract))
    return {
        "source": source,
        "contract": contract,
        "contract_path": contract_path,
        "paths": paths,
        "output": tmp_path / "analysis-results",
    }


def arguments(campaign: dict[str, Any]) -> list[str]:
    return [
        "--stage-a", *(str(path) for path in campaign["paths"]["stage_a"]),
        "--stage-b", *(str(path) for path in campaign["paths"]["stage_b"]),
        "--source-csv", str(campaign["source"]),
        "--input-contract", str(campaign["contract_path"]),
        "--output-dir", str(campaign["output"]),
        "--bootstrap-resamples", "20", "--seed", "17", "--no-figures",
    ]


def test_reconstruct_source_matches_exact_routed_bytes_without_modifying_csv(
    campaign: dict[str, Any],
) -> None:
    before = campaign["source"].read_bytes()
    source, audit = reconstruct_source(campaign["source"], dataset_contract())

    assert campaign["source"].read_bytes() == before
    assert audit["source_sha256"] == digest(SOURCE_TEXT)
    assert audit["routed_sha256"] == digest(ROUTED_TEXT)
    assert audit["assignment_sha256"] == digest(ASSIGNMENT_TEXT)
    assert audit["eligible_pairs"] == 4
    assert audit["eligible_split_counts"] == {"calibration": 2, "test": 2}
    assert audit["question_ids"] == {"calibration": ["q1", "q2"], "test": ["q3", "q4"]}
    assert audit["human_label_counts"]["test"] == {"A": 1, "B": 1}
    assert source[0]["prompt"] == "Question, one?"
    assert source[3]["human_winner"] == "B"


@pytest.mark.parametrize("hash_name", ("source_sha256", "routed_sha256", "assignment_sha256"))
def test_reconstruct_source_rejects_each_contract_hash_mismatch(
    campaign: dict[str, Any], hash_name: str,
) -> None:
    contract = {**dataset_contract(), hash_name: "0" * 64}
    with pytest.raises(ValueError, match="hash"):
        reconstruct_source(campaign["source"], contract)


def test_source_eligibility_preserves_original_row_indices(tmp_path: Path) -> None:
    text = SOURCE_TEXT + "q1,Incomplete answer,Answer A,,model_a\n"
    routed = ROUTED_TEXT + "q1,Incomplete answer,Answer A,,model_a,calibration\n"
    source_path = tmp_path / "source-with-ineligible-row.csv"
    source_path.write_text(text)
    contract = {**dataset_contract(), "source_sha256": digest(text), "routed_sha256": digest(routed)}
    source, audit = reconstruct_source(source_path, contract)

    assert set(source) == {0, 1, 2, 3}
    assert audit["raw_rows"] == 5
    assert audit["eligible_pairs"] == 4
    assert audit["skipped_rows"] == 1


@pytest.mark.parametrize("stage", ("stage_a", "stage_b"))
def test_load_stage_accepts_exact_models_and_records_with_receipts(
    campaign: dict[str, Any], stage: str,
) -> None:
    records, receipts = load_stage(
        campaign["paths"][stage], stage, campaign["contract"]["models"],
        source_rows(), digest(ROUTED_TEXT),
    )

    expected_per_model = 8 if stage == "stage_a" else 16
    assert len(records) == 2 * expected_per_model
    assert {receipt["model_name"] for receipt in receipts} == {"fixture-alpha", "fixture-beta"}
    for receipt in receipts:
        assert receipt["records"] == expected_per_model
        assert receipt["sha256"] == file_hash(Path(receipt["path"]))


@pytest.mark.parametrize(
    ("field", "value", "error"),
    (
        ("model_name", "unexpected-model", "unexpected model"),
        ("source_row_index", 99, "ineligible source row"),
        ("source_row_index", "0", "ineligible source row"),
        ("ordering", "unknown", "presentation order"),
        ("human_winner", "B", "human label/order mismatch"),
        ("question_id", "q-wrong", "question identity mismatch"),
        ("routing_split", "test", "routing mismatch"),
        ("input_file_hash", "f" * 64, "routed input hash mismatch"),
        ("bias_name", "authority", "non-clean Stage A"),
    ),
)
def test_load_stage_rejects_invalid_record_metadata_even_with_matching_file_hash(
    campaign: dict[str, Any], field: str, value: Any, error: str,
) -> None:
    records = stage_rows("fixture-alpha", "stage_a")
    records[0][field] = value
    models = copy.deepcopy(campaign["contract"]["models"])
    models["fixture-alpha"]["stage_a"] = write_stage(campaign["paths"]["stage_a"][0], records)

    with pytest.raises(ValueError, match=error):
        load_stage(campaign["paths"]["stage_a"], "stage_a", models, source_rows(), digest(ROUTED_TEXT))


def test_swapped_order_cannot_reuse_the_original_human_label(campaign: dict[str, Any]) -> None:
    records = stage_rows("fixture-alpha", "stage_a")
    assert records[1]["ordering"] == "ba"
    records[1]["human_winner"] = records[0]["human_winner"]
    models = copy.deepcopy(campaign["contract"]["models"])
    models["fixture-alpha"]["stage_a"] = write_stage(campaign["paths"]["stage_a"][0], records)

    with pytest.raises(ValueError, match="human label/order mismatch"):
        load_stage(campaign["paths"]["stage_a"], "stage_a", models, source_rows(), digest(ROUTED_TEXT))


def test_unrecognized_and_ambiguous_stage_hashes_are_rejected(campaign: dict[str, Any]) -> None:
    models = copy.deepcopy(campaign["contract"]["models"])
    models["fixture-alpha"]["stage_a"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="unrecognized or ambiguous"):
        load_stage(campaign["paths"]["stage_a"], "stage_a", models, source_rows(), digest(ROUTED_TEXT))

    models = copy.deepcopy(campaign["contract"]["models"])
    models["fixture-beta"]["stage_a"] = models["fixture-alpha"]["stage_a"].copy()
    with pytest.raises(ValueError, match="unrecognized or ambiguous"):
        load_stage(campaign["paths"]["stage_a"], "stage_a", models, source_rows(), digest(ROUTED_TEXT))


def test_missing_and_duplicate_model_files_are_rejected(campaign: dict[str, Any]) -> None:
    paths = campaign["paths"]["stage_a"]
    models = campaign["contract"]["models"]
    with pytest.raises(ValueError, match="model set differs"):
        load_stage(paths[:1], "stage_a", models, source_rows(), digest(ROUTED_TEXT))
    with pytest.raises(ValueError, match="duplicate stage_a input"):
        load_stage([paths[0], paths[0]], "stage_a", models, source_rows(), digest(ROUTED_TEXT))


@pytest.mark.parametrize("mutation", ("missing_order", "duplicate_source_order", "wrong_count"))
def test_stage_a_requires_the_exact_source_order_grid(campaign: dict[str, Any], mutation: str) -> None:
    rows = stage_rows("fixture-alpha", "stage_a")
    if mutation == "missing_order":
        rows.pop()
    elif mutation == "duplicate_source_order":
        rows.append({**rows[0], "record_id": "duplicate-source-order"})
    models = copy.deepcopy(campaign["contract"]["models"])
    models["fixture-alpha"]["stage_a"] = write_stage(campaign["paths"]["stage_a"][0], rows)
    if mutation == "wrong_count":
        models["fixture-alpha"]["stage_a"]["records"] += 1

    with pytest.raises(ValueError):
        load_stage(campaign["paths"]["stage_a"], "stage_a", models, source_rows(), digest(ROUTED_TEXT))


def test_stage_b_rejects_a_calibration_record_even_when_its_source_metadata_matches(
    campaign: dict[str, Any],
) -> None:
    rows = stage_rows("fixture-alpha", "stage_b")
    calibration = {
        **stage_rows("fixture-alpha", "stage_a")[0],
        "bias_name": "authority", "dose": 1, "cue_target": "A",
        "variant_id": "authority_congruent_1_ab", "cue_congruency": "congruent",
    }
    rows.append(calibration)
    models = copy.deepcopy(campaign["contract"]["models"])
    models["fixture-alpha"]["stage_b"] = write_stage(campaign["paths"]["stage_b"][0], rows)

    with pytest.raises(ValueError, match="non-test Stage B row"):
        load_stage(campaign["paths"]["stage_b"], "stage_b", models, source_rows(), digest(ROUTED_TEXT))


def test_source_row_index_cannot_be_a_json_boolean(campaign: dict[str, Any]) -> None:
    rows = stage_rows("fixture-alpha", "stage_a")
    assert rows[2]["source_row_index"] == 1
    rows[2]["source_row_index"] = True
    models = copy.deepcopy(campaign["contract"]["models"])
    models["fixture-alpha"]["stage_a"] = write_stage(campaign["paths"]["stage_a"][0], rows)

    with pytest.raises(ValueError, match="ineligible source row"):
        load_stage(campaign["paths"]["stage_a"], "stage_a", models, source_rows(), digest(ROUTED_TEXT))


def test_main_refuses_an_existing_output_directory_without_touching_it(
    campaign: dict[str, Any],
) -> None:
    campaign["output"].mkdir()
    marker = campaign["output"] / "existing-results.txt"
    marker.write_text("Preserve this result")

    with pytest.raises(SystemExit) as failure:
        main(arguments(campaign))

    assert failure.value.code == 2
    assert marker.read_text() == "Preserve this result"
    assert list(campaign["output"].iterdir()) == [marker]


def test_invalid_inputs_do_not_create_a_result_directory(campaign: dict[str, Any]) -> None:
    campaign["source"].write_text(SOURCE_TEXT + "\n")
    with pytest.raises(ValueError, match="source dataset hash"):
        main(arguments(campaign))
    assert not campaign["output"].exists()


def test_two_model_cli_run_produces_auditable_paired_outputs(campaign: dict[str, Any]) -> None:
    assert main(arguments(campaign)) == 0
    output = campaign["output"]
    completion = json.loads((output / "analysis_complete.json").read_text())
    audit = json.loads((output / "input_audit.json").read_text())
    summary = json.loads((output / "analysis.json").read_text())
    with (output / "paired_lowest_dose.csv").open(newline="") as handle:
        paired = list(csv.DictReader(handle))

    assert completion["status"] == "complete"
    assert completion["paired_lowest_dose_records"] == 32
    assert completion["models"] == ["fixture-alpha", "fixture-beta"]
    assert len(paired) == 32
    assert {row["question_id"] for row in paired} == {"q3", "q4"}
    assert {row["ordering"] for row in paired} == {"ab", "ba"}
    assert {row["family"] for row in paired} == {"authority", "bandwagon"}
    assert {row["human_direction"] for row in paired} == {"congruent", "incongruent"}
    assert {row["source_row_index"] for row in paired} == {"2", "3"}
    assert audit["source"]["question_ids"]["test"] == ["q3", "q4"]
    assert len(audit["inputs"]) == 4
    assert audit["input_contract_sha256"] == file_hash(campaign["contract_path"])
    assert summary["audit"]["n_questions"] == 2
    assert summary["audit"]["bootstrap_unit"] == "question"
    rates = [
        row for row in summary["transition_rates"]
        if row["model_name"] == "fixture-alpha" and row["ordering"] == "pooled"
        and row["human_direction"] == "incongruent" and row["clean_state"] == "agrees"
        and row["metric"] == "decisive_reversal_rate"
    ]
    assert {row["family"]: row["estimate"] for row in rates} == {"authority": 1.0, "bandwagon": 0.0}
    for filename, expected_hash in completion["outputs"].items():
        assert file_hash(output / filename) == expected_hash
    assert not list(output.glob("*.png"))
    assert not list(output.glob("*.pdf"))


def repeated_question_linkage_inputs(
    campaign: dict[str, Any], *, corruption: str | None = None,
) -> tuple:
    """Two distinct source rows share their question and human preference."""
    source = source_rows()
    source[3] = {**source[2]}
    models = copy.deepcopy(campaign["contract"]["models"])
    for model_index, model in enumerate(("fixture-alpha", "fixture-beta")):
        for stage in ("stage_a", "stage_b"):
            rows = stage_rows(model, stage)
            for row in rows:
                index = row["source_row_index"]
                if index == 3:
                    row["question_id"] = "q3"
                    row["human_winner"] = "A" if row["ordering"] == "ab" else "B"
                if model != "fixture-alpha":
                    continue
                if corruption == "cross_stage" and stage == "stage_b" and index == 2:
                    # The clean reference and metadata still name pair 2, while
                    # the archived cue record was sourced from different pair 3.
                    row["source_row_index"] = 3
                elif corruption == "order_twins" and index in {2, 3} and row["ordering"] == "ba":
                    # Both stages agree locally. Each nominal pair nevertheless
                    # combines AB from one source row with BA from the other.
                    row["source_row_index"] = 5 - index
            models[model][stage] = write_stage(campaign["paths"][stage][model_index], rows)
    clean_indices: dict[str, int] = {}
    cued_indices: dict[str, int] = {}
    clean, _ = load_stage(
        campaign["paths"]["stage_a"], "stage_a", models, source, digest(ROUTED_TEXT),
        source_indices=clean_indices,
    )
    cued, _ = load_stage(
        campaign["paths"]["stage_b"], "stage_b", models, source, digest(ROUTED_TEXT),
        source_indices=cued_indices,
    )
    return clean, cued, clean_indices, cued_indices


def test_source_linkage_accepts_distinct_pairs_from_the_same_question_and_label(
    campaign: dict[str, Any],
) -> None:
    clean, cued, clean_indices, cued_indices = repeated_question_linkage_inputs(campaign)

    assert len(clean_indices) == len(clean) == 16
    assert len(cued_indices) == len(cued) == 32
    assert clean_indices["fixture-alpha-clean-2-ab"] == 2
    assert clean_indices["fixture-alpha-clean-3-ab"] == 3
    validate_source_linkage(clean, cued, clean_indices, cued_indices)


def test_stage_b_cannot_link_a_different_source_pair_with_matching_question_and_label(
    campaign: dict[str, Any],
) -> None:
    clean, cued, clean_indices, cued_indices = repeated_question_linkage_inputs(
        campaign, corruption="cross_stage",
    )
    record = next(row for row in cued if row.record_id == "fixture-alpha-authority-2-ab-A")
    partner = next(row for row in clean if row.record_id == record.clean_record_id)
    assert record.question_id == partner.question_id == "q3"
    assert record.human_winner == partner.human_winner == "A"
    assert cued_indices[record.record_id] == 3
    assert clean_indices[partner.record_id] == 2

    with pytest.raises(ValueError):
        validate_source_linkage(clean, cued, clean_indices, cued_indices)


def test_order_twins_must_represent_the_same_source_pair_even_when_stages_agree(
    campaign: dict[str, Any],
) -> None:
    clean, cued, clean_indices, cued_indices = repeated_question_linkage_inputs(
        campaign, corruption="order_twins",
    )
    assert all(cued_indices[row.record_id] == clean_indices[row.clean_record_id] for row in cued)
    assert clean_indices["fixture-alpha-clean-2-ab"] == 2
    assert clean_indices["fixture-alpha-clean-2-ba"] == 3

    with pytest.raises(ValueError):
        validate_source_linkage(clean, cued, clean_indices, cued_indices)
