from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from biases.analysis.records import ConditionRecord, record_from_mapping
from biases.analysis.uncertainty_shift import (
    PredictorSpec,
    analyze_matched_group,
    controlled_uncertainty_shift_report,
    exact_test_pairs,
    question_split_from_records,
    validate_single_model_identity,
)
from biases.dataset_splits import routing_assignment_sha256
from biases.schemas import CueReferenceKind
from biases.social_cue_prompts import AUTHORITY_DOSES, BANDWAGON_DOSES
from scripts.analyze_controlled_uncertainty_shift import (
    build_score_table,
    load_frozen_routing_package,
    parse_predictor_declarations,
    validate_lowercase_sha256,
    validate_model_revision,
    validate_output_path,
)
from scripts.prepare_frozen_question_routing import (
    MANIFEST_FILENAME,
    build_routing_package,
)


MODEL_REVISION = "a" * 40


def raw_record(
    record_id: str,
    question_id: str,
    *,
    routing_split: str,
    verdict: str = "A",
    family: str = "clean",
    clean_record_id: str | None = None,
    score: float = 0.5,
    direction: str = "incongruent",
    dose: int = 4,
    clean_tie: bool = False,
    model_name: str = "model",
    model_revision: str | None = MODEL_REVISION,
    reference_kind: str | None = None,
) -> dict[str, object]:
    pair_key = f"pair-{question_id}"
    return {
        "record_id": record_id,
        "example_id": pair_key,
        "question_id": question_id,
        "pair_key": pair_key,
        "pair_identity_key": pair_key,
        "clean_record_id": clean_record_id,
        "ordering": "ab",
        "model_name": model_name,
        "model_revision": model_revision,
        "routing_split": routing_split,
        "bias_name": family,
        "cue_congruency": "clean" if family == "clean" else direction,
        "dose": None if family == "clean" else dose,
        "variant_id": (
            "clean" if family == "clean" else f"{family}_{direction}_{dose}_ab"
        ),
        "reference_kind": (
            reference_kind
            if reference_kind is not None
            else (
                None
                if family == "clean"
                else CueReferenceKind.MODEL_CLEAN_VERDICT.value
            )
        ),
        "human_winner": "A",
        "verdict": verdict,
        "clean_tie": clean_tie,
        "msp": score,
    }


def base_records() -> tuple[tuple[ConditionRecord, ...], tuple[ConditionRecord, ...]]:
    clean_raw = [
        raw_record("clean-q1", "q1", routing_split="calibration", verdict="A"),
        raw_record("clean-q2", "q2", routing_split="calibration", verdict="B"),
        raw_record("clean-q3", "q3", routing_split="test", verdict="A"),
        raw_record("clean-q4", "q4", routing_split="test", verdict="A"),
    ]
    cued_raw = []
    for family, doses in (
        ("authority", AUTHORITY_DOSES),
        ("bandwagon", BANDWAGON_DOSES),
    ):
        for direction in ("congruent", "incongruent"):
            for dose in doses:
                for question_id in ("q3", "q4"):
                    cued_raw.append(
                        raw_record(
                            f"cued-{family}-{direction}-{dose}-{question_id}",
                            question_id,
                            routing_split="test",
                            verdict="B",
                            family=family,
                            direction=direction,
                            dose=dose,
                            clean_record_id=f"clean-{question_id}",
                        )
                    )
    return (
        tuple(record_from_mapping(row) for row in clean_raw),
        tuple(record_from_mapping(row) for row in cued_raw),
    )


def score_table(
    cued: tuple[ConditionRecord, ...],
) -> dict[str, dict[str, float | None]]:
    table: dict[str, dict[str, float | None]] = {
        "clean-q1": {"p1": 0.9, "p2": 0.1},
        "clean-q2": {"p1": 0.8, "p2": 0.9},
        "clean-q3": {"p1": 0.95, "p2": 0.7},
        "clean-q4": {"p1": 0.6, "p2": 0.8},
    }
    for record in cued:
        table[record.record_id] = (
            {"p1": 0.7, "p2": 0.6}
            if record.question_id == "q3"
            else {"p1": 0.95, "p2": None}
        )
    return table


def write_routing_package_fixture(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    source_path = tmp_path / "source.csv"
    pd.DataFrame(
        [
            {
                "question_id": "q1",
                "prompt": "Question 1",
                "response_a": "Answer A1",
                "response_b": "Answer B1",
                "winner": "model_a",
            },
            {
                "question_id": "q2",
                "prompt": "Question 2",
                "response_a": "Answer A2",
                "response_b": "Answer B2",
                "winner": "model_b",
            },
            {
                "question_id": "q-skipped",
                "prompt": "Skipped",
                "response_a": "",
                "response_b": "Answer B",
                "winner": "model_b",
            },
        ]
    ).to_csv(source_path, index=False)
    package = tmp_path / "routing"
    manifest = build_routing_package(
        source_csv=source_path,
        output_dir=package,
        dataset_lineage={"dataset": "fixture", "revision": "pinned"},
    )
    return package, manifest


def test_split_consumes_frozen_routes_and_matches_shared_hash() -> None:
    clean, _ = base_records()
    expected = routing_assignment_sha256(
        pd.DataFrame(
            [
                {"question_id": "q1", "routing_split": "calibration"},
                {"question_id": "q2", "routing_split": "calibration"},
                {"question_id": "q3", "routing_split": "test"},
                {"question_id": "q4", "routing_split": "test"},
            ]
        )
    )

    split = question_split_from_records(
        tuple(reversed(clean)),
        expected_raw_assignment_sha256=expected,
        expected_eligible_assignment_sha256=expected,
    )

    assert split.calibration_question_ids == ("q1", "q2")
    assert split.test_question_ids == ("q3", "q4")
    assert split.raw_assignment_sha256 == expected
    assert split.eligible_assignment_sha256 == expected
    assert split.raw_question_count == 4
    assert split.eligible_question_count == 4
    with pytest.raises(ValueError, match="raw routing assignment SHA-256 mismatch"):
        question_split_from_records(
            clean,
            expected_raw_assignment_sha256="0" * 64,
        )


def test_split_validates_eligible_records_as_subset_of_raw_routing() -> None:
    clean, _ = base_records()
    frozen = {
        "q1": "calibration",
        "q2": "calibration",
        "q3": "test",
        "q4": "test",
        "q-skipped": "test",
    }
    expected = routing_assignment_sha256(
        pd.DataFrame(
            [
                {"question_id": question_id, "routing_split": routing_split}
                for question_id, routing_split in frozen.items()
            ]
        )
    )

    eligible = {key: value for key, value in frozen.items() if key != "q-skipped"}
    eligible_expected = routing_assignment_sha256(
        pd.DataFrame(
            [
                {"question_id": question_id, "routing_split": routing_split}
                for question_id, routing_split in eligible.items()
            ]
        )
    )
    split = question_split_from_records(
        clean,
        expected_raw_assignment_sha256=expected,
        expected_eligible_assignment_sha256=eligible_expected,
        frozen_raw_question_assignments=frozen,
        frozen_eligible_question_assignments=eligible,
    )

    assert split.raw_assignment_sha256 == expected
    assert split.eligible_assignment_sha256 != expected
    assert split.raw_question_count == 5
    assert split.eligible_question_count == 4
    with pytest.raises(ValueError, match="does not exactly match"):
        question_split_from_records(
            tuple(record for record in clean if record.question_id != "q2"),
            frozen_raw_question_assignments=frozen,
            frozen_eligible_question_assignments=eligible,
        )
    mismatched = tuple(
        replace(record, routing_split="test")
        if record.question_id == "q2"
        else record
        for record in clean
    )
    with pytest.raises(ValueError, match=r"mismatched=\['q2'\]"):
        question_split_from_records(
            mismatched,
            frozen_raw_question_assignments=frozen,
            frozen_eligible_question_assignments=eligible,
        )


def test_routing_package_loader_verifies_raw_question_universe(
    tmp_path: Path,
) -> None:
    package, manifest = write_routing_package_fixture(tmp_path)
    manifest_path = package / MANIFEST_FILENAME

    raw, eligible, provenance, observed_path = load_frozen_routing_package(manifest_path)

    assert set(raw) == {"q1", "q2", "q-skipped"}
    assert set(eligible) == {"q1", "q2"}
    assert provenance["raw_routing_assignment_sha256"] == manifest[
        "routing_assignment_sha256"
    ]
    assert provenance["eligible_routing_assignment_sha256"] != provenance[
        "raw_routing_assignment_sha256"
    ]
    assert provenance["raw_question_count"] == 3
    assert provenance["eligible_question_count"] == 2
    assert provenance["eligibility_sha256"] == manifest["eligibility"][
        "eligibility_sha256"
    ]
    assert observed_path == (package / "routed_full.csv").resolve()


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda manifest: manifest.__setitem__("seed", 43), "seed=42"),
        (
            lambda manifest: manifest.__setitem__("calibration_fraction", 0.4),
            "calibration_fraction=0.5",
        ),
        (
            lambda manifest: manifest["outputs"]["full"].__setitem__(
                "rows", manifest["outputs"]["full"]["rows"] + 1
            ),
            r"outputs\.full\.rows does not match",
        ),
        (
            lambda manifest: manifest["question_counts"].__setitem__(
                "test", manifest["question_counts"]["test"] + 1
            ),
            "question_counts does not match",
        ),
        (
            lambda manifest: manifest["counts"]["eligible_questions"].__setitem__(
                "total", manifest["counts"]["eligible_questions"]["total"] + 1
            ),
            "raw/eligible/skipped counts do not match",
        ),
        (
            lambda manifest: manifest["eligibility"].__setitem__(
                "eligibility_sha256", "0" * 64
            ),
            "eligibility hash/audit does not match",
        ),
    ),
)
def test_routing_package_loader_rejects_manifest_drift(
    tmp_path: Path,
    mutation: object,
    message: str,
) -> None:
    package, _ = write_routing_package_fixture(tmp_path)
    manifest_path = package / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert callable(mutation)
    mutation(manifest)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_frozen_routing_package(manifest_path)


def test_hash_and_revision_validators_reject_noncanonical_case() -> None:
    assert validate_lowercase_sha256("a" * 64, name="fixture") == "a" * 64
    assert validate_model_revision("b" * 40) == "b" * 40
    with pytest.raises(ValueError, match="lowercase"):
        validate_lowercase_sha256("A" * 64, name="fixture")
    with pytest.raises(ValueError, match="lowercase"):
        validate_model_revision("B" * 40)
    clean, _ = base_records()
    with pytest.raises(ValueError, match="lowercase"):
        question_split_from_records(
            clean,
            expected_raw_assignment_sha256="A" * 64,
        )


def test_analysis_requires_one_pinned_model_identity() -> None:
    clean, cued = base_records()
    assert validate_single_model_identity(
        clean,
        cued,
        expected_model_name="model",
        expected_model_revision=MODEL_REVISION,
    ) == ("model", MODEL_REVISION)

    with pytest.raises(ValueError, match="model name mismatch"):
        validate_single_model_identity(
            clean,
            cued,
            expected_model_name="another/model",
        )
    with pytest.raises(ValueError, match="model revision mismatch"):
        validate_single_model_identity(
            clean,
            cued,
            expected_model_revision="b" * 40,
        )
    with pytest.raises(ValueError, match="multiple model identities"):
        validate_single_model_identity(
            clean,
            (replace(cued[0], model_revision="b" * 40), *cued[1:]),
        )
    with pytest.raises(ValueError, match="multiple model identities"):
        validate_single_model_identity(
            clean,
            (replace(cued[0], model_name="another/model"), *cued[1:]),
        )
    with pytest.raises(ValueError, match="does not pin"):
        validate_single_model_identity(
            (replace(clean[0], model_revision=None), *clean[1:]),
            cued,
        )
    with pytest.raises(ValueError, match="does not pin"):
        validate_single_model_identity(
            (replace(clean[0], model_revision="A" * 40), *clean[1:]),
            cued,
        )


def test_split_rejects_a_question_in_both_routes() -> None:
    clean, _ = base_records()
    duplicate = raw_record(
        "other-q1", "q1", routing_split="test", verdict="A"
    )

    with pytest.raises(ValueError, match="both routing splits"):
        question_split_from_records((*clean, record_from_mapping(duplicate)))


def test_exact_pairing_requires_explicit_clean_record_id() -> None:
    clean, cued = base_records()
    split = question_split_from_records(clean)
    malformed = raw_record(
        "cued-q3-bad",
        "q3",
        routing_split="test",
        verdict="B",
        family="authority",
        clean_record_id="wrong-id",
    )

    with pytest.raises(ValueError, match="exact clean_record_id"):
        exact_test_pairs(clean, (record_from_mapping(malformed), *cued[1:]), split)

    with pytest.raises(ValueError, match="model_revision"):
        exact_test_pairs(
            clean,
            (replace(cued[0], model_revision="b" * 40), *cued[1:]),
            split,
        )


def test_fallback_references_are_robustness_only_and_cannot_enter_primary() -> None:
    clean, cued = base_records()
    split = question_split_from_records(clean)

    malformed = replace(
        cued[0],
        reference_kind=CueReferenceKind.HUMAN_LABEL_FALLBACK.value,
    )
    with pytest.raises(
        ValueError,
        match="fallback-referenced.*would enter the primary target-bias cohort",
    ):
        exact_test_pairs(clean, (malformed,), split, exclude_clean_ties=True)

    tied_clean = tuple(
        replace(record, verdict="tie", clean_tie=True)
        if record.record_id == "clean-q3"
        else record
        for record in clean
    )
    fallback = replace(
        cued[0],
        clean_tie=True,
        reference_kind=CueReferenceKind.HUMAN_LABEL_FALLBACK.value,
    )
    robustness_pairs = exact_test_pairs(
        tied_clean,
        (fallback,),
        question_split_from_records(tied_clean),
        exclude_clean_ties=False,
    )
    assert len(robustness_pairs) == 1
    assert exact_test_pairs(
        tied_clean,
        (fallback,),
        question_split_from_records(tied_clean),
        exclude_clean_ties=True,
    ) == ()


def test_report_labels_tie_fallback_rows_without_changing_full_grid() -> None:
    clean, cued = base_records()
    tied_clean = tuple(
        replace(record, verdict="tie", clean_tie=True)
        if record.record_id == "clean-q3"
        else record
        for record in clean
    )
    cued_with_fallback = tuple(
        replace(
            record,
            clean_tie=True,
            reference_kind=CueReferenceKind.HUMAN_LABEL_FALLBACK.value,
        )
        if record.question_id == "q3"
        else record
        for record in cued
    )

    report = controlled_uncertainty_shift_report(
        tied_clean,
        cued_with_fallback,
        score_table(cued_with_fallback),
        (PredictorSpec("p1"),),
        target_risks=(0.10,),
        n_resamples=2,
    )

    assert len(report["groups"]) == 16
    assert all(group["full_test_grid_pair_n"] == 2 for group in report["groups"])
    assert all(group["structural_pair_n"] == 1 for group in report["groups"])
    assert all(group["primary_target_bias_pair_n"] == 1 for group in report["groups"])
    assert all(
        group["fallback_reference_robustness_pair_n"] == 1
        for group in report["groups"]
    )
    assert all(
        group["full_test_reference_kind_counts"]
        == {
            CueReferenceKind.HUMAN_LABEL_FALLBACK.value: 1,
            CueReferenceKind.MODEL_CLEAN_VERDICT.value: 1,
        }
        for group in report["groups"]
    )
    assert all(
        group["estimand_kind"] == "primary_target_bias"
        and group["reference_kind"]
        == CueReferenceKind.MODEL_CLEAN_VERDICT.value
        for group in report["groups"]
    )


@pytest.mark.parametrize(
    ("question_id", "routing_split"),
    (("q1", "calibration"), ("q3", "calibration")),
)
def test_exact_pairing_rejects_every_non_test_cued_record(
    question_id: str,
    routing_split: str,
) -> None:
    clean, cued = base_records()
    split = question_split_from_records(clean)
    leaked = raw_record(
        "leaked-cue",
        question_id,
        routing_split=routing_split,
        verdict="B",
        family="authority",
        clean_record_id=f"clean-{question_id}",
    )

    with pytest.raises(ValueError, match="outside the frozen test split"):
        exact_test_pairs(clean, (*cued, record_from_mapping(leaked)), split)


def test_report_rejects_incomplete_or_duplicated_condition_grid() -> None:
    clean, cued = base_records()
    scores = score_table(cued)

    with pytest.raises(ValueError, match="does not match the structural test cohort"):
        controlled_uncertainty_shift_report(
            clean,
            cued[:-1],
            scores,
            (PredictorSpec("p1"),),
            target_risks=(0.10,),
            n_resamples=2,
        )

    duplicate = replace(cued[0], record_id="duplicate-cued-record")
    with pytest.raises(ValueError, match="duplicate clean/cued links"):
        controlled_uncertainty_shift_report(
            clean,
            (*cued, duplicate),
            {**scores, duplicate.record_id: {"p1": 0.7}},
            (PredictorSpec("p1"),),
            target_risks=(0.10,),
            n_resamples=2,
        )


def test_predictors_are_separate_and_zero_coverage_risk_is_undefined() -> None:
    clean, cued = base_records()
    report = controlled_uncertainty_shift_report(
        clean,
        cued,
        score_table(cued),
        (PredictorSpec("p1"), PredictorSpec("p2")),
        target_risks=(0.10,),
        n_resamples=40,
        seed=7,
    )

    assert report["combined_predictors"] == []
    assert report["model"] == {
        "name": "model",
        "revision": MODEL_REVISION,
    }
    results = {
        row["predictor"]: row
        for row in report["groups"][0]["predictor_results"]
    }
    assert set(results) == {"p1", "p2"}
    assert results["p1"]["matched_score_pair_n"] == 2
    assert results["p1"]["acceptance_transitions"] == {
        "clean_accepted__cued_accepted": 0,
        "clean_accepted__cued_rejected": 1,
        "clean_rejected__cued_accepted": 1,
        "clean_rejected__cued_rejected": 0,
    }
    assert results["p2"]["score_availability"]["transitions"] == {
        "both": 1,
        "clean_only": 1,
        "cued_only": 0,
        "neither": 0,
    }
    assert results["p2"]["rule"]["threshold"] is None
    assert results["p2"]["clean_test"]["coverage_among_matched_scores"] == 0.0
    assert results["p2"]["clean_test"]["coverage_among_structural_pairs"] == 0.0
    assert results["p2"]["clean_test"]["risk"] is None
    assert results["p2"]["cued_test"]["coverage_among_matched_scores"] == 0.0
    assert results["p2"]["cued_test"]["coverage_among_structural_pairs"] == 0.0
    assert results["p2"]["cued_test"]["risk"] is None
    assert results["p2"]["score_availability"] == {
        "structural_pair_n": 2,
        "clean_score_available_n": 2,
        "cued_score_available_n": 1,
        "jointly_available_n": 1,
        "clean_score_availability_among_structural_pairs": 1.0,
        "cued_score_availability_among_structural_pairs": 0.5,
        "joint_availability_among_structural_pairs": 0.5,
        "transitions": {
            "both": 1,
            "clean_only": 1,
            "cued_only": 0,
            "neither": 0,
        },
    }
    assert report["configuration"]["calibration_clean_ties"].startswith(
        "included"
    )
    assert len(report["groups"]) == 16
    assert len(
        {group["structural_clean_population_sha256"] for group in report["groups"]}
    ) == 1


def test_calibration_ties_are_included_and_bootstrap_schedule_is_shared() -> None:
    clean, cued = base_records()
    clean_with_calibration_tie = (
        clean[0],
        replace(clean[1], clean_tie=True),
        *clean[2:],
    )
    report = controlled_uncertainty_shift_report(
        clean_with_calibration_tie,
        cued,
        score_table(cued),
        (PredictorSpec("p1"),),
        target_risks=(0.10,),
        n_resamples=40,
        seed=11,
    )

    results = [group["predictor_results"][0] for group in report["groups"]]
    assert all(result["rule"]["calibration_population_n"] == 2 for result in results)
    assert len(
        {
            result["bootstrap"]["calibration_rule_schedule_sha256"]
            for result in results
        }
    ) == 1
    assert len(
        {json.dumps(result["bootstrap"]["threshold"], sort_keys=True) for result in results}
    ) == 1
    assert report["configuration"]["calibration_clean_ties"] == (
        "included_to_match_prior_threshold_fitting_estimand"
    )


def test_joint_bootstrap_refits_calibration_and_is_deterministic() -> None:
    clean, cued = base_records()
    split = question_split_from_records(clean)
    pairs = exact_test_pairs(clean, cued, split)
    calibration = tuple(
        record for record in clean if record.routing_split == "calibration"
    )

    first = analyze_matched_group(
        calibration,
        pairs,
        score_table(cued),
        PredictorSpec("p1"),
        target_risk=0.10,
        n_resamples=80,
        seed=17,
    )
    second = analyze_matched_group(
        calibration,
        pairs,
        score_table(cued),
        PredictorSpec("p1"),
        target_risk=0.10,
        n_resamples=80,
        seed=17,
    )

    assert first["bootstrap"] == second["bootstrap"]
    assert first["bootstrap"]["inference_scope"].startswith("shared_clean")
    assert first["bootstrap"]["calibration_clusters"] == 2
    assert 0 < first["bootstrap"]["threshold"]["finite_resamples"] < 80
    assert first["bootstrap"]["risk_difference"]["finite_resamples"] <= 80


def test_score_loader_rejects_composite_names_and_conflicting_sources() -> None:
    predictors, fields = parse_predictor_declarations(
        ["msp=msp", "entropy=scores.entropy"],
        {"entropy"},
    )
    assert [predictor.name for predictor in predictors] == ["msp", "entropy"]
    assert predictors[1].higher_is_more_confident is False
    assert fields == {"msp": "msp", "entropy": "scores.entropy"}
    with pytest.raises(ValueError, match="duplicate predictor"):
        parse_predictor_declarations(["msp=msp", "msp=other"], set())
    with pytest.raises(ValueError, match="conflicting msp"):
        build_score_table(
            [
                [{"record_id": "row", "msp": 0.8}],
                [{"record_id": "row", "msp": 0.9}],
            ],
            {"msp": "msp"},
        )


def test_output_is_no_clobber(tmp_path) -> None:
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(json.dumps({"record_id": "row"}) + "\n")
    output = tmp_path / "report.json"
    validate_output_path(output, [input_path])
    output.write_text("preserve")

    with pytest.raises(FileExistsError, match="overwrite"):
        validate_output_path(output, [input_path])
    with pytest.raises(ValueError, match="differ"):
        validate_output_path(input_path, [input_path])
