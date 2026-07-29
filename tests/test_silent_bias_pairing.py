from __future__ import annotations

from biases.pairing import (
    make_condition_group_id,
    make_ordering_twin_key,
    make_pair_identity_key,
    make_pair_key,
)
from biases.schemas import ConsistencyMetrics, PairOrdering, RunRecord


def test_legacy_run_record_remains_valid_without_linkage_fields() -> None:
    record = RunRecord.model_validate(
        {
            "record_id": "legacy-record",
            "spec": {
                "dataset_name": "fixture.csv",
                "dataset_split": "test",
                "model_name": "fixture-model",
                "backend_name": "manual",
                "bias_name": "authority",
                "output_mode": "choice_only",
                "uncertainty_methods": ["logit"],
                "consistency_runs": 1,
                "temperature": 0.0,
            },
            "example_id": "q1:original",
            "question_id": "q1",
            "condition": {
                "bias_type": "authority",
                "variant_id": "control",
            },
            "seed": 0,
            "verdict": "A",
            "raw_output": "A",
            "prompt_hash": "prompt-hash",
            "uncertainty": {},
        }
    )

    assert record.pair_key is None
    assert record.condition_group_id is None
    assert record.ordering_twin_key is None
    assert record.spec_hash is None
    assert record.input_file_hash is None


def test_consistency_majority_verdict_is_optional() -> None:
    legacy = ConsistencyMetrics(
        run_count=4,
        agreement_rate=0.75,
        vote_entropy=0.811,
        unique_verdict_count=2,
        flip_rate=0.25,
        verdict_counts={"A": 3, "B": 1},
    )
    enriched = ConsistencyMetrics(
        **legacy.model_dump(exclude={"majority_verdict"}),
        majority_verdict="A",
    )

    assert legacy.majority_verdict is None
    assert enriched.model_dump(mode="json")["majority_verdict"] == "A"


def test_pair_identity_is_collision_safe_for_repeated_question_rows() -> None:
    common = {
        "dataset_name": "mtbench_full.csv",
        "input_file_hash": "a" * 64,
        "question_id": "42",
        "turn": "2",
    }
    first = make_pair_identity_key(source_row_index=10, **common)
    second = make_pair_identity_key(source_row_index=11, **common)

    assert first != second
    assert first == make_pair_identity_key(source_row_index=10, **common)


def test_pair_keys_link_clean_conditions_and_ordering_twins() -> None:
    identity = make_pair_identity_key(
        dataset_name="fixture.csv",
        input_file_hash="b" * 64,
        source_row_index=3,
        question_id="q3",
    )
    ab = make_pair_key(
        pair_identity_key=identity,
        model_name="judge",
        ordering=PairOrdering.AB,
    )
    ba = make_pair_key(
        pair_identity_key=identity,
        model_name="judge",
        ordering=PairOrdering.BA,
    )

    assert ab != ba
    assert (
        make_ordering_twin_key(
            pair_identity_key=identity,
            model_name="judge",
            ordering=PairOrdering.AB,
        )
        == ba
    )
    condition_group = make_condition_group_id(
        pair_identity_key=identity,
        model_name="judge",
        family="authority",
        direction="incongruent",
        dose=1,
    )
    assert condition_group == make_condition_group_id(
        pair_identity_key=identity,
        model_name="judge",
        family="authority",
        direction="incongruent",
        dose=1,
    )
    assert condition_group != make_condition_group_id(
        pair_identity_key=identity,
        model_name="judge",
        family="authority",
        direction="incongruent",
        dose=2,
    )
