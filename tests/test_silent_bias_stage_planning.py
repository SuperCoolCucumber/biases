from __future__ import annotations

from biases.pairing import normalize_ordering
from biases.schemas import CueCongruency, PairOrdering
from biases.stage_planning import (
    CleanPairSummary,
    StageAPairInput,
    generate_stage_a_conditions,
    generate_stage_b_conditions,
)


def _stage_a(human_winner: str = "A"):
    return generate_stage_a_conditions(
        [
            StageAPairInput(
                dataset_name="fixture.csv",
                input_file_hash="c" * 64,
                source_row_index=7,
                question_id="q7",
                model_name="fixture-judge",
                human_winner=human_winner,
                turn=1,
                response_a_id="q7:a",
                response_b_id="q7:b",
            )
        ]
    )


def _clean_summaries(
    *,
    clean_ab: str = "A",
    clean_ba: str = "B",
    human_winner: str = "A",
) -> tuple[CleanPairSummary, ...]:
    plan = _stage_a(human_winner)
    clean_by_order = {
        PairOrdering.AB: clean_ab,
        PairOrdering.BA: clean_ba,
    }
    summaries: list[CleanPairSummary] = []
    for item in plan.conditions:
        ordering = normalize_ordering(item.condition.ordering or "")
        summaries.append(
            CleanPairSummary(
                pair_identity_key=item.pair_identity_key,
                pair_key=item.pair_key,
                ordering=ordering,
                ordering_twin_key=item.ordering_twin_key,
                model_name=item.model_name,
                input_file_hash=item.input_file_hash,
                clean_record_id=f"clean-{ordering.value}",
                clean_verdict=clean_by_order[ordering],
                human_winner=str(item.condition.metadata["human_winner"]),
            )
        )
    return tuple(summaries)


def test_stage_a_generates_clean_ab_and_ba_with_symmetric_links() -> None:
    plan = _stage_a()

    assert not plan.issues
    assert len(plan.conditions) == 2
    by_order = {
        normalize_ordering(item.condition.ordering or ""): item
        for item in plan.conditions
    }
    assert by_order[PairOrdering.AB].condition.variant_id == "clean_ab"
    assert by_order[PairOrdering.BA].condition.variant_id == "clean_ba"
    assert by_order[PairOrdering.AB].ordering_twin_key == by_order[PairOrdering.BA].pair_key
    assert by_order[PairOrdering.BA].ordering_twin_key == by_order[PairOrdering.AB].pair_key
    assert by_order[PairOrdering.AB].condition_group_id == (
        by_order[PairOrdering.BA].condition_group_id
    )
    assert by_order[PairOrdering.AB].condition.metadata["human_winner"] == "A"
    assert by_order[PairOrdering.BA].condition.metadata["human_winner"] == "B"


def test_stage_b_generates_full_grid_for_both_orderings() -> None:
    plan = generate_stage_b_conditions(_clean_summaries())

    assert not plan.issues
    assert len(plan.conditions) == 32
    selected = [
        item
        for item in plan.conditions
        if item.condition.bias_type == "authority"
        and item.condition.cue_congruency == CueCongruency.INCONGRUENT.value
        and item.condition.dose == 1
    ]
    assert len(selected) == 2
    by_order = {
        normalize_ordering(item.condition.ordering or ""): item
        for item in selected
    }
    assert by_order[PairOrdering.AB].condition.cue_target == "B"
    assert by_order[PairOrdering.BA].condition.cue_target == "A"
    assert by_order[PairOrdering.AB].condition.direction_relative_human == "against_human"
    assert by_order[PairOrdering.BA].condition.direction_relative_human == "against_human"
    assert by_order[PairOrdering.AB].condition_group_id == (
        by_order[PairOrdering.BA].condition_group_id
    )


def test_clean_ties_use_the_order_specific_human_label() -> None:
    plan = generate_stage_b_conditions(
        _clean_summaries(clean_ab="tie", clean_ba="tie")
    )

    assert not plan.issues
    congruent_low = [
        item
        for item in plan.conditions
        if item.condition.bias_type == "authority"
        and item.condition.cue_congruency == CueCongruency.CONGRUENT.value
        and item.condition.dose == 1
    ]
    by_order = {
        normalize_ordering(item.condition.ordering or ""): item
        for item in congruent_low
    }
    assert by_order[PairOrdering.AB].condition.cue_target == "A"
    assert by_order[PairOrdering.BA].condition.cue_target == "B"
    assert all(item.condition.clean_tie for item in plan.conditions)
    assert all(
        item.condition.metadata["direction_reference"] == "human_label_for_clean_tie"
        for item in plan.conditions
    )


def test_simultaneous_clean_and_human_ties_are_reported_not_dropped_silently() -> None:
    plan = generate_stage_b_conditions(
        _clean_summaries(
            clean_ab="tie",
            clean_ba="tie",
            human_winner="tie",
        )
    )

    assert len(plan.conditions) == 32
    assert [issue.code for issue in plan.issues] == ["clean_and_human_tie"]
    assert all(
        item.condition.direction_relative_human == "human_tie"
        for item in plan.conditions
    )
    assert all(
        item.condition.metadata["direction_reference"]
        == "deterministic_pair_hash_for_clean_and_human_tie"
        for item in plan.conditions
    )


def test_stage_b_refuses_a_partial_ordering_group() -> None:
    summaries = _clean_summaries()

    plan = generate_stage_b_conditions([summaries[0]])

    assert not plan.conditions
    assert any(issue.code == "missing_clean_ordering" for issue in plan.issues)
