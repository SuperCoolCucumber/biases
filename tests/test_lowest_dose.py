from __future__ import annotations

from dataclasses import replace
import math

import pytest

from biases.analysis.lowest_dose import (
    build_lowest_dose_rows,
    summarize_lowest_dose,
)
from biases.analysis.records import ConditionRecord, record_from_mapping


def clean_record(
    pair: str = "pair0",
    *,
    question: str = "q1",
    ordering: str = "ab",
    verdict: str = "A",
    human: str = "A",
    probabilities: tuple[float, float, float] = (0.80, 0.15, 0.05),
) -> ConditionRecord:
    return record_from_mapping(
        {
            "record_id": f"clean-{pair}-{ordering}",
            "example_id": f"{pair}-{ordering}",
            "question_id": question,
            "pair_identity_key": pair,
            "pair_key": f"judge-{pair}-{ordering}",
            "ordering": ordering,
            "model_name": "fixture-judge",
            "model_revision": "a" * 40,
            "routing_split": "test",
            "bias_name": "clean",
            "variant_id": "clean",
            "cue_congruency": "clean",
            "human_winner": human,
            "verdict": verdict,
            "label_prob_A": probabilities[0],
            "label_prob_B": probabilities[1],
            "label_prob_tie": probabilities[2],
            "msp": max(probabilities),
        }
    )


def cued_record(
    clean: ConditionRecord,
    target: str,
    *,
    verdict: str | None = None,
    probabilities: tuple[float, float, float] | None = None,
    family: str = "authority",
) -> ConditionRecord:
    if clean.verdict != "tie":
        reference, kind = clean.verdict, "model_clean_verdict"
    elif clean.human_winner != "tie":
        reference, kind = clean.human_winner, "human_label_fallback"
    else:
        reference, kind = "A", "deterministic_fallback"
    direction = "congruent" if target == reference else "incongruent"
    dose = 1 if family == "authority" else 55
    probs = probabilities or (
        clean.probability_a,
        clean.probability_b,
        clean.probability_tie,
    )
    return replace(
        clean,
        record_id=f"{family}-{clean.pair_identity_key}-{clean.ordering}-{target}",
        clean_record_id=clean.record_id,
        family=family,
        direction=direction,
        dose=float(dose),
        variant_id=f"{family}_{direction}_{dose}_{clean.ordering}",
        cue_target=target,
        reference_kind=kind,
        clean_tie=clean.verdict == "tie",
        verdict=verdict or clean.verdict,
        probability_a=probs[0],
        probability_b=probs[1],
        probability_tie=probs[2],
        msp=max(probs),
    )


def partial_rows(
    clean: tuple[ConditionRecord, ...], cued: tuple[ConditionRecord, ...]
) -> tuple[dict, ...]:
    return build_lowest_dose_rows(clean, cued, require_complete_grid=False)


@pytest.mark.parametrize("family", ("authority", "bandwagon"))
def test_confident_reversal_can_leave_msp_and_entropy_unchanged(family: str) -> None:
    clean = clean_record()
    cued = cued_record(
        clean,
        "B",
        verdict="B",
        probabilities=(0.15, 0.80, 0.05),
        family=family,
    )
    rows = partial_rows((clean,), (cued,))

    assert len(rows) == 1
    row = rows[0]
    assert row["event"] == "decisive_reversal"
    assert row["alignment_transition"] == "new_error"
    assert row["human_direction"] == "incongruent"
    assert row["toward_cue"] is True
    assert row["delta_msp_pp"] == pytest.approx(0)
    assert row["delta_entropy_bits"] == pytest.approx(0)
    assert row["delta_cue_probability_pp"] == pytest.approx(65)
    assert row["delta_human_probability_pp"] == pytest.approx(-65)


def test_human_alignment_uses_labels_in_each_answer_order() -> None:
    ab = clean_record()
    ba = clean_record(
        ordering="ba", verdict="B", human="B", probabilities=(0.15, 0.80, 0.05)
    )
    rows = partial_rows(
        (ab, ba),
        (
            cued_record(ab, "B", verdict="B", probabilities=(0.15, 0.80, 0.05)),
            cued_record(ba, "A", verdict="A", probabilities=(0.80, 0.15, 0.05)),
        ),
    )

    assert len(rows) == 2
    for row in rows:
        assert row["clean_state"] == "agrees"
        assert row["human_direction"] == "incongruent"
        assert row["alignment_transition"] == "new_error"
        assert row["delta_human_probability_pp"] == pytest.approx(-65)


def test_human_congruent_cue_corrects_an_initially_wrong_judgment() -> None:
    clean = clean_record(verdict="B", probabilities=(0.15, 0.80, 0.05))
    cued = cued_record(clean, "A", verdict="A", probabilities=(0.80, 0.15, 0.05))
    row = partial_rows((clean,), (cued,))[0]

    assert cued.direction == "incongruent"
    assert row["human_direction"] == "congruent"
    assert row["clean_state"] == "disagrees"
    assert row["alignment_transition"] == "correction"
    assert row["delta_human_probability_pp"] == pytest.approx(65)


@pytest.mark.parametrize(
    ("clean_verdict", "cued_verdict", "clean_probs", "cued_probs", "event"),
    (
        ("A", "tie", (0.80, 0.15, 0.05), (0.15, 0.05, 0.80), "to_tie"),
        ("tie", "B", (0.15, 0.05, 0.80), (0.15, 0.80, 0.05), "from_tie"),
        ("tie", "tie", (0.15, 0.05, 0.80), (0.20, 0.10, 0.70), "stable"),
    ),
)
def test_tie_transitions_are_not_decisive_reversals(
    clean_verdict: str,
    cued_verdict: str,
    clean_probs: tuple[float, float, float],
    cued_probs: tuple[float, float, float],
    event: str,
) -> None:
    clean = clean_record(verdict=clean_verdict, probabilities=clean_probs)
    cued = cued_record(clean, "B", verdict=cued_verdict, probabilities=cued_probs)
    row = partial_rows((clean,), (cued,))[0]

    assert row["event"] == event
    if clean_verdict == "tie":
        assert row["clean_state"] == "tie"


def test_human_ties_have_no_binary_human_congruence_or_probability_shift() -> None:
    clean = clean_record(human="tie")
    row = partial_rows((clean,), (cued_record(clean, "A"),))[0]

    assert row["human_direction"] == "human_tie"
    assert row["clean_state"] == "human_tie"
    assert row["delta_human_probability_pp"] is None


def test_verdict_probability_disagreement_is_preserved_and_auditable() -> None:
    clean = clean_record(verdict="B")
    row = partial_rows((clean,), (cued_record(clean, "A", verdict="A"),))[0]

    assert row["event"] == "decisive_reversal"
    assert row["clean_map_verdict"] == "A"
    assert row["clean_verdict_matches_map"] is False
    assert row["cued_verdict_matches_map"] is True
    assert row["delta_msp_pp"] == pytest.approx(0)


@pytest.mark.parametrize(
    "probs",
    (
        (-0.10, 0.80, 0.30),
        (1.10, 0.00, -0.10),
        (0.20, 0.20, 0.20),
        (0.00, 0.00, 0.00),
        (math.nan, 0.80, 0.20),
        (math.inf, 0.00, 0.00),
    ),
)
@pytest.mark.parametrize("invalid_side", ("clean", "cued"))
def test_invalid_probabilities_are_rejected_without_silent_normalization(
    probs: tuple[float, float, float], invalid_side: str
) -> None:
    clean = clean_record()
    cued = cued_record(clean, "B")
    invalid = replace(
        clean if invalid_side == "clean" else cued,
        probability_a=probs[0],
        probability_b=probs[1],
        probability_tie=probs[2],
    )
    if invalid_side == "clean":
        clean = invalid
    else:
        cued = invalid

    with pytest.raises(ValueError):
        partial_rows((clean,), (cued,))


def test_missing_probability_is_rejected() -> None:
    clean = clean_record()
    cued = replace(cued_record(clean, "B"), probability_tie=None)
    with pytest.raises(ValueError):
        partial_rows((clean,), (cued,))


def test_missing_or_wrong_clean_partner_is_rejected() -> None:
    clean = clean_record()
    cued = replace(cued_record(clean, "B"), clean_record_id="unavailable-clean")
    with pytest.raises(ValueError):
        partial_rows((clean,), (cued,))


@pytest.mark.parametrize("duplicate_side", ("clean", "cued"))
def test_duplicate_records_are_rejected(duplicate_side: str) -> None:
    clean = clean_record()
    cued = cued_record(clean, "B")
    with pytest.raises(ValueError):
        partial_rows(
            (clean, clean) if duplicate_side == "clean" else (clean,),
            (cued, cued) if duplicate_side == "cued" else (cued,),
        )


def test_duplicate_target_is_rejected_even_when_record_ids_differ() -> None:
    clean = clean_record()
    cued = cued_record(clean, "B")
    duplicate = replace(cued, record_id="second-record-with-the-same-condition")
    with pytest.raises(ValueError):
        partial_rows((clean,), (cued, duplicate))


def complete_grid() -> tuple[tuple[ConditionRecord, ...], tuple[ConditionRecord, ...]]:
    clean = (
        clean_record(),
        clean_record(
            ordering="ba", verdict="B", human="B", probabilities=(0.15, 0.80, 0.05)
        ),
    )
    cued = tuple(
        cued_record(record, target, family=family)
        for record in clean
        for family in ("authority", "bandwagon")
        for target in ("A", "B")
    )
    return clean, cued


def test_complete_grid_has_exactly_two_targets_for_each_family_and_order() -> None:
    clean, cued = complete_grid()
    rows = build_lowest_dose_rows(
        clean,
        cued,
        expected_test_pairs_per_model=1,
        expected_question_ids=("q1",),
    )

    assert len(rows) == 8

    with pytest.raises(ValueError):
        build_lowest_dose_rows(clean, cued[:-1])


def test_complete_grid_rejects_missing_order_even_with_all_remaining_cues() -> None:
    clean, cued = complete_grid()
    with pytest.raises(ValueError):
        build_lowest_dose_rows(clean[:1], tuple(r for r in cued if r.ordering == "ab"))


def test_manifest_population_expectations_are_enforced() -> None:
    clean, cued = complete_grid()
    with pytest.raises(ValueError):
        build_lowest_dose_rows(clean, cued, expected_test_pairs_per_model=2)
    with pytest.raises(ValueError):
        build_lowest_dose_rows(clean, cued, expected_question_ids=("q1", "q2"))


def test_higher_doses_cannot_substitute_for_a_missing_lowest_dose_target() -> None:
    clean, cued = complete_grid()
    higher_dose = replace(cued[-1], dose=70, variant_id="bandwagon_incongruent_70_ba")

    with pytest.raises(ValueError):
        build_lowest_dose_rows(clean, (*cued[:-1], higher_dose))


def test_calibration_and_higher_doses_do_not_enter_test_analysis() -> None:
    clean, cued = complete_grid()
    calibration = replace(
        clean_record("calibration0", question="q-calibration"),
        routing_split="calibration",
    )
    higher_dose = replace(
        cued[0], record_id="authority-dose-2", dose=2,
        variant_id="authority_congruent_2_ab",
    )
    rows = build_lowest_dose_rows(
        (*clean, calibration),
        (*cued, cued_record(calibration, "B"), higher_dose),
        expected_test_pairs_per_model=1,
        expected_question_ids=("q1",),
    )

    assert len(rows) == 8
    assert {row["question_id"] for row in rows} == {"q1"}


def two_pair_rows(
    *, copies: int = 1, same_question: bool = False
) -> tuple[dict, ...]:
    """One reversed judgment and one stable judgment, both with both cue targets."""
    clean = []
    cued = []
    for copy in range(copies):
        first = clean_record(f"reversal{copy}", question="q1")
        second = clean_record(
            f"stable{copy}", question="q1" if same_question else "q2"
        )
        clean.extend((first, second))
        cued.extend(
            (
                cued_record(first, "A", probabilities=(0.90, 0.05, 0.05)),
                cued_record(
                    first, "B", verdict="B", probabilities=(0.15, 0.80, 0.05)
                ),
                cued_record(second, "A", probabilities=(0.85, 0.10, 0.05)),
                cued_record(second, "B", probabilities=(0.70, 0.25, 0.05)),
            )
        )
    return partial_rows(tuple(clean), tuple(cued))


def find_summary(table: list[dict], **criteria: object) -> dict:
    matching = [row for row in table if all(row.get(key) == value for key, value in criteria.items())]
    assert len(matching) == 1, f"Expected exactly one summary for {criteria}: {matching}"
    return matching[0]


def test_reversal_rate_uses_all_eligible_judgments_while_shift_uses_reversals() -> None:
    summary = summarize_lowest_dose(two_pair_rows(), bootstrap_draws=100)
    group = dict(
        model_name="fixture-judge", family="authority", ordering="pooled",
        human_direction="incongruent", clean_state="agrees",
    )
    rate = find_summary(summary["transition_rates"], **group, metric="decisive_reversal_rate")
    shift = find_summary(
        summary["flip_shifts"], **group,
        event="decisive_reversal", metric="delta_human_probability_pp",
    )

    assert rate["n"] == 2
    assert rate["numerator"] == 1
    assert rate["estimate"] == pytest.approx(0.5)
    assert shift["n"] == 1
    assert shift["n_eligible"] == 2
    assert shift["estimate"] == pytest.approx(-65)
    assert shift["median"] == pytest.approx(-65)
    assert shift["q25"] == shift["q75"] == pytest.approx(-65)


@pytest.mark.parametrize(
    ("metric", "expected"),
    (("delta_msp_pp", 12.5), ("delta_human_probability_pp", 45.0)),
)
def test_paired_contrasts_include_stable_verdicts(metric: str, expected: float) -> None:
    summary = summarize_lowest_dose(two_pair_rows(), bootstrap_draws=100)
    contrast = find_summary(
        summary["paired_contrasts"],
        model_name="fixture-judge", family="authority", ordering="pooled",
        clean_state="agrees", metric=metric,
    )

    assert contrast["n"] == 2
    assert contrast["n_pairs"] == 2
    assert contrast["n_questions"] == 2
    assert contrast["estimate"] == pytest.approx(expected)


def test_bootstrap_keeps_repeated_pairs_with_their_question_cluster() -> None:
    original = summarize_lowest_dose(two_pair_rows(), bootstrap_draws=200, seed=17)
    repeated = summarize_lowest_dose(two_pair_rows(copies=4), bootstrap_draws=200, seed=17)
    filters = dict(
        model_name="fixture-judge", family="authority", ordering="pooled",
        clean_state="agrees", metric="delta_human_probability_pp",
    )
    first = find_summary(original["paired_contrasts"], **filters)
    second = find_summary(repeated["paired_contrasts"], **filters)

    assert first["n"] == 2
    assert second["n"] == 8
    assert first["n_questions"] == second["n_questions"] == 2
    assert first["n_bootstrap_valid"] == second["n_bootstrap_valid"] == 200
    assert first["estimate"] == second["estimate"] == pytest.approx(45)
    assert first["ci_low"] == second["ci_low"] == pytest.approx(15)
    assert first["ci_high"] == second["ci_high"] == pytest.approx(75)
    assert repeated["audit"]["bootstrap_unit"] == "question"


def test_empty_flip_groups_remain_explicit_without_an_invented_zero_shift() -> None:
    summary = summarize_lowest_dose(two_pair_rows(), bootstrap_draws=100)
    empty = find_summary(
        summary["flip_shifts"],
        model_name="fixture-judge", family="authority", ordering="pooled",
        human_direction="congruent", clean_state="agrees",
        event="decisive_reversal", metric="delta_msp_pp",
    )

    assert empty["n_eligible"] == 2
    assert empty["n"] == 0
    assert empty["estimate"] is None
    assert empty["ci_low"] is None
    assert empty["ci_high"] is None
    assert empty["n_bootstrap_valid"] == 0
    assert empty["n_bootstrap_nonfinite"] == 100


def test_paired_summary_rejects_missing_human_congruence_partner() -> None:
    clean = clean_record()
    rows = partial_rows((clean,), (cued_record(clean, "A"),))
    with pytest.raises(ValueError):
        summarize_lowest_dose(rows, bootstrap_draws=10)


def test_full_grid_summary_counts_underlying_pairs_once_across_answer_orders() -> None:
    clean, cued = complete_grid()
    rows = build_lowest_dose_rows(clean, cued)
    summary = summarize_lowest_dose(rows, bootstrap_draws=10)
    rate = find_summary(
        summary["transition_rates"],
        model_name="fixture-judge", family="authority", ordering="pooled",
        human_direction="all", clean_state="all", metric="stable_rate",
    )

    assert rate["n"] == 4
    assert rate["n_pairs"] == 1
    assert summary["audit"]["models"][0]["n_pairs"] == 1


def test_confidence_bin_boundary_is_counted_exactly_once() -> None:
    clean = clean_record(probabilities=(0.90, 0.05, 0.05))
    rows = partial_rows((clean,), (cued_record(clean, "A"), cued_record(clean, "B")))
    summary = summarize_lowest_dose(rows, bootstrap_draws=10)
    bins = [
        row for row in summary["confidence_bins"]
        if row["model_name"] == "fixture-judge" and row["family"] == "authority"
        and row["ordering"] == "pooled" and row["human_direction"] == "all"
        and row["clean_state"] == "all"
    ]

    assert sum(row["n"] for row in bins) == 2
    populated = [row for row in bins if row["n"]]
    assert len(populated) == 1
    assert populated[0]["bin_lower"] == 0.90


def test_equal_question_sensitivity_does_not_overweight_a_question_with_more_pairs() -> None:
    # q1 contributes two identical contrasts of 75pp; q2 contributes one of 15pp.
    rows = tuple(row for row in two_pair_rows(copies=2) if row["pair_identity_key"] != "stable1")
    summary = summarize_lowest_dose(rows, bootstrap_draws=100)
    contrast = find_summary(
        summary["paired_contrasts"],
        model_name="fixture-judge", family="authority", ordering="pooled",
        clean_state="agrees", metric="delta_human_probability_pp",
    )

    assert contrast["n"] == 3
    assert contrast["n_questions"] == 2
    assert contrast["estimate"] == pytest.approx(55)
    assert contrast["equal_question_estimate"] == pytest.approx(45)


def test_frozen_question_universe_retains_questions_without_eligible_rows() -> None:
    summary = summarize_lowest_dose(
        two_pair_rows(), bootstrap_draws=200, seed=17,
        expected_question_ids=("q1", "q2", "q3"),
    )
    contrast = find_summary(
        summary["paired_contrasts"],
        model_name="fixture-judge", family="authority", ordering="pooled",
        clean_state="agrees", metric="delta_human_probability_pp",
    )

    assert contrast["n_questions"] == 2
    assert contrast["n_bootstrap_questions"] == 3
    assert contrast["n_bootstrap_nonfinite"] > 0
    assert contrast["n_bootstrap_valid"] + contrast["n_bootstrap_nonfinite"] == 200
    assert contrast["estimate"] == pytest.approx(45)

    with pytest.raises(ValueError):
        summarize_lowest_dose(two_pair_rows(), expected_question_ids=("q1",))


def test_two_models_match_on_underlying_pairs_despite_distinct_condition_keys() -> None:
    clean, cued = complete_grid()
    other_clean = tuple(
        replace(
            record, model_name="another-fixture-judge",
            record_id=f"other-{record.record_id}", pair_key=f"other-{record.pair_key}",
        )
        for record in clean
    )
    other_cued = tuple(
        replace(
            record, model_name="another-fixture-judge",
            record_id=f"other-{record.record_id}", pair_key=f"other-{record.pair_key}",
            clean_record_id=f"other-{record.clean_record_id}",
        )
        for record in cued
    )
    rows = build_lowest_dose_rows(
        (*clean, *other_clean), (*cued, *other_cued),
        expected_test_pairs_per_model=1, expected_question_ids=("q1",),
    )

    assert len(rows) == 16
    assert {row["model_name"] for row in rows} == {"fixture-judge", "another-fixture-judge"}
