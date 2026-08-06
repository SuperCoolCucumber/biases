from __future__ import annotations

import csv
import math

import pytest

from scripts.analyze_repeatability_predictors import (
    PREDICTORS,
    attach_order_vote_scores,
    audit_row_from_mapping,
    question_disjoint_threshold_transfer,
    ranking_metrics,
    threshold_rule,
    threshold_transfer,
    validate_msp_primary_regression,
    validate_output_path,
)


def raw_row(
    *,
    record_id: str,
    ordering: str,
    verdict: str,
    human_winner: str,
    counts: dict[str, int],
    msp: float = 0.8,
) -> dict[str, object]:
    return {
        "record_id": record_id,
        "question_id": "q1",
        "model_name": "model",
        "pair_identity_key": "pair",
        "condition_group_id": "condition",
        "ordering": ordering,
        "routing_split": "calibration",
        "bias_name": "clean",
        "cue_congruency": "clean",
        "dose": None,
        "clean_tie": False,
        "cue_target": None,
        "human_winner": human_winner,
        "verdict": verdict,
        "consistency_verdict_counts": counts,
        "consistency_flip_rate": 1.0 - counts[verdict] / sum(counts.values()),
        "msp": msp,
    }


def test_audit_row_uses_anchor_reproducibility_not_modal_agreement() -> None:
    row = audit_row_from_mapping(
        raw_row(
            record_id="ab",
            ordering="ab",
            verdict="A",
            human_winner="A",
            counts={"A": 1, "B": 3, "tie": 0},
        )
    )

    assert row.scores["anchor_reproducibility"] == pytest.approx(0.25)
    assert row.scores["frequency_semantic_entropy_confidence"] == pytest.approx(
        1.0 - (-(0.25 * math.log(0.25) + 0.75 * math.log(0.75))) / math.log(3.0)
    )
    assert row.scores["degree_matrix_agreement"] == pytest.approx(0.625)


def test_audit_row_rejects_inconsistent_stored_flip_rate() -> None:
    payload = raw_row(
        record_id="ab",
        ordering="ab",
        verdict="A",
        human_winner="A",
        counts={"A": 1, "B": 3, "tie": 0},
    )
    payload["consistency_flip_rate"] = 0.5

    with pytest.raises(ValueError, match="inconsistent flip_rate"):
        audit_row_from_mapping(payload)


def test_order_vote_scores_align_candidate_identity_and_preserve_tie() -> None:
    ab = audit_row_from_mapping(
        raw_row(
            record_id="ab",
            ordering="ab",
            verdict="A",
            human_winner="A",
            counts={"A": 2, "B": 1, "tie": 1},
        )
    )
    ba = audit_row_from_mapping(
        raw_row(
            record_id="ba",
            ordering="ba",
            verdict="B",
            human_winner="B",
            counts={"A": 1, "B": 2, "tie": 1},
        )
    )

    diagnostics = attach_order_vote_scores([ab, ba])

    assert diagnostics == {
        "groups": 1,
        "complete_pairs": 1,
        "malformed_pairs": 0,
        "pairs_missing_counts": 0,
        "canonical_human_disagreements": 0,
        "canonical_cue_target_disagreements": 0,
        "unequal_repeat_count_pairs": 0,
    }
    assert ab.scores["order_vote_js_similarity"] == pytest.approx(1.0)
    assert ba.scores["order_vote_tv_similarity"] == pytest.approx(1.0)
    assert ab.scores["order_vote_expected_agreement"] == pytest.approx(0.375)


def test_order_vote_scores_reject_misaligned_cue_targets() -> None:
    ab_payload = raw_row(
        record_id="ab",
        ordering="ab",
        verdict="A",
        human_winner="A",
        counts={"A": 4, "B": 0, "tie": 0},
    )
    ba_payload = raw_row(
        record_id="ba",
        ordering="ba",
        verdict="B",
        human_winner="B",
        counts={"A": 0, "B": 4, "tie": 0},
    )
    ab_payload["cue_target"] = "A"
    ba_payload["cue_target"] = "A"

    diagnostics = attach_order_vote_scores(
        [
            audit_row_from_mapping(ab_payload),
            audit_row_from_mapping(ba_payload),
        ]
    )

    assert diagnostics["canonical_cue_target_disagreements"] == 1
    assert diagnostics["complete_pairs"] == 0


def test_order_vote_scores_reject_unequal_repeat_counts() -> None:
    ab = audit_row_from_mapping(
        raw_row(
            record_id="ab",
            ordering="ab",
            verdict="A",
            human_winner="A",
            counts={"A": 4, "B": 0, "tie": 0},
        )
    )
    ba = audit_row_from_mapping(
        raw_row(
            record_id="ba",
            ordering="ba",
            verdict="B",
            human_winner="B",
            counts={"A": 0, "B": 3, "tie": 0},
        )
    )

    diagnostics = attach_order_vote_scores([ab, ba])

    assert diagnostics["unequal_repeat_count_pairs"] == 1
    assert diagnostics["complete_pairs"] == 0


def test_empirical_rule_accepts_equal_scores_as_one_block() -> None:
    rows = [
        audit_row_from_mapping(
            raw_row(
                record_id=f"row-{index}",
                ordering="ab",
                verdict=verdict,
                human_winner="A",
                counts={"A": 4, "B": 0, "tie": 0},
                msp=score,
            )
        )
        for index, (score, verdict) in enumerate(
            [(0.9, "A"), (0.8, "A"), (0.8, "B")]
        )
    ]

    strict_rule = threshold_rule(rows, "msp", 0.10)
    relaxed_rule = threshold_rule(rows, "msp", 0.40)

    assert strict_rule["accepted"] == 1
    assert strict_rule["threshold"] == pytest.approx(0.9)
    assert relaxed_rule["accepted"] == 3
    assert relaxed_rule["threshold"] == pytest.approx(0.8)


def test_aurc_uses_right_continuous_tie_blocks() -> None:
    rows = [
        audit_row_from_mapping(
            raw_row(
                record_id=f"row-{index}",
                ordering="ab",
                verdict=verdict,
                human_winner="A",
                counts={"A": 4, "B": 0, "tie": 0},
                msp=score,
            )
        )
        for index, (score, verdict) in enumerate(
            [(0.9, "A"), (0.8, "B")]
        )
    ]
    unique = [(row, float(row.scores["msp"]), row.verdict == "A") for row in rows]
    tied = [(row, 0.8, row.verdict == "A") for row in rows]

    assert ranking_metrics(unique)["aurc"] == pytest.approx(0.25)
    assert ranking_metrics(tied)["aurc"] == pytest.approx(0.5)


def test_no_feasible_rule_has_zero_coverage_and_undefined_risk() -> None:
    row = audit_row_from_mapping(
        raw_row(
            record_id="error",
            ordering="ab",
            verdict="B",
            human_winner="A",
            counts={"A": 0, "B": 4, "tie": 0},
        )
    )

    rule = threshold_rule([row], "msp", 0.10)
    transfer = threshold_transfer([row], "msp", rule)

    assert rule["threshold"] is None
    assert rule["coverage"] == 0.0
    assert rule["risk"] is None
    assert transfer["accepted"] == 0
    assert transfer["errors"] == 0
    assert transfer["risk"] is None


def test_question_disjoint_transfer_applies_each_item_once() -> None:
    rows = []
    for question_id, score in (("q1", 0.9), ("q2", 0.7)):
        payload = raw_row(
            record_id=question_id,
            ordering="ab",
            verdict="A",
            human_winner="A",
            counts={"A": 4, "B": 0, "tie": 0},
            msp=score,
        )
        payload["question_id"] = question_id
        rows.append(audit_row_from_mapping(payload))

    result = question_disjoint_threshold_transfer(
        rows,
        "msp",
        {0: {"threshold": 0.8}, 1: {"threshold": 0.6}},
        {"q1": 0, "q2": 1},
    )

    assert result["n"] == 2
    assert result["accepted"] == 2
    assert result["fold_thresholds"] == {"0": 0.8, "1": 0.6}


def test_question_disjoint_transfer_rejects_missing_question_fold() -> None:
    row = audit_row_from_mapping(
        raw_row(
            record_id="row",
            ordering="ab",
            verdict="A",
            human_winner="A",
            counts={"A": 4, "B": 0, "tie": 0},
        )
    )

    with pytest.raises(ValueError, match="missing a fold assignment"):
        question_disjoint_threshold_transfer(
            [row],
            "msp",
            {0: {"threshold": 0.8}},
            {},
        )


def test_predictors_are_separate_and_have_no_composite() -> None:
    assert PREDICTORS == (
        "msp",
        "anchor_reproducibility",
        "frequency_semantic_entropy_confidence",
        "degree_matrix_agreement",
        "order_vote_js_similarity",
        "order_vote_tv_similarity",
        "order_vote_expected_agreement",
    )


def test_msp_regression_accepts_matching_primary_cell(tmp_path) -> None:
    oracle = tmp_path / "rq2_threshold_transfer.csv"
    fields = [
        "primary",
        "model_name",
        "ordering",
        "family",
        "dose",
        "calibration_n",
        "test_n",
        "test_accepted",
        "threshold",
        "calibration_coverage",
        "calibration_risk",
        "test_coverage",
        "test_realized_risk",
    ]
    with oracle.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "primary": "True",
                "model_name": "model",
                "ordering": "ab",
                "family": "authority",
                "dose": 4.0,
                "calibration_n": 10,
                "test_n": 5,
                "test_accepted": 0,
                "threshold": "inf",
                "calibration_coverage": 0.0,
                "calibration_risk": "",
                "test_coverage": 0.0,
                "test_realized_risk": "",
            }
        )
    cell = {
        "model_name": "model",
        "ordering": "ab",
        "family": "authority",
        "dose": 4.0,
        "predictor": "msp",
        "target_risk": 0.10,
        "rule_threshold": None,
        "rule_calibration_n": 10,
        "rule_calibration_coverage": 0.0,
        "rule_calibration_risk": None,
        "n": 5,
        "accepted": 0,
        "errors": 0,
        "coverage": 0.0,
        "risk": None,
    }

    result = validate_msp_primary_regression([cell], oracle)

    assert result["passed"] is True
    assert result["zero_coverage_cells"] == 1


def test_output_must_be_new_and_outside_campaign_root(tmp_path) -> None:
    campaign_root = tmp_path / "campaign"
    campaign_root.mkdir()

    with pytest.raises(ValueError, match="outside"):
        validate_output_path(campaign_root, campaign_root / "report.json")

    existing = tmp_path / "existing.json"
    existing.write_text("preserve me")
    with pytest.raises(FileExistsError, match="overwrite"):
        validate_output_path(campaign_root, existing)

    validate_output_path(campaign_root, tmp_path / "new.json")

    published_root = tmp_path / "published"
    published_root.mkdir()
    with pytest.raises(ValueError, match="immutable"):
        validate_output_path(
            campaign_root,
            published_root / "report.json",
            additional_immutable_roots=(published_root,),
        )
