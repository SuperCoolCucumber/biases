from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from scripts.analyze_lm_polygraph_inference import (
    COLLECTOR_FILENAME,
    COLLECTOR_MARKER_FILENAME,
    ESTIMANDS,
    LM_POLYGRAPH_COMMIT,
    MAP_AGREE_ESTIMAND,
    PREDICTORS,
    PRIMARY_ESTIMAND,
    SAME_ENGINE_ESTIMAND,
    _join_expected_rows,
    audit_row_from_mapping,
    collector_score_from_mapping,
    discover_collector_directories,
    expected_calibration_error,
    fit_isotonic,
    isotonic_predict,
    make_analysis_row,
    metric_summary,
    ranking_metrics,
    threshold_rule,
    threshold_transfer,
    valid_items,
    validate_output_path,
)


def campaign_row(
    record_id: str,
    *,
    score: float,
    verdict: str = "A",
    human_winner: str = "A",
    question_id: str = "q1",
):
    return audit_row_from_mapping(
        {
            "record_id": record_id,
            "question_id": question_id,
            "model_name": "model",
            "pair_identity_key": f"pair-{record_id}",
            "condition_group_id": f"condition-{record_id}",
            "ordering": "ab",
            "routing_split": "calibration",
            "bias_name": "clean",
            "cue_congruency": "clean",
            "dose": None,
            "clean_tie": False,
            "cue_target": None,
            "human_winner": human_winner,
            "verdict": verdict,
            "msp": score,
        }
    )


def collector_row(
    record_id: str,
    *,
    log_p_true: float = -0.2,
    entropy: float = 3.0,
    self_certainty: float = -8.0,
    msp: float | None = None,
    verdict: str = "A",
    hf_verdict: str | None = None,
):
    hf_verdict = hf_verdict or verdict
    normalized_hf = "tie" if hf_verdict.lower() in {"t", "tie"} else hf_verdict
    normalized_source = "tie" if verdict.lower() in {"t", "tie"} else verdict
    probabilities = {"A": 0.1, "B": 0.1, "tie": 0.1}
    probabilities[normalized_hf] = 0.8
    payload: dict[str, object] = {
        "record_id": record_id,
        "model_name": "model",
        "p_true_log_probability": log_p_true,
        "p_true_probability": math.exp(log_p_true),
        "p_true_uncertainty": -log_p_true,
        "mean_token_entropy": entropy,
        "self_certainty": self_certainty,
        "verdict": verdict,
        "hf_restricted_label_probabilities": probabilities,
        "hf_restricted_msp": 0.8,
        "hf_restricted_map_verdict": hf_verdict,
        "hf_restricted_map_matches_stored": normalized_hf == normalized_source,
        "hf_restricted_verdict_probability": probabilities[normalized_source],
        "hf_source_probability_max_abs_difference": 0.1,
    }
    if msp is not None:
        payload["msp"] = msp
    return collector_score_from_mapping(payload)


def analysis_row(
    record_id: str,
    *,
    msp: float,
    entropy: float,
    self_certainty: float,
    verdict: str = "A",
    human_winner: str = "A",
    hf_verdict: str | None = None,
):
    campaign = campaign_row(
        record_id,
        score=msp,
        verdict=verdict,
        human_winner=human_winner,
    )
    collector = collector_row(
        record_id,
        entropy=entropy,
        self_certainty=self_certainty,
        msp=msp,
        verdict=verdict,
        hf_verdict=hf_verdict,
    )
    return make_analysis_row(campaign, collector)


def test_predictors_remain_separate() -> None:
    assert PREDICTORS == (
        "msp",
        "p_true",
        "mean_token_entropy",
        "self_certainty",
    )


def test_estimands_are_explicit_and_separate() -> None:
    assert ESTIMANDS == (
        PRIMARY_ESTIMAND,
        SAME_ENGINE_ESTIMAND,
        MAP_AGREE_ESTIMAND,
    )


def test_cross_engine_same_engine_and_map_agree_targets_diverge() -> None:
    disagreeing = analysis_row(
        "disagreeing",
        msp=0.8,
        entropy=2.0,
        self_certainty=-3.0,
        verdict="A",
        hf_verdict="B",
        human_winner="A",
    )
    agreeing = analysis_row(
        "agreeing",
        msp=0.7,
        entropy=3.0,
        self_certainty=-4.0,
        verdict="A",
        hf_verdict="A",
        human_winner="A",
    )
    rows = [disagreeing, agreeing]

    assert [item[2] for item in valid_items(
        rows,
        "mean_token_entropy",
        PRIMARY_ESTIMAND,
    )] == [True, True]
    assert [item[2] for item in valid_items(
        rows,
        "mean_token_entropy",
        SAME_ENGINE_ESTIMAND,
    )] == [False, True]
    sensitivity = valid_items(
        rows,
        "mean_token_entropy",
        MAP_AGREE_ESTIMAND,
    )
    assert [item[0].record_id for item in sensitivity] == ["agreeing"]
    summary = metric_summary(rows, "mean_token_entropy", MAP_AGREE_ESTIMAND)
    assert summary["population_total"] == 2
    assert summary["excluded_by_estimand"] == 1
    assert summary["total"] == 1


def test_p_true_is_not_misapplied_to_hf_replay_correctness() -> None:
    row = analysis_row(
        "disagreeing",
        msp=0.8,
        entropy=2.0,
        self_certainty=-3.0,
        verdict="A",
        hf_verdict="B",
    )
    assert valid_items([row], "p_true", SAME_ENGINE_ESTIMAND) == []
    summary = metric_summary([row], "p_true", SAME_ENGINE_ESTIMAND)
    assert summary["predictor_applicable"] is False
    assert summary["n"] == 0
    assert "frozen vLLM verdict" in summary["predictor_estimand_caveat"]


def test_frozen_vllm_msp_is_not_mislabeled_as_same_engine() -> None:
    row = analysis_row(
        "disagreeing",
        msp=0.2,
        entropy=2.0,
        self_certainty=-3.0,
        verdict="A",
        hf_verdict="B",
    )
    assert row.collector.hf_restricted_msp == pytest.approx(0.8)
    assert row.confidence_scores["msp"] == pytest.approx(0.2)
    assert valid_items([row], "msp", SAME_ENGINE_ESTIMAND) == []
    summary = metric_summary([row], "msp", SAME_ENGINE_ESTIMAND)
    assert summary["predictor_applicable"] is False
    assert "published MSP" in summary["predictor_estimand_caveat"]


def test_thresholds_are_fit_independently_for_each_outcome() -> None:
    row = analysis_row(
        "disagreeing",
        msp=0.8,
        entropy=2.0,
        self_certainty=-3.0,
        verdict="A",
        hf_verdict="B",
        human_winner="A",
    )
    primary_rule = threshold_rule(
        [row],
        "mean_token_entropy",
        0.10,
        PRIMARY_ESTIMAND,
    )
    same_engine_rule = threshold_rule(
        [row],
        "mean_token_entropy",
        0.10,
        SAME_ENGINE_ESTIMAND,
    )
    assert primary_rule["threshold"] == pytest.approx(-2.0)
    assert same_engine_rule["threshold"] is None


def test_collector_rejects_forged_map_agreement() -> None:
    payload = {
        "record_id": "row",
        "model_name": "model",
        "p_true_log_probability": -0.2,
        "p_true_probability": math.exp(-0.2),
        "p_true_uncertainty": 0.2,
        "mean_token_entropy": 2.0,
        "self_certainty": -3.0,
        "verdict": "A",
        "hf_restricted_label_probabilities": {
            "A": 0.1,
            "B": 0.8,
            "tie": 0.1,
        },
        "hf_restricted_msp": 0.8,
        "hf_restricted_map_verdict": "B",
        "hf_restricted_map_matches_stored": True,
        "hf_restricted_verdict_probability": 0.1,
        "hf_source_probability_max_abs_difference": 0.3,
    }
    with pytest.raises(ValueError, match="does not match the two verdicts"):
        collector_score_from_mapping(payload)


def test_collector_requires_pinned_argmax_tie_breaking() -> None:
    payload = {
        "record_id": "row",
        "model_name": "model",
        "p_true_log_probability": -0.2,
        "p_true_probability": math.exp(-0.2),
        "p_true_uncertainty": 0.2,
        "mean_token_entropy": 2.0,
        "self_certainty": -3.0,
        "verdict": "A",
        "hf_restricted_label_probabilities": {
            "A": 0.45,
            "B": 0.45,
            "tie": 0.10,
        },
        "hf_restricted_msp": 0.45,
        "hf_restricted_map_verdict": "B",
        "hf_restricted_map_matches_stored": False,
        "hf_restricted_verdict_probability": 0.45,
        "hf_source_probability_max_abs_difference": 0.1,
    }
    with pytest.raises(ValueError, match="pinned A/B/tie argmax rule"):
        collector_score_from_mapping(payload)


def test_collector_rejects_inconsistent_p_true_representations() -> None:
    payload = {
        "record_id": "row",
        "model_name": "model",
        "p_true_log_probability": -0.2,
        "p_true_probability": 0.9,
        "p_true_uncertainty": 0.2,
        "mean_token_entropy": 2.0,
        "self_certainty": -3.0,
    }
    with pytest.raises(ValueError, match="p_true_probability"):
        collector_score_from_mapping(payload)


def test_collector_allows_explicitly_unavailable_p_true() -> None:
    score = collector_score_from_mapping(
        {
            "record_id": "row",
            "model_name": "model",
            "p_true_log_probability": None,
            "p_true_probability": None,
            "p_true_uncertainty": None,
            "mean_token_entropy": 2.0,
            "self_certainty": -3.0,
            "verdict": "A",
            "hf_restricted_label_probabilities": {
                "A": 0.8,
                "B": 0.1,
                "tie": 0.1,
            },
            "hf_restricted_msp": 0.8,
            "hf_restricted_map_verdict": "A",
            "hf_restricted_map_matches_stored": True,
            "hf_restricted_verdict_probability": 0.8,
            "hf_source_probability_max_abs_difference": 0.1,
        }
    )
    assert score.p_true_log_probability is None


def test_unbounded_scores_are_oriented_without_clipping() -> None:
    row = analysis_row(
        "row",
        msp=0.8,
        entropy=123.5,
        self_certainty=-456.25,
    )
    assert row.raw_scores["mean_token_entropy"] == pytest.approx(123.5)
    assert row.confidence_scores["mean_token_entropy"] == pytest.approx(-123.5)
    assert row.raw_scores["self_certainty"] == pytest.approx(-456.25)
    assert row.confidence_scores["self_certainty"] == pytest.approx(456.25)


def test_tie_batched_threshold_works_on_arbitrary_real_scale() -> None:
    rows = [
        analysis_row(
            f"row-{index}",
            msp=0.8,
            entropy=entropy,
            self_certainty=-2.0,
            verdict=verdict,
        )
        for index, (entropy, verdict) in enumerate(
            [(100.0, "A"), (200.0, "A"), (200.0, "B")]
        )
    ]
    strict = threshold_rule(rows, "mean_token_entropy", 0.10)
    relaxed = threshold_rule(rows, "mean_token_entropy", 0.40)

    assert strict["threshold"] == pytest.approx(-100.0)
    assert strict["accepted"] == 1
    assert relaxed["threshold"] == pytest.approx(-200.0)
    assert relaxed["accepted"] == 3


def test_zero_coverage_has_undefined_risk() -> None:
    row = analysis_row(
        "error",
        msp=0.7,
        entropy=10.0,
        self_certainty=-5.0,
        verdict="B",
    )
    rule = threshold_rule([row], "self_certainty", 0.10)
    transfer = threshold_transfer([row], "self_certainty", rule)
    assert rule["threshold"] is None
    assert rule["coverage"] == 0.0
    assert rule["risk"] is None
    assert transfer["accepted"] == 0
    assert transfer["risk"] is None


def test_right_continuous_aurc_respects_ties() -> None:
    correct = analysis_row(
        "correct",
        msp=0.8,
        entropy=1.0,
        self_certainty=-2.0,
    )
    wrong = analysis_row(
        "wrong",
        msp=0.8,
        entropy=2.0,
        self_certainty=-2.0,
        verdict="B",
    )
    tied = [(correct, 10.0, True), (wrong, 10.0, False)]
    separated = [(correct, 10.0, True), (wrong, 5.0, False)]
    assert ranking_metrics(tied)["aurc"] == pytest.approx(0.5)
    assert ranking_metrics(separated)["aurc"] == pytest.approx(0.25)


def test_isotonic_maps_arbitrary_scores_to_probabilities() -> None:
    rows = [
        analysis_row(
            f"row-{index}",
            msp=0.8,
            entropy=entropy,
            self_certainty=-2.0,
            verdict=verdict,
        )
        for index, (entropy, verdict) in enumerate(
            [(1000.0, "B"), (100.0, "B"), (10.0, "A"), (1.0, "A")]
        )
    ]
    items = [
        (row, row.confidence_scores["mean_token_entropy"], row.correct)
        for row in rows
    ]
    blocks = fit_isotonic(items)  # type: ignore[arg-type]
    predictions = [isotonic_predict(blocks, item[1]) for item in items]
    assert all(0.0 <= probability <= 1.0 for probability in predictions)
    assert predictions[0] <= predictions[-1]
    probability_items = [
        (item[0], probability, item[2])
        for item, probability in zip(items, predictions, strict=True)
    ]
    assert expected_calibration_error(probability_items) is not None


def test_join_requires_exact_record_id_set() -> None:
    campaign = [campaign_row("a", score=0.8), campaign_row("b", score=0.7)]
    collectors = {"a": collector_row("a")}
    with pytest.raises(ValueError, match="record_id set differs"):
        _join_expected_rows(campaign, collectors)


def test_collector_marker_pins_commit_and_verifies_hash(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    score_path = model_dir / COLLECTOR_FILENAME
    score_path.write_text(
        json.dumps(
            {
                "record_id": "row",
                "model_name": "model",
                "p_true_log_probability": -0.2,
                "p_true_probability": math.exp(-0.2),
                "p_true_uncertainty": 0.2,
                "mean_token_entropy": 2.0,
                "self_certainty": -3.0,
            }
        )
        + "\n"
    )
    sha = hashlib.sha256(score_path.read_bytes()).hexdigest()
    (model_dir / COLLECTOR_MARKER_FILENAME).write_text(
        json.dumps(
            {
                "status": "complete",
                "model_name": "model",
                "lm_polygraph_commit": LM_POLYGRAPH_COMMIT,
                "score_file_sha256": sha,
            }
        )
    )
    assert discover_collector_directories(tmp_path) == {"model": model_dir}

    marker_path = model_dir / COLLECTOR_MARKER_FILENAME
    marker = json.loads(marker_path.read_text())
    marker["lm_polygraph_commit"] = "wrong"
    marker_path.write_text(json.dumps(marker))
    with pytest.raises(ValueError, match="expected LM-Polygraph commit"):
        discover_collector_directories(tmp_path)


def test_output_must_be_no_clobber_and_outside_inputs(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    collector = tmp_path / "collector"
    campaign.mkdir()
    collector.mkdir()
    with pytest.raises(ValueError, match="outside"):
        validate_output_path(campaign, collector, campaign / "report.json")

    output = tmp_path / "report.json"
    output.write_text("existing")
    with pytest.raises(FileExistsError, match="overwrite"):
        validate_output_path(campaign, collector, output)
