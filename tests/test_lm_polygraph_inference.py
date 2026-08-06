from __future__ import annotations

import math

import numpy as np
import pytest

from scripts.collect_lm_polygraph_inference import (
    _json_safe_evidence,
    validate_stage_generation_provenance,
    write_batch_failure_evidence,
)

from biases.analysis.lm_polygraph_inference import (
    P_TRUE_TEMPLATE,
    FullVocabularyMetrics,
    PTrueMetrics,
    ReplayItem,
    ReplaySelection,
    RestrictedLabelMetrics,
    full_vocabulary_metrics,
    is_primary_stage_b_row,
    make_score_row,
    normalize_verdict_token,
    p_true_meta_prompt,
    p_true_metrics,
    prompt_token_length_preflight,
    restricted_label_metrics,
    restricted_pairwise_logit_gaps,
    token_ids_sha256,
    validate_existing_score_rows,
    validate_score_rows_against_selection,
    validate_scientific_score_gates,
)


def source_row(*, verdict: str = "A") -> dict[str, object]:
    return {
        "record_id": "record-1",
        "question_id": "question-1",
        "example_id": "example-1",
        "prompt_hash": "prompt-hash",
        "pair_key": "pair-key",
        "condition_group_id": "condition-group",
        "spec_hash": "source-spec",
        "input_file_hash": "f" * 64,
        "verdict": verdict,
        "raw_prompt_logprobs": {"A": 0.7, "B": 0.2, "tie": 0.1},
        "uncertainty": {"logit": {"msp": 0.7}},
        "spec": {
            "model_name": "fixture/model",
            "model_revision": "revision",
        },
        "condition": {
            "bias_type": "authority",
            "cue_congruency": "incongruent",
            "direction_relative_human": "against",
            "ordering": "ab",
            "dose": 4,
            "clean_tie": False,
        },
        "metadata": {
            "pair_identity_key": "pair-identity",
            "routing_split": "test",
            "human_winner": "A",
        },
    }


def replay_item(*, verdict: str = "A") -> ReplayItem:
    return ReplayItem(
        source_stage="stage_b",
        source_row=source_row(verdict=verdict),
        original_prompt="original",
        p_true_prompt="meta",
        verdict_token_text=normalize_verdict_token(verdict),
    )


def test_p_true_meta_prompt_is_exact_pinned_template() -> None:
    expected = (
        "Question: SYSTEM:\njudge\n"
        " Possible answer:T\n"
        " Is the possible answer:\n"
        " (A) True\n"
        " (B) False\n"
        " The possible answer is:"
    )
    assert p_true_meta_prompt("SYSTEM:\njudge", "tie") == expected
    assert P_TRUE_TEMPLATE.endswith("The possible answer is:")


@pytest.mark.parametrize(
    ("stored", "token"),
    [("A", "A"), ("b", "B"), ("tie", "T"), ("T", "T")],
)
def test_normalize_verdict_token(stored: str, token: str) -> None:
    assert normalize_verdict_token(stored) == token


def test_uniform_full_vocab_metrics_match_definitions() -> None:
    metrics = full_vocabulary_metrics(np.zeros(5))
    assert metrics.mean_token_entropy == pytest.approx(math.log(5))
    assert metrics.mean_token_entropy_confidence == pytest.approx(-math.log(5))
    assert metrics.self_certainty == pytest.approx(0.0, abs=1e-12)
    assert metrics.self_certainty_confidence == pytest.approx(0.0, abs=1e-12)


def test_peaked_distribution_increases_self_certainty_confidence() -> None:
    uniform = full_vocabulary_metrics([0.0, 0.0, 0.0])
    peaked = full_vocabulary_metrics([8.0, 0.0, 0.0])
    assert peaked.mean_token_entropy < uniform.mean_token_entropy
    assert peaked.self_certainty < uniform.self_certainty
    assert peaked.self_certainty_confidence > uniform.self_certainty_confidence


def test_p_true_uses_full_vocabulary_probability() -> None:
    metrics = p_true_metrics([math.log(3), 0.0], true_token_id=0)
    assert metrics.p_true_probability == pytest.approx(0.75)
    assert metrics.p_true_log_probability == pytest.approx(math.log(0.75))
    assert metrics.p_true_uncertainty == pytest.approx(-math.log(0.75))


def test_restricted_label_softmax_uses_only_a_b_t() -> None:
    metrics = restricted_label_metrics(
        [math.log(7), math.log(2), 100.0, 0.0],
        label_token_ids={"A": 0, "B": 1, "tie": 3},
    )
    assert sum(metrics.probabilities.values()) == pytest.approx(1.0)
    assert metrics.probabilities == pytest.approx({"A": 0.7, "B": 0.2, "tie": 0.1})
    assert metrics.msp == pytest.approx(0.7)


def test_primary_stage_b_filter_includes_clean_ties() -> None:
    row = source_row()
    condition = dict(row["condition"])  # type: ignore[arg-type]
    condition["clean_tie"] = True
    row["condition"] = condition
    assert is_primary_stage_b_row(row)
    condition["dose"] = 3
    assert not is_primary_stage_b_row(row)


def test_score_row_has_required_metadata_and_standardized_scores() -> None:
    row = make_score_row(
        item=replay_item(),
        full_vocab=FullVocabularyMetrics(0.5, -0.5, -2.0, 2.0),
        p_true=PTrueMetrics(math.log(0.8), 0.8, -math.log(0.8)),
        restricted=RestrictedLabelMetrics(
            probabilities={"A": 0.7, "B": 0.2, "tie": 0.1},
            msp=0.7,
        ),
        true_token_id=10,
        false_token_id=11,
        label_token_ids={"A": 1, "B": 2, "tie": 3},
        original_token_count=100,
        p_true_token_count=150,
        original_token_ids_sha256="1" * 64,
        p_true_token_ids_sha256="2" * 64,
        vocabulary_size=128,
        tokenizer_vocabulary_size=120,
        collector_spec_hash="spec-hash",
    )
    for key in (
        "record_id",
        "question_id",
        "model_name",
        "model_revision",
        "source_stage",
        "prompt_hash",
        "pair_identity_key",
        "condition_group_id",
        "ordering",
        "routing_split",
        "family",
        "bias_type",
        "direction",
        "cue_congruency",
        "dose",
        "clean_tie",
        "human_winner",
        "verdict",
        "msp",
        "p_true_log_probability",
        "p_true_probability",
        "p_true_uncertainty",
        "mean_token_entropy",
        "self_certainty",
    ):
        assert key in row
    assert row["hf_restricted_map_verdict"] == "A"
    assert row["hf_restricted_map_matches_stored"] is True
    assert row["hf_source_probability_max_abs_difference"] == pytest.approx(0.0)
    assert row["hf_source_probability_within_tolerance"] is True
    assert row["hf_source_pairwise_logit_gap_max_abs_difference"] == pytest.approx(
        0.0
    )
    assert row["padded_vocabulary_size_delta"] == 8


class CharacterTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return list(range(len(text)))

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        assert not skip_special_tokens
        assert not clean_up_tokenization_spaces
        return "".join(chr(token_id) for token_id in token_ids)


def test_token_length_preflight_checks_both_prompts_for_full_selection() -> None:
    selection = ReplaySelection(
        model_registry_name="fixture",
        model_name="fixture/model",
        model_revision="revision",
        input_file_hash="f" * 64,
        items=(replay_item(),),
        stage_a_count=0,
        full_stage_b_count=1,
        primary_stage_b_count=1,
        stage_a_pair_summary_count=0,
    )
    result = prompt_token_length_preflight(
        selection,
        tokenizer=CharacterTokenizer(),
        max_model_len=8,
    )
    assert result["checked_record_count"] == 1
    assert result["max_original_token_count"] == 8
    assert result["max_p_true_token_count"] == 4
    assert result["overlimit_count"] == 0


def test_resume_rows_must_be_deterministic_prefix() -> None:
    rows = [
        {"record_id": "one", "collector_spec_hash": "spec"},
        {"record_id": "three", "collector_spec_hash": "spec"},
    ]
    with pytest.raises(ValueError, match="deterministic selection prefix"):
        validate_existing_score_rows(
            rows,
            selected_record_ids=["one", "two", "three"],
            collector_spec_hash="spec",
        )


def test_failure_evidence_sanitizes_nonfinite_values_and_is_no_clobber(
    tmp_path,
) -> None:
    safe = _json_safe_evidence(
        {"nan": float("nan"), "positive": float("inf"), "negative": -float("inf")}
    )
    assert safe == {
        "nan": {"nonfinite_float": "nan"},
        "positive": {"nonfinite_float": "+inf"},
        "negative": {"nonfinite_float": "-inf"},
    }
    paths = [
        write_batch_failure_evidence(
            output_dir=tmp_path,
            processed_before_batch=8,
            batch_record_ids=["record"],
            stage="structural_batch_validation",
            exception=ValueError("nonfinite"),
            batch_rows=[{"score": float("nan")}],
            collector_spec_hash="a" * 64,
        )
        for _ in range(2)
    ]
    assert paths[0] != paths[1]
    assert all(path.exists() for path in paths)


def test_schema_2_stage_generation_allows_exact_mixed_stage_commits() -> None:
    stage_a_commit = "a" * 40
    stage_b_commit = "b" * 40
    marker = {
        "stage_generation": {
            "stage_a": {
                "code_commit": stage_a_commit,
                "mode": "preserved",
                "records": 6_674,
            },
            "stage_b": {
                "code_commit": stage_b_commit,
                "mode": "new_generation",
                "records": 106_784,
            },
        }
    }
    model_contract = {"stage_generation": marker["stage_generation"]}

    assert validate_stage_generation_provenance(
        marker=marker,
        model_contract=model_contract,
        contract_schema_version=2,
        legacy_generation_commit=stage_a_commit,
        expected_stage_counts={"stage_a": 6_674, "stage_b": 106_784},
    ) == marker["stage_generation"]


def test_schema_2_stage_generation_rejects_commit_or_mode_drift() -> None:
    expected = {
        "stage_a": {
            "code_commit": "a" * 40,
            "mode": "preserved",
            "records": 6_674,
        },
        "stage_b": {
            "code_commit": "b" * 40,
            "mode": "new_generation",
            "records": 106_784,
        },
    }
    marker = {"stage_generation": {key: dict(value) for key, value in expected.items()}}
    marker["stage_generation"]["stage_b"]["mode"] = "preserved"
    with pytest.raises(ValueError, match="stage_b generation mode drift"):
        validate_stage_generation_provenance(
            marker=marker,
            model_contract={"stage_generation": expected},
            contract_schema_version=2,
            legacy_generation_commit="a" * 40,
            expected_stage_counts={"stage_a": 6_674, "stage_b": 106_784},
        )

    marker["stage_generation"]["stage_b"] = dict(expected["stage_b"])
    marker["stage_generation"]["stage_b"]["code_commit"] = "c" * 40
    with pytest.raises(ValueError, match="stage_b generation commit drift"):
        validate_stage_generation_provenance(
            marker=marker,
            model_contract={"stage_generation": expected},
            contract_schema_version=2,
            legacy_generation_commit="a" * 40,
            expected_stage_counts={"stage_a": 6_674, "stage_b": 106_784},
        )


def test_schema_2_stage_generation_rejects_shape_count_and_mode_contract_drift() -> None:
    expected = {
        "stage_a": {
            "code_commit": "a" * 40,
            "mode": "preserved",
            "records": 1,
        },
        "stage_b": {
            "code_commit": "b" * 40,
            "mode": "new_generation",
            "records": 2,
        },
    }
    marker = {"stage_generation": expected}

    missing_stage = {"stage_generation": {"stage_a": expected["stage_a"]}}
    with pytest.raises(ValueError, match="must pin exactly stage_a and stage_b"):
        validate_stage_generation_provenance(
            marker=marker,
            model_contract=missing_stage,
            contract_schema_version=2,
            legacy_generation_commit="a" * 40,
            expected_stage_counts={"stage_a": 1, "stage_b": 2},
        )

    count_drift = {
        "stage_generation": {
            key: dict(value) for key, value in expected.items()
        }
    }
    count_drift["stage_generation"]["stage_b"]["records"] = 3
    with pytest.raises(ValueError, match="stage_b record-count drift"):
        validate_stage_generation_provenance(
            marker=marker,
            model_contract=count_drift,
            contract_schema_version=2,
            legacy_generation_commit="a" * 40,
            expected_stage_counts={"stage_a": 1, "stage_b": 2},
        )

    invalid_mode = {
        "stage_generation": {
            key: dict(value) for key, value in expected.items()
        }
    }
    invalid_mode["stage_generation"]["stage_b"]["mode"] = "inferred"
    with pytest.raises(ValueError, match="stage_b provenance is incomplete"):
        validate_stage_generation_provenance(
            marker=marker,
            model_contract=invalid_mode,
            contract_schema_version=2,
            legacy_generation_commit="a" * 40,
            expected_stage_counts={"stage_a": 1, "stage_b": 2},
        )

    extra_marker_stage = {
        "stage_generation": {**expected, "stage_c": expected["stage_b"]}
    }
    with pytest.raises(ValueError, match="campaign marker must pin exactly"):
        validate_stage_generation_provenance(
            marker=extra_marker_stage,
            model_contract={"stage_generation": expected},
            contract_schema_version=2,
            legacy_generation_commit="a" * 40,
            expected_stage_counts={"stage_a": 1, "stage_b": 2},
        )


def test_schema_1_stage_generation_retains_global_commit_contract() -> None:
    generation_commit = "a" * 40
    marker = {
        "stage_generation": {
            "stage_a": {
                "code_commit": generation_commit,
                "mode": "preserved",
                "records": 1,
            },
            "stage_b": {
                "code_commit": generation_commit,
                "mode": "preserved",
                "records": 2,
            },
        }
    }
    validated = validate_stage_generation_provenance(
        marker=marker,
        model_contract={},
        contract_schema_version=1,
        legacy_generation_commit=generation_commit,
        expected_stage_counts={"stage_a": 1, "stage_b": 2},
    )
    assert validated == marker["stage_generation"]

    with pytest.raises(ValueError, match="schema-1.*cannot override"):
        validate_stage_generation_provenance(
            marker=marker,
            model_contract={"stage_generation": marker["stage_generation"]},
            contract_schema_version=1,
            legacy_generation_commit=generation_commit,
            expected_stage_counts={"stage_a": 1, "stage_b": 2},
        )


def test_cross_backend_replay_is_reported_without_selective_rejection() -> None:
    row = {
        "record_id": "record",
        "verdict": "A",
        "hf_restricted_map_verdict": "A",
        "hf_source_probability_max_abs_difference": 0.01,
        "hf_source_pairwise_logit_gap_max_abs_difference": 0.25,
        "hf_source_pairwise_logit_gap_available_count": 3,
        "hf_source_pairwise_logit_gap_complete": True,
        "original_token_count": 100,
        "p_true_token_count": 200,
    }
    result = validate_scientific_score_gates([row])
    assert result["hard_gates_passed"] is True
    assert result["cross_backend_replay_role"] == "diagnostic_only"
    assert result["restricted_map_matches_stored"] is True
    assert result["max_source_probability_abs_difference"] == pytest.approx(0.01)
    assert result[
        "max_source_pairwise_logit_gap_abs_difference"
    ] == pytest.approx(0.25)
    row["hf_restricted_map_verdict"] = "B"
    row["hf_source_probability_max_abs_difference"] = 0.2
    row["hf_source_pairwise_logit_gap_max_abs_difference"] = 1.5
    diagnostic = validate_scientific_score_gates([row])
    assert diagnostic["restricted_map_matches_stored"] is False
    assert diagnostic["restricted_map_mismatch_count"] == 1
    assert diagnostic["source_probability_tolerance_enforced"] is False
    assert diagnostic["source_probability_tolerance_exceedance_count"] == 1
    assert diagnostic["max_source_probability_abs_difference"] == pytest.approx(0.2)
    assert diagnostic[
        "max_source_pairwise_logit_gap_abs_difference"
    ] == pytest.approx(1.5)


def test_zero_serialized_probabilities_make_logit_gap_diagnostic_nullable() -> None:
    assert restricted_pairwise_logit_gaps(
        {"A": 1.0, "B": 0.0, "tie": 0.0}
    ) == {
        "A_minus_B": None,
        "A_minus_tie": None,
        "B_minus_tie": None,
    }
    item = replay_item()
    item.source_row["raw_prompt_logprobs"] = {"A": 1.0, "B": 0.0, "tie": 0.0}
    row = make_score_row(
        item=item,
        full_vocab=FullVocabularyMetrics(0.5, -0.5, -2.0, 2.0),
        p_true=PTrueMetrics(math.log(0.8), 0.8, -math.log(0.8)),
        restricted=RestrictedLabelMetrics(
            probabilities={"A": 0.7, "B": 0.2, "tie": 0.1},
            msp=0.7,
        ),
        true_token_id=10,
        false_token_id=11,
        label_token_ids={"A": 1, "B": 2, "tie": 3},
        original_token_count=8,
        p_true_token_count=4,
        original_token_ids_sha256=token_ids_sha256(range(8)),
        p_true_token_ids_sha256=token_ids_sha256(range(4)),
        vocabulary_size=128,
        tokenizer_vocabulary_size=120,
        collector_spec_hash="spec-hash",
    )
    assert row["hf_source_pairwise_logit_gap_max_abs_difference"] is None
    assert row["hf_source_pairwise_logit_gap_available_count"] == 0
    assert row["hf_source_pairwise_logit_gap_complete"] is False
    validate_score_rows_against_selection(
        [row],
        selected_items=[item],
        collector_spec_hash="spec-hash",
        tokenizer=CharacterTokenizer(),
        true_token_id=10,
        false_token_id=11,
        label_token_ids={"A": 1, "B": 2, "tie": 3},
        vocabulary_size=128,
        tokenizer_vocabulary_size=120,
    )


def test_one_zero_probability_is_reported_as_partial_logit_gap_diagnostic() -> None:
    item = replay_item()
    item.source_row["raw_prompt_logprobs"] = {"A": 0.8, "B": 0.2, "tie": 0.0}
    row = make_score_row(
        item=item,
        full_vocab=FullVocabularyMetrics(0.5, -0.5, -2.0, 2.0),
        p_true=PTrueMetrics(math.log(0.8), 0.8, -math.log(0.8)),
        restricted=RestrictedLabelMetrics(
            probabilities={"A": 0.7, "B": 0.2, "tie": 0.1},
            msp=0.7,
        ),
        true_token_id=10,
        false_token_id=11,
        label_token_ids={"A": 1, "B": 2, "tie": 3},
        original_token_count=8,
        p_true_token_count=4,
        original_token_ids_sha256=token_ids_sha256(range(8)),
        p_true_token_ids_sha256=token_ids_sha256(range(4)),
        vocabulary_size=128,
        tokenizer_vocabulary_size=120,
        collector_spec_hash="spec-hash",
    )
    assert row["hf_source_pairwise_logit_gap_available_count"] == 1
    assert row["hf_source_pairwise_logit_gap_complete"] is False
    summary = validate_score_rows_against_selection(
        [row],
        selected_items=[item],
        collector_spec_hash="spec-hash",
        tokenizer=CharacterTokenizer(),
        true_token_id=10,
        false_token_id=11,
        label_token_ids={"A": 1, "B": 2, "tie": 3},
        vocabulary_size=128,
        tokenizer_vocabulary_size=120,
    )
    assert summary["source_pairwise_logit_gap_available_gap_count"] == 1
    assert summary["source_pairwise_logit_gap_total_gap_count"] == 3
    assert summary["source_pairwise_logit_gap_partial_row_count"] == 1
    assert summary["source_pairwise_logit_gap_complete_row_count"] == 0


def _strict_valid_score_row() -> dict[str, object]:
    return make_score_row(
        item=replay_item(),
        full_vocab=FullVocabularyMetrics(0.5, -0.5, -2.0, 2.0),
        p_true=PTrueMetrics(math.log(0.8), 0.8, -math.log(0.8)),
        restricted=RestrictedLabelMetrics(
            probabilities={"A": 0.7, "B": 0.2, "tie": 0.1},
            msp=0.7,
        ),
        true_token_id=10,
        false_token_id=11,
        label_token_ids={"A": 1, "B": 2, "tie": 3},
        original_token_count=8,
        p_true_token_count=4,
        original_token_ids_sha256=token_ids_sha256(range(8)),
        p_true_token_ids_sha256=token_ids_sha256(range(4)),
        vocabulary_size=128,
        tokenizer_vocabulary_size=120,
        collector_spec_hash="spec-hash",
    )


def _strict_validate(row: dict[str, object]) -> None:
    validate_score_rows_against_selection(
        [row],
        selected_items=[replay_item()],
        collector_spec_hash="spec-hash",
        tokenizer=CharacterTokenizer(),
        true_token_id=10,
        false_token_id=11,
        label_token_ids={"A": 1, "B": 2, "tie": 3},
        vocabulary_size=128,
        tokenizer_vocabulary_size=120,
    )


def test_strict_resume_validator_recomputes_all_derived_gates() -> None:
    _strict_validate(_strict_valid_score_row())

    forged = _strict_valid_score_row()
    forged["hf_restricted_label_probabilities"] = {
        "A": 0.6,
        "B": 0.3,
        "tie": 0.1,
    }
    forged["hf_restricted_msp"] = 0.6
    forged["hf_restricted_verdict_probability"] = 0.6
    with pytest.raises(ValueError, match="forged source probability drift"):
        _strict_validate(forged)

    forged_agreement = _strict_valid_score_row()
    forged_agreement["hf_restricted_map_matches_stored"] = False
    with pytest.raises(ValueError, match="forged MAP-agreement diagnostic"):
        _strict_validate(forged_agreement)

    forged_gap = _strict_valid_score_row()
    forged_gap["hf_source_pairwise_logit_gap_max_abs_difference"] = 1.0
    with pytest.raises(ValueError, match="forged logit-gap drift"):
        _strict_validate(forged_gap)


def test_strict_resume_validator_rejects_nan_and_prompt_hash_drift() -> None:
    nan_row = _strict_valid_score_row()
    nan_row["mean_token_entropy"] = float("nan")
    with pytest.raises(ValueError, match="finite number"):
        _strict_validate(nan_row)

    prompt_drift = _strict_valid_score_row()
    prompt_drift["original_prompt_text_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="original_prompt_text_sha256"):
        _strict_validate(prompt_drift)

    token_drift = _strict_valid_score_row()
    token_drift["original_token_ids_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="original_token_ids_sha256"):
        _strict_validate(token_drift)
