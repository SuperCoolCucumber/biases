from __future__ import annotations

from biases.stats import (
    benjamini_hochberg,
    bootstrap_bca_ci,
    delong_test,
    mann_whitney_u,
    mcnemar_exact,
    roc_auc,
    wilson_ci,
)


def test_mcnemar_exact_uses_discordant_pairs() -> None:
    result = mcnemar_exact(10, 0)

    assert result.statistic == 0
    assert abs(result.p_value - 0.001953125) < 1e-12


def test_mcnemar_exact_no_discordance_is_neutral() -> None:
    assert mcnemar_exact(0, 0).p_value == 1.0


def test_wilson_ci_for_half_success_rate() -> None:
    interval = wilson_ci(50, 100)

    assert interval.estimate == 0.5
    assert 0.40 < interval.low < 0.41
    assert 0.59 < interval.high < 0.60


def test_bootstrap_bca_ci_contains_mean() -> None:
    interval = bootstrap_bca_ci([1, 2, 3, 4, 5], n_resamples=500, seed=1)

    assert interval.estimate == 3.0
    assert interval.low <= 3.0 <= interval.high


def test_mann_whitney_reports_positive_rank_biserial_for_larger_first_sample() -> None:
    result = mann_whitney_u([3, 4, 5], [1, 2, 3])

    assert result.u > 4.5
    assert result.rank_biserial > 0.0
    assert 0.0 <= result.p_value <= 1.0


def test_roc_auc_handles_ties() -> None:
    auc = roc_auc([1, 1, 0, 0], [0.9, 0.5, 0.5, 0.1])

    assert auc == 0.875


def test_delong_identical_scores_are_neutral() -> None:
    labels = [1, 1, 0, 0, 1, 0]
    scores = [0.9, 0.8, 0.7, 0.2, 0.6, 0.1]

    result = delong_test(labels, scores, scores)

    assert result.auc_1 == result.auc_2
    assert result.p_value == 1.0


def test_delong_detects_auc_order() -> None:
    labels = [1, 1, 0, 0, 1, 0]
    stronger = [0.9, 0.8, 0.2, 0.1, 0.7, 0.3]
    weaker = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

    result = delong_test(labels, stronger, weaker)

    assert result.auc_1 > result.auc_2
    assert 0.0 <= result.p_value <= 1.0


def test_benjamini_hochberg_preserves_input_order() -> None:
    adjusted = benjamini_hochberg([0.01, 0.04, 0.03])

    assert adjusted == [0.03, 0.04, 0.04]
