from __future__ import annotations

import numpy as np

from biases.internal_uncertainty import (
    LinearProbeScorer,
    MahalanobisDistanceSeq,
    RelativeMahalanobisDistanceSeq,
    score_internal_states,
)


def test_mahalanobis_scores_far_point_higher() -> None:
    train = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1]])
    scorer = MahalanobisDistanceSeq().fit(train)

    near, far = scorer.score(np.array([[0.05, 0.05], [3.0, 3.0]]))

    assert far > near


def test_relative_mahalanobis_subtracts_background_distance() -> None:
    foreground = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]])
    background = np.array([[3.0, 3.0], [3.1, 3.0], [3.0, 3.1]])
    scorer = RelativeMahalanobisDistanceSeq().fit(foreground, background)

    foreground_like, background_like = scorer.score(np.array([[0.0, 0.0], [3.0, 3.0]]))

    assert foreground_like < background_like


def test_linear_probe_learns_separable_points() -> None:
    x = np.array([[0.0, 0.0], [0.1, 0.1], [2.0, 2.0], [2.1, 2.1]])
    y = [0, 0, 1, 1]
    probe = LinearProbeScorer(epochs=300, learning_rate=0.2).fit(x, y)

    low, high = probe.score(np.array([[0.0, 0.0], [2.2, 2.2]]))

    assert high > low
    assert high > 0.5


def test_score_internal_states_returns_joinable_rows() -> None:
    rows = [
        {"record_id": "r1", "pair_id": "p1", "variant_id": "control", "routing_split": "calibration"},
        {"record_id": "r2", "pair_id": "p2", "variant_id": "control", "routing_split": "calibration"},
        {"record_id": "r3", "pair_id": "p3", "variant_id": "control", "routing_split": "test"},
        {"record_id": "r4", "pair_id": "p4", "variant_id": "control", "routing_split": "test"},
    ]
    states = np.array([[0.0, 0.0], [2.0, 2.0], [0.1, 0.1], [2.1, 2.1]])
    scores = score_internal_states(
        record_rows=rows,
        hidden_states=states,
        calibration_mask=[True, True, False, False],
        event_labels=[False, True, False, True],
    )

    assert len(scores) == 4
    assert scores[0].record_id == "r1"
    assert "mahalanobis_distance" in scores[0].scores
    assert "linear_probe_probability" in scores[0].scores
