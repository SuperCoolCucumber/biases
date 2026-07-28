from __future__ import annotations

from biases.dataset_adapters import (
    adapt_llmbar_rows,
    adapt_rewardbench_rows,
    randomized_pair,
)


def test_randomized_pair_preserves_winner_after_assignment() -> None:
    pair = randomized_pair(
        source_dataset="fixture",
        source_id="1",
        prompt="Question?",
        chosen="Better",
        rejected="Worse",
        seed=42,
    )

    if pair.winner == "model_a":
        assert pair.response_a == "Better"
        assert pair.response_b == "Worse"
    else:
        assert pair.response_a == "Worse"
        assert pair.response_b == "Better"


def test_rewardbench_adapter_maps_chosen_rejected() -> None:
    pairs = adapt_rewardbench_rows(
        [
            {
                "id": "rb1",
                "prompt": "Who?",
                "chosen": "Correct",
                "rejected": "Incorrect",
            }
        ],
        seed=1,
    )

    assert len(pairs) == 1
    assert pairs[0].source_dataset == "rewardbench"
    assert pairs[0].winner in {"model_a", "model_b"}


def test_llmbar_adapter_uses_label() -> None:
    pairs = adapt_llmbar_rows(
        [
            {
                "id": "lb1",
                "input": "Summarize.",
                "output_1": "Good",
                "output_2": "Bad",
                "label": "1",
            }
        ],
        seed=1,
    )

    assert len(pairs) == 1
    pair = pairs[0]
    chosen = pair.response_a if pair.winner == "model_a" else pair.response_b
    assert chosen == "Good"
