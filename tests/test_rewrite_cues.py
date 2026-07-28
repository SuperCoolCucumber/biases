from __future__ import annotations

from pathlib import Path

from biases.cue_runner import apply_cue_variant, validate_within_pair_invariance
from biases.position_bias import load_position_pairs
from biases.rewrite_cues import (
    CueDose,
    CueFamily,
    CueRewriteRequest,
    build_rewrite_prompt,
    length_match_ok,
    make_cue_variant,
)


def test_rewrite_prompt_contains_guardrails() -> None:
    request = CueRewriteRequest(
        pair_id="p1",
        target_side="B",
        neutral_text="The answer is concise.",
        cue_family=CueFamily.CONFIDENCE,
        dose=CueDose.HIGH,
    )

    prompt = build_rewrite_prompt(request)

    assert "Preserve every factual claim" in prompt
    assert "confidence" in prompt


def test_make_cue_variant_tracks_length_and_content() -> None:
    request = CueRewriteRequest(
        pair_id="p1",
        target_side="A",
        neutral_text="Superman was created by Jerry Siegel and Joe Shuster in 1938.",
        cue_family=CueFamily.AUTHORITY_CITATION,
        dose=CueDose.LOW,
    )

    variant = make_cue_variant(request)

    assert variant.content_preserved
    assert length_match_ok(variant.rewritten_text, variant.neutral_text)


def test_apply_cue_variant_preserves_non_target_answer(tmp_path: Path) -> None:
    csv_path = tmp_path / "pairs.csv"
    csv_path.write_text(
        "\n".join(
            [
                "example_id,prompt,response_a,response_b,winner",
                "q1,Question?,Answer A,Answer B,A",
            ]
        ),
        encoding="utf-8",
    )
    pair = load_position_pairs(csv_path)[0]
    variant = make_cue_variant(
        CueRewriteRequest(
            pair_id=pair.pair_id,
            target_side="A",
            neutral_text="Answer A",
            cue_family=CueFamily.CONFIDENCE,
            dose=CueDose.HIGH,
        ),
        rewritten_text="Clearly, Answer A",
    )

    cued = apply_cue_variant(pair.original, variant)

    assert validate_within_pair_invariance(pair.original, cued, "A")
    assert cued.candidates["B"].response == "Answer B"
    assert cued.candidates["A"].response == "Clearly, Answer A"
