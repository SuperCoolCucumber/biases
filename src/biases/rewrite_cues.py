from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable

from biases.paths import data_path
from biases.utils import stable_hash, write_jsonl


class CueFamily(str, Enum):
    AUTHORITY_CREDENTIAL = "authority_credential"
    AUTHORITY_CITATION = "authority_citation"
    CONFIDENCE = "confidence"
    SOCIAL_PROOF = "social_proof"


class CueDose(str, Enum):
    NEUTRAL = "neutral"
    LOW = "low"
    HIGH = "high"


@dataclass(frozen=True)
class CueRewriteRequest:
    pair_id: str
    target_side: str
    neutral_text: str
    cue_family: CueFamily
    dose: CueDose
    seed: int = 42


@dataclass(frozen=True)
class CueVariant:
    variant_id: str
    pair_id: str
    target_side: str
    cue_family: str
    dose: str
    neutral_text: str
    rewritten_text: str
    content_preserved: bool
    length_ratio: float
    metadata: dict[str, str]


def build_rewrite_prompt(request: CueRewriteRequest) -> str:
    return (
        "Rewrite the target answer to add only the requested stylistic cue.\n"
        "Preserve every factual claim. Add no new information. Keep length within +/-15%.\n\n"
        f"Cue family: {request.cue_family.value}\n"
        f"Dose: {request.dose.value}\n"
        f"Target answer:\n{request.neutral_text}\n\n"
        "Return only the rewritten answer."
    )


def heuristic_rewrite(request: CueRewriteRequest) -> str:
    """Deterministic fallback rewriter for tests and dry runs.

    Production runs should replace this with a strong rewriter plus NLI
    validation. The heuristic keeps the original content and only adds compact
    style markers.
    """

    text = request.neutral_text.strip()
    if request.dose == CueDose.NEUTRAL:
        return text
    if request.cue_family == CueFamily.CONFIDENCE:
        prefix = "I am confident that " if request.dose == CueDose.LOW else "The answer is clear: "
        return _fit_length(prefix + _lowercase_first(text), text)
    if request.cue_family == CueFamily.SOCIAL_PROOF:
        prefix = "Many readers prefer this phrasing: " if request.dose == CueDose.LOW else "Most readers agree: "
        return _fit_length(prefix + text, text)
    if request.cue_family == CueFamily.AUTHORITY_CITATION:
        suffix = " This matches standard references." if request.dose == CueDose.LOW else " This is well documented in standard references."
        return _fit_length(text + suffix, text)
    prefix = "As an experienced evaluator, " if request.dose == CueDose.LOW else "As a recognized expert, "
    return _fit_length(prefix + _lowercase_first(text), text)


def _lowercase_first(text: str) -> str:
    if not text:
        return text
    return text[0].lower() + text[1:]


def _fit_length(rewrite: str, neutral: str, max_ratio: float = 1.15) -> str:
    neutral_words = neutral.split()
    rewrite_words = rewrite.split()
    max_words = max(1, int(len(neutral_words) * max_ratio))
    if len(rewrite_words) <= max_words:
        return rewrite
    return " ".join(rewrite_words[:max_words])


def length_ratio(candidate: str, reference: str) -> float:
    reference_len = max(1, len(reference.split()))
    return len(candidate.split()) / reference_len


def length_match_ok(candidate: str, reference: str, *, tolerance: float = 0.15) -> bool:
    ratio = length_ratio(candidate, reference)
    return (1.0 - tolerance) <= ratio <= (1.0 + tolerance)


def lexical_content_preserved(candidate: str, reference: str, *, min_overlap: float = 0.55) -> bool:
    reference_terms = _content_terms(reference)
    if not reference_terms:
        return True
    candidate_terms = _content_terms(candidate)
    return len(reference_terms & candidate_terms) / len(reference_terms) >= min_overlap


def _content_terms(text: str) -> set[str]:
    return {
        token.strip(".,:;!?()[]{}\"'").lower()
        for token in text.split()
        if len(token.strip(".,:;!?()[]{}\"'")) >= 4
    }


def make_cue_variant(request: CueRewriteRequest, rewritten_text: str | None = None) -> CueVariant:
    rewritten = rewritten_text if rewritten_text is not None else heuristic_rewrite(request)
    ratio = length_ratio(rewritten, request.neutral_text)
    preserved = lexical_content_preserved(rewritten, request.neutral_text) and length_match_ok(
        rewritten,
        request.neutral_text,
    )
    variant_id = stable_hash(
        {
            "pair_id": request.pair_id,
            "target_side": request.target_side,
            "cue_family": request.cue_family.value,
            "dose": request.dose.value,
            "seed": request.seed,
        }
    )
    return CueVariant(
        variant_id=variant_id,
        pair_id=request.pair_id,
        target_side=request.target_side,
        cue_family=request.cue_family.value,
        dose=request.dose.value,
        neutral_text=request.neutral_text,
        rewritten_text=rewritten,
        content_preserved=preserved,
        length_ratio=ratio,
        metadata={"seed": str(request.seed)},
    )


def write_cue_variants(
    variants: Iterable[CueVariant],
    path: Path = data_path("processed", "cued_variants.jsonl"),
) -> None:
    write_jsonl(path, [asdict(variant) for variant in variants])


def read_cue_variants(path: Path) -> list[CueVariant]:
    variants: list[CueVariant] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            variants.append(CueVariant(**row))
    return variants
