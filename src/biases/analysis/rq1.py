from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from biases.analysis.records import LABELS, ConditionRecord, PairedCondition
from biases.analysis.resampling import cluster_resamples, percentile


def _difference(biased: float | None, clean: float | None) -> float | None:
    if biased is None or clean is None:
        return None
    return biased - clean


def jensen_shannon_divergence(
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    if len(first) != len(second) or not first:
        raise ValueError("probability vectors must be non-empty and equal length")
    if any(value < 0.0 for value in (*first, *second)):
        raise ValueError("probabilities must be non-negative")
    first_total = sum(first)
    second_total = sum(second)
    if first_total <= 0.0 or second_total <= 0.0:
        raise ValueError("each probability vector must have positive mass")
    p = [value / first_total for value in first]
    q = [value / second_total for value in second]
    midpoint = [(left + right) / 2.0 for left, right in zip(p, q, strict=True)]

    def kl(left: Sequence[float], right: Sequence[float]) -> float:
        return sum(
            probability * math.log2(probability / reference)
            for probability, reference in zip(left, right, strict=True)
            if probability > 0.0
        )

    return (kl(p, midpoint) + kl(q, midpoint)) / 2.0


@dataclass(frozen=True, slots=True)
class PairedShift:
    clean_record_id: str
    cued_record_id: str
    example_id: str
    question_id: str
    pair_key: str
    condition_group_id: str | None
    ordering: str
    model_name: str
    routing_split: str | None
    family: str
    direction: str
    dose: float | None
    variant_id: str
    cue_target: str | None
    human_winner: str | None
    clean_verdict: str
    cued_verdict: str
    clean_tie: bool
    delta_entropy: float | None
    delta_normalized_entropy: float | None
    delta_msp: float | None
    delta_margin: float | None
    delta_verbalized_confidence: float | None
    delta_consistency_entropy: float | None
    signed_cue_mass: float | None
    js_divergence: float | None
    flip: bool
    error: bool | None
    clean_entropy: float | None
    clean_msp: float | None
    clean_margin: float | None
    clean_verbalized_confidence: float | None
    clean_consistency_entropy: float | None
    cued_entropy: float | None
    cued_msp: float | None
    cued_margin: float | None
    cued_verbalized_confidence: float | None
    cued_consistency_entropy: float | None

    @property
    def susceptibility_key(self) -> tuple[str, str, str, str, str]:
        return self.model_name, self.family, self.pair_key, self.ordering, self.direction

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def compute_paired_shift(pair: PairedCondition) -> PairedShift:
    clean = pair.clean
    cued = pair.cued
    if clean.key != cued.key and cued.clean_record_id != clean.record_id:
        raise ValueError("clean and cued records do not share a pairing key")
    clean_probs = clean.probabilities
    cued_probs = cued.probabilities
    js = (
        jensen_shannon_divergence(clean_probs, cued_probs)
        if clean_probs is not None and cued_probs is not None
        else None
    )
    clean_target_mass = clean.probability_for(cued.cue_target) if cued.cue_target else None
    cued_target_mass = cued.probability_for(cued.cue_target) if cued.cue_target else None
    signed_mass = _difference(cued_target_mass, clean_target_mass)
    human = cued.human_winner or clean.human_winner
    return PairedShift(
        clean_record_id=clean.record_id,
        cued_record_id=cued.record_id,
        example_id=cued.example_id,
        question_id=cued.question_id,
        pair_key=cued.pair_key,
        condition_group_id=cued.condition_group_id,
        ordering=cued.ordering,
        model_name=cued.model_name,
        routing_split=cued.routing_split or clean.routing_split,
        family=cued.family,
        direction=cued.direction,
        dose=cued.dose,
        variant_id=cued.variant_id,
        cue_target=cued.cue_target,
        human_winner=human,
        clean_verdict=clean.verdict,
        cued_verdict=cued.verdict,
        clean_tie=clean.verdict == "tie" or cued.clean_tie,
        delta_entropy=_difference(cued.entropy, clean.entropy),
        delta_normalized_entropy=_difference(cued.normalized_entropy, clean.normalized_entropy),
        delta_msp=_difference(cued.msp, clean.msp),
        delta_margin=_difference(cued.margin, clean.margin),
        delta_verbalized_confidence=_difference(
            cued.verbalized_confidence,
            clean.verbalized_confidence,
        ),
        delta_consistency_entropy=_difference(
            cued.consistency_entropy,
            clean.consistency_entropy,
        ),
        signed_cue_mass=signed_mass,
        js_divergence=js,
        flip=cued.verdict != clean.verdict,
        error=None if human is None else cued.verdict != human,
        clean_entropy=clean.entropy,
        clean_msp=clean.msp,
        clean_margin=clean.margin,
        clean_verbalized_confidence=clean.verbalized_confidence,
        clean_consistency_entropy=clean.consistency_entropy,
        cued_entropy=cued.entropy,
        cued_msp=cued.msp,
        cued_margin=cued.margin,
        cued_verbalized_confidence=cued.verbalized_confidence,
        cued_consistency_entropy=cued.consistency_entropy,
    )


def compute_paired_shifts(pairs: Sequence[PairedCondition]) -> tuple[PairedShift, ...]:
    return tuple(compute_paired_shift(pair) for pair in pairs)


def roc_auc(labels: Sequence[bool | int], scores: Sequence[float]) -> float | None:
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    positives = [float(score) for label, score in zip(labels, scores, strict=True) if bool(label)]
    negatives = [float(score) for label, score in zip(labels, scores, strict=True) if not bool(label)]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += float(positive > negative) + 0.5 * float(positive == negative)
    return wins / (len(positives) * len(negatives))


def shift_metric_value(shift: PairedShift, metric: str) -> float | None:
    aliases = {
        "entropy": "delta_entropy",
        "normalized_entropy": "delta_normalized_entropy",
        "msp": "delta_msp",
        "margin": "delta_margin",
        "verbalized_confidence": "delta_verbalized_confidence",
        "consistency_entropy": "delta_consistency_entropy",
        "cue_mass": "signed_cue_mass",
        "js": "js_divergence",
    }
    attribute = aliases.get(metric, metric)
    value = getattr(shift, attribute, None)
    return float(value) if value is not None else None


def clean_uncertainty_value(shift: PairedShift, channel: str) -> float | None:
    if channel in {"entropy", "normalized_entropy", "consistency_entropy"}:
        value = getattr(shift, f"clean_{channel}", None)
    elif channel == "msp":
        value = None if shift.clean_msp is None else 1.0 - shift.clean_msp
    elif channel == "margin":
        value = None if shift.clean_margin is None else 1.0 - shift.clean_margin
    elif channel == "verbalized_confidence":
        value = (
            None
            if shift.clean_verbalized_confidence is None
            else 1.0 - shift.clean_verbalized_confidence
        )
    else:
        raise ValueError(f"unknown uncertainty channel: {channel}")
    return float(value) if value is not None else None


@dataclass(frozen=True, slots=True)
class SusceptibilityAUC:
    model_name: str
    family: str
    direction: str
    low_dose: float
    high_dose: float
    shift_metric: str
    baseline_channel: str
    n: int
    positives: int
    shift_auc: float | None
    clean_baseline_auc: float | None
    auc_difference: float | None


@dataclass(frozen=True, slots=True)
class SusceptibilityExample:
    question_id: str
    event: bool
    shift_score: float
    baseline_score: float


def _matched_susceptibility_examples(
    shifts: Sequence[PairedShift],
    *,
    shift_metric: str,
    baseline_channel: str,
    low_dose: float,
    high_dose: float,
) -> tuple[SusceptibilityExample, ...]:
    low_rows = {
        shift.susceptibility_key: shift for shift in shifts if shift.dose == low_dose
    }
    high_rows = {
        shift.susceptibility_key: shift for shift in shifts if shift.dose == high_dose
    }
    examples: list[SusceptibilityExample] = []
    for key in sorted(low_rows, key=repr):
        low_row = low_rows[key]
        high_row = high_rows.get(key)
        shift_score = shift_metric_value(low_row, shift_metric)
        baseline_score = clean_uncertainty_value(low_row, baseline_channel)
        if high_row is None or shift_score is None or baseline_score is None:
            continue
        examples.append(
            SusceptibilityExample(
                question_id=low_row.question_id,
                event=high_row.flip,
                shift_score=shift_score,
                baseline_score=baseline_score,
            )
        )
    return tuple(examples)


def low_dose_susceptibility_auc(
    shifts: Sequence[PairedShift],
    *,
    shift_metric: str = "signed_cue_mass",
    baseline_channel: str = "entropy",
    low_dose: float | None = None,
    high_dose: float | None = None,
) -> SusceptibilityAUC:
    if not shifts:
        raise ValueError("shifts must not be empty")
    doses = sorted({shift.dose for shift in shifts if shift.dose is not None})
    if not doses:
        raise ValueError("dose values are required")
    low = doses[0] if low_dose is None else float(low_dose)
    high = doses[-1] if high_dose is None else float(high_dose)
    examples = _matched_susceptibility_examples(
        shifts,
        shift_metric=shift_metric,
        baseline_channel=baseline_channel,
        low_dose=low,
        high_dose=high,
    )
    labels = [example.event for example in examples]
    shift_scores = [example.shift_score for example in examples]
    baseline_scores = [example.baseline_score for example in examples]
    shift_auc = roc_auc(labels, shift_scores)
    baseline_auc = roc_auc(labels, baseline_scores)
    first = shifts[0]
    return SusceptibilityAUC(
        model_name=first.model_name,
        family=first.family,
        direction=first.direction,
        low_dose=low,
        high_dose=high,
        shift_metric=shift_metric,
        baseline_channel=baseline_channel,
        n=len(labels),
        positives=sum(labels),
        shift_auc=shift_auc,
        clean_baseline_auc=baseline_auc,
        auc_difference=(
            None if shift_auc is None or baseline_auc is None else shift_auc - baseline_auc
        ),
    )


@dataclass(frozen=True, slots=True)
class SusceptibilityAUCWithCI:
    estimate: SusceptibilityAUC
    shift_auc_ci_low: float | None
    shift_auc_ci_high: float | None
    clean_baseline_auc_ci_low: float | None
    clean_baseline_auc_ci_high: float | None
    auc_difference_ci_low: float | None
    auc_difference_ci_high: float | None
    confidence: float
    n_clusters: int
    n_resamples: int


def low_dose_susceptibility_auc_with_cluster_bootstrap(
    shifts: Sequence[PairedShift],
    *,
    shift_metric: str = "signed_cue_mass",
    baseline_channel: str = "entropy",
    low_dose: float | None = None,
    high_dose: float | None = None,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
) -> SusceptibilityAUCWithCI:
    estimate = low_dose_susceptibility_auc(
        shifts,
        shift_metric=shift_metric,
        baseline_channel=baseline_channel,
        low_dose=low_dose,
        high_dose=high_dose,
    )
    examples = _matched_susceptibility_examples(
        shifts,
        shift_metric=shift_metric,
        baseline_channel=baseline_channel,
        low_dose=estimate.low_dose,
        high_dose=estimate.high_dose,
    )
    shift_aucs: list[float] = []
    baseline_aucs: list[float] = []
    differences: list[float] = []
    if examples:
        for sample in cluster_resamples(
            examples,
            cluster_key=lambda example: example.question_id,
            n_resamples=n_resamples,
            seed=seed,
        ):
            labels = [example.event for example in sample]
            shift_auc = roc_auc(labels, [example.shift_score for example in sample])
            baseline_auc = roc_auc(labels, [example.baseline_score for example in sample])
            if shift_auc is not None:
                shift_aucs.append(shift_auc)
            if baseline_auc is not None:
                baseline_aucs.append(baseline_auc)
            if shift_auc is not None and baseline_auc is not None:
                differences.append(shift_auc - baseline_auc)
    alpha = 1.0 - confidence

    def bounds(values: Sequence[float]) -> tuple[float | None, float | None]:
        if not values:
            return None, None
        return percentile(values, alpha / 2.0), percentile(values, 1.0 - alpha / 2.0)

    shift_low, shift_high = bounds(shift_aucs)
    baseline_low, baseline_high = bounds(baseline_aucs)
    difference_low, difference_high = bounds(differences)
    return SusceptibilityAUCWithCI(
        estimate=estimate,
        shift_auc_ci_low=shift_low,
        shift_auc_ci_high=shift_high,
        clean_baseline_auc_ci_low=baseline_low,
        clean_baseline_auc_ci_high=baseline_high,
        auc_difference_ci_low=difference_low,
        auc_difference_ci_high=difference_high,
        confidence=confidence,
        n_clusters=len({example.question_id for example in examples}),
        n_resamples=n_resamples,
    )
