from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from biases.analysis.records import LABELS, ConditionRecord, normalize_label, opposite_label
from biases.analysis.resampling import cluster_resamples, percentile
from biases.stats import mcnemar_exact


CONFIDENCE_CHANNELS = (
    "msp",
    "consistency_agreement",
    "verbalized_confidence",
)


@dataclass(frozen=True, slots=True)
class ScoredPrediction:
    record_id: str
    question_id: str
    pair_key: str
    ordering: str
    model_name: str
    routing_split: str | None
    family: str
    direction: str
    dose: float | None
    variant_id: str
    verdict: str
    human_winner: str | None
    probabilities: tuple[float, float, float] | None
    confidence: float | None
    flip: bool | None = None
    clean_tie: bool | None = None
    msp: float | None = None
    consistency_agreement: float | None = None
    verbalized_confidence: float | None = None


def confidence_value(
    prediction: ScoredPrediction,
    confidence_channel: str = "msp",
) -> float | None:
    if confidence_channel not in CONFIDENCE_CHANNELS:
        raise ValueError(f"unknown confidence channel: {confidence_channel}")
    if confidence_channel == "msp":
        value = prediction.msp
        if value is None:
            value = prediction.confidence
    else:
        value = getattr(prediction, confidence_channel)
    if value is None or not math.isfinite(value):
        return None
    return min(1.0, max(0.0, float(value)))


def prediction_from_record(
    record: ConditionRecord,
    *,
    flip: bool | None = None,
) -> ScoredPrediction:
    probs = record.probabilities
    confidence = record.msp if record.msp is not None else (max(probs) if probs else None)
    return ScoredPrediction(
        record_id=record.record_id,
        question_id=record.question_id,
        pair_key=record.pair_key,
        ordering=record.ordering,
        model_name=record.model_name,
        routing_split=record.routing_split,
        family=record.family,
        direction=record.direction,
        dose=record.dose,
        variant_id=record.variant_id,
        verdict=record.verdict,
        human_winner=record.human_winner,
        probabilities=probs,
        confidence=confidence,
        flip=flip,
        clean_tie=record.clean_tie,
        msp=confidence,
        consistency_agreement=record.consistency_agreement,
        verbalized_confidence=record.verbalized_confidence,
    )


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    bin_index: int
    lower: float
    upper: float
    n: int
    mean_confidence: float | None
    accuracy: float | None


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    n: int
    brier_n: int
    ece: float | None
    brier: float | None
    accuracy: float | None
    bins: tuple[ReliabilityBin, ...]


def multiclass_brier(predictions: Sequence[ScoredPrediction]) -> tuple[float | None, int]:
    values: list[float] = []
    for prediction in predictions:
        label = normalize_label(prediction.human_winner)
        if prediction.probabilities is None or label is None:
            continue
        target = LABELS.index(label)
        values.append(
            sum(
                (probability - float(index == target)) ** 2
                for index, probability in enumerate(prediction.probabilities)
            )
        )
    return (sum(values) / len(values), len(values)) if values else (None, 0)


def reliability_diagram(
    predictions: Sequence[ScoredPrediction],
    *,
    n_bins: int = 10,
    confidence_channel: str = "msp",
) -> tuple[ReliabilityBin, ...]:
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    buckets: list[list[tuple[ScoredPrediction, float]]] = [
        [] for _ in range(n_bins)
    ]
    for prediction in predictions:
        confidence = confidence_value(prediction, confidence_channel)
        if confidence is None or prediction.human_winner is None:
            continue
        index = min(n_bins - 1, int(confidence * n_bins))
        buckets[index].append((prediction, confidence))
    result: list[ReliabilityBin] = []
    for index, bucket in enumerate(buckets):
        lower = index / n_bins
        upper = (index + 1) / n_bins
        result.append(
            ReliabilityBin(
                bin_index=index,
                lower=lower,
                upper=upper,
                n=len(bucket),
                mean_confidence=(
                    sum(confidence for _, confidence in bucket) / len(bucket)
                    if bucket
                    else None
                ),
                accuracy=(
                    sum(
                        item.verdict == item.human_winner
                        for item, _ in bucket
                    )
                    / len(bucket)
                    if bucket
                    else None
                ),
            )
        )
    return tuple(result)


def calibration_summary(
    predictions: Sequence[ScoredPrediction],
    *,
    n_bins: int = 10,
    confidence_channel: str = "msp",
) -> CalibrationResult:
    bins = reliability_diagram(
        predictions,
        n_bins=n_bins,
        confidence_channel=confidence_channel,
    )
    valid_n = sum(bin_.n for bin_ in bins)
    ece = (
        sum(
            bin_.n
            * abs(float(bin_.accuracy) - float(bin_.mean_confidence))
            for bin_ in bins
            if bin_.n and bin_.accuracy is not None and bin_.mean_confidence is not None
        )
        / valid_n
        if valid_n
        else None
    )
    brier, brier_n = multiclass_brier(predictions)
    correct = [
        prediction.verdict == prediction.human_winner
        for prediction in predictions
        if prediction.human_winner is not None
        and confidence_value(prediction, confidence_channel) is not None
    ]
    return CalibrationResult(
        n=valid_n,
        brier_n=brier_n,
        ece=ece,
        brier=brier,
        accuracy=sum(correct) / len(correct) if correct else None,
        bins=bins,
    )


@dataclass(frozen=True, slots=True)
class RiskCoveragePoint:
    threshold: float
    coverage: float
    risk: float
    accepted: int
    total: int


@dataclass(frozen=True, slots=True)
class RiskCoverageResult:
    n: int
    aurc: float | None
    points: tuple[RiskCoveragePoint, ...]


def risk_coverage_curve(
    predictions: Sequence[ScoredPrediction],
    *,
    confidence_channel: str = "msp",
) -> RiskCoverageResult:
    valid = [
        (prediction, confidence_value(prediction, confidence_channel))
        for prediction in predictions
        if prediction.human_winner is not None
        and confidence_value(prediction, confidence_channel) is not None
    ]
    if not valid:
        return RiskCoverageResult(n=0, aurc=None, points=())
    thresholds = sorted(
        {float(confidence) for _, confidence in valid if confidence is not None},
        reverse=True,
    )
    points = [RiskCoveragePoint(threshold=math.inf, coverage=0.0, risk=0.0, accepted=0, total=len(valid))]
    for threshold in thresholds:
        accepted = [
            (item, confidence)
            for item, confidence in valid
            if float(confidence) >= threshold
        ]
        errors = sum(
            item.verdict != item.human_winner for item, _ in accepted
        )
        points.append(
            RiskCoveragePoint(
                threshold=threshold,
                coverage=len(accepted) / len(valid),
                risk=errors / len(accepted),
                accepted=len(accepted),
                total=len(valid),
            )
        )
    area = sum(
        (right.coverage - left.coverage) * (left.risk + right.risk) / 2.0
        for left, right in zip(points, points[1:])
    )
    return RiskCoverageResult(n=len(valid), aurc=area, points=tuple(points))


@dataclass(frozen=True, slots=True)
class ThresholdRule:
    confidence_channel: str
    target_risk: float
    threshold: float
    calibration_n: int
    calibration_coverage: float
    calibration_risk: float | None


def calibrate_threshold_at_target_risk(
    clean_calibration: Sequence[ScoredPrediction],
    *,
    target_risk: float,
    confidence_channel: str = "msp",
) -> ThresholdRule:
    if not 0.0 <= target_risk <= 1.0:
        raise ValueError("target_risk must be in [0, 1]")
    curve = risk_coverage_curve(
        clean_calibration,
        confidence_channel=confidence_channel,
    )
    feasible = [
        point
        for point in curve.points
        if point.accepted > 0 and point.risk <= target_risk
    ]
    if not feasible:
        return ThresholdRule(
            confidence_channel=confidence_channel,
            target_risk=target_risk,
            threshold=math.inf,
            calibration_n=curve.n,
            calibration_coverage=0.0,
            calibration_risk=None,
        )
    selected = max(feasible, key=lambda point: (point.coverage, -point.threshold))
    return ThresholdRule(
        confidence_channel=confidence_channel,
        target_risk=target_risk,
        threshold=selected.threshold,
        calibration_n=curve.n,
        calibration_coverage=selected.coverage,
        calibration_risk=selected.risk,
    )


@dataclass(frozen=True, slots=True)
class ThresholdTransfer:
    confidence_channel: str
    target_risk: float
    threshold: float
    n: int
    accepted: int
    coverage: float | None
    realized_risk: float | None
    risk_inflation_vs_target: float | None
    risk_inflation_vs_clean_calibration: float | None
    flips: int
    accepted_flips: int
    accepted_flip_fraction: float | None


def evaluate_threshold_transfer(
    predictions: Sequence[ScoredPrediction],
    rule: ThresholdRule,
) -> ThresholdTransfer:
    valid = [
        (prediction, confidence_value(prediction, rule.confidence_channel))
        for prediction in predictions
        if prediction.human_winner is not None
        and confidence_value(prediction, rule.confidence_channel) is not None
    ]
    accepted = [
        (prediction, confidence)
        for prediction, confidence in valid
        if float(confidence) >= rule.threshold
    ]
    realized = (
        sum(item.verdict != item.human_winner for item, _ in accepted)
        / len(accepted)
        if accepted
        else None
    )
    flip_rows = [item for item, _ in valid if item.flip is True]
    accepted_flip_count = sum(
        item.flip is True and float(confidence) >= rule.threshold
        for item, confidence in valid
    )
    return ThresholdTransfer(
        confidence_channel=rule.confidence_channel,
        target_risk=rule.target_risk,
        threshold=rule.threshold,
        n=len(valid),
        accepted=len(accepted),
        coverage=len(accepted) / len(valid) if valid else None,
        realized_risk=realized,
        risk_inflation_vs_target=None if realized is None else realized - rule.target_risk,
        risk_inflation_vs_clean_calibration=(
            None
            if realized is None or rule.calibration_risk is None
            else realized - rule.calibration_risk
        ),
        flips=len(flip_rows),
        accepted_flips=accepted_flip_count,
        accepted_flip_fraction=(
            accepted_flip_count / len(flip_rows) if flip_rows else None
        ),
    )


@dataclass(frozen=True, slots=True)
class ThresholdTransferWithCI:
    rule: ThresholdRule
    estimate: ThresholdTransfer
    realized_risk_ci_low: float | None
    realized_risk_ci_high: float | None
    risk_inflation_vs_target_ci_low: float | None
    risk_inflation_vs_target_ci_high: float | None
    risk_inflation_vs_target_p_value_one_sided: float | None
    risk_inflation_vs_clean_calibration_ci_low: float | None
    risk_inflation_vs_clean_calibration_ci_high: float | None
    accepted_flip_fraction_ci_low: float | None
    accepted_flip_fraction_ci_high: float | None
    confidence: float
    n_calibration_clusters: int
    n_test_clusters: int
    n_resamples: int


def clean_calibrated_threshold_transfer_with_cluster_bootstrap(
    clean_calibration: Sequence[ScoredPrediction],
    biased_test: Sequence[ScoredPrediction],
    *,
    target_risk: float,
    confidence_channel: str = "msp",
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 42,
) -> ThresholdTransferWithCI:
    """Calibrate on clean data, then bootstrap calibration and test questions.

    Calibration and test splits are resampled independently. Every bootstrap
    draw recalibrates the threshold, so the intervals include threshold
    selection uncertainty as well as biased-test sampling uncertainty.
    """

    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    rule = calibrate_threshold_at_target_risk(
        clean_calibration,
        target_risk=target_risk,
        confidence_channel=confidence_channel,
    )
    estimate = evaluate_threshold_transfer(biased_test, rule)
    metrics: dict[str, list[float]] = {
        "realized_risk": [],
        "risk_inflation_vs_target": [],
        "risk_inflation_vs_clean_calibration": [],
        "accepted_flip_fraction": [],
    }
    if clean_calibration and biased_test:
        calibration_samples = cluster_resamples(
            clean_calibration,
            cluster_key=lambda prediction: prediction.question_id,
            n_resamples=n_resamples,
            seed=seed,
        )
        test_samples = cluster_resamples(
            biased_test,
            cluster_key=lambda prediction: prediction.question_id,
            n_resamples=n_resamples,
            seed=seed + 1,
        )
        for calibration_sample, test_sample in zip(
            calibration_samples,
            test_samples,
            strict=True,
        ):
            sampled_rule = calibrate_threshold_at_target_risk(
                calibration_sample,
                target_risk=target_risk,
                confidence_channel=confidence_channel,
            )
            sampled = evaluate_threshold_transfer(test_sample, sampled_rule)
            for field in metrics:
                value = getattr(sampled, field)
                if value is not None and math.isfinite(value):
                    metrics[field].append(float(value))
    alpha = 1.0 - confidence

    def bounds(name: str) -> tuple[float | None, float | None]:
        values = metrics[name]
        if not values:
            return None, None
        return percentile(values, alpha / 2.0), percentile(values, 1.0 - alpha / 2.0)

    risk_low, risk_high = bounds("realized_risk")
    target_low, target_high = bounds("risk_inflation_vs_target")
    clean_low, clean_high = bounds("risk_inflation_vs_clean_calibration")
    flips_low, flips_high = bounds("accepted_flip_fraction")
    return ThresholdTransferWithCI(
        rule=rule,
        estimate=estimate,
        realized_risk_ci_low=risk_low,
        realized_risk_ci_high=risk_high,
        risk_inflation_vs_target_ci_low=target_low,
        risk_inflation_vs_target_ci_high=target_high,
        risk_inflation_vs_target_p_value_one_sided=(
            (
                1
                + sum(
                    value <= 0.0
                    for value in metrics["risk_inflation_vs_target"]
                )
            )
            / (len(metrics["risk_inflation_vs_target"]) + 1)
            if metrics["risk_inflation_vs_target"]
            else None
        ),
        risk_inflation_vs_clean_calibration_ci_low=clean_low,
        risk_inflation_vs_clean_calibration_ci_high=clean_high,
        accepted_flip_fraction_ci_low=flips_low,
        accepted_flip_fraction_ci_high=flips_high,
        confidence=confidence,
        n_calibration_clusters=len(
            {prediction.question_id for prediction in clean_calibration}
        ),
        n_test_clusters=len({prediction.question_id for prediction in biased_test}),
        n_resamples=n_resamples,
    )


@dataclass(frozen=True, slots=True)
class PairedCorrectnessMcNemar:
    n: int
    clean_correct: int
    cued_correct: int
    b_clean_correct_cued_wrong: int
    c_clean_wrong_cued_correct: int
    statistic: int
    p_value: float


def paired_correctness_mcnemar(
    clean_verdicts: Sequence[str],
    cued_verdicts: Sequence[str],
    human_labels: Sequence[str | None],
) -> PairedCorrectnessMcNemar:
    if not (
        len(clean_verdicts) == len(cued_verdicts) == len(human_labels)
    ):
        raise ValueError("clean, cued, and human label arrays must have equal length")
    clean_status: list[bool] = []
    cued_status: list[bool] = []
    for clean, cued, human in zip(
        clean_verdicts,
        cued_verdicts,
        human_labels,
        strict=True,
    ):
        normalized_human = normalize_label(human)
        if normalized_human is None:
            continue
        clean_status.append(normalize_label(clean) == normalized_human)
        cued_status.append(normalize_label(cued) == normalized_human)
    b = sum(
        clean_correct and not cued_correct
        for clean_correct, cued_correct in zip(clean_status, cued_status, strict=True)
    )
    c = sum(
        not clean_correct and cued_correct
        for clean_correct, cued_correct in zip(clean_status, cued_status, strict=True)
    )
    result = mcnemar_exact(b, c)
    return PairedCorrectnessMcNemar(
        n=len(clean_status),
        clean_correct=sum(clean_status),
        cued_correct=sum(cued_status),
        b_clean_correct_cued_wrong=b,
        c_clean_wrong_cued_correct=c,
        statistic=result.statistic,
        p_value=result.p_value,
    )


def _canonical_label(label: str | None, ordering: str) -> str | None:
    normalized = normalize_label(label)
    if normalized is None:
        return None
    return opposite_label(normalized) if ordering in {"ba", "swapped"} else normalized


def _canonical_probabilities(record: ConditionRecord) -> tuple[float, float, float] | None:
    probs = record.probabilities
    if probs is None:
        return None
    return (probs[1], probs[0], probs[2]) if record.ordering in {"ba", "swapped"} else probs


def swap_average_pair(
    first: ConditionRecord,
    second: ConditionRecord,
    *,
    flip: bool | None = None,
) -> ScoredPrediction:
    if {first.ordering, second.ordering} not in ({"ab", "ba"}, {"original", "swapped"}):
        raise ValueError("swap averaging requires one AB/original and one BA/swapped record")
    if first.model_name != second.model_name or first.question_id != second.question_id:
        raise ValueError("ordering twins must share model_name and question_id")
    if (first.family, first.direction, first.dose) != (
        second.family,
        second.direction,
        second.dose,
    ):
        raise ValueError("ordering twins must represent the same condition")
    if (
        first.pair_identity_key is not None
        and second.pair_identity_key is not None
        and first.pair_identity_key != second.pair_identity_key
    ):
        raise ValueError("ordering twins disagree on pair_identity_key")
    first_probs = _canonical_probabilities(first)
    second_probs = _canonical_probabilities(second)
    if first_probs is None or second_probs is None:
        raise ValueError("swap averaging requires complete label probabilities")
    averaged = tuple(
        (left + right) / 2.0
        for left, right in zip(first_probs, second_probs, strict=True)
    )
    maximum = max(averaged)
    winners = [LABELS[index] for index, value in enumerate(averaged) if value == maximum]
    verdict = winners[0] if len(winners) == 1 else "tie"
    human_first = _canonical_label(first.human_winner, first.ordering)
    human_second = _canonical_label(second.human_winner, second.ordering)
    if human_first is not None and human_second is not None and human_first != human_second:
        raise ValueError("ordering twins disagree on the canonical human label")
    source_pair_key = re.sub(
        r"(?::|_)(?:original|swapped|ab|ba)$",
        "",
        first.example_id,
        flags=re.IGNORECASE,
    )
    return ScoredPrediction(
        record_id=f"swap-average:{first.record_id}:{second.record_id}",
        question_id=first.question_id,
        pair_key=first.pair_identity_key or source_pair_key or first.pair_key,
        ordering="swap_average",
        model_name=first.model_name,
        routing_split=first.routing_split,
        family=first.family,
        direction=first.direction,
        dose=first.dose,
        variant_id=re.sub(r"_(?:ab|ba|original|swapped)$", "", first.variant_id),
        verdict=verdict,
        human_winner=human_first or human_second,
        probabilities=averaged,
        confidence=maximum,
        flip=flip,
        clean_tie=bool(first.clean_tie or second.clean_tie),
        msp=maximum,
    )


def swap_average_records(records: Sequence[ConditionRecord]) -> tuple[ScoredPrediction, ...]:
    grouped: dict[tuple[object, ...], list[ConditionRecord]] = defaultdict(list)
    for record in records:
        base_pair = record.pair_identity_key or record.condition_group_id or re.sub(
            r"(?:[_:]?(?:ab|ba|original|swapped))$",
            "",
            record.pair_key,
            flags=re.IGNORECASE,
        )
        base_variant = re.sub(
            r"_(?:ab|ba|original|swapped)$",
            "",
            record.variant_id,
            flags=re.IGNORECASE,
        )
        key = (
            record.model_name,
            base_pair,
            record.family,
            record.direction,
            record.dose,
            base_variant,
        )
        grouped[key].append(record)
    averaged: list[ScoredPrediction] = []
    for key in sorted(grouped, key=repr):
        group = grouped[key]
        if len(group) != 2:
            continue
        averaged.append(swap_average_pair(group[0], group[1]))
    return tuple(averaged)


def dataclass_row(value: object) -> dict[str, object]:
    return asdict(value)  # type: ignore[arg-type]
