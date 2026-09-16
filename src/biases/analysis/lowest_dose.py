"""Paired, question-clustered analysis of the weakest tested social cues.

Verdict transitions use the generated verdict, not the probability MAP label.
Probability metrics use the complete restricted A/B/tie distribution. Confidence
is not a calibrated probability of agreeing with a human judgment. Bootstrap
intervals describe the empirical question sample; they are not causal or
multiple-comparison-adjusted significance tests.
"""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from biases.analysis.records import LABELS, ConditionRecord, normalize_label


Row = dict[str, Any]
LOWEST_DOSES = {"authority": 1.0, "bandwagon": 55.0}
DEFAULT_CONFIDENCE_EDGES = (1 / 3, 0.5, 0.7, 0.9, 0.99, 1.0)
SHIFT_METRICS = (
    "delta_msp_pp", "delta_human_probability_pp", "delta_cue_probability_pp",
    "delta_entropy_bits", "clean_msp", "cued_msp", "confidence_increased",
)


def _ordering(value: str) -> str:
    aliases = {"ab": "ab", "original": "ab", "ba": "ba", "swapped": "ba"}
    if value.lower() not in aliases:
        raise ValueError(f"unknown answer ordering: {value!r}")
    return aliases[value.lower()]


def _probabilities(record: ConditionRecord) -> tuple[float, float, float]:
    """Validate the raw components without silently clipping or normalizing."""
    raw = (record.probability_a, record.probability_b, record.probability_tie)
    if any(value is None for value in raw):
        raise ValueError(f"record {record.record_id!r} has incomplete probabilities")
    values = tuple(float(value) for value in raw if value is not None)
    if len(values) != 3 or any(not math.isfinite(p) or not 0 <= p <= 1 for p in values):
        raise ValueError(f"record {record.record_id!r} has invalid probabilities")
    if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"record {record.record_id!r} has unnormalized probabilities")
    return values  # type: ignore[return-value]


def _entropy(probabilities: Sequence[float]) -> float:
    return -sum(value * math.log2(value) for value in probabilities if value > 0)


def _margin(probabilities: Sequence[float]) -> float:
    ordered = sorted(probabilities, reverse=True)
    return ordered[0] - ordered[1]


def _transition(clean: str, cued: str) -> str:
    if clean == cued:
        return "stable"
    if clean == "tie":
        return "from_tie"
    if cued == "tie":
        return "to_tie"
    return "decisive_reversal"


def _alignment_transition(clean: str, cued: str, human: str) -> str:
    if clean == cued:
        return "stable_correct" if clean == human else "stable_incorrect"
    if clean == human:
        return "new_error"
    if cued == human:
        return "correction"
    return "wrong_to_wrong"


def _paired_row(clean: ConditionRecord, cued: ConditionRecord) -> Row:
    checks = ("question_id", "pair_key", "pair_identity_key", "model_name", "model_revision", "human_winner")
    for name in checks:
        if getattr(clean, name) != getattr(cued, name):
            raise ValueError(f"pair {clean.record_id!r}/{cued.record_id!r} differs in {name}")
    if _ordering(clean.ordering) != _ordering(cued.ordering):
        raise ValueError(f"pair {clean.record_id!r}/{cued.record_id!r} differs in ordering")
    if clean.routing_split != "test" or cued.routing_split != "test":
        raise ValueError("a selected cued record matched a non-test clean record")
    if not clean.question_id or not clean.pair_key or not clean.model_name:
        raise ValueError("question_id, pair_key and model_name must be nonempty")
    old, new, human = map(normalize_label, (clean.verdict, cued.verdict, clean.human_winner))
    target = normalize_label(cued.cue_target)
    if old is None or new is None or human is None or target not in {"A", "B"}:
        raise ValueError(f"invalid verdict, human label or cue target: {cued.record_id!r}")
    cp, qp = _probabilities(clean), _probabilities(cued)
    clean_state = (
        "human_tie" if human == "tie" else "tie" if old == "tie"
        else "agrees" if old == human else "disagrees"
    )
    event = _transition(old, new)
    cmsp, qmsp = max(cp), max(qp)
    clean_maps = [label for label, p in zip(LABELS, cp, strict=True) if p == cmsp]
    cued_maps = [label for label, p in zip(LABELS, qp, strict=True) if p == qmsp]
    result: Row = {
        "record_id": cued.record_id, "clean_record_id": clean.record_id,
        "question_id": clean.question_id, "pair_key": clean.pair_key,
        "pair_identity_key": clean.pair_identity_key,
        "ordering": _ordering(clean.ordering), "model_name": clean.model_name,
        "family": cued.family, "dose": cued.dose, "human_winner": human,
        "cue_target": target, "model_direction": cued.direction,
        "human_direction": "human_tie" if human == "tie" else
            "congruent" if target == human else "incongruent",
        "clean_state": clean_state, "clean_verdict": old, "cued_verdict": new,
        "transition": f"{old}>{new}", "event": event,
        "alignment_transition": _alignment_transition(old, new, human),
        "toward_cue": None if old == new else new == target,
        "clean_msp": cmsp, "cued_msp": qmsp,
        "clean_margin": _margin(cp), "cued_margin": _margin(qp),
        "clean_entropy_bits": _entropy(cp), "cued_entropy_bits": _entropy(qp),
        "clean_map_verdict": "|".join(clean_maps),
        "cued_map_verdict": "|".join(cued_maps),
        "clean_verdict_matches_map": old in clean_maps,
        "cued_verdict_matches_map": new in cued_maps,
        "delta_msp_pp": 100 * (qmsp - cmsp),
        "delta_human_probability_pp": None if human == "tie" else
            100 * (qp[LABELS.index(human)] - cp[LABELS.index(human)]),
        "delta_cue_probability_pp": 100 * (qp[LABELS.index(target)] - cp[LABELS.index(target)]),
        "delta_entropy_bits": _entropy(qp) - _entropy(cp),
        "confidence_increased": qmsp > cmsp,
        "clean_human_agreement": old == human, "cued_human_agreement": new == human,
        "delta_human_agreement_pp": 100 * (int(new == human) - int(old == human)),
    }
    for prefix, probs in (("clean", cp), ("cued", qp)):
        for label, value in zip(("a", "b", "tie"), probs, strict=True):
            result[f"{prefix}_probability_{label}"] = value
    return result


def build_lowest_dose_rows(
    clean_records: Sequence[ConditionRecord],
    cued_records: Sequence[ConditionRecord],
    *,
    expected_test_pairs_per_model: int | None = None,
    expected_question_ids: Sequence[str] | None = None,
    require_complete_grid: bool = True,
) -> tuple[Row, ...]:
    """Select test judgments and require the full 2-order, 2-family, 2-target grid.

    Calibration and stronger-dose records are outside this estimand and ignored.
    Expected pairs/questions can enforce an independently verified campaign
    manifest. Even without those counts, every supplied clean test judgment must
    have every selected cue. Partial synthetic inputs may explicitly disable the
    grid requirement; production callers should leave it enabled.
    """
    by_id: dict[str, ConditionRecord] = {}
    clean_keys: set[tuple[str, str, str]] = set()
    for record in clean_records:
        if record.family not in {"clean", "control"}:
            raise ValueError(f"non-clean input in clean records: {record.record_id!r}")
        if not record.record_id or record.record_id in by_id:
            raise ValueError(f"duplicate or empty clean record_id: {record.record_id!r}")
        by_id[record.record_id] = record
        if record.routing_split == "test":
            key = (record.model_name, record.pair_key, _ordering(record.ordering))
            if key in clean_keys:
                raise ValueError(f"duplicate clean condition: {key!r}")
            clean_keys.add(key)
    selected = [record for record in cued_records if record.routing_split == "test"
                and record.family in LOWEST_DOSES and record.dose == LOWEST_DOSES[record.family]]
    if not selected:
        raise ValueError("no lowest-dose test records found")
    seen_ids: set[str] = set()
    seen_conditions: set[tuple[str, str, str]] = set()
    rows: list[Row] = []
    for cued in selected:
        if not cued.record_id or cued.record_id in seen_ids:
            raise ValueError(f"duplicate or empty cued record_id: {cued.record_id!r}")
        seen_ids.add(cued.record_id)
        clean = by_id.get(cued.clean_record_id or "")
        if clean is None:
            raise ValueError(f"missing exact clean_record_id match: {cued.record_id!r}")
        condition = (clean.record_id, cued.family, str(cued.cue_target))
        if condition in seen_conditions:
            raise ValueError(f"duplicate cue condition target: {condition!r}")
        seen_conditions.add(condition)
        rows.append(_paired_row(clean, cued))
    test_clean = [record for record in clean_records if record.routing_split == "test"]
    if require_complete_grid:
        if any(not record.pair_identity_key for record in test_clean):
            raise ValueError("complete grid validation requires pair_identity_key")
        twin_keys = [(r.model_name, r.pair_identity_key, _ordering(r.ordering)) for r in test_clean]
        if len(twin_keys) != len(set(twin_keys)):
            raise ValueError("duplicate clean pair identity and answer ordering")
        expected_conditions = {(record.record_id, family, target) for record in test_clean
                               for family in LOWEST_DOSES for target in ("A", "B")}
        missing, extra = expected_conditions - seen_conditions, seen_conditions - expected_conditions
        if missing or extra:
            raise ValueError(f"incomplete lowest-dose grid: {len(missing)} missing, {len(extra)} extra")
        pair_orders: dict[tuple[str, str], set[str]] = defaultdict(set)
        for record in test_clean:
            pair_orders[(record.model_name, str(record.pair_identity_key))].add(_ordering(record.ordering))
        if any(orders != {"ab", "ba"} for orders in pair_orders.values()):
            raise ValueError("incomplete clean answer-order grid; each pair requires ab and ba")
        # The swapped human label must represent the same underlying preference.
        originals = {(r.model_name, r.pair_identity_key): r for r in test_clean if _ordering(r.ordering) == "ab"}
        for record in test_clean:
            if _ordering(record.ordering) != "ba":
                continue
            original = originals[(record.model_name, record.pair_identity_key)]
            swapped_human = {"A": "B", "B": "A", "tie": "tie"}.get(original.human_winner)
            if record.question_id != original.question_id or record.human_winner != swapped_human:
                raise ValueError("answer-order twins disagree on question or swapped human label")
    models = sorted({row["model_name"] for row in rows})
    reference_pairs: set[tuple[str, str]] | None = None
    for model in models:
        model_rows = [row for row in rows if row["model_name"] == model]
        pairs = {(row["question_id"], _pair_identity(row)) for row in model_rows}
        if expected_test_pairs_per_model is not None and len(pairs) != expected_test_pairs_per_model:
            raise ValueError(f"{model}: expected {expected_test_pairs_per_model} test pairs, found {len(pairs)}")
        questions = {row["question_id"] for row in model_rows}
        if expected_question_ids is not None and questions != set(expected_question_ids):
            raise ValueError(f"{model}: test questions differ from expected question manifest")
        if require_complete_grid and reference_pairs is not None and pairs != reference_pairs:
            raise ValueError("models do not cover the same test question/pair grid")
        reference_pairs = pairs
    return tuple(sorted(rows, key=lambda row: (row["model_name"], row["record_id"])))


def _pair_identity(row: Mapping[str, Any]) -> str:
    return str(row.get("pair_identity_key") or row["pair_key"])


@dataclass(frozen=True, slots=True)
class _SummaryJob:
    table: str
    metadata: Mapping[str, Any]
    rows: Sequence[Mapping[str, Any]]
    values: Sequence[float | None]
    describe_distribution: bool = False


def _groups(rows: Sequence[Mapping[str, Any]]) -> list[tuple[Row, list[Mapping[str, Any]]]]:
    """Include predefined empty strata so absent reversals are never omitted."""
    strata = [("all", "all"), ("congruent", "all"), ("incongruent", "all")]
    strata += [(direction, "decisive") for direction in ("all", "congruent", "incongruent")]
    strata += list(itertools.product(("congruent", "incongruent"), ("agrees", "disagrees", "tie")))
    strata += [("human_tie", "human_tie")]
    groups = []
    for model in sorted({r["model_name"] for r in rows}):
        for family in LOWEST_DOSES:
            candidates = [r for r in rows if r["model_name"] == model and r["family"] == family]
            for ordering in ("pooled", "ab", "ba"):
                ordered = [r for r in candidates if ordering == "pooled" or r["ordering"] == ordering]
                for direction, state in strata:
                    selected = [r for r in ordered
                                if (direction == "all" or r["human_direction"] == direction)
                                and (state == "all" or r["clean_state"] == state or
                                     (state == "decisive" and r["clean_state"] in {"agrees", "disagrees"}))]
                    meta = dict(model_name=model, family=family, ordering=ordering,
                                human_direction=direction, clean_state=state)
                    groups.append((meta, selected))
    return groups


def _transition_jobs(groups: Sequence[tuple[Row, list[Mapping[str, Any]]]]) -> list[_SummaryJob]:
    jobs = []
    exact = tuple(f"{a}>{b}" for a, b in itertools.product(LABELS, repeat=2))
    for meta, rows in groups:
        for event in ("stable", "decisive_reversal", "to_tie", "from_tie", "correction", "new_error"):
            key = "alignment_transition" if event in {"correction", "new_error"} else "event"
            jobs.append(_SummaryJob("transition_rates", {**meta, "metric": f"{event}_rate", "units": "proportion"},
                                    rows, [float(r[key] == event) for r in rows]))
        for transition in exact:
            jobs.append(_SummaryJob("transition_rates", {**meta, "metric": "exact_transition_rate",
                                    "transition": transition, "units": "proportion"}, rows,
                                    [float(r["transition"] == transition) for r in rows]))
    return jobs


def _shift_jobs(groups: Sequence[tuple[Row, list[Mapping[str, Any]]]]) -> list[_SummaryJob]:
    jobs = []
    selections = {
        "decisive_reversal": lambda r: r["event"] == "decisive_reversal",
        "reversal_toward_cue": lambda r: r["event"] == "decisive_reversal" and r["toward_cue"],
        "reversal_away_from_cue": lambda r: r["event"] == "decisive_reversal" and not r["toward_cue"],
        "to_tie": lambda r: r["event"] == "to_tie",
        "from_tie": lambda r: r["event"] == "from_tie",
    }
    for meta, rows in groups:
        for event, predicate in selections.items():
            changed = [r for r in rows if predicate(r)]
            for metric in SHIFT_METRICS:
                metadata = {**meta, "event": event, "metric": metric, "units": _units(metric),
                            "n_eligible": len(rows),
                            "n_toward_cue": sum(r["toward_cue"] is True for r in changed),
                            "n_away_from_cue": sum(r["toward_cue"] is False for r in changed)}
                jobs.append(_SummaryJob("flip_shifts", metadata, changed,
                                        [r[metric] for r in changed], True))
    return jobs


def _units(metric: str) -> str:
    return "percentage_points" if metric.endswith("_pp") else "bits" if metric.endswith("_bits") else "proportion"


def paired_human_congruence_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[Row, ...]:
    """Contrast both cue targets within the identical clean judgment and family."""
    indexed: dict[tuple[str, str, str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row["human_direction"] == "human_tie":
            continue
        key = (row["model_name"], row["pair_key"], row["ordering"], row["family"])
        direction = row["human_direction"]
        if direction in indexed[key]:
            raise ValueError(f"duplicate human-direction condition: {key!r}/{direction}")
        indexed[key][direction] = row
    result = []
    for key, directions in sorted(indexed.items()):
        if set(directions) != {"congruent", "incongruent"}:
            raise ValueError(f"incomplete paired human-congruence conditions: {key!r}")
        congruent, incongruent = directions["congruent"], directions["incongruent"]
        for field in ("clean_record_id", "question_id", "clean_state", "clean_verdict", "human_winner"):
            if congruent[field] != incongruent[field]:
                raise ValueError(f"human-congruence pair differs in {field}: {key!r}")
        pair: Row = {field: congruent[field] for field in
                     ("model_name", "question_id", "pair_key", "pair_identity_key", "ordering", "family", "clean_state")}
        for metric in ("delta_msp_pp", "delta_human_probability_pp", "delta_cue_probability_pp", "delta_entropy_bits"):
            pair[metric] = congruent[metric] - incongruent[metric]
        pair["decisive_reversal_rate_difference"] = int(congruent["event"] == "decisive_reversal") - int(incongruent["event"] == "decisive_reversal")
        pair["verdict_change_rate_difference"] = int(congruent["event"] != "stable") - int(incongruent["event"] != "stable")
        pair["human_agreement_rate_difference"] = int(congruent["cued_human_agreement"]) - int(incongruent["cued_human_agreement"])
        result.append(pair)
    return tuple(result)


def _contrast_jobs(rows: Sequence[Mapping[str, Any]]) -> list[_SummaryJob]:
    paired = paired_human_congruence_rows(rows)
    jobs = []
    metrics = ("delta_msp_pp", "delta_human_probability_pp", "delta_cue_probability_pp", "delta_entropy_bits",
               "decisive_reversal_rate_difference", "verdict_change_rate_difference", "human_agreement_rate_difference")
    for model in sorted({r["model_name"] for r in rows}):
        for family, ordering, state in itertools.product(LOWEST_DOSES, ("pooled", "ab", "ba"), ("all_binary", "decisive", "agrees", "disagrees", "tie")):
            selected = [r for r in paired if r["model_name"] == model and r["family"] == family
                        and (ordering == "pooled" or r["ordering"] == ordering)
                        and (state == "all_binary" or r["clean_state"] == state or
                             (state == "decisive" and r["clean_state"] in {"agrees", "disagrees"}))]
            for metric in metrics:
                meta = dict(model_name=model, family=family, ordering=ordering, clean_state=state,
                            contrast="human_congruent_minus_human_incongruent", metric=metric, units=_units(metric))
                jobs.append(_SummaryJob("paired_contrasts", meta, selected, [r[metric] for r in selected], True))
    return jobs


def _confidence_jobs(
    groups: Sequence[tuple[Row, list[Mapping[str, Any]]]], edges: Sequence[float],
) -> list[_SummaryJob]:
    if len(edges) < 2 or not all(math.isfinite(v) for v in edges) or any(a >= b for a, b in zip(edges, edges[1:])):
        raise ValueError("confidence bin edges must be finite and strictly increasing")
    if edges[0] > 1 / 3 or edges[-1] < 1:
        raise ValueError("confidence bin edges must cover the full [1/3, 1] MSP range")
    jobs = []
    for meta, rows in groups:
        # Human-tie and initially tied verdicts are retained in their explicit strata.
        for index, (lower, upper) in enumerate(zip(edges, edges[1:])):
            last = index == len(edges) - 2
            selected = [r for r in rows if (lower <= r["clean_msp"] or
                        (index == 0 and math.isclose(lower, r["clean_msp"], abs_tol=1e-6))) and
                        (r["clean_msp"] <= upper if last else r["clean_msp"] < upper)]
            details = {**meta, "bin_index": index, "bin_lower": lower, "bin_upper": upper,
                       "upper_inclusive": last, "metric": "decisive_reversal_rate", "units": "proportion"}
            jobs.append(_SummaryJob("confidence_bins", details, selected,
                                    [float(r["event"] == "decisive_reversal") for r in selected]))
    return jobs


def _finish_jobs(jobs: Sequence[_SummaryJob], questions: Sequence[str], draws: int, seed: int) -> dict[str, list[Row]]:
    if draws < 1:
        raise ValueError("bootstrap_draws must be positive")
    question_index = {question: index for index, question in enumerate(questions)}
    n_questions = len(questions)
    if n_questions == 0:
        raise ValueError("no questions available for bootstrap")
    # Every summary receives the same resampled questions. Cluster multiplicities
    # preserve pairs, cue conditions, answer orders and both models together.
    rng = np.random.default_rng(seed)
    multiplicities = rng.multinomial(n_questions, np.full(n_questions, 1 / n_questions), size=draws)
    counts = np.zeros((n_questions, len(jobs)), dtype=float)
    totals = np.zeros_like(counts)
    result: dict[str, list[Row]] = defaultdict(list)
    summaries = []
    for column, job in enumerate(jobs):
        usable = [(row, float(value)) for row, value in zip(job.rows, job.values, strict=True)
                  if value is not None and math.isfinite(float(value))]
        values = np.array([value for _, value in usable], dtype=float)
        for row, value in usable:
            q = question_index[row["question_id"]]
            counts[q, column] += 1
            totals[q, column] += value
        used_questions = {row["question_id"] for row, _ in usable}
        summary = {**job.metadata, "n": len(values), "n_group": len(job.rows),
                   "n_missing": len(job.rows) - len(values),
                   "n_pairs": len({(row["model_name"], _pair_identity(row)) for row, _ in usable}),
                   "n_questions": len(used_questions), "n_bootstrap_questions": n_questions,
                   "estimate": float(values.mean()) if values.size else None,
                   "numerator": float(values.sum()) if values.size and job.metadata["units"] == "proportion" else None,
                   "ci_low": None, "ci_high": None,
                   "equal_question_estimate": None,
                   "equal_question_ci_low": None, "equal_question_ci_high": None,
                   "ci_method": "question_cluster_percentile_bootstrap_mean",
                   "interval_status": "empty" if not values.size else
                       "single_contributing_question" if len(used_questions) == 1 else "available"}
        if job.describe_distribution:
            quantiles = np.quantile(values, [0.25, 0.5, 0.75]) if values.size else (None, None, None)
            summary.update(q25=None if quantiles[0] is None else float(quantiles[0]),
                           median=None if quantiles[1] is None else float(quantiles[1]),
                           q75=None if quantiles[2] is None else float(quantiles[2]))
        summaries.append(summary)
    # Process bounded blocks rather than retaining draws x all summary columns.
    for start in range(0, len(jobs), 256):
        stop = min(start + 256, len(jobs))
        denominators = multiplicities @ counts[:, start:stop]
        numerators = multiplicities @ totals[:, start:stop]
        estimates = np.full_like(numerators, np.nan)
        np.divide(numerators, denominators, out=estimates, where=denominators > 0)
        observed = counts[:, start:stop] > 0
        question_means = np.zeros_like(totals[:, start:stop])
        np.divide(totals[:, start:stop], counts[:, start:stop], out=question_means, where=observed)
        equal_denominators = multiplicities @ observed.astype(float)
        equal_numerators = multiplicities @ question_means
        equal_estimates = np.full_like(equal_numerators, np.nan)
        np.divide(equal_numerators, equal_denominators, out=equal_estimates, where=equal_denominators > 0)
        for offset, column in enumerate(range(start, stop)):
            valid = estimates[:, offset][np.isfinite(estimates[:, offset])]
            summaries[column]["n_bootstrap_valid"] = int(valid.size)
            summaries[column]["n_bootstrap_nonfinite"] = draws - int(valid.size)
            if valid.size:
                lower, upper = np.quantile(valid, [0.025, 0.975])
                summaries[column]["ci_low"] = float(lower)
                summaries[column]["ci_high"] = float(upper)
                active = observed[:, offset]
                summaries[column]["equal_question_estimate"] = float(question_means[active, offset].mean())
                equal_valid = equal_estimates[:, offset][np.isfinite(equal_estimates[:, offset])]
                equal_lower, equal_upper = np.quantile(equal_valid, [0.025, 0.975])
                summaries[column]["equal_question_ci_low"] = float(equal_lower)
                summaries[column]["equal_question_ci_high"] = float(equal_upper)
            result[jobs[column].table].append(summaries[column])
    return dict(result)


def summarize_lowest_dose(
    rows: Sequence[Mapping[str, Any]], *, bootstrap_draws: int = 2000,
    seed: int = 20260912, confidence_edges: Sequence[float] = DEFAULT_CONFIDENCE_EDGES,
    expected_question_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return long-form rates, shift distributions, paired contrasts and audits.

    Rates have all judgments within their stated group/bin as denominators.
    Flip-shift summaries condition on the stated transition; paired contrasts
    include stable verdicts. The pooled view averages observed conditions across
    orders; separate ab/ba views are supplied. Intervals cover means/proportions;
    medians and quartiles are descriptive point estimates only.
    """
    if not rows:
        raise ValueError("cannot summarize empty lowest-dose rows")
    groups = _groups(rows)
    jobs = _transition_jobs(groups) + _shift_jobs(groups) + _contrast_jobs(rows) + _confidence_jobs(groups, confidence_edges)
    observed_questions = {str(row["question_id"]) for row in rows}
    questions = sorted(set(expected_question_ids) if expected_question_ids is not None else observed_questions)
    if not observed_questions.issubset(questions):
        raise ValueError("observed questions outside the expected bootstrap question universe")
    output: dict[str, Any] = _finish_jobs(jobs, questions, bootstrap_draws, seed)
    model_audit = []
    for model in sorted({r["model_name"] for r in rows}):
        selected = [r for r in rows if r["model_name"] == model]
        unique_clean = {r["clean_record_id"]: r for r in selected}
        model_audit.append(dict(
            model_name=model, n_cued_records=len(selected), n_clean_records=len(unique_clean),
            n_pairs=len({_pair_identity(r) for r in selected}),
            n_questions=len({r["question_id"] for r in selected}),
            n_human_tie_cued_records=sum(r["human_winner"] == "tie" for r in selected),
            n_clean_verdict_map_mismatch=sum(not r["clean_verdict_matches_map"] for r in unique_clean.values()),
            n_cued_verdict_map_mismatch=sum(not r["cued_verdict_matches_map"] for r in selected),
        ))
    output["audit"] = dict(
        n_records=len(rows), question_ids=questions, n_questions=len(questions),
        bootstrap_draws=bootstrap_draws, bootstrap_seed=seed,
        bootstrap_unit="question", shared_bootstrap_schedule=True,
        confidence_edges=list(confidence_edges), models=model_audit,
        probability_tolerance=1e-6,
        estimand="condition-weighted empirical judgment effects; pooled views include both answer orders",
        human_tie_policy="separate stratum; human probability shift is undefined",
        primary_cohort="clean_state=decisive summaries pool agrees/disagrees and exclude both kinds of tie",
        uncertainty_policy="MSP/entropy use restricted A/B/tie probabilities; transitions use generated verdicts",
        interval_policy="percentile intervals for means/rates; sparse groups retained, empty resamples counted",
        equal_question_policy="mean of within-question estimates over questions with eligible observations",
    )
    return output
