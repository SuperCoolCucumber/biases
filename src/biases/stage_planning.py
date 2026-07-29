from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from biases.pairing import (
    direction_relative_to_human,
    make_condition_group_id,
    make_ordering_twin_key,
    make_pair_identity_key,
    make_pair_key,
    normalize_ordering,
    normalize_verdict,
    opposite_binary_label,
    swap_display_label,
)
from biases.schemas import (
    BiasCondition,
    BiasType,
    CueCongruency,
    HumanCueDirection,
    PairOrdering,
    VerdictLabel,
)
from biases.social_cue_prompts import (
    AUTHORITY_DOSES,
    BANDWAGON_DOSES,
    build_social_cue,
    format_clean_variant_id,
    format_variant_id,
)


@dataclass(frozen=True)
class StageAPairInput:
    dataset_name: str
    input_file_hash: str
    source_row_index: int | str
    question_id: int | str
    model_name: str
    human_winner: VerdictLabel | str
    turn: int | str | None = None
    response_a_id: str | None = None
    response_b_id: str | None = None


@dataclass(frozen=True)
class CleanPairSummary:
    pair_identity_key: str
    pair_key: str
    ordering: PairOrdering | str
    ordering_twin_key: str
    model_name: str
    input_file_hash: str
    clean_record_id: str
    clean_verdict: VerdictLabel | str
    human_winner: VerdictLabel | str

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "CleanPairSummary":
        return cls(
            pair_identity_key=str(row["pair_identity_key"]),
            pair_key=str(row["pair_key"]),
            ordering=str(row["ordering"]),
            ordering_twin_key=str(row["ordering_twin_key"]),
            model_name=str(row["model_name"]),
            input_file_hash=str(row["input_file_hash"]),
            clean_record_id=str(row.get("clean_record_id") or row["record_id"]),
            clean_verdict=str(row.get("clean_verdict") or row["verdict"]),
            human_winner=str(row["human_winner"]),
        )


@dataclass(frozen=True)
class PlannedCondition:
    pair_identity_key: str
    pair_key: str
    condition_group_id: str
    ordering_twin_key: str
    model_name: str
    input_file_hash: str
    condition: BiasCondition


@dataclass(frozen=True)
class PlanningIssue:
    code: str
    message: str
    pair_identity_key: str
    model_name: str
    ordering: PairOrdering | None = None


@dataclass(frozen=True)
class StagePlan:
    conditions: tuple[PlannedCondition, ...]
    issues: tuple[PlanningIssue, ...] = ()

    def __iter__(self) -> Iterator[PlannedCondition]:
        return iter(self.conditions)


def clean_summaries_from_rows(rows: Iterable[Mapping[str, Any]]) -> tuple[CleanPairSummary, ...]:
    return tuple(CleanPairSummary.from_mapping(row) for row in rows)


def generate_stage_a_conditions(pair_inputs: Iterable[StageAPairInput]) -> StagePlan:
    planned: list[PlannedCondition] = []
    issues: list[PlanningIssue] = []
    seen_identities: set[tuple[str, str]] = set()

    for pair_input in pair_inputs:
        pair_identity_key = make_pair_identity_key(
            dataset_name=pair_input.dataset_name,
            input_file_hash=pair_input.input_file_hash,
            source_row_index=pair_input.source_row_index,
            question_id=pair_input.question_id,
            turn=pair_input.turn,
            response_a_id=pair_input.response_a_id,
            response_b_id=pair_input.response_b_id,
        )
        identity = (pair_identity_key, pair_input.model_name)
        if identity in seen_identities:
            issues.append(
                PlanningIssue(
                    code="duplicate_stage_a_input",
                    message="The same source pair and model appeared more than once.",
                    pair_identity_key=pair_identity_key,
                    model_name=pair_input.model_name,
                )
            )
            continue
        seen_identities.add(identity)

        human_ab = normalize_verdict(pair_input.human_winner)
        condition_group_id = make_condition_group_id(
            pair_identity_key=pair_identity_key,
            model_name=pair_input.model_name,
            family=BiasType.CLEAN,
        )
        for ordering in (PairOrdering.AB, PairOrdering.BA):
            human_winner = (
                human_ab if ordering == PairOrdering.AB else swap_display_label(human_ab)
            )
            pair_key = make_pair_key(
                pair_identity_key=pair_identity_key,
                model_name=pair_input.model_name,
                ordering=ordering,
            )
            ordering_twin_key = make_ordering_twin_key(
                pair_identity_key=pair_identity_key,
                model_name=pair_input.model_name,
                ordering=ordering,
            )
            condition = BiasCondition(
                bias_type=BiasType.CLEAN,
                variant_id=format_clean_variant_id(ordering),
                cue_congruency=CueCongruency.CONTROL,
                ordering=ordering,
                direction_relative_human=HumanCueDirection.NONE,
                metadata={
                    "pair_identity_key": pair_identity_key,
                    "source_row_index": pair_input.source_row_index,
                    "turn": pair_input.turn,
                    "human_winner": human_winner.value,
                },
            )
            planned.append(
                PlannedCondition(
                    pair_identity_key=pair_identity_key,
                    pair_key=pair_key,
                    condition_group_id=condition_group_id,
                    ordering_twin_key=ordering_twin_key,
                    model_name=pair_input.model_name,
                    input_file_hash=pair_input.input_file_hash,
                    condition=condition,
                )
            )

    return StagePlan(conditions=tuple(planned), issues=tuple(issues))


def _issue(
    *,
    code: str,
    message: str,
    summary: CleanPairSummary,
    ordering: PairOrdering | None = None,
) -> PlanningIssue:
    return PlanningIssue(
        code=code,
        message=message,
        pair_identity_key=summary.pair_identity_key,
        model_name=summary.model_name,
        ordering=ordering,
    )


def _validate_summary_group(
    summaries: Sequence[CleanPairSummary],
) -> tuple[dict[PairOrdering, CleanPairSummary], list[PlanningIssue]]:
    by_ordering: dict[PairOrdering, CleanPairSummary] = {}
    issues: list[PlanningIssue] = []
    first = summaries[0]

    for summary in summaries:
        ordering = normalize_ordering(summary.ordering)
        if ordering in by_ordering:
            issues.append(
                _issue(
                    code="duplicate_clean_ordering",
                    message=f"Multiple clean summaries were supplied for ordering {ordering.value}.",
                    summary=summary,
                    ordering=ordering,
                )
            )
            continue
        if summary.input_file_hash != first.input_file_hash:
            issues.append(
                _issue(
                    code="input_hash_mismatch",
                    message="AB and BA clean summaries have different input-file hashes.",
                    summary=summary,
                    ordering=ordering,
                )
            )
        expected_pair_key = make_pair_key(
            pair_identity_key=summary.pair_identity_key,
            model_name=summary.model_name,
            ordering=ordering,
        )
        expected_twin_key = make_ordering_twin_key(
            pair_identity_key=summary.pair_identity_key,
            model_name=summary.model_name,
            ordering=ordering,
        )
        if summary.pair_key != expected_pair_key or summary.ordering_twin_key != expected_twin_key:
            issues.append(
                _issue(
                    code="invalid_pair_linkage",
                    message="Clean summary pair/twin keys do not match the canonical key helpers.",
                    summary=summary,
                    ordering=ordering,
                )
            )
        if not summary.clean_record_id.strip():
            issues.append(
                _issue(
                    code="missing_clean_record_id",
                    message="A Stage B condition cannot link to an empty clean record ID.",
                    summary=summary,
                    ordering=ordering,
                )
            )
        by_ordering[ordering] = summary

    for ordering in (PairOrdering.AB, PairOrdering.BA):
        if ordering not in by_ordering:
            issues.append(
                _issue(
                    code="missing_clean_ordering",
                    message=f"Stage B requires a clean summary for ordering {ordering.value}.",
                    summary=first,
                    ordering=ordering,
                )
            )

    if len(by_ordering) == 2:
        if by_ordering[PairOrdering.AB].pair_key != by_ordering[PairOrdering.BA].ordering_twin_key:
            issues.append(
                _issue(
                    code="asymmetric_twin_linkage",
                    message="The BA clean summary does not point back to the AB pair key.",
                    summary=first,
                )
            )
        if by_ordering[PairOrdering.BA].pair_key != by_ordering[PairOrdering.AB].ordering_twin_key:
            issues.append(
                _issue(
                    code="asymmetric_twin_linkage",
                    message="The AB clean summary does not point to the BA pair key.",
                    summary=first,
                )
            )

    return by_ordering, issues


def _cue_reference(
    summary: CleanPairSummary,
) -> tuple[VerdictLabel | None, bool, str, PlanningIssue | None]:
    clean_verdict = normalize_verdict(summary.clean_verdict)
    human_winner = normalize_verdict(summary.human_winner)
    if clean_verdict in {VerdictLabel.A, VerdictLabel.B}:
        return clean_verdict, False, "clean_verdict", None
    if human_winner in {VerdictLabel.A, VerdictLabel.B}:
        return human_winner, True, "human_label_for_clean_tie", None
    ordering = normalize_ordering(summary.ordering)
    deterministic_reference = (
        VerdictLabel.A
        if int(summary.pair_identity_key[-1], 16) % 2 == 0
        else VerdictLabel.B
    )
    return (
        deterministic_reference,
        True,
        "deterministic_pair_hash_for_clean_and_human_tie",
        _issue(
            code="clean_and_human_tie",
            message=(
                "Both clean and human labels are ties; cue targets use a "
                "deterministic pair-hash reference and this stratum must be "
                "reported separately."
            ),
            summary=summary,
            ordering=ordering,
        ),
    )


def _conditions_for_summary(
    *,
    summary: CleanPairSummary,
    bandwagon_doses: Sequence[int],
    authority_doses: Sequence[int],
) -> tuple[list[PlannedCondition], PlanningIssue | None]:
    ordering = normalize_ordering(summary.ordering)
    reference, clean_tie, direction_reference, issue = _cue_reference(summary)
    if reference is None:
        return [], issue

    human_winner = normalize_verdict(summary.human_winner)
    clean_verdict = normalize_verdict(summary.clean_verdict)
    planned: list[PlannedCondition] = []
    family_doses = (
        (BiasType.BANDWAGON, bandwagon_doses),
        (BiasType.AUTHORITY, authority_doses),
    )
    for direction in (CueCongruency.CONGRUENT, CueCongruency.INCONGRUENT):
        cue_target = (
            reference
            if direction == CueCongruency.CONGRUENT
            else opposite_binary_label(reference)
        )
        human_direction = direction_relative_to_human(
            cue_target=cue_target,
            human_winner=human_winner,
        )
        for family, doses in family_doses:
            for dose in doses:
                cue_text = build_social_cue(
                    family=family,
                    target=cue_target,
                    dose=dose,
                )
                condition_group_id = make_condition_group_id(
                    pair_identity_key=summary.pair_identity_key,
                    model_name=summary.model_name,
                    family=family,
                    direction=direction,
                    dose=dose,
                )
                condition = BiasCondition(
                    bias_type=family,
                    variant_id=format_variant_id(
                        family=family,
                        direction=direction,
                        dose=dose,
                        ordering=ordering,
                    ),
                    cue_target=cue_target,
                    cue_congruency=direction,
                    cue_text=cue_text,
                    ordering=ordering,
                    dose=dose,
                    direction_relative_human=human_direction,
                    clean_tie=clean_tie,
                    clean_record_id=summary.clean_record_id,
                    metadata={
                        "pair_identity_key": summary.pair_identity_key,
                        "direction_reference": direction_reference,
                        "clean_verdict": clean_verdict.value,
                        "human_winner": human_winner.value,
                    },
                )
                planned.append(
                    PlannedCondition(
                        pair_identity_key=summary.pair_identity_key,
                        pair_key=summary.pair_key,
                        condition_group_id=condition_group_id,
                        ordering_twin_key=summary.ordering_twin_key,
                        model_name=summary.model_name,
                        input_file_hash=summary.input_file_hash,
                        condition=condition,
                    )
                )
    return planned, issue


def generate_stage_b_conditions(
    clean_summaries: Iterable[CleanPairSummary],
    *,
    bandwagon_doses: Sequence[int] = BANDWAGON_DOSES,
    authority_doses: Sequence[int] = AUTHORITY_DOSES,
) -> StagePlan:
    """Generate the full cued grid from Stage A clean summaries.

    Groups with missing/invalid ordering links are returned as explicit issues
    and are not partially scheduled. A clean tie falls back to the human label;
    a simultaneous clean and human tie uses a deterministic pair-hash reference
    and is retained with an explicit reporting issue.
    """

    grouped: dict[tuple[str, str], list[CleanPairSummary]] = {}
    for summary in clean_summaries:
        grouped.setdefault((summary.pair_identity_key, summary.model_name), []).append(summary)

    planned: list[PlannedCondition] = []
    issues: list[PlanningIssue] = []
    for summaries in grouped.values():
        by_ordering, group_issues = _validate_summary_group(summaries)
        issues.extend(group_issues)
        if group_issues:
            continue

        group_conditions: list[PlannedCondition] = []
        group_tie_issues: list[PlanningIssue] = []
        for ordering in (PairOrdering.AB, PairOrdering.BA):
            conditions, tie_issue = _conditions_for_summary(
                summary=by_ordering[ordering],
                bandwagon_doses=bandwagon_doses,
                authority_doses=authority_doses,
            )
            group_conditions.extend(conditions)
            if tie_issue is not None:
                group_tie_issues.append(tie_issue)
        if group_tie_issues:
            issues.append(group_tie_issues[0])
        planned.extend(group_conditions)

    return StagePlan(conditions=tuple(planned), issues=tuple(issues))


__all__ = [
    "CleanPairSummary",
    "PlannedCondition",
    "PlanningIssue",
    "StageAPairInput",
    "StagePlan",
    "clean_summaries_from_rows",
    "generate_stage_a_conditions",
    "generate_stage_b_conditions",
]
