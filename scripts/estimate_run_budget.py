from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Literal, Sequence


ConsistencySchedule = Literal["all", "extremes"]

ORDERINGS_PER_EXAMPLE = 2
CLEAN_CONDITIONS_PER_EXAMPLE_MODEL = 2
BIAS_FAMILIES = 2
CUE_DIRECTIONS = 2
DOSES_PER_FAMILY = 4
EXTREME_DOSES_PER_FAMILY = 2
CUED_CONDITIONS_PER_EXAMPLE_MODEL = (
    BIAS_FAMILIES * CUE_DIRECTIONS * DOSES_PER_FAMILY * ORDERINGS_PER_EXAMPLE
)
TOTAL_CONDITIONS_PER_EXAMPLE_MODEL = (
    CLEAN_CONDITIONS_PER_EXAMPLE_MODEL + CUED_CONDITIONS_PER_EXAMPLE_MODEL
)
EXTREME_CUED_CONDITIONS_PER_EXAMPLE_MODEL = (
    BIAS_FAMILIES
    * CUE_DIRECTIONS
    * EXTREME_DOSES_PER_FAMILY
    * ORDERINGS_PER_EXAMPLE
)
EXTREME_CONSISTENCY_CONDITIONS_PER_EXAMPLE_MODEL = (
    CLEAN_CONDITIONS_PER_EXAMPLE_MODEL
    + EXTREME_CUED_CONDITIONS_PER_EXAMPLE_MODEL
)


@dataclass(frozen=True)
class StageBudget:
    conditions: int
    logit_generations: int
    consistency_generations: int
    verbalized_generations: int

    @property
    def total_generations(self) -> int:
        return (
            self.logit_generations
            + self.consistency_generations
            + self.verbalized_generations
        )

    def to_dict(self) -> dict[str, int]:
        return {
            **asdict(self),
            "total_generations": self.total_generations,
        }


@dataclass(frozen=True)
class RunBudget:
    examples: int
    models: int
    consistency_k: int
    consistency_schedule: ConsistencySchedule
    include_verbalized: bool
    consistency_conditions_per_example_model: int
    stage_a: StageBudget
    stage_b: StageBudget

    @property
    def conditions(self) -> int:
        return self.stage_a.conditions + self.stage_b.conditions

    @property
    def logit_generations(self) -> int:
        return self.stage_a.logit_generations + self.stage_b.logit_generations

    @property
    def consistency_generations(self) -> int:
        return (
            self.stage_a.consistency_generations
            + self.stage_b.consistency_generations
        )

    @property
    def verbalized_generations(self) -> int:
        return (
            self.stage_a.verbalized_generations
            + self.stage_b.verbalized_generations
        )

    @property
    def total_generations(self) -> int:
        return self.stage_a.total_generations + self.stage_b.total_generations

    def to_dict(self) -> dict[str, object]:
        return {
            "examples": self.examples,
            "models": self.models,
            "consistency_k": self.consistency_k,
            "consistency_schedule": self.consistency_schedule,
            "include_verbalized": self.include_verbalized,
            "orderings_per_example": ORDERINGS_PER_EXAMPLE,
            "clean_conditions_per_example_model": CLEAN_CONDITIONS_PER_EXAMPLE_MODEL,
            "cued_conditions_per_example_model": CUED_CONDITIONS_PER_EXAMPLE_MODEL,
            "conditions_per_example_model": TOTAL_CONDITIONS_PER_EXAMPLE_MODEL,
            "consistency_conditions_per_example_model": (
                self.consistency_conditions_per_example_model
            ),
            "conditions": self.conditions,
            "logit_generations": self.logit_generations,
            "consistency_generations": self.consistency_generations,
            "verbalized_generations": self.verbalized_generations,
            "total_generations": self.total_generations,
            "stage_a": self.stage_a.to_dict(),
            "stage_b": self.stage_b.to_dict(),
        }


def _validate_positive(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be at least 1")


def _validate_non_negative(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def estimate_run_budget(
    *,
    examples: int,
    models: int,
    consistency_k: int,
    consistency_schedule: ConsistencySchedule,
    include_verbalized: bool,
) -> RunBudget:
    """Return exact condition and generation counts for the two-stage grid."""

    _validate_positive("examples", examples)
    _validate_positive("models", models)
    _validate_non_negative("consistency_k", consistency_k)
    if consistency_schedule not in {"all", "extremes"}:
        raise ValueError("consistency_schedule must be 'all' or 'extremes'")

    scale = examples * models
    stage_a_conditions = CLEAN_CONDITIONS_PER_EXAMPLE_MODEL * scale
    stage_b_conditions = CUED_CONDITIONS_PER_EXAMPLE_MODEL * scale

    if consistency_schedule == "all":
        stage_b_consistency_conditions = CUED_CONDITIONS_PER_EXAMPLE_MODEL
        consistency_conditions = TOTAL_CONDITIONS_PER_EXAMPLE_MODEL
    else:
        stage_b_consistency_conditions = (
            EXTREME_CUED_CONDITIONS_PER_EXAMPLE_MODEL
        )
        consistency_conditions = (
            EXTREME_CONSISTENCY_CONDITIONS_PER_EXAMPLE_MODEL
        )

    stage_a = StageBudget(
        conditions=stage_a_conditions,
        logit_generations=stage_a_conditions,
        consistency_generations=stage_a_conditions * consistency_k,
        verbalized_generations=stage_a_conditions if include_verbalized else 0,
    )
    stage_b = StageBudget(
        conditions=stage_b_conditions,
        logit_generations=stage_b_conditions,
        consistency_generations=(
            stage_b_consistency_conditions * scale * consistency_k
        ),
        verbalized_generations=stage_b_conditions if include_verbalized else 0,
    )
    return RunBudget(
        examples=examples,
        models=models,
        consistency_k=consistency_k,
        consistency_schedule=consistency_schedule,
        include_verbalized=include_verbalized,
        consistency_conditions_per_example_model=consistency_conditions,
        stage_a=stage_a,
        stage_b=stage_b,
    )


def render_text(budget: RunBudget) -> str:
    payload = budget.to_dict()
    stage_a = budget.stage_a
    stage_b = budget.stage_b
    lines = [
        "Silent Bias run budget",
        f"Examples: {budget.examples:,}",
        f"Models: {budget.models:,}",
        f"Orderings per example: {ORDERINGS_PER_EXAMPLE}",
        (
            "Conditions per example/model: "
            f"{TOTAL_CONDITIONS_PER_EXAMPLE_MODEL} "
            f"({CLEAN_CONDITIONS_PER_EXAMPLE_MODEL} clean + "
            f"{CUED_CONDITIONS_PER_EXAMPLE_MODEL} cued)"
        ),
        f"Consistency samples per selected condition: {budget.consistency_k}",
        f"Consistency schedule: {budget.consistency_schedule}",
        (
            "Consistency-enabled conditions per example/model: "
            f"{budget.consistency_conditions_per_example_model}"
        ),
        f"Verbalized pass: {'enabled' if budget.include_verbalized else 'disabled'}",
        "",
        (
            "Stage A (clean): "
            f"{stage_a.conditions:,} conditions; "
            f"{stage_a.total_generations:,} generations"
        ),
        (
            "Stage B (cued): "
            f"{stage_b.conditions:,} conditions; "
            f"{stage_b.total_generations:,} generations"
        ),
        "",
        f"Logit generations: {int(payload['logit_generations']):,}",
        f"Consistency generations: {int(payload['consistency_generations']):,}",
        f"Verbalized generations: {int(payload['verbalized_generations']):,}",
        f"Total prompt conditions: {int(payload['conditions']):,}",
        f"Total generations: {int(payload['total_generations']):,}",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Print exact prompt-condition and generation counts for the "
            "two-stage Silent Bias experiment grid."
        ),
    )
    parser.add_argument("--examples", type=int, required=True)
    parser.add_argument("--models", type=int, default=1)
    parser.add_argument("--consistency-k", type=int, default=8)
    parser.add_argument(
        "--consistency-schedule",
        choices=("all", "extremes"),
        default="all",
        help=(
            "'all' samples every condition; 'extremes' samples clean plus the "
            "lowest and highest dose for each family/direction."
        ),
    )
    verbalized = parser.add_mutually_exclusive_group()
    verbalized.add_argument(
        "--verbalized",
        dest="include_verbalized",
        action="store_true",
        help="Include one verbalized-confidence generation per condition.",
    )
    verbalized.add_argument(
        "--no-verbalized",
        dest="include_verbalized",
        action="store_false",
        help="Do not count the verbalized-confidence pass.",
    )
    parser.set_defaults(include_verbalized=True)
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        budget = estimate_run_budget(
            examples=args.examples,
            models=args.models,
            consistency_k=args.consistency_k,
            consistency_schedule=args.consistency_schedule,
            include_verbalized=args.include_verbalized,
        )
    except ValueError as error:
        parser.error(str(error))

    if args.format == "json":
        print(json.dumps(budget.to_dict(), indent=2, sort_keys=True))
    else:
        print(render_text(budget))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
