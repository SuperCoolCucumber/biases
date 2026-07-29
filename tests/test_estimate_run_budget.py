from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "estimate_run_budget.py"
)


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("estimate_run_budget", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


budget_module = _load_module()


def test_full_schedule_matches_worked_example() -> None:
    budget = budget_module.estimate_run_budget(
        examples=1,
        models=1,
        consistency_k=8,
        consistency_schedule="all",
        include_verbalized=True,
    )

    assert budget_module.TOTAL_CONDITIONS_PER_EXAMPLE_MODEL == 34
    assert budget.stage_a.conditions == 2
    assert budget.stage_b.conditions == 32
    assert budget.logit_generations == 34
    assert budget.consistency_generations == 34 * 8
    assert budget.verbalized_generations == 34
    assert budget.total_generations == 340


def test_extremes_schedule_samples_clean_and_boundary_doses() -> None:
    budget = budget_module.estimate_run_budget(
        examples=1,
        models=1,
        consistency_k=8,
        consistency_schedule="extremes",
        include_verbalized=True,
    )

    assert budget.consistency_conditions_per_example_model == 18
    assert budget.stage_a.consistency_generations == 2 * 8
    assert budget.stage_b.consistency_generations == 16 * 8
    assert budget.total_generations == 212


def test_counts_scale_by_examples_and_models() -> None:
    budget = budget_module.estimate_run_budget(
        examples=198,
        models=3,
        consistency_k=8,
        consistency_schedule="all",
        include_verbalized=True,
    )

    assert budget.conditions == 198 * 3 * 34
    assert budget.total_generations == 198 * 3 * 340


def test_verbalized_pass_can_be_disabled() -> None:
    budget = budget_module.estimate_run_budget(
        examples=2,
        models=3,
        consistency_k=4,
        consistency_schedule="all",
        include_verbalized=False,
    )

    assert budget.verbalized_generations == 0
    assert budget.total_generations == 2 * 3 * 34 * 5


@pytest.mark.parametrize(
    ("argument", "value"),
    (("examples", 0), ("models", 0), ("consistency_k", -1)),
)
def test_invalid_counts_are_rejected(argument: str, value: int) -> None:
    kwargs = {
        "examples": 1,
        "models": 1,
        "consistency_k": 8,
        "consistency_schedule": "all",
        "include_verbalized": True,
    }
    kwargs[argument] = value

    with pytest.raises(ValueError):
        budget_module.estimate_run_budget(**kwargs)
