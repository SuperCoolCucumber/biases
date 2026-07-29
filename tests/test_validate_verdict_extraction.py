from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from biases.schemas import VerdictLabel


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "validate_verdict_extraction.py"
)


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "validate_verdict_extraction",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


validation_module = _load_module()


def test_smoke_validation_requires_all_twenty_rows_at_ninety_nine_percent() -> None:
    row = (
        VerdictLabel.A,
        "A",
        {"A": 0.8, "B": 0.15, "tie": 0.05},
    )

    passed = validation_module.validate_smoke_results(
        [row] * 20,
        expected_examples=20,
    )
    short = validation_module.validate_smoke_results(
        [row] * 19,
        expected_examples=20,
    )

    assert passed["passed"] is True
    assert short["passed"] is False


def test_smoke_validation_rejects_invalid_probability_support() -> None:
    result = validation_module.validate_smoke_results(
        [(VerdictLabel.A, "A", {"A": 1.0, "B": 0.0})],
        expected_examples=1,
    )

    assert result["passed"] is False
    assert result["valid_probability_examples"] == 0
