from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "prepare_mtbench_stratified.py"
)


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "prepare_mtbench_stratified",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


prepare = _load_module()


def test_pilot_is_stratified_by_winner_and_inherited_routing() -> None:
    frame = pd.DataFrame(
        {
            "question_id": list(range(12)),
            "model_a": ["left"] * 12,
            "model_b": ["right"] * 12,
            "winner": ["model_a"] * 4 + ["model_b"] * 4 + ["tie"] * 4,
            "turn": [1] * 12,
            "conversation_a": [f"a-{index}" for index in range(12)],
            "conversation_b": [f"b-{index}" for index in range(12)],
        }
    )
    pilot = prepare.stratified_sample_from_frame(
        frame,
        target_size=6,
        seed=42,
        calibration_fraction=0.5,
    )

    counts = pilot.groupby(["winner", "routing_split"]).size().to_dict()
    assert counts == {
        ("model_a", "calibration"): 1,
        ("model_a", "test"): 1,
        ("model_b", "calibration"): 1,
        ("model_b", "test"): 1,
        ("tie", "calibration"): 1,
        ("tie", "test"): 1,
    }
