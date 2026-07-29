from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

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


def test_validation_cli_accepts_optional_vllm_scheduler_tuning() -> None:
    parser = validation_module.build_parser()

    defaults = parser.parse_args(["--model-name", "qwen3-4b"])
    tuned = parser.parse_args(
        [
            "--model-name",
            "qwen3-4b",
            "--max-num-batched-tokens",
            "32768",
            "--max-num-seqs",
            "128",
        ]
    )

    assert defaults.max_num_batched_tokens is None
    assert defaults.max_num_seqs is None
    assert tuned.max_num_batched_tokens == 32768
    assert tuned.max_num_seqs == 128


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


@pytest.mark.parametrize(
    ("returned_verdict", "raw_output", "issue_fragment"),
    (
        (VerdictLabel.A, "not a verdict", "not a supported verdict form"),
        (VerdictLabel.A, "B", "does not match returned verdict"),
    ),
)
def test_smoke_parse_rate_requires_matching_raw_verdict(
    returned_verdict: VerdictLabel,
    raw_output: str,
    issue_fragment: str,
) -> None:
    result = validation_module.validate_smoke_results(
        [
            (
                returned_verdict,
                raw_output,
                {"A": 0.8, "B": 0.15, "tie": 0.05},
            )
        ],
        expected_examples=1,
    )

    assert result["parseable_examples"] == 0
    assert result["parse_rate"] == 0.0
    assert result["valid_probability_examples"] == 1
    assert result["passed"] is False
    assert issue_fragment in result["issues"][0]


def test_smoke_rejects_returned_verdict_that_is_not_probability_map() -> None:
    result = validation_module.validate_smoke_results(
        [
            (
                VerdictLabel.B,
                "B",
                {"A": 0.60, "B": 0.35, "tie": 0.05},
            )
        ],
        expected_examples=1,
    )

    assert result["parse_rate"] == 1.0
    assert result["valid_probability_rate"] == 1.0
    assert result["map_alignment_rate"] == 0.0
    assert result["passed"] is False
    assert "probability MAP 'A'" in result["issues"][0]


def _constrained_row(
    verdict: VerdictLabel = VerdictLabel.A,
) -> tuple[VerdictLabel, str, dict[str, float]]:
    return verdict, verdict.value, {"A": 0.8, "B": 0.15, "tie": 0.05}


def test_native_classifier_recognizes_bracketed_format_without_calling_it_direct() -> None:
    classified = validation_module.classify_native_verdict("[[B]]")

    assert classified.verdict == VerdictLabel.B
    assert classified.format_category == "double_bracket_label"


@pytest.mark.parametrize(
    "text",
    (
        "A or B",
        "Answer A is stronger.",
        "[[A]] and [[B]]",
        "I choose [[A]].",
    ),
)
def test_native_classifier_rejects_ambiguous_or_prose_forms(text: str) -> None:
    classified = validation_module.classify_native_verdict(text)

    assert classified.verdict is None
    assert classified.format_category == "unparseable"


def test_ambiguous_native_label_cannot_satisfy_contract() -> None:
    result = validation_module.validate_native_smoke_results(
        [
            validation_module.NativeGeneration(
                text="A or B",
                token_ids=(10, 20, 11),
            )
        ],
        [_constrained_row()],
        expected_examples=1,
        allowed_first_token_ids={10, 11, 12},
    )

    assert result["parse_rate"] == 0.0
    assert result["first_token_compatible_rate"] == 1.0
    assert result["contract_rate"] == 0.0
    assert result["passed"] is False


def test_native_contract_accepts_direct_label_at_declared_first_token() -> None:
    native_row = validation_module.NativeGeneration(text="A", token_ids=(10,))

    result = validation_module.validate_native_smoke_results(
        [native_row] * 20,
        [_constrained_row()] * 20,
        expected_examples=20,
        allowed_first_token_ids={10, 11, 12},
    )

    assert result["passed"] is True
    assert result["parse_rate"] == 1.0
    assert result["first_token_compatible_rate"] == 1.0
    assert result["verdict_agreement_rate"] == 1.0
    assert result["contract_rate"] == 1.0
    assert result["format_counts"] == {"direct_label": 20}


def test_bracketed_native_output_is_visible_but_fails_first_token_contract() -> None:
    native_row = validation_module.NativeGeneration(
        text="[[A]]",
        token_ids=(99, 99, 10),
    )

    result = validation_module.validate_native_smoke_results(
        [native_row] * 20,
        [_constrained_row()] * 20,
        expected_examples=20,
        allowed_first_token_ids={10, 11, 12},
    )

    assert result["passed"] is False
    assert result["parse_rate"] == 1.0
    assert result["verdict_agreement_rate"] == 1.0
    assert result["first_token_compatible_rate"] == 0.0
    assert result["contract_rate"] == 0.0
    assert result["format_counts"] == {"double_bracket_label": 20}
    serialized = json.dumps(result)
    assert "[[A]]" not in serialized
    assert len(result["sample_fingerprints_by_format"]["double_bracket_label"]) == 2


def test_top_level_pass_requires_native_diagnostic() -> None:
    constrained = validation_module.validate_smoke_results(
        [_constrained_row()],
        expected_examples=1,
    )
    native = validation_module.validate_native_smoke_results(
        [
            validation_module.NativeGeneration(
                text="[[A]]",
                token_ids=(99, 10),
            )
        ],
        [_constrained_row()],
        expected_examples=1,
        allowed_first_token_ids={10},
    )

    merged = validation_module.require_native_validation(constrained, native)

    assert merged["constrained_passed"] is True
    assert merged["native_diagnostic"]["passed"] is False
    assert merged["passed"] is False
    assert merged["issues"][-1] == "native unconstrained verdict diagnostic failed"


def test_native_generation_is_greedy_and_has_no_allowed_token_constraint(
    monkeypatch: Any,
) -> None:
    captured_sampling: dict[str, object] = {}
    captured_prompts: list[str] = []

    class _FakeSamplingParams:
        def __init__(self, **kwargs: object) -> None:
            captured_sampling.update(kwargs)

    class _Completion:
        text = "[[A]]"
        token_ids = [91, 91, 10]

    class _RequestOutput:
        outputs = [_Completion()]

    class _Model:
        def generate(
            self,
            prompts: list[str],
            _sampling_params: object,
            *,
            use_tqdm: bool,
        ) -> list[_RequestOutput]:
            assert use_tqdm is False
            captured_prompts.extend(prompts)
            return [_RequestOutput() for _prompt in prompts]

    class _Profile:
        stop_token_texts = ("</s>",)

    class _Judge:
        profile = _Profile()
        model = _Model()

        @staticmethod
        def _prepare_prompt(prompt: str) -> str:
            return f"prepared:{prompt}"

    monkeypatch.setattr(validation_module, "SamplingParams", _FakeSamplingParams)

    results = validation_module.generate_native_verdict_batch(
        _Judge(),
        ["prompt one", "prompt two"],
        seed=7,
        max_tokens=16,
    )

    assert captured_prompts == ["prepared:prompt one", "prepared:prompt two"]
    assert captured_sampling == {
        "max_tokens": 16,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 7,
        "stop": ["</s>"],
        "skip_special_tokens": True,
    }
    assert "allowed_token_ids" not in captured_sampling
    assert results == [
        validation_module.NativeGeneration(text="[[A]]", token_ids=(91, 91, 10)),
        validation_module.NativeGeneration(text="[[A]]", token_ids=(91, 91, 10)),
    ]
