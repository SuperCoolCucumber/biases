from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import biases.command_line as command_line
from biases.command_line import build_parser
from biases.schemas import BiasCondition, BiasType, VerdictLabel
from biases.models import get_model_profile
from biases.position_bias import (
    CONSTRAINED_LOGPROBS_MODE,
    JUDGE_OUTPUT_PARSER_VERSION,
    VERBALIZED_OUTPUT_PARSER_VERSION,
)
from biases.silent_bias_runner import (
    consistency_runs_for_condition,
    run_silent_bias_clean,
    run_silent_bias_cued,
)


class _FakeJudge:
    model_name = "Qwen/Qwen3-4B"
    logprobs_mode = CONSTRAINED_LOGPROBS_MODE
    decision_label_token_ids = {
        "A": [10],
        "B": [20],
        "tie": [30],
    }

    def render_messages(self, messages: list[dict[str, str]]) -> str:
        return "\n".join(
            f"{message['role']}:{message['content']}" for message in messages
        )

    def choose_verdict_batch(
        self,
        prompt_texts: list[str],
        seed: int,
        sampling_temperature: float,
    ) -> list[tuple[VerdictLabel, str, dict[str, float]]]:
        del seed, sampling_temperature
        results = []
        for prompt in prompt_texts:
            verdict = (
                VerdictLabel.B
                if "95% of annotators preferred answer B" in prompt
                else VerdictLabel.A
            )
            probs = (
                {"A": 0.1, "B": 0.85, "tie": 0.05}
                if verdict == VerdictLabel.B
                else {"A": 0.8, "B": 0.15, "tie": 0.05}
            )
            results.append((verdict, verdict.value, probs))
        return results

    def verbalize_confidence_batch(
        self,
        prompt_texts: list[str],
        seed: int = 0,
        max_tokens: int = 24,
    ) -> list[tuple[VerdictLabel | None, str, float | None]]:
        del seed, max_tokens
        return [(VerdictLabel.A, "A\nConfidence: 80", 80.0) for _ in prompt_texts]


class _InterruptingJudge(_FakeJudge):
    def __init__(self) -> None:
        self.batch_calls = 0

    def choose_verdict_batch(
        self,
        prompt_texts: list[str],
        seed: int,
        sampling_temperature: float,
    ) -> list[tuple[VerdictLabel, str, dict[str, float]]]:
        self.batch_calls += 1
        if self.batch_calls == 2:
            raise RuntimeError("simulated interruption")
        return super().choose_verdict_batch(
            prompt_texts,
            seed,
            sampling_temperature,
        )


class _UnavailableConfidenceJudge(_FakeJudge):
    def __init__(self, raw_output: str) -> None:
        self.raw_output = raw_output

    def verbalize_confidence_batch(
        self,
        prompt_texts: list[str],
        seed: int = 0,
        max_tokens: int = 24,
    ) -> list[tuple[VerdictLabel | None, str, float | None]]:
        del seed, max_tokens
        return [(None, self.raw_output, None) for _ in prompt_texts]


class _RawLogprobsJudge(_FakeJudge):
    logprobs_mode = "raw_logprobs"


def _write_fixture(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "question_id,prompt,response_a,response_b,winner,turn,routing_split",
                "q1,Question one?,Good answer,Bad answer,A,1,calibration",
                "q2,Question two?,First answer,Second answer,tie,2,test",
            ]
        ),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_silent_bias_cli_scheduler_tuning_is_optional() -> None:
    parser = build_parser()

    defaults = parser.parse_args(["run-silent-bias-clean"])
    tuned = parser.parse_args(
        [
            "run-silent-bias-cued",
            "--stage-a-summary",
            "clean.jsonl",
            "--max-num-batched-tokens",
            "32768",
            "--max-num-seqs",
            "128",
        ]
    )

    assert defaults.max_num_batched_tokens is None
    assert defaults.max_num_seqs is None
    assert defaults.enforce_eager is None
    assert defaults.disable_custom_all_reduce is None
    assert tuned.stage_b_routing_split == "all"
    assert tuned.max_num_batched_tokens == 32768
    assert tuned.max_num_seqs == 128


@pytest.mark.parametrize(
    ("command", "target_name", "required_args"),
    (
        ("run-silent-bias-clean", "run_silent_bias_clean", ()),
        (
            "run-silent-bias-cued",
            "run_silent_bias_cued",
            ("--stage-a-summary", "clean.jsonl"),
        ),
    ),
)
def test_silent_bias_cli_forwards_scheduler_tuning(
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    target_name: str,
    required_args: tuple[str, ...],
) -> None:
    captured: dict[str, object] = {}

    def _fake_runner(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(command_line, target_name, _fake_runner)

    exit_code = command_line.main(
        [
            command,
            *required_args,
            "--max-num-batched-tokens",
            "32768",
            "--max-num-seqs",
            "128",
            "--enforce-eager",
            "--disable-custom-all-reduce",
        ]
    )

    assert exit_code == 0
    assert captured["max_num_batched_tokens"] == 32768
    assert captured["max_num_seqs"] == 128
    assert captured["enforce_eager"] is True
    assert captured["disable_custom_all_reduce"] is True


def test_scheduler_tuning_is_runtime_provenance_and_stage_b_must_match(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "pilot.csv"
    _write_fixture(csv_path)

    default_summary = run_silent_bias_clean(
        csv_path=csv_path,
        output_dir=tmp_path / "default",
        model_name="qwen3-4b",
        consistency_runs=0,
        include_verbalized_confidence=False,
        judge=_FakeJudge(),
    )
    tuned_summary = run_silent_bias_clean(
        csv_path=csv_path,
        output_dir=tmp_path / "tuned",
        model_name="qwen3-4b",
        consistency_runs=0,
        include_verbalized_confidence=False,
        max_num_batched_tokens=32768,
        max_num_seqs=128,
        enforce_eager=True,
        disable_custom_all_reduce=True,
        judge=_FakeJudge(),
    )
    with pytest.raises(ValueError, match="inference-runtime contract"):
        run_silent_bias_cued(
            csv_path=csv_path,
            stage_a_summary_path=Path(tuned_summary["pair_summary_path"]),
            output_dir=tmp_path / "mismatched-stage-b",
            model_name="qwen3-4b",
            consistency_runs=0,
            include_verbalized_confidence=False,
            max_num_batched_tokens=16384,
            max_num_seqs=64,
            judge=_FakeJudge(),
        )
    cued_summary = run_silent_bias_cued(
        csv_path=csv_path,
        stage_a_summary_path=Path(tuned_summary["pair_summary_path"]),
        output_dir=tmp_path / "tuned",
        model_name="qwen3-4b",
        consistency_runs=0,
        include_verbalized_confidence=False,
        max_num_batched_tokens=32768,
        max_num_seqs=128,
        enforce_eager=True,
        disable_custom_all_reduce=True,
        judge=_FakeJudge(),
    )

    default_rows = _read_jsonl(Path(default_summary["raw_records_path"]))
    tuned_rows = _read_jsonl(Path(tuned_summary["raw_records_path"]))
    tuned_flat_rows = _read_jsonl(Path(tuned_summary["uncertainty_scores_path"]))
    tuned_pair_rows = _read_jsonl(Path(tuned_summary["pair_summary_path"]))
    cued_rows = _read_jsonl(Path(cued_summary["raw_records_path"]))
    cued_flat_rows = _read_jsonl(Path(cued_summary["uncertainty_scores_path"]))
    cued_pair_rows = _read_jsonl(Path(cued_summary["pair_summary_path"]))
    assert default_summary["max_num_batched_tokens"] is None
    assert default_summary["max_num_seqs"] is None
    assert tuned_summary["max_num_batched_tokens"] == 32768
    assert tuned_summary["max_num_seqs"] == 128
    runtime = tuned_summary["inference_runtime"]
    assert runtime["enforce_eager"] is True
    assert runtime["disable_custom_all_reduce"] is True
    assert set(runtime["engine_versions"]) == {
        "python",
        "torch",
        "transformers",
        "vllm",
    }
    runtime_without_digest = {
        key: value for key, value in runtime.items() if key != "runtime_sha256"
    }
    assert runtime["runtime_sha256"] == hashlib.sha256(
        json.dumps(
            runtime_without_digest,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    assert (
        tuned_summary["judge_output_parser_version"]
        == JUDGE_OUTPUT_PARSER_VERSION
    )
    assert JUDGE_OUTPUT_PARSER_VERSION == "strict_v3"
    assert (
        tuned_summary["verbalized_output_parser_version"]
        == VERBALIZED_OUTPUT_PARSER_VERSION
        == "strict_v3"
    )
    assert tuned_summary["logprobs_mode"] == CONSTRAINED_LOGPROBS_MODE
    assert tuned_summary["verbalized_parse_status_counts"] == {
        "not_requested": 4
    }
    assert cued_summary["max_num_batched_tokens"] == 32768
    assert cued_summary["max_num_seqs"] == 128
    assert cued_summary["inference_runtime"] == tuned_summary["inference_runtime"]
    assert cued_summary["verbalized_parse_status_counts"] == {
        "not_requested": 64
    }
    assert {
        (
            row["metadata"]["max_num_batched_tokens"],
            row["metadata"]["max_num_seqs"],
        )
        for row in default_rows
    } == {(None, None)}
    assert {
        (
            row["metadata"]["max_num_batched_tokens"],
            row["metadata"]["max_num_seqs"],
        )
        for row in tuned_rows
    } == {(32768, 128)}
    assert {
        (row["max_num_batched_tokens"], row["max_num_seqs"])
        for row in tuned_flat_rows
    } == {(32768, 128)}
    assert {
        (row["max_num_batched_tokens"], row["max_num_seqs"])
        for row in tuned_pair_rows
    } == {(32768, 128)}
    assert {
        row["metadata"]["verbalized_parse_status"] for row in tuned_rows
    } == {"not_requested"}
    assert {
        row["metadata"]["verbalized_output_parser_version"]
        for row in tuned_rows
    } == {VERBALIZED_OUTPUT_PARSER_VERSION}
    assert {
        row["verbalized_output_parser_version"] for row in tuned_flat_rows
    } == {VERBALIZED_OUTPUT_PARSER_VERSION}
    assert {
        row["verbalized_output_parser_version"] for row in tuned_pair_rows
    } == {VERBALIZED_OUTPUT_PARSER_VERSION}
    assert {
        (
            row["spec"]["logprobs_mode"],
            row["metadata"]["logprobs_mode"],
        )
        for row in tuned_rows
    } == {
        (
            CONSTRAINED_LOGPROBS_MODE,
            CONSTRAINED_LOGPROBS_MODE,
        )
    }
    assert {
        row["logprobs_mode"] for row in tuned_flat_rows
    } == {CONSTRAINED_LOGPROBS_MODE}
    assert {
        row["logprobs_mode"] for row in tuned_pair_rows
    } == {CONSTRAINED_LOGPROBS_MODE}
    expected_token_texts = {
        label: list(texts)
        for label, texts in get_model_profile("qwen3-4b").verdict_token_texts.items()
    }
    expected_token_ids = _FakeJudge.decision_label_token_ids
    assert tuned_summary["verdict_token_texts"] == expected_token_texts
    assert tuned_summary["verdict_token_ids"] == expected_token_ids
    assert {
        json.dumps(row["spec"]["verdict_token_texts"], sort_keys=True)
        for row in tuned_rows
    } == {json.dumps(expected_token_texts, sort_keys=True)}
    assert {
        json.dumps(row["spec"]["verdict_token_ids"], sort_keys=True)
        for row in tuned_rows
    } == {json.dumps(expected_token_ids, sort_keys=True)}
    for rows in (tuned_flat_rows, tuned_pair_rows):
        assert {
            json.dumps(row["verdict_token_texts"], sort_keys=True)
            for row in rows
        } == {json.dumps(expected_token_texts, sort_keys=True)}
        assert {
            json.dumps(row["verdict_token_ids"], sort_keys=True)
            for row in rows
        } == {json.dumps(expected_token_ids, sort_keys=True)}
    assert {
        row["verbalized_parse_status"] for row in tuned_flat_rows
    } == {"not_requested"}
    assert {
        row["verbalized_parse_status"] for row in tuned_pair_rows
    } == {"not_requested"}
    assert {
        (
            row["metadata"]["max_num_batched_tokens"],
            row["metadata"]["max_num_seqs"],
        )
        for row in cued_rows
    } == {(32768, 128)}
    assert {
        (row["max_num_batched_tokens"], row["max_num_seqs"])
        for row in cued_flat_rows
    } == {(32768, 128)}
    assert {
        (row["max_num_batched_tokens"], row["max_num_seqs"])
        for row in cued_pair_rows
    } == {(32768, 128)}
    assert {
        row["metadata"]["verbalized_parse_status"] for row in cued_rows
    } == {"not_requested"}
    assert {
        row["verbalized_parse_status"] for row in cued_flat_rows
    } == {"not_requested"}
    assert {
        row["verbalized_parse_status"] for row in cued_pair_rows
    } == {"not_requested"}
    assert [row["spec_hash"] for row in default_rows] == [
        row["spec_hash"] for row in tuned_rows
    ]
    assert [row["prompt_hash"] for row in default_rows] == [
        row["prompt_hash"] for row in tuned_rows
    ]
    with pytest.raises(
        ValueError,
        match="max_num_batched_tokens.*max_num_seqs",
    ):
        run_silent_bias_clean(
            csv_path=csv_path,
            output_dir=tmp_path / "tuned",
            model_name="qwen3-4b",
            consistency_runs=0,
            include_verbalized_confidence=False,
            max_num_batched_tokens=16384,
            max_num_seqs=64,
            judge=_FakeJudge(),
        )


def test_fake_backend_runs_both_stages_and_resume_is_idempotent(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "pilot.csv"
    output_dir = tmp_path / "outputs"
    _write_fixture(csv_path)
    judge = _FakeJudge()

    stage_a = run_silent_bias_clean(
        csv_path=csv_path,
        output_dir=output_dir,
        model_name="qwen3-4b",
        consistency_runs=0,
        include_verbalized_confidence=False,
        judge=judge,
    )
    assert stage_a["records_written"] == 4

    stage_a_again = run_silent_bias_clean(
        csv_path=csv_path,
        output_dir=output_dir,
        model_name="qwen3-4b",
        consistency_runs=0,
        include_verbalized_confidence=False,
        judge=judge,
    )
    assert stage_a_again["records_written"] == 4

    stage_b = run_silent_bias_cued(
        csv_path=csv_path,
        stage_a_summary_path=Path(stage_a["pair_summary_path"]),
        output_dir=output_dir,
        model_name="qwen3-4b",
        consistency_runs=0,
        include_verbalized_confidence=False,
        judge=judge,
    )
    assert stage_b["records_written"] == 64

    clean_rows = _read_jsonl(Path(stage_a["uncertainty_scores_path"]))
    cued_rows = _read_jsonl(Path(stage_b["uncertainty_scores_path"]))
    expected_revision = get_model_profile("qwen3-4b").revision
    assert stage_a["model_revision"] == expected_revision
    assert stage_b["model_revision"] == expected_revision
    assert {
        row["spec"]["model_revision"]
        for row in _read_jsonl(Path(stage_a["raw_records_path"]))
    } == {expected_revision}
    assert {
        row["metadata"]["judge_output_parser_version"]
        for row in _read_jsonl(Path(stage_a["raw_records_path"]))
    } == {JUDGE_OUTPUT_PARSER_VERSION}
    assert {
        row["judge_output_parser_version"] for row in clean_rows
    } == {JUDGE_OUTPUT_PARSER_VERSION}
    assert {row["ordering"] for row in clean_rows} == {"ab", "ba"}
    assert {row["question_id"] for row in clean_rows} == {"q1", "q2"}
    assert {row["source_row_index"] for row in clean_rows} == {0, 1}
    assert all(row["pair_key"] for row in cued_rows)
    assert all(row["clean_record_id"] for row in cued_rows)
    assert {
        row["variant_id"]
        for row in cued_rows
        if row["pair_id"] == "q1:turn-1"
    } == {
        f"{family}_{direction}_{dose}_{ordering}"
        for family, doses in (("bandwagon", (55, 70, 85, 95)), ("authority", (1, 2, 3, 4)))
        for direction in ("congruent", "incongruent")
        for dose in doses
        for ordering in ("ab", "ba")
    }


def test_stage_b_can_be_limited_to_the_held_out_routing_split(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "question_disjoint.csv"
    output_dir = tmp_path / "outputs"
    _write_fixture(csv_path)
    judge = _FakeJudge()

    stage_a = run_silent_bias_clean(
        csv_path=csv_path,
        output_dir=output_dir,
        model_name="qwen3-4b",
        consistency_runs=0,
        include_verbalized_confidence=False,
        judge=judge,
    )
    stage_b = run_silent_bias_cued(
        csv_path=csv_path,
        stage_a_summary_path=Path(stage_a["pair_summary_path"]),
        output_dir=output_dir,
        model_name="qwen3-4b",
        consistency_runs=0,
        include_verbalized_confidence=False,
        stage_b_routing_split="test",
        judge=judge,
    )

    assert stage_a["records_written"] == 4
    assert stage_b["records_written"] == 32
    assert stage_b["source_pairs"] == 2
    assert stage_b["stage_b_source_pairs"] == 1
    assert stage_b["stage_b_routing_split"] == "test"
    assert stage_a["inference_runtime"] == stage_b["inference_runtime"]
    cued_rows = _read_jsonl(Path(stage_b["uncertainty_scores_path"]))
    assert {row["question_id"] for row in cued_rows} == {"q2"}
    assert {row["routing_split"] for row in cued_rows} == {"test"}
    assert {row["model_revision"] for row in cued_rows} == {
        get_model_profile("qwen3-4b").revision
    }
    assert {
        json.dumps(row["inference_runtime"], sort_keys=True)
        for row in cued_rows
    } == {json.dumps(stage_b["inference_runtime"], sort_keys=True)}


def test_controlled_stage_b_requires_complete_stage_a_and_matching_runtime(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "question_disjoint.csv"
    _write_fixture(csv_path)
    stage_a = run_silent_bias_clean(
        csv_path=csv_path,
        output_dir=tmp_path / "stage-a",
        model_name="qwen3-4b",
        consistency_runs=0,
        include_verbalized_confidence=False,
        batch_size=64,
        judge=_FakeJudge(),
    )

    incomplete_path = tmp_path / "incomplete-stage-a.jsonl"
    rows = _read_jsonl(Path(stage_a["pair_summary_path"]))
    incomplete_path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows[1:]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="incomplete or duplicated"):
        run_silent_bias_cued(
            csv_path=csv_path,
            stage_a_summary_path=incomplete_path,
            output_dir=tmp_path / "incomplete-stage-b",
            model_name="qwen3-4b",
            consistency_runs=0,
            include_verbalized_confidence=False,
            stage_b_routing_split="test",
            judge=_FakeJudge(),
        )

    with pytest.raises(ValueError, match="inference-runtime contract"):
        run_silent_bias_cued(
            csv_path=csv_path,
            stage_a_summary_path=Path(stage_a["pair_summary_path"]),
            output_dir=tmp_path / "mismatched-stage-b",
            model_name="qwen3-4b",
            consistency_runs=0,
            include_verbalized_confidence=False,
            batch_size=32,
            stage_b_routing_split="test",
            judge=_FakeJudge(),
        )

    with pytest.raises(ValueError, match="inference-runtime contract"):
        run_silent_bias_cued(
            csv_path=csv_path,
            stage_a_summary_path=Path(stage_a["pair_summary_path"]),
            output_dir=tmp_path / "mismatched-all-stage-b",
            model_name="qwen3-4b",
            consistency_runs=0,
            include_verbalized_confidence=False,
            batch_size=32,
            stage_b_routing_split="all",
            judge=_FakeJudge(),
        )


def test_resume_rejects_changed_inference_runtime(tmp_path: Path) -> None:
    csv_path = tmp_path / "pilot.csv"
    output_dir = tmp_path / "outputs"
    _write_fixture(csv_path)
    run_silent_bias_clean(
        csv_path=csv_path,
        output_dir=output_dir,
        model_name="qwen3-4b",
        consistency_runs=0,
        include_verbalized_confidence=False,
        batch_size=64,
        judge=_FakeJudge(),
    )

    with pytest.raises(ValueError, match="inference_runtime"):
        run_silent_bias_clean(
            csv_path=csv_path,
            output_dir=output_dir,
            model_name="qwen3-4b",
            consistency_runs=0,
            include_verbalized_confidence=False,
            batch_size=32,
            judge=_FakeJudge(),
        )


def test_flat_scores_include_each_secondary_channel_verdict(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "pilot.csv"
    output_dir = tmp_path / "outputs"
    _write_fixture(csv_path)

    summary = run_silent_bias_clean(
        csv_path=csv_path,
        output_dir=output_dir,
        model_name="qwen3-4b",
        consistency_runs=1,
        include_verbalized_confidence=True,
        judge=_FakeJudge(),
    )
    flat_rows = _read_jsonl(Path(summary["uncertainty_scores_path"]))
    raw_rows = _read_jsonl(Path(summary["raw_records_path"]))
    pair_rows = _read_jsonl(Path(summary["pair_summary_path"]))

    assert {row["verbalized_verdict"] for row in flat_rows} == {"A"}
    assert {row["consistency_majority_verdict"] for row in flat_rows} == {"A"}
    assert summary["verbalized_parse_status_counts"] == {"parsed": 4}
    assert {
        row["metadata"]["verbalized_parse_status"] for row in raw_rows
    } == {"parsed"}
    assert {row["verbalized_parse_status"] for row in flat_rows} == {"parsed"}
    assert {row["verbalized_parse_status"] for row in pair_rows} == {"parsed"}
    assert {
        row["uncertainty"]["verbalized"]["verdict"] for row in raw_rows
    } == {"A"}

    for row in raw_rows:
        row["uncertainty"]["verbalized"].pop("verdict")
    raw_path = Path(summary["raw_records_path"])
    raw_path.write_text(
        "".join(
            f"{json.dumps(row, sort_keys=True)}\n"
            for row in raw_rows
        ),
        encoding="utf-8",
    )
    resumed = run_silent_bias_clean(
        csv_path=csv_path,
        output_dir=output_dir,
        model_name="qwen3-4b",
        consistency_runs=1,
        include_verbalized_confidence=True,
        judge=_FakeJudge(),
    )
    resumed_flat_rows = _read_jsonl(Path(resumed["uncertainty_scores_path"]))
    assert {row["verbalized_verdict"] for row in resumed_flat_rows} == {"A"}


@pytest.mark.parametrize(
    ("raw_output", "expected_status"),
    (
        ("", "missing"),
        ("I cannot provide an atomic confidence answer.", "unparseable"),
    ),
)
def test_fresh_runs_preserve_unavailable_verbalized_channel_status(
    tmp_path: Path,
    raw_output: str,
    expected_status: str,
) -> None:
    csv_path = tmp_path / "pilot.csv"
    output_dir = tmp_path / expected_status
    _write_fixture(csv_path)

    summary = run_silent_bias_clean(
        csv_path=csv_path,
        output_dir=output_dir,
        model_name="qwen3-4b",
        consistency_runs=0,
        include_verbalized_confidence=True,
        judge=_UnavailableConfidenceJudge(raw_output),
    )
    raw_rows = _read_jsonl(Path(summary["raw_records_path"]))
    flat_rows = _read_jsonl(Path(summary["uncertainty_scores_path"]))
    pair_rows = _read_jsonl(Path(summary["pair_summary_path"]))

    assert summary["verbalized_parse_status_counts"] == {
        expected_status: 4
    }
    assert {
        row["metadata"]["verbalized_parse_status"] for row in raw_rows
    } == {expected_status}
    assert {
        row["verbalized_parse_status"] for row in flat_rows
    } == {expected_status}
    assert {
        row["verbalized_parse_status"] for row in pair_rows
    } == {expected_status}
    assert {
        (
            row["uncertainty"]["verbalized"]["confidence"],
            row["uncertainty"]["verbalized"]["uncertainty"],
            row["uncertainty"]["verbalized"]["verdict"],
        )
        for row in raw_rows
    } == {(None, None, None)}


def test_extreme_schedule_skips_middle_doses() -> None:
    low = BiasCondition(
        bias_type=BiasType.BANDWAGON,
        variant_id="bandwagon_incongruent_55_ab",
        dose=55,
    )
    middle = BiasCondition(
        bias_type=BiasType.BANDWAGON,
        variant_id="bandwagon_incongruent_70_ab",
        dose=70,
    )

    assert (
        consistency_runs_for_condition(
            low,
            consistency_runs=8,
            consistency_schedule="extremes",
        )
        == 8
    )
    assert (
        consistency_runs_for_condition(
            middle,
            consistency_runs=8,
            consistency_schedule="extremes",
        )
        == 0
    )


def test_resume_rejects_incompatible_existing_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "pilot.csv"
    output_dir = tmp_path / "outputs"
    _write_fixture(csv_path)
    judge = _FakeJudge()

    summary = run_silent_bias_clean(
        csv_path=csv_path,
        output_dir=output_dir,
        model_name="qwen3-4b",
        consistency_runs=0,
        include_verbalized_confidence=False,
        judge=judge,
    )
    raw_path = Path(summary["raw_records_path"])
    rows = _read_jsonl(raw_path)
    rows[0]["input_file_hash"] = "stale"
    raw_path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="input_file_hash"):
        run_silent_bias_clean(
            csv_path=csv_path,
            output_dir=output_dir,
            model_name="qwen3-4b",
            consistency_runs=0,
            include_verbalized_confidence=False,
            judge=judge,
        )


def test_runner_and_resume_require_processed_logprobs_mode(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "pilot.csv"
    output_dir = tmp_path / "outputs"
    _write_fixture(csv_path)

    with pytest.raises(ValueError, match="processed constrained log probabilities"):
        run_silent_bias_clean(
            csv_path=csv_path,
            output_dir=output_dir,
            model_name="qwen3-4b",
            consistency_runs=0,
            include_verbalized_confidence=False,
            judge=_RawLogprobsJudge(),
        )

    summary = run_silent_bias_clean(
        csv_path=csv_path,
        output_dir=output_dir,
        model_name="qwen3-4b",
        consistency_runs=0,
        include_verbalized_confidence=False,
        judge=_FakeJudge(),
    )
    raw_path = Path(summary["raw_records_path"])
    rows = _read_jsonl(raw_path)
    rows[0]["spec"]["logprobs_mode"] = "raw_logprobs"
    rows[0]["metadata"]["logprobs_mode"] = "raw_logprobs"
    raw_path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="spec_logprobs_mode.*metadata_logprobs_mode",
    ):
        run_silent_bias_clean(
            csv_path=csv_path,
            output_dir=output_dir,
            model_name="qwen3-4b",
            consistency_runs=0,
            include_verbalized_confidence=False,
            judge=_FakeJudge(),
        )


def test_resume_rejects_changed_verdict_token_contract(tmp_path: Path) -> None:
    csv_path = tmp_path / "pilot.csv"
    output_dir = tmp_path / "outputs"
    _write_fixture(csv_path)
    judge = _FakeJudge()

    run_silent_bias_clean(
        csv_path=csv_path,
        output_dir=output_dir,
        model_name="qwen3-4b",
        consistency_runs=0,
        include_verbalized_confidence=False,
        judge=judge,
    )
    changed = _FakeJudge()
    changed.decision_label_token_ids = {
        **judge.decision_label_token_ids,
        "tie": [31],
    }

    with pytest.raises(
        ValueError,
        match="spec_hash.*verdict_token_ids",
    ):
        run_silent_bias_clean(
            csv_path=csv_path,
            output_dir=output_dir,
            model_name="qwen3-4b",
            consistency_runs=0,
            include_verbalized_confidence=False,
            judge=changed,
        )


def test_resume_rejects_stale_prompt_and_extraction_provenance(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "pilot.csv"
    output_dir = tmp_path / "outputs"
    _write_fixture(csv_path)
    judge = _FakeJudge()

    summary = run_silent_bias_clean(
        csv_path=csv_path,
        output_dir=output_dir,
        model_name="qwen3-4b",
        consistency_runs=0,
        include_verbalized_confidence=False,
        judge=judge,
    )
    raw_path = Path(summary["raw_records_path"])
    rows = _read_jsonl(raw_path)
    rows[0]["prompt_hash"] = "stale"
    rows[0]["spec_hash"] = "stale"
    rows[0]["metadata"]["conversation_extraction_mode"] = "stale"
    rows[0]["metadata"]["judge_output_parser_version"] = "stale"
    rows[0]["metadata"]["verbalized_output_parser_version"] = "stale"
    raw_path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=(
            "prompt_hash.*spec_hash.*conversation_extraction_mode"
            ".*judge_output_parser_version"
            ".*verbalized_output_parser_version"
        ),
    ):
        run_silent_bias_clean(
            csv_path=csv_path,
            output_dir=output_dir,
            model_name="qwen3-4b",
            consistency_runs=0,
            include_verbalized_confidence=False,
            judge=judge,
        )


def test_resume_rejects_missing_verbalized_parse_status(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "pilot.csv"
    output_dir = tmp_path / "outputs"
    _write_fixture(csv_path)
    judge = _FakeJudge()

    summary = run_silent_bias_clean(
        csv_path=csv_path,
        output_dir=output_dir,
        model_name="qwen3-4b",
        consistency_runs=0,
        include_verbalized_confidence=True,
        judge=judge,
    )
    raw_path = Path(summary["raw_records_path"])
    rows = _read_jsonl(raw_path)
    rows[0]["metadata"].pop("verbalized_parse_status")
    raw_path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="verbalized_parse_status"):
        run_silent_bias_clean(
            csv_path=csv_path,
            output_dir=output_dir,
            model_name="qwen3-4b",
            consistency_runs=0,
            include_verbalized_confidence=True,
            judge=judge,
        )


def test_resume_rejects_a_different_model_revision(tmp_path: Path) -> None:
    csv_path = tmp_path / "pilot.csv"
    output_dir = tmp_path / "outputs"
    _write_fixture(csv_path)
    judge = _FakeJudge()

    summary = run_silent_bias_clean(
        csv_path=csv_path,
        output_dir=output_dir,
        model_name="qwen3-4b",
        consistency_runs=0,
        include_verbalized_confidence=False,
        judge=judge,
    )
    raw_path = Path(summary["raw_records_path"])
    rows = _read_jsonl(raw_path)
    rows[0]["spec"]["model_revision"] = "different-revision"
    raw_path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="model_revision"):
        run_silent_bias_clean(
            csv_path=csv_path,
            output_dir=output_dir,
            model_name="qwen3-4b",
            consistency_runs=0,
            include_verbalized_confidence=False,
            judge=judge,
        )


def test_stage_b_rejects_stage_a_summary_from_an_old_output_parser(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "pilot.csv"
    output_dir = tmp_path / "outputs"
    _write_fixture(csv_path)
    summary = run_silent_bias_clean(
        csv_path=csv_path,
        output_dir=output_dir,
        model_name="qwen3-4b",
        consistency_runs=0,
        include_verbalized_confidence=False,
        judge=_FakeJudge(),
    )
    pair_summary_path = Path(summary["pair_summary_path"])
    rows = _read_jsonl(pair_summary_path)
    for row in rows:
        row.pop("judge_output_parser_version")
    pair_summary_path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="judge_output_parser_version"):
        run_silent_bias_cued(
            csv_path=csv_path,
            stage_a_summary_path=pair_summary_path,
            output_dir=output_dir,
            model_name="qwen3-4b",
            consistency_runs=0,
            include_verbalized_confidence=False,
            judge=_FakeJudge(),
        )


def test_stage_b_rejects_stage_a_summary_from_an_old_verbalized_parser(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "pilot.csv"
    output_dir = tmp_path / "outputs"
    _write_fixture(csv_path)
    summary = run_silent_bias_clean(
        csv_path=csv_path,
        output_dir=output_dir,
        model_name="qwen3-4b",
        consistency_runs=0,
        include_verbalized_confidence=True,
        judge=_FakeJudge(),
    )
    pair_summary_path = Path(summary["pair_summary_path"])
    rows = _read_jsonl(pair_summary_path)
    for row in rows:
        row["verbalized_output_parser_version"] = "strict_v2"
    pair_summary_path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="verbalized_output_parser_version",
    ):
        run_silent_bias_cued(
            csv_path=csv_path,
            stage_a_summary_path=pair_summary_path,
            output_dir=output_dir,
            model_name="qwen3-4b",
            consistency_runs=0,
            include_verbalized_confidence=True,
            judge=_FakeJudge(),
        )


def test_runner_checkpoints_completed_batches_for_resume(tmp_path: Path) -> None:
    csv_path = tmp_path / "pilot.csv"
    output_dir = tmp_path / "outputs"
    _write_fixture(csv_path)

    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_silent_bias_clean(
            csv_path=csv_path,
            output_dir=output_dir,
            model_name="qwen3-4b",
            consistency_runs=0,
            include_verbalized_confidence=False,
            batch_size=2,
            judge=_InterruptingJudge(),
        )

    checkpoint_path = output_dir / "silent_bias_stage_a_run_records.jsonl"
    checkpoint_rows = _read_jsonl(checkpoint_path)
    assert len(checkpoint_rows) == 2

    resumed = run_silent_bias_clean(
        csv_path=csv_path,
        output_dir=output_dir,
        model_name="qwen3-4b",
        consistency_runs=0,
        include_verbalized_confidence=False,
        batch_size=2,
        judge=_FakeJudge(),
    )
    assert resumed["records_written"] == 4
    assert len(_read_jsonl(checkpoint_path)) == 4


def test_runner_recovers_only_an_incomplete_checkpoint_tail(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "pilot.csv"
    output_dir = tmp_path / "outputs"
    _write_fixture(csv_path)

    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_silent_bias_clean(
            csv_path=csv_path,
            output_dir=output_dir,
            model_name="qwen3-4b",
            consistency_runs=0,
            include_verbalized_confidence=False,
            batch_size=2,
            judge=_InterruptingJudge(),
        )

    checkpoint_path = output_dir / "silent_bias_stage_a_run_records.jsonl"
    checkpoint_rows = _read_jsonl(checkpoint_path)
    with checkpoint_path.open("ab") as handle:
        handle.write(b'{"record_id": "incomplete')

    resumed = run_silent_bias_clean(
        csv_path=csv_path,
        output_dir=output_dir,
        model_name="qwen3-4b",
        consistency_runs=0,
        include_verbalized_confidence=False,
        batch_size=2,
        judge=_FakeJudge(),
    )

    resumed_rows = _read_jsonl(checkpoint_path)
    assert resumed["records_written"] == 4
    assert {row["record_id"] for row in checkpoint_rows}.issubset(
        {row["record_id"] for row in resumed_rows}
    )
    assert len(resumed_rows) == 4

    with checkpoint_path.open("ab") as handle:
        handle.write(b"not-json\n")
    with pytest.raises(ValueError, match="Invalid JSONL"):
        run_silent_bias_clean(
            csv_path=csv_path,
            output_dir=output_dir,
            model_name="qwen3-4b",
            consistency_runs=0,
            include_verbalized_confidence=False,
            batch_size=2,
            judge=_FakeJudge(),
        )


def test_final_materialization_is_sorted_and_byte_deterministic(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "pilot.csv"
    output_dir = tmp_path / "outputs"
    _write_fixture(csv_path)

    summary = run_silent_bias_clean(
        csv_path=csv_path,
        output_dir=output_dir,
        model_name="qwen3-4b",
        consistency_runs=0,
        include_verbalized_confidence=False,
        batch_size=2,
        judge=_FakeJudge(),
    )
    raw_path = Path(summary["raw_records_path"])
    uncertainty_path = Path(summary["uncertainty_scores_path"])
    pair_summary_path = Path(summary["pair_summary_path"])
    expected = {
        raw_path: raw_path.read_bytes(),
        uncertainty_path: uncertainty_path.read_bytes(),
        pair_summary_path: pair_summary_path.read_bytes(),
    }

    raw_lines = raw_path.read_bytes().splitlines(keepends=True)
    raw_path.write_bytes(b"".join(reversed(raw_lines)))
    assert raw_path.read_bytes() != expected[raw_path]

    resumed = run_silent_bias_clean(
        csv_path=csv_path,
        output_dir=output_dir,
        model_name="qwen3-4b",
        consistency_runs=0,
        include_verbalized_confidence=False,
        batch_size=2,
        judge=_FakeJudge(),
    )

    assert resumed["records_written"] == 4
    assert {path: path.read_bytes() for path in expected} == expected
    rows = _read_jsonl(raw_path)
    sort_keys = [
        (
            str(row["pair_key"]),
            str(row["condition"]["variant_id"]),
            str(row["record_id"]),
        )
        for row in rows
    ]
    assert sort_keys == sorted(sort_keys)


def test_pair_key_prevents_repeated_judgment_record_id_collisions(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "repeated.csv"
    csv_path.write_text(
        "\n".join(
            [
                "question_id,prompt,response_a,response_b,winner,turn,routing_split",
                "q1,Same question?,Same A,Same B,A,1,calibration",
                "q1,Same question?,Same A,Same B,B,1,test",
            ]
        ),
        encoding="utf-8",
    )
    summary = run_silent_bias_clean(
        csv_path=csv_path,
        output_dir=tmp_path / "outputs",
        model_name="qwen3-4b",
        consistency_runs=0,
        include_verbalized_confidence=False,
        judge=_FakeJudge(),
    )

    rows = _read_jsonl(Path(summary["raw_records_path"]))
    assert len(rows) == 4
    assert len({row["record_id"] for row in rows}) == 4
    assert len({row["pair_key"] for row in rows}) == 4
