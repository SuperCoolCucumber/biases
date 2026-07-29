from __future__ import annotations

import json
from pathlib import Path

import pytest

from biases.schemas import BiasCondition, BiasType, VerdictLabel
from biases.models import get_model_profile
from biases.silent_bias_runner import (
    consistency_runs_for_condition,
    run_silent_bias_clean,
    run_silent_bias_cued,
)


class _FakeJudge:
    model_name = "Qwen/Qwen3-4B"

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
    raw_path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="prompt_hash.*spec_hash.*conversation_extraction_mode",
    ):
        run_silent_bias_clean(
            csv_path=csv_path,
            output_dir=output_dir,
            model_name="qwen3-4b",
            consistency_runs=0,
            include_verbalized_confidence=False,
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
