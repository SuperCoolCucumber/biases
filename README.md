# biases

Research scaffold for studying how LLM judges change their uncertainty under
bias manipulations such as position, authority, and bandwagon effects.

## Current status

The repository currently provides an end-to-end experiment and analysis
pipeline:

- typed experiment schemas in `src/biases/schemas.py`
- MT-Bench loading and split preparation
- vLLM-based judge runners for position, authority, and bandwagon bias
- identical-answer and label-prior position controls
- logit, consistency, and optional verbalized uncertainty extraction
- routing and statistical analysis scripts
- portable artifact-root configuration and Slurm rendering helpers

## Environment setup

Use Python 3.12 and install the package:

```bash
source scripts/artifact_env.sh
uv sync --extra dev
```

This installs the declared dependencies from `pyproject.toml` and makes the
`biases` command available through `uv run`. The environment script keeps
package, Hugging Face, vLLM, torch, and Triton caches under one artifact root.

## Storage layout

Keep the repository for code and small text artifacts only. Datasets, model
caches, checkpoints, and experiment outputs should live outside Git.

Set an artifact root before running larger jobs:

```bash
export BIASES_ARTIFACT_ROOT=/path/to/biases-artifacts
source scripts/artifact_env.sh
```

The expected artifact layout is:

- `$BIASES_ARTIFACT_ROOT/data/processed/` for generated MT-Bench CSVs
- `$BIASES_ARTIFACT_ROOT/outputs/` for experiment outputs and analyses
- `$BIASES_ARTIFACT_ROOT/cache/` for Hugging Face, vLLM, and tool caches
- `/tmp/$USER/$SLURM_JOB_ID/` for node-local scratch caches such as Triton

If `BIASES_ARTIFACT_ROOT` is not set, the code defaults to a local
`artifacts/` directory, which is ignored by Git.

The repository ignores local `data/`, `outputs/`, `artifacts/`, `.cache/`,
`checkpoints/`, and `models/` directories to avoid storing heavy artifacts.

If you also want the virtualenv itself outside the repository, set this before
`uv sync`:

```bash
export UV_PROJECT_ENVIRONMENT="$BIASES_ARTIFACT_ROOT/venvs/biases"
uv sync --extra dev
```

## How to use the code

### 1. Inspect the CLI

The package exposes a basic CLI:

```bash
uv run biases --help
```

To print a sample serialized record shape:

```bash
uv run biases schema-demo
```

To run the position-bias experiment on the local CSV sample with vLLM:

```bash
uv sync --extra local --extra dev
python main.py run-position \
  --data-path "${BIASES_ARTIFACT_ROOT:-artifacts}/data/processed/mtbench_stratified_198.csv"
```

For Slurm-based infrastructure, see `docs/slurm.md`. Existing Slurm launchers
should be treated as templates: update scheduler directives, GPU counts, memory,
and artifact paths for the target cluster before submission.

### Full MT-Bench Qwen3 runs

Prepare the full MT-Bench human-judgment CSV and deterministic routing splits:

```bash
python scripts/prepare_mtbench_full_splits.py
```

This writes:

- `$BIASES_ARTIFACT_ROOT/data/processed/mtbench_full.csv`
- `$BIASES_ARTIFACT_ROOT/data/processed/mtbench_full_calibration.csv`
- `$BIASES_ARTIFACT_ROOT/data/processed/mtbench_full_test.csv`

The full CSV contains a `routing_split` column. The experiment runners preserve
that value in raw records, pair summaries, and flat uncertainty-score files, so
calibration/test routing analysis can be done after one full run.

Render Slurm scripts for the Qwen3 non-thinking full-dataset experiments on
systems that use Slurm:

```bash
python scripts/render_slurm_jobs.py --kind controls --output-dir slurm/generated
python scripts/render_slurm_jobs.py --kind phase3 --output-dir slurm/generated
```

Add `--partition`, `--qos`, or `--account` if the target scheduler requires
them. Review rendered scripts before submitting them with `sbatch`.

For Qwen3, the runner prefills an empty thinking block before generation so the
first generated token is the verdict label in non-thinking mode. The logit
uncertainty pass constrains that first token to tokenizer IDs for `A`, `B`, and
`T`, then saves the resulting label probabilities, entropy, MSP, margin,
verbalized confidence, and consistency entropy to
`*_uncertainty_scores.jsonl`.

See `docs/codex_handoff.md` for a compact handoff summary for future Codex
sessions.

### 2. Create experiment objects in Python

The main objects are `Candidate`, `JudgeExample`, `BiasCondition`,
`PromptPackage`, `JudgeRequest`, `JudgeResponse`, `UncertaintyBundle`, and
`RunRecord`.

Example:

```python
from biases.schemas import (
    BiasCondition,
    BiasType,
    Candidate,
    ExperimentSpec,
    JudgeExample,
    JudgeResponse,
    LogitMetrics,
    OutputMode,
    PromptPackage,
    RunRecord,
    UncertaintyBundle,
    VerdictLabel,
)
from biases.utils import stable_hash

example = JudgeExample(
    example_id="q1",
    question_id=1,
    prompt_messages=[{"role": "user", "content": "Explain overfitting."}],
    candidates={
        "A": Candidate(label=VerdictLabel.A, response="Overfitting is memorization."),
        "B": Candidate(label=VerdictLabel.B, response="Overfitting is when a model fits noise."),
    },
    human_winner=VerdictLabel.B,
)

condition = BiasCondition(
    bias_type=BiasType.POSITION,
    variant_id="swap_control",
)

prompt_text = "Judge which answer is better: A or B."
prompt = PromptPackage(
    prompt_text=prompt_text,
    output_mode=OutputMode.CHOICE_ONLY,
    allowed_labels=[VerdictLabel.A, VerdictLabel.B, VerdictLabel.TIE],
    prompt_hash=stable_hash({"prompt": prompt_text}),
)

response = JudgeResponse(
    verdict=VerdictLabel.B,
    raw_output="B",
    prompt_logprobs={"A": 0.20, "B": 0.70, "tie": 0.10},
)

uncertainty = UncertaintyBundle(
    logit=LogitMetrics.from_probs(response.prompt_logprobs or {}),
)

record = RunRecord(
    record_id=stable_hash({"example_id": example.example_id, "seed": 0}),
    spec=ExperimentSpec(
        dataset_name="fixture",
        dataset_split="train",
        model_name="demo-judge",
        backend_name="manual",
        bias_name=condition.bias_type,
        output_mode=prompt.output_mode,
        uncertainty_methods=["logit"],
        consistency_runs=1,
        temperature=0.0,
    ),
    example_id=example.example_id,
    question_id=str(example.question_id),
    condition=condition,
    seed=0,
    verdict=response.verdict,
    raw_output=response.raw_output,
    prompt_hash=prompt.prompt_hash,
    uncertainty=uncertainty,
    raw_prompt_logprobs=response.prompt_logprobs,
)

print(record.model_dump_json(indent=2))
```

### 3. Compute uncertainty metrics

The current built-in helpers already support two simple operations:

- `LogitMetrics.from_probs(...)` computes entropy, MSP, top-2 margin, and
  normalized entropy from label probabilities.
- `VerbalizedMetrics.from_confidence(...)` converts a 0-100 confidence score
  into normalized confidence and uncertainty.

Example:

```python
from biases.schemas import LogitMetrics, VerbalizedMetrics

print(LogitMetrics.from_probs({"A": 0.1, "B": 0.8, "tie": 0.1}))
print(VerbalizedMetrics.from_confidence(72))
```

### 4. Serialize outputs

Every schema is a Pydantic model, so you can serialize them consistently:

```python
record_dict = record.model_dump()
json_text = record.model_dump_json(indent=2)
```

For stable IDs, use:

```python
from biases.utils import stable_hash

stable_hash({"example_id": "q1", "seed": 0})
```

## What to extend next

The current foundation is designed so the next modules can plug into these
types without changing their public shape:

- dataset adapter returning `JudgeExample`
- prompt builder returning `PromptPackage`
- backend returning `JudgeResponse`
- uncertainty estimator returning `UncertaintyBundle`
- runner writing `RunRecord`

That separation is the intended way to keep experiments configurable and easy to
extend later.
