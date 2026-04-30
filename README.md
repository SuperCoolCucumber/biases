# biases

Research scaffold for studying how LLM judges change their uncertainty under
bias manipulations such as position, authority, bandwagon, and decoy effects.

## Current status

The repository currently provides the typed core data model:

- experiment and example schemas in `src/biases/schemas.py`
- utility helpers in `src/biases/utils.py`
- a small CLI entrypoint in `src/biases/cli.py`

The full experiment runner, dataset adapters, bias transforms, and backends are
the next layers to add on top of these schemas.

## Environment setup

Use Python 3.12 and install the package:

```bash
uv sync --extra dev
```

This installs the declared dependencies from `pyproject.toml` and makes the
`biases` command available through `uv run`.

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
python main.py run-position --data-path data/processed/mtbench_stratified_198.csv
```

To submit the same experiment to SLURM with the dedicated launcher:

```bash
sbatch slurm/position_bias_qwen2_5_14b_vllm.slurm
```

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
