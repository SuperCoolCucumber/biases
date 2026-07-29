# Slurm Notes

This project can run on Slurm, but committed Slurm material is a template layer,
not an infrastructure contract.

## Rules

- Keep `BIASES_ARTIFACT_ROOT` outside the Git repository for datasets, model
  caches, checkpoints, and outputs.
- Keep secrets out of scripts. Source them from an external env file through
  `BIASES_HF_ENV` if needed.
- Treat `#SBATCH` partition, QOS, account, GPU count, memory, and wall-time
  lines as infrastructure-specific.
- Run `bash -n` on scripts before submitting.
- Render generated jobs into `slurm/generated/`; that directory is ignored by
  Git.

## Rendering Jobs

Use the renderer for repeatable legacy job creation:

```bash
python3.12 scripts/render_slurm_jobs.py --kind controls --output-dir slurm/generated
python3.12 scripts/render_slurm_jobs.py --kind phase3 --output-dir slurm/generated
```

If the scheduler requires explicit routing fields:

```bash
python3.12 scripts/render_slurm_jobs.py \
  --kind phase3 \
  --output-dir slurm/generated \
  --partition <partition> \
  --qos <qos> \
  --account <account>
```

Review rendered files before submission.

## Silent Bias Stage A and Stage B

The Silent Bias renderer writes one Stage A and one Stage B job per selected
model. Stage A runs the clean AB/BA conditions. Stage B consumes Stage A's
pair-summary and runs the dose grid.

Estimate generation counts before rendering:

```bash
python3.12 scripts/estimate_run_budget.py \
  --examples 198 \
  --models 2 \
  --consistency-k 8 \
  --consistency-schedule all
```

Render a pilot for two model templates:

```bash
python3.12 scripts/render_slurm_jobs.py \
  --kind silent-bias \
  --models qwen3_14b mistral7b \
  --data-file mtbench_stratified_198.csv \
  --limit 198 \
  --consistency-runs 8 \
  --consistency-schedule all \
  --output-dir slurm/generated
```

The renderer does not set a partition, QOS, or account unless explicitly
provided. GPU, memory, wall-time, and tensor-parallel values are starting
points, not validated scheduler requirements. Override them while rendering or
edit the generated files after consulting the active infrastructure:

```bash
python3.12 scripts/render_slurm_jobs.py \
  --kind silent-bias \
  --models qwen3_14b \
  --gpus <count> \
  --mem <memory> \
  --time <walltime> \
  --tensor-parallel-size <count> \
  --partition <partition> \
  --qos <qos> \
  --account <account> \
  --output-dir slurm/generated
```

Validate every rendered file:

```bash
bash -n slurm/generated/silent_bias_stage_a_qwen3_14b.slurm
bash -n slurm/generated/silent_bias_stage_b_qwen3_14b.slurm
```

Submit Stage B only after Stage A succeeds. Use the same `RUN_GROUP` for both
jobs so their default artifact paths match:

```bash
export RUN_GROUP=<stable-run-group>
stage_a_job="$(
  sbatch --parsable slurm/generated/silent_bias_stage_a_qwen3_14b.slurm
)"
sbatch \
  --dependency="afterok:${stage_a_job}" \
  slurm/generated/silent_bias_stage_b_qwen3_14b.slurm
```

Generated jobs default to the CLI commands `run-silent-bias-clean` and
`run-silent-bias-cued`. Override `STAGE_A_COMMAND` or `STAGE_B_COMMAND` at
submission time if an integration branch uses another command name. Stage B
uses:

```bash
STAGE_A_SUMMARY=<pair-summary-path>
```

when the Stage A filename or location differs from the rendered default.
Other runtime overrides include `DATA_PATH`, `DATASET_SPLIT`, `MODEL_NAME`,
`OUTPUT_DIR`, `CONSISTENCY_RUNS`, `CONSISTENCY_SCHEDULE`,
`SAMPLING_TEMPERATURE`, `INCLUDE_VERBALIZED_CONFIDENCE`, `LIMIT`,
`TENSOR_PARALLEL_SIZE`, `MAX_MODEL_LEN`, `GPU_MEMORY_UTILIZATION`, `DTYPE`,
`PYTHON_BIN`, and `EXTRA_ARGS`.

The sanctioned reduced schedule is `CONSISTENCY_SCHEDULE=extremes`: clean plus
the lowest and highest cue doses receive consistency samples, while the logit
and enabled verbalized passes still cover the complete dose grid. Record this
choice in `ExperimentSpec`.

## Templates

`slurm/templates/generic_bias_job.slurm` is a minimal portable template. Copy it
or render jobs with `scripts/render_slurm_jobs.py`, then adapt the scheduler
directives for the active infrastructure.

`slurm/templates/silent_bias_job.slurm` is the renderer source for both stages.
It contains replacement markers and should be rendered rather than submitted
directly.
