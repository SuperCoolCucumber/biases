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

Use the renderer for repeatable job creation:

```bash
python scripts/render_slurm_jobs.py --kind controls --output-dir slurm/generated
python scripts/render_slurm_jobs.py --kind phase3 --output-dir slurm/generated
```

If the scheduler requires explicit routing fields:

```bash
python scripts/render_slurm_jobs.py \
  --kind phase3 \
  --output-dir slurm/generated \
  --partition <partition> \
  --qos <qos> \
  --account <account>
```

Review rendered files before submission.

## Generic Template

`slurm/templates/generic_bias_job.slurm` is a minimal portable template. Copy it
or render jobs with `scripts/render_slurm_jobs.py`, then adapt the scheduler
directives for the active infrastructure.
