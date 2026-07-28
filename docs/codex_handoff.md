# Codex Handoff

This file is the portable handoff for continuing the project in a local Codex
app or on new infrastructure. It intentionally avoids machine-specific paths and
cluster-specific operational notes.

## Project Goal

The project studies whether uncertainty signals can detect biased LLM-judge
verdicts in side-by-side evaluation. Current bias families:

- Position bias: judge verdict changes when answers are swapped.
- Authority bias: judge is exposed to an expert-style cue favoring one answer.
- Bandwagon bias: judge is exposed to a majority-preference cue favoring one
  answer.

The working thesis is that biased or fragile verdicts tend to have higher
internal/output uncertainty, making uncertainty useful for routing weak judges
to stronger judges or humans.

## Repository Entry Points

- `main.py`: compatibility entrypoint.
- `src/biases/command_line.py`: main CLI parser.
- `src/biases/position_bias.py`: shared vLLM judge wrapper, MT-Bench loading,
  position runner, and uncertainty extraction.
- `src/biases/authority_bias.py`: authority cue runner.
- `src/biases/bandwagon_bias.py`: bandwagon cue runner.
- `src/biases/position_controls.py`: identical-answer and label-prior controls.
- `src/biases/stats.py`: statistical tests and confidence intervals.
- `scripts/analyze_uncertainty_routing.py`: routing and escalation analysis.
- `scripts/analyze_bias_statistics.py`: statistical summaries.
- `scripts/prepare_mtbench_full_splits.py`: full MT-Bench CSV plus
  calibration/test split creation.
- `scripts/render_slurm_jobs.py`: portable Slurm script renderer.

## Data and Artifacts

Do not commit experiment artifacts. Put large files under an external artifact
root and set:

```bash
export BIASES_ARTIFACT_ROOT=/path/to/biases-artifacts
source scripts/artifact_env.sh
```

Expected artifact layout:

- `$BIASES_ARTIFACT_ROOT/data/processed/mtbench_full.csv`
- `$BIASES_ARTIFACT_ROOT/data/processed/mtbench_full_calibration.csv`
- `$BIASES_ARTIFACT_ROOT/data/processed/mtbench_full_test.csv`
- `$BIASES_ARTIFACT_ROOT/outputs/`
- `$BIASES_ARTIFACT_ROOT/cache/`

If `BIASES_ARTIFACT_ROOT` is unset, the code falls back to local `artifacts/`,
which is ignored by Git.

## Current Experimental State

The report in `reports/full_qwen3_results_report.tex` is the best compact
summary of the completed Qwen experiments and downstream analysis. Treat it as
the current narrative, not as a raw-data source.

Previously completed experiment families include:

- Qwen/Qwen3 and Qwen/Qwen3.5 size sweeps on MT-Bench full.
- Position, authority, and bandwagon runs for multiple judge sizes.
- Position controls for most Qwen judges.
- Routing and statistical analysis over available full-run outputs.
- Cross-family judge runs where output artifacts were available.

Known run gaps from the prior infrastructure should be revalidated after
artifact transfer rather than assumed:

- Qwen3-32B position controls.
- Mistral-7B full position, authority, and bandwagon runs.

## Recommended First Local-Codex Prompt

Use this when opening the local Codex app in this repository:

```text
Read AGENTS.md, README.md, docs/codex_handoff.md,
reports/metric_definitions.md, reports/reproducibility_checklist.md, and
reports/full_qwen3_results_report.tex. Then summarize the current project state,
the active artifact-root expectations, and the safest next task.
```

## Immediate Next Steps

1. Set `BIASES_ARTIFACT_ROOT` on the new infrastructure and copy or regenerate
   processed data.
2. Run `python scripts/prepare_mtbench_full_splits.py` if processed CSVs are not
   present.
3. Recompute analyses from available outputs:

```bash
python scripts/analyze_bias_statistics.py --help
python scripts/analyze_uncertainty_routing.py --help
```

4. Re-run only missing experiment cells after validating model access,
   scheduler settings, and artifact paths.
5. Update `reports/full_qwen3_results_report.tex` from analysis artifacts, not
   from manual log snippets.
