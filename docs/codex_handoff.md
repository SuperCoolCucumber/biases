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

## Silent Bias Paper Pipeline

The paired-condition pipeline now implements the preregistered RQ1--RQ3 design
in `docs/experiment_plan.md`.

- **P0 — Pairing:** optional backward-compatible `RunRecord` linkage fields,
  stable pair/group/twin keys, provenance hashes, and a migration helper for
  legacy JSONL.
- **P1 — Prompt ladders:** centralized, auditable authority and bandwagon
  templates with four doses, congruent/incongruent cue targets, clean-tie
  handling, and exact-prompt tests.
- **P2 — Models:** a registry owns chat-template policy, verdict-token
  candidates, stop tokens, system-role handling, and model-specific assistant
  prefill. The 20-example smoke command fails unless all constrained logit rows
  are valid and at least 99% of verdicts parse.
- **P3 — Staged execution:** `run-silent-bias-clean` produces Stage A clean
  summaries; `run-silent-bias-cued` is generated purely from those summaries.
  Both support provenance-checked resume, complete or boundary-dose consistency
  schedules, batching, and portable Slurm templates. The budget estimator gives
  exact condition and generation counts before submission.
- **P4 — Analysis:** tidy RQ1 paired shifts and susceptibility AUROCs, RQ2
  calibration/risk coverage/clean-threshold transfer/McNemar/swap averaging,
  and RQ3 psychometric dose fits, Gaussian GEE trends, permutation sensitivity,
  and a question-random-intercept mixed logistic model. All direct outputs
  carry analysis-spec and input-file hashes.
- **P5 — Assets:** deterministic PDF figures and booktabs tables are generated
  from conventional analysis CSVs with stable ordering and fixed PDF metadata.
- **P6 — Digest:** the generated claims-to-evidence report only states a result
  when its preregistered interval and adjusted test fields exist; missing
  evidence remains visibly unavailable.

The mandatory pilot dataset is balanced jointly by human winner and the
inherited deterministic calibration/test routing split. The default reduced
pilot uses `k=4` consistency sampling at clean and boundary doses while retaining
logit and verbalized-confidence passes across the full 34-condition grid. See
the README for the complete command sequence.

Before promoting a judge model to a full run, require:

1. a passing 20-example verdict-extraction artifact;
2. all 198 pilot rows in both orderings;
3. exactly one clean partner for each cued row and exactly two ordering twins
   per condition group;
4. observed generation counts matching the budget plan; and
5. two byte-identical paper-asset regenerations from the same inputs.

## Completed Silent-Bias Pilot

The real 198-row pilot completed end to end for `Qwen/Qwen3-4B` with the
reduced `k=4` boundary-dose consistency schedule:

- verdict-extraction smoke: 20/20 parseable verdicts and 20/20 normalized
  constrained label distributions;
- Stage A: 396 clean records (198 AB and 198 BA);
- Stage B: 6,336 cued records (3,168 authority and 3,168 bandwagon), evenly
  split between congruent and incongruent directions;
- all 396 clean pair keys have the expected 16 cued partners, all 3,168
  condition groups have both ordering twins, and both planning-issue files
  are empty;
- consistency sampling is present on 3,168 boundary-dose records and omitted
  on 3,168 sanctioned middle-dose records; logit and verbalized channels are
  complete throughout;
- the analysis contains 6,336 paired shifts with zero unmatched cued records
  and zero unused clean records.

Pilot provenance:

- processed-data SHA-256:
  `5983747255fdd73b4dd2375b80822629240e34778e5372d2cdfc4ec9278c0325`;
- Stage A score SHA-256:
  `27f35e5255dd4ca7dc77e5beb3d0240ff353acf3744cf16769209af164bd1418`;
- Stage B score SHA-256:
  `582f6e6ee0a9f5947b6a19fe0ac1f20444ade52981b72fb901eb84bd390b72fa`;
- analysis-spec hash:
  `785e6d18b8d202d531acf5a1906fbfce3402c355e530e40aa86bea42c4aa7df5`.

The 2,000-resample/10,000-permutation analysis package and all five paper
figures, booktabs tables, manifests, and claims-to-evidence digest regenerate
byte-identically from the copied inputs. This is a single-model pilot and
pipeline-validation result, not the multi-model/full-data evidence needed for
paper claims; full MT-Bench runs and the additional judge families remain
pending.
