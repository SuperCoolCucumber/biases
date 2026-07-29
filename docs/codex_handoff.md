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

## Retracted Pilot and 2026-07-29 Preflight Correction

Do not use the previous Qwen3-4B pilot for behavioral conclusions. Its
conversation columns were Python literal strings, but the old loader only
accepted JSON and therefore judged serialized conversations instead of the
selected MT-Bench turn. The old Stage A/B and analysis hashes remain in Git
history solely as an engineering audit trail. The tracked preliminary report
is marked retracted until a corrected pilot replaces it.

The repaired data path now:

- pins the upstream MT-Bench revision and writes conversation columns as
  canonical JSON while safely accepting legacy Python literals;
- selects question 1 / answer 1 for source turn 1;
- preserves both questions and assistant turns for source turn 2 while
  explicitly marking the second answer as the evaluation target;
- records the extraction mode and selected turn in run artifacts; and
- validates exact turn-1 and turn-2 prompt content in tests.

Regenerated inputs:

- pilot: 198/198 usable pairs, 99 calibration and 99 test rows, SHA-256
  `d0e2dd12c5c6a2b378b12ab0ab363850147f1fa501fd13d25860737fc80d6b7a`;
- full: 3,355 source rows, 3,346 usable pairs, 1,677 calibration and 1,678
  test rows, SHA-256
  `26cbf6de9985ddf6c5d7bacc7c46df8242882180a99b8489dab82abb90d13a54`.

The nine unusable full rows are all question 127, turn 1, with an empty source
candidate response. Full-run gates use 3,346 pairs: 6,692 Stage A and 107,072
Stage B records per model. Checkpointing now appends crash-safe batches instead
of rewriting the growing raw JSONL at every batch. RQ2 risk-coverage is
`O(n log n)`, and clean/test cluster-bootstrap work is reused through exact
question-level sufficient statistics. Rerun all four model smokes and the
entire 198-pair pilot before starting full inference.

The prescribed row-level split is not question-disjoint: all 80 full-data
question IDs occur in both calibration and test. Preserve it for the primary
analysis as required by `docs/experiment_plan.md`, but report this dependence
prominently and add a question-disjoint robustness analysis before claiming
transfer to unseen questions.
