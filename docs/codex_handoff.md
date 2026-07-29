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
- full: 3,355 source rows, 3,337 runnable pairs, 1,663 calibration and 1,674
  test rows, SHA-256
  `26cbf6de9985ddf6c5d7bacc7c46df8242882180a99b8489dab82abb90d13a54`.

The 18 non-runnable rows are all question 127: nine turn-1 comparisons have an
empty target candidate response, and the corresponding nine turn-2 comparisons
have an empty first-turn candidate response required as context. Full-run gates
use 3,337 pairs: 6,674 Stage A and 106,784 Stage B records per model.
Checkpointing now appends crash-safe batches instead of rewriting the growing
raw JSONL at every batch. RQ2 risk-coverage is `O(n log n)`, and clean/test
cluster-bootstrap work is reused through exact question-level sufficient
statistics. Rerun all four model smokes and the entire 198-pair pilot before
starting full inference.

The prescribed row-level split is not question-disjoint: all 80 full-data
question IDs occur in both calibration and test. Preserve it for the primary
analysis as required by `docs/experiment_plan.md`, but report this dependence
prominently and add a question-disjoint robustness analysis before claiming
transfer to unseen questions.

## 2026-07-29 RQ2 Channel-Semantics Correction

RQ2 secondary confidence channels now use the verdict from the same inference
pass. Consistency agreement is scored against its sampled majority verdict and
verbalized confidence against its free-pass verdict; neither silently falls
back to the constrained deterministic verdict when its own verdict is missing.
Accepted-flip fractions are also channel-specific for single-ordering analyses.
MSP, the preregistered primary channel, remains tied to the constrained
deterministic verdict and is unchanged.

The raw run records preserve both secondary raw outputs and verdicts. Derived
flat score files now expose `consistency_majority_verdict` and
`verbalized_verdict`. Current-parser raw records can be rematerialized without
GPU inference. Older records require an explicit migration that reparses the
stored raw output before assigning the current parser version; ordinary resume
correctly rejects them. Legacy flat files without a recoverable secondary
verdict remain usable for MSP and RQ1 uncertainty shifts, but that secondary
RQ2 channel is reported unavailable.

## 2026-07-29 Parser and Artifact Integrity Gate

Judge-output parsing is versioned as `strict_v2`. Deterministic constrained
outputs must agree with the aggregated A/B/tie probability MAP, while
verbalized outputs must contain an unambiguous verdict and confidence in the
declared two-line format. The verdict-extraction smoke artifact now checks the
same 20 prompts under constrained decoding and native greedy decoding before a
model can enter a full run.

Use `scripts/migrate_silent_bias_parser.py` to reparse pre-versioned stored raw
outputs and rematerialize their flat scores and pair summaries; in-place mode
requires backups. Use `scripts/validate_silent_bias_artifacts.py` after both
stages to verify the exact experimental grid, linkage and provenance fields,
and all parser-derived uncertainty values against their stored primitives.
vLLM scheduler overrides are optional, recorded throughout the artifact
bundle, and must match on resume.

## 2026-07-29 Constrained-Logprob Integrity Correction

The corrected-conversation Qwen3-4B pilot is also invalid for behavioral
analysis. vLLM 0.19.1 returned default `raw_logprobs` before applying the
registered A/B/T whitelist, so the stored normalized label probabilities can
omit allowed-token mass and cannot be repaired from the artifact.

New inference is fail-closed:

- `VLLMJudge` requests `processed_logprobs`;
- probability extraction accepts only registered token IDs and requires exact
  coverage of every allowed ID;
- the mode is recorded in the spec and every derived artifact layer, checked
  on resume, and enforced by smoke and artifact validators;
- migration marks undeclared legacy artifacts as `raw_logprobs`, preserves
  their original `spec_hash`, and does not make them eligible for analysis.

Mistral-7B-Instruct-v0.3 is deferred because its tokenizer does not validate
the runner's string render-and-reencode prompt transport. The registry now
fails closed for that model until a token-ID prompt adapter exists.
The initial replacement candidate matrix was Qwen3-4B, Qwen3-14B,
OLMo2-7B-Instruct, and Hermes-3-Llama-3.1-8B. A pinned OLMo tokenizer
preflight covered all 113,458 runnable full-data condition prompts in the
longer verbalized-confidence format: the maximum was 3,453 tokens, with zero
4,096-token context violations and zero string-transport token mismatches.
Including the 24-token confidence-generation allowance, the maximum request is
3,477 tokens. The campaign wrapper gates on persisted preflight artifacts.
Verification after this correction: 244 tests passed under Python 3.12,
changed Python files compiled, `git diff --check` passed, and the campaign
wrapper passed `bash -n`.

## 2026-07-30 OLMo2 and Phi-4 Gate Failures; OLMo3 Substitution

OLMo2-7B passed constrained extraction with complete
`processed_logprobs` coverage on 20/20 smoke examples, but failed the
independent native verdict gate. Only 18/20 examples both began with a
registered verdict token and agreed with the constrained verdict, below the
predeclared 99% minimum. Preserve the failed smoke artifact and do not run an
OLMo2 pilot or weaken the gate.

Phi-4-14B was the next public third-family candidate. `microsoft/phi-4` was
pinned to revision
`2db69c1c3e91a05d2c64a3185acfbaf36f744e25`; its pinned tokenizer preserves
canonical chat-template IDs through string transport, exposes single-token
A/B/T variants, supports the system role, and has a 16,384-token native
context. It passed constrained extraction on 20/20 examples, but only 6/20
native outputs both began with a registered verdict token and agreed with the
constrained verdict. Preserve this failed smoke as an exclusion artifact.
The predeclared 99% native gate remains unchanged.

Verification after the Phi-4 substitution: 246 tests passed under Python
3.12, changed Python files compiled, `git diff --check` passed, and the Slurm
template passed `bash -n`.

The current replacement required matrix is Qwen3-4B, Qwen3-14B,
OLMo3-7B-Instruct, and Hermes-3-Llama-3.1-8B, spanning Qwen3, OLMo3, and
Llama 3.1 families. `allenai/Olmo-3-7B-Instruct` is pinned to revision
`6e5971d9eba42665f5bd5a0fcf047f299ce1dccc`. Its tokenizer-only validation
passed canonical chat-template string transport and the registered A/B/T
single-token probes. The 20-example constrained and native GPU smoke remains
pending; OLMo3 cannot enter the pilot unless it passes the same unchanged
gates.

Verification after the OLMo3 substitution: 248 tests passed under Python
3.12, changed Python files compiled, `git diff --check` passed, and the
campaign wrapper passed `bash -n`.
