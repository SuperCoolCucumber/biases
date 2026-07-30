# Silent Bias Analysis

The analysis package consumes the flat Stage A and Stage B uncertainty JSONL
artifacts. It writes tidy CSVs under the active artifact root; generated data,
figures, and tables do not belong in Git.

## Run the analysis

Install the optional analysis dependencies and set the artifact root:

```bash
uv sync --extra analysis --extra dev
export BIASES_ARTIFACT_ROOT=/path/to/biases-artifacts
```

Generate the RQ1--RQ3 analysis package:

```bash
uv run --extra analysis python scripts/analyze_silent_bias.py \
  --stage-a "$BIASES_ARTIFACT_ROOT/outputs/pilot/silent_bias_stage_a_uncertainty_scores.jsonl" \
  --stage-b "$BIASES_ARTIFACT_ROOT/outputs/pilot/silent_bias_stage_b_uncertainty_scores.jsonl" \
  --output-dir "$BIASES_ARTIFACT_ROOT/outputs/analysis" \
  --bootstrap-resamples 2000 \
  --gee-bootstrap-workers 8 \
  --trend-permutations 10000 \
  --seed 42
```

`--gee-bootstrap-workers` parallelizes only the seeded GEE bootstrap refits
and is excluded from the scientific specification hash; serial and parallel
runs preserve the same draw stream and output order. Keep the worker count at
or below the allocated CPU count. When using multiple workers, set
`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and
`NUMEXPR_NUM_THREADS=1` so each worker does not create a second thread pool.

The conventional CSV outputs are:

- `paired_shifts.csv`
- `rq1_silent_shift.csv`
- `rq1_susceptibility.csv`
- `rq2_calibration.csv`
- `rq2_reliability.csv`
- `rq2_risk_coverage.csv`
- `rq2_threshold_transfer.csv`
- `rq2_mcnemar.csv`
- `rq3_dose_response.csv`
- `rq3_uncertainty_trend.csv`
- `rq3_uncertainty_by_dose.csv`
- `rq3_modeling.csv`

Every CSV row carries the analysis-spec hash and direct-input hashes.

## Routing-split population

`paired_shifts.csv` retains both deterministic source splits for auditability.
All aggregate RQ1 and RQ3 outputs are estimated from `routing_split=test`
only, and every emitted row declares that split. RQ2 retains split-stratified
descriptive calibration outputs, while calibration rows enter the headline
threshold-transfer analysis only to select the clean abstention thresholds.
The mixed-effects model also excludes `clean_tie=true` rows, matching the
preregistered primary population; tie-stratified RQ1 and family-specific RQ3
summaries remain explicit non-primary robustness rows on the test split.
Missing or unknown routing-split values make analysis fail rather than being
silently dropped.

## Mixed-effects model

The exact preregistered paper formula is:

```text
flip ~ dose * family * congruence + (1 | question)
```

`fit_flip_mixed_logit` implements the fixed component as
`flip ~ dose * family * congruence` and the question random intercept as
`0 + C(question_id)` through
`statsmodels.genmod.bayes_mixed_glm.BinomialBayesMixedGLM`. The output records
the estimator and fit method. If the optional dependency is unavailable or the
fit fails, the modeling CSV contains an explicit unavailable status instead of
a substituted model.

The primary within-question uncertainty trend is a Gaussian GEE with
exchangeable within-question correlation:

```text
uncertainty ~ normalized_dose
```

The output records convergence diagnostics, a 2,000-resample question-cluster
bootstrap interval for each primary GEE slope, and an explicit unavailable row
for degenerate or failed fits. `rq3_uncertainty_by_dose.csv` carries the
corresponding pre-first-flip entropy means and question-cluster bootstrap
intervals at every tested dose. A question-clustered sign-permutation trend test
is reported as a sensitivity analysis; neither trend estimator replaces the
mixed-effects flip model.

## Generate paper assets

Create deterministic figures, booktabs tables, an asset manifest, and the
claims-to-evidence digest:

```bash
uv run --extra analysis python scripts/make_paper_assets.py \
  --analysis-dir "$BIASES_ARTIFACT_ROOT/outputs/analysis" \
  --output-dir "$BIASES_ARTIFACT_ROOT/outputs/analysis/paper_assets" \
  --report-path reports/paper_results.md
```

Matplotlib is imported only when figures are requested. For table-only
validation, use `--skip-figures`. Missing conventional inputs fail the command
unless `--allow-missing` is explicitly supplied; that mode marks evidence as
unavailable and never invents a result.

Paper outputs are deterministic for identical input bytes:

- input rows and groups are stably sorted;
- headline RQ1 and RQ3 assets require the test split, including empirical
  distributions read from `paired_shifts.csv`;
- headline RQ2 assets use only MSP, the test split, and non-tie rows before
  selecting the highest incongruent dose, so confidence channels are never
  pooled;
- CSV and LaTeX formatting is fixed;
- PDF metadata omits creation and modification timestamps;
- the manifest contains content hashes and no absolute paths or runtime
  timestamps.

The threshold-transfer table and digest carry both the clustered
risk-inflation interval and the accepted-flip-fraction interval. RQ1
susceptibility follows its preregistered rule
`auc_difference_ci_low > 0`; RQ2 and RQ3 decisions likewise use the exact
decision-rule fields recorded by the analysis CSVs. A transferred threshold
with no accepted test examples has undefined realized risk and is reported as
unavailable rather than as a guarantee failure.

RQ2 keeps confidence and verdict semantics aligned. MSP is scored against the
constrained deterministic verdict, consistency agreement against the
consistency majority verdict, and verbalized confidence against the verdict
parsed from that same free-generation pass. Rows lacking the corresponding
secondary verdict are excluded only from that secondary channel. Flip counts
for threshold transfer compare clean and cued verdicts from the selected
channel.

ECE, reliability, risk--coverage, and AURC are emitted for each available
confidence channel. Multiclass Brier score is emitted only for MSP because only
the constrained-logit pass provides an A/B/tie probability vector; secondary
channel rows leave `brier` undefined and report `brier_n=0`.
Calibration summaries also report `total_n`, `missing_n`, and
`availability_rate` so verbalized parse failures remain visible per condition
instead of disappearing through complete-case filtering.

The dose-response output retains the raw fitted P25 extrapolation and records
the tested dose bounds plus `p25_range_status`. Paper assets render an
out-of-range P25 as below/above the tested ladder instead of presenting an
impossible cue dose as an observed sensitivity threshold.

Regenerate the assets in two clean output directories and compare their hashes
before a full paper run is accepted.

## Validate a completed package

First run `scripts/validate_silent_bias_artifacts.py` against the source CSV and
every model artifact directory. That is the raw Stage A/B semantic gate: it
checks JSONL schemas, experiment specs, record identities, channel
availability, and the experimental grid.

After analysis and both asset regenerations, run the package gate:

```bash
uv run --extra analysis python scripts/validate_silent_bias_analysis.py \
  --analysis-dir "$BIASES_ARTIFACT_ROOT/outputs/analysis" \
  --stage-a \
    "$BIASES_ARTIFACT_ROOT/outputs/full/model-a/silent_bias_stage_a_uncertainty_scores.jsonl" \
    "$BIASES_ARTIFACT_ROOT/outputs/full/model-b/silent_bias_stage_a_uncertainty_scores.jsonl" \
  --stage-b \
    "$BIASES_ARTIFACT_ROOT/outputs/full/model-a/silent_bias_stage_b_uncertainty_scores.jsonl" \
    "$BIASES_ARTIFACT_ROOT/outputs/full/model-b/silent_bias_stage_b_uncertainty_scores.jsonl" \
  --asset-package "$BIASES_ARTIFACT_ROOT/outputs/analysis/paper_assets" reports/paper_results.md \
  --asset-package "$BIASES_ARTIFACT_ROOT/outputs/analysis/paper_assets_repro" "$BIASES_ARTIFACT_ROOT/outputs/analysis/report_repro/paper_results.md" \
  --expected-model model-a \
  --expected-model model-b \
  --source-pairs 3337 \
  --report-path "$BIASES_ARTIFACT_ROOT/outputs/analysis/package_validation.json"
```

List one Stage A and Stage B path per judge, and repeat `--expected-model` once
per judge. This is a post-analysis package validator. The Stage paths bind the
analysis provenance to exact direct-input bytes; this command does not parse
those JSONL files or rederive the analysis. It checks CSV schemas and internal
equations, the emitted paired grid and model coverage, manifests, paper assets,
and preregistered primary selectors. It must therefore be layered after
`validate_silent_bias_artifacts.py`, not used as a substitute for it.

Missing estimates caused by legitimate data degeneracy (for example, zero
accepted examples at a transferred threshold) are reported separately as
availability warnings. Use `--require-primary-available` when a workflow
should promote those warnings to a nonzero exit code. Structural integrity
failures always return a nonzero exit code.
