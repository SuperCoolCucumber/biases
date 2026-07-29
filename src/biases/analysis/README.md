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
  --trend-permutations 10000 \
  --seed 42
```

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

The dose-response output retains the raw fitted P25 extrapolation and records
the tested dose bounds plus `p25_range_status`. Paper assets render an
out-of-range P25 as below/above the tested ladder instead of presenting an
impossible cue dose as an observed sensitivity threshold.

Regenerate the assets in two clean output directories and compare their hashes
before a full paper run is accepted.
