# Preliminary Silent Bias Pilot Report

## Status

This report summarizes the preregistered 198-pair pipeline-validation pilot for
`Qwen/Qwen3-4B`. It is preliminary single-model evidence, not the full-data,
multi-model result required for paper claims.

The pilot completed both answer orderings and the full authority/bandwagon dose
grid:

- 198 source answer pairs from 75 question clusters;
- 396 clean ordered judgments;
- 6,336 clean–cue paired comparisons;
- 20/20 parseable verdict-extraction smoke examples;
- no unmatched cued records, unused clean records, clean ties, or planning
  errors.

Consistency sampling used the sanctioned reduced schedule: `k=4` at clean and
boundary doses, with logit and verbalized-confidence passes at every dose.
Reported intervals use 2,000 question-cluster bootstrap resamples; the trend
analysis uses 10,000 permutations.

## Preliminary Findings

### RQ1 — Silent bias

The pilot supports the proposed silent-bias signal. Among incongruent examples
whose verdict did not flip, mean probability movement toward the cued answer
was positive at every dose:

- authority: `0.025` to `0.042`, with every clustered 95% interval above zero;
- bandwagon: `0.027` to `0.052`, with every clustered 95% interval above zero.

The lowest-dose signed probability shift also predicted highest-dose flips
better than clean entropy alone. The paired AUROC improvements were `0.093`
for authority (`0.972` versus `0.879`; 95% CI `[0.064, 0.121]`) and `0.133`
for bandwagon (`0.976` versus `0.843`; 95% CI `[0.087, 0.180]`).

### RQ2 — Selective evaluation under bias

The primary 10% clean-calibrated MSP threshold retained only 0–4% coverage on
the biased test conditions. Zero-coverage cells have undefined realized risk;
the remaining cells have wide intervals and are inconclusive after correction.
The pilot therefore does **not** establish either failure or survival of the
clean selective-risk guarantee. The full dataset is required to estimate this
claim at useful coverage.

### RQ3 — Dose response

The fitted flip-probability slope was positive for both families:

- authority: `0.226` (95% CI `[0.162, 0.299]`);
- bandwagon: `0.017` per percentage point (95% CI `[0.012, 0.022]`).

Both fitted 25% flip thresholds fall below the tested dose ranges, so they are
reported as extrapolations rather than observed thresholds. Pre-first-flip
entropy showed an early-warning trend for bandwagon (`0.084`, 95% CI
`[0.018, 0.152]`, Holm-adjusted `p=0.014`) but not authority (`-0.050`, 95% CI
`[-0.101, 0.002]`).

## Interpretation and Limitations

- Results cover one 4B judge model and a stratified pilot, not full MT-Bench.
- The prescribed row-level routing split places 41 of 75 question clusters
  across both calibration and test, weakening an independent-transfer
  interpretation without changing the preregistered split.
- Authority's empirical dose profile is non-monotone despite a positive fitted
  slope; that slope should not be described as monotone behavioral evidence.
- Near-separation warnings occur in exploratory mixed-effects models.
- Full runs must validate verdict extraction independently for every additional
  model before any result is pooled or compared.

## Reproducibility

The command sequence is documented in `README.md`; decision rules are frozen in
`docs/experiment_plan.md`. Generated data and results remain below
`$BIASES_ARTIFACT_ROOT` and are not committed.

Pilot provenance:

- processed dataset SHA-256:
  `5983747255fdd73b4dd2375b80822629240e34778e5372d2cdfc4ec9278c0325`;
- Stage A score SHA-256:
  `27f35e5255dd4ca7dc77e5beb3d0240ff353acf3744cf16769209af164bd1418`;
- Stage B score SHA-256:
  `582f6e6ee0a9f5947b6a19fe0ac1f20444ade52981b72fb901eb84bd390b72fa`;
- analysis specification hash:
  `785e6d18b8d202d531acf5a1906fbfce3402c355e530e40aa86bea42c4aa7df5`.

All 104 repository tests pass. The five PDF figures, six LaTeX tables,
manifest, and claims-to-evidence digest regenerate byte-identically from the
same analysis inputs.
