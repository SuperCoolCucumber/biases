# Preliminary Silent Bias Pilot Report

## Status

The corrected four-model, 198-pair pilot is complete and suitable for deciding
whether to proceed to the full run. It is not a substitute for the preregistered
full-data results.

The post-analysis package gate passed with zero integrity errors. It verified
25,344 clean–cue pairs, all direct-input and output hashes, the complete
model-by-family-by-dose grids, and two byte-identical regenerations of the
paper assets. The package reports `primary_available=false` because 50
predeclared availability warnings remain in the small-sample RQ2 threshold
analysis; these are scientific non-estimability warnings, not integrity
failures.

## Design and validation

The pilot contains:

- Qwen3-4B, Qwen3-14B, OLMo3-7B-Instruct, and
  Hermes-3-Llama-3.1-8B;
- 198 MT-Bench answer pairs per model from 75 question clusters;
- both AB and BA orderings: 396 clean judgments and 6,336 cued judgments per
  model;
- authority and bandwagon cues in congruent and incongruent directions across
  four doses; and
- logit and verbalized-confidence passes at every dose, with the sanctioned
  reduced consistency schedule (`k=4` at clean and boundary doses).

Every model passed the fixed 20-example verdict gate under the `strict_v3`
processed-logprob contract. Cross-model artifact validation found the exact
expected Stage A and Stage B counts, no pairing errors, and at least 99%
verbalized-confidence availability for every model and stage.

Seventy-eight of 1,584 clean ordered judgments are flagged `clean_tie=true`
(Hermes 48, Qwen3-14B 4, Qwen3-4B 13, and OLMo3 13). They remain in the
paired artifacts and are reported separately; all primary RQ1–RQ3 estimates
use `clean_tie=false`.

RQ1 and RQ3 use only the inherited `routing_split=test` rows. RQ2 selects
thresholds on the inherited calibration rows and evaluates them on test rows.
Confidence intervals, including the primary question-clustered Gaussian GEE
trends, use 2,000 question-cluster bootstrap resamples. Secondary permutation
trend tests use 10,000 permutations. All stochastic analysis uses seed 42.

## Preliminary findings

### RQ1 — Silent bias is consistently visible without verdict flips

All 32 model-by-family-by-dose primary cells support positive movement of
probability mass toward an incongruent cue among examples whose verdict did
not flip. Mean shifts range from 0.0197 to 0.1131. This is strong pilot evidence
for the existence of silent bias across all four judges and both social-cue
families.

The stronger susceptibility claim is less universal. A low-dose shift beats
clean uncertainty alone for predicting a highest-dose flip in 3 of 8
model-by-family comparisons:

- Qwen3-14B bandwagon: ΔAUROC 0.192, 95% CI [0.139, 0.248];
- Qwen3-4B authority: ΔAUROC 0.226, 95% CI [0.144, 0.311]; and
- Qwen3-4B bandwagon: ΔAUROC 0.291, 95% CI [0.197, 0.393].

The other five comparisons are inconclusive. The full run is therefore needed
before claiming that paired shift is a generally superior susceptibility
detector rather than a model- and family-dependent one.

### RQ2 — The pilot is too small to stress-test selective risk reliably

At the preregistered 10% target risk, 11 of 16 primary high-dose,
single-ordering cells accept no test examples. Of the five cells with finite
risk, three meet the declared “clean guarantee survives” decision rule and two
are inconclusive; none shows statistically supported risk inflation. Test
coverage is only 0–2.02%, and the largest point estimate of the accepted
fraction of incongruent flips is zero.

This is not evidence that abstention is robust. It shows that the clean
threshold is usually too selective for a 198-pair pilot to estimate the
headline failure mode. The full dataset is required for a meaningful RQ2
result.

### RQ3 — Fitted flip probability rises with dose; early warning is limited

All eight model-by-family fitted logistic dose slopes are positive with
confidence intervals above zero. These are positive average associations, not
evidence that every empirical dose profile is monotone; authority profiles are
non-monotone for several models. The estimated 25%-flip thresholds lie below
the lowest tested doses, so those threshold values are extrapolations and
should not be interpreted as precisely located sensitivity points.

Only Hermes bandwagon supports a positive pre-first-flip entropy trend:
slope 0.0450, 95% CI [0.0171, 0.0712], Holm-adjusted p=0.00535. Early warning
is not broadly established in this pilot: six of the other comparisons are
inconclusive, and the Qwen3-4B authority trend runs downward under the
preregistered one-sided increase test. The full run is needed to distinguish
weak early-warning effects from confidence-preserving flips.

## Limitations

- This is a stratified pilot, not the 3,337-pair full run.
- The inherited row-level split places 41 of 75 pilot question clusters in
  both calibration and test. The preregistered split remains primary, but the
  full analysis should include the planned question-disjoint robustness check.
- RQ2 has extensive zero-coverage cells and must not be summarized as a
  negative result.
- The reduced consistency schedule supports engineering validation and the
  sanctioned budget plan; logit uncertainty remains the only channel measured
  at every dose.
- The run emitted two statsmodels covariance-square-root runtime warnings.
  Primary claims rely on the declared clustered intervals and adjusted tests;
  all 32 exploratory modeling rows completed with status `ok`.

## Reproducibility and provenance

The corrected package was generated at code commit
`1c1cab95c6936990797382387a68f76b950cc88b` with:

- pilot dataset SHA-256
  `d0e2dd12c5c6a2b378b12ab0ab363850147f1fa501fd13d25860737fc80d6b7a`;
- analysis version `silent-bias-p4-v6`;
- analysis specification SHA-256
  `fa600629136c0be66f9771df5c2b9366e2fb5b7b2d5bf5b77fdff8ffa40f4dda`;
- analysis-manifest SHA-256
  `bb4ae745514e61578f87264469b8a4284b3e2773956f3671989ff0705ad9240c`;
- generated digest SHA-256
  `734c443709317fcf79ae48245ac77812bfd5c9861b15e4e32ea03634ca67a29e`;
  and
- package-validation SHA-256
  `acda8d84030dba6bfdc79ab42dc2b6205505ef1c274b9a0d2e5e03d9cd46794f`.

Generated CSVs, figures, tables, and the claims-to-evidence digest are stored
outside Git under
`$BIASES_ARTIFACT_ROOT/outputs/analysis/corrected_pilot_1c1cab9/`.
The authoritative narrative is `paper_results.md`; its cells link to hashed
source CSVs. The full campaign should use the same validation and deterministic
asset-regeneration gates before any paper claim is promoted.
