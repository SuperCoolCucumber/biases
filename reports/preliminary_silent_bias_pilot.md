# Preliminary Silent Bias Pilot Report

## Status

**Retracted pending rerun.** A preflight audit on 2026-07-29 found that the
processed MT-Bench conversation columns were serialized as Python literals,
while the loader only parsed JSON. The fallback treated each serialized
conversation as one text message. Consequently, the run did not reliably
separate the shared prompt from the candidate models' final assistant answers.
The numerical RQ1–RQ3 results from this pilot must not be used as scientific
evidence.

The run remains useful only as an engineering validation of the two-stage grid:

- 198 source answer pairs from 75 question clusters;
- 396 clean ordered judgments;
- 6,336 clean–cue paired comparisons;
- 20/20 parseable verdict-extraction smoke examples;
- no unmatched cued records, unused clean records, clean ties, or planning
  errors.

Consistency sampling used the sanctioned reduced schedule: `k=4` at clean and
boundary doses, with logit and verbalized-confidence passes at every dose.
The parser has been corrected with safe, backward-compatible literal parsing
and explicit MT-Bench turn selection. The regenerated 198-pair input has
SHA-256
`d0e2dd12c5c6a2b378b12ab0ab363850147f1fa501fd13d25860737fc80d6b7a`.
All smoke tests, the pilot, and its analyses will be rerun before the full-data
campaign begins.

## Invalidated Preliminary Findings

No RQ1, RQ2, or RQ3 numerical claim from the affected pilot is retained. The
previous values remain available in the immutable artifact files and Git
history solely for debugging and auditability.

## Interpretation and Limitations

- The affected run used malformed conversation interpretation and is invalid
  for behavioral conclusions.
- The replacement results will still cover one 4B judge model and a stratified
  pilot, not full MT-Bench.
- The prescribed row-level routing split places 41 of 75 question clusters
  across both calibration and test, weakening an independent-transfer
  interpretation without changing the preregistered split.
- In the full 3,355-row source file, all 80 question IDs occur in both routing
  splits. The primary analysis preserves that user-mandated split; a
  question-disjoint robustness analysis is required before interpreting RQ2
  as transfer to wholly unseen questions.
- Authority's empirical dose profile is non-monotone despite a positive fitted
  slope; that slope should not be described as monotone behavioral evidence.
- Near-separation warnings occur in exploratory mixed-effects models.
- Full runs must validate verdict extraction independently for every additional
  model before any result is pooled or compared.

## Reproducibility

The command sequence is documented in `README.md`; decision rules are frozen in
`docs/experiment_plan.md`. Generated data and results remain below
`$BIASES_ARTIFACT_ROOT` and are not committed.

Invalidated-run provenance (retained for audit only):

- processed dataset SHA-256:
  `5983747255fdd73b4dd2375b80822629240e34778e5372d2cdfc4ec9278c0325`;
- Stage A score SHA-256:
  `27f35e5255dd4ca7dc77e5beb3d0240ff353acf3744cf16769209af164bd1418`;
- Stage B score SHA-256:
  `582f6e6ee0a9f5947b6a19fe0ac1f20444ade52981b72fb901eb84bd390b72fa`;
- analysis specification hash:
  `785e6d18b8d202d531acf5a1906fbfce3402c355e530e40aa86bea42c4aa7df5`.

The replacement report will record new dataset, Stage A, Stage B, analysis, and
code hashes after the corrected pilot passes end to end.
