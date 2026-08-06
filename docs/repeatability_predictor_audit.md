# Repeatability Predictor Audit

This audit is exploratory. It does not replace the preregistered strict-v3 MSP
analysis, and it does not combine predictors. The published MSP zero-coverage
outcome remains part of the primary result.

## Predictors

### Same-order deterministic-anchor reproducibility

For deterministic verdict `y` and `K` stochastic consistency verdicts in the
same displayed ordering,

```text
anchor_reproducibility = count(sampled_verdict == y) / K
```

`A`, `B`, and `tie` are distinct outcomes. This statistic compares stochastic
consistency samples with the temperature-zero deterministic anchor. It is not
agreement among reruns made under one identical decoding configuration, and it
does not use the modal sampled verdict as its anchor.

### LM-Polygraph within-order adaptations

The audit evaluates two methods adapted from the pinned LM-Polygraph
implementation. For the empirical same-order repeat distribution
`p = (p_A, p_B, p_tie)`, it reports each method separately:

```text
frequency_semantic_entropy_confidence = 1 - [-sum_c p_c log(p_c)] / log(3)
degree_matrix_agreement = sum_c p_c^2
```

The first is the confidence-oriented, three-class normalization of
frequency-based Semantic Entropy with `A`, `B`, and `tie` treated as exact
semantic classes. The second is the confidence complement of the exact-label
categorical specialization of Degree Matrix. Its uncertainty is
`1 - sum_c p_c^2`, the disagreement probability for two draws with replacement
from the empirical repeat distribution. These scores use only retained
same-order repeat counts and require no new model inference.

With exactly four repeats and at most three labels, both scores induce the same
four ordered count partitions: `(4,0,0)`, `(3,1,0)`, `(2,2,0)`, and `(2,1,1)`.
The audit therefore regression-checks that their rank metrics, threshold
feasibility, selected items, coverage, and risk agree exactly. Their numerical
scales—and hence uncalibrated ECE and Brier scores—remain different.

### Identity-aligned ordering-distribution similarity

For each semantic AB/BA twin, the empirical stochastic verdict counts are
normalized to three-class distributions. The BA distribution swaps `A` and
`B` before comparison; `tie` is unchanged. The audit measures three views
separately:

- one minus normalized Jensen-Shannon divergence;
- one minus total-variation distance;
- the probability that independent draws from the two canonicalized
  distributions agree.

The first two isolate distributional change between orderings. Independent-draw
agreement also penalizes diffuse repeat distributions even when AB and BA are
identical.

Pairs are scored only when model, response-pair identity, condition group,
family, direction, dose, split, canonical human label, canonical cue target,
and repeat count agree. A nominal AB/BA pair whose cue targets refer to
different underlying candidates is reported as unavailable rather than being
treated as ordering disagreement.

## Evaluation

Each predictor is evaluated independently against deterministic-verdict
correctness. Higher scores mean more confidence or repeatability. The audit
reports:

- clean-calibration availability, score resolution, ECE, binary correctness
  Brier score, correctness AUROC, and AURC;
- the tie-batched empirical thresholds at 10% and 20% clean risk;
- transfer to held-out clean data and highest-dose incongruent authority and
  bandwagon test cells;
- question-disjoint five-fold clean checks, row-wise question-disjoint primary
  transfer, and isotonic calibration as diagnostics only.

Equal scores always enter a threshold together. A rule that accepts no item has
zero coverage and undefined realized risk. Isotonic normalization is never
silently substituted for a raw predictor. Each primary row receives the
threshold trained without its assigned question fold. Thresholds use all clean
calibration rows, including deterministic clean ties; strict primary transfer
excludes rows marked `clean_tie=true`, matching the published MSP estimand.
AURC uses a right-continuous empirical step integral over complete equal-score
blocks, so coarse scores do not receive artificial credit from interpolation
through an undefined zero-coverage risk.

The trusted strict-v3 invocation supplies the published provenance and primary
MSP table as regression oracles:

```bash
python3.12 scripts/analyze_repeatability_predictors.py \
  --campaign-root "$BIASES_ARTIFACT_ROOT/outputs/<campaign>/full" \
  --published-provenance "$PUBLISHED_ANALYSIS_DIR/provenance.json" \
  --published-msp-oracle "$PUBLISHED_ANALYSIS_DIR/rq2_threshold_transfer.csv" \
  --output "$EXPLORATORY_OUTPUT_DIR/repeatability_predictor_audit.json"
```

The command refuses to write inside the immutable campaign root or overwrite
an existing report. When both oracles are supplied, it requires all raw hashes
and every published primary MSP cell to reproduce before emitting a result.

## LM-Polygraph scope

The review was pinned to
[`IINemo/lm-polygraph@98dd675`](https://github.com/IINemo/lm-polygraph/tree/98dd675cc43e0f5da654c29940872ea913aea2bf).

The two directly mapped methods evaluated above are:

- frequency-based
  [Semantic Entropy](https://github.com/IINemo/lm-polygraph/blob/98dd675cc43e0f5da654c29940872ea913aea2bf/src/lm_polygraph/estimators/semantic_entropy.py),
  treating literal `A`, `B`, and `tie` as known semantic classes;
- exact-label pairwise disagreement, the categorical specialization of
  [Degree Matrix](https://github.com/IINemo/lm-polygraph/blob/98dd675cc43e0f5da654c29940872ea913aea2bf/src/lm_polygraph/estimators/deg_mat.py),
  whose exact-label uncertainty becomes the probability that two empirical
  repeat draws disagree.

The task-native constrained-label MSP and selected repeatability/order
statistics remain separate predictors; they are not LM-Polygraph sequence
scores.

Exact sequence MSP, perplexity, token entropy, Monte Carlo sequence entropy,
SAR, hidden-state methods, and attention methods cannot be reconstructed from
the retained three-label probabilities and aggregate repeat counts. They need
token-level likelihoods, sampled texts, hidden states, attentions, or new model
calls. P(True) and two-stage verbalized confidence likewise require new
inference. This audit does not create a composite LM-Polygraph-inspired score.
