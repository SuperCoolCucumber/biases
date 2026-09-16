# Controlled Uncertainty-Shift Exploratory Results

## Evidence status — 2026-09-16

This document preserves the historical August two-model exploratory results.
The original detailed analysis package and full stochastic run records are not
included in the local handoff, so the numerical RQ1–RQ3 claims and historical
validation receipts below have not been reverified from that package. The four
archived score inputs and separate September lowest-dose analysis are available
and were verified independently; see the [current handoff](../docs/codex_handoff.md).
The historical analysis-specification and completion hashes below identify
the earlier evidence; they do not establish its current local availability.

## Executive summary

The experiment supports a focused version of the idea:

> Authority and bandwagon cues shift probability toward the cued answer even among judgments whose categorical label remains unchanged at a given dose.

The effect appeared in both Qwen2.5-32B-Instruct and Llama-3.3-70B-Instruct. It was most pronounced for bandwagon cues. The results support a framing based on **latent directional susceptibility to social cues**, rather than a general theory that entropy must rise monotonically before a judgment changes.

This is a positive exploratory result, but not yet evidence for a reliable abstention or error-detection system.

## Data and validation

- Models: Qwen2.5-32B-Instruct and Llama-3.3-70B-Instruct.
- Stage A per model: 6,674 records and 33,370 generated sequences.
- Stage B per model: 54,496 records and 163,488 generated sequences.
- Primary analysis pairing: 108,992 matched clean/cued records.
- These are repeated judgments on source pairs, not independent questions or experiments.
- Unmatched cued records: 0.
- Expected unused clean rows: 6,536.
- Historical validation reported that all input, provenance, result-manifest, and output hashes matched.
- Historical validation reported that all 16 primary mixed-model result rows converged, were finite, and had `status=ok`.
- Historical frozen analysis specification SHA-256: `fa600629136c0be66f9771df5c2b9366e2fb5b7b2d5bf5b77fdff8ffa40f4dda`.
- Historical primary completion SHA-256: `8550e4b6b543258c1474e143702be9a7f1fcde84fa3fc107e935fffe93c1c30a`.

The historical strict-v3 MSP analysis remains an unchanged external baseline and was not used as a regression input for this new two-model analysis.

## RQ1: Does uncertainty shift while the verdict remains unchanged?

Yes, when uncertainty is measured directionally.

Among examples whose categorical judgment remained unchanged, all 16 primary model × cue-family × dose conditions shifted probability mass toward the incongruent cue. The reported confidence intervals excluded zero, and the corresponding Holm-adjusted p-values were 0.007996.

Dose conditions are separately evaluated prompts; these results do not observe
an internal reasoning trajectory over time.

| Model | Authority cue-mass shift | Bandwagon cue-mass shift |
|---|---:|---:|
| Qwen2.5-32B | +1.24 to +3.62 percentage points | +3.07 to +3.76 points |
| Llama-3.3-70B | +0.45 to +1.22 points | +1.23 to +1.47 points |

Directional cue-mass shift also discriminated examples that flipped at a stronger tested dose:

| Model | Authority AUROC | Bandwagon AUROC |
|---|---:|---:|
| Qwen2.5-32B | 0.826 | 0.931 |
| Llama-3.3-70B | 0.879 | 0.981 |

For bandwagon cues, directional shift significantly outperformed baseline clean entropy:

- Qwen: AUROC 0.931 versus 0.816; difference 0.116, 95% CI 0.059–0.172.
- Llama: AUROC 0.981 versus 0.838; difference 0.143, 95% CI 0.118–0.166.

Authority cue-mass shift predicted flips well in absolute terms, but did not improve significantly over clean entropy.

At the strongest incongruent dose, conventional uncertainty also moved relative to the clean prompt:

| Model/family | Entropy change | MSP change | Top-two margin change |
|---|---:|---:|---:|
| Qwen authority | +0.126 | −0.036 | −0.071 |
| Qwen bandwagon | +0.136 | −0.037 | −0.072 |
| Llama authority | +0.038 | −0.012 | −0.024 |
| Llama bandwagon | +0.048 | −0.015 | −0.030 |

Thus, strong cues increase uncertainty relative to clean prompts, while directional cue-mass reveals where the probability is moving.

## RQ2: Do cues reduce correctness, and can confidence support abstention?

Cues reduced correctness, defined here as agreement with the dataset's human
preference label, but the tested confidence measures did not provide a dependable
abstention rule. Human agreement does not independently establish objective truth.

- Cued correctness was lower in all 32 primary model × family × dose × ordering cells.
- Twenty-five of the 32 comparisons remained significant after Holm correction.
- All eight strongest-dose comparisons were significant.
- Both answer orderings exhibited the effect; ordering changed magnitude but did not reverse the conclusion.

Calibration was poor at the strongest doses:

- Raw MSP ECE was approximately 0.388–0.455.
- Raw MSP Brier score was approximately 0.816–0.915.
- Consistency-agreement ECE was approximately 0.419–0.468.

Clean-to-cued threshold transfer was not practically useful:

| Target risk | Accepted examples | Coverage | Realized pooled risk |
|---|---:|---:|---:|
| 10% | 42 / 13,384 | 0.31% | 14.3% |
| 20% | 1,988 / 13,384 | 14.85% | 21.2% |

Only four of eight primary cells met the 10% risk target, and only three of eight met the 20% target.

The separate repeatability/entropy predictor audit for this two-model campaign
also reported the following values; it is distinct from the historical
four-model strict-v3 audit:

- MSP median AUROC: approximately 0.661.
- Within-order repeatability and categorical-entropy predictors: approximately 0.515, close to chance.
- Cross-order vote-agreement predictors: approximately 0.612.
- None of the alternative predictors yielded a finite transferred rule at the tested risk targets.

These channels must remain separate: MSP, consistency-majority confidence, within-order repeatability, categorical entropy, and cross-order vote agreement are not interchangeable predictors.

## RQ3: Does cue strength produce a dose response?

Yes for categorical flips.

All four primary model × family dose slopes were positive, with confidence intervals above zero and Holm-adjusted `p = 0.001999`.

Bandwagon flip rates increased monotonically:

- Qwen: 22.9% → 36.0%.
- Llama: 25.3% → 39.0%.

Authority effects rose sharply and then plateaued:

- Qwen: 11.2% → 39.8% → 37.1% → 38.4%.
- Llama: 7.1% → 20.6% → 27.1% → 26.1%.

The fitted full-range incongruent-dose odds ratios were approximately:

| Model | Authority | Bandwagon |
|---|---:|---:|
| Qwen2.5-32B | 3.35× | 1.91× |
| Llama-3.3-70B | 3.72× | 1.97× |

However, entropy did not increase monotonically with dose among examples that had not yet flipped:

- All four primary pre-first-flip entropy-trend tests were null after correction (`Holm p = 1.0`).
- Slopes were near zero or negative.
- Qwen's bandwagon entropy decreased with dose within the surviving pre-flip subset.

This does not contradict the clean-versus-strong-cue uncertainty increase. It means that stronger cues do not produce a simple progressive entropy escalation among cases that remain unflipped.

## Viability conclusion

The idea is viable as an exploratory scientific result:

1. External social cues shift probability toward the cued answer among judgments whose categorical label remains unchanged at a given dose.
2. The effect was observed in two large model families within this exploratory campaign.
3. Stronger cues generally cause more flips.
4. Bandwagon effects are especially clear and predictable.

The current results do **not** support:

1. A general monotonic-entropy account of pre-flip uncertainty.
2. A dependable uncertainty threshold for selective prediction or abstention.
3. A publication-ready claim without broader replication.

The most defensible next framing is:

> LLM judges exhibit latent directional susceptibility to social cues: probability mass moves toward the cued answer even when the categorical judgment remains unchanged at a given dose.

Signed cue-mass should be the primary signal. Entropy and MSP changes are useful secondary evidence.

## Limitations and recommended next step

- RQ1 conditions on examples that have not yet flipped, so the surviving population changes with dose.
- Primary analyses exclude clean ties.
- Authority responses plateau rather than increasing strictly at every dose.
- The large number of row-level observations derives from only 40 test-question clusters per primary group.
- The mixed model used variational-Bayes logistic fitting; extremely small coefficient p-values should be treated as supportive exploratory evidence.

For now, no additional large inference run is necessary. The most useful next step is to refine the scientific claim and decide whether the directional pre-flip shift is sufficiently interesting to justify broader model and question replication. If it is taken toward publication, inference and analysis should then be rerun under publication-grade reproducibility controls.
