# Lowest-dose verdict reversals and confidence shifts

This is an exploratory reanalysis of existing inference. Definitions are fixed
before inspecting the newly requested subgroup results. It does not replace
the original RQ1–RQ3 analysis or require new GPU inference.

## Questions

1. How frequently do the lowest tested authority and bandwagon cues reverse a
   judge's preference, and how does this vary with initial confidence and
   agreement with human judgments?
2. Among observed reversals, how large are the changes in maximum confidence
   and probabilities assigned to the human-preferred and cued answers?
3. On the same answer pairs, how do human-congruent and human-incongruent cues
   differ, within strata of initial agreement with the human label?

## Population and comparisons

- Analyze Qwen2.5-32B-Instruct and Llama-3.3-70B-Instruct separately.
- Use held-out test questions only. Calibration rows do not contribute to
  effect estimates or selection of confidence bins.
- Authority uses dose 1, endorsement by another user. Bandwagon uses dose 55,
  a claimed 55% majority. These are the lowest tested levels within each
  family, not a common quantitative scale of influence.
- Pair every cued record to its exact clean record for the same model, source
  row, and presentation order.
- Derive human congruence from cue target and the order-specific human label,
  rather than interpreting model-relative `congruent` as human agreement.
- Human-tie rows and clean-model-tie rows remain explicitly labeled companion
  strata. Human-tie rows have no unique human-congruent cue.
- Report original and swapped presentation orders separately, and an equally
  weighted pooled summary of their repeated measurements. No observations
  become independent merely because their order or cue changes.

## Outcomes

The main reversal is A→B or B→A. Companion outcomes are stable judgments,
A/B→tie, and tie→A/B. Retain exact clean-to-cued labels as well as agreement
transitions: correction, new disagreement, continued disagreement, and
continued agreement. A reversal toward the cue must end at the cued answer;
changes away from the cue are retained and labeled separately.

Report rates using all eligible rows in the corresponding subgroup as the
denominator. Conditional confidence summaries among reversals describe that
selected population; they do not estimate a causal effect of being a reversal.

## Confidence and probability measures

For each paired observation calculate cued minus clean:

- Maximum-label-probability change, in percentage points.
- Human-winner probability change, in percentage points, for binary human
  labels only.
- Cue-target probability change, in percentage points.
- Shannon label-entropy change, in bits, as a secondary uncertainty measure.

Preserve both clean and cued levels. Keep parsed verdicts and probability-MAP
agreement diagnostics so an extraction discrepancy cannot silently become a
behavioral reversal. Never treat maximum label probability as a calibrated
probability of agreement with humans.

## Paired human-congruence contrasts

Within each model, family, ordering, source pair, and clean state, subtract the
human-incongruent outcome from the human-congruent outcome. Include stable
judgments as well as reversals. Summarize clean-agrees and clean-disagrees
strata separately, with clean ties as a companion stratum. A pooled comparison
of correction-only and newly-wrong-only subsets does not isolate human
congruence, because those subsets start with different clean judgments.

## Summaries and uncertainty

- Report sample size, contributing source pairs, and contributing questions.
- For shifts, report the mean and a pointwise 95% question-cluster bootstrap
  interval, plus median and interquartile range. Report the fraction with
  increased maximum confidence among each change category.
- Use 2,000 bootstrap draws, seed 20260912. Resample the same full set of test
  question identifiers for every cell, retaining all associated rows and
  repeated conditions. Recompute subgroup ratios within each draw. Report
  non-estimable resamples rather than converting missing risk or means to zero.
- Primary descriptive estimates weight source rows equally within each
  condition. An equally weighted question sensitivity estimate checks whether
  questions with more comparisons dominate the result.
- Explore susceptibility using fixed clean-MSP bins with edges 1/3, 0.5, 0.7,
  0.9, 0.99, and 1.0. These are descriptive bins, not fitted routing thresholds.
- The intervals are pointwise exploratory intervals, not simultaneous
  familywise guarantees. Do not infer equivalence from a nonsignificant change.
- Keep magnitude and uncertainty central; do not select metrics or strata
  according to which produce small p-values.

## Outputs and provenance

Produce an auditable per-condition table, transition-rate table, conditional
shift summaries, paired human-congruence contrasts, confidence-bin summaries,
and figures showing transitions and confidence distributions. Hash inputs,
the analysis specification, code, and produced outputs. Validate expected
model identities, the source dataset and question-routing assignment, exact
clean linkage, labels, and minimum-dose grid before creating result claims.

Execution requires the archived Stage A and Stage B score files for both
models, including A/B/tie probabilities. A narrative summary cannot substitute
for those inputs. Raw run records are additionally needed for analyses of
individual stochastic draws that cannot be recovered from the score files.
