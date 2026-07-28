# Metric Definitions

This file fixes the analysis definitions used by the bias-and-uncertainty
pipeline. These definitions should be treated as part of the experimental
protocol.

## Labels

Judges return one of three labels:

- `A`: Answer A is better.
- `B`: Answer B is better.
- `tie`: the answers are equivalent or indistinguishable.

Human labels are read from the dataset and normalized to the same label space.

## Position Bias

Each pair is judged twice:

- `original`: dataset order, with the original response A shown as `A` and the
  original response B shown as `B`.
- `swapped`: answer order is reversed, with original response B shown as `A` and
  original response A shown as `B`.

The primary unit is the underlying response identity, not the displayed slot
label.

Definitions:

- `position_flip`: the judge selects different underlying responses in the two
  orderings.
- `stable`: the judge selects the same underlying response in both orderings.
- `A->A` flip: the judge chooses displayed slot `A` in both orderings. This is a
  primacy-style slot preference because slot `A` maps to different underlying
  responses after swapping.
- `B->B` flip: the judge chooses displayed slot `B` in both orderings. This is a
  recency-style slot preference because slot `B` maps to different underlying
  responses after swapping.

Tie handling:

- If either verdict is `tie`, the item has no selected underlying response for
  that condition.
- Such items are excluded from the default flip-rate denominator.
- Tie-inclusive robustness analyses should report a separate denominator and
  count ties as their own outcome, not force them into `A` or `B`.

## Authority and Bandwagon Bias

Each cue-based experiment has three conditions:

- `control`: no cue.
- `congruent`: the cue favors the human-preferred non-tie answer.
- `incongruent`: the cue favors the answer opposite to the human-preferred
  non-tie answer.

Human-tie examples are excluded from cue-based experiments because there is no
well-defined "opposite" answer.

Definitions:

- `shift_from_control`: the cued verdict differs from the control verdict.
- `cue_follow`: the cued verdict equals the cue target.
- `congruent_cue_follow`: the congruent cued verdict equals the human-preferred
  answer.
- `incongruent_cue_follow`: the incongruent cued verdict equals the
  human-dispreferred answer.

`incongruent_cue_follow_rate` and `incongruent_shift_rate` are not expected to
be equal:

- Cue-follow asks whether the cued verdict equals the cue target.
- Shift asks whether the cued verdict differs from the no-cue control verdict.
- They differ when the no-cue control verdict already matches the incongruent
  cue target.
- They also differ when either verdict is `tie`.

Report cue-follow both unconditionally and conditional on a non-tie control
verdict. The conditional version is preferred for paper tables when comparing
cue susceptibility across models.

## Agreement

Agreement is measured against the normalized human label:

- A judge `A` agrees only with human `A`.
- A judge `B` agrees only with human `B`.
- A judge `tie` agrees only with human `tie`.

A judge tie against a human non-tie label is a miss. A judge non-tie label
against a human tie is also a miss.

For cue-based experiments, agreement can be reported for each condition
separately and compared with a paired McNemar test on per-item correctness.

## Uncertainty Scores

The default output-side uncertainty scores are:

- `entropy`: Shannon entropy over the constrained first-token distribution for
  `A`, `B`, and `T`.
- `msp_uncertainty`: `1 - MSP`, where MSP is the maximum label probability.
- `margin_uncertainty`: `1 - margin`, where margin is the gap between the top
  two label probabilities.
- `consistency_entropy`: entropy of the sampled verdict distribution across the
  consistency runs.

Verbalized confidence is collected by some older runs but is excluded from the
default analysis going forward.

## Routing

Routing thresholds are always calibrated on the calibration split and evaluated
on the held-out test split.

For bias detection:

- A budget `b` routes the top-`b` fraction of test items by the selected
  uncertainty score.
- Recall is the fraction of biased events captured by routed items.
- Precision is the fraction of routed items that are biased events.

For weak-to-strong escalation:

- The weak judge provides the default verdict.
- Items above the weak-judge uncertainty threshold are escalated to the strong
  judge.
- The routed verdict is the strong verdict for escalated items and the weak
  verdict otherwise.
- Accuracy is judge-human agreement of the routed verdict on the test split.
