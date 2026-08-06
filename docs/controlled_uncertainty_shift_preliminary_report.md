# Preliminary report: controlled uncertainty shift in pairwise LLM judging

## Purpose

This note records the evidence available before the new Qwen2.5-32B and
Llama-3.3-70B controlled-shift campaign. It is a pilot report, not the final
paper result. The new campaign has not yet been released for full inference,
and no result below should be interpreted as a comparison between the two new
models.

## Evidence retained from strict-v3

The completed strict-v3 study established that authority and bandwagon cues can
move a judge's probability distribution toward the cued answer without changing
its discrete verdict. Its clean-calibrated maximum-softmax-probability (MSP)
selective-prediction result is preserved unchanged:

- 16 primary RQ2 transfer cells;
- 10 cells with zero coverage;
- 141 accepted examples across the six nonzero-coverage cells; and
- 8 errors among those accepted examples.

Zero coverage is a substantive abstention outcome. It means that the frozen
clean rule accepted no examples in that biased test cell; realized selective
risk and risk inflation are therefore undefined, not zero.

Four-draw categorical repeatability predictors were evaluated separately from
MSP. Their score support was too coarse to produce a finite, positive-coverage
10% empirical-risk rule. P(True), Mean Token Entropy, and Self-Certainty were
also evaluated separately in a model- and engine-specific exploratory package.
Those experiments did not establish a replacement for MSP. They remain useful
method-development evidence, but the older Hermes and OLMo checkpoints are not
being carried forward as the main research objects.

## Why a new campaign is needed

The earlier model suite was adequate for detecting the phenomenon, but it is a
weak basis for claims about contemporary high-capability pairwise judges. The
replication therefore narrows the primary model scope to:

- `Qwen/Qwen2.5-32B-Instruct`, preserving a direct comparison point with recent
  selective pairwise-judging work; and
- `meta-llama/Llama-3.3-70B-Instruct`, replacing the older Llama-family
  checkpoint with a newer 70B instruction model.

The scientific target is controlled uncertainty shift under matched biased
conditions. SCOPE and BPE are not implemented in this campaign. MSP and each
within-order repeatability predictor are calibrated and evaluated separately;
no composite predictor or post-hoc predictor selection is permitted.

## Frozen design and denominators

The source has 3,355 raw MT-Bench rows. The explicit pair-eligibility contract
retains 3,337 rows and reports 18 rows with a missing extracted response. A
seeded, question-level 50/50 routing assigns all 80 questions to 40 calibration
and 40 test questions without overlap. All 18 skipped rows belong to test
question 127, but 26 other rows for that same question remain eligible. The
eligible analysis population therefore still contains 40 calibration and 40
test questions:

- 1,634 eligible calibration pairs;
- 1,703 eligible test pairs;
- 6,674 clean Stage-A records; and
- 54,496 biased Stage-B records per model.

The Stage-B number is the unchanged full generated workload, not necessarily
the primary target-bias denominator. After Stage A, only rows labeled
`reference_kind=model_clean_verdict` enter the primary analysis. Clean-tie rows
using `human_label_fallback` or `deterministic_fallback` remain in the grid as a
separately reported robustness estimand. Their model-specific count—and thus
the exact primary denominator—is not known before Stage A. The analyzer rejects
any fallback-referenced row that could enter the primary cohort.

The frozen repeatability schedule uses four stochastic draws for every clean
condition and for the minimum and maximum cue dose. Intermediate cue doses are
deterministic only. This produces 196,858 generated sequences per model. Stage
B is released only after the exact clean verdicts authorize and hash its
model-specific matched prompt grid.

## Completed feasibility evidence

Authenticated access to the pinned Llama-3.3-70B revision has been verified.
Its tokenizer satisfies the literal one-token `A`/`B`/`T` verdict contract and
the repository's text/chat-template transport check. A four-way tensor-parallel
BF16 load and constrained-verdict extraction smoke completed successfully,
including a 20-prompt batch, with no parser or probability-contract error.

That smoke proves access, model loading, and verdict extraction. It does not
prove full-campaign throughput or authorize scientific inference. The release
gate still requires the exact frozen routing package, a runtime-bound full-grid
preflight, and two fresh production-equivalent longest-prompt smokes for each
model. The required Stage-B throughput target is 1.261 end-to-end generated
sequences per second, including prompt prefill; a result below 0.946 sequences
per second cannot finish within the 48-hour allocation.

## Current interpretation

The strongest result so far is negative but informative: ordinary clean
confidence can respond to biased distribution shift by abstaining almost
everywhere, and four categorical repeats do not supply enough resolution to
repair that behavior. The next experiment asks a sharper and more relevant
question: whether this collapse, and the underlying score shift, persists for
current 32B and 70B judges when calibration and biased evaluation are
question-disjoint, matched pair by pair, and executed under one exact runtime
contract.

No answer to that question is reported yet. Full inference remains gated on
the reproducibility and throughput checks above.
