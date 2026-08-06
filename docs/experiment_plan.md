# Silent Bias Experiment Plan

This document preregisters the primary analyses for the Silent Bias paper. It
must be updated before a full run, not after inspecting full-run outcomes.
Pilot-driven implementation fixes may change the code without changing these
decision rules. Any scientific change requires an explicit amendment here.

## Experimental Unit and Pairing

The source unit is one MT-Bench `(question, answer pair)` row with its existing
human winner and deterministic `routing_split`. Never recompute or replace that
split. Every row is judged in both answer orderings, `AB` and `BA`.

### MT-Bench turn normalization

MT-Bench conversation columns are decoded as structured data before prompt
construction. Canonical JSON is the stored format; the loader also accepts
legacy Python literal representations via safe literal parsing only. Scalar or
unparseable values remain plain text and are flagged through extraction
metadata.

The source `turn` column selects the evaluation target:

- Turn 1 compares the two first-turn assistant answers to the shared first user
  question.
- Turn 2 includes both shared user questions and both assistant turns for each
  candidate, and explicitly asks the judge to focus on the second-turn answer
  in its preceding conversational context.

The loader records the extraction mode and selected turn in example metadata.
Pilot validation must inspect golden turn-1 and turn-2 prompts before any full
run. This normalization fixes source-data interpretation and does not amend the
scientific conditions or decision rules below.

Stage A runs the clean condition at temperature zero for every
`(example, ordering, model)`. Stage B is generated deterministically from the
Stage A pair-summary artifact. For each ordering, Stage B uses the clean verdict
from the same model and ordering to assign the social cue:

- `congruent`: cue the clean-selected answer.
- `incongruent`: cue the answer not selected under clean.

Store direction relative to the human label separately as `toward_human`,
`against_human`, or `human_tie`. Preserve the source `routing_split` through
both stages and include the same `pair_key`, `condition_group_id`, ordering,
model, and clean-partner identifier in every paired row.

### Tie rules

Do not silently discard ties.

- If the clean judge verdict is `T` and the human winner is `A` or `B`, set
  `clean_tie=true`; use the human winner and its opposite as the two cue
  targets, and analyze this stratum separately from non-tie clean verdicts.
- If both the clean verdict and human label are ties, set `clean_tie=true`,
  assign `A` and `B` cue targets with a deterministic pair-hash rule, and set
  direction relative to the human label to `human_tie`. Report this stratum
  separately.
- For correctness, a judge tie is correct only when the human label is also a
  tie. This is the existing label-prior-control convention.
- Primary RQ1--RQ3 analyses use the non-tie-clean stratum. Tie strata remain in
  artifacts and appear in dedicated robustness rows and counts.

## Conditions and Generation Budget

Each example/model has exactly 34 prompt conditions:

- 2 clean conditions: one per ordering.
- 32 cued conditions: 2 families x 2 directions x 4 doses x 2 orderings.

Bandwagon doses are `55`, `70`, `85`, and `95`. Authority doses are ordinal
levels `1` through `4`. Cue wording is centralized in the prompt-builder
module. The condition identifier is
`{family}_{direction}_{dose}_{ordering}`; clean identifiers include the
ordering.

Every condition receives one temperature-zero constrained logit pass. The
default also runs `k=8` consistency samples at temperature `0.7` and one
verbalized-confidence pass. Before submitting a run, record its exact budget:

```bash
python3.12 scripts/estimate_run_budget.py \
  --examples 198 \
  --models 1 \
  --consistency-k 8 \
  --consistency-schedule all
```

The only sanctioned grid reduction is `--consistency-schedule extremes`.
Under that schedule, logit and verbalized passes still cover all 34 conditions,
while consistency covers the 2 clean conditions and the lowest and highest
dose for every family, direction, and ordering: 18 conditions total. Reducing
`k` from 8 to 4 is also permitted. Select and record any reduction in
`ExperimentSpec` before the full run.

### Constrained probability and emitted-token contract (`strict_v3`)

The A/B/tie probability channel is the distribution after vLLM applies the
registered first-token whitelist, not raw full-vocabulary top-k output. The
current accepted parser and inference contract is `strict_v3`. Every required
model must use exactly one literal token surface per verdict: `A`, `B`, and
`T` (tie). Leading-space or other alternate token surfaces are not part of the
contract. Constrained emission, label-probability aggregation, and verdict
resolution must use the same three resolved token IDs.

Runs must use `logprobs_mode=processed_logprobs`, require all three registered
IDs to be present, and fail closed if the emitted-token verdict disagrees with
the label-probability MAP. Decoded-token string matching is not an admissible
fallback. The mode is recorded in the experiment specification, raw records,
flat scores, pair summaries, stage summaries, and smoke artifact.
`ExperimentSpec` also binds the exact verdict token texts and resolved token
IDs so a resumed or analyzed run cannot silently change this contract.

Artifacts produced from raw full-vocabulary top-k logprobs cannot be repaired
from their stored three-label probabilities: missing allowed-token logits are
not recoverable. They must be retained only as invalidated audit artifacts and
rerun before analysis.

### Verbalized-output contract (`strict_v3`)

The verbalized-confidence channel has a parser version independent of the
constrained verdict parser. `strict_v3` retains all `strict_v2` forms:

- the existing verdict/confidence line forms, retaining their backward-
  compatible allowance for a trailing rationale that contains no additional
  verdict- or confidence-like atom;
- `Line 1: {A|B|T}` followed by
  `Line 2: [Confidence:] {0--100}`;
- `1. {A|B|T}` followed by `2. {0--100}`; or
- the single-line form `{A|B|T}, {0--100}`.

It additionally accepts exactly two complete enumerated forms observed in the
preserved Hermes full-data clean run:

- `1: {A|B|T}` followed by `2: {0--100}`; or
- `1) {A|B|T}` followed by `2) {0--100}`.

Each enumerated or comma form must match the whole response. Missing
scores, answer numbers in place of `A`/`B`/`T`, out-of-range values, prose
continuations after a new form, and otherwise ambiguous responses remain
unparseable.

The earlier `strict_v2` extension was fixed after the corrected Hermes pilot
failed closed at 364/396 parsed clean responses: 29 of the 32 rejected
responses were exact instances of the three new atomic forms. The three
remaining responses are unavailable because they use an answer number, omit
the confidence score, or append explanatory prose to an otherwise parseable
pair. Applying `strict_v2` therefore gives 393/396 availability (99.24%)
without changing the preregistered 99% gate.

The full-data Hermes clean run then failed closed at 6,592/6,674 parsed
responses (98.771%). A read-only audit found that 28 of the 82 rejected
responses use only the two additional complete enumerated forms above.
`strict_v3` projects 6,620/6,674 parsed responses (99.1909%), leaving 54
responses unparseable. This parser-only recovery was approved after the
clean-stage failure and before any Hermes cued-condition inference. It does
not lower the 99% availability gate or add a prose/ambiguity fallback. All
preserved full-run artifacts must be rematerialized from their original raw
outputs under one parser version before cross-model analysis.

Every raw record, flat score, pair summary, and stage summary records
`verbalized_output_parser_version`. Resume and artifact validation reject a
missing or stale value. Stored raw verbalized text may be rematerialized with
the migration CLI; migration must preserve record IDs, pairing keys, input and
spec hashes, and the original raw outputs.

### Historical strict-v3 model matrix amendment (2026-07-30)

This section records the model-selection and gating decisions for the completed
strict-v3 evidence package. It is historical provenance, not the model matrix
for the active controlled uncertainty-shift campaign. References below to the
"full run," "required models," or pending strict-v3 gates apply only to that
historical campaign and are retained so its results remain interpretable.

The active controlled-shift replication instead evaluates exactly two current,
higher-capacity judges: `Qwen/Qwen2.5-32B-Instruct` and
`meta-llama/Llama-3.3-70B-Instruct`, each at the immutable revision specified in
`docs/controlled_uncertainty_shift_design.md`. That controlled design takes
precedence over this historical section for all new inference, calibration,
analysis, and reporting. Qwen3, OLMo3, and Hermes artifacts remain preserved
strict-v3 evidence; they are not active controlled-shift research objects and
must not be silently pooled with or substituted for the 32B/70B models.

The historical gated full-run minimum was four judges: Qwen3-4B, Qwen3-14B,
OLMo3-7B-Instruct (`allenai/Olmo-3-7B-Instruct`), and
Hermes3-Llama3.1-8B (`NousResearch/Hermes-3-Llama-3.1-8B`). This gives three
architecture families (Qwen3, OLMo3, and Llama 3.1). Every checkpoint is
pinned to an immutable model revision and must pass its own 20-example native
and constrained verdict-extraction gates. Skywork-Critic-Llama-3.1-8B is
optional and is included only if it passes those same gates; otherwise its
failed smoke remains an explicit exclusion artifact.
The historical campaign context limit was 4,096 tokens. Before its inference,
every model had to pass a full-grid prompt-length preflight that included
generation headroom; the persisted report is an operational gate, not merely a
diagnostic.

Mistral-7B-Instruct-v0.3 was a stretch model. Its pinned
`MistralCommonTokenizer` warns that string rendering with
`apply_chat_template(..., tokenize=False)` is unsafe, while the current runner
transports auditable string prompts to vLLM. Mistral was therefore excluded
until a token-ID prompt adapter preserves canonical chat-template IDs and
includes those IDs in prompt hashing and provenance. This is an implementation
integrity exclusion, not a result-driven model substitution.

OLMo2-7B-Instruct passed constrained extraction and full-grid prompt transport
but failed the preregistered native verdict contract (18/20 examples versus
the 99% minimum). It is retained as an exclusion artifact and is not replaced
by relaxing the gate. Phi-4-14B then passed constrained extraction on 20/20
examples but failed the same native verdict-token-and-agreement contract on
14 examples: only 6/20 passed. It is also retained as an exclusion artifact,
with the 99% gate unchanged.

OLMo3-7B-Instruct is the public third-family replacement, pinned to revision
`6e5971d9eba42665f5bd5a0fcf047f299ce1dccc`. Its tokenizer-only full-grid
preflight passed all 113,458 prompts, with a maximum prompt length of 3,450
tokens plus 24 generation tokens, zero string-transport mismatches, and zero
context overflows. Its first GPU smoke aborted before writing result records:
the earlier multi-surface whitelist allowed the emitted-token verdict to
diverge from the MAP after alternate token surfaces were aggregated by label.
This is a successful fail-closed detection, not an admissible result.

The four historical models were required to use the literal `A`/`B`/`T`
contract uniformly and rerun their 20-example constrained and native smokes
under `strict_v3`. Earlier smoke passes did not carry forward. The
preregistered 99% native gate was unchanged, and no pilot was to start until
all four models passed it.

The `strict_v3` reruns completed on 2026-07-30. Each of Qwen3-4B, Qwen3-14B,
OLMo3-7B-Instruct, and Hermes3-Llama3.1-8B passed 20/20 constrained examples
with valid three-label probabilities and MAP-aligned verdicts, and 20/20
native examples satisfied the first-token-and-verdict-agreement contract.
All artifacts declare `processed_logprobs`, `strict_v3`, and the literal
`A`/`B`/`T` token mapping. This clears only the model smoke prerequisite; the
198-pair pilot, artifact validation, analyses, and all RQ findings were still
pending at the time of this amendment. The per-model artifact paths and
SHA-256 ledger are recorded in
`docs/codex_handoff.md`.

## Splits and Analysis Population

- `calibration`: select confidence thresholds only.
- `test`: estimate all reported headline performance.
- Never tune a threshold, dose transformation, binning scheme, or metric
  direction on the test split.
- Cluster all bootstrap resampling by source question so multiple turns and
  both orderings move together.
- Preserve the existing routing split even if turns from one source question
  occur in different row-level splits; do not create a replacement split.

The 198-row stratified file is the mandatory end-to-end pilot. Full MT-Bench
runs begin only after all 198 rows, including tie strata, pass schema,
pairing, prompt, and verdict-extraction checks.

## RQ1: Silent Bias

### Primary metric

For each non-flipped incongruent pair, compute signed movement toward the cued
label:

`p_cued(biased) - p_cued(clean)`.

Report its mean and 95% question-cluster bootstrap interval by model, family,
and dose. The preregistered existence rule for silent bias is:

1. the Holm-adjusted one-sided test rejects a mean of zero at `alpha=0.05`;
2. the 95% interval is strictly above zero; and
3. the effect is positive at two or more adjacent doses, including at least
   one submaximal dose.

Also report paired changes in entropy, MSP, margin, verbalized confidence, and
consistency entropy, plus Jensen--Shannon divergence and flip/error indicators.
These are secondary unless explicitly identified below.

### Susceptibility prediction

The primary predictive comparison is AUROC of the lowest-dose signed mass
shift for predicting an incongruent flip at the highest dose. Compare it with
clean-condition entropy alone using paired question-cluster bootstrap
differences. Claim label-free susceptibility information beyond clean
uncertainty only when the AUROC difference has a strictly positive 95%
question-cluster bootstrap interval. The comparison is undefined, and the
claim remains unavailable, when either highest-dose outcome class is absent.

## RQ2: Selective Evaluation Under Bias

Use MSP as primary confidence. Consistency agreement and verbalized confidence
are secondary. On the clean calibration split, choose the threshold with
maximum coverage whose empirical risk is no greater than each target:

- primary target risk: 10%;
- confirmatory target risk: 20%.

Each confidence channel is scored against the verdict produced by the same
inference pass: MSP against the constrained deterministic verdict, consistency
agreement against the consistency majority verdict, and verbalized confidence
against the free verbalized-pass verdict. A secondary-channel observation with
an unparseable or missing same-pass verdict is unavailable for that channel and
is never silently scored against the deterministic verdict. Secondary-channel
accepted-flip fractions likewise compare the matching clean and cued
same-pass verdicts; the primary MSP flip definition is unchanged.

The pilot gate requires complete constrained-logit and scheduled-consistency
channels. Verbalized verdict/confidence parsing must succeed on at least 99% of
pilot records per model; failures are retained as explicit missing secondary
observations and their rate is reported by condition. A model below 99% must
have its verbalized prompt or parser contract corrected and rerun on the pilot
before full inference.

Resolve threshold ties by choosing the stricter threshold. Freeze each
model/ordering threshold, then evaluate clean and every cued condition on the
test split.

### Primary metrics and decision rule

The headline metrics are:

1. realized-risk inflation:
   `risk(cued, test, tau_clean) - target_risk`;
2. accepted confident flip rate: the fraction of incongruent flips whose MSP
   meets `tau_clean`.

The primary condition family is the highest incongruent dose, reported
separately by model and bias family at the 10% target. The 20% target and other
doses are confirmatory dose profiles.

Conclude that the clean selective guarantee fails in a primary cell when the
95% question-cluster bootstrap interval for realized-risk inflation is
strictly above zero after Holm correction. Conclude that it survives in a
primary cell only when the upper interval endpoint is no greater than zero;
otherwise report the cell as inconclusive. Always report coverage and accepted
flip rate, regardless of direction.

If the transferred threshold accepts no test examples, realized risk and risk
inflation are undefined: report zero coverage and do not classify the cell as
failure or survival, even if some recalibrated bootstrap draws have nonzero
coverage. Alternate confidence channels remain secondary and are excluded from
the MSP primary Holm family.

Compute ECE, reliability diagrams, risk--coverage curves, and AURC for every
available confidence channel and condition. Compute multiclass Brier score only
for MSP, whose constrained-logit pass supplies the required A/B/tie
probability vector; do not attach that vector to consistency or verbalized
predictions. Use deterministic confidence bins declared in the analysis
config. The standard mitigation baseline is swap averaging over AB/BA with the
same clean-calibrated abstention rule and tie policy.

## RQ3: Dose--Response

For each model and family, fit the preregistered psychometric model on
incongruent, non-tie-clean pairs:

`logit(P(flip)) = intercept + slope * dose`.

Use raw bandwagon percentage and authority ordinal dose for family-specific
fits. Report the slope and the dose at 25% predicted flip probability, with
95% question-cluster bootstrap intervals. A positive dose response requires a
strictly positive slope interval and Holm-adjusted `p < 0.05`.

For the cross-family model, normalize each four-level dose ladder to
`0, 1/3, 2/3, 1` and fit, separately per judge model, the exact mixed-effects
formula:

`flip ~ dose * family * congruence + (1 | question)`.

Document the statsmodels estimator, optimizer, convergence diagnostics, and
fallback behavior in the generated analysis metadata.

The early-warning test uses entropy among examples that have not yet flipped
at a given dose. Fit a question-clustered GEE trend and report dose-wise means
with bootstrap intervals. Report a question-cluster bootstrap interval for the
primary GEE slope. Treat a strictly positive slope interval together with a
Holm-significant positive entropy trend before the first flip as early warning.
A positive flip-dose slope
without a positive pre-flip entropy trend is evidence that flips occur with
confidence intact.

## Statistical Hygiene

- Use 2,000 question-cluster bootstrap resamples for every reported confidence
  interval.
- Set the bootstrap seed by CLI; the preregistered default is `42`, and record
  it in analysis metadata.
- Use exact McNemar tests for paired correctness/flip comparisons.
- Apply Holm--Bonferroni separately within each RQ's preregistered primary
  metric family. Label all other tests exploratory.
- Report effect estimates and intervals even when adjusted tests are not
  significant.
- Every tidy output row must carry a spec hash and hashes of all direct input
  files. Paper tables and figures must retain source artifact identifiers.

## Artifacts and Full-Run Gate

Write generated datasets, run outputs, analyses, figures, and tables below
`$BIASES_ARTIFACT_ROOT`; do not commit them. Tidy analysis CSVs live under
`$BIASES_ARTIFACT_ROOT/outputs/analysis/`.

Before a full run:

1. Require a passing 20-example verdict-extraction artifact per model. Every
   constrained raw output must reparse to its returned deterministic MAP
   verdict with valid `A`/`B`/`tie` probability support computed from complete
   post-whitelist (`processed_logprobs`) token coverage. On the exact same
   prompts, the unconstrained greedy native contract rate must be at least 99%:
   the output is parseable, its first generated token is a declared verdict
   token, and its verdict agrees with the constrained verdict.
2. Complete both stages on all 198 pilot rows in both orderings.
3. Require every raw and flat pilot row to carry the current judge-output
   parser version. An explicit migration from an older parser must reparse the
   stored raw outputs before stamping the new version.
4. Verify every cued record has exactly one clean partner and ordering twin.
5. Recompute logit, verbalized, and consistency metrics from their primitive
   raw fields and reject semantic mismatches.
6. Verify the estimator count matches the submitted run configuration.
7. Regenerate paper assets twice and require byte-identical outputs.
8. Record the Git commit, environment versions, model revisions, dataset
   hashes, spec hash, parser version, and chosen consistency schedule.
