# Controlled uncertainty shift under biased conditions

## Status and scope

This is a new, question-disjoint replication. It does not modify or replace the
reported strict-v3 MSP result. In particular, the strict-v3 result with 16
primary RQ2 cells, 10 zero-coverage cells, 141 accepted examples, and 8 errors
remains an immutable reported outcome.

The replication studies whether a confidence rule calibrated on clean
pairwise judgments transfers to the *same held-out pairs* after an authority or
bandwagon cue is introduced. SCOPE and BPE are explicitly out of scope for this
implementation.

## Models

- `qwen2.5-32b` uses `Qwen/Qwen2.5-32B-Instruct` at revision
  `5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd`.
- `llama3.3-70b-instruct` uses `meta-llama/Llama-3.3-70B-Instruct` at revision
  `6f6073b423013f6a7d4d9f39144961bfbfbc386b`.

Qwen2.5-32B preserves the direct SCOPE comparison point. Llama 3.3 deliberately
updates SCOPE's Llama 3.1 baseline to the newer 70B instruction checkpoint while
keeping the replication relevant to current pairwise-judge use without adopting
the SCOPE procedure. The Llama checkpoint is access-gated. Authenticated
config/tokenizer access at the pinned revision has been verified without
recording a credential in the experiment contract. Both tokenizers must pass
the frozen literal `A`/`B`/`T` one-token preflight, and each model must still
pass its load and verdict-extraction smoke, before an inference manifest can be
released.

The portable Llama renderer starts at tensor parallelism 4 with eager execution,
but this is a resource-planning template rather than a load-validated promise.
Its BF16 weights alone are about 131 GiB; 40 GiB accelerators therefore require a
conservative context/scheduler smoke or a wider tensor-parallel topology, while
four 80 GiB accelerators have substantially more headroom. Freeze the settings
that actually pass the preflight and pin their runtime digest before inference.

The two models are reported separately. Their scores, thresholds, or outcomes
are never pooled.

The frozen source contains 3,355 raw rows. The pair-loader eligibility
contract retains 3,337 rows and reports, rather than silently discards, the 18
rows whose extracted A or B response is empty. All workload denominators and
expected record counts below use the 3,337 eligible pairs; raw and skipped
counts remain first-class provenance.

## Question-disjoint routing

Questions, rather than rows, are deterministically assigned to calibration or
test. Every turn and pairwise row for a question receives the same assignment.
The assignment uses a seeded SHA-256 rank, preserves source row order, and emits
a canonical question-to-split digest.

The intended 50/50 split of the 80 MT-Bench questions is 40 calibration
questions and 40 test questions. Winner labels are not used to construct the
split because row-level winner stratification would violate question
disjointness. Realized winner, row, and turn distributions must instead be
reported as diagnostics.

Stage A evaluates clean AB and BA judgments for all rows. Stage B generates
authority and bandwagon conditions only for rows routed to `test`:

```text
calibration questions: clean Stage A -> threshold fitting only
test questions:        clean Stage A -> clean baseline evaluation
                                      -> matched biased Stage B evaluation
```

The frozen Stage-B workload retains every condition, including clean-tie rows.
Each planned condition and derived analysis row carries an exact
`reference_kind`: `model_clean_verdict` for a binary Stage-A verdict,
`human_label_fallback` when Stage A tied but the human label is binary, or
`deterministic_fallback` when both labels tied. Only `model_clean_verdict` is
eligible for the primary target-bias estimand. The two fallback kinds preserve
the full grid as a separately labeled clean-tie robustness estimand.

## Estimand

For each model, ordering, predictor, and target clean risk, fit one empirical
selective threshold using only clean calibration questions. Freeze that rule.
On the test questions, compare clean and cued judgments on their exact matched
pair intersection for every bias family, direction, and dose.

Report at least:

- clean and biased availability;
- matched sample and question counts;
- full-grid counts by `reference_kind`, with primary and fallback-robustness
  denominators named separately;
- clean and biased coverage;
- clean and biased selective risk;
- biased-minus-clean coverage change;
- biased-minus-clean risk change when both risks are defined;
- accepted-to-rejected and rejected-to-accepted transitions;
- clean-correct to biased-wrong transitions;
- predictor-score shift; and
- question-cluster bootstrap intervals for paired changes.

The calibration population includes clean ties, matching the prior MSP
threshold-fitting estimand. The primary held-out transfer population requires
`reference_kind=model_clean_verdict`; fallback-referenced clean ties remain in
the frozen full grid and are reported only as robustness rows/counts. The
analyzer validates the complete grid before this split and fails if a fallback
reference could enter the primary cohort. Bootstrap replicates resample
calibration questions and refit the threshold; the same calibration-rule
schedule is reused across every cue cell for a model, ordering, predictor, and
target risk.

For predictors with missing scores, report score availability and coverage both
among jointly scored pairs and relative to the full structural matched cohort.
The two denominators must remain explicitly named.

Zero accepted examples are a valid outcome. Coverage is then zero and selective
risk and risk change are undefined; they must not be converted to zero or
dropped.

## Predictor policy

MSP is the primary frozen comparator. Within-order repeatability and its
categorical entropy are supplemental predictors generated by this campaign.
Each predictor is calibrated, evaluated, and reported separately. No ensemble,
combined predictor, or post-hoc selection among predictors is part of this
design. Verbalized confidence is disabled: it is not one of the selected
predictors and enabling it would add a different prompt/output contract.

Mean token entropy, P(True), and Self-Certainty remain separately reported
LM-Polygraph analyses. They enter this controlled-shift analysis only if a
model-matched, condition-matched inference package passes its own immutable
gate; values from another model or inference engine are never substituted.

Predictors produced by a different inference engine require explicitly labeled
estimands. Cross-engine agreement subsets are sensitivity analyses and must
carry a selection-bias caveat.

## Validation gates

Before inference:

1. Freeze the routed CSV, routing manifest, source hash, model revisions,
   verdict-token contract, prompts, doses, and runtime settings.
2. Require zero question overlap and both routing splits to be present.
3. Run the verdict-extraction, tokenizer/context, and small-grid model
   preflights for each model.
4. Review the rendered Slurm resources for the active infrastructure.
5. Benchmark the production-equivalent deterministic plus four-seed
   repeatability schedule on a deterministic longest-per-stratum prompt set.
   A weight-load or one-row extraction check is necessary but not sufficient.

The frozen schedule is four repeatability draws for every clean condition and
for the minimum/maximum dose of each cued family, with the intermediate cued
doses evaluated deterministically. With 1,634 eligible calibration pairs and
1,703 eligible test pairs, one model therefore has:

- 6,674 Stage-A records and 33,370 generated sequences;
- 54,496 Stage-B records and 163,488 generated sequences; and
- 61,170 logical records / 196,858 generated sequences in total.

The 54,496 Stage-B count is the unchanged full-grid workload. The exact primary
denominator can be smaller because model-clean ties remain as robustness rows;
that model-dependent count is known only after Stage A.

A 48-hour Stage-B allocation needs at least 0.946 end-to-end generated
sequences per second; the release target is 1.261 sequences per second, which
leaves a 25% wall-time margin. Measure this using prompt-prefill-inclusive wall
time, not output-token speed. If the representative gate misses the minimum,
the unsharded Stage-B campaign is not released.

After inference:

1. Validate Stage A against the complete source grid.
2. Validate Stage B against the independently pinned `test` routing scope,
   rather than deriving the expected grid from the artifact declaration.
3. Reconfirm the full question-assignment digest from the immutable routing
   manifest; require valid, exactly equal Stage A/B inference-runtime mappings;
   and match both mappings to the runtime SHA-256 pinned before inference.
4. Require exact clean-record linkage for every biased record.
5. Require one exact `reference_kind` per biased record, permit fallback kinds
   only on clean-tie strata, and reject fallback rows from the primary cohort.
6. Require the complete authority/bandwagon, direction, and dose grid on the
   same structural test cohort, including labeled robustness rows.
7. Run analysis into a new empty output directory and preserve all input and
   output hashes.

The inference-free Stage-B grid is structural and uses provisional A/B target
realizations only. It cannot authorize Stage B. After Stage A completes, build
and hash the exact Stage-B plan from the model's clean verdicts, re-render its
actual prompts, recheck context bounds, and release Stage B only from that
post-Stage-A gate.

Fresh scientific launchers must fail when their stage output directory already
exists and must run with resume disabled. Any recovery uses a new immutable run
identifier and preserves the failed directory and logs.

## Portable command outline

Run this sequence independently for each model. `MODEL_RENDERER_KEY` is
`qwen25_32b` or `llama33_70b_instruct`; `MODEL_REGISTRY_NAME` is the
corresponding repository registry name. Before rendering, freeze a separate
model-specific `RUNTIME.json` containing the exact pinned model revision,
engine versions, verdict contract, repeatability schedule, and inference
settings. The routing and runtime manifests are inputs to every subsequent
gate; they are never reconstructed from generated records.

```bash
python3.12 scripts/prepare_frozen_question_routing.py \
  --source-csv FROZEN_SOURCE.csv \
  --output-dir ROUTING_PACKAGE \
  --dataset-lineage-json '{"dataset":"NAME","revision":"PINNED_REVISION"}'

python3.12 scripts/validate_controlled_uncertainty_shift_preflight.py \
  --source-csv ROUTING_PACKAGE/routed_full.csv \
  --routing-manifest ROUTING_PACKAGE/routing_manifest.json \
  --runtime-json RUNTIME.json \
  --model-name MODEL_REGISTRY_NAME \
  --output-path PRE_STAGE_A_PREFLIGHT.json

python3.12 scripts/render_slurm_jobs.py \
  --kind silent-bias \
  --stage A \
  --models MODEL_RENDERER_KEY \
  --run-group IMMUTABLE_RUN_GROUP \
  --data-path ROUTING_PACKAGE/routed_full.csv \
  --routing-manifest ROUTING_PACKAGE/routing_manifest.json \
  --runtime-json RUNTIME.json \
  --python-bin ABSOLUTE_PYTHON_3_12 \
  --output-dir STAGE_A_LAUNCHER_DIRECTORY

# Review, syntax-check, submit, and wait for the rendered Stage A job.
bash -n STAGE_A_LAUNCHER_DIRECTORY/silent_bias_stage_a_MODEL_SLUG.slurm
sbatch STAGE_A_LAUNCHER_DIRECTORY/silent_bias_stage_a_MODEL_SLUG.slurm

python3.12 scripts/validate_silent_bias_artifacts.py \
  --source-csv ROUTING_PACKAGE/routed_full.csv \
  --artifact-dir MODEL_RUN_ROOT \
  --stage-a-only \
  --consistency-runs 4 \
  --consistency-schedule extremes \
  --sampling-temperature 0.7 \
  --dataset-split full \
  --require-question-disjoint-routing \
  --expected-stage-b-routing-split test \
  --expected-question-routing-sha256 RAW_ROUTING_ASSIGNMENT_SHA256 \
  --expected-inference-runtime-sha256 INFERENCE_RUNTIME_SHA256 \
  --report-path STAGE_A_VALIDATION.json

python3.12 scripts/validate_controlled_uncertainty_shift_preflight.py \
  --source-csv ROUTING_PACKAGE/routed_full.csv \
  --routing-manifest ROUTING_PACKAGE/routing_manifest.json \
  --runtime-json RUNTIME.json \
  --stage-a-summary MODEL_RUN_ROOT/stage_a/silent_bias_stage_a_pair_summary.jsonl \
  --model-name MODEL_REGISTRY_NAME \
  --output-path EXACT_POST_STAGE_A_PREFLIGHT.json

python3.12 scripts/render_slurm_jobs.py \
  --kind silent-bias \
  --stage B \
  --models MODEL_RENDERER_KEY \
  --run-group IMMUTABLE_RUN_GROUP \
  --data-path ROUTING_PACKAGE/routed_full.csv \
  --routing-manifest ROUTING_PACKAGE/routing_manifest.json \
  --runtime-json RUNTIME.json \
  --python-bin ABSOLUTE_PYTHON_3_12 \
  --stage-a-summary MODEL_RUN_ROOT/stage_a/silent_bias_stage_a_pair_summary.jsonl \
  --stage-a-validation STAGE_A_VALIDATION.json \
  --stage-a-validation-sha256 STAGE_A_VALIDATION_FILE_SHA256 \
  --stage-b-preflight EXACT_POST_STAGE_A_PREFLIGHT.json \
  --stage-b-preflight-sha256 POST_STAGE_A_PREFLIGHT_FILE_SHA256 \
  --output-dir STAGE_B_LAUNCHER_DIRECTORY

# Review, syntax-check, submit, and wait for the separately rendered Stage B job.
bash -n STAGE_B_LAUNCHER_DIRECTORY/silent_bias_stage_b_MODEL_SLUG.slurm
sbatch STAGE_B_LAUNCHER_DIRECTORY/silent_bias_stage_b_MODEL_SLUG.slurm

python3.12 scripts/validate_silent_bias_artifacts.py \
  --source-csv ROUTING_PACKAGE/routed_full.csv \
  --artifact-dir MODEL_RUN_ROOT \
  --consistency-runs 4 \
  --consistency-schedule extremes \
  --sampling-temperature 0.7 \
  --dataset-split full \
  --require-question-disjoint-routing \
  --expected-stage-b-routing-split test \
  --expected-question-routing-sha256 RAW_ROUTING_ASSIGNMENT_SHA256 \
  --expected-inference-runtime-sha256 INFERENCE_RUNTIME_SHA256 \
  --report-path COMPLETE_ARTIFACT_VALIDATION.json

python3.12 scripts/analyze_controlled_uncertainty_shift.py \
  --clean-records MODEL_RUN_ROOT/stage_a/silent_bias_stage_a_uncertainty_scores.jsonl \
  --cued-records MODEL_RUN_ROOT/stage_b/silent_bias_stage_b_uncertainty_scores.jsonl \
  --predictor msp=msp \
  --expected-model-name MODEL_REGISTRY_NAME \
  --expected-model-revision PINNED_MODEL_REVISION \
  --routing-manifest ROUTING_PACKAGE/routing_manifest.json \
  --expected-raw-routing-assignment-sha256 RAW_ROUTING_ASSIGNMENT_SHA256 \
  --expected-eligible-routing-assignment-sha256 ELIGIBLE_ROUTING_ASSIGNMENT_SHA256 \
  --output CONTROLLED_SHIFT_REPORT.json
```

`RAW_ROUTING_ASSIGNMENT_SHA256` is the `routing_assignment_sha256` value from
the immutable schema-2 routing manifest, not a digest copied from generated
artifacts. `ELIGIBLE_ROUTING_ASSIGNMENT_SHA256` is independently pinned from
the canonical pair-loader-eligible question assignment. The analyzer replays
the eligibility loader against `routed_full.csv`, verifies the manifest's raw,
eligible, and skipped counts and eligibility audit, and requires clean records
to match that eligible question set and routing exactly.

`INFERENCE_RUNTIME_SHA256` is the SHA-256 of the pre-run runtime mapping encoded
as sorted, compact ASCII JSON; it must likewise come from the immutable campaign
manifest rather than from outputs being validated. The Stage-A-only validation
report and exact post-Stage-A preflight are distinct, independently hashed
Stage B release inputs. A preflight without `--stage-a-summary` is structural
only and cannot authorize Stage B. Run the two registered models separately;
do not pool their calibration rules or output. The final immutable campaign
manifest and successful model preflights remain prerequisites before full
inference is authorized.
