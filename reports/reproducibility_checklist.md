# Reproducibility Checklist

Record these fields for every paper run.

## Code and Environment

- Git commit hash.
- Python version.
- `uv.lock` checksum or committed lockfile state.
- `vllm`, `torch`, `transformers`, `datasets`, `numpy`, and `pandas` versions.
- CUDA driver/runtime versions.
- Hardware: GPU model/count, CPU count, memory, Slurm partition/QOS.

## Storage

- `BIASES_ARTIFACT_ROOT`.
- `HF_HOME`, `HF_HUB_CACHE`, `HF_DATASETS_CACHE`.
- `VLLM_CACHE_ROOT`, `TORCH_HOME`, `TMPDIR`, `TRITON_CACHE_DIR`, `UV_CACHE_DIR`.
- Dataset CSV path and checksum.

## Decoding

- Model name and revision if pinned.
- `dtype`.
- `tensor_parallel_size`.
- `max_model_len`.
- `gpu_memory_utilization`.
- Verdict decoding: greedy, `max_tokens=1`, constrained to tokenizer IDs for
  `A`, `B`, and `T`.
- Tokenizer-specific label-token IDs extracted by the runner.
- Qwen3/Qwen3.5 non-thinking prefill policy.

## Randomness

- Routing split seed.
- Consistency-run seeds.
- Sampling temperature for consistency runs.
- Rewrite seed for intrinsic cue variants.

## Metrics

- Tie policy from `reports/metric_definitions.md`.
- Calibration/test split definition.
- Routing budget grid.
- Bootstrap seed and number of resamples.
- Multiple-testing correction method.
