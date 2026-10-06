# Refactor validation checkpoint: 2026-10-05

This is a development checkpoint, not a completed model-compatibility or
performance release. Work was wrapped up after midnight on 2026-10-06; the
already-running revised full-checkpoint comparison completed during wrapup.

## Environment

- NVIDIA RTX 4070 Laptop GPU, 8 GiB; driver 581.42.
- WSL Ubuntu 24.04, CUDA toolkit 12.0.140, GCC 12.4, CMake 3.28.3.
- Release native build, CUDA architecture 89; Python 3.12 bindings.
- Reference environment: Torch 2.6.0+cu124, Transformers 4.51.3,
  safetensors 0.5.3 and NumPy 1.26.4.

## Completed checks

The latest Python-enabled native configuration passed all **20/20 CTest tests**,
including the two numerical tests added after the earlier 18-test checkpoint:

- Exact staged BF16 RMSNorm and eager/cached RoPE rounding fixtures, with
  in-place normalization and an unchanged FP32 normalization reference.
- FP32 decode-attention branch statistics against independent double-precision
  scalar gold; growing across partition thresholds and shrinking graph history.
- Shared-model session isolation, private cache bounds, reset/replay, held
  outputs, host-token/CUDA-view parity and rejection before output writes.
- Tensor ownership, workspace planning, AWQ, sampling, generation and Python
  callback/model-session integration.
- Pinned Qwen3-TTS conditioning fixture parity. This covers conditioning only,
  not talker, predictor, codec or waveform generation.

Earlier configurations passed portable top-level and standalone operator tests
4/4 each and the no-Python native tests 15/15. Those earlier results precede the
latest numerical corrections; a current-source no-Python rerun remains pending.
The full reference environment separately passed all five weight-conversion
checks; the normal local environment skips Torch-dependent cases when absent.
Offline frontend checkpoint loading passed 9/9 and TTS contracts passed 19/19.

The session regression again verifies zero application C++ allocation/free and
wrapped native CUDA allocation/free in measured warm decode submissions. This
does not count CUDA-library-private memory or establish end-to-end performance.

Latest commands:

```sh
cmake -S . -B build-engine-final
cmake --build build-engine-final --parallel 2
ctest --test-dir build-engine-final --output-on-failure
```

The existing build cache selects the compiler, architecture, Python interpreter
and external conditioning fixtures above. See the README for fresh-build flags.

## Real Qwen3 comparison: tokens pass, strict logits fail

The local dense `Qwen/Qwen3-0.6B` checkpoint is pinned to publisher revision
`c1899de289a04d12100db370d81485cdf75e47ca`. Its single safetensors payload has
1,503,300,328 bytes and SHA256
`f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.

The original strict comparison failed: eager generation diverged on one prompt,
and both eager/graph logits exceeded the declared numerical bounds. This exposed
missing BF16 stages in RMSNorm/RoPE and BF16 attention branch statistics. These
issues are corrected and covered by the latest operator regressions.

The revised full-checkpoint run completed two explicit token prompts, 16 greedy
steps and four fresh-prefix checks each. Eager and graph native generation now
both match all 32 reference tokens, including reset/replay and fresh-prefix
checks. Cache growth and all recorded row argmax checks match. Nevertheless,
full copied logits still fail the unchanged strict bound. Prefill maximum
absolute errors are 0.660888671875 and 0.43359375; graph incremental maxima are
0.4375 and 0.296875. The complete revised result is recorded outside the checkout
in `../qwen3-text-parity-fixed/source.json`. This remains a failed numerical
comparison, despite the passing generated-token checks.

The unchanged strict criterion is:

`abs(actual-reference) <= 0.015625 + 0.01 * max(abs(reference),abs(actual))`,
with exact row argmax, expected cache growth and reset/replay checks.

An independent attribution experiment used the same checkpoint and compared
Torch eager with Torch CUDA SDPA, whose profiler selected Flash Attention.
All 32 generated tokens matched, while 43%–57% of logits exceeded that same
bound. Backend-specific BF16 rounding therefore contributes to strict logit
differences. This evidence does not make the failed native comparison pass.
The maintained prefill path accumulates numerators in FP32, but its QK/scaling
and probability-rounding stages differ from the pinned Torch eager path.

Resume the reference command with a fresh output directory; the tool rejects
nonempty output directories to preserve previous evidence:

```sh
../qwen3-reference-venv/bin/python tools/validation/qwen3_reference.py \
  --checkpoint ../qwen3-0.6b-reference \
  --output ../qwen3-text-parity-next \
  --model-revision c1899de289a04d12100db370d81485cdf75e47ca \
  --expected-model-sha256 f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b \
  --native-module-dir build-engine-final \
  --native-logit-module-dir build-engine-final/bindings/python/tests \
  --steps 16 --native-capacity 128 \
  --prompt-ids 1,314,1234,42 --prompt-ids 3,7,11,19,23
```

## Remaining verification and scope

Docker was not built locally. The updated GPU-free CUDA CI configuration has
not yet run for this checkpoint. Skipped GPU tests establish no numerical
coverage. Native Windows, Jetson, optimized CPU execution, heterogeneous
placement, full Qwen TTS and broad checkpoint/performance matrices remain
unverified or unimplemented. No license is invented by this refactor.
