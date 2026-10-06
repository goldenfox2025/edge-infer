# Validation and numerical attribution: 2026-10-06

This checkpoint records current local tests and the investigation that motivated
the RoPE correction. Passing operator tests does not establish full-model logit
fidelity or a performance release. The original strict checkpoint comparison
remains a recorded failure; its tolerance has not changed.

## Environment and checkpoint

The environment matches [the October 5 record](validation-2026-10-05.md):
RTX 4070 Laptop GPU, 8 GiB, driver 581.42; WSL Ubuntu 24.04; CUDA toolkit
12.0.140, GCC 12.4, CMake 3.28.3; Release builds targeting CUDA architecture 89
and Python 3.12. The reference uses Torch 2.6.0+cu124, Transformers 4.51.3,
safetensors 0.5.3 and NumPy 1.26.4.

The dense `Qwen/Qwen3-0.6B` checkpoint is pinned to revision
`c1899de289a04d12100db370d81485cdf75e47ca`. Its 1,503,300,328-byte safetensors
payload has SHA256
`f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.

## Current independent validation

| Configuration | CTest result | CTest skips |
| --- | ---: | ---: |
| `build-engine-final`, Python enabled | 20/20 passed | 0 |
| `build-native-final`, `EDGE_INFER_BUILD_PYTHON=OFF` | 17/17 passed | 0 |

The ordinary Python environment skips Torch-dependent cases within the weight
conversion test when Torch is absent. The separate full reference environment
passed all five conversion cases at the preceding checkpoint.

All six runtime translation units were inspected without finding Python
includes. The rebuilt computation archive contains no Tensor ownership,
memory-pool or operator-factory symbols. Measured warm submissions retain zero
application C++ allocation/free and zero wrapped native CUDA allocation/free in
the tested scopes. This excludes CUDA-library-private allocation and is not an
end-to-end throughput measurement.

The current [BF16 contract test](../operators/tests/operators_cuda_bf16_contract_test.cu)
checks independent scalar expectations for:

- **RoPE:** prepared prefill, eager decode and graph now share the same cached
  trigonometric values. Standalone uncached RoPE uses double transcendental
  evaluation with explicit FP32 frequency/angle rounding, avoiding the old
  fast single-precision sine path. BF16 trig, product and sum staging is retained.
  Basis tests cover all 64 pairs at positions 0–20 against BF16-rounded FP64,
  host/device offsets, and rejection beyond the cache before launch.
- **Dense bias:** initialize the output with bias and use GEMM `beta=1`, so the
  FP32 dot product and bias are combined before the final BF16 rounding. The
  cancellation fixture preserves `(1 + 1/256) - 1 = 1/256` for both supported
  weight layouts. This fixes the dense biased path; the pinned Qwen3 checkpoint
  has no attention biases.
- **Fused SiLU × up:** round SiLU to the operand dtype, multiply by up, then
  round the product. Fusion preserves the intermediate BF16 conversion rather
  than replacing it with one final conversion. Independent double-based cases
  exercise BF16/FP32, 513-element tails, offset pointers, guards, in-place output,
  empty inputs and extent rejection.

The [attention precision test](../operators/tests/operators_cuda_attention_precision_test.cu)
checks eager and graph decode against full-sequence scalar double-precision
attention, including growing/shrinking histories and FP32 branch statistics.
The [AWQ test](../operators/tests/operators_cuda_awq_test.cu) also checks the
same bias-cancellation boundary in the aligned MMA, vectorized GEMV, scalar
and unaligned fallback paths. Active MMA now adds bias to its FP32 accumulators
before output conversion; its extra bias launch and unused v1 kernel are removed.
Session tests cover cache isolation, reset/replay and graph execution. The TTS
reference test covers conditioning only.

To reproduce with the existing configured build caches and local fixtures:

```sh
cmake --build build-engine-final --parallel 2
ctest --test-dir build-engine-final --output-on-failure
cmake --build build-native-final --parallel 2
ctest --test-dir build-native-final --output-on-failure
```

Fresh-build options and fixture setup are documented in the README. Test logs
are local build outputs under each build's `Testing/Temporary/` directory.

## Pre-fix attribution evidence

**Every measurement in this section predates the current RoPE correction.**
The research used a frozen operator archive with MD5
`d38d68f46e7fb10c2a189e043c8c8fc9`, 28 synthetic attention cases and 224 captured
Qwen3 attention calls from two short prompts and all 28 layers.

Torch eager rounds QK and scaled scores to BF16, then rounds normalized
probabilities to BF16. Native prefill retains FP32 scores/statistics and rounds
unnormalized tile probabilities for WMMA. Native decode retains FP32 branch
numerators and statistics until the output conversion.

On identical captured inputs, native prefill and Torch Flash differed in only
2 of 516,096 prefill output elements, maximum 0.00048828125. Across all 224
calls, RMSE against independent FP64 attention was 0.00132096 for native,
0.00132106 for Torch Flash and 0.01131048 for Torch eager. This demonstrates
backend staging differences; it does not make a failed logit comparison pass.

Attention alone did not explain the old full-model discrepancy. Substituting
native attention, RMSNorm and RoPE into Torch reproduced all 1,367,424 saved
native prefill logits exactly. The RoPE basis probe then isolated an avoidable
trigonometric error: at position 1/pair 63, the old native sine was
`1.125037670135498e-6`, while Torch and BF16-rounded FP64 gave
`1.2442469596862793e-6` (about 9.58% relative error). The old compiled kernel used
`MUFU.SIN`; 110 of 1,344 sampled sine values differed. The existing host cache
matched all 2,688 sampled sine/cosine values and all 8,192 values of the saved
first mismatching Q input against Torch. This evidence motivated the shared
cache and accurate uncached fallback.

Local research records are outside the checkout in `../precision-research/`:
`REPORT.md`, `checkpoint_attribution.json`, `nonattention_attribution.json`,
`rounding_probe.json` and `cached_rope_probe.json`. These are local evidence
locators, not published repository artifacts. The repository operator tests
above reproduce the relevant numerical contracts independently.

## Strict checkpoint comparison

The original comparison failed, and the October 5 corrected run still failed
full logits despite matching all 32 generated tokens. Its prefill maximum errors
were 0.660888671875 and 0.43359375; graph incremental maxima were 0.4375 and
0.296875. These are **historical pre-fix results**, preserved in
`../qwen3-text-parity-fixed/source.json` and the October 5 record.

The unchanged strict criterion remains:

`abs(actual-reference) <= 0.015625 + 0.01 * max(abs(reference),abs(actual))`,

with exact row argmax and separate cache-growth/reset/replay checks.

The current two-prompt, 16-step run completed. Both eager and graph generation
match all **32/32** reference tokens per mode, including reset/replay and four
fresh-prefix checks per prompt. All recorded logits are finite, their row argmax
matches, and interleaved session cache lengths grow as expected. Full logits
**still fail** the unchanged strict eager-reference criterion:

| Mode | Prompt | Prefill max absolute error | Incremental max absolute error |
| --- | ---: | ---: | ---: |
| Eager | 0 | 0.660888671875 | 0.59375 |
| Eager | 1 | 0.44921875 | 0.3125 |
| Graph | 0 | 0.660888671875 | 0.5625 |
| Graph | 1 | 0.44921875 | 0.359375 |

The [machine-readable result](validation/qwen3-0.6b-2026-10-06.json) preserves
versions, source/module hashes, prompt/token sequences, fixture hashes, the
unchanged tolerances and every recorded stage metric. Absolute local source and
module paths are normalized. Raw arrays remain local outputs; regenerate them
with the command below. Token agreement is not full numerical fidelity.

```sh
../qwen3-reference-venv/bin/python tools/validation/qwen3_reference.py \
  --checkpoint ../qwen3-0.6b-reference \
  --output ../qwen3-text-parity-final \
  --model-revision c1899de289a04d12100db370d81485cdf75e47ca \
  --expected-model-sha256 f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b \
  --native-module-dir build-engine-final \
  --native-logit-module-dir build-engine-final/bindings/python/tests \
  --steps 16 --native-capacity 128 \
  --prompt-ids 1,314,1234,42 --prompt-ids 3,7,11,19,23
```

Use a new empty output directory on another run. The tool preserves previous
evidence and exits with status 1 when strict numerical comparison fails.

An additional CPU-only comparison reused saved Torch SDPA/Flash fixtures after
checking checkpoint, software, prompt and continuation provenance. Prompt 0
prefill and reset now match all 607,744 SDPA values exactly in both modes;
prompt 1 prefill has maximum error 0.156982421875 and still fails the same
strict bound. Incremental outputs also fail that bound, with all row argmax
matching. These measurements neither replace the eager-reference failure nor
establish full SDPA fidelity. The local comparison script and detailed record
are `../precision-research/postfix_compare.py` and `postfix-vs-sdpa.json`.

## Remaining scope

These checks do not establish universal checkpoint correctness, long-context
accuracy, broad AWQ compatibility or a performance matrix. Jetson and native
Windows execution remain unverified. Optimized CPU execution and heterogeneous
CPU/GPU placement remain incomplete. TTS conditioning coverage does not cover
the full talker, predictor, codec or waveform pipeline. Docker and remote CI
results must be reported separately from these local runs.
