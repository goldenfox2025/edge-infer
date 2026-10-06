# Runtime and checkpoint contracts: 2026-10-06

This record covers the cleanup after the published
`6fafb0213197b08a13b379a73b81ab42be86bf29` baseline. Its earlier numerical
investigation and measurements remain in [the baseline record](validation-2026-10-06.md).
Operator and lifetime tests do not establish full-model numerical fidelity or
end-to-end performance.

## Environment and final checks

Local verification uses the same RTX 4070 Laptop GPU, WSL Ubuntu 24.04,
CUDA 12.0.140, GCC 12.4, CMake 3.28.3 and architecture 89 as the baseline.
The final source set was frozen before rebuilding both configurations and
checked by SHA256 after building and testing. GPU suites ran sequentially.

| Check | Result |
| --- | --- |
| Python-enabled native build and CTest | 22/22 passed; no CTest skips |
| Python-free native build and CTest | 19/19 passed; no CTest skips |
| Weight mapping/conversion in the full reference environment | 11/11 passed; no internal skips |
| Python Model/Session integration against the final module | 12/12 passed |
| Local checkpoint frontend contracts | 11/11 passed |
| Saved Qwen3 reference artifact contracts | 15/15 passed |
| Python AST and Bash syntax | 25 Python files and 5 shell scripts passed |

The ordinary build environment's weight-conversion test internally skips five
Torch cases; the separate full-environment run above executes all of them.
Multi-device workspace/release and cross-device callback checks cannot establish
multi-GPU behavior on this one-GPU machine. No CUDA numerical test was skipped
for missing hardware. Build logs contain GCC/pybind11, nvlink and subsecond
DrvFS clock-skew warnings; the changed runtime units, tests and Python modules
were rebuilt successfully.

All native runtime source units have no Python, pybind11, Torch or NumPy includes.
The rebuilt compute archive has no Tensor-ownership, memory-pool,
operator-factory or Python symbols.

## Storage and completion

CPU executors share a private checkpoint snapshot instead of copying the full
checkpoint on every fork. Native and Python tests cover caller mutation,
prepared-model destruction, independent outputs/caches, concurrent CPU execution
and independent sampling state. This remains an FP32 reference backend, not an
optimized CPU/GPU placement engine.

Hidden-only embedding prefill omits the vocabulary head request. The planning
fixture with 1,024 rows, hidden width 1,024 and vocabulary 151,936 uses
29,360,128 bytes (28 MiB) for hidden output versus 340,525,056 bytes
(324.75 MiB) for token output. It omits 311,164,928 bytes (296.75 MiB) of logits
requests. This is a synthetic workspace plan, not a checkpoint peak-VRAM
measurement. Same-row output switching is tested; arenas retain their peak
capacity after larger requests.

Measured steady eager, graph and embedding decode scopes retain zero
application C++ allocation/free and zero wrapped native CUDA allocation/free.
These probes exclude CUDA-library-private allocation. Alignment, peak retained
capacity, simultaneously live values and library workspaces still consume memory.

Ordinary callbacks run on the caller thread after a completed token operation,
before the next decode. Long-request callback failures stop immediately without
a producer queue. Ordinary, speculative and direct-session failure tests preserve
the original exception, complete all owned execution resources before clearing
history, and keep storage after a failed completion. A later successful wait
cannot revive an invalid instance. Direct decode never restores a cache extent
that an executor already discarded; capacity rejection occurs before execution.
Reset and warmup completion failures follow the same invalidation rule.

Direct-session regressions inject a GEMM failure after real KV writes, then
exercise device tokens, host tokens and embeddings with both successful and
failed completion. CUDA wrappers perform a real wait before a synthetic wait
failure; they do not deliberately poison the CUDA context. Tests check retained
storage, original errors, rejected operations, fresh-request recovery after
confirmed cleanup and unaffected sibling sessions.

## Checkpoint and scalar contracts

One mapper replaces the repeated Llama/Qwen/Qwen3 conversion implementations.
Tests cover exact keys, layer bounds, ranks, duplicate aliases, dense physical
layouts, all supported projection/head biases and explicit tied embeddings
without caller-dictionary mutation. AWQ admission requires compatible 4-bit
asymmetric GEMV metadata, consistent group sizes, signed int32 packing and
complete projection triples; dense/packed overlaps reject. Frontend and native
admission agree on declared formats. This does not establish compatibility with
arbitrary AWQ packers.

SiLU scalar tests now cover negative subnormal tails, tiny signed inputs,
large-product amplification and standalone/fused consistency under the build's
FTZ settings. For example, `x=-92` previously produced negative zero instead of
the independently rounded BF16 value `0x806f`. Stable tail evaluation and an
explicit rounding conversion preserve that value; IEEE multiplication preserves
subnormals and signed zero. The unchanged attention, RoPE, biased dense/AWQ,
sampling and conditioning contract tests also pass.

Reference validation can select eager or SDPA explicitly, verifies the actual
configured backend and disables attention-output requests that would cause a
fallback. Saved references require matching checkpoint/config/software/source
hashes and validated contained NPZ bytes, shapes, dtypes, finite values, token
continuations and teacher prefixes. Reuse loads no Torch model or tokenizer.
Exact native reset/replay and cross-mode prefill checks are separate from the
unchanged strict reference tolerance.

## Real Qwen3-0.6B rerun

The dense `Qwen/Qwen3-0.6B` checkpoint remains pinned to revision
`c1899de289a04d12100db370d81485cdf75e47ca`, with a 1,503,300,328-byte
safetensors payload and SHA256
`f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.
Two prompts (`1,314,1234,42` and `3,7,11,19,23`) each generate 16 greedy tokens
with capacity 128 and four fresh-prefix checks.

Both eager and graph match all **32/32 generated reference tokens per mode**,
including generation reset/replay and fresh-prefix token checks. Native logits
are finite with matching row argmax and expected interleaved cache growth.
Native reset/replay and eager/graph prefill match exactly in shape, dtype and
bytes, independently of the strict reference bound.

Full logits **still fail** the unchanged eager-reference criterion:

`abs(actual-reference) <= 0.015625 + 0.01 * max(abs(reference),abs(actual))`.

| Mode | Prompt | Prefill max absolute error | Incremental max absolute error |
| --- | ---: | ---: | ---: |
| Eager | 0 | 0.660888671875 | 0.59375 |
| Eager | 1 | 0.44921875 | 0.3125 |
| Graph | 0 | 0.660888671875 | 0.5625 |
| Graph | 1 | 0.44921875 | 0.359375 |

These maxima equal the baseline measurements. Token agreement does not establish
full numerical fidelity. The tool exits with status 1 for the strict failure.
No tolerance was widened and no reference failure was reclassified as success.

The [machine-readable result](validation/qwen3-0.6b-runtime-contracts-2026-10-06.json)
preserves all stage metrics, module hashes, fixture hashes, token sequences,
exact invariants and a SHA256 snapshot of the 114 tested source/CMake files.
Source hashes identify the exact local build bytes; repository text uses LF,
so Git normalization can change checkout byte hashes for CRLF working files.
The raw local result manifest has SHA256
`baf8adaab0c868e777520a1e24b336c5ca562e66ca52b7f5af53bccab7c48eaa`.
The reused eager reference manifest retains its original SHA256
`2844cd02a8a814e6e18e506ca389e76e8efcd87483ea2e5e733710472e406a0c`
and prior strict failure status. Absolute local source/module/reuse paths are
normalized in the published result. Array fixtures remain local outputs.

## Reproduction

With the existing configured build caches and local fixtures:

```sh
cmake --build build-engine-final --parallel 3
cmake --build build-native-final --parallel 3
ctest --test-dir build-engine-final --output-on-failure
ctest --test-dir build-native-final --output-on-failure
../qwen3-reference-venv/bin/python bindings/python/tests/weight_conversion_test.py \
  build-engine-final/bindings/python/tests
../qwen3-reference-venv/bin/python bindings/python/tests/model_session_test.py \
  build-engine-final
../qwen3-reference-venv/bin/python -m unittest discover \
  -s frontend/tests -p test_checkpoint.py -v
../qwen3-reference-venv/bin/python -m unittest discover \
  -s tools/validation/tests -p test_qwen3_reference_artifacts.py -v
```

The reference environment uses Torch 2.6.0+cu124, Transformers 4.51.3 and
NumPy 1.26.4. Fresh-build prerequisites are in the README. Test logs and
full array fixtures are local outputs, not committed build products.

To rerun native checks against the verified saved eager gold:

```sh
../qwen3-reference-venv/bin/python tools/validation/qwen3_reference.py \
  --checkpoint ../qwen3-0.6b-reference \
  --output ../qwen3-text-parity-contracts \
  --model-revision c1899de289a04d12100db370d81485cdf75e47ca \
  --expected-model-sha256 f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b \
  --native-module-dir build-engine-final \
  --native-logit-module-dir build-engine-final/bindings/python/tests \
  --reuse-reference ../qwen3-text-parity-final/source.json \
  --reference-attention eager --steps 16 --native-capacity 128 \
  --prompt-ids 1,314,1234,42 --prompt-ids 3,7,11,19,23
```

Use a new empty output directory for each run. On another machine, generate
fresh local reference fixtures first using the baseline command or omit
`--reuse-reference`; reuse must pass all provenance checks. A fresh reference
is a new measurement and must not overwrite the saved failed comparison.

## Remaining scope

Full Qwen3-TTS talker, MRoPE, code predictor, codec and waveform validation,
MiniMax H3 execution, optimized CPU operators, CPU/GPU placement, Jetson,
multi-GPU validation and performance measurements remain unimplemented or
unverified. The existing TTS reference covers conditioning only. Licensing and
upstream attribution remain a release decision. Remote CI and Docker scope must
be reported separately from local GPU verification.
