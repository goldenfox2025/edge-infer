# Qwen3 session refactor validation: 2026-10-04

This record covers the shared prepared Qwen3 model, dedicated execution session,
typed eager/prefill decoder, persistent cache ownership, borrowed cuBLAS context
and CUDA graph fixes committed with this document.

## Environment

WSL Ubuntu 24.04, x86-64; RTX 4070 Laptop GPU; NVIDIA driver 581.42;
CUDA 12.0.140; architecture 89; GCC 12.4; CMake 3.28.3; Debug configuration.
The binding configuration uses Python 3.12.3 and pybind11 2.13.6.
CUTLASS is pinned at `853ad93d60b23b4f87bc46dfbc3c9ce757773ed7`.

## Results

| Check | Result |
| --- | --- |
| Native runtime, Python and pybind11 discovery disabled | Build passed; 10/10 CTest tests passed |
| Runtime compilation dependency boundary | All eight runtime sources compile without Python/pybind11 include paths |
| Python-enabled runtime and extension | Build passed; 11/11 CTest tests passed |
| Offline Qwen TTS checkpoint contracts | 19/19 tests passed |
| Python 3.10 syntax, repository Markdown links and recorded TTS revisions | Passed |

The native and Python-enabled counts include the optional real Qwen TTS
conditioning fixture test. Without those local fixtures, the reproduction below
registers nine native tests. No registered test was skipped in these runs.

The focused session regression passed after the final layout and migration
fixes. All four eager/graph stages compared for history A had observed maximum
absolute error zero for logits, K and V. This is a synthetic-model result;
full checkpoint parity remains a separate validation milestone.

## Numerical and ownership coverage

`qwen3_session_test` uses nonzero deterministic BF16 synthetic weights: one
transformer layer, hidden/head width 128, one query/KV head, intermediate width
256, vocabulary 64 and context capacity 16. It needs no downloaded model.

Two prompt histories are compared against isolated baselines, checking every
logit and all initialized KV capacity after prefill and three decode steps.
The test checks separate streams, stable/disjoint output addresses, retained
prefill logits after another session and a legacy pool reset, shared model
lifetime, preserved rectangular transposed weight layout, same-cache reset
replay and continuation after another session is destroyed.

The managed API is checked in both eager and graph modes with capacities six
and seven. A child shares the exact prepared model, starts empty and maintains
its own history and fixed decode output. Automatic prefill/decode logits match
the isolated baselines. Tests cover replacing a history, reset/replay, parent
continuation after child destruction, full-capacity refusal and invalid input
or capacity requests preserving logical length and held logits.

Greedy sampled prefill checks the final prompt row, then sampled decode checks
argmax of complete logits. Sampled-token addresses stay private and fixed.
Out-of-vocabulary tokens, invalid pointers/ranks/devices/extents, malformed
weight shapes, incompatible caches and changed cache bindings reject before
model writes. A cache constructed during the legacy global prefill phase
survives arena reset and scratch overwrite.

A two-layer cache with capacity three and width seven preserves every K/V value,
active length and view layout through CPU-to-CUDA-to-CPU-to-CUDA migration.
Both uploads occur during the legacy prefill phase, followed by arena reset
and scratch overwrite. Fixed-output sampling rejects zero or excessive top-k
before kernel launch and preserves logits and an output sentinel.

AWQ preparation accepts contiguous output-major packed weights and padded
scales. It rejects the old input/group-major layouts, incomplete groups,
undersized scales and transposed packed tensors. Noncontiguous embeddings and
token inputs, and embedding widths unsupported by the vectorized gather kernel,
also reject. These AWQ checks validate metadata; they do not execute quantized
matmul or establish its numerical correctness.

Eager/graph comparisons check all logits and KV values with BF16 tolerance
`0.015625 + 0.01 * max(abs(reference), abs(actual))` and require matching greedy tokens. Isolation
comparisons within the same mode require exact values. Public session calls
synchronize before returning; interleaving does not test simultaneous full
model execution.

`operators_cuda_context_test` separately tests simultaneous dense float/BF16
GEMM submissions through two independent handles/nonblocking streams, including
rectangular logical weight views, direct/named prepared calls, bias/no-bias,
null-stream reset, singleton isolation and handle ownership.

The prior real Qwen TTS conditioning reference test remains part of the native
suite; its scope is recorded [separately](validation-qwen-tts-2026-10-04.md).

## Reproduction

```sh
cmake -S . -B build-native \
  -DEDGE_INFER_BUILD_RUNTIME=ON -DEDGE_INFER_BUILD_PYTHON=OFF \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-12 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 \
  -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON
cmake --build build-native --parallel 2
ctest --test-dir build-native --output-on-failure
./build-native/runtime/qwen3_session_test
```

Select a compiler/toolkit/device combination appropriate for your GPU. The
optional speech reference test additionally needs locally exported fixtures as
described in the conditioning validation record. CI tests the portable and
callback boundaries; native GPU validation was performed locally.

## Limits

This synthetic regression establishes the tested state/lifetime boundaries.
It does not establish full checkpoint parity, AWQ numerical correctness,
probabilistic sampling distributions, speculative resampling, multi-device
migration, asynchronous full-session concurrency, Jetson support or a speedup.
Token validation currently copies IDs to the host and session calls synchronize;
performance must be measured before changing these safety boundaries.

Legacy sampling scratch and Qwen2 runtime paths still need migration. Python
retains one guarded session. The speech talker, residual-code predictor, codec
and waveform generation have not been implemented by this refactor.
