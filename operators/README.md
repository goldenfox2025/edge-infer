# Operator library

The operator layer can be built separately from the model runtime and Python
bindings. It provides direct CPU reference operations and a standalone CUDA
compute library. Historical Tensor/factory adapters are preserved in Git history.

## Direct CPU reference API

`operators/core/cpu_reference.hpp` defines inline `op::cpu::add`, `multiply`,
`silu`, and `rms_norm` functions. They take borrowed `op::ArrayView<T>` values:
each view contains a pointer and an element count, with no ownership,
allocation, factory lookup, or virtual dispatch.

The CPU templates accept native C++ arithmetic types and are tested with
`float`. Vendor device types such as CUDA BF16 must be converted to float for
these host references; unsupported types are rejected at compile time.

```cpp
#include "operators/core/cpu_reference.hpp"

float a[] = {1.0f, 2.0f};
float b[] = {3.0f, 4.0f};
float result[2];
op::cpu::add<float>({a, 2}, {b, 2}, {result, 2});
```

Callers provide matching extents, valid storage and lifetimes. Exact aliasing
between an input and output is supported; partial overlap is unsupported.
RMSNorm additionally requires a positive feature dimension and weights that do
not overlap the output. These are CPU reference functions, with float
accumulation for SiLU/RMSNorm. No GPU performance
claim follows from this API.

From the repository root, a standalone build needs only CMake and a C++17
compiler:

```sh
cmake -S operators -B build-operators-cpu \
  -DEDGE_INFER_OPERATORS_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-operators-cpu -j
ctest --test-dir build-operators-cpu --output-on-failure
```

This builds the workspace planner, liveness, and direct operator tests without
CUDA, CUTLASS, Python, model weights, or the model runtime. Link the
`EdgeInfer::operators_core` interface target from CMake. This target exports the
`operators/include` and sibling `core/include` directories.

## Direct CUDA compute library

`EdgeInfer::operators_cuda` is the default CUDA static library. It requires the
CUDA toolkit with cudart/cuBLAS and the sibling core headers. It does not require
CUTLASS, Python, model weights or the core memory runtime. It excludes the
historical Tensor/factory adapters and resolves its CUDA device symbols during
the library build, so C++ consumers need no extra CUDA source for device linking.

```sh
cmake -S operators -B build-operators-cuda \
  -DEDGE_INFER_OPERATORS_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release
cmake --build build-operators-cuda -j
ctest --test-dir build-operators-cuda --output-on-failure
```

Choose the CUDA architecture for the target GPU; `89` is the existing desktop
default. The maintained compute library has no CUTLASS, Tensor/factory adapter
or process-global allocator dependency.

`operators/cuda/direct.hpp` exports `op::cuda::add`, `multiply`, `silu`,
`silu_multiply`, and `rms_norm` for borrowed float/BF16 device views and a
caller-supplied stream.
They launch concrete CUDA kernels and do not
allocate operand/workspace storage, register operators or use virtual dispatch.
Matching extents are checked. Elementwise counts must fit a signed integer;
the current RMSNorm kernel supports feature dimensions from 1 to 10240 and
rejects larger nonempty rows before launch. Empty views perform no launch.
RMSNorm weights must not overlap the output. Exact input/output aliasing is
supported; partial overlap is unsupported.

`silu_multiply(gate, up, output, stream)` performs the gated activation in one
kernel, with no intermediate device buffer. It preserves the two-stage dtype
contract: first round `SiLU(gate)` to the operand dtype, then multiply by `up`
and round the product. BF16 therefore keeps the same intermediate activation
rounding as separate SiLU and multiplication calls. FP32 retains FP32 arithmetic.
The decoder uses this operation with `output` exactly aliasing `gate`.

Launches are asynchronous. Keep device buffers alive until the supplied stream
completes. Enqueue uploads on that stream or establish a CUDA event dependency,
then synchronize or wait for an event before consuming outputs or reusing
storage from another stream.

```cpp
#include "operators/cuda/direct.hpp"

// The caller owns device_a, device_b, device_output, and stream.
op::cuda::add<float>({device_a, count}, {device_b, count},
                     {device_output, count}, stream);
```

Prepared model execution calls the direct CUDA API with fixed views and resolved
weight layouts. CUDA calls retain their launch and library overhead. Jetson
builds and end-to-end inference remain unverified.

The direct GPU test compares float/BF16 outputs against CPU references, including
in-place operations, vector tails, empty inputs, the RMSNorm cache boundary and
its size guard. It uses the `gpu`/`cuda` labels and skips with code 77 when no
CUDA device is available. Build success or a skipped test does not establish
GPU correctness on a target device.

`operators_cuda_bf16_contract_test` checks the fused activation against an
independent scalar reference, comparing BF16 bits exactly and FP32 values within
tolerance. Its mixed-sign inputs, cancelling output pairs, offset pointers and
513-element extent cover intermediate rounding, in-place execution, scalar tails,
nonblocking streams and output guards. Invalid extents and empty calls are also
checked. The same test covers BF16 RMSNorm, RoPE and biased dense-linear rounding.

## Direct CUDA conditioning primitives

`operators/cuda/conditioning.hpp` adds borrowed float/BF16 `op::cuda::linear`
and `sum_embeddings` primitives. These are tensor operations independent of
model names, text tokenization, audio decoding, and Python. Link the same
`EdgeInfer::operators_cuda` target.

`linear` accepts contiguous row-major input `[rows, in_features]`, physical
weights `[out_features, in_features]`, an optional bias `[out_features]`, and
output `[rows, out_features]`. Pass an empty bias view to omit it. The caller
provides at least `rows * out_features` floats of device scratch and a cuBLAS
handle in host pointer mode with `CUBLAS_DEFAULT_MATH`. The call explicitly sets
that handle's stream to the supplied stream; do not use the handle concurrently.
GEMM uses FP32 computation and writes FP32 scratch, then adds bias before casting
to the output dtype. This preserves the biased linear layer's BF16 rounding
boundary instead of rounding the GEMM result before adding bias. Nonempty
dimensions must be positive and fit cuBLAS's signed-int interface.

```cpp
#include "operators/cuda/conditioning.hpp"

// The caller owns the device buffers, FP32 scratch, handle, and stream.
op::cuda::linear<float>(
    {device_input, rows * in_features},
    {device_weight, out_features * in_features},
    {device_bias, out_features}, {device_output, rows * out_features},
    {device_scratch, rows * out_features},
    rows, in_features, out_features, handle, stream);
```

`sum_embeddings` takes a host array of `EmbeddingTable<T>` descriptors, each
containing a borrowed device table `[vocab_size, features]`, and host code IDs
in `[frames, codebooks]` order. Codebooks may have different vocabulary sizes.
It writes `[frames, features]` output using at least `frames * features` floats
of caller-owned device scratch. Every ID and table extent is validated before
the first launch. All codebooks accumulate in FP32 and the result is cast once;
repeated BF16 additions would introduce extra rounding. IDs travel in bounded
64-frame launch arguments, with no device metadata allocation. The primitive
consumes the host descriptors and IDs during the call, so these host arrays may
be released on return. Their referenced device tables must remain alive until
the supplied stream completes.

Both primitives require valid storage on the current CUDA device. Matrix,
table, ID, and output views have exact extents; scratch may be larger than the
required count. Extent products and byte counts are checked for overflow.
Operands and scratch must not overlap. Output may exactly alias scratch for
`float`; other output overlap is unsupported. These are primitive-level alias
rules: a composed operation may need stricter rules when one stage's output
becomes a later stage's input. Empty operations launch no work. Keep all device
buffers and the borrowed handle alive through stream completion, and establish
stream/event dependencies before consuming results elsewhere.

These functions perform no operand or workspace allocation, factory lookup,
shared ownership copies, or virtual dispatch. CUDA launch and cuBLAS overhead,
including any cuBLAS-managed workspace, remain. They do not establish a speedup
or complete speech inference support.

`operators_cuda_conditioning_test` checks float/BF16 biased and unbiased linear
layers, odd feature widths, different codebook vocabularies, multiple blocks
and the 64-frame chunk boundary. Cancellation fixtures verify the BF16 bias and
codebook sum rounding boundaries. It also checks nonblocking stream use, host
metadata lifetime, permitted float scratch aliasing, empty calls, handle modes,
and rejected IDs/extents/overflow before output or scratch writes. It uses the
`operators`/`cuda`/`gpu` labels and skips with code 77 when no GPU is available.
These primitive tests use small deterministic inputs; checkpoint-based model
conditioning parity is a separate runtime validation step.

## Prepared CUDA execution

`operators/cuda/execution.hpp` provides direct linear/AWQ, gather, RoPE, attention,
KV stores and sampling on fixed `TensorView<T, Rank>` descriptors. Views contain
only a borrowed pointer and fixed extents/strides. Preparation resolves matrix
layout and device launch limits once. Bind the borrowed execution context at the
start of a decoder submission, then keep all buffers and its handle alive through
stream completion. No operator factory or model runtime is needed.

Sampling preparation queries its CUB scratch requirement once; the caller owns
that scratch, output tokens/probabilities and RNG state. Temperature/top-k/top-p
are applied explicitly. A probability query writes to caller-owned storage.
BF16 head width 128 reuses the optimized attention implementation. Other supported
head widths and FP32 use a scalar correctness fallback. That fallback establishes
functional coverage; no performance improvement is implied.

Prepared models own one shared FP32 sine/cosine table. The common decoder calls
`rope_precomputed` with that table in prefill, eager decode and CUDA Graph mode;
the modes differ in how they supply the logical position offset. BF16 rotation
rounds trigonometric values, each product and the final sum to BF16. The uncached
`rope` primitive remains available for callers that provide a theta instead.
This table implements ordinary RoPE. Qwen TTS MRoPE still requires multimodal
position support in its model adapter.

Maintained optimized kernel bodies live in `operators/src/cuda/kernels` under
concrete entry points. Unused prefill variants and historical adapters are kept
in Git history rather than compiled into the compute archive.

`EdgeInfer::operators_cuda` links portable headers, cudart and cuBLAS only. The
normal compute archive excludes Tensor adapters, shared ownership and global
memory-pool symbols. Valid prepared submissions allocate no application operand
or scratch storage. CUDA library-private allocations remain outside that claim.

## CMake integration

- `EDGE_INFER_OPERATORS_ENABLE_CUDA`: build the direct CUDA compute library
  (default `ON`). Set it to `OFF` for the direct CPU reference path.
- `EDGE_INFER_CORE_ENABLE_CUDA`: expose the sibling core CUDA header interface
  with cudart linkage for owning Tensor consumers. The direct compute archive
  needs the portable core headers only.
- `BUILD_TESTING`: include tests (default `ON` for standalone builds).
- `EdgeInfer::operators_core`: header interface for the direct CPU operations
  and workspace utilities.
- `EdgeInfer::operators_cuda`: direct CUDA compute library; links portable
  headers, cudart and cuBLAS.

The sibling `core` directory is part of this library's dependency boundary.
Model code, Python bindings, frontend code and engine include paths are outside
it. Add `operators` once to a parent CMake project; its standalone configuration
adds `core` only when the core targets are absent.
