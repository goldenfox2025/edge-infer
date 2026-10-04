# Operator library

The operator layer can be built separately from the model runtime and Python
bindings. It provides a small direct CPU reference API and the existing
CPU/CUDA compatibility adapters.

## Direct CPU reference API

`operators/core/cpu_reference.hpp` defines inline `op::cpu::add`, `multiply`,
`silu`, and `rms_norm` functions. They take borrowed `op::ArrayView<T>` values:
each view contains a pointer and an element count, with no ownership,
allocation, factory lookup, or virtual dispatch. The CPU adapter classes reuse
these functions for their float implementations.

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
accumulation for SiLU/RMSNorm as in the existing adapters. No GPU performance
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

## CUDA and compatibility adapters

The `EdgeInfer::unified_operators` static library includes the existing
`Tensor`, factory, CPU and CUDA adapters, CUDA kernels, and legacy kernel bridge.
It requires the CUDA toolkit, the sibling `core` directory, and the pinned
CUTLASS submodule. It does not require Python or model weights.
The static operator library resolves its CUDA device symbols when it is built,
so a C++ application can link it through the native runtime without adding a
CUDA source merely to trigger final device linking.

```sh
git submodule update --init cutlass
cmake -S operators -B build-operators-cuda \
  -DEDGE_INFER_OPERATORS_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release
cmake --build build-operators-cuda -j
ctest --test-dir build-operators-cuda --output-on-failure
```

Choose the CUDA architecture for the target GPU; `89` is the existing desktop
default. `CUTLASS_DIR` can point to another compatible CUTLASS checkout. Building
this library includes a CPU compatibility test that still links the CUDA
toolkit because the existing `Tensor` API uses CUDA types and memory helpers.
It performs no model inference.

`operators/cuda/direct.hpp` exports `op::cuda::add`, `multiply`, `silu`, and
`rms_norm` for borrowed float/BF16 device views and a caller-supplied stream.
They share the existing CUDA kernels with the compatibility adapters and do not
allocate operand/workspace storage, register operators or use virtual dispatch.
Matching extents are checked. Elementwise counts must fit a signed integer;
the current RMSNorm kernel supports feature dimensions from 1 to 10240 and
rejects larger nonempty rows before launch. Empty views perform no launch.
RMSNorm weights must not overlap the output; exact input/output aliasing is
supported for all four functions.

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

`UnifiedOperators` uses these direct CUDA functions for the same four operations
and statically calls their CPU adapters. Those methods bypass factory lookup,
shared pointer copies and virtual dispatch. Factory metadata, the remaining
compatibility operations, and Qwen3's prepared packed executor retain dynamic
dispatch. Custom factory registrations affect those dynamic paths, while the
four direct facade methods use the built-in implementations.

CUDA calls retain their launch and library overhead. CPU sampling
uses AVX when enabled by the compiler and scalar temperature/max loops
otherwise. This fallback removes an x86-only header dependency; Jetson builds
and end-to-end inference remain unverified.

The GPU test compares float/BF16 outputs against CPU references, including
in-place operations, vector tails, empty inputs, the RMSNorm cache boundary and
its size guard. It uses the `gpu`/`cuda` labels and skips with code 77 when no
CUDA device is available. Build success or a skipped test does not establish
GPU correctness on a target device.

## CMake integration

- `EDGE_INFER_OPERATORS_ENABLE_CUDA`: build compatibility adapters and CUDA kernels
  (default `ON`). Set it to `OFF` for the direct CPU reference path.
- `EDGE_INFER_CORE_ENABLE_CUDA`: build the sibling core memory runtime. A standalone
  operators build selects the same value as `EDGE_INFER_OPERATORS_ENABLE_CUDA`. A
  parent project that adds `core` first must enable it for CUDA operators.
- `BUILD_TESTING`: include tests (default `ON` for standalone builds).
- `EdgeInfer::operators_core`: header interface for the direct CPU operations
  and workspace utilities.
- `EdgeInfer::unified_operators`: compatibility library, available with CUDA
  enabled; links `EdgeInfer::core_cuda` and cuBLAS.

The sibling `core` directory is part of this library's dependency boundary.
Model code, Python bindings, frontend code and engine include paths are outside
it. Add `operators` once to a parent CMake project; its standalone configuration
adds `core` only when the core targets are absent.
