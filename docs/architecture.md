# Architecture

The responsibilities are organized as `core -> operators -> runtime -> bindings`.
Core and operators have independent build targets. Runtime sources currently
build into the Python module, and callback handling in `runtime/src/inference.cpp`
still depends on pybind11. Moving GIL handling fully into the bindings is future
work. The Python frontend calls the bindings; development tools do not belong
in operator or model execution code.

## Shared core

`core/include/execution/` owns portable workspace planning and lifetime analysis.
`core_headers` exposes these without CUDA or Python. Tensor, WeightTensor and
the CUDA memory pool also share one implementation in `core/`. `core_cuda` owns
the pool's out-of-line definitions and links the CUDA runtime and driver.
The owning Tensor API retains its allocation and shared-ownership costs.

## Operator library

`operators_core` exposes non-owning reference operations through headers. Calls
use concrete types and inline functions rather than a registry, virtual methods
or shared ownership. This API introduces no required dispatch or ownership
machinery; optimized code and production performance still need measurement.

`unified_operators` retains CUDA kernels and Tensor-based adapters. It consumes
core and CUDA/CUTLASS without depending on models, Python or frontend code.
OperatorFactory and UnifiedOperators remain dynamic compatibility APIs.
Add, multiply, SiLU and RMSNorm also expose direct borrowed-buffer CUDA calls;
their existing eager facade methods use these calls. Matmul and prepared-node
execution retain dynamic dispatch. Migrating these remaining paths is future work.

Fused kernels belong beside primitive kernels. A fused projection/RoPE/KV-write
operation should declare its layout and workspace requirements rather than hide
model state in a global registry. Validate a fusion against an unfused reference
before replacing a call site.

`operators/src/cuda/legacy/` contains kernels still used by adapters. These
sources remain necessary until their consumers are migrated and validated.

## Runtime and bindings

`runtime/` owns model sequencing, cache state, execution programs, graph capture
and decoding. Model code may select an operation but must not become a dependency
of that operation's implementation. `bindings/python/` owns weight preprocessing,
initialization and Python exposure.

## Future speech models

[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) includes an audio tokenizer/codec
and streaming speech generation. Transformer primitives can be shared, while
speech-specific execution and audio decoding need separate runtime components.
An output API will need audio chunks, sample-rate metadata and cancellation or
backpressure. Add these contracts when a concrete integration requires them;
this cleanup does not implement TTS or add an unused modality hierarchy.
