# Architecture

The responsibilities are organized as `core -> operators -> runtime -> bindings`.
Core, operators and the native runtime have independent build targets. The
`EdgeInfer::runtime` static library depends on CUDA, operators and native threads;
it has no Python or pybind11 dependency. The optional Python module links that
library and owns Python callback/GIL handling. The Python frontend calls the
bindings; development tools do not belong in operator or model execution code.

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

The native `model_factory.hpp` belongs to `runtime/include/` and consumes Tensor
weight maps rather than Python dictionaries. The current frontend and bindings
still load/convert checkpoint weights. The native library exposes the existing
LLM model classes and C++ token callback interface; it does not yet have a native
checkpoint loader or a speech-model interface.

The current Python module owns one process-global model/decoder session. It
releases the GIL during native generation and reacquires it for Python callbacks.
Overlapping generation, initialization and device mutation are rejected before
changing that session. Native callbacks are joined before return, and callback
exceptions retain their Python type and traceback at the binding boundary.
Applications must serialize operations on a native engine instance; independent
concurrent sessions and stream-safe global CUDA pool use remain future work.

## Future speech models

[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) includes an audio tokenizer/codec
and streaming speech generation. Transformer primitives can be shared, while
speech-specific execution and audio decoding need separate runtime components.
An output API will need audio chunks, sample-rate metadata and cancellation or
backpressure. Add these contracts when a concrete integration requires them;
The [speech integration plan](speech-integration.md) defines the migration
sequence from a Torch reference. TTS execution and waveform output are future
work; the shared foundation is already used by the language-model runtime.
