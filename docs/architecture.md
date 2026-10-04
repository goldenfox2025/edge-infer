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
their existing eager facade methods use these calls. `conditioning.hpp` adds
concrete float/BF16 linear projections and multi-table embedding sums with
caller-owned FP32 scratch. Speech model code composes these directly, without
factory lookup or virtual dispatch. Existing text matmul adapters and
prepared-node execution retain dynamic dispatch.

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
LLM model classes and C++ token callback interface. `runtime/include/speech/`
adds a separate typed conditioning API. Native checkpoint loading and complete
speech decoding remain future work.

The current Python module owns one process-global model/decoder session. It
releases the GIL during native generation and reacquires it for Python callbacks.
Overlapping generation, initialization and device mutation are rejected before
changing that session. Native callbacks are joined before return, and callback
exceptions retain their Python type and traceback at the binding boundary.
Applications must serialize operations on a native engine instance; independent
concurrent sessions and stream-safe global CUDA pool use remain future work.

## Adding a model

The extension target is a model implementation (`.cpp` and its public `.hpp`),
typed configuration and weight mapping, plus a small build/registration entry.
A model covered by existing operations should not require edits to core memory
code, operator implementations, execution scheduling or generic bindings.
Architecture variants that only change dimensions or checkpoint names should
reuse an existing model implementation through configuration and weight mapping.

Keep model sequencing and variant-specific decisions in its runtime component.
Use concrete operator calls in the hot path; registration happens once at
initialization. When a model introduces a genuinely new operation, add that
generic primitive to operators with a numerical reference. Avoid duplicating
kernels per model or adding speculative plugin machinery. Model-level parity
fixtures accompany new implementations.

The current text factory and per-model Python weight processors still require
explicit edits when registering a new language model. The embedding-to-hidden
backbone and declarative weight mapping are the next steps toward the extension
target; a single-file model addition is not yet guaranteed by the current API.

## Dedicated sessions and fixed decode storage

The intended runtime separates an immutable model (configuration, weights and
read-only prepared data) from mutable session state. Sessions share model weights
and each owns its KV cache, current positions, decode workspace, sampling state,
CUDA stream/events, cuBLAS handle and graph execution state. Model classes
compose operators; session management does not belong in the operator library.

Prepare each session once with explicit model/precision, maximum context, batch
capacity and decode workspace sizes. Allocate KV capacity and decode scratch
before generation; addresses remain stable during each session's decode loop.
Capacity exhaustion returns an error or ends the session instead of silently
reallocating while graph pointers remain live. Reset positions and transient
state for a new request after outstanding GPU work completes; reuse allocations.
Release the session only after its streams and consumers finish.

Fixed decode scratch and persistent KV storage have different size rules. A
one-token decode shape can reuse scratch, while KV storage is reserved up to a
configured maximum context. Prefill needs a separate bounded budget or prompt
chunks, and TTS needs a persistent talker cache plus a short predictor cache
reset each frame. Sequential talker/predictor scratch can share a session-local
arena when their lifetimes are proven disjoint. Waveform buffers and codec state
are accounted for when the codec is implemented.

For batch-one dense KV storage, bytes are
`2 * layers * context_capacity * kv_heads * head_dim * sizeof(dtype)`.
The recorded 0.6B TTS talker uses 112 KiB per BF16 context position: a 4096-position
session reserves 448 MiB, while 32768 positions reserve 3.5 GiB, before weights
and scratch. Its five-layer code predictor needs at most 16 cached positions
for a frame's residual prediction, or 320 KiB BF16 KV. Reserve these actual
session bounds instead of each stage's checkpoint maximum position setting.

Device memory is budgeted as shared model weights plus each session's KV,
workspace and auxiliary storage, plus the permitted concurrent prefill budget.
Separate streams provide isolation and potential overlap; faster total decoding
requires measurements on the selected GPU and concurrent session count.

The existing Qwen3 path already preplans a fixed one-token workspace, but it is
owned by the model object and uses the global CUDA pool. The current bindings
also own one global session. Moving mutable state and allocator/graph resources
to explicit sessions is required before claiming concurrent full-model support.
Current Qwen3 graph capture binds attention to a particular KV cache, and
global allocation tags can resolve different model instances to the same
storage. Each session therefore needs its own captured graph and private arena;
removing the binding's busy guard alone would make these shared states unsafe.
The new speech conditioning stage already accepts caller-owned buffers, handles
and streams and stores only immutable configuration/borrowed weights.

## Speech models

[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) includes an audio tokenizer/codec
and streaming speech generation. The first native stage is
`edge_infer::speech::QwenTtsConditioner<T>`: text projection and composition of
audio-codebook embeddings with aligned text. Model dimensions and borrowed
weights belong to this runtime component; its generic linear and embedding
operations belong to operators. Torch is used only by the developer reference
exporter, never by this native stage.

The next shared transformer boundary should accept `inputs_embeds` and return
normalized hidden states with explicit caller-owned cache state. Token lookup
and task-specific output heads belong outside that backbone. The current text
`BaseModel` returns sampled token IDs and is not the speech execution contract.
Qwen TTS needs a talker hidden state and first code, then a short per-frame
predictor session with a different output head for each residual codebook.
Its explicit attention head dimension must remain independent of hidden width.

The [speech integration plan](speech-integration.md) records the pinned model
configuration and staged migration. A later waveform output API will need
chunk ownership, sample-rate metadata, cancellation and bounded buffering;
these are added with the codec and streaming implementation.
