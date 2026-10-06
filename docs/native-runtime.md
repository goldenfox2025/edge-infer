# Native runtime integration

`EdgeInfer::runtime` is a C++17 static library. It links the independent operator
library, CUDA runtime, cuBLAS and native threads. It does not depend on
Python, pybind11, Torch or a tokenizer. The current implementation is for Linux
or WSL with a supported CUDA toolchain.

## CMake source integration

The runtime build uses CUDA/cuBLAS and C++17. The maintained source does not
require CUTLASS or historical Tensor/operator-factory adapters.

```cmake
cmake_minimum_required(VERSION 3.20)
project(my_edge_app LANGUAGES CXX)

set(EDGE_INFER_BUILD_RUNTIME ON CACHE BOOL "Build native runtime")
set(EDGE_INFER_BUILD_PYTHON OFF CACHE BOOL "Build Python bindings")
set(BUILD_TESTING OFF CACHE BOOL "Build dependency tests")
set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "Target GPU architecture")

add_subdirectory(/absolute/path/to/edge-infer edge-infer-build)
add_executable(my_edge_app main.cpp)
target_link_libraries(my_edge_app PRIVATE EdgeInfer::runtime)
```

Select the compiler and CUDA architecture for the actual device. This is source
integration; an installed `find_package(EdgeInfer)` package is not provided.

## Prepared weights and generation

The existing factory consumes `ModelConfig` and native Tensor maps. Weight names,
layouts, quantization packing and configuration must follow the native model's
contracts; a raw checkpoint map is not automatically compatible. The Python
weight processors currently perform these conversions. Native checkpoint loading
and tokenization still need an implementation or an application-provided adapter.

With correctly prepared BF16 Qwen3 weights, configuration and input token IDs,
the native call boundary is:

```cpp
#include "inference.hpp"
#include "model_factory.hpp"

// weights, config and input_ids are supplied by the application.
auto model = ModelFactory::create_model_bf16(
    ModelType::QWEN3_BF16, weights, config);
InferenceEngine<__nv_bfloat16> engine(model, Device::CUDA, 1024);
engine.generate_with_callback(
    input_ids, 128, 1.0f, 0.9f, 1,
    [](uint32_t token) { /* consume a token ID */ });
```

`max_length` is the existing total sequence-length limit, including prompt
tokens. Model EOS terminates generation; the ordinary engine does not emit the
EOS token to its callback. Every high-level generation call starts from the
complete supplied prompt and replaces the previous request. Applications retain
and render conversation history themselves. Low-level `prefill()` and `decode()`
remain available for explicit continuation. Applications decode token IDs separately.

The third engine constructor argument bounds the KV cache. A positive capacity
must fit the model context limit; zero uses the model limit. SpeculativeDecoder
accepts capacity as its fifth constructor argument, after thread count, and
requires capacity within both models' limits. It verifies exact greedy tokens
only (`top_k=1`); stochastic sampling belongs to an ordinary engine.

## Lifetime and concurrency

The ordinary engine runs model generation on a worker and invokes token
callbacks on the calling thread. Generation joins its worker before returning,
including on callback/worker exceptions. A callback failure currently waits for
the worker to finish; early cancellation and bounded buffering are future work.
Serialize calls and state changes on an engine instance. Each engine forks
independent mutable execution state while retaining prepared CUDA weights.
Interleaved session isolation is tested; simultaneous GPU throughput requires
separate measurement.

`BaseModel::synchronize()` completes an executor's submitted work before engine
reset or warmup reuses state. Its default waits for the selected CUDA device;
Qwen3 sessions override it to wait for their private stream, and Qwen models
delegate to their session. Input transfers, migration and engine teardown may
still use default-stream or device-wide waits.

Engine device migration is an explicit compatibility operation. If `cuda()` or
`cpu()` fails, that engine is permanently invalidated: generation, reset and
state access then reject with an error. Construct another engine rather than
reusing partially migrated state. Prepared CUDA models do not support CPU migration.

Python bindings manage the GIL outside this library. Explicit Python models
create independent bounded sessions; each session rejects overlapping generation
or reset. The procedural compatibility wrapper also rejects replacement while
its generation is active. Python callbacks retain their original exception type,
value and traceback. See the [Python API](python-api.md).

The token-oriented API does not model audio frames or PCM chunks. A Qwen TTS
implementation can consume the same core/operators and execution mechanisms
while introducing speech-specific model state and outputs as described in the
[speech integration plan](speech-integration.md).

## Shared Qwen3 weights and dedicated execution

For direct logits integration, prepare weights once and create separate native
sessions. The managed API creates a matching fixed-capacity KV cache and owns
its lifetime:

```cpp
#include "qwen3.hpp"

using BF16 = __nv_bfloat16;
using Session = Qwen3Session<BF16>;
auto model = std::make_shared<Qwen3Model<BF16>>(weights, config);
auto first = Session::create(model, 4096);
auto second = first->new_session(1024);  // Shared weights, empty history.

// Contiguous rank-one CUDA uint32_t tensors; next_input contains one token.
auto prompt_logits = first->prefill(prompt);
auto decode_logits = first->decode(next_input);
// A CPU token value is also accepted: first->decode(uint32_t{token});
const auto active_tokens = first->context_size();
const auto capacity = first->context_capacity();
first->reset();  // Empty history; retain KV and decode allocations.
```

The capacity must be positive and at most the model's position limit. Creation
shares prepared weights and allocates private KV and execution buffers.
`prefill()` starts a fresh history; `decode()` appends one token after prefill.
Capacity overflow and invalid inputs reject before model writes. The default
mode is eager; pass `true` as the final creation argument to use graph decode.
Prefill remains eager in either mode. Calls are synchronous and logits borrow
session storage until its next operation.

The application or harness chooses summaries, truncation and retrieved history.
After changing the token history, prefill the resulting tokens to rebuild the
corresponding KV state. The runtime enforces storage limits and positions.
KV quantization or other storage compression would be runtime mechanisms with
separate numerical validation.

The lower-level API still accepts a caller-owned cache. Each session binds to
one fixed-capacity cache; logical resize never moves its CUDA allocations:

```cpp
#include "qwen3.hpp"

using BF16 = __nv_bfloat16;
auto model = std::make_shared<Qwen3Model<BF16>>(weights, config);
const auto& dimensions = model->config();
const size_t capacity = 4096;  // Must not exceed max_position_embeddings.
KVCache<BF16> cache(dimensions.n_layers, capacity,
                    dimensions.n_kv_heads * dimensions.head_dim, Device::CUDA);
Qwen3Session<BF16> session(model, false);  // Explicit eager mode.

// prompt and next_input are contiguous rank-one CUDA uint32_t tensors.
cache.resize(prompt.numel());
auto prompt_logits = session.prefill_eager(&prompt, &cache);
cache.resize(cache.size() + 1);
auto decode_logits = session.forward_eager(&next_input, &cache);
```

Create another cache/session pair with the same model to serve another history.
The model owns copies of prepared weights, retaining supported contiguous or
two-dimensional transposed layouts. Required dimensions are checked before
execution. Dense linear weights have logical shape `[input, output]` and may
use a transposed view. Token embeddings must be contiguous `[vocabulary, hidden]`.
BF16 attention with head dimension 128 uses the optimized implementation.
Other supported head widths and FP32 use a scalar correctness fallback.

The current AWQ kernel consumes contiguous N-major packed weights:
`qweight[output, ceil(input/8)]`, `qzeros[output, ceil(groups/8)]` and
`scales[output, padded_groups]`, where `groups = input / group_size` and
`padded_groups >= groups`. The input width must divide into whole groups.
These are prepared native layouts; standard AWQ checkpoint layouts may need
conversion before construction.

Logits borrow session storage. Consume them before its next operation or copy
them into application-owned output. Calls complete GPU work before returning;
serialize access to one session. A cache must retain its object, device,
capacity and backing addresses throughout execution. After `cache.clear()`,
prefill starts another history in the same allocations. CUDA graph mode uses
the same binding. `session.set_graph_enabled(true)` selects graph decode for
the sampled `forward()` method. For logits, call
`forward_for_graph_logits_only()` to use graph decode; `forward_eager()` always
uses the eager decoder.

The token factory returns a Qwen3 session through the `BaseModel` interface.
Native and Python consumers can create independent bounded generation engines.
Native interleaved logits/KV isolation and greedy sampling are tested with
synthetic weights; full checkpoint parity, asynchronous throughput and
probabilistic sampling isolation still require separate validation.

## Host tokens and model adapters

Managed sessions accept host token IDs without an owning CUDA Tensor:

```cpp
auto first = Qwen3Session<__nv_bfloat16>::create(model, 1024);
auto second = first->new_session(512);
auto logits = first->prefill(std::vector<uint32_t>{3, 7, 11});
logits = first->decode(uint32_t{17});
```

Host prefill validates IDs and uploads on the session's stream. It grows private
input storage as needed; subsequent one-token decode uses fixed storage.

An embedding or output-head adapter can use the same transformer backbone:

```cpp
// prompt_embeddings and frame_embedding are contiguous CUDA TensorViews.
auto hidden = second->prefill_embeddings(prompt_embeddings, 0);
hidden = second->decode_embeddings(frame_embedding, next_logical_position);
// Apply the adapter's output head to hidden on second->stream().
```

Prefill returns `[rows, hidden_size]`; decode returns `[1, hidden_size]`, both
after final normalization. No token lookup, language-model head or sampling runs
in these embedding calls. They use eager execution even when token graph mode is
enabled. Input storage must be independent of session workspace. Establish stream
dependencies for uploads from another stream; the calls complete before returning.

Embedding positions are explicit and must fit the model's position limit. The
session determines physical KV slots independently. Token calls use their normal
cache-position sequence; callers continuing custom positions use the embedding
API. This is single-axis RoPE, not an implementation of TTS MRoPE.
