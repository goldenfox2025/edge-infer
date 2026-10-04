# Native runtime integration

`EdgeInfer::runtime` is a C++17 static library. It links the independent operator
library, CUDA runtime/driver, cuBLAS and native threads. It does not depend on
Python, pybind11, Torch or a tokenizer. The current implementation is for Linux
or WSL with a supported CUDA toolchain.

## CMake source integration

Use the complete checkout, including its recorded CUTLASS submodule:

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
InferenceEngine<__nv_bfloat16> engine(model, Device::CUDA);
engine.generate_with_callback(
    input_ids, 128, 1.0f, 0.9f, 1,
    [](uint32_t token) { /* consume a token ID */ });
```

`max_length` is the existing total sequence-length limit, including prompt
tokens. Model EOS terminates generation; the ordinary engine does not emit the
EOS token to its callback. Applications decode token IDs separately.

## Lifetime and concurrency

The ordinary engine runs model generation on a worker and invokes token
callbacks on the calling thread. Generation joins its worker before returning,
including on callback/worker exceptions. A callback failure currently waits for
the worker to finish; early cancellation and bounded buffering are future work.
Serialize calls and state changes on an engine instance. Simultaneous full
generation using legacy sampling scratch has not been validated.

Python bindings manage the GIL outside this library. Their current process-global
session rejects overlapping generation, initialization or device changes. Python
callbacks retain their original exception type, value and traceback.

The token-oriented API does not model audio frames or PCM chunks. A Qwen TTS
implementation can consume the same core/operators and execution mechanisms
while introducing speech-specific model state and outputs as described in the
[speech integration plan](speech-integration.md).

## Shared Qwen3 weights and dedicated execution

For direct logits integration, prepare weights once and create separate native
sessions. Each session binds to one fixed-capacity cache; the cache owns private
CUDA allocations and logical resize never moves them.

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
use a transposed view. Token embeddings must be contiguous `[vocabulary, hidden]`,
with hidden width divisible by eight. The current attention path requires BF16
and head dimension 128.

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

The token factory returns a Qwen3 session through the legacy `BaseModel`
interface. Python continues to expose one guarded session. Native interleaved
logits/KV isolation and greedy sampling are tested with synthetic weights;
full checkpoint parity, asynchronous generation and probabilistic sampling
isolation still require separate validation.
