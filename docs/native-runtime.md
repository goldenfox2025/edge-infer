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
Serialize calls and state changes on an engine instance. Shared CUDA pool
concurrency and independent request sessions have not been validated.

Python bindings manage the GIL outside this library. Their current process-global
session rejects overlapping generation, initialization or device changes. Python
callbacks retain their original exception type, value and traceback.

The token-oriented API does not model audio frames or PCM chunks. A Qwen TTS
implementation can consume the same core/operators and execution mechanisms
while introducing speech-specific model state and outputs as described in the
[speech integration plan](speech-integration.md).
