# edge-infer

A C++17/CUDA inference library for language models on NVIDIA devices, with
shared prepared weights, independent bounded sessions and direct operators.
The maintained branch is `master`. The repository was previously named
`LLM_infer`; its history is preserved.

## Structure

```text
core/                Tensor ownership, fixed borrowed views, workspace planning
operators/           Independent CPU references and direct CUDA kernels
runtime/             Prepared models, decoder, sessions, KV cache and generation
bindings/python/     Optional model loading boundary and Model/Session bindings
frontend/            Local safetensors loading and terminal chat
tools/validation/    Reproducible checkpoint and reference checks
```

The compute library depends on CUDA/cuBLAS and portable core headers. It has no
model-runtime, Python, Tensor-ownership, operator-factory or global-pool dependency.
Operators receive fixed-rank borrowed views, explicit execution resources and
caller-owned scratch. Models compose those operators through one transformer
decoder, shared by prefill, eager decode and CUDA Graph decode.

A prepared model owns immutable weights. Each session owns its stream, handle,
KV cache and planned workspace. Decode storage is fixed; prefill storage grows
at a preparation boundary and is reused. Workspace lifetimes allow dead
intermediates to share storage. Alignment, peak prefill capacity and library
workspaces still consume memory.

See [architecture](docs/architecture.md), [native API](docs/native-runtime.md),
[Python API](docs/python-api.md) and [operators](operators/README.md).

## Implemented scope

| Area | Current scope |
| --- | --- |
| Language models | Qwen3 and Qwen2/Llama-style dense decoder paths; BF16 and AWQ CUDA operators |
| Sessions | Shared weights, independent caller-sized caches, reset/reuse and fixed decode storage |
| Generation | Greedy and top-k/top-p sampling; native speculation uses exact greedy verification |
| Speech | Qwen3-TTS text projection and audio-code embedding composition; embedding-to-hidden decoder boundary |
| CPU | FP32 model reference and portable operator references; no optimized heterogeneous executor yet |
| Platforms | Local verification on RTX 4070 Laptop / WSL Ubuntu 24.04; Jetson validation pending |

Full Qwen3-TTS talker, MRoPE, code predictor, codec and waveform generation are
not implemented. A new checkpoint variant can reuse configuration and weight
mapping. A new architecture may require additional operators and validation.
The harness decides history summarization and truncation; the runtime enforces
storage capacity and owns inference state.

The [runtime-contract validation record](docs/validation-runtime-contracts-2026-10-06.md)
describes current tests and their limits. Older measurements in `docs/` describe historical
revisions and are not current performance guarantees.

## Portable tests

Requirements: CMake 3.20+ and a C++17 compiler. No CUDA, Python or model download
is required.

```sh
cmake -S . -B build-cpu-tests \
  -DEDGE_INFER_BUILD_RUNTIME=OFF -DEDGE_INFER_BUILD_PYTHON=OFF \
  -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build-cpu-tests --parallel 2
ctest --test-dir build-cpu-tests --output-on-failure
```

On Linux/WSL, `bash scripts/test.sh` runs this workflow. The operator API also
builds independently:

```sh
cmake -S operators -B build-operators \
  -DEDGE_INFER_OPERATORS_ENABLE_CUDA=OFF -DBUILD_TESTING=ON
cmake --build build-operators --parallel 2
ctest --test-dir build-operators --output-on-failure
```

## Native CUDA build

Use a supported Linux CUDA/compiler combination and an NVIDIA driver. Local
verification uses CUDA 12.0, GCC 12 and compute capability 89. Native Windows
and Jetson/JetPack support have not been established.

```sh
cmake -S . -B build-native \
  -DEDGE_INFER_BUILD_RUNTIME=ON -DEDGE_INFER_BUILD_PYTHON=OFF \
  -DCMAKE_CXX_COMPILER=g++-12 -DCMAKE_CUDA_HOST_COMPILER=g++-12 \
  -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release
cmake --build build-native --parallel 2
ctest --test-dir build-native --output-on-failure
```

Applications can add this checkout with CMake and link `EdgeInfer::runtime` or
the independent `EdgeInfer::operators_cuda` target. Checkpoint parsing and
tokenization belong to the application/frontend; the native API accepts
validated Tensor weight maps and configuration.

```cpp
auto model = std::make_shared<Qwen3Model<__nv_bfloat16>>(weights, config);
auto first = Qwen3Session<__nv_bfloat16>::create(model, 1024);
auto second = first->new_session(512);

auto logits = first->prefill(std::vector<uint32_t>{3, 7, 11});
logits = first->decode(uint32_t{19});
first->reset();
```

Each session starts with empty history. `prefill` replaces history, `decode`
appends, and `reset` preserves allocations. Returned logits borrow session
storage until its next operation. Successful calls complete before returning;
applications serialize operations within a session. Failed completion makes
that session unusable; see native integration for recovery and borrowed-storage
lifetimes.

## Python and local chat

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-build.txt -r requirements-runtime.txt
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
PYTHON_BIN=python bash scripts/build.sh -- \
  -DCMAKE_CXX_COMPILER=g++-12 -DCMAKE_CUDA_HOST_COMPILER=g++-12 \
  -DCMAKE_CUDA_ARCHITECTURES=89
ctest --test-dir build --output-on-failure
```

The CPU Torch wheel is sufficient for host-side checkpoint loading; the native
module performs CUDA execution. Provide an existing local directory containing
configuration, tokenizer files and safetensors weights:

```sh
PYTHON_BIN=python bash scripts/run.sh \
  --model_path /absolute/path/to/Qwen3-0.6B \
  --model_type qwen3_bf16 --device cuda \
  --context_capacity 1024 --max_new_tokens 32 --top_k 1
```

`model_bridge.Model` prepares weights without allocating a generation cache.
`model.new_session(capacity)` creates independent state. Python
`session.generate` receives a complete fresh prompt and a total sequence-length
limit; see [API examples](docs/python-api.md). Loading never downloads models or
executes checkpoint-provided code.

## Development and validation

Keep code, comments and documentation in English. Add numerical operator checks
for new semantics and reference-model checks for new architectures. Record the
device, checkpoint revision, commands and measured scope before reporting
performance or compatibility. See [roadmap](docs/roadmap.md),
[speech integration](docs/speech-integration.md),
[CPU/GPU cooperation](docs/heterogeneous-inference.md) and [validation tools](tools/README.md).

CI runs portable tests and a GPU-free CUDA 12.0.1/GCC 12 build with CPU binding
integration. GPU tests use explicit skip code 77 when hardware is absent;
compilation and skipped tests do not establish GPU numerical correctness.
The [Docker recipe](Dockerfile) uses the same desktop toolchain. Its local build
status is recorded separately from native testing.

The project originated from the
[LearningInfiniTensor Rust exercises](https://github.com/LearningInfiniTensor/learning-lm-rs)
and evolved into this implementation. There is currently no top-level license;
licensing and upstream attribution remain a release decision.
