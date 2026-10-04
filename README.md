# edge-infer

A C++17/CUDA inference project for on-device language and speech generation.
The current implementation contains Qwen/Qwen3 language-model paths for NVIDIA GPUs;
Qwen TTS is a planned extension. NVIDIA desktops are the first validation target,
followed by a specific Jetson/JetPack configuration.

Shared memory utilities, independent operators and a native C++ runtime form the
inference foundation. Python bindings and the frontend are optional consumers.
The project explores BF16 and AWQ kernels, KV-cache management, CUDA Graphs and
workspace planning. It is experimental and has not established full model parity.

The repository began with the
[LearningInfiniTensor Rust inference exercises](https://github.com/LearningInfiniTensor/learning-lm-rs)
and evolved into this C++/CUDA implementation. Upstream exercises and external
libraries must be distinguished from the engine's own implementation.

## Current status

`master` is the maintained development branch. The repository was previously
named `LLM_infer`; its history remains intact. Historical source is preserved in
Git history and the `legacy-before-restructure` tag.

| Area | Status |
| --- | --- |
| Workspace planning and lifetime analysis | Standalone C++17 tests; no CUDA or Python dependency |
| Native C++ runtime | Separate `EdgeInfer::runtime` target; Python bindings are optional |
| CUDA operators and `model_bridge` | CUDA build and direct operator tests validated; full model correctness remains pending |
| Qwen/Qwen3 BF16 and AWQ paths | Implemented; not an established compatibility matrix |
| Speculative decoding | Experimental; probability rejection/resampling needs a correctness review |
| Desktop NVIDIA GPU | Current development target; historical measurements used RTX 4070 Laptop |
| Jetson ARM64 / JetPack | Planned validation target; not verified or supported by this README |
| Qwen TTS | Planned; shared transformer operators are available, speech execution and audio codec remain to be implemented |

Passing the workspace tests does not establish model correctness, GPU performance,
or Jetson support. The existing Tensor-based operator test still requires the CUDA build
because `Tensor` currently includes CUDA types. A CPU model product is not implied.

## Start with the workspace tests

Requirements: CMake 3.20 or newer and a C++17 compiler. From this checkout:

```sh
cmake -S . -B build-cpu-tests \
  -DEDGE_INFER_BUILD_RUNTIME=OFF -DEDGE_INFER_BUILD_PYTHON=OFF \
  -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build-cpu-tests --parallel 2
ctest --test-dir build-cpu-tests --output-on-failure
```

On Linux or WSL, `bash scripts/test.sh` runs the same workflow. It does not fetch
CUTLASS, install Python packages, download models, or initialize CUDA.

Recorded check on 2026-10-04: three portable tests passed in WSL Ubuntu 24.04,
using GCC 13.3 and CMake 3.28.3. The standalone CUDA library also built with
GCC 12.4 / CUDA 12.0; all five tests passed, including float/BF16 GPU checks
on RTX 4070 Laptop. See [validation](docs/validation-2026-10-04.md) for scope.
After the edge-infer rename and runtime extraction, the native configuration
passed 6/6 tests and the Python-enabled configuration passed 7/7. The isolated
callback suite passed eight cases without CUDA. See the
[native runtime validation](docs/validation-native-runtime-2026-10-04.md).
The Docker recipe was not built.

`scripts/test.sh` now runs these workspace unit tests instead of the historical matrix
benchmark. That benchmark remains a separate target after an engine build:

```sh
cmake --build build --target avx_matmul_bench --parallel 2
./build/operators/benchmarks/avx_matmul_bench 1 512 512 512 3 5
```

The historical target name is retained; it does not establish AVX acceleration
or GPU inference performance.

## CUDA development setup

Use a Linux toolchain supported by the selected CUDA toolkit. The desktop
container recipe uses CUDA 12.6.3 and Ubuntu 24.04. These are recipe inputs,
and this CUDA 12.6 container configuration has not been validated. The recorded
full engine build used CUDA 12.0; see the validation report above.

Prerequisites: a suitable NVIDIA driver, CUDA toolkit with cuBLAS, CMake,
a supported C++ compiler, Python development headers and a Python environment.
The engine contains Linux-specific host code; native Windows support is not
established. The compiler is selected by CMake instead of a hard-coded path.

```sh
git submodule update --init --recursive -- cutlass
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-build.txt -r requirements-runtime.txt
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
PYTHON_BIN=python bash scripts/build.sh -- -DCMAKE_CUDA_ARCHITECTURES=89
ctest --test-dir build --output-on-failure
```

The torch CPU wheel is sufficient for the current frontend's host-side weight
loading; CUDA execution happens in the native module. Model quantization tools
have additional dependencies and are outside this quickstart. Python versions
and runtime dependency pins still need frontend and model-level validation.
The Python 3.12 binding build and import have been checked.

`89` is the original Ada GPU target. Select the architecture for the actual
device using `CMAKE_CUDA_ARCHITECTURES`; changing this number alone does not
adapt the engine to Jetson. Jetson needs a matching JetPack/CUDA/compiler stack,
ARM64 dependency wheels, model checks and measurements on the device.

CUTLASS is pinned by the repository's gitlink. `scripts/build.sh` initializes that
checkout when its headers are absent. An exported source tree can use an
existing pinned checkout through `CUTLASS_DIR=/path/to/cutlass`. Build directories
are reused; `--clean` invokes the build system's clean target.

## Native C++ runtime

Build the native runtime without Python, pybind11 or frontend dependencies:

```sh
git submodule update --init --recursive -- cutlass
cmake -S . -B build-native \
  -DEDGE_INFER_BUILD_RUNTIME=ON -DEDGE_INFER_BUILD_PYTHON=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Debug
cmake --build build-native --parallel 2
ctest --test-dir build-native --output-on-failure
```

`bash scripts/build.sh --native` selects the same dependency boundary. Use a
compiler supported by your CUDA toolkit. An application adding this checkout
as a CMake subdirectory can link `EdgeInfer::runtime`; the target exports the
runtime and core headers and its CUDA/operator dependencies.

`model_factory.hpp` accepts native Tensor weight maps and `ModelConfig`, and
`infer_base` streams token IDs through a C++ callback. Checkpoint loading and
tokenization still belong to the current Python frontend. Python GIL handling
is confined to the bindings. This creates a usable C++ integration boundary;
it does not provide a native checkpoint loader or TTS model implementation.
See [native runtime integration](docs/native-runtime.md) for CMake usage,
prepared-weight contracts and callback lifetimes.

## Model smoke test

After validating the full engine build, provide a local model directory with
`config.json`, tokenizer files and safetensors weights. No model is downloaded
by the build. This is a suggested manual smoke test, not a recorded passing run:

```sh
PYTHON_BIN=python bash scripts/run.sh \
  --model_path /absolute/path/to/Qwen3-1.7B \
  --model_type qwen3_bf16 --device cuda \
  --top_k 1 --temperature 1.0 --max_length 32
```

Compare greedy outputs/logits with a reference implementation before making
performance claims. AWQ and speculative decoding need separate numerical and
sampling-distribution checks.

## Desktop container

Initialize CUTLASS first so the Docker build uses the recorded revision:

```sh
git submodule update --init --recursive -- cutlass
docker build --build-arg CUDA_ARCHITECTURES=89 -t edge-infer-dev .
docker run --rm -it --gpus all -v /absolute/path/to/models:/models:ro edge-infer-dev
```

The recipe copies `operators/` and its test sources, and does not clone a moving
CUTLASS revision. It has not been built as part of the standalone test check.
It is a desktop development image, not a Jetson image.

## Code map and next work

- `core/`: shared tensors, weight views, CUDA memory and portable workspace planning.
- `runtime/include/execution/`: model execution programs and CUDA workspace integration.
- `runtime/src/`: models, inference flow, KV-cache and CUDA Graph integration.
- `runtime/CMakeLists.txt`: the native `EdgeInfer::runtime` library.
- `operators/`: operator interfaces, CPU utilities, CUDA implementations and legacy kernels.
- `bindings/python/`: model initialization, weight conversion and Python bindings.
- `frontend/`: local model loading and terminal interaction.
- `operators/tests/`: portable workspace and static reference checks, plus CUDA-dependent Tensor tests.
- `tools/`: model analysis, quantization and manual reference checks.
- `scripts/`: build, test, run and profiling entry points.
- `docs/`: architecture, development directions and historical measurements.

Keep execution planning separate from model logic and CUDA kernel implementation.
The next release gates are repeatable CUDA CI, broader GPU operator coverage,
model-level greedy parity, a review of speculative resampling, and reproducible
benchmarks. Platform-specific changes should follow measurements on the chosen
NVIDIA device instead of speculative abstraction work.

## Extending to speech

Use a separate Torch Qwen TTS implementation as the reference for a future native
implementation; it is not bundled here. Reuse the shared operator, memory and execution
layers where tensor layouts and semantics match. Speech-specific model
sequencing, code prediction, codec operators and waveform output need their own
implementation; a text-model name change cannot provide them.

Start with staged comparisons against a pinned Torch checkpoint, then move the
talker and audio-codec stages into the native runtime. Measure time to first
audio, real-time factor, memory use and streaming continuity. See the
[speech integration plan](docs/speech-integration.md) for component boundaries
and validation gates.

## Measurements and license

The [2025 experiment report](docs/benchmarks-2025.md) is preserved as historical
material. It reported the best of five runs, without a complete baseline version
and command record. Its speedups have not been reproduced for the maintained revision and
are not current performance guarantees.

The project currently has no top-level license. Third-party components retain
their own terms; the engine's license and attribution need to be resolved before
a release intended for reuse.

## Standalone operator library

Build the portable API without the inference runtime, Python or CUDA:

```sh
cmake -S operators -B build-operators \
  -DEDGE_INFER_OPERATORS_ENABLE_CUDA=OFF -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build-operators --parallel 2
ctest --test-dir build-operators --output-on-failure
```

This build uses the sibling `core/` directory. `operators_core` exposes
non-owning views and inline reference functions. `unified_operators` contains
CUDA kernels and the existing Tensor adapters. Direct CUDA calls are available for add, multiply, SiLU and RMSNorm.
Those eager facade calls avoid factory lookup and virtual dispatch. Matmul and
prepared-node execution retain dynamic dispatch. The static API adds no required
shared ownership or operand allocation; CUDA launch overhead still applies. See [architecture](docs/architecture.md) and [roadmap](docs/roadmap.md).

Keep documentation, comments and diagnostics in English. Model input data may
contain any language. Develop against `master`, and validate fused operations
against unfused references before replacing production calls.

## Profiling

Use the same model arguments as the run script; reports go to `data/` or the
`PROFILE_DIR` you provide. The tools need the appropriate profiler permissions
on your system.

```sh
bash scripts/profile.sh nsys --model_path /absolute/path/to/model --model_type qwen3_bf16
bash scripts/profile.sh ncu --model_path /absolute/path/to/model --model_type qwen3_bf16
```
