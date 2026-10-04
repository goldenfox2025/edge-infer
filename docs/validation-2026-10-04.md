# Validation record: 2026-10-04

This record covers the initial repository restructuring committed alongside this
document. The source before restructuring is preserved at
`legacy-before-restructure` (`e8ed78ca5a3d94e366c7fa52bfc06f1ad6747a1f`).

## Environment

| Component | Recorded value |
| --- | --- |
| OS | WSL, Ubuntu 24.04, x86-64 |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU |
| NVIDIA driver | 581.42 |
| CUDA toolkit | 12.0.140 |
| CUDA architecture | 89 |
| CUDA host / engine compiler | GCC 12.4 |
| Portable compiler | GCC 13.3 |
| CMake | 3.28.3 |
| Build configuration | Debug |
| Python for bindings | 3.12.3 |
| pybind11 | 2.13.6 |
| CUTLASS | `853ad93d60b23b4f87bc46dfbc3c9ce757773ed7` (recorded gitlink) |

## Results and scope

| Check | Result |
| --- | --- |
| Root portable build | 3/3 CTest tests passed |
| Standalone portable operator build | 3/3 CTest tests passed |
| Standalone CUDA operator build | Build succeeded; 5/5 CTest tests passed |
| Full engine and Python module build | Build succeeded; 5/5 CTest tests passed |
| Python extension import | `model_bridge` imported and its public entry points were present |
| Header dependencies | 22 core/runtime headers compiled independently; 313 project quoted includes resolved |
| Python source syntax | 16 current sources parsed with Python 3.10 grammar |
| Repository checks | English source/documentation review and staged whitespace check passed |

The five-test CUDA configurations include the three portable tests
(`decode_workspace_plan_test`, `workspace_liveness_test`, `operators_core_test`),
the Tensor-based CPU adapter test (`unified_operators_cpu_test`), and the actual
GPU test (`operators_cuda_direct_test`). The GPU test ran on the device and was
not skipped. The portable configurations do not initialize CUDA or load Python.

Direct GPU numerical checks cover add, multiply, SiLU and RMSNorm for float and
BF16, odd/even lengths, multi-block tails (17, 18, 4098 and 4099 elements),
in-place operations, a nonblocking stream, empty inputs, and invalid extents.
RMSNorm includes feature dimensions 7 and 10240 and rejection outside the
supported range of 1 through 10240. Reference BF16 values are calculated with
decoded float arithmetic and rounded to BF16 for comparison.

The Python import check covered `init_model`, `generate_text_stream`,
`set_default_device`, `get_default_device`, `init_speculative_decoder` and
`generate_text_stream_speculative`. It did not load weights or generate tokens.

## Reproduction

Run from the repository root. Portable tests need only CMake and a C++17 compiler:

```sh
cmake -S . -B build-cpu-tests \
  -DLLM_INFER_BUILD_ENGINE=OFF -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build-cpu-tests --parallel 2
ctest --test-dir build-cpu-tests --output-on-failure

cmake -S operators -B build-operators-standalone \
  -DLLM_OPERATORS_ENABLE_CUDA=OFF -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build-operators-standalone --parallel 2
ctest --test-dir build-operators-standalone --output-on-failure
```

For the recorded CUDA configuration, install the toolkit and GCC 12, initialize
CUTLASS, and create a Python environment with development headers and pybind11.
The frontend dependencies are not needed for the extension import check.

```sh
git submodule update --init --recursive -- cutlass
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-build.txt

cmake -S operators -B build-operators-cuda \
  -DLLM_OPERATORS_ENABLE_CUDA=ON -DBUILD_TESTING=ON \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-12 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 \
  -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Debug
cmake --build build-operators-cuda --parallel 2
ctest --test-dir build-operators-cuda --output-on-failure

cmake -S . -B build-engine-final \
  -DLLM_INFER_BUILD_ENGINE=ON -DBUILD_TESTING=ON \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-12 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 \
  -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Debug \
  -DPython_EXECUTABLE="$PWD/.venv/bin/python" \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build build-engine-final --parallel 2
ctest --test-dir build-engine-final --output-on-failure

PYTHONPATH="$PWD/build-engine-final" python - <<'PY'
import model_bridge
required = (
    'init_model', 'generate_text_stream', 'set_default_device',
    'get_default_device', 'init_speculative_decoder',
    'generate_text_stream_speculative',
)
assert all(hasattr(model_bridge, name) for name in required)
assert model_bridge.get_default_device() in ('cpu', 'cuda')
print('Python extension import and API smoke check passed')
PY
```

The GitHub workflow runs both portable configurations and Python/shell syntax
checks. CUDA validation in this report was local; hosted portable CI does not
establish GPU support.

## Remaining validation

- Full model inference, reference logits, greedy token parity, KV-cache growth
  and CUDA Graph replay.
- AWQ numerical correctness, all legacy/fused kernels, and speculative sampling
  distributions.
- Release builds and performance reproduction, including overhead comparisons.
- Frontend dependency pins and the documented Docker image.
- Jetson ARM64 / JetPack and native Windows builds.
- Qwen TTS integration and audio generation; neither is implemented by this change.

Existing CUDA compiler/linker warnings and sub-second WSL filesystem clock-skew
warnings remained in the local build logs. Builds and tests succeeded. These
results establish the recorded build and tested operator behavior, not a general
model compatibility or performance guarantee.
