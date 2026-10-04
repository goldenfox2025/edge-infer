# Native runtime validation: 2026-10-04

This record covers the edge-infer rename, independent native runtime, Python
callback boundary and CUDA static-library linkage changes committed alongside
this document. The earlier restructuring is recorded separately at
[bbe9b42](validation-2026-10-04.md).

## Environment

WSL Ubuntu 24.04, x86-64; NVIDIA GeForce RTX 4070 Laptop GPU; driver 581.42;
CUDA toolkit 12.0.140; CUDA architecture 89; CMake 3.28.3. The native and binding
builds used GCC 12.4, Debug configuration, Python 3.12.3 and pybind11 2.13.6.
Portable and isolated binding tests used GCC 13.3. CUTLASS remained pinned at
`853ad93d60b23b4f87bc46dfbc3c9ce757773ed7`.

## Results

| Configuration | Result |
| --- | --- |
| Root portable, runtime/Python disabled | 3/3 CTest tests passed |
| Standalone portable operators | 3/3 CTest tests passed |
| Native runtime, Python discovery disabled | Build succeeded; 6/6 CTest tests passed |
| Native runtime plus Python bindings | Build succeeded; 7/7 CTest tests passed |
| Standalone Python callback tests, no CUDA | 1/1 CTest passed, covering eight unittest cases |
| Native callback test with `CUDA_VISIBLE_DEVICES=''` | Passed without a visible GPU |
| Python extension import/API smoke check | Passed for the six public generation/device/initialization entry points |
| Native runtime compile commands | All seven runtime sources had no Python or pybind11 include paths |

The native configuration includes the three portable tests, CUDA direct operator
test, Tensor-based CPU adapter test and native runtime callback test. The binding
configuration adds `python_callback_boundary_test`. The direct GPU test ran on
the RTX 4070 Laptop and was not skipped; its numerical scope remains the four
direct operations described in the earlier validation report.

The native callback test links the real runtime library with a deterministic CPU
model. It checks token ordering, caller-thread callbacks, worker completion/join,
and preservation of callback and worker exception types. It does not load weights
or validate an actual language-model forward pass.

The Python suite tests the production callback/operation helpers: caller-thread
and native-worker callbacks, GIL acquisition, original Python exception identity,
type, message and traceback, errors swallowed by native code, native exceptions,
reference lifetime, rejection of nested/concurrent mutation and guard cleanup
after callback failure. It uses a small test-only module with no model or CUDA
dependency.

## Reproduction

The following selects the recorded local CUDA toolchain. Choose compiler and
architecture values supported by the intended device/toolkit.

```sh
git submodule update --init --recursive -- cutlass
cmake -S . -B build-native \
  -DEDGE_INFER_BUILD_RUNTIME=ON -DEDGE_INFER_BUILD_PYTHON=OFF \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-12 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 \
  -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON
cmake --build build-native --parallel 2
ctest --test-dir build-native --output-on-failure
CUDA_VISIBLE_DEVICES='' ./build-native/runtime/runtime_callback_test

python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-build.txt
cmake -S . -B build-native \
  -DEDGE_INFER_BUILD_PYTHON=ON \
  -DPython_EXECUTABLE="$PWD/.venv/bin/python" \
  -Dpybind11_DIR="$(.venv/bin/python -m pybind11 --cmakedir)"
cmake --build build-native --parallel 2
ctest --test-dir build-native --output-on-failure
PYTHONPATH="$PWD/build-native" .venv/bin/python -c 'import model_bridge'
```

The callback suite also builds independently with no CUDA toolkit:

```sh
cmake -S bindings/python/tests -B build-bindings-tests \
  -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Debug \
  -DPython_EXECUTABLE="$PWD/.venv/bin/python" \
  -Dpybind11_DIR="$(.venv/bin/python -m pybind11 --cmakedir)"
cmake --build build-bindings-tests --parallel 2
ctest --test-dir build-bindings-tests --output-on-failure
```

The GitHub workflow runs both portable configurations, this isolated callback
suite and source syntax checks. GPU/native runtime validation above was local.

## Limits

Full model parity, AWQ and remaining kernel coverage, speculative distributions,
release performance, native checkpoint/tokenizer loading, Docker, Jetson and
speech execution/audio codec remain pending. The runtime still has existing
dynamic dispatch and global CUDA pool/session constraints documented in the
[architecture](architecture.md) and [native API](native-runtime.md).

Existing CUDA linker warnings and sub-second WSL filesystem clock-skew warnings
remained; builds and tests succeeded. These checks establish the tested build,
linkage and callback boundary, not general model, TTS or device compatibility.
