# Qwen TTS conditioning validation: 2026-10-04

This record covers the first native speech stage: text projection and composition
of codec embeddings with aligned text. It does not cover autoregressive talker
or code-predictor inference, prompt preprocessing or audio generation.

## Reference and inputs

Official Qwen3-TTS source: `022e286b98fbec7e1e916cb940cdf532cd9f488e`.
Checkpoint: `Qwen/Qwen3-TTS-12Hz-0.6B-Base` at
`5d83992436eae1d760afd27aff78a71d676296fc`. Reference export used PyTorch
2.6.0+cu124 on CUDA with TF32 disabled. The exporter executes only the official
`Qwen3TTSTalkerResizeMLP` class from that pinned Git object. The frame reference
uses the official `cat(...).sum(dim=1)` composition, then adds projected text.

Complete BF16 projection weights and biases were retrieved with exact HTTP byte
ranges; four real text rows and four rows per codec table were selected:

- Text IDs: `[0,77091,151671,151672]`.
- Codec IDs in each of 16 tables: `[0,17,2047,127]`.
- Projection weights: `[2048,2048]` and `[1024,2048]`, with both biases.
- Fixture inputs: four rows of text width 2048 and four audio frames of width
  1024; compact fixture codec IDs are remapped to `[0,1,2,3]`.

The downloaded BF16 subset is about 12 MiB of projections plus selected
embedding rows. No full checkpoint or tokenizer payload was downloaded for this
stage. The [reference metadata and fixture hashes](validation/qwen-tts-conditioning-source.json)
record exact source IDs/revisions and SHA256 values; model payloads are not
committed to this repository. This is a small deterministic real-weight check,
not broad model parity across inputs.

## Environment

WSL Ubuntu 24.04, x86-64; NVIDIA GeForce RTX 4070 Laptop GPU; driver 581.42;
native CUDA toolkit 12.0.140; CUDA architecture 89; GCC 12.4; CMake 3.28.3;
Debug configuration; Python 3.12.3 and pybind11 2.13.6 for optional bindings.
CUTLASS remained pinned at `853ad93d60b23b4f87bc46dfbc3c9ce757773ed7`.

## Results

| Check | Result |
| --- | --- |
| Native build, Python/pybind11 discovery disabled | 8/8 CTest tests passed |
| Native runtime with Python bindings | 9/9 CTest tests passed |
| Generic CUDA conditioning primitives | Float/BF16 linear and embedding-sum tests passed on the GPU |
| Official Base text projection, FP32 | Maximum absolute error `1.19209e-7` |
| Official Base codec sum plus text, FP32 | Maximum absolute error `1.19209e-7` |
| Official Base text projection, BF16 | Exact output match; maximum error 0 |
| Official Base codec sum plus text, BF16 | Exact output match; maximum error 0 |
| Shared conditioner with two caller states | Independent preallocated buffers, streams and handles; original and rotated rows checked separately |
| Offline Base/CustomVoice checkpoint contracts | 19 standard-library unittest cases passed |
| Native runtime compilation | All eight runtime sources compile without Python/pybind11 headers |

FP32 parity allows `1e-6 + 1e-5 * abs(reference)`; BF16 permits at most one
representable BF16 step. The observed values above are stricter than those
limits. No GPU test was skipped. The wrapper rejects text/output/workspace
overlap and hidden/workspace overlap before launches, avoiding corruption
across composed operations. Config/extent and downstream operator limits are
checked before execution.

Primitive tests additionally cover rectangular/odd shapes, biased and unbiased
projection, cancellation cases that expose premature BF16 rounding, 16 tables,
65-frame chunks, caller-owned host metadata lifetime, invalid IDs/extents and
overflow, explicit nonblocking streams, cuBLAS pointer/math mode, and primitive
float output/scratch exact alias. Compound runtime operations use stricter
disjointness contracts.

The caller-state isolation case reuses one immutable conditioner and weights.
Two callers enqueue both stages on different streams, each with its own cuBLAS
handle and preallocated intermediates, scratch and outputs, before either
readback. The second caller rotates text/code rows and reference outputs so
cross-caller state corruption would be visible. This checks only the
conditioning stage; it does not establish concurrent full-model sessions or a
throughput improvement.

The offline tests use attributed real configuration and safetensors headers for
both 0.6B variants. They validate shapes, BF16 dtype, explicit head dimensions,
source offsets, shard indexes/paths, truncation and malformed metadata. They do
not validate CustomVoice payload values. The GitHub portable workflow includes
these tests; CUDA and real-weight parity were run locally.

## Reproduction

Provide a complete local checkpoint and the separate official source checkout;
the tool does not download them. A previously prepared subset with the recorded
`source.json` may be used via `--weights-subset` instead of `--checkpoint`.

```sh
python3 tools/validation/qwen_tts_conditioning_reference.py \
  --checkpoint /absolute/path/to/Qwen3-TTS-12Hz-0.6B-Base \
  --reference-source /absolute/path/to/Qwen3-TTS \
  --model-id Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --model-revision 5d83992436eae1d760afd27aff78a71d676296fc \
  --output /absolute/path/to/conditioning-fixtures --device cuda

git submodule update --init --recursive -- cutlass
cmake -S . -B build-tts-native \
  -DEDGE_INFER_BUILD_RUNTIME=ON -DEDGE_INFER_BUILD_PYTHON=OFF \
  -DCMAKE_DISABLE_FIND_PACKAGE_Python=ON \
  -DCMAKE_DISABLE_FIND_PACKAGE_pybind11=ON \
  -DEDGE_INFER_QWEN_TTS_FIXTURES=/absolute/path/to/conditioning-fixtures \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-12 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 \
  -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON
cmake --build build-tts-native --parallel 2
ctest --test-dir build-tts-native --output-on-failure
ctest --test-dir build-tts-native -R '^qwen_tts_conditioning_parity$' -V
python3 -m unittest discover -s tools/validation/tests -p test_qwen_tts_checkpoint.py -v
```

The equivalent Python-enabled configuration enables
`EDGE_INFER_BUILD_PYTHON`, Python discovery and pybind11 discovery, and supplies
the Python executable/pybind11 CMake directory as in the
[native runtime check](validation-native-runtime-2026-10-04.md).

## Limits and next stage

The native stage borrows weights, device storage, handles and streams. It has no
Torch dependency or owned operand/workspace allocations. This establishes a
composition boundary, not an allocation-free or faster full-model decode loop.
The existing text runtime's global pool and single binding session still need
separation before concurrent full-model execution can be supported.

Talker/code predictor, aligned prompt construction, voice conditioning, codec,
waveform streaming, native checkpoint loading and Jetson validation remain
pending. Full language-model parity and other kernel/quantization checks are
also unaffected by this stage. Existing native linker and sub-second WSL
filesystem clock-skew warnings remained; the builds and tests succeeded.
See [speech integration](speech-integration.md) and [architecture](architecture.md)
for the next execution and dedicated-session boundaries.
