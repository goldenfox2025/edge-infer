# Developer tools

These scripts inspect local model files, compare diagnostic dumps, and run manual
experiments. Run the commands below from the repository root: quantization,
embedding comparisons, and logits visualization use paths relative to the current
working directory. They are outside the portable CTest suite.

| Directory | Purpose | Dependencies and inputs |
| --- | --- | --- |
| `analysis/` | Inspect weight shapes/AWQ layouts and compare tensor dumps | NumPy for binary comparisons; PyTorch and safetensors for model inspection |
| `analysis/visualize_logits.py` | Plot target/draft logits and KL divergence | NumPy, Matplotlib, Transformers; saved `logits_data/` and tokenizer files |
| `analysis/qwen3_inspect.py` | Load a Transformers model, inspect it, and generate text | PyTorch, Transformers, Accelerate; full model weights |
| `conversion/qwen_tts_checkpoint.py` | Export local Qwen3-TTS conditioning weight metadata | Python standard library; complete local safetensors checkpoint |
| `quantization/` | Prepare AutoAWQ GEMV/GEMM weights | AutoAWQ, PyTorch, Transformers, Accelerate, safetensors; source model weights |
| `validation/reference_generation.py` | Run Transformers generation and print elapsed time | PyTorch, Transformers, Accelerate; full model weights |
| `validation/test_speculative_decoding.py` | Compare standard/speculative generation and collect logits | Built `model_bridge`, CUDA device, PyTorch, safetensors, Transformers; target/draft model directories |
| `validation/qwen_tts_conditioning_reference.py` | Export conditioning reference fixtures for native parity tests | PyTorch, NumPy, Git; local weights and the pinned official source checkout |

`requirements-runtime.txt` pins the frontend dependencies; install a suitable
PyTorch wheel separately. Matplotlib, Accelerate and AutoAWQ are optional tool
dependencies and are not included in that file. `device_map="auto"` in the
Transformers generation tools requires Accelerate; GPU execution also needs a
CUDA-enabled PyTorch wheel. AutoAWQ's dependency compatibility with the frontend
pins has not been validated; use a separate environment for quantization.

## Model paths

`analyze_qwen_model.py` and `qwen_model_structure.py` read `MODEL_PATH`, defaulting
to `<repository>/models/Qwen3-1.7B`. They use Qwen3 weight names and indexed
safetensors; `qwen_model_structure.py` also expects the original two-shard
filenames. `print_model_shapes.py` accepts a positional model directory, then falls
back to `MODEL_PATH` and the same default. It writes a shapes JSON file in the
current working directory.

```sh
MODEL_PATH=/absolute/path/to/Qwen3-1.7B python tools/analysis/analyze_qwen_model.py
python tools/analysis/print_model_shapes.py /absolute/path/to/model
python tools/analysis/qwen3_inspect.py --model_path /absolute/path/to/Qwen3-1.7B
python tools/validation/reference_generation.py --model_path /absolute/path/to/Qwen2.5-1.5B
```

The two Transformers tools accept `--model_path`, with `MODEL_PATH` as the next
choice. Their defaults are `<repository>/models/Qwen3-1.7B` for inspection and
`<repository>/models/Qwen2.5-1.5B` for reference generation. They also accept Hugging
Face model IDs. The inspection script retains its historical fallback to
`Qwen/Qwen3-1.7B` when local loading fails and uses `trust_remote_code=True`;
it may fetch model files and execute model-provided code.

Other tools retain their existing positional arguments or internal experiment
settings. Check them before running: quantization replaces its existing output
directory, and speculative validation uses model paths configured inside the
script. An alternate native build directory can be provided through `BUILD_DIR`.

## Qwen TTS checkpoint metadata

The standard-library inspector supports the official 12Hz 0.6B Base and
CustomVoice configuration profiles. Provide a complete local checkpoint with
`config.json` and `model.safetensors`, or an indexed set of safetensors shards.
Shard paths must stay inside the checkpoint directory. The inspector reads JSON
and safetensors headers, checks dtype/shape/byte extents, and exports source file
offsets for the conditioning weights. It performs no downloads, pickle loading,
tensor conversion or payload checksum verification.

```sh
python3 tools/conversion/qwen_tts_checkpoint.py /absolute/path/to/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --output /absolute/path/to/conditioning-manifest.json \
  --model-id Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --model-revision 85e237c12c027371202489a0ec509ded67b5e4b5 \
  --reference-revision 022e286b98fbec7e1e916cb940cdf532cd9f488e
```

`--output` defaults to standard output. The provenance options are optional;
revision arguments require full Git SHAs and record the supplied values. The
recorded Base checkpoint revision is
`5d83992436eae1d760afd27aff78a71d676296fc`. The manifest maps 21 conditioning
tensors: the text embedding, four projection weight/bias tensors, and 16 codec
embedding tables. It preserves explicit attention dimensions and variant
metadata while marking talker/code-predictor execution as future work.

Run the 19 offline contract tests with only Python:

```sh
python3 -m unittest discover -s tools/validation/tests \
  -p test_qwen_tts_checkpoint.py -v
```

The recorded fixture contains official configurations, weight-header metadata,
file sizes and source revisions. It contains no tensor payloads. These tests
exercise both variants, required tensors, dtype/shape checks, shard paths,
source offsets and malformed/truncated headers.

## Conditioning reference and native parity

The optional reference generator requires NumPy and PyTorch; `--device cuda`
requires a CUDA-enabled PyTorch wheel and an available NVIDIA GPU. Provide a
complete local checkpoint and an official Qwen3-TTS Git checkout containing
commit `022e286b98fbec7e1e916cb940cdf532cd9f488e`. The generator reads the
projection class from that recorded Git object. It does not install the
`qwen-tts` package or download source/weights.

```sh
python3 tools/validation/qwen_tts_conditioning_reference.py \
  --checkpoint /absolute/path/to/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --reference-source /absolute/path/to/Qwen3-TTS \
  --output /absolute/path/to/conditioning-fixtures \
  --model-id Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --model-revision 85e237c12c027371202489a0ec509ded67b5e4b5 \
  --device cuda
```

The output directory contains selected text/codec embedding rows and complete
projection tensors as little-endian `.f32` files, `codec_ids.u32`,
`dimensions.txt`, float32/BF16 expected outputs, and `source.json` with reference
provenance and fixture hashes. `--device cpu` is available for reference export.
`--model-id` and `--model-revision` record supplied local checkpoint provenance;
they do not verify a payload checksum or the publisher's revision.
`--weights-subset` accepts a previously prepared BF16 subset instead of
`--checkpoint`; that subset must include the recorded `source.json` and tensor
files expected by the generator.

Use the [CUDA build prerequisites](../README.md#cuda-development-setup), initialize
the pinned CUTLASS submodule, and register the fixture directory for CTest:

```sh
cmake -S . -B build-tts-conditioning \
  -DEDGE_INFER_BUILD_RUNTIME=ON -DEDGE_INFER_BUILD_PYTHON=OFF \
  -DEDGE_INFER_QWEN_TTS_FIXTURES=/absolute/path/to/conditioning-fixtures \
  -DCMAKE_CUDA_ARCHITECTURES=89 -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build-tts-conditioning --target qwen_tts_conditioning_parity --parallel 2
ctest --test-dir build-tts-conditioning -R '^qwen_tts_conditioning_parity$' --output-on-failure
```

Choose the CUDA architecture and compiler for the actual device. The parity
test checks the two-layer SiLU text projection and codec-frame embedding
composition in float32/BF16. It skips when no CUDA device is available; a skip
does not establish parity. Autoregressive talker/code prediction, the audio
codec and waveform generation remain beyond this conditioning stage.

## Validation scope

Run `bash scripts/test.sh` for automated portable workspace and operator checks;
see the root README for the equivalent CMake commands. CUDA operator tests require
a CUDA-enabled build and a CUDA device. The checkpoint contract tests and
conditioning parity check cover their stated boundaries. The remaining model
diagnostics are manual: timing output, printed comparisons, and generated text
do not establish end-to-end model correctness, sampling correctness, or a
reproducible benchmark.
