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
| `quantization/` | Prepare AutoAWQ GEMV/GEMM weights | AutoAWQ, PyTorch, Transformers, Accelerate, safetensors; source model weights |
| `validation/reference_generation.py` | Run Transformers generation and print elapsed time | PyTorch, Transformers, Accelerate; full model weights |
| `validation/test_speculative_decoding.py` | Compare standard/speculative generation and collect logits | Built `model_bridge`, CUDA device, PyTorch, safetensors, Transformers; target/draft model directories |

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

## Validation scope

Run `bash scripts/test.sh` for automated portable workspace and operator checks;
see the root README for the equivalent CMake commands. CUDA operator tests require
a CUDA-enabled build and a CUDA device. The scripts here are manual diagnostics:
timing output, printed comparisons, and generated text do not establish end-to-end
model correctness, sampling correctness, or a reproducible benchmark.
