# Qwen TTS integration

edge-infer shares `core -> operators -> runtime -> bindings` between language
and speech workloads. Keep tensor storage and lifetimes in core, generic tensor
algebra in operators, model execution in runtime, and Python objects and
preprocessing at the frontend/binding boundary.

## Reference and checkpoint contract

The initial architecture target is Qwen3-TTS 12Hz 0.6B Base and CustomVoice.
The first numerical reference uses Base weights; CustomVoice has the same
conditioning layouts, checked through its configuration and safetensors header.

| Input | Recorded revision |
| --- | --- |
| [Official source](https://github.com/QwenLM/Qwen3-TTS) | `022e286b98fbec7e1e916cb940cdf532cd9f488e` |
| [0.6B Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) | `5d83992436eae1d760afd27aff78a71d676296fc` |
| [0.6B CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice) | `85e237c12c027371202489a0ec509ded67b5e4b5` |
| [12Hz audio tokenizer](https://huggingface.co/Qwen/Qwen3-TTS-Tokenizer-12Hz) | `7dd38ad4e9bad454aae9cd937d0cd577604fe229` |

These upstream resources are Apache-2.0; their terms do not supply a license for
edge-infer itself. The exporter reads the pinned official projection class from
a separate reference checkout. Upstream code and weights are not bundled in the
engine. Offline tests include attributed JSON configuration and safetensors
metadata, with no tensor payloads.

| Actual 0.6B field | Value |
| --- | --- |
| Text vocabulary / embedding width | 151936 / 2048 |
| Text projection | Biased Linear 2048 -> 2048, SiLU, biased Linear 2048 -> 1024 |
| Talker hidden width / layers | 1024 / 28 |
| Predictor hidden width / layers | 1024 / 5 |
| Query heads / KV heads / explicit head dimension | 16 / 8 / 128 |
| Q / KV projection widths | 2048 / 1024 |
| Audio codebooks | 16; primary vocabulary 3072, residual vocabularies 2048 |
| Codec embedding width | 1024 |
| Talker position encoding | Interleaved MRoPE, sections [24,20,20], theta 1000000 |
| Audio output sample rate | 24000 Hz |

`head_dim` is **not** `hidden_size / num_attention_heads`. The existing Qwen3
text path cannot be reused unchanged. Preserve nested configurations, positions
and group-specific output heads instead of encoding speech settings in the
text model's flat numeric configuration. Official 0.6B CustomVoice does not
enable instruction control; this milestone does not promise that feature.

Capture deterministic stage inputs and outputs where possible. A listening
comparison alone cannot establish numerical correctness. Conversely, logits
agreement alone does not establish waveform quality or continuous streaming.

## Implemented conditioning stage

`runtime/include/speech/qwen_tts_conditioning.hpp` exposes concrete float/BF16
stages:

- `project_text`: biased Linear -> SiLU -> biased Linear, including model-dtype
  rounding at each boundary.
- `compose_frame_embeddings`: sum all codec-table lookups in FP32, convert once
  to the model dtype, then add optional aligned projected text or a projected
  `tts_pad` embedding in that dtype.

Weights keep contiguous PyTorch `[out,in]` layout. The caller supplies device
storage, FP32 workspace, a cuBLAS handle and a stream. Host code IDs use
`[frames,codebooks]` order; each table retains its own vocabulary size. The stage
owns no model weights or operand/workspace allocation and does not use virtual
dispatch. Buffer lifetimes and disjointness are part of its public contract.
CUDA launches and cuBLAS still have their usual overhead.

The checkpoint tool maps 21 conditioning tensors from a complete local Base or
CustomVoice checkpoint and records offsets, dtype, shapes and provenance. It
checks headers and file extents without reading/converting payload values.
It is not a native checkpoint loader. See [developer tools](../tools/README.md)
for the manifest and reference fixture commands.

The first parity check uses four selected real text rows, four rows from each
codec table and complete text projection weights from the pinned Base checkpoint.
This validates conditioning, not prompt construction, transformer inference,
voice cloning or waveform quality. Results and commands are in the
[validation record](validation-qwen-tts-2026-10-04.md).

## Component boundary

| Component | edge-infer boundary | Integration work |
| --- | --- | --- |
| Tensor storage, CUDA buffers and workspace lifetimes | `core/` | Reuse ownership and stream contracts; verify memory budgets |
| Dense projections, normalization, activations, RoPE and attention | `operators/` | Match dtype/layout/semantics and add references for each reused path |
| KV-cache and graph execution | `runtime/` execution utilities | Share mechanisms while giving each speech stage its own state |
| Talker and code-prediction stages | Speech model code in `runtime/` | Implement checkpoint-specific sequencing, embeddings, prediction heads and stopping rules |
| Audio codec | Codec model in `runtime/`, primitives in `operators/` | Implement required convolution/upsampling/decoding operations from the selected codec |
| Checkpoint conversion and text/reference-audio processing | Conversion tools and frontend | Export a recorded native weight/layout contract and match Torch preprocessing |
| Waveform streaming | Runtime output and frontend/bindings | Define chunk ownership, sample-rate metadata, cancellation and bounded buffering |

Add, multiply, SiLU, RMSNorm, linear and embedding sums have direct GPU tests. Other existing
kernels are candidates for reuse, with broader numerical validation still
pending. Keep speech state out of operator implementations and keep Python
objects/GIL handling out of the native runtime.

The existing text-oriented `BaseModel` interface returns sampled token IDs. It
must not be assumed to represent multi-codebook audio frames or waveform chunks.
Introduce speech-specific stage/output contracts with their actual consumers;
avoid a generic modality framework before those requirements are established.

The implemented `Qwen3Session::prefill_embeddings()` and `decode_embeddings()`
boundary accepts contiguous CUDA embeddings and returns borrowed normalized
hidden states through the shared eager decoder. Each managed session owns its
bounded KV cache; callers supply logical position offsets independently of
physical cache slots. Lookup, output heads and sampling stay outside these
calls. This is a reusable execution boundary, not a completed speech model:
prepared weights and positions still follow the text backbone's contract, and
the talker requires MRoPE and stage-specific weights/heads.

A talker step produces a hidden state and primary
code; the predictor starts with `[talker hidden, primary-code embedding]` and
predicts residual groups with their own heads. Its cache resets each frame,
while the talker's cache continues across frames. Later variants with different
predictor/talker widths require the biased `small_to_mtp_projection` adapter.

The official talker path constructs three identical position axes in some
generation cases. Test plain-RoPE equivalence under explicit input restrictions
before using it as an optimization; preserve the recorded MRoPE configuration.

## Migration sequence

1. Conditioning is implemented. Extend deterministic references to actual
   prompt/preprocessing outputs and both supported variants before exposing
   user-facing generation.
2. Connect talker and code-prediction adapters to the implemented
   embedding-to-hidden boundary, adding MRoPE and stage-specific prepared weights
   and heads. Compare
   deterministic logits/codes with references and verify cache reset, growth,
   sequence boundaries and stopping rules.
3. Keep the Torch codec temporarily as a bridge to audible outputs while the
   native transformer stages are validated. Measure transfer costs explicitly.
   This mixed implementation is a development step, not a fully native result.
4. Implement and validate the native codec against identical audio-code inputs.
   Establish waveform error tolerances and check audible quality.
5. Add streaming tests for chunk boundaries, overlap/state, cancellation, slow
   consumers and repeated requests. Validate long inputs and conditioning modes
   supported by the selected checkpoint.
6. Compare repeated measurements on the same NVIDIA device: first-audio latency,
   real-time factor, peak/resident memory and steady streaming behavior. Record
   warmup, clocks/power settings and full commands.
7. Port the validated path to a chosen Jetson board and matching JetPack release.
   Verify ARM64 builds, memory/precision constraints and sustained thermals.

The current native stage has no Torch dependency. Complete speech generation,
audio output, native loading and Jetson compatibility remain separate gates.
