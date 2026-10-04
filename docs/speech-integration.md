# Qwen TTS integration

edge-infer is intended to share an inference foundation between on-device LLM
and speech workloads. The present native runtime implements language-model
paths. This document describes the integration work required for Qwen TTS;
it does not claim that a speech model or audio codec is already supported.

## Reference and checkpoint contract

Use a separate Torch Qwen TTS implementation as a reference; it is not bundled
in this repository. Before porting a model,
record the checkpoint variant/revision, tokenizer and codec revisions, model
configuration, preprocessing, precision, generation options, framework versions
and test inputs. Different Qwen TTS variants may need different conditioning.
Derive dimensions, codebook counts, sample rates and special-token rules from
the selected checkpoint rather than embedding assumptions in shared operators.

Capture deterministic stage inputs and outputs where possible. A listening
comparison alone cannot establish numerical correctness. Conversely, logits
agreement alone does not establish waveform quality or continuous streaming.

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

Add, multiply, SiLU and RMSNorm have direct GPU numerical tests. Other existing
kernels are candidates for reuse, with broader numerical validation still
pending. Keep speech state out of operator implementations and keep Python
objects/GIL handling out of the native runtime.

The existing text-oriented `BaseModel` interface returns sampled token IDs. It
must not be assumed to represent multi-codebook audio frames or waveform chunks.
Introduce speech-specific stage/output contracts with their actual consumers;
avoid a generic modality framework before those requirements are established.

## Migration sequence

1. Freeze Torch reference inputs and preprocessing; record intermediate
   embeddings, logits, audio codes and waveform outputs for selected examples.
2. Integrate the talker and code-prediction stages incrementally. Compare
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

The operator library and native runtime can support this work without depending
on Torch. Native checkpoint loading, speech execution and the codec are separate
deliverables beyond the current extraction.
