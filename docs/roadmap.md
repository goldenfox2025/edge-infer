# Development roadmap

`edge-infer` targets on-device language and speech inference. `master` is the
maintained branch. Native runtime extraction and the first Qwen3-TTS conditioning
stage are implemented. The current speech milestone is an audible 0.6B path on
NVIDIA desktop GPUs; its remaining work is:

1. Connect speech adapters to the shared embedding-to-hidden-state backbone
   and private session storage, adding the talker's multi-axis positions. Validate
   the talker and code predictor separately against the pinned reference.
2. Implement prompt alignment, language/speaker tokens, per-frame predictor
   cache reset, group-specific heads and stopping rules for the chosen variant.
3. Connect the reference Torch codec for initial audible comparisons, record
   transfer costs, then migrate the codec and waveform output to native code.
4. Validate streaming boundaries, cancellation, long inputs and repeated
   requests; measure first-audio latency, real-time factor and memory use.

See [speech integration](speech-integration.md) and the
[first-stage validation record](validation-qwen-tts-2026-10-04.md). The following
language-model and platform work also remains:

1. Validate BF16 greedy inference against a pinned reference: logits, generated
   tokens, KV-cache growth and graph replay. Record model/tokenizer revisions,
   compiler, driver, CUDA toolkit and commands.
2. Validate AWQ packing, scales, zeros and fused kernels across supported shapes.
   Establish correctness before comparing performance.
3. Measure the shared direct decoder and private sampling workspaces on real
   checkpoints: call overhead, allocations, memory use and end-to-end latency.
4. Review speculative rejection and residual resampling with distribution tests.
5. Publish repeated NVIDIA desktop benchmarks, including warmup, memory use,
   prefill/decode latency and complete reference configurations.
6. Validate language and speech on a specific Jetson board/JetPack release,
   including ARM64 dependencies, memory budgets, thermal limits and supported
   CUDA kernels. Select board-specific precision and fusion settings based on
   correctness and measurements.

Keep operators independently consumable. Introduce shared abstractions when
existing implementations require them.
