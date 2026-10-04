# Development roadmap

`master` is the maintained branch. This list describes future work.

1. Validate BF16 greedy inference against a pinned reference: logits, generated
   tokens, KV-cache growth and graph replay. Record model/tokenizer revisions,
   compiler, driver, CUDA toolkit and commands.
2. Validate AWQ packing, scales, zeros and fused kernels across supported shapes.
   Establish correctness before comparing performance.
3. Migrate runtime hot paths from factory/virtual dispatch to concrete backend
   calls where useful. Measure call overhead, allocations and end-to-end latency.
4. Review speculative rejection and residual resampling with distribution tests.
5. Publish repeated NVIDIA desktop benchmarks, including warmup, memory use,
   prefill/decode latency and complete reference configurations.
6. Validate a specific Jetson board/JetPack release, including ARM64 dependencies,
   unified memory, thermal limits and supported CUDA kernels.
7. Select a Qwen TTS checkpoint, map transformer/audio-codec components, establish
   an audio baseline, then implement streaming and measure time to first audio
   and real-time factor.

Keep operators independently consumable. Introduce shared abstractions when
existing implementations require them.
