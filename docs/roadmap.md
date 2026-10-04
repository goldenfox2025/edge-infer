# Development roadmap

`edge-infer` targets on-device language and speech inference. `master` is the
maintained branch. This list describes future work, in dependency order.

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
6. Pin a separate Torch Qwen TTS reference checkpoint and export staged
   inputs/outputs. Integrate talker/code prediction using shared operators where
   semantics match, then implement the audio codec and streaming output. See
   [speech integration](speech-integration.md). Measure time to first audio,
   real-time factor and peak memory alongside the Torch reference.
7. Validate language and speech on a specific Jetson board/JetPack release,
   including ARM64 dependencies, memory budgets, thermal limits and supported
   CUDA kernels. Select board-specific precision and fusion settings based on
   correctness and measurements.

Keep operators independently consumable. Introduce shared abstractions when
existing implementations require them.
