# Decoder refactor checkpoint: 2026-10-04

Work stopped before midnight Asia/Shanghai at the owner's request. This is a
local checkpoint, not a validated release. The existing remote master was not
updated with this checkpoint.

The refactor introduces fixed borrowed tensor views, independent direct CUDA
operators, one shared decoder backbone, private session arenas and requested-size
KV caches. Models share immutable prepared weights between independent sessions.
The Qwen2/Llama compatibility adapter and generation frontends use that backbone.
Default builds no longer require CUTLASS; historical operators remain opt-in.

## Verification completed

- Portable planner/view tests: 4/4 passed.
- Native Release build with Python disabled: succeeded on RTX 4070 Laptop,
  WSL Ubuntu 24.04, CUDA 12.0, GCC 12.4, architecture 89.
- Native CTest before the final fixes: 10/13 passed. AWQ, sampling, direct
  operators, conditioning, workspace/view and callback checks passed.
- The direct CUDA archive had no owning Tensor, global-pool, operator-factory
  or operator-vtable symbols in its inspected symbol table.

## Resume here

Three native tests failed: qwen3_session_test, qwen2_float_test and
inference_generation_test. The first expected an obsolete embedding-width
diagnostic; its expectation now matches the actual malformed fixture. The other
two reported token IDs outside the vocabulary. The final patch completes legacy
default-stream input uploads before private nonblocking session streams consume
them. This is a suspected ordering fix; it has not been retested.

Rebuild and rerun all native tests first using the existing build-engine-final
configuration. Only after they pass, enable the Python binding, rebuild, rerun
CTest and the offline TTS Python suite, then review and publish master.

No final allocation-count, full-checkpoint parity or speedup claim is established
by this checkpoint. Full Qwen TTS talker, predictor, codec/waveform integration
and Jetson validation remain outstanding. KV capacity is exact; alignment, live
scratch, retained prefill capacity and CUDA-library memory still consume space.
