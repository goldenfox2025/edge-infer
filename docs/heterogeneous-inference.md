# CPU/GPU cooperation: evidence and future direction

This document records a possible direction for `edge-infer`. The current runtime
does not implement a heterogeneous CPU/GPU executor, expert offloading or an
automatic hardware placement policy. Its prepared weights, independent bounded
sessions, concrete operators and offline workspace lifetimes provide useful
boundaries for that work. The FP32 CPU Qwen path is a correctness reference;
its executors share privately snapshotted weights and own independent mutable
scratch. Having a thread pool does not make it an optimized parallel CPU backend.

## Shared resource contract

The common layer should manage storage and execution resources while each model
declares its concrete computation and state transitions. A model can describe
fixed state or bounded dynamic state/workspaces; preparation reserves within
those bounds and resolves the views used by execution. The Qwen decoder is one
such sequence. An iterative audio/video model needs its own sequence rather than
an autoregressive interface with unused KV state.

CPU, GPU and pinned host storage need explicit byte capacity, addressability,
residency, scalar/quantization format, physical layout, alignment and lifetime.
Model weights can be shared read-only; session/request state remains private.
Transfer staging and any shared residency table need an owner, including the
completion rules for their readers and writers. A slot is reusable only after
its last read and write complete. Copy submission alone cannot publish a resident
weight, and mapped host storage cannot be budgeted as GPU-resident storage.

CPU jobs, copy streams and compute streams must agree on dependencies. Start with
static placement and ordinary completion events. On cancellation or failure,
drain submitted CPU work and transfers before releasing their backing storage or
starting another request in it. Adaptive replacement additionally needs to prevent
an outstanding reader from observing overwritten bytes; it must not reduce an
existing session's promised capacity. This is a resource contract, not a proposal
for a generic scheduler or automatic placement framework.

## Reference and scope

This review considers
[Niko1221/Strata](https://github.com/Niko1221/Strata).
The CUDA sparse-MoE path was inspected at commit
[`6f32ec070f23ced9f50e704d854d775da52591ab`](https://github.com/Niko1221/Strata/tree/6f32ec070f23ced9f50e704d854d775da52591ab).
Its repository paper, [*Running a 125-Billion-Parameter Model on a Normal PC*][paper],
describes a specialized Qwen3.8-Flash-Next engine. The inspected source has evolved
beyond that paper; this is a mechanism review of its CUDA path, not a complete
support-matrix audit or an independent benchmark reproduction.

The separate [*Strata: Hierarchical Context Caching for Long Context Language
Model Serving*](https://www.usenix.org/conference/osdi26/presentation/xie-zhiqiang)
is a serving/KV-cache paper. It is not the CPU expert engine discussed here.

## What the inspected engine actually does

| Component | Role and relevant evidence |
| --- | --- |
| Memory plan | Accounts for fixed weights, KV/indexer storage, recurrent state and workspace before allocating expert-cache capacity; refuses an impossible budget. Its geometry and default budget are model-specific. [Planner][plan] |
| Host expert arena | Stores expert weights where CPU kernels can read them directly. Attempts large-page backing and CUDA mapped registration; reports backing, registration limits and OS fallbacks. Pinning may cover only a prefix or per-layer slices, not the whole arena. [Arena contract][pinned-header], [implementation][pinned-source] |
| Expert residency | Owns VRAM slots and the `(layer, expert) -> slot` table, separately from computation. Slots can have layer-specific byte sizes. Copies and publication of residency must agree; a pending copy is not a usable resident expert. [Cache contract][cache] |
| CPU arithmetic | Uses a repacked Q2_0 VNNI path and native GGUF-format dot products through ggml CPU type traits. Native expert execution checks ISA support before selecting AVX-512 or AVX2 kernels, with a build-dependent ggml fallback. [Q2_0 kernel][q2], [native formats and dispatch][native] |
| CPU workers | Uses a dedicated expert batch pool with physical-core topology/affinity, per-worker scratch and optional host participation. Epoch-tagged job claims and completion/parking rules protect batch reuse. Workers spin briefly and then sleep. This is a different workload and contract from a general task queue. [Pool][pool] |
| GPU/host handoff | The router publishes activation and selected expert IDs to mapped pinned memory before publishing a sequence word. A CPU host loop observes that word and submits the corresponding expert work while the GPU continues resident/shared work. [Publication][layer], [host loop][verify-host] |
| GPU transfers | A separate copy stream issues asynchronous DMA for part of the misses. A stream-ordered host callback publishes completion; captured computation waits before reading that staging region. GPU compute, CPU expert evaluation and DMA can overlap, but each layer still waits for its required results. [DMA and completion][verify-dma] |
| Prefill | Uses chunks and bounded expert staging rings, with a copy stream and per-slot DMA events. A larger token batch changes the preferred expert compute/transfer strategy. [Prefill][prefill] |

The central advantage is sparsity: the routed experts needed by a token are a
small part of the whole MoE weight set. GPU-resident experts avoid host traffic;
CPU kernels compute other experts beside their RAM storage; a selected share of
misses can use the GPU copy engine. The CPU and GPU are doing different expert
jobs, rather than evaluating the same job twice.

Mapped memory and a sequence word are a synchronization protocol, not permission
to assume that a copy has finished after a delay. Buffer ownership, publication
ordering, completion, reuse and failure handling are required together. Strata's
busy polling and graph-internal waits should not be copied into the current
synchronous session interface without a separate failure/cancellation design.

Numerical dispatch also matters. The inspected [technical notes][details] explain
that CPU/GPU expert placement and single-/multi-token arithmetic can round
differently. Their reproducible-output controls disable or constrain some adaptive
behavior. Exact speculative verification within a chosen arithmetic path does not
establish identical outputs across every placement, cache state or backend.

## MiniMax H3: dense multimodal denoising

This review uses the official [MiniMax H3 release][h3-release] at
`d21241f0a4b3acbb34c97dae47fa417b7065e438`, the official-linked Diffusers
implementation at `36438e2ee44a7b8939a06e9c635085d96ae83a3e`, and SGLang's H3
implementation at `f3b5a28f4315767261d439baf4705f2049d8871f`. No H3 checkpoint
has been executed by `edge-infer`.

H3 uses a Qwen3-VL-32B encoder, a dense 33B Omni Transformer and separate visual
and audio VAEs. The [released transformer config][h3-config] specifies 50 layers,
hidden width 5376, 56 attention heads of width 128, and FFN width 14336. The
[transformer implementation][h3-transformer] performs noncausal self-attention
over a packed text/video/audio sequence, with partial three-axis `(t, h, w)`
MM-RoPE and AdaLN selected by each row's timestep and modality. There is no
cross-attention in that block stack.

Each [denoising iteration][h3-denoise] predicts velocities for the full sequence,
then updates only generated audio/video rows. Conditioning rows remain fixed.
Audio and video use separate [rectified-flow schedules][h3-scheduler], with
released shifts of 3 and 12 respectively. Position construction, timestep
conventions, modulation, packing and latent updates belong to the H3 model
implementation. They cannot be supplied by changing the weight names in the
current causal Qwen/KV loop.

The official release reports about 13B parameters in AdaLN branches whose outputs
can be precomputed for inference. SGLang's [AdaLN cache][h3-adaln] is a relevant
resource example: exact FP32 timestep bit patterns identify plans; cache metadata
validates the format and model variant; read fences and completed copies govern
publication and reuse. Its host/GPU tiers have byte budgets and an explicit
arithmetic mode. These precomputed values can be shared when their model,
timestep and precision contracts agree; request latents remain private.

For H3, first consider encoder, transformer and VAE stage residency, then measured
dense-block streaming with bounded staging. Unlike sparse MoE, the dense blocks
are used on every denoising iteration, so repeatedly moving all nonresident
weights can dominate runtime. A useful transfer accounting term is
`nonresident dense bytes per iteration * actual model iterations`; sustainable
copy bandwidth and overlap must be measured. Full attention still has quadratic
arithmetic cost even if a tiled kernel avoids materializing its score matrix.

As an illustration, 33B BF16 parameters occupy about 66 GB before metadata and
alignment. Removing the reported 13B AdaLN branches leaves roughly 40 GB of dense
weights; even an ideal four-bit representation of that remainder is about 10 GB,
before scales, encoder/VAEs and workspaces. This arithmetic is not a supported
quantization format, a complete memory plan or a promise that H3 fits an 8 GB GPU.
The initial release provides full attention only. Its sparse-attention
implementation, hosted Context-IR workflow and Regenerate-2K module are not
included in that release.

## Fit for dense models, speech, video and Jetson

| Workload | Useful ideas | Required boundary |
| --- | --- | --- |
| Current dense Qwen backbone | Explicit memory budget, stable session arenas, measured prefill/decode policies, transfer completion and bounded staging | Dense weights are broadly used on every token. An MoE hot-expert cache does not translate directly; CPU layer offloading needs optimized CPU operators and a measured transfer/compute cost model. |
| Qwen TTS architectures | Separate placement of conditioning, transformer, prediction and codec stages; budget the state of each stage; measure first audio and streaming behavior | Multi-axis positions, group-specific heads and cache reset/stopping rules remain model semantics. A codec boundary is not an MoE expert boundary. See [speech integration](speech-integration.md) for maintained scope. |
| MiniMax H3 | Stage/block residency, bounded copies, reusable dense operators/workspaces and precomputed AdaLN plans | Dense iterative full-sequence execution needs noncausal attention, MM-RoPE, modulation and separate audio/video schedules. An MoE hot-expert policy or append KV loop does not provide these semantics. |
| Future sparse MoE | Independent router/grouped-expert operators, explicit expert residency and CPU/GPU job partitioning | First implement and numerically validate the architecture, supported weight formats and optimized CPU expert backend. Residency policy alone cannot provide inference. |
| Jetson | Joint memory budget and phase-specific execution choices | ARM64 needs its own CPU backend. Shared DRAM changes the cost model: CPU, GPU and transfers contend for bandwidth, and desktop PCIe/AVX assumptions do not apply. Validate one board and JetPack release. |

Multiple sessions should continue sharing immutable prepared weights while owning
their history and mutable workspaces. Any future transfer slots, completion events
and CPU scratch need an explicit owner and a bounded reuse protocol. Expert
residency may be shared as model storage, but outstanding session reads must
finish before a slot is replaced. No future placement policy should silently
change a session's promised context capacity.

## Proposed sequence

1. **Account before placing.** Extend preparation reporting to include retained
   weights, per-session KV and peak prefill/decode workspaces, library-private
   memory, CPU scratch and any pinned staging. Reserve resources at preparation
   boundaries and report the chosen plan. Keep this device budget separate from
   the existing operator-lifetime arena planner. Do not import Strata's geometry
   or fixed budget defaults.
2. **Measure concrete phase choices.** Establish an optimized CPU backend and
   compare complete GPU execution with selected dense layer/stage placements.
   Add bounded pinned staging and stream/event dependencies only where required.
   Compare prefill and decode separately; changing the placement every token is
   not the initial goal. For speech, measure the codec bridge and stage transfers
   before deciding which components to move to native code.
3. **Validate a concrete multimodal block.** For an H3 path, first compare one
   block against the pinned reference on identical inputs, including noncausal
   attention, MM-RoPE, AdaLN and the gated MLP. Report its weight, workspace and
   staging budgets and test repeated independent requests. Then measure static
   stage/block placement and copy overlap before attempting a full checkpoint.
4. **Add MoE policy with a real MoE model.** Keep routing and grouped expert
   computation in independently usable operators. Put residency, job partitioning
   and transfer ownership in preparation/session execution. Start with static
   byte-based placement and explicit completion; add adaptive placement only after
   numerical parity, safe slot reuse and measured benefit.

These changes do not require a generic model factory or dynamic graph framework.
Resolve model semantics and placement during preparation, then call concrete
operators with borrowed views and explicit execution resources. Introduce a
shared abstraction only when implemented paths need the same contract.

Acceptance requires real checkpoints and repeatable measurements: numerical
error and token/audio/video behavior; peak RAM/VRAM/pinned bytes; cold/warm
first-token or first-audio latency; prefill, steady decode and full denoising
iterations; CPU/GPU/copy activity; and
multiple sessions, cancellation and allocation/transfer failures. Record CPU ISA,
worker policy, device, driver, toolkit, model revision and placements. The external
project's throughput is not an `edge-infer` result, and this document makes no
performance claim.

[paper]: https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/paper/Strata-Paper.pdf
[plan]: https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/plan/plan.hpp
[pinned-header]: https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/core/pinned.hpp#L23-L59
[pinned-source]: https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/core/pinned.cu#L327-L365
[cache]: https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/core/expert_cache.hpp
[q2]: https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cpu/expert.cpp#L144-L191
[native]: https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cpu/native_expert.cpp#L32-L123
[pool]: https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/cpu/pool.hpp#L121-L218
[layer]: https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/core/layer.cpp#L374-L389
[verify-host]: https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/core/verify.cpp#L1420-L1490
[verify-dma]: https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/core/verify.cpp#L1618-L1637
[prefill]: https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/prefill/prefill.cpp
[details]: https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/DETAILS.md#L87-L100
[h3-release]: https://github.com/MiniMax-AI/MiniMax-H3/blob/d21241f0a4b3acbb34c97dae47fa417b7065e438/README.md
[h3-config]: https://github.com/MiniMax-AI/MiniMax-H3/blob/d21241f0a4b3acbb34c97dae47fa417b7065e438/FL2VA/transformer/config.json
[h3-transformer]: https://github.com/huggingface/diffusers/blob/36438e2ee44a7b8939a06e9c635085d96ae83a3e/src/diffusers/models/transformers/transformer_minimax_h3.py
[h3-denoise]: https://github.com/huggingface/diffusers/blob/36438e2ee44a7b8939a06e9c635085d96ae83a3e/src/diffusers/modular_pipelines/minimax_h3/denoise.py
[h3-scheduler]: https://github.com/huggingface/diffusers/blob/36438e2ee44a7b8939a06e9c635085d96ae83a3e/src/diffusers/schedulers/scheduling_minimax_h3.py
[h3-adaln]: https://github.com/sgl-project/sglang/blob/f3b5a28f4315767261d439baf4705f2049d8871f/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3_adaln_cache.py
