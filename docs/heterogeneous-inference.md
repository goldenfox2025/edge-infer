# CPU/GPU cooperation: evidence and future direction

This document records a possible direction for `edge-infer`. The current runtime
does not implement a heterogeneous CPU/GPU executor, expert offloading or an
automatic hardware placement policy. Its prepared weights, independent bounded
sessions, concrete operators and offline workspace lifetimes provide useful
boundaries for that work. The FP32 CPU Qwen path is a correctness reference;
having a thread pool does not make it an optimized parallel CPU backend.

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

## Fit for dense models, speech and Jetson

| Workload | Useful ideas | Required boundary |
| --- | --- | --- |
| Current dense Qwen backbone | Explicit memory budget, stable session arenas, measured prefill/decode policies, transfer completion and bounded staging | Dense weights are broadly used on every token. An MoE hot-expert cache does not translate directly; CPU layer offloading needs optimized CPU operators and a measured transfer/compute cost model. |
| Qwen TTS architectures | Separate placement of conditioning, transformer, prediction and codec stages; budget the state of each stage; measure first audio and streaming behavior | Multi-axis positions, group-specific heads and cache reset/stopping rules remain model semantics. A codec boundary is not an MoE expert boundary. See [speech integration](speech-integration.md) for maintained scope. |
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
3. **Add MoE policy with a real MoE model.** Keep routing and grouped expert
   computation in independently usable operators. Put residency, job partitioning
   and transfer ownership in preparation/session execution. Start with static
   byte-based placement and explicit completion; add adaptive placement only after
   numerical parity, safe slot reuse and measured benefit.

These changes do not require a generic model factory or dynamic graph framework.
Resolve model semantics and placement during preparation, then call concrete
operators with borrowed views and explicit execution resources. Introduce a
shared abstraction only when implemented paths need the same contract.

Acceptance requires real checkpoints and repeatable measurements: numerical
error and token/audio behavior; peak RAM/VRAM/pinned bytes; cold/warm first-token
or first-audio latency; prefill and steady decode; CPU/GPU/copy activity; and
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
