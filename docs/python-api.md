# Python models and sessions

`model_bridge.Model` prepares native weights on an explicit device without
allocating a generation engine or KV cache. It keeps the native preparation
prototype private. CUDA sessions share immutable prepared weights and own
independent caches, execution storage and streams. The FP32 CPU reference
snapshots caller weights once, shares that private snapshot across executors,
and keeps scratch and temperature/top-k/top-p sampling state independent.
The reference numerical comparisons cover greedy generation.

```python
from frontend.checkpoint import load_model
from model_bridge import Model

config, weights, model_type = load_model("/absolute/path/to/model", "qwen3_bf16")
model = Model(config, weights, model_type, device="cuda")
first = model.new_session(capacity=1024)
second = model.new_session(capacity=1024)
del weights

# prompt_ids contains the entire rendered prompt, including chat history.
# max_length counts prompt tokens plus generated tokens.
tokens = []
first.generate(prompt_ids, tokens.append,
               max_length=len(prompt_ids) + 32, top_k=1)
first.reset()
```

Supported model types are `llama`, `qwen`, `qwen_bf16`, `qwen_awq`,
`qwen3_bf16` and `qwen3_awq`. BF16 and AWQ require an available CUDA device.
`device=None` uses the configured default for new models; selecting a device
does not migrate existing models or sessions. Invalid devices and construction
failures raise exceptions. Text models currently require plain RoPE and full
attention; nonempty `rope_scaling`, active sliding windows and other layer types
are rejected before weight preparation. `is_cuda_available()` reports actual device
availability independently of the selected default.

Checkpoint admission requires an explicit embedding tensor and an output head.
A missing head is derived from the embedding only when `tie_word_embeddings`
is explicitly boolean `True`; caller dictionaries remain unchanged. One shared
mapper validates exact decoder names, layer indices, ranks and aliases, and
preserves supported projection and head biases. Unsupported keys reject rather
than disappearing silently.

AWQ model types accept the native 4-bit asymmetric GEMV packing layout. Declared
`quant_method`, `bits`/`w_bit`, `zero_point` and `version` must agree with that
contract. All `group_size`/`q_group_size` declarations must agree and be positive
integers within the native range; absent declarations default to 128. Packed
weights and zeros require signed 32-bit storage. Dense and packed projections
can coexist, but one projection cannot declare both and every packed projection
requires its scales and zeros.

`new_session()` defaults to `min(4096, model.max_context_length)` tokens.
An explicit capacity must be positive and within the model context limit.
The session exposes read-only `capacity`, `device` and `speculative` properties.
Its cache allocation follows the supplied capacity rather than reserving the
model's full advertised context. Session objects retain prepared weights after
the Python Model is released.

Each `generate()` call accepts a complete fresh prompt and replaces request
state. Repeated calls do not implicitly append prompts. The frontend retains
system, user and assistant messages, renders the complete conversation for each
turn, and passes a total length within the session capacity. The chat option
`--max_new_tokens` controls generated-token count; `--max_length` remains an alias
for that frontend option. The native and binding `max_length` argument continues
to count the total sequence, including the prompt.

Callbacks receive token IDs on the thread that called generation. The binding
releases the GIL while native generation runs and reacquires it for callbacks.
Python callback exceptions retain their original type, value and traceback.
Generation or reset on the same session while it is busy raises an error;
another session has an independent guard. Applications should serialize calls
to each session. `reset()` clears request state while retaining allocations.
If native completion fails, the session is permanently invalid; `reset()`
cannot recover it. Create a replacement with `model.new_session(capacity)`,
retaining the prepared model.

## Exact greedy speculation

```python
draft_config, draft_weights, draft_type = load_model(
    "/absolute/path/to/draft", "qwen3_bf16")
draft = Model(draft_config, draft_weights, draft_type, device="cuda")
speculative = model.new_speculative_session(
    draft, capacity=1024, spec_length=4)
speculative.generate(prompt_ids, tokens.append,
                     max_length=len(prompt_ids) + 32, top_k=1)
```

Target and draft must use compatible tokenization, BF16/AWQ execution and the
same vocabulary. Capacity must fit both models. `spec_length` must be between
1 and 8. The native speculative decoder supports exact greedy verification
only. In Python, `top_k != 1` creates an ordinary target engine lazily and uses
its separate cache; this request performs ordinary sampling. There is no
probability-ratio verification mode or adaptive speculation policy.

## Procedural compatibility

`init_model(config, weights, model_type)`, `generate_text_stream(...)`,
`init_speculative_decoder(...)` and `generate_text_stream_speculative(...)`
remain wrappers over one explicit model and bounded session. Initialization
returns a boolean and publishes replacement objects only on success; failure
preserves the previous usable model and sessions. Overlapping compatibility
generation, initialization and default-device changes are rejected.

`generate_text_stream_speculative(..., top_k=1)` uses the initialized
speculative session, or ordinary inference if no speculative session exists.
Requests with `top_k != 1` use the existing ordinary session. All generation
functions require the same complete-prompt and total-length contract.
