# Historical measurements from 2025

Archived RTX 4070 Laptop observations, translated from the original report.
Values are the best of five runs. Complete baseline revisions, commands, logs
and numerical comparisons were not included. These results have not been
reproduced on the maintained branch and are not performance guarantees.
The original report remains in Git history.

## Qwen2.5 1.5B Instruct BF16

Flash Attention enabled; top-k 20; top-p disabled; output limit 201 tokens,
including the first token produced during prefill.

| Metric | llama.cpp | LLM_infer |
| --- | --- | --- |
| Initial prefill latency, 23 tokens | 21.60 ms | 17.94 ms |
| Initial prefill throughput | 1065.01 tokens/s | 1282.05 tokens/s |
| Decode latency, 200 tokens | 2883.04 ms | 2640.19 ms |
| Decode throughput | 69.37 tokens/s | 75.75 tokens/s |
| Warmed benchmark prefill latency, 200 tokens | 33.44 ms | 28.02 ms |
| Warmed benchmark prefill throughput | 5980.84 tokens/s | 7137.75 tokens/s |
| Warmed benchmark decode throughput | 69.47 tokens/s | 75.75 tokens/s |

The original report described approximately 1.20x prefill and 1.09x decode
ratios. Benchmark mode warmed up with the same input before timing. The prefill
arena was sized to avoid additional physical allocations during measurement.
Missing baseline details limit comparisons across revisions.

## Qwen3 1.7B BF16 and AWQ

| Metric | BF16 | AWQ |
| --- | --- | --- |
| Prefill latency, 23 tokens | 24.26 ms | 50.73 ms |
| Prefill throughput | 956.31 tokens/s | 453.38 tokens/s |
| Decode latency, 200 tokens | 3104.76 ms | 1980.37 ms |
| Decode throughput | 64.42 tokens/s | 100.99 tokens/s |

AWQ decode was reported as 1.57x faster, while prefill was slower. The original
analysis attributed slower prefill to an insufficiently optimized WMMA GEMM
and faster decode to reduced memory traffic. Those explanations were not
independently profiled in the archived report.

## Speculative decoding

Qwen3 0.6B AWQ drafted for Qwen3 1.7B AWQ. Approximate reported timings were
9 ms per baseline token, 30 ms to draft 6-8 tokens and 55 ms for verification.

```text
iteration cost = 30 ms + 55 ms = 85 ms
85 ms < accepted_tokens * 9 ms
accepted_tokens > 9.4
```

The experiment did not achieve the required acceptance rate. The simple model
also omits details such as bonus-token accounting. Probability rejection and
residual resampling still require a correctness review; these observations do
not establish speculative sampling correctness.
