#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "qwen3.hpp"

// -------------------------------
// prefill: Prefill entry point
// -------------------------------
template <typename T>
uint32_t *Qwen3Session<T>::prefill(const Tensor<uint32_t> *input, ThreadPool &thread_pool, KVCacheBase *kv_cache,
                                 size_t top_k, float temperature, float top_p, curandState *d_states) {
    KVCache<T> *typed_cache = dynamic_cast<KVCache<T> *>(kv_cache);

    auto logits = prefill_eager(input, typed_cache);
    const size_t rows = logits.sizes()[0];
    const size_t vocabulary = logits.sizes()[1];
    auto final_logits = logits.slice({rows - 1, 0}, {rows, vocabulary});
    operators_->sample_to_fixed(std::move(final_logits), sampled_token_.data_ptr(),
                                temperature, top_p, top_k, d_states, execution_stream_);
    synchronize();
    return sampled_token_.data_ptr();
}

// -------------------------------
// prefill_eager: Device-independent eager prefill implementation
// -------------------------------
template <typename T>
Tensor<T> Qwen3Session<T>::prefill_eager(const Tensor<uint32_t> *input, KVCache<T> *kv_cache) {
    validate_and_bind(input, kv_cache, false);
    synchronize();
    const size_t rows = input->numel();
    if (prefill_rows_ != rows) {
        prefill_buffers_ = prepare_buffers(rows, prefill_workspace_);
        prefill_rows_ = rows;
    }
    run_decoder(*input, *kv_cache, prefill_buffers_, true);
    synchronize();
    return prefill_buffers_.logits;
}

// Explicit template instantiations
template uint32_t *Qwen3Session<__nv_bfloat16>::prefill(const Tensor<uint32_t> *input, ThreadPool &thread_pool,
                                                      KVCacheBase *kv_cache, size_t top_k, float temperature,
                                                      float top_p, curandState *d_states);

template Tensor<__nv_bfloat16> Qwen3Session<__nv_bfloat16>::prefill_eager(const Tensor<uint32_t> *input,
                                                                        KVCache<__nv_bfloat16> *kv_cache);
