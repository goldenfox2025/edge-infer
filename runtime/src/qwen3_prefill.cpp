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
uint32_t *Qwen3Model<T>::prefill(const Tensor<uint32_t> *input, ThreadPool &thread_pool, KVCacheBase *kv_cache,
                                 size_t top_k, float temperature, float top_p, curandState *d_states) {
    KVCache<T> *typed_cache = dynamic_cast<KVCache<T> *>(kv_cache);

    return operators_->sample(prefill_eager(input, typed_cache), temperature, top_p, top_k, d_states);
}

// -------------------------------
// prefill_eager: Device-independent eager prefill implementation
// -------------------------------
template <typename T>
Tensor<T> Qwen3Model<T>::prefill_eager(const Tensor<uint32_t> *input, KVCache<T> *kv_cache) {
    if (input->device() != Device::CUDA) {
        throw std::runtime_error("Input tensor must be on CUDA device");
    }

    const size_t seq_len = input->sizes()[0];
    auto artifacts = build_decoder_runtime_artifacts(seq_len, AttentionMode::Prefill);
    const auto workspace_plan =
        build_workspace_plan_from_execution_program<T>(artifacts.program);
    CudaWorkspaceArena workspace_arena;
    workspace_arena.reserve_for_plan(workspace_plan);
    DecodeEagerBuffers buffers;
    buffers.tensors.reserve(artifacts.program.values.size());
    auto workspace_tensor = [&](const std::string& name,
                                const std::vector<size_t>& shape) {
        const auto& allocation = workspace_plan.at(name);
        return Tensor<T>::from_external_buffer(workspace_arena.template ptr_at<T>(allocation.offset), shape,
                                               Device::CUDA);
    };
    for (const auto& value : artifacts.program.values) {
        buffers.tensors.emplace(value.name, workspace_tensor(value.name, value.shape));
    }

    size_t offset = 0;
    if (kv_cache) {
        if (kv_cache->device() != Device::CUDA) {
            throw std::runtime_error("KVCache must be on CUDA device");
        }
        offset = kv_cache->size() - seq_len;
    } else {
        throw std::runtime_error("KVCache is required for Qwen3 prefill");
    }

    for (const auto& node : artifacts.nodes) {
        node.execute(
            [&](const std::string& name) -> Tensor<T>& { return buffers.require(name); },
            [&](size_t layer) {
              return kv_cache->get_contiguous_tensor(layer);
            },
            input, seq_len, offset, n_heads_, n_kv_heads_, head_dim_,
            rms_norm_eps_, rope_theta_);
    }

    return buffers.require("logits");
}

// Explicit template instantiations
template uint32_t *Qwen3Model<__nv_bfloat16>::prefill(const Tensor<uint32_t> *input, ThreadPool &thread_pool,
                                                      KVCacheBase *kv_cache, size_t top_k, float temperature,
                                                      float top_p, curandState *d_states);

template Tensor<__nv_bfloat16> Qwen3Model<__nv_bfloat16>::prefill_eager(const Tensor<uint32_t> *input,
                                                                        KVCache<__nv_bfloat16> *kv_cache);
