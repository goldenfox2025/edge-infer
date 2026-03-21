#pragma once

#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

#include "operators/cuda/legacy/legacy_bridge.cuh"
#include "operators/cuda/sampling_runtime.cuh"
#include "operators/operator_factory.hpp"

namespace op::detail {

template <typename T>
class UnifiedRuntimeBridge {
 public:
  explicit UnifiedRuntimeBridge(OperatorPlatform platform)
      : platform_(platform) {}

  void set_platform(OperatorPlatform platform) { platform_ = platform; }

  uint32_t sample_cpu(Tensor<T>&& logits, float temperature, float top_p,
                      size_t top_k) const {
    auto op = OperatorFactory<T>::getSampleOperator(OperatorPlatform::CPU);
    if (!op) {
      throw std::runtime_error("CPU sample operator not registered");
    }

    uint32_t* result_ptr =
        (*op)(std::move(logits), temperature, top_p, top_k, nullptr);
    const uint32_t result = *result_ptr;
    delete result_ptr;
    return result;
  }

  void init_curand(curandState* d_states, unsigned long long seed, int offset,
                   cudaStream_t stream = nullptr) const {
    require_cuda("init_curand");
    cuda::init_curand(d_states, seed, offset, stream);
  }

  void sample_to_fixed(Tensor<T>&& logits, uint32_t* output_ptr,
                       float temperature, float top_p, size_t top_k,
                       curandState* d_states,
                       cudaStream_t stream = nullptr) const {
    if (platform_ == OperatorPlatform::CPU) {
      if (!output_ptr) {
        throw std::runtime_error(
            "sample_to_fixed requires a valid CPU output pointer");
      }
      *output_ptr = sample_cpu(std::move(logits), temperature, top_p, top_k);
      return;
    }
    cuda::sample_to_fixed(std::move(logits), output_ptr, temperature, top_p,
                          top_k, d_states, stream);
  }

  void sample_batch_to_fixed(Tensor<T>&& logits, uint32_t* output_ptr,
                             float temperature, float top_p, size_t top_k,
                             curandState* d_states,
                             cudaStream_t stream = nullptr) const {
    if (platform_ == OperatorPlatform::CPU) {
      if (!output_ptr) {
        throw std::runtime_error(
            "sample_batch_to_fixed requires a valid CPU output pointer");
      }
      if (logits.sizes().empty()) {
        throw std::runtime_error("sample_batch_to_fixed received empty logits");
      }

      const size_t batch_size = logits.sizes()[0];
      const size_t vocab_size =
          logits.sizes().size() > 1 ? logits.sizes()[1] : logits.numel();
      for (size_t i = 0; i < batch_size; ++i) {
        Tensor<T> row = logits.slice({i, 0}, {i + 1, vocab_size}).squeeze(0);
        output_ptr[i] = sample_cpu(std::move(row), temperature, top_p, top_k);
      }
      return;
    }

    cuda::sample_batch_to_fixed(std::move(logits), output_ptr, temperature,
                                top_p, top_k, d_states, stream);
  }

  void sample_to_fixed_with_prob(Tensor<T>&& logits, uint32_t* token_ptr,
                                 float* prob_ptr, float temperature,
                                 float top_p, size_t top_k,
                                 curandState* d_states,
                                 cudaStream_t stream = nullptr) const {
    if (!token_ptr || !prob_ptr) {
      throw std::runtime_error(
          "sample_to_fixed_with_prob requires valid output pointers");
    }

    if (platform_ == OperatorPlatform::CPU) {
      Tensor<T> logits_copy = logits;
      *token_ptr = sample_cpu(std::move(logits), temperature, top_p, top_k);
      *prob_ptr = get_token_probability(logits_copy, 0, *token_ptr, stream);
      return;
    }

    cuda::sample_to_fixed_with_prob(std::move(logits), token_ptr, prob_ptr,
                                    temperature, top_p, top_k, d_states,
                                    stream);
  }

  void generate_random_values(float* values, size_t count, curandState* states,
                              cudaStream_t stream = nullptr) const {
    if (!values) {
      throw std::runtime_error(
          "generate_random_values requires a valid output buffer");
    }

    if (platform_ == OperatorPlatform::CPU) {
      static thread_local std::mt19937 rng(std::random_device{}());
      std::uniform_real_distribution<float> dist(0.0f, 1.0f);
      for (size_t i = 0; i < count; ++i) {
        values[i] = dist(rng);
      }
      return;
    }

    cuda::generate_random_values(values, count, states, stream);
  }

  float get_token_probability(const Tensor<T>& logits, int position,
                              uint32_t token_id,
                              cudaStream_t stream = nullptr) const {
    if (platform_ == OperatorPlatform::CPU) {
      if (logits.device() != Device::CPU) {
        throw std::runtime_error(
            "CPU probability query requires CPU logits tensor");
      }
      if (logits.sizes().empty()) {
        throw std::runtime_error("get_token_probability received empty logits");
      }

      const size_t vocab_size =
          logits.sizes().size() > 1 ? logits.sizes()[1] : logits.numel();
      const size_t row_count =
          logits.sizes().size() > 1 ? logits.sizes()[0] : static_cast<size_t>(1);
      if (position < 0 || static_cast<size_t>(position) >= row_count) {
        throw std::runtime_error("get_token_probability position out of range");
      }
      if (token_id >= vocab_size) {
        throw std::runtime_error("get_token_probability token_id out of range");
      }

      const T* data = logits.data_ptr();
      const size_t row_offset = static_cast<size_t>(position) * vocab_size;
      float max_logit = -std::numeric_limits<float>::infinity();
      for (size_t i = 0; i < vocab_size; ++i) {
        max_logit =
            std::max(max_logit, static_cast<float>(data[row_offset + i]));
      }

      float denom = 0.0f;
      for (size_t i = 0; i < vocab_size; ++i) {
        denom += std::exp(static_cast<float>(data[row_offset + i]) - max_logit);
      }

      const float token_logit = static_cast<float>(data[row_offset + token_id]);
      return std::exp(token_logit - max_logit) / denom;
    }

    return cuda::get_token_probability(logits, position, token_id, stream);
  }

  void gemv_qkv_rope(Tensor<T>* hidden_states,
                     const Tensor<T>* merged_qkv_weight, Tensor<T>* q_buf,
                     Tensor<T>* k_buf, Tensor<T>* v_buf,
                     const Tensor<T>* merged_qkv_bias, size_t* d_rope_offset,
                     const Tensor<float>* rope_sin_cos_cache,
                     int* d_offset_array, int layer_idx, int q_dim, int k_dim,
                     int v_dim, int n_heads, int n_kv_heads, int head_dim,
                     cudaStream_t stream = nullptr, int n_layers = 0,
                     int* pingpong = nullptr) const {
    require_cuda("gemv_qkv_rope");
    legacy::gemv_qkv_rope(hidden_states, merged_qkv_weight, q_buf, k_buf, v_buf,
                          merged_qkv_bias, d_rope_offset, rope_sin_cos_cache,
                          d_offset_array, layer_idx, q_dim, k_dim, v_dim,
                          n_heads, n_kv_heads, head_dim, stream, n_layers,
                          pingpong);
  }

  void rope_with_precomputed_cache(Tensor<T>* tensor, const size_t* d_offset,
                                   const Tensor<float>* rope_sin_cos_cache,
                                   cudaStream_t stream = nullptr,
                                   int* d_offset_array = nullptr,
                                   int layer_idx = 0, int n_layers = 0,
                                   int* pingpong = nullptr) const {
    require_cuda("rope_with_precomputed_cache");
    legacy::rope_with_precomputed_cache(tensor, d_offset, rope_sin_cos_cache,
                                        stream, d_offset_array, layer_idx,
                                        n_layers, pingpong);
  }

  void flash_attention_graph_fixed(Tensor<T>& q, const Tensor<T>& k,
                                   const Tensor<T>& v, T** d_output_ptrs,
                                   int* d_segment_info, int n_kv_heads,
                                   cudaStream_t stream = nullptr,
                                   int* pingpong = nullptr) const {
    require_cuda("flash_attention_graph_fixed");
    legacy::flash_attention_graph_fixed(q, k, v, d_output_ptrs, d_segment_info,
                                        n_kv_heads, stream, pingpong);
  }

  void gather_fa_graph_fixed(T** d_output_ptrs, Tensor<T>& att_heads,
                             int* d_segment_info,
                             cudaStream_t stream = nullptr) const {
    require_cuda("gather_fa_graph_fixed");
    legacy::gather_fa_graph_fixed(d_output_ptrs, att_heads, d_segment_info,
                                  stream);
  }

  void gemv_mlp_fused(Tensor<T>* hidden_states,
                      const Tensor<T>* merged_mlp_weight,
                      Tensor<T>* gate_buf_silu,
                      cudaStream_t stream = nullptr) const {
    require_cuda("gemv_mlp_fused");
    legacy::gemv_mlp_fused(hidden_states, merged_mlp_weight, gate_buf_silu,
                           stream);
  }

 private:
  void require_cuda(const char* op_name) const {
    if (platform_ != OperatorPlatform::CUDA) {
      throw std::runtime_error(std::string(op_name) +
                               " is only available on CUDA");
    }
  }

  OperatorPlatform platform_;
};

}  // namespace op::detail
