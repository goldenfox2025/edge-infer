#pragma once

#include <cuda_bf16.h>

#include <cstring>
#include <memory>
#include <optional>

#include "tensor.hpp"
#include "weight_tensor.hpp"
#include "operators/operator_factory.hpp"
#include "operators/unified/composite_ops.hpp"
#include "operators/unified/runtime_bridge.hpp"
#include "operators/unified/support.hpp"

namespace op {

// 统一算子接口
template <typename T>
class UnifiedOperators {
 public:
  explicit UnifiedOperators(Device device = Device::CPU)
      : device_(device),
        platform_(detail::platform_from_device(device)),
        runtime_bridge_(platform_) {
    detail::configure_backend<T>(device_, platform_);
    runtime_bridge_.set_platform(platform_);
  }

  // 切换到CUDA设备
  void cuda() {
    if (device_ == Device::CUDA) return;
    device_ = Device::CUDA;
    detail::configure_backend<T>(device_, platform_);
    runtime_bridge_.set_platform(platform_);
  }

  // 切换到CPU设备
  void cpu() {
    if (device_ == Device::CPU) return;
    device_ = Device::CPU;
    detail::configure_backend<T>(device_, platform_);
    runtime_bridge_.set_platform(platform_);
  }

  Device device() const { return device_; }
  OperatorPlatform platform() const { return platform_; }

  std::shared_ptr<OperatorBase> get_operator_base(
      const std::string& op_name) const {
    return OperatorFactory<T>::getOperatorBaseByName(op_name, platform_);
  }

  bool has_operator(const std::string& op_name) const {
    return static_cast<bool>(get_operator_base(op_name));
  }

  // RoPE算子
  void rope(Tensor<T>* tensor, size_t offset, float theta,
            cudaStream_t stream = nullptr) {
    auto op = detail::require_operator<T>(
        OperatorFactory<T>::getRopeOperator(platform_), platform_, "RoPE");
    (*op)(tensor, offset, theta, stream);
  }

  // RMS Norm算子
  void rms_norm(Tensor<T>* output, Tensor<T>* input, Tensor<T>* weight,
                float eps, cudaStream_t stream = nullptr) {
    auto op = detail::require_operator<T>(
        OperatorFactory<T>::getRmsNormOperator(platform_), platform_,
        "RMS Norm");
    (*op)(output, input, weight, eps, stream);
  }

  // Multiply算子
  void multiply(Tensor<T>* output, Tensor<T>* input_a, Tensor<T>* input_b,
                cudaStream_t stream = nullptr) {
    auto op = detail::require_operator<T>(
        OperatorFactory<T>::getMultiplyOperator(platform_), platform_,
        "Multiply");
    (*op)(output, input_a, input_b, stream);
  }

  // SiLU算子
  void silu(Tensor<T>* output, Tensor<T>* input,
            cudaStream_t stream = nullptr) {
    auto op = detail::require_operator<T>(
        OperatorFactory<T>::getSiluOperator(platform_), platform_, "SiLU");
    (*op)(output, input, stream);
  }

  // Add算子
  void add(Tensor<T>* output, Tensor<T>* input_a, Tensor<T>* input_b,
           cudaStream_t stream = nullptr) {
    auto op = detail::require_operator<T>(
        OperatorFactory<T>::getAddOperator(platform_), platform_, "Add");
    (*op)(output, input_a, input_b, stream);
  }

  // 组合算子：residual += update; hidden_states = rms_norm(residual)
  void add_rms(Tensor<T>* hidden_states, Tensor<T>* residual, Tensor<T>* update,
               Tensor<T>* norm_weight, float eps,
               cudaStream_t stream = nullptr) {
    detail::UnifiedCompositeOps<T, UnifiedOperators>::add_rms(
        *this, hidden_states, residual, update, norm_weight, eps, stream);
  }

  // 组合算子：output = silu(input_a) * input_b
  void silu_multiply(Tensor<T>* output, Tensor<T>* input_a, Tensor<T>* input_b,
                     cudaStream_t stream = nullptr) {
    detail::UnifiedCompositeOps<T, UnifiedOperators>::silu_multiply(
        *this, output, input_a, input_b, stream);
  }

  void matmul(Tensor<T>* output, Tensor<T>* input,
              const WeightTensor<T>& weight, const Tensor<T>* bias = nullptr,
              cudaStream_t stream = nullptr) {
    auto op = detail::require_operator<T>(
        OperatorFactory<T>::getMatmulOperator(platform_), platform_, "MatMul");
    (*op)(output, input, weight, bias, stream);
  }

 public:
  // 从嵌入表中根据索引获取嵌入向量
  void gather(Tensor<T>* output, const Tensor<uint32_t>* input,
              const Tensor<T>* embedding_table, cudaStream_t stream = nullptr) {
    auto op = detail::require_operator<T>(
        OperatorFactory<T>::getGatherOperator(platform_), platform_, "Gather");
    (*op)(output, input, embedding_table, stream);
  }

  // 从logits中采样下一个token
  uint32_t* sample(Tensor<T>&& logits, float temperature, float top_p,
                   size_t top_k, curandState* d_states,
                   cudaStream_t stream = nullptr) {
    auto op = detail::require_operator<T>(
        OperatorFactory<T>::getSampleOperator(platform_), platform_, "Sample");
    return (*op)(std::move(logits), temperature, top_p, top_k, d_states,
                 stream);
  }

  // CPU版本的sample方法
  uint32_t sample_cpu(Tensor<T>&& logits, float temperature, float top_p,
                      size_t top_k) {
    return runtime_bridge_.sample_cpu(std::move(logits), temperature, top_p,
                                      top_k);
  }

  void init_curand(curandState* d_states, unsigned long long seed, int offset,
                   cudaStream_t stream = nullptr) {
    runtime_bridge_.init_curand(d_states, seed, offset, stream);
  }

  void sample_to_fixed(Tensor<T>&& logits, uint32_t* output_ptr,
                       float temperature, float top_p, size_t top_k,
                       curandState* d_states,
                       cudaStream_t stream = nullptr) {
    runtime_bridge_.sample_to_fixed(std::move(logits), output_ptr, temperature,
                                    top_p, top_k, d_states, stream);
  }

  void sample_batch_to_fixed(Tensor<T>&& logits, uint32_t* output_ptr,
                             float temperature, float top_p, size_t top_k,
                             curandState* d_states,
                             cudaStream_t stream = nullptr) {
    runtime_bridge_.sample_batch_to_fixed(std::move(logits), output_ptr,
                                          temperature, top_p, top_k, d_states,
                                          stream);
  }

  void sample_to_fixed_with_prob(Tensor<T>&& logits, uint32_t* token_ptr,
                                 float* prob_ptr, float temperature,
                                 float top_p, size_t top_k,
                                 curandState* d_states,
                                 cudaStream_t stream = nullptr) {
    runtime_bridge_.sample_to_fixed_with_prob(std::move(logits), token_ptr,
                                              prob_ptr, temperature, top_p,
                                              top_k, d_states, stream);
  }

  void generate_random_values(float* values, size_t count, curandState* states,
                              cudaStream_t stream = nullptr) {
    runtime_bridge_.generate_random_values(values, count, states, stream);
  }

  float get_token_probability(const Tensor<T>& logits, int position,
                              uint32_t token_id,
                              cudaStream_t stream = nullptr) {
    return runtime_bridge_.get_token_probability(logits, position, token_id,
                                                 stream);
  }

  // 动态Flash Attention包装函数
  void dynamic_flash_attention(Tensor<T>& Q, const Tensor<T>& K,
                               const Tensor<T>& V, Tensor<T>& output,
                               int n_kv_heads, cudaStream_t stream = nullptr) {
    auto op = OperatorFactory<T>::getDynamicFlashAttentionOperator(platform_);
    if (!op) {
      if (platform_ == OperatorPlatform::CPU) {
        detail::UnifiedCompositeOps<T, UnifiedOperators>::
            dynamic_flash_attention_fallback(*this, Q, K, V, output,
                                             n_kv_heads, stream);
        return;
      }
      throw std::runtime_error(
          "Dynamic Flash Attention operator not registered for the current "
          "platform");
    }
    (*op)(Q, K, V, output, n_kv_heads, stream);
  }

  // Prefill阶段计算注意力分数
  void compute_attention_scores_prefill(const Tensor<T>& Q, const Tensor<T>& K,
                                        Tensor<T>& att_scores, size_t head_dim,
                                        cudaStream_t stream = nullptr) {
    auto op = detail::require_operator<T>(
        OperatorFactory<T>::getAttentionScoresPrefillOperator(platform_),
        platform_, "Attention Scores Prefill");
    (*op)(Q, K, att_scores, head_dim, stream);
  }

  // Prefill阶段计算注意力输出
  void compute_attention_output_prefill(const Tensor<T>& att_scores,
                                        const Tensor<T>& V,
                                        Tensor<T>& att_output, size_t n_heads,
                                        size_t head_dim, size_t total_seq_len,
                                        size_t n_kv_heads,
                                        cudaStream_t stream = nullptr) {
    auto op = detail::require_operator<T>(
        OperatorFactory<T>::getAttentionOutputPrefillOperator(platform_),
        platform_, "Attention Output Prefill");
    (*op)(att_scores, V, att_output, n_heads, head_dim, total_seq_len,
          n_kv_heads, stream);
  }

  // Softmax算子
  void softmax(Tensor<T>* output, const Tensor<T>* input, int dim,
               bool mask = false, int offset = 0,
               cudaStream_t stream = nullptr) {
    auto op = detail::require_operator<T>(
        OperatorFactory<T>::getSoftmaxOperator(platform_), platform_,
        "Softmax");
    (*op)(output, input, dim, mask, offset, stream);
  }

  // Flash Attention Prefill算子
  void flash_attention_prefill(const Tensor<T>& Q, const Tensor<T>& K,
                               const Tensor<T>& V, Tensor<T>& output,
                               int n_heads, int n_kv_heads, int head_dim,
                               int seq_len, int total_seq_len, int offset,
                               cudaStream_t stream = nullptr) {
    if (platform_ == OperatorPlatform::CUDA && head_dim != 128) {
      detail::UnifiedCompositeOps<T, UnifiedOperators>::
          flash_attention_prefill_fallback(*this, Q, K, V, output, n_heads,
                                           n_kv_heads, head_dim, seq_len,
                                           total_seq_len, offset, stream);
      return;
    }

    auto op = OperatorFactory<T>::getFlashAttentionPrefillOperator(platform_);
    if (!op) {
      if (platform_ == OperatorPlatform::CPU) {
        detail::UnifiedCompositeOps<T, UnifiedOperators>::
            flash_attention_prefill_fallback(*this, Q, K, V, output, n_heads,
                                             n_kv_heads, head_dim, seq_len,
                                             total_seq_len, offset, stream);
        return;
      }
      throw std::runtime_error(
          "Flash Attention Prefill operator not registered for the current "
          "platform");
    }
    (*op)(Q, K, V, output, n_heads, n_kv_heads, head_dim, seq_len,
          total_seq_len, offset, stream);
  }

  // CUDA 图/融合路径保留桥接，模型层不再直接调用 cuda_OP
  void gemv_qkv_rope(Tensor<T>* hidden_states, const Tensor<T>* merged_qkv_weight,
                     Tensor<T>* q_buf, Tensor<T>* k_buf, Tensor<T>* v_buf,
                     const Tensor<T>* merged_qkv_bias, size_t* d_rope_offset,
                     const Tensor<float>* rope_sin_cos_cache,
                     int* d_offset_array, int layer_idx, int q_dim, int k_dim,
                     int v_dim, int n_heads, int n_kv_heads, int head_dim,
                     cudaStream_t stream = nullptr, int n_layers = 0,
                     int* pingpong = nullptr) {
    runtime_bridge_.gemv_qkv_rope(hidden_states, merged_qkv_weight, q_buf,
                                  k_buf, v_buf, merged_qkv_bias,
                                  d_rope_offset, rope_sin_cos_cache,
                                  d_offset_array, layer_idx, q_dim, k_dim,
                                  v_dim, n_heads, n_kv_heads, head_dim, stream,
                                  n_layers, pingpong);
  }

  void rope_with_precomputed_cache(Tensor<T>* tensor, const size_t* d_offset,
                                   const Tensor<float>* rope_sin_cos_cache,
                                   cudaStream_t stream = nullptr,
                                   int* d_offset_array = nullptr,
                                   int layer_idx = 0, int n_layers = 0,
                                   int* pingpong = nullptr) {
    runtime_bridge_.rope_with_precomputed_cache(
        tensor, d_offset, rope_sin_cos_cache, stream, d_offset_array,
        layer_idx, n_layers, pingpong);
  }

  void flash_attention_graph_fixed(Tensor<T>& q, const Tensor<T>& k,
                                   const Tensor<T>& v, T** d_output_ptrs,
                                   int* d_segment_info, int n_kv_heads,
                                   cudaStream_t stream = nullptr,
                                   int* pingpong = nullptr) {
    runtime_bridge_.flash_attention_graph_fixed(q, k, v, d_output_ptrs,
                                                d_segment_info, n_kv_heads,
                                                stream, pingpong);
  }

  void gather_fa_graph_fixed(T** d_output_ptrs, Tensor<T>& att_heads,
                             int* d_segment_info,
                             cudaStream_t stream = nullptr) {
    runtime_bridge_.gather_fa_graph_fixed(d_output_ptrs, att_heads,
                                          d_segment_info, stream);
  }

  void gemv_mlp_fused(Tensor<T>* hidden_states, const Tensor<T>* merged_mlp_weight,
                      Tensor<T>* gate_buf_silu,
                      cudaStream_t stream = nullptr) {
    runtime_bridge_.gemv_mlp_fused(hidden_states, merged_mlp_weight,
                                   gate_buf_silu, stream);
  }

 private:
  Device device_;
  OperatorPlatform platform_;
  detail::UnifiedRuntimeBridge<T> runtime_bridge_;
};

}  // namespace op
