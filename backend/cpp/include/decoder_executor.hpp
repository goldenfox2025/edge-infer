#pragma once

#include <algorithm>
#include <stdexcept>

#include "cuda_runtime.h"
#include "inference.hpp"
#include "operators/unified_operators.hpp"
#include "tensor.hpp"
#include "weight_tensor.hpp"

namespace decoder {

struct DecoderGeometry {
  size_t n_layers = 0;
  size_t n_heads = 0;
  size_t n_kv_heads = 0;
  size_t hidden_size = 0;
  size_t head_dim = 0;
  size_t intermediate_size = 0;
  size_t vocab_size = 0;
  float rms_norm_eps = 0.0f;
  float rope_theta = 10000.0f;
};

template <typename T>
struct CommonDecoderWeights {
  Tensor<T>* embedding_table = nullptr;
  Tensor<T>* final_norm_weight = nullptr;
  op::WeightTensor<T> lm_head;
  const Tensor<T>* lm_head_bias = nullptr;
};

template <typename T>
struct DecoderLayerWeights {
  Tensor<T>* input_norm_weight = nullptr;
  op::WeightTensor<T> q_proj;
  const Tensor<T>* q_bias = nullptr;
  op::WeightTensor<T> k_proj;
  const Tensor<T>* k_bias = nullptr;
  op::WeightTensor<T> v_proj;
  const Tensor<T>* v_bias = nullptr;
  op::WeightTensor<T> o_proj;
  const Tensor<T>* o_bias = nullptr;
  Tensor<T>* post_attention_norm_weight = nullptr;
  op::WeightTensor<T> gate_proj;
  const Tensor<T>* gate_bias = nullptr;
  op::WeightTensor<T> up_proj;
  const Tensor<T>* up_bias = nullptr;
  op::WeightTensor<T> down_proj;
  const Tensor<T>* down_bias = nullptr;
};

template <typename T>
inline void write_kv_cache(Device device, KVCache<T>* kv_cache, size_t layer_idx,
                           size_t offset, size_t n_kv_heads, size_t head_dim,
                           const Tensor<T>& k_buf_view,
                           const Tensor<T>& v_buf_view) {
  if (!kv_cache) {
    return;
  }

  const size_t seq_len = k_buf_view.sizes()[0];
  const size_t row_size = n_kv_heads * head_dim;

  for (size_t token_idx = 0; token_idx < seq_len; ++token_idx) {
    Tensor<T>& k_slice = kv_cache->k_cache(layer_idx, offset + token_idx);
    Tensor<T>& v_slice = kv_cache->v_cache(layer_idx, offset + token_idx);

    if (device == Device::CUDA) {
      cudaMemcpy(k_slice.data_ptr(), k_buf_view.data_ptr() + token_idx * row_size,
                 row_size * sizeof(T), cudaMemcpyDeviceToDevice);
      cudaMemcpy(v_slice.data_ptr(), v_buf_view.data_ptr() + token_idx * row_size,
                 row_size * sizeof(T), cudaMemcpyDeviceToDevice);
    } else {
      std::copy(k_buf_view.data_ptr() + token_idx * row_size,
                k_buf_view.data_ptr() + (token_idx + 1) * row_size,
                k_slice.data_ptr());
      std::copy(v_buf_view.data_ptr() + token_idx * row_size,
                v_buf_view.data_ptr() + (token_idx + 1) * row_size,
                v_slice.data_ptr());
    }
  }
}

template <typename T, typename LayerResolver>
Tensor<T> run_generic_decoder(const Tensor<uint32_t>* input, KVCache<T>* kv_cache,
                              Device device, op::UnifiedOperators<T>* operators,
                              const DecoderGeometry& geometry,
                              const CommonDecoderWeights<T>& common_weights,
                              LayerResolver&& resolve_layer) {
  if (!input) {
    throw std::invalid_argument("Input tensor cannot be null");
  }
  if (!kv_cache) {
    throw std::runtime_error("KVCache cannot be null");
  }
  if (!operators) {
    throw std::runtime_error("UnifiedOperators cannot be null");
  }
  if (input->device() != device) {
    throw std::runtime_error("Input tensor device does not match model device");
  }
  if (kv_cache->device() != device) {
    throw std::runtime_error("KVCache device does not match model device");
  }
  if (!common_weights.embedding_table || !common_weights.final_norm_weight) {
    throw std::runtime_error("Decoder common weights are incomplete");
  }

  const size_t seq_len = input->numel();
  const size_t offset = kv_cache->size() - seq_len;

  Tensor<T> residual({seq_len, geometry.hidden_size}, device);
  Tensor<T> hidden_states({seq_len, geometry.hidden_size}, device);
  operators->gather(&residual, input, common_weights.embedding_table);

  for (size_t layer_idx = 0; layer_idx < geometry.n_layers; ++layer_idx) {
    DecoderLayerWeights<T> layer = resolve_layer(layer_idx);
    if (!layer.input_norm_weight || !layer.post_attention_norm_weight) {
      throw std::runtime_error("Decoder layer weights are incomplete");
    }

    operators->rms_norm(&hidden_states, &residual, layer.input_norm_weight,
                        geometry.rms_norm_eps);

    Tensor<T> q_buf({seq_len, geometry.n_heads * geometry.head_dim}, device);
    Tensor<T> k_buf({seq_len, geometry.n_kv_heads * geometry.head_dim}, device);
    Tensor<T> v_buf({seq_len, geometry.n_kv_heads * geometry.head_dim}, device);

    operators->matmul(&q_buf, &hidden_states, layer.q_proj, layer.q_bias);
    operators->matmul(&k_buf, &hidden_states, layer.k_proj, layer.k_bias);
    operators->matmul(&v_buf, &hidden_states, layer.v_proj, layer.v_bias);

    Tensor<T> q_buf_view = q_buf.view({seq_len, geometry.n_heads, geometry.head_dim});
    Tensor<T> k_buf_view = k_buf.view({seq_len, geometry.n_kv_heads, geometry.head_dim});
    Tensor<T> v_buf_view = v_buf.view({seq_len, geometry.n_kv_heads, geometry.head_dim});

    operators->rope(&q_buf_view, offset, geometry.rope_theta);
    operators->rope(&k_buf_view, offset, geometry.rope_theta);
    write_kv_cache(device, kv_cache, layer_idx, offset, geometry.n_kv_heads,
                   geometry.head_dim, k_buf_view, v_buf_view);

    auto [total_k_raw, total_v_raw] = kv_cache->get_contiguous_tensor(layer_idx);
    Tensor<T> total_k =
        total_k_raw.view({offset + seq_len, geometry.n_kv_heads, geometry.head_dim});
    Tensor<T> total_v =
        total_v_raw.view({offset + seq_len, geometry.n_kv_heads, geometry.head_dim});

    Tensor<T> att_heads({seq_len, geometry.n_heads, geometry.head_dim}, device);
    if (seq_len == 1) {
      operators->dynamic_flash_attention(q_buf_view, total_k, total_v, att_heads,
                                         geometry.n_kv_heads);
    } else {
      operators->flash_attention_prefill(
          q_buf_view, total_k, total_v, att_heads, geometry.n_heads,
          geometry.n_kv_heads, geometry.head_dim, seq_len, offset + seq_len,
          offset);
    }

    Tensor<T> att_proj({seq_len, geometry.hidden_size}, device);
    Tensor<T> att_heads_reshaped =
        att_heads.view({seq_len, geometry.n_heads * geometry.head_dim});
    operators->matmul(&att_proj, &att_heads_reshaped, layer.o_proj, layer.o_bias);
    operators->add(&residual, &residual, &att_proj);
    operators->rms_norm(&hidden_states, &residual, layer.post_attention_norm_weight,
                        geometry.rms_norm_eps);

    Tensor<T> gate_buf({seq_len, geometry.intermediate_size}, device);
    Tensor<T> up_buf({seq_len, geometry.intermediate_size}, device);
    operators->matmul(&gate_buf, &hidden_states, layer.gate_proj, layer.gate_bias);
    operators->matmul(&up_buf, &hidden_states, layer.up_proj, layer.up_bias);
    operators->silu(&gate_buf, &gate_buf);
    operators->multiply(&gate_buf, &gate_buf, &up_buf);

    Tensor<T> ffn_out({seq_len, geometry.hidden_size}, device);
    operators->matmul(&ffn_out, &gate_buf, layer.down_proj, layer.down_bias);
    operators->add(&residual, &residual, &ffn_out);
  }

  Tensor<T> final_h({seq_len, geometry.hidden_size}, device);
  operators->rms_norm(&final_h, &residual, common_weights.final_norm_weight,
                      geometry.rms_norm_eps);

  Tensor<T> logits({seq_len, geometry.vocab_size}, device);
  operators->matmul(&logits, &final_h, common_weights.lm_head,
                    common_weights.lm_head_bias);
  return logits;
}

}  // namespace decoder
