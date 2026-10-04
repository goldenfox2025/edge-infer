#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "qwen3.hpp"
#include "operators/cuda/gather_cuda.cuh"
#include "operators/cuda/rope_cuda.cuh"
#include "operators/cuda/flash_attention_prefill_cuda.cuh"
#include "tensor.hpp"
#include "graph_runner.hpp"

// The model contributes prepared weights; the session contributes buffers and
// execution resources. Decode and prefill share this explicit operator sequence.
template <typename T>
void Qwen3Session<T>::run_decoder(const Tensor<uint32_t>& input, KVCache<T>& cache,
                                  DecoderBuffers& buffers, bool prefill) {
  const auto& config = model_->config();
  const size_t rows = input.numel();
  const size_t offset = cache.size() - rows;
  auto read = [](const Tensor<T>& tensor) {
    return op::ArrayView<const T>{tensor.data_ptr(), tensor.numel()};
  };
  auto write = [](Tensor<T>& tensor) {
    return op::ArrayView<T>{tensor.data_ptr(), tensor.numel()};
  };
  auto norm = [&](Tensor<T>& output, const Tensor<T>& source,
                  const Tensor<T>& weight, size_t norm_rows, size_t width) {
    op::cuda::rms_norm<T>(read(source), read(weight), write(output),
                          norm_rows, width, config.rms_norm_eps, execution_stream_);
  };
  op::GatherCUDAOperator<T> gather;
  op::RopeCUDAOperator<T> rope;
  op::FlashAttentionPrefillCUDAOperator<T> prompt_attention;
  gather.op::GatherCUDAOperator<T>::operator()(
      &buffers.residual, &input, &model_->embedding(), execution_stream_);

  for (size_t index = 0; index < model_->layers().size(); ++index) {
    const auto& layer = model_->layers()[index];
    norm(buffers.hidden, buffers.residual, *layer.attention_norm, rows, config.hidden_size);
    operators_->matmul(&buffers.q, &buffers.hidden, layer.q, layer.q_bias, execution_stream_);
    operators_->matmul(&buffers.k, &buffers.hidden, layer.k, layer.k_bias, execution_stream_);
    operators_->matmul(&buffers.v, &buffers.hidden, layer.v, layer.v_bias, execution_stream_);
    norm(buffers.q, buffers.q, *layer.q_norm, rows * config.n_heads, config.head_dim);
    norm(buffers.k, buffers.k, *layer.k_norm, rows * config.n_kv_heads, config.head_dim);

    auto q = static_cast<const Tensor<T>&>(buffers.q).view({rows, config.n_heads, config.head_dim});
    auto k = static_cast<const Tensor<T>&>(buffers.k).view({rows, config.n_kv_heads, config.head_dim});
    auto v = static_cast<const Tensor<T>&>(buffers.v).view({rows, config.n_kv_heads, config.head_dim});
    // view on a const source makes a new descriptor, preserving typed buffer shapes.
    rope.op::RopeCUDAOperator<T>::operator()(&q, offset, config.rope_theta, execution_stream_);
    rope.op::RopeCUDAOperator<T>::operator()(&k, offset, config.rope_theta, execution_stream_);
    copy_kv_cache(index, offset, &cache, k, v, execution_stream_);
    const auto cached = cache.get_contiguous_tensor(index);
    auto cached_k = cached.first.view({cache.size(), config.n_kv_heads, config.head_dim});
    auto cached_v = cached.second.view({cache.size(), config.n_kv_heads, config.head_dim});
    auto attention = static_cast<const Tensor<T>&>(buffers.attention).view({rows, config.n_heads, config.head_dim});
    if (prefill) {
      prompt_attention.op::FlashAttentionPrefillCUDAOperator<T>::operator()(
          q, cached_k, cached_v, attention, static_cast<int>(config.n_heads),
          static_cast<int>(config.n_kv_heads), static_cast<int>(config.head_dim),
          static_cast<int>(rows), static_cast<int>(cache.size()),
          static_cast<int>(offset), execution_stream_);
    } else {
      decode_attention_->op::DynamicFlashAttentionCUDAOperator<T>::operator()(
          q, cached_k, cached_v, attention, static_cast<int>(config.n_kv_heads), execution_stream_);
    }
    operators_->matmul(&buffers.attention_projected, &buffers.attention,
                        layer.o, layer.o_bias, execution_stream_);
    op::cuda::add<T>(read(buffers.residual), read(buffers.attention_projected),
                     write(buffers.residual), execution_stream_);
    norm(buffers.hidden, buffers.residual, *layer.ffn_norm, rows, config.hidden_size);
    operators_->matmul(&buffers.gate, &buffers.hidden, layer.gate, layer.gate_bias, execution_stream_);
    operators_->matmul(&buffers.up, &buffers.hidden, layer.up, layer.up_bias, execution_stream_);
    op::cuda::silu<T>(read(buffers.gate), write(buffers.gate), execution_stream_);
    op::cuda::multiply<T>(read(buffers.gate), read(buffers.up), write(buffers.gate), execution_stream_);
    operators_->matmul(&buffers.ffn, &buffers.gate, layer.down, layer.down_bias, execution_stream_);
    op::cuda::add<T>(read(buffers.residual), read(buffers.ffn), write(buffers.residual), execution_stream_);
  }
  norm(buffers.hidden, buffers.residual, model_->output_norm(), rows, config.hidden_size);
  operators_->matmul(&buffers.logits, &buffers.hidden, model_->output_weight(),
                      nullptr, execution_stream_);
}

// -------------------------------
// forward: Forward entry point
// -------------------------------
template <typename T>
uint32_t *Qwen3Session<T>::forward(const Tensor<uint32_t> *input,
                                 ThreadPool &thread_pool, KVCacheBase *kv_cache,
                                 size_t top_k, float temperature, float top_p,
                                 curandState *d_states) {
  KVCache<T> *typed_cache = dynamic_cast<KVCache<T> *>(kv_cache);
  Tensor<T> logits;
  cudaStream_t sample_stream = execution_stream_;

  if (use_cuda_graph_) {
    logits = forward_for_graph_logits_only(input, typed_cache);
    sample_stream = graph_runtime_ ? graph_runtime_->graph_stream : nullptr;
  } else {
    logits = forward_eager(input, typed_cache);
  }

  operators_->sample_to_fixed(std::move(logits), sampled_token_.data_ptr(),
                               temperature, top_p, top_k, d_states, sample_stream);
  synchronize();
  return sampled_token_.data_ptr();
}

// -------------------------------
// forward_eager: Device-independent eager forward implementation
// -------------------------------
template <typename T>
Tensor<T> Qwen3Session<T>::forward_eager(const Tensor<uint32_t> *input,
                                       KVCache<T> *kv_cache) {
  validate_and_bind(input, kv_cache, true);
  run_decoder(*input, *kv_cache, decode_buffers_, false);
  synchronize();
  return decode_buffers_.logits;
}

template <typename T>
Tensor<T> Qwen3Session<T>::forward_graph_cuda(const Tensor<uint32_t> *input,
                                            KVCache<T> *kv_cache,
                                            cudaStream_t stream) {
  auto &graph = *graph_runtime_;
  if (input->device() != Device::CUDA) {
    throw std::runtime_error("Input tensor must be on CUDA device");
  }

  const size_t seq_len = 1;
  const size_t offset = kv_cache->size() - seq_len;

  Tensor<T>& residual = graph_tensor("residual");
  Tensor<T>& hidden_states = graph_tensor("hidden_states");

  operators_->gather(&residual, &graph.graph_input_tensor,
                     &model_->get_params().at("token_embeddings.weight"), stream);

  for (size_t i = 0; i < model_->config().n_layers; i++) {
    auto &attention_norm_weight = model_->get_params().at("rms_att_w" + std::to_string(i));
    operators_->rms_norm(&hidden_states, &residual, &attention_norm_weight,
                         model_->config().rms_norm_eps, stream);

    const auto& layer = model_->layers()[i];

    Tensor<T>& q_buf = graph_tensor("q_buf_" + std::to_string(i));
    Tensor<T> &k_buf = graph.fixed_k_buffers[i];
    Tensor<T> &v_buf = graph.fixed_v_buffers[i];

    auto q_weight = layer.q;
    auto k_weight = layer.k;
    auto v_weight = layer.v;

    operators_->matmul(&q_buf, &hidden_states, q_weight, layer.q_bias, stream);
    operators_->matmul(&k_buf, &hidden_states, k_weight, layer.k_bias, stream);
    operators_->matmul(&v_buf, &hidden_states, v_weight, layer.v_bias, stream);

    auto &q_norm_weight = model_->get_params().at("q_norm" + std::to_string(i));
    auto &k_norm_weight = model_->get_params().at("k_norm" + std::to_string(i));
    Tensor<T> q_buf_view = static_cast<const Tensor<T>&>(q_buf).view({seq_len, model_->config().n_heads, model_->config().head_dim});
    Tensor<T> k_buf_view = static_cast<const Tensor<T>&>(k_buf).view({seq_len, model_->config().n_kv_heads, model_->config().head_dim});
    Tensor<T> v_buf_view = static_cast<const Tensor<T>&>(v_buf).view({seq_len, model_->config().n_kv_heads, model_->config().head_dim});

    operators_->rms_norm(&q_buf_view, &q_buf_view, &q_norm_weight,
                         model_->config().rms_norm_eps, stream);
    operators_->rms_norm(&k_buf_view, &k_buf_view, &k_norm_weight,
                         model_->config().rms_norm_eps, stream);
    operators_->rope_with_precomputed_cache(&q_buf_view, graph.d_rope_offset,
                                            &model_->rope_cache(), stream,
                                            nullptr, static_cast<int>(i),
                                            static_cast<int>(model_->config().n_layers),
                                            graph.pingpong);
    operators_->rope_with_precomputed_cache(&k_buf_view, graph.d_rope_offset,
                                            &model_->rope_cache(), stream,
                                            nullptr, static_cast<int>(i),
                                            static_cast<int>(model_->config().n_layers),
                                            graph.pingpong);

    copy_kv_cache(i, offset, kv_cache, k_buf_view, v_buf_view, stream);

    auto [k_cache_tensor, v_cache_tensor] = kv_cache->get_contiguous_tensor(i);
    Tensor<T> k_cache_view =
        k_cache_tensor.view({kv_cache->size(), model_->config().n_kv_heads, model_->config().head_dim});
    Tensor<T> v_cache_view =
        v_cache_tensor.view({kv_cache->size(), model_->config().n_kv_heads, model_->config().head_dim});
    Tensor<T> attn_output =
        run_decode_graph_attention(q_buf_view, k_cache_view, v_cache_view, i,
                                   stream);
    Tensor<T>& attn_proj = graph_tensor("attn_proj_" + std::to_string(i));
    auto o_weight = layer.o;
    operators_->matmul(&attn_proj, &attn_output, o_weight, layer.o_bias,
                       stream);
    operators_->add(&residual, &residual, &attn_proj, stream);

    auto &ffn_norm_weight = model_->get_params().at("rms_ffn_w" + std::to_string(i));
    operators_->rms_norm(&hidden_states, &residual, &ffn_norm_weight,
                         model_->config().rms_norm_eps, stream);


    Tensor<T>& gate_buf = graph_tensor("gate_buf_" + std::to_string(i));
    Tensor<T>& up_buf = graph_tensor("up_buf_" + std::to_string(i));
    auto gate_weight = layer.gate;
    auto up_weight = layer.up;
    operators_->matmul(&gate_buf, &hidden_states, gate_weight, layer.gate_bias,
                       stream);
    operators_->matmul(&up_buf, &hidden_states, up_weight, layer.up_bias,
                       stream);
    operators_->silu(&gate_buf, &gate_buf, stream);
    operators_->multiply(&gate_buf, &gate_buf, &up_buf, stream);

    Tensor<T>& ffn_out = graph_tensor("ffn_out_" + std::to_string(i));
    auto down_weight = layer.down;
    operators_->matmul(&ffn_out, &gate_buf, down_weight, layer.down_bias,
                       stream);
    operators_->add(&residual, &residual, &ffn_out, stream);
  }

  auto &norm_weight = model_->get_params().at("rms_out_w");
  Tensor<T>& final_h = graph_tensor("final_h");
  operators_->rms_norm(&final_h, &residual, &norm_weight, model_->config().rms_norm_eps,
                       stream);

  auto lm_head_weight = get_weight("lm_head");
  operators_->matmul(&graph.graph_output_tensor, &final_h, lm_head_weight,
                     nullptr, stream);
  return graph.graph_output_tensor;
}

template <typename T>
Tensor<T> Qwen3Session<T>::forward_for_graph_logits_only(
    const Tensor<uint32_t> *input, KVCache<T> *kv_cache) {
  validate_and_bind(input, kv_cache, true);
  auto &graph = *graph_runtime_;
  if (!graph.graph_initialized) {
    initialize_cuda_graph_with_kv_cache(kv_cache);
  }

  cudaMemcpyAsync(graph.graph_input_tensor.data_ptr(), input->data_ptr(),
                  sizeof(uint32_t), cudaMemcpyDeviceToDevice,
                  graph.graph_stream);

  const size_t offset = kv_cache->size() - 1;
  const size_t total_seq_len = kv_cache->size();
  GraphRunner<T>::update_kv_copy_nodes(graph, kv_cache, offset, model_->config().n_layers);
  prepare_graph_execution(offset, total_seq_len, graph.graph_stream,
                          graph.pingpong_index);

  GraphRunner<T>::launch(graph, "Qwen3");
  synchronize();
  return graph.graph_output_tensor;
}

template uint32_t* Qwen3Session<__nv_bfloat16>::forward(
    const Tensor<uint32_t>*, ThreadPool&, KVCacheBase*, size_t, float, float, curandState*);
template void Qwen3Session<__nv_bfloat16>::run_decoder(
    const Tensor<uint32_t>&, KVCache<__nv_bfloat16>&, DecoderBuffers&, bool);
template Tensor<__nv_bfloat16> Qwen3Session<__nv_bfloat16>::forward_eager(
    const Tensor<uint32_t>*, KVCache<__nv_bfloat16>*);
template Tensor<__nv_bfloat16> Qwen3Session<__nv_bfloat16>::forward_graph_cuda(
    const Tensor<uint32_t>*, KVCache<__nv_bfloat16>*, cudaStream_t);
template Tensor<__nv_bfloat16> Qwen3Session<__nv_bfloat16>::forward_for_graph_logits_only(
    const Tensor<uint32_t>*, KVCache<__nv_bfloat16>*);
