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
#include "execution/context.hpp"
#include "tensor.hpp"
#include "graph_runner.hpp"

namespace {

}  // namespace

template <typename T>
void Qwen3Model<T>::append_attention_node(
    DecodeRuntimeArtifacts& artifacts, AttentionMode mode,
    const std::string& out_name, std::vector<std::string> inputs,
    size_t layer) const {
  const std::string op_name =
      mode == AttentionMode::Decode ? "dynamic_flash_attention"
                                    : "flash_attention_prefill";
  auto op = operators_->get_operator_base(op_name);
  if (!op) {
    throw std::runtime_error(std::string(
        "Qwen3 decoder eager operator not registered: ") + op_name);
  }
  DecodePreparedNode node{std::move(op), std::move(inputs), {out_name}, {},
                          nullptr, std::nullopt, nullptr,
                          static_cast<uint16_t>(n_heads_), layer};
  node.tensor_args =
      node.op->packed_tensor_args(false, static_cast<uint16_t>(n_heads_));
  artifacts.nodes.push_back(std::move(node));
}

template <typename T>
typename Qwen3Model<T>::DecodeRuntimeArtifacts
Qwen3Model<T>::build_decoder_runtime_artifacts(size_t seq_len,
                                               AttentionMode mode) {
  constexpr size_t kAlignment = CudaMemoryPool::kAllocationAlignment;
  const std::string token_residual_name = "residual_0";
  const std::string final_h_name = "final_h";
  const std::string logits_name = "logits";
  DecodeRuntimeArtifacts artifacts;
  ExecutionBuilder<T> builder(kAlignment);
  auto residual_in_name = [&](size_t layer) {
    return layer == 0 ? token_residual_name
                      : "residual_" + std::to_string(layer);
  };
  auto residual_out_name = [&](size_t layer) {
    return "residual_" + std::to_string(layer + 1);
  };
  auto layer_name = [&](size_t layer, const char* suffix) {
    return "layer_" + std::to_string(layer) + "." + suffix;
  };
  auto append_node = [&](const std::string& op_name, const std::string&,
                         std::vector<std::string> inputs,
                         std::vector<std::string> outputs,
                         const Tensor<T>* tensor_arg = nullptr,
                         std::optional<op::WeightTensor<T>> weight = std::nullopt,
                         const Tensor<T>* bias = nullptr, size_t layer = 0,
                         bool inplace = false, uint16_t view_heads = 0) {
    auto op = operators_->get_operator_base(op_name);
    if (!op) {
      throw std::runtime_error(
          std::string("Qwen3 decoder eager operator not registered: ") +
          op_name);
    }
    DecodePreparedNode node{std::move(op), std::move(inputs), std::move(outputs),
                            {}, tensor_arg, std::move(weight), bias,
                            view_heads, layer};
    node.tensor_args = node.op->packed_tensor_args(inplace, view_heads);
    artifacts.nodes.push_back(std::move(node));
  };

  auto residual = builder.call("gather", token_residual_name,
                               {seq_len, hidden_size_}, {}, "token_embedding");
  append_node("gather", "token_embedding", {}, {residual.name},
              &params_.at("token_embeddings.weight"));

  for (size_t layer = 0; layer < n_layers_; ++layer) {
    auto q_weight = get_weight("wq" + std::to_string(layer));
    auto k_weight = get_weight("wk" + std::to_string(layer));
    auto v_weight = get_weight("wv" + std::to_string(layer));
    auto o_weight = get_weight("wo" + std::to_string(layer));
    auto gate_weight = get_weight("w_gate" + std::to_string(layer));
    auto up_weight = get_weight("w_up" + std::to_string(layer));
    auto down_weight = get_weight("w_down" + std::to_string(layer));
    const auto attn_biases = load_attention_biases(layer);
    const auto mlp_biases = load_mlp_biases(layer);
    const Tensor<T>* attn_norm_weight =
        &params_.at("rms_att_w" + std::to_string(layer));
    const Tensor<T>* ffn_norm_weight =
        &params_.at("rms_ffn_w" + std::to_string(layer));
    const Tensor<T>* q_norm_weight =
        &params_.at("q_norm" + std::to_string(layer));
    const Tensor<T>* k_norm_weight =
        &params_.at("k_norm" + std::to_string(layer));

    auto hidden_attn =
        builder.call("rms_norm", layer_name(layer, "hidden_attn"), {},
                     {residual}, "attn_input_norm");
    append_node("rms_norm", "attn_input_norm", {residual.name},
                {hidden_attn.name}, attn_norm_weight);

    auto q = builder.call("matmul", layer_name(layer, "q_buf"), {seq_len, n_heads_ * head_dim_},
                          {hidden_attn}, "q_proj");
    append_node("matmul", "q_proj", {hidden_attn.name}, {q.name},
                nullptr, q_weight,
                attn_biases.q);

    auto k = builder.call("matmul", layer_name(layer, "k_buf"), {seq_len, n_kv_heads_ * head_dim_},
                          {hidden_attn}, "k_proj");
    append_node("matmul", "k_proj", {hidden_attn.name}, {k.name},
                nullptr, k_weight,
                attn_biases.k);

    auto v = builder.call("matmul", layer_name(layer, "v_buf"), {seq_len, n_kv_heads_ * head_dim_},
                          {hidden_attn}, "v_proj");
    append_node("matmul", "v_proj", {hidden_attn.name}, {v.name},
                nullptr, v_weight,
                attn_biases.v);

    q = builder.call_inplace("rms_norm", q, {q}, "q_norm");
    append_node("rms_norm", "q_norm", {q.name}, {q.name},
                q_norm_weight, std::nullopt, nullptr, 0, true,
                static_cast<uint16_t>(n_heads_));

    k = builder.call_inplace("rms_norm", k, {k}, "k_norm");
    append_node("rms_norm", "k_norm", {k.name}, {k.name},
                k_norm_weight, std::nullopt, nullptr, 0, true,
                static_cast<uint16_t>(n_kv_heads_));

    q = builder.call_inplace("rope", q, {q}, "q_rope");
    append_node("rope", "q_rope", {q.name}, {q.name},
                nullptr, std::nullopt, nullptr, 0, true,
                static_cast<uint16_t>(n_heads_));

    k = builder.call_inplace("rope", k, {k}, "k_rope");
    append_node("rope", "k_rope", {k.name}, {k.name},
                nullptr, std::nullopt, nullptr, 0, true,
                static_cast<uint16_t>(n_kv_heads_));

    builder.call_void("kv_cache_write", {k, v}, "kv_cache_write");
    append_node("kv_cache_write", "kv_cache_write", {k.name, v.name}, {},
                nullptr, std::nullopt, nullptr, layer, false,
                static_cast<uint16_t>(n_kv_heads_));

    auto attn_output = build_attention_value(builder, mode,
                                             layer_name(layer, "attn_output"), {q, k, v});
    append_attention_node(artifacts, mode, attn_output.name,
                          {q.name, k.name, v.name}, layer);

    auto attn_proj = builder.call("matmul", layer_name(layer, "attn_proj"), {seq_len, hidden_size_},
                                  {attn_output}, "o_proj");
    append_node("matmul", "o_proj", {attn_output.name}, {attn_proj.name},
                nullptr, o_weight,
                attn_biases.o);

    auto residual_mid =
        builder.call("add", layer_name(layer, "residual_mid"), {}, {residual, attn_proj},
                     "attn_residual");
    append_node("add", "attn_residual", {residual.name, attn_proj.name},
                {residual_mid.name});

    auto hidden_ffn =
        builder.call("rms_norm", layer_name(layer, "hidden_ffn"), {}, {residual_mid},
                     "ffn_input_norm");
    append_node("rms_norm", "ffn_input_norm", {residual_mid.name},
                {hidden_ffn.name}, ffn_norm_weight);

    auto gate = builder.call("matmul", layer_name(layer, "gate_buf"), {seq_len, intermediate_size_},
                             {hidden_ffn}, "gate_proj");
    append_node("matmul", "gate_proj", {hidden_ffn.name}, {gate.name},
                nullptr, gate_weight,
                mlp_biases.gate);

    auto up = builder.call("matmul", layer_name(layer, "up_buf"), {seq_len, intermediate_size_},
                           {hidden_ffn}, "up_proj");
    append_node("matmul", "up_proj", {hidden_ffn.name}, {up.name},
                nullptr, up_weight,
                mlp_biases.up);

    gate = builder.call_inplace("silu", gate, {gate}, "gate_silu");
    append_node("silu", "gate_silu", {gate.name}, {gate.name},
                nullptr, std::nullopt, nullptr, 0, true);

    gate = builder.call_inplace("multiply", gate, {gate, up}, "gate_up_mul");
    append_node("multiply", "gate_up_mul", {gate.name, up.name}, {gate.name},
                nullptr, std::nullopt, nullptr, 0, true);

    auto ffn_out = builder.call("matmul", layer_name(layer, "ffn_out"), {seq_len, hidden_size_},
                                {gate}, "down_proj");
    append_node("matmul", "down_proj", {gate.name}, {ffn_out.name},
                nullptr, down_weight,
                mlp_biases.down);

    residual = builder.call("add", residual_out_name(layer), {}, {residual_mid, ffn_out},
                            "ffn_residual");
    append_node("add", "ffn_residual", {residual_mid.name, ffn_out.name},
                {residual.name});
  }

  auto final_h = builder.call("rms_norm", final_h_name, {}, {residual},
                              "final_norm");
  append_node("rms_norm", "final_norm", {residual.name}, {final_h.name},
              &params_.at("rms_out_w"));

  auto logits = builder.call("matmul", logits_name, {seq_len, vocab_size_},
                             {final_h}, "lm_head");
  append_node("matmul", "lm_head", {final_h.name}, {logits.name},
              nullptr, get_weight("lm_head"),
              nullptr);

  artifacts.program = builder.program();
  return artifacts;
}

// -------------------------------
// forward: 前向传播接口
// -------------------------------
template <typename T>
uint32_t *Qwen3Model<T>::forward(const Tensor<uint32_t> *input,
                                 ThreadPool &thread_pool, KVCacheBase *kv_cache,
                                 size_t top_k, float temperature, float top_p,
                                 curandState *d_states) {
  KVCache<T> *typed_cache = dynamic_cast<KVCache<T> *>(kv_cache);
  Tensor<T> logits;
  cudaStream_t sample_stream = nullptr;

  if (device_ == Device::CUDA && use_cuda_graph_) {
    logits = forward_for_graph_logits_only(input, typed_cache);
    sample_stream = graph_runtime_ ? graph_runtime_->graph_stream : nullptr;
  } else {
    logits = forward_eager(input, typed_cache);
  }

  return operators_->sample(std::move(logits), temperature, top_p, top_k, d_states,
                            sample_stream);
}

// -------------------------------
// forward_eager: 设备无关的即时执行前向传播实现
// -------------------------------
template <typename T>
Tensor<T> Qwen3Model<T>::forward_eager(const Tensor<uint32_t> *input,
                                       KVCache<T> *kv_cache) {
  if (input->device() != Device::CUDA) {
    throw std::runtime_error("Input tensor must be on CUDA device");
  }

  const size_t seq_len = 1;
  size_t offset = 0;
  if (kv_cache) {
    if (kv_cache->device() != Device::CUDA) {
      throw std::runtime_error("KVCache must be on CUDA device");
    }

    offset = kv_cache->size() - seq_len;
  }

  auto buffers = materialize_decode_eager_buffers();
  if (decode_eager_nodes_.empty()) {
    throw std::runtime_error("Qwen3 decode eager prepared nodes are not initialized");
  }
  decode_eager_frame_.input = input;
  decode_eager_frame_.kv_cache = kv_cache;
  decode_eager_frame_.seq_len = seq_len;
  decode_eager_frame_.offset = offset;

  for (const auto& node : decode_eager_nodes_) {
    node.execute(
        [&](const std::string& name) -> Tensor<T>& { return buffers.require(name); },
        [&](size_t layer) {
          return decode_eager_frame_.kv_cache->get_contiguous_tensor(layer);
        },
        decode_eager_frame_.input, decode_eager_frame_.seq_len,
        decode_eager_frame_.offset, n_heads_, n_kv_heads_, head_dim_,
        rms_norm_eps_, rope_theta_);
  }

  return buffers.require("logits");
}

template <typename T>
Tensor<T> Qwen3Model<T>::forward_graph_cuda(const Tensor<uint32_t> *input,
                                            KVCache<T> *kv_cache,
                                            cudaStream_t stream) {
  auto &graph = *graph_runtime_;
  if (input->device() != Device::CUDA) {
    throw std::runtime_error("Input tensor must be on CUDA device");
  }

  const size_t seq_len = 1;
  const size_t offset = kv_cache->size() - seq_len;

  Tensor<T> residual(decode_tensor_shape("residual"), Device::CUDA, false,
                     graph_tensor_tag("residual"));
  Tensor<T> hidden_states(decode_tensor_shape("hidden_states"), Device::CUDA, false,
                          graph_tensor_tag("hidden_states"));

  operators_->gather(&residual, &graph.graph_input_tensor,
                     &params_.at("token_embeddings.weight"), stream);

  for (size_t i = 0; i < n_layers_; i++) {
    auto &attention_norm_weight = params_.at("rms_att_w" + std::to_string(i));
    operators_->rms_norm(&hidden_states, &residual, &attention_norm_weight,
                         rms_norm_eps_, stream);

    const auto attn_biases = load_attention_biases(i);

    Tensor<T> q_buf(decode_tensor_shape("q_buf"), Device::CUDA, false,
                    graph_layer_tensor_tag("q_buf", i));
    Tensor<T> &k_buf = graph.fixed_k_buffers[i];
    Tensor<T> &v_buf = graph.fixed_v_buffers[i];

    auto q_weight = get_weight("wq" + std::to_string(i));
    auto k_weight = get_weight("wk" + std::to_string(i));
    auto v_weight = get_weight("wv" + std::to_string(i));

    operators_->matmul(&q_buf, &hidden_states, q_weight, attn_biases.q, stream);
    operators_->matmul(&k_buf, &hidden_states, k_weight, attn_biases.k, stream);
    operators_->matmul(&v_buf, &hidden_states, v_weight, attn_biases.v, stream);

    auto &q_norm_weight = params_.at("q_norm" + std::to_string(i));
    auto &k_norm_weight = params_.at("k_norm" + std::to_string(i));
    Tensor<T> q_buf_view = q_buf.view({seq_len, n_heads_, head_dim_});
    Tensor<T> k_buf_view = k_buf.view({seq_len, n_kv_heads_, head_dim_});
    Tensor<T> v_buf_view = v_buf.view({seq_len, n_kv_heads_, head_dim_});

    operators_->rms_norm(&q_buf_view, &q_buf_view, &q_norm_weight,
                         rms_norm_eps_, stream);
    operators_->rms_norm(&k_buf_view, &k_buf_view, &k_norm_weight,
                         rms_norm_eps_, stream);
    operators_->rope_with_precomputed_cache(&q_buf_view, graph.d_rope_offset,
                                            &rope_sin_cos_cache_, stream,
                                            nullptr, static_cast<int>(i),
                                            static_cast<int>(n_layers_),
                                            graph.pingpong);
    operators_->rope_with_precomputed_cache(&k_buf_view, graph.d_rope_offset,
                                            &rope_sin_cos_cache_, stream,
                                            nullptr, static_cast<int>(i),
                                            static_cast<int>(n_layers_),
                                            graph.pingpong);

    copy_kv_cache(i, offset, kv_cache, k_buf_view, v_buf_view, stream);

    auto [k_cache_tensor, v_cache_tensor] = kv_cache->get_contiguous_tensor(i);
    Tensor<T> k_cache_view =
        k_cache_tensor.view({kv_cache->size(), n_kv_heads_, head_dim_});
    Tensor<T> v_cache_view =
        v_cache_tensor.view({kv_cache->size(), n_kv_heads_, head_dim_});
    Tensor<T> attn_output =
        run_decode_graph_attention(q_buf_view, k_cache_view, v_cache_view, i,
                                   stream);
    Tensor<T> attn_proj(decode_tensor_shape("attn_proj"), Device::CUDA, false,
                        graph_layer_tensor_tag("attn_proj", i));
    auto o_weight = get_weight("wo" + std::to_string(i));
    operators_->matmul(&attn_proj, &attn_output, o_weight, attn_biases.o,
                       stream);
    operators_->add(&residual, &residual, &attn_proj, stream);

    auto &ffn_norm_weight = params_.at("rms_ffn_w" + std::to_string(i));
    operators_->rms_norm(&hidden_states, &residual, &ffn_norm_weight,
                         rms_norm_eps_, stream);

    const auto mlp_biases = load_mlp_biases(i);
    Tensor<T> gate_buf(decode_tensor_shape("gate_buf"), Device::CUDA, false,
                       graph_layer_tensor_tag("gate_buf", i));
    Tensor<T> up_buf(decode_tensor_shape("up_buf"), Device::CUDA, false,
                     graph_layer_tensor_tag("up_buf", i));
    auto gate_weight = get_weight("w_gate" + std::to_string(i));
    auto up_weight = get_weight("w_up" + std::to_string(i));
    operators_->matmul(&gate_buf, &hidden_states, gate_weight, mlp_biases.gate,
                       stream);
    operators_->matmul(&up_buf, &hidden_states, up_weight, mlp_biases.up,
                       stream);
    operators_->silu(&gate_buf, &gate_buf, stream);
    operators_->multiply(&gate_buf, &gate_buf, &up_buf, stream);

    Tensor<T> ffn_out(decode_tensor_shape("ffn_out"), Device::CUDA, false,
                      graph_layer_tensor_tag("ffn_out", i));
    auto down_weight = get_weight("w_down" + std::to_string(i));
    operators_->matmul(&ffn_out, &gate_buf, down_weight, mlp_biases.down,
                       stream);
    operators_->add(&residual, &residual, &ffn_out, stream);
  }

  auto &norm_weight = params_.at("rms_out_w");
  Tensor<T> final_h(decode_tensor_shape("final_h"), Device::CUDA, false,
                    graph_tensor_tag("final_h"));
  operators_->rms_norm(&final_h, &residual, &norm_weight, rms_norm_eps_,
                       stream);

  auto lm_head_weight = get_weight("lm_head");
  operators_->matmul(&graph.graph_output_tensor, &final_h, lm_head_weight,
                     nullptr, stream);
  return graph.graph_output_tensor;
}

template <typename T>
Tensor<T> Qwen3Model<T>::forward_for_graph_logits_only(
    const Tensor<uint32_t> *input, KVCache<T> *kv_cache) {
  auto &graph = *graph_runtime_;
  if (!graph.graph_initialized) {
    initialize_cuda_graph_with_kv_cache(kv_cache);
  }

  cudaMemcpyAsync(graph.graph_input_tensor.data_ptr(), input->data_ptr(),
                  sizeof(uint32_t), cudaMemcpyDeviceToDevice,
                  graph.graph_stream);

  const size_t offset = kv_cache->size() - 1;
  const size_t total_seq_len = kv_cache->size();
  GraphRunner<T>::update_kv_copy_nodes(graph, kv_cache, offset, n_layers_);
  prepare_graph_execution(offset, total_seq_len, graph.graph_stream,
                          graph.pingpong_index);

  GraphRunner<T>::launch(graph, "Qwen3");
  return graph.graph_output_tensor;
}

// 显式实例化模板函数
template uint32_t *Qwen3Model<__nv_bfloat16>::forward(
    const Tensor<uint32_t> *input, ThreadPool &thread_pool,
    KVCacheBase *kv_cache, size_t top_k, float temperature, float top_p,
    curandState *d_states);
template typename Qwen3Model<__nv_bfloat16>::DecodeRuntimeArtifacts
Qwen3Model<__nv_bfloat16>::build_decoder_runtime_artifacts(
    size_t seq_len, AttentionMode mode);
template void Qwen3Model<__nv_bfloat16>::append_attention_node(
    DecodeRuntimeArtifacts& artifacts, AttentionMode mode,
    const std::string& out_name, std::vector<std::string> inputs,
    size_t layer) const;

template Tensor<__nv_bfloat16> Qwen3Model<__nv_bfloat16>::forward_eager(
    const Tensor<uint32_t> *input, KVCache<__nv_bfloat16> *kv_cache);
template Tensor<__nv_bfloat16> Qwen3Model<__nv_bfloat16>::forward_graph_cuda(
    const Tensor<uint32_t> *input, KVCache<__nv_bfloat16> *kv_cache,
    cudaStream_t stream);
template Tensor<__nv_bfloat16>
Qwen3Model<__nv_bfloat16>::forward_for_graph_logits_only(
    const Tensor<uint32_t> *input, KVCache<__nv_bfloat16> *kv_cache);
