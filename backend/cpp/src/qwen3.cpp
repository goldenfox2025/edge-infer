#include "../include/qwen3.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "graph_runner.hpp"
#include "execution/context.hpp"
#include "execution/program.hpp"
#include "tensor.hpp"

namespace {

bool verbose_weight_debug() {
  const char *value = std::getenv("LLM_INFER_VERBOSE_WEIGHTS");
  return value != nullptr && std::string(value) == "1";
}

bool qwen3_graph_enabled_by_default() {
  const char* value = std::getenv("LLM_INFER_ENABLE_QWEN3_GRAPH");
  return value != nullptr && std::string(value) == "1";
}

}  // namespace

// -------------------------------
// Qwen3Model<T> 构造函数
// -------------------------------
template <typename T>
Qwen3Model<T>::Qwen3Model(
    const std::unordered_map<std::string, Tensor<T>> &params,
    const ModelConfig &config)
    : params_(params) {
  load_config(config);
  init_runtime_state();
}

// 带量化参数的构造函数
template <typename T>
Qwen3Model<T>::Qwen3Model(
    const std::unordered_map<std::string, Tensor<T>> &params,
    const std::unordered_map<std::string, Tensor<int32_t>> &qweight_params,
    const std::unordered_map<std::string, Tensor<T>> &scales_params,
    const std::unordered_map<std::string, Tensor<int32_t>> &qzeros_params,
    const ModelConfig &config)
    : params_(params), qweight_params_(qweight_params),
      scales_params_(scales_params), qzeros_params_(qzeros_params) {
  quant_type_ = 1;
  if (config.find("group_size") != config.end()) {
    group_size_ = config.at("group_size");
  }
  load_config(config);
  init_runtime_state();
}

template <typename T>
void Qwen3Model<T>::load_config(
    const ModelConfig &config) {
  vocab_size_ = config.at("vocab_size");
  n_layers_ = config.at("n_layers");
  n_heads_ = config.at("n_heads");
  n_kv_heads_ = config.at("n_kv_heads");
  hidden_size_ = config.at("hidden_size");
  intermediate_size_ = config.at("intermediate_size");
  max_position_embeddings_ = config.at("max_position_embeddings");
  bos_token_id_ = static_cast<uint32_t>(config.at("bos_token_id"));
  eos_token_id_ = static_cast<uint32_t>(config.at("eos_token_id"));
  rms_norm_eps_ = static_cast<float>(config.at("rms_norm_eps"));
  rope_theta_ = static_cast<float>(config.at("rope_theta"));

  if (config.find("head_dim") != config.end()) {
    head_dim_ = config.at("head_dim");
    std::cout << "使用配置中的head_dim: " << head_dim_ << std::endl;
  } else {
    if (hidden_size_ % n_heads_ != 0) {
      throw std::runtime_error(
          "hidden_size must be divisible by n_heads when head_dim is absent");
    }
    head_dim_ = hidden_size_ / n_heads_;
  }
}

template <typename T> void Qwen3Model<T>::init_runtime_state() {
  device_ = Device::CUDA;
  use_cuda_graph_ = qwen3_graph_enabled_by_default();
  operators_ = std::make_unique<op::UnifiedOperators<T>>(device_);
  graph_runtime_ = std::make_unique<CudaGraphRuntime<T>>();
  decode_workspace_ = std::make_unique<CudaWorkspaceArena>();

  for (auto &stream : compute_streams_) {
    cudaStreamCreate(&stream);
  }

  for (auto &event : fa_done_events_) {
    cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
  }

  auto &graph = *graph_runtime_;
  if (!graph.graph_stream) {
    cudaStreamCreate(&graph.graph_stream);
  }
  if (!graph.prep_stream) {
    cudaStreamCreate(&graph.prep_stream);
  }
  precompute_rope_cache();
  initialize_decode_workspace();
}

template <typename T> void Qwen3Model<T>::precompute_rope_cache() {
  if (device_ != Device::CUDA || head_dim_ == 0 || max_position_embeddings_ == 0) {
    return;
  }

  std::vector<float> freq(head_dim_ / 2);
  for (size_t i = 0; i < head_dim_ / 2; ++i) {
    freq[i] = 1.0f / powf(rope_theta_, (2.0f * static_cast<float>(i)) /
                                           static_cast<float>(head_dim_));
  }

  std::vector<float> sin_cos_cpu(max_position_embeddings_ * head_dim_);
  for (size_t pos = 0; pos < max_position_embeddings_; ++pos) {
    for (size_t i = 0; i < head_dim_ / 2; ++i) {
      float angle = static_cast<float>(pos) * freq[i];
      size_t base_idx = pos * head_dim_ + i * 2;
      sin_cos_cpu[base_idx] = sinf(angle);
      sin_cos_cpu[base_idx + 1] = cosf(angle);
    }
  }

  rope_sin_cos_cache_ = Tensor<float>({max_position_embeddings_, head_dim_},
                                      Device::CUDA, false,
                                      "qwen3_rope_sin_cos_cache");
  cudaError_t err =
      cudaMemcpy(rope_sin_cos_cache_.data_ptr(), sin_cos_cpu.data(),
                 sin_cos_cpu.size() * sizeof(float), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to initialize Qwen3 RoPE cache: " +
                             std::string(cudaGetErrorString(err)));
  }
}

template <typename T>
size_t Qwen3Model<T>::estimate_prefill_workspace_bytes(size_t seq_len) const {
  if (seq_len == 0) {
    return 0;
  }
  auto artifacts = const_cast<Qwen3Model<T>*>(this)->build_decoder_runtime_artifacts(
      seq_len, AttentionMode::Prefill);
  const auto plan =
      build_workspace_plan_from_execution_program<T>(artifacts.program);
  return plan.total_bytes();
}

template <typename T> void Qwen3Model<T>::initialize_decode_workspace() {
  auto artifacts = build_decoder_runtime_artifacts(1, AttentionMode::Decode);
  decode_workspace_plan_ =
      std::make_unique<WorkspacePlan>(
          build_workspace_plan_from_execution_program<T>(artifacts.program));
  decode_values_ = std::move(artifacts.program.values);
  decode_eager_nodes_ = std::move(artifacts.nodes);
  if (!decode_workspace_ || !decode_workspace_plan_ || decode_workspace_plan_->empty()) {
    return;
  }
  decode_workspace_->reserve_for_plan(*decode_workspace_plan_);
}

template <typename T>
Tensor<T> Qwen3Model<T>::decode_workspace_tensor(const std::string& name,
                                                 const std::vector<size_t>& shape) const {
  if (!decode_workspace_ || !decode_workspace_plan_) {
    throw std::runtime_error("Qwen3 decode workspace is not initialized");
  }
  const auto& allocation = decode_workspace_plan_->at(name);
  const size_t requested_bytes = Tensor<T>::from_external_buffer(
                                     decode_workspace_->template ptr_at<T>(allocation.offset), shape, Device::CUDA)
                                     .nbytes();
  if (requested_bytes > allocation.bytes) {
    throw std::runtime_error("Qwen3 decode workspace allocation too small for " + name);
  }
  return Tensor<T>::from_external_buffer(
      decode_workspace_->template ptr_at<T>(allocation.offset), shape, Device::CUDA);
}

template <typename T>
typename Qwen3Model<T>::DecodeEagerBuffers
Qwen3Model<T>::materialize_decode_eager_buffers() const {
  if (decode_values_.empty()) {
    throw std::runtime_error("Qwen3 decode value layout is not initialized");
  }

  DecodeEagerBuffers buffers;
  buffers.tensors.reserve(decode_values_.size());
  for (const auto& value : decode_values_) {
    buffers.tensors.emplace(value.name, decode_workspace_tensor(value.name, value.shape));
  }
  return buffers;
}

template typename Qwen3Model<__nv_bfloat16>::DecodeEagerBuffers
Qwen3Model<__nv_bfloat16>::materialize_decode_eager_buffers() const;

template <typename T>
std::vector<size_t> Qwen3Model<T>::decode_tensor_shape(const std::string& name) const {
  if (name == "residual" || name == "hidden_states" || name == "attn_proj" ||
      name == "ffn_out" || name == "final_h") {
    return {1, hidden_size_};
  }
  if (name == "q_buf" || name == "attn_output") {
    return {1, n_heads_ * head_dim_};
  }
  if (name == "k_buf" || name == "v_buf") {
    return {1, n_kv_heads_ * head_dim_};
  }
  if (name == "gate_buf" || name == "up_buf") {
    return {1, intermediate_size_};
  }
  if (name == "logits") {
    return {1, vocab_size_};
  }
  if (name == "att_heads") {
    return {n_heads_, head_dim_};
  }
  if (name == "fa_output") {
    return {n_heads_, head_dim_ + 2};
  }
  throw std::runtime_error("Unknown Qwen3 decode tensor shape request: " + name);
}

template <typename T>
std::string Qwen3Model<T>::graph_tensor_tag(const std::string& name) const {
  return "qwen3_graph_" + name;
}

template <typename T>
std::string Qwen3Model<T>::graph_layer_tensor_tag(const std::string& name, size_t layer) const {
  return graph_tensor_tag(name) + "_" + std::to_string(layer);
}

template <typename T>
const Tensor<T> *Qwen3Model<T>::find_optional_param(
    const std::string &key) const {
  auto it = params_.find(key);
  return it == params_.end() ? nullptr : &it->second;
}

template <typename T>
typename Qwen3Model<T>::AttentionBiases Qwen3Model<T>::load_attention_biases(
    size_t layer) const {
  const std::string layer_prefix = "layers." + std::to_string(layer) + ".";
  return {
      find_optional_param(layer_prefix + "self_attn.q_proj.bias"),
      find_optional_param(layer_prefix + "self_attn.k_proj.bias"),
      find_optional_param(layer_prefix + "self_attn.v_proj.bias"),
      find_optional_param(layer_prefix + "self_attn.o_proj.bias"),
  };
}

template <typename T>
typename Qwen3Model<T>::MlpBiases Qwen3Model<T>::load_mlp_biases(
    size_t layer) const {
  const std::string layer_prefix = "layers." + std::to_string(layer) + ".";
  return {
      find_optional_param(layer_prefix + "mlp.gate_proj.bias"),
      find_optional_param(layer_prefix + "mlp.up_proj.bias"),
      find_optional_param(layer_prefix + "mlp.down_proj.bias"),
  };
}

template <typename T>
ExecutionValueHandle Qwen3Model<T>::build_attention_value(
    ExecutionBuilder<T>& builder, AttentionMode mode, const std::string& out_name,
    std::vector<ExecutionValueHandle> inputs) const {
  return builder.call(mode == AttentionMode::Decode ? "dynamic_flash_attention"
                                                    : "flash_attention_prefill",
                      out_name, {}, std::move(inputs), attention_semantic(mode));
}

template <typename T>
Tensor<T> Qwen3Model<T>::run_decode_graph_attention(Tensor<T>& q_buf_view,
                                                    const Tensor<T>& k_cache_view,
                                                    const Tensor<T>& v_cache_view,
                                                    size_t layer,
                                                    cudaStream_t stream) {
  auto& graph = *graph_runtime_;
  Tensor<T> att_heads(decode_tensor_shape("att_heads"), Device::CUDA, false,
                      graph_layer_tensor_tag("att_heads", layer));
  operators_->flash_attention_graph_fixed(q_buf_view, k_cache_view, v_cache_view,
                                          graph.d_output_ptrs,
                                          graph.d_segment_info, n_kv_heads_,
                                          stream, graph.pingpong);
  operators_->gather_fa_graph_fixed(graph.d_output_ptrs, att_heads,
                                    graph.d_segment_info, stream);
  return att_heads.view({1, n_heads_ * head_dim_});
}
template Tensor<__nv_bfloat16>
Qwen3Model<__nv_bfloat16>::run_decode_graph_attention(
    Tensor<__nv_bfloat16>& q_buf_view,
    const Tensor<__nv_bfloat16>& k_cache_view,
    const Tensor<__nv_bfloat16>& v_cache_view, size_t layer,
    cudaStream_t stream);

template ExecutionValueHandle
Qwen3Model<__nv_bfloat16>::build_attention_value(
    ExecutionBuilder<__nv_bfloat16>& builder, AttentionMode mode,
    const std::string& out_name,
    std::vector<ExecutionValueHandle> inputs) const;

template <typename T>
void Qwen3Model<T>::copy_kv_cache(size_t layer, size_t offset,
                                  KVCache<T> *kv_cache,
                                  const Tensor<T> &k_buf_view,
                                  const Tensor<T> &v_buf_view,
                                  cudaStream_t stream) const {
  const size_t seq_len = k_buf_view.sizes().at(0);
  const size_t head_size = n_kv_heads_ * head_dim_;

  for (size_t token_idx = 0; token_idx < seq_len; ++token_idx) {
    Tensor<T> &k_cache_slice = kv_cache->k_cache(layer, offset + token_idx);
    Tensor<T> &v_cache_slice = kv_cache->v_cache(layer, offset + token_idx);

    cudaMemcpyAsync(k_cache_slice.data_ptr(),
                    k_buf_view.data_ptr() + token_idx * head_size,
                    head_size * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(v_cache_slice.data_ptr(),
                    v_buf_view.data_ptr() + token_idx * head_size,
                    head_size * sizeof(T), cudaMemcpyDeviceToDevice, stream);
  }
}

template <typename T> Qwen3Model<T>::~Qwen3Model() {
  if (graph_runtime_) {
    auto &graph = *graph_runtime_;
    graph.release_fixed_memory();
    graph.release_graph_objects();
    graph.release_streams();
    graph.release_pingpong();
    graph.reset_state();
  }
  for (cudaStream_t stream : compute_streams_) {
    if (stream) {
      // 最好在销毁流之前同步它，确保所有工作完成
      cudaStreamSynchronize(stream);
      cudaStreamDestroy(stream);
    }
  }

  for (int i = 0; i < 3; ++i) {
    if (fa_done_events_[i]) {
      cudaEventDestroy(fa_done_events_[i]);
    }
  }
}

// -------------------------------
// 参数验证：检查全局与层级关键参数是否存在
// -------------------------------
template <typename T> bool Qwen3Model<T>::verify_params() const {
  // 检查基本权重是否存在
  std::vector<std::string> base_weights = {"token_embeddings.weight",
                                           "rms_out_w", "lm_head"};

  for (const auto &weight_name : base_weights) {
    if (params_.find(weight_name) == params_.end()) {
      std::cerr << "缺少基本权重: " << weight_name << std::endl;
      return false;
    }
  }

  // 调试信息: 打印量化参数中的键名
  if (quant_type_ == 1 && verbose_weight_debug()) {
    std::cout << "\n=== 量化参数键名调试信息 ===" << std::endl;
    std::cout << "qweight_params_ 键名 (" << qweight_params_.size()
              << " 项):" << std::endl;
    for (const auto &[key, _] : qweight_params_) {
      std::cout << "  " << key << std::endl;
    }

    std::cout << "scales_params_ 键名 (" << scales_params_.size()
              << " 项):" << std::endl;
    for (const auto &[key, _] : scales_params_) {
      std::cout << "  " << key << std::endl;
    }

    std::cout << "qzeros_params_ 键名 (" << qzeros_params_.size()
              << " 项):" << std::endl;
    for (const auto &[key, _] : qzeros_params_) {
      std::cout << "  " << key << std::endl;
    }
    std::cout << "================================\n" << std::endl;
  }

  // 检查每层权重是否存在
  for (size_t i = 0; i < n_layers_; i++) {
    std::string layer_id = std::to_string(i);
    std::vector<std::string> layer_weights = {
        "rms_att_w" + layer_id, "rms_ffn_w" + layer_id, "wq" + layer_id,
        "wk" + layer_id,        "wv" + layer_id,        "wo" + layer_id,
        "q_norm" + layer_id,    "k_norm" + layer_id,    "w_gate" + layer_id,
        "w_up" + layer_id,      "w_down" + layer_id};

    for (const auto &weight_name : layer_weights) {
      // 非量化权重或者非线性层权重直接在params_中查找
      if (weight_name.find("rms_") == 0 || weight_name.find("q_norm") == 0 ||
          weight_name.find("k_norm") == 0) {
        if (params_.find(weight_name) == params_.end()) {
          std::cerr << "缺少层权重: " << weight_name << std::endl;
          return false;
        }
        continue; // 对于非量化的层级权重，直接检查完成，进入下一个循环
      }

      // 对于可能量化的线性层权重，检查是否存在于params_或量化参数中
      if (params_.find(weight_name) == params_.end()) {
        // 对于量化模型，检查是否存在量化版本的权重
        if (quant_type_ == 1) {
          // 针对线性层权重检查量化版本
          if (weight_name.find("wq") == 0 || weight_name.find("wk") == 0 ||
              weight_name.find("wv") == 0 || weight_name.find("wo") == 0 ||
              weight_name.find("w_gate") == 0 ||
              weight_name.find("w_up") == 0 ||
              weight_name.find("w_down") == 0) {
            // 尝试多种可能的键名格式
            std::vector<std::string> possible_qweight_keys = {
                weight_name,
                weight_name + ".qweight",
            };

            std::vector<std::string> possible_scales_keys = {
                weight_name,
                weight_name + ".scales",
            };

            std::vector<std::string> possible_qzeros_keys = {
                weight_name,
                weight_name + ".qzeros",
            };

            bool found_qweight = false;
            bool found_scales = false;
            bool found_qzeros = false;

            // 检查是否存在任何一种可能的键名
            for (const auto &key : possible_qweight_keys) {
              if (qweight_params_.find(key) != qweight_params_.end()) {
                found_qweight = true;
                break;
              }
            }

            for (const auto &key : possible_scales_keys) {
              if (scales_params_.find(key) != scales_params_.end()) {
                found_scales = true;
                break;
              }
            }

            for (const auto &key : possible_qzeros_keys) {
              if (qzeros_params_.find(key) != qzeros_params_.end()) {
                found_qzeros = true;
                break;
              }
            }

            // 如果找到了所有三种权重，那么认为权重存在
            if (found_qweight && found_scales && found_qzeros) {
              continue; // 权重存在，继续检查下一个权重
            }

            // 打印调试信息
            std::cerr << "缺少层权重: " << weight_name << std::endl;
            if (!found_qweight) {
              std::cerr << "  缺少qweight: ";
              for (const auto &key : possible_qweight_keys) {
                std::cerr << key << " ";
              }
              std::cerr << std::endl;
            }
            if (!found_scales) {
              std::cerr << "  缺少scales: ";
              for (const auto &key : possible_scales_keys) {
                std::cerr << key << " ";
              }
              std::cerr << std::endl;
            }
            if (!found_qzeros) {
              std::cerr << "  缺少qzeros: ";
              for (const auto &key : possible_qzeros_keys) {
                std::cerr << key << " ";
              }
              std::cerr << std::endl;
            }
            return false;
          }
        } else {
          std::cerr << "缺少层权重: " << weight_name << std::endl;
          return false;
        }
      }
    }
  }

  return true;
}

// -------------------------------
// 打印模型信息
// -------------------------------
template <typename T> void Qwen3Model<T>::print_model_info() const {
  std::cout << "\n=== Qwen3 Model Information ===" << std::endl;
  std::cout << "Vocab Size: " << vocab_size_ << std::endl;
  std::cout << "Hidden Size: " << hidden_size_ << std::endl;
  std::cout << "Num Layers: " << n_layers_ << std::endl;
  std::cout << "Num Attention Heads: " << n_heads_ << std::endl;
  std::cout << "Num KV Heads: " << n_kv_heads_ << std::endl;
  std::cout << "Head Dimension: " << head_dim_ << std::endl;
  std::cout << "Intermediate Size: " << intermediate_size_ << std::endl;
  std::cout << "Max Position Embeddings: " << max_position_embeddings_
            << std::endl;
  std::cout << "RMS Norm Epsilon: " << rms_norm_eps_ << std::endl;
  std::cout << "RoPE Theta: " << rope_theta_ << std::endl;

  if (quant_type_ > 0) {
    std::cout << "Quantization: AWQ (group_size=" << group_size_ << ")"
              << std::endl;
  } else {
    std::cout << "Quantization: None" << std::endl;
  }

  std::cout << "Device: " << (device_ == Device::CUDA ? "CUDA" : "CPU")
            << std::endl;
  std::cout << "CUDA Graph: " << (use_cuda_graph_ ? "Enabled" : "Disabled")
            << std::endl;
  std::cout << "================================\n" << std::endl;
}

template <typename T> void Qwen3Model<T>::initialize_graph_fixed_memory() {
  auto &graph = *graph_runtime_;
  cudaError_t err = cudaMalloc(&graph.d_rope_offset, sizeof(size_t) * 2);
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to allocate Qwen3 graph rope offset: " +
                             std::string(cudaGetErrorString(err)));
  }

  cudaMalloc(&graph.pingpong, sizeof(int));
  cudaMemset(graph.pingpong, 0, sizeof(int));
  graph.pingpong_index = 0;

  graph.fixed_k_buffers.clear();
  graph.fixed_v_buffers.clear();
  graph.fixed_k_buffers.reserve(n_layers_);
  graph.fixed_v_buffers.reserve(n_layers_);
  for (size_t i = 0; i < n_layers_; ++i) {
    graph.fixed_k_buffers.emplace_back(decode_tensor_shape("k_buf"), Device::CUDA, false,
                                       graph_layer_tensor_tag("k_buf", i));
    graph.fixed_v_buffers.emplace_back(decode_tensor_shape("v_buf"), Device::CUDA, false,
                                       graph_layer_tensor_tag("v_buf", i));
  }

  graph.kv_copy_nodes.clear();
  constexpr int kFixedBranches = 3;
  graph.segment_info_tensor =
      Tensor<int>({2}, Device::CUDA, false, graph_tensor_tag("segment_info"));
  graph.d_segment_info = graph.segment_info_tensor.data_ptr();
  graph.output_ptrs_tensor = Tensor<T *>({kFixedBranches}, Device::CUDA, false,
                                         graph_tensor_tag("output_ptrs"));
  graph.d_output_ptrs = graph.output_ptrs_tensor.data_ptr();

  graph.fixed_fa_outputs.clear();
  graph.fixed_fa_outputs.reserve(kFixedBranches);
  std::vector<T *> host_output_ptrs(kFixedBranches);
  for (int i = 0; i < kFixedBranches; ++i) {
    graph.fixed_fa_outputs.emplace_back(decode_tensor_shape("fa_output"), Device::CUDA, false,
                                        graph_layer_tensor_tag("fa_output", i));
    host_output_ptrs[i] = graph.fixed_fa_outputs.back().data_ptr();
  }

  cudaMemcpyAsync(graph.d_output_ptrs, host_output_ptrs.data(),
                  kFixedBranches * sizeof(T *), cudaMemcpyHostToDevice,
                  graph.graph_stream);
}

template <typename T>
void Qwen3Model<T>::prepare_graph_execution(size_t rope_offset,
                                            size_t total_seq_len,
                                            cudaStream_t stream,
                                            int pingpong_index) {
  auto &graph = *graph_runtime_;
  cudaMemcpyAsync(graph.pingpong, &pingpong_index, sizeof(int),
                  cudaMemcpyHostToDevice, stream);
  if (graph.d_rope_offset) {
    cudaMemcpyAsync(graph.d_rope_offset + pingpong_index, &rope_offset,
                    sizeof(size_t), cudaMemcpyHostToDevice, stream);
  }
  if (graph.d_segment_info) {
    int value = static_cast<int>(total_seq_len);
    cudaMemcpyAsync(graph.d_segment_info + pingpong_index, &value, sizeof(int),
                    cudaMemcpyHostToDevice, stream);
  }
}

template <typename T>
void Qwen3Model<T>::initialize_cuda_graph_with_kv_cache(KVCache<T> *kv_cache) {
  auto &graph = *graph_runtime_;
  if (graph.graph_initialized) {
    return;
  }

  initialize_graph_fixed_memory();
  graph.graph_input_tensor = Tensor<uint32_t>({1}, Device::CUDA, false,
                                              graph_tensor_tag("input_token"));
  graph.graph_output_tensor =
      Tensor<T>(decode_tensor_shape("logits"), Device::CUDA, false, graph_tensor_tag("logits"));

  uint32_t init_token = 9707;
  cudaMemcpy(graph.graph_input_tensor.data_ptr(), &init_token, sizeof(uint32_t),
             cudaMemcpyHostToDevice);

  GraphRunner<T>::initialize(
      graph, "Qwen3",
      [&]() {
        prepare_graph_execution(kv_cache->size() - 1, kv_cache->size(), nullptr,
                                graph.pingpong_index);
        Tensor<T> warmup_output =
            forward_graph_cuda(&graph.graph_input_tensor, kv_cache, nullptr);
        (void)warmup_output;
        cudaDeviceSynchronize();
      },
      [&]() {
        return forward_graph_cuda(&graph.graph_input_tensor, kv_cache,
                                  graph.graph_stream);
      });
  GraphRunner<T>::extract_kv_copy_nodes(graph, n_layers_, n_kv_heads_,
                                        head_dim_);
}

// -------------------------------
// cuda()：将所有参数移到 CUDA，并设置设备
// -------------------------------
template <typename T> Qwen3Model<T> &Qwen3Model<T>::cuda() {
  for (auto &kv : params_) {
    if (kv.second.device() != Device::CUDA) {
      kv.second.cuda();
    }
  }

  // 移动量化参数到CUDA
  if (quant_type_ == 1) {
    for (auto &kv : qweight_params_) {
      if (kv.second.device() != Device::CUDA) {
        kv.second.cuda();
      }
    }

    for (auto &kv : scales_params_) {
      if (kv.second.device() != Device::CUDA) {
        kv.second.cuda();
      }
    }

    for (auto &kv : qzeros_params_) {
      if (kv.second.device() != Device::CUDA) {
        kv.second.cuda();
      }
    }
  }

  device_ = Device::CUDA;

  // 更新算子接口
  if (operators_) {
    operators_->cuda();
  } else {
    operators_ = std::make_unique<op::UnifiedOperators<T>>(Device::CUDA);
  }

  auto &graph = *graph_runtime_;
  if (!graph.graph_stream) {
    cudaStreamCreate(&graph.graph_stream);
  }
  if (!graph.prep_stream) {
    cudaStreamCreate(&graph.prep_stream);
  }
  precompute_rope_cache();
  use_cuda_graph_ = qwen3_graph_enabled_by_default();

  return *this;
}

// -------------------------------
// cpu()：Qwen3 模型仅支持 CUDA，故调用 cpu() 抛出异常
// -------------------------------
template <typename T> Qwen3Model<T> &Qwen3Model<T>::cpu() {
  use_cuda_graph_ = false;
  if (graph_runtime_) {
    auto &graph = *graph_runtime_;
    graph.release_fixed_memory();
    graph.release_graph_objects();
    graph.release_streams();
    graph.release_pingpong();
    graph.reset_state();
  }
  // 更新算子接口（虽然会抛出异常，但保持一致性）
  if (operators_) {
    operators_->cpu();
  }

  throw std::runtime_error("Qwen3Model only supports CUDA execution.");
  return *this;
}

// 显式实例化模板类
template class Qwen3Model<__nv_bfloat16>;
