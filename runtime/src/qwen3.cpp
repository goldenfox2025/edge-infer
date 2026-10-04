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
#include "tensor.hpp"

namespace {

bool qwen3_graph_enabled_by_default() {
  const char* value = std::getenv("EDGE_INFER_ENABLE_QWEN3_GRAPH");
  return value != nullptr && std::string(value) == "1";
}

}  // namespace

// -------------------------------

// -------------------------------
template <typename T>
Qwen3Session<T>::Qwen3Session(std::shared_ptr<const Qwen3Model<T>> model)
    : Qwen3Session(std::move(model), qwen3_graph_enabled_by_default()) {}

template <typename T>
Qwen3Session<T>::Qwen3Session(std::shared_ptr<const Qwen3Model<T>> model,
                             bool enable_graph)
    : model_(std::move(model)), use_cuda_graph_(enable_graph) {
  if (!model_) throw std::invalid_argument("Qwen3 session requires a model");
  int device_id = 0;
  CUDA_CHECK(cudaGetDevice(&device_id));
  if (device_id != model_->cuda_device_id()) {
    throw std::invalid_argument("Qwen3 session must be constructed on its model's CUDA device");
  }
  try {
    init_runtime_state();
  } catch (...) {
    if (execution_stream_) cudaStreamSynchronize(execution_stream_);
    if (graph_runtime_) graph_runtime_->release_streams();
    decode_attention_.reset();
    operators_.reset();
    if (cublas_handle_) cublasDestroy(cublas_handle_);
    if (execution_stream_) cudaStreamDestroy(execution_stream_);
    throw;
  }
}

template <typename T>
Qwen3Session<T>::Qwen3Session(
    const std::unordered_map<std::string, Tensor<T>>& params,
    const ModelConfig& config)
    : Qwen3Session(std::make_shared<Qwen3Model<T>>(params, config)) {}

template <typename T>
Qwen3Session<T>::Qwen3Session(
    const std::unordered_map<std::string, Tensor<T>>& params,
    const std::unordered_map<std::string, Tensor<int32_t>>& qweights,
    const std::unordered_map<std::string, Tensor<T>>& scales,
    const std::unordered_map<std::string, Tensor<int32_t>>& qzeros,
    const ModelConfig& config)
    : Qwen3Session(std::make_shared<Qwen3Model<T>>(params, qweights, scales,
                                                 qzeros, config)) {}

template <typename T>
std::unique_ptr<Qwen3Session<T>> Qwen3Session<T>::create(
    std::shared_ptr<const Qwen3Model<T>> model, size_t context_capacity,
    bool enable_graph) {
  if (!model) throw std::invalid_argument("Qwen3 session requires a model");
  const auto& config = model->config();
  if (!context_capacity || context_capacity > config.max_position_embeddings) {
    throw std::invalid_argument("Qwen3 session context capacity must be positive and not exceed the model limit");
  }
  auto session = std::make_unique<Qwen3Session<T>>(std::move(model), enable_graph);
  session->managed_cache_ = std::make_unique<KVCache<T>>(
      config.n_layers, context_capacity, config.n_kv_heads * config.head_dim,
      Device::CUDA);
  return session;
}

template <typename T>
std::unique_ptr<Qwen3Session<T>> Qwen3Session<T>::new_session(
    size_t context_capacity, bool enable_graph) const {
  return create(model_, context_capacity, enable_graph);
}

template <typename T>
void Qwen3Session<T>::require_managed_cache() const {
  if (!managed_cache_) {
    throw std::logic_error("Qwen3 context operations require a managed session created with create or new_session");
  }
}

template <typename T>
size_t Qwen3Session<T>::context_size() const {
  require_managed_cache();
  return managed_cache_->size();
}

template <typename T>
size_t Qwen3Session<T>::context_capacity() const {
  require_managed_cache();
  return managed_cache_->get_max_seq_len();
}

template <typename T>
void Qwen3Session<T>::reset() {
  require_managed_cache();
  synchronize();
  managed_cache_->clear();
  managed_history_ready_ = false;
}

template <typename T>
Tensor<T> Qwen3Session<T>::prefill(const Tensor<uint32_t>& input) {
  require_managed_cache();
  if (!input.numel()) {
    throw std::invalid_argument("Qwen3 managed prefill requires a nonempty prompt");
  }
  if (input.numel() > managed_cache_->get_max_seq_len()) {
    throw std::length_error("Qwen3 prompt exceeds session context capacity");
  }
  const size_t previous_size = managed_cache_->size();
  try {
    // The prompt begins at cache offset zero, retaining the fixed backing storage.
    managed_cache_->resize(input.numel());
    auto logits = prefill_eager(&input, managed_cache_.get());
    managed_history_ready_ = true;
    return logits;
  } catch (...) {
    // Validation rejects before model writes. CUDA execution failures do not
    // promise preservation of existing KV values, even when length is restored.
    managed_cache_->resize(previous_size);
    throw;
  }
}

template <typename T>
Tensor<T> Qwen3Session<T>::decode(const Tensor<uint32_t>& input) {
  require_managed_cache();
  if (!managed_history_ready_) {
    throw std::logic_error("Qwen3 managed decode requires a successful prefill");
  }
  if (input.numel() != 1) {
    throw std::invalid_argument("Qwen3 managed decode requires exactly one token");
  }
  const size_t previous_size = managed_cache_->size();
  if (previous_size >= managed_cache_->get_max_seq_len()) {
    throw std::length_error("Qwen3 session context capacity is exhausted");
  }
  try {
    managed_cache_->resize(previous_size + 1);
    return use_cuda_graph_
        ? forward_for_graph_logits_only(&input, managed_cache_.get())
        : forward_eager(&input, managed_cache_.get());
  } catch (...) {
    managed_cache_->resize(previous_size);
    throw;
  }
}

template <typename T>
void Qwen3Session<T>::init_runtime_state() {
  CUDA_CHECK(cudaStreamCreateWithFlags(&execution_stream_, cudaStreamNonBlocking));
  CUBLAS_CHECK(cublasCreate(&cublas_handle_));
  CUBLAS_CHECK(cublasSetPointerMode(cublas_handle_, CUBLAS_POINTER_MODE_HOST));
  CUBLAS_CHECK(cublasSetStream(cublas_handle_, execution_stream_));
  operators_ = std::make_unique<op::UnifiedOperators<T>>(Device::CUDA, cublas_handle_);
  graph_runtime_ = std::make_unique<CudaGraphRuntime<T>>();
  CUDA_CHECK(cudaStreamCreateWithFlags(&graph_runtime_->graph_stream,
                                       cudaStreamNonBlocking));
  decode_workspace_ = std::make_unique<CudaWorkspaceArena>();
  const size_t heads = model_->config().n_heads;
  const size_t head_dim = model_->config().head_dim;
  if (head_dim > std::numeric_limits<size_t>::max() - 2 ||
      heads > std::numeric_limits<size_t>::max() / (head_dim + 2) / 5 / sizeof(T)) {
    throw std::overflow_error("Qwen3 attention workspace size overflow");
  }
  const size_t attention_elements = 5 * heads * (head_dim + 2);
  attention_storage_.reserve(attention_elements * sizeof(T));
  attention_workspace_ = Tensor<T>::from_external_buffer(
      attention_storage_.template ptr_at<T>(0), {attention_elements}, Device::CUDA);
  decode_attention_ = std::make_unique<op::DynamicFlashAttentionCUDAOperator<T>>(
      attention_workspace_);
  decode_buffers_ = prepare_buffers(1, *decode_workspace_);
  sampling_workspace_.reserve(sizeof(uint32_t));
  sampled_token_ = Tensor<uint32_t>::from_external_buffer(
      sampling_workspace_.template ptr_at<uint32_t>(0), {1}, Device::CUDA);
}

template <typename T>
void Qwen3Session<T>::synchronize() const {
  if (execution_stream_) CUDA_CHECK(cudaStreamSynchronize(execution_stream_));
  if (graph_runtime_ && graph_runtime_->graph_stream) {
    CUDA_CHECK(cudaStreamSynchronize(graph_runtime_->graph_stream));
  }
}

template <typename T>
void Qwen3Session<T>::set_graph_enabled(bool enabled) {
  synchronize();
  use_cuda_graph_ = enabled;
}

template <typename T>
void Qwen3Session<T>::validate_and_bind(const Tensor<uint32_t>* input,
                                       KVCache<T>* cache, bool decode) {
  if (!input || !cache) throw std::invalid_argument("Qwen3 session requires input and KV cache");
  if (managed_cache_ && cache != managed_cache_.get()) {
    throw std::invalid_argument("Qwen3 managed session requires its owned KV cache");
  }
  int device_id = 0;
  CUDA_CHECK(cudaGetDevice(&device_id));
  if (device_id != model_->cuda_device_id()) {
    throw std::invalid_argument("Qwen3 session cannot execute on another CUDA device");
  }
  if (input->device() != Device::CUDA || input->sizes().size() != 1 ||
      !input->is_contiguous() || !input->numel() || (decode && input->numel() != 1)) {
    throw std::invalid_argument("Qwen3 session input must be a nonempty contiguous rank-one CUDA token tensor; decode requires one token");
  }
  const auto& config = model_->config();
  if (cache->device() != Device::CUDA || cache->get_n_layers() != config.n_layers ||
      cache->get_head_dim() != config.n_kv_heads * config.head_dim ||
      !cache->get_max_seq_len() || cache->get_max_seq_len() > config.max_position_embeddings ||
      cache->size() < input->numel() || cache->size() > cache->get_max_seq_len()) {
    throw std::invalid_argument("Qwen3 session KV cache has incompatible device, layers, width, capacity or extent");
  }
  if (config.head_dim != 128) {
    throw std::invalid_argument("Qwen3 session attention currently requires head_dim 128");
  }
  auto require_device_pointer = [&](const void* pointer) {
    cudaPointerAttributes attributes{};
    const auto status = cudaPointerGetAttributes(&attributes, pointer);
    if (status != cudaSuccess) {
      cudaGetLastError();
      throw std::invalid_argument("Qwen3 session requires valid CUDA input and cache pointers");
    }
    if (attributes.type != cudaMemoryTypeDevice || attributes.device != model_->cuda_device_id()) {
      throw std::invalid_argument("Qwen3 session input and KV cache must reside on its model's CUDA device");
    }
  };
  require_device_pointer(input->data_ptr());
  std::vector<std::pair<const T*, const T*>> bases;
  bases.reserve(config.n_layers);
  for (size_t layer = 0; layer < config.n_layers; ++layer) {
    const auto tensors = cache->get_contiguous_tensor(layer);
    require_device_pointer(tensors.first.data_ptr());
    require_device_pointer(tensors.second.data_ptr());
    bases.emplace_back(tensors.first.data_ptr(), tensors.second.data_ptr());
  }
  if (bound_cache_ && (bound_cache_ != cache || bound_capacity_ != cache->get_max_seq_len() ||
                       bases != bound_kv_bases_)) {
    throw std::invalid_argument("Qwen3 session KV cache binding cannot change; create another session");
  }
  // Gather skips invalid ids. Reject them before any activation or KV write.
  std::vector<uint32_t> host_ids(input->numel());
  CUDA_CHECK(cudaMemcpy(host_ids.data(), input->data_ptr(),
                         host_ids.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  for (uint32_t id : host_ids) {
    if (id >= config.vocab_size) {
      throw std::invalid_argument("Qwen3 session token id exceeds model vocabulary");
    }
  }
  if (!bound_cache_) {
    bound_cache_ = cache;
    bound_capacity_ = cache->get_max_seq_len();
    bound_kv_bases_ = std::move(bases);
  }
}

template <typename T>
size_t Qwen3Session<T>::decoder_workspace_bytes(size_t rows) const {
  if (!rows) return 0;
  const auto& config = model_->config();
  if (rows > config.max_position_embeddings) {
    throw std::invalid_argument("Qwen3 workspace row count exceeds model context capacity");
  }
  constexpr size_t alignment = 256;
  size_t bytes = 0;
  const size_t q_width = config.n_heads * config.head_dim;
  const size_t kv_width = config.n_kv_heads * config.head_dim;
  for (size_t width : {config.hidden_size, config.hidden_size, q_width,
                       kv_width, kv_width, q_width, config.hidden_size,
                       config.intermediate_size, config.intermediate_size,
                       config.hidden_size, config.vocab_size}) {
    if (rows > static_cast<size_t>(std::numeric_limits<int>::max()) / width ||
        rows * width > (std::numeric_limits<size_t>::max() - alignment) / sizeof(T)) {
      throw std::overflow_error("Qwen3 decoder tensor extent exceeds supported range");
    }
    const size_t allocation = (rows * width * sizeof(T) + alignment - 1) & ~(alignment - 1);
    if (bytes > std::numeric_limits<size_t>::max() - allocation) {
      throw std::overflow_error("Qwen3 decoder workspace size overflow");
    }
    bytes += allocation;
  }
  return bytes;
}

template <typename T>
size_t Qwen3Session<T>::estimate_prefill_workspace_bytes(size_t rows) const {
  return decoder_workspace_bytes(rows);
}

template <typename T>
typename Qwen3Session<T>::DecoderBuffers
Qwen3Session<T>::prepare_buffers(size_t rows, CudaWorkspaceArena& arena) {
  arena.reserve(decoder_workspace_bytes(rows));
  const auto& config = model_->config();
  constexpr size_t alignment = 256;
  size_t offset = 0;
  auto take = [&](size_t width) {
    auto result = Tensor<T>::from_external_buffer(arena.template ptr_at<T>(offset),
                                                  {rows, width}, Device::CUDA);
    offset += (rows * width * sizeof(T) + alignment - 1) & ~(alignment - 1);
    return result;
  };
  DecoderBuffers buffers;
  buffers.residual = take(config.hidden_size);
  buffers.hidden = take(config.hidden_size);
  buffers.q = take(config.n_heads * config.head_dim);
  buffers.k = take(config.n_kv_heads * config.head_dim);
  buffers.v = take(config.n_kv_heads * config.head_dim);
  buffers.attention = take(config.n_heads * config.head_dim);
  buffers.attention_projected = take(config.hidden_size);
  buffers.gate = take(config.intermediate_size);
  buffers.up = take(config.intermediate_size);
  buffers.ffn = take(config.hidden_size);
  buffers.logits = take(config.vocab_size);
  return buffers;
}

template <typename T>
std::vector<size_t> Qwen3Session<T>::decode_tensor_shape(const std::string& name) const {
  if (name == "residual" || name == "hidden_states" || name == "attn_proj" ||
      name == "ffn_out" || name == "final_h") {
    return {1, model_->config().hidden_size};
  }
  if (name == "q_buf" || name == "attn_output") {
    return {1, model_->config().n_heads * model_->config().head_dim};
  }
  if (name == "k_buf" || name == "v_buf") {
    return {1, model_->config().n_kv_heads * model_->config().head_dim};
  }
  if (name == "gate_buf" || name == "up_buf") {
    return {1, model_->config().intermediate_size};
  }
  if (name == "logits") {
    return {1, model_->config().vocab_size};
  }
  if (name == "att_heads") {
    return {model_->config().n_heads, model_->config().head_dim};
  }
  if (name == "fa_output") {
    return {model_->config().n_heads, model_->config().head_dim + 2};
  }
  throw std::runtime_error("Unknown Qwen3 decode tensor shape request: " + name);
}

template <typename T>
Tensor<T>& Qwen3Session<T>::graph_tensor(const std::string& name) {
  return graph_tensors_.at(name);
}

template <typename T>
Tensor<T> Qwen3Session<T>::run_decode_graph_attention(Tensor<T>& q_buf_view,
                                                    const Tensor<T>& k_cache_view,
                                                    const Tensor<T>& v_cache_view,
                                                    size_t layer,
                                                    cudaStream_t stream) {
  auto& graph = *graph_runtime_;
  Tensor<T>& att_heads = graph_tensor("att_heads_" + std::to_string(layer));
  operators_->flash_attention_graph_fixed(q_buf_view, k_cache_view, v_cache_view,
                                          graph.d_output_ptrs,
                                          graph.d_segment_info, model_->config().n_kv_heads,
                                          stream, graph.pingpong);
  operators_->gather_fa_graph_fixed(graph.d_output_ptrs, att_heads,
                                    graph.d_segment_info, stream);
  return static_cast<const Tensor<T>&>(att_heads).view(
      {1, model_->config().n_heads * model_->config().head_dim});
}
template Tensor<__nv_bfloat16>
Qwen3Session<__nv_bfloat16>::run_decode_graph_attention(
    Tensor<__nv_bfloat16>& q_buf_view,
    const Tensor<__nv_bfloat16>& k_cache_view,
    const Tensor<__nv_bfloat16>& v_cache_view, size_t layer,
    cudaStream_t stream);

template <typename T>
void Qwen3Session<T>::copy_kv_cache(size_t layer, size_t offset,
                                  KVCache<T> *kv_cache,
                                  const Tensor<T> &k_buf_view,
                                  const Tensor<T> &v_buf_view,
                                  cudaStream_t stream) const {
  const size_t seq_len = k_buf_view.sizes().at(0);
  const size_t head_size = model_->config().n_kv_heads * model_->config().head_dim;

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

template <typename T>
Qwen3Session<T>::~Qwen3Session() {
  int previous_device = model_->cuda_device_id();
  cudaGetDevice(&previous_device);
  if (previous_device != model_->cuda_device_id()) cudaSetDevice(model_->cuda_device_id());
  if (execution_stream_) cudaStreamSynchronize(execution_stream_);
  if (graph_runtime_) {
    auto& graph = *graph_runtime_;
    if (graph.graph_stream) cudaStreamSynchronize(graph.graph_stream);
    graph.release_graph_objects();
    graph.release_fixed_memory();
    graph.release_pingpong();
    graph.release_streams();
  }
  decode_attention_.reset();
  operators_.reset();
  if (cublas_handle_) cublasDestroy(cublas_handle_);
  if (execution_stream_) cudaStreamDestroy(execution_stream_);
  decode_workspace_->release();
  prefill_workspace_.release();
  graph_workspace_.release();
  sampling_workspace_.release();
  attention_storage_.release();
  // Captured graph operations must be destroyed before their cache addresses.
  managed_cache_.reset();
  if (previous_device != model_->cuda_device_id()) cudaSetDevice(previous_device);
}

template <typename T>
bool Qwen3Session<T>::verify_params() const {
  return model_->verify_params();
}

template <typename T>
void Qwen3Session<T>::print_model_info() const {
  const auto& config = model_->config();
  std::cout << "Qwen3 CUDA session: layers=" << config.n_layers
            << ", hidden=" << config.hidden_size
            << ", heads=" << config.n_heads << "/" << config.n_kv_heads
            << ", head_dim=" << config.head_dim
            << ", quantization=" << config.quant_type
            << ", CUDA graph=" << (use_cuda_graph_ ? "enabled" : "disabled")
            << std::endl;
}

template <typename T>
void Qwen3Session<T>::initialize_graph_fixed_memory() {
  auto& graph = *graph_runtime_;
  synchronize();
  graph.release_graph_objects();
  graph.release_fixed_memory();
  graph.release_pingpong();
  graph_tensors_.clear();
  constexpr size_t alignment = 256;
  size_t bytes = 0;
  std::vector<std::pair<std::string, std::vector<size_t>>> layout;
  auto add = [&](const std::string& name, const std::vector<size_t>& shape) {
    size_t elements = 1;
    for (size_t extent : shape) {
      if (extent && elements > std::numeric_limits<size_t>::max() / extent) {
        throw std::overflow_error("Qwen3 graph tensor size overflow");
      }
      elements *= extent;
    }
    if (elements > (std::numeric_limits<size_t>::max() - alignment) / sizeof(T) ||
        bytes > std::numeric_limits<size_t>::max() - elements * sizeof(T) - alignment) {
      throw std::overflow_error("Qwen3 graph workspace size overflow");
    }
    bytes += (elements * sizeof(T) + alignment - 1) & ~(alignment - 1);
    layout.emplace_back(name, shape);
  };
  for (const char* name : {"residual", "hidden_states", "final_h", "logits"}) {
    add(name, decode_tensor_shape(name));
  }
  for (size_t layer = 0; layer < model_->config().n_layers; ++layer) {
    for (const char* name : {"q_buf", "k_buf", "v_buf", "att_heads", "attn_proj", "gate_buf", "up_buf", "ffn_out"}) {
      add(std::string(name) + "_" + std::to_string(layer), decode_tensor_shape(name));
    }
  }
  constexpr int branches = 3;
  for (int branch = 0; branch < branches; ++branch) {
    add("fa_output_" + std::to_string(branch), decode_tensor_shape("fa_output"));
  }
  const size_t input_offset = bytes;
  const size_t segment_offset = bytes + alignment;
  const size_t pointers_offset = bytes + alignment * 2;
  if (bytes > std::numeric_limits<size_t>::max() - alignment * 3) {
    throw std::overflow_error("Qwen3 graph workspace size overflow");
  }
  graph_workspace_.reserve(bytes + alignment * 3);
  size_t offset = 0;
  for (const auto& entry : layout) {
    auto tensor = Tensor<T>::from_external_buffer(
        graph_workspace_.template ptr_at<T>(offset), entry.second, Device::CUDA);
    offset += (tensor.nbytes() + alignment - 1) & ~(alignment - 1);
    graph_tensors_.emplace(entry.first, std::move(tensor));
  }
  CUDA_CHECK(cudaMalloc(&graph.d_rope_offset, sizeof(size_t) * 2));
  CUDA_CHECK(cudaMalloc(&graph.pingpong, sizeof(int)));
  CUDA_CHECK(cudaMemsetAsync(graph.pingpong, 0, sizeof(int), graph.graph_stream));
  graph.pingpong_index = 0;
  graph.graph_input_tensor = Tensor<uint32_t>::from_external_buffer(
      graph_workspace_.template ptr_at<uint32_t>(input_offset), {1}, Device::CUDA);
  graph.graph_output_tensor = graph_tensor("logits");
  for (size_t layer = 0; layer < model_->config().n_layers; ++layer) {
    graph.fixed_k_buffers.push_back(graph_tensor("k_buf_" + std::to_string(layer)));
    graph.fixed_v_buffers.push_back(graph_tensor("v_buf_" + std::to_string(layer)));
  }
  graph.segment_info_tensor = Tensor<int>::from_external_buffer(
      graph_workspace_.template ptr_at<int>(segment_offset), {2}, Device::CUDA);
  graph.d_segment_info = graph.segment_info_tensor.data_ptr();
  graph.output_ptrs_tensor = Tensor<T*>::from_external_buffer(
      graph_workspace_.template ptr_at<T*>(pointers_offset), {branches}, Device::CUDA);
  graph.d_output_ptrs = graph.output_ptrs_tensor.data_ptr();
  std::vector<T*> output_ptrs;
  for (int branch = 0; branch < branches; ++branch) {
    graph.fixed_fa_outputs.push_back(graph_tensor("fa_output_" + std::to_string(branch)));
    output_ptrs.push_back(graph.fixed_fa_outputs.back().data_ptr());
  }
  CUDA_CHECK(cudaMemcpyAsync(graph.d_output_ptrs, output_ptrs.data(), branches * sizeof(T*),
                             cudaMemcpyHostToDevice, graph.graph_stream));
  CUDA_CHECK(cudaStreamSynchronize(graph.graph_stream));
}

template <typename T>
void Qwen3Session<T>::prepare_graph_execution(size_t rope_offset,
                                            size_t total_seq_len,
                                            cudaStream_t stream,
                                            int pingpong_index) {
  auto &graph = *graph_runtime_;
  graph_pingpong_host_ = pingpong_index;
  graph_rope_offsets_[pingpong_index] = rope_offset;
  graph_segment_lengths_[pingpong_index] = static_cast<int>(total_seq_len);
  CUDA_CHECK(cudaMemcpyAsync(graph.pingpong, &graph_pingpong_host_, sizeof(int),
                             cudaMemcpyHostToDevice, stream));
  if (graph.d_rope_offset) {
    CUDA_CHECK(cudaMemcpyAsync(graph.d_rope_offset + pingpong_index,
                               graph_rope_offsets_ + pingpong_index,
                               sizeof(size_t), cudaMemcpyHostToDevice, stream));
  }
  if (graph.d_segment_info) {
    CUDA_CHECK(cudaMemcpyAsync(graph.d_segment_info + pingpong_index,
                               graph_segment_lengths_ + pingpong_index, sizeof(int),
                               cudaMemcpyHostToDevice, stream));
  }
}

template <typename T>
void Qwen3Session<T>::initialize_cuda_graph_with_kv_cache(KVCache<T> *kv_cache) {
  auto &graph = *graph_runtime_;
  if (graph.graph_initialized) {
    return;
  }

  initialize_graph_fixed_memory();
  const uint32_t init_token = model_->config().bos_token_id < model_->config().vocab_size
                                  ? model_->config().bos_token_id : 0;
  CUDA_CHECK(cudaMemcpyAsync(graph.graph_input_tensor.data_ptr(), &init_token, sizeof(uint32_t),
                             cudaMemcpyHostToDevice, graph.graph_stream));
  CUDA_CHECK(cudaStreamSynchronize(graph.graph_stream));

  GraphRunner<T>::initialize(
      graph, "Qwen3",
      [&]() {
        prepare_graph_execution(kv_cache->size() - 1, kv_cache->size(), graph.graph_stream,
                                graph.pingpong_index);
        Tensor<T> warmup_output =
            forward_graph_cuda(&graph.graph_input_tensor, kv_cache, graph.graph_stream);
        (void)warmup_output;
        CUDA_CHECK(cudaStreamSynchronize(graph.graph_stream));
      },
      [&]() {
        return forward_graph_cuda(&graph.graph_input_tensor, kv_cache,
                                  graph.graph_stream);
      });
  GraphRunner<T>::extract_kv_copy_nodes(graph, model_->config().n_layers, model_->config().n_kv_heads,
                                        model_->config().head_dim);
}

// -------------------------------

// -------------------------------
template <typename T>
Qwen3Session<T>& Qwen3Session<T>::cuda() {
  return *this;
}

template <typename T>
Qwen3Session<T>& Qwen3Session<T>::cpu() {
  throw std::runtime_error("Qwen3Session only supports CUDA execution");
}

template class Qwen3Session<__nv_bfloat16>;
