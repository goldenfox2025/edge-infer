#include "qwen3.hpp"

#include <cstdlib>
#include <iostream>

#include "common.hpp"

namespace {
bool graph_default() {
  const char* v = std::getenv("EDGE_INFER_ENABLE_QWEN3_GRAPH");
  return v && v[0] == '1' && v[1] == '\0';
}
TensorView<const uint32_t, 1> tokens_from(const Tensor<uint32_t>* in) {
  if (!in || in->device() != Device::CUDA)
    throw std::invalid_argument("Session requires CUDA tokens");
  return borrow_tensor_view<1>(*in);
}
void require_cuda_pointer(const void* ptr, int device) {
  cudaPointerAttributes attributes{};
  const auto status = cudaPointerGetAttributes(&attributes, ptr);
  if (status != cudaSuccess) {
    cudaGetLastError();
    throw std::invalid_argument("Invalid CUDA session pointer");
  }
  if (attributes.type != cudaMemoryTypeDevice || attributes.device != device)
    throw std::invalid_argument("Session pointers must reside on model CUDA device");
}
}  // namespace
template <typename T>
Qwen3Session<T>::Qwen3Session(std::shared_ptr<const Qwen3Model<T>> m)
    : Qwen3Session(std::move(m), graph_default()) {}
template <typename T>
Qwen3Session<T>::Qwen3Session(std::shared_ptr<const Qwen3Model<T>> m, bool graph)
    : model_(std::move(m)), use_cuda_graph_(graph) {
  if (!model_) throw std::invalid_argument("Session requires a prepared model");
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  if (dev != model_->cuda_device_id())
    throw std::invalid_argument("Session must be created on its model CUDA device");
  try {
    initialize();
  } catch (...) {
    release();
    throw;
  }
}
template <typename T>
Qwen3Session<T>::Qwen3Session(const typename Qwen3Model<T>::Parameters& p, const ModelConfig& c)
    : Qwen3Session(std::make_shared<Qwen3Model<T>>(p, c)) {}
template <typename T>
Qwen3Session<T>::Qwen3Session(const typename Qwen3Model<T>::Parameters& p,
                              const typename Qwen3Model<T>::IntegerParameters& q,
                              const typename Qwen3Model<T>::Parameters& s,
                              const typename Qwen3Model<T>::IntegerParameters& z,
                              const ModelConfig& c)
    : Qwen3Session(std::make_shared<Qwen3Model<T>>(p, q, s, z, c)) {}
template <typename T>
void Qwen3Session<T>::initialize() {
  CUDA_CHECK(cudaStreamCreateWithFlags(&context_.stream, cudaStreamNonBlocking));
  CUBLAS_CHECK(cublasCreate(&context_.handle));
  context_ = op::cuda::prepare_execution_context(context_.handle, context_.stream);
  op::cuda::bind_execution_context(context_);
  decode_plan_ = plan_decoder_workspace<T>(model_->config(), 1);
  decode_buffers_ =
      resolve_decoder_buffers<T>(model_->config(), 1, decode_plan_, decode_workspace_);
  graph_storage_.reserve(1280 + 5 * sizeof(float*));
  graph_input_ = {graph_storage_.template ptr_at<uint32_t>(0), {1}, {1}};
  device_offset_ = graph_storage_.template ptr_at<size_t>(256);
  device_lengths_ = graph_storage_.template ptr_at<int>(512);
  device_pingpong_ = graph_storage_.template ptr_at<int>(768);
  graph_branches_ = graph_storage_.template ptr_at<float*>(1024);
  CUDA_CHECK(cudaMemsetAsync(device_pingpong_, 0, sizeof(int), context_.stream));
  const auto& c = model_->config();
  float* branches[5];
  for (size_t i = 0; i < 5; ++i)
    branches[i] = decode_buffers_.attention_scratch.data + i * c.n_heads * (c.head_dim + 2);
  CUDA_CHECK(cudaMemcpyAsync(graph_branches_, branches, sizeof(branches), cudaMemcpyHostToDevice,
                             context_.stream));
  sampling_plan_ = op::cuda::prepare_sampling(context_, c.vocab_size);
  sampling_storage_.reserve(sampling_plan_.total_bytes + 512);
  sampling_scratch_ = {
      sampling_storage_.template ptr_at<unsigned char>(0), {sampling_plan_.total_bytes}, {1}};
  sampled_token_ = {
      sampling_storage_.template ptr_at<uint32_t>(sampling_plan_.total_bytes), {1}, {1}};
  sampled_probability_ = {
      sampling_storage_.template ptr_at<float>(sampling_plan_.total_bytes + 256), {1}, {1}};
  host_ids_.resize(1);
  bound_bases_.reserve(c.n_layers);
  synchronize();
}
template <typename T>
void Qwen3Session<T>::release() noexcept {
  if (!model_) return;
  int prev = model_->cuda_device_id();
  cudaGetDevice(&prev);
  if (prev != model_->cuda_device_id()) cudaSetDevice(model_->cuda_device_id());
  if (context_.stream) cudaStreamSynchronize(context_.stream);
  if (graph_exec_) {
    cudaGraphExecDestroy(graph_exec_);
    graph_exec_ = nullptr;
  }
  if (graph_) {
    cudaGraphDestroy(graph_);
    graph_ = nullptr;
  }
  managed_cache_.reset();
  decode_workspace_.release();
  prefill_workspace_.release();
  graph_storage_.release();
  sampling_storage_.release();
  host_token_storage_.release();
  if (context_.handle) {
    cublasDestroy(context_.handle);
    context_.handle = nullptr;
  }
  if (context_.stream) {
    cudaStreamDestroy(context_.stream);
    context_.stream = nullptr;
  }
  if (prev != model_->cuda_device_id()) cudaSetDevice(prev);
}
template <typename T>
Qwen3Session<T>::~Qwen3Session() {
  release();
}
template <typename T>
void Qwen3Session<T>::synchronize() const {
  // Cleanup owners may retry completion on a poisoned instance. A successful
  // retry releases borrowed-input lifetime obligations but never revives it.
  if (context_.stream) {
    const auto status = cudaStreamSynchronize(context_.stream);
    if (status != cudaSuccess) poisoned_ = true;
    CUDA_CHECK(status);
  }
}
template <typename T>
void Qwen3Session<T>::require_usable() const {
  if (poisoned_)
    throw std::runtime_error("Session is unusable after failed CUDA stream completion");
}
template <typename T>
bool Qwen3Session<T>::drain_failed_execution(KVCache<T>* modified_cache) noexcept {
  // synchronize() already records a failed completion. Never retry it through
  // an outer catch or let a later successful API call revive that instance.
  if (poisoned_) return false;
  if (context_.stream && cudaStreamSynchronize(context_.stream) != cudaSuccess) {
    poisoned_ = true;
    return false;
  }
  if (modified_cache) {
    try {
      modified_cache->clear();
      if (modified_cache == managed_cache_.get()) history_ready_ = false;
    } catch (...) {
      // A custom external cache must not replace the original launch error.
      poisoned_ = true;
      return false;
    }
  }
  return true;
}
template <typename T>
void Qwen3Session<T>::set_graph_enabled(bool enabled) {
  require_usable();
  synchronize();
  use_cuda_graph_ = enabled;
}
template <typename T>
Qwen3Session<T>& Qwen3Session<T>::cuda() {
  return *this;
}
template <typename T>
Qwen3Session<T>& Qwen3Session<T>::cpu() {
  throw std::runtime_error("Prepared CUDA sessions cannot migrate to CPU");
}
template <typename T>
void Qwen3Session<T>::print_model_info() const {
  std::cout << "Prepared decoder: " << get_n_layers() << " layers, " << get_hidden_size()
            << " hidden, " << get_vocab_size() << " vocabulary; workspace "
            << decode_workspace_bytes() << " bytes\n";
}
template <typename T>
size_t Qwen3Session<T>::estimate_prefill_workspace_bytes(size_t rows) const {
  return plan_decoder_workspace<T>(model_->config(), rows).total_bytes();
}

template <typename T>
size_t Qwen3Session<T>::estimate_embedding_prefill_workspace_bytes(size_t rows) const {
  return plan_decoder_workspace<T>(model_->config(), rows, DecoderOutput::Hidden).total_bytes();
}

template <typename T>
void Qwen3Session<T>::prepare_prefill(size_t rows, DecoderOutput output) {
  // A same-length request can change its output contract. Cache that contract
  // in the resolved views, so an embedding plan never serves a token head.
  if (prefill_buffers_.residual.shape[0] == rows &&
      (prefill_buffers_.logits.data != nullptr) == (output == DecoderOutput::Logits))
    return;
  const auto plan = plan_decoder_workspace<T>(model_->config(), rows, output);
  prefill_buffers_ = resolve_decoder_buffers<T>(model_->config(), rows, plan, prefill_workspace_);
}
template <typename T>
std::unique_ptr<Qwen3Session<T>> Qwen3Session<T>::create(std::shared_ptr<const Qwen3Model<T>> m,
                                                         size_t n, bool graph) {
  if (!m || !n || n > m->config().max_position_embeddings)
    throw std::invalid_argument("Session capacity must be positive and within the model limit");
  auto s = std::make_unique<Qwen3Session<T>>(std::move(m), graph);
  s->managed_cache_ = std::make_unique<KVCache<T>>(
      s->get_n_layers(), n, s->get_n_kv_heads() * s->get_head_dim(), Device::CUDA);
  return s;
}
template <typename T>
std::unique_ptr<Qwen3Session<T>> Qwen3Session<T>::new_session(size_t n, bool graph) const {
  require_usable();
  return create(model_, n, graph);
}
template <typename T>
void Qwen3Session<T>::require_managed() const {
  if (!managed_cache_) throw std::logic_error("Context operations require create or new_session");
}
template <typename T>
size_t Qwen3Session<T>::context_size() const {
  require_managed();
  return managed_cache_->size();
}
template <typename T>
size_t Qwen3Session<T>::context_capacity() const {
  require_managed();
  return managed_cache_->get_max_seq_len();
}
template <typename T>
void Qwen3Session<T>::reset() {
  require_usable();
  require_managed();
  synchronize();
  managed_cache_->clear();
  history_ready_ = false;
}
template <typename T>
void Qwen3Session<T>::validate_cache(KVCache<T>* cache, size_t rows, bool decode) {
  if (!cache || !rows || (decode && rows != 1))
    throw std::invalid_argument(
        "Session requires a cache and positive row count; decode takes one row");
  if (managed_cache_ && managed_cache_.get() != cache)
    throw std::invalid_argument("Managed session requires its own cache");
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  if (dev != model_->cuda_device_id()) throw std::invalid_argument("Session CUDA device changed");
  const auto& c = model_->config();
  if (cache->device() != Device::CUDA || cache->get_n_layers() != c.n_layers ||
      cache->get_head_dim() != c.n_kv_heads * c.head_dim || !cache->get_max_seq_len() ||
      cache->get_max_seq_len() > c.max_position_embeddings || cache->size() < rows ||
      cache->size() > cache->get_max_seq_len())
    throw std::invalid_argument("Session cache shape, device or extent is incompatible");
  if (bound_cache_ && (bound_cache_ != cache || bound_capacity_ != cache->get_max_seq_len()))
    throw std::invalid_argument("Session cache binding cannot change");
  for (size_t i = 0; i < c.n_layers; ++i) {
    auto k = cache->k_capacity_view(i), v = cache->v_capacity_view(i);
    if (bound_cache_) {
      if (bound_bases_[i].first != k.data || bound_bases_[i].second != v.data)
        throw std::invalid_argument("Session cache storage changed");
    } else {
      require_cuda_pointer(k.data, dev);
      require_cuda_pointer(v.data, dev);
    }
  }
}
template <typename T>
void Qwen3Session<T>::bind_cache(KVCache<T>* cache) {
  if (!bound_cache_) {
    for (size_t i = 0; i < model_->config().n_layers; ++i)
      bound_bases_.emplace_back(cache->k_capacity_view(i).data, cache->v_capacity_view(i).data);
    bound_cache_ = cache;
    bound_capacity_ = cache->get_max_seq_len();
  }
}
template <typename T>
void Qwen3Session<T>::validate_and_bind(TensorView<const uint32_t, 1> in, KVCache<T>* cache,
                                        bool decode) {
  if (!in.data || !in.shape[0] || in.stride[0] != 1)
    throw std::invalid_argument("Session requires nonempty contiguous CUDA tokens");
  validate_cache(cache, in.shape[0], decode);
  require_cuda_pointer(in.data, model_->cuda_device_id());
  if (host_ids_.size() < in.shape[0]) host_ids_.resize(in.shape[0]);
  CUDA_CHECK(cudaMemcpyAsync(host_ids_.data(), in.data, in.shape[0] * sizeof(uint32_t),
                             cudaMemcpyDeviceToHost, context_.stream));
  synchronize();
  for (size_t i = 0; i < in.shape[0]; ++i)
    if (host_ids_[i] >= model_->config().vocab_size)
      throw std::invalid_argument("Session token id exceeds vocabulary");
  bind_cache(cache);
}
template <typename T>
void Qwen3Session<T>::run(TensorView<const uint32_t, 1> in, KVCache<T>& cache, DecoderBuffers<T>& b,
                          const DecoderStep& step) {
  op::cuda::gather<T>(context_, in, model_->embedding_view(), b.residual);
  auto h = execute_decoder<T>(context_, *model_, cache, b, step, graph_branches_);
  execute_linear<T>(context_, h.as_const(), model_->output_weight(), b.logits);
}
template <typename T>
void Qwen3Session<T>::capture_graph(KVCache<T>& cache) {
  DecoderStep step{host_offset_,   host_offset_,    DecoderMode::Graph, device_offset_,
                   device_offset_, device_lengths_, device_pingpong_};
  op::cuda::bind_execution_context(context_);
  run(graph_input_.as_const(), cache, decode_buffers_, step);
  synchronize();
  CUDA_CHECK(cudaStreamBeginCapture(context_.stream, cudaStreamCaptureModeThreadLocal));
  bool capturing = true;
  try {
    run(graph_input_.as_const(), cache, decode_buffers_, step);
    auto status = cudaStreamEndCapture(context_.stream, &graph_);
    capturing = false;
    CUDA_CHECK(status);
    CUDA_CHECK(cudaGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0));
  } catch (...) {
    if (capturing) cudaStreamEndCapture(context_.stream, &graph_);
    if (graph_exec_) {
      cudaGraphExecDestroy(graph_exec_);
      graph_exec_ = nullptr;
    }
    if (graph_) {
      cudaGraphDestroy(graph_);
      graph_ = nullptr;
    }
    throw;
  }
}
template <typename T>
TensorView<T, 2> Qwen3Session<T>::execute(TensorView<const uint32_t, 1> in, KVCache<T>& cache,
                                          bool prefill, bool graph) {
  require_usable();
  try {
    validate_and_bind(in, &cache, !prefill);
  } catch (...) {
    // Device-ID validation can already have queued a read of borrowed tokens.
    // It does not write the cache, so successful cleanup preserves its history.
    drain_failed_execution();
    throw;
  }
  return execute_prevalidated(in, cache, prefill, graph);
}
template <typename T>
TensorView<T, 2> Qwen3Session<T>::execute_prevalidated(TensorView<const uint32_t, 1> in,
                                                       KVCache<T>& cache, bool prefill,
                                                       bool graph) {
  require_usable();
  const size_t rows = in.shape[0], offset = cache.size() - rows;
  if (prefill) prepare_prefill(rows, DecoderOutput::Logits);
  try {
    if (prefill) {
      op::cuda::bind_execution_context(context_);
      run(in, cache, prefill_buffers_, {offset, offset, DecoderMode::Prefill});
      synchronize();
      return prefill_buffers_.logits;
    }
    if (graph) {
      host_offset_ = offset;
      host_length_ = static_cast<int>(cache.size());
      if (in.data != graph_input_.data)
        CUDA_CHECK(cudaMemcpyAsync(graph_input_.data, in.data, sizeof(uint32_t),
                                   cudaMemcpyDeviceToDevice, context_.stream));
      CUDA_CHECK(cudaMemcpyAsync(device_offset_, &host_offset_, sizeof(size_t),
                                 cudaMemcpyHostToDevice, context_.stream));
      CUDA_CHECK(cudaMemcpyAsync(device_lengths_, &host_length_, sizeof(int),
                                 cudaMemcpyHostToDevice, context_.stream));
      if (!graph_exec_) capture_graph(cache);
      CUDA_CHECK(cudaGraphLaunch(graph_exec_, context_.stream));
    } else {
      op::cuda::bind_execution_context(context_);
      run(in, cache, decode_buffers_, {offset, offset, DecoderMode::Decode});
    }
    synchronize();
    return decode_buffers_.logits;
  } catch (...) {
    // Submitted kernels can partially overwrite KV. Never restore a logical
    // history whose contents are no longer known after an execution failure.
    drain_failed_execution(&cache);
    throw;
  }
}
template <typename T>
TensorView<T, 2> Qwen3Session<T>::prefill(TensorView<const uint32_t, 1> in) {
  require_usable();
  require_managed();
  if (!in.shape[0]) throw std::invalid_argument("Prompt must be nonempty");
  if (in.shape[0] > context_capacity()) throw std::length_error("Prompt exceeds session capacity");
  const size_t previous = context_size();
  try {
    managed_cache_->resize(in.shape[0]);
    auto out = execute(in, *managed_cache_, true, false);
    history_ready_ = true;
    return out;
  } catch (...) {
    if (!poisoned_ && (history_ready_ || !previous)) managed_cache_->resize(previous);
    throw;
  }
}
template <typename T>
TensorView<T, 2> Qwen3Session<T>::prefill(const uint32_t* tokens, size_t rows) {
  return execute_host_tokens(tokens, rows, true);
}
template <typename T>
TensorView<T, 2> Qwen3Session<T>::execute_host_tokens(const uint32_t* tokens, size_t rows,
                                                      bool prefill) {
  require_usable();
  require_managed();
  if (!tokens || !rows || (!prefill && rows != 1))
    throw std::invalid_argument("Host input requires tokens; decode takes one token");
  if (!prefill && !history_ready_) throw std::logic_error("Decode requires successful prefill");
  const size_t previous = context_size();
  if (rows > context_capacity() || (!prefill && previous >= context_capacity()))
    throw std::length_error("Host input exceeds session capacity");
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  if (device != model_->cuda_device_id())
    throw std::invalid_argument("Session CUDA device changed");
  for (size_t i = 0; i < rows; ++i)
    if (tokens[i] >= get_vocab_size()) throw std::invalid_argument("Token exceeds vocabulary");
  bool upload_attempted = false;
  try {
    managed_cache_->resize(prefill ? rows : previous + 1);
    validate_cache(managed_cache_.get(), rows, !prefill);
    bind_cache(managed_cache_.get());
    uint32_t* destination = graph_input_.data;
    if (prefill) {
      host_token_storage_.reserve(rows * sizeof(uint32_t));
      destination = host_token_storage_.template ptr_at<uint32_t>(0);
    }
    upload_attempted = true;
    CUDA_CHECK(cudaMemcpyAsync(destination, tokens, rows * sizeof(uint32_t), cudaMemcpyHostToDevice,
                               context_.stream));
    // Host IDs were checked before upload. External CUDA views retain full
    // device-ID validation; this private path avoids a redundant download.
    auto out = execute_prevalidated({destination, {rows}, {1}}, *managed_cache_, prefill,
                                    !prefill && use_cuda_graph_);
    if (prefill) history_ready_ = true;
    return out;
  } catch (...) {
    // The caller may release host storage immediately, including scalar stack
    // tokens. Drain any attempted upload even if later workspace planning fails.
    const bool completed = !upload_attempted || drain_failed_execution();
    if (completed && !poisoned_ && (history_ready_ || !previous)) {
      managed_cache_->resize(previous);
    }
    throw;
  }
}
template <typename T>
TensorView<T, 2> Qwen3Session<T>::decode(TensorView<const uint32_t, 1> in) {
  require_usable();
  require_managed();
  if (!history_ready_) throw std::logic_error("Decode requires successful prefill");
  if (in.shape[0] != 1) throw std::invalid_argument("Decode takes one token");
  const size_t previous = context_size();
  if (previous >= context_capacity()) throw std::length_error("Session context capacity exhausted");
  try {
    managed_cache_->resize(previous + 1);
    return execute(in, *managed_cache_, false, use_cuda_graph_);
  } catch (...) {
    if (!poisoned_ && history_ready_) managed_cache_->resize(previous);
    throw;
  }
}
template <typename T>
TensorView<T, 2> Qwen3Session<T>::decode(uint32_t token) {
  return execute_host_tokens(&token, 1, false);
}
template <typename T>
TensorView<T, 2> Qwen3Session<T>::execute_embeddings(TensorView<const T, 2> input, size_t position,
                                                     bool prefill) {
  require_usable();
  require_managed();
  const auto& config = model_->config();
  if (!input.data || !input.shape[0] || input.shape[1] != config.hidden_size ||
      !input.is_contiguous() || (!prefill && input.shape[0] != 1))
    throw std::invalid_argument(
        "Session embeddings require contiguous CUDA rows with model hidden width");
  if (position > config.max_position_embeddings ||
      input.shape[0] > config.max_position_embeddings - position)
    throw std::out_of_range("Embedding positions exceed model limit");
  if (!prefill && !history_ready_) throw std::logic_error("Decode requires successful prefill");
  const size_t previous = context_size(), rows = input.shape[0];
  if (rows > context_capacity() || (!prefill && previous >= context_capacity()))
    throw std::length_error("Embedding input exceeds session capacity");
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  if (device != model_->cuda_device_id())
    throw std::invalid_argument("Session CUDA device changed");
  require_cuda_pointer(input.data, device);
  const auto address = reinterpret_cast<uintptr_t>(input.data);
  const auto bytes = rows * config.hidden_size * sizeof(T);
  const auto overlaps_workspace = [&](const CudaWorkspaceArena& arena) {
    if (arena.empty()) return false;
    const auto base = reinterpret_cast<uintptr_t>(arena.base_ptr());
    return address <= base ? base - address < bytes : address - base < arena.capacity_bytes();
  };
  if (overlaps_workspace(prefill_workspace_) || overlaps_workspace(decode_workspace_))
    throw std::invalid_argument("Embedding inputs must not alias session workspace");
  managed_cache_->resize(prefill ? rows : previous + 1);
  try {
    validate_cache(managed_cache_.get(), rows, !prefill);
  } catch (...) {
    managed_cache_->resize(previous);
    throw;
  }
  bind_cache(managed_cache_.get());
  if (prefill) {
    try {
      prepare_prefill(rows, DecoderOutput::Hidden);
    } catch (...) {
      managed_cache_->resize(previous);
      throw;
    }
  }
  auto& buffers = prefill ? prefill_buffers_ : decode_buffers_;
  try {
    op::cuda::bind_execution_context(context_);
    CUDA_CHECK(cudaMemcpyAsync(buffers.residual.data, input.data,
                               rows * config.hidden_size * sizeof(T), cudaMemcpyDeviceToDevice,
                               context_.stream));
    auto hidden = execute_decoder<T>(
        context_, *model_, *managed_cache_, buffers,
        {prefill ? 0 : previous, position, prefill ? DecoderMode::Prefill : DecoderMode::Decode});
    synchronize();
    history_ready_ = true;
    return hidden;
  } catch (...) {
    drain_failed_execution(managed_cache_.get());
    throw;
  }
}
template <typename T>
TensorView<T, 2> Qwen3Session<T>::prefill_embeddings(TensorView<const T, 2> input,
                                                     size_t position) {
  return execute_embeddings(input, position, true);
}
template <typename T>
TensorView<T, 2> Qwen3Session<T>::decode_embeddings(TensorView<const T, 2> input, size_t position) {
  return execute_embeddings(input, position, false);
}
template <typename T>
TensorView<T, 2> Qwen3Session<T>::forward_eager(const Tensor<uint32_t>* in, KVCache<T>* kv) {
  if (!kv) throw std::invalid_argument("Session requires typed cache");
  return execute(tokens_from(in), *kv, false, false);
}
template <typename T>
TensorView<T, 2> Qwen3Session<T>::prefill_eager(const Tensor<uint32_t>* in, KVCache<T>* kv) {
  if (!kv) throw std::invalid_argument("Session requires typed cache");
  return execute(tokens_from(in), *kv, true, false);
}
template <typename T>
TensorView<T, 2> Qwen3Session<T>::forward_for_graph_logits_only(const Tensor<uint32_t>* in,
                                                                KVCache<T>* kv) {
  if (!kv) throw std::invalid_argument("Session requires typed cache");
  return execute(tokens_from(in), *kv, false, true);
}
template <typename T>
uint32_t* Qwen3Session<T>::sample_logits(TensorView<const T, 2> logits, float temp, float p,
                                         size_t k, curandState* states) {
  require_usable();
  try {
    op::cuda::sample<T>(context_, logits, sampled_token_, sampled_probability_, sampling_scratch_,
                        sampling_plan_, temp, p, k, states);
    synchronize();
  } catch (...) {
    drain_failed_execution();
    throw;
  }
  return sampled_token_.data;
}
template <typename T>
uint32_t* Qwen3Session<T>::forward(const Tensor<uint32_t>* in, ThreadPool&, KVCacheBase* base,
                                   size_t k, float temp, float p, curandState* states) {
  op::cuda::validate_sampling_policy(get_vocab_size(), temp, p, k, states != nullptr);
  auto* kv = dynamic_cast<KVCache<T>*>(base);
  auto logits = use_cuda_graph_ ? forward_for_graph_logits_only(in, kv) : forward_eager(in, kv);
  return sample_logits(logits.as_const(), temp, p, k, states);
}
template <typename T>
uint32_t* Qwen3Session<T>::prefill(const Tensor<uint32_t>* in, ThreadPool&, KVCacheBase* base,
                                   size_t k, float temp, float p, curandState* states) {
  op::cuda::validate_sampling_policy(get_vocab_size(), temp, p, k, states != nullptr);
  auto logits = prefill_eager(in, dynamic_cast<KVCache<T>*>(base));
  return sample_logits(logits.subview({logits.shape[0] - 1, 0}, {1, logits.shape[1]}).as_const(),
                       temp, p, k, states);
}
template class Qwen3Session<__nv_bfloat16>;
template class Qwen3Session<float>;
