#include "inference.hpp"

#include <chrono>
#include <limits>
#include <vector>

#include "base_model.hpp"
#include "common.hpp"
#include "operators/cuda/random.hpp"
#include "operators/cuda/execution.hpp"

namespace {

uint32_t read_token_from_device(uint32_t* token_ptr, Device device) {
  if (!token_ptr) throw std::runtime_error("Model returned a null token pointer");
  if (device == Device::CUDA) {
    uint32_t token = 0;
    checkCudaErrors(cudaMemcpyAsync(&token, token_ptr, sizeof(uint32_t),
                                    cudaMemcpyDeviceToHost,
                                    cudaStreamDefault));
    checkCudaErrors(cudaStreamSynchronize(cudaStreamDefault));
    return token;
  }

  return *token_ptr;
}

void validate_token_id(const BaseModel* model, uint32_t token,
                       const char* phase) {
  if (!model) {
    throw std::runtime_error("Model is null while validating token id");
  }

  const size_t vocab_size = model->get_vocab_size();
  if (token >= vocab_size) {
    throw std::runtime_error(std::string("Invalid token id from ") + phase +
                             ": " + std::to_string(token) +
                             " >= vocab_size " +
                             std::to_string(vocab_size));
  }
}

std::shared_ptr<BaseModel> require_model(std::shared_ptr<BaseModel> model, Device device) {
  if (!model) throw std::invalid_argument("Inference engine requires a model");
  auto executor = model->fork_executor();
  if (executor) model = std::move(executor);
  // Resolve unsupported devices before allocating a cache for that backend.
  if (model->device() != device) {
    if (device == Device::CUDA) model->cuda();
    else model->cpu();
  }
  return model;
}

size_t engine_capacity(const BaseModel& model, size_t capacity) {
  const size_t limit = model.get_max_seq_len();
  if (!limit || capacity > limit) throw std::invalid_argument("Engine capacity exceeds the model context limit");
  return capacity ? capacity : limit;
}

void validate_request_sampling(const BaseModel& model, Device device,
                               float temperature, float top_p, size_t top_k) {
  if (device == Device::CUDA) {
    op::cuda::validate_sampling_policy(model.get_vocab_size(), temperature, top_p, top_k, true);
  } else if (!std::isfinite(temperature) || !std::isfinite(top_p) ||
             top_p <= 0.0f || top_p > 1.0f || !top_k) {
    throw std::invalid_argument("Invalid generation sampling policy");
  }
}

class EngineDeviceScope {
 public:
  explicit EngineDeviceScope(int device) : device_(device) {
    if (device_ < 0) return;
    checkCudaErrors(cudaGetDevice(&previous_));
    if (previous_ != device_) checkCudaErrors(cudaSetDevice(device_));
  }
  ~EngineDeviceScope() {
    if (device_ >= 0 && previous_ != device_) cudaSetDevice(previous_);
  }
 private:
  int device_;
  int previous_ = -1;
};
}  // namespace

template <typename T>
KVCache<T>::KVCache(size_t n_layers, size_t max_seq_len, size_t head_dim, Device device, size_t initial_size)
    : n_layers_(n_layers), storage_capacity_(max_seq_len),
      head_dim_(head_dim), current_len_(0), device_(device) {

    if (!n_layers_ || !storage_capacity_ || !head_dim_ ||
        n_layers_ > std::numeric_limits<size_t>::max() / storage_capacity_ ||
        n_layers_ * storage_capacity_ > std::numeric_limits<size_t>::max() / head_dim_ / sizeof(T)) {
        throw std::invalid_argument("KVCache dimensions are empty or exceed addressable storage");
    }
    if (initial_size > storage_capacity_) {
        throw std::invalid_argument("Initial size cannot exceed max_seq_len");
    }
    const std::vector<size_t> shape{n_layers_, storage_capacity_, head_dim_};
    if (device_ == Device::CUDA) {
        const size_t bytes = n_layers_ * storage_capacity_ * head_dim_ * sizeof(T);
        k_storage_.reserve(bytes);
        v_storage_.reserve(bytes);
        k_cache_contiguous_ = Tensor<T>::from_external_buffer(k_storage_.template ptr_at<T>(0), shape, device_);
        v_cache_contiguous_ = Tensor<T>::from_external_buffer(v_storage_.template ptr_at<T>(0), shape, device_);
    } else {
        k_cache_contiguous_ = Tensor<T>(shape, device_);
        v_cache_contiguous_ = Tensor<T>(shape, device_);
    }

    current_len_ = initial_size;
}

template <typename T>
void KVCache<T>::resize(size_t new_size) {
    if (new_size > storage_capacity_) {
        throw std::runtime_error("KVCache: Attempted to resize beyond max_seq_len");
    }
    current_len_ = new_size;
}

template <typename T>
void KVCache<T>::clear() {
    current_len_ = 0;
}
template <typename T>
Tensor<T>& KVCache<T>::k_cache(size_t layer, size_t pos) {
    if (layer >= n_layers_) {
        throw std::runtime_error("KVCache: Layer index out of range");
    }
    if (pos >= storage_capacity_) {
        throw std::runtime_error("KVCache: Position index out of range");
    }
    const size_t idx = layer * storage_capacity_ + pos;
    auto found = k_cache_slices_.find(idx);
    if (found == k_cache_slices_.end()) {
        found = k_cache_slices_.emplace(idx,
            k_cache_contiguous_.slice({layer, pos, 0}, {layer + 1, pos + 1, head_dim_})).first;
    }
    return found->second;
}

template <typename T>
Tensor<T>& KVCache<T>::v_cache(size_t layer, size_t pos) {
    if (layer >= n_layers_) {
        throw std::runtime_error("KVCache: Layer index out of range");
    }
    if (pos >= storage_capacity_) {
        throw std::runtime_error("KVCache: Position index out of range");
    }
    const size_t idx = layer * storage_capacity_ + pos;
    auto found = v_cache_slices_.find(idx);
    if (found == v_cache_slices_.end()) {
        found = v_cache_slices_.emplace(idx,
            v_cache_contiguous_.slice({layer, pos, 0}, {layer + 1, pos + 1, head_dim_})).first;
    }
    return found->second;
}

template <typename T>
void KVCache<T>::refresh_legacy_slices() {
    for (auto& entry : k_cache_slices_) {
        const size_t layer = entry.first / storage_capacity_;
        const size_t pos = entry.first % storage_capacity_;
        entry.second = k_cache_contiguous_.slice({layer, pos, 0}, {layer + 1, pos + 1, head_dim_});
    }
    for (auto& entry : v_cache_slices_) {
        const size_t layer = entry.first / storage_capacity_;
        const size_t pos = entry.first % storage_capacity_;
        entry.second = v_cache_contiguous_.slice({layer, pos, 0}, {layer + 1, pos + 1, head_dim_});
    }
}

template <typename T>
KVCache<T>& KVCache<T>::cuda() {
    if (device_ == Device::CUDA)
        return *this;

    const size_t k_bytes = k_cache_contiguous_.nbytes();
    const size_t v_bytes = v_cache_contiguous_.nbytes();
    k_storage_.reserve(k_bytes);
    v_storage_.reserve(v_bytes);
    auto next_k = Tensor<T>::from_external_buffer(k_storage_.template ptr_at<T>(0),
                                                 k_cache_contiguous_.sizes(), Device::CUDA);
    auto next_v = Tensor<T>::from_external_buffer(v_storage_.template ptr_at<T>(0),
                                                 v_cache_contiguous_.sizes(), Device::CUDA);
    checkCudaErrors(cudaMemcpy(next_k.data_ptr(), k_cache_contiguous_.data_ptr(),
                               k_bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(next_v.data_ptr(), v_cache_contiguous_.data_ptr(),
                               v_bytes, cudaMemcpyHostToDevice));
    k_cache_contiguous_ = std::move(next_k);
    v_cache_contiguous_ = std::move(next_v);
    device_ = Device::CUDA;

    refresh_legacy_slices();
    return *this;
}

template <typename T>
KVCache<T>& KVCache<T>::cpu() {
    if (device_ == Device::CPU)
        return *this;

    // Prepare both copies before changing the live cache's device/backing state.
    Tensor<T> next_k(k_cache_contiguous_.sizes(), Device::CPU);
    Tensor<T> next_v(v_cache_contiguous_.sizes(), Device::CPU);
    checkCudaErrors(cudaMemcpy(next_k.data_ptr(), k_cache_contiguous_.data_ptr(),
                               next_k.numel() * sizeof(T), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(next_v.data_ptr(), v_cache_contiguous_.data_ptr(),
                               next_v.numel() * sizeof(T), cudaMemcpyDeviceToHost));
    k_cache_contiguous_ = std::move(next_k);
    v_cache_contiguous_ = std::move(next_v);
    device_ = Device::CPU;
    refresh_legacy_slices();
    k_storage_.release();
    v_storage_.release();
    return *this;
}

template <typename T>
std::pair<const Tensor<T>, const Tensor<T>> KVCache<T>::get_contiguous_tensor(size_t layer) const {
    if (layer >= n_layers_) {
        throw std::runtime_error("KVCache: Layer index out of range in get_contiguous_tensor");
    }
    Tensor<T> K = k_cache_contiguous_.slice({layer, 0, 0}, {layer + 1, current_len_, head_dim_}).squeeze(0);
    Tensor<T> V = v_cache_contiguous_.slice({layer, 0, 0}, {layer + 1, current_len_, head_dim_}).squeeze(0);

    return {K, V};
}

template <typename T>
std::pair<Tensor<T>, Tensor<T>> KVCache<T>::get_layer_view(size_t layer) {
    if (layer >= n_layers_) {
        throw std::runtime_error("KVCache: Layer index out of range in get_layer_view");
    }
    // Return a view (slice) of the specified layer.
    // The view is writable and points to the continuous memory block of that layer.
    Tensor<T> k_view = k_cache_contiguous_.slice({layer, 0, 0}, {layer + 1, storage_capacity_, head_dim_}).squeeze(0);
    Tensor<T> v_view = v_cache_contiguous_.slice({layer, 0, 0}, {layer + 1, storage_capacity_, head_dim_}).squeeze(0);
    return {k_view, v_view};
}

template class KVCache<float>;
template class KVCache<__nv_bfloat16>;

template <typename T>
InferenceEngine<T>::InferenceEngine(std::shared_ptr<BaseModel> model, Device device, size_t capacity)
    : thread_pool_(device == Device::CUDA ? 0 : 4), model_(require_model(std::move(model), device)),
      kv_cache_(model_->get_n_layers(), engine_capacity(*model_, capacity),
                model_->get_head_dim() * model_->get_n_kv_heads(), device),
      device_(device), benchmark_mode_(false), benchmark_warmup_tokens_(64) {
  if (device_ == Device::CUDA) {
    init_cuda_resources();
  } else {
    decode_input_ = Tensor<uint32_t>({1}, Device::CPU);
  }
}

template <typename T>
void InferenceEngine<T>::init_cuda_resources() {
  checkCudaErrors(cudaGetDevice(&cuda_device_id_));
  WorkspacePlanner planner;
  planner.add_request("rng", sizeof(curandState), 0, 1);
  planner.add_request("decode_input", sizeof(uint32_t), 0, 1);
  const auto plan = planner.build();
  cuda_resources_.reserve_for_plan(plan);
  d_states = cuda_resources_.template ptr_at<curandState>(plan.at("rng").offset);
  decode_input_ = Tensor<uint32_t>::from_external_buffer(
      cuda_resources_.template ptr_at<uint32_t>(plan.at("decode_input").offset), {1}, Device::CUDA);
  prompt_storage_.reserve(kv_cache_.get_max_seq_len() * sizeof(uint32_t));
  const auto seed = static_cast<unsigned long long>(
      std::chrono::system_clock::now().time_since_epoch().count());
  op::cuda::init_curand(d_states, seed, 0);
  checkCudaErrors(cudaStreamSynchronize(nullptr));
}

template <typename T>
void InferenceEngine<T>::release_cuda_resources() noexcept {
  if (cuda_device_id_ < 0) return;
  int previous = cuda_device_id_;
  cudaGetDevice(&previous);
  if (previous != cuda_device_id_) cudaSetDevice(cuda_device_id_);
  cudaDeviceSynchronize();
  // The borrowed decode descriptor remains unused until migration replaces it
  // or member destruction drops its no-op ownership wrapper.
  d_states = nullptr;
  prompt_storage_.release();
  cuda_resources_.release();
  if (previous != cuda_device_id_) cudaSetDevice(previous);
  cuda_device_id_ = -1;
}

template <typename T>
InferenceEngine<T>::~InferenceEngine() {
  release_cuda_resources();
  // Session graphs borrow this engine's cache, which is still alive here.
  model_.reset();
}

template <typename T>
uint32_t* InferenceEngine<T>::generate_next_token(ThreadPool& thread_pool, uint32_t* input_ids,
                                                float temperature, float top_p, size_t top_k) {
  require_valid();
  if (!input_ids) throw std::invalid_argument("Decode requires a token pointer");
  validate_request_sampling(*model_, device_, temperature, top_p, top_k);
  if (kv_cache_.size() >= kv_cache_.get_max_seq_len())
    throw std::length_error("Engine context capacity exhausted");
  EngineDeviceScope device_scope(device_ == Device::CUDA ? cuda_device_id_ : -1);
  try {
    if (device_ == Device::CUDA) {
      checkCudaErrors(cudaMemcpy(decode_input_.data_ptr(), input_ids, sizeof(uint32_t), cudaMemcpyDeviceToDevice));
      checkCudaErrors(cudaStreamSynchronize(nullptr));
    } else {
      decode_input_.data_ptr()[0] = *input_ids;
    }
    kv_cache_.resize(kv_cache_.size() + 1);
    return model_->forward(&decode_input_, thread_pool, &kv_cache_, top_k, temperature, top_p, d_states);
  } catch (...) {
    const auto original_error = std::current_exception();
    try { model_->synchronize(); } catch (...) { valid_ = false; }
    // Execution may already have overwritten or explicitly discarded history.
    // Never resurrect it by restoring the previous logical length.
    if (valid_) kv_cache_.clear();
    std::rethrow_exception(original_error);
  }
}

template <typename T>
void InferenceEngine<T>::warmup(size_t warmup_tokens, bool force_warmup,
                                float temperature, float top_p, size_t top_k) {
  require_valid();
  if (device_ != Device::CUDA || (has_warmed_up_ && !force_warmup) || kv_cache_.size()) return;
  validate_request_sampling(*model_, device_, temperature, top_p, top_k);
  EngineDeviceScope device_scope(cuda_device_id_);
  const size_t count = std::min(warmup_tokens, kv_cache_.get_max_seq_len());
  if (!count) return;
  std::vector<uint32_t> tokens(count, 0);
  auto input = Tensor<uint32_t>::from_external_buffer(
      prompt_storage_.template ptr_at<uint32_t>(0), {count}, Device::CUDA);
  checkCudaErrors(cudaMemcpy(input.data_ptr(), tokens.data(), count * sizeof(uint32_t), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaStreamSynchronize(nullptr));
  try {
    kv_cache_.resize(count);
    auto* token = model_->prefill(&input, thread_pool_, &kv_cache_, top_k, temperature, top_p, d_states);
    if (!token) throw std::runtime_error("Warmup prefill returned a null token");
    if (count < kv_cache_.get_max_seq_len())
      generate_next_token(thread_pool_, token, temperature, top_p, top_k);
    try { model_->synchronize(); } catch (...) { valid_ = false; throw; }
    kv_cache_.clear();
    has_warmed_up_ = true;
  } catch (...) {
    const auto original_error = std::current_exception();
    try { model_->synchronize(); } catch (...) { valid_ = false; }
    if (valid_) kv_cache_.clear();
    std::rethrow_exception(original_error);
  }
}

template <typename T>
void InferenceEngine<T>::set_benchmark_mode(bool enabled, size_t warmup_tokens) {
  require_valid();
  benchmark_mode_ = enabled;
  benchmark_warmup_tokens_ = warmup_tokens;
}

template <typename T>
void InferenceEngine<T>::generate_with_callback(const std::vector<uint32_t>& input_ids, size_t max_length,
                                               float temperature, float top_p, size_t top_k,
                                               std::function<void(uint32_t)> callback) {
  require_valid();
  if (input_ids.empty()) throw std::invalid_argument("Prompt must be nonempty");
  if (!callback) throw std::invalid_argument("Generation requires a callback");
  if (input_ids.size() > kv_cache_.get_max_seq_len())
    throw std::length_error("Prompt exceeds engine context capacity");
  const size_t limit = std::min(max_length, kv_cache_.get_max_seq_len());
  if (limit <= input_ids.size()) return;
  for (auto token : input_ids) validate_token_id(model_.get(), token, "prompt");
  validate_request_sampling(*model_, device_, temperature, top_p, top_k);
  reset();
  if (device_ == Device::CUDA)
    warmup(benchmark_mode_ ? benchmark_warmup_tokens_ : 64, benchmark_mode_, temperature, top_p, top_k);

  try {
    uint32_t* token_ptr = nullptr;
    Tensor<uint32_t> prompt;
    {
      // Select the executor's device only for native work. A callback may use
      // another device; its caller-thread CUDA selection remains its own.
      EngineDeviceScope device_scope(device_ == Device::CUDA ? cuda_device_id_ : -1);
      if (device_ == Device::CUDA) {
        prompt = Tensor<uint32_t>::from_external_buffer(
            prompt_storage_.template ptr_at<uint32_t>(0), {input_ids.size()}, Device::CUDA);
        checkCudaErrors(cudaMemcpy(prompt.data_ptr(), input_ids.data(),
                                    input_ids.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaStreamSynchronize(nullptr));
      } else {
        prompt = Tensor<uint32_t>(std::vector<uint32_t>(input_ids), {input_ids.size()}, Device::CPU);
      }
      kv_cache_.resize(input_ids.size());
      token_ptr = model_->prefill(&prompt, thread_pool_, &kv_cache_, top_k, temperature, top_p, d_states);
    }
    std::unique_ptr<uint32_t> cpu_token(device_ == Device::CPU ? token_ptr : nullptr);
    size_t total_length = input_ids.size();
    while (true) {
      uint32_t token;
      {
        EngineDeviceScope device_scope(device_ == Device::CUDA ? cuda_device_id_ : -1);
        token = read_token_from_device(token_ptr, device_);
        validate_token_id(model_.get(), token, "generation");
      }
      ++total_length;
      if (token == model_->get_eos_token_id()) break;
      callback(token);
      if (total_length >= limit || kv_cache_.size() >= kv_cache_.get_max_seq_len()) break;
      token_ptr = generate_next_token(thread_pool_, token_ptr, temperature, top_p, top_k);
      if (device_ == Device::CPU) cpu_token.reset(token_ptr);
    }
  } catch (...) {
    const auto original_error = std::current_exception();
    // Preserve the operation/callback failure even if completion also fails.
    // A completion failure keeps backing storage in place and invalidates the
    // engine, so no public operation can reuse potentially unfinished state.
    try {
      EngineDeviceScope device_scope(device_ == Device::CUDA ? cuda_device_id_ : -1);
      model_->synchronize();
    } catch (...) { valid_ = false; }
    if (valid_) kv_cache_.clear();
    std::rethrow_exception(original_error);
  }
}

template <typename T>
void InferenceEngine<T>::reset() {
  require_valid();
  EngineDeviceScope device_scope(device_ == Device::CUDA ? cuda_device_id_ : -1);
  try { model_->synchronize(); } catch (...) { valid_ = false; throw; }
  kv_cache_.clear();
}

template <typename T>
InferenceEngine<T>& InferenceEngine<T>::cuda() {
  require_valid();
  if (device_ == Device::CUDA) return *this;
  try {
    model_->cuda();
    kv_cache_.cuda();
    init_cuda_resources();
    device_ = Device::CUDA;
    has_warmed_up_ = false;
  } catch (...) {
    // Model migration may already have changed its state before throwing.
    valid_ = false;
    throw;
  }
  return *this;
}

template <typename T>
InferenceEngine<T>& InferenceEngine<T>::cpu() {
  require_valid();
  if (device_ == Device::CPU) return *this;
  try {
    EngineDeviceScope device_scope(cuda_device_id_);
    checkCudaErrors(cudaDeviceSynchronize());
    model_->cpu();
    kv_cache_.cpu();
    release_cuda_resources();
    decode_input_ = Tensor<uint32_t>({1}, Device::CPU);
    device_ = Device::CPU;
    has_warmed_up_ = false;
  } catch (...) {
    valid_ = false;
    throw;
  }
  return *this;
}

template class InferenceEngine<float>;
template class InferenceEngine<__nv_bfloat16>;
