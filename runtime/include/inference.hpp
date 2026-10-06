#pragma once
#include <cuda_bf16.h>
#include <curand_kernel.h>  // Device-side random number generation

#include <chrono>
#include <cmath>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "kvcache_base.hpp"
#include "execution/cuda_workspace_arena.hpp"
#include "tensor.hpp"
#include "tensor_view.hpp"
#include "thread_pool.hpp"

// BaseModel forward declaration
class BaseModel;

template <typename T>
class InferenceEngine;

template <typename T>
class KVCache : public KVCacheBase {
 public:
  // n_layers: layer count; max_seq_len: capacity; head_dim: elements per cache entry (usually
  // n_kv_heads * dqkv). initial_size: initial cached token count; device:
  // KV-cache device (CPU or CUDA)
  KVCache(size_t n_layers, size_t max_seq_len, size_t head_dim,
          Device device = Device::CPU, size_t initial_size = 0);

  // Grow the cache length without exceeding max_seq_len.
  void resize(size_t new_size) override;
  // Clear the cache by setting its current length to zero.
  void clear() override;
  // Current cached token count
  size_t size() const override { return current_len_; }

  // Return the K-cache view for layer and position pos.
  Tensor<T>& k_cache(size_t layer, size_t pos);
  // Return the V-cache view for layer and position pos.
  Tensor<T>& v_cache(size_t layer, size_t pos);

  // Move the KV cache to CUDA.
  KVCache<T>& cuda();
  // Move the KV cache back to the CPU.
  KVCache<T>& cpu();
  Device device() const override { return device_; }

  // KVCacheBase property accessors
  size_t get_n_layers() const override { return n_layers_; }
  size_t get_head_dim() const override { return head_dim_; }
  size_t get_max_seq_len() const override { return storage_capacity_; }

  // Return base pointers for contiguous K/V storage.
  std::pair<const Tensor<T>, const Tensor<T>> get_contiguous_tensor(
      size_t layer) const;

  // Gets a writable view of a single layer's K/V cache.
  std::pair<Tensor<T>, Tensor<T>> get_layer_view(size_t layer);

  // Allocation-free borrowed views. Callers validate layer/position indices
  // before the execution path; these accessors never retain storage ownership.
  TensorView<T, 2> k_view(size_t layer) noexcept {
    return {k_cache_contiguous_.data_ptr() + layer * storage_capacity_ * head_dim_,
            {current_len_, head_dim_}, {head_dim_, 1}};
  }
  TensorView<const T, 2> k_view(size_t layer) const noexcept {
    return {k_cache_contiguous_.data_ptr() + layer * storage_capacity_ * head_dim_,
            {current_len_, head_dim_}, {head_dim_, 1}};
  }
  TensorView<T, 2> v_view(size_t layer) noexcept {
    return {v_cache_contiguous_.data_ptr() + layer * storage_capacity_ * head_dim_,
            {current_len_, head_dim_}, {head_dim_, 1}};
  }
  TensorView<const T, 2> v_view(size_t layer) const noexcept {
    return {v_cache_contiguous_.data_ptr() + layer * storage_capacity_ * head_dim_,
            {current_len_, head_dim_}, {head_dim_, 1}};
  }
  TensorView<T, 2> k_capacity_view(size_t layer) noexcept {
    auto result = k_view(layer);
    result.shape[0] = storage_capacity_;
    return result;
  }
  TensorView<const T, 2> k_capacity_view(size_t layer) const noexcept {
    auto result = k_view(layer);
    result.shape[0] = storage_capacity_;
    return result;
  }
  TensorView<T, 2> v_capacity_view(size_t layer) noexcept {
    auto result = v_view(layer);
    result.shape[0] = storage_capacity_;
    return result;
  }
  TensorView<const T, 2> v_capacity_view(size_t layer) const noexcept {
    auto result = v_view(layer);
    result.shape[0] = storage_capacity_;
    return result;
  }
  TensorView<T, 1> k_token(size_t layer, size_t pos) noexcept {
    return k_capacity_view(layer).template select<0>(pos);
  }
  TensorView<const T, 1> k_token(size_t layer, size_t pos) const noexcept {
    return k_capacity_view(layer).template select<0>(pos);
  }
  TensorView<T, 1> v_token(size_t layer, size_t pos) noexcept {
    return v_capacity_view(layer).template select<0>(pos);
  }
  TensorView<const T, 1> v_token(size_t layer, size_t pos) const noexcept {
    return v_capacity_view(layer).template select<0>(pos);
  }

 private:
  // Persistent CUDA storage must never come from a global prefill arena.
  CudaWorkspaceArena k_storage_;
  CudaWorkspaceArena v_storage_;
  // Store all layers contiguously with shape [n_layers, max_seq_len, head_dim].
  Tensor<T> k_cache_contiguous_;
  Tensor<T> v_cache_contiguous_;

  // Stable legacy references are created lazily, outside the direct-view path.
  std::unordered_map<size_t, Tensor<T>> k_cache_slices_;
  std::unordered_map<size_t, Tensor<T>> v_cache_slices_;
  void refresh_legacy_slices();

  size_t n_layers_;
  const size_t storage_capacity_;
  size_t head_dim_;
  size_t current_len_;
  Device device_;  // Current device
};

class infer_base {
 public:
  virtual ~infer_base() = default;
  virtual void generate_with_callback(
      const std::vector<uint32_t>& input_ids, size_t max_length,
      float temperature, float top_p, size_t top_k,
      std::function<void(uint32_t)> callback) = 0;

  virtual Device device() const = 0;
  virtual void reset() { throw std::logic_error("This executor does not support reset"); }
  virtual size_t context_size() const {
    throw std::logic_error("This executor does not expose context size");
  }
  virtual size_t context_capacity() const {
    throw std::logic_error("This executor does not expose context capacity");
  }
};
template <typename T>
class InferenceEngine : public infer_base {
 public:
  // Warmup belongs to this engine's prepared session and cache.
  bool has_warmed_up_ = false;

  // Construct with a shared BaseModel instance.
  // device: inference device (CPU or CUDA)
  // capacity: fixed context storage; zero selects the model's context limit.
  InferenceEngine(std::shared_ptr<BaseModel> model,
                  Device device = Device::CUDA, size_t capacity = 0);

  // Release CUDA resources.
  virtual ~InferenceEngine();

  // Generate one token.
  uint32_t* generate_next_token(ThreadPool& thread_pool, uint32_t* input_ids,
                               float temperature = 1.0f, float top_p = 0.9f,
                               size_t top_k = 50);
  // Generate from a complete fresh prompt, until max_length, capacity, or EOS.
  // Callbacks run on the calling thread between completed token operations.
  // A callback exception stops the request before the next decode is submitted.
  void generate_with_callback(const std::vector<uint32_t>& input_ids,
                              size_t max_length, float temperature, float top_p,
                              size_t top_k,
                              std::function<void(uint32_t)> callback) override;
  // Reset inference state and clear the KV cache.
  void reset() override;
  size_t context_size() const override {
    require_valid();
    return kv_cache_.size();
  }
  size_t context_capacity() const override {
    require_valid();
    return kv_cache_.get_max_seq_len();
  }

  // CUDA warmup
  // warmup_tokens: token count for warmup
  // force_warmup: rerun warmup even if it has already completed
  void warmup(size_t warmup_tokens = 64, bool force_warmup = false, float temperature = 1.0f, float top_p = 0.9f, size_t top_k = 50);

  // Configure benchmark mode.
  // enable_benchmark: warm up before each benchmark call
  // benchmark_warmup_tokens: token count used for benchmark warmup
  void set_benchmark_mode(bool enable_benchmark, size_t benchmark_warmup_tokens = 64);

  // A failed migration or completion invalidates the engine; construct a new
  // engine to retry. Invalid engines retain their backing storage until teardown.
  // Move the engine, model, and KV cache to CUDA.
  InferenceEngine& cuda();
  // Move the engine, model, and KV cache to the CPU.
  InferenceEngine& cpu();
  Device device() const override {
    require_valid();
    return device_;
  }

 private:
  void require_valid() const {
    if (!valid_)
      throw std::logic_error("Executor completion or device migration failed; construct a new inference engine");
  }
  bool valid_ = true;
  ThreadPool thread_pool_;
  std::shared_ptr<BaseModel> model_;
  KVCache<T> kv_cache_;
  Device device_;
  int cuda_device_id_ = -1;
  CudaWorkspaceArena cuda_resources_;
  CudaWorkspaceArena prompt_storage_;
  Tensor<uint32_t> decode_input_;
  curandState* d_states = nullptr;
  void init_cuda_resources();
  void release_cuda_resources() noexcept;

  // Benchmark settings
  bool benchmark_mode_;
  size_t benchmark_warmup_tokens_;
};
