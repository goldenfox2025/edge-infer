#pragma once
#include <cuda_bf16.h>
#include <curand_kernel.h>  // Device-side random number generation

#include <chrono>
#include <cmath>
#include <condition_variable>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "kvcache_base.hpp"
#include "tensor.hpp"
#include "thread_pool.hpp"

namespace op {
template <typename T>
class UnifiedOperators;
}
template <typename T>
class ThreadSafeQueue {
 public:
  void push(T value) {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push(std::move(value));
    cv_.notify_one();
  }
  T pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return !queue_.empty(); });
    T value = std::move(queue_.front());
    queue_.pop();
    return value;
  }

 private:
  std::mutex mutex_;

  std::queue<T> queue_;
  std::condition_variable cv_;
};

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
  size_t max_seq_len_;

  // Move the KV cache to CUDA.
  KVCache<T>& cuda();
  // Move the KV cache back to the CPU.
  KVCache<T>& cpu();
  Device device() const override { return device_; }

  // KVCacheBase property accessors
  size_t get_n_layers() const override { return n_layers_; }
  size_t get_head_dim() const override { return head_dim_; }
  size_t get_max_seq_len() const override { return max_seq_len_; }

  // Return base pointers for contiguous K/V storage.
  std::pair<const Tensor<T>, const Tensor<T>> get_contiguous_tensor(
      size_t layer) const;

  // Gets a writable view of a single layer's K/V cache.
  std::pair<Tensor<T>, Tensor<T>> get_layer_view(size_t layer);

 private:
  // Store all layers contiguously with shape [n_layers, max_seq_len, head_dim].
  Tensor<T> k_cache_contiguous_;
  Tensor<T> v_cache_contiguous_;

  // Store n_layers * max_seq_len slice views in a flat vector;
  // each slice references the corresponding contiguous allocation.
  std::vector<Tensor<T>> k_cache_slices_;
  std::vector<Tensor<T>> v_cache_slices_;

  size_t n_layers_;
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
};
template <typename T>
class InferenceEngine : public infer_base {
 public:
  // Track whether warmup has already run.
  static bool has_warmed_up_;

  // Construct with a shared BaseModel instance.
  // device: inference device (CPU or CUDA)
  InferenceEngine(std::shared_ptr<BaseModel> model,
                  Device device = Device::CUDA);

  // Release CUDA resources.
  virtual ~InferenceEngine();

  // Generate one token.
  uint32_t* generate_next_token(ThreadPool& thread_pool, uint32_t* input_ids,
                               float temperature = 1.0f, float top_p = 0.9f,
                               size_t top_k = 50);
  // Generate until max_length or EOS.
  void generate_with_callback(const std::vector<uint32_t>& input_ids,
                              size_t max_length, float temperature, float top_p,
                              size_t top_k,
                              std::function<void(uint32_t)> callback);
  // Reset inference state and clear the KV cache.
  void reset();

  // CUDA warmup
  // warmup_tokens: token count for warmup
  // force_warmup: rerun warmup even if it has already completed
  void warmup(size_t warmup_tokens = 64, bool force_warmup = false, float temperature = 1.0f, float top_p = 0.9f, size_t top_k = 50);

  // Configure benchmark mode.
  // enable_benchmark: warm up before each benchmark call
  // benchmark_warmup_tokens: token count used for benchmark warmup
  void set_benchmark_mode(bool enable_benchmark, size_t benchmark_warmup_tokens = 64);

  // Move the engine, model, and KV cache to CUDA.
  InferenceEngine& cuda();
  // Move the engine, model, and KV cache to the CPU.
  InferenceEngine& cpu();
  Device device() const { return device_; }

 private:
  ThreadPool thread_pool_;
  std::shared_ptr<BaseModel> model_;
  KVCache<T> kv_cache_;
  Device device_;
  curandState* d_states;
  std::unique_ptr<op::UnifiedOperators<T>> operators_;

  // Benchmark settings
  bool benchmark_mode_;
  size_t benchmark_warmup_tokens_;
};
