#pragma once
#include <cuda_bf16.h>
#include <curand_kernel.h>  // Device-side random number generation

#include <chrono>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "kvcache_base.hpp"
#include "tensor.hpp"
#include "thread_pool.hpp"

// Forward declaration of KVCache
template <typename T>
class KVCache;
constexpr int kNumStreams = 5;
using ModelConfig = std::unordered_map<std::string, double>;

// Base model class that will be used for both LlamaModel and QwenModel
class BaseModel {
 public:
  BaseModel() = default;
  virtual ~BaseModel() = default;

  // Core inference methods that must be implemented by derived classes
  virtual uint32_t* forward(const Tensor<uint32_t>* input,
                            ThreadPool& thread_pool, KVCacheBase* kv_cache,
                            size_t top_k, float temperature, float top_p,
                            curandState* d_states = nullptr) = 0;
  virtual uint32_t* prefill(const Tensor<uint32_t>* input,
                            ThreadPool& thread_pool, KVCacheBase* kv_cache,
                            size_t top_k, float temperature, float top_p,
                            curandState* d_states = nullptr) = 0;

  // Common methods for all model types
  virtual bool verify_params() const = 0;
  virtual void print_model_info() const = 0;

  // Device management
  virtual BaseModel& cuda() = 0;
  virtual BaseModel& cpu() = 0;
  virtual Device device() const = 0;

  // Getters for model properties
  virtual size_t get_n_layers() const = 0;
  virtual size_t get_max_seq_len() const = 0;
  virtual size_t get_head_dim() const = 0;
  virtual size_t get_n_kv_heads() const = 0;
  virtual size_t get_vocab_size() const = 0;
  virtual uint32_t get_eos_token_id() const = 0;

  // Return the hidden size used by speculative decoding.
  virtual size_t get_hidden_size() const = 0;

  // Estimate peak prefill workspace to reserve arena capacity before execution.
  virtual size_t estimate_prefill_workspace_bytes(size_t seq_len) const {
    return 0;
  }

  // Compatibility engines need the global arena only for legacy executors.
  virtual bool owns_execution_workspaces() const { return false; }

  // Print the model device.
  void print_device_info() const {
    std::cout << "Current model device: " << (device() == Device::CUDA ? "CUDA" : "CPU")
              << std::endl;
  }
};
