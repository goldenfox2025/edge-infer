#pragma once
#include <cuda_bf16.h>
#include <curand_kernel.h>  // 用于设备端随机数生成

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

  // 获取隐藏层大小（用于投机解码）
  virtual size_t get_hidden_size() const = 0;

  // 估算prefill阶段临时workspace峰值，用于提前准备arena容量。
  virtual size_t estimate_prefill_workspace_bytes(size_t seq_len) const {
    return 0;
  }

  // 打印模型设备信息
  void print_device_info() const {
    std::cout << "模型当前设备: " << (device() == Device::CUDA ? "CUDA" : "CPU")
              << std::endl;
  }
};
