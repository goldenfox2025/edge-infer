#pragma once
#include <cuda_bf16.h>
#include <curand_kernel.h>  // 用于设备端随机数生成

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

// 前向声明 BaseModel
class BaseModel;

template <typename T>
class InferenceEngine;

template <typename T>
class KVCache : public KVCacheBase {
 public:
  // n_layers：模型层数，max_seq_len：最大序列长度，head_dim：每个缓存张量的元素数（通常为
  // n_kv_heads * dqkv） initial_size：初始缓存 token 数（可选） device: 指定
  // KVCache 所在的设备 (CPU or CUDA)
  KVCache(size_t n_layers, size_t max_seq_len, size_t head_dim,
          Device device = Device::CPU, size_t initial_size = 0);

  // 调整缓存长度，必须大于当前长度且不超过 max_seq_len
  void resize(size_t new_size) override;
  // 清空缓存（置当前长度为 0）
  void clear() override;
  // 当前缓存 token 数
  size_t size() const override { return current_len_; }

  // 访问第 layer 层、位置 pos 的 K 缓存（返回引用）
  Tensor<T>& k_cache(size_t layer, size_t pos);
  // 访问第 layer 层、位置 pos 的 V 缓存（返回引用）
  Tensor<T>& v_cache(size_t layer, size_t pos);
  size_t max_seq_len_;

  // 移动 KVCache 到 CUDA 设备
  KVCache<T>& cuda();
  // 移动 KVCache 到 CPU 设备 (可选，如果需要显式移回 CPU)
  KVCache<T>& cpu();
  Device device() const override { return device_; }

  // 实现 KVCacheBase 中的获取属性方法
  size_t get_n_layers() const override { return n_layers_; }
  size_t get_head_dim() const override { return head_dim_; }
  size_t get_max_seq_len() const override { return max_seq_len_; }

  // 新增方法：返回连续存储的 k 和 v 的起始指针
  std::pair<const Tensor<T>, const Tensor<T>> get_contiguous_tensor(
      size_t layer) const;

  // Gets a writable view of a single layer's K/V cache.
  std::pair<Tensor<T>, Tensor<T>> get_layer_view(size_t layer);

 private:
  // 使用连续内存存储所有层的 KV，形状为 [n_layers, max_seq_len, head_dim]
  Tensor<T> k_cache_contiguous_;
  Tensor<T> v_cache_contiguous_;

  // 存储所有 slice 的一维 vector，大小为 n_layers * max_seq_len，
  // 每个 slice 都是从对应连续内存中切出的一部分。
  std::vector<Tensor<T>> k_cache_slices_;
  std::vector<Tensor<T>> v_cache_slices_;

  size_t n_layers_;
  size_t head_dim_;
  size_t current_len_;
  Device device_;  // 当前设备
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
  // 静态变量，用于跟踪是否已经执行过预热
  static bool has_warmed_up_;

  // 构造时传入共享的 BaseModel 实例
  // device: 指定 InferenceEngine 运行的设备 (CPU or CUDA)
  InferenceEngine(std::shared_ptr<BaseModel> model,
                  Device device = Device::CUDA);

  // 析构函数：释放CUDA资源
  virtual ~InferenceEngine();

  // 生成单个 token
  uint32_t* generate_next_token(ThreadPool& thread_pool, uint32_t* input_ids,
                               float temperature = 1.0f, float top_p = 0.9f,
                               size_t top_k = 50);
  // 批量生成 token，直到达到 max_length 或遇到 eos
  void generate_with_callback(const std::vector<uint32_t>& input_ids,
                              size_t max_length, float temperature, float top_p,
                              size_t top_k,
                              std::function<void(uint32_t)> callback);
  // 重置推理状态（清空 KV 缓存）
  void reset();

  // CUDA 预热函数
  // warmup_tokens: 预热使用的token数量
  // force_warmup: 是否强制预热（即使已经预热过）
  void warmup(size_t warmup_tokens = 64, bool force_warmup = false, float temperature = 1.0f, float top_p = 0.9f, size_t top_k = 50);

  // 设置基准测试模式
  // enable_benchmark: 是否启用基准测试模式（每次调用前都预热）
  // benchmark_warmup_tokens: 基准测试模式下的预热token数量
  void set_benchmark_mode(bool enable_benchmark, size_t benchmark_warmup_tokens = 64);

  // 移动 InferenceEngine (及其模型和 KV Cache) 到 CUDA 设备
  InferenceEngine& cuda();
  // 移动 InferenceEngine (及其模型和 KV Cache) 到 CPU 设备 (可选)
  InferenceEngine& cpu();
  Device device() const { return device_; }

 private:
  ThreadPool thread_pool_;
  std::shared_ptr<BaseModel> model_;
  KVCache<T> kv_cache_;
  Device device_;
  curandState* d_states;
  std::unique_ptr<op::UnifiedOperators<T>> operators_;
  
  // 基准测试模式相关变量
  bool benchmark_mode_;
  size_t benchmark_warmup_tokens_;
};
