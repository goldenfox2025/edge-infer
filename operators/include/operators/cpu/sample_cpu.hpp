#pragma once

#include <cmath>
#include <random>
#include <vector>
#include <algorithm>
#include <numeric>
#include <immintrin.h>  // For AVX2/FMA intrinsics
#include <chrono>
#include <cstdint>

#include "operators/operator_base.hpp"

namespace op {

template <typename T>
class SampleCPUOperator : public SampleOperator<T> {
private:
  // 预分配buffer，避免每次采样时的内存分配
  static thread_local std::vector<float> temp_float_buffer_;
  static thread_local std::vector<float> temp_probs_buffer_;
  
public:
  SampleCPUOperator() = default;
  ~SampleCPUOperator() override = default;

  // 实现CPU版本的Sample
  uint32_t* operator()(Tensor<T>&& logits, float temperature, float top_p, size_t top_k,
                       curandState* d_states = nullptr, cudaStream_t stream = nullptr) override {
    // 对于bf16，转换为float进行处理
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      return sample_cpu_bf16_impl(std::move(logits), temperature, top_p, top_k);
    } else {
      return sample_cpu_impl(std::move(logits), temperature, top_p, top_k);
    }
  }

  // 获取算子平台
  OperatorPlatform platform() const override { return OperatorPlatform::CPU; }

 private:
  // bf16专用的sample实现，转换为float处理，使用SIMD优化和预分配buffer
  uint32_t* sample_cpu_bf16_impl(Tensor<T>&& logits, float temperature, float top_p, size_t top_k) {
    size_t vocab_size = logits.numel();
    T* logits_data = logits.data_ptr();

    // 使用预分配buffer，避免内存分配开销
    if (temp_float_buffer_.size() < vocab_size) {
      temp_float_buffer_.resize(vocab_size);
    }
    if (temp_probs_buffer_.size() < vocab_size) {
      temp_probs_buffer_.resize(vocab_size);
    }
    
    float* float_logits = temp_float_buffer_.data();
    float* probs = temp_probs_buffer_.data();

    // 将bf16转换为float进行处理，使用SIMD优化
    convert_bf16_to_float_simd(logits_data, float_logits, vocab_size);

    // 1. 应用temperature scaling（SIMD优化）
    if (temperature != 1.0f && temperature > 0.0f) {
      apply_temperature_simd(float_logits, vocab_size, temperature);
    }

    // 2. 找到最大值用于数值稳定性（SIMD优化）
    float max_logit = find_max_simd(float_logits, vocab_size);

    // 3. 计算softmax概率（SIMD优化）
    float sum_exp = compute_softmax_simd(float_logits, probs, vocab_size, max_logit);

    // 4. 执行采样（优化版本）
    uint32_t selected_token = perform_sampling_optimized(probs, top_k, top_p, vocab_size);

    // 5. 分配内存并返回结果指针
    uint32_t* result = new uint32_t(selected_token);
    return result;
  }

  uint32_t* sample_cpu_impl(Tensor<T>&& logits, float temperature, float top_p, size_t top_k) {
    size_t vocab_size = logits.numel();
    T* logits_data = logits.data_ptr();

    // 使用预分配buffer
    if (temp_float_buffer_.size() < vocab_size) {
      temp_float_buffer_.resize(vocab_size);
    }
    if (temp_probs_buffer_.size() < vocab_size) {
      temp_probs_buffer_.resize(vocab_size);
    }
    
    float* float_logits = temp_float_buffer_.data();
    float* probs = temp_probs_buffer_.data();

    // 将数据转换为float进行高效处理
    for (size_t i = 0; i < vocab_size; ++i) {
      float_logits[i] = static_cast<float>(logits_data[i]);
    }

    // 1. 应用temperature scaling（SIMD优化）
    if (temperature != 1.0f && temperature > 0.0f) {
      apply_temperature_simd(float_logits, vocab_size, temperature);
    }

    // 2. 找到最大值用于数值稳定性（SIMD优化）
    float max_logit = find_max_simd(float_logits, vocab_size);

    // 3. 计算softmax概率（SIMD优化）
    float sum_exp = compute_softmax_simd(float_logits, probs, vocab_size, max_logit);

    // 4. 执行采样（优化版本）
    uint32_t selected_token = perform_sampling_optimized(probs, top_k, top_p, vocab_size);

    // 5. 分配内存并返回结果指针
    uint32_t* result = new uint32_t(selected_token);
    return result;
  }

  // SIMD优化的高性能采样逻辑
  uint32_t perform_sampling_optimized(const float* probs, size_t top_k, float top_p, size_t vocab_size) {
    if (top_k == 1) {
      // Greedy sampling - 使用SIMD找到最大值索引
      return find_argmax_simd(probs, vocab_size);
    } else {
      // 优化的Top-k/Top-p sampling
      return sample_top_k_p_optimized(probs, vocab_size, top_k, top_p);
    }
  }

private:
  // SIMD优化的工具函数
  
  // bf16转float的SIMD优化
  void convert_bf16_to_float_simd(const T* bf16_data, float* float_data, size_t size) {
    if constexpr (!std::is_same_v<T, __nv_bfloat16>) {
      // 非bf16类型，直接转换
      for (size_t i = 0; i < size; ++i) {
        float_data[i] = static_cast<float>(bf16_data[i]);
      }
      return;
    }
    
    // bf16到float的转换
    for (size_t i = 0; i < size; ++i) {
      float_data[i] = static_cast<float>(bf16_data[i]);
    }
  }
  
  // 应用temperature的SIMD优化
  void apply_temperature_simd(float* data, size_t size, float temperature) {
    const float inv_temp = 1.0f / temperature;
    const __m256 inv_temp_vec = _mm256_set1_ps(inv_temp);
    
    size_t simd_size = (size / 8) * 8;
    
    // SIMD处理8个元素为一组
    for (size_t i = 0; i < simd_size; i += 8) {
      __m256 data_vec = _mm256_loadu_ps(&data[i]);
      data_vec = _mm256_mul_ps(data_vec, inv_temp_vec);
      _mm256_storeu_ps(&data[i], data_vec);
    }
    
    // 处理剩余元素
    for (size_t i = simd_size; i < size; ++i) {
      data[i] *= inv_temp;
    }
  }
  
  // 查找最大值的SIMD优化
  float find_max_simd(const float* data, size_t size) {
    if (size == 0) return 0.0f;
    
    __m256 max_vec = _mm256_set1_ps(-INFINITY);
    size_t simd_size = (size / 8) * 8;
    
    // SIMD处理
    for (size_t i = 0; i < simd_size; i += 8) {
      __m256 data_vec = _mm256_loadu_ps(&data[i]);
      max_vec = _mm256_max_ps(max_vec, data_vec);
    }
    
    // 提取最大值
    alignas(32) float max_array[8];
    _mm256_store_ps(max_array, max_vec);
    
    float max_val = max_array[0];
    for (int i = 1; i < 8; ++i) {
      max_val = std::max(max_val, max_array[i]);
    }
    
    // 处理剩余元素
    for (size_t i = simd_size; i < size; ++i) {
      max_val = std::max(max_val, data[i]);
    }
    
    return max_val;
  }
  
  // 优化的softmax计算 - 减少内存访问
  float compute_softmax_simd(const float* logits, float* probs, size_t size, float max_val) {
    // 使用Kahan求和算法提高数值稳定性并减少误差累积
    float sum_exp = 0.0f;
    float sum_compensation = 0.0f;
    
    // 第一遍：计算exp并累加
    for (size_t i = 0; i < size; ++i) {
      float exp_val = expf(logits[i] - max_val);
      probs[i] = exp_val;
      
      // Kahan求和算法
      float y = exp_val - sum_compensation;
      float t = sum_exp + y;
      sum_compensation = (t - sum_exp) - y;
      sum_exp = t;
    }
    
    // 第二遍：归一化（使用逆乘法避免除法）
    const float inv_sum = 1.0f / sum_exp;
    for (size_t i = 0; i < size; ++i) {
      probs[i] *= inv_sum;
    }
    
    return sum_exp;
  }
  
  // 改用标准exp函数，确保精度
  void compute_exp_simd(const float* input, float* output, size_t size, float max_val) {
    for (size_t i = 0; i < size; ++i) {
      output[i] = expf(input[i] - max_val);
    }
  }
  
  // 简化的argmax查找，确保正确性
  uint32_t find_argmax_simd(const float* data, size_t size) {
    if (size == 0) return 0;
    
    uint32_t max_idx = 0;
    float max_val = data[0];
    
    for (size_t i = 1; i < size; ++i) {
      if (data[i] > max_val) {
        max_val = data[i];
        max_idx = i;
      }
    }
    
    return max_idx;
  }
  
  // 超高速的top-k/top-p采样，基于llama.cpp优化
  uint32_t sample_top_k_p_optimized(const float* probs, size_t vocab_size, size_t top_k, float top_p) {
    // 如果是greedy sampling (top_k=1)，直接返回最大值索引
    if (top_k == 1) {
      return find_argmax_simd(probs, vocab_size);
    }
    
    // 预分配候选vector，避免动态扩容
    static thread_local std::vector<std::pair<float, uint32_t>> candidates;
    candidates.clear();
    candidates.reserve(std::min(vocab_size, top_k > 0 ? top_k : vocab_size));
    
    // 快速预过滤：只考虑概率大于某个阈值的token
    const float min_prob_threshold = 1e-7f;
    
    for (size_t i = 0; i < vocab_size; ++i) {
      if (probs[i] > min_prob_threshold) {
        candidates.emplace_back(probs[i], static_cast<uint32_t>(i));
      }
    }
    
    if (candidates.empty()) {
      return 0;  // fallback
    }
    
    // 优化1: 如果候选数量很少，直接排序
    if (candidates.size() <= 32) {
      std::sort(candidates.begin(), candidates.end(),
               [](const auto& a, const auto& b) { return a.first > b.first; });
    } else {
      // 优化2: 使用nth_element进行部分排序，只排序需要的部分
      size_t k_limit = (top_k > 0 && top_k < candidates.size()) ? top_k : candidates.size();
      
      // 使用introspective selection，比完整排序快很多
      std::nth_element(candidates.begin(), candidates.begin() + k_limit - 1, candidates.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
      
      // 只排序top-k部分
      std::sort(candidates.begin(), candidates.begin() + k_limit,
               [](const auto& a, const auto& b) { return a.first > b.first; });
      
      candidates.resize(k_limit);
    }
    
    // 优化3: Top-p过滤 - early termination
    if (top_p < 1.0f) {
      float cumsum = 0.0f;
      size_t cutoff = 0;
      
      for (size_t i = 0; i < candidates.size(); ++i) {
        cumsum += candidates[i].first;
        cutoff = i + 1;
        if (cumsum >= top_p) {
          break;  // early termination
        }
      }
      
      candidates.resize(cutoff);
    }
    
    if (candidates.empty()) {
      return 0;
    }
    
    // 优化4: 快速累积概率计算 - 避免重新归一化
    float total_prob = 0.0f;
    for (const auto& candidate : candidates) {
      total_prob += candidate.first;
    }
    
    // 优化5: 使用快速随机数生成器
    static thread_local uint64_t rng_state = std::chrono::steady_clock::now().time_since_epoch().count();
    
    // XORShift64* - 比std::mt19937快很多
    auto xorshift64star = [](uint64_t& state) -> uint64_t {
      state ^= state >> 12;
      state ^= state << 25;
      state ^= state >> 27;
      return state * 0x2545F4914F6CDD1DULL;
    };
    
    float rand_val = (float)(xorshift64star(rng_state) >> 11) / (float)(1ULL << 53) * total_prob;
    
    // 优化6: 线性搜索而不是二分搜索（对于小数组更快）
    float cumulative = 0.0f;
    for (const auto& candidate : candidates) {
      cumulative += candidate.first;
      if (rand_val <= cumulative) {
        return candidate.second;
      }
    }
    
    // 保险起见，返回第一个候选
    return candidates[0].second;
  }
};

// 静态成员定义
template<typename T>
thread_local std::vector<float> SampleCPUOperator<T>::temp_float_buffer_;

template<typename T>
thread_local std::vector<float> SampleCPUOperator<T>::temp_probs_buffer_;

}  // namespace op