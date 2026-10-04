#pragma once

#include <cmath>
#include <random>
#include <vector>
#include <algorithm>
#include <numeric>
#if defined(__AVX__)
#include <immintrin.h>
#endif
#include <chrono>
#include <cstdint>

#include "operators/operator_base.hpp"

namespace op {

template <typename T>
class SampleCPUOperator : public SampleOperator<T> {
private:
  // Reuse thread-local scratch buffers; growing them may allocate.
  static thread_local std::vector<float> temp_float_buffer_;
  static thread_local std::vector<float> temp_probs_buffer_;

public:
  SampleCPUOperator() = default;
  ~SampleCPUOperator() override = default;

  uint32_t* operator()(Tensor<T>&& logits, float temperature, float top_p, size_t top_k,
                       curandState* d_states = nullptr, cudaStream_t stream = nullptr) override {

    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      return sample_cpu_bf16_impl(std::move(logits), temperature, top_p, top_k);
    } else {
      return sample_cpu_impl(std::move(logits), temperature, top_p, top_k);
    }
  }

  OperatorPlatform platform() const override { return OperatorPlatform::CPU; }

 private:
  // Convert BF16 logits to float and reuse thread-local scratch buffers.
  uint32_t* sample_cpu_bf16_impl(Tensor<T>&& logits, float temperature, float top_p, size_t top_k) {
    size_t vocab_size = logits.numel();
    T* logits_data = logits.data_ptr();

    if (temp_float_buffer_.size() < vocab_size) {
      temp_float_buffer_.resize(vocab_size);
    }
    if (temp_probs_buffer_.size() < vocab_size) {
      temp_probs_buffer_.resize(vocab_size);
    }

    float* float_logits = temp_float_buffer_.data();
    float* probs = temp_probs_buffer_.data();

    // Convert BF16 values to float with scalar casts.
    convert_bf16_to_float_simd(logits_data, float_logits, vocab_size);

    if (temperature != 1.0f && temperature > 0.0f) {
      apply_temperature_simd(float_logits, vocab_size, temperature);
    }

    float max_logit = find_max_simd(float_logits, vocab_size);

    float sum_exp = compute_softmax_simd(float_logits, probs, vocab_size, max_logit);

    uint32_t selected_token = perform_sampling_optimized(probs, top_k, top_p, vocab_size);

    // The caller owns the returned token pointer and must delete it.
    uint32_t* result = new uint32_t(selected_token);
    return result;
  }

  uint32_t* sample_cpu_impl(Tensor<T>&& logits, float temperature, float top_p, size_t top_k) {
    size_t vocab_size = logits.numel();
    T* logits_data = logits.data_ptr();

    if (temp_float_buffer_.size() < vocab_size) {
      temp_float_buffer_.resize(vocab_size);
    }
    if (temp_probs_buffer_.size() < vocab_size) {
      temp_probs_buffer_.resize(vocab_size);
    }

    float* float_logits = temp_float_buffer_.data();
    float* probs = temp_probs_buffer_.data();

    for (size_t i = 0; i < vocab_size; ++i) {
      float_logits[i] = static_cast<float>(logits_data[i]);
    }

    if (temperature != 1.0f && temperature > 0.0f) {
      apply_temperature_simd(float_logits, vocab_size, temperature);
    }

    float max_logit = find_max_simd(float_logits, vocab_size);

    float sum_exp = compute_softmax_simd(float_logits, probs, vocab_size, max_logit);

    uint32_t selected_token = perform_sampling_optimized(probs, top_k, top_p, vocab_size);

    // The caller owns the returned token pointer and must delete it.
    uint32_t* result = new uint32_t(selected_token);
    return result;
  }

  uint32_t perform_sampling_optimized(const float* probs, size_t top_k, float top_p, size_t vocab_size) {
    if (top_k == 1) {

      return find_argmax_simd(probs, vocab_size);
    } else {

      return sample_top_k_p_optimized(probs, vocab_size, top_k, top_p);
    }
  }

private:

  // BF16 conversion currently uses scalar casts.
  void convert_bf16_to_float_simd(const T* bf16_data, float* float_data, size_t size) {
    if constexpr (!std::is_same_v<T, __nv_bfloat16>) {

      for (size_t i = 0; i < size; ++i) {
        float_data[i] = static_cast<float>(bf16_data[i]);
      }
      return;
    }

    for (size_t i = 0; i < size; ++i) {
      float_data[i] = static_cast<float>(bf16_data[i]);
    }
  }

  void apply_temperature_simd(float* data, size_t size, float temperature) {
    const float inv_temp = 1.0f / temperature;
#if defined(__AVX__)
    const __m256 inv_temp_vec = _mm256_set1_ps(inv_temp);

    size_t simd_size = (size / 8) * 8;

    for (size_t i = 0; i < simd_size; i += 8) {
      __m256 data_vec = _mm256_loadu_ps(&data[i]);
      data_vec = _mm256_mul_ps(data_vec, inv_temp_vec);
      _mm256_storeu_ps(&data[i], data_vec);
    }

    for (size_t i = simd_size; i < size; ++i) {
      data[i] *= inv_temp;
    }
#else
    for (size_t i = 0; i < size; ++i) {
      data[i] *= inv_temp;
    }
#endif
  }

  float find_max_simd(const float* data, size_t size) {
    if (size == 0) return 0.0f;

#if defined(__AVX__)
    __m256 max_vec = _mm256_set1_ps(-INFINITY);
    size_t simd_size = (size / 8) * 8;

    for (size_t i = 0; i < simd_size; i += 8) {
      __m256 data_vec = _mm256_loadu_ps(&data[i]);
      max_vec = _mm256_max_ps(max_vec, data_vec);
    }

    alignas(32) float max_array[8];
    _mm256_store_ps(max_array, max_vec);

    float max_val = max_array[0];
    for (int i = 1; i < 8; ++i) {
      max_val = std::max(max_val, max_array[i]);
    }

    for (size_t i = simd_size; i < size; ++i) {
      max_val = std::max(max_val, data[i]);
    }
#else
    float max_val = -INFINITY;
    for (size_t i = 0; i < size; ++i) {
      max_val = std::max(max_val, data[i]);
    }
#endif

    return max_val;
  }

  float compute_softmax_simd(const float* logits, float* probs, size_t size, float max_val) {
    // Use Kahan summation for the softmax denominator.
    float sum_exp = 0.0f;
    float sum_compensation = 0.0f;

    for (size_t i = 0; i < size; ++i) {
      float exp_val = expf(logits[i] - max_val);
      probs[i] = exp_val;

      // Compensated summation.
      float y = exp_val - sum_compensation;
      float t = sum_exp + y;
      sum_compensation = (t - sum_exp) - y;
      sum_exp = t;
    }

    // Normalize the stored exponentials using the reciprocal sum.
    const float inv_sum = 1.0f / sum_exp;
    for (size_t i = 0; i < size; ++i) {
      probs[i] *= inv_sum;
    }

    return sum_exp;
  }

  void compute_exp_simd(const float* input, float* output, size_t size, float max_val) {
    for (size_t i = 0; i < size; ++i) {
      output[i] = expf(input[i] - max_val);
    }
  }

  // Scalar argmax; ties select the first token.
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

  // Filter candidates by probability, then apply top-k/top-p sampling.
  uint32_t sample_top_k_p_optimized(const float* probs, size_t vocab_size, size_t top_k, float top_p) {

    if (top_k == 1) {
      return find_argmax_simd(probs, vocab_size);
    }

    // Reuse the thread-local candidate vector; reserve may still allocate.
    static thread_local std::vector<std::pair<float, uint32_t>> candidates;
    candidates.clear();
    candidates.reserve(std::min(vocab_size, top_k > 0 ? top_k : vocab_size));

    // Discard candidates below the fixed probability threshold.
    const float min_prob_threshold = 1e-7f;

    for (size_t i = 0; i < vocab_size; ++i) {
      if (probs[i] > min_prob_threshold) {
        candidates.emplace_back(probs[i], static_cast<uint32_t>(i));
      }
    }

    if (candidates.empty()) {
      return 0;  // fallback
    }

    if (candidates.size() <= 32) {
      std::sort(candidates.begin(), candidates.end(),
               [](const auto& a, const auto& b) { return a.first > b.first; });
    } else {

      size_t k_limit = (top_k > 0 && top_k < candidates.size()) ? top_k : candidates.size();

      std::nth_element(candidates.begin(), candidates.begin() + k_limit - 1, candidates.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });

      // Sort the selected top-k prefix.
      std::sort(candidates.begin(), candidates.begin() + k_limit,
               [](const auto& a, const auto& b) { return a.first > b.first; });

      candidates.resize(k_limit);
    }

    // Retain the shortest sorted prefix reaching top_p.
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

    // Draw from the retained probability mass without renormalizing the array.
    float total_prob = 0.0f;
    for (const auto& candidate : candidates) {
      total_prob += candidate.first;
    }

    static thread_local uint64_t rng_state = std::chrono::steady_clock::now().time_since_epoch().count();

    // Thread-local XORShift64* generator.
    auto xorshift64star = [](uint64_t& state) -> uint64_t {
      state ^= state >> 12;
      state ^= state << 25;
      state ^= state >> 27;
      return state * 0x2545F4914F6CDD1DULL;
    };

    float rand_val = (float)(xorshift64star(rng_state) >> 11) / (float)(1ULL << 53) * total_prob;

    // Select a token by scanning cumulative probability.
    float cumulative = 0.0f;
    for (const auto& candidate : candidates) {
      cumulative += candidate.first;
      if (rand_val <= cumulative) {
        return candidate.second;
      }
    }

    // Rounding fallback: return the first retained candidate.
    return candidates[0].second;
  }
};

// Per-thread scratch buffer definitions.
template<typename T>
thread_local std::vector<float> SampleCPUOperator<T>::temp_float_buffer_;

template<typename T>
thread_local std::vector<float> SampleCPUOperator<T>::temp_probs_buffer_;

}  // namespace op
