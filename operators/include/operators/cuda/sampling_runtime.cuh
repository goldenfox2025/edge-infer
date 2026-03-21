#pragma once

#include <curand_kernel.h>

#include <cstddef>
#include <cstdint>
#include <utility>

#include "tensor.hpp"

namespace op::cuda {

void init_curand(curandState* d_states, unsigned long long seed, int offset,
                 cudaStream_t stream = nullptr);

void generate_random_values(float* values, size_t count, curandState* states,
                            cudaStream_t stream = nullptr);

template <typename T>
uint32_t* sample(Tensor<T>&& logits, float temperature, float top_p,
                 size_t top_k, curandState* d_states,
                 cudaStream_t stream = nullptr);

template <typename T>
void sample_to_fixed(Tensor<T>&& logits, uint32_t* output_ptr,
                     float temperature, float top_p, size_t top_k,
                     curandState* d_states, cudaStream_t stream = nullptr);

template <typename T>
void sample_batch_to_fixed(Tensor<T>&& logits, uint32_t* output_ptr,
                           float temperature, float top_p, size_t top_k,
                           curandState* d_states,
                           cudaStream_t stream = nullptr);

template <typename T>
void sample_to_fixed_with_prob(Tensor<T>&& logits, uint32_t* token_ptr,
                               float* prob_ptr, float temperature, float top_p,
                               size_t top_k, curandState* d_states,
                               cudaStream_t stream = nullptr);

template <typename T>
float get_token_probability(const Tensor<T>& logits, int position,
                            uint32_t token_id,
                            cudaStream_t stream = nullptr);

}  // namespace op::cuda
