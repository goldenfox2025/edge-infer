#include "operators/cuda/sampling_runtime.cuh"

#include <cuda_bf16.h>

#include "cuda/legacy/legacy_cuda_api.cuh"

namespace op::cuda {

void init_curand(curandState* d_states, unsigned long long seed, int offset,
                 cudaStream_t stream) {
  cuda_OP::init_curand(d_states, seed, offset, stream);
}

void generate_random_values(float* values, size_t count, curandState* states,
                            cudaStream_t stream) {
  cuda_OP::generate_random_values(values, count, states, stream);
}

template <typename T>
uint32_t* sample(Tensor<T>&& logits, float temperature, float top_p,
                 size_t top_k, curandState* d_states, cudaStream_t stream) {
  return cuda_OP::sample(std::move(logits), temperature, top_p, top_k, d_states,
                         stream);
}

template <typename T>
void sample_to_fixed(Tensor<T>&& logits, uint32_t* output_ptr,
                     float temperature, float top_p, size_t top_k,
                     curandState* d_states, cudaStream_t stream) {
  cuda_OP::sample_to_fixed(std::move(logits), output_ptr, temperature, top_p,
                           top_k, d_states, stream);
}

template <typename T>
void sample_batch_to_fixed(Tensor<T>&& logits, uint32_t* output_ptr,
                           float temperature, float top_p, size_t top_k,
                           curandState* d_states, cudaStream_t stream) {
  cuda_OP::sample_batch_to_fixed(std::move(logits), output_ptr, temperature,
                                 top_p, top_k, d_states, stream);
}

template <typename T>
void sample_to_fixed_with_prob(Tensor<T>&& logits, uint32_t* token_ptr,
                               float* prob_ptr, float temperature, float top_p,
                               size_t top_k, curandState* d_states,
                               cudaStream_t stream) {
  cuda_OP::sample_to_fixed_with_prob(std::move(logits), token_ptr, prob_ptr,
                                     temperature, top_p, top_k, d_states,
                                     stream);
}

template <typename T>
float get_token_probability(const Tensor<T>& logits, int position,
                            uint32_t token_id, cudaStream_t stream) {
  return cuda_OP::get_token_probability(logits, position, token_id, stream);
}

template uint32_t* sample<float>(Tensor<float>&&, float, float, size_t,
                                 curandState*, cudaStream_t);
template uint32_t* sample<__nv_bfloat16>(Tensor<__nv_bfloat16>&&, float, float,
                                         size_t, curandState*, cudaStream_t);

template void sample_to_fixed<float>(Tensor<float>&&, uint32_t*, float, float,
                                     size_t, curandState*, cudaStream_t);
template void sample_to_fixed<__nv_bfloat16>(Tensor<__nv_bfloat16>&&, uint32_t*,
                                             float, float, size_t,
                                             curandState*, cudaStream_t);

template void sample_batch_to_fixed<float>(Tensor<float>&&, uint32_t*, float,
                                           float, size_t, curandState*,
                                           cudaStream_t);
template void sample_batch_to_fixed<__nv_bfloat16>(
    Tensor<__nv_bfloat16>&&, uint32_t*, float, float, size_t, curandState*,
    cudaStream_t);

template void sample_to_fixed_with_prob<float>(Tensor<float>&&, uint32_t*,
                                               float*, float, float, size_t,
                                               curandState*, cudaStream_t);
template void sample_to_fixed_with_prob<__nv_bfloat16>(
    Tensor<__nv_bfloat16>&&, uint32_t*, float*, float, float, size_t,
    curandState*, cudaStream_t);

template float get_token_probability<float>(const Tensor<float>&, int, uint32_t,
                                            cudaStream_t);
template float get_token_probability<__nv_bfloat16>(
    const Tensor<__nv_bfloat16>&, int, uint32_t, cudaStream_t);

}  // namespace op::cuda
