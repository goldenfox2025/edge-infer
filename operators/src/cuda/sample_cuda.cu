#include "operators/cuda/sampling_runtime.cuh"
#include "operators/cuda/sample_cuda.cuh"

namespace op {

template <typename T>
uint32_t* SampleCUDAOperator<T>::operator()(Tensor<T>&& logits,
                                            float temperature, float top_p,
                                            size_t top_k,
                                            curandState* d_states,
                                            cudaStream_t stream) {
  return cuda::sample(std::move(logits), temperature, top_p, top_k, d_states,
                      stream);
}

template class SampleCUDAOperator<float>;
template class SampleCUDAOperator<__nv_bfloat16>;

}  // namespace op
