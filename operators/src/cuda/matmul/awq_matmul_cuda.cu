#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <stdexcept>

#include "operators/cuda/legacy/legacy_bridge.cuh"
#include "operators/cuda/matmul/awq_matmul_cuda.cuh"

namespace op {

template <typename T>
void AwqMatmulCUDAOperator<T>::operator()(Tensor<T>* output, Tensor<T>* input, const WeightTensor<T>& weight,
                                          const Tensor<T>* bias, cudaStream_t stream) {

    if (!weight.is_quantized()) {
        throw std::runtime_error("AWQ MatMul operator requires quantized weights");
    }

    legacy::matmul_quantized_gemv(*input, *weight.qweight(), *weight.scales(), *weight.qzeros(), weight.group_size(),
                                   output, stream, bias);
}

template class AwqMatmulCUDAOperator<float>;
template class AwqMatmulCUDAOperator<__nv_bfloat16>;

}  // namespace op
