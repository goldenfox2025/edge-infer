#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <stdexcept>

#include "operators/cuda/legacy/legacy_bridge.cuh"
#include "operators/cuda/matmul/awq_matmul_cuda.cuh"

namespace op {

// 实现AwqMatmulCUDAOperator的operator()方法
template <typename T>
void AwqMatmulCUDAOperator<T>::operator()(Tensor<T>* output, Tensor<T>* input, const WeightTensor<T>& weight,
                                          const Tensor<T>* bias, cudaStream_t stream) {
    // 确保权重是量化的
    if (!weight.is_quantized()) {
        throw std::runtime_error("AWQ MatMul operator requires quantized weights");
    }

    // 调用AWQ量化矩阵乘法实现
    // 从weight对象获取所需的量化参数
    legacy::matmul_quantized_gemv(*input, *weight.qweight(), *weight.scales(), *weight.qzeros(), weight.group_size(),
                                   output, stream, bias);
}

// 显式模板实例化
template class AwqMatmulCUDAOperator<float>;
template class AwqMatmulCUDAOperator<__nv_bfloat16>;

}  // namespace op