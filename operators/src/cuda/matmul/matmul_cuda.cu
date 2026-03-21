#include <cuda_runtime.h>

#include <stdexcept>

#include "operators/cuda/matmul/matmul_cuda.cuh"
#include "operators/cuda/matmul/matmul_selector.hpp"

namespace op {

// 实现MatmulCUDAOperator的operator()方法
template <typename T>
void MatmulCUDAOperator<T>::operator()(Tensor<T>* output, Tensor<T>* input, const WeightTensor<T>& weight,
                                       const Tensor<T>* bias, cudaStream_t stream) {
    // 使用MatmulSelector选择合适的实现
    auto& selector = MatmulSelector<T>::instance();

    // 根据权重特性选择合适的实现
    auto impl = selector.selectImpl(weight, OperatorPlatform::CUDA);

    if (!impl) {
        throw std::runtime_error("No suitable MatMul implementation found");
    }

    // 调用选择的实现
    (*impl)(output, input, weight, bias, stream);
}

// 显式模板实例化
template class MatmulCUDAOperator<float>;
template class MatmulCUDAOperator<__nv_bfloat16>;

}  // namespace op