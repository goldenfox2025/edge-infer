#include <cuda_runtime.h>

#include <stdexcept>

#include "operators/cuda/matmul/matmul_cuda.cuh"
#include "operators/cuda/matmul/matmul_selector.hpp"

namespace op {

template <typename T>
MatmulCUDAOperator<T>::MatmulCUDAOperator(cublasHandle_t borrowed_handle)
    : borrowed_handle_(borrowed_handle) {
    if (!borrowed_handle_) {
        throw std::invalid_argument("Borrowed cuBLAS handle must not be null");
    }
}

template <typename T>
void MatmulCUDAOperator<T>::operator()(Tensor<T>* output, Tensor<T>* input, const WeightTensor<T>& weight,
                                       const Tensor<T>* bias, cudaStream_t stream) {

    if (borrowed_handle_ && !weight.is_quantized()) {
        CublasMatmulCUDAOperator<T> dense_matmul(borrowed_handle_);
        dense_matmul(output, input, weight, bias, stream);
        return;
    }

    auto& selector = MatmulSelector<T>::instance();

    auto impl = selector.selectImpl(weight, OperatorPlatform::CUDA);

    if (!impl) {
        throw std::runtime_error("No suitable MatMul implementation found");
    }

    (*impl)(output, input, weight, bias, stream);
}

template class MatmulCUDAOperator<float>;
template class MatmulCUDAOperator<__nv_bfloat16>;

}  // namespace op
