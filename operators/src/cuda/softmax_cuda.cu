#include "operators/cuda/softmax_cuda.cuh"
#include "operators/cuda/legacy/legacy_bridge.cuh"

namespace op {

template <typename T>
void SoftmaxCUDAOperator<T>::operator()(Tensor<T>* output, const Tensor<T>* input, int dim, 
                                       bool mask, int offset, cudaStream_t stream) {
    // 调用现有的CUDA OP实现
    legacy::softmax(output, input, dim, mask, offset, stream);
}

// 显式模板实例化
template class SoftmaxCUDAOperator<float>;
template class SoftmaxCUDAOperator<__nv_bfloat16>;

} // namespace op
