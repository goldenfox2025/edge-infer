#include "operators/cuda/legacy/legacy_bridge.cuh"
#include "operators/cuda/gather_cuda.cuh"

namespace op {

template <typename T>
void GatherCUDAOperator<T>::operator()(Tensor<T>* output,
                                       const Tensor<uint32_t>* input,
                                       const Tensor<T>* embedding_table,
                                       cudaStream_t stream) {
  legacy::gather(output, input, embedding_table, stream);
}

template class GatherCUDAOperator<float>;
template class GatherCUDAOperator<__nv_bfloat16>;

}  // namespace op
