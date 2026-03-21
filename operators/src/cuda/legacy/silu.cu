#include <cuda_runtime.h>
#include <math.h>

#include <cstdio>
#include <iostream>
#include <vector>

#include "cuda/legacy/legacy_cuda_api.cuh"

namespace cuda_OP {
// --------------------------------------------------
// SiLU 内核与包装函数（模板化）
// --------------------------------------------------
template <typename T>
__global__ void silu_kernel(T *data, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        float x = static_cast<float>(data[idx]);
        data[idx] = static_cast<T>(x / (1.0f + expf(-x)));
    }
}

template <typename T>
void silu(Tensor<T> *output, const Tensor<T> *input, cudaStream_t stream) {
    size_t total = 1;
    for (auto s : input->sizes())
        total *= s;
    if (output->data_ptr() != input->data_ptr()) {
        checkCudaError(cudaMemcpy(output->data_ptr(), input->data_ptr(), total * sizeof(T), cudaMemcpyDeviceToDevice));
    }
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    silu_kernel<<<blocks, threads, 0, stream>>>(output->data_ptr(), total);
    checkCudaError(cudaGetLastError());
    // if (stream == nullptr) {
    //   checkCudaError(cudaDeviceSynchronize());
    // }
}
template void silu<float>(Tensor<float> *, const Tensor<float> *, cudaStream_t);
template void silu<nvbf16>(Tensor<nvbf16> *, const Tensor<nvbf16> *, cudaStream_t);
}  // namespace cuda_OP