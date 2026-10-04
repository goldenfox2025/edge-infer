#include <cmath>
#include <iostream>
#include <stdexcept>
#include <limits>

#include "operators/cuda/direct.hpp"
#include "operators/cuda/rms_norm_cuda.cuh"

namespace op {

__device__ inline float warp_reduce_sum(float val) {
    // Reduce only lanes included in the active mask.
    for (int offset = 32 / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(__activemask(), val, offset);
    }
    return val;
}

template <typename T>
__global__ void rms_norm_kernel(const T* input, T* output, const T* __restrict__ weight,
                                float eps, size_t row_size) {
    // Each block normalizes one row.
    int row = blockIdx.x;
    const T* in_row = input + row * row_size;
    T* out_row = output + row * row_size;

    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    float local_sum = 0.0f;
    float val[10];
    int flag = 0;
    for (size_t i_base = 0; i_base < row_size; i_base += nthreads) {
        size_t i = i_base + tid;

        if (i < row_size) {
            val[flag++] = static_cast<float>(in_row[i]);
            local_sum += val[flag - 1] * val[flag - 1];
        }
    }

    local_sum = warp_reduce_sum(local_sum);

    // Shared storage holds one partial sum per warp (up to 32 warps per block).

    __shared__ float s_warp_sums[32];

    int lane = tid % warpSize;
    int warp_id = tid / warpSize;

    // Lane 0 of each warp publishes its partial sum.
    if (lane == 0) {
        s_warp_sums[warp_id] = local_sum;
    }

    // Wait until all warp partial sums are visible.
    __syncthreads();

    // Warp 0 reduces the partial sums from all warps.
    float block_sum = 0.0f;
    if (warp_id == 0) {
        int num_warps_in_block = (nthreads + warpSize - 1) / warpSize;

        float warp_partial_sum = (tid < num_warps_in_block) ? s_warp_sums[tid] : 0.0f;

        block_sum = warp_reduce_sum(warp_partial_sum);
        // Only lane 0 of warp 0 holds the final block sum.
    }

    // Broadcast the inverse RMS through shared memory.
    __shared__ float s_inv_rms;
    if (tid == 0) {

        s_inv_rms = rsqrtf(block_sum / row_size + eps);
    }

    // Wait for thread 0 to publish the inverse RMS.
    __syncthreads();

    // All threads read the shared inverse RMS.
    float inv_rms = s_inv_rms;
    // float val;
    flag = 0;

    for (size_t i = tid; i < row_size; i += nthreads) {
        if (i < row_size) {
            // val = static_cast<float>(in_row[i]);
            float x = val[flag++];
            float w = static_cast<float>(weight[i]);

            out_row[i] = static_cast<T>((x * inv_rms) * w);
        }
    }
}
template <typename T>
void cuda::rms_norm(ArrayView<const T> input, ArrayView<const T> weight,
                    ArrayView<T> output, std::size_t batch_size,
                    std::size_t feature_dim, float eps, cudaStream_t stream) {
    if (output.size != input.size) {
        throw std::runtime_error("RMSNorm operator: input and output views must have the same size");
    }
    if (input.size == 0) {
        return;
    }
    if (feature_dim == 0 || feature_dim > 10240) {
        throw std::runtime_error("RMSNorm CUDA feature_dim must be between 1 and 10240");
    }
    if (batch_size > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("RMSNorm CUDA row count exceeds the kernel index range");
    }
    if (input.size % feature_dim != 0 || input.size / feature_dim != batch_size ||
        weight.size != feature_dim) {
        throw std::runtime_error("RMSNorm CUDA view extents do not match rows and feature_dim");
    }

    int threads_per_block = 1024;

    if (threads_per_block > 1024)
        threads_per_block = 1024;

    dim3 block_dim(threads_per_block);
    dim3 grid_dim(batch_size);  // grid_dim.x = batch_size

    rms_norm_kernel<T><<<grid_dim, block_dim, 0, stream>>>(input.data, output.data, weight.data,
                                                           eps, feature_dim);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error after rms_norm kernel launch: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA rms_norm kernel launch failed");
    }
}

template <typename T>
void RmsNormCUDAOperator<T>::operator()(Tensor<T>* output, Tensor<T>* input,
                                       Tensor<T>* weight, float eps, cudaStream_t stream) {
    const auto& sizes = input->sizes();
    if (sizes.empty()) {
        throw std::runtime_error("RMSNorm CUDA input requires a feature dimension");
    }
    const size_t feature_dim = sizes.back();
    size_t batch_size = 1;
    for (size_t i = 0; i + 1 < sizes.size(); ++i) {
        batch_size *= sizes[i];
    }
    cuda::rms_norm<T>({input->data_ptr(), input->numel()},
                      {weight->data_ptr(), weight->numel()},
                      {output->data_ptr(), output->numel()},
                      batch_size, feature_dim, eps, stream);
}

template void cuda::rms_norm<float>(ArrayView<const float>, ArrayView<const float>, ArrayView<float>, std::size_t, std::size_t, float, cudaStream_t);
template void cuda::rms_norm<__nv_bfloat16>(ArrayView<const __nv_bfloat16>, ArrayView<const __nv_bfloat16>, ArrayView<__nv_bfloat16>, std::size_t, std::size_t, float, cudaStream_t);

template class RmsNormCUDAOperator<float>;
template class RmsNormCUDAOperator<__nv_bfloat16>;

}  // namespace op
