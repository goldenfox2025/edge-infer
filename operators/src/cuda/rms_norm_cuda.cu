#include <cmath>
#include <iostream>
#include <stdexcept>

#include "operators/cuda/rms_norm_cuda.cuh"

namespace op {

__device__ inline float warp_reduce_sum(float val) {
    // 注意：__activemask() 会返回当前活跃线程的掩码
    for (int offset = 32 / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(__activemask(), val, offset);
    }
    return val;
}

template <typename T>
__global__ void rms_norm_kernel(const T* __restrict__ input, T* __restrict__ output, const T* __restrict__ weight,
                                float eps, size_t row_size) {
    // 每个 block 处理一行数据
    int row = blockIdx.x;
    const T* __restrict__ in_row = input + row * row_size;
    T* __restrict__ out_row = output + row * row_size;

    int tid = threadIdx.x;
    int nthreads = blockDim.x;  // 使用 blockDim.x 获取块大小

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

    // 为每个 warp 的部分和分配共享内存
    // 需要足够容纳块中所有 warp 的 leader 线程写入
    // 例如，如果最多 1024 线程，则最多 32 个 warp
    __shared__ float s_warp_sums[32];

    int lane = tid % warpSize;
    int warp_id = tid / warpSize;

    // 每个 warp 的第一个线程 (lane 0) 将其 warp 的归约结果写入共享内存
    if (lane == 0) {
        s_warp_sums[warp_id] = local_sum;
    }

    // 同步，确保所有 warp 的结果都已写入共享内存
    __syncthreads();

    // 让第一个 warp (warp_id == 0) 读取所有 warp 的部分和并进行最终归约
    float block_sum = 0.0f;
    if (warp_id == 0) {
        int num_warps_in_block = (nthreads + warpSize - 1) / warpSize;
        // 读取其他 warp (包括自己) 的部分和
        // 注意：这里的读取操作是分散的，但因为只由一个 warp 执行，影响相对较小
        // 并且读取的数据量很小 (最多 32 个 float)
        float warp_partial_sum = (tid < num_warps_in_block) ? s_warp_sums[tid] : 0.0f;

        // 在第一个 warp 内部再次使用 warp_reduce_sum 进行最终归约
        block_sum = warp_reduce_sum(warp_partial_sum);
        // 此时，block_sum 只在 warp 0 的 lane 0 中持有最终结果
    }

    // 使用共享内存的第一个元素广播最终的 RMS 值或其倒数
    __shared__ float s_inv_rms;
    if (tid == 0) {  // 只有线程 0 计算最终的 rms 并写入共享内存
        // 计算 1 / rms，使用乘法通常比除法快
        s_inv_rms = rsqrtf(block_sum / row_size + eps);
    }

    // 同步，确保 s_inv_rms 已被线程 0 写入
    __syncthreads();

    // 所有线程从共享内存读取广播后的 1/rms 值
    float inv_rms = s_inv_rms;
    // float val;
    flag = 0;
    // 归一化和加权
    for (size_t i = tid; i < row_size; i += nthreads) {
        if (i < row_size) {
            // val = static_cast<float>(in_row[i]);
            float x = val[flag++];
            float w = static_cast<float>(weight[i]);
            // 使用乘法代替除法
            out_row[i] = static_cast<T>((x * inv_rms) * w);
        }
    }
}
template <typename T>
void RmsNormCUDAOperator<T>::operator()(Tensor<T>* output, Tensor<T>* input, Tensor<T>* weight, float eps,
                                        cudaStream_t stream) {
    // 获取输入张量的形状
    const auto& sizes = input->sizes();

    // 确定特征维度和批次维度
    // 最后一个维度是特征维度，其余是批次维度
    size_t feature_dim = sizes.back();
    size_t batch_size = 1;

    // 计算批次大小（所有除最后一维外的维度的乘积）
    for (size_t i = 0; i < sizes.size() - 1; ++i) {
        batch_size *= sizes[i];
    }
    // std::cout << "batch_size: " << batch_size << std::endl;
    // std::cout << "feature_dim: " << feature_dim << std::endl;
    // 配置CUDA核函数的启动参数
    int threads_per_block = 1024;  // 可以根据需要调整

    // 确保线程数不超过设备限制
    if (threads_per_block > 1024)
        threads_per_block = 1024;

    // 网格大小 = 批次大小（每个block处理一个样本）
    dim3 block_dim(threads_per_block);
    dim3 grid_dim(batch_size);  // grid_dim.x = batch_size

    // 启动核函数
    rms_norm_kernel<T><<<grid_dim, block_dim, 0, stream>>>(input->data_ptr(), output->data_ptr(), weight->data_ptr(),
                                                           eps, feature_dim);

    // 错误检查
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error after rms_norm kernel launch: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA rms_norm kernel launch failed");
    }
}

// 显式模板实例化
template class RmsNormCUDAOperator<float>;
template class RmsNormCUDAOperator<__nv_bfloat16>;

}  // namespace op
