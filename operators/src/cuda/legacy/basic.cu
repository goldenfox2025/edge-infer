#include <cublas_v2.h>
#include <cuda_bf16.h>  // 提供 __nv_bfloat16 定义
#include <cuda_runtime.h>
#include <math.h>

#include <cstdio>  // printf
#include <iostream>
#include <stdexcept>
#include <vector>

#include "cuda/legacy/legacy_cuda_api.cuh"

namespace cuda_OP {

// --------------------------------------------------
// 工具函数：检查 CUDA 错误
// --------------------------------------------------
void checkCudaError(cudaError_t error) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA错误: " << cudaGetErrorString(error) << std::endl;
        throw std::runtime_error("CUDA操作失败: " + std::string(cudaGetErrorString(error)));
    }
}

// --------------------------------------------------
// RMSNorm 内核与包装函数（模板化）
// --------------------------------------------------

__device__ inline float warp_reduce_sum(float val) {
    // 注意：__activemask() 会返回当前活跃线程的掩码
    for (int offset = 32 / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(__activemask(), val, offset);
    }
    return val;
}

// v2 版本: 修复内存合并问题，优化块内归约
template <typename T>
__global__ void rms_norm_kernel_v2(const T *__restrict__ input, T *__restrict__ output, const T *__restrict__ weight,
                                   float eps, size_t row_size) {
    // 每个 block 处理一行数据
    int row = blockIdx.x;
    const T *__restrict__ in_row = input + row * row_size;
    T *__restrict__ out_row = output + row * row_size;

    int tid = threadIdx.x;
    int nthreads = blockDim.x;  // 使用 blockDim.x 获取块大小

    float local_sum = 0.0f;
    float val[5];
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

// v1版本

template <typename T>
__global__ void rms_norm_kernel_v1(const T *input, T *output, const T *weight, float eps, size_t row_size) {
    // 每个 block 处理一行数据
    int row = blockIdx.x;
    const T *in_row = input + row * row_size;
    T *out_row = output + row * row_size;

    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    float local_sum = 0.0f;
    for (size_t i = tid; i < row_size; i += nthreads) {
        float val = static_cast<float>(in_row[i]);
        local_sum += val * val;
    }

    local_sum = warp_reduce_sum(local_sum);

    // 每个 warp 的第一个线程将归约结果写入共享内存
    // 最多支持 32 个 warp
    __shared__ float shared[32];
    int lane = tid % 32;
    int warp_id = tid / 32;
    if (lane == 0) {
        shared[warp_id] = local_sum;
    }
    __syncthreads();

    float block_sum = 0.0f;
    int num_warps = (nthreads + 32 - 1) / 32;
    if (warp_id == 0) {
        float warp_partial_sum = 0.0f;
        if (tid < num_warps) {
            warp_partial_sum = shared[lane];
        }
        block_sum = warp_reduce_sum(warp_partial_sum);
    }

    // 将最终归约结果通过共享内存广播到所有线程
    // 使用共享内存的第一个元素来存储最终结果
    if (tid == 0) {
        shared[0] = block_sum;
    }
    __syncthreads();

    float final_sum = shared[0];

    float rms = sqrtf(final_sum / row_size + eps);

    // 归一化，每个线程负责处理多个元素
    for (size_t i = tid; i < row_size; i += nthreads) {
        float val = static_cast<float>(in_row[i]);
        float w = static_cast<float>(weight[i]);
        out_row[i] = static_cast<T>((val / rms) * w);
    }
}

// 无印版本
template <typename T>
__global__ void rms_norm_kernel(const T *input, T *output, const T *weight, float eps, size_t row_size) {
    int row = blockIdx.x;  // 每个 block 处理一行
    const T *in_row = input + row * row_size;
    T *out_row = output + row * row_size;
    float sum = 0.0f;
    for (int i = 0; i < row_size; i++) {
        float val = static_cast<float>(in_row[i]);
        sum += val * val;
    }
    float rms = sqrtf(sum / row_size + eps);
    for (int i = 0; i < row_size; i++) {
        float val = static_cast<float>(in_row[i]);
        float w = static_cast<float>(weight[i]);
        out_row[i] = static_cast<T>((val / rms) * w);
    }
}

template <typename T>
void rms_norm(Tensor<T> *output, const Tensor<T> *input, const Tensor<T> *weight, float eps, cudaStream_t stream) {
    // input/output shape 均为 [seq_len, d]
    size_t seq_len = input->sizes()[0];
    size_t d = input->sizes()[1];  // row_size

    // --- 可调参数 ---
    // 块大小（每行用多少线程处理）
    // 常见选择: 128, 256, 512. 需要根据 GPU 架构和 d 的大小进行调整测试
    // 256 是一个比较通用的起点
    int threads_per_block = 1024;

    // 确保线程数不超过设备限制 (通常是 1024)
    // 同时考虑 d 的大小，如果 d 很小，用太多线程可能浪费
    // if (d < threads_per_block) {
    //     threads_per_block = next_power_of_2(d); // 或者选择一个合理的较小值
    // }
    // 这里暂时不加动态调整逻辑，假设 256 是一个可接受的起点

    // 检查 threads_per_block 是否是 warpSize (32) 的倍数通常更好，但非必须
    // 检查是否超过 1024 (大多数设备的最大值)
    if (threads_per_block > 1024)
        threads_per_block = 1024;

    // --- Kernel 启动 ---
    // 网格大小 gridDim.x = seq_len (一个 block 处理一行)
    // 块大小 blockDim.x = threads_per_block
    dim3 block_dim(threads_per_block);
    dim3 grid_dim(seq_len);  // grid_dim.x = seq_len

    rms_norm_kernel_v2<T>
        <<<grid_dim, block_dim, 0, stream>>>(input->data_ptr(), output->data_ptr(), weight->data_ptr(), eps, d);

    // --- 错误检查和同步 ---
    checkCudaError(cudaGetLastError());
    // 对于性能分析或确保后续 CPU 操作能看到结果，需要同步
    // if (stream == nullptr) {
    //   checkCudaError(cudaDeviceSynchronize());
    // }
}

// Helper function (optional, if you want dynamic adjustment based on d)
// int next_power_of_2(int n) {
//     n--;
//     n |= n >> 1;
//     n |= n >> 2;
//     n |= n >> 4;
//     n |= n >> 8;
//     n |= n >> 16;
//     n++;
//     // Ensure it's at least warpSize for efficiency maybe?
//     return (n < 32) ? 32 : n;
// }

// --------------------------------------------------
// rope 内核与包装函数（模板化）
// --------------------------------------------------

template <typename T>
__global__ void rope_kernel_v1(T *tensor, size_t seq_len, size_t n_heads, size_t head_dim, size_t offset, float theta) {
    // Each thread handles one dimension pair (i, i + head_dim/2) for a specific
    // (seq_idx, head_idx) Grid dimension should be (seq_len * n_heads * head_dim
    // / 2)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_rotations = seq_len * n_heads * (head_dim / 2);

    if (idx < total_rotations) {
        size_t head_dim_half = head_dim / 2;
        size_t i = idx % head_dim_half;              // Dimension index (0 to head_dim/2 - 1)
        size_t head_flat_idx = idx / head_dim_half;  // Flat index for (seq_idx, head_idx)
        size_t seq_idx = head_flat_idx / n_heads;    // Sequence index
        size_t head_idx = head_flat_idx % n_heads;   // Head index

        size_t head_offset = seq_idx * n_heads * head_dim + head_idx * head_dim;
        T *head_ptr = tensor + head_offset;

        // Calculate frequency and rotation angle
        // Inverse frequency calculation is stable across threads working on the
        // same head/seq pos
        float freq = 1.0f / powf(theta, static_cast<float>(2 * i) / static_cast<float>(head_dim));
        float val = (static_cast<float>(seq_idx) + offset) * freq;
        float cos_val;
        float sin_val;
        __sincosf(val, &sin_val, &cos_val);  // Compute sin and cos together

        // Load the pair of elements
        // Accesses head_ptr[i] and head_ptr[i + head_dim_half]
        // Consecutive threads access consecutive 'i', improving coalescing for both
        // reads.
        float x0_f = static_cast<float>(head_ptr[i]);
        float x1_f = static_cast<float>(head_ptr[i + head_dim_half]);

        // Perform rotation
        float rotated_x0 = x0_f * cos_val - x1_f * sin_val;
        float rotated_x1 = x0_f * sin_val + x1_f * cos_val;

        // Store the rotated pair back
        // Consecutive threads access consecutive 'i', improving coalescing for
        // writes.
        head_ptr[i] = static_cast<T>(rotated_x0);
        head_ptr[i + head_dim_half] = static_cast<T>(rotated_x1);
    }
}

// --- BF16 Specialization using __nv_bfloat162 ---
// This leverages vector types and intrinsics for BF16

// Check if CUDA version supports __nv_bfloat162 intrinsics (usually >= 11.0)
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)  // Ampere or later recommended

template <>
__global__ void rope_kernel_v1<__nv_bfloat16>(__nv_bfloat16 *tensor, size_t seq_len, size_t n_heads, size_t head_dim,
                                              size_t offset, float theta) {
    // Each thread handles TWO dimension pairs using bfloat162 type
    // Total number of bfloat162 pairs to process per head is head_dim / 4
    // Grid dimension should be (seq_len * n_heads * head_dim / 4)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t head_dim_half = head_dim / 2;
    size_t pairs_per_head_half = head_dim_half / 2;  // Each bfloat162 holds 2 elements
    size_t total_vec_rotations = seq_len * n_heads * pairs_per_head_half;

    if (idx < total_vec_rotations) {
        size_t vec_i = idx % pairs_per_head_half;          // Index of the bfloat162 pair (0
                                                           // to pairs_per_head_half - 1)
        size_t head_flat_idx = idx / pairs_per_head_half;  // Flat index for (seq_idx, head_idx)
        size_t seq_idx = head_flat_idx / n_heads;          // Sequence index
        size_t head_idx = head_flat_idx % n_heads;         // Head index

        size_t head_offset = seq_idx * n_heads * head_dim + head_idx * head_dim;
        __nv_bfloat16 *head_ptr = tensor + head_offset;

        // Calculate dimension indices for the two elements in the vector
        size_t i0 = vec_i * 2;
        size_t i1 = vec_i * 2 + 1;

        // Calculate frequencies and rotation angles for both elements
        // float freq0 = 1.0f / powf(theta, static_cast<float>(2 * i0) /
        //                                      static_cast<float>(head_dim));
        // float freq1 = 1.0f / powf(theta, static_cast<float>(2 * i1) /
        //                                      static_cast<float>(head_dim));

        float log_theta = logf(theta);
        float neg_two_log_theta_div_hd = -2.0f * log_theta / static_cast<float>(head_dim);

        // --- Inside the loop or calculation for specific i0/i1 ---
        // Calculate the argument for expf
        float exp_arg0 = neg_two_log_theta_div_hd * static_cast<float>(i0);
        float exp_arg1 = neg_two_log_theta_div_hd * static_cast<float>(i1);

        // Calculate frequencies using expf
        float freq0 = expf(exp_arg0);
        float freq1 = expf(exp_arg1);

        float val0 = (static_cast<float>(seq_idx) + offset) * freq0;
        float val1 = (static_cast<float>(seq_idx) + offset) * freq1;

        float cos_val0, sin_val0, cos_val1, sin_val1;
        __sincosf(val0, &sin_val0, &cos_val0);
        __sincosf(val1, &sin_val1, &cos_val1);

        // Load a pair of bfloat162 (4 elements total) using vector load
        // reinterpret_cast is necessary for vectorized loads/stores
        __nv_bfloat162 x0_vec = *reinterpret_cast<__nv_bfloat162 *>(&head_ptr[i0]);  // Loads elements at i0, i1
        __nv_bfloat162 x1_vec =
            *reinterpret_cast<__nv_bfloat162 *>(&head_ptr[i0 + head_dim_half]);  // Loads elements at i0+d/2, i1+d/2

        // Convert bf16 vectors to float vectors for calculation
        float2 x0_fvec = __bfloat1622float2(x0_vec);
        float2 x1_fvec = __bfloat1622float2(x1_vec);

        // Pack sin/cos values into float2 for potential vector operations (though
        // used component-wise here)
        float2 cos_vec = make_float2(cos_val0, cos_val1);
        float2 sin_vec = make_float2(sin_val0, sin_val1);

        // Perform rotation using float components (or use bf16 intrinsics if
        // preferred, requires bf16 sin/cos) If using intrinsics, convert cos/sin to
        // bf162 first:
        // __nv_bfloat162 cos_bf16 = __float22bfloat162_rn(cos_vec);
        // __nv_bfloat162 sin_bf16 = __float22bfloat162_rn(sin_vec);
        // __nv_bfloat162 neg_x1_vec = __hnegb2(x1_vec); // Negate x1 for FMA
        // __nv_bfloat162 rotated_x0_bf16 = __hfma2(x0_vec, cos_bf16,
        // __hmul2(neg_x1_vec, sin_bf16));
        // __nv_bfloat162 rotated_x1_bf16 = __hfma2(x0_vec, sin_bf16,
        // __hmul2(x1_vec, cos_bf16));

        // Component-wise rotation using float:
        float rot_x0_f0 = x0_fvec.x * cos_vec.x - x1_fvec.x * sin_vec.x;
        float rot_x0_f1 = x0_fvec.y * cos_vec.y - x1_fvec.y * sin_vec.y;
        float rot_x1_f0 = x0_fvec.x * sin_vec.x + x1_fvec.x * cos_vec.x;
        float rot_x1_f1 = x0_fvec.y * sin_vec.y + x1_fvec.y * cos_vec.y;

        // Convert results back to bfloat162
        __nv_bfloat162 rotated_x0_bf16 = __float22bfloat162_rn(make_float2(rot_x0_f0, rot_x0_f1));
        __nv_bfloat162 rotated_x1_bf16 = __float22bfloat162_rn(make_float2(rot_x1_f0, rot_x1_f1));

        // Store the rotated pairs back using vector store
        *reinterpret_cast<__nv_bfloat162 *>(&head_ptr[i0]) = rotated_x0_bf16;
        *reinterpret_cast<__nv_bfloat162 *>(&head_ptr[i0 + head_dim_half]) = rotated_x1_bf16;
    }
}
#endif  // __CUDA_ARCH__ >= 800

template <typename T>
__global__ void rope_kernel(T *tensor, size_t seq_len, size_t n_heads, size_t head_dim, size_t offset, float theta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < seq_len * n_heads) {
        size_t seq_idx = idx / n_heads;
        size_t head_idx = idx % n_heads;
        T *head_ptr = tensor + seq_idx * n_heads * head_dim + head_idx * head_dim;
        size_t dim_half = head_dim / 2;
        for (size_t i = 0; i < dim_half; i++) {
            float freq = 1.0f / powf(theta, (2.0f * i) / head_dim);
            float val = (seq_idx + offset) * freq;
            float cos_val = cosf(val);
            float sin_val = sinf(val);
            float x0 = static_cast<float>(head_ptr[i]);
            float x1 = static_cast<float>(head_ptr[i + dim_half]);
            head_ptr[i] = static_cast<T>(x0 * cos_val - x1 * sin_val);
            head_ptr[i + dim_half] = static_cast<T>(x0 * sin_val + x1 * cos_val);
        }
    }
}

template <typename T>
void rope(Tensor<T> *x, size_t offset, float theta, cudaStream_t stream) {
    const auto &sizes = x->sizes();
    if (sizes.size() < 3) {
        throw std::runtime_error("rope: tensor must be at least 3D");
    }
    size_t seq_len = sizes[0];
    size_t n_heads = sizes[1];
    size_t head_dim = sizes[2];

    if (head_dim == 0)
        return;  // Nothing to do
    if (head_dim % 2 != 0) {
        throw std::runtime_error("rope: head_dim must be even");
    }

    int threads = 256;  // Common block size, can be tuned (128, 512, etc.)
    int blocks = 0;
    void *kernel_ptr = nullptr;

    // Select kernel and grid based on type
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        if (head_dim % 4 != 0) {
            // BF16 kernel requires head_dim to be a multiple of 4 for bfloat162
            // loads/stores Fallback to generic kernel or throw error. Here we throw.
            throw std::runtime_error(
                "rope: BF16 requires head_dim to be a multiple of 4 for optimized "
                "kernel");
        }
        size_t total_vec_rotations = seq_len * n_heads * (head_dim / 4);  // Each thread handles a bfloat162 pair
        blocks = (total_vec_rotations + threads - 1) / threads;
        kernel_ptr = (void *)rope_kernel_v1<__nv_bfloat16>;
        // std::cout << "Using BF16 Optimized Kernel" << std::endl; // For debugging
    } else
#endif
    {
        // Generic FP32/FP16 path
        size_t total_rotations = seq_len * n_heads * (head_dim / 2);  // Each thread handles one pair
        blocks = (total_rotations + threads - 1) / threads;
        kernel_ptr = (void *)rope_kernel_v1<T>;
        // std::cout << "Using Generic Optimized Kernel" << std::endl; // For
        // debugging
    }

    if (blocks == 0 && (seq_len * n_heads * head_dim) > 0) {
        // Handle cases where total rotations might be 0 but tensor isn't empty
        // or very small head_dim resulted in 0 rotations/blocks.
        // If head_dim was 0, we returned earlier. If head_dim is > 0, blocks should
        // be > 0. This calculation should ensure blocks > 0 if work needs to be
        // done. If blocks is still 0, it likely means seq_len or n_heads is 0.
        if (seq_len > 0 && n_heads > 0 && head_dim > 0) {
            blocks = 1;  // Launch at least one block if there's data
        } else {
            return;  // No work to do
        }
    }

    // --- Kernel Launch ---
    // We use a function pointer to avoid repeating the launch code inside
    // if/else. Note: Directly using the kernel function name is usually preferred
    // for type safety, but this shows how to handle it if selecting dynamically.
    if (kernel_ptr == (void *)rope_kernel_v1<T>) {
        rope_kernel_v1<T><<<blocks, threads, 0, stream>>>(x->data_ptr(), seq_len, n_heads, head_dim, offset, theta);
    }
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    else if (kernel_ptr == (void *)rope_kernel_v1<__nv_bfloat16>) {
        rope_kernel_v1<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16 *>(x->data_ptr()),  // Cast needed if T is bf16
            seq_len, n_heads, head_dim, offset, theta);
    }
#endif
    else {
        throw std::runtime_error("Internal error: No valid kernel selected.");
    }

    // --- Error Checking ---
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error after rope kernel launch: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA rope kernel launch failed");
    }

    if (stream == nullptr) {
        // err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cerr << "CUDA synchronization error after rope: " << cudaGetErrorString(err) << std::endl;
            throw std::runtime_error("CUDA rope synchronization failed");
        }
    }
}

template <typename T>
__global__ void attention_scores_kernel(const T *__restrict__ Q,  // Q 指针，假定布局为 [n_q_h, dqkv] 或等效
                                        const int n_q_h, const int dqkv,
                                        const T *__restrict__ K,  // K 指针，布局 [cache_length, n_kv_h, dqkv]
                                        const int cache_length,
                                        T *__restrict__ att_scores,  // 输出，布局 [n_q_h, cache_length]
                                        const int n_kv_h) {
    // 使用 extern 声明共享内存，大小在启动时指定
    // 需要存储一个 Q 向量
    __shared__ T smemQ[128];

    // Grid 映射: blockIdx.x -> q (query head), blockIdx.y -> pos_block
    // (块内位置的起始) Block 映射: threadIdx.x -> pos_offset (块内位置的偏移)
    const int q = blockIdx.x;
    const int pos_base = blockIdx.y * blockDim.x;
    const int tid = threadIdx.x;  // Thread ID within the block (0 to blockDim.x - 1)
    const int block_size = blockDim.x;

    // 计算当前线程负责的 cache position
    const int pos = pos_base + tid;

    // --- 边界检查 ---
    // 检查 q 是否有效 (理论上 gridDim.x == n_q_h)
    if (q >= n_q_h) {
        return;
    }

    // --- 加载 Q 到共享内存 ---
    // 让块内的线程协作加载 Q[q] 到 smemQ
    // Q 的布局假定为 [n_q_h, dqkv] (或等效 [1, n_q_h, dqkv])
    // q_vec 指向当前 query head 的数据起始位置
    const T *q_vec = Q + static_cast<size_t>(q) * dqkv;

    // 每个线程负责加载 Q 向量的一部分
    // 使用循环确保所有 dqkv 维度都被加载，即使 dqkv > block_size
    for (int i = tid; i < dqkv; i += block_size) {
        // 检查 i 是否越界（理论上不需要，因为上层循环保证）
        smemQ[i] = q_vec[i];
    }
    // 等待块内所有线程完成加载 Q 到共享内存
    __syncthreads();

    // --- 计算 Attention Score ---
    // 再次检查 pos 是否越界 (因为 cache_length 可能不是 block_size 的整数倍)
    if (pos < cache_length) {
        // 计算对应的 KV head
        const int n_groups = n_q_h / n_kv_h;
        const int kv_head = q / n_groups;

        // 计算 K 向量的起始地址
        // K 布局: [cache_length, n_kv_h, dqkv]
        // 索引: pos * n_kv_h * dqkv + kv_head * dqkv + i
        //      = (pos * n_kv_h + kv_head) * dqkv + i
        const size_t k_vec_offset = (static_cast<size_t>(pos) * n_kv_h + kv_head) * dqkv;
        const T *k_vec = K + k_vec_offset;

        // 计算点积 (从共享内存读取 Q, 从全局内存读取 K)
        float dot = 0.0f;
        for (int i = 0; i < dqkv; ++i) {
            // 从共享内存读取 Q 值
            float q_val = static_cast<float>(smemQ[i]);
            // 从全局内存读取 K 值 (访问 k_vec[i] 是连续的)
            float k_val = static_cast<float>(k_vec[i]);
            // 累加点积
            dot = fmaf(q_val, k_val, dot);  // 使用 FMA 指令
                                            // dot += q_val * k_val; // 等价写法
        }

        // 计算缩放因子 (1 / sqrt(dqkv))
        // 使用 rsqrtf (更快，但精度可能略低) 或 1.0f / sqrtf
        const float scale = rsqrtf(static_cast<float>(dqkv));
        // const float scale = 1.0f / sqrtf(static_cast<float>(dqkv)); //
        // 备选，精度更高

        // 写入结果到全局内存 (写入 att_scores[q * cache_length + pos])
        // 由于 pos = pos_base + threadIdx.x，块内的写入是合并的
        att_scores[static_cast<size_t>(q) * cache_length + pos] = static_cast<T>(dot * scale);
    }
    // 注意：这里不需要最后的 __syncthreads()，因为线程之间没有后续依赖
}

// 2. 修改后的 C++ 封装函数 (compute_attention_scores)
template <typename T>
void compute_attention_scores(const Tensor<T> &Q, const Tensor<T> &K, size_t n_q_h, size_t dqkv, Tensor<T> &att_scores,
                              size_t n_kv_h, cudaStream_t stream) {
    // --- 输入检查 ---
    if (n_q_h == 0 || dqkv == 0 || n_kv_h == 0) {
        throw std::runtime_error("Head counts (n_q_h, n_kv_h) and dimension (dqkv) must be non-zero.");
    }
    if (n_q_h % n_kv_h != 0) {
        char msg[256];
        snprintf(msg, sizeof(msg), "n_q_h (%zu) must be divisible by n_kv_h (%zu)", n_q_h, n_kv_h);
        throw std::runtime_error(msg);
    }

    const auto &k_sizes = K.sizes();
    if (k_sizes.size() != 3) {
        throw std::runtime_error("K tensor must have 3 dimensions [cache_length, n_kv_h, dqkv]");
    }
    const size_t cache_length = k_sizes[0];
    if (k_sizes[1] != n_kv_h || k_sizes[2] != dqkv) {
        char msg[512];
        snprintf(msg, sizeof(msg), "K tensor shape mismatch. Expected [*, %zu, %zu], got [%zu, %zu, %zu]", n_kv_h, dqkv,
                 k_sizes[0], k_sizes[1], k_sizes[2]);
        throw std::runtime_error(msg);
    }

    const auto &q_sizes = Q.sizes();
    bool is_3d_q = (q_sizes.size() == 3);
    // size_t expected_q_elems = n_q_h * dqkv;
    size_t actual_q_elems = 1;
    for (size_t dim : q_sizes) {
        actual_q_elems *= dim;
    }

    if (is_3d_q) {
        // 允许 [1, n_q_h, dqkv]
        if (q_sizes[0] != 1 || q_sizes[1] != n_q_h || q_sizes[2] != dqkv) {
            char msg[512];
            snprintf(msg, sizeof(msg),
                     "Q tensor shape mismatch (3D). Expected [1, %zu, %zu], got "
                     "[%zu, %zu, %zu]",
                     n_q_h, dqkv, q_sizes[0], q_sizes[1], q_sizes[2]);
            throw std::runtime_error(msg);
        }
    } else if (q_sizes.size() == 2) {
        // 允许 [n_q_h, dqkv]
        if (q_sizes[0] != n_q_h || q_sizes[1] != dqkv) {
            char msg[512];
            snprintf(msg, sizeof(msg), "Q tensor shape mismatch (2D). Expected [%zu, %zu], got [%zu, %zu]", n_q_h, dqkv,
                     q_sizes[0], q_sizes[1]);
            throw std::runtime_error(msg);
        }
    } else {
        char msg[512];
        snprintf(msg, sizeof(msg), "Q tensor must have 2 or 3 dimensions, got %zu dimensions.", q_sizes.size());
        throw std::runtime_error(msg);
    }
    // Kernel 假定 Q 是连续的 n_q_h * dqkv 数据，无论是 [1, n_q_h, dqkv] 还是
    // [n_q_h, dqkv] 所以我们不需要根据 is_3d_q 传递不同的指针或标志

    const auto &score_sizes = att_scores.sizes();
    if (score_sizes.size() != 2 || score_sizes[0] != n_q_h || score_sizes[1] != cache_length) {
        char msg[512];
        snprintf(msg, sizeof(msg),
                 "Attention scores tensor shape mismatch. Expected [%zu, %zu], got "
                 "[%zu, %zu]",
                 n_q_h, cache_length, score_sizes.size() > 0 ? score_sizes[0] : 0,
                 score_sizes.size() > 1 ? score_sizes[1] : 0);
        throw std::runtime_error(msg);
    }

    if (cache_length == 0) {
        // 如果 cache length 为 0，输出应该为空或形状匹配但无数据。
        // 核函数在这种情况下 gridDim.y 会是 0，不会启动。
        // 或者可以在这里直接返回，确保 att_scores 状态正确。
        return;
    }

    // --- CUDA Kernel 启动配置 ---
    // 选择 Block 大小 (线程数)
    // 256 是一个常用的、通常性能不错的选择，可以调整测试
    const int block_size_x = 256;  // 使用 1D block

    // Grid 维度
    // gridDim.x 对应 q (n_q_h)
    // gridDim.y 对应 cache_length 的块数
    // 需要向上取整来覆盖所有的 cache_length
    dim3 gridDim(static_cast<unsigned int>(n_q_h),
                 static_cast<unsigned int>((cache_length + block_size_x - 1) / block_size_x), 1);

    // Block 维度 (1D)
    dim3 blockDim(block_size_x, 1, 1);

    // 计算共享内存大小：需要存储 dqkv 个 T 类型的元素
    // size_t shared_mem_size = dqkv * sizeof(T);
    // // 注意：共享内存大小有限制 (e.g., 48KB, 96KB per SM)。如果 dqkv *
    // sizeof(T)
    // // 过大，此策略会失败。 例如，如果 T=float(4B), dqkv=128, 需要 512
    // // Bytes，非常小。 如果 T=float(4B), dqkv=16384 (非常大), 需要
    // // 64KB，可能会超过限制或影响占用率。
    // cudaDeviceProp prop;
    // cudaGetDeviceProperties(&prop, 0);  // Get properties of device 0
    // if (shared_mem_size > prop.sharedMemPerBlock) {
    //   char msg[512];
    //   snprintf(msg, sizeof(msg),
    //            "Required shared memory size (%zu bytes) exceeds device limit
    //            per " "block (%zu bytes) for dqkv=%zu.", shared_mem_size,
    //            prop.sharedMemPerBlock, dqkv);
    //   throw std::runtime_error(msg);
    // }
    // if (shared_mem_size > prop.sharedMemPerMultiprocessor) {
    //   // Technically okay if occupancy allows, but good to be aware
    //   // printf("Warning: Shared memory size (%zu bytes) is large relative to
    //   SM
    //   // limit (%zu bytes).\n",
    //   //        shared_mem_size, prop.sharedMemPerMultiprocessor);
    // }

    // 启动内核
    attention_scores_kernel<<<gridDim, blockDim, 0, stream>>>(Q.data_ptr(),  // 直接传递 Q 的数据指针
                                                              static_cast<int>(n_q_h), static_cast<int>(dqkv),
                                                              K.data_ptr(), static_cast<int>(cache_length),
                                                              att_scores.data_ptr(), static_cast<int>(n_kv_h));

    // 检查错误
    checkCudaError(cudaGetLastError());
    // checkCudaError(cudaDeviceSynchronize()); // 通常不需要立即同步
}

// --------------------------------------------------
// 注意力计算：decode 版本 — 计算注意力输出（模板化）
// --------------------------------------------------
template <typename T>
__global__ void att_output_kernel(const T *__restrict__ att_probs,  // 使用 __restrict__ 提示编译器指针不混叠
                                  const int n_q_h,                  // 总 query head 数量
                                  const int cache_length,           // K/V cache 的长度 (seq len)
                                  const int dqkv,                   // 每个 head 的维度
                                  const T *__restrict__ V, T *__restrict__ att_output,
                                  const int n_kv_h)  // 总 key/value head 数量
{
    // 每个 block 处理一个 query head (q)
    const int q = blockIdx.x;
    // 每个 thread 处理一个或多个 d 维度
    const int d_start = threadIdx.x;
    const int d_step = blockDim.x;  // 线程块中处理 d 维度的线程数

    // 检查 q 是否越界 (虽然 gridDim.x 通常等于 n_q_h, 但以防万一)
    if (q >= n_q_h) {
        return;
    }

    // 计算这个 query head 对应的 key/value head 索引
    // 注意：整型除法会自动向下取整，这正是我们需要的
    const int n_groups = n_q_h / n_kv_h;  // 每个 KV head 对应的 Q head 数量
    const int kv_head = q / n_groups;

    // 计算 V 张量中对应 kv_head 的起始地址偏移 (不包含 d 维度)
    // V 的形状是 [cache_length, n_kv_h, dqkv]
    // 访问 V[pos, kv_head, d] 的线性地址是:
    // pos * (n_kv_h * dqkv) + kv_head * dqkv + d
    // = (pos * n_kv_h + kv_head) * dqkv + d

    // 指向当前 query head 的 attention probabilities 行的指针
    const T *current_att_probs_row = att_probs + static_cast<size_t>(q) * cache_length;

    // 指向当前 query head 的 output 行的指针
    T *current_att_output_row = att_output + static_cast<size_t>(q) * dqkv;

    // 每个线程负责计算多个 d 维度 (如果 dqkv > blockDim.x)
    for (int d = d_start; d < dqkv; d += d_step) {
        float sum = 0.0f;  // 使用 float 进行累加，保证精度

        // V 张量的指针，固定 kv_head 和 d，随 pos 变化
        // const T* v_ptr_for_d = V + v_base_offset + d; // 不对， V 的布局是 (pos,
        // kv_head, d)

        // 遍历 K/V cache (序列长度)
        for (int pos = 0; pos < cache_length; ++pos) {
            // 读取 attention probability (所有处理相同 q
            // 的线程读取相同的值，形成广播)
            const float prob = static_cast<float>(current_att_probs_row[pos]);

            // 计算 V 中元素的索引
            // V 的形状: [cache_length, n_kv_h, dqkv]
            // 线性索引: pos * n_kv_h * dqkv + kv_head * dqkv + d
            // 优化索引: (pos * n_kv_h + kv_head) * dqkv + d
            const size_t v_index = (static_cast<size_t>(pos) * n_kv_h + kv_head) * dqkv + d;
            const float val = static_cast<float>(V[v_index]);  // 读取 V 值 (线程块内对 d 的访问是合并的)

            // 累加
            // 使用 fmaf (fused multiply-add)
            // 可能稍微提高性能，但现代编译器通常会自动优化
            sum = fmaf(prob, val, sum);
            // sum += prob * val; // 等价写法
        }

        // 将结果写回 global memory (线程块内对 d 的访问是合并的)
        current_att_output_row[d] = static_cast<T>(sum);
    }
}

template <typename T>
void compute_att_output(const Tensor<T> &att_probs, const Tensor<T> &V, size_t n_q_h, size_t dqkv,
                        Tensor<T> &att_output, size_t n_kv_h, cudaStream_t stream) {
    // --- 输入检查 ---
    // 检查维度信息是否匹配
    if (n_q_h == 0 || dqkv == 0 || n_kv_h == 0) {
        throw std::runtime_error("Head counts (n_q_h, n_kv_h) and dimension (dqkv) must be non-zero.");
    }
    if (n_q_h % n_kv_h != 0) {
        char msg[256];
        snprintf(msg, sizeof(msg), "n_q_h (%zu) must be divisible by n_kv_h (%zu)", n_q_h, n_kv_h);
        throw std::runtime_error(msg);
    }

    const auto &v_sizes = V.sizes();
    if (v_sizes.size() != 3) {
        throw std::runtime_error("V tensor must have 3 dimensions [cache_length, n_kv_h, dqkv]");
    }
    const size_t cache_length = v_sizes[0];
    if (v_sizes[1] != n_kv_h || v_sizes[2] != dqkv) {
        char msg[512];
        snprintf(msg, sizeof(msg), "V tensor shape mismatch. Expected [*, %zu, %zu], got [%zu, %zu, %zu]", n_kv_h, dqkv,
                 v_sizes[0], v_sizes[1], v_sizes[2]);
        throw std::runtime_error(msg);
    }
    // if (cache_length == 0) {
    // 如果 cache length 为 0，输出应该全为
    // 0，可以提前处理或允许核函数运行（循环次数为0）
    // 这里假设允许核函数处理，它会正确地将 sum 初始化为 0 并写回。
    // 如果需要严格处理，可以在这里添加逻辑填充 att_output 为 0 并返回。
    // 例如: cudaMemsetAsync(att_output.data_ptr(), 0, att_output.numel() *
    // sizeof(T), stream); return;
    // }

    const auto &probs_sizes = att_probs.sizes();
    if (probs_sizes.size() != 2 || probs_sizes[0] != n_q_h || probs_sizes[1] != cache_length) {
        char msg[512];
        snprintf(msg, sizeof(msg),
                 "Attention probabilities tensor shape mismatch. Expected [%zu, "
                 "%zu], got [%zu, %zu]",
                 n_q_h, cache_length, probs_sizes.size() > 0 ? probs_sizes[0] : 0,
                 probs_sizes.size() > 1 ? probs_sizes[1] : 0);
        throw std::runtime_error(msg);
    }

    const auto &output_sizes = att_output.sizes();
    if (output_sizes.size() != 2 || output_sizes[0] != n_q_h || output_sizes[1] != dqkv) {
        char msg[512];
        snprintf(msg, sizeof(msg),
                 "Attention output tensor shape mismatch. Expected [%zu, %zu], got "
                 "[%zu, %zu]",
                 n_q_h, dqkv, output_sizes.size() > 0 ? output_sizes[0] : 0,
                 output_sizes.size() > 1 ? output_sizes[1] : 0);
        throw std::runtime_error(msg);
    }

    // --- CUDA Kernel 启动配置 ---
    // 每个 block 处理一个 query head (q)
    // block 内的线程处理 d 维度
    // 选择一个合适的 block 大小，通常是 128, 256, 512 等，取决于 dqkv 和 GPU 架构
    // 这里选择 256 作为示例，可以根据实际情况调整
    const int block_dim_d = 128;

    // Grid 维度：我们需要 n_q_h 个 block，每个 block 负责一个 q
    dim3 gridDim(static_cast<unsigned int>(n_q_h), 1, 1);
    // Block 维度：我们用 block_dim_d 个线程来并行处理 d 维度
    dim3 blockDim(block_dim_d, 1, 1);

    // 启动内核
    att_output_kernel<<<gridDim, blockDim, 0, stream>>>(att_probs.data_ptr(), static_cast<int>(n_q_h),
                                                        static_cast<int>(cache_length), static_cast<int>(dqkv),
                                                        V.data_ptr(), att_output.data_ptr(), static_cast<int>(n_kv_h));

    checkCudaError(cudaGetLastError());
}

template <typename T>
void compute_attention_scores_prefill(const Tensor<T> &Q, const Tensor<T> &K, Tensor<T> &att_scores, size_t dqkv,
                                      cudaStream_t stream) {
    // 使用launch_gqa_gemm计算注意力分数
    // 这个函数专门处理GQA场景，一个KV头对应多个Q头
    // 通过索引映射而非复制来处理这种关系，提高内存效率
    // 打印Q的sizes数组

    cuda_OP::launch_gqa_gemm(Q, K, att_scores, stream);
}

/**
 * @brief 修正后的高效注意力输出计算内核。
 *
 * 此内核严格遵循原始计算逻辑，采用分块方法优化内存访问。
 * 网格划分: 1个线程块 -> 1个查询头(q)的输出向量。
 * 线程协同: 块内线程协同计算完整的dqkv维输出向量。
 * 共享内存: 对内层循环(cache_length)和外层循环(dqkv)进行分块，
 * 将V和att_probs的切片加载到共享内存中以实现数据重用。
 *
 * @tparam T 数据类型 (e.g., float, half)。
 * @tparam BLOCK_THREADS 每个块的线程数，推荐为128或256。
 * @tparam TILE_J_DIM J维度（cache_length）的分块大小，推荐为8或16。
 */
template <typename T, int BLOCK_THREADS, int TILE_J_DIM>
__global__ void att_output_prefill_revised_kernel(const T *__restrict__ att_probs, const T *__restrict__ V,
                                                  T *att_output, int n_q, int cache_length, int dqkv, int n_kv_h,
                                                  int n_q_h) {
    // 共享内存，用于缓存V和att_probs的切片
    __shared__ T v_tile[TILE_J_DIM][BLOCK_THREADS];
    __shared__ T prob_tile[TILE_J_DIM];

    // --- 1. 索引和常量计算 ---
    // 每个线程块处理一个查询头
    const int q = blockIdx.x;
    const int tx = threadIdx.x;

    // 计算此查询块共享的索引
    const int s = q / n_q_h;
    const int qh = q % n_q_h;
    const int n_groups = n_q_h / n_kv_h;
    const int kv_head = qh / n_groups;

    // --- 2. 主计算循环 ---
    // 块内线程协同处理dqkv维度，每次处理一个大小为BLOCK_THREADS的块
    for (int d_base = 0; d_base < dqkv; d_base += BLOCK_THREADS) {
        const int d = d_base + tx;
        float sum = 0.0f;

        // J维度(cache_length)的分块循环
        for (int j_base = 0; j_base < cache_length; j_base += TILE_J_DIM) {
// --- 2a. 从全局内存协同加载数据到共享内存 ---
// 使用块内线程加载V的二维切片 v_tile[TILE_J_DIM][BLOCK_THREADS]
#pragma unroll
            for (int j_offset = 0; j_offset < TILE_J_DIM; ++j_offset) {
                const int j = j_base + j_offset;
                if (j < cache_length && d < dqkv) {
                    const int v_idx = (j * n_kv_h + kv_head) * dqkv + d;
                    v_tile[j_offset][tx] = V[v_idx];
                } else {
                    v_tile[j_offset][tx] = static_cast<T>(0.0f);
                }
            }

            // 使用每个warp的第一个线程加载att_probs切片 (避免冗余读取)
            if (tx < TILE_J_DIM) {
                const int j = j_base + tx;
                if (j < cache_length) {
                    const int att_idx = s * (n_q_h * cache_length) + qh * cache_length + j;
                    prob_tile[tx] = att_probs[att_idx];
                } else {
                    prob_tile[tx] = static_cast<T>(0.0f);
                }
            }
            __syncthreads();  // 确保所有数据加载完毕

// --- 2b. 在共享内存中进行计算 ---
#pragma unroll
            for (int j_offset = 0; j_offset < TILE_J_DIM; ++j_offset) {
                float prob = static_cast<float>(prob_tile[j_offset]);
                float val = static_cast<float>(v_tile[j_offset][tx]);
                sum += prob * val;
            }
            __syncthreads();  // 确保计算完成，准备加载下一块
        }

        // --- 2c. 将结果写回全局内存 ---
        if (d < dqkv) {
            const int out_idx = q * dqkv + d;
            att_output[out_idx] = static_cast<T>(sum);
        }
    }
}

/**
 * @brief 主机端启动函数 (接口保持不变)
 *
 * 调用修正后的、高效的 `att_output_prefill_revised_kernel` 内核。
 */
template <typename T>
void compute_att_output_prefill(const Tensor<T> &att_probs, const Tensor<T> &V, Tensor<T> &att_output, size_t n_q_h,
                                size_t dqkv, size_t total_seq_len, size_t n_kv_h, cudaStream_t stream) {
    // Tensor维度检查
    size_t batch_size = att_probs.sizes()[0];
    size_t cache_length = att_probs.sizes()[2];

    if (att_probs.sizes()[1] != n_q_h) {
        throw std::runtime_error("attention probabilities head dimension mismatch");
    }
    if (V.sizes()[0] != cache_length || V.sizes()[1] != n_kv_h || V.sizes()[2] != dqkv) {
        throw std::runtime_error("V tensor dimension mismatch");
    }
    if (att_output.sizes()[0] != batch_size || att_output.sizes()[1] != n_q_h || att_output.sizes()[2] != dqkv) {
        throw std::runtime_error("attention output tensor shape mismatch");
    }

    // 总查询数 (等效于输出矩阵的行数或批次数)
    const size_t total_q = batch_size * n_q_h;

    // 定义内核启动参数
    // 每个块的线程数
    const int BLOCK_THREADS = 128;
    // J维度(cache_length)的平铺大小。这会影响共享内存使用量和循环展开效果。
    const int TILE_J_DIM = 8;

    // 网格维度：每个查询头启动一个线程块
    const dim3 grid_dim(static_cast<unsigned int>(total_q));
    // 块维度：固定的线程数
    const dim3 block_dim(BLOCK_THREADS);

    // 启动修正后的优化内核
    att_output_prefill_revised_kernel<T, BLOCK_THREADS, TILE_J_DIM><<<grid_dim, block_dim, 0, stream>>>(
        att_probs.data_ptr(), V.data_ptr(), att_output.data_ptr(), static_cast<int>(total_q),
        static_cast<int>(cache_length), static_cast<int>(dqkv), static_cast<int>(n_kv_h), static_cast<int>(n_q_h));

    checkCudaError(cudaGetLastError());
}

__global__ void init_curand_state_kernel(curandState *states, unsigned long long seed, unsigned long long offset) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(seed, 0, offset, &states[0]);
    }
}
void init_curand(curandState *d_states, unsigned long long seed, int offset, cudaStream_t stream) {
    int blocks = 1;
    int threads = 1;
    init_curand_state_kernel<<<blocks, threads, 0, stream>>>(d_states, seed, offset);
    checkCudaError(cudaGetLastError());
    // if (stream == nullptr) {
    //   checkCudaError(cudaDeviceSynchronize());
    // }
}

template void rope<float>(Tensor<float> *, size_t, float, cudaStream_t);
template void rms_norm<float>(Tensor<float> *, const Tensor<float> *, const Tensor<float> *, float, cudaStream_t);

template void compute_attention_scores<float>(const Tensor<float> &, const Tensor<float> &, size_t, size_t,
                                              Tensor<float> &, size_t, cudaStream_t);
template void compute_att_output<float>(const Tensor<float> &, const Tensor<float> &, size_t, size_t, Tensor<float> &,
                                        size_t, cudaStream_t);
template void compute_attention_scores_prefill<float>(const Tensor<float> &, const Tensor<float> &, Tensor<float> &,
                                                      size_t, cudaStream_t);
template void compute_att_output_prefill<float>(const Tensor<float> &, const Tensor<float> &, Tensor<float> &, size_t,
                                                size_t, size_t, size_t, cudaStream_t);

// 对 __half 类型的实例化
template void rope<__half>(Tensor<__half> *, size_t, float, cudaStream_t);
template void rms_norm<__half>(Tensor<__half> *, const Tensor<__half> *, const Tensor<__half> *, float, cudaStream_t);

template void compute_attention_scores<__half>(const Tensor<__half> &, const Tensor<__half> &, size_t, size_t,
                                               Tensor<__half> &, size_t, cudaStream_t);
template void compute_att_output<__half>(const Tensor<__half> &, const Tensor<__half> &, size_t, size_t,
                                         Tensor<__half> &, size_t, cudaStream_t);
template void compute_attention_scores_prefill<__half>(const Tensor<__half> &, const Tensor<__half> &, Tensor<__half> &,
                                                       size_t, cudaStream_t);
template void compute_att_output_prefill<__half>(const Tensor<__half> &, const Tensor<__half> &, Tensor<__half> &,
                                                 size_t, size_t, size_t, size_t, cudaStream_t);

// 对 nvbf16 类型的实例化

template void rope<nvbf16>(Tensor<nvbf16> *, size_t, float, cudaStream_t);
template void rms_norm<nvbf16>(Tensor<nvbf16> *, const Tensor<nvbf16> *, const Tensor<nvbf16> *, float, cudaStream_t);

template void compute_attention_scores<nvbf16>(const Tensor<nvbf16> &, const Tensor<nvbf16> &, size_t, size_t,
                                               Tensor<nvbf16> &, size_t, cudaStream_t);
template void compute_att_output<nvbf16>(const Tensor<nvbf16> &, const Tensor<nvbf16> &, size_t, size_t,
                                         Tensor<nvbf16> &, size_t, cudaStream_t);
template void compute_attention_scores_prefill<nvbf16>(const Tensor<nvbf16> &, const Tensor<nvbf16> &, Tensor<nvbf16> &,
                                                       size_t, cudaStream_t);
template void compute_att_output_prefill<nvbf16>(const Tensor<nvbf16> &, const Tensor<nvbf16> &, Tensor<nvbf16> &,
                                                 size_t, size_t, size_t, size_t, cudaStream_t);
}  // namespace cuda_OP
