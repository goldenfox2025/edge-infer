#include <cmath>
#include <iostream>
#include <stdexcept>

#include "cuda/legacy/legacy_cuda_api.cuh"

namespace cuda_OP {

// 每个线程负责处理N对元素
template <typename T, int actual_pairs_per_thread = 2>
__global__ void rope_kernel_device_offset(T *tensor, size_t batch_size, size_t seq_len, size_t n_heads, size_t head_dim,
                                          const size_t *d_offset, float theta, int stride) {
    // 从设备内存读取offset
    size_t offset = *d_offset;

    // 确认当前线程的位置
    // size_t b_idx = blockIdx.x;
    size_t head_idx = blockIdx.y;
    size_t seq_idx_in_batch = blockIdx.z;

    // RoPE作用于向量的前一半和后一半
    size_t head_dim_half = head_dim / 2;

    // 当前线程的职责
    // threadIdx.x 标记这个线程是处理当前头向量中的"第几组"旋转对
    size_t group_idx = threadIdx.x;
    // Token在原始完整序列中的绝对位置
    size_t absolute_seq_pos = seq_idx_in_batch + offset;

    // 输入的顺序是 seqlen, n_heads, head_dim
    T *current_head_ptr = tensor + seq_idx_in_batch * stride +  // 序列位置偏移（使用stride）
                          head_idx * head_dim;                  // 头偏移

    // 执行旋转
    // 每个线程根据 actual_pairs_per_thread 循环处理它负责的旋转对
    for (int i = 0; i < actual_pairs_per_thread; ++i) {
        // 当前线程要处理的旋转维度索引 (相对于 head_dim_half)
        size_t rot_dim = group_idx * actual_pairs_per_thread + i;

        // 边界检查：确保 rot_dim 没有超出当前头向量的一半长度
        if (rot_dim < head_dim_half) {
            // 计算 RoPE 频率和角度 (与原始逻辑相同)
            float freq = 1.0f / powf(theta, (2.0f * rot_dim) / head_dim);
            float val = (float)absolute_seq_pos * freq;
            float cos_val = cosf(val);
            float sin_val = sinf(val);

            // 取出要旋转的一对数
            float x0 = static_cast<float>(current_head_ptr[rot_dim]);
            float x1 = static_cast<float>(current_head_ptr[rot_dim + head_dim_half]);

            // 执行旋转并写回
            current_head_ptr[rot_dim] = static_cast<T>(x0 * cos_val - x1 * sin_val);
            current_head_ptr[rot_dim + head_dim_half] = static_cast<T>(x0 * sin_val + x1 * cos_val);
        }
    }
}

// CUDA图优化版本：使用设备端固定内存的offset
template <typename T>
void rope_with_device_offset(Tensor<T> *tensor, const size_t *d_offset, float theta, cudaStream_t stream) {
    if (tensor->device() != Device::CUDA) {
        throw std::runtime_error("RoPE: Input tensor must be on CUDA device.");
    }
    if (d_offset == nullptr) {
        throw std::runtime_error("RoPE: Device offset pointer cannot be null.");
    }

    const auto &sizes = tensor->sizes();
    if (sizes.size() < 3) {
        throw std::runtime_error("RoPE: Input tensor needs at least 3D (seq_len, n_heads, head_dim).");
    }
    // 默认为1
    size_t batch_size = 1;
    size_t seq_len, n_heads, head_dim;
    if (sizes.size() == 3) {
        seq_len = sizes[0];
        n_heads = sizes[1];
        head_dim = sizes[2];
    } else {
        throw std::runtime_error("RoPE: Input tensor must be 3D (seq_len, n_heads, head_dim).");
    }

    // 如果任何维度为0，或头维度不是偶数，则提前返回或报错
    if (batch_size == 0 || seq_len == 0 || n_heads == 0 || head_dim == 0)
        return;
    if (head_dim % 2 != 0) {
        throw std::runtime_error("RoPE: head_dim must be even.");
    }

    // RoPE作用于向量的一半，总共有这么多"对"
    size_t head_dim_half = head_dim / 2;
    if (head_dim_half == 0)
        return;  // 没有可旋转的维度对

    // 根据数据类型T，决定每个线程处理多少"对"元素
    constexpr int actual_pairs_per_thread = 2;

    // 一个Block需要多少线程来处理这 head_dim_half 对元素？
    // 使用向上取整的整数除法
    int threads_per_block_dim = (head_dim_half + actual_pairs_per_thread - 1) / actual_pairs_per_thread;

    if (threads_per_block_dim > 1024) {
        throw std::runtime_error(
            "RoPE: Calculated threads per block > 1024. Head dimension might be "
            "too large.");
    }

    // Grid维度：每个(样本, 头, 序列位置) 对应一个Block
    dim3 grid_dim(batch_size, n_heads, seq_len);
    // Block维度：刚才计算出的、用于处理一个头内部所有旋转对的线程数
    dim3 block_dim(threads_per_block_dim);

    // 启动Kernel - 添加stride支持
    int stride = tensor->strides()[0];
    rope_kernel_device_offset<T, actual_pairs_per_thread><<<grid_dim, block_dim, 0, stream>>>(
        tensor->data_ptr(), batch_size, seq_len, n_heads, head_dim, d_offset, theta, stride);

    // 错误检查
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error after RoPE kernel launch: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("RoPE CUDA kernel launch failed");
    }
}

// 使用预计算sin/cos缓存的RoPE kernel
template <typename T, int actual_pairs_per_thread = 2>
__global__ void rope_kernel_precomputed_cache(T *tensor, size_t batch_size, size_t seq_len, size_t n_heads,
                                              size_t head_dim, const size_t *d_offset, const float *sin_cos_cache,
                                              size_t cache_stride, int stride, int *offset_array, int layer_index,
                                              int n_layers, int *pingpong_index) {
    size_t offset;

    // 从设备内存读取offset
    if (pingpong_index != nullptr) {
        offset = d_offset[*pingpong_index];
    } else {
        offset = d_offset[0];
    }

    int t_off = 0;
    if (offset_array != nullptr) {
        t_off = offset_array[layer_index + n_layers * (*pingpong_index)];
    }
    // 确认当前线程的位置
    // size_t b_idx = blockIdx.x;
    size_t head_idx = blockIdx.y;
    size_t seq_idx_in_batch = blockIdx.z;

    // RoPE作用于向量的前一半和后一半
    size_t head_dim_half = head_dim / 2;

    // 当前线程的职责
    size_t group_idx = threadIdx.x;
    // Token在原始完整序列中的绝对位置
    size_t absolute_seq_pos = seq_idx_in_batch + offset;
    T *current_head_ptr;
    // 输入的顺序是 seqlen, n_heads, head_dim

    current_head_ptr = tensor + t_off + seq_idx_in_batch * stride +  // 序列位置偏移（使用stride）
                       head_idx * head_dim;                          // 头偏移

    // 执行旋转
    for (int i = 0; i < actual_pairs_per_thread; ++i) {
        // 当前线程要处理的旋转维度索引 (相对于 head_dim_half)
        size_t rot_dim = group_idx * actual_pairs_per_thread + i;

        // 边界检查：确保 rot_dim 没有超出当前头向量的一半长度
        if (rot_dim < head_dim_half) {
            size_t cache_idx = absolute_seq_pos * cache_stride + rot_dim * 2;
            float2 sincos = *reinterpret_cast<const float2 *>(sin_cos_cache + cache_idx);

            float x0 = static_cast<float>(current_head_ptr[rot_dim]);
            float x1 = static_cast<float>(current_head_ptr[rot_dim + head_dim_half]);
            current_head_ptr[rot_dim] = static_cast<T>(x0 * sincos.y - x1 * sincos.x);
            current_head_ptr[rot_dim + head_dim_half] = static_cast<T>(x0 * sincos.x + x1 * sincos.y);
        }
    }
}

// 使用预计算sin/cos缓存的RoPE版本
template <typename T>
void rope_with_precomputed_cache(Tensor<T> *tensor, const size_t *d_offset, const Tensor<float> *sin_cos_cache,
                                 cudaStream_t stream, int *offset_array, int layer_index, int n_layers,
                                 int *pingpong_index) {
    if (tensor->device() != Device::CUDA) {
        throw std::runtime_error("RoPE: Input tensor must be on CUDA device.");
    }
    if (d_offset == nullptr) {
        throw std::runtime_error("RoPE: Device offset pointer cannot be null.");
    }
    if (sin_cos_cache == nullptr || sin_cos_cache->device() != Device::CUDA) {
        throw std::runtime_error("RoPE: sin_cos_cache must be on CUDA device.");
    }

    const auto &sizes = tensor->sizes();
    if (sizes.size() < 3) {
        throw std::runtime_error("RoPE: Input tensor needs at least 3D (seq_len, n_heads, head_dim).");
    }

    // 解析张量维度
    size_t batch_size = 1;
    size_t seq_len, n_heads, head_dim;
    if (sizes.size() == 3) {
        seq_len = sizes[0];
        n_heads = sizes[1];
        head_dim = sizes[2];
    } else {
        throw std::runtime_error("RoPE: Input tensor must be 3D (seq_len, n_heads, head_dim).");
    }

    // 验证缓存维度
    const auto &cache_sizes = sin_cos_cache->sizes();
    if (cache_sizes.size() != 2) {
        throw std::runtime_error("RoPE: sin_cos_cache must be 2D (max_seq_len, head_dim).");
    }

    size_t cache_max_seq_len = cache_sizes[0];
    size_t cache_head_dim = cache_sizes[1];
    const int stride = tensor->strides()[0];

    if (cache_head_dim != head_dim) {
        throw std::runtime_error("RoPE: sin_cos_cache head_dim mismatch.");
    }

    // 基本检查
    if (batch_size == 0 || seq_len == 0 || n_heads == 0 || head_dim == 0)
        return;
    if (head_dim % 2 != 0) {
        throw std::runtime_error("RoPE: head_dim must be even.");
    }

    size_t head_dim_half = head_dim / 2;
    if (head_dim_half == 0)
        return;

    // 线程配置
    constexpr int actual_pairs_per_thread = 2;
    int threads_per_block_dim = (head_dim_half + actual_pairs_per_thread - 1) / actual_pairs_per_thread;

    if (threads_per_block_dim > 1024) {
        throw std::runtime_error("RoPE: Calculated threads per block > 1024. Head dimension might be too large.");
    }

    // Grid维度：每个(样本, 头, 序列位置) 对应一个Block
    dim3 grid_dim(batch_size, n_heads, seq_len);
    // Block维度：用于处理一个头内部所有旋转对的线程数
    dim3 block_dim(threads_per_block_dim);

    // 启动Kernel
    rope_kernel_precomputed_cache<T, actual_pairs_per_thread><<<grid_dim, block_dim, 0, stream>>>(
        tensor->data_ptr(), batch_size, seq_len, n_heads, head_dim, d_offset, sin_cos_cache->data_ptr(), cache_head_dim,
        stride, offset_array, layer_index, n_layers, pingpong_index);

    // 错误检查
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error after RoPE precomputed cache kernel launch: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("RoPE precomputed cache CUDA kernel launch failed");
    }
}

template void rope_with_device_offset<float>(Tensor<float> *tensor, const size_t *d_offset, float theta,
                                             cudaStream_t stream);
template void rope_with_device_offset<__nv_bfloat16>(Tensor<__nv_bfloat16> *tensor, const size_t *d_offset, float theta,
                                                     cudaStream_t stream);
template void rope_with_precomputed_cache<float>(Tensor<float> *tensor, const size_t *d_offset,
                                                 const Tensor<float> *sin_cos_cache, cudaStream_t stream, int *, int,
                                                 int, int *);
template void rope_with_precomputed_cache<__nv_bfloat16>(Tensor<__nv_bfloat16> *tensor, const size_t *d_offset,
                                                         const Tensor<float> *sin_cos_cache, cudaStream_t stream, int *,
                                                         int, int, int *);
}  // namespace cuda_OP
