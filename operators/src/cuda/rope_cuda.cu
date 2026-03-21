#include <cmath>
#include <iostream>
#include <stdexcept>
#include <type_traits>  // 需要包含这个来使用 std::is_same_v

#include "operators/cuda/rope_cuda.cuh"

namespace op {

// --- CUDA Kernel ---
template <typename T, int actual_pairs_per_thread = 2>
__global__ void rope_kernel(T *tensor, size_t batch_size, size_t seq_len, size_t n_heads, size_t head_dim,
                            size_t offset, float theta) {
    // 每个Block负责处理一个样本(b_idx)中一个头(head_idx)在一个序列位置(seq_idx_in_batch)的旋转操作。
    size_t b_idx = blockIdx.x;
    size_t head_idx = blockIdx.y;
    size_t seq_idx_in_batch = blockIdx.z;

    size_t head_dim_half = head_dim / 2;  // RoPE作用于向量的前一半和后一半

    // threadIdx.x 标记这个线程是处理当前头向量中的“第几组”旋转对
    size_t group_idx = threadIdx.x;
    size_t absolute_seq_pos = seq_idx_in_batch + offset;  // Token在原始完整序列中的绝对位置

    // 计算指向当前 (样本, 位置, 头) 的数据指针
    T *current_head_ptr = tensor + b_idx * seq_len * n_heads * head_dim +  // 批次偏移
                          seq_idx_in_batch * n_heads * head_dim +          // 序列内位置偏移
                          head_idx * head_dim;                             // 头偏移

    // 每个线程根据 actual_pairs_per_thread 循环处理它负责的旋转对
    for (int i = 0; i < actual_pairs_per_thread; ++i) {
        // 当前线程要处理的旋转维度索引 (相对于 head_dim_half)
        size_t rot_dim = group_idx * actual_pairs_per_thread + i;

        // 边界检查：确保 rot_dim 没有超出当前头向量的一半长度
        // (Block的线程数是根据 head_dim_half 和 actual_pairs_per_thread
        // 精确计算的，
        //  所以这里主要是为了处理 head_dim_half 不是 actual_pairs_per_thread
        //  整数倍的尾部情况)
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

template <typename T>
void RopeCUDAOperator<T>::operator()(Tensor<T> *x, size_t offset, float theta, cudaStream_t stream) {
    if (x->device() != Device::CUDA) {
        throw std::runtime_error("RoPE: Input tensor must be on CUDA device.");
    }
    const auto &sizes = x->sizes();
    if (sizes.size() < 3) {
        throw std::runtime_error("RoPE: Input tensor needs at least 3D (seq_len, n_heads, head_dim).");
    }
    size_t batch_size = 1;
    size_t seq_len, n_heads, head_dim;
    if (sizes.size() == 3) {
        seq_len = sizes[0];
        n_heads = sizes[1];
        head_dim = sizes[2];
    } else {
        for (size_t i = 0; i < sizes.size() - 3; ++i) {
            batch_size *= sizes[i];
        }
        seq_len = sizes[sizes.size() - 3];
        n_heads = sizes[sizes.size() - 2];
        head_dim = sizes[sizes.size() - 1];
    }

    // 如果任何维度为0，或头维度不是偶数，则提前返回或报错
    if (batch_size == 0 || seq_len == 0 || n_heads == 0 || head_dim == 0)
        return;
    if (head_dim % 2 != 0) {
        throw std::runtime_error("RoPE: head_dim must be even.");
    }

    // 计算Kernel启动配置
    size_t head_dim_half = head_dim / 2;  // RoPE作用于向量的一半，总共有这么多“对”
    if (head_dim_half == 0)
        return;  // 没有可旋转的维度对

    // 根据数据类型T，决定每个线程处理多少“对”元素
    constexpr int actual_pairs_per_thread = 2;

    // 简化计算：一个Block需要多少线程来处理这 head_dim_half 对元素？
    // 使用向上取整的整数除法: (numerator + denominator - 1) / denominator
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

    // 启动Kernel
    rope_kernel<T, actual_pairs_per_thread>
        <<<grid_dim, block_dim, 0, stream>>>(x->data_ptr(), batch_size, seq_len, n_heads, head_dim, offset, theta);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error after RoPE kernel launch: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("RoPE CUDA kernel launch failed");
    }
}

// 显式模板实例化
template class RopeCUDAOperator<float>;
template class RopeCUDAOperator<__nv_bfloat16>;
// template class RopeCUDAOperator<__half>; // 如果需要支持半精度

}  // namespace op