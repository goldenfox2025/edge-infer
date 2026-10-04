#include <cmath>
#include <iostream>
#include <stdexcept>
#include <type_traits>

#include "operators/cuda/rope_cuda.cuh"

namespace op {

// --- CUDA Kernel ---
template <typename T, int actual_pairs_per_thread = 2>
__global__ void rope_kernel(T *tensor, size_t batch_size, size_t seq_len, size_t n_heads, size_t head_dim,
                            size_t offset, float theta) {
    // One block rotates one head at one sequence position in one batch item.
    size_t b_idx = blockIdx.x;
    size_t head_idx = blockIdx.y;
    size_t seq_idx_in_batch = blockIdx.z;

    size_t head_dim_half = head_dim / 2;  // Pair corresponding elements from the first and second halves.

    // Threads distribute rotation pairs within this head.
    size_t group_idx = threadIdx.x;
    size_t absolute_seq_pos = seq_idx_in_batch + offset;  // Absolute token position in the full sequence.

    T *current_head_ptr = tensor + b_idx * seq_len * n_heads * head_dim +
                          seq_idx_in_batch * n_heads * head_dim +
                          head_idx * head_dim;

    for (int i = 0; i < actual_pairs_per_thread; ++i) {

        size_t rot_dim = group_idx * actual_pairs_per_thread + i;

        if (rot_dim < head_dim_half) {

            float freq = 1.0f / powf(theta, (2.0f * rot_dim) / head_dim);
            float val = (float)absolute_seq_pos * freq;
            float cos_val = cosf(val);
            float sin_val = sinf(val);

            float x0 = static_cast<float>(current_head_ptr[rot_dim]);
            float x1 = static_cast<float>(current_head_ptr[rot_dim + head_dim_half]);

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

    // RoPE requires an even head dimension; empty tensors return without launching.
    if (batch_size == 0 || seq_len == 0 || n_heads == 0 || head_dim == 0)
        return;
    if (head_dim % 2 != 0) {
        throw std::runtime_error("RoPE: head_dim must be even.");
    }

    size_t head_dim_half = head_dim / 2;  // Number of pairs spanning the first and second halves.
    if (head_dim_half == 0)
        return;

    constexpr int actual_pairs_per_thread = 2;

    int threads_per_block_dim = (head_dim_half + actual_pairs_per_thread - 1) / actual_pairs_per_thread;

    if (threads_per_block_dim > 1024) {
        throw std::runtime_error(
            "RoPE: Calculated threads per block > 1024. Head dimension might be "
            "too large.");
    }

    // One block per (batch item, head, sequence position).
    dim3 grid_dim(batch_size, n_heads, seq_len);

    dim3 block_dim(threads_per_block_dim);

    rope_kernel<T, actual_pairs_per_thread>
        <<<grid_dim, block_dim, 0, stream>>>(x->data_ptr(), batch_size, seq_len, n_heads, head_dim, offset, theta);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error after RoPE kernel launch: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("RoPE CUDA kernel launch failed");
    }
}

template class RopeCUDAOperator<float>;
template class RopeCUDAOperator<__nv_bfloat16>;

}  // namespace op
