// Rotary position embedding with a device-resident position offset for graph replay.
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "cuda/legacy/legacy_cuda_api.cuh"

namespace cuda_OP {


template <typename T, int actual_pairs_per_thread = 2>
__global__ void rope_kernel_device_offset(T *tensor, size_t batch_size, size_t seq_len, size_t n_heads, size_t head_dim,
                                          const size_t *d_offset, float theta, int stride) {

    size_t offset = *d_offset;


    // size_t b_idx = blockIdx.x;
    size_t head_idx = blockIdx.y;
    size_t seq_idx_in_batch = blockIdx.z;


    size_t head_dim_half = head_dim / 2;


    size_t group_idx = threadIdx.x;

    size_t absolute_seq_pos = seq_idx_in_batch + offset;


    T *current_head_ptr = tensor + seq_idx_in_batch * stride +
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

    size_t batch_size = 1;
    size_t seq_len, n_heads, head_dim;
    if (sizes.size() == 3) {
        seq_len = sizes[0];
        n_heads = sizes[1];
        head_dim = sizes[2];
    } else {
        throw std::runtime_error("RoPE: Input tensor must be 3D (seq_len, n_heads, head_dim).");
    }


    if (batch_size == 0 || seq_len == 0 || n_heads == 0 || head_dim == 0)
        return;
    if (head_dim % 2 != 0) {
        throw std::runtime_error("RoPE: head_dim must be even.");
    }


    size_t head_dim_half = head_dim / 2;
    if (head_dim_half == 0)
        return;


    constexpr int actual_pairs_per_thread = 2;


    int threads_per_block_dim = (head_dim_half + actual_pairs_per_thread - 1) / actual_pairs_per_thread;

    if (threads_per_block_dim > 1024) {
        throw std::runtime_error(
            "RoPE: Calculated threads per block > 1024. Head dimension might be "
            "too large.");
    }


    dim3 grid_dim(batch_size, n_heads, seq_len);

    dim3 block_dim(threads_per_block_dim);


    int stride = tensor->strides()[0];
    rope_kernel_device_offset<T, actual_pairs_per_thread><<<grid_dim, block_dim, 0, stream>>>(
        tensor->data_ptr(), batch_size, seq_len, n_heads, head_dim, d_offset, theta, stride);


    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error after RoPE kernel launch: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("RoPE CUDA kernel launch failed");
    }
}


template <typename T, int actual_pairs_per_thread = 2>
__global__ void rope_kernel_precomputed_cache(T *tensor, size_t batch_size, size_t seq_len, size_t n_heads,
                                              size_t head_dim, const size_t *d_offset, const float *sin_cos_cache,
                                              size_t cache_stride, int stride, int *offset_array, int layer_index,
                                              int n_layers, int *pingpong_index) {
    size_t offset;


    if (pingpong_index != nullptr) {
        offset = d_offset[*pingpong_index];
    } else {
        offset = d_offset[0];
    }

    int t_off = 0;
    if (offset_array != nullptr) {
        t_off = offset_array[layer_index + n_layers * (*pingpong_index)];
    }

    // size_t b_idx = blockIdx.x;
    size_t head_idx = blockIdx.y;
    size_t seq_idx_in_batch = blockIdx.z;


    size_t head_dim_half = head_dim / 2;


    size_t group_idx = threadIdx.x;

    size_t absolute_seq_pos = seq_idx_in_batch + offset;
    T *current_head_ptr;


    current_head_ptr = tensor + t_off + seq_idx_in_batch * stride +
                       head_idx * head_dim;


    for (int i = 0; i < actual_pairs_per_thread; ++i) {

        size_t rot_dim = group_idx * actual_pairs_per_thread + i;


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


    size_t batch_size = 1;
    size_t seq_len, n_heads, head_dim;
    if (sizes.size() == 3) {
        seq_len = sizes[0];
        n_heads = sizes[1];
        head_dim = sizes[2];
    } else {
        throw std::runtime_error("RoPE: Input tensor must be 3D (seq_len, n_heads, head_dim).");
    }


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


    if (batch_size == 0 || seq_len == 0 || n_heads == 0 || head_dim == 0)
        return;
    if (head_dim % 2 != 0) {
        throw std::runtime_error("RoPE: head_dim must be even.");
    }

    size_t head_dim_half = head_dim / 2;
    if (head_dim_half == 0)
        return;


    constexpr int actual_pairs_per_thread = 2;
    int threads_per_block_dim = (head_dim_half + actual_pairs_per_thread - 1) / actual_pairs_per_thread;

    if (threads_per_block_dim > 1024) {
        throw std::runtime_error("RoPE: Calculated threads per block > 1024. Head dimension might be too large.");
    }


    dim3 grid_dim(batch_size, n_heads, seq_len);

    dim3 block_dim(threads_per_block_dim);


    rope_kernel_precomputed_cache<T, actual_pairs_per_thread><<<grid_dim, block_dim, 0, stream>>>(
        tensor->data_ptr(), batch_size, seq_len, n_heads, head_dim, d_offset, sin_cos_cache->data_ptr(), cache_head_dim,
        stride, offset_array, layer_index, n_layers, pingpong_index);


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
