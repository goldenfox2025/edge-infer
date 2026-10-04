// CUDA sampling kernels. top_k controls candidate selection; sequence and vocabulary dimensions must be nonzero.
#ifndef CUDA_OP_CUH
#define CUDA_OP_CUH

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <float.h>

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>


#include <cub/cub.cuh>


#include "CudaMemoryPool.hpp"
#include "cuda/legacy/legacy_cuda_api.cuh"
#include "tensor.hpp"


#define MAX_TOPK 1024


namespace cuda_OP {


template <typename T>
__global__ void scale_logits_and_init_indices_kernel(const T* __restrict__ logits,
                                                     T* d_scaled_logits,
                                                     int* d_indices,
                                                     size_t vocab_size,
                                                     float temperature
) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < vocab_size; i += stride) {

        float logit_f;

        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            logit_f = __bfloat162float(__ldg(&logits[i]));
        } else {
            logit_f = static_cast<float>(__ldg(&logits[i]));
        }
        float scaled_logit_f = logit_f / temperature;


        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            d_scaled_logits[i] = __float2bfloat16(scaled_logit_f);
        } else {
            d_scaled_logits[i] = static_cast<T>(scaled_logit_f);
        }


        d_indices[i] = i;
    }
}


template <typename T>
__global__ void scale_logits_to_float_and_init_indices_kernel(const T* __restrict__ logits,
                                                              float* __restrict__ scaled_logits_f,
                                                              int* __restrict__ indices, size_t vocab_size,
                                                              float temperature) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < static_cast<int>(vocab_size); i += stride) {
        float v = 0.0f;
        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            v = __bfloat162float(__ldg(&logits[i]));
        } else {
            v = static_cast<float>(__ldg(&logits[i]));
        }
        scaled_logits_f[i] = v / temperature;
        indices[i] = i;
    }
}


template <typename T, int BLOCK_DIM_X>
__global__ void sample_from_sorted_topk_kernel(
    const T* __restrict__ d_sorted_topk_logits,
    const int* __restrict__ d_sorted_topk_indices,
    size_t k,                                       // Top-K
    const float* __restrict__ d_max_val_ptr,
    curandState* states,
    uint32_t* d_sampled_index) {

    using BlockReduce = cub::BlockReduce<float, BLOCK_DIM_X>;


    __shared__ union SharedStorage {
        typename BlockReduce::TempStorage reduce_storage;

        struct Combined {
            typename BlockReduce::TempStorage reduce_storage;
            float exp_vals[MAX_TOPK];
        } combined;
    } shared_storage;

    int tid = threadIdx.x;


    __shared__ float max_val_shared;
    if (tid == 0) {
        max_val_shared = *d_max_val_ptr;
    }
    __syncthreads();


    float thread_exp_sum = 0.0f;

    for (int i = tid; i < k; i += BLOCK_DIM_X) {
        float scaled_logit_f;
        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            scaled_logit_f = __bfloat162float(d_sorted_topk_logits[i]);
        } else {
            scaled_logit_f = static_cast<float>(d_sorted_topk_logits[i]);
        }

        float exp_val = expf(scaled_logit_f - max_val_shared);


        if (i < MAX_TOPK) {
            shared_storage.combined.exp_vals[i] = exp_val;
        }

        thread_exp_sum += exp_val;
    }
    __syncthreads();


    float block_total_exp_sum = BlockReduce(shared_storage.combined.reduce_storage).Sum(thread_exp_sum);


    if (tid == 0) {
        float total_exp_sum = block_total_exp_sum;
        curandState localState = states[0];

        uint32_t selected_final_index = 0;


        if (total_exp_sum <= 1e-9f || k == 0) {
            if (k > 0) {
                selected_final_index = static_cast<uint32_t>(d_sorted_topk_indices[0]);
            } else {
                selected_final_index = 0;
            }
        } else {

            float r = curand_uniform(&localState) * total_exp_sum;
            float cumulative = 0.0f;


            selected_final_index = static_cast<uint32_t>(d_sorted_topk_indices[0]);
            float* s_exp_vals = shared_storage.combined.exp_vals;
            for (int i = 0; i < k; ++i) {

                cumulative += s_exp_vals[i];

                if (cumulative >= r) {
                    selected_final_index = static_cast<uint32_t>(d_sorted_topk_indices[i]);
                    break;
                }
            }
        }

        *d_sampled_index = selected_final_index;

        states[0] = localState;
    }
}


template <typename Tin>
struct ConvertToFloatFunctor {
    __device__ __forceinline__ float operator()(const Tin& x) const {
        if constexpr (std::is_same_v<Tin, __nv_bfloat16>) {
            return __bfloat162float(x);
        } else {
            return static_cast<float>(x);
        }
    }
};


template <typename T>
uint32_t* sample(Tensor<T>&& logits, float temperature,
                 float top_p,
                 size_t top_k, curandState* d_states, cudaStream_t stream) {

    if (logits.device() != Device::CUDA) {
        throw std::runtime_error("Input tensor must be on a CUDA device");
    }

    if (top_k == 0) {
        throw std::runtime_error("top_k must be at least 1");
    }

    const auto& shape = logits.sizes();
    if (shape.size() != 2 || shape[0] == 0 || shape[1] == 0) {
        throw std::runtime_error("Input tensor must be 2D with nonzero dimensions [seq_len, vocab_size]");
    }

    const size_t seq_len = shape[0];
    const size_t vocab_size = shape[1];


    if (top_k > vocab_size) {
        top_k = vocab_size;
    }

    if (top_k > MAX_TOPK) {
        throw std::runtime_error("Requested top_k (" + std::to_string(top_k) + ") exceeds Kernel 2 MAX_TOPK (" +
                                 std::to_string(MAX_TOPK) + ") and cannot fit in shared memory");
    }


    const T* d_logits_ptr = logits.data_ptr() + (seq_len - 1) * vocab_size;


    auto& pool = GlobalCudaMemoryPool::instance();

    T* d_scaled_logits = static_cast<T*>(pool.allocate(vocab_size * sizeof(T)));
    float* d_max_val = static_cast<float*>(pool.allocate(sizeof(float)));
    int* d_indices = static_cast<int*>(pool.allocate(vocab_size * sizeof(int)));
    T* d_sorted_logits = static_cast<T*>(pool.allocate(vocab_size * sizeof(T)));
    int* d_sorted_indices = static_cast<int*>(pool.allocate(vocab_size * sizeof(int)));


    uint32_t* d_sampled_index = static_cast<uint32_t*>(pool.allocate_tagged("graph_input_token", sizeof(uint32_t)));


    void* d_reduce_temp_storage = nullptr;
    size_t reduce_temp_storage_bytes = 0;
    void* d_sort_temp_storage = nullptr;
    size_t sort_temp_storage_bytes = 0;


    const int scale_init_block_size = 256;
    const int scale_init_grid_size =
        (vocab_size + scale_init_block_size - 1) / scale_init_block_size;


    scale_logits_and_init_indices_kernel<T><<<scale_init_grid_size, scale_init_block_size, 0, stream>>>(
        d_logits_ptr, d_scaled_logits, d_indices, vocab_size, temperature);
    CUDA_CHECK(cudaGetLastError());


    cub::TransformInputIterator<float, ConvertToFloatFunctor<T>, const T*> itr(d_scaled_logits,
                                                                               ConvertToFloatFunctor<T>());
    CUDA_CHECK(
        cub::DeviceReduce::Max(d_reduce_temp_storage, reduce_temp_storage_bytes, itr, d_max_val, vocab_size, stream));

    d_reduce_temp_storage = pool.allocate(reduce_temp_storage_bytes);

    CUDA_CHECK(
        cub::DeviceReduce::Max(d_reduce_temp_storage, reduce_temp_storage_bytes, itr, d_max_val, vocab_size, stream));


    CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(d_sort_temp_storage, sort_temp_storage_bytes, d_scaled_logits,
                                                         d_sorted_logits, d_indices, d_sorted_indices, vocab_size, 0,
                                                         sizeof(T) * 8, stream));

    d_sort_temp_storage = pool.allocate(sort_temp_storage_bytes);

    CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(d_sort_temp_storage, sort_temp_storage_bytes, d_scaled_logits,
                                                         d_sorted_logits, d_indices, d_sorted_indices, vocab_size, 0,
                                                         sizeof(T) * 8, stream));


    const int sample_block_size = 128;


    size_t reduce_storage_size_est = sizeof(cub::BlockReduce<float,
                                                             sample_block_size>::TempStorage);
    size_t exp_values_size = MAX_TOPK * sizeof(float);
    size_t sample_shared_mem = reduce_storage_size_est + exp_values_size;


    sample_from_sorted_topk_kernel<T, sample_block_size><<<1, sample_block_size, sample_shared_mem, stream>>>(
        d_sorted_logits,
        d_sorted_indices,
        top_k,
        d_max_val,
        d_states,
        d_sampled_index
    );
    CUDA_CHECK(cudaGetLastError());


    pool.free(d_scaled_logits);
    pool.free(d_max_val);
    pool.free(d_indices);
    pool.free(d_sorted_logits);
    pool.free(d_sorted_indices);

    pool.free(d_reduce_temp_storage);
    pool.free(d_sort_temp_storage);


    return d_sampled_index;
}


template <typename T>
void fast_sample_to_fixed(Tensor<T>&& logits, uint32_t* output_ptr, float* /*prob_ptr*/, float temperature,
                          float /*top_p*/, size_t top_k, curandState* d_states, cudaStream_t stream) {
    if (logits.device() != Device::CUDA) {
        throw std::runtime_error("Input tensor must be on a CUDA device");
    }

    const auto& shape = logits.sizes();
    if (shape.size() != 2 || shape[0] != 1) {
        throw std::runtime_error("Input tensor must have shape [1, vocab_size]");
    }

    const size_t vocab_size = shape[1];
    top_k = std::min(top_k, vocab_size);

    const T* d_logits_ptr = logits.data_ptr();

    auto& pool = GlobalCudaMemoryPool::instance();
    T* d_scaled_logits = static_cast<T*>(pool.allocate(vocab_size * sizeof(T)));
    float* d_max_val = static_cast<float*>(pool.allocate(sizeof(float)));
    int* d_indices = static_cast<int*>(pool.allocate(vocab_size * sizeof(int)));
    T* d_sorted_logits = static_cast<T*>(pool.allocate(vocab_size * sizeof(T)));
    int* d_sorted_indices = static_cast<int*>(pool.allocate(vocab_size * sizeof(int)));

    void* d_reduce_temp_storage = nullptr;
    size_t reduce_temp_storage_bytes = 0;
    void* d_sort_temp_storage = nullptr;
    size_t sort_temp_storage_bytes = 0;


    const int threads = 256;
    const int blocks = static_cast<int>((vocab_size + threads - 1) / threads);
    scale_logits_and_init_indices_kernel<T>
        <<<blocks, threads, 0, stream>>>(d_logits_ptr, d_scaled_logits, d_indices, vocab_size, temperature);
    CUDA_CHECK(cudaGetLastError());

    // 2) Max
    cub::TransformInputIterator<float, ConvertToFloatFunctor<T>, const T*> itr2(d_scaled_logits,
                                                                                ConvertToFloatFunctor<T>());
    CUDA_CHECK(
        cub::DeviceReduce::Max(d_reduce_temp_storage, reduce_temp_storage_bytes, itr2, d_max_val, vocab_size, stream));
    d_reduce_temp_storage = pool.allocate(reduce_temp_storage_bytes);
    CUDA_CHECK(
        cub::DeviceReduce::Max(d_reduce_temp_storage, reduce_temp_storage_bytes, itr2, d_max_val, vocab_size, stream));

    // 3) SortPairsDescending (float keys)
    CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(d_sort_temp_storage, sort_temp_storage_bytes, d_scaled_logits,
                                                         d_sorted_logits, d_indices, d_sorted_indices, vocab_size, 0,
                                                         sizeof(T) * 8, stream));
    d_sort_temp_storage = pool.allocate(sort_temp_storage_bytes);
    CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(d_sort_temp_storage, sort_temp_storage_bytes, d_scaled_logits,
                                                         d_sorted_logits, d_indices, d_sorted_indices, vocab_size, 0,
                                                         sizeof(T) * 8, stream));


    const int sample_block_size = 128;
    size_t sample_shared_mem =
        sizeof(cub::BlockReduce<float, sample_block_size>::TempStorage) + MAX_TOPK * sizeof(float);
    sample_from_sorted_topk_kernel<T, sample_block_size><<<1, sample_block_size, sample_shared_mem, stream>>>(
        d_sorted_logits, d_sorted_indices, top_k, d_max_val, d_states, output_ptr);
    CUDA_CHECK(cudaGetLastError());

    pool.free(d_scaled_logits);
    pool.free(d_max_val);
    pool.free(d_indices);
    pool.free(d_sorted_logits);
    pool.free(d_sorted_indices);
    pool.free(d_reduce_temp_storage);
    pool.free(d_sort_temp_storage);
}

template <typename T>
__global__ void sample_batch_parallel_kernel(
    const T* __restrict__ logits_data,
    uint32_t* output_ptr,
    size_t seq_len,
    size_t vocab_size,
    float temperature,
    size_t top_k,
    curandState* d_states
) {

    int seq_idx = blockIdx.x;
    if (seq_idx >= seq_len)
        return;

    int tid = threadIdx.x;
    int block_size = blockDim.x;


    extern __shared__ float shared_mem[];
    float* s_scaled_logits = shared_mem;


    const T* seq_logits = logits_data + seq_idx * vocab_size;


    float max_val = -FLT_MAX;


    for (int i = tid; i < vocab_size; i += block_size) {
        float logit_f;
        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            logit_f = __bfloat162float(seq_logits[i]);
        } else {
            logit_f = static_cast<float>(seq_logits[i]);
        }
        s_scaled_logits[i] = logit_f / temperature;
        max_val = fmaxf(max_val, s_scaled_logits[i]);
    }


    __shared__ float s_max_val;
    __syncthreads();


    for (int stride = block_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            max_val = fmaxf(max_val, __shfl_down_sync(0xFFFFFFFF, max_val, stride));
        }
    }
    if (tid == 0) {
        s_max_val = max_val;
    }
    __syncthreads();


    float exp_sum = 0.0f;


    for (int i = tid; i < vocab_size; i += block_size) {
        float exp_val = expf(s_scaled_logits[i] - s_max_val);
        s_scaled_logits[i] = exp_val;
        exp_sum += exp_val;
    }


    for (int stride = block_size / 2; stride > 0; stride /= 2) {
        exp_sum += __shfl_down_sync(0xFFFFFFFF, exp_sum, stride);
    }

    __shared__ float s_exp_sum;
    if (tid == 0) {
        s_exp_sum = exp_sum;
    }
    __syncthreads();


    if (tid == 0) {
        curandState local_state = d_states[seq_idx % 1];

        float r = curand_uniform(&local_state) * s_exp_sum;
        float cumulative = 0.0f;
        uint32_t selected_idx = 0;


        for (int i = 0; i < vocab_size; ++i) {
            cumulative += s_scaled_logits[i];
            if (cumulative >= r) {
                selected_idx = i;
                break;
            }
        }

        output_ptr[seq_idx] = selected_idx;
        d_states[seq_idx % 1] = local_state;
    }
}


template <typename T>
void sample_batch_to_fixed(Tensor<T>&& logits, uint32_t* output_ptr, float temperature, float top_p, size_t top_k,
                           curandState* d_states, cudaStream_t stream) {
    if (logits.device() != Device::CUDA) {
        throw std::runtime_error("Input tensor must be on a CUDA device");
    }

    if (top_k == 0) {
        throw std::runtime_error("top_k must be at least 1");
    }

    const auto& shape = logits.sizes();
    if (shape.size() != 2 || shape[0] == 0 || shape[1] == 0) {
        throw std::runtime_error("Input tensor must be 2D with nonzero dimensions [seq_len, vocab_size]");
    }

    const size_t seq_len = shape[0];
    const size_t vocab_size = shape[1];


    for (size_t i = 0; i < seq_len; i++) {

        std::vector<size_t> start = {i, 0};
        std::vector<size_t> end = {i + 1, vocab_size};
        Tensor<T> logit_view = logits.slice(start, end);


        sample_to_fixed(std::move(logit_view), output_ptr + i, temperature, top_p, top_k, d_states, stream);
    }
}


template <typename T>
void sample_to_fixed(Tensor<T>&& logits, uint32_t* output_ptr, float temperature, float top_p, size_t top_k,
                     curandState* d_states, cudaStream_t stream) {

    fast_sample_to_fixed(std::move(logits), output_ptr, nullptr, temperature, top_p, top_k, d_states, stream);
}

template uint32_t* sample<float>(Tensor<float>&&, float, float, size_t, curandState*, cudaStream_t);
template uint32_t* sample<__nv_bfloat16>(Tensor<__nv_bfloat16>&&, float, float, size_t, curandState*, cudaStream_t);

template void sample_to_fixed<float>(Tensor<float>&&, uint32_t*, float, float, size_t, curandState*, cudaStream_t);
template void sample_to_fixed<__nv_bfloat16>(Tensor<__nv_bfloat16>&&, uint32_t*, float, float, size_t, curandState*,
                                             cudaStream_t);

template void sample_batch_to_fixed<float>(Tensor<float>&&, uint32_t*, float, float, size_t, curandState*,
                                           cudaStream_t);
template void sample_batch_to_fixed<__nv_bfloat16>(Tensor<__nv_bfloat16>&&, uint32_t*, float, float, size_t,
                                                   curandState*, cudaStream_t);

template void fast_sample_to_fixed<float>(Tensor<float>&&, uint32_t*, float*, float, float, size_t, curandState*,
                                          cudaStream_t);
template void fast_sample_to_fixed<__nv_bfloat16>(Tensor<__nv_bfloat16>&&, uint32_t*, float*, float, float, size_t,
                                                  curandState*, cudaStream_t);

}  // namespace cuda_OP

#endif  // CUDA_OP_CUH