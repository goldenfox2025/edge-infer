// Sampling with probability outputs for speculative decoding. These kernels do not establish rejection/resampling correctness.
#ifndef CUDA_OP_SAMPLE_WITH_PROB_CUH
#define CUDA_OP_SAMPLE_WITH_PROB_CUH

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


__global__ void generate_random_values_kernel(float* values, size_t count, curandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < count; i += stride) {

        if (idx == 0) {
            curandState localState = states[0];
            for (int j = 0; j < count; j++) {
                values[j] = curand_uniform(&localState);
            }
            states[0] = localState;
            break;
        }
    }
}


void generate_random_values(float* values, size_t count, curandState* states, cudaStream_t stream) {
    const int block_size = 256;
    const int grid_size = (count + block_size - 1) / block_size;

    generate_random_values_kernel<<<grid_size, block_size, 0, stream>>>(values, count, states);
    CUDA_CHECK(cudaGetLastError());
}


template <typename T>
__global__ void get_token_probability_kernel(const T* logits, int position, uint32_t token_id, float* output_prob,
                                             size_t vocab_size, float temperature) {

    if (threadIdx.x == 0 && blockIdx.x == 0) {

        const T* pos_logits = logits + position * vocab_size;


        float max_val = -FLT_MAX;
        for (size_t i = 0; i < vocab_size; i++) {
            float val;
            if constexpr (std::is_same_v<T, __nv_bfloat16>) {
                val = __bfloat162float(pos_logits[i]) / temperature;
            } else {
                val = static_cast<float>(pos_logits[i]) / temperature;
            }
            max_val = fmaxf(max_val, val);
        }


        float sum_exp = 0.0f;
        for (size_t i = 0; i < vocab_size; i++) {
            float val;
            if constexpr (std::is_same_v<T, __nv_bfloat16>) {
                val = __bfloat162float(pos_logits[i]) / temperature;
            } else {
                val = static_cast<float>(pos_logits[i]) / temperature;
            }
            sum_exp += expf(val - max_val);
        }


        float token_val;
        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            token_val = __bfloat162float(pos_logits[token_id]) / temperature;
        } else {
            token_val = static_cast<float>(pos_logits[token_id]) / temperature;
        }


        *output_prob = expf(token_val - max_val) / sum_exp;
    }
}


template <typename T>
float get_token_probability(const Tensor<T>& logits, int position, uint32_t token_id, cudaStream_t stream) {
    if (logits.device() != Device::CUDA) {
        throw std::runtime_error("Input tensor must be on a CUDA device");
    }

    const auto& shape = logits.sizes();
    if (shape.size() != 2) {
        throw std::runtime_error("Input tensor must be 2D [seq_len, vocab_size]");
    }

    size_t seq_len = shape[0];
    size_t vocab_size = shape[1];

    if (position < 0 || position >= seq_len) {
        throw std::runtime_error("Position is out of range");
    }

    if (token_id >= vocab_size) {
        throw std::runtime_error("token_id is outside the vocabulary");
    }


    float* d_prob;
    cudaMalloc(&d_prob, sizeof(float));


    get_token_probability_kernel<<<1, 1, 0, stream>>>(logits.data_ptr(), position, token_id, d_prob, vocab_size, 1.0f);
    CUDA_CHECK(cudaGetLastError());


    float h_prob;
    cudaMemcpyAsync(&h_prob, d_prob, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);


    cudaFree(d_prob);

    return h_prob;
}


template <int BLOCK_DIM_X>
__global__ void sample_from_sorted_topk_with_prob_kernel(const float* __restrict__ d_sorted_topk_logits,
                                                         const int* __restrict__ d_sorted_topk_indices, size_t k,
                                                         const float* __restrict__ d_max_val_ptr, curandState* states,
                                                         uint32_t* d_sampled_index, float* d_sampled_prob) {

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
        float scaled_logit_f = d_sorted_topk_logits[i];

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
        float selected_prob = 0.0f;

        if (total_exp_sum <= 1e-9f || k == 0) {
            if (k > 0) {
                selected_final_index = static_cast<uint32_t>(d_sorted_topk_indices[0]);
                selected_prob = 1.0f;
            } else {
                selected_final_index = 0;
                selected_prob = 1.0f;
            }
        } else {
            float r = curand_uniform(&localState) * total_exp_sum;
            float cumulative = 0.0f;

            selected_final_index = static_cast<uint32_t>(d_sorted_topk_indices[0]);
            float* s_exp_vals = shared_storage.combined.exp_vals;
            selected_prob = s_exp_vals[0] / total_exp_sum;
            for (int i = 0; i < k; ++i) {
                cumulative += s_exp_vals[i];

                if (cumulative >= r) {
                    selected_final_index = static_cast<uint32_t>(d_sorted_topk_indices[i]);

                    selected_prob = s_exp_vals[i] / total_exp_sum;
                    break;
                }
            }
        }


        *d_sampled_index = selected_final_index;
        *d_sampled_prob = selected_prob;


        states[0] = localState;
    }
}


template <typename T>
std::pair<uint32_t, float> sample_with_prob(Tensor<T>&& logits, float temperature, float top_p, size_t top_k,
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

    if (top_k > vocab_size) {
        top_k = vocab_size;
    }

    if (top_k > MAX_TOPK) {
        throw std::runtime_error("Requested top_k (" + std::to_string(top_k) + ") exceeds MAX_TOPK (" +
                                 std::to_string(MAX_TOPK) + ")");
    }


    const T* d_logits_ptr = logits.data_ptr() + (seq_len - 1) * vocab_size;


    auto& pool = GlobalCudaMemoryPool::instance();

    float* d_scaled_logits_f = static_cast<float*>(pool.allocate(vocab_size * sizeof(float)));
    float* d_max_val = static_cast<float*>(pool.allocate(sizeof(float)));
    int* d_indices = static_cast<int*>(pool.allocate(vocab_size * sizeof(int)));
    float* d_sorted_logits_f = static_cast<float*>(pool.allocate(vocab_size * sizeof(float)));
    int* d_sorted_indices = static_cast<int*>(pool.allocate(vocab_size * sizeof(int)));


    uint32_t* d_sampled_index;
    float* d_sampled_prob;
    cudaMalloc(&d_sampled_index, sizeof(uint32_t));
    cudaMalloc(&d_sampled_prob, sizeof(float));


    void* d_reduce_temp_storage = nullptr;
    size_t reduce_temp_storage_bytes = 0;
    void* d_sort_temp_storage = nullptr;
    size_t sort_temp_storage_bytes = 0;


    const int scale_init_block_size = 256;
    const int scale_init_grid_size = (vocab_size + scale_init_block_size - 1) / scale_init_block_size;


    scale_logits_to_float_and_init_indices_kernel<T><<<scale_init_grid_size, scale_init_block_size, 0, stream>>>(
        d_logits_ptr, d_scaled_logits_f, d_indices, vocab_size, temperature);
    CUDA_CHECK(cudaGetLastError());


    CUDA_CHECK(cub::DeviceReduce::Max(d_reduce_temp_storage, reduce_temp_storage_bytes, d_scaled_logits_f, d_max_val,
                                      vocab_size, stream));
    d_reduce_temp_storage = pool.allocate(reduce_temp_storage_bytes);
    CUDA_CHECK(cub::DeviceReduce::Max(d_reduce_temp_storage, reduce_temp_storage_bytes, d_scaled_logits_f, d_max_val,
                                      vocab_size, stream));
    CUDA_CHECK(cudaGetLastError());


    CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(d_sort_temp_storage, sort_temp_storage_bytes,
                                                         d_scaled_logits_f, d_sorted_logits_f, d_indices,
                                                         d_sorted_indices, vocab_size, 0, sizeof(float) * 8, stream));

    d_sort_temp_storage = pool.allocate(sort_temp_storage_bytes);

    CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(d_sort_temp_storage, sort_temp_storage_bytes,
                                                         d_scaled_logits_f, d_sorted_logits_f, d_indices,
                                                         d_sorted_indices, vocab_size, 0, sizeof(float) * 8, stream));
    CUDA_CHECK(cudaGetLastError());


    const int sample_block_size = 128;

    size_t reduce_storage_size_est = sizeof(cub::BlockReduce<float, sample_block_size>::TempStorage);
    size_t exp_values_size = MAX_TOPK * sizeof(float);
    size_t sample_shared_mem = reduce_storage_size_est + exp_values_size;

    int max_shared_mem_per_block = 0;
    int cur_dev = 0;
    CUDA_CHECK(cudaGetDevice(&cur_dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&max_shared_mem_per_block, cudaDevAttrMaxSharedMemoryPerBlock, cur_dev));
    if (sample_shared_mem > max_shared_mem_per_block) {
        throw std::runtime_error("Required shared memory exceeds the device limit");
    }

    sample_from_sorted_topk_with_prob_kernel<sample_block_size><<<1, sample_block_size, sample_shared_mem, stream>>>(
        d_sorted_logits_f, d_sorted_indices, top_k, d_max_val, d_states, d_sampled_index, d_sampled_prob);
    CUDA_CHECK(cudaGetLastError());


    uint32_t h_sampled_index;
    float h_sampled_prob;

    cudaMemcpyAsync(&h_sampled_index, d_sampled_index, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&h_sampled_prob, d_sampled_prob, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);


    float real_prob = get_token_probability(logits, seq_len - 1, h_sampled_index, stream);


    pool.free(d_scaled_logits_f);
    pool.free(d_max_val);
    pool.free(d_indices);
    pool.free(d_sorted_logits_f);
    pool.free(d_sorted_indices);
    pool.free(d_reduce_temp_storage);
    pool.free(d_sort_temp_storage);

    cudaFree(d_sampled_index);
    cudaFree(d_sampled_prob);


    return {h_sampled_index, real_prob};
}


template <typename T>
void sample_to_fixed_with_prob(Tensor<T>&& logits, uint32_t* token_ptr, float* prob_ptr, float temperature, float top_p,
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
        throw std::runtime_error("Requested top_k (" + std::to_string(top_k) + ") exceeds MAX_TOPK (" +
                                 std::to_string(MAX_TOPK) + ")");
    }


    const T* d_logits_ptr = logits.data_ptr() + (seq_len - 1) * vocab_size;


    auto& pool = GlobalCudaMemoryPool::instance();
    float* d_scaled_logits_f = static_cast<float*>(pool.allocate(vocab_size * sizeof(float)));
    float* d_max_val = static_cast<float*>(pool.allocate(sizeof(float)));
    int* d_indices = static_cast<int*>(pool.allocate(vocab_size * sizeof(int)));
    float* d_sorted_logits_f = static_cast<float*>(pool.allocate(vocab_size * sizeof(float)));
    int* d_sorted_indices = static_cast<int*>(pool.allocate(vocab_size * sizeof(int)));


    void* d_reduce_temp_storage = nullptr;
    size_t reduce_temp_storage_bytes = 0;
    void* d_sort_temp_storage = nullptr;
    size_t sort_temp_storage_bytes = 0;


    const int scale_init_block_size = 256;
    const int scale_init_grid_size = (vocab_size + scale_init_block_size - 1) / scale_init_block_size;

    scale_logits_to_float_and_init_indices_kernel<T><<<scale_init_grid_size, scale_init_block_size, 0, stream>>>(
        d_logits_ptr, d_scaled_logits_f, d_indices, vocab_size, temperature);
    CUDA_CHECK(cudaGetLastError());


    CUDA_CHECK(cub::DeviceReduce::Max(d_reduce_temp_storage, reduce_temp_storage_bytes, d_scaled_logits_f, d_max_val,
                                      vocab_size, stream));
    d_reduce_temp_storage = pool.allocate(reduce_temp_storage_bytes);
    CUDA_CHECK(cub::DeviceReduce::Max(d_reduce_temp_storage, reduce_temp_storage_bytes, d_scaled_logits_f, d_max_val,
                                      vocab_size, stream));
    CUDA_CHECK(cudaGetLastError());


    CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(d_sort_temp_storage, sort_temp_storage_bytes,
                                                         d_scaled_logits_f, d_sorted_logits_f, d_indices,
                                                         d_sorted_indices, vocab_size, 0, sizeof(float) * 8, stream));

    d_sort_temp_storage = pool.allocate(sort_temp_storage_bytes);

    CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(d_sort_temp_storage, sort_temp_storage_bytes,
                                                         d_scaled_logits_f, d_sorted_logits_f, d_indices,
                                                         d_sorted_indices, vocab_size, 0, sizeof(float) * 8, stream));
    CUDA_CHECK(cudaGetLastError());


    const int sample_block_size = 128;

    size_t reduce_storage_size_est = sizeof(cub::BlockReduce<float, sample_block_size>::TempStorage);
    size_t exp_values_size = MAX_TOPK * sizeof(float);
    size_t sample_shared_mem = reduce_storage_size_est + exp_values_size;

    int max_shared_mem_per_block = 0;
    int cur_dev2 = 0;
    CUDA_CHECK(cudaGetDevice(&cur_dev2));
    CUDA_CHECK(cudaDeviceGetAttribute(&max_shared_mem_per_block, cudaDevAttrMaxSharedMemoryPerBlock, cur_dev2));
    if (sample_shared_mem > max_shared_mem_per_block) {
        throw std::runtime_error("Required shared memory exceeds the device limit");
    }

    sample_from_sorted_topk_with_prob_kernel<sample_block_size><<<1, sample_block_size, sample_shared_mem, stream>>>(
        d_sorted_logits_f, d_sorted_indices, top_k, d_max_val, d_states, token_ptr, prob_ptr);
    CUDA_CHECK(cudaGetLastError());


    pool.free(d_scaled_logits_f);
    pool.free(d_max_val);
    pool.free(d_indices);
    pool.free(d_sorted_logits_f);
    pool.free(d_sorted_indices);
    pool.free(d_reduce_temp_storage);
    pool.free(d_sort_temp_storage);
}


template <typename T>
__global__ void batch_scale_logits_and_init_indices_kernel(const T* __restrict__ logits, T* d_scaled_logits,
                                                           int* d_indices, size_t seq_len, size_t vocab_size,
                                                           float temperature) {
    int seq_idx = blockIdx.x;
    int token_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (seq_idx >= seq_len || token_idx >= vocab_size)
        return;

    size_t global_idx = seq_idx * vocab_size + token_idx;


    float logit_f;
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        logit_f = __bfloat162float(__ldg(&logits[global_idx]));
    } else {
        logit_f = static_cast<float>(__ldg(&logits[global_idx]));
    }
    float scaled_logit_f = logit_f / temperature;

    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        d_scaled_logits[global_idx] = __float2bfloat16(scaled_logit_f);
    } else {
        d_scaled_logits[global_idx] = static_cast<T>(scaled_logit_f);
    }


    d_indices[global_idx] = token_idx;
}


template <typename T, int BLOCK_DIM_X>
__global__ void batch_sample_from_sorted_topk_with_prob_kernel(
    const T* __restrict__ d_sorted_topk_logits, const int* __restrict__ d_sorted_topk_indices, size_t seq_len, size_t k,
    const float* __restrict__ d_max_vals, curandState* states, uint32_t* d_sampled_indices, float* d_sampled_probs) {
    int seq_idx = blockIdx.x;
    if (seq_idx >= seq_len)
        return;

    using BlockReduce = cub::BlockReduce<float, BLOCK_DIM_X>;

    __shared__ union SharedStorage {
        typename BlockReduce::TempStorage reduce_storage;
        struct Combined {
            typename BlockReduce::TempStorage reduce_storage;
            float exp_vals[MAX_TOPK];
        } combined;
    } shared_storage;

    int tid = threadIdx.x;


    const T* seq_logits = d_sorted_topk_logits + seq_idx * k;
    const int* seq_indices = d_sorted_topk_indices + seq_idx * k;
    float max_val = d_max_vals[seq_idx];


    float thread_exp_sum = 0.0f;
    for (int i = tid; i < k; i += BLOCK_DIM_X) {
        T scaled_logit_T = seq_logits[i];
        float scaled_logit_f;

        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            scaled_logit_f = __bfloat162float(scaled_logit_T);
        } else {
            scaled_logit_f = static_cast<float>(scaled_logit_T);
        }

        float exp_val = expf(scaled_logit_f - max_val);

        if (i < MAX_TOPK) {
            shared_storage.combined.exp_vals[i] = exp_val;
        }

        thread_exp_sum += exp_val;
    }
    __syncthreads();


    float block_total_exp_sum = BlockReduce(shared_storage.combined.reduce_storage).Sum(thread_exp_sum);


    if (tid == 0) {
        float total_exp_sum = block_total_exp_sum;
        curandState localState = states[seq_idx];

        uint32_t selected_final_index = 0;
        float selected_prob = 0.0f;

        if (total_exp_sum <= 1e-9f || k == 0) {
            if (k > 0) {
                selected_final_index = static_cast<uint32_t>(seq_indices[0]);
                selected_prob = 1.0f;
            } else {
                selected_final_index = 0;
                selected_prob = 1.0f;
            }
        } else {
            float r = curand_uniform(&localState) * total_exp_sum;
            float cumulative = 0.0f;

            selected_final_index = static_cast<uint32_t>(seq_indices[0]);
            float* s_exp_vals = shared_storage.combined.exp_vals;
            selected_prob = s_exp_vals[0] / total_exp_sum;

            for (int i = 0; i < k; ++i) {
                cumulative += s_exp_vals[i];
                if (cumulative >= r) {
                    selected_final_index = static_cast<uint32_t>(seq_indices[i]);
                    selected_prob = s_exp_vals[i] / total_exp_sum;
                    break;
                }
            }
        }

        d_sampled_indices[seq_idx] = selected_final_index;
        d_sampled_probs[seq_idx] = selected_prob;
        states[seq_idx] = localState;
    }
}


__global__ void set_offsets_kernel(int* offsets, size_t seq_len, size_t vocab_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= seq_len) {
        offsets[idx] = idx * vocab_size;
    }
}


template <typename T>
void sample_batch_to_fixed_with_prob(Tensor<T>&& logits, uint32_t* token_ptr, float* prob_ptr, float temperature,
                                     float top_p, size_t top_k, curandState* d_states, cudaStream_t stream) {

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
        throw std::runtime_error("Requested top_k (" + std::to_string(top_k) + ") exceeds MAX_TOPK (" +
                                 std::to_string(MAX_TOPK) + ")");
    }


    auto& pool = GlobalCudaMemoryPool::instance();
    size_t total_elements = seq_len * vocab_size;
    // size_t topk_elements = seq_len * top_k;

    T* d_scaled_logits = static_cast<T*>(pool.allocate(total_elements * sizeof(T)));
    int* d_indices = static_cast<int*>(pool.allocate(total_elements * sizeof(int)));
    T* d_sorted_logits = static_cast<T*>(pool.allocate(total_elements * sizeof(T)));
    int* d_sorted_indices = static_cast<int*>(pool.allocate(total_elements * sizeof(int)));
    float* d_max_vals = static_cast<float*>(pool.allocate(seq_len * sizeof(float)));


    void* d_reduce_temp_storage = nullptr;
    size_t reduce_temp_storage_bytes = 0;
    void* d_sort_temp_storage = nullptr;
    size_t sort_temp_storage_bytes = 0;


    const int block_size = 256;
    dim3 grid_size(seq_len, (vocab_size + block_size - 1) / block_size);

    batch_scale_logits_and_init_indices_kernel<T><<<grid_size, block_size, 0, stream>>>(
        logits.data_ptr(), d_scaled_logits, d_indices, seq_len, vocab_size, temperature);
    CUDA_CHECK(cudaGetLastError());


    int* d_offsets = static_cast<int*>(pool.allocate((seq_len + 1) * sizeof(int)));


    const int offset_block_size = 256;
    const int offset_grid_size = (seq_len + 1 + offset_block_size - 1) / offset_block_size;
    set_offsets_kernel<<<offset_grid_size, offset_block_size, 0, stream>>>(d_offsets, seq_len, vocab_size);
    CUDA_CHECK(cudaGetLastError());

    cub::TransformInputIterator<float, ConvertToFloatFunctor<T>, const T*> itr(d_scaled_logits,
                                                                               ConvertToFloatFunctor<T>());

    CUDA_CHECK(cub::DeviceSegmentedReduce::Max(d_reduce_temp_storage, reduce_temp_storage_bytes, itr, d_max_vals,
                                               seq_len, d_offsets, d_offsets + 1, stream));
    d_reduce_temp_storage = pool.allocate(reduce_temp_storage_bytes);
    CUDA_CHECK(cub::DeviceSegmentedReduce::Max(d_reduce_temp_storage, reduce_temp_storage_bytes, itr, d_max_vals,
                                               seq_len, d_offsets, d_offsets + 1, stream));


    CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairsDescending(
        d_sort_temp_storage, sort_temp_storage_bytes, d_scaled_logits, d_sorted_logits, d_indices, d_sorted_indices,
        total_elements, seq_len, d_offsets, d_offsets + 1, 0, sizeof(T) * 8, stream));
    d_sort_temp_storage = pool.allocate(sort_temp_storage_bytes);
    CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairsDescending(
        d_sort_temp_storage, sort_temp_storage_bytes, d_scaled_logits, d_sorted_logits, d_indices, d_sorted_indices,
        total_elements, seq_len, d_offsets, d_offsets + 1, 0, sizeof(T) * 8, stream));


    const int sample_block_size = 128;
    batch_sample_from_sorted_topk_with_prob_kernel<T, sample_block_size><<<seq_len, sample_block_size, 0, stream>>>(
        d_sorted_logits, d_sorted_indices, seq_len, top_k, d_max_vals, d_states, token_ptr, prob_ptr);
    CUDA_CHECK(cudaGetLastError());


    pool.free(d_scaled_logits);
    pool.free(d_indices);
    pool.free(d_sorted_logits);
    pool.free(d_sorted_indices);
    pool.free(d_max_vals);
    pool.free(d_offsets);
    pool.free(d_reduce_temp_storage);
    pool.free(d_sort_temp_storage);
}


template float get_token_probability<float>(const Tensor<float>&, int, uint32_t, cudaStream_t);
template float get_token_probability<__nv_bfloat16>(const Tensor<__nv_bfloat16>&, int, uint32_t, cudaStream_t);


template std::pair<uint32_t, float> sample_with_prob<float>(Tensor<float>&&, float, float, size_t, curandState*,
                                                            cudaStream_t);
template std::pair<uint32_t, float> sample_with_prob<__nv_bfloat16>(Tensor<__nv_bfloat16>&&, float, float, size_t,
                                                                    curandState*, cudaStream_t);


template void sample_to_fixed_with_prob<float>(Tensor<float>&&, uint32_t*, float*, float, float, size_t, curandState*,
                                               cudaStream_t);
template void sample_to_fixed_with_prob<__nv_bfloat16>(Tensor<__nv_bfloat16>&&, uint32_t*, float*, float, float, size_t,
                                                       curandState*, cudaStream_t);


template void sample_batch_to_fixed_with_prob<float>(Tensor<float>&&, uint32_t*, float*, float, float, size_t,
                                                     curandState*, cudaStream_t);
template void sample_batch_to_fixed_with_prob<__nv_bfloat16>(Tensor<__nv_bfloat16>&&, uint32_t*, float*, float, float,
                                                             size_t, curandState*, cudaStream_t);

}  // namespace cuda_OP

#endif  // CUDA_OP_SAMPLE_WITH_PROB_CUH