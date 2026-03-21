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

// 包含 CUB 头文件
#include <cub/cub.cuh>

#include "CudaMemoryPool.hpp"
#include "cuda/legacy/legacy_cuda_api.cuh"
#include "tensor.hpp"

// 使用与sample.cu相同的MAX_TOPK定义
#define MAX_TOPK 1024

namespace cuda_OP {

// 从sample.cu引入所需的函数和类型定义
// Kernel 1: 缩放 Logits 并初始化索引
template <typename T>
__global__ void scale_logits_and_init_indices_kernel(const T* __restrict__ logits,  // 输入: 原始 logits (设备指针)
                                                     T* d_scaled_logits,            // 输出: 缩放后的 logits (设备指针)
                                                     int* d_indices,                // 输出: 初始化的索引数组 (设备指针)
                                                     size_t vocab_size,             // 输入: 词汇表大小
                                                     float temperature              // 输入: 温度系数
) {
    // 使用网格跨步循环处理所有词汇
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < vocab_size; i += stride) {
        // 1. 缩放 Logits
        float logit_f;
        // 使用 __ldg 进行缓存的全局内存读取
        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            logit_f = __bfloat162float(__ldg(&logits[i]));
        } else {
            logit_f = static_cast<float>(__ldg(&logits[i]));
        }
        float scaled_logit_f = logit_f / temperature;  // 应用温度缩放

        // 写回缩放后的值 (根据类型转换)
        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            d_scaled_logits[i] = __float2bfloat16(scaled_logit_f);
        } else {
            d_scaled_logits[i] = static_cast<T>(scaled_logit_f);
        }

        // 2. 初始化索引
        d_indices[i] = i;
    }
}

// 将 logits 缩放为 float 并初始化索引
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

// CUB TransformIterator 的辅助 Functor
// 功能: 在 CUB 操作中动态地将输入类型 Tin (如 bfloat16) 转换为 float。
template <typename Tin>
struct ConvertToFloatFunctor {
    __device__ __forceinline__ float operator()(const Tin& x) const {
        if constexpr (std::is_same_v<Tin, __nv_bfloat16>) {
            return __bfloat162float(x);  // bfloat16 转 float
        } else {
            return static_cast<float>(x);  // 其他类型直接转 float
        }
    }
};

// Kernel：生成随机值数组
__global__ void generate_random_values_kernel(float* values, size_t count, curandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < count; i += stride) {
        // 使用线程0的随机状态生成随机数
        if (idx == 0) {
            curandState localState = states[0];
            for (int j = 0; j < count; j++) {
                values[j] = curand_uniform(&localState);
            }
            states[0] = localState;
            break;  // 只需要一个线程执行
        }
    }
}

// 生成随机值数组的函数实现
void generate_random_values(float* values, size_t count, curandState* states, cudaStream_t stream) {
    const int block_size = 256;
    const int grid_size = (count + block_size - 1) / block_size;

    generate_random_values_kernel<<<grid_size, block_size, 0, stream>>>(values, count, states);
    CUDA_CHECK(cudaGetLastError());
}

// Kernel：计算指定token的概率（考虑temperature）
template <typename T>
__global__ void get_token_probability_kernel(const T* logits, int position, uint32_t token_id, float* output_prob,
                                             size_t vocab_size, float temperature) {
    // 单线程执行，计算指定token的概率
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // 计算该位置的logits起始位置
        const T* pos_logits = logits + position * vocab_size;

        // 找到最大logit值（用于数值稳定性）
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

        // 计算分母（所有exp值之和）
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

        // 计算目标token的概率
        float token_val;
        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            token_val = __bfloat162float(pos_logits[token_id]) / temperature;
        } else {
            token_val = static_cast<float>(pos_logits[token_id]) / temperature;
        }

        // 计算softmax概率
        *output_prob = expf(token_val - max_val) / sum_exp;
    }
}

// 获取指定token在logits中的概率
template <typename T>
float get_token_probability(const Tensor<T>& logits, int position, uint32_t token_id, cudaStream_t stream) {
    if (logits.device() != Device::CUDA) {
        throw std::runtime_error("输入张量必须在CUDA设备上");
    }

    const auto& shape = logits.sizes();
    if (shape.size() != 2) {
        throw std::runtime_error("输入张量必须是二维 [seq_len, vocab_size]");
    }

    size_t seq_len = shape[0];
    size_t vocab_size = shape[1];

    if (position < 0 || position >= seq_len) {
        throw std::runtime_error("位置超出范围");
    }

    if (token_id >= vocab_size) {
        throw std::runtime_error("token_id超出词汇表大小");
    }

    // 分配设备内存用于存储概率结果
    float* d_prob;
    cudaMalloc(&d_prob, sizeof(float));

    // 启动kernel计算概率
    get_token_probability_kernel<<<1, 1, 0, stream>>>(logits.data_ptr(), position, token_id, d_prob, vocab_size, 1.0f);
    CUDA_CHECK(cudaGetLastError());

    // 将结果复制回主机
    float h_prob;
    cudaMemcpyAsync(&h_prob, d_prob, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // 释放设备内存
    cudaFree(d_prob);

    return h_prob;
}

// Kernel：从排序后的top-k结果中采样，并记录概率（float logits）
template <int BLOCK_DIM_X>
__global__ void sample_from_sorted_topk_with_prob_kernel(const float* __restrict__ d_sorted_topk_logits,
                                                         const int* __restrict__ d_sorted_topk_indices, size_t k,
                                                         const float* __restrict__ d_max_val_ptr, curandState* states,
                                                         uint32_t* d_sampled_index, float* d_sampled_prob) {
    // 使用与sample.cu相同的CUB块内归约
    using BlockReduce = cub::BlockReduce<float, BLOCK_DIM_X>;

    // 共享内存设置
    __shared__ union SharedStorage {
        typename BlockReduce::TempStorage reduce_storage;
        struct Combined {
            typename BlockReduce::TempStorage reduce_storage;
            float exp_vals[MAX_TOPK];
        } combined;
    } shared_storage;

    int tid = threadIdx.x;

    // 线程0读取最大值并存入共享内存
    __shared__ float max_val_shared;
    if (tid == 0) {
        max_val_shared = *d_max_val_ptr;
    }
    __syncthreads();

    // 并行计算exp(logit - max_val)
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

    // 使用CUB在块内归约求和
    float block_total_exp_sum = BlockReduce(shared_storage.combined.reduce_storage).Sum(thread_exp_sum);

    // 线程0执行加权采样
    if (tid == 0) {
        float total_exp_sum = block_total_exp_sum;
        curandState localState = states[0];

        uint32_t selected_final_index = 0;
        float selected_prob = 0.0f;

        if (total_exp_sum <= 1e-9f || k == 0) {
            if (k > 0) {
                selected_final_index = static_cast<uint32_t>(d_sorted_topk_indices[0]);
                selected_prob = 1.0f;  // 如果总和接近0，选择概率最高的那个，概率为1
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
                    // 计算选中token的概率
                    selected_prob = s_exp_vals[i] / total_exp_sum;
                    break;
                }
            }
        }

        // 将最终选定的索引和概率写入输出指针
        *d_sampled_index = selected_final_index;
        *d_sampled_prob = selected_prob;

        // 更新cuRAND状态
        states[0] = localState;
    }
}

// 采样函数，返回token和其概率
template <typename T>
std::pair<uint32_t, float> sample_with_prob(Tensor<T>&& logits, float temperature, float top_p, size_t top_k,
                                            curandState* d_states, cudaStream_t stream) {
    // --- 输入验证 ---
    if (logits.device() != Device::CUDA) {
        throw std::runtime_error("输入张量必须在CUDA设备上");
    }

    if (top_k == 0) {
        throw std::runtime_error("top_k必须至少为1");
    }

    const auto& shape = logits.sizes();
    if (shape.size() != 2 || shape[0] == 0 || shape[1] == 0) {
        throw std::runtime_error("输入张量必须是二维且维度非零 [seq_len, vocab_size]");
    }

    const size_t seq_len = shape[0];
    const size_t vocab_size = shape[1];

    if (top_k > vocab_size) {
        top_k = vocab_size;
    }

    if (top_k > MAX_TOPK) {
        throw std::runtime_error("请求的top_k (" + std::to_string(top_k) + ") 超过了MAX_TOPK限制 (" +
                                 std::to_string(MAX_TOPK) + ")");
    }

    // 获取指向最后一个token的logits数据的设备指针
    const T* d_logits_ptr = logits.data_ptr() + (seq_len - 1) * vocab_size;

    // --- 内存管理 ---
    auto& pool = GlobalCudaMemoryPool::instance();
    // 使用 float 作为排序键
    float* d_scaled_logits_f = static_cast<float*>(pool.allocate(vocab_size * sizeof(float)));
    float* d_max_val = static_cast<float*>(pool.allocate(sizeof(float)));
    int* d_indices = static_cast<int*>(pool.allocate(vocab_size * sizeof(int)));
    float* d_sorted_logits_f = static_cast<float*>(pool.allocate(vocab_size * sizeof(float)));
    int* d_sorted_indices = static_cast<int*>(pool.allocate(vocab_size * sizeof(int)));

    // 为token和概率分配内存
    uint32_t* d_sampled_index;
    float* d_sampled_prob;
    cudaMalloc(&d_sampled_index, sizeof(uint32_t));
    cudaMalloc(&d_sampled_prob, sizeof(float));

    // CUB临时存储
    void* d_reduce_temp_storage = nullptr;
    size_t reduce_temp_storage_bytes = 0;
    void* d_sort_temp_storage = nullptr;
    size_t sort_temp_storage_bytes = 0;

    // --- 步骤1: 缩放Logits并初始化索引 ---
    const int scale_init_block_size = 256;
    const int scale_init_grid_size = (vocab_size + scale_init_block_size - 1) / scale_init_block_size;

    // 缩放为 float 并初始化索引
    scale_logits_to_float_and_init_indices_kernel<T><<<scale_init_grid_size, scale_init_block_size, 0, stream>>>(
        d_logits_ptr, d_scaled_logits_f, d_indices, vocab_size, temperature);
    CUDA_CHECK(cudaGetLastError());

    // --- 步骤2: 查找最大缩放Logit ---
    CUDA_CHECK(cub::DeviceReduce::Max(d_reduce_temp_storage, reduce_temp_storage_bytes, d_scaled_logits_f, d_max_val,
                                      vocab_size, stream));
    d_reduce_temp_storage = pool.allocate(reduce_temp_storage_bytes);
    CUDA_CHECK(cub::DeviceReduce::Max(d_reduce_temp_storage, reduce_temp_storage_bytes, d_scaled_logits_f, d_max_val,
                                      vocab_size, stream));
    CUDA_CHECK(cudaGetLastError());

    // --- 步骤3: 按Logit值降序排序 ---
    CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(d_sort_temp_storage, sort_temp_storage_bytes,
                                                         d_scaled_logits_f, d_sorted_logits_f, d_indices,
                                                         d_sorted_indices, vocab_size, 0, sizeof(float) * 8, stream));

    d_sort_temp_storage = pool.allocate(sort_temp_storage_bytes);

    CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(d_sort_temp_storage, sort_temp_storage_bytes,
                                                         d_scaled_logits_f, d_sorted_logits_f, d_indices,
                                                         d_sorted_indices, vocab_size, 0, sizeof(float) * 8, stream));
    CUDA_CHECK(cudaGetLastError());

    // --- 步骤4: 从Top-K结果中进行最终加权采样 ---
    const int sample_block_size = 128;

    size_t reduce_storage_size_est = sizeof(cub::BlockReduce<float, sample_block_size>::TempStorage);
    size_t exp_values_size = MAX_TOPK * sizeof(float);
    size_t sample_shared_mem = reduce_storage_size_est + exp_values_size;

    int max_shared_mem_per_block = 0;
    int cur_dev = 0;
    CUDA_CHECK(cudaGetDevice(&cur_dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&max_shared_mem_per_block, cudaDevAttrMaxSharedMemoryPerBlock, cur_dev));
    if (sample_shared_mem > max_shared_mem_per_block) {
        throw std::runtime_error("计算出的所需共享内存超过设备限制");
    }

    sample_from_sorted_topk_with_prob_kernel<sample_block_size><<<1, sample_block_size, sample_shared_mem, stream>>>(
        d_sorted_logits_f, d_sorted_indices, top_k, d_max_val, d_states, d_sampled_index, d_sampled_prob);
    CUDA_CHECK(cudaGetLastError());

    // --- 结果处理 ---
    uint32_t h_sampled_index;
    float h_sampled_prob;

    cudaMemcpyAsync(&h_sampled_index, d_sampled_index, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&h_sampled_prob, d_sampled_prob, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // 计算选中token在整个词汇表中的真实概率（考虑temperature）
    float real_prob = get_token_probability(logits, seq_len - 1, h_sampled_index, stream);

    // --- 释放临时内存 ---
    pool.free(d_scaled_logits_f);
    pool.free(d_max_val);
    pool.free(d_indices);
    pool.free(d_sorted_logits_f);
    pool.free(d_sorted_indices);
    pool.free(d_reduce_temp_storage);
    pool.free(d_sort_temp_storage);

    cudaFree(d_sampled_index);
    cudaFree(d_sampled_prob);

    // 返回token和其在整个词汇表中的真实概率
    return {h_sampled_index, real_prob};
}

// 采样函数的变体，将token和概率写入指定的GPU内存位置
template <typename T>
void sample_to_fixed_with_prob(Tensor<T>&& logits, uint32_t* token_ptr, float* prob_ptr, float temperature, float top_p,
                               size_t top_k, curandState* d_states, cudaStream_t stream) {
    // 直接在GPU上进行采样，避免GPU-CPU-GPU内存复制
    // --- 输入验证 ---
    if (logits.device() != Device::CUDA) {
        throw std::runtime_error("输入张量必须在CUDA设备上");
    }

    if (top_k == 0) {
        throw std::runtime_error("top_k必须至少为1");
    }

    const auto& shape = logits.sizes();
    if (shape.size() != 2 || shape[0] == 0 || shape[1] == 0) {
        throw std::runtime_error("输入张量必须是二维且维度非零 [seq_len, vocab_size]");
    }

    const size_t seq_len = shape[0];
    const size_t vocab_size = shape[1];

    if (top_k > vocab_size) {
        top_k = vocab_size;
    }

    if (top_k > MAX_TOPK) {
        throw std::runtime_error("请求的top_k (" + std::to_string(top_k) + ") 超过了MAX_TOPK限制 (" +
                                 std::to_string(MAX_TOPK) + ")");
    }

    // 获取指向最后一个token的logits数据的设备指针
    const T* d_logits_ptr = logits.data_ptr() + (seq_len - 1) * vocab_size;

    // --- 内存管理 ---
    auto& pool = GlobalCudaMemoryPool::instance();
    float* d_scaled_logits_f = static_cast<float*>(pool.allocate(vocab_size * sizeof(float)));
    float* d_max_val = static_cast<float*>(pool.allocate(sizeof(float)));
    int* d_indices = static_cast<int*>(pool.allocate(vocab_size * sizeof(int)));
    float* d_sorted_logits_f = static_cast<float*>(pool.allocate(vocab_size * sizeof(float)));
    int* d_sorted_indices = static_cast<int*>(pool.allocate(vocab_size * sizeof(int)));

    // CUB临时存储
    void* d_reduce_temp_storage = nullptr;
    size_t reduce_temp_storage_bytes = 0;
    void* d_sort_temp_storage = nullptr;
    size_t sort_temp_storage_bytes = 0;

    // --- 步骤1: 缩放Logits并初始化索引 ---
    const int scale_init_block_size = 256;
    const int scale_init_grid_size = (vocab_size + scale_init_block_size - 1) / scale_init_block_size;

    scale_logits_to_float_and_init_indices_kernel<T><<<scale_init_grid_size, scale_init_block_size, 0, stream>>>(
        d_logits_ptr, d_scaled_logits_f, d_indices, vocab_size, temperature);
    CUDA_CHECK(cudaGetLastError());

    // --- 步骤2: 查找最大缩放Logit ---
    CUDA_CHECK(cub::DeviceReduce::Max(d_reduce_temp_storage, reduce_temp_storage_bytes, d_scaled_logits_f, d_max_val,
                                      vocab_size, stream));
    d_reduce_temp_storage = pool.allocate(reduce_temp_storage_bytes);
    CUDA_CHECK(cub::DeviceReduce::Max(d_reduce_temp_storage, reduce_temp_storage_bytes, d_scaled_logits_f, d_max_val,
                                      vocab_size, stream));
    CUDA_CHECK(cudaGetLastError());

    // --- 步骤3: 按Logit值降序排序 ---
    CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(d_sort_temp_storage, sort_temp_storage_bytes,
                                                         d_scaled_logits_f, d_sorted_logits_f, d_indices,
                                                         d_sorted_indices, vocab_size, 0, sizeof(float) * 8, stream));

    d_sort_temp_storage = pool.allocate(sort_temp_storage_bytes);

    CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(d_sort_temp_storage, sort_temp_storage_bytes,
                                                         d_scaled_logits_f, d_sorted_logits_f, d_indices,
                                                         d_sorted_indices, vocab_size, 0, sizeof(float) * 8, stream));
    CUDA_CHECK(cudaGetLastError());

    // --- 步骤4: 从Top-K结果中进行最终加权采样，直接写入输出位置 ---
    const int sample_block_size = 128;

    size_t reduce_storage_size_est = sizeof(cub::BlockReduce<float, sample_block_size>::TempStorage);
    size_t exp_values_size = MAX_TOPK * sizeof(float);
    size_t sample_shared_mem = reduce_storage_size_est + exp_values_size;

    int max_shared_mem_per_block = 0;
    int cur_dev2 = 0;
    CUDA_CHECK(cudaGetDevice(&cur_dev2));
    CUDA_CHECK(cudaDeviceGetAttribute(&max_shared_mem_per_block, cudaDevAttrMaxSharedMemoryPerBlock, cur_dev2));
    if (sample_shared_mem > max_shared_mem_per_block) {
        throw std::runtime_error("计算出的所需共享内存超过设备限制");
    }

    sample_from_sorted_topk_with_prob_kernel<sample_block_size><<<1, sample_block_size, sample_shared_mem, stream>>>(
        d_sorted_logits_f, d_sorted_indices, top_k, d_max_val, d_states, token_ptr, prob_ptr);
    CUDA_CHECK(cudaGetLastError());

    // --- 释放临时内存 ---
    pool.free(d_scaled_logits_f);
    pool.free(d_max_val);
    pool.free(d_indices);
    pool.free(d_sorted_logits_f);
    pool.free(d_sorted_indices);
    pool.free(d_reduce_temp_storage);
    pool.free(d_sort_temp_storage);
}

// 批量缩放logits并初始化索引的kernel
template <typename T>
__global__ void batch_scale_logits_and_init_indices_kernel(const T* __restrict__ logits, T* d_scaled_logits,
                                                           int* d_indices, size_t seq_len, size_t vocab_size,
                                                           float temperature) {
    int seq_idx = blockIdx.x;
    int token_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (seq_idx >= seq_len || token_idx >= vocab_size)
        return;

    size_t global_idx = seq_idx * vocab_size + token_idx;

    // 缩放logits
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

    // 初始化索引
    d_indices[global_idx] = token_idx;
}

// 批量采样kernel - 每个块处理一个序列
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

    // 获取当前序列的数据指针
    const T* seq_logits = d_sorted_topk_logits + seq_idx * k;
    const int* seq_indices = d_sorted_topk_indices + seq_idx * k;
    float max_val = d_max_vals[seq_idx];

    // 并行计算exp值
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

    // 块内归约
    float block_total_exp_sum = BlockReduce(shared_storage.combined.reduce_storage).Sum(thread_exp_sum);

    // 线程0执行采样
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

// 设置分段偏移量的kernel
__global__ void set_offsets_kernel(int* offsets, size_t seq_len, size_t vocab_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= seq_len) {
        offsets[idx] = idx * vocab_size;
    }
}

// 批量采样函数的变体，将token和概率写入指定的GPU内存位置
template <typename T>
void sample_batch_to_fixed_with_prob(Tensor<T>&& logits, uint32_t* token_ptr, float* prob_ptr, float temperature,
                                     float top_p, size_t top_k, curandState* d_states, cudaStream_t stream) {
    // --- 输入验证 ---
    if (logits.device() != Device::CUDA) {
        throw std::runtime_error("输入张量必须在CUDA设备上");
    }

    if (top_k == 0) {
        throw std::runtime_error("top_k必须至少为1");
    }

    const auto& shape = logits.sizes();
    if (shape.size() != 2 || shape[0] == 0 || shape[1] == 0) {
        throw std::runtime_error("输入张量必须是二维且维度非零 [seq_len, vocab_size]");
    }

    const size_t seq_len = shape[0];
    const size_t vocab_size = shape[1];

    if (top_k > vocab_size) {
        top_k = vocab_size;
    }

    if (top_k > MAX_TOPK) {
        throw std::runtime_error("请求的top_k (" + std::to_string(top_k) + ") 超过了MAX_TOPK限制 (" +
                                 std::to_string(MAX_TOPK) + ")");
    }

    // --- 内存管理 ---
    auto& pool = GlobalCudaMemoryPool::instance();
    size_t total_elements = seq_len * vocab_size;
    // size_t topk_elements = seq_len * top_k;

    T* d_scaled_logits = static_cast<T*>(pool.allocate(total_elements * sizeof(T)));
    int* d_indices = static_cast<int*>(pool.allocate(total_elements * sizeof(int)));
    T* d_sorted_logits = static_cast<T*>(pool.allocate(total_elements * sizeof(T)));
    int* d_sorted_indices = static_cast<int*>(pool.allocate(total_elements * sizeof(int)));
    float* d_max_vals = static_cast<float*>(pool.allocate(seq_len * sizeof(float)));

    // CUB临时存储
    void* d_reduce_temp_storage = nullptr;
    size_t reduce_temp_storage_bytes = 0;
    void* d_sort_temp_storage = nullptr;
    size_t sort_temp_storage_bytes = 0;

    // --- 步骤1: 批量缩放logits并初始化索引 ---
    const int block_size = 256;
    dim3 grid_size(seq_len, (vocab_size + block_size - 1) / block_size);

    batch_scale_logits_and_init_indices_kernel<T><<<grid_size, block_size, 0, stream>>>(
        logits.data_ptr(), d_scaled_logits, d_indices, seq_len, vocab_size, temperature);
    CUDA_CHECK(cudaGetLastError());

    // --- 步骤2: 批量查找最大值 ---
    // 创建分段偏移量数组
    int* d_offsets = static_cast<int*>(pool.allocate((seq_len + 1) * sizeof(int)));

    // 设置分段偏移量
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

    // --- 步骤3: 批量排序 ---
    CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairsDescending(
        d_sort_temp_storage, sort_temp_storage_bytes, d_scaled_logits, d_sorted_logits, d_indices, d_sorted_indices,
        total_elements, seq_len, d_offsets, d_offsets + 1, 0, sizeof(T) * 8, stream));
    d_sort_temp_storage = pool.allocate(sort_temp_storage_bytes);
    CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairsDescending(
        d_sort_temp_storage, sort_temp_storage_bytes, d_scaled_logits, d_sorted_logits, d_indices, d_sorted_indices,
        total_elements, seq_len, d_offsets, d_offsets + 1, 0, sizeof(T) * 8, stream));

    // --- 步骤4: 批量采样 ---
    const int sample_block_size = 128;
    batch_sample_from_sorted_topk_with_prob_kernel<T, sample_block_size><<<seq_len, sample_block_size, 0, stream>>>(
        d_sorted_logits, d_sorted_indices, seq_len, top_k, d_max_vals, d_states, token_ptr, prob_ptr);
    CUDA_CHECK(cudaGetLastError());

    // --- 释放临时内存 ---
    pool.free(d_scaled_logits);
    pool.free(d_indices);
    pool.free(d_sorted_logits);
    pool.free(d_sorted_indices);
    pool.free(d_max_vals);
    pool.free(d_offsets);
    pool.free(d_reduce_temp_storage);
    pool.free(d_sort_temp_storage);
}

// --- 模板显式实例化 ---
// 为float和__nv_bfloat16类型实例化get_token_probability函数
template float get_token_probability<float>(const Tensor<float>&, int, uint32_t, cudaStream_t);
template float get_token_probability<__nv_bfloat16>(const Tensor<__nv_bfloat16>&, int, uint32_t, cudaStream_t);

// 为float和__nv_bfloat16类型实例化sample_with_prob函数
template std::pair<uint32_t, float> sample_with_prob<float>(Tensor<float>&&, float, float, size_t, curandState*,
                                                            cudaStream_t);
template std::pair<uint32_t, float> sample_with_prob<__nv_bfloat16>(Tensor<__nv_bfloat16>&&, float, float, size_t,
                                                                    curandState*, cudaStream_t);

// 为float和__nv_bfloat16类型实例化sample_to_fixed_with_prob函数
template void sample_to_fixed_with_prob<float>(Tensor<float>&&, uint32_t*, float*, float, float, size_t, curandState*,
                                               cudaStream_t);
template void sample_to_fixed_with_prob<__nv_bfloat16>(Tensor<__nv_bfloat16>&&, uint32_t*, float*, float, float, size_t,
                                                       curandState*, cudaStream_t);

// 为float和__nv_bfloat16类型实例化sample_batch_to_fixed_with_prob函数
template void sample_batch_to_fixed_with_prob<float>(Tensor<float>&&, uint32_t*, float*, float, float, size_t,
                                                     curandState*, cudaStream_t);
template void sample_batch_to_fixed_with_prob<__nv_bfloat16>(Tensor<__nv_bfloat16>&&, uint32_t*, float*, float, float,
                                                             size_t, curandState*, cudaStream_t);

}  // namespace cuda_OP

#endif  // CUDA_OP_SAMPLE_WITH_PROB_CUH