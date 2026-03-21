#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>  // printf
#include <iostream>
#include <stdexcept>
#include <vector>

#include "cuda/legacy/legacy_cuda_api.cuh"
#include "tensor.hpp"
namespace cuda_OP {
template <typename T>
__global__ void gather_kernel_v2(const uint32_t* input, const T* embedding_table, T* output, int seq_len, int embed_dim,
                                 int vocab_size) {
    // 每个 float4 载入 16 字节，单位 T 的个数：
    constexpr int vec_unit = sizeof(float4) / sizeof(T);  // =4 for float, =8 for __nv_bfloat16
    // 每行有多少个 vectorized 块
    int num_vec = embed_dim / vec_unit;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    // 总共需要处理的 vectorized 块数
    int total = seq_len * num_vec;

    // 将 embedding_table 和 output 当作 float4 数组处理
    const float4* embedding_table_vec = reinterpret_cast<const float4*>(embedding_table);
    float4* output_vec = reinterpret_cast<float4*>(output);

    for (int idx = tid; idx < total; idx += stride) {
        int row = idx / num_vec;
        int vec_idx = idx % num_vec;
        uint32_t token_id = input[row];
        if (token_id >= vocab_size) {
            continue;
        }
        // 计算当前 vector 块在 embedding_table 中起始元素的全局偏移
        int emb_index = token_id * embed_dim + vec_idx * vec_unit;
        if (emb_index + vec_unit > vocab_size * embed_dim) {
            continue;
        }
        // 将元素索引转换为 vector 索引
        int vec_index = emb_index / vec_unit;
        float4 vec_data = embedding_table_vec[vec_index];
        output_vec[row * num_vec + vec_idx] = vec_data;
    }
}

// 模板化的 CUDA kernel，output[i, :] = embedding_table[input[i], :]
template <typename T>
__global__ void gather_kernel_v1(const uint32_t* input, const T* embedding_table, T* output, int seq_len, int embed_dim,
                                 int vocab_size) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int total = seq_len * embed_dim;
    for (int idx = tid; idx < total; idx += stride) {
        int row = idx / embed_dim;
        int col = idx % embed_dim;
        // 越界检查（理论上 grid-stride loop
        // 已经保证了，不需要再次判断，但增加健壮性）
        if (row >= seq_len || col >= embed_dim) {
            continue;
        }
        uint32_t token_id = input[row];
        if (token_id >= vocab_size) {
            continue;
        }
        int emb_index = token_id * embed_dim + col;
        // 再次检查防止溢出
        if (emb_index >= vocab_size * embed_dim) {
            continue;
        }
        T value = embedding_table[emb_index];
        output[idx] = value;
    }
}

template <typename T>
void gather(Tensor<T>* output, const Tensor<uint32_t>* input, const Tensor<T>* embedding_table, cudaStream_t stream) {
    int seq_len = static_cast<int>(input->numel());
    int embed_dim = static_cast<int>(output->sizes()[1]);
    int vocab_size = static_cast<int>(embedding_table->sizes()[0]);

    // 验证设备一致性
    if (input->device() != Device::CUDA) {
        throw std::runtime_error("Input must be on CUDA device");
    }
    if (output->device() != Device::CUDA) {
        throw std::runtime_error("Output must be on CUDA device");
    }
    if (embedding_table->device() != Device::CUDA) {
        throw std::runtime_error("Embedding table must be on CUDA device");
    }

    // 验证输入数据有效性
    if (seq_len <= 0 || embed_dim <= 0 || vocab_size <= 0) {
        throw std::runtime_error("Invalid dimensions in gather");
    }



    int threadsPerBlock = 256;
    int total = seq_len * embed_dim;
    // For gather_kernel_v2, the 'total' is actually seq_len * num_vec.
    // However, blocks calculation based on seq_len * embed_dim is a safe upper bound
    // and common practice for grid-stride loops that cover all elements.
    // If embed_dim is not a multiple of vec_unit, there might be some threads
    // doing extra work, but the kernel's boundary checks will handle it.
    // A more precise block calculation for v2 would be based on seq_len * num_vec.
    // Here, we stick to the original logic assuming embed_dim is large enough for v2.
    int blocks = (total + threadsPerBlock - 1) / threadsPerBlock;

    // 启动 CUDA kernel
    gather_kernel_v2<T><<<blocks, threadsPerBlock, 0, stream>>>(input->data_ptr(), embedding_table->data_ptr(),
                                                                output->data_ptr(), seq_len, embed_dim, vocab_size);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA gather kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }
    // 注意：这里没有同步，函数是异步的
}
template void gather<nvbf16>(Tensor<nvbf16>*, const Tensor<uint32_t>*, const Tensor<nvbf16>*, cudaStream_t);
template void gather<float>(Tensor<float>*, const Tensor<uint32_t>*, const Tensor<float>*, cudaStream_t);
}  // namespace cuda_OP
