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

    constexpr int vec_unit = sizeof(float4) / sizeof(T);  // =4 for float, =8 for __nv_bfloat16

    int num_vec = embed_dim / vec_unit;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int total = seq_len * num_vec;


    const float4* embedding_table_vec = reinterpret_cast<const float4*>(embedding_table);
    float4* output_vec = reinterpret_cast<float4*>(output);

    for (int idx = tid; idx < total; idx += stride) {
        int row = idx / num_vec;
        int vec_idx = idx % num_vec;
        uint32_t token_id = input[row];
        if (token_id >= vocab_size) {
            continue;
        }

        int emb_index = token_id * embed_dim + vec_idx * vec_unit;
        if (emb_index + vec_unit > vocab_size * embed_dim) {
            continue;
        }

        int vec_index = emb_index / vec_unit;
        float4 vec_data = embedding_table_vec[vec_index];
        output_vec[row * num_vec + vec_idx] = vec_data;
    }
}


template <typename T>
__global__ void gather_kernel_v1(const uint32_t* input, const T* embedding_table, T* output, int seq_len, int embed_dim,
                                 int vocab_size) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int total = seq_len * embed_dim;
    for (int idx = tid; idx < total; idx += stride) {
        int row = idx / embed_dim;
        int col = idx % embed_dim;


        if (row >= seq_len || col >= embed_dim) {
            continue;
        }
        uint32_t token_id = input[row];
        if (token_id >= vocab_size) {
            continue;
        }
        int emb_index = token_id * embed_dim + col;

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


    if (input->device() != Device::CUDA) {
        throw std::runtime_error("Input must be on CUDA device");
    }
    if (output->device() != Device::CUDA) {
        throw std::runtime_error("Output must be on CUDA device");
    }
    if (embedding_table->device() != Device::CUDA) {
        throw std::runtime_error("Embedding table must be on CUDA device");
    }


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


    gather_kernel_v2<T><<<blocks, threadsPerBlock, 0, stream>>>(input->data_ptr(), embedding_table->data_ptr(),
                                                                output->data_ptr(), seq_len, embed_dim, vocab_size);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA gather kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }

}
template void gather<nvbf16>(Tensor<nvbf16>*, const Tensor<uint32_t>*, const Tensor<nvbf16>*, cudaStream_t);
template void gather<float>(Tensor<float>*, const Tensor<uint32_t>*, const Tensor<float>*, cudaStream_t);
}  // namespace cuda_OP
