#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdint.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "cuda/legacy/legacy_cuda_api.cuh"

// Define WARP_SIZE if not already defined
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

namespace cuda_OP {

// WMMA-optimized GQA GEMM kernel for attention score computation
// Computes Q @ K^T for attention scores
template <typename T,      // Data type (half, nv_bfloat16)
          int BM,          // Block M dimension (Q sequence length)
          int BN,          // Block N dimension (K sequence length)
          int BK,          // Block K dimension (head dimension)
          int WMMA_M,      // WMMA M dimension
          int WMMA_N,      // WMMA N dimension
          int WMMA_K,      // WMMA K dimension
          int WARP_CNT>    // Number of warps per block
__global__ void gqa_gemm_wmma_kernel(
    const T *__restrict__ Q,          // Query [seq_len, n_q_heads, head_dim]
    const T *__restrict__ K,          // Key [total_seq_len, n_kv_heads, head_dim]
    T *__restrict__ scores,           // Output [seq_len, n_q_heads, total_seq_len]
    int seq_len,                      // Q sequence length
    int head_dim,                     // Head dimension
    int n_q_heads,                    // Number of query heads
    int n_kv_heads,                   // Number of key-value heads
    int ratio,                        // n_q_heads / n_kv_heads
    int total_seq_len,                // K sequence length
    float scale                       // Scale factor (1.0f / sqrtf(head_dim))
) {
    using namespace nvcuda;
    using FragmentA = wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, T, wmma::row_major>;
    using FragmentB = wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, T, wmma::row_major>;
    using FragmentC = wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>;

    // Block and thread indexing
    int block_k_seq_idx = blockIdx.x;
    int block_q_seq_idx = blockIdx.y;
    int q_head_idx = blockIdx.z;
    int kv_head_idx = q_head_idx / ratio;

    int block_q_seq_start = block_q_seq_idx * BM;
    int block_k_seq_start = block_k_seq_idx * BN;

    // Warp indexing
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_m = warp_id / (BN / WMMA_N);
    int warp_n = warp_id % (BN / WMMA_N);

    // Shared memory for tiling
    __shared__ T smemQ[BM * BK];
    __shared__ T smemK[BN * BK];
    __shared__ float smem_out[BM * BN];

    // WMMA fragments
    FragmentC fragC;
    wmma::fill_fragment(fragC, 0.0f);

    // Compute strides
    int q_stride_seq = n_q_heads * head_dim;
    int k_stride_seq = n_kv_heads * head_dim;
    const T *q_head_ptr = Q + q_head_idx * head_dim;
    const T *k_head_ptr = K + kv_head_idx * head_dim;

    // Tile over K dimension
    for (int k_tile = 0; k_tile < head_dim; k_tile += BK) {
        // Load Q tile cooperatively
        for (int load_idx = threadIdx.x; load_idx < BM * BK; load_idx += blockDim.x) {
            int load_row = load_idx / BK;
            int load_col = load_idx % BK;
            int global_q_seq = block_q_seq_start + load_row;
            int global_q_dim = k_tile + load_col;

            if (global_q_seq < seq_len && global_q_dim < head_dim) {
                smemQ[load_idx] = q_head_ptr[global_q_seq * q_stride_seq + global_q_dim];
            } else {
                smemQ[load_idx] = static_cast<T>(0.0f);
            }
        }

        // Load K tile cooperatively (transposed for K^T)
        for (int load_idx = threadIdx.x; load_idx < BN * BK; load_idx += blockDim.x) {
            int load_row = load_idx / BK;
            int load_col = load_idx % BK;
            int global_k_seq = block_k_seq_start + load_row;
            int global_k_dim = k_tile + load_col;

            if (global_k_seq < total_seq_len && global_k_dim < head_dim) {
                // Load K^T: transpose K matrix
                smemK[load_col * BN + load_row] = k_head_ptr[global_k_seq * k_stride_seq + global_k_dim];
            } else {
                smemK[load_col * BN + load_row] = static_cast<T>(0.0f);
            }
        }

        __syncthreads();

        // WMMA computation
        for (int k_step = 0; k_step < BK; k_step += WMMA_K) {
            FragmentA fragA;
            FragmentB fragB;

            // Load A fragment (Q)
            const T *smemA_ptr = smemQ + (warp_m * WMMA_M * BK) + k_step;
            wmma::load_matrix_sync(fragA, smemA_ptr, BK);

            // Load B fragment (K^T)
            const T *smemB_ptr = smemK + (k_step * BN) + (warp_n * WMMA_N);
            wmma::load_matrix_sync(fragB, smemB_ptr, BN);

            // Perform WMMA
            wmma::mma_sync(fragC, fragA, fragB, fragC);
        }

        __syncthreads();
    }

    // Store result to shared memory
    const int out_m_base = warp_m * WMMA_M;
    const int out_n_base = warp_n * WMMA_N;
    wmma::store_matrix_sync(&smem_out[out_m_base * BN + out_n_base], fragC, BN, wmma::mem_row_major);

    __syncthreads();

    // Write back to global memory with scaling
    for (int write_idx = threadIdx.x; write_idx < BM * BN; write_idx += blockDim.x) {
        int m_local = write_idx / BN;
        int n_local = write_idx % BN;

        int global_q_seq = block_q_seq_start + m_local;
        int global_k_seq = block_k_seq_start + n_local;

        if (global_q_seq < seq_len && global_k_seq < total_seq_len) {
            int scores_offset = global_q_seq * (n_q_heads * total_seq_len) +
                               q_head_idx * total_seq_len + global_k_seq;
            float result = smem_out[write_idx] * scale;
            scores[scores_offset] = static_cast<T>(result);
        }
    }
}

// WMMA-optimized attention output kernel
// Computes attention_scores @ V
template <typename T,      // Data type (half, nv_bfloat16)
          int BM,          // Block M dimension (sequence length)
          int BN,          // Block N dimension (head dimension)
          int BK,          // Block K dimension (cache/sequence length)
          int WMMA_M,      // WMMA M dimension
          int WMMA_N,      // WMMA N dimension
          int WMMA_K,      // WMMA K dimension
          int WARP_CNT>    // Number of warps per block
__global__ void att_output_wmma_kernel(
    const T *__restrict__ att_probs,  // Attention probabilities [seq_len, n_q_heads, cache_length]
    const T *__restrict__ V,          // Value [cache_length, n_kv_heads, head_dim]
    T *__restrict__ att_output,       // Output [seq_len, n_q_heads, head_dim]
    int seq_len,                      // Sequence length
    int cache_length,                 // Cache length
    int head_dim,                     // Head dimension
    int n_q_heads,                    // Number of query heads
    int n_kv_heads                    // Number of key-value heads
) {
    using namespace nvcuda;
    using FragmentA = wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, T, wmma::row_major>;
    using FragmentB = wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, T, wmma::row_major>;
    using FragmentC = wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>;

    // Block and thread indexing
    int block_seq_idx = blockIdx.x;
    int block_dim_idx = blockIdx.y;
    int q_head_idx = blockIdx.z;
    int kv_head_idx = q_head_idx / (n_q_heads / n_kv_heads);

    int block_seq_start = block_seq_idx * BM;
    int block_dim_start = block_dim_idx * BN;

    // Warp indexing
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_m = warp_id / (BN / WMMA_N);
    int warp_n = warp_id % (BN / WMMA_N);

    // Shared memory for tiling
    __shared__ T smemA[BM * BK];  // Attention probabilities
    __shared__ T smemB[BK * BN];  // Value matrix
    __shared__ float smem_out[BM * BN];

    // WMMA fragments
    FragmentC fragC;
    wmma::fill_fragment(fragC, 0.0f);

    // Tile over cache_length dimension
    for (int k_tile = 0; k_tile < cache_length; k_tile += BK) {
        // Load attention probabilities tile
        for (int load_idx = threadIdx.x; load_idx < BM * BK; load_idx += blockDim.x) {
            int load_row = load_idx / BK;
            int load_col = load_idx % BK;
            int global_seq = block_seq_start + load_row;
            int global_cache = k_tile + load_col;

            if (global_seq < seq_len && global_cache < cache_length) {
                int att_idx = global_seq * (n_q_heads * cache_length) +
                             q_head_idx * cache_length + global_cache;
                smemA[load_idx] = att_probs[att_idx];
            } else {
                smemA[load_idx] = static_cast<T>(0.0f);
            }
        }

        // Load V tile
        for (int load_idx = threadIdx.x; load_idx < BK * BN; load_idx += blockDim.x) {
            int load_row = load_idx / BN;
            int load_col = load_idx % BN;
            int global_cache = k_tile + load_row;
            int global_dim = block_dim_start + load_col;

            if (global_cache < cache_length && global_dim < head_dim) {
                int v_idx = global_cache * (n_kv_heads * head_dim) +
                           kv_head_idx * head_dim + global_dim;
                smemB[load_idx] = V[v_idx];
            } else {
                smemB[load_idx] = static_cast<T>(0.0f);
            }
        }

        __syncthreads();

        // WMMA computation
        for (int k_step = 0; k_step < BK; k_step += WMMA_K) {
            FragmentA fragA;
            FragmentB fragB;

            // Load A fragment (attention probabilities)
            const T *smemA_ptr = smemA + (warp_m * WMMA_M * BK) + k_step;
            wmma::load_matrix_sync(fragA, smemA_ptr, BK);

            // Load B fragment (V)
            const T *smemB_ptr = smemB + (k_step * BN) + (warp_n * WMMA_N);
            wmma::load_matrix_sync(fragB, smemB_ptr, BN);

            // Perform WMMA
            wmma::mma_sync(fragC, fragA, fragB, fragC);
        }

        __syncthreads();
    }

    // Store result to shared memory
    const int out_m_base = warp_m * WMMA_M;
    const int out_n_base = warp_n * WMMA_N;
    wmma::store_matrix_sync(&smem_out[out_m_base * BN + out_n_base], fragC, BN, wmma::mem_row_major);

    __syncthreads();

    // Write back to global memory
    for (int write_idx = threadIdx.x; write_idx < BM * BN; write_idx += blockDim.x) {
        int m_local = write_idx / BN;
        int n_local = write_idx % BN;

        int global_seq = block_seq_start + m_local;
        int global_dim = block_dim_start + n_local;

        if (global_seq < seq_len && global_dim < head_dim) {
            int output_idx = global_seq * (n_q_heads * head_dim) +
                           q_head_idx * head_dim + global_dim;
            att_output[output_idx] = static_cast<T>(smem_out[write_idx]);
        }
    }
}

// Host-side launcher for WMMA-optimized GQA GEMM
template <typename T>
void launch_gqa_gemm_wmma(
    const Tensor<T> &Q,
    const Tensor<T> &K,
    Tensor<T> &scores,
    cudaStream_t stream
) {
    // Extract dimensions
    const auto &q_sizes = Q.sizes();
    const auto &k_sizes = K.sizes();

    int seq_len = q_sizes[0];
    int n_q_heads = q_sizes[1];
    int head_dim = q_sizes[2];
    int total_seq_len = k_sizes[0];
    int n_kv_heads = k_sizes[1];
    int ratio = n_q_heads / n_kv_heads;

    // WMMA configuration
    constexpr int BM = 64, BN = 64, BK = 32;
    constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
    constexpr int WARP_CNT = (BM / WMMA_M) * (BN / WMMA_N);

    // Launch configuration
    dim3 blockDim(WARP_CNT * WARP_SIZE);
    dim3 gridDim((total_seq_len + BN - 1) / BN,
                 (seq_len + BM - 1) / BM,
                 n_q_heads);

    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    // Launch kernel
    if constexpr (std::is_same_v<T, __nv_bfloat16> || std::is_same_v<T, __half>) {
        gqa_gemm_wmma_kernel<T, BM, BN, BK, WMMA_M, WMMA_N, WMMA_K, WARP_CNT>
            <<<gridDim, blockDim, 0, stream>>>(
                Q.data_ptr(), K.data_ptr(), scores.data_ptr(),
                seq_len, head_dim, n_q_heads, n_kv_heads, ratio, total_seq_len, scale);
    } else {
        throw std::runtime_error("WMMA requires half or bfloat16 precision");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("WMMA GQA GEMM launch failed: " +
                                std::string(cudaGetErrorString(err)));
    }
}

// Host-side launcher for WMMA-optimized attention output
template <typename T>
void launch_att_output_wmma(
    const Tensor<T> &att_probs,
    const Tensor<T> &V,
    Tensor<T> &att_output,
    int n_q_heads,
    int n_kv_heads,
    cudaStream_t stream
) {

    // Extract dimensions
    int seq_len = att_probs.sizes()[0];
    int cache_length = att_probs.sizes()[2];
    int head_dim = V.sizes()[2];

    // WMMA configuration
    constexpr int BM = 32, BN = 64, BK = 32;
    constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
    constexpr int WARP_CNT = (BM / WMMA_M) * (BN / WMMA_N);

    // Launch configuration
    dim3 blockDim(WARP_CNT * WARP_SIZE);
    dim3 gridDim((seq_len + BM - 1) / BM,
                 (head_dim + BN - 1) / BN,
                 n_q_heads);

    // Launch kernel
    if constexpr (std::is_same_v<T, __nv_bfloat16> || std::is_same_v<T, __half>) {
        att_output_wmma_kernel<T, BM, BN, BK, WMMA_M, WMMA_N, WMMA_K, WARP_CNT>
            <<<gridDim, blockDim, 0, stream>>>(
                att_probs.data_ptr(), V.data_ptr(), att_output.data_ptr(),
                seq_len, cache_length, head_dim, n_q_heads, n_kv_heads);
    } else {
        throw std::runtime_error("WMMA requires half or bfloat16 precision");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("WMMA attention output launch failed: " +
                                std::string(cudaGetErrorString(err)));
    }
}

// Updated wrapper functions that can choose between original and WMMA versions
template <typename T>
void compute_attention_scores_prefill_wmma(
    const Tensor<T> &Q,
    const Tensor<T> &K,
    Tensor<T> &att_scores,
    size_t head_dim,
    cudaStream_t stream
) {
    // Use WMMA version for supported types and dimensions
    if constexpr (std::is_same_v<T, __nv_bfloat16> || std::is_same_v<T, __half>) {
        if (head_dim % 16 == 0) {  // WMMA requires dimensions divisible by 16
            launch_gqa_gemm_wmma(Q, K, att_scores, stream);
            return;
        }
    }

    // Fallback to original implementation
    launch_gqa_gemm(Q, K, att_scores, stream);
}

template <typename T>
void compute_att_output_prefill_wmma(
    const Tensor<T> &att_probs,
    const Tensor<T> &V,
    Tensor<T> &att_output,
    size_t n_q_heads,
    size_t head_dim,
    size_t total_seq_len,
    size_t n_kv_heads,
    cudaStream_t stream
) {
    // Use WMMA version for supported types and dimensions
    if constexpr (std::is_same_v<T, __nv_bfloat16> || std::is_same_v<T, __half>) {
        if (head_dim % 16 == 0) {  // WMMA requires dimensions divisible by 16
            launch_att_output_wmma(att_probs, V, att_output, n_q_heads, n_kv_heads, stream);
            return;
        }
    }

    // Fallback to original implementation
    compute_att_output_prefill(att_probs, V, att_output, n_q_heads, head_dim, total_seq_len, n_kv_heads, stream);
}

// Template instantiations
template void compute_attention_scores_prefill_wmma<__nv_bfloat16>(
    const Tensor<__nv_bfloat16> &, const Tensor<__nv_bfloat16> &, Tensor<__nv_bfloat16> &, size_t, cudaStream_t);
template void compute_attention_scores_prefill_wmma<__half>(
    const Tensor<__half> &, const Tensor<__half> &, Tensor<__half> &, size_t, cudaStream_t);

template void compute_att_output_prefill_wmma<__nv_bfloat16>(
    const Tensor<__nv_bfloat16> &, const Tensor<__nv_bfloat16> &, Tensor<__nv_bfloat16> &,
    size_t, size_t, size_t, size_t, cudaStream_t);
template void compute_att_output_prefill_wmma<__half>(
    const Tensor<__half> &, const Tensor<__half> &, Tensor<__half> &,
    size_t, size_t, size_t, size_t, cudaStream_t);

// Add missing float template instantiation for WMMA functions
template void compute_attention_scores_prefill_wmma<float>(
    const Tensor<float> &, const Tensor<float> &, Tensor<float> &, size_t, cudaStream_t);
template void compute_att_output_prefill_wmma<float>(
    const Tensor<float> &, const Tensor<float> &, Tensor<float> &,
    size_t, size_t, size_t, size_t, cudaStream_t);

}  // namespace cuda_OP