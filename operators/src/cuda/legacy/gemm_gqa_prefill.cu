// Grouped-query prefill: query heads map to shared KV heads; tensor layout is [sequence, heads, dimension].


#ifndef CUDA_GQA_GEMM_CUH
#define CUDA_GQA_GEMM_CUH

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuda/legacy/legacy_cuda_api.cuh"

namespace cuda_OP {


template <typename T,
          int BM,
          int BN,
          int BK,
          int TM,
          int TN
          >
__global__ void gqa_gemm_kernel_v2(
    const T *__restrict__ Q,
    const T *__restrict__ K,
    T *__restrict__ scores,
    int seq_len,
    int head_dim,

    int n_q_heads,
    int n_kv_heads,
    int ratio,
    int total_seq_len,
    float scale
) {


  int block_k_seq_idx = blockIdx.x;
  int block_q_seq_idx = blockIdx.y;


  int q_head_idx = blockIdx.z;


  int kv_head_idx = q_head_idx / ratio;


  int block_q_seq_start = block_q_seq_idx * BM;
  int block_k_seq_start = block_k_seq_idx * BN;


  int tid = threadIdx.x;

  constexpr int THREADS_PER_BLOCK_N = BN / TN;


  int thread_m_idx = tid / THREADS_PER_BLOCK_N;
  int thread_n_idx = tid % THREADS_PER_BLOCK_N;


  int thread_q_seq_start = block_q_seq_start + thread_m_idx * TM;
  int thread_k_seq_start = block_k_seq_start + thread_n_idx * TN;

  int ph = 1;
  __shared__ float smemQ[2][BM][BK + 1];
  __shared__ float smemK[2][BN][BK + 1];


  float accum[TM][TN] = {{0.0f}};


  int q_stride_seq = n_q_heads * head_dim;
  int q_stride_head = head_dim;

  int k_stride_seq = n_kv_heads * head_dim;
  int k_stride_head = head_dim;


  const T *q_head_ptr = Q + q_head_idx * q_stride_head;
  const T *k_head_ptr = K + kv_head_idx * k_stride_head;


  constexpr int vec_unit = 16 / sizeof(T);
  int k_tile_start = 0;


// all_tid = BM*BN/(TM*TN)


// TM=2 TN=2 BK=32 BM=32 BN=32


#pragma unroll(1)
  for (int load_idx = tid; load_idx < BM * (BK / vec_unit);
       load_idx += blockDim.x) {

    int smem_q_row = load_idx / (BK / vec_unit);
    int vec_idx_in_row = load_idx % (BK / vec_unit);
    int smem_q_col_start = vec_idx_in_row * vec_unit;


    int global_q_seq = block_q_seq_start + smem_q_row;
    int global_q_dim_vec_start =
        k_tile_start + smem_q_col_start;


    if (global_q_seq < seq_len) {

      if (global_q_dim_vec_start + vec_unit - 1 < head_dim) {

        Vec<T, vec_unit> vq;
        vq.f4 = *reinterpret_cast<const float4 *>(
            &q_head_ptr[global_q_seq * q_stride_seq + global_q_dim_vec_start]);


#pragma unroll
        for (int i = 0; i < vec_unit; ++i) {

          if (smem_q_col_start + i < BK) {
            smemQ[ph & 1][smem_q_row][smem_q_col_start + i] =
                static_cast<float>(vq.t[i]);
          }
        }
      } else {

#pragma unroll
        for (int i = 0; i < vec_unit; ++i) {
          int current_smem_col = smem_q_col_start + i;
          if (current_smem_col < BK) {
            int current_global_dim = k_tile_start + current_smem_col;
            if (current_global_dim < head_dim) {

              T element =
                  q_head_ptr[global_q_seq * q_stride_seq + current_global_dim];
              smemQ[ph & 1][smem_q_row][current_smem_col] =
                  static_cast<float>(element);
            } else {

              smemQ[ph & 1][smem_q_row][current_smem_col] = 0.0f;
            }
          }
        }
      }
    } else {


#pragma unroll
      for (int i = 0; i < vec_unit; ++i) {
        int current_smem_col = smem_q_col_start + i;
        if (current_smem_col < BK) {
          smemQ[ph & 1][smem_q_row][current_smem_col] = 0.0f;
        }
      }
    }
  }


// const int THREADS_PER_ROW_LOAD_GROUP_K = blockDim.x / VEC_LOADS_PER_ROW_K; //


#pragma unroll(1)

  for (int load_idx = tid; load_idx < BN * (BK / vec_unit);
       load_idx += blockDim.x) {

    int smem_k_row = load_idx / (BK / vec_unit);
    int vec_idx_in_row = load_idx % (BK / vec_unit);
    int smem_k_col_start = vec_idx_in_row * vec_unit;


    int global_k_seq = block_k_seq_start + smem_k_row;
    int global_k_dim_vec_start = k_tile_start + smem_k_col_start;


    if (global_k_seq < total_seq_len) {
      if (global_k_dim_vec_start + vec_unit - 1 < head_dim) {

        Vec<T, vec_unit> vk;
        vk.f4 = *reinterpret_cast<const float4 *>(
            &k_head_ptr[global_k_seq * k_stride_seq + global_k_dim_vec_start]);


#pragma unroll
        for (int i = 0; i < vec_unit; ++i) {
          if (smem_k_col_start + i < BK) {
            smemK[ph & 1][smem_k_row][smem_k_col_start + i] =
                static_cast<float>(vk.t[i]);
          }
        }
      } else {

#pragma unroll
        for (int i = 0; i < vec_unit; ++i) {
          int current_smem_col = smem_k_col_start + i;
          if (current_smem_col < BK) {
            int current_global_dim = k_tile_start + current_smem_col;
            if (current_global_dim < head_dim) {
              T element =
                  k_head_ptr[global_k_seq * k_stride_seq + current_global_dim];
              smemK[ph & 1][smem_k_row][current_smem_col] =
                  static_cast<float>(element);
            } else {
              smemK[ph & 1][smem_k_row][current_smem_col] = 0.0f;
            }
          }
        }
      }
    } else {

#pragma unroll
      for (int i = 0; i < vec_unit; ++i) {
        int current_smem_col = smem_k_col_start + i;
        if (current_smem_col < BK) {
          smemK[ph & 1][smem_k_row][current_smem_col] = 0.0f;
        }
      }
    }
  }


  for (k_tile_start = BK; k_tile_start <= head_dim; k_tile_start += BK) {
    ph ^= 1;
#pragma unroll 1
    for (int load_idx = tid; load_idx < BM * (BK / vec_unit);
         load_idx += blockDim.x) {

      int smem_q_row = load_idx / (BK / vec_unit);
      int vec_idx_in_row = load_idx % (BK / vec_unit);
      int smem_q_col_start = vec_idx_in_row * vec_unit;


      int global_q_seq = block_q_seq_start + smem_q_row;
      int global_q_dim_vec_start =
          k_tile_start + smem_q_col_start;


      if (global_q_seq < seq_len) {

        if (global_q_dim_vec_start + vec_unit - 1 < head_dim) {

          Vec<T, vec_unit> vq;
          vq.f4 = *reinterpret_cast<const float4 *>(
              &q_head_ptr[global_q_seq * q_stride_seq +
                          global_q_dim_vec_start]);


#pragma unroll
          for (int i = 0; i < vec_unit; ++i) {

            if (smem_q_col_start + i < BK) {
              smemQ[ph & 1][smem_q_row][smem_q_col_start + i] =
                  static_cast<float>(vq.t[i]);
            }
          }
        } else {

#pragma unroll
          for (int i = 0; i < vec_unit; ++i) {
            int current_smem_col = smem_q_col_start + i;
            if (current_smem_col < BK) {
              int current_global_dim = k_tile_start + current_smem_col;
              if (current_global_dim < head_dim) {

                T element = q_head_ptr[global_q_seq * q_stride_seq +
                                       current_global_dim];
                smemQ[ph & 1][smem_q_row][current_smem_col] =
                    static_cast<float>(element);
              } else {

                smemQ[ph & 1][smem_q_row][current_smem_col] = 0.0f;
              }
            }
          }
        }
      } else {


#pragma unroll
        for (int i = 0; i < vec_unit; ++i) {
          int current_smem_col = smem_q_col_start + i;
          smemQ[ph & 1][smem_q_row][current_smem_col] = 0.0f;
        }
      }
    }


#pragma unroll(1)

    for (int load_idx = tid; load_idx < BN * (BK / vec_unit);
         load_idx += blockDim.x) {

      int smem_k_row = load_idx / (BK / vec_unit);
      int vec_idx_in_row = load_idx % (BK / vec_unit);
      int smem_k_col_start = vec_idx_in_row * vec_unit;


      int global_k_seq = block_k_seq_start + smem_k_row;
      int global_k_dim_vec_start = k_tile_start + smem_k_col_start;


      if (global_k_seq < total_seq_len) {
        if (global_k_dim_vec_start + vec_unit - 1 < head_dim) {

          Vec<T, vec_unit> vk;
          vk.f4 = *reinterpret_cast<const float4 *>(
              &k_head_ptr[global_k_seq * k_stride_seq +
                          global_k_dim_vec_start]);


#pragma unroll
          for (int i = 0; i < vec_unit; ++i) {
            if (smem_k_col_start + i < BK) {
              smemK[ph & 1][smem_k_row][smem_k_col_start + i] =
                  static_cast<float>(vk.t[i]);
            }
          }
        } else {

#pragma unroll
          for (int i = 0; i < vec_unit; ++i) {
            int current_smem_col = smem_k_col_start + i;
            if (current_smem_col < BK) {
              int current_global_dim = k_tile_start + current_smem_col;
              if (current_global_dim < head_dim) {
                T element = k_head_ptr[global_k_seq * k_stride_seq +
                                       current_global_dim];
                smemK[ph & 1][smem_k_row][current_smem_col] =
                    static_cast<float>(element);
              } else {
                smemK[ph & 1][smem_k_row][current_smem_col] = 0.0f;
              }
            }
          }
        }
      } else {

#pragma unroll
        for (int i = 0; i < vec_unit; ++i) {
          int current_smem_col = smem_k_col_start + i;
          if (current_smem_col < BK) {
            smemK[ph & 1][smem_k_row][current_smem_col] = 0.0f;
          }
        }
      }
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < BK; ++k) {
#pragma unroll
      for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {


          int smem_q_row = thread_m_idx * TM + i;
          int smem_k_col = thread_n_idx * TN + j;
          accum[i][j] +=
              smemQ[(ph) ^ 1][smem_q_row][k] * smemK[(ph) ^ 1][smem_k_col][k];
        }
      }
    }
  }
  __syncthreads();


  int scores_stride_seq = n_q_heads * total_seq_len;
  int scores_stride_head = total_seq_len;

#pragma unroll
  for (int i = 0; i < TM; ++i) {
#pragma unroll
    for (int j = 0; j < TN; ++j) {

      int global_scores_q_seq = thread_q_seq_start + i;
      int global_scores_k_seq = thread_k_seq_start + j;


      if (global_scores_q_seq < seq_len &&
          global_scores_k_seq < total_seq_len) {

        // scores[global_scores_q_seq][q_head_idx][global_scores_k_seq]
        int scores_offset = global_scores_q_seq * scores_stride_seq +
                            q_head_idx * scores_stride_head +
                            global_scores_k_seq;

        scores[scores_offset] = static_cast<T>(accum[i][j] * scale);
      }
    }
  }
}


template <typename T,
          int BM,
          int BN,
          int BK,
          int TM,
          int TN
          >
__global__ void gqa_gemm_kernel_v1(
    const T *__restrict__ Q,
    const T *__restrict__ K,
    T *__restrict__ scores,
    int seq_len,
    int head_dim,

    int n_q_heads,
    int n_kv_heads,
    int ratio,
    int total_seq_len,
    float scale
) {


  int block_k_seq_idx = blockIdx.x;
  int block_q_seq_idx = blockIdx.y;


  int q_head_idx = blockIdx.z;


  int kv_head_idx = q_head_idx / ratio;


  int block_q_seq_start = block_q_seq_idx * BM;
  int block_k_seq_start = block_k_seq_idx * BN;


  int tid = threadIdx.x;

  constexpr int THREADS_PER_BLOCK_N = BN / TN;


  int thread_m_idx = tid / THREADS_PER_BLOCK_N;
  int thread_n_idx = tid % THREADS_PER_BLOCK_N;


  int thread_q_seq_start = block_q_seq_start + thread_m_idx * TM;
  int thread_k_seq_start = block_k_seq_start + thread_n_idx * TN;

  __shared__ float smemQ[BM][BK];
  __shared__ float smemK[BN][BK];


  float accum[TM][TN] = {{0.0f}};


  int q_stride_seq = n_q_heads * head_dim;
  int q_stride_head = head_dim;

  int k_stride_seq = n_kv_heads * head_dim;
  int k_stride_head = head_dim;


  const T *q_head_ptr = Q + q_head_idx * q_stride_head;
  const T *k_head_ptr = K + kv_head_idx * k_stride_head;

  for (int k_tile_start = 0; k_tile_start < head_dim; k_tile_start += BK) {
#pragma unroll

    for (int load_idx = tid; load_idx < BM * BK; load_idx += blockDim.x) {


      int load_row = load_idx / BK;
      int load_col = load_idx % BK;
      int global_q_seq = block_q_seq_start + load_row;
      int global_q_dim = k_tile_start + load_col;
      if (global_q_seq < seq_len && global_q_dim < head_dim) {
        smemQ[load_row][load_col] = static_cast<float>(
            q_head_ptr[global_q_seq * q_stride_seq + global_q_dim]);
      } else {
        smemQ[load_row][load_col] = 0.0f;
      }
    }
#pragma unroll
    for (int load_idx = tid; load_idx < BN * BK; load_idx += blockDim.x) {


      int load_row = load_idx / BK;
      int load_col = load_idx % BK;
      int global_k_seq = block_k_seq_start + load_row;
      int global_k_dim = k_tile_start + load_col;
      if (global_k_seq < total_seq_len && global_k_dim < head_dim) {
        smemK[load_row][load_col] = static_cast<float>(
            k_head_ptr[global_k_seq * k_stride_seq + global_k_dim]);
      } else {
        smemK[load_row][load_col] = 0.0f;
      }
    }
    __syncthreads();


#pragma unroll
    for (int k = 0; k < BK; ++k) {
#pragma unroll
      for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {


          int smem_q_row = thread_m_idx * TM + i;
          int smem_k_col = thread_n_idx * TN + j;
          accum[i][j] += smemQ[smem_q_row][k] * smemK[smem_k_col][k];
        }
      }
    }
    __syncthreads();
  }


  int scores_stride_seq = n_q_heads * total_seq_len;
  int scores_stride_head = total_seq_len;

#pragma unroll
  for (int i = 0; i < TM; ++i) {
#pragma unroll
    for (int j = 0; j < TN; ++j) {

      int global_scores_q_seq = thread_q_seq_start + i;
      int global_scores_k_seq = thread_k_seq_start + j;


      if (global_scores_q_seq < seq_len &&
          global_scores_k_seq < total_seq_len) {

        // scores[global_scores_q_seq][q_head_idx][global_scores_k_seq]
        int scores_offset = global_scores_q_seq * scores_stride_seq +
                            q_head_idx * scores_stride_head +
                            global_scores_k_seq;

        scores[scores_offset] = static_cast<T>(accum[i][j] * scale);
      }
    }
  }
}


template <typename T>
void launch_gqa_gemm(
    const Tensor<T> &Q,
    const Tensor<T> &K,
    Tensor<T> &scores,
    cudaStream_t stream)
{

  const auto &q_sizes = Q.sizes();
  const auto &k_sizes = K.sizes();
  const auto &scores_sizes = scores.sizes();


  if (q_sizes.size() != 3 || k_sizes.size() != 3 || scores_sizes.size() != 3) {
    throw std::runtime_error(
        "Input/output tensors must be 3D with [seq, heads, "
        "dim/seq_k] layout");
  }


  int seq_len = q_sizes[0];
  int n_q_heads = q_sizes[1];
  int head_dim = q_sizes[2];

  int total_seq_len = k_sizes[0];
  int n_kv_heads = k_sizes[1];
  if (k_sizes[2] != head_dim) {
    throw std::runtime_error("Q (" + std::to_string(head_dim) + ") and K (" +
                             std::to_string(k_sizes[2]) +
                             ") have different head dimensions");
  }


  if (n_q_heads == 0 || n_kv_heads == 0) {
    throw std::runtime_error("The head count must be nonzero");
  }
  if (n_q_heads % n_kv_heads != 0) {
    throw std::runtime_error("n_q_heads (" + std::to_string(n_q_heads) +
                             ") must be divisible by n_kv_heads (" +
                             std::to_string(n_kv_heads) + ")");
  }
  int ratio = n_q_heads / n_kv_heads;


  if (scores_sizes[0] != seq_len || scores_sizes[1] != n_q_heads ||
      scores_sizes[2] != total_seq_len) {
    throw std::runtime_error(
        "Scores tensor shape does not match; expected [" + std::to_string(seq_len) + ", " +
        std::to_string(n_q_heads) + ", " + std::to_string(total_seq_len) +
        "], got [" + std::to_string(scores_sizes[0]) + ", " +
        std::to_string(scores_sizes[1]) + ", " +
        std::to_string(scores_sizes[2]) + "]");
  }


  if (seq_len == 0 || total_seq_len == 0 || n_q_heads == 0 || head_dim == 0) {
    throw std::runtime_error("Invalid gqa_gemm data");
  }


  constexpr int BM = 32;
  constexpr int BN = 32;
  constexpr int BK = 32;
  constexpr int TM = 2;
  constexpr int TN = 2;


  constexpr int THREADS_M = BM / TM;

  constexpr int THREADS_N = BN / TN;
  constexpr int THREADS_PER_BLOCK = THREADS_M * THREADS_N;
  static_assert(THREADS_PER_BLOCK > 0 && THREADS_PER_BLOCK <= 1024,
                "Invalid thread count per block");

  static_assert(BM % TM == 0, "BM must be divisible by TM");
  static_assert(BN % TN == 0, "BN must be divisible by TN");


  dim3 blockDim(THREADS_PER_BLOCK);
  dim3 gridDim((total_seq_len + BN - 1) / BN,
               (seq_len + BM - 1) / BM,
               n_q_heads
  );


  static float scale = (head_dim > 0)
                           ? (1.0f / sqrtf(static_cast<float>(head_dim)))
                           : 1.0f;


  gqa_gemm_kernel_v2<T, BM, BN, BK, TM, TN><<<gridDim, blockDim, 0, stream>>>(
      Q.data_ptr(), K.data_ptr(), scores.data_ptr(), seq_len, head_dim,

      n_q_heads, n_kv_heads, ratio, total_seq_len, scale);


  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {

    std::string error_msg = "CUDA error in launch_gqa_gemm (Grid: ";
    error_msg += std::to_string(gridDim.x) + "," + std::to_string(gridDim.y) +
                 "," + std::to_string(gridDim.z);
    error_msg += ", Block: " + std::to_string(blockDim.x) +
                 "): " + std::string(cudaGetErrorString(err));
    throw std::runtime_error(error_msg);
  }
}


template void launch_gqa_gemm<float>(const Tensor<float> &Q,
                                     const Tensor<float> &K,
                                     Tensor<float> &scores,
                                     cudaStream_t stream);

template void launch_gqa_gemm<half>(const Tensor<half> &Q,
                                    const Tensor<half> &K, Tensor<half> &scores,
                                    cudaStream_t stream);

template void launch_gqa_gemm<nv_bfloat16>(const Tensor<nv_bfloat16> &Q,
                                           const Tensor<nv_bfloat16> &K,
                                           Tensor<nv_bfloat16> &scores,
                                           cudaStream_t stream);

}  // namespace cuda_OP

#endif  // CUDA_GQA_GEMM_TILED_CUH

// #ifndef CUDA_GQA_GEMM_CUH
// #define CUDA_GQA_GEMM_CUH


// #include "cuda/legacy/legacy_cuda_api.cuh"

// namespace cuda_OP
// {
//     /**

//      *


//      */
//     template <typename T, int BLOCK_SIZE = 16>
//     __global__ void gqa_gemm_kernel(


//     {
//         int batch_idx = blockIdx.z / n_q_heads;
//         int q_head_idx = blockIdx.z % n_q_heads;


//         int kv_head_idx = q_head_idx / ratio;


//         int seq_pos = blockIdx.y * BLOCK_SIZE + threadIdx.y;
//         int t_seq_pos = blockIdx.x * BLOCK_SIZE + threadIdx.x;
//         int scores_offset = batch_idx * total_seq_len * n_q_heads * seq_len +
//                             seq_pos * total_seq_len * n_q_heads +
//                             q_head_idx * total_seq_len + t_seq_pos;
//         const T *q = Q + batch_idx * n_q_heads * head_dim * seq_len +
//                      seq_pos * head_dim * n_q_heads + q_head_idx * head_dim;
//         const T *k = K + batch_idx * n_kv_heads * head_dim * total_seq_len +
//                      t_seq_pos * head_dim * n_kv_heads + kv_head_idx *
//                      head_dim;
//         float sum = 0.0f;
//         int tid_t = threadIdx.x;
//         int tid = threadIdx.y;
//         int tile = (head_dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
//         __shared__ float smemQ[16][16];
//         __shared__ float smemK[16][16];
//         for (int i = 0; i < tile; i++)
//         {
//             if (tid_t + i * BLOCK_SIZE < head_dim && tid + i * BLOCK_SIZE <
//             head_dim)
//             {
//                 smemQ[tid][tid_t] = static_cast<float>(q[tid_t + i *
//                 BLOCK_SIZE]); smemK[tid_t][tid] = static_cast<float>(k[tid +
//                 i * BLOCK_SIZE]);
//             }
//             __syncthreads();
//             for (int j = 0; j < BLOCK_SIZE; j++)
//             {
//                 sum += smemQ[tid][j] * smemK[tid_t][j];
//             }
//             __syncthreads();
//         }

//         sum *= rsqrtf(static_cast<float>(head_dim));

//         if (seq_pos < seq_len && t_seq_pos < total_seq_len)
//         {
//             scores[scores_offset] = static_cast<T>(sum);
//         }
//     }

//     /**

//      *


//      */
//     template <typename T>
//     void launch_gqa_gemm(


//         seq_len, head_dim] Tensor<T>

//             seq_len]


//     {
//         int batch_size = 1;


//         int seq_len = Q.sizes()[0];
//         int n_q_heads = Q.sizes()[1];
//         int head_dim = Q.sizes()[2];
//         int total_seq_len = K.sizes()[0];
//         int n_kv_heads = K.sizes()[1];


//         if (n_q_heads % n_kv_heads != 0)
//         {
//             throw std::runtime_error("n_q_heads must be divisible by
//             n_kv_heads");
//         }
//         int ratio = n_q_heads / n_kv_heads;


//         if (scores.sizes()[0] != seq_len || scores.sizes()[1] != n_q_heads ||
//             scores.sizes()[2] != total_seq_len)
//         {
//             throw std::runtime_error("scores tensor shape mismatch");
//         }


//         dim3 grid(
//             (total_seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE, // x:


//         );


//         dim3 block(BLOCK_SIZE, BLOCK_SIZE);


//         //

//         gqa_gemm_kernel<T, BLOCK_SIZE><<<grid, block, 0, stream>>>(


//         );


//         cudaError_t err = cudaGetLastError();
//         if (err != cudaSuccess)
//         {
//             throw std::runtime_error("CUDA error in launch_gqa_gemm: " +
//                                      std::string(cudaGetErrorString(err)));
//         }
//     }


//     template void launch_gqa_gemm<float>(const Tensor<float> &Q,
//                                          const Tensor<float> &K,
//                                          Tensor<float> &scores,
//                                          cudaStream_t stream);

//     template void launch_gqa_gemm<half>(const Tensor<half> &Q,
//                                         const Tensor<half> &K, Tensor<half>
//                                         &scores, cudaStream_t stream);

//     template void launch_gqa_gemm<nv_bfloat16>(const Tensor<nv_bfloat16> &Q,
//                                                const Tensor<nv_bfloat16> &K,
//                                                Tensor<nv_bfloat16> &scores,
//                                                cudaStream_t stream);

// } // namespace cuda_OP

// #endif // CUDA_GQA_GEMM_CUH