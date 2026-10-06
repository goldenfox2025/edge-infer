// Packed INT4 AWQ kernels. Keep the weight/zero packing order and group-size layout consistent with the loader.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdint.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "ptx_common.h"
#include "operators/cuda/execution_kernels.cuh"


namespace cuda_OP {

constexpr int BITS = 4;
constexpr int PACK_FACTOR = 32 / BITS;
constexpr int WARP_SIZE = 32;
template <typename T, typename S, int BM, int BN, int BK, int WMMA_M, int WMMA_N, int WMMA_K, int WARP_NUM, int K_STAGE,
          int WARP_TILE_M, int WARP_TILE_N>
__global__ void awq_gemm_kernel_mma(const T* A,          // Input matrix [M, K]
                                    const int32_t* qwt,  // Quantized weights [N, K/8] (N-Major)
                                    const S* scl,        // Scale factors [N, G_padded] (N-Major)
                                    const int32_t* zos,  // Zero points [N, G/8] (N-Major)
                                    T* C,                // Output matrix [M, N]
                                    int M, int N, int K, int group_size, int G_PADDED,
                                    const T* bias) {
    // Dequantization constants
    constexpr int BITS = 4;
    constexpr int PACK_FACTOR = 32 / BITS;

    const int G = K / group_size;
    const int K_PACKED = K / PACK_FACTOR;
    const int G_PACKED = G / PACK_FACTOR;

    int warp_id = threadIdx.x / 32;
    constexpr int WARP_N_NUM = BN / (WMMA_N * WARP_TILE_N);
    int warp_n_id = warp_id % WARP_N_NUM;
    int warp_m_id = warp_id / WARP_N_NUM;
    int global_m_base = blockIdx.x * BM;
    int global_n_base = blockIdx.y * BN;
    constexpr int SA_SIZE = BM * BK;
    constexpr int SB_SIZE = BN * BK;
    const int lane_id = threadIdx.x % 32;

    // Shared memory
    __shared__ T smemA[K_STAGE * BM * BK];
    __shared__ T smemB[K_STAGE * BN * BK];
    uint32_t smem_a_base_ptr = __cvta_generic_to_shared(smemA);
    uint32_t smem_b_base_ptr = __cvta_generic_to_shared(smemB);

    constexpr int vec_size = sizeof(float4) / sizeof(T);

    // Registers for MMA
    uint32_t RC[WARP_TILE_M][WARP_TILE_N][8];

#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            RC[i][j][0] = 0;
            RC[i][j][1] = 0;
            RC[i][j][2] = 0;
            RC[i][j][3] = 0;
            RC[i][j][4] = 0;
            RC[i][j][5] = 0;
            RC[i][j][6] = 0;
            RC[i][j][7] = 0;
        }
    }

    // Pre-loading loop with swizzle
    for (int k_load_stage = 0; k_load_stage < (K_STAGE - 1); ++k_load_stage) {
        // Load A matrix data with swizzle
        for (int load_idx = threadIdx.x * vec_size; load_idx < BM * BK; load_idx += blockDim.x * vec_size) {
            int smem_row = load_idx / BK;
            int smem_col = load_idx % BK;
            int global_row = global_m_base + smem_row;
            int global_col = k_load_stage * BK + smem_col;

            if (global_row < M && (global_col + vec_size - 1) < K) {
                int load_gmem_a_addr = global_row * K + global_col;
                int swizzled_col = swizzle_permuted_A_j(smem_row, smem_col);
                int swizzled_idx = smem_row * BK + swizzled_col;
                uint32_t load_smem_a_ptr = smem_a_base_ptr + (swizzled_idx + k_load_stage * SA_SIZE) * sizeof(T);
                CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);
            }
        }


        constexpr int CHUNK_SIZE = 8;
        for (int load_idx = threadIdx.x / CHUNK_SIZE; load_idx < BN * BK / CHUNK_SIZE;
             load_idx += blockDim.x / CHUNK_SIZE) {
            const int inner_idx = threadIdx.x % CHUNK_SIZE;

            int smem_row = load_idx / (BK / CHUNK_SIZE);
            int smem_col_base = (load_idx % (BK / CHUNK_SIZE)) * CHUNK_SIZE;
            int global_row = global_n_base + smem_row;
            int global_col_base = k_load_stage * BK + smem_col_base;

            int32_t qwt_val = (global_row < N && global_col_base < K)
                                  ? qwt[global_row * K_PACKED + global_col_base / PACK_FACTOR]
                                  : 0;
            int base_group_idx = global_col_base / group_size;
            S scale_val = (global_row < N && global_col_base < K) ? scl[global_row * G_PADDED + base_group_idx] : S(0);
            int32_t zeros_val =
                (global_row < N && global_col_base < K) ? zos[global_row * G_PACKED + base_group_idx / PACK_FACTOR] : 0;

            int swizzled_col_base = swizzle_permuted_B_j(smem_row, smem_col_base);
            int swizzled_idx_base = smem_row * BK + swizzled_col_base;


            int global_col = global_col_base + inner_idx;
            T dequantized_val = T(0);

            if (global_row < N && global_col < K) {
                uint32_t w = (qwt_val >> ((global_col % PACK_FACTOR) * BITS)) & 0xF;
                int current_group = global_col / group_size;
                S current_scale =
                    (current_group == base_group_idx) ? scale_val : scl[global_row * G_PADDED + current_group];
                uint32_t z = (zeros_val >> ((current_group % PACK_FACTOR) * BITS)) & 0xF;
                dequantized_val =
                    static_cast<T>((static_cast<float>(w) - static_cast<float>(z)) * static_cast<float>(current_scale));
            }


            int store_idx = swizzled_idx_base + inner_idx + k_load_stage * SB_SIZE;
            smemB[store_idx] = dequantized_val;
        }

        CP_ASYNC_COMMIT_GROUP();
    }
    CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
    __syncthreads();

    uint32_t RA[WARP_TILE_M][4];
    uint32_t RB[WARP_TILE_N][4];
    for (int k_load_base = (K_STAGE - 1) * BK; k_load_base < K; k_load_base += BK) {
        const int k_load_stage = k_load_base / BK;
        int smem_sel = (k_load_stage + 1) % K_STAGE;
        int smem_sel_next = k_load_stage % K_STAGE;

        // Load A matrix data with swizzle
        for (int load_idx = threadIdx.x * vec_size; load_idx < BM * BK; load_idx += blockDim.x * vec_size) {
            int smem_row = load_idx / BK;
            int smem_col = load_idx % BK;
            int global_row = global_m_base + smem_row;
            int global_col = k_load_base + smem_col;

            if (global_row < M && (global_col + vec_size - 1) < K) {
                int load_gmem_a_addr = global_row * K + global_col;
                int swizzled_col = swizzle_permuted_A_j(smem_row, smem_col);
                int swizzled_idx = smem_row * BK + swizzled_col;
                uint32_t load_smem_a_ptr = smem_a_base_ptr + (swizzled_idx + smem_sel_next * SA_SIZE) * sizeof(T);
                CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);
            }
        }

        constexpr int CHUNK_SIZE = 8;
        for (int load_idx = threadIdx.x / CHUNK_SIZE; load_idx < BN * BK / CHUNK_SIZE;
             load_idx += blockDim.x / CHUNK_SIZE) {
            const int inner_idx = threadIdx.x % CHUNK_SIZE;

            int smem_row = load_idx / (BK / CHUNK_SIZE);
            int smem_col_base = (load_idx % (BK / CHUNK_SIZE)) * CHUNK_SIZE;
            int global_row = global_n_base + smem_row;
            int global_col_base = k_load_base + smem_col_base;

            int32_t qwt_val = (global_row < N && global_col_base < K)
                                  ? qwt[global_row * K_PACKED + global_col_base / PACK_FACTOR]
                                  : 0;
            int base_group_idx = global_col_base / group_size;
            S scale_val = (global_row < N && global_col_base < K) ? scl[global_row * G_PADDED + base_group_idx] : S(0);
            int32_t zeros_val =
                (global_row < N && global_col_base < K) ? zos[global_row * G_PACKED + base_group_idx / PACK_FACTOR] : 0;

            int swizzled_col_base = swizzle_permuted_B_j(smem_row, smem_col_base);
            int swizzled_idx_base = smem_row * BK + swizzled_col_base;


            int global_col = global_col_base + inner_idx;
            T dequantized_val = T(0);

            if (global_row < N && global_col < K) {
                uint32_t w = (qwt_val >> ((global_col % PACK_FACTOR) * BITS)) & 0xF;
                int current_group = global_col / group_size;
                S current_scale =
                    (current_group == base_group_idx) ? scale_val : scl[global_row * G_PADDED + current_group];
                uint32_t z = (zeros_val >> ((current_group % PACK_FACTOR) * BITS)) & 0xF;
                dequantized_val =
                    static_cast<T>((static_cast<float>(w) - static_cast<float>(z)) * static_cast<float>(current_scale));
            }


            int store_idx = swizzled_idx_base + inner_idx + smem_sel_next * SB_SIZE;
            smemB[store_idx] = dequantized_val;
        }

        CP_ASYNC_COMMIT_GROUP();

        for (int TILE_K = 0; TILE_K < BK; TILE_K += WMMA_K) {
            // Read A matrix data from swizzled layout
            for (int i = 0; i < WARP_TILE_M; ++i) {
                int warp_smem_a_m = warp_m_id * WMMA_M * WARP_TILE_M + i * WMMA_M;
                int warp_smem_a_k = TILE_K;
                int base_row = warp_smem_a_m + (lane_id % 16);
                int base_col = warp_smem_a_k + (lane_id / 16) * vec_size;
                int swizzled_col = swizzle_permuted_A_j(base_row, base_col);
                T* lane_smem_a_ptr = smemA + base_row * BK + swizzled_col + smem_sel * SA_SIZE;
                uint32_t ptr = __cvta_generic_to_shared(lane_smem_a_ptr);
                LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], ptr);
            }
            // Read B matrix data from swizzled layout
            for (int i = 0; i < WARP_TILE_N; ++i) {
                int warp_smem_b_n = warp_n_id * WMMA_N * WARP_TILE_N + i * WMMA_N;
                int warp_smem_b_k = TILE_K;
                int base_row1 = warp_smem_b_n + (lane_id % 8);
                int base_col1 = warp_smem_b_k + (lane_id / 8) * vec_size;
                int swizzled_col1 = swizzle_permuted_B_j(base_row1, base_col1);
                T* lane_smem_b_ptr1 = smemB + base_row1 * BK + swizzled_col1 + smem_sel * SB_SIZE;
                uint32_t ptr1 = __cvta_generic_to_shared(lane_smem_b_ptr1);
                LDMATRIX_X2(RB[i][0], RB[i][1], ptr1);
                int base_row2 = warp_smem_b_n + 8 + (lane_id % 8);
                int base_col2 = warp_smem_b_k + (lane_id / 8) * vec_size;
                int swizzled_col2 = swizzle_permuted_B_j(base_row2, base_col2);
                T* lane_smem_b_ptr2 = smemB + base_row2 * BK + swizzled_col2 + smem_sel * SB_SIZE;
                uint32_t ptr2 = __cvta_generic_to_shared(lane_smem_b_ptr2);
                LDMATRIX_X2(RB[i][2], RB[i][3], ptr2);
            }
            for (int i = 0; i < WARP_TILE_M; ++i) {
                for (int j = 0; j < WARP_TILE_N; ++j) {
                    MMA161616_BF16(RC[i][j][0], RC[i][j][1], RC[i][j][2], RC[i][j][3], RC[i][j][4], RC[i][j][5],
                                   RC[i][j][6], RC[i][j][7], RA[i][0], RA[i][1], RA[i][2], RA[i][3], RB[j][0], RB[j][1],
                                   RB[j][2], RB[j][3], RC[i][j][0], RC[i][j][1], RC[i][j][2], RC[i][j][3], RC[i][j][4],
                                   RC[i][j][5], RC[i][j][6], RC[i][j][7]);
                }
            }
        }
        CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
        __syncthreads();
    }

    if ((K_STAGE - 2) > 0) {
        CP_ASYNC_WAIT_GROUP(0);
        __syncthreads();
    }

    // Compute remaining stages
    for (int k_load = 0; k_load < K_STAGE - 1; ++k_load) {
        const int stage_sel = ((K / BK - (K_STAGE - 1) + k_load) % K_STAGE);
        for (int TILE_K = 0; TILE_K < BK; TILE_K += WMMA_K) {
            for (int i = 0; i < WARP_TILE_M; ++i) {
                int warp_smem_a_m = warp_m_id * WMMA_M * WARP_TILE_M + i * WMMA_M;
                int warp_smem_a_k = TILE_K;
                int base_row = warp_smem_a_m + (lane_id % 16);
                int base_col = warp_smem_a_k + (lane_id / 16) * vec_size;
                int swizzled_col = swizzle_permuted_A_j(base_row, base_col);
                T* lane_smem_a_ptr = smemA + base_row * BK + swizzled_col + stage_sel * SA_SIZE;
                uint32_t ptr = __cvta_generic_to_shared(lane_smem_a_ptr);
                LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], ptr);
            }
            for (int i = 0; i < WARP_TILE_N; ++i) {
                int warp_smem_b_n = warp_n_id * WMMA_N * WARP_TILE_N + i * WMMA_N;
                int warp_smem_b_k = TILE_K;
                int base_row1 = warp_smem_b_n + (lane_id % 8);
                int base_col1 = warp_smem_b_k + (lane_id / 8) * vec_size;
                int swizzled_col1 = swizzle_permuted_B_j(base_row1, base_col1);
                T* lane_smem_b_ptr1 = smemB + base_row1 * BK + swizzled_col1 + stage_sel * SB_SIZE;
                uint32_t ptr1 = __cvta_generic_to_shared(lane_smem_b_ptr1);
                LDMATRIX_X2(RB[i][0], RB[i][1], ptr1);
                int base_row2 = warp_smem_b_n + 8 + (lane_id % 8);
                int base_col2 = warp_smem_b_k + (lane_id / 8) * vec_size;
                int swizzled_col2 = swizzle_permuted_B_j(base_row2, base_col2);
                T* lane_smem_b_ptr2 = smemB + base_row2 * BK + swizzled_col2 + stage_sel * SB_SIZE;
                uint32_t ptr2 = __cvta_generic_to_shared(lane_smem_b_ptr2);
                LDMATRIX_X2(RB[i][2], RB[i][3], ptr2);
            }
            for (int i = 0; i < WARP_TILE_M; ++i) {
                for (int j = 0; j < WARP_TILE_N; ++j) {
                    MMA161616_BF16(RC[i][j][0], RC[i][j][1], RC[i][j][2], RC[i][j][3], RC[i][j][4], RC[i][j][5],
                                   RC[i][j][6], RC[i][j][7], RA[i][0], RA[i][1], RA[i][2], RA[i][3], RB[j][0], RB[j][1],
                                   RB[j][2], RB[j][3], RC[i][j][0], RC[i][j][1], RC[i][j][2], RC[i][j][3], RC[i][j][4],
                                   RC[i][j][5], RC[i][j][6], RC[i][j][7]);
                }
            }
        }
    }

    // Store results
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            const int tile_m0 = global_m_base + (warp_m_id * WMMA_M * WARP_TILE_M) + i * WMMA_M;
            const int tile_n0 = global_n_base + (warp_n_id * WMMA_N * WARP_TILE_N) + j * WMMA_N;

            int group = lane_id >> 2;
            int tid4 = lane_id & 3;
            int row0 = group;
            int row1 = group + 8;
            int col0 = 2 * tid4;
            int col1 = 2 * tid4 + 1;
            int col2 = 2 * tid4 + 8;
            int col3 = 2 * tid4 + 9;

            float v0 = __uint_as_float(RC[i][j][0]);
            float v1 = __uint_as_float(RC[i][j][1]);
            float v2 = __uint_as_float(RC[i][j][2]);
            float v3 = __uint_as_float(RC[i][j][3]);
            float v4 = __uint_as_float(RC[i][j][4]);
            float v5 = __uint_as_float(RC[i][j][5]);
            float v6 = __uint_as_float(RC[i][j][6]);
            float v7 = __uint_as_float(RC[i][j][7]);

            // Add bias to FP32 accumulators before the single output rounding.
            // Partial output tiles must not read bias past the live columns.
            if (bias) {
                const float b0 = tile_n0 + col0 < N ? static_cast<float>(bias[tile_n0 + col0]) : 0.0f;
                const float b1 = tile_n0 + col1 < N ? static_cast<float>(bias[tile_n0 + col1]) : 0.0f;
                const float b2 = tile_n0 + col2 < N ? static_cast<float>(bias[tile_n0 + col2]) : 0.0f;
                const float b3 = tile_n0 + col3 < N ? static_cast<float>(bias[tile_n0 + col3]) : 0.0f;
                v0 += b0; v2 += b0;
                v1 += b1; v3 += b1;
                v4 += b2; v6 += b2;
                v5 += b3; v7 += b3;
            }

            if ((tile_m0 + row0) < M && (tile_n0 + col0) < N)
                C[(tile_m0 + row0) * N + (tile_n0 + col0)] = static_cast<T>(v0);
            if ((tile_m0 + row0) < M && (tile_n0 + col1) < N)
                C[(tile_m0 + row0) * N + (tile_n0 + col1)] = static_cast<T>(v1);
            if ((tile_m0 + row1) < M && (tile_n0 + col0) < N)
                C[(tile_m0 + row1) * N + (tile_n0 + col0)] = static_cast<T>(v2);
            if ((tile_m0 + row1) < M && (tile_n0 + col1) < N)
                C[(tile_m0 + row1) * N + (tile_n0 + col1)] = static_cast<T>(v3);
            if ((tile_m0 + row0) < M && (tile_n0 + col2) < N)
                C[(tile_m0 + row0) * N + (tile_n0 + col2)] = static_cast<T>(v4);
            if ((tile_m0 + row0) < M && (tile_n0 + col3) < N)
                C[(tile_m0 + row0) * N + (tile_n0 + col3)] = static_cast<T>(v5);
            if ((tile_m0 + row1) < M && (tile_n0 + col2) < N)
                C[(tile_m0 + row1) * N + (tile_n0 + col2)] = static_cast<T>(v6);
            if ((tile_m0 + row1) < M && (tile_n0 + col3) < N)
                C[(tile_m0 + row1) * N + (tile_n0 + col3)] = static_cast<T>(v7);
        }
    }
}

template <typename T,
          typename S,
          int BLOCK_N_GEMV>
__global__ void matmul_awq_gemv_kernel_M_1(const T* __restrict__ inp,
                                           const int32_t* __restrict__ qwt,
                                           const S* __restrict__ scl,        // Scales [N, G_padded]
                                           const int32_t* __restrict__ zos,  // Zeros [N, G/8]
                                           T* __restrict__ out,
                                           int K, int N, int group_size,
                                           int G_PADDED,
                                           const T* __restrict__ bias) {

    static_assert(BLOCK_N_GEMV % WARP_SIZE == 0, "BLOCK_N_GEMV must be a multiple of WARP_SIZE");
    const int G = K / group_size;
    const int K_PACKED = (K + PACK_FACTOR - 1) / PACK_FACTOR;
    const int G_PACKED = (G + PACK_FACTOR - 1) / PACK_FACTOR;


    const int warps_per_block = BLOCK_N_GEMV / WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int n = blockIdx.x * warps_per_block + warp_id;

    if (n >= N)
        return;


    extern __shared__ char sh_mem_raw[];
    T* sh_inp = reinterpret_cast<T*>(sh_mem_raw);  // sh_inp [K]
    S* sh_scl = reinterpret_cast<S*>(&sh_inp[K]);  // sh_scl [warps_per_block][G_PADDED]
    int32_t* sh_zos =
        reinterpret_cast<int32_t*>(&sh_scl[warps_per_block * G_PADDED]);  // sh_zos [warps_per_block][G_PACKED]


    for (int k_idx = threadIdx.x; k_idx < K; k_idx += BLOCK_N_GEMV) {
        sh_inp[k_idx] = inp[k_idx];
    }

    for (int g = lane; g < G; g += WARP_SIZE) {

        sh_scl[warp_id * G_PADDED + g] = scl[n * G_PADDED + g];
    }

    for (int packed_g = lane; packed_g < G_PACKED; packed_g += WARP_SIZE) {
        sh_zos[warp_id * G_PACKED + packed_g] = zos[n * G_PACKED + packed_g];
    }
    __syncthreads();


    float acc = 0.0f;
    for (int k = lane; k < K; k += WARP_SIZE) {
        float iv = static_cast<float>(sh_inp[k]);

        int g = k / group_size;

        S current_s = sh_scl[warp_id * G_PADDED + g];

        int packed_g = g / PACK_FACTOR;
        int32_t packed_z_val = sh_zos[warp_id * G_PACKED + packed_g];
        int inner_g = g % PACK_FACTOR;
        int shift_z = inner_g * BITS;
        uint32_t z = (packed_z_val >> shift_z) & ((1 << BITS) - 1);


        int packed_k = k / PACK_FACTOR;
        int32_t packed_w_val = qwt[n * K_PACKED + packed_k];
        int inner_k = k % PACK_FACTOR;
        int shift_w = inner_k * BITS;
        uint32_t w = (packed_w_val >> shift_w) & ((1 << BITS) - 1);

        float scale_val = static_cast<float>(current_s);
        acc = __fmaf_rn(iv, (static_cast<float>(w) - static_cast<float>(z)) * scale_val, acc);
    }


#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }


    if (lane == 0) {
        if (bias) {
            acc += static_cast<float>(bias[n]);
        }
        out[n] = static_cast<T>(acc);
    }
}
template <int BLOCK_N_GEMV, typename ScaleType = float>
__global__ void matmul_awq_gemv_bf16_vectorized_kernel(
    const __nv_bfloat16* __restrict__ inp,
    const int32_t* __restrict__ qwt,
    const ScaleType* __restrict__ scl,      // Scales [N, G_padded] (ScaleType)
    const int32_t* __restrict__ zos,        // Zeros [N, G/8]
    __nv_bfloat16* __restrict__ out,
    int K, int N, int group_size,
    int G_PADDED,
    const __nv_bfloat16* __restrict__ bias)
{

    static_assert(BLOCK_N_GEMV % WARP_SIZE == 0, "BLOCK_N_GEMV must be a multiple of WARP_SIZE");
    const int G = K / group_size;
    const int K_PACKED = (K + PACK_FACTOR - 1) / PACK_FACTOR;
    const int G_PACKED = (G + PACK_FACTOR - 1) / PACK_FACTOR;
    constexpr int K_PER_THREAD = 8;
    constexpr int K_PER_WARP = WARP_SIZE * K_PER_THREAD;


    const int warps_per_block = BLOCK_N_GEMV / WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int n = blockIdx.x * warps_per_block + warp_id;

    if (n >= N)
        return;


    extern __shared__ char sh_mem_raw[];
    __nv_bfloat16* sh_inp = reinterpret_cast<__nv_bfloat16*>(sh_mem_raw);
    ScaleType* sh_scl = reinterpret_cast<ScaleType*>(&sh_inp[K]);
    int32_t* sh_zos = reinterpret_cast<int32_t*>(&sh_scl[warps_per_block * G_PADDED]);
    constexpr int vec_unit = 16 / sizeof(__nv_bfloat16);


    for (int k_idx = threadIdx.x; k_idx < K / vec_unit; k_idx += BLOCK_N_GEMV) {
        reinterpret_cast<float4*>(sh_inp)[k_idx] = reinterpret_cast<const float4*>(inp)[k_idx];
    }

    for (int g = lane; g < G; g += WARP_SIZE) {
        sh_scl[warp_id * G_PADDED + g] = scl[n * G_PADDED + g];
    }

    for (int packed_g = lane; packed_g < G_PACKED; packed_g += WARP_SIZE) {
        sh_zos[warp_id * G_PACKED + packed_g] = zos[n * G_PACKED + packed_g];
    }
    __syncthreads();


    float acc = 0.0f;


    for (int k_block_start = 0; k_block_start < K; k_block_start += K_PER_WARP) {

        int k_thread_start = k_block_start + lane * K_PER_THREAD;


        int packed_k_idx = k_thread_start / PACK_FACTOR;
        int32_t packed_w_val = 0;

        if (k_thread_start < K && packed_k_idx < K_PACKED) {
            packed_w_val = qwt[n * K_PACKED + packed_k_idx];
        }


        int g = k_thread_start / group_size;
        float current_s = 0.0f;
        uint32_t z_vals[K_PER_THREAD];


        if (k_thread_start < K && g < G) {
            current_s = sh_scl[warp_id * G_PADDED + g];
            int packed_g = g / PACK_FACTOR;
            int32_t packed_z_val = sh_zos[warp_id * G_PACKED + packed_g];


            int inner_g = g % PACK_FACTOR;
            int shift_z = inner_g * BITS;
            uint32_t z = (packed_z_val >> shift_z) & ((1 << BITS) - 1);


#pragma unroll
            for (int i = 0; i < K_PER_THREAD; ++i) {
                z_vals[i] = z;
            }
        } else {

#pragma unroll
            for (int i = 0; i < K_PER_THREAD; ++i) {
                z_vals[i] = 0;
            }
        }


#pragma unroll
        for (int i = 0; i < K_PER_THREAD; ++i) {
            int current_k = k_thread_start + i;


            if (current_k < K) {

                float iv = __bfloat162float(sh_inp[current_k]);


                int inner_k = current_k % PACK_FACTOR;
                int shift_w = inner_k * BITS;
                uint32_t w = (packed_w_val >> shift_w) & ((1 << BITS) - 1);


                uint32_t z = z_vals[i];


                float scale_val = static_cast<float>(current_s);
                acc = __fmaf_rn(iv, (static_cast<float>(w) - static_cast<float>(z)) * scale_val, acc);
            }

        }
    }


#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }


    if (lane == 0) {
        if (bias) {
            acc += __bfloat162float(bias[n]);
        }
        out[n] = __float2bfloat16_rn(acc);
    }
}

}  // namespace cuda_OP

namespace op::cuda::detail {
namespace {

template <typename T>
__global__ void awq_scalar_kernel(const T* input, const int32_t* packed,
    const T* scales, const int32_t* zeros, const T* bias, T* output,
    int rows, int width, int features, int group_size, int padded_groups) {
  const std::size_t count = static_cast<std::size_t>(rows) * width;
  const std::size_t begin = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t step = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  const int weight_stride = (features + 7) / 8;
  const int zero_stride = (features / group_size + 7) / 8;
  for (std::size_t index = begin; index < count; index += step) {
    const int row = index / width, column = index % width;
    float result = bias ? static_cast<float>(bias[column]) : 0.0f;
    for (int feature = 0; feature < features; ++feature) {
      const int group = feature / group_size;
      const uint32_t value = (static_cast<uint32_t>(packed[column * weight_stride + feature / 8]) >>
                                ((feature % 8) * 4)) & 15;
      const uint32_t zero = (static_cast<uint32_t>(zeros[column * zero_stride + group / 8]) >>
                               ((group % 8) * 4)) & 15;
      float weight = (static_cast<float>(value) - static_cast<float>(zero)) *
                      static_cast<float>(scales[column * padded_groups + group]);
      if (rows > 1) weight = static_cast<float>(static_cast<T>(weight));
      result = fmaf(static_cast<float>(input[row * features + feature]), weight, result);
    }
    output[index] = static_cast<T>(result);
  }
}
}  // namespace

template <typename T>
void launch_awq(const ExecutionContext& context, TensorView<const T, 2> input,
                 const AwqLinearWeight<T>& weight, TensorView<T, 2> output) {
  const int rows = input.shape[0], features = weight.input_features, width = weight.output_features;
  const int groups = features / weight.group_size;
  const std::size_t scratch = static_cast<std::size_t>(features) * sizeof(T) +
      8 * static_cast<std::size_t>(weight.padded_groups) * sizeof(T) +
      8 * static_cast<std::size_t>((groups + 7) / 8) * sizeof(int32_t);
  bool launched = false;
  if (rows == 1 && features % 8 == 0 && width % 8 == 0 &&
      reinterpret_cast<std::uintptr_t>(input.data) % 16 == 0 &&
      (!std::is_same_v<T, __nv_bfloat16> || weight.group_size % 8 == 0) &&
      scratch <= context.shared_memory_limit) {
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      cuda_OP::matmul_awq_gemv_bf16_vectorized_kernel<256, T>
          <<<(width + 7) / 8, 256, scratch, context.stream>>>(input.data, weight.qweight,
              weight.scales, weight.zeros, output.data, features, width, weight.group_size,
              weight.padded_groups, weight.bias);
    } else {
      cuda_OP::matmul_awq_gemv_kernel_M_1<T, T, 256>
          <<<(width + 7) / 8, 256, scratch, context.stream>>>(input.data, weight.qweight,
              weight.scales, weight.zeros, output.data, features, width, weight.group_size,
              weight.padded_groups, weight.bias);
    }
    launched = true;
  }
  if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    if (!launched && rows > 1 && features >= 32 && features % 16 == 0 && groups % 8 == 0 &&
        reinterpret_cast<std::uintptr_t>(input.data) % 16 == 0) {
      constexpr int BM = 32, BN = 128, BK = 16, WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
      constexpr int WARPS = BM / WMMA_M * BN / WMMA_N;
      cuda_OP::awq_gemm_kernel_mma<T, T, BM, BN, BK, WMMA_M, WMMA_N, WMMA_K, WARPS, 2, 1, 1>
          <<<dim3((rows + BM - 1) / BM, (width + BN - 1) / BN), WARPS * 32, 0, context.stream>>>(
              input.data, weight.qweight, weight.scales, weight.zeros, output.data,
              rows, width, features, weight.group_size, weight.padded_groups, weight.bias);
      launched = true;
    }
  }
  if (!launched) {
    const auto blocks = std::min<std::size_t>((output.numel() + 255) / 256,
                                              context.multiprocessors * 32);
    awq_scalar_kernel<T><<<blocks, 256, 0, context.stream>>>(input.data, weight.qweight,
        weight.scales, weight.zeros, weight.bias, output.data, rows, width, features,
        weight.group_size, weight.padded_groups);
  }
  const auto result = cudaGetLastError();
  if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

template void launch_awq<float>(const ExecutionContext&, TensorView<const float, 2>, const AwqLinearWeight<float>&, TensorView<float, 2>);
template void launch_awq<__nv_bfloat16>(const ExecutionContext&, TensorView<const __nv_bfloat16, 2>, const AwqLinearWeight<__nv_bfloat16>&, TensorView<__nv_bfloat16, 2>);

}  // namespace op::cuda::detail
