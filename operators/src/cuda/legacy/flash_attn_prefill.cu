#include <float.h>
#include <mma.h>

#include <stdexcept>
#include <string>

#include "cuda/legacy/legacy_cuda_api.cuh"
#include "ptx_common.h"

__device__ __forceinline__ float warpReduceSum(float val) {
#pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return __shfl_sync(0xFFFFFFFF, val, 0);
}

__device__ __forceinline__ float warpReduceMax(float val) {
#pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val = max(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return __shfl_sync(0xFFFFFFFF, val, 0);
}

namespace cuda_OP {


template <typename T, int B_c, int B_r, int T_r, int WARP_NUM = 4, int DQKV = 128>
__global__ void flash_attn_prefill_kernel_v0(const T* __restrict__ q_global, const T* __restrict__ k_global,
                                             const T* __restrict__ v_global, T* __restrict__ out_global,
                                             int num_q_heads_total, int num_kv_heads_total, int GQA_n_group,
                                             int current_prefill_q_length, int current_kv_cache_total_len,
                                             int q_offset_in_kv_timeline, int q_stride) {

    const int q_segment_idx = blockIdx.x;
    const int q_head_idx_global = blockIdx.y;
    const int kv_head_idx_global = q_head_idx_global / GQA_n_group;


    const int tid = threadIdx.x;
    const int warp_id = tid / warpSize;
    const int lane_id = tid % warpSize;


    const int q_segment_start_idx = q_segment_idx * T_r;


    __shared__ T q_smem[B_r][DQKV];
    __shared__ T k_smem[B_c][DQKV];
    __shared__ T v_smem[B_c][DQKV];
    __shared__ float scores_smem[B_r][B_c];
    __shared__ T o_smem[B_r][DQKV];
    __shared__ float m_stats[B_r];
    __shared__ float l_stats[B_r];


    for (int q_block_offset = 0; q_block_offset < T_r; q_block_offset += B_r) {

        for (int row_idx = tid; row_idx < B_r; row_idx += blockDim.x) {
            m_stats[row_idx] = -FLT_MAX;
            l_stats[row_idx] = 0.0f;
            for (int dim_idx = 0; dim_idx < DQKV; ++dim_idx) {
                o_smem[row_idx][dim_idx] = static_cast<T>(0.0f);
            }
        }
        __syncthreads();


        for (int q_smem_row = 0; q_smem_row < B_r; ++q_smem_row) {
            int q_token_idx = q_segment_start_idx + q_block_offset + q_smem_row;

            bool is_valid_q = (q_token_idx < current_prefill_q_length);
            const T* q_global_ptr = q_global + q_token_idx * q_stride + q_head_idx_global * DQKV;
            for (int dim_idx = tid; dim_idx < DQKV; dim_idx += blockDim.x) {
                if (is_valid_q) {
                    q_smem[q_smem_row][dim_idx] = q_global_ptr[dim_idx];
                } else {
                    q_smem[q_smem_row][dim_idx] = static_cast<T>(0.0f);
                }
            }
        }
        __syncthreads();


        for (int kv_block_offset = 0; kv_block_offset < current_kv_cache_total_len; kv_block_offset += B_c) {

            for (int smem_row = 0; smem_row < B_c; ++smem_row) {
                int k_token_idx = kv_block_offset + smem_row;
                bool is_valid_k = (k_token_idx < current_kv_cache_total_len);
                const T* k_global_ptr = k_global + (k_token_idx * num_kv_heads_total + kv_head_idx_global) * DQKV;
                for (int dim_idx = tid; dim_idx < DQKV; dim_idx += blockDim.x) {
                    if (is_valid_k) {
                        k_smem[smem_row][dim_idx] = k_global_ptr[dim_idx];
                    } else {
                        k_smem[smem_row][dim_idx] = static_cast<T>(0.0f);
                    }
                }
            }
            for (int smem_row = 0; smem_row < B_c; ++smem_row) {
                int v_token_idx = kv_block_offset + smem_row;
                bool is_valid_v = (v_token_idx < current_kv_cache_total_len);
                const T* v_global_ptr = v_global + (v_token_idx * num_kv_heads_total + kv_head_idx_global) * DQKV;
                for (int dim_idx = tid; dim_idx < DQKV; dim_idx += blockDim.x) {
                    if (is_valid_v) {
                        v_smem[smem_row][dim_idx] = v_global_ptr[dim_idx];
                    } else {
                        v_smem[smem_row][dim_idx] = static_cast<T>(0.0f);
                    }
                }
            }
            __syncthreads();


            constexpr int Q_ROWS_PER_WARP = (B_r + WARP_NUM - 1) / WARP_NUM;
            for (int q_row_in_warp = 0; q_row_in_warp < Q_ROWS_PER_WARP; ++q_row_in_warp) {

                int q_smem_row = warp_id * Q_ROWS_PER_WARP + q_row_in_warp;

                if (q_smem_row < B_r) {
                    int q_token_idx = q_segment_start_idx + q_block_offset + q_smem_row;
                    int q_abs_pos = q_offset_in_kv_timeline + q_token_idx;
                    for (int k_smem_row = 0; k_smem_row < B_c; ++k_smem_row) {
                        float dot_sum = 0.0f;
                        for (int d_idx = lane_id; d_idx < DQKV; d_idx += warpSize) {
                            dot_sum += static_cast<float>(q_smem[q_smem_row][d_idx]) *
                                       static_cast<float>(k_smem[k_smem_row][d_idx]);
                        }
                        float score = warpReduceSum(dot_sum);
                        score *= (1.0f / sqrtf(static_cast<float>(DQKV)));

                        if (lane_id == 0) {
                            int k_token_idx = kv_block_offset + k_smem_row;
                            bool q_is_padding = (q_token_idx >= current_prefill_q_length);
                            bool k_is_padding = (k_token_idx >= current_kv_cache_total_len);
                            bool masked_by_causal = (k_token_idx > q_abs_pos);

                            if (q_is_padding || k_is_padding || masked_by_causal) {
                                scores_smem[q_smem_row][k_smem_row] = -FLT_MAX;
                            } else {
                                scores_smem[q_smem_row][k_smem_row] = score;
                            }
                        }
                    }
                }
            }
            __syncthreads();


            for (int q_row_in_warp = 0; q_row_in_warp < Q_ROWS_PER_WARP; ++q_row_in_warp) {
                int q_smem_row = warp_id * Q_ROWS_PER_WARP + q_row_in_warp;
                if (q_smem_row >= B_r)
                    continue;

                int q_token_idx = q_segment_start_idx + q_block_offset + q_smem_row;
                if (q_token_idx >= current_prefill_q_length)
                    continue;

                float m_prev = m_stats[q_smem_row];
                float l_prev = l_stats[q_smem_row];

                float m_block = -FLT_MAX;
                for (int col_idx = lane_id; col_idx < B_c; col_idx += warpSize) {
                    m_block = max(m_block, scores_smem[q_smem_row][col_idx]);
                }
                m_block = warpReduceMax(m_block);


                float m_new = max(m_prev, m_block);


                // online softmax
                // m_new = max(m_old, m_block)
                // scale = exp(m_old - m_new)
                // l_new = new_scale * l_old + l_block_sum


                float l_block = 0.0f;
                for (int col_idx = lane_id; col_idx < B_c; col_idx += warpSize) {
                    float s = scores_smem[q_smem_row][col_idx];
                    float p = (s == -FLT_MAX) ? 0.0f : expf(s - m_new);
                    scores_smem[q_smem_row][col_idx] = p;
                    l_block += p;
                }
                l_block = warpReduceSum(l_block);


                float scale_prev;
                if (m_prev == -FLT_MAX) {
                    scale_prev = 0.0f;
                } else if (m_prev == m_new) {
                    scale_prev = 1.0f;
                } else {
                    scale_prev = expf(m_prev - m_new);
                }


                float l_new = l_prev * scale_prev + l_block;


                for (int d_idx = lane_id; d_idx < DQKV; d_idx += warpSize) {
                    float pv_sum = 0.0f;
                    for (int k_smem_row = 0; k_smem_row < B_c; ++k_smem_row) {
                        pv_sum += scores_smem[q_smem_row][k_smem_row] * static_cast<float>(v_smem[k_smem_row][d_idx]);
                    }


                    float o_old = static_cast<float>(o_smem[q_smem_row][d_idx]);
                    float o_new = o_old * scale_prev + pv_sum;
                    o_smem[q_smem_row][d_idx] = static_cast<T>(o_new);
                }


                if (lane_id == 0) {
                    m_stats[q_smem_row] = m_new;
                    l_stats[q_smem_row] = l_new;
                }
            }
            __syncthreads();
        }


        constexpr int Q_ROWS_PER_WARP_WRITE = (B_r + WARP_NUM - 1) / WARP_NUM;
        for (int q_row_in_warp = 0; q_row_in_warp < Q_ROWS_PER_WARP_WRITE; ++q_row_in_warp) {
            int q_smem_row = warp_id * Q_ROWS_PER_WARP_WRITE + q_row_in_warp;

            if (q_smem_row >= B_r)
                continue;

            int q_token_idx = q_segment_start_idx + q_block_offset + q_smem_row;

            if (q_token_idx < current_prefill_q_length) {
                float l_final = l_stats[q_smem_row];

                float inv_l_final = (l_final > 1e-6f) ? (1.0f / l_final) : 0.0f;
                T* out_global_ptr = out_global + (q_token_idx * num_q_heads_total + q_head_idx_global) * DQKV;
                for (int d_idx = lane_id; d_idx < DQKV; d_idx += warpSize) {
                    float final_output = static_cast<float>(o_smem[q_smem_row][d_idx]) * inv_l_final;
                    out_global_ptr[d_idx] = static_cast<T>(final_output);
                }
            }
        }
        __syncthreads();
    }
}

template <typename T, int B_c, int B_r, int T_r, int WARP_NUM = 4, int DQKV = 128>
__global__ void flash_attn_prefill_kernel_v1(const T* __restrict__ q_global, const T* __restrict__ k_global,
                                             const T* __restrict__ v_global, T* __restrict__ out_global,
                                             int num_q_heads_total, int num_kv_heads_total, int GQA_n_group,
                                             int current_prefill_q_length, int current_kv_cache_total_len,
                                             int q_offset_in_kv_timeline, int q_stride) {

    const int q_segment_idx = blockIdx.x;
    const int q_head_idx_global = blockIdx.y;
    const int kv_head_idx_global = q_head_idx_global / GQA_n_group;
    using namespace nvcuda;
    using FragmentA = wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major>;
    using FragmentB = wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::col_major>;
    using FragmentC = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;


    const int tid = threadIdx.x;
    const int warp_id = tid / warpSize;
    const int lane_id = tid % warpSize;


    const int q_segment_start_idx = q_segment_idx * T_r;


    __shared__ T q_smem[B_r][DQKV];
    __shared__ T k_smem[B_c][DQKV];
    __shared__ T v_smem[B_c][DQKV];
    __shared__ float scores_smem[B_r][B_c];
    __shared__ T o_smem[B_r][DQKV];
    __shared__ float m_stats[B_r];
    __shared__ float l_stats[B_r];


    for (int q_block_offset = 0; q_block_offset < T_r; q_block_offset += B_r) {

        for (int row_idx = 0; row_idx < B_r; ++row_idx) {
            m_stats[row_idx] = -FLT_MAX;
            l_stats[row_idx] = 0.0f;
            for (int dim_idx = tid; dim_idx < DQKV; dim_idx += blockDim.x) {
                o_smem[row_idx][dim_idx] = static_cast<T>(0.0f);
            }
        }


        for (int q_smem_row = 0; q_smem_row < B_r; ++q_smem_row) {
            int q_token_idx = q_segment_start_idx + q_block_offset + q_smem_row;
            bool is_valid_q = (q_token_idx < current_prefill_q_length);
            const T* q_global_ptr = q_global + q_token_idx * q_stride + q_head_idx_global * DQKV;
            for (int dim_idx = tid; dim_idx < DQKV; dim_idx += blockDim.x) {
                if (is_valid_q) {
                    q_smem[q_smem_row][dim_idx] = q_global_ptr[dim_idx];
                } else {
                    q_smem[q_smem_row][dim_idx] = static_cast<T>(0.0f);
                }
            }
        }


        for (int kv_block_offset = 0; kv_block_offset < current_kv_cache_total_len; kv_block_offset += B_c) {

            for (int smem_row = 0; smem_row < B_c; ++smem_row) {
                int k_token_idx = kv_block_offset + smem_row;
                bool is_valid_kv = (k_token_idx < current_kv_cache_total_len);
                const T* k_global_ptr = k_global + (k_token_idx * num_kv_heads_total + kv_head_idx_global) * DQKV;
                const T* v_global_ptr = v_global + (k_token_idx * num_kv_heads_total + kv_head_idx_global) * DQKV;
                for (int dim_idx = tid; dim_idx < DQKV; dim_idx += blockDim.x) {
                    if (is_valid_kv) {
                        k_smem[smem_row][dim_idx] = k_global_ptr[dim_idx];
                        v_smem[smem_row][dim_idx] = v_global_ptr[dim_idx];
                    } else {
                        k_smem[smem_row][dim_idx] = static_cast<T>(0.0f);
                        v_smem[smem_row][dim_idx] = static_cast<T>(0.0f);
                    }
                }
            }
            __syncthreads();


            const int WARP_M = B_r / 16;
            const int WARP_N = B_c / 16;
            const int warp_m_id = warp_id / WARP_N;
            const int warp_n_id = warp_id % WARP_N;

            if (warp_m_id < WARP_M) {
                FragmentC fragC;
                wmma::fill_fragment(fragC, 0.0f);

                for (int k_step = 0; k_step < DQKV; k_step += 16) {
                    FragmentA fragA_load;
                    FragmentB fragB_load;
                    const T* smemA_ptr = &q_smem[warp_m_id * 16][k_step];
                    const T* smemB_ptr = &k_smem[warp_n_id * 16][k_step];

                    wmma::load_matrix_sync(fragA_load, smemA_ptr, DQKV);


                    wmma::load_matrix_sync(fragB_load, smemB_ptr, DQKV);
                    wmma::mma_sync(fragC, fragA_load, fragB_load, fragC);
                }

                const int out_m_base = warp_m_id * 16;
                const int out_n_base = warp_n_id * 16;
                wmma::store_matrix_sync(&scores_smem[out_m_base][out_n_base], fragC, B_c, wmma::mem_row_major);
            }
            __syncthreads();

            const float inv_dk = (1.0f / sqrtf(static_cast<float>(DQKV)));
            for (int i = tid; i < B_r * B_c; i += blockDim.x) {
                int q_smem_row = i / B_c;
                int k_smem_row = i % B_c;

                int q_token_idx_local = q_segment_start_idx + q_block_offset + q_smem_row;
                int k_token_idx_local = kv_block_offset + k_smem_row;


                int q_abs_pos = q_offset_in_kv_timeline + q_token_idx_local;

                bool is_q_padding = (q_token_idx_local >= current_prefill_q_length);
                bool is_k_padding = (k_token_idx_local >= current_kv_cache_total_len);
                bool is_masked_by_causal = (k_token_idx_local > q_abs_pos);

                if (is_q_padding || is_k_padding || is_masked_by_causal) {
                    scores_smem[q_smem_row][k_smem_row] = -FLT_MAX;
                } else {
                    scores_smem[q_smem_row][k_smem_row] *= inv_dk;
                }
            }
            __syncthreads();


            constexpr int Q_ROWS_PER_WARP = (B_r + WARP_NUM - 1) / WARP_NUM;
            for (int q_row_in_warp = 0; q_row_in_warp < Q_ROWS_PER_WARP; ++q_row_in_warp) {
                int q_smem_row = warp_id * Q_ROWS_PER_WARP + q_row_in_warp;
                if (q_smem_row >= B_r)
                    continue;

                int q_token_idx = q_segment_start_idx + q_block_offset + q_smem_row;
                if (q_token_idx >= current_prefill_q_length)
                    continue;

                float m_prev = m_stats[q_smem_row];
                float l_prev = l_stats[q_smem_row];

                float m_block = -FLT_MAX;
                for (int col_idx = lane_id; col_idx < B_c; col_idx += warpSize) {
                    m_block = max(m_block, scores_smem[q_smem_row][col_idx]);
                }
                m_block = warpReduceMax(m_block);

                float m_new = max(m_prev, m_block);

                float l_block = 0.0f;
                for (int col_idx = lane_id; col_idx < B_c; col_idx += warpSize) {
                    float s = scores_smem[q_smem_row][col_idx];

                    float p = expf(s - m_new);
                    scores_smem[q_smem_row][col_idx] = p;
                    l_block += p;
                }
                l_block = warpReduceSum(l_block);

                float scale_prev = expf(m_prev - m_new);
                float l_new = l_prev * scale_prev + l_block;

                for (int d_idx = lane_id; d_idx < DQKV; d_idx += warpSize) {
                    float pv_sum = 0.0f;
                    for (int k_smem_row = 0; k_smem_row < B_c; ++k_smem_row) {
                        pv_sum += scores_smem[q_smem_row][k_smem_row] * static_cast<float>(v_smem[k_smem_row][d_idx]);
                    }
                    float o_old = static_cast<float>(o_smem[q_smem_row][d_idx]);
                    float o_new = o_old * scale_prev + pv_sum;
                    o_smem[q_smem_row][d_idx] = static_cast<T>(o_new);
                }

                if (lane_id == 0) {
                    m_stats[q_smem_row] = m_new;
                    l_stats[q_smem_row] = l_new;
                }
            }
            __syncthreads();
        }


        for (int q_smem_row = 0; q_smem_row < B_r; ++q_smem_row) {
            int q_token_idx = q_segment_start_idx + q_block_offset + q_smem_row;

            if (q_token_idx < current_prefill_q_length) {
                float l_final = l_stats[q_smem_row];
                float inv_l_final = (l_final > 1e-6f) ? (1.0f / l_final) : 0.0f;
                T* out_global_ptr = out_global + (q_token_idx * num_q_heads_total + q_head_idx_global) * DQKV;
                for (int d_idx = tid; d_idx < DQKV; d_idx += blockDim.x) {
                    float final_output = static_cast<float>(o_smem[q_smem_row][d_idx]) * inv_l_final;
                    out_global_ptr[d_idx] = static_cast<T>(final_output);
                }
            }
        }
        __syncthreads();
    }
}
template <typename T, int B_c, int B_r, int T_r, int WARP_NUM = 4, int DQKV = 128, int STAGE = 1>
__global__ void flash_attn_prefill_kernel_v2(const T* __restrict__ q_global, const T* __restrict__ k_global,
                                             const T* __restrict__ v_global, T* __restrict__ out_global,
                                             int num_q_heads_total, int num_kv_heads_total, int GQA_n_group,
                                             int current_prefill_q_length, int current_kv_cache_total_len,
                                             int q_offset_in_kv_timeline, int q_stride) {

    const int q_segment_idx = blockIdx.x;
    const int q_head_idx_global = blockIdx.y;
    const int kv_head_idx_global = q_head_idx_global / GQA_n_group;
    using namespace nvcuda;
    using FragmentA = wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major>;
    using FragmentB = wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::col_major>;
    using FragmentB_t = wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::row_major>;
    using FragmentC = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;


    const int tid = threadIdx.x;
    const int warp_id = tid / warpSize;
    const int lane_id = tid % warpSize;


    const int q_segment_start_idx = q_segment_idx * T_r;

    constexpr auto align_size = [](size_t size) { return ((size + 15) / 16) * 16; };

    const size_t q_smem_size = align_size(B_r * DQKV * sizeof(T));
    const size_t k_smem_size = align_size(B_c * DQKV * sizeof(T));
    const size_t v_smem_size = align_size(B_c * DQKV * sizeof(T));
    const size_t o_smem_size = align_size(B_r * DQKV * sizeof(float));
    const size_t stats_size = align_size(B_r * 2 * sizeof(float));
    const size_t scores_smem_size = align_size(B_r * B_c * sizeof(float));
    const size_t p_smem_size = align_size(B_r * B_c * sizeof(T));
    const size_t pv_smem_size = align_size(B_r * DQKV * sizeof(float));

    const size_t k_smem_offset = q_smem_size;
    const size_t v_smem_offset = k_smem_offset + k_smem_size;
    const size_t o_smem_offset = v_smem_offset + v_smem_size;
    const size_t stats_offset = o_smem_offset + o_smem_size;
    const size_t scores_smem_offset = stats_offset + stats_size;
    const size_t p_smem_offset = scores_smem_offset + scores_smem_size;
    const size_t pv_smem_offset = p_smem_offset + p_smem_size;

    extern __shared__ unsigned char smem_buffer[];

    T* q_smem = reinterpret_cast<T*>(smem_buffer);
    T* k_smem = reinterpret_cast<T*>(smem_buffer + k_smem_offset);
    T* v_smem = reinterpret_cast<T*>(smem_buffer + v_smem_offset);
    float* o_smem = reinterpret_cast<float*>(smem_buffer + o_smem_offset);
    float* m_stats = reinterpret_cast<float*>(smem_buffer + stats_offset);
    float* l_stats = reinterpret_cast<float*>(smem_buffer + stats_offset + B_r * sizeof(float));
    float* scores_smem = reinterpret_cast<float*>(smem_buffer + scores_smem_offset);
    T* p_smem = reinterpret_cast<T*>(smem_buffer + p_smem_offset);
    float* pv_smem = reinterpret_cast<float*>(smem_buffer + pv_smem_offset);


    for (int q_block_offset = 0; q_block_offset < T_r; q_block_offset += B_r) {

        for (int row_idx = 0; row_idx < B_r; ++row_idx) {
            m_stats[row_idx] = -FLT_MAX;
            l_stats[row_idx] = 0.0f;
            for (int dim_idx = tid; dim_idx < DQKV; dim_idx += blockDim.x) {
                o_smem[row_idx * DQKV + dim_idx] = 0.0f;
            }
        }


        for (int q_smem_row = 0; q_smem_row < B_r; ++q_smem_row) {
            int q_token_idx = q_segment_start_idx + q_block_offset + q_smem_row;
            bool is_valid_q = (q_token_idx < current_prefill_q_length);
            const T* q_global_ptr = q_global + q_token_idx * q_stride + q_head_idx_global * DQKV;
            for (int dim_idx = tid; dim_idx < DQKV; dim_idx += blockDim.x) {
                if (is_valid_q) {
                    q_smem[q_smem_row * DQKV + dim_idx] = q_global_ptr[dim_idx];
                } else {
                    q_smem[q_smem_row * DQKV + dim_idx] = static_cast<T>(0.0f);
                }
            }
        }


        for (int kv_block_offset = 0; kv_block_offset < current_kv_cache_total_len; kv_block_offset += B_c) {

            for (int smem_row = 0; smem_row < B_c; ++smem_row) {
                int k_token_idx = kv_block_offset + smem_row;
                bool is_valid_kv = (k_token_idx < current_kv_cache_total_len);
                const T* k_global_ptr = k_global + (k_token_idx * num_kv_heads_total + kv_head_idx_global) * DQKV;
                const T* v_global_ptr = v_global + (k_token_idx * num_kv_heads_total + kv_head_idx_global) * DQKV;
                for (int dim_idx = tid; dim_idx < DQKV; dim_idx += blockDim.x) {
                    if (is_valid_kv) {
                        k_smem[smem_row * DQKV + dim_idx] = k_global_ptr[dim_idx];
                        v_smem[smem_row * DQKV + dim_idx] = v_global_ptr[dim_idx];
                    } else {
                        k_smem[smem_row * DQKV + dim_idx] = static_cast<T>(0.0f);
                        v_smem[smem_row * DQKV + dim_idx] = static_cast<T>(0.0f);
                    }
                }
            }
            __syncthreads();


            const int WARP_M = B_r / 16;
            const int WARP_N = B_c / 16;
            const int warp_m_id = warp_id / WARP_N;
            const int warp_n_id = warp_id % WARP_N;

            if (warp_m_id < WARP_M) {
                FragmentC fragC;
                wmma::fill_fragment(fragC, 0.0f);

                for (int k_step = 0; k_step < DQKV; k_step += 16) {
                    FragmentA fragA_load;
                    FragmentB fragB_load;
                    // Fixed: Correct 1D indexing and stride for Q@K^T computation
                    const T* smemA_ptr = &q_smem[warp_m_id * 16 * DQKV + k_step];
                    const T* smemB_ptr = &k_smem[warp_n_id * 16 * DQKV + k_step];

                    wmma::load_matrix_sync(fragA_load, smemA_ptr, DQKV);
                    wmma::load_matrix_sync(fragB_load, smemB_ptr, DQKV);
                    wmma::mma_sync(fragC, fragA_load, fragB_load, fragC);
                }

                const int out_m_base = warp_m_id * 16;
                const int out_n_base = warp_n_id * 16;
                // Fixed: Store with correct 1D indexing
                float* scores_ptr = &scores_smem[out_m_base * B_c + out_n_base];
                wmma::store_matrix_sync(scores_ptr, fragC, B_c, wmma::mem_row_major);
            }
            __syncthreads();

            // Apply scaling, masking, and causal mask
            const float inv_dk = (1.0f / sqrtf(static_cast<float>(DQKV)));
            for (int i = tid; i < B_r * B_c; i += blockDim.x) {
                int q_smem_row = i / B_c;
                int k_smem_row = i % B_c;
                int q_token_idx_local = q_segment_start_idx + q_block_offset + q_smem_row;
                int k_token_idx_local = kv_block_offset + k_smem_row;
                int q_abs_pos = q_offset_in_kv_timeline + q_token_idx_local;
                bool is_q_padding = (q_token_idx_local >= current_prefill_q_length);
                bool is_k_padding = (k_token_idx_local >= current_kv_cache_total_len);
                bool is_masked_by_causal = (k_token_idx_local > q_abs_pos);

                if (is_q_padding || is_k_padding || is_masked_by_causal) {
                    scores_smem[i] = -FLT_MAX;
                } else {
                    scores_smem[i] *= inv_dk;
                }
            }
            __syncthreads();


            __shared__ float scale_prev[B_r];

            for (int q_smem_row = warp_id; q_smem_row < B_r; q_smem_row += WARP_NUM) {
                float m_prev = m_stats[q_smem_row];
                float l_prev = l_stats[q_smem_row];
                float m_block = -FLT_MAX;
                for (int col_idx = lane_id; col_idx < B_c; col_idx += warpSize) {
                    m_block = max(m_block, scores_smem[q_smem_row * B_c + col_idx]);
                }
                for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
                    m_block = max(m_block, __shfl_xor_sync(0xffffffff, m_block, offset));
                }
                float m_new = max(m_prev, m_block);

                float l_block = 0.0f;
                // Fixed: Use separate p_smem (no longer aliased with scores_smem)
                for (int col_idx = lane_id; col_idx < B_c; col_idx += warpSize) {
                    float s = scores_smem[q_smem_row * B_c + col_idx];
                    float p = expf(s - m_new);
                    p_smem[q_smem_row * B_c + col_idx] = static_cast<T>(p);
                    l_block += p;
                }
                for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
                    l_block += __shfl_xor_sync(0xffffffff, l_block, offset);
                }

                float scale = expf(m_prev - m_new);
                scale_prev[q_smem_row] = scale;

                if (lane_id == 0) {
                    m_stats[q_smem_row] = m_new;
                    l_stats[q_smem_row] = l_prev * scale + l_block;
                }
            }
            __syncthreads();


            constexpr int cur_WARP_M = B_r / 16;
            constexpr int cur_WARP_N = DQKV / 16;
            const int cur_warp_m_id = warp_id / cur_WARP_N;
            const int cur_warp_n_id = warp_id % cur_WARP_N;

            if (cur_warp_m_id < cur_WARP_M) {
                FragmentC fragC;
                wmma::fill_fragment(fragC, 0.0f);

                for (int k_base = 0; k_base < B_c; k_base += 16) {
                    FragmentA fragA_load;
                    FragmentB_t fragB_load;
                    // Fixed: Correct 1D indexing for P matrix (B_r x B_c)
                    const T* smemA_ptr = &p_smem[cur_warp_m_id * 16 * B_c + k_base];
                    // Fixed: Correct 1D indexing for V matrix (B_c x DQKV)
                    const T* smemB_ptr = &v_smem[k_base * DQKV + cur_warp_n_id * 16];

                    wmma::load_matrix_sync(fragA_load, smemA_ptr, B_c);
                    wmma::load_matrix_sync(fragB_load, smemB_ptr, DQKV);
                    wmma::mma_sync(fragC, fragA_load, fragB_load, fragC);
                }

                const int out_m_base = cur_warp_m_id * 16;
                const int out_n_base = cur_warp_n_id * 16;
                // Fixed: Store with correct 1D indexing
                float* pv_ptr = &pv_smem[out_m_base * DQKV + out_n_base];
                wmma::store_matrix_sync(pv_ptr, fragC, DQKV, wmma::mem_row_major);
            }
            __syncthreads();


            for (int q_smem_row = warp_id; q_smem_row < B_r; q_smem_row += WARP_NUM) {
                for (int d_idx = lane_id; d_idx < DQKV; d_idx += warpSize) {
                    float current_o = o_smem[q_smem_row * DQKV + d_idx];
                    float pv_val = pv_smem[q_smem_row * DQKV + d_idx];
                    o_smem[q_smem_row * DQKV + d_idx] = current_o * scale_prev[q_smem_row] + pv_val;
                }
            }
            __syncthreads();
        }


        for (int q_smem_row = 0; q_smem_row < B_r; ++q_smem_row) {
            int q_token_idx = q_segment_start_idx + q_block_offset + q_smem_row;
            if (q_token_idx < current_prefill_q_length) {
                float l_final = l_stats[q_smem_row];
                float inv_l_final = (l_final > 1e-6f) ? (1.0f / l_final) : 0.0f;
                T* out_global_ptr = out_global + (q_token_idx * num_q_heads_total + q_head_idx_global) * DQKV;
                for (int d_idx = tid; d_idx < DQKV; d_idx += blockDim.x) {
                    float final_output = o_smem[q_smem_row * DQKV + d_idx] * inv_l_final;
                    out_global_ptr[d_idx] = static_cast<T>(final_output);
                }
            }
        }
        __syncthreads();
    }
}

template <typename T, int B_c, int B_r, int T_r, int WARP_NUM = 4, int DQKV = 128, int STAGE = 1>
__global__ void flash_attn_prefill_kernel_v3(const T* __restrict__ q_global, const T* __restrict__ k_global,
                                             const T* __restrict__ v_global, T* __restrict__ out_global,
                                             int num_q_heads_total, int num_kv_heads_total, int GQA_n_group,
                                             int current_prefill_q_length, int current_kv_cache_total_len,
                                             int q_offset_in_kv_timeline, int q_stride, T* zero_vec) {

    const int q_segment_idx = blockIdx.x;
    const int q_head_idx_global = blockIdx.y;
    const int kv_head_idx_global = q_head_idx_global / GQA_n_group;
    using namespace nvcuda;
    using FragmentA = wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major>;
    using FragmentB = wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::col_major>;
    using FragmentB_t = wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::row_major>;
    using FragmentC = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;


    const int tid = threadIdx.x;
    const int warp_id = tid / warpSize;
    const int lane_id = tid % warpSize;


    const int q_segment_start_idx = q_segment_idx * T_r;

    constexpr auto align_size = [](size_t size) { return ((size + 15) / 16) * 16; };

    const size_t q_smem_size = align_size(B_r * DQKV * sizeof(T));
    const size_t k_smem_size = align_size(B_c * DQKV * sizeof(T) * STAGE);
    const size_t v_smem_size = align_size(B_c * DQKV * sizeof(T) * STAGE);

    const size_t kv_stage_size = align_size(B_c * DQKV);

    const size_t o_smem_size = align_size(B_r * DQKV * sizeof(float));
    const size_t stats_size = align_size(B_r * 2 * sizeof(float));
    const size_t scores_smem_size = align_size(B_r * B_c * sizeof(float));
    const size_t p_smem_size = align_size(B_r * B_c * sizeof(T));
    const size_t pv_smem_size = align_size(B_r * DQKV * sizeof(float));

    const size_t k_smem_offset = q_smem_size;
    const size_t v_smem_offset = k_smem_offset + k_smem_size;
    const size_t o_smem_offset = v_smem_offset + v_smem_size;
    const size_t stats_offset = o_smem_offset + o_smem_size;
    const size_t scores_smem_offset = stats_offset + stats_size;
    const size_t p_smem_offset = scores_smem_offset + scores_smem_size;
    const size_t pv_smem_offset = p_smem_offset + p_smem_size;

    extern __shared__ unsigned char smem_buffer[];

    T* q_smem = reinterpret_cast<T*>(smem_buffer);
    T* k_smem = reinterpret_cast<T*>(smem_buffer + k_smem_offset);
    T* v_smem = reinterpret_cast<T*>(smem_buffer + v_smem_offset);
    float* o_smem = reinterpret_cast<float*>(smem_buffer + o_smem_offset);
    float* m_stats = reinterpret_cast<float*>(smem_buffer + stats_offset);
    float* l_stats = reinterpret_cast<float*>(smem_buffer + stats_offset + B_r * sizeof(float));
    float* scores_smem = reinterpret_cast<float*>(smem_buffer + scores_smem_offset);
    T* p_smem = reinterpret_cast<T*>(smem_buffer + p_smem_offset);
    float* pv_smem = reinterpret_cast<float*>(smem_buffer + pv_smem_offset);
    constexpr int VEC_SIZE = 16 / sizeof(T);

    __shared__ float scale_prev[B_r];


    for (int q_block_offset = 0; q_block_offset < T_r; q_block_offset += B_r) {

        for (int load_idx = tid; load_idx < B_r; load_idx += blockDim.x) {
            m_stats[load_idx] = -FLT_MAX;
            l_stats[load_idx] = 0.0f;
        }
        for (int load_idx = tid; load_idx < DQKV * B_r; load_idx += blockDim.x) {
            o_smem[load_idx] = 0.0f;
        }

        for (int q_load_idx = tid * VEC_SIZE; q_load_idx < B_r * DQKV; q_load_idx += blockDim.x * VEC_SIZE) {
            int q_smem_row = q_load_idx / DQKV;
            int dim_idx = q_load_idx % DQKV;
            int q_token_idx = q_segment_start_idx + q_block_offset + q_smem_row;
            bool is_valid_q = (q_token_idx < current_prefill_q_length);
            const T* q_global_ptr = q_global + q_token_idx * q_stride + q_head_idx_global * DQKV;
            // if (is_valid_q) {
            //     uint32_t q_smem_cp = __cvta_generic_to_shared(q_smem + q_smem_row * DQKV + dim_idx);
            //     CP_ASYNC_CG(q_smem_cp, q_global_ptr + dim_idx, 16);
            // } else {
            //     uint32_t q_smem_cp = __cvta_generic_to_shared(q_smem + q_smem_row * DQKV + dim_idx);
            //     CP_ASYNC_CG(q_smem_cp, zero_vec, 16);
            // }
            if (is_valid_q) {
                *reinterpret_cast<float4*>(&q_smem[q_smem_row * DQKV + dim_idx]) =
                    *reinterpret_cast<const float4*>(&q_global_ptr[dim_idx]);
            } else {
                *reinterpret_cast<float4*>(&q_smem[q_smem_row * DQKV + dim_idx]) =
                    *reinterpret_cast<const float4*>(zero_vec);
            }
        }
        // CP_ASYNC_COMMIT_GROUP();

        for (int load_stage = 0; load_stage < STAGE - 1; load_stage++) {
            for (int load_idx = tid * VEC_SIZE; load_idx < B_c * DQKV; load_idx += blockDim.x * VEC_SIZE) {
                int smem_row = load_idx / DQKV;
                int dim_idx = load_idx % DQKV;
                int k_token_idx = load_stage * B_c + smem_row;
                bool is_valid_kv = (k_token_idx < current_kv_cache_total_len);
                const T* k_global_ptr = k_global + (k_token_idx * num_kv_heads_total + kv_head_idx_global) * DQKV;
                const T* v_global_ptr = v_global + (k_token_idx * num_kv_heads_total + kv_head_idx_global) * DQKV;
                if (is_valid_kv) {
                    uint32_t k_smem_cp =
                        __cvta_generic_to_shared(k_smem + smem_row * DQKV + dim_idx + load_stage * kv_stage_size);
                    uint32_t v_smem_cp =
                        __cvta_generic_to_shared(v_smem + smem_row * DQKV + dim_idx + load_stage * kv_stage_size);
                    CP_ASYNC_CG(k_smem_cp, k_global_ptr + dim_idx, 16);
                    CP_ASYNC_CG(v_smem_cp, v_global_ptr + dim_idx, 16);
                } else {
                    uint32_t k_smem_cp =
                        __cvta_generic_to_shared(k_smem + smem_row * DQKV + dim_idx + load_stage * kv_stage_size);
                    uint32_t v_smem_cp =
                        __cvta_generic_to_shared(v_smem + smem_row * DQKV + dim_idx + load_stage * kv_stage_size);
                    CP_ASYNC_CG(k_smem_cp, zero_vec, 16);
                    CP_ASYNC_CG(v_smem_cp, zero_vec, 16);
                }
            }
            CP_ASYNC_COMMIT_GROUP();
        }


        for (int kv_block_offset_load = (STAGE - 1) * B_c; kv_block_offset_load < current_kv_cache_total_len;
             kv_block_offset_load += B_c) {
            CP_ASYNC_WAIT_GROUP(STAGE - 2);
            __syncthreads();
            int kv_block_offset = kv_block_offset_load - (STAGE - 1) * B_c;

            int smem_sel_next = (kv_block_offset_load / B_c) % STAGE;
            int smem_sel = (kv_block_offset_load / B_c + 1) % STAGE;

            for (int load_idx = tid * VEC_SIZE; load_idx < B_c * DQKV; load_idx += blockDim.x * VEC_SIZE) {
                int smem_row = load_idx / DQKV;
                int dim_idx = load_idx % DQKV;
                int k_token_idx = kv_block_offset_load + smem_row;
                bool is_valid_kv = (k_token_idx < current_kv_cache_total_len);
                const T* k_global_ptr = k_global + (k_token_idx * num_kv_heads_total + kv_head_idx_global) * DQKV;
                const T* v_global_ptr = v_global + (k_token_idx * num_kv_heads_total + kv_head_idx_global) * DQKV;
                if (is_valid_kv) {
                    uint32_t k_smem_cp =
                        __cvta_generic_to_shared(k_smem + smem_row * DQKV + dim_idx + smem_sel_next * kv_stage_size);
                    uint32_t v_smem_cp =
                        __cvta_generic_to_shared(v_smem + smem_row * DQKV + dim_idx + smem_sel_next * kv_stage_size);
                    CP_ASYNC_CG(k_smem_cp, k_global_ptr + dim_idx, 16);
                    CP_ASYNC_CG(v_smem_cp, v_global_ptr + dim_idx, 16);
                } else {
                    uint32_t k_smem_cp =
                        __cvta_generic_to_shared(k_smem + smem_row * DQKV + dim_idx + smem_sel_next * kv_stage_size);
                    uint32_t v_smem_cp =
                        __cvta_generic_to_shared(v_smem + smem_row * DQKV + dim_idx + smem_sel_next * kv_stage_size);
                    CP_ASYNC_CG(k_smem_cp, zero_vec, 16);
                    CP_ASYNC_CG(v_smem_cp, zero_vec, 16);
                }
            }
            CP_ASYNC_COMMIT_GROUP();


            const int WARP_M = B_r / 16;
            const int WARP_N = B_c / 16;
            const int warp_m_id = warp_id / WARP_N;
            const int warp_n_id = warp_id % WARP_N;


            if (warp_m_id < WARP_M) {
                FragmentC fragC;
                wmma::fill_fragment(fragC, 0.0f);

                for (int k_step = 0; k_step < DQKV; k_step += 16) {
                    FragmentA fragA_load;
                    FragmentB fragB_load;
                    // Fixed: Correct 1D indexing and stride for Q@K^T computation
                    const T* smemA_ptr = &q_smem[warp_m_id * 16 * DQKV + k_step];
                    const T* smemB_ptr = &k_smem[warp_n_id * 16 * DQKV + k_step + smem_sel * kv_stage_size];

                    wmma::load_matrix_sync(fragA_load, smemA_ptr, DQKV);
                    wmma::load_matrix_sync(fragB_load, smemB_ptr, DQKV);
                    wmma::mma_sync(fragC, fragA_load, fragB_load, fragC);
                }

                const int out_m_base = warp_m_id * 16;
                const int out_n_base = warp_n_id * 16;

                float* scores_ptr = &scores_smem[out_m_base * B_c + out_n_base];
                wmma::store_matrix_sync(scores_ptr, fragC, B_c, wmma::mem_row_major);
            }
            __syncthreads();

            // mask
            const float inv_dk = (1.0f / sqrtf(static_cast<float>(DQKV)));
            for (int i = tid; i < B_r * B_c; i += blockDim.x) {
                int q_smem_row = i / B_c;
                int k_smem_row = i % B_c;
                int q_token_idx_local = q_segment_start_idx + q_block_offset + q_smem_row;
                int k_token_idx_local = kv_block_offset + k_smem_row;
                int q_abs_pos = q_offset_in_kv_timeline + q_token_idx_local;
                bool is_q_padding = (q_token_idx_local >= current_prefill_q_length);
                bool is_k_padding = (k_token_idx_local >= current_kv_cache_total_len);
                bool is_masked_by_causal = (k_token_idx_local > q_abs_pos);

                if (is_q_padding || is_k_padding || is_masked_by_causal) {
                    scores_smem[i] = -FLT_MAX;
                } else {
                    scores_smem[i] *= inv_dk;
                }
            }
            __syncthreads();
            // softmax
            for (int q_smem_row = warp_id; q_smem_row < B_r; q_smem_row += WARP_NUM) {
                float m_prev = m_stats[q_smem_row];
                float l_prev = l_stats[q_smem_row];
                float m_block = -FLT_MAX;
                for (int col_idx = lane_id; col_idx < B_c; col_idx += warpSize) {
                    m_block = max(m_block, scores_smem[q_smem_row * B_c + col_idx]);
                }
                for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
                    m_block = max(m_block, __shfl_xor_sync(0xffffffff, m_block, offset));
                }
                float m_new = max(m_prev, m_block);

                float l_block = 0.0f;

                for (int col_idx = lane_id; col_idx < B_c; col_idx += warpSize) {
                    float s = scores_smem[q_smem_row * B_c + col_idx];
                    float p = expf(s - m_new);
                    p_smem[q_smem_row * B_c + col_idx] = static_cast<T>(p);
                    l_block += p;
                }
                for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
                    l_block += __shfl_xor_sync(0xffffffff, l_block, offset);
                }

                float scale = expf(m_prev - m_new);
                scale_prev[q_smem_row] = scale;

                if (lane_id == 0) {
                    m_stats[q_smem_row] = m_new;
                    l_stats[q_smem_row] = l_prev * scale + l_block;
                }
            }
            __syncthreads();


            constexpr int cur_WARP_M = B_r / 16;
            constexpr int cur_WARP_N = DQKV / 16;
            const int cur_warp_m_id = warp_id / cur_WARP_N;
            const int cur_warp_n_id = warp_id % cur_WARP_N;
            if (cur_warp_m_id < cur_WARP_M) {
                FragmentC fragC;
                wmma::fill_fragment(fragC, 0.0f);
                for (int k_base = 0; k_base < B_c; k_base += 16) {
                    FragmentA fragA_load;
                    FragmentB_t fragB_load;
                    const T* smemA_ptr = &p_smem[cur_warp_m_id * 16 * B_c + k_base];
                    const T* smemB_ptr = &v_smem[k_base * DQKV + cur_warp_n_id * 16 + smem_sel * kv_stage_size];

                    wmma::load_matrix_sync(fragA_load, smemA_ptr, B_c);
                    wmma::load_matrix_sync(fragB_load, smemB_ptr, DQKV);
                    wmma::mma_sync(fragC, fragA_load, fragB_load, fragC);
                }

                const int out_m_base = cur_warp_m_id * 16;
                const int out_n_base = cur_warp_n_id * 16;
                float* pv_ptr = &pv_smem[out_m_base * DQKV + out_n_base];
                wmma::store_matrix_sync(pv_ptr, fragC, DQKV, wmma::mem_row_major);
            }
            __syncthreads();

            for (int q_smem_row = warp_id; q_smem_row < B_r; q_smem_row += WARP_NUM) {
                for (int d_idx = lane_id; d_idx < DQKV; d_idx += warpSize) {
                    float current_o = o_smem[q_smem_row * DQKV + d_idx];
                    float pv_val = pv_smem[q_smem_row * DQKV + d_idx];
                    o_smem[q_smem_row * DQKV + d_idx] = current_o * scale_prev[q_smem_row] + pv_val;
                }
            }
            __syncthreads();
        }
        CP_ASYNC_WAIT_GROUP(0);

        const int num_kv_blocks = (current_kv_cache_total_len + B_c - 1) / B_c;
        const int main_loop_computed_blocks = (num_kv_blocks > STAGE - 1) ? (num_kv_blocks - (STAGE - 1)) : 0;
        for (int i = 0; i < STAGE - 1; ++i) {
            int current_block_idx = main_loop_computed_blocks + i;
            if (current_block_idx >= num_kv_blocks) {
                break;
            }
            int kv_block_offset = current_block_idx * B_c;
            int smem_sel = current_block_idx % STAGE;
            __syncthreads();

            const int WARP_M = B_r / 16;
            const int WARP_N = B_c / 16;
            const int warp_m_id = warp_id / WARP_N;
            const int warp_n_id = warp_id % WARP_N;

            if (warp_m_id < WARP_M) {
                FragmentC fragC;
                wmma::fill_fragment(fragC, 0.0f);
                for (int k_step = 0; k_step < DQKV; k_step += 16) {
                    FragmentA fragA_load;
                    FragmentB fragB_load;
                    const T* smemA_ptr = &q_smem[warp_m_id * 16 * DQKV + k_step];
                    const T* smemB_ptr =
                        &k_smem[warp_n_id * 16 * DQKV + k_step + smem_sel * kv_stage_size];
                    wmma::load_matrix_sync(fragA_load, smemA_ptr, DQKV);
                    wmma::load_matrix_sync(fragB_load, smemB_ptr, DQKV);
                    wmma::mma_sync(fragC, fragA_load, fragB_load, fragC);
                }
                const int out_m_base = warp_m_id * 16;
                const int out_n_base = warp_n_id * 16;
                float* scores_ptr = &scores_smem[out_m_base * B_c + out_n_base];
                wmma::store_matrix_sync(scores_ptr, fragC, B_c, wmma::mem_row_major);
            }
            __syncthreads();

            const float inv_dk = (1.0f / sqrtf(static_cast<float>(DQKV)));
            for (int j = tid; j < B_r * B_c; j += blockDim.x) {
                int q_smem_row = j / B_c;
                int k_smem_row = j % B_c;
                int q_token_idx_local = q_segment_start_idx + q_block_offset + q_smem_row;
                int k_token_idx_local = kv_block_offset + k_smem_row;
                int q_abs_pos = q_offset_in_kv_timeline + q_token_idx_local;
                bool is_q_padding = (q_token_idx_local >= current_prefill_q_length);
                bool is_k_padding = (k_token_idx_local >= current_kv_cache_total_len);
                bool is_masked_by_causal = (k_token_idx_local > q_abs_pos);
                if (is_q_padding || is_k_padding || is_masked_by_causal) {
                    scores_smem[j] = -FLT_MAX;
                } else {
                    scores_smem[j] *= inv_dk;
                }
            }
            __syncthreads();
            for (int q_smem_row = warp_id; q_smem_row < B_r; q_smem_row += WARP_NUM) {
                float m_prev = m_stats[q_smem_row];
                float l_prev = l_stats[q_smem_row];
                float m_block = -FLT_MAX;
                for (int col_idx = lane_id; col_idx < B_c; col_idx += warpSize) {
                    m_block = max(m_block, scores_smem[q_smem_row * B_c + col_idx]);
                }
                for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
                    m_block = max(m_block, __shfl_xor_sync(0xffffffff, m_block, offset));
                }
                float m_new = max(m_prev, m_block);
                float l_block = 0.0f;
                for (int col_idx = lane_id; col_idx < B_c; col_idx += warpSize) {
                    float s = scores_smem[q_smem_row * B_c + col_idx];
                    float p = expf(s - m_new);
                    p_smem[q_smem_row * B_c + col_idx] = static_cast<T>(p);
                    l_block += p;
                }
                for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
                    l_block += __shfl_xor_sync(0xffffffff, l_block, offset);
                }
                float scale = expf(m_prev - m_new);
                scale_prev[q_smem_row] = scale;
                if (lane_id == 0) {
                    m_stats[q_smem_row] = m_new;
                    l_stats[q_smem_row] = l_prev * scale + l_block;
                }
            }
            __syncthreads();


            constexpr int cur_WARP_M = B_r / 16;
            constexpr int cur_WARP_N = DQKV / 16;
            const int cur_warp_m_id = warp_id / cur_WARP_N;
            const int cur_warp_n_id = warp_id % cur_WARP_N;

            if (cur_warp_m_id < cur_WARP_M) {
                FragmentC fragC;
                wmma::fill_fragment(fragC, 0.0f);
                for (int k_base = 0; k_base < B_c; k_base += 16) {
                    FragmentA fragA_load;
                    FragmentB_t fragB_load;
                    const T* smemA_ptr = &p_smem[cur_warp_m_id * 16 * B_c + k_base];
                    const T* smemB_ptr =
                        &v_smem[k_base * DQKV + cur_warp_n_id * 16 + smem_sel * kv_stage_size];
                    wmma::load_matrix_sync(fragA_load, smemA_ptr, B_c);
                    wmma::load_matrix_sync(fragB_load, smemB_ptr, DQKV);
                    wmma::mma_sync(fragC, fragA_load, fragB_load, fragC);
                }
                const int out_m_base = cur_warp_m_id * 16;
                const int out_n_base = cur_warp_n_id * 16;
                float* pv_ptr = &pv_smem[out_m_base * DQKV + out_n_base];
                wmma::store_matrix_sync(pv_ptr, fragC, DQKV, wmma::mem_row_major);
            }
            __syncthreads();


            for (int q_smem_row = warp_id; q_smem_row < B_r; q_smem_row += WARP_NUM) {
                for (int d_idx = lane_id; d_idx < DQKV; d_idx += warpSize) {
                    float current_o = o_smem[q_smem_row * DQKV + d_idx];
                    float pv_val = pv_smem[q_smem_row * DQKV + d_idx];
                    o_smem[q_smem_row * DQKV + d_idx] = current_o * scale_prev[q_smem_row] + pv_val;
                }
            }
            __syncthreads();
        }
        __syncthreads();
        for (int q_smem_row = 0; q_smem_row < B_r; ++q_smem_row) {
            int q_token_idx = q_segment_start_idx + q_block_offset + q_smem_row;
            if (q_token_idx < current_prefill_q_length) {
                float l_final = l_stats[q_smem_row];
                float inv_l_final = (l_final > 1e-6f) ? (1.0f / l_final) : 0.0f;
                T* out_global_ptr = out_global + (q_token_idx * num_q_heads_total + q_head_idx_global) * DQKV;
                for (int d_idx = tid; d_idx < DQKV; d_idx += blockDim.x) {
                    float final_output = o_smem[q_smem_row * DQKV + d_idx] * inv_l_final;
                    out_global_ptr[d_idx] = static_cast<T>(final_output);
                }
            }
        }
        __syncthreads();
    }
}
template <typename T>
void flash_attention_prefill(const Tensor<T>& Q, const Tensor<T>& K, const Tensor<T>& V, Tensor<T>& output, int n_heads,
                             int n_kv_heads, int head_dim, int seq_len, int total_seq_len, int offset,
                             cudaStream_t stream) {
    if (head_dim != 128) {
        throw std::runtime_error("Flash attention prefill currently only supports head_dim=128");
    }

    int n_groups = n_heads / n_kv_heads;

    auto& pool = GlobalCudaMemoryPool::instance();
    static std::once_flag init_flag;
    constexpr int VEC_SIZE = 16 / sizeof(T);
    T* zero_vec = static_cast<T*>(pool.allocate_tagged("zero_vec", VEC_SIZE * sizeof(T)));
    std::call_once(init_flag, [zero_vec, stream]() mutable {
        std::vector<T> vec(VEC_SIZE, static_cast<T>(0.0f));
        cudaMemcpyAsync(zero_vec, vec.data(), VEC_SIZE * sizeof(T), cudaMemcpyHostToDevice, stream);
    });

    constexpr int B_c = 16;
    constexpr int B_r = 16;
    constexpr int DQKV_val = 128;

    constexpr int WARP_NUM = (B_r / 16) * (DQKV_val / 16);
    constexpr int T_r = B_r;

    int num_q_segments = (seq_len + T_r - 1) / T_r;
    dim3 grid(num_q_segments, n_heads);
    dim3 block(WARP_NUM * 32);
    int q_stride = Q.strides()[0];

    constexpr auto align_size = [](size_t size) {
        constexpr size_t alignment = 16;
        return ((size + alignment - 1) / alignment) * alignment;
    };

    constexpr int STAGE = 2;
    // Fixed: Calculate separate buffer sizes (no more aliasing)
    const size_t q_smem_size = align_size(B_r * DQKV_val * sizeof(T));
    const size_t k_smem_size = align_size(B_c * DQKV_val * sizeof(T) * STAGE);
    const size_t v_smem_size = align_size(B_c * DQKV_val * sizeof(T) * STAGE);
    const size_t o_smem_size = align_size(B_r * DQKV_val * sizeof(float));
    const size_t stats_size = align_size(B_r * 2 * sizeof(float));
    const size_t scores_smem_size = align_size(B_r * B_c * sizeof(float));
    const size_t p_smem_size = align_size(B_r * B_c * sizeof(T));
    const size_t pv_smem_size = align_size(B_r * DQKV_val * sizeof(float));

    const size_t total_smem_size = q_smem_size + k_smem_size + v_smem_size + o_smem_size + stats_size +
                                   scores_smem_size + p_smem_size + pv_smem_size;

    flash_attn_prefill_kernel_v3<T, B_c, B_r, T_r, WARP_NUM, DQKV_val, STAGE><<<grid, block, total_smem_size, stream>>>(
        Q.data_ptr(), K.data_ptr(), V.data_ptr(), output.data_ptr(), n_heads, n_kv_heads, n_groups, seq_len,
        total_seq_len, offset, q_stride, zero_vec);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error in flash_attention_prefill: " + std::string(cudaGetErrorString(err)));
    }
}
template <>
void flash_attention_prefill<float>(const Tensor<float>& Q, const Tensor<float>& K, const Tensor<float>& V,
                                    Tensor<float>& output, int n_heads, int n_kv_heads, int head_dim, int seq_len,
                                    int total_seq_len, int offset, cudaStream_t stream) {
    throw std::runtime_error("Flash attention prefill with mma.h does not support FP32. Use bfloat16.");
}

template void flash_attention_prefill<float>(const Tensor<float>& Q, const Tensor<float>& K, const Tensor<float>& V,
                                             Tensor<float>& output, int n_heads, int n_kv_heads, int head_dim,
                                             int seq_len, int total_seq_len, int offset, cudaStream_t stream);

template void flash_attention_prefill<__nv_bfloat16>(const Tensor<__nv_bfloat16>& Q, const Tensor<__nv_bfloat16>& K,
                                                     const Tensor<__nv_bfloat16>& V, Tensor<__nv_bfloat16>& output,
                                                     int n_heads, int n_kv_heads, int head_dim, int seq_len,
                                                     int total_seq_len, int offset, cudaStream_t stream);

}  // namespace cuda_OP