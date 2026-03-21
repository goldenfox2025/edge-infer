#include <cublas_v2.h>
#include <cuda_bf16.h>  // 提供 __nv_bfloat16 定义
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

#include <algorithm>  // min
#include <cstdio>     // printf
#include <cstring>    // memcpy
#include <iostream>
#include <stdexcept>
#include <vector>

#include "cuda/legacy/legacy_cuda_api.cuh"

#define DQKV_VALUE 128
#define B_C_VALUE 8

constexpr int WARP_SIZE = 32;

namespace cuda_OP {

template <typename T>
__global__ void flash_attention_kernel_v5(T *q, const T *k1, const T *k2, const T *k3, const T *v1, const T *v2,
                                          const T *v3, T *att_output1, T *att_output2,
                                          T *att_output3,  // v4 输出是 T 类型
                                          int n_q_h, int cache_length1, int cache_length2, int cache_length3,
                                          int n_kv_h,
                                          int dqkv,  // 运行时 dqkv，用于验证
                                          int B_c,   // 运行时 B_c，用于验证
                                          int B_r, int n_groups, int T_r, int T_c1, int T_c2, int T_c3,
                                          T softmax_scale) {
    int T_c, cache_length;
    const T *k, *v;
    T *att_output;
    if (blockIdx.y == 0) {
        k = k1;
        v = v1;
        att_output = att_output1;
        cache_length = cache_length1;
        T_c = T_c1;
    } else if (blockIdx.y == 1) {
        k = k2;
        v = v2;
        att_output = att_output2;
        cache_length = cache_length2;
        T_c = T_c2;
    } else if (blockIdx.y == 2) {
        k = k3;
        v = v3;
        att_output = att_output3;
        cache_length = cache_length3;
        T_c = T_c3;
    }
    if (dqkv != DQKV_VALUE || B_c != B_C_VALUE)
        return;

    __shared__ float s_qi[DQKV_VALUE];
    __shared__ float s_vj[B_C_VALUE * DQKV_VALUE];
    __shared__ float s_score_buf[B_C_VALUE];
    __shared__ float s_lm[2];
    __shared__ float s_s_score[B_C_VALUE];
    __shared__ float s_o[DQKV_VALUE];

    const int d_tid = threadIdx.x;
    const int token_tid = threadIdx.y;
    const int head_id = blockIdx.x;
    const int q_offset = head_id * dqkv;
    const int kv_head = head_id / n_groups;

    constexpr int vec_unit = 16 / sizeof(T);
    Vec<T, vec_unit> vq, vk, vv;

    const int vecCount = dqkv / vec_unit;
    for (int i = d_tid; i < vecCount; i += blockDim.x) {
        vq.f4 = *reinterpret_cast<const float4 *>(&q[q_offset + i * vec_unit]);
        // #pragma unroll
        if (token_tid < vec_unit)
            s_qi[i * vec_unit + token_tid] = static_cast<float>(vq.t[token_tid]);
    }
    __syncthreads();

    float &global_m = s_lm[0];
    float &global_l = s_lm[1];

    // --------------------------
    // 遍历 KV 分块
    // --------------------------

    for (int j = 0; j < T_c; ++j) {
        int token_index = j * B_c + token_tid;
        bool valid = (token_index < cache_length);
        float local_score = 0.0f;

        for (int i = d_tid; i < vecCount; i += blockDim.x) {
            int index = (token_index * n_kv_h + kv_head) * dqkv + i * vec_unit;
            if (valid) {
                vk.f4 = *reinterpret_cast<const float4 *>(&k[index]);
                vv.f4 = *reinterpret_cast<const float4 *>(&v[index]);
#pragma unroll
                for (int l = 0; l < vec_unit; l++) {
                    float k_val = static_cast<float>(vk.t[l]);
                    float v_val = static_cast<float>(vv.t[l]);
                    local_score += s_qi[i * vec_unit + l] * k_val;
                    s_vj[token_tid * DQKV_VALUE + i * vec_unit + l] = v_val;
                }
            } else {
#pragma unroll
                for (int l = 0; l < vec_unit; l++) {
                    s_vj[token_tid * DQKV_VALUE + i * vec_unit + l] = 0.0f;
                }
            }
        }

        __syncthreads();

        // Warp 内归约 QK Score
        // TODO：这里只支持dqkv最大为128（float）， 或者256（bf16）
        // 加载更大模型得补跨warp归约 暂时懒得写 后面也是一样
        if (valid) {
            unsigned int mask = 0xFFFFFFFF;
            for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                local_score += __shfl_down_sync(mask, local_score, offset);
            }
            if (d_tid == 0) {
                s_score_buf[token_tid] = local_score * static_cast<float>(softmax_scale);
            }
        } else {
            if (d_tid == 0) {
                s_score_buf[token_tid] = -FLT_MAX;
            }
        }
        __syncthreads();

        __shared__ float cur_m_s;  // 使用临时 shared 变量传递归约结果

        float warp_val = (d_tid < B_c && threadIdx.y == 0) ? s_score_buf[d_tid] : -FLT_MAX;
        unsigned int mask_max = 0xFFFFFFFF;
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            warp_val = fmaxf(warp_val, __shfl_down_sync(mask_max, warp_val, offset));
        }
        if (d_tid == 0 && threadIdx.y == 0) {
            cur_m_s = warp_val;
        }
        __syncthreads();
        float cur_m = cur_m_s;

        __shared__ float cur_l_s;
        float warp_val_l = 0.0f;
        if (d_tid < B_c && threadIdx.y == 0) {
            float score_val = s_score_buf[d_tid];
            float exp_val = expf(score_val - cur_m);
            s_s_score[d_tid] = exp_val;  // 写入一项
            warp_val_l = exp_val;        // warp_val_l 等于这一项的值
        }
        // 后续 Warp Reduce 会正确地将 8 个 warp_val_l 加起来
        else {
            // 其他线程不参与计算，但 warp_val_l 需初始化为 0 用于归约
            warp_val_l = 0.0f;
        }

        __syncthreads();  // 必须确保 s_s_score 完全写入

        // 求和归约
        unsigned int mask_sum = 0xFFFFFFFF;
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            warp_val_l += __shfl_down_sync(mask_sum, warp_val_l, offset);
        }
        if (d_tid == 0 && threadIdx.y == 0) {
            cur_l_s = warp_val_l;
        }
        __syncthreads();
        float cur_l = cur_l_s;  // 所有线程读取 cur_l

        if (j == 0) {
            // 第一个块: 计算并直接写入 s_o
            if (token_tid == 0) {  // 只有 y=0 线程工作
                // 外层循环：遍历当前线程负责的维度 k
                for (int k_dim = d_tid; k_dim < DQKV_VALUE; k_dim += blockDim.x) {
                    float current_dim_partial_out = 0.0f;  // 初始化该维度的累加器

                    // 内层循环：遍历 B_c 个 token，计算 sum(score * V[k])
                    for (int i_tok = 0; i_tok < B_c; ++i_tok) {
                        float exp_score = s_s_score[i_tok];
                        float v_val = s_vj[i_tok * DQKV_VALUE + k_dim];  // 读取 token i 在维度 k 的 V 值
                        current_dim_partial_out = fmaf(exp_score, v_val, current_dim_partial_out);
                    }
                    // 将第一个块计算出的结果直接写入 s_o
                    s_o[k_dim] = current_dim_partial_out;
                }
            }
            // 初始化全局 m, l (由线程 0,0 完成)
            if (token_tid == 0 && d_tid == 0) {
                global_m = cur_m;
                global_l = cur_l;
            }
        } else {
            // 后续块: Online update
            // 读取旧的全局 m, l (所有线程需要)
            float old_global_m = global_m;
            float old_global_l = global_l;
            // 计算新的全局 m 和缩放因子 (所有线程需要)
            float new_global_m = fmaxf(old_global_m, cur_m);
            float exp_old = __expf(old_global_m - new_global_m);
            float exp_cur = __expf(cur_m - new_global_m);

            if (token_tid == 0) {  // 只有 y=0 线程更新 s_o
                // 外层循环：遍历当前线程负责的维度 k
                for (int k_dim = d_tid; k_dim < DQKV_VALUE; k_dim += blockDim.x) {
                    float current_dim_partial_out = 0.0f;  // 初始化该维度的累加器

                    // 内层循环：遍历 B_c 个 token，计算当前块对维度 k 的贡献
                    for (int i_tok = 0; i_tok < B_c; ++i_tok) {
                        float exp_score = s_s_score[i_tok];
                        float v_val = s_vj[i_tok * DQKV_VALUE + k_dim];
                        current_dim_partial_out = fmaf(exp_score, v_val, current_dim_partial_out);
                    }

                    // 读取旧的 s_o 值
                    float old_out_val = s_o[k_dim];
                    // 执行 Online Update
                    float new_out_val = old_out_val * exp_old + current_dim_partial_out * exp_cur;
                    // 写回新的 s_o 值
                    s_o[k_dim] = new_out_val;
                }
            }

            // 更新全局 m, l (由线程 0,0 完成)
            if (token_tid == 0 && d_tid == 0) {
                float new_global_l = old_global_l * exp_old + cur_l * exp_cur;
                global_m = new_global_m;
                global_l = new_global_l;
            }
        }
        __syncthreads();  // 确保 s_o, m, l 更新对下一轮或写回可见

    }  // end for each chunk (T_c)

    // 写回 global memory
    if (threadIdx.y == 0) {
        int out_offset = head_id * (dqkv + 2);
        for (int i = d_tid; i < DQKV_VALUE; i += blockDim.x) {
            att_output[out_offset + i] = static_cast<T>(s_o[i]);
        }
        if (d_tid == 0) {
            att_output[out_offset + dqkv] = static_cast<T>(global_m);
            att_output[out_offset + dqkv + 1] = static_cast<T>(global_l);
        }
    }
}

template <typename T>
void flash_attention(Tensor<T> &Q, const Tensor<T> &&K1, const Tensor<T> &&K2, const Tensor<T> &&K3,
                     const Tensor<T> &&V1, const Tensor<T> &&V2, const Tensor<T> &&V3, Tensor<T> &att_output1,
                     Tensor<T> &att_output2, Tensor<T> &att_output3, cudaStream_t stream) {
    int dqkv = Q.sizes()[2];  // 每个 head 内维度
    if (dqkv != DQKV_VALUE) {
        throw std::runtime_error("dqkv 不匹配预定义的值");
    }
    float softmax_scale = 1.0f / sqrtf(static_cast<float>(dqkv));
    int n_q_h = Q.sizes()[1];           // query head 数
    int cache_length1 = K1.sizes()[0];  // 总的 kv token 数
    int cache_length2 = K2.sizes()[0];  // 总的 kv token 数
    int cache_length3 = K3.sizes()[0];  // 总的 kv token 数
    int n_kv_h = K1.sizes()[1];
    int n_groups = n_q_h / n_kv_h;
    int B_r = 1;
    int T_r = 1;

    // 每个 chunk 读取的 kv token 数（预设为偶数 B_C_VALUE）
    int B_c = B_C_VALUE;
    int T_c1 = (cache_length1 + B_c - 1) / B_c;
    int T_c2 = (cache_length2 + B_c - 1) / B_c;
    int T_c3 = (cache_length3 + B_c - 1) / B_c;
    // 每个 block 处理一个 query head
    dim3 grid(n_q_h, 3);
    int threads_x = 32;   // dqkv = DQKV_VALUE
    int threads_y = B_c;  // B_c = B_C_VALUE
    dim3 block(threads_x, threads_y);

    // 只有v5适应于新的分块fa模式
    flash_attention_kernel_v5<T><<<grid, block, 0, stream>>>(
        Q.data_ptr(), K1.data_ptr(), K2.data_ptr(), K3.data_ptr(), V1.data_ptr(), V2.data_ptr(), V3.data_ptr(),
        att_output1.data_ptr(), att_output2.data_ptr(), att_output3.data_ptr(), n_q_h, cache_length1, cache_length2,
        cache_length3, n_kv_h, dqkv, B_c, B_r, n_groups, T_r, T_c1, T_c2, T_c3, static_cast<T>(softmax_scale));
}

// 显式实例化
template void flash_attention<float>(Tensor<float> &, const Tensor<float> &&, const Tensor<float> &&,
                                     const Tensor<float> &&, const Tensor<float> &&, const Tensor<float> &&,
                                     const Tensor<float> &&, Tensor<float> &, Tensor<float> &, Tensor<float> &,
                                     cudaStream_t stream);
template void flash_attention<nvbf16>(Tensor<nvbf16> &, const Tensor<nvbf16> &&, const Tensor<nvbf16> &&,
                                      const Tensor<nvbf16> &&, const Tensor<nvbf16> &&, const Tensor<nvbf16> &&,
                                      const Tensor<nvbf16> &&, Tensor<nvbf16> &, Tensor<nvbf16> &, Tensor<nvbf16> &,
                                      cudaStream_t stream);

}  // namespace cuda_OP
