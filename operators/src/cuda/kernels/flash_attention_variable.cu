// Attention kernels for variable sequence lengths. Preserve the causal mask and per-row normalization.
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

#include <algorithm>  // min
#include <cstdio>     // printf
#include <cstring>    // memcpy
#include <iostream>
#include <stdexcept>
#include <vector>

#include "operators/cuda/execution_kernels.cuh"
#include "cuda/vector_pack.cuh"

#define DQKV_VALUE 128
#define B_C_VALUE 8

constexpr int WARP_SIZE = 32;

namespace cuda_OP {


template <typename T>
__global__ void flash_attention_kernel_variable(
    T *q,
    const T *k1, const T *k2, const T *k3, const T *k4, const T *k5,
    const T *v1, const T *v2, const T *v3, const T *v4, const T *v5,
    float *att_output1, float *att_output2, float *att_output3, float *att_output4, float *att_output5,
    int n_q_h, int n_kv_h, int dqkv, int B_c, int B_r, int n_groups, int T_r,
    int cache_length1, int cache_length2, int cache_length3, int cache_length4, int cache_length5,
    int T_c1, int T_c2, int T_c3, int T_c4, int T_c5,
    int branch_count, float softmax_scale) {


  if (blockIdx.y >= branch_count) return;


  const T *k;
  const T *v;
  float *att_output;
  int cache_length;
  int T_c;


  switch (blockIdx.y) {
    case 0:
      k = k1;
      v = v1;
      att_output = att_output1;
      cache_length = cache_length1;
      T_c = T_c1;
      break;
    case 1:
      k = k2;
      v = v2;
      att_output = att_output2;
      cache_length = cache_length2;
      T_c = T_c2;
      break;
    case 2:
      k = k3;
      v = v3;
      att_output = att_output3;
      cache_length = cache_length3;
      T_c = T_c3;
      break;
    case 3:
      k = k4;
      v = v4;
      att_output = att_output4;
      cache_length = cache_length4;
      T_c = T_c4;
      break;
    case 4:
      k = k5;
      v = v5;
      att_output = att_output5;
      cache_length = cache_length5;
      T_c = T_c5;
      break;
  }


  if (dqkv != DQKV_VALUE || B_c != B_C_VALUE) return;

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


    if (valid) {
      unsigned int mask = 0xFFFFFFFF;
      for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        local_score += __shfl_down_sync(mask, local_score, offset);
      }
      if (d_tid == 0) {
        s_score_buf[token_tid] =
            local_score * static_cast<float>(softmax_scale);
      }
    } else {
      if (d_tid == 0) {
        s_score_buf[token_tid] = -FLT_MAX;
      }
    }
    __syncthreads();

    // --------------------------
    // Local Softmax
    // --------------------------
    __shared__ float cur_m_s;

    float warp_val =
        (d_tid < B_c && threadIdx.y == 0) ? s_score_buf[d_tid] : -FLT_MAX;
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
      s_s_score[d_tid] = exp_val;
      warp_val_l = exp_val;
    }

    else {

      warp_val_l = 0.0f;
    }

    __syncthreads();


    unsigned int mask_sum = 0xFFFFFFFF;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
      warp_val_l += __shfl_down_sync(mask_sum, warp_val_l, offset);
    }
    if (d_tid == 0 && threadIdx.y == 0) {
      cur_l_s = warp_val_l;
    }
    __syncthreads();
    float cur_l = cur_l_s;

    // --------------------------

    // --------------------------
    if (j == 0) {

      if (token_tid == 0) {

        for (int k_dim = d_tid; k_dim < DQKV_VALUE; k_dim += blockDim.x) {
          float current_dim_partial_out = 0.0f;


          for (int i_tok = 0; i_tok < B_c; ++i_tok) {
            float exp_score = s_s_score[i_tok];
            float v_val = s_vj[i_tok * DQKV_VALUE +
                             k_dim];
            current_dim_partial_out =
                fmaf(exp_score, v_val, current_dim_partial_out);
          }

          s_o[k_dim] = current_dim_partial_out;
        }
      }

      if (token_tid == 0 && d_tid == 0) {
        global_m = cur_m;
        global_l = cur_l;
      }
    } else {


      float old_global_m = global_m;
      float old_global_l = global_l;

      float new_global_m = fmaxf(old_global_m, cur_m);
      float exp_old = __expf(old_global_m - new_global_m);
      float exp_cur = __expf(cur_m - new_global_m);

      if (token_tid == 0) {

        for (int k_dim = d_tid; k_dim < DQKV_VALUE; k_dim += blockDim.x) {
          float current_dim_partial_out = 0.0f;


          for (int i_tok = 0; i_tok < B_c; ++i_tok) {
            float exp_score = s_s_score[i_tok];
            float v_val = s_vj[i_tok * DQKV_VALUE + k_dim];
            current_dim_partial_out =
                fmaf(exp_score, v_val, current_dim_partial_out);
          }


          float old_out_val = s_o[k_dim];

          float new_out_val =
              old_out_val * exp_old + current_dim_partial_out * exp_cur;

          s_o[k_dim] = new_out_val;
        }
      }


      if (token_tid == 0 && d_tid == 0) {
        float new_global_l = old_global_l * exp_old + cur_l * exp_cur;
        global_m = new_global_m;
        global_l = new_global_l;
      }
    }
    __syncthreads();

  }  // end for each chunk (T_c)

  // --------------------------

  // --------------------------


  if (threadIdx.y == 0) {
    int out_offset = head_id * (dqkv + 2);
    for (int i = d_tid; i < DQKV_VALUE; i += blockDim.x) {
      att_output[out_offset + i] = s_o[i];
    }
    if (d_tid == 0) {
      att_output[out_offset + dqkv] = global_m;
      att_output[out_offset + dqkv + 1] = global_l;
    }
  }
}


}  // namespace cuda_OP

namespace op::cuda::detail {

template <typename T>
void launch_decode_128(const ExecutionContext& context, TensorView<const T, 3> q,
    TensorView<const T, 3> k, TensorView<const T, 3> v, TensorView<T, 3> output,
    TensorView<float, 1> workspace) {
  const int heads = q.shape[1], kv_heads = k.shape[1], width = q.shape[2];
  const int length = k.shape[0];
  const int branches = std::min((length + B_C_VALUE - 1) / B_C_VALUE, 5);
  const int per_branch = (length + branches - 1) / branches;
  const std::size_t output_elements = static_cast<std::size_t>(heads) * (width + 2);
  const T* keys[5]{};
  const T* values[5]{};
  float* results[5]{};
  int lengths[5]{};
  int tiles[5]{};
  for (int branch = 0; branch < branches; ++branch) {
    const int start = branch * per_branch;
    lengths[branch] = std::min(per_branch, length - start);
    tiles[branch] = (lengths[branch] + B_C_VALUE - 1) / B_C_VALUE;
    keys[branch] = k.data + static_cast<std::size_t>(start) * k.stride[0];
    values[branch] = v.data + static_cast<std::size_t>(start) * v.stride[0];
    results[branch] = workspace.data + branch * output_elements;
  }
  cuda_OP::flash_attention_kernel_variable<T>
      <<<dim3(heads, branches), dim3(32, B_C_VALUE), 0, context.stream>>>(
          const_cast<T*>(q.data), keys[0], keys[1], keys[2], keys[3], keys[4],
          values[0], values[1], values[2], values[3], values[4],
          results[0], results[1], results[2], results[3], results[4],
          heads, kv_heads, width, B_C_VALUE, 1, heads / kv_heads, 1,
          lengths[0], lengths[1], lengths[2], lengths[3], lengths[4],
          tiles[0], tiles[1], tiles[2], tiles[3], tiles[4], branches,
          1.0f / sqrtf(static_cast<float>(width)));
  const auto result = cudaGetLastError();
  if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
  launch_attention_gather(context, results[0], results[1], results[2], results[3], results[4], branches, output);
}

template void launch_decode_128<float>(const ExecutionContext&, TensorView<const float, 3>, TensorView<const float, 3>, TensorView<const float, 3>, TensorView<float, 3>, TensorView<float, 1>);
template void launch_decode_128<__nv_bfloat16>(const ExecutionContext&, TensorView<const __nv_bfloat16, 3>, TensorView<const __nv_bfloat16, 3>, TensorView<const __nv_bfloat16, 3>, TensorView<__nv_bfloat16, 3>, TensorView<float, 1>);

}  // namespace op::cuda::detail
