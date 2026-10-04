// Fused projection and rotary position embedding. Rotation pairs must use the same positions and frequency convention as the model.
#include "cuda/legacy/legacy_cuda_api.cuh"

namespace cuda_OP {


template <typename T, int WARP_SIZE = 32>
__global__ void gemv_qkv_rope_fused_kernel(const T *A, const T *B, T *q, T *k, T *v, const T *bias, int *offset_array,
                                           int layer_index, int N, int K, int Q_len, int K_len, int V_len, int n_layers,
                                           int *pingpong_index,
                                           // RoPE specific params
                                           const size_t *d_offset, const float *sin_cos_cache, size_t n_heads,
                                           size_t n_kv_heads, size_t head_dim, size_t cache_stride, int q_stride,
                                           int k_stride) {

    int out_off = 0;
    if (offset_array != nullptr) {
        out_off = offset_array[layer_index + n_layers * (*pingpong_index)];
    }

    const int tid_x = threadIdx.x;
    const int tid_y = threadIdx.y;

    const int global_n = blockIdx.x * blockDim.y + tid_y;

    if (global_n >= N) {
        return;
    }


    size_t head_dim_half = head_dim / 2;


    if (global_n >= Q_len + K_len) {
        float acc = 0.f;
        const T *b_row_ptr = B + global_n * K;

        constexpr int VEC_UNIT = sizeof(float4) / sizeof(T);
        for (int k_idx = tid_x * VEC_UNIT; k_idx < K; k_idx += WARP_SIZE * VEC_UNIT) {
            Vec<T, VEC_UNIT> va, vb;
            va.f4 = *reinterpret_cast<const float4 *>(A + k_idx);
            vb.f4 = *reinterpret_cast<const float4 *>(b_row_ptr + k_idx);
#pragma unroll
            for (int j = 0; j < VEC_UNIT; ++j) {
                acc += static_cast<float>(va.t[j]) * static_cast<float>(vb.t[j]);
            }
        }

        // Warp-level reduction
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            acc += __shfl_xor_sync(0xffffffff, acc, offset);
        }

        if (tid_x == 0) {
            if (bias != nullptr) {
                acc += static_cast<float>(bias[global_n]);
            }
            T *out_ptr = v + out_off + (global_n - Q_len - K_len);
            *out_ptr = static_cast<T>(acc);
        }
        return;
    }


    size_t local_idx, current_n_heads;
    bool is_q;

    if (global_n < Q_len) {  // In Q
        is_q = true;
        local_idx = global_n;
        current_n_heads = n_heads;
    } else {  // In K
        is_q = false;
        local_idx = global_n - Q_len;
        current_n_heads = n_kv_heads;
    }

    size_t dim_in_head = local_idx % head_dim;


    if (dim_in_head >= head_dim_half) {
        return;
    }


    const int global_n_partner = global_n + head_dim_half;


    float acc0 = 0.f;
    const T *b_row_ptr0 = B + global_n * K;
    constexpr int VEC_UNIT = sizeof(float4) / sizeof(T);
    for (int k_idx = tid_x * VEC_UNIT; k_idx < K; k_idx += WARP_SIZE * VEC_UNIT) {
        Vec<T, VEC_UNIT> va, vb;
        va.f4 = *reinterpret_cast<const float4 *>(A + k_idx);
        vb.f4 = *reinterpret_cast<const float4 *>(b_row_ptr0 + k_idx);
#pragma unroll
        for (int j = 0; j < VEC_UNIT; ++j) {
            acc0 += static_cast<float>(va.t[j]) * static_cast<float>(vb.t[j]);
        }
    }


    float acc1 = 0.f;
    const T *b_row_ptr1 = B + global_n_partner * K;
    for (int k_idx = tid_x * VEC_UNIT; k_idx < K; k_idx += WARP_SIZE * VEC_UNIT) {
        Vec<T, VEC_UNIT> va, vb;
        va.f4 = *reinterpret_cast<const float4 *>(A + k_idx);
        vb.f4 = *reinterpret_cast<const float4 *>(b_row_ptr1 + k_idx);
#pragma unroll
        for (int j = 0; j < VEC_UNIT; ++j) {
            acc1 += static_cast<float>(va.t[j]) * static_cast<float>(vb.t[j]);
        }
    }


    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc0 += __shfl_xor_sync(0xffffffff, acc0, offset);
        acc1 += __shfl_xor_sync(0xffffffff, acc1, offset);
    }


    if (tid_x == 0) {

        if (bias != nullptr) {
            acc0 += static_cast<float>(bias[global_n]);
            acc1 += static_cast<float>(bias[global_n_partner]);
        }


        size_t offset = d_offset[*pingpong_index];
        size_t seq_idx_in_batch = 0;
        size_t absolute_seq_pos = seq_idx_in_batch + offset;

        size_t rot_dim = dim_in_head;
        size_t cache_idx = absolute_seq_pos * cache_stride + rot_dim * 2;
        float2 sincos = *reinterpret_cast<const float2 *>(sin_cos_cache + cache_idx);

        float x0_rot = acc0 * sincos.y - acc1 * sincos.x;
        float x1_rot = acc0 * sincos.x + acc1 * sincos.y;


        if (is_q) {
            q[global_n] = static_cast<T>(x0_rot);
            q[global_n_partner] = static_cast<T>(x1_rot);
        } else {

            T *k_base_ptr = k + out_off;
            size_t k_local_idx = global_n - Q_len;
            size_t k_partner_local_idx = global_n_partner - Q_len;
            k_base_ptr[k_local_idx] = static_cast<T>(x0_rot);
            k_base_ptr[k_partner_local_idx] = static_cast<T>(x1_rot);
        }
    }
}

template <typename T>
void gemv_qkv_rope(const Tensor<T> *A, const Tensor<T> *B, Tensor<T> *q, Tensor<T> *k, Tensor<T> *v,
                   const Tensor<T> *bias, const size_t *d_offset, const Tensor<float> *sin_cos_cache, int *offset_array,
                   int layer_index, size_t Q_len, size_t K_len, size_t V_len, size_t n_heads, size_t n_kv_heads,
                   size_t head_dim, cudaStream_t stream, int n_layers, int *pingpong_index) {
    // Input validation
    if (!A || !B || !q || !k || !v || !d_offset || !sin_cos_cache) {
        throw std::runtime_error("gemv_qkv_rope: null pointer arguments");
    }

    if (head_dim % 2 != 0) {
        throw std::runtime_error("gemv_qkv_rope: head_dim must be even for RoPE");
    }

    const int N = B->sizes()[1];  // Total output dimension (Q_len + K_len + V_len)
    const int K = B->sizes()[0];  // Input dimension

    const T *d_A = A->data_ptr();
    const T *d_B = B->data_ptr();
    T *d_q = q->data_ptr();
    T *d_k = k->data_ptr();
    T *d_v = v->data_ptr();
    const T *d_bias = bias ? bias->data_ptr() : nullptr;


    const auto &cache_sizes = sin_cos_cache->sizes();
    size_t cache_stride = cache_sizes[1];
    int q_stride = q->strides()[0];
    int k_stride = k->strides()[0];


    constexpr int ROWS_PER_BLOCK = 4;
    dim3 blockDim(32, ROWS_PER_BLOCK);
    dim3 gridDim((N + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, 1);


    gemv_qkv_rope_fused_kernel<T><<<gridDim, blockDim, 0, stream>>>(
        d_A, d_B, d_q, d_k, d_v, d_bias, offset_array, layer_index, N, K, Q_len, K_len, V_len, n_layers, pingpong_index,

        d_offset, sin_cos_cache->data_ptr(), n_heads, n_kv_heads, head_dim, cache_stride, q_stride, k_stride);

    // Error checking
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("gemv_qkv_rope_fused_kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }
}

// Explicit template instantiations
template void gemv_qkv_rope<nv_bfloat16>(const Tensor<nv_bfloat16> *A, const Tensor<nv_bfloat16> *B,
                                         Tensor<nv_bfloat16> *q, Tensor<nv_bfloat16> *k, Tensor<nv_bfloat16> *v,
                                         const Tensor<nv_bfloat16> *bias, const size_t *d_offset,
                                         const Tensor<float> *sin_cos_cache, int *offset_array, int layer_index,
                                         size_t Q_len, size_t K_len, size_t V_len, size_t n_heads, size_t n_kv_heads,
                                         size_t head_dim, cudaStream_t stream, int n_layers, int *pingpong_index);

template void gemv_qkv_rope<float>(const Tensor<float> *A, const Tensor<float> *B, Tensor<float> *q, Tensor<float> *k,
                                   Tensor<float> *v, const Tensor<float> *bias, const size_t *d_offset,
                                   const Tensor<float> *sin_cos_cache, int *offset_array, int layer_index, size_t Q_len,
                                   size_t K_len, size_t V_len, size_t n_heads, size_t n_kv_heads, size_t head_dim,
                                   cudaStream_t stream, int n_layers, int *pingpong_index);

}  // namespace cuda_OP