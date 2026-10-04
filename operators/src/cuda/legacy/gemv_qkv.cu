#include "cuda/legacy/legacy_cuda_api.cuh"

namespace cuda_OP {
template <typename T, int WARP_SIZE = 32>
__global__ void gemv_qkv_kernel(const T *A, const T *B, T *q, T *k, T *v, const T *bias, int *offset_array,
                                int layer_index, int N, int K, int Q_len, int K_len, int V_len, int n_layers,
                                int *pingpong_index) {

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

    constexpr int VEC_UNIT = sizeof(float4) / sizeof(T);
    float acc = 0;


    const T *b_row_ptr = B + global_n * K;

    for (int k_idx = tid_x * VEC_UNIT; k_idx < K; k_idx += WARP_SIZE * VEC_UNIT) {

        if (k_idx + VEC_UNIT <= K) {
            Vec<T, VEC_UNIT> va, vb;

            va.f4 = *reinterpret_cast<const float4 *>(A + k_idx);
            vb.f4 = *reinterpret_cast<const float4 *>(b_row_ptr + k_idx);


#pragma unroll
            for (int j = 0; j < VEC_UNIT; ++j) {
                acc += static_cast<float>(va.t[j]) * static_cast<float>(vb.t[j]);
            }
        } else {
            for (int j = k_idx; j < K; ++j) {
                acc += static_cast<float>(A[j]) * static_cast<float>(b_row_ptr[j]);
            }
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

        if (global_n < Q_len) {
            T *out_ptr = q + global_n;
            *out_ptr = static_cast<T>(acc);
        } else if (global_n < Q_len + K_len) {
            T *out_ptr = k + out_off + (global_n - Q_len);
            *out_ptr = static_cast<T>(acc);
        } else if (global_n < Q_len + K_len + V_len) {
            T *out_ptr = v + out_off + (global_n - Q_len - K_len);
            *out_ptr = static_cast<T>(acc);
        }
    }
}

template <typename T>
void gemv_qkv(const Tensor<T> *A, const Tensor<T> *B, Tensor<T> *q, Tensor<T> *k, Tensor<T> *v, const Tensor<T> *bias,
              int *offset_array, int layer_index, size_t Q_len, size_t K_len, size_t V_len, cudaStream_t stream,
              int n_layers, int *pingpong_index) {
    const int N = B->sizes()[1];
    const int K = B->sizes()[0];

    const T *d_A = A->data_ptr();
    const T *d_B = B->data_ptr();
    T *d_q = q->data_ptr();
    T *d_k = k->data_ptr();
    T *d_v = v->data_ptr();
    const T *d_bias = bias->data_ptr();

    constexpr int ROWS_PER_BLOCK = 4;
    dim3 blockDim(32, ROWS_PER_BLOCK);
    dim3 gridDim((N + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, 1);


    gemv_qkv_kernel<T><<<gridDim, blockDim, 0, stream>>>(d_A, d_B, d_q, d_k, d_v, d_bias, offset_array, layer_index, N,
                                                         K, Q_len, K_len, V_len, n_layers, pingpong_index);
}


template void gemv_qkv<nv_bfloat16>(const Tensor<nv_bfloat16> *A, const Tensor<nv_bfloat16> *B, Tensor<nv_bfloat16> *q,
                                    Tensor<nv_bfloat16> *k, Tensor<nv_bfloat16> *v, const Tensor<nv_bfloat16> *bias,
                                    int *offset_array, int layer_index, size_t Q_len, size_t K_len, size_t V_len,
                                    cudaStream_t stream, int n_layers, int *pingpong_index);
template void gemv_qkv<float>(const Tensor<float> *A, const Tensor<float> *B, Tensor<float> *q, Tensor<float> *k,
                              Tensor<float> *v, const Tensor<float> *bias, int *offset_array, int layer_index,
                              size_t Q_len, size_t K_len, size_t V_len, cudaStream_t stream, int n_layers,
                              int *pingpong_index);

}  // namespace cuda_OP