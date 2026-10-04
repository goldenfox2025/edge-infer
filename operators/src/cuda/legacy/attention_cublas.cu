#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

#include "cuda/legacy/legacy_cuda_api.cuh"

namespace cuda_OP {

// cuBLAS-based GQA optimization using batched GEMM
// This treats groups of query heads as batches, leveraging cuBLAS's highly optimized batched operations
template <typename T>
class GQABatchedGEMM {
private:
    cublasHandle_t handle_;
    std::vector<const T*> a_array_;
    std::vector<const T*> b_array_;
    std::vector<T*> c_array_;
    const T** d_a_array_;
    const T** d_b_array_;
    T** d_c_array_;
    size_t max_batch_size_;

public:
    GQABatchedGEMM(size_t max_batch_size = 1024) : max_batch_size_(max_batch_size) {
        cublasCreate(&handle_);

        // Allocate device arrays for batch pointers
        cudaMalloc(&d_a_array_, max_batch_size * sizeof(T*));
        cudaMalloc(&d_b_array_, max_batch_size * sizeof(T*));
        cudaMalloc(&d_c_array_, max_batch_size * sizeof(T*));

        // Reserve host arrays
        a_array_.reserve(max_batch_size);
        b_array_.reserve(max_batch_size);
        c_array_.reserve(max_batch_size);
    }

    ~GQABatchedGEMM() {
        cublasDestroy(handle_);
        cudaFree(d_a_array_);
        cudaFree(d_b_array_);
        cudaFree(d_c_array_);
    }

    void setStream(cudaStream_t stream) {
        cublasSetStream(handle_, stream);
    }

    // Batched attention score computation: Q @ K^T
    void computeAttentionScoresBatched(
        const Tensor<T> &Q,           // [seq_len, n_q_heads, head_dim]
        const Tensor<T> &K,           // [total_seq_len, n_kv_heads, head_dim]
        Tensor<T> &scores,            // [seq_len, n_q_heads, total_seq_len]
        float scale,
        cudaStream_t stream = 0
    ) {
        int seq_len = Q.sizes()[0];
        int n_q_heads = Q.sizes()[1];
        int head_dim = Q.sizes()[2];
        int total_seq_len = K.sizes()[0];
        int n_kv_heads = K.sizes()[1];
        int ratio = n_q_heads / n_kv_heads;

        // Clear arrays
        a_array_.clear();
        b_array_.clear();
        c_array_.clear();

        // Setup batch pointers for each query head
        for (int q_head = 0; q_head < n_q_heads; ++q_head) {
            int kv_head = q_head / ratio;

            // Q matrix for this head: [seq_len, head_dim]
            const T* q_ptr = Q.data_ptr() + q_head * head_dim;

            // K matrix for corresponding KV head: [total_seq_len, head_dim]
            const T* k_ptr = K.data_ptr() + kv_head * head_dim;

            // Output scores for this head: [seq_len, total_seq_len]
            T* scores_ptr = scores.data_ptr() + q_head * total_seq_len;

            a_array_.push_back(q_ptr);
            b_array_.push_back(k_ptr);
            c_array_.push_back(scores_ptr);
        }

        // Copy pointers to device
        cudaMemcpyAsync(d_a_array_, a_array_.data(), n_q_heads * sizeof(T*),
                       cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_b_array_, b_array_.data(), n_q_heads * sizeof(T*),
                       cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_c_array_, c_array_.data(), n_q_heads * sizeof(T*),
                       cudaMemcpyHostToDevice, stream);

        // Set cuBLAS stream
        setStream(stream);

        // Perform batched GEMM: C = alpha * A @ B^T + beta * C
        const T alpha = static_cast<T>(scale);
        const T beta = static_cast<T>(0.0);

        // Compute strides
        int lda = n_q_heads * head_dim;  // Leading dimension of Q
        int ldb = n_kv_heads * head_dim; // Leading dimension of K
        int ldc = n_q_heads * total_seq_len; // Leading dimension of scores

        if constexpr (std::is_same_v<T, __half>) {
            cublasHgemmBatched(
                handle_,
                CUBLAS_OP_N, CUBLAS_OP_T,  // Q @ K^T
                seq_len, total_seq_len, head_dim,
                reinterpret_cast<const __half*>(&alpha),
                reinterpret_cast<const __half* const*>(d_a_array_), lda,
                reinterpret_cast<const __half* const*>(d_b_array_), ldb,
                reinterpret_cast<const __half*>(&beta),
                reinterpret_cast<__half**>(d_c_array_), ldc,
                n_q_heads
            );
        } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            // Use cuBLAS bfloat16 support (CUDA 11.0+)
            #if CUDA_VERSION >= 11000
            cublasGemmStridedBatchedEx(
                handle_,
                CUBLAS_OP_N, CUBLAS_OP_T,  // Q @ K^T
                seq_len, total_seq_len, head_dim,
                &alpha,
                d_a_array_, CUDA_R_16BF, lda, seq_len * head_dim,
                d_b_array_, CUDA_R_16BF, ldb, total_seq_len * head_dim,
                &beta,
                d_c_array_, CUDA_R_16BF, ldc, seq_len * total_seq_len,
                n_q_heads,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
            );
            #else
            throw std::runtime_error("cuBLAS bfloat16 requires CUDA 11.0 or higher");
            #endif
        } else {
            throw std::runtime_error("Unsupported type for cuBLAS batched operations");
        }
    }

    // Batched attention output computation: attention_scores @ V
    void computeAttentionOutputBatched(
        const Tensor<T> &att_probs,   // [seq_len, n_q_heads, cache_length]
        const Tensor<T> &V,           // [cache_length, n_kv_heads, head_dim]
        Tensor<T> &att_output,        // [seq_len, n_q_heads, head_dim]
        cudaStream_t stream = 0
    ) {
        int seq_len = att_probs.sizes()[0];
        int n_q_heads = att_probs.sizes()[1];
        int cache_length = att_probs.sizes()[2];
        int n_kv_heads = V.sizes()[1];
        int head_dim = V.sizes()[2];
        int ratio = n_q_heads / n_kv_heads;

        // Clear arrays
        a_array_.clear();
        b_array_.clear();
        c_array_.clear();

        // Setup batch pointers for each query head
        for (int q_head = 0; q_head < n_q_heads; ++q_head) {
            int kv_head = q_head / ratio;

            // Attention probs for this head: [seq_len, cache_length]
            const T* att_ptr = att_probs.data_ptr() + q_head * cache_length;

            // V matrix for corresponding KV head: [cache_length, head_dim]
            const T* v_ptr = V.data_ptr() + kv_head * head_dim;

            // Output for this head: [seq_len, head_dim]
            T* out_ptr = att_output.data_ptr() + q_head * head_dim;

            a_array_.push_back(att_ptr);
            b_array_.push_back(v_ptr);
            c_array_.push_back(out_ptr);
        }

        // Copy pointers to device
        cudaMemcpyAsync(d_a_array_, a_array_.data(), n_q_heads * sizeof(T*),
                       cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_b_array_, b_array_.data(), n_q_heads * sizeof(T*),
                       cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_c_array_, c_array_.data(), n_q_heads * sizeof(T*),
                       cudaMemcpyHostToDevice, stream);

        // Set cuBLAS stream
        setStream(stream);

        // Perform batched GEMM: C = alpha * A @ B + beta * C
        const T alpha = static_cast<T>(1.0);
        const T beta = static_cast<T>(0.0);

        // Compute strides
        int lda = n_q_heads * cache_length;  // Leading dimension of att_probs
        int ldb = n_kv_heads * head_dim;     // Leading dimension of V
        int ldc = n_q_heads * head_dim;      // Leading dimension of output

        if constexpr (std::is_same_v<T, __half>) {
            cublasHgemmBatched(
                handle_,
                CUBLAS_OP_N, CUBLAS_OP_N,  // att_probs @ V
                seq_len, head_dim, cache_length,
                reinterpret_cast<const __half*>(&alpha),
                reinterpret_cast<const __half* const*>(d_a_array_), lda,
                reinterpret_cast<const __half* const*>(d_b_array_), ldb,
                reinterpret_cast<const __half*>(&beta),
                reinterpret_cast<__half**>(d_c_array_), ldc,
                n_q_heads
            );
        } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            // Use cuBLAS bfloat16 support (CUDA 11.0+)
            #if CUDA_VERSION >= 11000
            cublasGemmStridedBatchedEx(
                handle_,
                CUBLAS_OP_N, CUBLAS_OP_N,  // att_probs @ V
                seq_len, head_dim, cache_length,
                &alpha,
                d_a_array_, CUDA_R_16BF, lda, seq_len * cache_length,
                d_b_array_, CUDA_R_16BF, ldb, cache_length * head_dim,
                &beta,
                d_c_array_, CUDA_R_16BF, ldc, seq_len * head_dim,
                n_q_heads,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
            );
            #else
            throw std::runtime_error("cuBLAS bfloat16 requires CUDA 11.0 or higher");
            #endif
        } else {
            throw std::runtime_error("Unsupported type for cuBLAS batched operations");
        }
    }
};

// Global instance for reuse
thread_local std::unique_ptr<GQABatchedGEMM<__half>> g_gqa_batched_half = nullptr;

// Wrapper functions using cuBLAS batched operations
template <typename T>
void compute_attention_scores_prefill_cublas(
    const Tensor<T> &Q,
    const Tensor<T> &K,
    Tensor<T> &att_scores,
    size_t head_dim,
    cudaStream_t stream
) {
    if constexpr (std::is_same_v<T, __half>) {
        if (!g_gqa_batched_half) {
            g_gqa_batched_half = std::make_unique<GQABatchedGEMM<__half>>();
        }

        float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
        g_gqa_batched_half->computeAttentionScoresBatched(Q, K, att_scores, scale, stream);
    } else {
        // Fallback to WMMA implementation
        compute_attention_scores_prefill_wmma(Q, K, att_scores, head_dim, stream);
    }
}

template <typename T>
void compute_att_output_prefill_cublas(
    const Tensor<T> &att_probs,
    const Tensor<T> &V,
    Tensor<T> &att_output,
    size_t n_q_heads,
    size_t head_dim,
    size_t total_seq_len,
    size_t n_kv_heads,
    cudaStream_t stream
) {
    if constexpr (std::is_same_v<T, __half>) {
        if (!g_gqa_batched_half) {
            g_gqa_batched_half = std::make_unique<GQABatchedGEMM<__half>>();
        }

        g_gqa_batched_half->computeAttentionOutputBatched(att_probs, V, att_output, stream);
    } else {
        // Fallback to WMMA implementation
        compute_att_output_prefill_wmma(att_probs, V, att_output, n_q_heads, head_dim, total_seq_len, n_kv_heads, stream);
    }
}

// Adaptive function that chooses the best implementation
template <typename T>
void compute_attention_scores_prefill_adaptive(
    const Tensor<T> &Q,
    const Tensor<T> &K,
    Tensor<T> &att_scores,
    size_t head_dim,
    cudaStream_t stream
) {
    int seq_len = Q.sizes()[0];
    int n_q_heads = Q.sizes()[1];
    int total_seq_len = K.sizes()[0];

    // Use cuBLAS for larger problems with many heads
    if (n_q_heads >= 8 && seq_len >= 128 && total_seq_len >= 128) {
        compute_attention_scores_prefill_cublas(Q, K, att_scores, head_dim, stream);
    }
    // Use WMMA for medium-sized problems
    else if (head_dim % 16 == 0 && seq_len >= 32 && total_seq_len >= 32) {
        compute_attention_scores_prefill_wmma(Q, K, att_scores, head_dim, stream);
    }
    // Use original implementation for small problems
    else {
        launch_gqa_gemm(Q, K, att_scores, stream);
    }
}

template <typename T>
void compute_att_output_prefill_adaptive(
    const Tensor<T> &att_probs,
    const Tensor<T> &V,
    Tensor<T> &att_output,
    size_t n_q_heads,
    size_t head_dim,
    size_t total_seq_len,
    size_t n_kv_heads,
    cudaStream_t stream
) {
    int seq_len = att_probs.sizes()[0];
    int cache_length = att_probs.sizes()[2];

    // Use cuBLAS for larger problems with many heads
    if (n_q_heads >= 8 && seq_len >= 128 && cache_length >= 128) {
        compute_att_output_prefill_cublas(att_probs, V, att_output, n_q_heads, head_dim, total_seq_len, n_kv_heads, stream);
    }
    // Use WMMA for medium-sized problems
    else if (head_dim % 16 == 0 && seq_len >= 32 && cache_length >= 32) {
        compute_att_output_prefill_wmma(att_probs, V, att_output, n_q_heads, head_dim, total_seq_len, n_kv_heads, stream);
    }
    // Use original implementation for small problems
    else {
        compute_att_output_prefill(att_probs, V, att_output, n_q_heads, head_dim, total_seq_len, n_kv_heads, stream);
    }
}

// Template instantiations
template void compute_attention_scores_prefill_cublas<__half>(
    const Tensor<__half> &, const Tensor<__half> &, Tensor<__half> &, size_t, cudaStream_t);
template void compute_att_output_prefill_cublas<__half>(
    const Tensor<__half> &, const Tensor<__half> &, Tensor<__half> &, size_t, size_t, size_t, size_t, cudaStream_t);

template void compute_attention_scores_prefill_cublas<__nv_bfloat16>(
    const Tensor<__nv_bfloat16> &, const Tensor<__nv_bfloat16> &, Tensor<__nv_bfloat16> &, size_t, cudaStream_t);
template void compute_att_output_prefill_cublas<__nv_bfloat16>(
    const Tensor<__nv_bfloat16> &, const Tensor<__nv_bfloat16> &, Tensor<__nv_bfloat16> &, size_t, size_t, size_t, size_t, cudaStream_t);

template void compute_attention_scores_prefill_adaptive<__half>(
    const Tensor<__half> &, const Tensor<__half> &, Tensor<__half> &, size_t, cudaStream_t);
template void compute_att_output_prefill_adaptive<__half>(
    const Tensor<__half> &, const Tensor<__half> &, Tensor<__half> &, size_t, size_t, size_t, size_t, cudaStream_t);

template void compute_attention_scores_prefill_adaptive<__nv_bfloat16>(
    const Tensor<__nv_bfloat16> &, const Tensor<__nv_bfloat16> &, Tensor<__nv_bfloat16> &, size_t, cudaStream_t);
template void compute_att_output_prefill_adaptive<__nv_bfloat16>(
    const Tensor<__nv_bfloat16> &, const Tensor<__nv_bfloat16> &, Tensor<__nv_bfloat16> &, size_t, size_t, size_t, size_t, cudaStream_t);

// Add missing float template instantiation
template void compute_attention_scores_prefill_adaptive<float>(
    const Tensor<float> &, const Tensor<float> &, Tensor<float> &, size_t, cudaStream_t);
template void compute_att_output_prefill_adaptive<float>(
    const Tensor<float> &, const Tensor<float> &, Tensor<float> &, size_t, size_t, size_t, size_t, cudaStream_t);

}  // namespace cuda_OP