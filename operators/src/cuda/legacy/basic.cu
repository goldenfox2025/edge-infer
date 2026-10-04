#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>

#include <cstdio>  // printf
#include <iostream>
#include <stdexcept>
#include <vector>

#include "cuda/legacy/legacy_cuda_api.cuh"

namespace cuda_OP {

// --------------------------------------------------

// --------------------------------------------------
void checkCudaError(cudaError_t error) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        throw std::runtime_error("CUDA operation failed: " + std::string(cudaGetErrorString(error)));
    }
}

// --------------------------------------------------

// --------------------------------------------------

__device__ inline float warp_reduce_sum(float val) {

    for (int offset = 32 / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(__activemask(), val, offset);
    }
    return val;
}


template <typename T>
__global__ void rms_norm_kernel_v2(const T *__restrict__ input, T *__restrict__ output, const T *__restrict__ weight,
                                   float eps, size_t row_size) {

    int row = blockIdx.x;
    const T *__restrict__ in_row = input + row * row_size;
    T *__restrict__ out_row = output + row * row_size;

    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    float local_sum = 0.0f;
    float val[5];
    int flag = 0;
    for (size_t i_base = 0; i_base < row_size; i_base += nthreads) {
        size_t i = i_base + tid;

        if (i < row_size) {
            val[flag++] = static_cast<float>(in_row[i]);
            local_sum += val[flag - 1] * val[flag - 1];
        }
    }

    local_sum = warp_reduce_sum(local_sum);


    __shared__ float s_warp_sums[32];

    int lane = tid % warpSize;
    int warp_id = tid / warpSize;


    if (lane == 0) {
        s_warp_sums[warp_id] = local_sum;
    }


    __syncthreads();


    float block_sum = 0.0f;
    if (warp_id == 0) {
        int num_warps_in_block = (nthreads + warpSize - 1) / warpSize;


        float warp_partial_sum = (tid < num_warps_in_block) ? s_warp_sums[tid] : 0.0f;


        block_sum = warp_reduce_sum(warp_partial_sum);

    }


    __shared__ float s_inv_rms;
    if (tid == 0) {

        s_inv_rms = rsqrtf(block_sum / row_size + eps);
    }


    __syncthreads();


    float inv_rms = s_inv_rms;
    // float val;
    flag = 0;

    for (size_t i = tid; i < row_size; i += nthreads) {
        if (i < row_size) {
            // val = static_cast<float>(in_row[i]);
            float x = val[flag++];
            float w = static_cast<float>(weight[i]);

            out_row[i] = static_cast<T>((x * inv_rms) * w);
        }
    }
}


template <typename T>
__global__ void rms_norm_kernel_v1(const T *input, T *output, const T *weight, float eps, size_t row_size) {

    int row = blockIdx.x;
    const T *in_row = input + row * row_size;
    T *out_row = output + row * row_size;

    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    float local_sum = 0.0f;
    for (size_t i = tid; i < row_size; i += nthreads) {
        float val = static_cast<float>(in_row[i]);
        local_sum += val * val;
    }

    local_sum = warp_reduce_sum(local_sum);


    __shared__ float shared[32];
    int lane = tid % 32;
    int warp_id = tid / 32;
    if (lane == 0) {
        shared[warp_id] = local_sum;
    }
    __syncthreads();

    float block_sum = 0.0f;
    int num_warps = (nthreads + 32 - 1) / 32;
    if (warp_id == 0) {
        float warp_partial_sum = 0.0f;
        if (tid < num_warps) {
            warp_partial_sum = shared[lane];
        }
        block_sum = warp_reduce_sum(warp_partial_sum);
    }


    if (tid == 0) {
        shared[0] = block_sum;
    }
    __syncthreads();

    float final_sum = shared[0];

    float rms = sqrtf(final_sum / row_size + eps);


    for (size_t i = tid; i < row_size; i += nthreads) {
        float val = static_cast<float>(in_row[i]);
        float w = static_cast<float>(weight[i]);
        out_row[i] = static_cast<T>((val / rms) * w);
    }
}


template <typename T>
__global__ void rms_norm_kernel(const T *input, T *output, const T *weight, float eps, size_t row_size) {
    int row = blockIdx.x;
    const T *in_row = input + row * row_size;
    T *out_row = output + row * row_size;
    float sum = 0.0f;
    for (int i = 0; i < row_size; i++) {
        float val = static_cast<float>(in_row[i]);
        sum += val * val;
    }
    float rms = sqrtf(sum / row_size + eps);
    for (int i = 0; i < row_size; i++) {
        float val = static_cast<float>(in_row[i]);
        float w = static_cast<float>(weight[i]);
        out_row[i] = static_cast<T>((val / rms) * w);
    }
}

template <typename T>
void rms_norm(Tensor<T> *output, const Tensor<T> *input, const Tensor<T> *weight, float eps, cudaStream_t stream) {

    size_t seq_len = input->sizes()[0];
    size_t d = input->sizes()[1];  // row_size


    int threads_per_block = 1024;


    // if (d < threads_per_block) {

    // }


    if (threads_per_block > 1024)
        threads_per_block = 1024;


    dim3 block_dim(threads_per_block);
    dim3 grid_dim(seq_len);  // grid_dim.x = seq_len

    rms_norm_kernel_v2<T>
        <<<grid_dim, block_dim, 0, stream>>>(input->data_ptr(), output->data_ptr(), weight->data_ptr(), eps, d);


    checkCudaError(cudaGetLastError());

    // if (stream == nullptr) {
    //   checkCudaError(cudaDeviceSynchronize());
    // }
}

// Helper function (optional, if you want dynamic adjustment based on d)
// int next_power_of_2(int n) {
//     n--;
//     n |= n >> 1;
//     n |= n >> 2;
//     n |= n >> 4;
//     n |= n >> 8;
//     n |= n >> 16;
//     n++;
//     // Ensure it's at least warpSize for efficiency maybe?
//     return (n < 32) ? 32 : n;
// }

// --------------------------------------------------

// --------------------------------------------------

template <typename T>
__global__ void rope_kernel_v1(T *tensor, size_t seq_len, size_t n_heads, size_t head_dim, size_t offset, float theta) {
    // Each thread handles one dimension pair (i, i + head_dim/2) for a specific
    // (seq_idx, head_idx) Grid dimension should be (seq_len * n_heads * head_dim
    // / 2)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_rotations = seq_len * n_heads * (head_dim / 2);

    if (idx < total_rotations) {
        size_t head_dim_half = head_dim / 2;
        size_t i = idx % head_dim_half;              // Dimension index (0 to head_dim/2 - 1)
        size_t head_flat_idx = idx / head_dim_half;  // Flat index for (seq_idx, head_idx)
        size_t seq_idx = head_flat_idx / n_heads;    // Sequence index
        size_t head_idx = head_flat_idx % n_heads;   // Head index

        size_t head_offset = seq_idx * n_heads * head_dim + head_idx * head_dim;
        T *head_ptr = tensor + head_offset;

        // Calculate frequency and rotation angle
        // Inverse frequency calculation is stable across threads working on the
        // same head/seq pos
        float freq = 1.0f / powf(theta, static_cast<float>(2 * i) / static_cast<float>(head_dim));
        float val = (static_cast<float>(seq_idx) + offset) * freq;
        float cos_val;
        float sin_val;
        __sincosf(val, &sin_val, &cos_val);  // Compute sin and cos together

        // Load the pair of elements
        // Accesses head_ptr[i] and head_ptr[i + head_dim_half]
        // Consecutive threads access consecutive 'i', improving coalescing for both
        // reads.
        float x0_f = static_cast<float>(head_ptr[i]);
        float x1_f = static_cast<float>(head_ptr[i + head_dim_half]);

        // Perform rotation
        float rotated_x0 = x0_f * cos_val - x1_f * sin_val;
        float rotated_x1 = x0_f * sin_val + x1_f * cos_val;

        // Store the rotated pair back
        // Consecutive threads access consecutive 'i', improving coalescing for
        // writes.
        head_ptr[i] = static_cast<T>(rotated_x0);
        head_ptr[i + head_dim_half] = static_cast<T>(rotated_x1);
    }
}

// --- BF16 Specialization using __nv_bfloat162 ---
// This leverages vector types and intrinsics for BF16

// Check if CUDA version supports __nv_bfloat162 intrinsics (usually >= 11.0)
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)  // Ampere or later recommended

template <>
__global__ void rope_kernel_v1<__nv_bfloat16>(__nv_bfloat16 *tensor, size_t seq_len, size_t n_heads, size_t head_dim,
                                              size_t offset, float theta) {
    // Each thread handles TWO dimension pairs using bfloat162 type
    // Total number of bfloat162 pairs to process per head is head_dim / 4
    // Grid dimension should be (seq_len * n_heads * head_dim / 4)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t head_dim_half = head_dim / 2;
    size_t pairs_per_head_half = head_dim_half / 2;  // Each bfloat162 holds 2 elements
    size_t total_vec_rotations = seq_len * n_heads * pairs_per_head_half;

    if (idx < total_vec_rotations) {
        size_t vec_i = idx % pairs_per_head_half;          // Index of the bfloat162 pair (0
                                                           // to pairs_per_head_half - 1)
        size_t head_flat_idx = idx / pairs_per_head_half;  // Flat index for (seq_idx, head_idx)
        size_t seq_idx = head_flat_idx / n_heads;          // Sequence index
        size_t head_idx = head_flat_idx % n_heads;         // Head index

        size_t head_offset = seq_idx * n_heads * head_dim + head_idx * head_dim;
        __nv_bfloat16 *head_ptr = tensor + head_offset;

        // Calculate dimension indices for the two elements in the vector
        size_t i0 = vec_i * 2;
        size_t i1 = vec_i * 2 + 1;

        // Calculate frequencies and rotation angles for both elements
        // float freq0 = 1.0f / powf(theta, static_cast<float>(2 * i0) /
        //                                      static_cast<float>(head_dim));
        // float freq1 = 1.0f / powf(theta, static_cast<float>(2 * i1) /
        //                                      static_cast<float>(head_dim));

        float log_theta = logf(theta);
        float neg_two_log_theta_div_hd = -2.0f * log_theta / static_cast<float>(head_dim);

        // --- Inside the loop or calculation for specific i0/i1 ---
        // Calculate the argument for expf
        float exp_arg0 = neg_two_log_theta_div_hd * static_cast<float>(i0);
        float exp_arg1 = neg_two_log_theta_div_hd * static_cast<float>(i1);

        // Calculate frequencies using expf
        float freq0 = expf(exp_arg0);
        float freq1 = expf(exp_arg1);

        float val0 = (static_cast<float>(seq_idx) + offset) * freq0;
        float val1 = (static_cast<float>(seq_idx) + offset) * freq1;

        float cos_val0, sin_val0, cos_val1, sin_val1;
        __sincosf(val0, &sin_val0, &cos_val0);
        __sincosf(val1, &sin_val1, &cos_val1);

        // Load a pair of bfloat162 (4 elements total) using vector load
        // reinterpret_cast is necessary for vectorized loads/stores
        __nv_bfloat162 x0_vec = *reinterpret_cast<__nv_bfloat162 *>(&head_ptr[i0]);  // Loads elements at i0, i1
        __nv_bfloat162 x1_vec =
            *reinterpret_cast<__nv_bfloat162 *>(&head_ptr[i0 + head_dim_half]);  // Loads elements at i0+d/2, i1+d/2

        // Convert bf16 vectors to float vectors for calculation
        float2 x0_fvec = __bfloat1622float2(x0_vec);
        float2 x1_fvec = __bfloat1622float2(x1_vec);

        // Pack sin/cos values into float2 for potential vector operations (though
        // used component-wise here)
        float2 cos_vec = make_float2(cos_val0, cos_val1);
        float2 sin_vec = make_float2(sin_val0, sin_val1);

        // Perform rotation using float components (or use bf16 intrinsics if
        // preferred, requires bf16 sin/cos) If using intrinsics, convert cos/sin to
        // bf162 first:
        // __nv_bfloat162 cos_bf16 = __float22bfloat162_rn(cos_vec);
        // __nv_bfloat162 sin_bf16 = __float22bfloat162_rn(sin_vec);
        // __nv_bfloat162 neg_x1_vec = __hnegb2(x1_vec); // Negate x1 for FMA
        // __nv_bfloat162 rotated_x0_bf16 = __hfma2(x0_vec, cos_bf16,
        // __hmul2(neg_x1_vec, sin_bf16));
        // __nv_bfloat162 rotated_x1_bf16 = __hfma2(x0_vec, sin_bf16,
        // __hmul2(x1_vec, cos_bf16));

        // Component-wise rotation using float:
        float rot_x0_f0 = x0_fvec.x * cos_vec.x - x1_fvec.x * sin_vec.x;
        float rot_x0_f1 = x0_fvec.y * cos_vec.y - x1_fvec.y * sin_vec.y;
        float rot_x1_f0 = x0_fvec.x * sin_vec.x + x1_fvec.x * cos_vec.x;
        float rot_x1_f1 = x0_fvec.y * sin_vec.y + x1_fvec.y * cos_vec.y;

        // Convert results back to bfloat162
        __nv_bfloat162 rotated_x0_bf16 = __float22bfloat162_rn(make_float2(rot_x0_f0, rot_x0_f1));
        __nv_bfloat162 rotated_x1_bf16 = __float22bfloat162_rn(make_float2(rot_x1_f0, rot_x1_f1));

        // Store the rotated pairs back using vector store
        *reinterpret_cast<__nv_bfloat162 *>(&head_ptr[i0]) = rotated_x0_bf16;
        *reinterpret_cast<__nv_bfloat162 *>(&head_ptr[i0 + head_dim_half]) = rotated_x1_bf16;
    }
}
#endif  // __CUDA_ARCH__ >= 800

template <typename T>
__global__ void rope_kernel(T *tensor, size_t seq_len, size_t n_heads, size_t head_dim, size_t offset, float theta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < seq_len * n_heads) {
        size_t seq_idx = idx / n_heads;
        size_t head_idx = idx % n_heads;
        T *head_ptr = tensor + seq_idx * n_heads * head_dim + head_idx * head_dim;
        size_t dim_half = head_dim / 2;
        for (size_t i = 0; i < dim_half; i++) {
            float freq = 1.0f / powf(theta, (2.0f * i) / head_dim);
            float val = (seq_idx + offset) * freq;
            float cos_val = cosf(val);
            float sin_val = sinf(val);
            float x0 = static_cast<float>(head_ptr[i]);
            float x1 = static_cast<float>(head_ptr[i + dim_half]);
            head_ptr[i] = static_cast<T>(x0 * cos_val - x1 * sin_val);
            head_ptr[i + dim_half] = static_cast<T>(x0 * sin_val + x1 * cos_val);
        }
    }
}

template <typename T>
void rope(Tensor<T> *x, size_t offset, float theta, cudaStream_t stream) {
    const auto &sizes = x->sizes();
    if (sizes.size() < 3) {
        throw std::runtime_error("rope: tensor must be at least 3D");
    }
    size_t seq_len = sizes[0];
    size_t n_heads = sizes[1];
    size_t head_dim = sizes[2];

    if (head_dim == 0)
        return;  // Nothing to do
    if (head_dim % 2 != 0) {
        throw std::runtime_error("rope: head_dim must be even");
    }

    int threads = 256;  // Common block size, can be tuned (128, 512, etc.)
    int blocks = 0;
    void *kernel_ptr = nullptr;

    // Select kernel and grid based on type
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        if (head_dim % 4 != 0) {
            // BF16 kernel requires head_dim to be a multiple of 4 for bfloat162
            // loads/stores Fallback to generic kernel or throw error. Here we throw.
            throw std::runtime_error(
                "rope: BF16 requires head_dim to be a multiple of 4 for optimized "
                "kernel");
        }
        size_t total_vec_rotations = seq_len * n_heads * (head_dim / 4);  // Each thread handles a bfloat162 pair
        blocks = (total_vec_rotations + threads - 1) / threads;
        kernel_ptr = (void *)rope_kernel_v1<__nv_bfloat16>;
        // std::cout << "Using BF16 Optimized Kernel" << std::endl; // For debugging
    } else
#endif
    {
        // Generic FP32/FP16 path
        size_t total_rotations = seq_len * n_heads * (head_dim / 2);  // Each thread handles one pair
        blocks = (total_rotations + threads - 1) / threads;
        kernel_ptr = (void *)rope_kernel_v1<T>;
        // std::cout << "Using Generic Optimized Kernel" << std::endl; // For
        // debugging
    }

    if (blocks == 0 && (seq_len * n_heads * head_dim) > 0) {
        // Handle cases where total rotations might be 0 but tensor isn't empty
        // or very small head_dim resulted in 0 rotations/blocks.
        // If head_dim was 0, we returned earlier. If head_dim is > 0, blocks should
        // be > 0. This calculation should ensure blocks > 0 if work needs to be
        // done. If blocks is still 0, it likely means seq_len or n_heads is 0.
        if (seq_len > 0 && n_heads > 0 && head_dim > 0) {
            blocks = 1;  // Launch at least one block if there's data
        } else {
            return;  // No work to do
        }
    }

    // --- Kernel Launch ---
    // We use a function pointer to avoid repeating the launch code inside
    // if/else. Note: Directly using the kernel function name is usually preferred
    // for type safety, but this shows how to handle it if selecting dynamically.
    if (kernel_ptr == (void *)rope_kernel_v1<T>) {
        rope_kernel_v1<T><<<blocks, threads, 0, stream>>>(x->data_ptr(), seq_len, n_heads, head_dim, offset, theta);
    }
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    else if (kernel_ptr == (void *)rope_kernel_v1<__nv_bfloat16>) {
        rope_kernel_v1<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16 *>(x->data_ptr()),  // Cast needed if T is bf16
            seq_len, n_heads, head_dim, offset, theta);
    }
#endif
    else {
        throw std::runtime_error("Internal error: No valid kernel selected.");
    }

    // --- Error Checking ---
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error after rope kernel launch: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA rope kernel launch failed");
    }

    if (stream == nullptr) {
        // err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cerr << "CUDA synchronization error after rope: " << cudaGetErrorString(err) << std::endl;
            throw std::runtime_error("CUDA rope synchronization failed");
        }
    }
}

template <typename T>
__global__ void attention_scores_kernel(const T *__restrict__ Q,
                                        const int n_q_h, const int dqkv,
                                        const T *__restrict__ K,
                                        const int cache_length,
                                        T *__restrict__ att_scores,
                                        const int n_kv_h) {


    __shared__ T smemQ[128];


    const int q = blockIdx.x;
    const int pos_base = blockIdx.y * blockDim.x;
    const int tid = threadIdx.x;  // Thread ID within the block (0 to blockDim.x - 1)
    const int block_size = blockDim.x;


    const int pos = pos_base + tid;


    if (q >= n_q_h) {
        return;
    }


    const T *q_vec = Q + static_cast<size_t>(q) * dqkv;


    for (int i = tid; i < dqkv; i += block_size) {

        smemQ[i] = q_vec[i];
    }

    __syncthreads();


    if (pos < cache_length) {

        const int n_groups = n_q_h / n_kv_h;
        const int kv_head = q / n_groups;


        //      = (pos * n_kv_h + kv_head) * dqkv + i
        const size_t k_vec_offset = (static_cast<size_t>(pos) * n_kv_h + kv_head) * dqkv;
        const T *k_vec = K + k_vec_offset;


        float dot = 0.0f;
        for (int i = 0; i < dqkv; ++i) {

            float q_val = static_cast<float>(smemQ[i]);

            float k_val = static_cast<float>(k_vec[i]);

            dot = fmaf(q_val, k_val, dot);

        }


        const float scale = rsqrtf(static_cast<float>(dqkv));
        // const float scale = 1.0f / sqrtf(static_cast<float>(dqkv)); //


        att_scores[static_cast<size_t>(q) * cache_length + pos] = static_cast<T>(dot * scale);
    }

}


template <typename T>
void compute_attention_scores(const Tensor<T> &Q, const Tensor<T> &K, size_t n_q_h, size_t dqkv, Tensor<T> &att_scores,
                              size_t n_kv_h, cudaStream_t stream) {

    if (n_q_h == 0 || dqkv == 0 || n_kv_h == 0) {
        throw std::runtime_error("Head counts (n_q_h, n_kv_h) and dimension (dqkv) must be non-zero.");
    }
    if (n_q_h % n_kv_h != 0) {
        char msg[256];
        snprintf(msg, sizeof(msg), "n_q_h (%zu) must be divisible by n_kv_h (%zu)", n_q_h, n_kv_h);
        throw std::runtime_error(msg);
    }

    const auto &k_sizes = K.sizes();
    if (k_sizes.size() != 3) {
        throw std::runtime_error("K tensor must have 3 dimensions [cache_length, n_kv_h, dqkv]");
    }
    const size_t cache_length = k_sizes[0];
    if (k_sizes[1] != n_kv_h || k_sizes[2] != dqkv) {
        char msg[512];
        snprintf(msg, sizeof(msg), "K tensor shape mismatch. Expected [*, %zu, %zu], got [%zu, %zu, %zu]", n_kv_h, dqkv,
                 k_sizes[0], k_sizes[1], k_sizes[2]);
        throw std::runtime_error(msg);
    }

    const auto &q_sizes = Q.sizes();
    bool is_3d_q = (q_sizes.size() == 3);
    // size_t expected_q_elems = n_q_h * dqkv;
    size_t actual_q_elems = 1;
    for (size_t dim : q_sizes) {
        actual_q_elems *= dim;
    }

    if (is_3d_q) {

        if (q_sizes[0] != 1 || q_sizes[1] != n_q_h || q_sizes[2] != dqkv) {
            char msg[512];
            snprintf(msg, sizeof(msg),
                     "Q tensor shape mismatch (3D). Expected [1, %zu, %zu], got "
                     "[%zu, %zu, %zu]",
                     n_q_h, dqkv, q_sizes[0], q_sizes[1], q_sizes[2]);
            throw std::runtime_error(msg);
        }
    } else if (q_sizes.size() == 2) {

        if (q_sizes[0] != n_q_h || q_sizes[1] != dqkv) {
            char msg[512];
            snprintf(msg, sizeof(msg), "Q tensor shape mismatch (2D). Expected [%zu, %zu], got [%zu, %zu]", n_q_h, dqkv,
                     q_sizes[0], q_sizes[1]);
            throw std::runtime_error(msg);
        }
    } else {
        char msg[512];
        snprintf(msg, sizeof(msg), "Q tensor must have 2 or 3 dimensions, got %zu dimensions.", q_sizes.size());
        throw std::runtime_error(msg);
    }


    const auto &score_sizes = att_scores.sizes();
    if (score_sizes.size() != 2 || score_sizes[0] != n_q_h || score_sizes[1] != cache_length) {
        char msg[512];
        snprintf(msg, sizeof(msg),
                 "Attention scores tensor shape mismatch. Expected [%zu, %zu], got "
                 "[%zu, %zu]",
                 n_q_h, cache_length, score_sizes.size() > 0 ? score_sizes[0] : 0,
                 score_sizes.size() > 1 ? score_sizes[1] : 0);
        throw std::runtime_error(msg);
    }

    if (cache_length == 0) {


        return;
    }


    const int block_size_x = 256;


    dim3 gridDim(static_cast<unsigned int>(n_q_h),
                 static_cast<unsigned int>((cache_length + block_size_x - 1) / block_size_x), 1);


    dim3 blockDim(block_size_x, 1, 1);


    // size_t shared_mem_size = dqkv * sizeof(T);

    // sizeof(T)


    // cudaDeviceProp prop;
    // cudaGetDeviceProperties(&prop, 0);  // Get properties of device 0
    // if (shared_mem_size > prop.sharedMemPerBlock) {
    //   char msg[512];
    //   snprintf(msg, sizeof(msg),
    //            "Required shared memory size (%zu bytes) exceeds device limit
    //            per " "block (%zu bytes) for dqkv=%zu.", shared_mem_size,
    //            prop.sharedMemPerBlock, dqkv);
    //   throw std::runtime_error(msg);
    // }
    // if (shared_mem_size > prop.sharedMemPerMultiprocessor) {
    //   // Technically okay if occupancy allows, but good to be aware
    //   // printf("Warning: Shared memory size (%zu bytes) is large relative to
    //   SM
    //   // limit (%zu bytes).\n",
    //   //        shared_mem_size, prop.sharedMemPerMultiprocessor);
    // }


    attention_scores_kernel<<<gridDim, blockDim, 0, stream>>>(Q.data_ptr(),
                                                              static_cast<int>(n_q_h), static_cast<int>(dqkv),
                                                              K.data_ptr(), static_cast<int>(cache_length),
                                                              att_scores.data_ptr(), static_cast<int>(n_kv_h));


    checkCudaError(cudaGetLastError());

}

// --------------------------------------------------

// --------------------------------------------------
template <typename T>
__global__ void att_output_kernel(const T *__restrict__ att_probs,
                                  const int n_q_h,
                                  const int cache_length,
                                  const int dqkv,
                                  const T *__restrict__ V, T *__restrict__ att_output,
                                  const int n_kv_h)
{

    const int q = blockIdx.x;

    const int d_start = threadIdx.x;
    const int d_step = blockDim.x;


    if (q >= n_q_h) {
        return;
    }


    const int n_groups = n_q_h / n_kv_h;
    const int kv_head = q / n_groups;


    // pos * (n_kv_h * dqkv) + kv_head * dqkv + d
    // = (pos * n_kv_h + kv_head) * dqkv + d


    const T *current_att_probs_row = att_probs + static_cast<size_t>(q) * cache_length;


    T *current_att_output_row = att_output + static_cast<size_t>(q) * dqkv;


    for (int d = d_start; d < dqkv; d += d_step) {
        float sum = 0.0f;


        // kv_head, d)


        for (int pos = 0; pos < cache_length; ++pos) {


            const float prob = static_cast<float>(current_att_probs_row[pos]);


            const size_t v_index = (static_cast<size_t>(pos) * n_kv_h + kv_head) * dqkv + d;
            const float val = static_cast<float>(V[v_index]);


            sum = fmaf(prob, val, sum);

        }


        current_att_output_row[d] = static_cast<T>(sum);
    }
}

template <typename T>
void compute_att_output(const Tensor<T> &att_probs, const Tensor<T> &V, size_t n_q_h, size_t dqkv,
                        Tensor<T> &att_output, size_t n_kv_h, cudaStream_t stream) {


    if (n_q_h == 0 || dqkv == 0 || n_kv_h == 0) {
        throw std::runtime_error("Head counts (n_q_h, n_kv_h) and dimension (dqkv) must be non-zero.");
    }
    if (n_q_h % n_kv_h != 0) {
        char msg[256];
        snprintf(msg, sizeof(msg), "n_q_h (%zu) must be divisible by n_kv_h (%zu)", n_q_h, n_kv_h);
        throw std::runtime_error(msg);
    }

    const auto &v_sizes = V.sizes();
    if (v_sizes.size() != 3) {
        throw std::runtime_error("V tensor must have 3 dimensions [cache_length, n_kv_h, dqkv]");
    }
    const size_t cache_length = v_sizes[0];
    if (v_sizes[1] != n_kv_h || v_sizes[2] != dqkv) {
        char msg[512];
        snprintf(msg, sizeof(msg), "V tensor shape mismatch. Expected [*, %zu, %zu], got [%zu, %zu, %zu]", n_kv_h, dqkv,
                 v_sizes[0], v_sizes[1], v_sizes[2]);
        throw std::runtime_error(msg);
    }
    // if (cache_length == 0) {


    // sizeof(T), stream); return;
    // }

    const auto &probs_sizes = att_probs.sizes();
    if (probs_sizes.size() != 2 || probs_sizes[0] != n_q_h || probs_sizes[1] != cache_length) {
        char msg[512];
        snprintf(msg, sizeof(msg),
                 "Attention probabilities tensor shape mismatch. Expected [%zu, "
                 "%zu], got [%zu, %zu]",
                 n_q_h, cache_length, probs_sizes.size() > 0 ? probs_sizes[0] : 0,
                 probs_sizes.size() > 1 ? probs_sizes[1] : 0);
        throw std::runtime_error(msg);
    }

    const auto &output_sizes = att_output.sizes();
    if (output_sizes.size() != 2 || output_sizes[0] != n_q_h || output_sizes[1] != dqkv) {
        char msg[512];
        snprintf(msg, sizeof(msg),
                 "Attention output tensor shape mismatch. Expected [%zu, %zu], got "
                 "[%zu, %zu]",
                 n_q_h, dqkv, output_sizes.size() > 0 ? output_sizes[0] : 0,
                 output_sizes.size() > 1 ? output_sizes[1] : 0);
        throw std::runtime_error(msg);
    }


    const int block_dim_d = 128;


    dim3 gridDim(static_cast<unsigned int>(n_q_h), 1, 1);

    dim3 blockDim(block_dim_d, 1, 1);


    att_output_kernel<<<gridDim, blockDim, 0, stream>>>(att_probs.data_ptr(), static_cast<int>(n_q_h),
                                                        static_cast<int>(cache_length), static_cast<int>(dqkv),
                                                        V.data_ptr(), att_output.data_ptr(), static_cast<int>(n_kv_h));

    checkCudaError(cudaGetLastError());
}

template <typename T>
void compute_attention_scores_prefill(const Tensor<T> &Q, const Tensor<T> &K, Tensor<T> &att_scores, size_t dqkv,
                                      cudaStream_t stream) {


    cuda_OP::launch_gqa_gemm(Q, K, att_scores, stream);
}


template <typename T, int BLOCK_THREADS, int TILE_J_DIM>
__global__ void att_output_prefill_revised_kernel(const T *__restrict__ att_probs, const T *__restrict__ V,
                                                  T *att_output, int n_q, int cache_length, int dqkv, int n_kv_h,
                                                  int n_q_h) {

    __shared__ T v_tile[TILE_J_DIM][BLOCK_THREADS];
    __shared__ T prob_tile[TILE_J_DIM];


    const int q = blockIdx.x;
    const int tx = threadIdx.x;


    const int s = q / n_q_h;
    const int qh = q % n_q_h;
    const int n_groups = n_q_h / n_kv_h;
    const int kv_head = qh / n_groups;


    for (int d_base = 0; d_base < dqkv; d_base += BLOCK_THREADS) {
        const int d = d_base + tx;
        float sum = 0.0f;


        for (int j_base = 0; j_base < cache_length; j_base += TILE_J_DIM) {


#pragma unroll
            for (int j_offset = 0; j_offset < TILE_J_DIM; ++j_offset) {
                const int j = j_base + j_offset;
                if (j < cache_length && d < dqkv) {
                    const int v_idx = (j * n_kv_h + kv_head) * dqkv + d;
                    v_tile[j_offset][tx] = V[v_idx];
                } else {
                    v_tile[j_offset][tx] = static_cast<T>(0.0f);
                }
            }


            if (tx < TILE_J_DIM) {
                const int j = j_base + tx;
                if (j < cache_length) {
                    const int att_idx = s * (n_q_h * cache_length) + qh * cache_length + j;
                    prob_tile[tx] = att_probs[att_idx];
                } else {
                    prob_tile[tx] = static_cast<T>(0.0f);
                }
            }
            __syncthreads();


#pragma unroll
            for (int j_offset = 0; j_offset < TILE_J_DIM; ++j_offset) {
                float prob = static_cast<float>(prob_tile[j_offset]);
                float val = static_cast<float>(v_tile[j_offset][tx]);
                sum += prob * val;
            }
            __syncthreads();
        }


        if (d < dqkv) {
            const int out_idx = q * dqkv + d;
            att_output[out_idx] = static_cast<T>(sum);
        }
    }
}


template <typename T>
void compute_att_output_prefill(const Tensor<T> &att_probs, const Tensor<T> &V, Tensor<T> &att_output, size_t n_q_h,
                                size_t dqkv, size_t total_seq_len, size_t n_kv_h, cudaStream_t stream) {

    size_t batch_size = att_probs.sizes()[0];
    size_t cache_length = att_probs.sizes()[2];

    if (att_probs.sizes()[1] != n_q_h) {
        throw std::runtime_error("attention probabilities head dimension mismatch");
    }
    if (V.sizes()[0] != cache_length || V.sizes()[1] != n_kv_h || V.sizes()[2] != dqkv) {
        throw std::runtime_error("V tensor dimension mismatch");
    }
    if (att_output.sizes()[0] != batch_size || att_output.sizes()[1] != n_q_h || att_output.sizes()[2] != dqkv) {
        throw std::runtime_error("attention output tensor shape mismatch");
    }


    const size_t total_q = batch_size * n_q_h;


    const int BLOCK_THREADS = 128;

    const int TILE_J_DIM = 8;


    const dim3 grid_dim(static_cast<unsigned int>(total_q));

    const dim3 block_dim(BLOCK_THREADS);


    att_output_prefill_revised_kernel<T, BLOCK_THREADS, TILE_J_DIM><<<grid_dim, block_dim, 0, stream>>>(
        att_probs.data_ptr(), V.data_ptr(), att_output.data_ptr(), static_cast<int>(total_q),
        static_cast<int>(cache_length), static_cast<int>(dqkv), static_cast<int>(n_kv_h), static_cast<int>(n_q_h));

    checkCudaError(cudaGetLastError());
}

__global__ void init_curand_state_kernel(curandState *states, unsigned long long seed, unsigned long long offset) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(seed, 0, offset, &states[0]);
    }
}
void init_curand(curandState *d_states, unsigned long long seed, int offset, cudaStream_t stream) {
    int blocks = 1;
    int threads = 1;
    init_curand_state_kernel<<<blocks, threads, 0, stream>>>(d_states, seed, offset);
    checkCudaError(cudaGetLastError());
    // if (stream == nullptr) {
    //   checkCudaError(cudaDeviceSynchronize());
    // }
}

template void rope<float>(Tensor<float> *, size_t, float, cudaStream_t);
template void rms_norm<float>(Tensor<float> *, const Tensor<float> *, const Tensor<float> *, float, cudaStream_t);

template void compute_attention_scores<float>(const Tensor<float> &, const Tensor<float> &, size_t, size_t,
                                              Tensor<float> &, size_t, cudaStream_t);
template void compute_att_output<float>(const Tensor<float> &, const Tensor<float> &, size_t, size_t, Tensor<float> &,
                                        size_t, cudaStream_t);
template void compute_attention_scores_prefill<float>(const Tensor<float> &, const Tensor<float> &, Tensor<float> &,
                                                      size_t, cudaStream_t);
template void compute_att_output_prefill<float>(const Tensor<float> &, const Tensor<float> &, Tensor<float> &, size_t,
                                                size_t, size_t, size_t, cudaStream_t);


template void rope<__half>(Tensor<__half> *, size_t, float, cudaStream_t);
template void rms_norm<__half>(Tensor<__half> *, const Tensor<__half> *, const Tensor<__half> *, float, cudaStream_t);

template void compute_attention_scores<__half>(const Tensor<__half> &, const Tensor<__half> &, size_t, size_t,
                                               Tensor<__half> &, size_t, cudaStream_t);
template void compute_att_output<__half>(const Tensor<__half> &, const Tensor<__half> &, size_t, size_t,
                                         Tensor<__half> &, size_t, cudaStream_t);
template void compute_attention_scores_prefill<__half>(const Tensor<__half> &, const Tensor<__half> &, Tensor<__half> &,
                                                       size_t, cudaStream_t);
template void compute_att_output_prefill<__half>(const Tensor<__half> &, const Tensor<__half> &, Tensor<__half> &,
                                                 size_t, size_t, size_t, size_t, cudaStream_t);


template void rope<nvbf16>(Tensor<nvbf16> *, size_t, float, cudaStream_t);
template void rms_norm<nvbf16>(Tensor<nvbf16> *, const Tensor<nvbf16> *, const Tensor<nvbf16> *, float, cudaStream_t);

template void compute_attention_scores<nvbf16>(const Tensor<nvbf16> &, const Tensor<nvbf16> &, size_t, size_t,
                                               Tensor<nvbf16> &, size_t, cudaStream_t);
template void compute_att_output<nvbf16>(const Tensor<nvbf16> &, const Tensor<nvbf16> &, size_t, size_t,
                                         Tensor<nvbf16> &, size_t, cudaStream_t);
template void compute_attention_scores_prefill<nvbf16>(const Tensor<nvbf16> &, const Tensor<nvbf16> &, Tensor<nvbf16> &,
                                                       size_t, cudaStream_t);
template void compute_att_output_prefill<nvbf16>(const Tensor<nvbf16> &, const Tensor<nvbf16> &, Tensor<nvbf16> &,
                                                 size_t, size_t, size_t, size_t, cudaStream_t);
}  // namespace cuda_OP
