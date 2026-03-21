#include <cmath>
#include <iostream>
#include <stdexcept>

#include "cuda/legacy/legacy_cuda_api.cuh"
#include "ptx_common.h"

namespace cuda_OP {

// 一个简化和优化的融合RoPE和KV缓存写入的内核。
// 它对Q和K应用RoPE变换，然后将K和V写入到各自层的缓存中。
// 设计为每层调用一次，接收指向该层缓存的指针。
template <typename T, int actual_pairs_per_thread = 2>
__global__ void rope_kv_fused_kernel(T* q_inout,            // Q张量数据，原地修改
                                     const T* k_input,      // K张量数据
                                     const T* v_input,      // V张量数据
                                     T* k_cache_layer_ptr,  // 指向当前层K缓存起始位置的指针
                                     T* v_cache_layer_ptr,  // 指向当前层V缓存起始位置的指针
                                     size_t seq_len, size_t n_heads, size_t n_kv_heads, size_t head_dim,
                                     const size_t* d_rope_offset,  // 过去序列长度（缓存中的偏移量）
                                     const float* sin_cos_cache,   // 预计算的sin/cos值
                                     size_t sin_cos_stride,        // sin_cos_cache的步长
                                     size_t kv_cache_seq_stride,   // 缓存中一个序列元素的步长
                                     size_t q_stride_s,            // Q张量的序列步长
                                     size_t k_stride_s,            // K张量的序列步长
                                     size_t v_stride_s             // V张量的序列步长
) {
    // 使用共享内存进行高效的V缓存写入
    extern __shared__ char s_raw_buffer[];
    T* s_buffer = reinterpret_cast<T*>(s_raw_buffer);

    const size_t rope_offset = *d_rope_offset;
    const size_t head_idx = blockIdx.x;
    const size_t seq_idx = blockIdx.y;

    if (seq_idx >= seq_len) {
        return;
    }

    const size_t head_dim_half = head_dim / 2;
    const size_t group_idx = threadIdx.x;
    const size_t absolute_seq_pos = seq_idx + rope_offset;

    // 指向Q输入/输出张量中当前头的指针
    T* q_head_ptr = q_inout + seq_idx * q_stride_s + head_idx * head_dim;

    const T* k_head_ptr = nullptr;
    T* k_cache_ptr = nullptr;
    const T* v_head_ptr = nullptr;
    T* v_cache_ptr = nullptr;

    // 设置K和V的指针，并开始V到共享内存的异步复制
    if (head_idx < n_kv_heads) {
        k_head_ptr = k_input + seq_idx * k_stride_s + head_idx * head_dim;
        v_head_ptr = v_input + seq_idx * v_stride_s + head_idx * head_dim;

        const size_t cache_seq_pos = rope_offset + seq_idx;
        k_cache_ptr = k_cache_layer_ptr + cache_seq_pos * kv_cache_seq_stride + head_idx * head_dim;
        v_cache_ptr = v_cache_layer_ptr + cache_seq_pos * kv_cache_seq_stride + head_idx * head_dim;

        // 异步将V从全局内存复制到共享内存以获得更快的访问速度
        unsigned int s_buffer_base_addr = __cvta_generic_to_shared(s_raw_buffer);
        constexpr int copy_bytes = 16;
        int copy_chunks = (head_dim * sizeof(T)) / copy_bytes;
        for (int i = threadIdx.x; i < copy_chunks; i += blockDim.x) {
            unsigned int dst_addr = s_buffer_base_addr + i * copy_bytes;
            CP_ASYNC_CA(dst_addr, v_head_ptr + (i * copy_bytes / sizeof(T)), copy_bytes);
        }
        CP_ASYNC_COMMIT_GROUP();
    }

    // 对Q和K应用RoPE变换
    for (int i = 0; i < actual_pairs_per_thread; ++i) {
        size_t rot_dim = group_idx * actual_pairs_per_thread + i;
        if (rot_dim < head_dim_half) {
            size_t cache_idx = absolute_seq_pos * sin_cos_stride + rot_dim * 2;
            float2 sincos = *reinterpret_cast<const float2*>(sin_cos_cache + cache_idx);

            // 对所有头应用Q变换
            if (head_idx < n_heads) {
                float q0 = static_cast<float>(q_head_ptr[rot_dim]);
                float q1 = static_cast<float>(q_head_ptr[rot_dim + head_dim_half]);
                q_head_ptr[rot_dim] = static_cast<T>(q0 * sincos.y - q1 * sincos.x);
                q_head_ptr[rot_dim + head_dim_half] = static_cast<T>(q0 * sincos.x + q1 * sincos.y);
            }

            // 对K应用变换并直接写入缓存（仅对KV头）
            if (k_cache_ptr) {
                float k0 = static_cast<float>(k_head_ptr[rot_dim]);
                float k1 = static_cast<float>(k_head_ptr[rot_dim + head_dim_half]);
                k_cache_ptr[rot_dim] = static_cast<T>(k0 * sincos.y - k1 * sincos.x);
                k_cache_ptr[rot_dim + head_dim_half] = static_cast<T>(k0 * sincos.x + k1 * sincos.y);
            }
        }
    }

    // 将V从共享内存写入全局缓存
    if (v_cache_ptr) {
        CP_ASYNC_WAIT_ALL();  // 等待异步复制完成
        __syncthreads();

        // 使用向量化写入以提高性能
        constexpr int vec_size = sizeof(float4) / sizeof(T);
        if (head_dim % vec_size == 0) {
            const float4* s_v_vec_ptr = reinterpret_cast<const float4*>(s_buffer);
            float4* v_cache_vec_ptr = reinterpret_cast<float4*>(v_cache_ptr);
            const size_t head_dim_vec = head_dim / vec_size;
            for (size_t i = threadIdx.x; i < head_dim_vec; i += blockDim.x) {
                v_cache_vec_ptr[i] = s_v_vec_ptr[i];
            }
        } else {  // 非向量化大小的回退方案
            for (size_t dim_idx = threadIdx.x; dim_idx < head_dim; dim_idx += blockDim.x) {
                v_cache_ptr[dim_idx] = s_buffer[dim_idx];
            }
        }
    }
}

// 融合RoPE和KV缓存内核的包装函数。
// 此函数通过从Tensor对象中提取参数来简化内核启动。
template <typename T>
void rope_kv_fused(Tensor<T>& q_tensor,        // 输入/输出Q张量，形状 [seq_len, n_heads, head_dim]
                   const Tensor<T>& k_tensor,  // 输入K张量，形状 [seq_len, n_kv_heads, head_dim]
                   const Tensor<T>& v_tensor,  // 输入V张量，形状 [seq_len, n_kv_heads, head_dim]
                   Tensor<T>& k_cache_layer,   // 单层的K缓存，形状 [max_len, n_kv_heads * head_dim]
                   Tensor<T>& v_cache_layer,   // 单层的V缓存，形状 [max_len, n_kv_heads * head_dim]
                   const size_t* d_rope_offset, const Tensor<float>& sin_cos_cache, cudaStream_t stream) {
    // --- 输入验证 ---
    if (q_tensor.device() != Device::CUDA || k_tensor.device() != Device::CUDA || v_tensor.device() != Device::CUDA ||
        k_cache_layer.device() != Device::CUDA || v_cache_layer.device() != Device::CUDA ||
        sin_cos_cache.device() != Device::CUDA) {
        throw std::runtime_error("rope_kv_fused: 所有张量必须在CUDA设备上。");
    }

    const auto& q_sizes = q_tensor.sizes();
    const auto& k_sizes = k_tensor.sizes();
    const auto& v_sizes = v_tensor.sizes();

    if (q_sizes.size() != 3 || k_sizes.size() != 3 || v_sizes.size() != 3) {
        throw std::runtime_error("rope_kv_fused: 输入张量Q、K、V必须是3D的。");
    }

    const size_t seq_len = q_sizes[0];
    const size_t n_heads = q_sizes[1];
    const size_t head_dim = q_sizes[2];
    const size_t n_kv_heads = k_sizes[1];

    if (seq_len == 0)
        return;  // 无事可做
    if (head_dim % 2 != 0)
        throw std::runtime_error("head_dim必须是偶数才能进行RoPE。");
    if (k_sizes[0] != seq_len || v_sizes[0] != seq_len || k_sizes[2] != head_dim || v_sizes[2] != head_dim) {
        throw std::runtime_error("rope_kv_fused: Q、K、V张量维度不一致。");
    }

    // --- 内核配置 ---
    constexpr int actual_pairs_per_thread = 2;
    size_t head_dim_half = head_dim / 2;
    int threads_per_block = (head_dim_half + actual_pairs_per_thread - 1) / actual_pairs_per_thread;
    threads_per_block = std::max(threads_per_block, (int)((head_dim + 3) / 4));  // 确保有足够的线程进行V复制
    threads_per_block = std::min(threads_per_block, 1024);

    dim3 grid_dim(n_heads, seq_len);  // 网格覆盖所有Q头和序列元素
    dim3 block_dim(threads_per_block);
    size_t smem_size = head_dim * sizeof(T);  // 一个V头的共享内存

    // --- 内核启动 ---
    rope_kv_fused_kernel<T, actual_pairs_per_thread><<<grid_dim, block_dim, smem_size, stream>>>(
        q_tensor.data_ptr(), k_tensor.data_ptr(), v_tensor.data_ptr(), k_cache_layer.data_ptr(),
        v_cache_layer.data_ptr(), seq_len, n_heads, n_kv_heads, head_dim, d_rope_offset, sin_cos_cache.data_ptr(),
        sin_cos_cache.strides()[0], k_cache_layer.strides()[0], q_tensor.strides()[0], k_tensor.strides()[0],
        v_tensor.strides()[0]);

    // --- 错误检查 ---
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("rope_kv_fused内核启动失败: " + std::string(cudaGetErrorString(err)));
    }
}

// --- 模板实例化 ---
template void rope_kv_fused<float>(Tensor<float>& q_tensor, const Tensor<float>& k_tensor,
                                   const Tensor<float>& v_tensor, Tensor<float>& k_cache_layer,
                                   Tensor<float>& v_cache_layer, const size_t* d_rope_offset,
                                   const Tensor<float>& sin_cos_cache, cudaStream_t stream);

template void rope_kv_fused<__nv_bfloat16>(Tensor<__nv_bfloat16>& q_tensor, const Tensor<__nv_bfloat16>& k_tensor,
                                           const Tensor<__nv_bfloat16>& v_tensor, Tensor<__nv_bfloat16>& k_cache_layer,
                                           Tensor<__nv_bfloat16>& v_cache_layer, const size_t* d_rope_offset,
                                           const Tensor<float>& sin_cos_cache, cudaStream_t stream);

}  // namespace cuda_OP
