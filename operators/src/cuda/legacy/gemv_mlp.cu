#include <cmath>

#include "cuda/legacy/legacy_cuda_api.cuh"

namespace cuda_OP {

// 每个 warp 计算 MLP 层的一个输出元素。
// 此核函数融合了两个 GEMV（门控和上行投影）、一个 SiLU 激活函数以及一个逐元素乘法。
template <typename T, int WARP_SIZE = 32>
__global__ void gemv_mlp_fused_kernel(
    const T *hidden_states,  // 输入张量 [1, hidden_size]
    const T *merged_weight,  // 合并后的门控和上行权重 [2 * intermediate_size, hidden_size]
    T *output,               // 输出张量 [1, intermediate_size]
    int hidden_size, int intermediate_size) {
    // 每个 warp 负责计算中间张量中的一个输出元素 `i`。
    const int i = blockIdx.x * blockDim.y + threadIdx.y;
    if (i >= intermediate_size) {
        return;
    }

    const int lane = threadIdx.x;

    // 指向当前输出元素 `i` 对应行的起始位置的指针。
    const T *gate_weight_row = merged_weight + i * hidden_size;
    const T *up_weight_row = merged_weight + (i + intermediate_size) * hidden_size;

    float gate_val_acc = 0.0f;
    float up_val_acc = 0.0f;

    // 使用 float4 进行向量化内存访问。
    constexpr int VEC_UNIT = sizeof(float4) / sizeof(T);

    // 沿 hidden_size 维度循环，每个线程处理一个 VEC_UNIT 大小的数据块。
    for (int k = lane * VEC_UNIT; k < hidden_size; k += WARP_SIZE * VEC_UNIT) {
        // 从 hidden_states、gate_weight 和 up_weight 加载向量。
        Vec<T, VEC_UNIT> v_hidden, v_gate, v_up;
        v_hidden.f4 = *reinterpret_cast<const float4 *>(hidden_states + k);
        v_gate.f4 = *reinterpret_cast<const float4 *>(gate_weight_row + k);
        v_up.f4 = *reinterpret_cast<const float4 *>(up_weight_row + k);

// 对两个投影执行点积运算。
#pragma unroll
        for (int j = 0; j < VEC_UNIT; ++j) {
            gate_val_acc += static_cast<float>(v_hidden.t[j]) * static_cast<float>(v_gate.t[j]);
            up_val_acc += static_cast<float>(v_hidden.t[j]) * static_cast<float>(v_up.t[j]);
        }
    }

    // 在 warp 内部对部分和进行归约。
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        gate_val_acc += __shfl_xor_sync(0xffffffff, gate_val_acc, offset);
        up_val_acc += __shfl_xor_sync(0xffffffff, up_val_acc, offset);
    }
    if (lane == 0) {
        float silu_val = gate_val_acc * (1.0f / (1.0f + expf(-gate_val_acc)));

        // 逐元素乘法
        float final_val = silu_val * up_val_acc;

        output[i] = static_cast<T>(final_val);
    }
}

template <typename T>
void gemv_mlp_fused(const Tensor<T> *hidden_states, const Tensor<T> *merged_mlp_weight, Tensor<T> *output,
                    cudaStream_t stream) {
    const int hidden_size = hidden_states->sizes()[1];

    const int intermediate_size = merged_mlp_weight->sizes()[1] / 2;
    const T *d_hidden = hidden_states->data_ptr();
    const T *d_merged_w = merged_mlp_weight->data_ptr();
    T *d_output = output->data_ptr();

    // 类 GEMV 核函数的标准启动配置。
    constexpr int ROWS_PER_BLOCK = 4;
    dim3 blockDim(32, ROWS_PER_BLOCK);
    dim3 gridDim((intermediate_size + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, 1);

    gemv_mlp_fused_kernel<T>
        <<<gridDim, blockDim, 0, stream>>>(d_hidden, d_merged_w, d_output, hidden_size, intermediate_size);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("gemv_mlp_fused_kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }
}

// 显式模板实例化
template void gemv_mlp_fused<nv_bfloat16>(const Tensor<nv_bfloat16> *hidden_states,
                                          const Tensor<nv_bfloat16> *merged_mlp_weight, Tensor<nv_bfloat16> *output,
                                          cudaStream_t stream);

template void gemv_mlp_fused<float>(const Tensor<float> *hidden_states, const Tensor<float> *merged_mlp_weight,
                                    Tensor<float> *output, cudaStream_t stream);

}  // namespace cuda_OP