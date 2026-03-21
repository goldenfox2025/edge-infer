#include <float.h>

#include <cmath>  // For std::isfinite, fmaxf, __expf
#include <limits> // For std::numeric_limits
#include <vector>

#include "cuda/legacy/legacy_cuda_api.cuh"

namespace cuda_OP
{

// 最多支持5个分支的gather_fa kernel
template <typename T>
__global__ void gather_fa_kernel_variable(const T *T1_ptr, const T *T2_ptr, const T *T3_ptr,
                                         const T *T4_ptr, const T *T5_ptr,
                                         T *output_ptr,
                                         int branch_count,
                                         int q_h, int dqkv)
{
  // 每个 block 对应一个 head，每个线程处理该 head 中一个维度
  int head_id = blockIdx.x;
  int tid = threadIdx.x;
  if (head_id >= q_h || tid >= dqkv)
  {
    return;
  }

  // 每个 head 中 T 张量的 stride（最后 2 个元素用于存储 m 与 l）
  int input_stride = dqkv + 2;
  int output_stride = dqkv;
  int base_in = head_id * input_stride;
  int base_out = head_id * output_stride;

  // 初始化归约变量
  float global_m = -FLT_MAX;
  float global_l = 0.0f;
  float global_o = 0.0f;

  // 处理第一个分支
  if (branch_count > 0 && T1_ptr != nullptr) {
    float m1 = static_cast<float>(T1_ptr[base_in + dqkv]);
    float l1 = static_cast<float>(T1_ptr[base_in + dqkv + 1]);
    float o1 = static_cast<float>(T1_ptr[base_in + tid]);

    global_m = m1;
    global_l = l1;
    global_o = o1;
  }

  // 处理第二个分支
  if (branch_count > 1 && T2_ptr != nullptr) {
    float m2 = static_cast<float>(T2_ptr[base_in + dqkv]);
    float l2 = static_cast<float>(T2_ptr[base_in + dqkv + 1]);
    float o2 = static_cast<float>(T2_ptr[base_in + tid]);

    float old_global_m = global_m;
    float old_global_l = global_l;
    float new_global_m = fmaxf(old_global_m, m2);
    float exp_old = __expf(old_global_m - new_global_m);
    float exp_cur = __expf(m2 - new_global_m);
    global_l = old_global_l * exp_old + l2 * exp_cur;
    global_o = global_o * exp_old + o2 * exp_cur;
    global_m = new_global_m;
  }

  // 处理第三个分支
  if (branch_count > 2 && T3_ptr != nullptr) {
    float m3 = static_cast<float>(T3_ptr[base_in + dqkv]);
    float l3 = static_cast<float>(T3_ptr[base_in + dqkv + 1]);
    float o3 = static_cast<float>(T3_ptr[base_in + tid]);

    float old_global_m = global_m;
    float old_global_l = global_l;
    float new_global_m = fmaxf(old_global_m, m3);
    float exp_old = __expf(old_global_m - new_global_m);
    float exp_cur = __expf(m3 - new_global_m);
    global_l = old_global_l * exp_old + l3 * exp_cur;
    global_o = global_o * exp_old + o3 * exp_cur;
    global_m = new_global_m;
  }

  // 处理第四个分支
  if (branch_count > 3 && T4_ptr != nullptr) {
    float m4 = static_cast<float>(T4_ptr[base_in + dqkv]);
    float l4 = static_cast<float>(T4_ptr[base_in + dqkv + 1]);
    float o4 = static_cast<float>(T4_ptr[base_in + tid]);

    float old_global_m = global_m;
    float old_global_l = global_l;
    float new_global_m = fmaxf(old_global_m, m4);
    float exp_old = __expf(old_global_m - new_global_m);
    float exp_cur = __expf(m4 - new_global_m);
    global_l = old_global_l * exp_old + l4 * exp_cur;
    global_o = global_o * exp_old + o4 * exp_cur;
    global_m = new_global_m;
  }

  // 处理第五个分支
  if (branch_count > 4 && T5_ptr != nullptr) {
    float m5 = static_cast<float>(T5_ptr[base_in + dqkv]);
    float l5 = static_cast<float>(T5_ptr[base_in + dqkv + 1]);
    float o5 = static_cast<float>(T5_ptr[base_in + tid]);

    float old_global_m = global_m;
    float old_global_l = global_l;
    float new_global_m = fmaxf(old_global_m, m5);
    float exp_old = __expf(old_global_m - new_global_m);
    float exp_cur = __expf(m5 - new_global_m);
    global_l = old_global_l * exp_old + l5 * exp_cur;
    global_o = global_o * exp_old + o5 * exp_cur;
    global_m = new_global_m;
  }

  // 最终归一化
  float final_out = (global_l > 0.0f) ? global_o / global_l : 0.0f;
  output_ptr[base_out + tid] = static_cast<T>(final_out);
}

// 特化版本的gather_fa实现 - 1分支
template <typename T>
void gather_fa_specialized_1branch(const std::vector<Tensor<T>>& inputs, Tensor<T>& output,
                                 cudaStream_t stream)
{
  // 验证输入
  if (inputs.size() != 1) {
    throw std::runtime_error("Mismatch in number of input tensors for 1-branch specialization");
  }

  int dqkv = output.sizes()[1];
  int q_h = output.sizes()[0];

  // 准备各分支的指针
  const T* T1_ptr = inputs[0].data_ptr();

  // 设置kernel参数
  dim3 grid(q_h);
  dim3 block(dqkv);

  // 启动kernel - 直接传递指针，不需要额外的内存复制
  gather_fa_kernel_variable<T><<<grid, block, 0, stream>>>(
      T1_ptr, nullptr, nullptr, nullptr, nullptr,
      output.data_ptr(), 1, q_h, dqkv);
}

// 特化版本的gather_fa实现 - 2分支
template <typename T>
void gather_fa_specialized_2branch(const std::vector<Tensor<T>>& inputs, Tensor<T>& output,
                                 cudaStream_t stream)
{
  // 验证输入
  if (inputs.size() != 2) {
    throw std::runtime_error("Mismatch in number of input tensors for 2-branch specialization");
  }

  int dqkv = output.sizes()[1];
  int q_h = output.sizes()[0];

  // 准备各分支的指针
  const T* T1_ptr = inputs[0].data_ptr();
  const T* T2_ptr = inputs[1].data_ptr();

  // 设置kernel参数
  dim3 grid(q_h);
  dim3 block(dqkv);

  // 启动kernel - 直接传递指针，不需要额外的内存复制
  gather_fa_kernel_variable<T><<<grid, block, 0, stream>>>(
      T1_ptr, T2_ptr, nullptr, nullptr, nullptr,
      output.data_ptr(), 2, q_h, dqkv);
}

// 特化版本的gather_fa实现 - 3分支
template <typename T>
void gather_fa_specialized_3branch(const std::vector<Tensor<T>>& inputs, Tensor<T>& output,
                                 cudaStream_t stream)
{
  // 验证输入
  if (inputs.size() != 3) {
    throw std::runtime_error("Mismatch in number of input tensors for 3-branch specialization");
  }

  int dqkv = output.sizes()[1];
  int q_h = output.sizes()[0];

  // 准备各分支的指针
  const T* T1_ptr = inputs[0].data_ptr();
  const T* T2_ptr = inputs[1].data_ptr();
  const T* T3_ptr = inputs[2].data_ptr();

  // 设置kernel参数
  dim3 grid(q_h);
  dim3 block(dqkv);

  // 启动kernel - 直接传递指针，不需要额外的内存复制
  gather_fa_kernel_variable<T><<<grid, block, 0, stream>>>(
      T1_ptr, T2_ptr, T3_ptr, nullptr, nullptr,
      output.data_ptr(), 3, q_h, dqkv);
}

// 特化版本的gather_fa实现 - 4分支
template <typename T>
void gather_fa_specialized_4branch(const std::vector<Tensor<T>>& inputs, Tensor<T>& output,
                                 cudaStream_t stream)
{
  // 验证输入
  if (inputs.size() != 4) {
    throw std::runtime_error("Mismatch in number of input tensors for 4-branch specialization");
  }

  int dqkv = output.sizes()[1];
  int q_h = output.sizes()[0];

  // 准备各分支的指针
  const T* T1_ptr = inputs[0].data_ptr();
  const T* T2_ptr = inputs[1].data_ptr();
  const T* T3_ptr = inputs[2].data_ptr();
  const T* T4_ptr = inputs[3].data_ptr();

  // 设置kernel参数
  dim3 grid(q_h);
  dim3 block(dqkv);

  // 启动kernel - 直接传递指针，不需要额外的内存复制
  gather_fa_kernel_variable<T><<<grid, block, 0, stream>>>(
      T1_ptr, T2_ptr, T3_ptr, T4_ptr, nullptr,
      output.data_ptr(), 4, q_h, dqkv);
}

// 特化版本的gather_fa实现 - 5分支
template <typename T>
void gather_fa_specialized_5branch(const std::vector<Tensor<T>>& inputs, Tensor<T>& output,
                                 cudaStream_t stream)
{
  // 验证输入
  if (inputs.size() != 5) {
    throw std::runtime_error("Mismatch in number of input tensors for 5-branch specialization");
  }

  int dqkv = output.sizes()[1];
  int q_h = output.sizes()[0];

  // 准备各分支的指针
  const T* T1_ptr = inputs[0].data_ptr();
  const T* T2_ptr = inputs[1].data_ptr();
  const T* T3_ptr = inputs[2].data_ptr();
  const T* T4_ptr = inputs[3].data_ptr();
  const T* T5_ptr = inputs[4].data_ptr();

  // 设置kernel参数
  dim3 grid(q_h);
  dim3 block(dqkv);

  // 启动kernel - 直接传递指针，不需要额外的内存复制
  gather_fa_kernel_variable<T><<<grid, block, 0, stream>>>(
      T1_ptr, T2_ptr, T3_ptr, T4_ptr, T5_ptr,
      output.data_ptr(), 5, q_h, dqkv);
}

// 可变分支数量的gather_fa实现
template <typename T>
void gather_fa_variable(const std::vector<Tensor<T>>& inputs, Tensor<T>& output,
                       cudaStream_t stream)
{
  int branch_count = inputs.size();
  if (branch_count == 0) {
    throw std::runtime_error("No input tensors provided to gather_fa_variable");
  }

  if (branch_count > 5) {
    throw std::runtime_error("gather_fa_variable supports at most 5 branches");
  }

  // 根据分支数量选择特化版本
  switch (branch_count) {
    case 1:
      gather_fa_specialized_1branch(inputs, output, stream);
      break;
    case 2:
      gather_fa_specialized_2branch(inputs, output, stream);
      break;
    case 3:
      gather_fa_specialized_3branch(inputs, output, stream);
      break;
    case 4:
      gather_fa_specialized_4branch(inputs, output, stream);
      break;
    case 5:
      gather_fa_specialized_5branch(inputs, output, stream);
      break;
    default:
      // 不应该到达这里，因为我们已经限制了分支数量
      throw std::runtime_error("Unexpected branch count: " + std::to_string(branch_count));
  }

  // 检查错误
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA error in gather_fa_variable: " +
                            std::string(cudaGetErrorString(err)));
  }
}

// 模板实例化 - 原始函数
template void gather_fa_variable<float>(
    const std::vector<Tensor<float>>& inputs,
    Tensor<float>& output,
    cudaStream_t stream);

template void gather_fa_variable<__nv_bfloat16>(
    const std::vector<Tensor<__nv_bfloat16>>& inputs,
    Tensor<__nv_bfloat16>& output,
    cudaStream_t stream);

// 模板实例化 - 特化版本 (1分支)
template void gather_fa_specialized_1branch<float>(
    const std::vector<Tensor<float>>& inputs,
    Tensor<float>& output,
    cudaStream_t stream);

template void gather_fa_specialized_1branch<__nv_bfloat16>(
    const std::vector<Tensor<__nv_bfloat16>>& inputs,
    Tensor<__nv_bfloat16>& output,
    cudaStream_t stream);

// 模板实例化 - 特化版本 (2分支)
template void gather_fa_specialized_2branch<float>(
    const std::vector<Tensor<float>>& inputs,
    Tensor<float>& output,
    cudaStream_t stream);

template void gather_fa_specialized_2branch<__nv_bfloat16>(
    const std::vector<Tensor<__nv_bfloat16>>& inputs,
    Tensor<__nv_bfloat16>& output,
    cudaStream_t stream);

// 模板实例化 - 特化版本 (3分支)
template void gather_fa_specialized_3branch<float>(
    const std::vector<Tensor<float>>& inputs,
    Tensor<float>& output,
    cudaStream_t stream);

template void gather_fa_specialized_3branch<__nv_bfloat16>(
    const std::vector<Tensor<__nv_bfloat16>>& inputs,
    Tensor<__nv_bfloat16>& output,
    cudaStream_t stream);

// 模板实例化 - 特化版本 (4分支)
template void gather_fa_specialized_4branch<float>(
    const std::vector<Tensor<float>>& inputs,
    Tensor<float>& output,
    cudaStream_t stream);

template void gather_fa_specialized_4branch<__nv_bfloat16>(
    const std::vector<Tensor<__nv_bfloat16>>& inputs,
    Tensor<__nv_bfloat16>& output,
    cudaStream_t stream);

// 模板实例化 - 特化版本 (5分支)
template void gather_fa_specialized_5branch<float>(
    const std::vector<Tensor<float>>& inputs,
    Tensor<float>& output,
    cudaStream_t stream);

template void gather_fa_specialized_5branch<__nv_bfloat16>(
    const std::vector<Tensor<__nv_bfloat16>>& inputs,
    Tensor<__nv_bfloat16>& output,
    cudaStream_t stream);

} // namespace cuda_OP
