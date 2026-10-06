#include <float.h>

#include <cmath>  // For std::isfinite, fmaxf, __expf
#include <limits> // For std::numeric_limits
#include <vector>

#include "operators/cuda/execution_kernels.cuh"

namespace cuda_OP
{


template <typename T>
__global__ void gather_fa_kernel_variable(const float *T1_ptr, const float *T2_ptr, const float *T3_ptr,
                                         const float *T4_ptr, const float *T5_ptr,
                                         T *output_ptr,
                                         int branch_count,
                                         int q_h, int dqkv)
{

  int head_id = blockIdx.x;
  int tid = threadIdx.x;
  if (head_id >= q_h || tid >= dqkv)
  {
    return;
  }


  int input_stride = dqkv + 2;
  int output_stride = dqkv;
  int base_in = head_id * input_stride;
  int base_out = head_id * output_stride;


  float global_m = -FLT_MAX;
  float global_l = 0.0f;
  float global_o = 0.0f;


  if (branch_count > 0 && T1_ptr != nullptr) {
    float m1 = static_cast<float>(T1_ptr[base_in + dqkv]);
    float l1 = static_cast<float>(T1_ptr[base_in + dqkv + 1]);
    float o1 = static_cast<float>(T1_ptr[base_in + tid]);

    global_m = m1;
    global_l = l1;
    global_o = o1;
  }


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


  float final_out = (global_l > 0.0f) ? global_o / global_l : 0.0f;
  output_ptr[base_out + tid] = static_cast<T>(final_out);
}


} // namespace cuda_OP

namespace op::cuda::detail {

template <typename T>
void launch_attention_gather(const ExecutionContext& context, const float* first,
    const float* second, const float* third, const float* fourth, const float* fifth,
    int branches, TensorView<T, 3> output) {
  cuda_OP::gather_fa_kernel_variable<T><<<output.shape[1], output.shape[2], 0, context.stream>>>(
      first, second, third, fourth, fifth, output.data, branches, output.shape[1], output.shape[2]);
  const auto result = cudaGetLastError();
  if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

template void launch_attention_gather<float>(const ExecutionContext&, const float*, const float*, const float*, const float*, const float*, int, TensorView<float, 3>);
template void launch_attention_gather<__nv_bfloat16>(const ExecutionContext&, const float*, const float*, const float*, const float*, const float*, int, TensorView<__nv_bfloat16, 3>);

}  // namespace op::cuda::detail
