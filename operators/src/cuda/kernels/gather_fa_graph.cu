#include <float.h>

#include <cmath>
#include <limits>
#include <vector>

#include "operators/cuda/execution_kernels.cuh"

#define MAX_BRANCHES 3

namespace cuda_OP {


template <typename T>
__global__ void gather_fa_kernel_graph_fixed(float **input_ptrs, T *output_ptr, int *segment_info, int q_h, int dqkv) {

    //   int total_seq_len = segment_info[0];


    // const int FIXED_BRANCHES = 3;


    int head_id = blockIdx.x;
    int tid = threadIdx.x;
    if (head_id >= q_h || tid >= dqkv) {
        return;
    }


    int input_stride = dqkv + 2;
    int output_stride = dqkv;
    int base_in = head_id * input_stride;
    int base_out = head_id * output_stride;


    float global_m;
    float global_l;
    float global_o;

    float *T1_ptr = input_ptrs[0];
    float m1 = static_cast<float>(T1_ptr[base_in + dqkv]);
    float l1 = static_cast<float>(T1_ptr[base_in + dqkv + 1]);
    float o1 = static_cast<float>(T1_ptr[base_in + tid]);

    global_m = m1;
    global_l = l1;
    global_o = o1;

    float *T2_ptr = input_ptrs[1];
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

    float *T3_ptr = input_ptrs[2];
    float m3 = static_cast<float>(T3_ptr[base_in + dqkv]);
    float l3 = static_cast<float>(T3_ptr[base_in + dqkv + 1]);
    float o3 = static_cast<float>(T3_ptr[base_in + tid]);

    old_global_m = global_m;
    old_global_l = global_l;
    new_global_m = fmaxf(old_global_m, m3);
    exp_old = __expf(old_global_m - new_global_m);
    exp_cur = __expf(m3 - new_global_m);
    global_l = old_global_l * exp_old + l3 * exp_cur;
    global_o = global_o * exp_old + o3 * exp_cur;
    global_m = new_global_m;

    float final_out = (global_l > 0.0f) ? global_o / global_l : 0.0f;
    output_ptr[base_out + tid] = static_cast<T>(final_out);
}


}  // namespace cuda_OP

namespace op::cuda::detail {

template <typename T>
void launch_graph_gather(const ExecutionContext& context, float** branches,
                          TensorView<T, 3> output, int* lengths) {
  cuda_OP::gather_fa_kernel_graph_fixed<T><<<output.shape[1], output.shape[2], 0, context.stream>>>(
      branches, output.data, lengths, output.shape[1], output.shape[2]);
  const auto result = cudaGetLastError();
  if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
template void launch_graph_gather<float>(const ExecutionContext&, float**, TensorView<float, 3>, int*);
template void launch_graph_gather<__nv_bfloat16>(const ExecutionContext&, float**, TensorView<__nv_bfloat16, 3>, int*);

}  // namespace op::cuda::detail
