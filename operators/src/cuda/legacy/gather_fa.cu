#include <float.h>

#include <cmath>  // For std::isfinite, fmaxf, __expf
#include <limits> // For std::numeric_limits

#include "cuda/legacy/legacy_cuda_api.cuh"

namespace cuda_OP
{


  template <typename T>
  __global__ void gather_fa_kernel_3_inputs_optimized(const T *T1_ptr,
                                                      const T *T2_ptr,
                                                      const T *T3_ptr, T *T5_ptr,
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


    float m1 = static_cast<float>(T1_ptr[base_in + dqkv]);
    float l1 = static_cast<float>(T1_ptr[base_in + dqkv + 1]);
    float o1 = static_cast<float>(T1_ptr[base_in + tid]);


    float global_m = -FLT_MAX;
    float global_l = 0.0f;
    float global_o = 0.0f;

    global_m = m1;
    global_l = l1;
    global_o = o1;

    // --- Chunk 2 ---
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

    // --- Chunk 3 ---
    float m3 = static_cast<float>(T3_ptr[base_in + dqkv]);
    float l3 = static_cast<float>(T3_ptr[base_in + dqkv + 1]);
    float o3 = static_cast<float>(T3_ptr[base_in + tid]);

    old_global_m = global_m;
    old_global_l = global_l;
    new_global_m =
        fmaxf(old_global_m, m3);
    exp_old =
        __expf(old_global_m - new_global_m);
    exp_cur = __expf(m3 - new_global_m);
    global_l = old_global_l * exp_old + l3 * exp_cur;
    global_o = global_o * exp_old + o3 * exp_cur;
    global_m = new_global_m;


    float final_out =
        (global_l > 0.0f) ? global_o / global_l : 0.0f;
    T5_ptr[base_out + tid] = static_cast<T>(final_out);
  }


  template <typename T>
  void gather_fa(const Tensor<T> &T1, const Tensor<T> &T2, const Tensor<T> &T3,
                 Tensor<T> &T5, cudaStream_t stream)
  {
    int dqkv = T5.sizes()[1];
    int q_h = T5.sizes()[0];

    dim3 grid(q_h);
    dim3 block(dqkv);

    gather_fa_kernel_3_inputs_optimized<T><<<grid, block, 0, stream>>>(
        T1.data_ptr(), T2.data_ptr(), T3.data_ptr(), T5.data_ptr(), q_h, dqkv);


  }


  template void gather_fa<float>(const Tensor<float> &T1, const Tensor<float> &T2,
                                 const Tensor<float> &T3, Tensor<float> &T5,
                                 cudaStream_t stream);
  template void gather_fa<__nv_bfloat16>(const Tensor<__nv_bfloat16> &T1,
                                         const Tensor<__nv_bfloat16> &T2,
                                         const Tensor<__nv_bfloat16> &T3,
                                         Tensor<__nv_bfloat16> &T5,
                                         cudaStream_t stream);

} // namespace cuda_OP
