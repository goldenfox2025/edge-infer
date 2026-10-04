#ifndef CUDA_GEMMV_OP_CUH
#define CUDA_GEMMV_OP_CUH

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

#include <cstdio>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "cuda/legacy/legacy_cuda_api.cuh"

namespace cuda_OP {


__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  // Butterfly reduction using shuffle down
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {


    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  }

  return val;
}

//----------------------------------------------------------------------------//

//----------------------------------------------------------------------------//


template <typename T, typename type_acc, int block_size>
static __global__ void gemv_s(
    const T* x,
    const T* y,
    type_acc* dst,
    const int channel_ratio,
    const int stride_channel_x,
    const int stride_channel_y,
    const int stride_channel_dst
) {

  const int64_t seq_idx = blockIdx.x;
  const int64_t channel = blockIdx.y;
  const int tid = threadIdx.x;
  constexpr int warp_size = 32;


  x += seq_idx * (gridDim.y / channel_ratio) *
           stride_channel_x +
       (channel / channel_ratio) * stride_channel_x;

  y += channel * stride_channel_y;

  dst += channel * stride_channel_dst;


  __shared__ float smem[warp_size];
  if (block_size > warp_size) {

    if (tid < warp_size) {
      smem[tid] = 0.0f;
    }
    __syncthreads();
  }


  float sumf = 0.0f;


  if constexpr (std::is_same<T, nv_bfloat16>::value) {

    constexpr int vec_unit =
        16 / sizeof(T);


    const Vec<T, vec_unit>* x_vec =
        reinterpret_cast<const Vec<T, vec_unit>*>(x);
    const Vec<T, vec_unit>* y_vec =
        reinterpret_cast<const Vec<T, vec_unit>*>(y);


    for (int64_t col_vec = tid; col_vec < stride_channel_y / vec_unit;
         col_vec += block_size) {
      Vec<T, vec_unit> xi = x_vec[col_vec];
      Vec<T, vec_unit> yi = y_vec[col_vec];

      for (int i = 0; i < vec_unit; ++i) {

        sumf += static_cast<float>(xi.t[i]) * static_cast<float>(yi.t[i]);
      }
    }

  } else if constexpr (std::is_same<T, float>::value) {

    constexpr int vec_unit = 4;
    const float4* x_vec = reinterpret_cast<const float4*>(x);
    const float4* y_vec = reinterpret_cast<const float4*>(y);

    for (int64_t col_vec = tid; col_vec < stride_channel_y / vec_unit;
         col_vec += block_size) {
      float4 xi = x_vec[col_vec];
      float4 yi = y_vec[col_vec];

      sumf += xi.x * yi.x + xi.y * yi.y + xi.z * yi.z + xi.w * yi.w;
    }

  } else {

    static_assert(std::is_same<T, void>::value,
                  "gemmv_s: unsupported data type T");
  }


  sumf = warp_reduce_sum(sumf);


  if (block_size > warp_size) {

    if (tid % warp_size == 0) {
      smem[tid / warp_size] = sumf;
    }
    __syncthreads();


    if (tid < warp_size) {

      sumf = (tid < block_size / warp_size) ? smem[tid] : 0.0f;

      sumf = warp_reduce_sum(sumf);
    }

  }


  if (tid == 0) {


    const float scale_factor = rsqrtf(128.0f);


    dst[seq_idx] = static_cast<type_acc>(sumf * scale_factor);
  }
}

//----------------------------------------------------------------------------//

//----------------------------------------------------------------------------//


template <typename T, typename AccT>
void launch_gemv_scores(
    const T* x,
    const T* y,
    AccT* dst,
    const int channel_size,
    const int channel_ratio,
    const int row_size,
    const int stride_channel_x,
    const int stride_channel_y,
    const int stride_channel_dst,
    cudaStream_t stream) {


  dim3 grid(row_size, channel_size);

  constexpr int block_size =
      128;
  dim3 block(block_size);


  gemv_s<T, AccT, block_size>
      <<<grid, block, 0, stream>>>(x, y, dst, channel_ratio, stride_channel_x,
                                   stride_channel_y, stride_channel_dst);


}


template void launch_gemv_scores<float, float>(
    const float* x, const float* y, float* dst, const int channel_size,
    const int channel_ratio, const int row_size, const int stride_channel_x,
    const int stride_channel_y, const int stride_channel_dst,
    cudaStream_t stream);


template void launch_gemv_scores<nv_bfloat16, nv_bfloat16>(
    const nv_bfloat16* x, const nv_bfloat16* y, nv_bfloat16* dst,
    const int channel_size, const int channel_ratio, const int row_size,
    const int stride_channel_x, const int stride_channel_y,
    const int stride_channel_dst, cudaStream_t stream);


template void launch_gemv_scores<nv_bfloat16, float>(
    const nv_bfloat16* x, const nv_bfloat16* y, float* dst,
    const int channel_size, const int channel_ratio, const int row_size,
    const int stride_channel_x, const int stride_channel_y,
    const int stride_channel_dst, cudaStream_t stream);

}  // namespace cuda_OP

#endif  // CUDA_GEMMV_OP_CUH