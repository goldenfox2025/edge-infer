// Elementwise addition: use vector pairs only when pointers are aligned and the element count is even.
#ifndef CUDA_ADD_OP_CUH
#define CUDA_ADD_OP_CUH

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <vector>


#include "cuda/legacy/legacy_cuda_api.cuh"


namespace cuda_OP {

// --------------------------------------------------


// --------------------------------------------------
template <typename T>
__global__ void add_kernel_v1(const T* A,
                              const T* B,
                              T* out,
                              size_t total) {

  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;


  for (size_t i = idx; i < total; i += stride) {

    out[i] = A[i] + B[i];
  }
}

// --------------------------------------------------


// --------------------------------------------------


__global__ void add_kernel_float_v2(
    const float2* A,
    const float2* B,
    float2* out,
    size_t total_vec2)
{

  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;


  for (size_t i = idx; i < total_vec2; i += stride) {
    float2 a_val = A[i];
    float2 b_val = B[i];
    float2 result;

    result.x = a_val.x + b_val.x;
    result.y = a_val.y + b_val.y;
    out[i] = result;
  }
}


__global__ void add_kernel_bf16_v2(
    const nv_bfloat162* A,
    const nv_bfloat162* B,
    nv_bfloat162* out,
    size_t total_vec2)
{

  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;


  for (size_t i = idx; i < total_vec2; i += stride) {
#if __CUDA_ARCH__ >= 800

    out[i] = __hadd2(A[i], B[i]);
#else

    nv_bfloat162 a_val = A[i];
    nv_bfloat162 b_val = B[i];
    nv_bfloat162 result;

    result.x = a_val.x + b_val.x;
    result.y = a_val.y + b_val.y;
    out[i] = result;
#endif
  }
}

// --------------------------------------------------


// --------------------------------------------------
template <typename T>
void add(Tensor<T>* output,
         Tensor<T>* A,
         Tensor<T>* B,
         cudaStream_t stream) {

  if (A->numel() != B->numel() || A->numel() != output->numel()) {
    throw std::runtime_error("Tensor addition requires matching shapes");
  }
  if (A->device() != Device::CUDA || B->device() != Device::CUDA ||
      output->device() != Device::CUDA) {
    throw std::runtime_error("All tensors must be on a CUDA device");
  }

  size_t total = A->numel();
  if (total == 0) {
    return;
  }


  int threads = 256;


  bool can_vectorize = false;
  size_t vec_alignment = 0;


  if constexpr (std::is_same_v<T, float>) {
    vec_alignment = alignof(float2);
    can_vectorize = true;
  } else if constexpr (std::is_same_v<T, nvbf16>) {
    vec_alignment = alignof(nv_bfloat162);
    can_vectorize = true;
  }


  bool use_vectorized_kernel =
      can_vectorize && (total % 2 == 0) &&
      (reinterpret_cast<uintptr_t>(A->data_ptr()) % vec_alignment == 0) &&
      (reinterpret_cast<uintptr_t>(B->data_ptr()) % vec_alignment == 0) &&
      (reinterpret_cast<uintptr_t>(output->data_ptr()) % vec_alignment == 0);


  if (use_vectorized_kernel) {

    size_t total_vec2 = total / 2;


    int device;
    checkCudaError(cudaGetDevice(&device));
    int numSMs;
    checkCudaError(cudaDeviceGetAttribute(
        &numSMs, cudaDevAttrMultiProcessorCount, device));

    int blocks = std::max(
        1, std::min((int)((total_vec2 + threads - 1) / threads), numSMs * 32));


    if constexpr (std::is_same_v<T, float>) {
      add_kernel_float_v2<<<blocks, threads, 0, stream>>>(
          reinterpret_cast<const float2*>(A->data_ptr()),
          reinterpret_cast<const float2*>(B->data_ptr()),
          reinterpret_cast<float2*>(output->data_ptr()),
          total_vec2);
    } else if constexpr (std::is_same_v<T, nvbf16>) {
      add_kernel_bf16_v2<<<blocks, threads, 0, stream>>>(
          reinterpret_cast<const nv_bfloat162*>(A->data_ptr()),
          reinterpret_cast<const nv_bfloat162*>(B->data_ptr()),
          reinterpret_cast<nv_bfloat162*>(output->data_ptr()),
          total_vec2);
    }
  } else {


    int device;
    checkCudaError(cudaGetDevice(&device));
    int numSMs;
    checkCudaError(cudaDeviceGetAttribute(
        &numSMs, cudaDevAttrMultiProcessorCount, device));
    int blocks = std::max(
        1, std::min((int)((total + threads - 1) / threads), numSMs * 32));


    add_kernel_v1<T><<<blocks, threads, 0, stream>>>(
        A->data_ptr(), B->data_ptr(), output->data_ptr(), total);
  }


  checkCudaError(cudaGetLastError());

}


template void add<float>(Tensor<float>*, Tensor<float>*, Tensor<float>*, cudaStream_t);
template void add<nvbf16>(Tensor<nvbf16>*, Tensor<nvbf16>*, Tensor<nvbf16>*, cudaStream_t);

} // namespace cuda_OP

#endif // CUDA_ADD_OP_CUH