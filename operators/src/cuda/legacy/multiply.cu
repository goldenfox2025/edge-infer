#include <cuda_runtime.h>
#include <math.h>

#include <cstdio>
#include <iostream>
#include <vector>

#include "cuda/legacy/legacy_cuda_api.cuh"

namespace cuda_OP {

template <typename T>
__global__ void multiply_kernel_v3(const T *A, const T *B, T *out, int total) {

  constexpr int vec_unit = 16 / sizeof(T);
  typedef Vec<T, vec_unit> VecT;


  int total_vec = total / vec_unit;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;


  const VecT *A_vec = reinterpret_cast<const VecT *>(A);
  const VecT *B_vec = reinterpret_cast<const VecT *>(B);
  VecT *out_vec = reinterpret_cast<VecT *>(out);


  for (int i = tid; i < total_vec; i += stride) {
    VecT a_val = A_vec[i];
    VecT b_val = B_vec[i];
    VecT result;
#pragma unroll
    for (int j = 0; j < vec_unit; j++) {
      result.t[j] = a_val.t[j] * b_val.t[j];
    }
    out_vec[i] = result;
  }


  int remaining = total - total_vec * vec_unit;
  int offset = total_vec * vec_unit;
  for (int i = tid; i < remaining; i += stride) {
    out[offset + i] = A[offset + i] * B[offset + i];
  }
}

// --------------------------------------------------

// --------------------------------------------------
template <typename T>
__global__ void multiply_kernel_v2(const T *A, const T *B, T *out, int total) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total) {
    out[idx] = __ldg(&A[idx]) * __ldg(&B[idx]);
  }
}

// --------------------------------------------------

// --------------------------------------------------
template <typename T>
__global__ void multiply_kernel_v1(const T *A, const T *B, T *out, int total) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total) {
    out[idx] = A[idx] * B[idx];
  }
}

// --------------------------------------------------


// --------------------------------------------------
template <typename T>
void multiply(Tensor<T> *output, const Tensor<T> *A, const Tensor<T> *B,
              cudaStream_t stream) {
  size_t total = A->numel();
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  multiply_kernel_v3<T><<<blocks, threads, 0, stream>>>(
      A->data_ptr(), B->data_ptr(), output->data_ptr(), total);
  checkCudaError(cudaGetLastError());
  // if (stream == nullptr) {
  //   checkCudaError(cudaDeviceSynchronize());
  // }
}
template void multiply<float>(Tensor<float> *, const Tensor<float> *,
                              const Tensor<float> *, cudaStream_t);

template void multiply<nvbf16>(Tensor<nvbf16> *, const Tensor<nvbf16> *,
                               const Tensor<nvbf16> *, cudaStream_t);
}  // namespace cuda_OP
