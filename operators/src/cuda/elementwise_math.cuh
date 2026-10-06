#pragma once

#include <cuda_runtime.h>

namespace op::cuda::detail {

__device__ __forceinline__ float multiply_preserving_subnormals(float a, float b) {
  // A signed zero addend makes this exactly a rounded FP32 multiply, including
  // zero signs. The IEEE intrinsic retains subnormal operands and results even
  // under -ftz=true; a tiny activation times a large value can become normal.
  const float zero = __uint_as_float((__float_as_uint(a) ^ __float_as_uint(b)) & 0x80000000U);
  return __fmaf_ieee_rn(a, b, zero);
}

}  // namespace op::cuda::detail
