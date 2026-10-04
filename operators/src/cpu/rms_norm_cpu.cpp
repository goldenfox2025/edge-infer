#include "operators/cpu/rms_norm_cpu.hpp"

#include <cuda_bf16.h>

namespace op {

template class RmsNormCPUOperator<float>;

// template class RmsNormCPUOperator<__nv_bfloat16>;

}  // namespace op
