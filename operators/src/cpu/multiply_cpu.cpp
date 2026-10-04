#include "operators/cpu/multiply_cpu.hpp"

#include <cuda_bf16.h>

namespace op {

template class MultiplyCPUOperator<float>;

// template class MultiplyCPUOperator<__nv_bfloat16>;

}  // namespace op
