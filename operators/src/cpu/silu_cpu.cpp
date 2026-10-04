#include "operators/cpu/silu_cpu.hpp"

#include <cuda_bf16.h>

namespace op {

template class SiluCPUOperator<float>;

// template class SiluCPUOperator<__nv_bfloat16>;

}  // namespace op
