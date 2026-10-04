#include "operators/cpu/rope_cpu.hpp"

#include <cuda_bf16.h>

namespace op {

template class RopeCPUOperator<float>;

// template class RopeCPUOperator<__nv_bfloat16>;

}  // namespace op
