#include "operators/cpu/add_cpu.hpp"

#include <cuda_bf16.h>

namespace op {

template class AddCPUOperator<float>;

// template class AddCPUOperator<__nv_bfloat16>;

}  // namespace op
