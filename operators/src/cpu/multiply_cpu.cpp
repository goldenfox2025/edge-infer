#include "operators/cpu/multiply_cpu.hpp"

#include <cuda_bf16.h>  // 添加对 __nv_bfloat16 的支持

namespace op {

// 显式模板实例化
template class MultiplyCPUOperator<float>;

// 在CPU实现中，我们只需要支持float类型
// 如果需要支持__nv_bfloat16，需要确保CPU代码能够处理这种类型
// template class MultiplyCPUOperator<__nv_bfloat16>;

}  // namespace op
