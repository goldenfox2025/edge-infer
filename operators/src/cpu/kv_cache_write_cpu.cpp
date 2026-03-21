#include "operators/cpu/kv_cache_write_cpu.hpp"

#include <cuda_bf16.h>

namespace op {

template class KvCacheWriteCPUOperator<float>;

}  // namespace op
