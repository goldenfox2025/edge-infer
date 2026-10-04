#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace test_cuda {
void initialize_random_state(curandState* state, unsigned long long seed,
                             cudaStream_t stream = nullptr);
}
