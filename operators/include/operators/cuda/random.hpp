#pragma once
#include <curand_kernel.h>
#include <cstddef>
namespace op::cuda {
// Borrowed RNG state is owned by the caller and serialized on its stream.
void init_curand(curandState*, unsigned long long seed, int offset, cudaStream_t stream = nullptr);
void generate_random_values(float*, size_t count, curandState*, cudaStream_t stream = nullptr);
}
