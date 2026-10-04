#include "test_cuda_rng.hpp"

#include <stdexcept>

namespace test_cuda {
namespace {
__global__ void initialize_kernel(curandState* state, unsigned long long seed) {
    if (!threadIdx.x && !blockIdx.x) curand_init(seed, 0, 0, state);
}
}
void initialize_random_state(curandState* state, unsigned long long seed,
                             cudaStream_t stream) {
    initialize_kernel<<<1, 1, 0, stream>>>(state, seed);
    const auto result = cudaGetLastError();
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
}
