#include "operators/cuda/random.hpp"
#include <stdexcept>
namespace op::cuda {
namespace {
__global__ void initialize(curandState* state, unsigned long long seed, int offset) {
    curand_init(seed, 0, offset, state);
}
__global__ void values(float* out, size_t count, curandState* state) {
    auto local = *state;
    for (size_t i = 0; i < count; ++i) out[i] = curand_uniform(&local);
    *state = local;
}
void check() {
    const auto status = cudaGetLastError();
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
}
void init_curand(curandState* state, unsigned long long seed, int offset, cudaStream_t stream) {
    if (!state || offset < 0) throw std::invalid_argument("RNG requires valid state and nonnegative offset");
    initialize<<<1,1,0,stream>>>(state,seed,offset); check();
}
void generate_random_values(float* out, size_t count, curandState* state, cudaStream_t stream) {
    if (!count) return;
    if (!out || !state) throw std::invalid_argument("RNG requires caller-owned output and state");
    values<<<1,1,0,stream>>>(out,count,state); check();
}
}
