#include "CudaMemoryPool.hpp"

// Define the static members declared by GlobalCudaMemoryPool.
// The out-of-line definitions provide one instance across the program.

CudaMemoryPool* GlobalCudaMemoryPool::pool_instance_ptr = nullptr;

std::mutex GlobalCudaMemoryPool::init_mutex_;


// Methods, including instance(), are defined inline in CudaMemoryPool.hpp.
// Keep the pool implementation inline in the header.
// No additional implementation is needed here.
