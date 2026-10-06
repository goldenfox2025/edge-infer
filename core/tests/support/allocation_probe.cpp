#include "allocation_probe.hpp"

#include <cstdlib>
#include <limits>
#include <new>

#ifdef _WIN32
#include <malloc.h>
#endif

#ifdef EDGE_INFER_TEST_CUDA_WRAPPING
#include <cuda_runtime.h>
#endif

namespace test_alloc {
thread_local Counts counts;
thread_local bool active = false;
}

namespace {

void* allocate(std::size_t bytes) {
  if (test_alloc::active) ++test_alloc::counts.host_allocations;
  if (void* pointer = std::malloc(bytes == 0 ? 1 : bytes)) return pointer;
  throw std::bad_alloc();
}

void deallocate(void* pointer) noexcept {
  if (pointer && test_alloc::active) ++test_alloc::counts.host_frees;
  std::free(pointer);
}

void* allocate_aligned(std::size_t bytes, std::size_t alignment) {
  if (test_alloc::active) ++test_alloc::counts.host_allocations;
  bytes = bytes == 0 ? alignment : bytes;
  if (bytes > std::numeric_limits<std::size_t>::max() - alignment + 1) {
    throw std::bad_alloc();
  }
  const auto rounded = (bytes + alignment - 1) & ~(alignment - 1);
#ifdef _WIN32
  if (void* pointer = _aligned_malloc(rounded, alignment)) return pointer;
#else
  if (void* pointer = std::aligned_alloc(alignment, rounded)) return pointer;
#endif
  throw std::bad_alloc();
}

void deallocate_aligned(void* pointer) noexcept {
  if (pointer && test_alloc::active) ++test_alloc::counts.host_frees;
#ifdef _WIN32
  _aligned_free(pointer);
#else
  std::free(pointer);
#endif
}

}  // namespace

void* operator new(std::size_t bytes) { return allocate(bytes); }
void* operator new[](std::size_t bytes) { return allocate(bytes); }
void operator delete(void* pointer) noexcept { deallocate(pointer); }
void operator delete[](void* pointer) noexcept { deallocate(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { deallocate(pointer); }
void operator delete[](void* pointer, std::size_t) noexcept { deallocate(pointer); }
void* operator new(std::size_t bytes, const std::nothrow_t&) noexcept {
  try { return allocate(bytes); } catch (...) { return nullptr; }
}
void* operator new[](std::size_t bytes, const std::nothrow_t&) noexcept {
  try { return allocate(bytes); } catch (...) { return nullptr; }
}
void operator delete(void* pointer, const std::nothrow_t&) noexcept { deallocate(pointer); }
void operator delete[](void* pointer, const std::nothrow_t&) noexcept { deallocate(pointer); }
void* operator new(std::size_t bytes, std::align_val_t alignment) {
  return allocate_aligned(bytes, static_cast<std::size_t>(alignment));
}
void* operator new[](std::size_t bytes, std::align_val_t alignment) {
  return allocate_aligned(bytes, static_cast<std::size_t>(alignment));
}
void operator delete(void* pointer, std::align_val_t) noexcept { deallocate_aligned(pointer); }
void operator delete[](void* pointer, std::align_val_t) noexcept { deallocate_aligned(pointer); }
void operator delete(void* pointer, std::size_t, std::align_val_t) noexcept { deallocate_aligned(pointer); }
void operator delete[](void* pointer, std::size_t, std::align_val_t) noexcept { deallocate_aligned(pointer); }
void* operator new(std::size_t bytes, std::align_val_t alignment,
                   const std::nothrow_t&) noexcept {
  try { return allocate_aligned(bytes, static_cast<std::size_t>(alignment)); }
  catch (...) { return nullptr; }
}
void* operator new[](std::size_t bytes, std::align_val_t alignment,
                     const std::nothrow_t&) noexcept {
  try { return allocate_aligned(bytes, static_cast<std::size_t>(alignment)); }
  catch (...) { return nullptr; }
}
void operator delete(void* pointer, std::align_val_t,
                     const std::nothrow_t&) noexcept { deallocate_aligned(pointer); }
void operator delete[](void* pointer, std::align_val_t,
                       const std::nothrow_t&) noexcept { deallocate_aligned(pointer); }

#ifdef EDGE_INFER_TEST_CUDA_WRAPPING
// GNU --wrap intercepts calls emitted by the native runtime and operator
// archives. Library-private CUDA/driver allocators are outside this contract.
extern "C" cudaError_t __real_cudaMalloc(void**, std::size_t);
extern "C" cudaError_t __real_cudaMallocAsync(void**, std::size_t, cudaStream_t);
extern "C" cudaError_t __real_cudaFree(void*);
extern "C" cudaError_t __real_cudaFreeAsync(void*, cudaStream_t);
extern "C" cudaError_t __wrap_cudaMalloc(void** pointer, std::size_t bytes) {
  if (test_alloc::active) ++test_alloc::counts.device_allocations;
  return __real_cudaMalloc(pointer, bytes);
}
extern "C" cudaError_t __wrap_cudaMallocAsync(void** pointer, std::size_t bytes,
                                            cudaStream_t stream) {
  if (test_alloc::active) ++test_alloc::counts.device_allocations;
  return __real_cudaMallocAsync(pointer, bytes, stream);
}
extern "C" cudaError_t __wrap_cudaFree(void* pointer) {
  if (pointer && test_alloc::active) ++test_alloc::counts.device_frees;
  return __real_cudaFree(pointer);
}
extern "C" cudaError_t __wrap_cudaFreeAsync(void* pointer, cudaStream_t stream) {
  if (pointer && test_alloc::active) ++test_alloc::counts.device_frees;
  return __real_cudaFreeAsync(pointer, stream);
}
#endif
