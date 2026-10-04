#pragma once

// System headers
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

// Standard library headers
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

// Project headers
#include "tensor.hpp"

//------------------------------------------------------------------------------
// CUDA error checking
//------------------------------------------------------------------------------

/**
 * @brief Check CUDA status and throw on failure.
 *
 * Check a CUDA API result and, on failure,
 * throw runtime_error with the source file and line.
 */
#define CUDA_CHECK(call)                                                                                \
    do {                                                                                                \
        cudaError_t err = call;                                                                         \
        if (err != cudaSuccess) {                                                                       \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            throw std::runtime_error(cudaGetErrorString(err));                                          \
        }                                                                                               \
    } while (0)

/**
 * @brief Backward-compatible alias for CUDA_CHECK.
 */
#define checkCudaErrors(call) CUDA_CHECK(call)

/**
 * @brief Check cuBLAS status and throw on failure.
 */
#define CUBLAS_CHECK(call)                                                                                 \
    do {                                                                                                   \
        cublasStatus_t status = call;                                                                      \
        if (status != CUBLAS_STATUS_SUCCESS) {                                                             \
            fprintf(stderr, "cuBLAS Error at %s:%d - %d\n", __FILE__, __LINE__, static_cast<int>(status)); \
            const char* error_string = "Unknown cuBLAS error";                                             \
            switch (status) {                                                                              \
                case CUBLAS_STATUS_NOT_INITIALIZED:                                                        \
                    error_string = "CUBLAS_STATUS_NOT_INITIALIZED";                                        \
                    break;                                                                                 \
                case CUBLAS_STATUS_ALLOC_FAILED:                                                           \
                    error_string = "CUBLAS_STATUS_ALLOC_FAILED";                                           \
                    break;                                                                                 \
                case CUBLAS_STATUS_INVALID_VALUE:                                                          \
                    error_string = "CUBLAS_STATUS_INVALID_VALUE";                                          \
                    break;                                                                                 \
                case CUBLAS_STATUS_ARCH_MISMATCH:                                                          \
                    error_string = "CUBLAS_STATUS_ARCH_MISMATCH";                                          \
                    break;                                                                                 \
                case CUBLAS_STATUS_MAPPING_ERROR:                                                          \
                    error_string = "CUBLAS_STATUS_MAPPING_ERROR";                                          \
                    break;                                                                                 \
                case CUBLAS_STATUS_EXECUTION_FAILED:                                                       \
                    error_string = "CUBLAS_STATUS_EXECUTION_FAILED";                                       \
                    break;                                                                                 \
                case CUBLAS_STATUS_INTERNAL_ERROR:                                                         \
                    error_string = "CUBLAS_STATUS_INTERNAL_ERROR";                                         \
                    break;                                                                                 \
                case CUBLAS_STATUS_NOT_SUPPORTED:                                                          \
                    error_string = "CUBLAS_STATUS_NOT_SUPPORTED";                                          \
                    break;                                                                                 \
                case CUBLAS_STATUS_LICENSE_ERROR:                                                          \
                    error_string = "CUBLAS_STATUS_LICENSE_ERROR";                                          \
                    break;                                                                                 \
            }                                                                                              \
            throw std::runtime_error(std::string("cuBLAS error: ") + error_string);                        \
        }                                                                                                  \
    } while (0)

/**
 * @brief Function wrapper for cuBLAS status checking.
 *
 * Use this helper when a function is more appropriate
 * than a status-checking macro.
 */
inline void checkCublasStatus(cublasStatus_t status, const char* file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        char errorMsg[256];
        const char* error_string = "Unknown cuBLAS error";
        switch (status) {
            case CUBLAS_STATUS_NOT_INITIALIZED:
                error_string = "CUBLAS_STATUS_NOT_INITIALIZED";
                break;
            case CUBLAS_STATUS_ALLOC_FAILED:
                error_string = "CUBLAS_STATUS_ALLOC_FAILED";
                break;
            case CUBLAS_STATUS_INVALID_VALUE:
                error_string = "CUBLAS_STATUS_INVALID_VALUE";
                break;
            case CUBLAS_STATUS_ARCH_MISMATCH:
                error_string = "CUBLAS_STATUS_ARCH_MISMATCH";
                break;
            case CUBLAS_STATUS_MAPPING_ERROR:
                error_string = "CUBLAS_STATUS_MAPPING_ERROR";
                break;
            case CUBLAS_STATUS_EXECUTION_FAILED:
                error_string = "CUBLAS_STATUS_EXECUTION_FAILED";
                break;
            case CUBLAS_STATUS_INTERNAL_ERROR:
                error_string = "CUBLAS_STATUS_INTERNAL_ERROR";
                break;
            case CUBLAS_STATUS_NOT_SUPPORTED:
                error_string = "CUBLAS_STATUS_NOT_SUPPORTED";
                break;
            case CUBLAS_STATUS_LICENSE_ERROR:
                error_string = "CUBLAS_STATUS_LICENSE_ERROR";
                break;
        }
        snprintf(errorMsg, sizeof(errorMsg), "cuBLAS error %d (%s) at %s:%d", static_cast<int>(status), error_string,
                 file, line);
        fprintf(stderr, "%s\n", errorMsg);
        throw std::runtime_error(errorMsg);
    }
}

/**
 * @brief Check CUTLASS status and throw on failure.
 *
 * Disabled to avoid duplicate definitions in legacy CUDA headers.
 */
/*
#define CUTLASS_CHECK(status)                                             \
  {                                                                       \
    cutlass::Status error = status;                                       \
    if (error != cutlass::Status::kSuccess) {                             \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) \
                << " at: " << __FILE__ << ":" << __LINE__ << std::endl;   \
      throw std::runtime_error("CUTLASS error");                          \
    }                                                                     \
  }
*/

//------------------------------------------------------------------------------
// Debugging helpers
//------------------------------------------------------------------------------

/**
 * @brief Print tensor information for diagnostics.
 *
 * @param tensor Tensor to print
 * @param tensor_name Tensor name for display
 * @param num_to_print Maximum number of elements to print
 */
template <typename T>
inline void debugPrintTensor(const Tensor<T>& tensor, const std::string& tensor_name, size_t num_to_print = 10) {
    std::cout << "[Debug] " << tensor_name << ":\n";

    // 1) Print shape
    std::cout << "  shape: [";
    for (auto s : tensor.sizes()) {
        std::cout << s << " ";
    }
    std::cout << "]\n";

    // 2) Print strides
    std::cout << "  strides: [";
    for (auto st : tensor.strides()) {
        std::cout << st << " ";
    }
    std::cout << "]\n";

    // 3) Print device
    std::cout << "  device: ";
    if (tensor.device() == Device::CPU) {
        std::cout << "CPU";
    } else if (tensor.device() == Device::CUDA) {
        std::cout << "CUDA";
    } else {
        std::cout << "UNKNOWN";
    }
    std::cout << "\n";

    // 4) Print elements starting from offset 0
    size_t offset = 0;  // Start printing from the beginning
    size_t total_elements = tensor.numel();
    size_t n_print = std::min(num_to_print, total_elements - offset);

    std::cout << "  elements from offset " << offset << " (" << n_print << " element(s)): ";
    if (tensor.device() == Device::CPU) {
        const T* ptr = tensor.data_ptr();
        for (size_t i = 0; i < n_print; i++) {
            std::cout << ptr[offset + i] << " ";
        }
        std::cout << "\n";
    } else {
        // Copy from GPU to CPU, then print
        std::vector<T> host_buffer(n_print);
        cudaError_t err =
            cudaMemcpy(host_buffer.data(), tensor.data_ptr() + offset, n_print * sizeof(T), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cout << "  [Error] cudaMemcpy failed\n";
            return;
        }
        for (size_t i = 0; i < n_print; i++) {
            std::cout << host_buffer[i] << " ";
        }
        std::cout << "\n";
    }
}

/**
 * @brief debugPrintTensor specialization for __nv_bfloat16.
 */
template <>
inline void debugPrintTensor<__nv_bfloat16>(const Tensor<__nv_bfloat16>& tensor, const std::string& tensor_name,
                                            size_t num_to_print) {
    std::cout << "[Debug] " << tensor_name << ":\n";

    // 1) Print shape
    std::cout << "  shape: [";
    for (auto s : tensor.sizes()) {
        std::cout << s << " ";
    }
    std::cout << "]\n";

    // 2) Print strides
    std::cout << "  strides: [";
    for (auto st : tensor.strides()) {
        std::cout << st << " ";
    }
    std::cout << "]\n";

    // 3) Print device
    std::cout << "  device: ";
    if (tensor.device() == Device::CPU) {
        std::cout << "CPU";
    } else if (tensor.device() == Device::CUDA) {
        std::cout << "CUDA";
    } else {
        std::cout << "UNKNOWN";
    }
    std::cout << "\n";

    // 4) Print elements starting from offset 0
    size_t offset = 0;
    size_t total_elements = tensor.numel();
    size_t n_print = std::min(num_to_print, total_elements - offset);

    std::cout << "  elements from offset " << offset << " (" << n_print << " element(s)): ";
    if (tensor.device() == Device::CPU) {
        const __nv_bfloat16* ptr = tensor.data_ptr();
        for (size_t i = 0; i < n_print; i++) {
            std::cout << static_cast<float>(ptr[offset + i]) << " ";
        }
        std::cout << "\n";
    } else {
        // Copy from GPU to CPU, then print
        std::vector<__nv_bfloat16> host_buffer(n_print);
        cudaError_t err = cudaMemcpy(host_buffer.data(), tensor.data_ptr() + offset, n_print * sizeof(__nv_bfloat16),
                                     cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cout << "  [Error] cudaMemcpy failed\n";
            return;
        }
        for (size_t i = 0; i < n_print; i++) {
            std::cout << static_cast<float>(host_buffer[i]) << " ";
        }
        std::cout << "\n";
    }
}

/**
 * @brief debugPrintTensor specialization for __half.
 */
template <>
inline void debugPrintTensor<__half>(const Tensor<__half>& tensor, const std::string& tensor_name,
                                     size_t num_to_print) {
    std::cout << "[Debug] " << tensor_name << ":\n";

    // 1) Print shape
    std::cout << "  shape: [";
    for (auto s : tensor.sizes()) {
        std::cout << s << " ";
    }
    std::cout << "]\n";

    // 2) Print strides
    std::cout << "  strides: [";
    for (auto st : tensor.strides()) {
        std::cout << st << " ";
    }
    std::cout << "]\n";

    // 3) Print device
    std::cout << "  device: ";
    if (tensor.device() == Device::CPU) {
        std::cout << "CPU";
    } else if (tensor.device() == Device::CUDA) {
        std::cout << "CUDA";
    } else {
        std::cout << "UNKNOWN";
    }
    std::cout << "\n";

    // 4) Print elements starting from offset 0
    size_t offset = 0;
    size_t total_elements = tensor.numel();
    size_t n_print = std::min(num_to_print, total_elements - offset);

    std::cout << "  elements from offset " << offset << " (" << n_print << " element(s)): ";
    if (tensor.device() == Device::CPU) {
        const __half* ptr = tensor.data_ptr();
        for (size_t i = 0; i < n_print; i++) {
            std::cout << static_cast<float>(ptr[offset + i]) << " ";
        }
        std::cout << "\n";
    } else {
        // Copy from GPU to CPU, then print
        std::vector<__half> host_buffer(n_print);
        cudaError_t err = cudaMemcpy(host_buffer.data(), tensor.data_ptr() + offset, n_print * sizeof(__half),
                                     cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cout << "  [Error] cudaMemcpy failed\n";
            return;
        }
        for (size_t i = 0; i < n_print; i++) {
            std::cout << static_cast<float>(host_buffer[i]) << " ";
        }
        std::cout << "\n";
    }
}

//------------------------------------------------------------------------------
// Timing helpers
//------------------------------------------------------------------------------

/**
 * @brief CUDA-event timer for kernel execution.
 *
 * Measure GPU elapsed time with CUDA events.
 */
class GpuTimer {
   private:
    cudaEvent_t start_;
    cudaEvent_t stop_;

   public:
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }

    ~GpuTimer() noexcept {
        // Report errors instead of throwing from the destructor.
        cudaError_t err = cudaEventDestroy(start_);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error in ~GpuTimer(): %s\n", cudaGetErrorString(err));
        }

        err = cudaEventDestroy(stop_);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error in ~GpuTimer(): %s\n", cudaGetErrorString(err));
        }
    }

    void start(cudaStream_t stream = nullptr) {
        CUDA_CHECK(cudaEventRecord(start_, stream));
    }

    void stop(cudaStream_t stream = nullptr) {
        CUDA_CHECK(cudaEventRecord(stop_, stream));
    }

    float milliseconds() {
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float time;
        CUDA_CHECK(cudaEventElapsedTime(&time, start_, stop_));
        return time;
    }

    float seconds() {
        return milliseconds() * 1e-3f;
    }
};

/**
 * @brief CPU execution timer.
 *
 * Measure CPU elapsed time with std::chrono.
 */
class CpuTimer {
   private:
    std::chrono::high_resolution_clock::time_point start_;
    std::chrono::high_resolution_clock::time_point stop_;

   public:
    void start() {
        start_ = std::chrono::high_resolution_clock::now();
    }

    void stop() {
        stop_ = std::chrono::high_resolution_clock::now();
    }

    double milliseconds() {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_ - start_);
        return duration.count() / 1000.0;
    }

    double seconds() {
        return milliseconds() / 1000.0;
    }
};

//------------------------------------------------------------------------------
// Thread helpers
//------------------------------------------------------------------------------

/**
 * @brief Pin the current thread to a CPU core.
 *
 * @param core_id Core ID to bind
 */
inline void bind_this_thread_to_core(int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_t current_thread = pthread_self();

    int rc = pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        std::cerr << "Error calling pthread_setaffinity_np: " << strerror(rc) << "\n";
    }
}

/**
 * @brief Return the number of available CPU cores.
 *
 * @return CPU core count
 */
inline int get_num_cores() {
    return sysconf(_SC_NPROCESSORS_ONLN);
}

//------------------------------------------------------------------------------
// Memory helpers
//------------------------------------------------------------------------------

/**
 * @brief Print current CUDA memory usage.
 *
 * @param location Label identifying the caller
 */
inline void print_cuda_memory_usage(const char* location = "Current") {
    size_t free_memory = 0, total_memory = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_memory, &total_memory));

    float free_gb = free_memory / (1024.0f * 1024.0f * 1024.0f);
    float total_gb = total_memory / (1024.0f * 1024.0f * 1024.0f);
    float used_gb = total_gb - free_gb;

    printf("[%s] CUDA Memory: Used = %.2f GB, Free = %.2f GB, Total = %.2f GB\n", location, used_gb, free_gb, total_gb);
}

/**
 * @brief Allocate device memory.
 *
 * @param size Number of bytes to allocate
 * @return Pointer to allocated memory
 */
inline void* cuda_malloc(size_t size) {
    void* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, size));
    return ptr;
}

/**
 * @brief Free device memory.
 *
 * @param ptr Pointer to free
 */
inline void cuda_free(void* ptr) {
    if (ptr) {
        CUDA_CHECK(cudaFree(ptr));
    }
}

/**
 * @brief Copy host memory to the device.
 *
 * @param dst Device destination pointer
 * @param src Host source pointer
 * @param size Number of bytes to copy
 * @param stream Optional CUDA stream
 */
inline void cuda_h2d(void* dst, const void* src, size_t size, cudaStream_t stream = nullptr) {
    if (stream) {
        CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream));
    } else {
        CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
    }
}

/**
 * @brief Copy device memory to the host.
 *
 * @param dst Host destination pointer
 * @param src Device source pointer
 * @param size Number of bytes to copy
 * @param stream Optional CUDA stream
 */
inline void cuda_d2h(void* dst, const void* src, size_t size, cudaStream_t stream = nullptr) {
    if (stream) {
        CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream));
    } else {
        CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
    }
}

/**
 * @brief Copy between device allocations.
 *
 * @param dst Device destination pointer
 * @param src Device source pointer
 * @param size Number of bytes to copy
 * @param stream Optional CUDA stream
 */
inline void cuda_d2d(void* dst, const void* src, size_t size, cudaStream_t stream = nullptr) {
    if (stream) {
        CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream));
    } else {
        CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
    }
}

/**
 * @brief Set device memory to a byte value.
 *
 * @param ptr Pointer to initialize
 * @param value Value to write
 * @param size Number of bytes to initialize
 * @param stream Optional CUDA stream
 */
inline void cuda_memset(void* ptr, int value, size_t size, cudaStream_t stream = nullptr) {
    if (stream) {
        CUDA_CHECK(cudaMemsetAsync(ptr, value, size, stream));
    } else {
        CUDA_CHECK(cudaMemset(ptr, value, size));
    }
}

/**
 * @brief Manage paired host and device allocations.
 *
 * Provide allocation, copying, and release helpers.
 */
template <typename T>
struct MemoryUnit {
    T* host_ptr;
    T* device_ptr;
    size_t size_bytes;
    size_t elements;

    MemoryUnit(size_t elements_) : size_bytes(elements_ * sizeof(T)), elements(elements_) {
        host_ptr = static_cast<T*>(malloc(elements_ * sizeof(T)));
        if (!host_ptr) {
            throw std::runtime_error("Failed to allocate host memory");
        }
        device_ptr = static_cast<T*>(cuda_malloc(elements_ * sizeof(T)));
    }

    ~MemoryUnit() {
        free_all();
    }

    void h2d(cudaStream_t stream = nullptr) {
        cuda_h2d(device_ptr, host_ptr, size_bytes, stream);
    }

    void d2h(cudaStream_t stream = nullptr) {
        cuda_d2h(host_ptr, device_ptr, size_bytes, stream);
    }

    void free_all() {
        if (host_ptr) {
            free(host_ptr);
            host_ptr = nullptr;
        }
        if (device_ptr) {
            cuda_free(device_ptr);
            device_ptr = nullptr;
        }
    }

    void init(int abs_range = 1) {
        for (size_t i = 0; i < elements; i++) {
            host_ptr[i] = static_cast<T>(rand() % 100 / static_cast<float>(100) * 2 * abs_range - abs_range);
        }
        h2d();
    }

    // Reallocate while preserving existing data.
    void resize(size_t new_elements) {
        if (new_elements == elements) {
            return;
        }

        // Allocate new host storage.
        T* new_host_ptr = static_cast<T*>(malloc(new_elements * sizeof(T)));
        if (!new_host_ptr) {
            throw std::runtime_error("Failed to allocate host memory during resize");
        }

        // Copy the existing data.
        size_t copy_elements = std::min(elements, new_elements);
        memcpy(new_host_ptr, host_ptr, copy_elements * sizeof(T));

        // Allocate new device storage.
        T* new_device_ptr = static_cast<T*>(cuda_malloc(new_elements * sizeof(T)));

        // Copy the preserved data to the device.
        if (copy_elements > 0) {
            cuda_d2d(new_device_ptr, device_ptr, copy_elements * sizeof(T));
        }

        // Release the old allocations.
        free(host_ptr);
        cuda_free(device_ptr);

        // Update pointers and size.
        host_ptr = new_host_ptr;
        device_ptr = new_device_ptr;
        elements = new_elements;
        size_bytes = new_elements * sizeof(T);
    }

    // Zero device storage.
    void zero_device(cudaStream_t stream = nullptr) {
        cuda_memset(device_ptr, 0, size_bytes, stream);
    }

    // Zero host storage.
    void zero_host() {
        memset(host_ptr, 0, size_bytes);
    }

    // Zero both allocations.
    void zero_all(cudaStream_t stream = nullptr) {
        zero_host();
        zero_device(stream);
    }
};

//------------------------------------------------------------------------------
// Utility helpers
//------------------------------------------------------------------------------

/**
 * @brief Test whether a value is a power of two.
 *
 * @param x Value to test
 * @return True if x is a power of two
 */
template <typename T>
bool is_power_of_2(T x) {
    return x > 0 && (x & (x - 1)) == 0;
}

/**
 * @brief Compute the greatest common divisor.
 *
 * @param a First operand
 * @param b Second operand
 * @return Greatest common divisor
 */
template <typename T>
T gcd(T a, T b) {
    while (b != 0) {
        T temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

/**
 * @brief Compute the least common multiple.
 *
 * @param a First operand
 * @param b Second operand
 * @return Least common multiple
 */
template <typename T>
T lcm(T a, T b) {
    return (a / gcd(a, b)) * b;
}

/**
 * @brief Round a value up to the next multiple.
 *
 * @param value Value to round up
 * @param multiple Alignment multiple
 * @return Rounded value
 */
template <typename T>
T round_up(T value, T multiple) {
    if (multiple == 0)
        return value;
    T remainder = value % multiple;
    if (remainder == 0)
        return value;
    return value + multiple - remainder;
}

//------------------------------------------------------------------------------
// CUDA helpers
//------------------------------------------------------------------------------

/**
 * @brief Return CUDA device properties.
 *
 * @param device_id Device ID
 * @return Device properties
 */
inline cudaDeviceProp get_device_properties(int device_id = 0) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    return prop;
}

/**
 * @brief Print CUDA device information.
 *
 * @param device_id Device ID
 */
inline void print_device_info(int device_id = 0) {
    cudaDeviceProp prop = get_device_properties(device_id);

    printf("Device: %s\n", prop.name);
    printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("  Total Global Memory: %.2f GB\n", static_cast<float>(prop.totalGlobalMem) / (1024.0f * 1024.0f * 1024.0f));
    printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("  Max Threads Dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1],
           prop.maxThreadsDim[2]);
    printf("  Max Grid Size: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("  Warp Size: %d\n", prop.warpSize);
    printf("  Memory Clock Rate: %.0f MHz\n", prop.memoryClockRate / 1000.0f);
    printf("  Memory Bus Width: %d bits\n", prop.memoryBusWidth);
    printf("  L2 Cache Size: %d KB\n", prop.l2CacheSize / 1024);
}

/**
 * @brief Initialize a CUDA device.
 *
 * @param device_id Device ID to initialize
 * @param print_info Whether to print device information
 */
inline void init_cuda_device(int device_id = 0, bool print_info = false) {
    CUDA_CHECK(cudaSetDevice(device_id));

    // Optionally configure the cache.
    CUDA_CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

    if (print_info) {
        print_device_info(device_id);
    }
}

/**
 * @brief Return the CUDA device count.
 *
 * @return CUDA device count
 */
inline int get_cuda_device_count() {
    int count;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    return count;
}

/**
 * @brief Return the current CUDA device.
 *
 * @return Current device ID
 */
inline int get_current_cuda_device() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    return device;
}

/**
 * @brief Synchronize the current CUDA device.
 */
inline void sync_cuda_device() {
    CUDA_CHECK(cudaDeviceSynchronize());
}

/**
 * @brief Reset the current CUDA device.
 */
inline void reset_cuda_device() {
    CUDA_CHECK(cudaDeviceReset());
}

//------------------------------------------------------------------------------
// Logging helpers
//------------------------------------------------------------------------------

/**
 * @brief Log levels
 */
enum class LogLevel { DEBUG, INFO, WARNING, ERROR, FATAL };

/**
 * @brief Logger
 */
class Logger {
   private:
    static LogLevel current_level_;
    static std::mutex log_mutex_;

    static const char* level_to_string(LogLevel level) {
        switch (level) {
            case LogLevel::DEBUG:
                return "DEBUG";
            case LogLevel::INFO:
                return "INFO";
            case LogLevel::WARNING:
                return "WARNING";
            case LogLevel::ERROR:
                return "ERROR";
            case LogLevel::FATAL:
                return "FATAL";
            default:
                return "UNKNOWN";
        }
    }

   public:
    static void set_level(LogLevel level) {
        current_level_ = level;
    }

    static LogLevel get_level() {
        return current_level_;
    }

    template <typename... Args>
    static void log(LogLevel level, const char* format, Args... args) {
        if (level < current_level_)
            return;

        std::lock_guard<std::mutex> lock(log_mutex_);

        // Get current time
        auto now = std::chrono::system_clock::now();
        auto now_c = std::chrono::system_clock::to_time_t(now);
        char time_buf[20];
        std::strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", std::localtime(&now_c));

        // Print header
        fprintf(stderr, "[%s] [%s] ", time_buf, level_to_string(level));

        // Print message
        fprintf(stderr, format, args...);
        fprintf(stderr, "\n");

        // If fatal, exit
        if (level == LogLevel::FATAL) {
            exit(EXIT_FAILURE);
        }
    }

    template <typename... Args>
    static void debug(const char* format, Args... args) {
        log(LogLevel::DEBUG, format, args...);
    }

    template <typename... Args>
    static void info(const char* format, Args... args) {
        log(LogLevel::INFO, format, args...);
    }

    template <typename... Args>
    static void warning(const char* format, Args... args) {
        log(LogLevel::WARNING, format, args...);
    }

    template <typename... Args>
    static void error(const char* format, Args... args) {
        log(LogLevel::ERROR, format, args...);
    }

    template <typename... Args>
    static void fatal(const char* format, Args... args) {
        log(LogLevel::FATAL, format, args...);
    }
};

// Initialize static members
inline LogLevel Logger::current_level_ = LogLevel::INFO;
inline std::mutex Logger::log_mutex_;

//------------------------------------------------------------------------------
// Formatting helpers
//------------------------------------------------------------------------------

/**
 * @brief Print a vector.
 *
 * @param vec Vector to print
 * @param name Vector name
 */
template <typename T>
inline void print_vector(const std::vector<T>& vec, const std::string& name = "") {
    if (!name.empty()) {
        std::cout << name << ": ";
    }

    std::cout << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << vec[i];
        if (i < vec.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}

/**
 * @brief Print a tensor shape.
 *
 * @param shape Shape to print
 * @param name Shape name
 */
template <typename T>
inline void print_shape(const std::vector<T>& shape, const std::string& name = "") {
    if (!name.empty()) {
        std::cout << name << " shape: ";
    }

    std::cout << "[";
    if (shape.empty()) {
        std::cout << "<empty>";
    } else {
        for (size_t i = 0; i < shape.size(); ++i) {
            std::cout << shape[i];
            if (i < shape.size() - 1) {
                std::cout << ", ";
            }
        }
    }
    std::cout << "]" << std::endl;
}

/**
 * @brief Format a number with thousands separators.
 *
 * @param value Number to format
 * @return Formatted string
 */
template <typename T>
inline std::string format_with_commas(T value) {
    std::string result;
    std::string str = std::to_string(value);

    int count = 0;
    for (auto it = str.rbegin(); it != str.rend(); ++it) {
        if (count > 0 && count % 3 == 0) {
            result.push_back(',');
        }
        result.push_back(*it);
        ++count;
    }

    std::reverse(result.begin(), result.end());
    return result;
}

/**
 * @brief Format a byte count for display.
 *
 * @param bytes Byte count
 * @return Formatted string
 */
inline std::string format_bytes(size_t bytes) {
    static const char* suffixes[] = {"B", "KB", "MB", "GB", "TB", "PB"};

    int suffix_idx = 0;
    double size = static_cast<double>(bytes);

    while (size >= 1024 && suffix_idx < 5) {
        size /= 1024;
        ++suffix_idx;
    }

    char buffer[32];
    if (size - static_cast<size_t>(size) == 0) {
        snprintf(buffer, sizeof(buffer), "%zu %s", static_cast<size_t>(size), suffixes[suffix_idx]);
    } else {
        snprintf(buffer, sizeof(buffer), "%.2f %s", size, suffixes[suffix_idx]);
    }

    return std::string(buffer);
}

//------------------------------------------------------------------------------
// Tensor export helpers
//------------------------------------------------------------------------------

/**
 * @brief Save a tensor to a file.
 *
 * @param tensor Tensor to save
 * @param filename Destination filename
 * @param mode File mode ("w" to write, "a" to append)
 * @return Whether saving succeeded
 */
template <typename T>
inline bool saveTensorToFile(const Tensor<T>& tensor, const std::string& filename, const std::string& mode = "w") {
    // Create the output directory if needed.
    std::string dir_path = filename.substr(0, filename.find_last_of('/'));
    if (!dir_path.empty()) {
        std::string cmd = "mkdir -p " + dir_path;
        int res = system(cmd.c_str());
        if (res != 0) {
            std::cerr << "Failed to create directory: " << dir_path << std::endl;
            return false;
        }
    }

    // Open the file.
    FILE* file = fopen(filename.c_str(), mode.c_str());
    if (!file) {
        std::cerr << "Cannot open file for writing: " << filename << std::endl;
        return false;
    }

    // Save the tensor shape.
    std::vector<size_t> shape = tensor.sizes();
    size_t ndim = shape.size();
    fwrite(&ndim, sizeof(size_t), 1, file);
    fwrite(shape.data(), sizeof(size_t), ndim, file);

    // Compute the element count.
    size_t total_elements = tensor.numel();

    // Copy GPU data to the CPU if needed.
    if (tensor.device() == Device::CUDA) {
        // Allocate a host buffer.
        std::vector<T> host_buffer(total_elements);
        // Copy the data.
        cudaError_t err =
            cudaMemcpy(host_buffer.data(), tensor.data_ptr(), total_elements * sizeof(T), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "Failed to copy data from the GPU: " << cudaGetErrorString(err) << std::endl;
            fclose(file);
            return false;
        }

        // Write the data.
        fwrite(host_buffer.data(), sizeof(T), total_elements, file);
    } else {
        // Write CPU data directly.
        fwrite(tensor.data_ptr(), sizeof(T), total_elements, file);
    }

    fclose(file);
    return true;
}

/**
 * @brief saveTensorToFile specialization for __nv_bfloat16.
 */
template <>
inline bool saveTensorToFile<__nv_bfloat16>(const Tensor<__nv_bfloat16>& tensor, const std::string& filename,
                                            const std::string& mode) {
    // Create the output directory if needed.
    std::string dir_path = filename.substr(0, filename.find_last_of('/'));
    if (!dir_path.empty()) {
        std::string cmd = "mkdir -p " + dir_path;
        int res = system(cmd.c_str());
        if (res != 0) {
            std::cerr << "Failed to create directory: " << dir_path << std::endl;
            return false;
        }
    }

    // Open the file.
    FILE* file = fopen(filename.c_str(), mode.c_str());
    if (!file) {
        std::cerr << "Cannot open file for writing: " << filename << std::endl;
        return false;
    }

    // Save the tensor shape.
    std::vector<size_t> shape = tensor.sizes();
    size_t ndim = shape.size();
    fwrite(&ndim, sizeof(size_t), 1, file);
    fwrite(shape.data(), sizeof(size_t), ndim, file);

    // Compute the element count.
    size_t total_elements = tensor.numel();

    // Copy GPU data to the CPU if needed and convert to float.
    if (tensor.device() == Device::CUDA) {
        // Allocate a host buffer.
        std::vector<__nv_bfloat16> bf16_buffer(total_elements);
        std::vector<float> float_buffer(total_elements);

        // Copy the data.
        cudaError_t err = cudaMemcpy(bf16_buffer.data(), tensor.data_ptr(), total_elements * sizeof(__nv_bfloat16),
                                     cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "Failed to copy data from the GPU: " << cudaGetErrorString(err) << std::endl;
            fclose(file);
            return false;
        }

        // Convert to float.
        for (size_t i = 0; i < total_elements; i++) {
            float_buffer[i] = static_cast<float>(bf16_buffer[i]);
        }

        // Write the data as float.
        fwrite(float_buffer.data(), sizeof(float), total_elements, file);
    } else {
        // Convert CPU data to float.
        std::vector<float> float_buffer(total_elements);
        for (size_t i = 0; i < total_elements; i++) {
            float_buffer[i] = static_cast<float>(tensor.data_ptr()[i]);
        }

        // Write the data.
        fwrite(float_buffer.data(), sizeof(float), total_elements, file);
    }

    fclose(file);
    return true;
}

/**
 * @brief saveTensorToFile specialization for __half.
 */
template <>
inline bool saveTensorToFile<__half>(const Tensor<__half>& tensor, const std::string& filename,
                                     const std::string& mode) {
    // Create the output directory if needed.
    std::string dir_path = filename.substr(0, filename.find_last_of('/'));
    if (!dir_path.empty()) {
        std::string cmd = "mkdir -p " + dir_path;
        int tmp = system(cmd.c_str());
    }

    // Open the file.
    FILE* file = fopen(filename.c_str(), mode.c_str());
    if (!file) {
        std::cerr << "Cannot open file for writing: " << filename << std::endl;
        return false;
    }

    // Save the tensor shape.
    std::vector<size_t> shape = tensor.sizes();
    size_t ndim = shape.size();
    fwrite(&ndim, sizeof(size_t), 1, file);
    fwrite(shape.data(), sizeof(size_t), ndim, file);

    // Compute the element count.
    size_t total_elements = tensor.numel();

    // Copy GPU data to the CPU if needed and convert to float.
    if (tensor.device() == Device::CUDA) {
        // Allocate a host buffer.
        std::vector<__half> half_buffer(total_elements);
        std::vector<float> float_buffer(total_elements);

        // Copy the data.
        cudaError_t err =
            cudaMemcpy(half_buffer.data(), tensor.data_ptr(), total_elements * sizeof(__half), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "Failed to copy data from the GPU: " << cudaGetErrorString(err) << std::endl;
            fclose(file);
            return false;
        }

        // Convert to float.
        for (size_t i = 0; i < total_elements; i++) {
            float_buffer[i] = static_cast<float>(half_buffer[i]);
        }

        // Write the data as float.
        fwrite(float_buffer.data(), sizeof(float), total_elements, file);
    } else {
        // Convert CPU data to float.
        std::vector<float> float_buffer(total_elements);
        for (size_t i = 0; i < total_elements; i++) {
            float_buffer[i] = static_cast<float>(tensor.data_ptr()[i]);
        }

        // Write the data.
        fwrite(float_buffer.data(), sizeof(float), total_elements, file);
    }

    fclose(file);
    return true;
}
