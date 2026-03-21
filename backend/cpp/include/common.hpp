#pragma once

// 系统包含
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

// 标准库包含
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

// 项目包含
#include "tensor.hpp"

//------------------------------------------------------------------------------
// CUDA 错误检查宏
//------------------------------------------------------------------------------

/**
 * @brief 检查CUDA错误并在发生错误时抛出异常
 *
 * 此宏检查CUDA API调用的结果，如果调用失败，
 * 则抛出包含文件和行信息的runtime_error异常。
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
 * @brief CUDA_CHECK的别名，用于向后兼容
 */
#define checkCudaErrors(call) CUDA_CHECK(call)

/**
 * @brief 检查cuBLAS状态并在发生错误时抛出异常
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
 * @brief 检查cuBLAS状态的辅助函数
 *
 * 当需要在函数中检查cuBLAS状态而不是直接使用宏时，
 * 可以使用此函数。
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
 * @brief 检查CUTLASS状态并在发生错误时抛出异常
 *
 * 注意：这部分被注释掉以避免与 legacy CUDA 头中的定义重复
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
// 调试工具
//------------------------------------------------------------------------------

/**
 * @brief 打印张量信息用于调试
 *
 * @param tensor 要打印的张量
 * @param tensor_name 用于显示的张量名称
 * @param num_to_print 要打印的最大元素数量
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
 * @brief debugPrintTensor针对__nv_bfloat16类型的特化版本
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
 * @brief debugPrintTensor针对__half类型的特化版本
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
// 计时工具
//------------------------------------------------------------------------------

/**
 * @brief GPU计时器，用于测量内核执行时间
 *
 * 使用CUDA事件来测量GPU上的经过时间
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
        // 在析构函数中不抛出异常，而是打印错误信息
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
 * @brief CPU计时器，用于测量执行时间
 *
 * 使用std::chrono来测量CPU上的经过时间
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
// 线程管理工具
//------------------------------------------------------------------------------

/**
 * @brief 将当前线程绑定到特定的CPU核心
 *
 * @param core_id 要绑定的核心ID
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
 * @brief 获取可用的CPU核心数量
 *
 * @return CPU核心数量
 */
inline int get_num_cores() {
    return sysconf(_SC_NPROCESSORS_ONLN);
}

//------------------------------------------------------------------------------
// 内存管理工具
//------------------------------------------------------------------------------

/**
 * @brief 打印当前CUDA内存使用情况
 *
 * @param location 用于标识此函数从何处调用的字符串
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
 * @brief 分配设备内存
 *
 * @param size 要分配的字节大小
 * @return 指向已分配内存的指针
 */
inline void* cuda_malloc(size_t size) {
    void* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, size));
    return ptr;
}

/**
 * @brief 释放设备内存
 *
 * @param ptr 要释放的内存指针
 */
inline void cuda_free(void* ptr) {
    if (ptr) {
        CUDA_CHECK(cudaFree(ptr));
    }
}

/**
 * @brief 将内存从主机复制到设备
 *
 * @param dst 目标指针（设备）
 * @param src 源指针（主机）
 * @param size 要复制的字节大小
 * @param stream 要使用的CUDA流（可选）
 */
inline void cuda_h2d(void* dst, const void* src, size_t size, cudaStream_t stream = nullptr) {
    if (stream) {
        CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream));
    } else {
        CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
    }
}

/**
 * @brief 将内存从设备复制到主机
 *
 * @param dst 目标指针（主机）
 * @param src 源指针（设备）
 * @param size 要复制的字节大小
 * @param stream 要使用的CUDA流（可选）
 */
inline void cuda_d2h(void* dst, const void* src, size_t size, cudaStream_t stream = nullptr) {
    if (stream) {
        CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream));
    } else {
        CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
    }
}

/**
 * @brief 将内存从设备复制到设备
 *
 * @param dst 目标指针（设备）
 * @param src 源指针（设备）
 * @param size 要复制的字节大小
 * @param stream 要使用的CUDA流（可选）
 */
inline void cuda_d2d(void* dst, const void* src, size_t size, cudaStream_t stream = nullptr) {
    if (stream) {
        CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream));
    } else {
        CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
    }
}

/**
 * @brief 将设备内存设置为指定值
 *
 * @param ptr 要设置的内存指针
 * @param value 要设置的值
 * @param size 要设置的字节大小
 * @param stream 要使用的CUDA流（可选）
 */
inline void cuda_memset(void* ptr, int value, size_t size, cudaStream_t stream = nullptr) {
    if (stream) {
        CUDA_CHECK(cudaMemsetAsync(ptr, value, size, stream));
    } else {
        CUDA_CHECK(cudaMemset(ptr, value, size));
    }
}

/**
 * @brief 用于管理主机和设备内存的简单内存单元
 *
 * 提供用于分配、复制和释放内存的工具
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

    // 重新分配内存，保留原有数据
    void resize(size_t new_elements) {
        if (new_elements == elements) {
            return;
        }

        // 分配新的主机内存
        T* new_host_ptr = static_cast<T*>(malloc(new_elements * sizeof(T)));
        if (!new_host_ptr) {
            throw std::runtime_error("Failed to allocate host memory during resize");
        }

        // 复制原有数据
        size_t copy_elements = std::min(elements, new_elements);
        memcpy(new_host_ptr, host_ptr, copy_elements * sizeof(T));

        // 分配新的设备内存
        T* new_device_ptr = static_cast<T*>(cuda_malloc(new_elements * sizeof(T)));

        // 复制原有数据到设备
        if (copy_elements > 0) {
            cuda_d2d(new_device_ptr, device_ptr, copy_elements * sizeof(T));
        }

        // 释放旧内存
        free(host_ptr);
        cuda_free(device_ptr);

        // 更新指针和大小
        host_ptr = new_host_ptr;
        device_ptr = new_device_ptr;
        elements = new_elements;
        size_bytes = new_elements * sizeof(T);
    }

    // 清零设备内存
    void zero_device(cudaStream_t stream = nullptr) {
        cuda_memset(device_ptr, 0, size_bytes, stream);
    }

    // 清零主机内存
    void zero_host() {
        memset(host_ptr, 0, size_bytes);
    }

    // 清零所有内存
    void zero_all(cudaStream_t stream = nullptr) {
        zero_host();
        zero_device(stream);
    }
};

//------------------------------------------------------------------------------
// 杂项工具
//------------------------------------------------------------------------------

/**
 * @brief 检查一个数字是否为2的幂
 *
 * @param x 要检查的数字
 * @return 如果x是2的幂则返回true，否则返回false
 */
template <typename T>
bool is_power_of_2(T x) {
    return x > 0 && (x & (x - 1)) == 0;
}

/**
 * @brief 计算两个数的最大公约数
 *
 * @param a 第一个数
 * @param b 第二个数
 * @return 最大公约数
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
 * @brief 计算两个数的最小公倍数
 *
 * @param a 第一个数
 * @param b 第二个数
 * @return 最小公倍数
 */
template <typename T>
T lcm(T a, T b) {
    return (a / gcd(a, b)) * b;
}

/**
 * @brief 向上取整到给定数字的下一个倍数
 *
 * @param value 要向上取整的值
 * @param multiple 要取整到的倍数
 * @return 向上取整后的值
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
// CUDA特定工具
//------------------------------------------------------------------------------

/**
 * @brief 获取CUDA设备属性
 *
 * @param device_id 设备ID
 * @return 设备属性
 */
inline cudaDeviceProp get_device_properties(int device_id = 0) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    return prop;
}

/**
 * @brief 打印CUDA设备信息
 *
 * @param device_id 设备ID
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
 * @brief 初始化CUDA设备
 *
 * @param device_id 要初始化的设备ID
 * @param print_info 是否打印设备信息
 */
inline void init_cuda_device(int device_id = 0, bool print_info = false) {
    CUDA_CHECK(cudaSetDevice(device_id));

    // 可选：设置缓存配置
    CUDA_CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

    if (print_info) {
        print_device_info(device_id);
    }
}

/**
 * @brief 获取CUDA设备数量
 *
 * @return CUDA设备数量
 */
inline int get_cuda_device_count() {
    int count;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    return count;
}

/**
 * @brief 获取当前CUDA设备
 *
 * @return 当前设备ID
 */
inline int get_current_cuda_device() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    return device;
}

/**
 * @brief 同步当前CUDA设备
 */
inline void sync_cuda_device() {
    CUDA_CHECK(cudaDeviceSynchronize());
}

/**
 * @brief 重置当前CUDA设备
 */
inline void reset_cuda_device() {
    CUDA_CHECK(cudaDeviceReset());
}

//------------------------------------------------------------------------------
// 调试和日志工具
//------------------------------------------------------------------------------

/**
 * @brief 日志级别枚举
 */
enum class LogLevel { DEBUG, INFO, WARNING, ERROR, FATAL };

/**
 * @brief 简单的日志记录器类
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
// 打印和格式化工具
//------------------------------------------------------------------------------

/**
 * @brief 打印向量
 *
 * @param vec 要打印的向量
 * @param name 向量的名称
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
 * @brief 打印形状（维度向量）
 *
 * @param shape 要打印的形状
 * @param name 形状的名称
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
 * @brief 格式化数字，使用逗号作为千位分隔符
 *
 * @param value 要格式化的数字
 * @return 格式化后的字符串
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
 * @brief 将字节大小格式化为人类可读的字符串
 *
 * @param bytes 字节大小
 * @return 格式化后的字符串
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
// Tensor保存工具
//------------------------------------------------------------------------------

/**
 * @brief 将张量保存到文件
 *
 * @param tensor 要保存的张量
 * @param filename 目标文件名
 * @param mode 文件打开模式 ("w" 为写入, "a" 为追加)
 * @return 是否成功保存
 */
template <typename T>
inline bool saveTensorToFile(const Tensor<T>& tensor, const std::string& filename, const std::string& mode = "w") {
    // 创建输出目录（如果不存在）
    std::string dir_path = filename.substr(0, filename.find_last_of('/'));
    if (!dir_path.empty()) {
        std::string cmd = "mkdir -p " + dir_path;
        int res = system(cmd.c_str());
        if (res != 0) {
            std::cerr << "创建目录失败: " << dir_path << std::endl;
            return false;
        }
    }

    // 打开文件
    FILE* file = fopen(filename.c_str(), mode.c_str());
    if (!file) {
        std::cerr << "无法打开文件进行写入: " << filename << std::endl;
        return false;
    }

    // 保存张量形状
    std::vector<size_t> shape = tensor.sizes();
    size_t ndim = shape.size();
    fwrite(&ndim, sizeof(size_t), 1, file);
    fwrite(shape.data(), sizeof(size_t), ndim, file);

    // 计算数据总量
    size_t total_elements = tensor.numel();

    // 从GPU复制数据到CPU（如果需要）
    if (tensor.device() == Device::CUDA) {
        // 分配主机缓冲区
        std::vector<T> host_buffer(total_elements);
        // 拷贝数据
        cudaError_t err =
            cudaMemcpy(host_buffer.data(), tensor.data_ptr(), total_elements * sizeof(T), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "从GPU复制数据失败: " << cudaGetErrorString(err) << std::endl;
            fclose(file);
            return false;
        }

        // 写入数据
        fwrite(host_buffer.data(), sizeof(T), total_elements, file);
    } else {
        // 直接写入CPU数据
        fwrite(tensor.data_ptr(), sizeof(T), total_elements, file);
    }

    fclose(file);
    return true;
}

/**
 * @brief 针对__nv_bfloat16类型的saveTensorToFile特化版本
 */
template <>
inline bool saveTensorToFile<__nv_bfloat16>(const Tensor<__nv_bfloat16>& tensor, const std::string& filename,
                                            const std::string& mode) {
    // 创建输出目录（如果不存在）
    std::string dir_path = filename.substr(0, filename.find_last_of('/'));
    if (!dir_path.empty()) {
        std::string cmd = "mkdir -p " + dir_path;
        int res = system(cmd.c_str());
        if (res != 0) {
            std::cerr << "创建目录失败: " << dir_path << std::endl;
            return false;
        }
    }

    // 打开文件
    FILE* file = fopen(filename.c_str(), mode.c_str());
    if (!file) {
        std::cerr << "无法打开文件进行写入: " << filename << std::endl;
        return false;
    }

    // 保存张量形状
    std::vector<size_t> shape = tensor.sizes();
    size_t ndim = shape.size();
    fwrite(&ndim, sizeof(size_t), 1, file);
    fwrite(shape.data(), sizeof(size_t), ndim, file);

    // 计算数据总量
    size_t total_elements = tensor.numel();

    // 从GPU复制数据到CPU（如果需要），并转换为float
    if (tensor.device() == Device::CUDA) {
        // 分配主机缓冲区
        std::vector<__nv_bfloat16> bf16_buffer(total_elements);
        std::vector<float> float_buffer(total_elements);

        // 拷贝数据
        cudaError_t err = cudaMemcpy(bf16_buffer.data(), tensor.data_ptr(), total_elements * sizeof(__nv_bfloat16),
                                     cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "从GPU复制数据失败: " << cudaGetErrorString(err) << std::endl;
            fclose(file);
            return false;
        }

        // 转换为float
        for (size_t i = 0; i < total_elements; i++) {
            float_buffer[i] = static_cast<float>(bf16_buffer[i]);
        }

        // 写入数据（作为float保存）
        fwrite(float_buffer.data(), sizeof(float), total_elements, file);
    } else {
        // CPU数据转换为float
        std::vector<float> float_buffer(total_elements);
        for (size_t i = 0; i < total_elements; i++) {
            float_buffer[i] = static_cast<float>(tensor.data_ptr()[i]);
        }

        // 写入数据
        fwrite(float_buffer.data(), sizeof(float), total_elements, file);
    }

    fclose(file);
    return true;
}

/**
 * @brief 针对__half类型的saveTensorToFile特化版本
 */
template <>
inline bool saveTensorToFile<__half>(const Tensor<__half>& tensor, const std::string& filename,
                                     const std::string& mode) {
    // 创建输出目录（如果不存在）
    std::string dir_path = filename.substr(0, filename.find_last_of('/'));
    if (!dir_path.empty()) {
        std::string cmd = "mkdir -p " + dir_path;
        int tmp = system(cmd.c_str());
    }

    // 打开文件
    FILE* file = fopen(filename.c_str(), mode.c_str());
    if (!file) {
        std::cerr << "无法打开文件进行写入: " << filename << std::endl;
        return false;
    }

    // 保存张量形状
    std::vector<size_t> shape = tensor.sizes();
    size_t ndim = shape.size();
    fwrite(&ndim, sizeof(size_t), 1, file);
    fwrite(shape.data(), sizeof(size_t), ndim, file);

    // 计算数据总量
    size_t total_elements = tensor.numel();

    // 从GPU复制数据到CPU（如果需要），并转换为float
    if (tensor.device() == Device::CUDA) {
        // 分配主机缓冲区
        std::vector<__half> half_buffer(total_elements);
        std::vector<float> float_buffer(total_elements);

        // 拷贝数据
        cudaError_t err =
            cudaMemcpy(half_buffer.data(), tensor.data_ptr(), total_elements * sizeof(__half), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "从GPU复制数据失败: " << cudaGetErrorString(err) << std::endl;
            fclose(file);
            return false;
        }

        // 转换为float
        for (size_t i = 0; i < total_elements; i++) {
            float_buffer[i] = static_cast<float>(half_buffer[i]);
        }

        // 写入数据（作为float保存）
        fwrite(float_buffer.data(), sizeof(float), total_elements, file);
    } else {
        // CPU数据转换为float
        std::vector<float> float_buffer(total_elements);
        for (size_t i = 0; i < total_elements; i++) {
            float_buffer[i] = static_cast<float>(tensor.data_ptr()[i]);
        }

        // 写入数据
        fwrite(float_buffer.data(), sizeof(float), total_elements, file);
    }

    fclose(file);
    return true;
}
