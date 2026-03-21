#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <memory>
#include <stdexcept>
#include <type_traits>

#include "operators/operator_base.hpp"
#include "operators/operator_factory.hpp"
#include "operators/cuda/cuda_resource_manager.cuh"

namespace op::detail {

inline OperatorPlatform platform_from_device(Device device) {
  return device == Device::CUDA ? OperatorPlatform::CUDA
                                : OperatorPlatform::CPU;
}

template <typename T>
inline void register_platform_operators(OperatorPlatform platform) {
  if (platform == OperatorPlatform::CUDA) {
    OperatorFactory<T>::registerCUDAOperators();
    return;
  }
  OperatorFactory<T>::registerCPUOperators();
}

template <typename T>
inline void configure_backend(Device& device, OperatorPlatform& platform) {
  platform = platform_from_device(device);
  register_platform_operators<T>(platform);

  if (platform != OperatorPlatform::CUDA) {
    return;
  }

  try {
    int device_count = 0;
    const cudaError_t count_status = cudaGetDeviceCount(&device_count);
    if (count_status != cudaSuccess || device_count <= 0) {
      std::fprintf(stderr,
                   "CUDA runtime unavailable: %s. Falling back to CPU.\n",
                   cudaGetErrorString(count_status));
      cudaGetLastError();  // 清理 runtime error 状态，避免污染后续调用
      device = Device::CPU;
      platform = OperatorPlatform::CPU;
      register_platform_operators<T>(platform);
      return;
    }

    auto& resource_manager = CUDAResourceManager::instance();
    resource_manager.getCublasHandle();
  } catch (const std::exception& e) {
    std::fprintf(stderr,
                 "CUDA initialization failed: %s. Falling back to CPU.\n",
                 e.what());
    device = Device::CPU;
    platform = OperatorPlatform::CPU;
    register_platform_operators<T>(platform);
  }
}

template <typename T, typename OperatorPtr>
inline OperatorPtr require_operator(OperatorPtr op, OperatorPlatform platform,
                                    const char* op_name) {
  if (op) {
    return op;
  }

  if (platform == OperatorPlatform::CPU &&
      std::is_same_v<T, __nv_bfloat16>) {
    throw std::runtime_error(std::string(op_name) +
                             " for __nv_bfloat16 not supported on CPU");
  }

  throw std::runtime_error(std::string(op_name) +
                           " operator not registered for current platform");
}

}  // namespace op::detail
