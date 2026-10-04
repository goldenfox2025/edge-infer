#pragma once
#include "tensor.hpp"
#include <mutex>


class DeviceManager {
public:
    static DeviceManager& instance() {
        static DeviceManager instance;
        return instance;
    }
    void setDefaultDevice(Device device) {
        std::lock_guard<std::mutex> lock(mutex_);
        default_device_ = device;
    }

    Device getDefaultDevice() const {
        return default_device_;
    }

    bool isCudaAvailable() const {
        int deviceCount = 0;
        cudaError_t error = cudaGetDeviceCount(&deviceCount);
        return (error == cudaSuccess && deviceCount > 0);
    }

private:
    DeviceManager() : default_device_(Device::CUDA) {
        if (!isCudaAvailable()) {
            default_device_ = Device::CPU;
        }
    }

    // Prevent copying and assignment
    DeviceManager(const DeviceManager&) = delete;
    DeviceManager& operator=(const DeviceManager&) = delete;

    Device default_device_;
    std::mutex mutex_;
};
