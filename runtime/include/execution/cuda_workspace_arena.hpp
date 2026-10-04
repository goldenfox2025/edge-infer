#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>

#include "execution/workspace_plan.hpp"
#include "tensor_view.hpp"

class CudaWorkspaceArena {
   public:
    CudaWorkspaceArena() = default;

    ~CudaWorkspaceArena() {
        release();
    }

    CudaWorkspaceArena(const CudaWorkspaceArena&) = delete;
    CudaWorkspaceArena& operator=(const CudaWorkspaceArena&) = delete;
    CudaWorkspaceArena(CudaWorkspaceArena&&) = delete;
    CudaWorkspaceArena& operator=(CudaWorkspaceArena&&) = delete;

    bool reserve(size_t bytes) {
        if (bytes == 0) {
            return false;
        }
        if (capacity_bytes_ >= bytes) {
            return false;
        }

        int current_device = 0;
        auto result = cudaGetDevice(&current_device);
        if (result != cudaSuccess) {
            throw std::runtime_error("Failed to identify CUDA workspace device: " +
                                     std::string(cudaGetErrorString(result)));
        }
        if (base_ptr_ && current_device != device_id_) {
            throw std::invalid_argument("CUDA workspace must grow on its owning device");
        }
        void* new_ptr = nullptr;
        result = cudaMalloc(&new_ptr, bytes);
        if (result != cudaSuccess) {
            throw std::runtime_error("Failed to allocate CUDA workspace arena: " +
                                     std::string(cudaGetErrorString(result)));
        }

        release();
        base_ptr_ = new_ptr;
        capacity_bytes_ = bytes;
        device_id_ = current_device;
        return true;
    }

    bool reserve_for_plan(const WorkspacePlan& plan) {
        return reserve(plan.total_bytes());
    }

    // Resolve planned offsets once during preparation; returned descriptors borrow
    // the arena and are invalidated by an arena growth or release.
    template <typename T, size_t Rank>
    TensorView<T, Rank> bind_view(const WorkspaceAllocation& allocation,
                                 const std::array<size_t, Rank>& shape) {
        if (allocation.offset > capacity_bytes_ ||
            allocation.bytes > capacity_bytes_ - allocation.offset ||
            allocation.offset % alignof(T)) {
            throw std::invalid_argument("Workspace allocation cannot be bound to this arena");
        }
        auto result = TensorView<T, Rank>::contiguous(ptr_at<T>(allocation.offset), shape);
        if (result.numel() > allocation.bytes / sizeof(T)) {
            throw std::invalid_argument("TensorView exceeds its planned workspace allocation");
        }
        return result;
    }

    // The owner must finish work on its stream before resizing or releasing.
    void release() noexcept {
        if (base_ptr_ != nullptr) {
            int previous_device = device_id_;
            cudaGetDevice(&previous_device);
            if (previous_device != device_id_) cudaSetDevice(device_id_);
            cudaFree(base_ptr_);
            if (previous_device != device_id_) cudaSetDevice(previous_device);
            base_ptr_ = nullptr;
        }
        capacity_bytes_ = 0;
    }

    bool empty() const {
        return base_ptr_ == nullptr;
    }

    size_t capacity_bytes() const {
        return capacity_bytes_;
    }

    void* base_ptr() {
        return base_ptr_;
    }

    const void* base_ptr() const {
        return base_ptr_;
    }

    template <typename T>
    T* ptr_at(size_t offset_bytes) {
        validate_offset(offset_bytes);
        return reinterpret_cast<T*>(static_cast<char*>(base_ptr_) + offset_bytes);
    }

    template <typename T>
    const T* ptr_at(size_t offset_bytes) const {
        validate_offset(offset_bytes);
        return reinterpret_cast<const T*>(static_cast<const char*>(base_ptr_) + offset_bytes);
    }

   private:
    void validate_offset(size_t offset_bytes) const {
        if (base_ptr_ == nullptr) {
            throw std::runtime_error("CUDA workspace arena is not allocated");
        }
        if (offset_bytes >= capacity_bytes_) {
            throw std::runtime_error("CUDA workspace arena offset out of range");
        }
    }

    void* base_ptr_ = nullptr;
    size_t capacity_bytes_ = 0;
    int device_id_ = 0;
};
