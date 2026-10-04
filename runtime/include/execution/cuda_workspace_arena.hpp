#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>

#include "execution/workspace_plan.hpp"

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

        void* new_ptr = nullptr;
        const auto result = cudaMalloc(&new_ptr, bytes);
        if (result != cudaSuccess) {
            throw std::runtime_error("Failed to allocate CUDA workspace arena: " +
                                     std::string(cudaGetErrorString(result)));
        }

        release();
        base_ptr_ = new_ptr;
        capacity_bytes_ = bytes;
        return true;
    }

    bool reserve_for_plan(const WorkspacePlan& plan) {
        return reserve(plan.total_bytes());
    }

    // The owner must finish work on its stream before resizing or releasing.
    void release() noexcept {
        if (base_ptr_ != nullptr) {
            cudaFree(base_ptr_);
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
};
