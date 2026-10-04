#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <unordered_map>
#include <vector>

class PrefillWorkspaceArena {
   public:
    struct Stats {
        size_t reserved_bytes = 0;
        size_t committed_bytes = 0;
        size_t used_bytes = 0;
        size_t peak_used_bytes = 0;
    };

    PrefillWorkspaceArena()
        : enabled_(false),
          phase_(false),
          base_ptr_(nullptr),
          reserved_bytes_(0),
          used_bytes_(0),
          committed_bytes_(0),
          max_bytes_(256 * 1024 * 1024),
          granularity_(0),
          peak_used_bytes_(0) {}

    bool enabled() const { return enabled_; }
    bool phase() const { return phase_; }

    void set_phase(bool is_prefill) { phase_ = is_prefill; }

    bool enable(size_t initial_size, size_t max_size) {
        if (enabled_) {
            disable_no_tracking();
        }

        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location = {CU_MEM_LOCATION_TYPE_DEVICE, 0};
        cuMemGetAllocationGranularity(&granularity_, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
        if (granularity_ == 0) {
            return false;
        }

        max_bytes_ = align_to_granularity(max_size);
        if (max_bytes_ == 0) {
            return false;
        }

        CUdeviceptr reserved_ptr = 0;
        if (cuMemAddressReserve(&reserved_ptr, max_bytes_, 0, 0, 0) != CUDA_SUCCESS) {
            return false;
        }

        base_ptr_ = reinterpret_cast<void*>(reserved_ptr);
        reserved_bytes_ = max_bytes_;
        used_bytes_ = 0;
        committed_bytes_ = 0;
        peak_used_bytes_ = 0;

        const size_t adjusted_initial_size = adjust_initial_size(initial_size);
        if (adjusted_initial_size > 0 && !commit_until(adjusted_initial_size)) {
            disable_no_tracking();
            return false;
        }

        enabled_ = true;
        std::cerr << "PrefillWorkspaceArena: enabled. reserved=" << (reserved_bytes_ / (1024.0 * 1024.0))
                  << " MB, committed=" << (committed_bytes_ / (1024.0 * 1024.0)) << " MB" << std::endl;
        return true;
    }

    void disable(std::unordered_map<void*, size_t>& active_allocations) {
        clear_active_allocations(active_allocations);
        disable_no_tracking();
    }

    void reset(std::unordered_map<void*, size_t>& active_allocations) {
        if (!enabled_ || base_ptr_ == nullptr) {
            return;
        }
        clear_active_allocations(active_allocations);
        peak_used_bytes_ = std::max(peak_used_bytes_, used_bytes_);
        used_bytes_ = 0;
    }

    bool prepare(size_t requested_bytes, std::unordered_map<void*, size_t>& active_allocations) {
        if (!enabled_ || base_ptr_ == nullptr) {
            return false;
        }

        clear_active_allocations(active_allocations);
        peak_used_bytes_ = std::max(peak_used_bytes_, used_bytes_);
        used_bytes_ = 0;
        if (requested_bytes == 0) {
            return true;
        }

        return commit_until(requested_bytes);
    }

    void* try_allocate(size_t aligned_size, std::unordered_map<void*, size_t>& active_allocations) {
        if (!enabled_ || base_ptr_ == nullptr) {
            return nullptr;
        }

        const size_t required_used = used_bytes_ + aligned_size;
        if (!commit_until(required_used)) {
            return nullptr;
        }

        void* ptr = static_cast<char*>(base_ptr_) + used_bytes_;
        used_bytes_ = required_used;
        peak_used_bytes_ = std::max(peak_used_bytes_, used_bytes_);
        active_allocations[ptr] = aligned_size;
        return ptr;
    }

    bool owns(const void* ptr) const {
        if (!enabled_ || base_ptr_ == nullptr) {
            return false;
        }

        const char* p = static_cast<const char*>(ptr);
        const char* start = static_cast<const char*>(base_ptr_);
        const char* end = start + reserved_bytes_;
        return p >= start && p < end;
    }

    void free_allocation(void*, size_t) {
        // Arena allocations are reclaimed in bulk by reset()/disable().
    }

    Stats stats() const {
        return {reserved_bytes_, committed_bytes_, used_bytes_, peak_used_bytes_};
    }

   private:
    struct VmmChunk {
        CUmemGenericAllocationHandle handle = {};
        size_t size = 0;
    };

    bool enabled_;
    bool phase_;
    void* base_ptr_;
    size_t reserved_bytes_;
    size_t used_bytes_;
    size_t committed_bytes_;
    size_t max_bytes_;
    size_t granularity_;
    size_t peak_used_bytes_;
    std::vector<VmmChunk> chunks_;

    size_t align_to_granularity(size_t size) const {
        if (granularity_ == 0) {
            return 0;
        }
        return (std::max(size, granularity_) + granularity_ - 1) & ~(granularity_ - 1);
    }

    size_t adjust_initial_size(size_t requested_size) const {
        if (granularity_ == 0) {
            return 0;
        }

        size_t free_memory = 0;
        size_t total_memory = 0;
        if (cudaMemGetInfo(&free_memory, &total_memory) == cudaSuccess) {
            if (requested_size > free_memory * 0.8) {
                requested_size = static_cast<size_t>(free_memory * 0.7);
            }
        }

        return align_to_granularity(requested_size);
    }

    bool commit_until(size_t requested_bytes) {
        if (requested_bytes <= committed_bytes_) {
            return true;
        }
        if (requested_bytes > reserved_bytes_) {
            return false;
        }

        while (committed_bytes_ < requested_bytes) {
            const size_t remaining_bytes = reserved_bytes_ - committed_bytes_;
            const size_t min_chunk_size = std::max(granularity_, requested_bytes - committed_bytes_);
            size_t new_chunk_size = 0;
            if (committed_bytes_ == 0) {
                new_chunk_size = align_to_granularity(min_chunk_size);
            } else {
                const size_t growth_target = std::max(min_chunk_size * 2, committed_bytes_ / 4);
                new_chunk_size = align_to_granularity(growth_target);
            }
            new_chunk_size = std::min(new_chunk_size, remaining_bytes);
            if (new_chunk_size < min_chunk_size) {
                return false;
            }

            CUmemAllocationProp prop = {};
            prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location = {CU_MEM_LOCATION_TYPE_DEVICE, 0};

            CUmemGenericAllocationHandle handle = {};
            if (cuMemCreate(&handle, new_chunk_size, &prop, 0) != CUDA_SUCCESS) {
                return false;
            }

            CUdeviceptr va_ptr = reinterpret_cast<CUdeviceptr>(base_ptr_);
            if (cuMemMap(va_ptr + committed_bytes_, new_chunk_size, 0, handle, 0) != CUDA_SUCCESS) {
                cuMemRelease(handle);
                return false;
            }

            CUmemAccessDesc access_desc = {};
            access_desc.location = prop.location;
            access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            if (cuMemSetAccess(va_ptr + committed_bytes_, new_chunk_size, &access_desc, 1) != CUDA_SUCCESS) {
                cuMemUnmap(va_ptr + committed_bytes_, new_chunk_size);
                cuMemRelease(handle);
                return false;
            }

            chunks_.push_back({handle, new_chunk_size});
            committed_bytes_ += new_chunk_size;
        }

        return true;
    }

    void clear_active_allocations(std::unordered_map<void*, size_t>& active_allocations) {
        if (!enabled_ || base_ptr_ == nullptr) {
            return;
        }

        for (auto it = active_allocations.begin(); it != active_allocations.end();) {
            if (owns(it->first)) {
                it = active_allocations.erase(it);
            } else {
                ++it;
            }
        }
    }

    void disable_no_tracking() {
        if (!enabled_ && base_ptr_ == nullptr) {
            return;
        }

        if (base_ptr_ != nullptr) {
            CUdeviceptr va_ptr = reinterpret_cast<CUdeviceptr>(base_ptr_);
            size_t offset = 0;
            for (const auto& chunk : chunks_) {
                cuMemUnmap(va_ptr + offset, chunk.size);
                cuMemRelease(chunk.handle);
                offset += chunk.size;
            }

            if (reserved_bytes_ > 0) {
                cuMemAddressFree(va_ptr, reserved_bytes_);
            }
        }

        chunks_.clear();
        enabled_ = false;
        phase_ = false;
        base_ptr_ = nullptr;
        reserved_bytes_ = 0;
        used_bytes_ = 0;
        committed_bytes_ = 0;
        peak_used_bytes_ = 0;
    }
};
