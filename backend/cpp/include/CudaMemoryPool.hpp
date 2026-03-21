#pragma once

#include <cuda.h>  // <--- 引入CUDA Driver API以使用VMM
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include "prefill_workspace_arena.hpp"

class CudaMemoryPool;

class CudaMemoryPool {
   public:
    static constexpr size_t kAllocationAlignment = 256;

    CudaMemoryPool()
        : is_shutting_down_(false) {
        // 调用cuInit(0)确保Driver API已初始化, 多次调用是安全的。
        // 调用cudaFree(0)可初始化CUDA Runtime和上下文。
        cuInit(0);
        cudaError_t err = cudaFree(0);
        if (err != cudaSuccess && err != cudaErrorInvalidDevicePointer) {
            // 初始化检查失败，可能表示CUDA环境存在问题。
        }
    }

    ~CudaMemoryPool() {
        std::lock_guard<std::mutex> lock(mutex_);
        is_shutting_down_ = true;

        // 优先清理VMM资源
        if (prefill_arena_.enabled()) {
            prefill_arena_.disable(active_allocations_);
        }

        bool driver_available = is_cuda_driver_available();

        if (driver_available) {
            // 释放所有缓存的内存块
            for (auto& [size, blocks] : free_blocks_) {
                for (void* ptr : blocks) {
                    safe_cuda_free(ptr, driver_available);
                }
            }

            // 释放所有标签化内存块
            for (auto const& [tag, block_info] : tagged_memory_) {
                safe_cuda_free(block_info.ptr, driver_available);
            }

            // VMM prefill_buffer_的释放已由disable_prefill_mode_internal处理
            // 无需再调用 safe_cuda_free(prefill_buffer_, driver_available);
        }

        // 无论驱动状态如何，都清空所有跟踪记录
        free_blocks_.clear();
        tagged_memory_.clear();
        memory_tags_.clear();
        active_allocations_.clear();
    }

    // 删除拷贝和移动操作，保证单例。
    CudaMemoryPool(const CudaMemoryPool&) = delete;
    CudaMemoryPool& operator=(const CudaMemoryPool&) = delete;
    CudaMemoryPool(CudaMemoryPool&&) = delete;
    CudaMemoryPool& operator=(CudaMemoryPool&&) = delete;

    // --- 公开接口 ---

    void* allocate(size_t size, bool is_prefill_request = false, const std::string& tag = "") {
        if (size == 0)
            return nullptr;
        if (!tag.empty()) {
            // 重定向到专用的标签分配函数。
            return allocate_tagged(tag, size, is_prefill_request);
        }

        std::lock_guard<std::mutex> lock(mutex_);
        if (is_shutting_down_)
            return nullptr;

        size_t aligned_size = align_size(size);
        void* ptr = nullptr;

        // Prefill阶段的分配优先使用prefill缓冲区。
        bool use_prefill = is_prefill_request || prefill_arena_.phase();
        if (use_prefill) {
            ptr = prefill_arena_.try_allocate(aligned_size, active_allocations_);
            if (ptr)
                return ptr;
        }

        // 尝试从缓存中寻找合适的内存块。
        ptr = try_allocate_from_cache_internal(aligned_size);
        if (ptr)
            return ptr;

        // 最后再分配新的内存块。
        return allocate_new_block_internal(aligned_size, "");
    }

    void free(void* ptr) {
        if (!ptr)
            return;

        std::lock_guard<std::mutex> lock(mutex_);
        if (is_shutting_down_) {
            // 关闭时，仅停止跟踪分配，由析构函数统一释放。
            active_allocations_.erase(ptr);
            return;
        }

        auto it_active = active_allocations_.find(ptr);
        if (it_active == active_allocations_.end()) {
            // 未找到指针，可能是重复释放或无效指针。
            return;
        }
        size_t aligned_size = it_active->second;

        // 检查指针是否属于标签内存块，若是则标记为非活动。
        auto it_tag = memory_tags_.find(ptr);
        if (it_tag != memory_tags_.end()) {
            tagged_memory_[it_tag->second].is_active = false;
            active_allocations_.erase(it_active);
            return;
        }

        // 检查指针是否来自prefill缓冲区。
        if (prefill_arena_.owns(ptr)) {
            prefill_arena_.free_allocation(ptr, aligned_size);
            active_allocations_.erase(it_active);
            return;
        }

        // 对常规分配，进行缓存或直接释放。
        if (should_cache_block_internal(aligned_size)) {
            free_blocks_[aligned_size].push_back(ptr);
        } else {
            bool driver_ok = is_cuda_driver_available();
            safe_cuda_free(ptr, driver_ok);
        }

        active_allocations_.erase(it_active);
        perform_periodic_cleanup_internal();
    }

    void* allocate_tagged(const std::string& tag, size_t size, bool is_prefill_request = false) {
        if (tag.empty())
            return allocate(size, is_prefill_request);
        if (size == 0)
            return nullptr;

        std::lock_guard<std::mutex> lock(mutex_);
        if (is_shutting_down_)
            return nullptr;

        size_t aligned_size = align_size(size);

        auto it = tagged_memory_.find(tag);
        if (it != tagged_memory_.end()) {
            TaggedBlockInfo& block_info = it->second;

            // 内存块已存在，检查是否可用。
            if (block_info.size >= aligned_size) {
                if (!block_info.is_active) {
                    block_info.is_active = true;
                    active_allocations_[block_info.ptr] = block_info.size;
                }
                return block_info.ptr;
            } else {
                // 现有内存块太小，释放它再分配新的。
                if (block_info.is_active) {
                    active_allocations_.erase(block_info.ptr);
                }
                bool driver_ok = is_cuda_driver_available();
                safe_cuda_free(block_info.ptr, driver_ok);
                memory_tags_.erase(block_info.ptr);
                tagged_memory_.erase(it);
            }
        }

        // 为标签分配一个新的内存块 (标签内存不使用prefill缓冲区)。
        return allocate_new_block_internal(aligned_size, tag);
    }

    void* get_tagged_memory(const std::string& tag) {
        if (tag.empty())
            return nullptr;
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = tagged_memory_.find(tag);
        return (it != tagged_memory_.end()) ? it->second.ptr : nullptr;
    }

    bool has_tag(const std::string& tag) {
        if (tag.empty())
            return false;
        std::lock_guard<std::mutex> lock(mutex_);
        return tagged_memory_.count(tag) > 0;
    }

    void trim(size_t size = 0) {
        std::lock_guard<std::mutex> lock(mutex_);
        trim_internal(size);
    }

    void trim_threshold(size_t max_blocks_per_size) {
        std::lock_guard<std::mutex> lock(mutex_);
        trim_threshold_internal(max_blocks_per_size);
    }

    bool enable_prefill_mode(size_t initial_size = 48 * 1024 * 1024, size_t max_size = 512 * 1024 * 1024) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (is_shutting_down_) {
            return false;
        }
        return prefill_arena_.enable(initial_size, max_size);
    }

    void disable_prefill_mode() {
        std::lock_guard<std::mutex> lock(mutex_);
        prefill_arena_.disable(active_allocations_);
    }

    void reset_prefill_buffer() {
        std::lock_guard<std::mutex> lock(mutex_);
        prefill_arena_.reset(active_allocations_);
    }

    void set_prefill_phase(bool is_prefill) {
        std::lock_guard<std::mutex> lock(mutex_);
        prefill_arena_.set_phase(is_prefill);
    }

    bool prepare_prefill_capacity(size_t requested_bytes) {
        std::lock_guard<std::mutex> lock(mutex_);
        return prefill_arena_.prepare(requested_bytes, active_allocations_);
    }

    void prepare_for_shutdown() {
        std::lock_guard<std::mutex> lock(mutex_);
        is_shutting_down_ = true;
        if (prefill_arena_.enabled()) {
            prefill_arena_.disable(active_allocations_);
        }
        trim_internal(0);  // 清理所有缓存
        std::cerr << "CudaMemoryPool: 已准备安全关闭，将限制后续CUDA操作。" << std::endl;
    }

    bool is_shutting_down() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return is_shutting_down_;
    }

    // --- 统计信息 ---
    struct PoolStats {
        size_t total_cached_blocks = 0;
        size_t total_cached_bytes = 0;
        size_t active_allocations_count = 0;
        size_t active_bytes = 0;
        size_t size_categories_in_cache = 0;
        size_t tagged_allocations_count = 0;
        size_t tagged_active_count = 0;
        size_t tagged_total_bytes = 0;
        size_t prefill_buffer_reserved_bytes = 0;
        size_t prefill_buffer_committed_bytes = 0;
        size_t prefill_buffer_used_bytes = 0;
    };

    PoolStats getStats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        PoolStats stats;

        stats.active_allocations_count = active_allocations_.size();
        stats.active_bytes = std::accumulate(active_allocations_.begin(), active_allocations_.end(), 0ULL,
                                             [](size_t sum, const auto& pair) { return sum + pair.second; });

        stats.size_categories_in_cache = free_blocks_.size();
        for (const auto& [size_val, blocks] : free_blocks_) {
            stats.total_cached_blocks += blocks.size();
            stats.total_cached_bytes += size_val * blocks.size();
        }

        stats.tagged_allocations_count = tagged_memory_.size();
        for (const auto& [tag, block_info] : tagged_memory_) {
            stats.tagged_total_bytes += block_info.size;
            if (block_info.is_active) {
                stats.tagged_active_count++;
            }
        }

        const auto prefill_stats = prefill_arena_.stats();
        stats.prefill_buffer_reserved_bytes = prefill_stats.reserved_bytes;
        stats.prefill_buffer_committed_bytes = prefill_stats.committed_bytes;
        stats.prefill_buffer_used_bytes = prefill_stats.used_bytes;
        return stats;
    }

   private:
    // --- 内部数据结构 ---

    struct TaggedBlockInfo {
        void* ptr = nullptr;
        size_t size = 0;
        bool is_active = false;
    };

    // --- 内部状态变量 ---

    mutable std::mutex mutex_;
    bool is_shutting_down_;

    // 常规缓存状态
    std::unordered_map<size_t, std::vector<void*>> free_blocks_;
    std::unordered_map<void*, size_t> active_allocations_;

    // 标签内存状态
    std::map<std::string, TaggedBlockInfo> tagged_memory_;
    std::map<void*, std::string> memory_tags_;

    PrefillWorkspaceArena prefill_arena_;

    // --- 内部辅助方法 ---

    static size_t align_size(size_t size) {
        return (size + kAllocationAlignment - 1) & ~(kAllocationAlignment - 1);
    }

    // ... [其它未改动的内部方法: try_allocate_from_cache_internal, allocate_new_block_internal, etc.]
    void* try_allocate_from_cache_internal(size_t aligned_size) {
        auto it = free_blocks_.find(aligned_size);
        if (it != free_blocks_.end() && !it->second.empty()) {
            void* ptr = it->second.back();
            it->second.pop_back();
            if (it->second.empty()) {
                free_blocks_.erase(it);
            }
            active_allocations_[ptr] = aligned_size;
            return ptr;
        }
        return nullptr;
    }

    void* allocate_new_block_internal(size_t aligned_size, const std::string& tag) {
        size_t free_memory = 0, total_memory = 0;
        if (cudaMemGetInfo(&free_memory, &total_memory) == cudaSuccess) {
            if (aligned_size > free_memory) {
                trim_threshold_internal(2);
                if (cudaMemGetInfo(&free_memory, &total_memory) == cudaSuccess && aligned_size > free_memory) {
                    trim_internal(0);
                }
            }
        }

        void* ptr = nullptr;
        if (cudaMalloc(&ptr, aligned_size) != cudaSuccess) {
            return nullptr;
        }

        active_allocations_[ptr] = aligned_size;

        if (!tag.empty()) {
            tagged_memory_[tag] = {ptr, aligned_size, true};
            memory_tags_[ptr] = tag;
        }
        return ptr;
    }

    bool should_cache_block_internal(size_t aligned_size) const {
        size_t max_blocks = 8;
        if (aligned_size >= 1024 * 1024)
            max_blocks = 2;  // >= 1MB
        else if (aligned_size >= 65536)
            max_blocks = 4;  // >= 64KB

        auto it = free_blocks_.find(aligned_size);
        if (it == free_blocks_.end())
            return true;
        return it->second.size() < max_blocks;
    }

    void perform_periodic_cleanup_internal() {
        size_t total_cached_blocks = 0;
        for (const auto& [size_key, blocks] : free_blocks_) {
            total_cached_blocks += blocks.size();
        }
        if (total_cached_blocks > 100) {
            trim_threshold_internal(4);
        }
    }

    void trim_internal(size_t size) {
        bool driver_ok = is_cuda_driver_available();
        if (size == 0) {
            for (auto& [block_size, blocks] : free_blocks_) {
                for (void* ptr : blocks) {
                    safe_cuda_free(ptr, driver_ok);
                }
            }
            free_blocks_.clear();
        } else {
            size_t aligned_size = align_size(size);
            auto it = free_blocks_.find(aligned_size);
            if (it != free_blocks_.end()) {
                for (void* ptr : it->second) {
                    safe_cuda_free(ptr, driver_ok);
                }
                free_blocks_.erase(it);
            }
        }
    }

    void trim_threshold_internal(size_t max_blocks_per_size) {
        bool driver_ok = is_cuda_driver_available();
        for (auto it = free_blocks_.begin(); it != free_blocks_.end();) {
            while (it->second.size() > max_blocks_per_size) {
                void* ptr = it->second.back();
                it->second.pop_back();
                safe_cuda_free(ptr, driver_ok);
            }
            if (it->second.empty()) {
                it = free_blocks_.erase(it);
            } else {
                ++it;
            }
        }
    }

    bool is_cuda_driver_available() {
        cudaError_t err = cudaFree(0);
        return err == cudaSuccess || err == cudaErrorInvalidDevicePointer;
    }

    void safe_cuda_free(void* ptr, bool& driver_available_flag) {
        if (!ptr || !driver_available_flag)
            return;

        cudaError_t err = cudaFree(ptr);
        if (err != cudaSuccess && err == cudaErrorCudartUnloading) {
            driver_available_flag = false;
        }
    }
};

// ==================================================================
// 全局单例包装器
// ==================================================================

class GlobalCudaMemoryPool {
   public:
    static CudaMemoryPool& instance() {
        std::lock_guard<std::mutex> lock(init_mutex_);
        if (!pool_instance_ptr) {
            pool_instance_ptr = new CudaMemoryPool();
        }
        return *pool_instance_ptr;
    }

    // --- 静态接口转发 ---

    static void* allocate(size_t size, bool is_prefill_request = false, const std::string& tag = "") {
        return instance().allocate(size, is_prefill_request, tag);
    }

    static void free(void* ptr) {
        instance().free(ptr);
    }

    static bool enable_prefill_mode(size_t initial_size = 48 * 1024 * 1024, size_t max_size = 512 * 1024 * 1024) {
        return instance().enable_prefill_mode(initial_size, max_size);
    }

    static void disable_prefill_mode() {
        instance().disable_prefill_mode();
    }

    static void reset_prefill_buffer() {
        instance().reset_prefill_buffer();
    }

    static void set_prefill_phase(bool is_prefill) {
        instance().set_prefill_phase(is_prefill);
    }

    static bool prepare_prefill_capacity(size_t requested_bytes) {
        return instance().prepare_prefill_capacity(requested_bytes);
    }

    static void* allocate_tagged(const std::string& tag, size_t size, bool is_prefill_request = false) {
        return instance().allocate_tagged(tag, size, is_prefill_request);
    }

    static void* get_tagged_memory(const std::string& tag) {
        return instance().get_tagged_memory(tag);
    }

    static bool has_tag(const std::string& tag) {
        return instance().has_tag(tag);
    }

    static void prepare_for_shutdown() {
        std::lock_guard<std::mutex> lock(init_mutex_);
        if (pool_instance_ptr != nullptr) {
            pool_instance_ptr->prepare_for_shutdown();
        }
    }

    static bool is_shutting_down() {
        std::lock_guard<std::mutex> lock(init_mutex_);
        if (pool_instance_ptr != nullptr) {
            return pool_instance_ptr->is_shutting_down();
        }
        return false;
    }

    static void explicitly_delete_pool_instance() {
        std::lock_guard<std::mutex> lock(init_mutex_);
        if (pool_instance_ptr) {
            delete pool_instance_ptr;
            pool_instance_ptr = nullptr;
        }
    }

   private:
    GlobalCudaMemoryPool() = default;
    ~GlobalCudaMemoryPool() = default;
    GlobalCudaMemoryPool(const GlobalCudaMemoryPool&) = delete;
    GlobalCudaMemoryPool& operator=(const GlobalCudaMemoryPool&) = delete;
    GlobalCudaMemoryPool(GlobalCudaMemoryPool&&) = delete;
    GlobalCudaMemoryPool& operator=(GlobalCudaMemoryPool&&) = delete;

    static CudaMemoryPool* pool_instance_ptr;
    static std::mutex init_mutex_;
};
