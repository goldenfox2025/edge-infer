#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

struct WorkspaceRequest {
    std::string name;
    size_t bytes = 0;
    size_t first_use = 0;
    size_t last_use = 0;
    size_t alignment = 256;
};

struct WorkspaceAllocation {
    std::string name;
    size_t offset = 0;
    size_t bytes = 0;
    size_t first_use = 0;
    size_t last_use = 0;
    size_t alignment = 256;
};

class WorkspacePlan {
   public:
    WorkspacePlan() = default;

    WorkspacePlan(std::vector<WorkspaceAllocation> allocations, size_t total_bytes)
        : allocations_(std::move(allocations)), total_bytes_(total_bytes) {
        for (size_t i = 0; i < allocations_.size(); ++i) {
            index_[allocations_[i].name] = i;
        }
    }

    bool empty() const {
        return allocations_.empty();
    }

    size_t total_bytes() const {
        return total_bytes_;
    }

    const std::vector<WorkspaceAllocation>& allocations() const {
        return allocations_;
    }

    bool has_allocation(const std::string& name) const {
        return index_.find(name) != index_.end();
    }

    const WorkspaceAllocation& at(const std::string& name) const {
        auto it = index_.find(name);
        if (it == index_.end()) {
            throw std::runtime_error("Workspace allocation not found: " + name);
        }
        return allocations_[it->second];
    }

   private:
    std::vector<WorkspaceAllocation> allocations_;
    std::unordered_map<std::string, size_t> index_;
    size_t total_bytes_ = 0;
};

// Offline planner: values with disjoint lifetimes share storage slots.
// This constructs a plan; it does not allocate or free runtime memory.
class WorkspacePlanner {
   public:
    void clear() {
        requests_.clear();
    }

    void add_request(const std::string& name, size_t bytes, size_t first_use, size_t last_use, size_t alignment = 256) {
        if (name.empty()) {
            throw std::runtime_error("Workspace request name must not be empty");
        }
        if (bytes == 0) {
            throw std::runtime_error("Workspace request bytes must be > 0 for " + name);
        }
        if (first_use > last_use) {
            throw std::runtime_error("Workspace request has invalid lifetime for " + name);
        }
        if (alignment == 0) {
            throw std::runtime_error("Workspace alignment must be > 0 for " + name);
        }
        if (!is_power_of_two(alignment)) {
            throw std::runtime_error("Workspace alignment must be a power of two for " + name);
        }

        requests_.push_back({name, bytes, first_use, last_use, alignment});
    }

    WorkspacePlan build() const {
        std::vector<WorkspaceRequest> ordered = requests_;
        std::sort(ordered.begin(), ordered.end(), [](const auto& lhs, const auto& rhs) {
            if (lhs.first_use != rhs.first_use) {
                return lhs.first_use < rhs.first_use;
            }
            if (lhs.last_use != rhs.last_use) {
                return lhs.last_use < rhs.last_use;
            }
            return lhs.name < rhs.name;
        });

        std::vector<WorkspaceAllocation> allocations;
        std::vector<Slot> slots;
        size_t total_bytes = 0;

        for (const auto& request : ordered) {
            // Reuse a sufficiently large slot whose last value is already dead.
            size_t slot_index = find_reusable_slot(slots, request);
            if (slot_index == kInvalidOffset) {
                // Append a slot only when existing storage cannot be reused.
                const size_t offset = align_up(total_bytes, request.alignment);
                slots.push_back({offset, request.bytes, request.last_use});
                total_bytes = offset + request.bytes;
                slot_index = slots.size() - 1;
            }

            auto& slot = slots[slot_index];
            allocations.push_back(
                {request.name, slot.offset, request.bytes, request.first_use, request.last_use, request.alignment});
            slot.last_use = request.last_use;
        }

        return WorkspacePlan(std::move(allocations), total_bytes);
    }

   private:
    static constexpr size_t kInvalidOffset = static_cast<size_t>(-1);

    struct Slot {
        size_t offset = 0;
        size_t capacity = 0;
        size_t last_use = 0;
    };

    static bool is_power_of_two(size_t value) {
        return value != 0 && (value & (value - 1)) == 0;
    }

    static size_t align_up(size_t value, size_t alignment) {
        return (value + alignment - 1) & ~(alignment - 1);
    }

    static size_t find_reusable_slot(const std::vector<Slot>& slots, const WorkspaceRequest& request) {
        size_t best_index = kInvalidOffset;
        size_t best_size = 0;

        for (size_t i = 0; i < slots.size(); ++i) {
            const auto& slot = slots[i];
            // A last use at the new first use still overlaps; do not reuse it.
            if (slot.last_use >= request.first_use) {
                continue;
            }
            if (slot.capacity < request.bytes) {
                continue;
            }
            if (slot.offset % request.alignment != 0) {
                continue;
            }
            if (best_index == kInvalidOffset || slot.capacity < best_size ||
                (slot.capacity == best_size && slot.offset < slots[best_index].offset)) {
                best_index = i;
                best_size = slot.capacity;
            }
        }
        return best_index;
    }

    std::vector<WorkspaceRequest> requests_;
};
