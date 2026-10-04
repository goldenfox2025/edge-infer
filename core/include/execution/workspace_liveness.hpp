#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "execution/workspace_plan.hpp"

struct WorkspaceStep {
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
};

// Infer lifetimes from the values read and written by each execution step.
class WorkspaceLivenessBuilder {
   public:
    void clear() {
        values_.clear();
        steps_.clear();
    }

    void add_value(const std::string& name, std::vector<size_t> shape, size_t alignment = 256) {
        if (name.empty()) {
            throw std::runtime_error("Workspace value name must not be empty");
        }
        if (shape.empty()) {
            throw std::runtime_error("Workspace value shape must not be empty for " + name);
        }
        if (values_.find(name) != values_.end()) {
            throw std::runtime_error("Duplicate workspace value: " + name);
        }
        for (size_t extent : shape) {
            if (!extent) throw std::invalid_argument("Workspace value has an empty extent: " + name);
        }
        if (!alignment || (alignment & (alignment - 1))) {
            throw std::invalid_argument("Workspace value alignment must be a power of two: " + name);
        }
        values_[name] = {std::move(shape), alignment};
    }

    void add_step(std::vector<std::string> inputs, std::vector<std::string> outputs) {
        steps_.push_back({std::move(inputs), std::move(outputs)});
    }

    template <typename T>
    WorkspacePlan build_plan() const {
        WorkspacePlanner planner;
        std::unordered_map<std::string, Lifetime> lifetimes;

        // Record the first and last execution step that uses each value.
        for (size_t step_index = 0; step_index < steps_.size(); ++step_index) {
            const auto& step = steps_[step_index];

            auto touch = [&](const std::string& name) {
                const auto value_it = values_.find(name);
                if (value_it == values_.end()) {
                    throw std::runtime_error("Workspace step references unknown value: " + name);
                }

                auto& lifetime = lifetimes[name];
                if (!lifetime.seen) {
                    lifetime.first_use = step_index;
                    lifetime.seen = true;
                }
                lifetime.last_use = step_index;
            };

            for (const auto& name : step.inputs) {
                touch(name);
            }
            for (const auto& name : step.outputs) {
                touch(name);
            }
        }

        for (const auto& [name, value] : values_) {
            const auto lifetime_it = lifetimes.find(name);
            if (lifetime_it == lifetimes.end() || !lifetime_it->second.seen) {
                throw std::runtime_error("Workspace value was declared but never used: " + name);
            }
            // Let the storage planner assign slots to the inferred lifetimes.
            planner.add_request(name, bytes_for<T>(value.shape), lifetime_it->second.first_use,
                                lifetime_it->second.last_use, value.alignment);
        }

        return planner.build();
    }

   private:
    struct ValueInfo {
        std::vector<size_t> shape;
        size_t alignment = 256;
    };

    struct Lifetime {
        size_t first_use = 0;
        size_t last_use = 0;
        bool seen = false;
    };

    template <typename T>
    static size_t bytes_for(const std::vector<size_t>& shape) {
        size_t bytes = sizeof(T);
        for (size_t dim : shape) {
            if (!dim || bytes > std::numeric_limits<size_t>::max() / dim) {
                throw std::overflow_error("Workspace value byte extent overflow");
            }
            bytes *= dim;
        }
        return bytes;
    }

    std::unordered_map<std::string, ValueInfo> values_;
    std::vector<WorkspaceStep> steps_;
};
