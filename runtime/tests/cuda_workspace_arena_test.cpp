#include "execution/cuda_workspace_arena.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {

void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

template <typename Operation>
void expect_rejected(Operation operation, const char* message) {
    bool rejected = false;
    try { operation(); } catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected, message);
}

void arena_test(int devices) {
    CudaWorkspaceArena arena;
    expect(!arena.reserve(0) && arena.empty() && !arena.capacity_bytes(),
           "Empty reserve must leave the arena unallocated");
    WorkspacePlanner planner;
    planner.add_request("activation", 16 * sizeof(float), 0, 1, 64);
    const auto plan = planner.build();
    expect(arena.reserve_for_plan(plan), "First plan must allocate caller-owned arena storage");
    auto view = arena.bind_view<float, 2>(plan.at("activation"), {4, 4});
    expect(reinterpret_cast<std::uintptr_t>(view.data) % 64 == 0 &&
               view.is_contiguous() && view.numel() == 16,
           "Bound view must satisfy physical pointer alignment and dimensions");
    const auto* original = arena.base_ptr();
    expect(!arena.reserve(16) && arena.base_ptr() == original,
           "A smaller reserve on the owning device must retain storage");
    std::array<float, 16> sentinel{};
    for (std::size_t index = 0; index < sentinel.size(); ++index) sentinel[index] = static_cast<float>(index) + 0.25f;
    check(cudaMemcpy(view.data, sentinel.data(), sizeof(sentinel), cudaMemcpyHostToDevice));

    auto misaligned = plan.at("activation");
    misaligned.offset = sizeof(float);
    misaligned.bytes -= sizeof(float);
    expect_rejected([&] { arena.bind_view<float, 1>(misaligned, {15}); },
                    "Binding must reject an element-aligned pointer that violates requested alignment");
    auto invalid_alignment = plan.at("activation");
    invalid_alignment.alignment = 3;
    expect_rejected([&] { arena.bind_view<float, 1>(invalid_alignment, {16}); },
                    "Binding must reject non-power-of-two requested alignment");
    invalid_alignment.alignment = 0;
    expect_rejected([&] { arena.bind_view<float, 1>(invalid_alignment, {16}); },
                    "Binding must reject zero requested alignment");
    auto unaligned_element = plan.at("activation");
    unaligned_element.alignment = 1;
    unaligned_element.offset = 1;
    unaligned_element.bytes = sizeof(float);
    expect_rejected([&] { arena.bind_view<float, 1>(unaligned_element, {1}); },
                    "Binding must satisfy the element's natural alignment even when the plan requests one byte");
    auto empty_allocation = plan.at("activation");
    empty_allocation.bytes = 0;
    expect_rejected([&] { arena.bind_view<float, 1>(empty_allocation, {0}); },
                    "Binding must reject a zero-byte allocation");
    expect_rejected([&] { arena.bind_view<float, 1>(plan.at("activation"), {17}); },
                    "Binding must reject a view larger than its planned allocation");
    WorkspacePlanner oversized_alignment;
    oversized_alignment.add_request("overaligned", 16, 0, 1, 512);
    const auto unsupported = oversized_alignment.build();
    expect_rejected([&] { arena.reserve_for_plan(unsupported); },
                    "CUDA arena must reject plans beyond its guaranteed base alignment");
    expect_rejected([&] { arena.bind_view<float, 1>(unsupported.at("overaligned"), {4}); },
                    "Binding must consistently reject unsupported alignment");
    std::array<float, 16> actual{};
    check(cudaMemcpy(actual.data(), view.data, sizeof(actual), cudaMemcpyDeviceToHost));
    expect(actual == sentinel && arena.base_ptr() == original && arena.capacity_bytes() == plan.total_bytes(),
           "Rejected preparation must preserve caller storage and previous bindings");

    if (devices > 1) {
        int owner = 0;
        check(cudaGetDevice(&owner));
        const int other = (owner + 1) % devices;
        check(cudaSetDevice(other));
        bool reuse_rejected = false;
        try { arena.reserve(16); } catch (const std::invalid_argument&) { reuse_rejected = true; }
        // Release on a different current device must free on the owner and
        // restore the caller's device selection.
        arena.release();
        int after = -1;
        check(cudaGetDevice(&after));
        check(cudaSetDevice(owner));
        expect(reuse_rejected && after == other && arena.empty(),
               "Cross-device reuse must reject and destruction must preserve caller device");
    } else {
        arena.release();
        std::cout << "CUDA workspace multi-device checks skipped: one CUDA device\n";
    }
    expect(arena.empty() && !arena.capacity_bytes(), "Release must clear arena ownership and capacity");
}

}  // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) {
        std::cout << "cuda_workspace_arena_test skipped: no CUDA device\n";
        return 77;
    }
    try {
        arena_test(devices);
        std::cout << "cuda_workspace_arena_test passed\n";
    } catch (const std::exception& error) {
        std::cerr << "cuda_workspace_arena_test failed: " << error.what() << '\n';
        return 1;
    }
}
