#include "tensor.hpp"
#include "allocation_probe.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

template <typename Error, typename Operation>
void rejected(Operation operation, const char* message) {
    bool caught = false;
    try { operation(); } catch (const Error&) { caught = true; }
    require(caught, message);
}

std::vector<int> sequence() {
    return {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
}

void equal_cpu(const Tensor<int>& tensor, const std::vector<int>& expected,
               const char* message) {
    require(tensor.device() == Device::CPU && tensor.is_contiguous() &&
                tensor.numel() == expected.size(), message);
    require(expected.empty() || std::equal(expected.begin(), expected.end(), tensor.data_ptr()),
            message);
}

void cpu_storage_test() {
    Tensor<int> uninitialized;
    auto empty_default = uninitialized.slice({}, {});
    require(!empty_default.numel() && !empty_default.data_ptr() && empty_default.sizes().empty(),
            "Slicing default empty storage must not create a nonempty null scalar");
    Tensor<int> base(sequence(), {3, 4});
    auto sliced = base.slice({1, 1}, {3, 4});
    require(sliced.sizes() == std::vector<std::size_t>({2, 3}) &&
                sliced.offset() == 5 && !sliced.is_contiguous(),
            "An interior slice must preserve its physical row stride");
    rejected<std::invalid_argument>([&] { sliced.view({6}); },
                                    "A strided slice must reject contiguous reshape");
    require(sliced.sizes() == std::vector<std::size_t>({2, 3}) && sliced.offset() == 5,
            "Rejected reshape must preserve the view");
    sliced.view({2, 3});
    auto transposed = base.transpose(-1, -2);
    require(transposed.sizes() == std::vector<std::size_t>({4, 3}) &&
                transposed.strides() == std::vector<std::size_t>({1, 4}),
            "Negative transpose axes must resolve within the rank");
    rejected<std::out_of_range>([&] { base.transpose(-3, 0); },
                               "Transpose must reject an excessively negative first axis");
    rejected<std::out_of_range>([&] { base.transpose(0, -3); },
                               "Transpose must reject an excessively negative second axis");
    rejected<std::out_of_range>([&] { base.transpose(2, 0); },
                               "Transpose must reject an axis at the rank boundary");
    rejected<std::invalid_argument>([&] { transposed.view({12}); },
                                    "Transposed storage must reject contiguous reshape");

    auto empty = base.slice({3, 4}, {3, 4});
    require(!empty.numel() && empty.data_ptr() == base.data_ptr() && !empty.offset(),
            "An empty corner slice must preserve the source pointer without arithmetic");
    auto empty_offset = sliced.slice({2, 3}, {2, 3});
    require(!empty_offset.numel() && empty_offset.data_ptr() == sliced.data_ptr() &&
                empty_offset.offset() == sliced.offset(),
            "An empty slice of an offset view must preserve its existing pointer");
    rejected<std::out_of_range>([&] { base.slice({4, 4}, {4, 4}); },
                               "Empty slices must still reject out-of-bounds origins");
    Tensor<int> zero({2, 0, 3});
    auto zero_view = zero.slice({2, 0, 3}, {2, 0, 3});
    require(!zero.numel() && !zero_view.numel() && zero_view.data_ptr() == zero.data_ptr(),
            "Zero-extent host storage must admit safe empty views");
    rejected<std::overflow_error>([] {
        Tensor<int> oversized({std::numeric_limits<std::size_t>::max(), 2});
    }, "Element or stride overflow must reject before storage allocation");
    rejected<std::overflow_error>([] {
        Tensor<int> oversized({std::numeric_limits<std::size_t>::max() / sizeof(int) + 1});
    }, "Byte-size overflow must reject before storage allocation");

    // This runs before CUDA availability is queried. It must reject the input
    // shape before attempting any allocation, including on hosts without a GPU.
    test_alloc::Scope allocation_probe;
    rejected<std::invalid_argument>([] {
        Tensor<int> short_input(std::vector<int>{1, 2}, {3}, Device::CUDA);
    }, "Short CUDA vector input must reject before migration or device allocation");
    const auto invalid_counts = allocation_probe.finish();
#ifdef EDGE_INFER_TEST_CUDA_WRAPPING
    require(!invalid_counts.device_allocations && !invalid_counts.device_frees,
            "Short vector validation must not call a native CUDA allocator");
#else
    (void)invalid_counts;
#endif

    Tensor<int> retained;
    {
        Tensor<int> owner(sequence(), {3, 4});
        retained = owner.slice({1, 0}, {3, 4});
    }
    equal_cpu(retained, {4, 5, 6, 7, 8, 9, 10, 11},
              "A host view must retain shared storage after its parent is destroyed");
    auto shared = retained;
    shared.data_ptr()[0] = 99;
    require(retained.data_ptr()[0] == 99, "Tensor copies must share ownership and data");
}

void migration_test() {
    const std::vector<int> sliced_values{5, 6, 7, 9, 10, 11};
    const std::vector<int> transposed_values{5, 9, 6, 10, 7, 11};
    Tensor<int> host(sequence(), {3, 4});
    auto sliced = host.slice({1, 1}, {3, 4});
    sliced.cuda();
    require(sliced.device() == Device::CUDA && !sliced.offset() && sliced.is_contiguous(),
            "Host-to-device migration must materialize a strided slice and reset offset");
    sliced.cpu();
    equal_cpu(sliced, sliced_values, "Sliced host-to-device-to-host values must follow logical order");
    equal_cpu(host, sequence(), "Migrating a view must preserve the shared original host storage");

    auto host_transpose = host.slice({1, 1}, {3, 4}).transpose(0, 1);
    host_transpose.cuda().cpu();
    equal_cpu(host_transpose, transposed_values,
              "Transposed host-to-device migration must use logical row-major values");
    require(host_transpose.sizes() == std::vector<std::size_t>({3, 2}) && !host_transpose.offset(),
            "Migration must preserve logical dimensions and remove the source offset");

    Tensor<int> device(sequence(), {3, 4}, Device::CUDA);
    auto device_slice = device.slice({1, 1}, {3, 4});
    device_slice.cpu();
    equal_cpu(device_slice, sliced_values,
              "Device-to-host migration must materialize a strided interior slice");
    auto device_transpose = device.slice({1, 1}, {3, 4}).transpose(0, 1);
    device_transpose.cpu();
    equal_cpu(device_transpose, transposed_values,
              "Device-to-host migration must materialize a transposed slice");
    device.cpu();
    equal_cpu(device, sequence(), "Migrating GPU views must preserve the shared original device storage");

    auto contiguous_tail = host.slice({1, 0}, {3, 4});
    contiguous_tail.cuda().cpu();
    equal_cpu(contiguous_tail, {4, 5, 6, 7, 8, 9, 10, 11},
              "Contiguous offset views must migrate from their logical start");
    require(!contiguous_tail.offset(), "Contiguous migration must also reset the offset");

    test_alloc::Scope allocation_probe;
    Tensor<int> empty_device({0, 4}, Device::CUDA);
    Tensor<int> empty_host({2, 0, 3});
    empty_host.cuda().cpu();
    const auto empty_counts = allocation_probe.finish();
    require(!empty_device.numel() && !empty_device.data_ptr() && !empty_host.numel(),
            "Empty CUDA tensors and migrations must retain zero logical extent");
#ifdef EDGE_INFER_TEST_CUDA_WRAPPING
    require(!empty_counts.device_allocations && !empty_counts.device_frees,
            "Zero-extent storage must not allocate or free native CUDA memory");
#else
    (void)empty_counts;
#endif
}

void cuda_ownership_test() {
    int* adopted_raw = nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&adopted_raw), 4 * sizeof(int)));
    const std::vector<int> expected{41, 42, 43, 44};
    check(cudaMemcpy(adopted_raw, expected.data(), 4 * sizeof(int), cudaMemcpyHostToDevice));
    Tensor<int> owner(adopted_raw, {2, 2}, Device::CUDA);
    auto retained = owner.slice({1, 0}, {2, 2});
    test_alloc::Scope parent_release;
    owner = Tensor<int>();
    const auto parent_counts = parent_release.finish();
#ifdef EDGE_INFER_TEST_CUDA_WRAPPING
    require(!parent_counts.device_frees,
            "Destroying the parent must not free adopted storage while a shared slice lives");
#else
    (void)parent_counts;
#endif
    int actual[2]{};
    check(cudaMemcpy(actual, retained.data_ptr(), sizeof(actual), cudaMemcpyDeviceToHost));
    require(actual[0] == 43 && actual[1] == 44,
            "An adopted CUDA view must retain its allocation after parent destruction");
    test_alloc::Scope final_release;
    retained = Tensor<int>();
    const auto adopted_counts = final_release.finish();
#ifdef EDGE_INFER_TEST_CUDA_WRAPPING
    require(adopted_counts.device_frees == 1,
            "The final adopted storage owner must free its CUDA allocation exactly once");
#else
    (void)adopted_counts;
#endif

    int* borrowed_raw = nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&borrowed_raw), 4 * sizeof(int)));
    check(cudaMemcpy(borrowed_raw, expected.data(), 4 * sizeof(int), cudaMemcpyHostToDevice));
    test_alloc::Scope borrowed_release;
    {
        auto borrowed = Tensor<int>::from_external_buffer(borrowed_raw, {2, 2}, Device::CUDA);
        auto alias = borrowed.slice({1, 0}, {2, 2});
        require(alias.data_ptr() == borrowed_raw + 2,
                "Borrowed CUDA views must retain the external allocation address");
    }
    const auto borrowed_counts = borrowed_release.finish();
#ifdef EDGE_INFER_TEST_CUDA_WRAPPING
    require(!borrowed_counts.device_frees,
            "Destroying borrowed CUDA storage must never call cudaFree");
#else
    (void)borrowed_counts;
#endif
    std::vector<int> after_borrow(4);
    check(cudaMemcpy(after_borrow.data(), borrowed_raw, 4 * sizeof(int), cudaMemcpyDeviceToHost));
    require(after_borrow == expected, "External storage must remain usable after borrowed views are destroyed");
    check(cudaFree(borrowed_raw));
}

void owner_device_test(int devices) {
    if (devices < 2) {
        std::cout << "Tensor owner-device release checks skipped: one CUDA device\n";
        return;
    }
    int original_device = 0;
    check(cudaGetDevice(&original_device));
    Tensor<int> storage({4}, Device::CUDA);
    const int other_device = (original_device + 1) % devices;
    check(cudaSetDevice(other_device));
    test_alloc::Scope release;
    storage = Tensor<int>();
    const auto counts = release.finish();
    int after = -1;
    check(cudaGetDevice(&after));
    check(cudaSetDevice(original_device));
    require(after == other_device,
            "Owned CUDA storage release must restore the caller's selected device");
#ifdef EDGE_INFER_TEST_CUDA_WRAPPING
    require(counts.device_frees == 1, "Cross-device release must free owned storage exactly once");
#else
    (void)counts;
#endif
}

}  // namespace

int main() {
    try {
        cpu_storage_test();
        std::cout << "tensor_storage_test CPU checks passed\n";
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) {
            std::cout << "tensor_storage_test CUDA checks skipped: no CUDA device\n";
            return 77;
        }
        migration_test();
        cuda_ownership_test();
        owner_device_test(devices);
#ifndef EDGE_INFER_TEST_CUDA_WRAPPING
        std::cout << "Tensor native CUDA allocation counter checks skipped: linker wrapping unavailable\n";
#endif
        std::cout << "tensor_storage_test passed\n";
    } catch (const std::exception& error) {
        std::cerr << "tensor_storage_test failed: " << error.what() << '\n';
        return 1;
    }
}
