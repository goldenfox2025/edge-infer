#include "tensor_view.hpp"
#include "allocation_probe.hpp"

#include <array>
#include <cstddef>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>
#include <type_traits>

static_assert(std::is_aggregate_v<TensorView<float, 3>>);
static_assert(std::is_standard_layout_v<TensorView<float, 3>>);
static_assert(std::is_trivially_copyable_v<TensorView<const float, 3>>);
static_assert(!std::is_polymorphic_v<TensorView<float, 3>>);
static_assert(sizeof(TensorView<float, 3>) == sizeof(float*) + 6 * sizeof(std::size_t));
static_assert(std::is_same_v<decltype(std::declval<TensorView<const float, 2>>()(0, 0)),
                             const float&>);
constexpr int kCompileValues[]{1, 2, 3, 4, 5, 6};
constexpr TensorView<const int, 2> kCompileView{kCompileValues, {2, 3}, {3, 1}};
static_assert(kCompileView.numel() == 6 && kCompileView(1, 2) == 6);
static_assert(kCompileView.transpose<0, 1>()(2, 1) == 6);

namespace {

void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Operation>
void expect_rejected(Operation operation, const char* message) {
    bool rejected = false;
    try { operation(); } catch (const std::exception&) { rejected = true; }
    expect(rejected, message);
}

void test_borrowed_offsets_constness_and_layouts() {
    std::array<float, 24> data{};
    for (std::size_t index = 0; index < data.size(); ++index) data[index] = static_cast<float>(index);
    auto view = TensorView<float, 3>::contiguous(data.data(), {2, 3, 4});
    expect(view.strides() == std::array<std::size_t, 3>{12, 4, 1} &&
               view.is_contiguous() && view.numel() == 24,
           "contiguous view metadata differs");
    auto slab = view.select<0>(1);
    auto column = slab.select<1>(2);
    expect(slab.data_ptr() == data.data() + 12 && slab(1, 2) == 18 &&
               column.data_ptr() == data.data() + 14 && column(1) == 18 &&
               column.strides() == std::array<std::size_t, 1>{4} &&
               !column.is_contiguous(),
           "selection must preserve offsets and noncontiguous strides");
    const auto transpose = slab.transpose<0, 1>();
    expect(transpose.sizes() == std::array<std::size_t, 2>{4, 3} &&
               transpose.strides() == std::array<std::size_t, 2>{1, 4} &&
               transpose(2, 1) == 18 && !transpose.is_contiguous(),
           "transpose must retain its borrowed physical layout");
    auto region = view.checked_subview({1, 1, 1}, {1, 2, 3});
    expect(region.data_ptr() == data.data() + 17 && region(0, 1, 2) == 23,
           "subview offsets differ");
    region(0, 1, 2) = 100.0f;
    expect(data[23] == 100.0f && region.as_const()(0, 1, 2) == 100.0f,
           "mutable and const views must borrow the same caller storage");
    const auto flat = view.reshape_contiguous<1>({24});
    expect(flat.data_ptr() == data.data() && flat(23) == 100.0f &&
               flat.is_contiguous(), "contiguous reshape must retain backing storage");
    const auto strided = TensorView<const float, 2>::strided(data.data(), {2, 3}, {12, 4});
    expect(strided(1, 2) == data[20] && !strided.is_contiguous(),
           "checked strided construction must preserve caller layout");
    const auto singleton = TensorView<float, 3>::strided(data.data(), {1, 2, 3}, {999, 3, 1});
    expect(singleton.is_contiguous(), "singleton strides must not defeat contiguity");
    expect_rejected([&] { view.checked_subview({2, 0, 0}, {1, 1, 1}); },
                    "checked subviews must reject origins/extents beyond their source");
    const auto empty_corner = view.checked_subview({2, 3, 4}, {0, 0, 0});
    const auto empty_edge = view.checked_subview({1, 3, 4}, {1, 0, 0});
    expect(empty_corner.numel() == 0 && empty_edge.numel() == 0 &&
               empty_corner.data_ptr() == data.data() && empty_edge.data_ptr() == data.data(),
           "empty boundary slices must not form pointers beyond caller storage");
}

void test_empty_null_and_extent_validation() {
    const auto empty = TensorView<float, 3>::contiguous(nullptr, {2, 0, 3});
    expect(empty.numel() == 0 && empty.is_contiguous() && empty.data_ptr() == nullptr,
           "empty views may borrow null storage");
    const auto null_region = empty.checked_subview({2, 0, 3}, {0, 0, 0});
    expect(null_region.numel() == 0 && null_region.data_ptr() == nullptr,
           "empty null slices must preserve null without pointer arithmetic");
    expect_rejected([&] { empty.checked_subview({3, 0, 0}, {0, 0, 0}); },
                    "empty slices must still reject out-of-bounds origins");
    float value = 0;
    expect_rejected([] { TensorView<float, 1>::contiguous(nullptr, {1}); },
                    "nonempty views require caller storage");
    expect_rejected([&] {
        TensorView<float, 2>::contiguous(&value,
            {std::numeric_limits<std::size_t>::max(), 2});
    }, "contiguous shape products must reject overflow");
    expect_rejected([&] {
        TensorView<float, 2>::strided(&value, {3, 2},
            {std::numeric_limits<std::size_t>::max(), 1});
    }, "strided address extents must reject overflow");
    expect_rejected([&] {
        TensorView<float, 1>::contiguous(&value,
            {std::numeric_limits<std::size_t>::max() / sizeof(float) + 1});
    }, "view byte extents must reject overflow");
}

void test_view_operations_allocate_nothing() {
    {
        test_alloc::Scope probe;
        void* pointer = ::operator new(64);
        ::operator delete(pointer);
        const auto count = probe.finish();
        expect(count.host_allocations == 1 && count.host_frees == 1,
               "allocation probe must detect explicit C++ allocation/free");
    }
    std::array<float, 24> data{};
    const auto view = TensorView<float, 3>::contiguous(data.data(), {2, 3, 4});
    volatile float checksum = 0;
    test_alloc::Scope probe;
    for (int iteration = 0; iteration < 1000; ++iteration) {
        auto slab = view.select<0>(1);
        auto region = view.subview({1, 1, 1}, {1, 2, 3});
        auto transpose = slab.transpose<0, 1>();
        auto flat = view.reshape_contiguous<1>({24});
        checksum += region.as_const()(0, 0, 0) + transpose(1, 1) + flat(0);
    }
    const auto count = probe.finish();
    expect(checksum == 0 && count.host_allocations == 0 && count.host_frees == 0,
           "fixed-rank view operations must not allocate or own heap storage");
}

}  // namespace

int main() {
    test_borrowed_offsets_constness_and_layouts();
    test_empty_null_and_extent_validation();
    test_view_operations_allocate_nothing();
    std::cout << "tensor_view_test passed (portable, no CUDA dependency)\n";
}
