#pragma once

#include <cstddef>

namespace op {

// A borrowed contiguous range. It owns no storage and performs no allocation.
// The caller keeps the storage alive and supplies a valid pointer for size > 0.
template <typename T>
struct ArrayView {
    T* data = nullptr;
    std::size_t size = 0;

    constexpr T& operator[](std::size_t index) const noexcept {
        return data[index];
    }
};

}  // namespace op
