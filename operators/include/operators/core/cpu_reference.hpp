#pragma once

#include <cmath>
#include <cstddef>
#include <type_traits>

#include "operators/core/array_view.hpp"

namespace op::cpu {

// Direct, statically dispatched reference operations. There is no factory,
// virtual dispatch, allocation, or owning Tensor in this API. Views must have
// matching extents. Each input may alias output exactly, but partial overlap is
// unsupported. The RMSNorm weights have the stricter precondition below.
// These CPU references make no claim about CUDA launch costs or throughput.
// T must be a native C++ arithmetic type. Convert vendor device types (such as
// CUDA BF16) to float before using these portable host references.
template <typename T>
inline void add(ArrayView<const T> a, ArrayView<const T> b, ArrayView<T> output) {
    static_assert(std::is_arithmetic_v<T>, "CPU references require native C++ arithmetic types");
    for (std::size_t i = 0; i < a.size; ++i) {
        output[i] = a[i] + b[i];
    }
}

template <typename T>
inline void multiply(ArrayView<const T> a, ArrayView<const T> b, ArrayView<T> output) {
    static_assert(std::is_arithmetic_v<T>, "CPU references require native C++ arithmetic types");
    for (std::size_t i = 0; i < a.size; ++i) {
        output[i] = a[i] * b[i];
    }
}

template <typename T>
inline void silu(ArrayView<const T> input, ArrayView<T> output) {
    static_assert(std::is_arithmetic_v<T>, "CPU references require native C++ arithmetic types");
    for (std::size_t i = 0; i < input.size; ++i) {
        const float value = static_cast<float>(input[i]);
        output[i] = static_cast<T>(value / (1.0f + std::exp(-value)));
    }
}

// Rows are contiguous, feature_dim is positive, weights contain feature_dim
// elements and do not overlap output, and input/output contain rows * feature_dim
// elements. Accumulation stays float to preserve the existing CPU adapter's
// arithmetic. In-place input/output is safe.
template <typename T>
inline void rms_norm(ArrayView<const T> input, ArrayView<const T> weights,
                     ArrayView<T> output, std::size_t rows,
                     std::size_t feature_dim, float eps) {
    static_assert(std::is_arithmetic_v<T>, "CPU references require native C++ arithmetic types");
    for (std::size_t row = 0; row < rows; ++row) {
        float sum_squares = 0.0f;
        for (std::size_t j = 0; j < feature_dim; ++j) {
            const float value = static_cast<float>(input[row * feature_dim + j]);
            sum_squares += value * value;
        }
        const float rms = std::sqrt(sum_squares / feature_dim + eps);
        for (std::size_t j = 0; j < feature_dim; ++j) {
            const float normalized = static_cast<float>(input[row * feature_dim + j]) / rms;
            output[row * feature_dim + j] = static_cast<T>(normalized * static_cast<float>(weights[j]));
        }
    }
}

}  // namespace op::cpu
