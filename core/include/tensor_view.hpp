#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>

#if defined(__CUDACC__)
#define EDGE_INFER_VIEW_HD __host__ __device__
#else
#define EDGE_INFER_VIEW_HD
#endif

// A borrowed descriptor: no device selection, allocation or storage ownership.
// Checked factories belong to preparation. Aggregate construction and the
// unchecked helpers below require already-validated extents and live storage.
template <typename T, std::size_t Rank>
struct TensorView {
    static_assert(Rank > 0, "TensorView requires a positive rank");
    using value_type = std::remove_const_t<T>;
    using dimensions_type = std::array<std::size_t, Rank>;

    T* data = nullptr;
    dimensions_type shape{};
    dimensions_type stride{};

    EDGE_INFER_VIEW_HD constexpr T* data_ptr() const noexcept { return data; }
    EDGE_INFER_VIEW_HD constexpr const dimensions_type& sizes() const noexcept { return shape; }
    EDGE_INFER_VIEW_HD constexpr const dimensions_type& strides() const noexcept { return stride; }

    EDGE_INFER_VIEW_HD constexpr std::size_t numel() const noexcept {
        std::size_t count = 1;
        for (std::size_t extent : shape) count *= extent;
        return count;
    }

    EDGE_INFER_VIEW_HD constexpr bool is_contiguous() const noexcept {
        if (!numel()) return true;
        std::size_t expected = 1;
        for (std::size_t axis = Rank; axis-- > 0;) {
            if (shape[axis] != 1 && stride[axis] != expected) return false;
            expected *= shape[axis];
        }
        return true;
    }

    EDGE_INFER_VIEW_HD constexpr TensorView<const value_type, Rank> as_const() const noexcept {
        return {data, shape, stride};
    }

    template <typename... Indices>
    EDGE_INFER_VIEW_HD constexpr T& operator()(Indices... indices) const noexcept {
        static_assert(sizeof...(Indices) == Rank, "TensorView index rank mismatch");
        const dimensions_type coordinates{static_cast<std::size_t>(indices)...};
        std::size_t offset = 0;
        for (std::size_t axis = 0; axis < Rank; ++axis) offset += coordinates[axis] * stride[axis];
        return data[offset];
    }

    EDGE_INFER_VIEW_HD constexpr TensorView subview(
        const dimensions_type& origin, const dimensions_type& extent) const noexcept {
        std::size_t offset = 0;
        for (std::size_t axis = 0; axis < Rank; ++axis) offset += origin[axis] * stride[axis];
        return {offset ? data + offset : data, extent, stride};
    }

    template <std::size_t Axis, std::size_t R = Rank,
              std::enable_if_t<(R > 1), int> = 0>
    EDGE_INFER_VIEW_HD constexpr TensorView<T, Rank - 1> select(std::size_t index) const noexcept {
        static_assert(Axis < Rank, "TensorView selection axis out of range");
        TensorView<T, Rank - 1> result{};
        const std::size_t offset = index * stride[Axis];
        result.data = offset ? data + offset : data;
        for (std::size_t source = 0, target = 0; source < Rank; ++source) {
            if (source != Axis) {
                result.shape[target] = shape[source];
                result.stride[target++] = stride[source];
            }
        }
        return result;
    }

    template <std::size_t First, std::size_t Second>
    EDGE_INFER_VIEW_HD constexpr TensorView transpose() const noexcept {
        static_assert(First < Rank && Second < Rank, "TensorView transpose axis out of range");
        TensorView result = *this;
        result.shape[First] = shape[Second];
        result.shape[Second] = shape[First];
        result.stride[First] = stride[Second];
        result.stride[Second] = stride[First];
        return result;
    }

    // Caller establishes contiguity and equal element count before using this.
    template <std::size_t NewRank>
    EDGE_INFER_VIEW_HD constexpr TensorView<T, NewRank> reshape_contiguous(
        const std::array<std::size_t, NewRank>& extent) const noexcept {
        TensorView<T, NewRank> result{data, extent, {}};
        std::size_t next = 1;
        for (std::size_t axis = NewRank; axis-- > 0;) {
            result.stride[axis] = next;
            next *= extent[axis];
        }
        return result;
    }

    static TensorView contiguous(T* pointer, const dimensions_type& extent) {
        dimensions_type strides{};
        std::size_t next = 1;
        for (std::size_t axis = Rank; axis-- > 0;) {
            strides[axis] = next;
            if (extent[axis] && next > std::numeric_limits<std::size_t>::max() / extent[axis]) {
                throw std::overflow_error("TensorView contiguous extent overflow");
            }
            next *= extent[axis];
        }
        return strided(pointer, extent, strides);
    }

    static TensorView strided(T* pointer, const dimensions_type& extent,
                              const dimensions_type& strides) {
        std::size_t count = 1;
        bool empty = false;
        for (std::size_t size : extent) empty = empty || size == 0;
        if (empty) return {pointer, extent, strides};
        std::size_t last_offset = 0;
        for (std::size_t axis = 0; axis < Rank; ++axis) {
            if (count > std::numeric_limits<std::size_t>::max() / extent[axis]) {
                throw std::overflow_error("TensorView element count overflow");
            }
            count *= extent[axis];
            const std::size_t last = extent[axis] - 1;
            if (last && strides[axis] > std::numeric_limits<std::size_t>::max() / last) {
                throw std::overflow_error("TensorView stride extent overflow");
            }
            const std::size_t contribution = last * strides[axis];
            if (last_offset > std::numeric_limits<std::size_t>::max() - contribution) {
                throw std::overflow_error("TensorView address extent overflow");
            }
            last_offset += contribution;
        }
        if (!pointer) throw std::invalid_argument("TensorView requires non-null storage for nonempty extents");
        const auto maximum = std::numeric_limits<std::size_t>::max() / sizeof(value_type);
        if (count > maximum || last_offset >= maximum) {
            throw std::overflow_error("TensorView byte extent overflow");
        }
        return {pointer, extent, strides};
    }

    TensorView checked_subview(const dimensions_type& origin,
                               const dimensions_type& extent) const {
        for (std::size_t axis = 0; axis < Rank; ++axis) {
            if (origin[axis] > shape[axis] || extent[axis] > shape[axis] - origin[axis]) {
                throw std::out_of_range("TensorView subview exceeds its source extent");
            }
        }
        return subview(origin, extent);
    }
};

static_assert(std::is_aggregate_v<TensorView<float, 2>>);
static_assert(std::is_trivially_copyable_v<TensorView<float, 2>>);
static_assert(!std::is_polymorphic_v<TensorView<float, 2>>);

#undef EDGE_INFER_VIEW_HD
