#pragma once

#include <cuda_bf16.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "tensor.hpp"

namespace py = pybind11;

namespace weight_processor_utils {

inline std::vector<size_t> get_tensor_shape(const py::object& tensor) {
    const auto shape_tuple = tensor.attr("shape").cast<py::tuple>();
    std::vector<size_t> shape;
    for (const auto& extent : shape_tuple) shape.push_back(py::cast<size_t>(extent));
    return shape;
}

// Materialize caller arrays in logical C order before raw storage copies.
// PyTorch tensors are detached and moved to CPU; NumPy arrays keep their
// logical values when positive, negative or transposed strides are normalized.
template <typename T>
using ContiguousArray = py::array_t<T, py::array::c_style | py::array::forcecast>;

template <typename T>
inline ContiguousArray<T> as_contiguous_array(py::handle input) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, int32_t>);
    py::object values = py::reinterpret_borrow<py::object>(input);
    if (py::hasattr(values, "detach")) {
        const auto torch = py::module::import("torch");
        const char* dtype = std::is_same_v<T, float> ? "float32" : "int32";
        values = values.attr("detach")().attr("to")(
            py::arg("device") = "cpu", py::arg("dtype") = torch.attr(dtype)).attr("contiguous")().attr("numpy")();
    }
    return ContiguousArray<T>(values);
}

inline Tensor<__nv_bfloat16> convert_bf16_tensor(const py::object& tensor) {
    const auto torch = py::module::import("torch");
    py::object source = tensor;
    if (!py::hasattr(source, "detach")) {
        // Preserve NumPy dtype before the ordinary BF16 cast. A C-order copy
        // also lets Torch consume reversed NumPy views without changing values.
        const auto array = py::array::ensure(source, py::array::c_style);
        if (!array) throw std::invalid_argument("BF16 conversion requires a tensor or array");
        source = torch.attr("as_tensor")(array);
    }
    auto values = source.attr("detach")().attr("to")(
        py::arg("device") = "cpu", py::arg("dtype") = torch.attr("bfloat16")).attr("contiguous")();
    const auto elements = values.attr("numel")().cast<size_t>();
    std::vector<__nv_bfloat16> data(elements);
    if (elements) {
        const auto pointer = values.attr("data_ptr")().cast<uintptr_t>();
        std::memcpy(data.data(), reinterpret_cast<const void*>(pointer), elements * sizeof(__nv_bfloat16));
    }
    // Checkpoint values are never clamped, rescaled or replaced by a heuristic.
    return Tensor<__nv_bfloat16>(std::move(data), get_tensor_shape(values));
}

inline Tensor<float> convert_float_tensor(const py::object& tensor) {
    const auto values = as_contiguous_array<float>(tensor);
    std::vector<size_t> shape;
    for (int axis = 0; axis < values.ndim(); ++axis) shape.push_back(values.shape(axis));
    std::vector<float> data(values.size());
    if (!data.empty()) std::memcpy(data.data(), values.data(), data.size() * sizeof(float));
    return Tensor<float>(std::move(data), shape);
}

}  // namespace weight_processor_utils
