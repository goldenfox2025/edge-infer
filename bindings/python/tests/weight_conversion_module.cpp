#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "weight_processor_utils.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_edge_weight_conversion_test, module) {
    module.def("bf16_bits", [](const py::object& source) {
        const auto tensor = weight_processor_utils::convert_bf16_tensor(source);
        std::vector<std::uint16_t> bits(tensor.numel());
        if (!bits.empty()) std::memcpy(bits.data(), tensor.data_ptr(), bits.size() * sizeof(bits[0]));
        return py::make_tuple(tensor.sizes(), bits);
    });
    module.def("float_values", [](const py::object& source) {
        const auto tensor = weight_processor_utils::convert_float_tensor(source);
        std::vector<float> values(tensor.numel());
        if (!values.empty()) std::memcpy(values.data(), tensor.data_ptr(), values.size() * sizeof(float));
        return py::make_tuple(tensor.sizes(), values);
    });
    module.def("packed_int_values", [](const py::object& source) {
        const auto array = weight_processor_utils::as_contiguous_array<std::int32_t>(source);
        std::vector<std::size_t> shape;
        for (int axis = 0; axis < array.ndim(); ++axis) shape.push_back(array.shape(axis));
        std::vector<std::int32_t> values(array.size());
        if (!values.empty()) std::memcpy(values.data(), array.data(), values.size() * sizeof(values[0]));
        return py::make_tuple(shape, values);
    });
}
