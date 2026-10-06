#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

#include "weight_processor_utils.hpp"
#include "decoder_weight_mapper.hpp"

namespace py = pybind11;

namespace {
template <typename T>
py::dict mapped_values(const decoder_weight_mapper::Parameters<T>& tensors) {
    py::dict result;
    for (const auto& entry : tensors) {
        const auto& tensor = entry.second;
        py::list values;
        for (size_t flat = 0; flat < tensor.numel(); ++flat) {
            size_t remainder = flat, offset = 0;
            for (size_t axis = tensor.sizes().size(); axis-- > 0;) {
                offset += (remainder % tensor.sizes()[axis]) * tensor.strides()[axis];
                remainder /= tensor.sizes()[axis];
            }
            if constexpr (std::is_same_v<T, int32_t>) values.append(tensor.data_ptr()[offset]);
            else values.append(static_cast<float>(tensor.data_ptr()[offset]));
        }
        result[py::str(entry.first)] = py::make_tuple(tensor.sizes(), tensor.strides(), values);
    }
    return result;
}
}  // namespace

PYBIND11_MODULE(_edge_weight_conversion_test, module) {
    module.def("mapped_dense", [](const py::dict& source, bool qk_norm, bool bf16, size_t n_layers) {
        if (bf16) return mapped_values(decoder_weight_mapper::dense<__nv_bfloat16>(source, qk_norm, n_layers));
        return mapped_values(decoder_weight_mapper::dense<float>(source, qk_norm, n_layers));
    }, py::arg("source"), py::arg("qk_norm") = false, py::arg("bf16") = false,
       py::arg("n_layers") = 0);
    module.def("mapped_awq", [](const py::dict& source, bool qk_norm) {
        const auto [dense, packed, scales, zeros] = decoder_weight_mapper::awq(source, qk_norm);
        return py::make_tuple(mapped_values(dense), mapped_values(packed), mapped_values(scales), mapped_values(zeros));
    }, py::arg("source"), py::arg("qk_norm") = false);
    module.def("awq_group_size", &decoder_weight_mapper::awq_group_size);
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
