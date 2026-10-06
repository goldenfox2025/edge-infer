#pragma once

#include <cuda_bf16.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <cstring>
#include <iomanip>
#include <type_traits>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensor.hpp"

namespace py = pybind11;

/**
 * Namespace for weight-processing utilities
 * Provide shared weight-processing utilities, including tensor conversion, Weight-format processing and progress reporting
 */
namespace weight_processor_utils {

inline size_t total_weights = 0;
inline size_t processed_weights = 0;
inline std::string current_model_type = "";
inline bool progress_initialized = false;
inline size_t total_params_count = 0;

inline void update_progress(const std::string& key, const std::string& dst_key);

/**
 * from PyTorch Extract shape information from tensors
 * @param tensor PyTorch Tensor object
 * @return Vector containing shape information
 */
inline std::vector<size_t> get_tensor_shape(const py::object& tensor) {
    py::tuple shape_tuple = tensor.attr("shape");
    std::vector<size_t> shape;
    for (size_t i = 0; i < py::len(shape_tuple); ++i) {
        shape.push_back(shape_tuple[i].cast<size_t>());
    }
    return shape;
}

/**
 * Calculate the parameter count from a tensor shape
 * @param shape Shape vector
 * @return Total parameters
 */
inline size_t calculate_params_from_shape(const std::vector<size_t>& shape) {
    if (shape.empty()) {
        return 0;
    }
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
}

/**
 * Count parameters in a tensor
 * @param tensor PyTorch Tensor object
 * @return Total parameters
 */
inline size_t calculate_params_count(const py::object& tensor) {
    std::vector<size_t> shape = get_tensor_shape(tensor);
    return calculate_params_from_shape(shape);
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
    auto values = tensor.attr("detach")().attr("to")(
        py::arg("device") = "cpu", py::arg("dtype") = torch.attr("bfloat16")).attr("contiguous")();
    const auto elements = values.attr("numel")().cast<size_t>();
    std::vector<__nv_bfloat16> data(elements);
    if (elements) {
        const auto pointer = values.attr("data_ptr")().cast<uintptr_t>();
        std::memcpy(data.data(), reinterpret_cast<const void*>(pointer), elements * sizeof(__nv_bfloat16));
    }
    // Casting uses PyTorch's ordinary BF16 conversion. Checkpoint values are
    // never clamped, rescaled or replaced by a conversion heuristic.
    return Tensor<__nv_bfloat16>(std::move(data), get_tensor_shape(values));
}

inline Tensor<float> convert_float_tensor(const py::object& tensor) {
    auto values = as_contiguous_array<float>(tensor);
    std::vector<size_t> shape;
    for (int axis = 0; axis < values.ndim(); ++axis) shape.push_back(values.shape(axis));
    std::vector<float> data(values.size());
    if (!data.empty()) std::memcpy(data.data(), values.data(), data.size() * sizeof(float));
    return Tensor<float>(std::move(data), shape);
}

/**
 * Update the progress bar
 * @param key Source key
 * @param dst_key Target key
 */
inline void update_progress(const std::string& key, const std::string& dst_key) {
    if (!progress_initialized) {
        return;
    }

    processed_weights++;

    float percentage = static_cast<float>(processed_weights) / total_weights * 100.0f;
    int bar_width = static_cast<int>(percentage / 2.0f);

    std::cout << "\r";

    std::cout << " Progress: [";
    std::cout << std::string(bar_width, '=');
    if (bar_width < 50) {
        std::cout << ">";
        std::cout << std::string(49 - bar_width, ' ');
    } else {
        std::cout << "=";
    }
    std::cout << "] " << std::fixed << std::setprecision(1) << percentage << "%";

    // if (key.length() > 30) {
    //   std::cout << " " << key.substr(0, 27) << "...";
    // } else {
    //   std::cout << " " << key;
    // }

    std::cout << std::flush;
}

/**
 * Print weight-processing progress
 * @param key Source key
 * @param dst_key Target key
 */
inline void print_processing_info(const std::string& key, const std::string& dst_key) {
    if (progress_initialized) {

        update_progress(key, dst_key);
    } else {

        std::cout << "Processing key: " << key << " -> " << dst_key << std::endl;
    }
}

/**
 * Initialize the progress bar
 * @param total_weights Total weights
 * @param model_type Description of the model type
 */
inline void init_progress(size_t total_weight_count, const std::string& model_type) {

    if (progress_initialized) {
        std::cout << "\r Progress: [" << std::string(50, '=') << "] 100%";
        std::cout << "\n\033[1;32m✓ Previous weight processing was forced to completion!\033[0m\n" << std::endl;
    }

    total_weights = total_weight_count;
    processed_weights = 0;
    total_params_count = 0;
    current_model_type = model_type;
    progress_initialized = true;

    std::cout << "\n\033[1;36m Process " << model_type << " Model weights \033[0m" << std::endl;
    std::cout << " Total weights: " << total_weights << std::endl;
    std::cout << " Progress: [" << std::string(50, ' ') << "] 0%" << std::flush;
}

/**
 * Complete the progress bar
 */
inline void finish_progress() {
    if (!progress_initialized) {
        return;
    }

    std::cout << "\r Progress: [" << std::string(50, '=') << "] 100%";
    std::cout << "\n\033[1;32m✓ Weight processing complete!\033[0m" << std::endl;

    if (total_params_count > 0) {
        double params_in_millions = static_cast<double>(total_params_count) / 1000000.0;
        std::cout << " Total parameters: " << std::fixed << std::setprecision(2) << params_in_millions << " million ("
                  << total_params_count << " parameters )\n"
                  << std::endl;
    } else {
        std::cout << std::endl;
    }

    progress_initialized = false;
    processed_weights = 0;
    total_weights = 0;
    total_params_count = 0;
    current_model_type = "";
}

/**
 * Update the parameter count
 * @param count Number of parameters to add
 */
inline void update_params_count(size_t count) {
    total_params_count += count;
}

}  // namespace weight_processor_utils
