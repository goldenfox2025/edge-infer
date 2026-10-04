#pragma once

#include <cuda_bf16.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <iomanip>
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

/**
 * will PyTorch Convert tensors to __nv_bfloat16 type Tensor
 * @param tensor PyTorch Tensor object
 * @return Converted bf16 Tensor
 */
inline Tensor<__nv_bfloat16> convert_bf16_tensor(const py::object& tensor) {
    try {
        py::object torch_module = py::module::import("torch");

        py::object cpu_tensor = tensor.attr("detach")().attr("cpu")();

        std::vector<size_t> shape = get_tensor_shape(cpu_tensor);

        size_t numel = 1;
        for (auto dim : shape) {
            numel *= dim;
        }

        std::vector<__nv_bfloat16> data;
        data.reserve(numel);

        if (py::hasattr(cpu_tensor, "element_size") && py::hasattr(cpu_tensor, "data_ptr")) {

            size_t element_size = cpu_tensor.attr("element_size")().cast<size_t>();

            std::string dtype_str = py::str(cpu_tensor.attr("dtype")).cast<std::string>();

            // check whether it is bfloat16 type or another 2 byte type ( such as fp16)
            if (element_size == 2) {

                if (dtype_str.find("bfloat16") != std::string::npos) {
                    // Yes bfloat16 Type, can be copied directly
                    uintptr_t data_ptr = cpu_tensor.attr("data_ptr")().cast<uintptr_t>();
                    const __nv_bfloat16* ptr = reinterpret_cast<const __nv_bfloat16*>(data_ptr);

                    for (size_t i = 0; i < numel; ++i) {
                        __nv_bfloat16 bits = ptr[i];
                        data.push_back(bits);
                    }
                } else if (dtype_str.find("float16") != std::string::npos ||
                           dtype_str.find("half") != std::string::npos) {

                    py::object float_tensor = cpu_tensor.attr("to")(torch_module.attr("float"));

                    py::object max_val = torch_module.attr("max")(float_tensor);
                    py::object min_val = torch_module.attr("min")(float_tensor);
                    float max_value = max_val.cast<float>();
                    float min_value = min_val.cast<float>();

                    if (max_value > 0 && max_value < 0.3) {

                        py::array_t<float> np_array = float_tensor.attr("numpy")().cast<py::array_t<float>>();
                        py::buffer_info buffer = np_array.request();
                        float* float_ptr = static_cast<float*>(buffer.ptr);

                        std::vector<float> fixed_data(numel);
                        size_t zeroes_fixed = 0;

                        for (size_t i = 0; i < numel; ++i) {
                            float val = float_ptr[i];

                            if (val != 0.0f && std::abs(val) < 0.001f) {

                                // among small values 0 is often quantized to 0, no processing needed
                                fixed_data[i] = val < 0 ? -0.001f : 0.001f;
                                zeroes_fixed++;
                            } else {
                                fixed_data[i] = val;
                            }
                        }

                        for (size_t i = 0; i < numel; ++i) {
                            data.push_back(__nv_bfloat16(fixed_data[i]));
                        }
                    } else {

                        py::array_t<float> np_array = float_tensor.attr("numpy")().cast<py::array_t<float>>();
                        py::buffer_info buffer = np_array.request();
                        float* float_ptr = static_cast<float*>(buffer.ptr);

                        for (size_t i = 0; i < numel; ++i) {
                            data.push_back(__nv_bfloat16(float_ptr[i]));
                        }
                    }
                } else {

                    py::object float_tensor = cpu_tensor.attr("to")(torch_module.attr("float"));

                    py::array_t<float> np_array = float_tensor.attr("numpy")().cast<py::array_t<float>>();
                    py::buffer_info buffer = np_array.request();
                    float* float_ptr = static_cast<float*>(buffer.ptr);

                    for (size_t i = 0; i < numel; ++i) {
                        data.push_back(__nv_bfloat16(float_ptr[i]));
                    }
                }
            } else {

                py::object float_tensor = cpu_tensor.attr("to")(torch_module.attr("float"));

                py::array_t<float> np_array = float_tensor.attr("numpy")().cast<py::array_t<float>>();
                py::buffer_info buffer = np_array.request();
                float* float_ptr = static_cast<float*>(buffer.ptr);

                for (size_t i = 0; i < numel; ++i) {
                    data.push_back(__nv_bfloat16(float_ptr[i]));
                }
            }
        } else {

            std::cerr << "Warning: Using fallback element-wise access for conversion" << std::endl;

            py::object float_tensor = cpu_tensor.attr("to")(torch_module.attr("float"));
            for (size_t i = 0; i < numel; ++i) {

                py::object item = float_tensor.attr("flatten")()[py::int_(i)];
                float value = item.cast<float>();
                data.push_back(__nv_bfloat16(value));
            }
        }

        return Tensor<__nv_bfloat16>(std::move(data), shape);
    } catch (const std::exception& e) {
        std::cerr << "Exception in convert_bf16_tensor: " << e.what() << std::endl;
        throw;
    }
}

/**
 * will PyTorch Convert tensors to float type Tensor
 * @param tensor PyTorch Tensor object
 * @return Converted float Tensor
 */
inline Tensor<float> convert_float_tensor(const py::object& tensor) {
    try {

        py::object cpu_tensor = tensor.attr("detach")().attr("cpu")();

        py::array_t<float> np_array = cpu_tensor.attr("numpy")().cast<py::array_t<float>>();

        std::vector<size_t> shape;
        for (int i = 0; i < np_array.ndim(); i++) {
            shape.push_back(np_array.shape(i));
        }

        std::vector<float> data(np_array.data(), np_array.data() + np_array.size());

        return Tensor<float>(std::move(data), shape);
    } catch (const std::exception& e) {
        std::cerr << "Exception in convert_float_tensor: " << e.what() << std::endl;
        throw;
    }
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
