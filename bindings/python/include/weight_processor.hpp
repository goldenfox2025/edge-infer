#pragma once

#include <cuda_bf16.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "decoder_weight_mapper.hpp"

namespace py = pybind11;

namespace weight_processor {

inline std::unordered_map<std::string, Tensor<float>> process_llama_weights(
    const py::dict& weights, size_t n_layers = 0) {
  return decoder_weight_mapper::dense<float>(weights, false, n_layers);
}

inline std::unordered_map<std::string, Tensor<float>>
process_qwen_weights_fp32(const py::dict& weights, size_t n_layers = 0) {
  return decoder_weight_mapper::dense<float>(weights, false, n_layers);
}

inline std::unordered_map<std::string, Tensor<__nv_bfloat16>>
process_qwen_weights_bf16(const py::dict& weights, size_t n_layers = 0) {
  return decoder_weight_mapper::dense<__nv_bfloat16>(weights, false, n_layers);
}

inline decoder_weight_mapper::AwqParameters
process_qwen_weights_awq(const py::dict& weights, size_t n_layers = 0) {
  return decoder_weight_mapper::awq(weights, false, n_layers);
}

inline std::unordered_map<std::string, Tensor<__nv_bfloat16>>
process_qwen3_weights_bf16(const py::dict& weights, size_t n_layers = 0) {
  return decoder_weight_mapper::dense<__nv_bfloat16>(weights, true, n_layers);
}

inline decoder_weight_mapper::AwqParameters
process_qwen3_weights_awq(const py::dict& weights, size_t n_layers = 0) {
  return decoder_weight_mapper::awq(weights, true, n_layers);
}

// Ordinary checkpoint BF16 conversion preserves values and logical order.
inline Tensor<__nv_bfloat16> convert_bf16_tensor(const py::object& tensor) {
  return weight_processor_utils::convert_bf16_tensor(tensor);
}

} // namespace weight_processor
