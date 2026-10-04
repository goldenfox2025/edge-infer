#pragma once

#include <pybind11/pybind11.h>

#include <memory>
#include <string>
#include <unordered_map>

#include "base_model.hpp"
#include "inference.hpp"

namespace py = pybind11;

class ModelInitializer {
 public:

  static bool init_llama_model(py::dict config, py::dict weights,
                               std::shared_ptr<BaseModel>& model,
                               std::unique_ptr<infer_base>& engine);

  static bool init_qwen_fp32_model(py::dict config, py::dict weights,
                                   std::shared_ptr<BaseModel>& model,
                                   std::unique_ptr<infer_base>& engine);

  static bool init_qwen_bf16_model(py::dict config, py::dict weights,
                                   std::shared_ptr<BaseModel>& model,
                                   std::unique_ptr<infer_base>& engine);

  static bool init_qwen_awq_model(py::dict config, py::dict weights,
                                  std::shared_ptr<BaseModel>& model,
                                  std::unique_ptr<infer_base>& engine);

  static bool init_qwen3_bf16_model(py::dict config, py::dict weights,
                                    std::shared_ptr<BaseModel>& model,
                                    std::unique_ptr<infer_base>& engine);

  static bool init_qwen3_awq_model(py::dict config, py::dict weights,
                                   std::shared_ptr<BaseModel>& model,
                                   std::unique_ptr<infer_base>& engine);

  static bool init_cuda_memory_pool(
      const ModelConfig& config);

  static void print_config_and_weights_info(py::dict config, py::dict weights);

  static ModelConfig build_base_config(py::dict config);
};
