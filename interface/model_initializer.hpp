#pragma once

#include <pybind11/pybind11.h>

#include <memory>
#include <string>
#include <unordered_map>

#include "base_model.hpp"
#include "inference.hpp"

namespace py = pybind11;

// 模型初始化器类，负责初始化不同类型的模型
class ModelInitializer {
 public:
  // 初始化 Llama 模型
  static bool init_llama_model(py::dict config, py::dict weights,
                               std::shared_ptr<BaseModel>& model,
                               std::unique_ptr<infer_base>& engine);

  // 初始化 Qwen FP32 模型
  static bool init_qwen_fp32_model(py::dict config, py::dict weights,
                                   std::shared_ptr<BaseModel>& model,
                                   std::unique_ptr<infer_base>& engine);

  // 初始化 Qwen BF16 模型
  static bool init_qwen_bf16_model(py::dict config, py::dict weights,
                                   std::shared_ptr<BaseModel>& model,
                                   std::unique_ptr<infer_base>& engine);

  // 初始化 Qwen AWQ 模型
  static bool init_qwen_awq_model(py::dict config, py::dict weights,
                                  std::shared_ptr<BaseModel>& model,
                                  std::unique_ptr<infer_base>& engine);

  // 初始化 Qwen3 BF16 模型
  static bool init_qwen3_bf16_model(py::dict config, py::dict weights,
                                    std::shared_ptr<BaseModel>& model,
                                    std::unique_ptr<infer_base>& engine);

  // 初始化 Qwen3 AWQ 模型
  static bool init_qwen3_awq_model(py::dict config, py::dict weights,
                                   std::shared_ptr<BaseModel>& model,
                                   std::unique_ptr<infer_base>& engine);

  // 初始化 CUDA 内存池
  static bool init_cuda_memory_pool(
      const ModelConfig& config);

  // 打印配置和权重信息
  static void print_config_and_weights_info(py::dict config, py::dict weights);

  // 构建基础配置
  static ModelConfig build_base_config(py::dict config);
};
