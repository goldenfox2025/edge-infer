#pragma once

#include <pybind11/pybind11.h>

#include <memory>

#include "base_model.hpp"
#include "model_factory.hpp"

namespace py = pybind11;

class ModelInitializer {
 public:
  // Prepare weights without allocating a generation engine or a KV cache.
  // Failures propagate to Python; callers publish only fully prepared models.
  static std::shared_ptr<BaseModel> prepare_model(
      py::dict config, py::dict weights, ModelType type, Device device);

  static void print_config_and_weights_info(py::dict config, py::dict weights);
  static ModelConfig build_base_config(py::dict config);
};
