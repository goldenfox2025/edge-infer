// model_bridge.cpp
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "CudaMemoryPool.hpp"
#include "base_model.hpp"
#include "device_manager.hpp"
#include "include/weight_processor.hpp"
#include "inference.hpp"
#include "model_factory.hpp"
#include "model_initializer.hpp"
#include "speculative_decoder.hpp"

namespace py = pybind11;
class infer_base;

namespace {

using InitModelFn = bool (*)(py::dict, py::dict, std::shared_ptr<BaseModel>&,
                             std::unique_ptr<infer_base>&);

InitModelFn resolve_initializer(ModelType type) {
  switch (type) {
    case ModelType::LLAMA:
      return &ModelInitializer::init_llama_model;
    case ModelType::QWEN:
      return &ModelInitializer::init_qwen_fp32_model;
    case ModelType::QWEN_BF16:
      return &ModelInitializer::init_qwen_bf16_model;
    case ModelType::QWEN_AWQ:
      return &ModelInitializer::init_qwen_awq_model;
    case ModelType::QWEN3_BF16:
      return &ModelInitializer::init_qwen3_bf16_model;
    case ModelType::QWEN3_AWQ:
      return &ModelInitializer::init_qwen3_awq_model;
    default:
      throw std::runtime_error("Unsupported model type");
  }
}

InitModelFn resolve_draft_initializer(ModelType type) {
  switch (type) {
    case ModelType::QWEN_BF16:
      return &ModelInitializer::init_qwen_bf16_model;
    case ModelType::QWEN_AWQ:
      return &ModelInitializer::init_qwen_awq_model;
    case ModelType::QWEN3_BF16:
      return &ModelInitializer::init_qwen3_bf16_model;
    case ModelType::QWEN3_AWQ:
      return &ModelInitializer::init_qwen3_awq_model;
    default:
      throw std::runtime_error("Unsupported draft model type");
  }
}

ModelConfig build_runtime_pool_config(py::dict config) {
  ModelConfig cpp_config = ModelInitializer::build_base_config(config);
  cpp_config["n_layers"] = config["num_hidden_layers"].cast<int>();
  return cpp_config;
}

}  // namespace

// Global model and inference-engine instances to keep object lifetimes manageable
std::shared_ptr<BaseModel> g_model;
std::unique_ptr<infer_base> g_engine;
std::unique_ptr<infer_base> g_spec_engine;

std::shared_ptr<BaseModel> g_draft_model;

infer_base* resolve_generation_engine(bool prefer_speculative) {
  if (prefer_speculative && g_spec_engine) {
    return g_spec_engine.get();
  }
  return g_engine.get();
}

void dispatch_generation(infer_base* engine, const std::vector<uint32_t>& input_ids,
                         py::function callback, size_t max_length, float temperature,
                         float top_p, size_t top_k) {
  if (!engine) {
    throw std::runtime_error("Generation engine not initialized");
  }

  engine->generate_with_callback(input_ids, max_length, temperature, top_p,
                                 top_k, [callback](uint32_t token) {
                                   py::gil_scoped_acquire acquire;
                                   callback(token);
                                 });
}

bool init_model(py::dict config, py::dict weights,
                const std::string& model_type) {
  try {

    ModelInitializer::print_config_and_weights_info(config, weights);

    const ModelType parsed_model_type = model_type_from_string(model_type);
    const InitModelFn initializer = resolve_initializer(parsed_model_type);
    const bool result = initializer(config, weights, g_model, g_engine);
    g_spec_engine.reset();
    g_draft_model.reset();

    if (result) {

      std::cout << "\n=== Model initialization complete, Device information ===" << std::endl;
      g_model->print_device_info();
      std::cout << " Model type: " << model_type << std::endl;
      std::cout << "==============================" << std::endl;

      ModelInitializer::init_cuda_memory_pool(build_runtime_pool_config(config));
    }

    return result;
  } catch (const std::exception& e) {
    std::cerr << "Error initializing model: " << e.what() << std::endl;
    return false;
  }
}

void generate_text_stream(const std::vector<uint32_t>& input_ids,
                          py::function callback, size_t max_length = 100,
                          float temperature = 1.0f, float top_p = 0.9f,
                          size_t top_k = 50) {
  dispatch_generation(resolve_generation_engine(false), input_ids, callback,
                      max_length, temperature, top_p, top_k);
}

bool set_default_device(const std::string& device_str) {
  try {
    Device device;
    if (device_str == "cuda" || device_str == "CUDA") {
      device = Device::CUDA;

      if (!DeviceManager::instance().isCudaAvailable()) {
        std::cerr << "CUDA requested but not available. Falling back to CPU."
                  << std::endl;
        device = Device::CPU;
      }
    } else if (device_str == "cpu" || device_str == "CPU") {
      device = Device::CPU;
    } else {
      std::cerr << "Invalid device: " << device_str
                << ". Valid options are 'cuda' or 'cpu'." << std::endl;
      return false;
    }

    DeviceManager::instance().setDefaultDevice(device);
    std::cout << "Default device set to: "
              << (device == Device::CUDA ? "CUDA" : "CPU") << std::endl;
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Error setting default device: " << e.what() << std::endl;
    return false;
  }
}

std::string get_default_device() {
  Device device = DeviceManager::instance().getDefaultDevice();
  return (device == Device::CUDA) ? "cuda" : "cpu";
}

bool init_speculative_decoder(py::dict config, py::dict weights,
                              const std::string& draft_model_type,
                              size_t spec_length = 4) {
  try {

    if (!g_model) {
      throw std::runtime_error(
          "Target model not initialized. Please initialize target model "
          "first.");
    }

    std::cout << "\n=== Initialize the draft model ===" << std::endl;
    ModelInitializer::print_config_and_weights_info(config, weights);

    std::unique_ptr<infer_base> draft_engine;
    const ModelType parsed_draft_type = model_type_from_string(draft_model_type);
    const InitModelFn initializer = resolve_draft_initializer(parsed_draft_type);
    const bool result =
        initializer(config, weights, g_draft_model, draft_engine);

    if (!result) {
      throw std::runtime_error("Failed to initialize draft model");
    }

    g_spec_engine = std::make_unique<SpeculativeDecoder<__nv_bfloat16>>(
        g_model, g_draft_model, spec_length);

    std::cout << "\n=== Speculative decoder initialization complete ===" << std::endl;
    std::cout << " Target model: " << g_model->get_n_layers() << " layer " << std::endl;
    std::cout << " Draft model: " << g_draft_model->get_n_layers() << " layer "
              << std::endl;
    std::cout << " Speculative length: " << spec_length << std::endl;
    std::cout << "==============================" << std::endl;

    return true;
  } catch (const std::exception& e) {
    std::cerr << "Error initializing speculative decoder: " << e.what()
              << std::endl;
    return false;
  }
}

void generate_text_stream_speculative(const std::vector<uint32_t>& input_ids,
                                      py::function callback,
                                      size_t max_length = 100,
                                      float temperature = 1.0f,
                                      float top_p = 0.9f, size_t top_k = 50) {
  dispatch_generation(resolve_generation_engine(true), input_ids, callback,
                      max_length, temperature, top_p, top_k);
}

PYBIND11_MODULE(model_bridge, m) {
  m.def("init_model", &init_model, py::arg("config"), py::arg("weights"),
        py::arg("model_type") = "llama", "Initialize and verify the model");
  m.def("generate_text_stream", &generate_text_stream, py::arg("input_ids"),
        py::arg("callback"), py::arg("max_length") = 100,
        py::arg("temperature") = 1.0f, py::arg("top_p") = 0.9f,
        py::arg("top_k") = 50, "Stream generated tokens via callback");
  m.def("set_default_device", &set_default_device, py::arg("device"),
        "Set the default device for model execution (cuda or cpu)");
  m.def("get_default_device", &get_default_device,
        "Get the current default device for model execution");

  m.def("init_speculative_decoder", &init_speculative_decoder,
        py::arg("config"), py::arg("weights"), py::arg("draft_model_type"),
        py::arg("spec_length") = 4,
        "Initialize speculative decoder with a draft model");
  m.def("generate_text_stream_speculative", &generate_text_stream_speculative,
        py::arg("input_ids"), py::arg("callback"), py::arg("max_length") = 100,
        py::arg("temperature") = 1.0f, py::arg("top_p") = 0.9f,
        py::arg("top_k") = 50,
        "Stream generated tokens via speculative decoding");
}
