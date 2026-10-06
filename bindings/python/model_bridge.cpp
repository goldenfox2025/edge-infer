#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "base_model.hpp"
#include "device_manager.hpp"
#include "include/python_callback.hpp"
#include "inference.hpp"
#include "model_factory.hpp"
#include "model_initializer.hpp"
#include "speculative_decoder.hpp"

namespace py = pybind11;

namespace {

Device parse_device(const std::optional<std::string>& value) {
  if (!value) return DeviceManager::instance().getDefaultDevice();
  if (*value == "cuda" || *value == "CUDA") return Device::CUDA;
  if (*value == "cpu" || *value == "CPU") return Device::CPU;
  throw std::invalid_argument("Device must be 'cuda' or 'cpu'");
}

std::string device_name(Device device) {
  return device == Device::CUDA ? "cuda" : "cpu";
}

class PythonSession;

class PythonModel {
 public:
  PythonModel(py::dict config, py::dict weights, const std::string& model_type,
              const std::optional<std::string>& device)
      : type_(model_type_from_string(model_type)) {
    ModelInitializer::print_config_and_weights_info(config, weights);
    prototype_ = ModelInitializer::prepare_model(
        config, weights, type_, parse_device(device));
  }

  std::shared_ptr<PythonSession> new_session(std::optional<size_t> capacity) const;
  std::shared_ptr<PythonSession> new_speculative_session(
      const PythonModel& draft, std::optional<size_t> capacity,
      size_t spec_length) const;

  size_t max_context_length() const { return prototype_->get_max_seq_len(); }
  std::string device() const { return device_name(prototype_->device()); }

 private:
  friend class PythonSession;
  std::shared_ptr<BaseModel> prototype_;
  ModelType type_;
};

class PythonSession {
 public:
  PythonSession(const PythonModel& model, size_t capacity)
      : target_(model.prototype_), type_(model.type_), capacity_(capacity) {
    normal_engine_ = create_normal_engine();
  }

  PythonSession(const PythonModel& model, const PythonModel& draft,
                size_t capacity, size_t spec_length)
      : target_(model.prototype_), draft_(draft.prototype_), type_(model.type_),
        capacity_(capacity) {
    if (type_ == ModelType::LLAMA || type_ == ModelType::QWEN ||
        draft.type_ == ModelType::LLAMA || draft.type_ == ModelType::QWEN) {
      throw std::invalid_argument("Speculative sessions require BF16 or AWQ models");
    }
    if (target_->device() != Device::CUDA || draft_->device() != Device::CUDA) {
      throw std::invalid_argument("Speculative sessions require CUDA models");
    }
    speculative_engine_ = std::make_unique<SpeculativeDecoder<__nv_bfloat16>>(
        target_, draft_, spec_length, 8, capacity_);
  }

  void generate(const std::vector<uint32_t>& input_ids, py::function callback,
                size_t max_length, float temperature, float top_p,
                size_t top_k) {
    edge_infer::python::RuntimeOperation operation(busy_);
    // Exact speculative verification is greedy. Sampling uses a separate
    // ordinary engine with the same immutable target weights and capacity.
    infer_base* engine;
    if (speculative_engine_ && top_k == 1) {
      engine = speculative_engine_.get();
    } else {
      if (!normal_engine_) normal_engine_ = create_normal_engine();
      engine = normal_engine_.get();
    }
    edge_infer::python::dispatch_callback(
        callback, [&](std::function<void(uint32_t)> native_callback) {
          engine->generate_with_callback(input_ids, max_length, temperature,
                                         top_p, top_k, std::move(native_callback));
        });
  }

  void reset() {
    edge_infer::python::RuntimeOperation operation(busy_);
    if (normal_engine_) normal_engine_->reset();
    if (speculative_engine_) speculative_engine_->reset();
  }

  size_t capacity() const { return capacity_; }
  bool speculative() const { return static_cast<bool>(speculative_engine_); }
  std::string device() const { return device_name(target_->device()); }

 private:
  std::unique_ptr<infer_base> create_normal_engine() const {
    if (type_ == ModelType::LLAMA || type_ == ModelType::QWEN) {
      return std::make_unique<InferenceEngine<float>>(
          target_, target_->device(), capacity_);
    }
    return std::make_unique<InferenceEngine<__nv_bfloat16>>(
        target_, target_->device(), capacity_);
  }

  // Retain prepared weights even after the Python Model is released. Native
  // engines fork their own executors; no generation mutates these prototypes.
  std::shared_ptr<BaseModel> target_;
  std::shared_ptr<BaseModel> draft_;
  ModelType type_;
  size_t capacity_;
  bool busy_ = false;  // Read and written only while holding the Python GIL.
  std::unique_ptr<infer_base> normal_engine_;
  std::unique_ptr<SpeculativeDecoder<__nv_bfloat16>> speculative_engine_;
};

size_t resolve_capacity(std::optional<size_t> requested, size_t maximum) {
  const size_t result = requested.value_or(std::min<size_t>(4096, maximum));
  if (!result || result > maximum) {
    throw std::invalid_argument("Session capacity must be positive and within the model context limit");
  }
  return result;
}

std::shared_ptr<PythonSession> PythonModel::new_session(
    std::optional<size_t> capacity) const {
  return std::make_shared<PythonSession>(
      *this, resolve_capacity(capacity, max_context_length()));
}

std::shared_ptr<PythonSession> PythonModel::new_speculative_session(
    const PythonModel& draft, std::optional<size_t> capacity,
    size_t spec_length) const {
  return std::make_shared<PythonSession>(
      *this, draft,
      resolve_capacity(capacity, std::min(max_context_length(),
                                        draft.max_context_length())),
      spec_length);
}

// The procedural API is a compatibility wrapper over one explicit session.
// Replacement is atomic: a failed construction preserves the previous model,
// ordinary session and speculative session.
bool g_compatibility_busy = false;
std::shared_ptr<PythonModel> g_model;
std::shared_ptr<PythonSession> g_session;
std::shared_ptr<PythonSession> g_speculative_session;

bool init_model(py::dict config, py::dict weights,
                const std::string& model_type) {
  edge_infer::python::RuntimeOperation operation(g_compatibility_busy);
  try {
    auto model = std::make_shared<PythonModel>(
        config, weights, model_type, std::nullopt);
    auto session = model->new_session(std::nullopt);
    g_speculative_session.reset();
    g_session = std::move(session);
    g_model = std::move(model);
    return true;
  } catch (const std::exception& error) {
    std::cerr << "Error initializing model: " << error.what() << std::endl;
    return false;
  }
}

void generate_text_stream(const std::vector<uint32_t>& input_ids,
                          py::function callback, size_t max_length,
                          float temperature, float top_p, size_t top_k) {
  edge_infer::python::RuntimeOperation operation(g_compatibility_busy);
  if (!g_session) throw std::runtime_error("Model not initialized");
  g_session->generate(input_ids, std::move(callback), max_length,
                      temperature, top_p, top_k);
}

bool set_default_device(const std::string& device) {
  edge_infer::python::RuntimeOperation operation(g_compatibility_busy);
  try {
    const Device parsed = parse_device(device);
    if (parsed == Device::CUDA && !DeviceManager::instance().isCudaAvailable()) {
      throw std::runtime_error("CUDA requested but no CUDA device is available");
    }
    DeviceManager::instance().setDefaultDevice(parsed);
    return true;
  } catch (const std::exception& error) {
    std::cerr << "Error setting default device: " << error.what() << std::endl;
    return false;
  }
}

std::string get_default_device() {
  return device_name(DeviceManager::instance().getDefaultDevice());
}

bool init_speculative_decoder(py::dict config, py::dict weights,
                              const std::string& draft_model_type,
                              size_t spec_length) {
  edge_infer::python::RuntimeOperation operation(g_compatibility_busy);
  try {
    if (!g_model) throw std::runtime_error("Target model not initialized");
    auto draft = std::make_shared<PythonModel>(
        config, weights, draft_model_type, g_model->device());
    auto session = g_model->new_speculative_session(
        *draft, g_session->capacity(), spec_length);
    g_speculative_session = std::move(session);
    return true;
  } catch (const std::exception& error) {
    std::cerr << "Error initializing speculative decoder: " << error.what()
              << std::endl;
    return false;
  }
}

void generate_text_stream_speculative(
    const std::vector<uint32_t>& input_ids, py::function callback,
    size_t max_length, float temperature, float top_p, size_t top_k) {
  edge_infer::python::RuntimeOperation operation(g_compatibility_busy);
  // Keep stochastic compatibility calls on the existing ordinary session.
  auto session = top_k == 1 && g_speculative_session
                     ? g_speculative_session : g_session;
  if (!session) throw std::runtime_error("Model not initialized");
  session->generate(input_ids, std::move(callback), max_length,
                    temperature, top_p, top_k);
}

}  // namespace

PYBIND11_MODULE(model_bridge, m) {
  m.doc() = "Prepared models and independent bounded generation sessions.";
  py::class_<PythonSession, std::shared_ptr<PythonSession>>(m, "Session")
      .def("generate", &PythonSession::generate, py::arg("input_ids"),
           py::arg("callback"), py::arg("max_length") = 100,
           py::arg("temperature") = 1.0f, py::arg("top_p") = 0.9f,
           py::arg("top_k") = 1,
           "Generate from a complete fresh prompt; max_length includes prompt tokens.")
      .def("reset", &PythonSession::reset, "Clear request state while retaining allocations.")
      .def_property_readonly("capacity", &PythonSession::capacity)
      .def_property_readonly("device", &PythonSession::device)
      .def_property_readonly("speculative", &PythonSession::speculative);

  py::class_<PythonModel, std::shared_ptr<PythonModel>>(m, "Model")
      .def(py::init<py::dict, py::dict, const std::string&,
                    const std::optional<std::string>&>(),
           py::arg("config"), py::arg("weights"),
           py::arg("model_type") = "llama", py::arg("device") = py::none(),
           "Prepare immutable model weights without a generation KV cache.")
      .def("new_session", &PythonModel::new_session,
           py::arg("capacity") = py::none(),
           "Create an independent session; default capacity is min(4096, model limit).")
      .def("new_speculative_session", &PythonModel::new_speculative_session,
           py::arg("draft_model"), py::arg("capacity") = py::none(),
           py::arg("spec_length") = 4,
           "Create an exact greedy speculative session; stochastic requests use ordinary inference.")
      .def_property_readonly("max_context_length", &PythonModel::max_context_length)
      .def_property_readonly("device", &PythonModel::device);

  m.def("init_model", &init_model, py::arg("config"), py::arg("weights"),
        py::arg("model_type") = "llama", "Replace the compatibility model on success.");
  m.def("generate_text_stream", &generate_text_stream, py::arg("input_ids"),
        py::arg("callback"), py::arg("max_length") = 100,
        py::arg("temperature") = 1.0f, py::arg("top_p") = 0.9f,
        py::arg("top_k") = 50, "Generate from a complete fresh prompt.");
  m.def("set_default_device", &set_default_device, py::arg("device"),
        "Set the default device for future model construction.");
  m.def("get_default_device", &get_default_device);
  m.def("is_cuda_available", [] { return DeviceManager::instance().isCudaAvailable(); });

  m.def("init_speculative_decoder", &init_speculative_decoder,
        py::arg("config"), py::arg("weights"), py::arg("draft_model_type"),
        py::arg("spec_length") = 4,
        "Replace the compatibility greedy speculative session on success.");
  m.def("generate_text_stream_speculative", &generate_text_stream_speculative,
        py::arg("input_ids"), py::arg("callback"), py::arg("max_length") = 100,
        py::arg("temperature") = 1.0f, py::arg("top_p") = 0.9f,
        py::arg("top_k") = 1,
        "Generate from a complete prompt; top_k != 1 selects ordinary inference.");
}
