#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include "model_initializer.hpp"
#include "qwen3.hpp"

namespace py = pybind11;

namespace {

using BFloat16 = __nv_bfloat16;
using Session = Qwen3Session<BFloat16>;
using Model = Qwen3Model<BFloat16>;

py::array_t<float> copy_logits(TensorView<BFloat16, 2> logits) {
    if (!logits.is_contiguous()) throw std::runtime_error("Diagnostic logits must be contiguous");
    std::vector<BFloat16> host(logits.numel());
    if (!host.empty()) {
        const auto status = cudaMemcpy(host.data(), logits.data, host.size() * sizeof(BFloat16), cudaMemcpyDeviceToHost);
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    }
    py::array_t<float> result({static_cast<py::ssize_t>(logits.shape[0]),
                              static_cast<py::ssize_t>(logits.shape[1])});
    auto* destination = result.mutable_data();
    for (std::size_t index = 0; index < host.size(); ++index)
        destination[index] = __bfloat162float(host[index]);
    return result;
}

class DiagnosticSession {
 public:
    DiagnosticSession(std::shared_ptr<const Model> model, std::size_t capacity, bool graph)
        : session_(Session::create(std::move(model), capacity, graph)) {}
    py::array_t<float> prefill(const std::vector<std::uint32_t>& tokens) {
        return copy_logits(session_->prefill(tokens));
    }
    py::array_t<float> decode(std::uint32_t token) {
        return copy_logits(session_->decode(token));
    }
    void reset() { session_->reset(); }
    std::size_t context_size() const { return session_->context_size(); }
    std::size_t capacity() const { return session_->context_capacity(); }

 private:
    std::unique_ptr<Session> session_;
};

class DiagnosticModel {
 public:
    DiagnosticModel(py::dict config, py::dict weights) {
        auto prepared = ModelInitializer::prepare_model(config, weights, ModelType::QWEN3_BF16, Device::CUDA);
        const auto typed = std::dynamic_pointer_cast<Session>(prepared);
        if (!typed) throw std::runtime_error("Prepared diagnostic model is not a Qwen3 session");
        model_ = typed->model();
    }
    std::shared_ptr<DiagnosticSession> new_session(std::size_t capacity, bool graph) const {
        return std::make_shared<DiagnosticSession>(model_, capacity, graph);
    }

 private:
    std::shared_ptr<const Model> model_;
};

}  // namespace

PYBIND11_MODULE(_edge_qwen3_logits_test, module) {
    module.doc() = "Test-only copied Qwen3 logits for local reference checks; no production API.";
    py::class_<DiagnosticSession, std::shared_ptr<DiagnosticSession>>(module, "Session")
        .def("prefill", &DiagnosticSession::prefill, py::arg("tokens"))
        .def("decode", &DiagnosticSession::decode, py::arg("token"))
        .def("reset", &DiagnosticSession::reset)
        .def_property_readonly("context_size", &DiagnosticSession::context_size)
        .def_property_readonly("capacity", &DiagnosticSession::capacity);
    py::class_<DiagnosticModel>(module, "Model")
        .def(py::init<py::dict, py::dict>(), py::arg("config"), py::arg("weights"))
        .def("new_session", &DiagnosticModel::new_session, py::arg("capacity"), py::arg("graph") = false);
}
