#include "python_callback.hpp"

#include <pybind11/pybind11.h>

#include <cstdint>
#include <exception>
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>

namespace py = pybind11;

namespace {

bool runtime_busy = false;

void generate(const py::function& callback, const std::string& mode) {
    edge_infer::python::dispatch_callback(
        callback, [&](std::function<void(uint32_t)> emit) {
          if (mode == "native_error") {
            throw std::runtime_error("native generation failed");
          }
          if (mode == "swallow") {
            for (uint32_t token : {2u, 3u}) {
              try {
                emit(token);
              } catch (const std::exception&) {
                // Mirror a native decoder that catches callback failures.
              }
            }
            return;
          }
          if (mode == "worker") {
            std::exception_ptr worker_error;
            std::thread worker([&] {
              try {
                emit(2);
                emit(3);
              } catch (...) {
                worker_error = std::current_exception();
              }
            });
            worker.join();
            if (worker_error) {
              std::rethrow_exception(worker_error);
            }
            return;
          }
          if (mode != "normal") {
            throw std::runtime_error("Unknown test generation mode");
          }
          emit(2);
          emit(3);
        });
}

}  // namespace

PYBIND11_MODULE(_edge_callback_boundary_test, module) {
  module.def("generate", &generate);
  module.def("generate_guarded", [](const py::function& callback,
                                    const std::string& mode) {
    edge_infer::python::RuntimeOperation operation(runtime_busy);
    generate(callback, mode);
  });
  module.def("mutate_guarded", [] {
    edge_infer::python::RuntimeOperation operation(runtime_busy);
  });
}
