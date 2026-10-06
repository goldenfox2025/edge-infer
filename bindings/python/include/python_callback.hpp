#pragma once

#include <pybind11/pybind11.h>

#include <cstdint>
#include <exception>
#include <functional>
#include <stdexcept>
#include <utility>

namespace edge_infer::python {

// Enter and leave while holding the GIL. Each mutable session has its own guard;
// the procedural compatibility wrapper has another guard for atomic replacement.
// Reject overlapping operations instead of waiting while holding the GIL.
class RuntimeOperation {
 public:
  explicit RuntimeOperation(bool& busy) : busy_(busy) {
    if (busy_) {
      throw std::runtime_error(
          "Inference runtime is busy; wait for the current operation to finish");
    }
    busy_ = true;
  }

  ~RuntimeOperation() { busy_ = false; }
  RuntimeOperation(const RuntimeOperation&) = delete;
  RuntimeOperation& operator=(const RuntimeOperation&) = delete;

 private:
  bool& busy_;
};

// Native generation sees only this marker. The binding retains the original
// Python exception and restores it after generation has returned with the GIL.
class CallbackFailure : public std::runtime_error {
 public:
  CallbackFailure() : std::runtime_error("Python generation callback failed") {}
};

// Generation must finish or join all callback work before returning. Neither
// the native callback nor any of its copies owns a Python reference, so copying
// or destroying it while the GIL is released cannot change Python refcounts.
template <typename Generate>
void dispatch_callback(const pybind11::function& callback, Generate&& generate) {
  std::exception_ptr callback_error;
  std::exception_ptr generation_error;
  std::function<void(uint32_t)> native_callback =
      [&callback, &callback_error](uint32_t token) {
        pybind11::gil_scoped_acquire acquire;
        // Access under the GIL also serializes callbacks from native threads.
        if (callback_error) {
          throw CallbackFailure{};
        }
        try {
          callback(token);
        } catch (...) {
          callback_error = std::current_exception();
          throw CallbackFailure{};
        }
      };

  {
    pybind11::gil_scoped_release release;
    try {
      std::forward<Generate>(generate)(std::move(native_callback));
    } catch (...) {
      generation_error = std::current_exception();
    }
  }

  // Some native generation paths catch callback failures internally. Preserve
  // the original Python type, value, and traceback even in that case. Both
  // exception pointers are destroyed only after the GIL has been restored.
  if (callback_error) {
    std::rethrow_exception(callback_error);
  }
  if (generation_error) {
    std::rethrow_exception(generation_error);
  }
}

}  // namespace edge_infer::python
