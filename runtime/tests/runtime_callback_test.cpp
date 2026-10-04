#include "base_model.hpp"
#include "inference.hpp"

#include <atomic>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

class CallbackError : public std::runtime_error {
 public:
  CallbackError() : std::runtime_error("native callback failed") {}
};

class WorkerError : public std::runtime_error {
 public:
  WorkerError() : std::runtime_error("native prefill failed") {}
};

// A deterministic model verifies the public native runtime boundary without
// checkpoints, a tokenizer, Python, or GPU execution.
class TokenModel final : public BaseModel {
 public:
  explicit TokenModel(bool fail_prefill = false) : fail_prefill_(fail_prefill) {}

  uint32_t* prefill(const Tensor<uint32_t>*, ThreadPool&, KVCacheBase*, size_t,
                    float, float, curandState*) override {
    if (fail_prefill_) {
      throw WorkerError{};
    }
    return new uint32_t(2);
  }

  uint32_t* forward(const Tensor<uint32_t>*, ThreadPool&, KVCacheBase*, size_t,
                    float, float, curandState*) override {
    const int call = ++decode_calls;
    return new uint32_t(call == 1 ? 3 : get_eos_token_id());
  }

  bool verify_params() const override { return true; }
  void print_model_info() const override {}
  BaseModel& cuda() override { throw std::runtime_error("CPU-only test model"); }
  BaseModel& cpu() override { return *this; }
  Device device() const override { return Device::CPU; }
  size_t get_n_layers() const override { return 1; }
  size_t get_max_seq_len() const override { return 16; }
  size_t get_head_dim() const override { return 1; }
  size_t get_n_kv_heads() const override { return 1; }
  size_t get_vocab_size() const override { return 16; }
  uint32_t get_eos_token_id() const override { return 9; }
  size_t get_hidden_size() const override { return 1; }

  std::atomic<int> decode_calls{0};

 private:
  bool fail_prefill_;
};

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void test_tokens_and_thread() {
  auto model = std::make_shared<TokenModel>();
  InferenceEngine<float> engine(model, Device::CPU);
  const auto caller = std::this_thread::get_id();
  std::vector<uint32_t> tokens;
  engine.generate_with_callback({1}, 4, 1.0f, 0.9f, 1, [&](uint32_t token) {
    require(std::this_thread::get_id() == caller,
            "The callback must run on the native caller thread");
    tokens.push_back(token);
  });
  require(tokens == std::vector<uint32_t>({2, 3}), "Incorrect native token stream");
  require(model->decode_calls == 2, "Generation must finish before returning");
}

void test_callback_exception() {
  auto model = std::make_shared<TokenModel>();
  InferenceEngine<float> engine(model, Device::CPU);
  bool caught = false;
  try {
    engine.generate_with_callback({1}, 4, 1.0f, 0.9f, 1,
                                  [](uint32_t) { throw CallbackError{}; });
  } catch (const CallbackError& error) {
    caught = std::string(error.what()) == "native callback failed";
  }
  require(caught, "The original native callback exception must reach the caller");
  require(model->decode_calls == 2, "The generation worker must be joined on failure");
}

void test_worker_exception() {
  auto model = std::make_shared<TokenModel>(true);
  InferenceEngine<float> engine(model, Device::CPU);
  bool caught = false;
  try {
    engine.generate_with_callback({1}, 4, 1.0f, 0.9f, 1,
                                  [](uint32_t) {
                                    throw std::runtime_error("Unexpected callback");
                                  });
  } catch (const WorkerError& error) {
    caught = std::string(error.what()) == "native prefill failed";
  }
  require(caught, "The original worker exception must reach the native caller");
}

}  // namespace

int main() {
  try {
    test_tokens_and_thread();
    test_callback_exception();
    test_worker_exception();
    std::cout << "Native runtime callback tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
