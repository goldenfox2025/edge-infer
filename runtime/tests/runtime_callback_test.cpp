#include "base_model.hpp"
#include "inference.hpp"
#include "qwen.hpp"

#include <algorithm>
#include <atomic>
#include <iostream>
#include <limits>
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

class DecodeError : public std::runtime_error {
 public:
  DecodeError() : std::runtime_error("native decode failed") {}
};

// A deterministic model verifies the public native runtime boundary without
// checkpoints, a tokenizer, Python, or GPU execution.
class TokenModel final : public BaseModel {
 public:
  explicit TokenModel(bool fail_prefill = false, bool mutate_on_cuda = false,
                      bool endless = false)
      : fail_prefill_(fail_prefill), mutate_on_cuda_(mutate_on_cuda), endless_(endless) {}

  uint32_t* prefill(const Tensor<uint32_t>* input, ThreadPool&, KVCacheBase* cache, size_t,
                    float, float, curandState*) override {
    ++prefill_calls;
    prefill_thread = std::this_thread::get_id();
    last_cache = cache;
    prefill_offset = cache->size() - input->numel();
    if (fail_prefill_) {
      throw WorkerError{};
    }
    return new uint32_t(2);
  }

  uint32_t* forward(const Tensor<uint32_t>*, ThreadPool&, KVCacheBase*, size_t,
                    float, float, curandState*) override {
    const int call = ++decode_calls;
    decode_thread = std::this_thread::get_id();
    if (fail_decode) throw DecodeError{};
    return new uint32_t(endless_ || call == 1 ? 3 : get_eos_token_id());
  }

  bool verify_params() const override { return true; }
  void print_model_info() const override {}
  BaseModel& cuda() override {
    ++migration_calls;
    if (mutate_on_cuda_) device_ = Device::CUDA;
    throw std::runtime_error("CPU-only test model");
  }
  BaseModel& cpu() override { return *this; }
  Device device() const override { return device_; }
  void synchronize() const override {
    ++synchronize_calls;
    if (fail_synchronize) throw WorkerError{};
  }
  size_t get_n_layers() const override { return 1; }
  size_t get_max_seq_len() const override { return endless_ ? 1024 : 16; }
  size_t get_head_dim() const override { return 1; }
  size_t get_n_kv_heads() const override { return 1; }
  size_t get_vocab_size() const override { return 16; }
  uint32_t get_eos_token_id() const override { return 9; }
  size_t get_hidden_size() const override { return 1; }

  std::atomic<int> decode_calls{0};
  std::atomic<int> prefill_calls{0};
  std::atomic<size_t> prefill_offset{0};
  int migration_calls = 0;
  mutable int synchronize_calls = 0;
  bool fail_synchronize = false;
  bool fail_decode = false;
  KVCacheBase* last_cache = nullptr;
  std::thread::id prefill_thread, decode_thread;

 private:
  bool fail_prefill_;
  bool mutate_on_cuda_;
  bool endless_;
  Device device_ = Device::CPU;
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
  require(model->prefill_thread == caller && model->decode_thread == caller,
          "Native token operations must run on the calling thread");
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
  require(model->decode_calls == 0 && engine.context_size() == 0,
          "Callback failure must stop before decode and discard the failed request");
}

void test_long_request_failure_and_reuse() {
  auto model = std::make_shared<TokenModel>(false, false, true);
  InferenceEngine<float> engine(model, Device::CPU);
  int callbacks = 0;
  bool caught = false;
  try {
    engine.generate_with_callback({1}, 1024, 1.0f, 0.9f, 1, [&](uint32_t) {
      ++callbacks;
      throw CallbackError{};
    });
  } catch (const CallbackError& error) {
    caught = std::string(error.what()) == "native callback failed";
  }
  require(caught && callbacks == 1 && model->prefill_calls == 1 && model->decode_calls == 0 &&
              engine.context_size() == 0,
          "A first callback failure must not drain a long request or queue later tokens");
  std::vector<uint32_t> tokens;
  engine.generate_with_callback({1}, 8, 1.0f, 0.9f, 1,
                                [&](uint32_t token) { tokens.push_back(token); });
  require(tokens == std::vector<uint32_t>({2, 3, 3, 3, 3, 3, 3}) &&
              model->decode_calls == 6 && engine.context_size() == 7,
          "An engine must accept a fresh request after a callback failure");
}

void test_failed_callback_completion_invalidates_engine() {
  auto model = std::make_shared<TokenModel>(false, false, true);
  InferenceEngine<float> engine(model, Device::CPU, 4);
  bool caught = false;
  try {
    engine.generate_with_callback({1}, 4, 1.0f, 0.9f, 1, [&](uint32_t) {
      model->fail_synchronize = true;
      throw CallbackError{};
    });
  } catch (const CallbackError& error) {
    caught = std::string(error.what()) == "native callback failed";
  }
  require(caught && model->prefill_calls == 1 && model->decode_calls == 0,
          "Completion failure must preserve the original callback exception");
  // Even if a backend can later complete, the failed engine cannot reuse its
  // existing cache or execution resources. Only a fresh engine can retry.
  model->fail_synchronize = false;
  const auto require_invalid = [](auto operation) {
    bool rejected = false;
    try { operation(); }
    catch (const std::logic_error& error) {
      rejected = std::string(error.what()).find("construct a new inference engine") != std::string::npos;
    }
    require(rejected, "Failed completion must invalidate every public engine operation");
  };
  ThreadPool unused(0);
  uint32_t token = 1;
  int callbacks = 0;
  require_invalid([&] { engine.generate_with_callback({1}, 0, 1.0f, 0.9f, 1,
                                                     [&](uint32_t) { ++callbacks; }); });
  require_invalid([&] { engine.generate_next_token(unused, &token); });
  require_invalid([&] { engine.warmup(); });
  require_invalid([&] { engine.reset(); });
  require_invalid([&] { engine.cuda(); });
  require_invalid([&] { engine.cpu(); });
  require_invalid([&] { engine.device(); });
  require_invalid([&] { engine.context_size(); });
  require_invalid([&] { engine.context_capacity(); });
  require_invalid([&] { engine.set_benchmark_mode(true); });
  require(callbacks == 0 && model->prefill_calls == 1 && model->decode_calls == 0,
          "Invalid engine methods must reject before callbacks or native execution");
  InferenceEngine<float> fresh(model, Device::CPU, 4);
  std::vector<uint32_t> output;
  fresh.generate_with_callback({1}, 4, 1.0f, 0.9f, 1,
                               [&](uint32_t next) { output.push_back(next); });
  require(output == std::vector<uint32_t>({2, 3, 3}),
          "A fresh engine must be able to retry after backend completion recovers");
}

void test_decode_failure_and_reuse() {
  auto model = std::make_shared<TokenModel>(false, false, true);
  InferenceEngine<float> engine(model, Device::CPU);
  model->fail_decode = true;
  std::vector<uint32_t> tokens;
  bool caught = false;
  try {
    engine.generate_with_callback({1}, 8, 1.0f, 0.9f, 1,
                                  [&](uint32_t token) { tokens.push_back(token); });
  } catch (const DecodeError& error) {
    caught = std::string(error.what()) == "native decode failed";
  }
  require(caught && tokens == std::vector<uint32_t>{2} && model->decode_calls == 1 &&
              engine.context_size() == 0,
          "Native decode failure must preserve its error and discard the failed request");
  model->fail_decode = false;
  tokens.clear();
  engine.generate_with_callback({1}, 4, 1.0f, 0.9f, 1,
                                [&](uint32_t token) { tokens.push_back(token); });
  require(tokens == std::vector<uint32_t>({2, 3, 3}),
          "An engine must accept a fresh request after a native decode failure");
}

void test_prefill_exception() {
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
  require(caught, "The original prefill exception must reach the native caller");
  require(model->prefill_calls == 1 && model->decode_calls == 0 && engine.context_size() == 0,
          "Native prefill failure must discard the failed request before returning");
}

void test_fresh_request_and_capacity() {
  auto model = std::make_shared<TokenModel>();
  InferenceEngine<float> engine(model, Device::CPU, 4);
  require(engine.context_capacity() == 4, "Engine must reserve its explicit context capacity");
  std::vector<uint32_t> tokens;
  engine.generate_with_callback({1, 2}, 3, 1.0f, 0.9f, 1,
                                [&](uint32_t token) { tokens.push_back(token); });
  require(tokens == std::vector<uint32_t>{2} && engine.context_size() == 2,
          "Initial short request has the wrong token or cache extent");
  tokens.clear();
  engine.generate_with_callback({3, 4, 5}, 4, 1.0f, 0.9f, 1,
                                [&](uint32_t token) { tokens.push_back(token); });
  require(tokens == std::vector<uint32_t>{2} && engine.context_size() == 3 && model->prefill_offset == 0,
          "A complete new prompt must replace the previous request history");
  const int calls = model->prefill_calls;
  bool rejected = false;
  try { engine.generate_with_callback({1}, 3, 1.0f, 0.9f, 0, [](uint32_t) {}); }
  catch (const std::invalid_argument&) { rejected = true; }
  require(rejected && model->prefill_calls == calls && engine.context_size() == 3,
          "Invalid sampling must reject before clearing or executing the current request");
  engine.generate_with_callback({1, 2, 3, 4}, 5, 1.0f, 0.9f, 1,
                                [](uint32_t) { throw std::runtime_error("Full capacity generated a token"); });
  require(model->prefill_calls == calls, "Generation limit must respect the requested context capacity");
  engine.reset();
  require(engine.context_size() == 0, "Engine reset must retain an empty context");
  rejected = false;
  try { InferenceEngine<float> invalid(model, Device::CPU, 17); }
  catch (const std::invalid_argument&) { rejected = true; }
  require(rejected, "Capacity beyond the model context must reject");
}

void test_cpu_sampling_rejects_before_writes() {
  using Parameters = QwenModel<float>::Parameters;
  Parameters weights;
  weights.emplace("token_embeddings.weight", Tensor<float>(std::vector<float>(8, 1.0f), {4, 2}));
  for (const char* name : {"rms_out_w", "rms_att_w0", "rms_ffn_w0"})
    weights.emplace(name, Tensor<float>(std::vector<float>(2, 1.0f), {2}));
  for (const char* name : {"wq0", "wk0", "wv0", "wo0", "w_gate0", "w_up0", "w_down0"})
    weights.emplace(name, Tensor<float>(std::vector<float>(4, 0.0f), {2, 2}));
  weights.emplace("lm_head", Tensor<float>(std::vector<float>(8, 0.0f), {2, 4}));
  ModelConfig config{{"vocab_size", 4}, {"n_layers", 1}, {"n_heads", 1}, {"n_kv_heads", 1},
                    {"hidden_size", 2}, {"head_dim", 2}, {"intermediate_size", 2},
                    {"max_position_embeddings", 8}, {"bos_token_id", 1}, {"eos_token_id", 3},
                    {"rms_norm_eps", 1.0e-6}, {"rope_theta", 10000}};
  QwenModel<float> model(weights, config);
  KVCache<float> cache(1, 8, 2, Device::CPU, 3);
  auto keys = cache.k_capacity_view(0), values = cache.v_capacity_view(0);
  std::fill_n(keys.data, keys.numel(), 77.0f);
  std::fill_n(values.data, values.numel(), 66.0f);
  Tensor<uint32_t> prompt(std::vector<uint32_t>{1, 2, 1}, {3});
  Tensor<uint32_t> token(std::vector<uint32_t>{1}, {1});
  ThreadPool unused(0);
  for (int policy = 0; policy < 3; ++policy) {
    const float temperature = policy == 1 ? std::numeric_limits<float>::quiet_NaN() : 1.0f;
    const float top_p = policy == 2 ? 0.0f : 0.9f;
    const size_t top_k = policy == 0 ? 0 : 1;
    for (bool prefill : {true, false}) {
      bool rejected = false;
      try {
        std::unique_ptr<uint32_t> unexpected(prefill
            ? model.prefill(&prompt, unused, &cache, top_k, temperature, top_p)
            : model.forward(&token, unused, &cache, top_k, temperature, top_p));
      } catch (const std::invalid_argument&) { rejected = true; }
      require(rejected, "Invalid CPU sampling policy must reject");
      require(std::all_of(keys.data, keys.data + keys.numel(), [](float x) { return x == 77.0f; }) &&
                  std::all_of(values.data, values.data + values.numel(), [](float x) { return x == 66.0f; }),
              "Rejected CPU sampling policy must preserve all cache contents");
    }
  }
}

void test_failed_migration_invalidates_engine() {
  auto model = std::make_shared<TokenModel>(false, true);
  InferenceEngine<float> engine(model, Device::CPU, 4);
  engine.generate_with_callback({1}, 2, 1.0f, 0.9f, 1, [](uint32_t) {});
  bool caught = false;
  try { engine.cuda(); }
  catch (const std::runtime_error& error) {
    caught = std::string(error.what()) == "CPU-only test model";
  }
  require(caught && model->device() == Device::CUDA,
          "Migration must preserve the original exception after model state changes");
  const auto require_invalid = [](auto operation) {
    bool rejected = false;
    try { operation(); }
    catch (const std::logic_error& error) {
      rejected = std::string(error.what()).find("construct a new inference engine") != std::string::npos;
    }
    require(rejected, "Every public operation must reject an engine after failed migration");
  };
  ThreadPool unused(0);
  uint32_t token = 1;
  int callbacks = 0;
  require_invalid([&] { engine.generate_with_callback({1}, 0, 1.0f, 0.9f, 1,
                                                     [&](uint32_t) { ++callbacks; }); });
  require_invalid([&] { engine.generate_next_token(unused, &token); });
  require_invalid([&] { engine.warmup(); });
  require_invalid([&] { engine.reset(); });
  require_invalid([&] { engine.cuda(); });
  require_invalid([&] { engine.cpu(); });
  require_invalid([&] { engine.device(); });
  require_invalid([&] { engine.context_size(); });
  require_invalid([&] { engine.context_capacity(); });
  require_invalid([&] { engine.set_benchmark_mode(true); });
  require(callbacks == 0 && model->prefill_calls == 1 && model->decode_calls == 0 &&
              model->migration_calls == 1,
          "Invalid engine operations must fail before callbacks or model execution");
}

void test_thread_pool_failure_and_drain() {
  for (size_t workers : {size_t{0}, size_t{1}, size_t{2}}) {
    ThreadPool pool(workers);
    const auto caller = std::this_thread::get_id();
    std::atomic<int> completed{0};
    std::atomic<bool> wrong_inline_thread{false};
    pool.enqueueTask([] { throw WorkerError{}; });
    for (int index = 0; index < 40; ++index) {
      pool.enqueueTask([&] {
        if (!workers && std::this_thread::get_id() != caller) wrong_inline_thread = true;
        ++completed;
      });
    }
    if (!workers)
      require(completed == 40 && !wrong_inline_thread,
              "Zero-worker pool must execute synchronously on the submitting thread");
    bool caught = false;
    try { pool.waitForAllTasks(); }
    catch (const WorkerError&) { caught = true; }
    require(caught && completed == 40,
            "Pool must drain all tasks and preserve the original task exception");
    pool.waitForAllTasks();
    pool.enqueueTask([&] { ++completed; });
    pool.waitForAllTasks();
    require(completed == 41, "Pool must remain usable after reporting a task exception");
    bool rejected = false;
    try { pool.enqueueTask({}); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "Empty tasks must reject without changing outstanding work");
    pool.enqueueTask([&] { ++completed; });
    pool.stopThreadPool();
    require(completed == 42, "Stopping a pool must drain accepted tasks before returning");
    pool.stopThreadPool();
    rejected = false;
    try { pool.enqueueTask([] {}); }
    catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "Stopped pools must reject new tasks");
  }
  std::atomic<int> destroyed_tasks{0};
  {
    ThreadPool pool(1);
    pool.enqueueTask([] { throw WorkerError{}; });
    pool.enqueueTask([&] { ++destroyed_tasks; });
  }
  require(destroyed_tasks == 1, "Destructor must drain accepted work without throwing task failures");
}

void test_executor_completion_delegation() {
  auto model = std::make_shared<TokenModel>();
  // The conservative base default is a no-op for CPU executors; calling it
  // explicitly must not invoke a CUDA API on CPU-only test hosts.
  model->BaseModel::synchronize();
  require(model->synchronize_calls == 0, "Base CPU completion must remain a no-op");
  InferenceEngine<float> engine(model, Device::CPU, 4);
  engine.generate_with_callback({1}, 2, 1.0f, 0.9f, 1, [](uint32_t) {});
  require(model->synchronize_calls == 1 && engine.context_size() == 1,
          "Fresh requests must delegate state completion to their own executor");
  model->fail_synchronize = true;
  bool caught = false;
  try { engine.reset(); }
  catch (const WorkerError&) { caught = true; }
  require(caught && model->synchronize_calls == 2 && model->last_cache->size() == 1,
          "Reset must preserve the executor's completion failure and existing context");
  model->fail_synchronize = false;
  model->synchronize();
  const auto require_invalid = [](auto operation) {
    bool rejected = false;
    try { operation(); }
    catch (const std::logic_error&) { rejected = true; }
    require(rejected, "A successful backend cleanup cannot revive a failed engine");
  };
  require_invalid([&] { engine.reset(); });
  require_invalid([&] { engine.context_size(); });
  require(model->synchronize_calls == 3 && model->last_cache->size() == 1,
          "Invalid engines must reject before clearing history or touching the executor");
  InferenceEngine<float> replacement(model, Device::CPU, 4);
  replacement.generate_with_callback({1}, 2, 1.0f, 0.9f, 1, [](uint32_t) {});
  replacement.reset();
  require(model->synchronize_calls == 5 && replacement.context_size() == 0,
          "Reset must clear history only after its executor completes");
}

}  // namespace

int main() {
  try {
    test_tokens_and_thread();
    test_callback_exception();
    test_long_request_failure_and_reuse();
    test_failed_callback_completion_invalidates_engine();
    test_decode_failure_and_reuse();
    test_prefill_exception();
    test_fresh_request_and_capacity();
    test_cpu_sampling_rejects_before_writes();
    test_failed_migration_invalidates_engine();
    test_thread_pool_failure_and_drain();
    test_executor_completion_delegation();
    std::cout << "Native runtime callback tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
