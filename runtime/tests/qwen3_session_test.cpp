#include "qwen3.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using BFloat16 = __nv_bfloat16;
using Weights = std::unordered_map<std::string, Tensor<BFloat16>>;
using Session = Qwen3Session<BFloat16>;
using Model = Qwen3Model<BFloat16>;
using Cache = KVCache<BFloat16>;

constexpr std::size_t kHidden = 128;
constexpr std::size_t kIntermediate = 256;
constexpr std::size_t kVocabulary = 64;
constexpr std::size_t kCapacity = 16;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void cuda_check(cudaError_t status) {
  if (status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(status));
  }
}

ModelConfig config() {
  return {{"vocab_size", kVocabulary},
          {"n_layers", 1},
          {"n_heads", 1},
          {"n_kv_heads", 1},
          {"hidden_size", kHidden},
          {"head_dim", kHidden},
          {"intermediate_size", kIntermediate},
          {"max_position_embeddings", kCapacity},
          {"bos_token_id", 1},
          {"eos_token_id", 2},
          {"rms_norm_eps", 1.0e-6},
          {"rope_theta", 10000}};
}

Tensor<BFloat16> norm(std::size_t width) {
  return Tensor<BFloat16>(
      std::vector<BFloat16>(width, __float2bfloat16(1.0f)), {width});
}

Tensor<BFloat16> matrix(std::size_t in, std::size_t out,
                       std::size_t seed, float diagonal) {
  // Match the checkpoint loader: logical [in, out], physical [out, in].
  std::vector<BFloat16> values(in * out);
  for (std::size_t row = 0; row < out; ++row) {
    for (std::size_t column = 0; column < in; ++column) {
      const int residue = static_cast<int>(
          (row * 37 + column * 19 + seed * 11) % 23) - 11;
      float value = (static_cast<float>(residue) + 0.25f) * 0.001f;
      if (column == row % in) value += diagonal;
      values[row * in + column] = __float2bfloat16(value);
    }
  }
  return Tensor<BFloat16>(std::move(values), {out, in}).transpose(0, 1);
}

Weights weights() {
  Weights result;
  std::vector<BFloat16> embeddings(kVocabulary * kHidden);
  for (std::size_t token = 0; token < kVocabulary; ++token) {
    for (std::size_t column = 0; column < kHidden; ++column) {
      const int residue = static_cast<int>(
          (token * 29 + column * 13 + token * column * 3) % 71) - 35;
      embeddings[token * kHidden + column] =
          __float2bfloat16((static_cast<float>(residue) + 0.5f) * 0.02f);
    }
  }
  result.emplace("token_embeddings.weight",
                 Tensor<BFloat16>(std::move(embeddings),
                                  {kVocabulary, kHidden}));
  result.emplace("rms_out_w", norm(kHidden));
  result.emplace("rms_att_w0", norm(kHidden));
  result.emplace("rms_ffn_w0", norm(kHidden));
  result.emplace("q_norm0", norm(kHidden));
  result.emplace("k_norm0", norm(kHidden));
  result.emplace("wq0", matrix(kHidden, kHidden, 1, 0.125f));
  result.emplace("wk0", matrix(kHidden, kHidden, 2, 0.125f));
  result.emplace("wv0", matrix(kHidden, kHidden, 3, 0.125f));
  result.emplace("wo0", matrix(kHidden, kHidden, 4, 0.125f));
  result.emplace("w_gate0", matrix(kHidden, kIntermediate, 5, 0.08f));
  result.emplace("w_up0", matrix(kHidden, kIntermediate, 6, 0.08f));
  result.emplace("w_down0", matrix(kIntermediate, kHidden, 7, 0.08f));
  result.emplace("lm_head", matrix(kHidden, kVocabulary, 8, 0.25f));
  return result;
}

Tensor<std::uint32_t> input(const std::vector<std::uint32_t>& values,
                            const std::vector<std::size_t>& shape = {}) {
  return Tensor<std::uint32_t>(std::vector<std::uint32_t>(values),
                               shape.empty()
                                   ? std::vector<std::size_t>{values.size()}
                                   : shape,
                               Device::CUDA);
}

void initialize_cache(Cache& cache) {
  // Initialize the entire capacity so accidental writes to future positions are
  // visible as well as changes to active K/V entries.
  for (std::size_t layer = 0; layer < cache.get_n_layers(); ++layer) {
    auto views = cache.get_layer_view(layer);
    std::vector<BFloat16> sentinel(views.first.numel(),
                                  __float2bfloat16(0.375f));
    cuda_check(cudaMemcpy(views.first.data_ptr(), sentinel.data(),
                          sentinel.size() * sizeof(BFloat16),
                          cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(views.second.data_ptr(), sentinel.data(),
                          sentinel.size() * sizeof(BFloat16),
                          cudaMemcpyHostToDevice));
  }
}

std::vector<BFloat16> download(const Tensor<BFloat16>& tensor) {
  if (tensor.device() == Device::CPU) {
    return std::vector<BFloat16>(tensor.data_ptr(),
                                 tensor.data_ptr() + tensor.numel());
  }
  std::vector<BFloat16> result(tensor.numel());
  if (!result.empty()) {
    cuda_check(cudaMemcpy(result.data(), tensor.data_ptr(),
                          result.size() * sizeof(BFloat16),
                          cudaMemcpyDeviceToHost));
  }
  return result;
}

struct Snapshot {
  std::vector<BFloat16> logits;
  std::vector<BFloat16> keys;
  std::vector<BFloat16> values;
};

Snapshot snapshot(Session& session, Cache& cache,
                  const Tensor<BFloat16>* logits = nullptr) {
  session.synchronize();
  Snapshot result;
  if (logits) result.logits = download(*logits);
  for (std::size_t layer = 0; layer < cache.get_n_layers(); ++layer) {
    auto views = cache.get_layer_view(layer);
    auto keys = download(views.first);
    auto values = download(views.second);
    result.keys.insert(result.keys.end(), keys.begin(), keys.end());
    result.values.insert(result.values.end(), values.begin(), values.end());
  }
  return result;
}

void equal_values(const std::vector<BFloat16>& expected,
                  const std::vector<BFloat16>& actual,
                  const std::string& label) {
  require(expected.size() == actual.size(), label + " extent differs");
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const float left = __bfloat162float(expected[index]);
    const float right = __bfloat162float(actual[index]);
    require(std::isfinite(left) && std::isfinite(right),
            label + " contains a non-finite value");
    // Identical execution modes on identical weights must be deterministic;
    // comparing every BF16 value catches aliasing that sampled tokens conceal.
    if (left != right) {
      throw std::runtime_error(label + " differs at " + std::to_string(index) +
                               ": " + std::to_string(left) + " vs " +
                               std::to_string(right));
    }
  }
}

void equal_snapshot(const Snapshot& expected, const Snapshot& actual,
                    const std::string& label) {
  equal_values(expected.logits, actual.logits, label + " logits");
  equal_values(expected.keys, actual.keys, label + " K cache");
  equal_values(expected.values, actual.values, label + " V cache");
}

Tensor<BFloat16> decode(Session& session,
                        const Tensor<std::uint32_t>& token, Cache& cache,
                        bool graph) {
  cache.resize(cache.size() + 1);
  return graph ? session.forward_for_graph_logits_only(&token, &cache)
               : session.forward_eager(&token, &cache);
}

struct History {
  std::vector<std::uint32_t> prompt;
  std::vector<std::uint32_t> continuation;
};

const History kA{{3, 7, 11}, {13, 19, 2}};
const History kB{{17, 23, 29, 31}, {37, 41, 5}};

std::vector<Snapshot> isolated(const Weights& source, const History& history,
                               bool graph) {
  auto model = std::make_shared<Model>(source, config());
  Session session(model, graph);
  Cache cache(1, kCapacity, kHidden, Device::CUDA);
  initialize_cache(cache);
  auto prompt = input(history.prompt);
  cache.resize(history.prompt.size());
  auto logits = session.prefill_eager(&prompt, &cache);
  std::vector<Snapshot> result{snapshot(session, cache, &logits)};
  for (std::uint32_t value : history.continuation) {
    auto token = input({value});
    logits = decode(session, token, cache, graph);
    result.push_back(snapshot(session, cache, &logits));
  }
  return result;
}

std::uint32_t last_row_argmax(const Snapshot& state) {
  require(!state.logits.empty() && state.logits.size() % kVocabulary == 0,
          "Argmax requires complete vocabulary rows");
  const auto first = state.logits.size() - kVocabulary;
  std::size_t selected = 0;
  for (std::size_t index = 1; index < kVocabulary; ++index) {
    if (__bfloat162float(state.logits[first + index]) >
        __bfloat162float(state.logits[first + selected])) {
      selected = index;
    }
  }
  return static_cast<std::uint32_t>(selected);
}

std::uint32_t download_token(const std::uint32_t* pointer) {
  require(pointer != nullptr, "Sampling returned a null token pointer");
  std::uint32_t token = 0;
  cuda_check(cudaMemcpy(&token, pointer, sizeof(token), cudaMemcpyDeviceToHost));
  return token;
}

void equal_cache(const Snapshot& expected, const Snapshot& actual,
                 const std::string& label) {
  equal_values(expected.keys, actual.keys, label + " K cache");
  equal_values(expected.values, actual.values, label + " V cache");
}

void equal_active_cache(const Snapshot& expected, const Snapshot& actual,
                        std::size_t tokens, const std::string& label) {
  const auto active = tokens * kHidden;
  require(expected.keys.size() >= active && actual.keys.size() >= active &&
              expected.values.size() >= active && actual.values.size() >= active,
          label + " active cache extent differs");
  equal_values(std::vector<BFloat16>(expected.keys.begin(), expected.keys.begin() + active),
                std::vector<BFloat16>(actual.keys.begin(), actual.keys.begin() + active),
                label + " active K cache");
  equal_values(std::vector<BFloat16>(expected.values.begin(), expected.values.begin() + active),
                std::vector<BFloat16>(actual.values.begin(), actual.values.begin() + active),
                label + " active V cache");
}

float near_values(const std::vector<BFloat16>& expected,
                  const std::vector<BFloat16>& actual,
                  const std::string& label) {
  require(expected.size() == actual.size(), label + " extent differs");
  float maximum_error = 0.0f;
  constexpr float absolute_tolerance = 0.015625f;
  constexpr float relative_tolerance = 0.01f;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const float left = __bfloat162float(expected[index]);
    const float right = __bfloat162float(actual[index]);
    require(std::isfinite(left) && std::isfinite(right),
            label + " contains a non-finite value");
    const float error = std::fabs(left - right);
    maximum_error = std::max(maximum_error, error);
    const float tolerance = absolute_tolerance +
        relative_tolerance * std::max(std::fabs(left), std::fabs(right));
    if (error > tolerance) {
      throw std::runtime_error(label + " exceeds BF16 tolerance at " +
                               std::to_string(index) + ": " +
                               std::to_string(left) + " vs " +
                               std::to_string(right));
    }
  }
  return maximum_error;
}

void cross_mode_test(const Weights& source) {
  const auto eager = isolated(source, kA, false);
  const auto graph = isolated(source, kA, true);
  require(eager.size() == graph.size(), "Eager/graph history length differs");
  for (std::size_t step = 0; step < eager.size(); ++step) {
    const auto prefix = "Eager/graph step " + std::to_string(step);
    const float logits_error = near_values(eager[step].logits, graph[step].logits,
                                          prefix + " logits");
    const float keys_error = near_values(eager[step].keys, graph[step].keys,
                                        prefix + " K cache");
    const float values_error = near_values(eager[step].values, graph[step].values,
                                          prefix + " V cache");
    require(last_row_argmax(eager[step]) == last_row_argmax(graph[step]),
            prefix + " greedy token differs");
    std::cout << prefix << ": max_abs_logits=" << logits_error
              << ", max_abs_K=" << keys_error
              << ", max_abs_V=" << values_error << '\n';
  }
}

void reset_test(const std::shared_ptr<const Model>& model,
                const std::vector<Snapshot>& expected, bool graph) {
  Session session(model, graph);
  Cache cache(1, kCapacity, kHidden, Device::CUDA);
  initialize_cache(cache);
  auto prompt = input(kA.prompt);
  for (int replay = 0; replay < 2; ++replay) {
    cache.clear();
    cache.resize(kA.prompt.size());
    auto logits = session.prefill_eager(&prompt, &cache);
    auto actual = snapshot(session, cache, &logits);
    equal_values(expected[0].logits, actual.logits, "Reset/replayed prefill logits");
    equal_active_cache(expected[0], actual, cache.size(), "Reset/replayed prefill");
    for (std::size_t step = 0; step < kA.continuation.size(); ++step) {
      auto token = input({kA.continuation[step]});
      logits = decode(session, token, cache, graph);
      actual = snapshot(session, cache, &logits);
      equal_values(expected[step + 1].logits, actual.logits,
                    "Reset/replayed decode logits");
      equal_active_cache(expected[step + 1], actual, cache.size(),
                           "Reset/replayed decode");
    }
    // clear() resets length, without erasing old bytes beyond the active range.
    // The second replay must ignore those stale entries and reset graph offsets.
  }
  std::cout << (graph ? "Graph" : "Eager")
            << " session reset: repeated prompt/decode matches original\n";
}

void destruction_test(const std::shared_ptr<const Model>& model,
                      const std::vector<Snapshot>& expected, bool graph) {
  Session a(model, graph);
  Cache cache_a(1, kCapacity, kHidden, Device::CUDA);
  initialize_cache(cache_a);
  auto prompt_a = input(kA.prompt);
  cache_a.resize(kA.prompt.size());
  auto logits_a = a.prefill_eager(&prompt_a, &cache_a);
  {
    Cache cache_b(1, kCapacity, kHidden, Device::CUDA);
    initialize_cache(cache_b);
    Session b(model, graph);
    auto prompt_b = input(kB.prompt);
    cache_b.resize(kB.prompt.size());
    b.prefill_eager(&prompt_b, &cache_b);
    auto token_a = input({kA.continuation[0]});
    auto token_b = input({kB.continuation[0]});
    logits_a = decode(a, token_a, cache_a, graph);
    decode(b, token_b, cache_b, graph);
    equal_snapshot(expected[1], snapshot(a, cache_a, &logits_a),
                    "A decode before B destruction");
  }
  // B has released its handle, stream, graph and workspaces. A must continue
  // using its private resources and the still-shared immutable model.
  equal_snapshot(expected[1], snapshot(a, cache_a, &logits_a),
                  "Held A output after B destruction");
  for (std::size_t step = 1; step < kA.continuation.size(); ++step) {
    auto token = input({kA.continuation[step]});
    logits_a = decode(a, token, cache_a, graph);
    equal_snapshot(expected[step + 1], snapshot(a, cache_a, &logits_a),
                    "A decode after B destruction");
  }
  std::cout << (graph ? "Graph" : "Eager")
            << " A continues correctly after B destruction\n";
}

void sampling_test(const std::shared_ptr<const Model>& model,
                   const std::vector<Snapshot>& expected_a,
                   const std::vector<Snapshot>& expected_b, bool graph) {
  Session a(model, graph);
  Session b(model, graph);
  ThreadPool thread_pool(1);
  Cache cache_a(1, kCapacity, kHidden, Device::CUDA);
  Cache cache_b(1, kCapacity, kHidden, Device::CUDA);
  initialize_cache(cache_a);
  initialize_cache(cache_b);

  // The legacy sampler consumes a valid RNG state even with top_k=1.
  CudaWorkspaceArena random_storage;
  random_storage.reserve(2 * sizeof(curandState));
  auto* random_a = random_storage.ptr_at<curandState>(0);
  auto* random_b = random_a + 1;
  op::UnifiedOperators<BFloat16> random_operators(Device::CUDA);
  random_operators.init_curand(random_a, 123, 0);
  random_operators.init_curand(random_b, 456, 0);
  cuda_check(cudaDeviceSynchronize());

  auto prompt_a = input(kA.prompt);
  auto prompt_b = input(kB.prompt);
  cache_a.resize(kA.prompt.size());
  auto* sampled_a = a.prefill(&prompt_a, thread_pool, &cache_a, 1, 1.0f,
                             1.0f, random_a);
  const auto expected_prefill_a = last_row_argmax(expected_a[0]);
  require(download_token(sampled_a) == expected_prefill_a,
          "Sampled A prefill differs from final-row logits argmax");
  const auto held_a_prefill = snapshot(a, cache_a);
  equal_cache(expected_a[0], held_a_prefill, "Sampled A prefill");
  cache_b.resize(kB.prompt.size());
  auto* sampled_b = b.prefill(&prompt_b, thread_pool, &cache_b, 1, 1.0f,
                             1.0f, random_b);
  require(sampled_a != sampled_b,
          "Sessions must own separate sampled-token storage");
  require(download_token(sampled_b) == last_row_argmax(expected_b[0]),
          "Sampled B prefill differs from final-row logits argmax");
  require(download_token(sampled_a) == expected_prefill_a,
          "B prefill modified held A sampled token");
  equal_cache(expected_b[0], snapshot(b, cache_b), "Sampled B prefill");
  equal_cache(held_a_prefill, snapshot(a, cache_a),
               "Held sampled A prefill after B");

  for (std::size_t step = 0; step < kA.continuation.size(); ++step) {
    // Use fixed history tokens to compare sampling with the all-logits run.
    auto token_a = input({kA.continuation[step]});
    auto token_b = input({kB.continuation[step]});
    cache_a.resize(cache_a.size() + 1);
    auto* next_a = a.forward(&token_a, thread_pool, &cache_a, 1, 1.0f,
                             1.0f, random_a);
    require(next_a == sampled_a, "A sampling storage changed during decode");
    const auto expected_token_a = last_row_argmax(expected_a[step + 1]);
    require(download_token(next_a) == expected_token_a,
            "Sampled A decode differs from logits argmax");
    const auto held_a = snapshot(a, cache_a);
    equal_cache(expected_a[step + 1], held_a, "Sampled A decode");
    cache_b.resize(cache_b.size() + 1);
    auto* next_b = b.forward(&token_b, thread_pool, &cache_b, 1, 1.0f,
                             1.0f, random_b);
    require(next_b == sampled_b && next_a != next_b,
            "Sampled-token storage must stay fixed and session-specific");
    require(download_token(next_b) == last_row_argmax(expected_b[step + 1]),
            "Sampled B decode differs from logits argmax");
    require(download_token(next_a) == expected_token_a,
            "B decode modified held A sampled token");
    equal_cache(expected_b[step + 1], snapshot(b, cache_b), "Sampled B decode");
    equal_cache(held_a, snapshot(a, cache_a), "Held sampled A decode after B");
  }
  std::cout << (graph ? "Graph" : "Eager")
            << " sampled sessions: greedy tokens match all-logits argmax\n";
}

void isolation_test(const Weights& source, bool graph) {
  const auto expected_a = isolated(source, kA, graph);
  const auto expected_b = isolated(source, kB, graph);
  require(expected_a.back().logits.size() == kVocabulary,
          "Decode must return every vocabulary logit");
  bool distinct = false;
  for (std::size_t index = 0; index < kVocabulary; ++index) {
    if (__bfloat162float(expected_a.back().logits[index]) !=
        __bfloat162float(expected_b.back().logits[index])) {
      distinct = true;
    }
  }
  require(distinct, "Synthetic histories must produce distinct logits");

  // The model must own uploaded weights after the loader's temporary map dies.
  auto model = std::make_shared<Model>(weights(), config());
  Session a(model, graph);
  Session b(model, graph);
  require(a.model().get() == model.get() && b.model().get() == model.get(),
          "Sessions must share the requested immutable model");
  require(a.stream() != nullptr && b.stream() != nullptr &&
              a.stream() != b.stream(),
          "Sessions must own separate execution streams");
  const auto a_workspace = a.decode_workspace_bytes();
  const auto b_workspace = b.decode_workspace_bytes();
  require(a_workspace > 0 && b_workspace > 0,
          "Decode workspaces must be allocated during session construction");

  Cache cache_a(1, kCapacity, kHidden, Device::CUDA);
  Cache cache_b(1, kCapacity, kHidden, Device::CUDA);
  initialize_cache(cache_a);
  initialize_cache(cache_b);
  auto prompt_a = input(kA.prompt);
  auto prompt_b = input(kB.prompt);
  cache_a.resize(kA.prompt.size());
  auto logits_a = a.prefill_eager(&prompt_a, &cache_a);
  const auto held_prefill = snapshot(a, cache_a, &logits_a);
  equal_snapshot(expected_a[0], held_prefill, "Interleaved A prefill");

  // Legacy generation changes a process-wide allocation phase after prefill.
  // That phase/reset must no longer invalidate this session's borrowed logits.
  GlobalCudaMemoryPool::set_prefill_phase(false);
  GlobalCudaMemoryPool::reset_prefill_buffer();
  cache_b.resize(kB.prompt.size());
  auto logits_b = b.prefill_eager(&prompt_b, &cache_b);
  equal_snapshot(expected_b[0], snapshot(b, cache_b, &logits_b),
                  "Interleaved B prefill");
  equal_snapshot(held_prefill, snapshot(a, cache_a, &logits_a),
                  "Held A prefill after B and legacy reset");

  const BFloat16* first_decode_a = nullptr;
  const BFloat16* first_decode_b = nullptr;
  for (std::size_t step = 0; step < kA.continuation.size(); ++step) {
    auto token_a = input({kA.continuation[step]});
    auto token_b = input({kB.continuation[step]});
    logits_a = decode(a, token_a, cache_a, graph);
    const auto held_a = snapshot(a, cache_a, &logits_a);
    // Public session calls complete GPU work before returning. This verifies
    // interleaved sessions; it makes no claim about overlapping their kernels.
    logits_b = decode(b, token_b, cache_b, graph);
    equal_snapshot(expected_a[step + 1], held_a,
                    "Interleaved A decode " + std::to_string(step));
    equal_snapshot(expected_b[step + 1], snapshot(b, cache_b, &logits_b),
                    "Interleaved B decode " + std::to_string(step));
    equal_snapshot(held_a, snapshot(a, cache_a, &logits_a),
                    "Held A decode after B " + std::to_string(step));
    if (step == 0) {
      first_decode_a = logits_a.data_ptr();
      first_decode_b = logits_b.data_ptr();
      require(first_decode_a != first_decode_b,
              "Session logits must not alias another session's workspace");
    } else {
      require(logits_a.data_ptr() == first_decode_a &&
                  logits_b.data_ptr() == first_decode_b,
              "Decode must reuse fixed session output storage");
    }
    require(a.decode_workspace_bytes() == a_workspace &&
                b.decode_workspace_bytes() == b_workspace,
            "Decode workspace capacity changed during generation");
  }
  std::cout << (graph ? "Graph" : "Eager")
            << " sessions: isolated/interleaved logits and full KV match\n";
  sampling_test(model, expected_a, expected_b, graph);
  reset_test(model, expected_a, graph);
  destruction_test(model, expected_a, graph);
}

void expect_rejected(Session& session, Cache& cache,
                     const std::function<void()>& operation,
                     const std::string& label,
                     const Tensor<BFloat16>* held_logits = nullptr,
                     Cache* extra_cache = nullptr) {
  const auto before = snapshot(session, cache, held_logits);
  Snapshot extra_before;
  if (extra_cache) extra_before = snapshot(session, *extra_cache);
  bool rejected = false;
  try {
    operation();
  } catch (const std::exception& error) {
    require(std::strlen(error.what()) > 0, label + " missing diagnostic");
    rejected = true;
  }
  require(rejected, label + " was accepted");
  equal_snapshot(before, snapshot(session, cache, held_logits),
                  label + " modified live session storage");
  if (extra_cache) {
    equal_snapshot(extra_before, snapshot(session, *extra_cache),
                    label + " modified invalid cache storage");
  }
}

void expect_preparation_rejected(const std::function<void()>& operation,
                                 const std::string& diagnostic,
                                 const std::string& label) {
  bool rejected = false;
  try {
    operation();
  } catch (const std::invalid_argument& error) {
    require(std::string(error.what()).find(diagnostic) != std::string::npos,
            label + " missing the expected diagnostic: " + error.what());
    rejected = true;
  }
  require(rejected, label + " was accepted");
}

void model_preparation_test() {
  auto loader_weights = weights();
  const auto& source_gate = loader_weights.at("w_gate0");
  const auto source_shape = source_gate.sizes();
  const auto source_strides = source_gate.strides();
  const std::vector<BFloat16> source_values(
      source_gate.data_ptr(), source_gate.data_ptr() + source_gate.numel());
  auto model = std::make_shared<Model>(loader_weights, config());
  const auto& frozen_gate = model->get_params().at("w_gate0");
  require(frozen_gate.device() == Device::CUDA &&
              frozen_gate.sizes() == source_shape &&
              frozen_gate.strides() == source_strides &&
              !frozen_gate.is_contiguous(),
          "Model preparation lost rectangular gate transpose metadata");
  equal_values(source_values, download(frozen_gate),
                "Prepared rectangular gate physical values");
  auto& mutable_gate = loader_weights.at("w_gate0");
  std::fill(mutable_gate.data_ptr(),
            mutable_gate.data_ptr() + mutable_gate.numel(),
            __float2bfloat16(0.0f));
  loader_weights.clear();
  equal_values(source_values, download(frozen_gate),
                "Frozen gate after loader modification and release");

  auto malformed = weights();
  malformed.at("w_gate0") = matrix(kHidden, kIntermediate / 2, 5, 0.08f);
  bool rejected = false;
  try {
    Model invalid_model(malformed, config());
  } catch (const std::invalid_argument& error) {
    require(std::string(error.what()).find("w_gate0") != std::string::npos,
            "Invalid model shape diagnostic must identify its weight");
    rejected = true;
  }
  require(rejected, "Invalid required gate matrix shape was accepted");

  auto strided_embeddings = weights();
  strided_embeddings.at("token_embeddings.weight") = Tensor<BFloat16>(
      std::vector<BFloat16>(kHidden * kVocabulary, __float2bfloat16(0.5f)),
      {kHidden, kVocabulary}).transpose(0, 1);
  expect_preparation_rejected(
      [&] { Model invalid(strided_embeddings, config()); },
      "contiguous: token_embeddings.weight", "Noncontiguous embeddings");
  auto unsupported_width = config();
  unsupported_width.at("hidden_size") = 130;
  expect_preparation_rejected(
      [&] { Model invalid(weights(), unsupported_width); },
      "embedding width must be divisible by 8", "Unsupported embedding width");
  std::cout << "Model preparation preserves owned transposed weights and validates shapes\n";
}

struct AwqFixture {
  Weights dense;
  Model::IntegerParameters packed, zeros;
  Weights scales;
  ModelConfig configuration;
};

AwqFixture awq_fixture(std::size_t padded_groups = 2) {
  AwqFixture fixture;
  fixture.dense = weights();
  fixture.dense.erase("w_gate0");
  fixture.configuration = config();
  fixture.configuration.emplace("group_size", 64);
  fixture.packed.emplace("w_gate0", Tensor<std::int32_t>(
      std::vector<std::int32_t>(kIntermediate * (kHidden / 8), 0x77777777),
      {kIntermediate, kHidden / 8}));
  fixture.zeros.emplace("w_gate0", Tensor<std::int32_t>(
      std::vector<std::int32_t>(kIntermediate, 0x77777777),
      {kIntermediate, 1}));
  fixture.scales.emplace("w_gate0", Tensor<BFloat16>(
      std::vector<BFloat16>(kIntermediate * padded_groups,
                            __float2bfloat16(0.02f)),
      {kIntermediate, padded_groups}));
  return fixture;
}

void awq_preparation_test() {
  // These cases validate checkpoint layout only. They do not execute AWQ math
  // or claim numerical parity for its kernels.
  for (std::size_t padded_groups : {std::size_t{2}, std::size_t{8}}) {
    const auto fixture = awq_fixture(padded_groups);
    Model model(fixture.dense, fixture.packed, fixture.scales, fixture.zeros,
                 fixture.configuration);
    require(model.verify_params() && model.config().quant_type == 1 &&
                model.layers()[0].gate.is_quantized(),
            "Valid AWQ gate was not prepared as quantized weights");
    require(model.get_qweight_params().at("w_gate0").sizes() ==
                std::vector<std::size_t>{kIntermediate, kHidden / 8} &&
                model.get_qzeros_params().at("w_gate0").sizes() ==
                std::vector<std::size_t>{kIntermediate, 1} &&
                model.get_scales_params().at("w_gate0").sizes() ==
                std::vector<std::size_t>{kIntermediate, padded_groups},
            "AWQ preparation changed output-major packed tensor layouts");
  }
  auto reject_fixture = [](const AwqFixture& fixture,
                            const std::string& diagnostic,
                            const std::string& label) {
    expect_preparation_rejected(
        [&] { Model invalid(fixture.dense, fixture.packed, fixture.scales,
                              fixture.zeros, fixture.configuration); },
        diagnostic, label);
  };
  {
    auto fixture = awq_fixture();
    fixture.packed.at("w_gate0") = Tensor<std::int32_t>(
        std::vector<std::int32_t>(kHidden * (kIntermediate / 8), 0x77777777),
        {kHidden, kIntermediate / 8});
    reject_fixture(fixture, "w_gate0.qweight", "Old AWQ input-major layout");
  }
  {
    auto fixture = awq_fixture();
    fixture.zeros.at("w_gate0") = Tensor<std::int32_t>(
        std::vector<std::int32_t>(2 * (kIntermediate / 8), 0x77777777),
        {2, kIntermediate / 8});
    reject_fixture(fixture, "w_gate0.qzeros", "Old AWQ group-major zeros layout");
  }
  {
    auto fixture = awq_fixture();
    fixture.scales.at("w_gate0") = Tensor<BFloat16>(
        std::vector<BFloat16>(2 * kIntermediate, __float2bfloat16(0.02f)),
        {2, kIntermediate});
    reject_fixture(fixture, "w_gate0.scales", "Old AWQ group-major scales layout");
  }
  {
    auto fixture = awq_fixture();
    fixture.configuration.at("group_size") = 96;
    reject_fixture(fixture, "divisible by group_size: w_gate0",
                    "Indivisible AWQ input groups");
  }
  {
    auto fixture = awq_fixture(1);
    reject_fixture(fixture, "w_gate0.scales", "Undersized AWQ scales");
  }
  {
    auto fixture = awq_fixture();
    fixture.packed.at("w_gate0") = Tensor<std::int32_t>(
        std::vector<std::int32_t>((kHidden / 8) * kIntermediate, 0x77777777),
        {kHidden / 8, kIntermediate}).transpose(0, 1);
    reject_fixture(fixture, "contiguous: w_gate0.qweight",
                    "Noncontiguous AWQ qweight");
  }
  {
    auto fixture = awq_fixture();
    fixture.scales.at("w_gate0") = Tensor<BFloat16>(
        std::vector<BFloat16>(2 * kIntermediate, __float2bfloat16(0.02f)),
        {2, kIntermediate}).transpose(0, 1);
    reject_fixture(fixture, "contiguous: w_gate0.scales",
                    "Noncontiguous AWQ scales");
  }
  {
    // Sixteen groups require two packed zero columns, making a transpose
    // noncontiguous even when its logical output-major shape is correct.
    auto fixture = awq_fixture(16);
    fixture.configuration.at("group_size") = 8;
    fixture.zeros.at("w_gate0") = Tensor<std::int32_t>(
        std::vector<std::int32_t>(2 * kIntermediate, 0x77777777),
        {2, kIntermediate}).transpose(0, 1);
    reject_fixture(fixture, "contiguous: w_gate0.qzeros",
                    "Noncontiguous AWQ qzeros");
  }
  std::cout << "AWQ preparation accepts output-major packing and validates groups/layouts\n";
}

void cache_roundtrip_test() {
  constexpr std::size_t layers = 2;
  constexpr std::size_t capacity = 3;
  constexpr std::size_t width = 7;
  constexpr std::size_t active = 2;
  constexpr std::size_t layer_elements = capacity * width;
  Cache cache(layers, capacity, width, Device::CPU, active);
  std::vector<BFloat16> expected_k(layers * layer_elements);
  std::vector<BFloat16> expected_v(layers * layer_elements);
  for (std::size_t index = 0; index < expected_k.size(); ++index) {
    expected_k[index] = __float2bfloat16((static_cast<float>(index) + 1) * 0.125f);
    expected_v[index] = __float2bfloat16(-(static_cast<float>(index) + 1) * 0.25f);
  }
  for (std::size_t layer = 0; layer < layers; ++layer) {
    auto views = cache.get_layer_view(layer);
    std::copy_n(expected_k.data() + layer * layer_elements,
                 layer_elements, views.first.data_ptr());
    std::copy_n(expected_v.data() + layer * layer_elements,
                 layer_elements, views.second.data_ptr());
  }

  auto verify = [&](Device device, const std::string& phase) {
    require(cache.device() == device && cache.size() == active &&
                cache.get_n_layers() == layers &&
                cache.get_max_seq_len() == capacity &&
                cache.get_head_dim() == width,
            phase + " cache geometry or active length changed");
    const auto base = cache.get_layer_view(0);
    for (std::size_t layer = 0; layer < layers; ++layer) {
      const auto views = cache.get_layer_view(layer);
      require(views.first.device() == device && views.second.device() == device &&
                  views.first.sizes() == std::vector<std::size_t>{capacity, width} &&
                  views.second.sizes() == std::vector<std::size_t>{capacity, width} &&
                  views.first.strides() == std::vector<std::size_t>{width, 1} &&
                  views.second.strides() == std::vector<std::size_t>{width, 1},
              phase + " layer view shape, stride or device changed");
      require(views.first.data_ptr() == base.first.data_ptr() + layer * layer_elements &&
                  views.second.data_ptr() == base.second.data_ptr() + layer * layer_elements,
              phase + " layer views no longer share contiguous backing");
      const auto first = layer * layer_elements;
      equal_values(std::vector<BFloat16>(expected_k.begin() + first,
                                        expected_k.begin() + first + layer_elements),
                    download(views.first), phase + " full layer K values");
      equal_values(std::vector<BFloat16>(expected_v.begin() + first,
                                        expected_v.begin() + first + layer_elements),
                    download(views.second), phase + " full layer V values");
      const auto live = cache.get_contiguous_tensor(layer);
      require(live.first.sizes() == std::vector<std::size_t>{active, width} &&
                  live.second.sizes() == std::vector<std::size_t>{active, width} &&
                  live.first.data_ptr() == views.first.data_ptr() &&
                  live.second.data_ptr() == views.second.data_ptr(),
              phase + " active backing view changed");
      for (std::size_t position = 0; position < capacity; ++position) {
        const auto& key = cache.k_cache(layer, position);
        const auto& value = cache.v_cache(layer, position);
        require(key.device() == device && value.device() == device &&
                    key.sizes() == std::vector<std::size_t>{1, 1, width} &&
                    value.sizes() == std::vector<std::size_t>{1, 1, width} &&
                    key.data_ptr() == views.first.data_ptr() + position * width &&
                    value.data_ptr() == views.second.data_ptr() + position * width,
                phase + " token slice shape, pointer or device changed");
        const auto offset = first + position * width;
        equal_values(std::vector<BFloat16>(expected_k.begin() + offset,
                                          expected_k.begin() + offset + width),
                      download(key), phase + " token slice K values");
        equal_values(std::vector<BFloat16>(expected_v.begin() + offset,
                                          expected_v.begin() + offset + width),
                      download(value), phase + " token slice V values");
      }
    }
  };

  verify(Device::CPU, "Initial CPU cache");
  for (int upload = 0; upload < 2; ++upload) {
    GlobalCudaMemoryPool::reset_prefill_buffer();
    GlobalCudaMemoryPool::set_prefill_phase(true);
    cache.cuda();
    GlobalCudaMemoryPool::set_prefill_phase(false);
    verify(Device::CUDA, "Uploaded CUDA cache");
    GlobalCudaMemoryPool::reset_prefill_buffer();
    {
      Tensor<BFloat16> overwrite_k({layers, capacity, width}, Device::CUDA, true);
      Tensor<BFloat16> overwrite_v({layers, capacity, width}, Device::CUDA, true);
      cuda_check(cudaMemset(overwrite_k.data_ptr(), 0,
                            overwrite_k.numel() * sizeof(BFloat16)));
      cuda_check(cudaMemset(overwrite_v.data_ptr(), 0,
                            overwrite_v.numel() * sizeof(BFloat16)));
      cuda_check(cudaDeviceSynchronize());
      verify(Device::CUDA, "CUDA cache after global prefill reset/reuse");
    }
    if (upload == 0) {
      cache.cpu();
      verify(Device::CPU, "Returned CPU cache");
    }
  }
  std::cout << "Cache CPU/CUDA/CPU/CUDA roundtrip preserves layers and slices\n";
}

void cache_lifetime_test() {
  auto model = std::make_shared<Model>(weights(), config());
  Session session(model, false);
  const bool arena_enabled = GlobalCudaMemoryPool::enable_prefill_mode(
      2 * 1024 * 1024, 16 * 1024 * 1024);
  {
    GlobalCudaMemoryPool::set_prefill_phase(true);
    Cache cache(1, kCapacity, kHidden, Device::CUDA);
    initialize_cache(cache);
    const auto views_before = cache.get_layer_view(0);
    const auto before = snapshot(session, cache);
    GlobalCudaMemoryPool::set_prefill_phase(false);
    GlobalCudaMemoryPool::reset_prefill_buffer();
    {
      // Reuse and overwrite the reset legacy arena. If cache storage came from
      // that arena, this destroys its sentinels instead of merely resetting a
      // bookkeeping counter while leaving the bytes temporarily unchanged.
      Tensor<BFloat16> overwrite_k({1, kCapacity, kHidden}, Device::CUDA, true);
      Tensor<BFloat16> overwrite_v({1, kCapacity, kHidden}, Device::CUDA, true);
      cuda_check(cudaMemset(overwrite_k.data_ptr(), 0,
                            overwrite_k.numel() * sizeof(BFloat16)));
      cuda_check(cudaMemset(overwrite_v.data_ptr(), 0,
                            overwrite_v.numel() * sizeof(BFloat16)));
      cuda_check(cudaDeviceSynchronize());
      const auto views_after = cache.get_layer_view(0);
      require(views_before.first.data_ptr() == views_after.first.data_ptr() &&
                  views_before.second.data_ptr() == views_after.second.data_ptr(),
              "Global prefill reset changed persistent cache backing");
      equal_snapshot(before, snapshot(session, cache),
                      "Cache created during global prefill after arena overwrite");
    }
  }
  cache_roundtrip_test();
  GlobalCudaMemoryPool::disable_prefill_mode();
  std::cout << "Persistent cache survives global prefill phase/reset"
            << (arena_enabled ? " and arena reuse\n" : " (legacy arena unavailable)\n");
}

void sampling_validation_test() {
  constexpr std::size_t vocabulary = 2048;
  constexpr std::uint32_t sentinel = 0xa5c39e71U;
  Tensor<BFloat16> logits(std::vector<BFloat16>(vocabulary,
                                              __float2bfloat16(0.25f)),
                          {1, vocabulary}, Device::CUDA);
  const auto original_logits = download(logits);
  CudaWorkspaceArena output_storage;
  output_storage.reserve(sizeof(std::uint32_t));
  auto* output = output_storage.ptr_at<std::uint32_t>(0);
  op::UnifiedOperators<BFloat16> operators(Device::CUDA);
  cuda_check(cudaGetLastError());
  for (std::size_t top_k : {std::size_t{0}, std::size_t{1025}}) {
    cuda_check(cudaMemcpy(output, &sentinel, sizeof(sentinel), cudaMemcpyHostToDevice));
    bool rejected = false;
    try {
      // The null RNG would be invalid for an actual sampling launch. Invalid
      // top_k must fail in the facade before touching device state or output.
      operators.sample_to_fixed(Tensor<BFloat16>(logits), output,
                                 1.0f, 1.0f, top_k, nullptr);
    } catch (const std::runtime_error& error) {
      require(std::string(error.what()).find("top_k") != std::string::npos,
              "Invalid sampling diagnostic must identify top_k");
      rejected = true;
    }
    require(rejected, "Invalid fixed-output sampling top_k was accepted");
    cuda_check(cudaPeekAtLastError());
    cuda_check(cudaDeviceSynchronize());
    require(download_token(output) == sentinel,
            "Rejected fixed-output sampling modified output storage");
    equal_values(original_logits, download(logits),
                  "Rejected fixed-output sampling modified logits");
  }
  std::cout << "Fixed-output sampling rejects top_k=0 and top_k>1024 before writes\n";
}

void validation_test(const Weights& source) {
  auto model = std::make_shared<Model>(source, config());
  Session session(model, false);
  Cache cache(1, kCapacity, kHidden, Device::CUDA);
  initialize_cache(cache);
  cache.resize(1);
  auto token = input({3});
  auto multi = input({3, 7});
  auto rank_two = input({3}, {1, 1});
  auto empty = token.slice({0}, {0});
  Tensor<std::uint32_t> cpu(std::vector<std::uint32_t>{3}, {1});

  expect_rejected(session, cache,
                  [&] { session.forward_eager(nullptr, &cache); },
                  "Null decode input");
  expect_rejected(session, cache,
                  [&] { session.forward_eager(&token, nullptr); },
                  "Null decode cache");
  expect_rejected(session, cache,
                  [&] { session.prefill_eager(nullptr, &cache); },
                  "Null prefill input");
  expect_rejected(session, cache,
                  [&] { session.prefill_eager(&token, nullptr); },
                  "Null prefill cache");
  expect_rejected(session, cache,
                  [&] { session.forward_eager(&cpu, &cache); },
                  "CPU input");
  expect_rejected(session, cache,
                  [&] { session.forward_eager(&multi, &cache); },
                  "Multiple decode tokens");
  expect_rejected(session, cache,
                  [&] { session.forward_eager(&rank_two, &cache); },
                  "Wrong input rank");
  expect_rejected(session, cache,
                  [&] { session.prefill_eager(&empty, &cache); },
                  "Empty prefill input");
  cache.clear();
  expect_rejected(session, cache,
                  [&] { session.forward_eager(&token, &cache); },
                  "Missing decode cache extent");
  cache.resize(1);

  Cache wrong_layers(2, kCapacity, kHidden, Device::CUDA, 1);
  Cache wrong_width(1, kCapacity, kHidden / 2, Device::CUDA, 1);
  Cache wrong_capacity(1, kCapacity + 1, kHidden, Device::CUDA, 1);
  initialize_cache(wrong_layers);
  initialize_cache(wrong_width);
  initialize_cache(wrong_capacity);
  for (auto item : std::vector<std::pair<Cache*, std::string>>{
           {&wrong_layers, "Wrong layer count"},
           {&wrong_width, "Wrong KV width"},
           {&wrong_capacity, "Cache capacity exceeds model context"}}) {
    expect_rejected(session, cache,
                    [&] { session.forward_eager(&token, item.first); },
                    item.second, nullptr, item.first);
  }
  Cache cpu_cache(1, kCapacity, kHidden, Device::CPU, 1);
  expect_rejected(session, cache,
                  [&] { session.forward_eager(&token, &cpu_cache); },
                  "CPU KV cache");
  expect_rejected(session, cache,
                  [&] { session.prefill_eager(&multi, &cache); },
                  "Prefill exceeds caller-provided cache extent");

  // Rejected calls must not bind a cache or poison a later valid execution.
  auto prompt = input(kA.prompt);
  cache.resize(kA.prompt.size());
  auto logits = session.prefill_eager(&prompt, &cache);
  const auto baseline = isolated(source, kA, false);
  equal_snapshot(baseline[0], snapshot(session, cache, &logits),
                  "Valid call after input validation failures");
  auto valid_decode_token = input({kA.continuation.front()});
  logits = decode(session, valid_decode_token, cache, false);
  equal_snapshot(baseline[1], snapshot(session, cache, &logits),
                  "Valid decode after input validation failures");
  expect_rejected(session, cache,
                  [&] { session.forward_eager(nullptr, &cache); },
                  "Null input with live decode logits", &logits);
  expect_rejected(session, cache,
                  [&] { session.forward_eager(&multi, &cache); },
                  "Wrong shape with live decode logits", &logits);
  auto invalid_token = input({static_cast<std::uint32_t>(kVocabulary)});
  expect_rejected(session, cache,
                  [&] { session.forward_eager(&invalid_token, &cache); },
                  "Out-of-vocabulary decode token", &logits);
  auto invalid_prompt = input({3, static_cast<std::uint32_t>(kVocabulary), 11});
  expect_rejected(session, cache,
                  [&] { session.prefill_eager(&invalid_prompt, &cache); },
                  "Out-of-vocabulary prefill token", &logits);
  auto token_pairs = input({3, 17, 7, 23, 11, 29}, {3, 2});
  auto noncontiguous_prompt = token_pairs.slice({0, 0}, {3, 1}).squeeze(1);
  require(noncontiguous_prompt.sizes() == std::vector<std::size_t>{3} &&
              !noncontiguous_prompt.is_contiguous(),
          "Strided token fixture must be noncontiguous with valid rank/extent");
  expect_rejected(session, cache,
                  [&] { session.prefill_eager(&noncontiguous_prompt, &cache); },
                  "Noncontiguous prefill tokens", &logits);
  Cache other_cache(1, kCapacity, kHidden, Device::CUDA, 1);
  initialize_cache(other_cache);
  expect_rejected(session, cache,
                  [&] { session.forward_eager(&token, &other_cache); },
                  "Switching bound cache", &logits, &other_cache);

  // This legacy cache property is public. A session must detect its mutation
  // even though the cache object and backing pointers are still identical.
  const auto before = snapshot(session, cache, &logits);
  cache.max_seq_len_ = kCapacity - 1;
  bool rejected = false;
  try {
    session.forward_eager(&token, &cache);
  } catch (const std::exception&) {
    rejected = true;
  }
  cache.max_seq_len_ = kCapacity;
  require(rejected, "Changing bound cache capacity was accepted");
  equal_snapshot(before, snapshot(session, cache, &logits),
                  "Changing bound cache capacity modified storage");
  std::cout << "Invalid inputs and cache changes reject before writes\n";
}

}  // namespace

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::cout << "Qwen3 session test skipped: no CUDA device\n";
    return 77;
  }
  try {
    GlobalCudaMemoryPool::set_prefill_phase(false);
    const auto source = weights();
    model_preparation_test();
    awq_preparation_test();
    cache_lifetime_test();
    sampling_validation_test();
    isolation_test(source, false);
    validation_test(source);
    isolation_test(source, true);
    cross_mode_test(source);
    cuda_check(cudaDeviceSynchronize());
    std::cout << "Qwen3 shared-model session regression passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Qwen3 session regression failed: " << error.what() << '\n';
    return 1;
  }
}
