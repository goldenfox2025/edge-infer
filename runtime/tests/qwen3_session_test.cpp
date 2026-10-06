#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "allocation_probe.hpp"
#include "operators/cuda/execution.hpp"
#include "qwen3.hpp"
#include "tensor_view_adapter.hpp"
#include "test_cuda_rng.hpp"

#ifdef EDGE_INFER_TEST_CUDA_WRAPPING
namespace test_session_failure {
thread_local cudaStream_t stream = nullptr;
thread_local cudaEvent_t submitted = nullptr;
thread_local bool fail_gemm = false, gemm_failed = false, fail_wait = false;
thread_local size_t completion_attempts = 0, actual_completions = 0;
thread_local size_t gemms_until_failure = 0;
}  // namespace test_session_failure

extern "C" cublasStatus_t __real_cublasGemmEx(
    cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
    const void*, const void*, cudaDataType_t, int, const void*, cudaDataType_t, int,
    const void*, void*, cudaDataType_t, int, cublasComputeType_t, cublasGemmAlgo_t);
extern "C" cublasStatus_t __wrap_cublasGemmEx(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
    const void* alpha, const void* a, cudaDataType_t a_type, int lda,
    const void* b, cudaDataType_t b_type, int ldb, const void* beta,
    void* c, cudaDataType_t c_type, int ldc, cublasComputeType_t compute, cublasGemmAlgo_t algorithm) {
  cudaStream_t stream = nullptr;
  if (test_session_failure::fail_gemm &&
      cublasGetStream(handle, &stream) == CUBLAS_STATUS_SUCCESS &&
      stream == test_session_failure::stream && --test_session_failure::gemms_until_failure == 0) {
    test_session_failure::fail_gemm = false;
    test_session_failure::gemm_failed = true;
    // Copy, normalization and KV writes were queued before this rejected GEMM.
    // An event marks their completion without causing a real CUDA fault.
    if (cudaEventRecord(test_session_failure::submitted, stream) != cudaSuccess)
      return CUBLAS_STATUS_INTERNAL_ERROR;
    return CUBLAS_STATUS_EXECUTION_FAILED;
  }
  return __real_cublasGemmEx(handle, transa, transb, m, n, k, alpha, a, a_type, lda,
                           b, b_type, ldb, beta, c, c_type, ldc, compute, algorithm);
}
extern "C" cudaError_t __real_cudaStreamSynchronize(cudaStream_t);
extern "C" cudaError_t __wrap_cudaStreamSynchronize(cudaStream_t stream) {
  if (test_session_failure::gemm_failed && stream == test_session_failure::stream) {
    ++test_session_failure::completion_attempts;
    if (test_session_failure::fail_wait) return cudaErrorUnknown;
    const auto status = __real_cudaStreamSynchronize(stream);
    if (status == cudaSuccess) ++test_session_failure::actual_completions;
    return status;
  }
  return __real_cudaStreamSynchronize(stream);
}
#endif

namespace {

using BFloat16 = __nv_bfloat16;
using Weights = std::unordered_map<std::string, Tensor<BFloat16>>;
using Session = Qwen3Session<BFloat16>;
using Model = Qwen3Model<BFloat16>;
using Cache = KVCache<BFloat16>;
using Logits = TensorView<BFloat16, 2>;

template <typename T, typename = void>
struct HasPublicMutableCapacity : std::false_type {};
template <typename T>
struct HasPublicMutableCapacity<T, std::void_t<decltype(std::declval<T&>().max_seq_len_)>>
    : std::true_type {};
static_assert(!HasPublicMutableCapacity<Cache>::value,
              "KV capacity must not expose a mutable public storage limit");
static_assert(!std::is_reference_v<decltype(std::declval<Cache&>().get_max_seq_len())>);

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
  return Tensor<BFloat16>(std::vector<BFloat16>(width, __float2bfloat16(1.0f)), {width});
}

Tensor<BFloat16> matrix(std::size_t in, std::size_t out, std::size_t seed, float diagonal) {
  // Match the checkpoint loader: logical [in, out], physical [out, in].
  std::vector<BFloat16> values(in * out);
  for (std::size_t row = 0; row < out; ++row) {
    for (std::size_t column = 0; column < in; ++column) {
      const int residue = static_cast<int>((row * 37 + column * 19 + seed * 11) % 23) - 11;
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
      const int residue =
          static_cast<int>((token * 29 + column * 13 + token * column * 3) % 71) - 35;
      embeddings[token * kHidden + column] =
          __float2bfloat16((static_cast<float>(residue) + 0.5f) * 0.02f);
    }
  }
  result.emplace("token_embeddings.weight",
                 Tensor<BFloat16>(std::move(embeddings), {kVocabulary, kHidden}));
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
                               shape.empty() ? std::vector<std::size_t>{values.size()} : shape,
                               Device::CUDA);
}

void initialize_cache(Cache& cache) {
  // Initialize the entire capacity so accidental writes to future positions are
  // visible as well as changes to active K/V entries.
  for (std::size_t layer = 0; layer < cache.get_n_layers(); ++layer) {
    auto views = cache.get_layer_view(layer);
    std::vector<BFloat16> sentinel(views.first.numel(), __float2bfloat16(0.375f));
    cuda_check(cudaMemcpy(views.first.data_ptr(), sentinel.data(),
                          sentinel.size() * sizeof(BFloat16), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(views.second.data_ptr(), sentinel.data(),
                          sentinel.size() * sizeof(BFloat16), cudaMemcpyHostToDevice));
  }
}

std::vector<BFloat16> download(const Tensor<BFloat16>& tensor) {
  if (tensor.device() == Device::CPU) {
    return std::vector<BFloat16>(tensor.data_ptr(), tensor.data_ptr() + tensor.numel());
  }
  std::vector<BFloat16> result(tensor.numel());
  if (!result.empty()) {
    cuda_check(cudaMemcpy(result.data(), tensor.data_ptr(), result.size() * sizeof(BFloat16),
                          cudaMemcpyDeviceToHost));
  }
  return result;
}

template <typename T, std::size_t Rank>
std::vector<std::remove_const_t<T>> download(TensorView<T, Rank> view) {
  require(view.is_contiguous(), "Downloaded GPU view must be contiguous");
  std::vector<std::remove_const_t<T>> result(view.numel());
  if (!result.empty()) {
    cuda_check(cudaMemcpy(result.data(), view.data_ptr(),
                          result.size() * sizeof(std::remove_const_t<T>), cudaMemcpyDeviceToHost));
  }
  return result;
}

struct Snapshot {
  std::vector<BFloat16> logits;
  std::vector<BFloat16> keys;
  std::vector<BFloat16> values;
};

Snapshot snapshot(Session& session, Cache& cache, const Logits* logits = nullptr) {
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

void equal_values(const std::vector<BFloat16>& expected, const std::vector<BFloat16>& actual,
                  const std::string& label) {
  require(expected.size() == actual.size(), label + " extent differs");
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const float left = __bfloat162float(expected[index]);
    const float right = __bfloat162float(actual[index]);
    require(std::isfinite(left) && std::isfinite(right), label + " contains a non-finite value");
    // Identical execution modes on identical weights must be deterministic;
    // comparing every BF16 value catches aliasing that sampled tokens conceal.
    if (left != right) {
      throw std::runtime_error(label + " differs at " + std::to_string(index) + ": " +
                               std::to_string(left) + " vs " + std::to_string(right));
    }
  }
}

void equal_snapshot(const Snapshot& expected, const Snapshot& actual, const std::string& label) {
  equal_values(expected.logits, actual.logits, label + " logits");
  equal_values(expected.keys, actual.keys, label + " K cache");
  equal_values(expected.values, actual.values, label + " V cache");
}

Logits decode(Session& session, const Tensor<std::uint32_t>& token, Cache& cache, bool graph) {
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

std::vector<Snapshot> isolated(const Weights& source, const History& history, bool graph) {
  auto model = std::make_shared<Model>(source, config());
  Cache cache(1, kCapacity, kHidden, Device::CUDA);
  Session session(model, graph);
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

void equal_cache(const Snapshot& expected, const Snapshot& actual, const std::string& label) {
  equal_values(expected.keys, actual.keys, label + " K cache");
  equal_values(expected.values, actual.values, label + " V cache");
}

void equal_active_cache(const Snapshot& expected, const Snapshot& actual, std::size_t tokens,
                        const std::string& label) {
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

float near_values(const std::vector<BFloat16>& expected, const std::vector<BFloat16>& actual,
                  const std::string& label) {
  require(expected.size() == actual.size(), label + " extent differs");
  float maximum_error = 0.0f;
  constexpr float absolute_tolerance = 0.015625f;
  constexpr float relative_tolerance = 0.01f;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const float left = __bfloat162float(expected[index]);
    const float right = __bfloat162float(actual[index]);
    require(std::isfinite(left) && std::isfinite(right), label + " contains a non-finite value");
    const float error = std::fabs(left - right);
    maximum_error = std::max(maximum_error, error);
    const float tolerance =
        absolute_tolerance + relative_tolerance * std::max(std::fabs(left), std::fabs(right));
    if (error > tolerance) {
      throw std::runtime_error(label + " exceeds BF16 tolerance at " + std::to_string(index) +
                               ": " + std::to_string(left) + " vs " + std::to_string(right));
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
    const float logits_error =
        near_values(eager[step].logits, graph[step].logits, prefix + " logits");
    const float keys_error = near_values(eager[step].keys, graph[step].keys, prefix + " K cache");
    const float values_error =
        near_values(eager[step].values, graph[step].values, prefix + " V cache");
    require(last_row_argmax(eager[step]) == last_row_argmax(graph[step]),
            prefix + " greedy token differs");
    std::cout << prefix << ": max_abs_logits=" << logits_error << ", max_abs_K=" << keys_error
              << ", max_abs_V=" << values_error << '\n';
  }
}

void reset_test(const std::shared_ptr<const Model>& model, const std::vector<Snapshot>& expected,
                bool graph) {
  Cache cache(1, kCapacity, kHidden, Device::CUDA);
  Session session(model, graph);
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
      equal_values(expected[step + 1].logits, actual.logits, "Reset/replayed decode logits");
      equal_active_cache(expected[step + 1], actual, cache.size(), "Reset/replayed decode");
    }
    // clear() resets length, without erasing old bytes beyond the active range.
    // The second replay must ignore those stale entries and reset graph offsets.
  }
  std::cout << (graph ? "Graph" : "Eager")
            << " session reset: repeated prompt/decode matches original\n";
}

void destruction_test(const std::shared_ptr<const Model>& model,
                      const std::vector<Snapshot>& expected, bool graph) {
  Cache cache_a(1, kCapacity, kHidden, Device::CUDA);
  Session a(model, graph);
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
    equal_snapshot(expected[1], snapshot(a, cache_a, &logits_a), "A decode before B destruction");
  }
  // B has released its handle, stream, graph and workspaces. A must continue
  // using its private resources and the still-shared immutable model.
  equal_snapshot(expected[1], snapshot(a, cache_a, &logits_a), "Held A output after B destruction");
  for (std::size_t step = 1; step < kA.continuation.size(); ++step) {
    auto token = input({kA.continuation[step]});
    logits_a = decode(a, token, cache_a, graph);
    equal_snapshot(expected[step + 1], snapshot(a, cache_a, &logits_a),
                   "A decode after B destruction");
  }
  std::cout << (graph ? "Graph" : "Eager") << " A continues correctly after B destruction\n";
}

void expect_managed_rejected(Session& session, const std::function<void()>& operation,
                             const std::string& label, const Logits* held_logits = nullptr) {
  session.synchronize();
  const auto before_size = session.context_size();
  const auto before_capacity = session.context_capacity();
  const auto* before_model = session.model().get();
  const auto before_logits = held_logits ? download(*held_logits) : std::vector<BFloat16>{};
  bool rejected = false;
  try {
    operation();
  } catch (const std::exception& error) {
    require(std::strlen(error.what()) > 0, label + " missing diagnostic");
    rejected = true;
  }
  require(rejected, label + " was accepted");
  session.synchronize();
  require(session.context_size() == before_size && session.context_capacity() == before_capacity &&
              session.model().get() == before_model,
          label + " changed managed history or model ownership");
  if (held_logits) {
    equal_values(before_logits, download(*held_logits), label + " modified held managed logits");
  }
}

void allocation_probe_self_test() {
  {
    test_alloc::Scope probe;
    void* pointer = ::operator new(64);
    ::operator delete(pointer);
    const auto count = probe.finish();
    require(count.host_allocations == 1 && count.host_frees == 1,
            "Scoped allocation probe missed C++ allocation/free");
  }
#ifdef EDGE_INFER_TEST_CUDA_WRAPPING
  void* pointer = nullptr;
  test_alloc::Scope probe;
  const auto allocation_status = cudaMalloc(&pointer, 256);
  const auto release_status = cudaFree(pointer);
  const auto count = probe.finish();
  cuda_check(allocation_status);
  cuda_check(release_status);
  require(count.device_allocations == 1 && count.device_frees == 1,
          "Scoped CUDA wrapper missed native allocation/free");
#endif
}

void allocation_test(const std::shared_ptr<const Model>& model,
                     const std::vector<Snapshot>& expected_a,
                     const std::vector<Snapshot>& expected_b, bool graph) {
  auto a = Session::create(model, kCapacity, graph);
  auto b = Session::create(model, kCapacity, graph);
  auto prompt_a = input(kA.prompt);
  auto prompt_b = input(kB.prompt);
  a->prefill(borrow_tensor_view<1>(prompt_a).as_const());
  b->prefill(borrow_tensor_view<1>(prompt_b).as_const());
  std::array<Tensor<std::uint32_t>, 3> tokens_a;
  std::array<Tensor<std::uint32_t>, 3> tokens_b;
  std::array<TensorView<const std::uint32_t, 1>, 3> views_a;
  std::array<TensorView<const std::uint32_t, 1>, 3> views_b;
  for (std::size_t step = 0; step < 3; ++step) {
    tokens_a[step] = input({kA.continuation[step]});
    tokens_b[step] = input({kB.continuation[step]});
    views_a[step] = borrow_tensor_view<1>(tokens_a[step]).as_const();
    views_b[step] = borrow_tensor_view<1>(tokens_b[step]).as_const();
  }
  // Warm library launch paths and, for graph mode, capture outside the scope.
  auto logits_a = a->decode(views_a[0]);
  auto logits_b = b->decode(views_b[0]);
  equal_values(expected_a[1].logits, download(logits_a), "Allocation warmup A logits");
  equal_values(expected_b[1].logits, download(logits_b), "Allocation warmup B logits");
  const auto* output_a = logits_a.data_ptr();
  const auto* output_b = logits_b.data_ptr();
  const auto capacity_a = a->decode_workspace_bytes();
  const auto capacity_b = b->decode_workspace_bytes();
  test_alloc::Counts total;
  for (std::size_t step = 1; step < 3; ++step) {
    test_alloc::Scope probe_a;
    logits_a = a->decode(views_a[step]);
    const auto count_a = probe_a.finish();
    test_alloc::Scope probe_b;
    logits_b = step == 2 ? b->decode(kB.continuation[step]) : b->decode(views_b[step]);
    const auto count_b = probe_b.finish();
    for (const auto& count : {count_a, count_b}) {
      total.host_allocations += count.host_allocations;
      total.host_frees += count.host_frees;
      total.device_allocations += count.device_allocations;
      total.device_frees += count.device_frees;
    }
    require(logits_a.data_ptr() == output_a && logits_b.data_ptr() == output_b &&
                output_a != output_b && a->decode_workspace_bytes() == capacity_a &&
                b->decode_workspace_bytes() == capacity_b,
            "Steady decode changed private output/workspace storage");
    equal_values(expected_a[step + 1].logits, download(logits_a), "Allocation-measured A logits");
    equal_values(expected_b[step + 1].logits, download(logits_b), "Allocation-measured B logits");
  }
  std::cout << (graph ? "Graph" : "Eager")
            << " steady decode: C++ allocations=" << total.host_allocations
            << ", C++ frees=" << total.host_frees;
#ifdef EDGE_INFER_TEST_CUDA_WRAPPING
  std::cout << ", native CUDA allocations=" << total.device_allocations
            << ", native CUDA frees=" << total.device_frees;
  require(total.device_allocations == 0 && total.device_frees == 0,
          "Steady decode allocated/freed native device storage");
#else
  std::cout << ", CUDA allocation tracing unavailable in this build";
#endif
  std::cout << '\n';
  require(total.host_allocations == 0 && total.host_frees == 0,
          "Steady decode allocated/freed C++ heap storage");
}

void managed_test(const std::shared_ptr<const Model>& model,
                  const std::vector<Snapshot>& expected_a, const std::vector<Snapshot>& expected_b,
                  bool graph) {
  constexpr std::size_t parent_capacity = 6;
  constexpr std::size_t child_capacity = 7;
  auto host_session = Session::create(model, parent_capacity, graph);
  auto host_logits = host_session->prefill(kA.prompt);
  equal_values(expected_a[0].logits, download(host_logits),
               "Host upload and external CUDA-view prefill parity");
  expect_managed_rejected(
      *host_session, [&] { host_session->decode(static_cast<std::uint32_t>(kVocabulary)); },
      "Scalar host decode OOV before upload", &host_logits);
  for (std::size_t step = 0; step < kA.continuation.size(); ++step) {
    host_logits = host_session->decode(kA.continuation[step]);
    equal_values(expected_a[step + 1].logits, download(host_logits),
                 "Scalar host and external CUDA-view decode parity");
    require(host_session->context_size() == kA.prompt.size() + step + 1,
            "Scalar host decode history growth differs");
  }
  auto parent = Session::create(model, parent_capacity, graph);
  require(parent->model().get() == model.get() && parent->context_size() == 0 &&
              parent->context_capacity() == parent_capacity,
          "Managed factory must retain the shared model and start empty");
  auto token = input({kA.continuation[0]});
  expect_managed_rejected(*parent, [&] { parent->decode(token); }, "Managed decode before prefill");
  auto prompt_a = input(kA.prompt);
  auto logits_a = parent->prefill(prompt_a);
  equal_values(expected_a[0].logits, download(logits_a), "Managed parent prefill logits");
  require(parent->context_size() == kA.prompt.size(),
          "Managed prefill must record its prompt length");

  for (std::size_t invalid_capacity : {std::size_t{0}, kCapacity + 1}) {
    expect_managed_rejected(
        *parent, [&] { auto invalid = Session::create(model, invalid_capacity, graph); },
        "Invalid managed factory capacity", &logits_a);
    expect_managed_rejected(
        *parent, [&] { auto invalid = parent->new_session(invalid_capacity, graph); },
        "Invalid child capacity", &logits_a);
  }
  auto bad_token = input({static_cast<std::uint32_t>(kVocabulary)});
  auto bad_decode_shape = input({3, 7});
  auto bad_prompt = input({3, static_cast<std::uint32_t>(kVocabulary)});
  auto bad_prompt_rank = input({3}, {1, 1});
  auto oversized_prompt = input({3, 7, 11, 13, 19, 2, 5});
  expect_managed_rejected(
      *parent, [&] { parent->decode(bad_token); }, "Managed invalid decode token", &logits_a);
  expect_managed_rejected(
      *parent, [&] { parent->decode(bad_decode_shape); }, "Managed invalid decode shape",
      &logits_a);
  expect_managed_rejected(
      *parent, [&] { parent->prefill(bad_prompt); }, "Managed invalid prefill token", &logits_a);
  expect_managed_rejected(
      *parent, [&] { parent->prefill(bad_prompt_rank); }, "Managed invalid prefill rank",
      &logits_a);
  expect_managed_rejected(
      *parent, [&] { parent->prefill(oversized_prompt); }, "Managed oversized prompt", &logits_a);

  // A child receives private empty history even though its parent has a prefix.
  auto child = parent->new_session(child_capacity, graph);
  require(child->model().get() == model.get() && child->context_size() == 0 &&
              child->context_capacity() == child_capacity &&
              parent->context_size() == kA.prompt.size(),
          "Managed child must share weights without copying its parent's prefix");
  expect_managed_rejected(
      *child, [&] { child->decode(token); }, "Fresh child decode before prefill");
  auto prompt_b = input(kB.prompt);
  const auto held_prefill_a = download(logits_a);
  auto logits_b = child->prefill(prompt_b);
  equal_values(expected_b[0].logits, download(logits_b), "Managed child prefill logits");
  equal_values(held_prefill_a, download(logits_a),
               "Managed child prefill changed held parent logits");
  require(child->context_size() == kB.prompt.size(),
          "Managed child prefill must record its own prompt length");

  const BFloat16* fixed_parent_output = nullptr;
  const BFloat16* fixed_child_output = nullptr;
  for (std::size_t step = 0; step < 2; ++step) {
    auto token_a = input({kA.continuation[step]});
    auto token_b = input({kB.continuation[step]});
    logits_a = parent->decode(token_a);
    equal_values(expected_a[step + 1].logits, download(logits_a), "Managed parent decode logits");
    const auto held_a = download(logits_a);
    logits_b = child->decode(token_b);
    equal_values(expected_b[step + 1].logits, download(logits_b), "Managed child decode logits");
    equal_values(held_a, download(logits_a), "Managed child decode changed held parent logits");
    require(parent->context_size() == kA.prompt.size() + step + 1 &&
                child->context_size() == kB.prompt.size() + step + 1,
            "Managed decode must advance only its own history");
    if (step == 0) {
      fixed_parent_output = logits_a.data_ptr();
      fixed_child_output = logits_b.data_ptr();
      require(fixed_parent_output != fixed_child_output,
              "Managed sessions must own separate fixed decode outputs");
    } else {
      require(
          logits_a.data_ptr() == fixed_parent_output && logits_b.data_ptr() == fixed_child_output,
          "Managed decode must reuse fixed session output storage");
    }
  }
  const auto held_parent = download(logits_a);
  auto last_child_token = input({kB.continuation[2]});
  logits_b = child->decode(last_child_token);
  equal_values(expected_b[3].logits, download(logits_b), "Managed child final decode logits");
  require(child->context_size() == child_capacity,
          "Managed child must fill its requested capacity");
  expect_managed_rejected(
      *child, [&] { child->decode(token); }, "Managed child full-capacity decode", &logits_b);
  child.reset();
  require(parent->context_size() == kA.prompt.size() + 2 &&
              parent->context_capacity() == parent_capacity,
          "Child destruction changed parent history/capacity");
  equal_values(held_parent, download(logits_a),
               "Child execution/destruction changed parent output");
  auto last_parent_token = input({kA.continuation[2]});
  logits_a = parent->decode(last_parent_token);
  equal_values(expected_a[3].logits, download(logits_a),
               "Managed parent decode after child destruction");
  require(logits_a.data_ptr() == fixed_parent_output && parent->context_size() == parent_capacity,
          "Managed parent final decode must reuse storage and fill capacity");
  expect_managed_rejected(
      *parent, [&] { parent->decode(token); }, "Managed parent full-capacity decode", &logits_a);

  // Re-prefill replaces an existing history; explicit reset also starts empty.
  logits_a = parent->prefill(prompt_a);
  equal_values(expected_a[0].logits, download(logits_a),
               "Managed fresh prefill over existing history");
  require(parent->context_size() == kA.prompt.size(),
          "Managed prefill appended instead of replacing history");
  parent->reset();
  require(parent->context_size() == 0 && parent->context_capacity() == parent_capacity,
          "Managed reset must clear history while retaining capacity");
  expect_managed_rejected(
      *parent, [&] { parent->decode(token); }, "Managed decode after reset without prefill");
  logits_a = parent->prefill(prompt_a);
  equal_values(expected_a[0].logits, download(logits_a), "Managed reset/replayed prefill logits");
  for (std::size_t step = 0; step < kA.continuation.size(); ++step) {
    auto next = input({kA.continuation[step]});
    logits_a = parent->decode(next);
    equal_values(expected_a[step + 1].logits, download(logits_a),
                 "Managed reset/replayed decode logits");
    require(parent->context_size() == kA.prompt.size() + step + 1,
            "Managed replay context length differs");
  }
  std::cout << (graph ? "Graph" : "Eager")
            << " managed sessions: owned history, children and replay match baseline\n";
}

void sampling_test(const std::shared_ptr<const Model>& model,
                   const std::vector<Snapshot>& expected_a, const std::vector<Snapshot>& expected_b,
                   bool graph) {
  Cache cache_a(1, kCapacity, kHidden, Device::CUDA);
  Cache cache_b(1, kCapacity, kHidden, Device::CUDA);
  Session a(model, graph);
  Session b(model, graph);
  ThreadPool thread_pool(1);
  initialize_cache(cache_a);
  initialize_cache(cache_b);

  // Own RNG storage explicitly; the test never uses the compatibility facade.
  CudaWorkspaceArena random_storage;
  random_storage.reserve(2 * sizeof(curandState));
  auto* random_a = random_storage.ptr_at<curandState>(0);
  auto* random_b = random_a + 1;
  test_cuda::initialize_random_state(random_a, 123, a.stream());
  test_cuda::initialize_random_state(random_b, 456, b.stream());
  cuda_check(cudaDeviceSynchronize());

  auto prompt_a = input(kA.prompt);
  auto prompt_b = input(kB.prompt);
  cache_a.resize(kA.prompt.size());
  auto* sampled_a = a.prefill(&prompt_a, thread_pool, &cache_a, 1, 1.0f, 1.0f, random_a);
  const auto expected_prefill_a = last_row_argmax(expected_a[0]);
  require(download_token(sampled_a) == expected_prefill_a,
          "Sampled A prefill differs from final-row logits argmax");
  const auto held_a_prefill = snapshot(a, cache_a);
  equal_cache(expected_a[0], held_a_prefill, "Sampled A prefill");
  cache_b.resize(kB.prompt.size());
  auto* sampled_b = b.prefill(&prompt_b, thread_pool, &cache_b, 1, 1.0f, 1.0f, random_b);
  require(sampled_a != sampled_b, "Sessions must own separate sampled-token storage");
  require(download_token(sampled_b) == last_row_argmax(expected_b[0]),
          "Sampled B prefill differs from final-row logits argmax");
  require(download_token(sampled_a) == expected_prefill_a,
          "B prefill modified held A sampled token");
  equal_cache(expected_b[0], snapshot(b, cache_b), "Sampled B prefill");
  equal_cache(held_a_prefill, snapshot(a, cache_a), "Held sampled A prefill after B");

  for (std::size_t step = 0; step < kA.continuation.size(); ++step) {
    // Use fixed history tokens to compare sampling with the all-logits run.
    auto token_a = input({kA.continuation[step]});
    auto token_b = input({kB.continuation[step]});
    cache_a.resize(cache_a.size() + 1);
    auto* next_a = a.forward(&token_a, thread_pool, &cache_a, 1, 1.0f, 1.0f, random_a);
    require(next_a == sampled_a, "A sampling storage changed during decode");
    const auto expected_token_a = last_row_argmax(expected_a[step + 1]);
    require(download_token(next_a) == expected_token_a,
            "Sampled A decode differs from logits argmax");
    const auto held_a = snapshot(a, cache_a);
    equal_cache(expected_a[step + 1], held_a, "Sampled A decode");
    cache_b.resize(cache_b.size() + 1);
    auto* next_b = b.forward(&token_b, thread_pool, &cache_b, 1, 1.0f, 1.0f, random_b);
    require(next_b == sampled_b && next_a != next_b,
            "Sampled-token storage must stay fixed and session-specific");
    require(download_token(next_b) == last_row_argmax(expected_b[step + 1]),
            "Sampled B decode differs from logits argmax");
    require(download_token(next_a) == expected_token_a, "B decode modified held A sampled token");
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
  Cache cache_a(1, kCapacity, kHidden, Device::CUDA);
  Cache cache_b(1, kCapacity, kHidden, Device::CUDA);
  Session a(model, graph);
  Session b(model, graph);
  require(a.model().get() == model.get() && b.model().get() == model.get(),
          "Sessions must share the requested immutable model");
  require(a.stream() != nullptr && b.stream() != nullptr && a.stream() != b.stream(),
          "Sessions must own separate execution streams");
  const auto a_workspace = a.decode_workspace_bytes();
  const auto b_workspace = b.decode_workspace_bytes();
  require(a_workspace > 0 && b_workspace > 0,
          "Decode workspaces must be allocated during session construction");
  const auto& plan = a.decode_workspace_plan();
  require(a.decode_reused_bytes() > 0 && a.decode_workspace_bytes() < a.decode_unaliased_bytes() &&
              a.decode_workspace_bytes() == plan.total_bytes(),
          "Automatic decoder memory must reclaim dead activation storage");
  for (std::size_t left_index = 0; left_index < plan.allocations().size(); ++left_index) {
    const auto& left = plan.allocations()[left_index];
    for (std::size_t right_index = left_index + 1; right_index < plan.allocations().size();
         ++right_index) {
      const auto& right = plan.allocations()[right_index];
      const bool live_overlap =
          left.first_use <= right.last_use && right.first_use <= left.last_use;
      const bool byte_overlap =
          left.offset < right.offset + right.bytes && right.offset < left.offset + left.bytes;
      require(!(live_overlap && byte_overlap), "Live decoder activations share planned bytes");
    }
  }

  initialize_cache(cache_a);
  initialize_cache(cache_b);
  auto prompt_a = input(kA.prompt);
  auto prompt_b = input(kB.prompt);
  cache_a.resize(kA.prompt.size());
  auto logits_a = a.prefill_eager(&prompt_a, &cache_a);
  const auto held_prefill = snapshot(a, cache_a, &logits_a);
  equal_snapshot(expected_a[0], held_prefill, "Interleaved A prefill");

  cache_b.resize(kB.prompt.size());
  auto logits_b = b.prefill_eager(&prompt_b, &cache_b);
  equal_snapshot(expected_b[0], snapshot(b, cache_b, &logits_b), "Interleaved B prefill");
  equal_snapshot(held_prefill, snapshot(a, cache_a, &logits_a), "Held A prefill after B");

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
    equal_snapshot(expected_a[step + 1], held_a, "Interleaved A decode " + std::to_string(step));
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
      require(logits_a.data_ptr() == first_decode_a && logits_b.data_ptr() == first_decode_b,
              "Decode must reuse fixed session output storage");
    }
    require(a.decode_workspace_bytes() == a_workspace && b.decode_workspace_bytes() == b_workspace,
            "Decode workspace capacity changed during generation");
  }
  std::cout << (graph ? "Graph" : "Eager")
            << " sessions: isolated/interleaved logits and full KV match\n";
  sampling_test(model, expected_a, expected_b, graph);
  reset_test(model, expected_a, graph);
  destruction_test(model, expected_a, graph);
  managed_test(model, expected_a, expected_b, graph);
  allocation_test(model, expected_a, expected_b, graph);
}

void expect_rejected(Session& session, Cache& cache, const std::function<void()>& operation,
                     const std::string& label, const Logits* held_logits = nullptr,
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
                                 const std::string& diagnostic, const std::string& label) {
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
  const std::vector<BFloat16> source_values(source_gate.data_ptr(),
                                            source_gate.data_ptr() + source_gate.numel());
  auto model = std::make_shared<Model>(loader_weights, config());
  const auto& frozen_gate = model->get_params().at("w_gate0");
  require(frozen_gate.device() == Device::CUDA && frozen_gate.sizes() == source_shape &&
              frozen_gate.strides() == source_strides && !frozen_gate.is_contiguous(),
          "Model preparation lost rectangular gate transpose metadata");
  equal_values(source_values, download(frozen_gate), "Prepared rectangular gate physical values");
  auto& mutable_gate = loader_weights.at("w_gate0");
  std::fill(mutable_gate.data_ptr(), mutable_gate.data_ptr() + mutable_gate.numel(),
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
  strided_embeddings.at("token_embeddings.weight") =
      Tensor<BFloat16>(std::vector<BFloat16>(kHidden * kVocabulary, __float2bfloat16(0.5f)),
                       {kHidden, kVocabulary})
          .transpose(0, 1);
  expect_preparation_rejected([&] { Model invalid(strided_embeddings, config()); },
                              "contiguous: token_embeddings.weight", "Noncontiguous embeddings");
  auto mismatched_width = config();
  mismatched_width.at("hidden_size") = 130;
  expect_preparation_rejected([&] { Model invalid(weights(), mismatched_width); },
                              "shape mismatch: token_embeddings.weight",
                              "Configured embedding width mismatch");
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
  fixture.packed.emplace(
      "w_gate0",
      Tensor<std::int32_t>(std::vector<std::int32_t>(kIntermediate * (kHidden / 8), 0x77777777),
                           {kIntermediate, kHidden / 8}));
  fixture.zeros.emplace("w_gate0",
                        Tensor<std::int32_t>(std::vector<std::int32_t>(kIntermediate, 0x77777777),
                                             {kIntermediate, 1}));
  fixture.scales.emplace(
      "w_gate0", Tensor<BFloat16>(
                     std::vector<BFloat16>(kIntermediate * padded_groups, __float2bfloat16(0.02f)),
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
  auto reject_fixture = [](const AwqFixture& fixture, const std::string& diagnostic,
                           const std::string& label) {
    expect_preparation_rejected(
        [&] {
          Model invalid(fixture.dense, fixture.packed, fixture.scales, fixture.zeros,
                        fixture.configuration);
        },
        diagnostic, label);
  };
  {
    auto fixture = awq_fixture();
    fixture.packed.at("w_gate0") =
        Tensor<std::int32_t>(std::vector<std::int32_t>(kHidden * (kIntermediate / 8), 0x77777777),
                             {kHidden, kIntermediate / 8});
    reject_fixture(fixture, "w_gate0.qweight", "Old AWQ input-major layout");
  }
  {
    auto fixture = awq_fixture();
    fixture.zeros.at("w_gate0") = Tensor<std::int32_t>(
        std::vector<std::int32_t>(2 * (kIntermediate / 8), 0x77777777), {2, kIntermediate / 8});
    reject_fixture(fixture, "w_gate0.qzeros", "Old AWQ group-major zeros layout");
  }
  {
    auto fixture = awq_fixture();
    fixture.scales.at("w_gate0") = Tensor<BFloat16>(
        std::vector<BFloat16>(2 * kIntermediate, __float2bfloat16(0.02f)), {2, kIntermediate});
    reject_fixture(fixture, "w_gate0.scales", "Old AWQ group-major scales layout");
  }
  {
    auto fixture = awq_fixture();
    fixture.configuration.at("group_size") = 96;
    reject_fixture(fixture, "divisible by group_size: w_gate0", "Indivisible AWQ input groups");
  }
  {
    auto fixture = awq_fixture(1);
    reject_fixture(fixture, "w_gate0.scales", "Undersized AWQ scales");
  }
  {
    auto fixture = awq_fixture();
    fixture.packed.at("w_gate0") =
        Tensor<std::int32_t>(std::vector<std::int32_t>((kHidden / 8) * kIntermediate, 0x77777777),
                             {kHidden / 8, kIntermediate})
            .transpose(0, 1);
    reject_fixture(fixture, "contiguous: w_gate0.qweight", "Noncontiguous AWQ qweight");
  }
  {
    auto fixture = awq_fixture();
    fixture.scales.at("w_gate0") =
        Tensor<BFloat16>(std::vector<BFloat16>(2 * kIntermediate, __float2bfloat16(0.02f)),
                         {2, kIntermediate})
            .transpose(0, 1);
    reject_fixture(fixture, "contiguous: w_gate0.scales", "Noncontiguous AWQ scales");
  }
  {
    // Sixteen groups require two packed zero columns, making a transpose
    // noncontiguous even when its logical output-major shape is correct.
    auto fixture = awq_fixture(16);
    fixture.configuration.at("group_size") = 8;
    fixture.zeros.at("w_gate0") =
        Tensor<std::int32_t>(std::vector<std::int32_t>(2 * kIntermediate, 0x77777777),
                             {2, kIntermediate})
            .transpose(0, 1);
    reject_fixture(fixture, "contiguous: w_gate0.qzeros", "Noncontiguous AWQ qzeros");
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
    std::copy_n(expected_k.data() + layer * layer_elements, layer_elements, views.first.data_ptr());
    std::copy_n(expected_v.data() + layer * layer_elements, layer_elements,
                views.second.data_ptr());
  }

  auto verify = [&](Device device, const std::string& phase) {
    require(cache.device() == device && cache.size() == active && cache.get_n_layers() == layers &&
                cache.get_max_seq_len() == capacity && cache.get_head_dim() == width,
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
        equal_values(
            std::vector<BFloat16>(expected_k.begin() + offset, expected_k.begin() + offset + width),
            download(key), phase + " token slice K values");
        equal_values(
            std::vector<BFloat16>(expected_v.begin() + offset, expected_v.begin() + offset + width),
            download(value), phase + " token slice V values");
      }
    }
  };

  verify(Device::CPU, "Initial CPU cache");
  for (int upload = 0; upload < 2; ++upload) {
    cache.cuda();
    verify(Device::CUDA, "Uploaded CUDA cache");
    {
      Tensor<BFloat16> overwrite_k({layers, capacity, width}, Device::CUDA);
      Tensor<BFloat16> overwrite_v({layers, capacity, width}, Device::CUDA);
      cuda_check(cudaMemset(overwrite_k.data_ptr(), 0, overwrite_k.numel() * sizeof(BFloat16)));
      cuda_check(cudaMemset(overwrite_v.data_ptr(), 0, overwrite_v.numel() * sizeof(BFloat16)));
      cuda_check(cudaDeviceSynchronize());
      verify(Device::CUDA, "CUDA cache during independent tensor writes");
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
  Cache cache(1, kCapacity, kHidden, Device::CUDA);
  Session session(model, false);
  initialize_cache(cache);
  auto prompt = input(kA.prompt);
  cache.resize(kA.prompt.size());
  auto logits = session.prefill_eager(&prompt, &cache);
  const auto views_before = cache.get_layer_view(0);
  const auto before = snapshot(session, cache, &logits);
  {
    Tensor<BFloat16> unrelated_k({1, kCapacity, kHidden}, Device::CUDA);
    Tensor<BFloat16> unrelated_v({1, kCapacity, kHidden}, Device::CUDA);
    require(unrelated_k.data_ptr() != views_before.first.data_ptr() &&
                unrelated_v.data_ptr() != views_before.second.data_ptr(),
            "Ordinary tensors must own storage separate from persistent cache");
    cuda_check(cudaMemset(unrelated_k.data_ptr(), 0, unrelated_k.numel() * sizeof(BFloat16)));
    cuda_check(cudaMemset(unrelated_v.data_ptr(), 0, unrelated_v.numel() * sizeof(BFloat16)));
    Cache other_cache(1, kCapacity, kHidden, Device::CUDA);
    Session other(model, false);
    auto other_prompt = input(kB.prompt);
    other_cache.resize(kB.prompt.size());
    other.prefill_eager(&other_prompt, &other_cache);
    cuda_check(cudaDeviceSynchronize());
    const auto views_after = cache.get_layer_view(0);
    require(views_before.first.data_ptr() == views_after.first.data_ptr() &&
                views_before.second.data_ptr() == views_after.second.data_ptr(),
            "Independent tensor/session allocation changed persistent cache backing");
    equal_snapshot(before, snapshot(session, cache, &logits),
                   "Held cache/logits during independent tensor writes and session execution");
  }
  equal_snapshot(before, snapshot(session, cache, &logits),
                 "Held cache/logits after independent storage destruction");
  cache_roundtrip_test();
  std::cout
      << "Persistent cache and session storage survive independent tensor/session lifetimes\n";
}

void sampling_validation_test() {
  constexpr std::size_t vocabulary = 2048;
  constexpr std::uint32_t sentinel = 0xa5c39e71U;
  Tensor<BFloat16> logits(std::vector<BFloat16>(vocabulary, __float2bfloat16(0.25f)),
                          {1, vocabulary}, Device::CUDA);
  const auto original_logits = download(logits);
  CudaWorkspaceArena output_storage;
  output_storage.reserve(sizeof(std::uint32_t));
  auto* output = output_storage.ptr_at<std::uint32_t>(0);
  const auto context = op::cuda::prepare_execution_context(nullptr, nullptr);
  const auto plan = op::cuda::prepare_sampling(context, vocabulary);
  CudaWorkspaceArena scratch;
  scratch.reserve(plan.total_bytes);
  const auto workspace = TensorView<unsigned char, 1>::contiguous(scratch.ptr_at<unsigned char>(0),
                                                                  {plan.total_bytes});
  const auto tokens = TensorView<std::uint32_t, 1>::contiguous(output, {1});
  cuda_check(cudaGetLastError());
  for (std::size_t top_k : {std::size_t{0}, std::size_t{1025}}) {
    cuda_check(cudaMemcpy(output, &sentinel, sizeof(sentinel), cudaMemcpyHostToDevice));
    bool rejected = false;
    try {
      // The null RNG would be invalid for an actual sampling launch. Invalid
      // top_k must fail validation before touching device state or output.
      op::cuda::sample(context, borrow_tensor_view<2>(static_cast<const Tensor<BFloat16>&>(logits)),
                       tokens, {}, workspace, plan, 1.0f, 1.0f, top_k, nullptr);
    } catch (const std::invalid_argument& error) {
      require(std::strlen(error.what()) > 0, "Invalid sampling diagnostic is empty");
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
  Cache cache(1, kCapacity, kHidden, Device::CUDA);
  Session session(model, false);
  initialize_cache(cache);
  cache.resize(1);
  auto token = input({3});
  auto multi = input({3, 7});
  auto rank_two = input({3}, {1, 1});
  auto empty = token.slice({0}, {0});
  Tensor<std::uint32_t> cpu(std::vector<std::uint32_t>{3}, {1});

  expect_rejected(
      session, cache, [&] { session.forward_eager(nullptr, &cache); }, "Null decode input");
  expect_rejected(
      session, cache, [&] { session.forward_eager(&token, nullptr); }, "Null decode cache");
  expect_rejected(
      session, cache, [&] { session.prefill_eager(nullptr, &cache); }, "Null prefill input");
  expect_rejected(
      session, cache, [&] { session.prefill_eager(&token, nullptr); }, "Null prefill cache");
  expect_rejected(session, cache, [&] { session.forward_eager(&cpu, &cache); }, "CPU input");
  expect_rejected(
      session, cache, [&] { session.forward_eager(&multi, &cache); }, "Multiple decode tokens");
  expect_rejected(
      session, cache, [&] { session.forward_eager(&rank_two, &cache); }, "Wrong input rank");
  expect_rejected(
      session, cache, [&] { session.prefill_eager(&empty, &cache); }, "Empty prefill input");
  cache.clear();
  expect_rejected(
      session, cache, [&] { session.forward_eager(&token, &cache); },
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
    expect_rejected(
        session, cache, [&] { session.forward_eager(&token, item.first); }, item.second, nullptr,
        item.first);
  }
  Cache cpu_cache(1, kCapacity, kHidden, Device::CPU, 1);
  expect_rejected(
      session, cache, [&] { session.forward_eager(&token, &cpu_cache); }, "CPU KV cache");
  expect_rejected(
      session, cache, [&] { session.prefill_eager(&multi, &cache); },
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
  expect_rejected(
      session, cache, [&] { session.forward_eager(nullptr, &cache); },
      "Null input with live decode logits", &logits);
  expect_rejected(
      session, cache, [&] { session.forward_eager(&multi, &cache); },
      "Wrong shape with live decode logits", &logits);
  auto invalid_token = input({static_cast<std::uint32_t>(kVocabulary)});
  expect_rejected(
      session, cache, [&] { session.forward_eager(&invalid_token, &cache); },
      "Out-of-vocabulary decode token", &logits);
  auto invalid_prompt = input({3, static_cast<std::uint32_t>(kVocabulary), 11});
  expect_rejected(
      session, cache, [&] { session.prefill_eager(&invalid_prompt, &cache); },
      "Out-of-vocabulary prefill token", &logits);
  auto token_pairs = input({3, 17, 7, 23, 11, 29}, {3, 2});
  auto noncontiguous_prompt = token_pairs.slice({0, 0}, {3, 1}).squeeze(1);
  require(noncontiguous_prompt.sizes() == std::vector<std::size_t>{3} &&
              !noncontiguous_prompt.is_contiguous(),
          "Strided token fixture must be noncontiguous with valid rank/extent");
  expect_rejected(
      session, cache, [&] { session.prefill_eager(&noncontiguous_prompt, &cache); },
      "Noncontiguous prefill tokens", &logits);
  Cache other_cache(1, kCapacity, kHidden, Device::CUDA, 1);
  initialize_cache(other_cache);
  expect_rejected(
      session, cache, [&] { session.forward_eager(&token, &other_cache); }, "Switching bound cache",
      &logits, &other_cache);

  require(cache.get_max_seq_len() == kCapacity && cache.k_capacity_view(0).shape[0] == kCapacity &&
              cache.v_capacity_view(0).shape[0] == kCapacity,
          "Cache capacity must remain the immutable physical storage limit");
  std::cout << "Invalid inputs and cache changes reject before writes\n";
}

Tensor<BFloat16> gathered_embeddings(const Weights& source,
                                     const std::vector<std::uint32_t>& tokens) {
  const auto& table = source.at("token_embeddings.weight");
  require(table.device() == Device::CPU && table.is_contiguous(),
          "Embedding fixture must be independently gathered from host checkpoint data");
  std::vector<BFloat16> values(tokens.size() * kHidden);
  for (std::size_t row = 0; row < tokens.size(); ++row) {
    require(tokens[row] < kVocabulary, "Embedding fixture token is outside vocabulary");
    std::copy_n(table.data_ptr() + tokens[row] * kHidden, kHidden, values.data() + row * kHidden);
  }
  return Tensor<BFloat16>(std::move(values), {tokens.size(), kHidden}, Device::CUDA);
}

std::vector<BFloat16> scalar_head(const Weights& source, Logits hidden) {
  require(hidden.shape[1] == kHidden, "Decoder hidden width differs from model width");
  const auto values = download(hidden);
  const auto& head = source.at("lm_head");
  require(head.device() == Device::CPU &&
              head.sizes() == std::vector<std::size_t>({kHidden, kVocabulary}),
          "Scalar head requires logical host checkpoint weights");
  std::vector<BFloat16> logits(hidden.shape[0] * kVocabulary);
  for (std::size_t row = 0; row < hidden.shape[0]; ++row) {
    for (std::size_t token = 0; token < kVocabulary; ++token) {
      double sum = 0;
      for (std::size_t column = 0; column < kHidden; ++column) {
        sum += static_cast<double>(__bfloat162float(values[row * kHidden + column])) *
               __bfloat162float(
                   head.data_ptr()[column * head.strides()[0] + token * head.strides()[1]]);
      }
      logits[row * kVocabulary + token] = __float2bfloat16(static_cast<float>(sum));
    }
  }
  return logits;
}

void embedding_session_test(const Weights& source, bool graph) {
  constexpr std::size_t capacity = 8;
  auto model = std::make_shared<Model>(source, config());
  auto a = Session::create(model, capacity, graph);
  auto b = Session::create(model, capacity, graph);
  auto token_a = Session::create(model, capacity, graph);
  auto token_b = Session::create(model, capacity, graph);
  auto embedding_a = gathered_embeddings(source, kA.prompt);
  auto embedding_b = gathered_embeddings(source, kB.prompt);
  const auto view_a = borrow_tensor_view<2>(embedding_a).as_const();
  const auto view_b = borrow_tensor_view<2>(embedding_b).as_const();
  const auto original_input = download(embedding_a);
  std::array<Tensor<BFloat16>, 3> continuation_a;
  std::array<Tensor<BFloat16>, 3> continuation_b;
  std::array<TensorView<const BFloat16, 2>, 3> continuation_views_a;
  std::array<TensorView<const BFloat16, 2>, 3> continuation_views_b;
  for (std::size_t step = 0; step < 3; ++step) {
    continuation_a[step] = gathered_embeddings(source, {kA.continuation[step]});
    continuation_b[step] = gathered_embeddings(source, {kB.continuation[step]});
    continuation_views_a[step] = borrow_tensor_view<2>(continuation_a[step]).as_const();
    continuation_views_b[step] = borrow_tensor_view<2>(continuation_b[step]).as_const();
  }
  expect_managed_rejected(
      *a, [&] { a->decode_embeddings(continuation_views_a[0], 0); },
      "Embedding decode before prefill");
  auto hidden_a = a->prefill_embeddings(view_a);
  const auto held_a = download(hidden_a);
  auto hidden_b = b->prefill_embeddings(view_b);
  require(a->prefill_workspace_bytes() ==
              a->estimate_embedding_prefill_workspace_bytes(kA.prompt.size()),
          "Embedding-only prefill must allocate its hidden-output plan");
  auto expected_a = token_a->prefill(kA.prompt);
  auto expected_b = token_b->prefill(kB.prompt.data(), kB.prompt.size());
  require(hidden_a.shape == std::array<std::size_t, 2>{kA.prompt.size(), kHidden} &&
              hidden_b.shape == std::array<std::size_t, 2>{kB.prompt.size(), kHidden} &&
              a->context_size() == kA.prompt.size() && b->context_size() == kB.prompt.size(),
          "Embedding prefill must return all normalized hidden rows and record private history");
  equal_values(held_a, download(hidden_a), "Session B prefill modified held A hidden state");
  near_values(download(expected_a), scalar_head(source, hidden_a),
              "Independent scalar head on A embedding prefill");
  near_values(download(expected_b), scalar_head(source, hidden_b),
              "Independent scalar head on B embedding prefill");

  const auto null_input = TensorView<const BFloat16, 2>{nullptr, {1, kHidden}, {kHidden, 1}};
  const auto empty_input = TensorView<const BFloat16, 2>::contiguous(nullptr, {0, kHidden});
  const auto wrong_width = TensorView<const BFloat16, 2>::contiguous(view_a.data, {1, kHidden - 1});
  const auto multiple_rows = view_a.checked_subview({0, 0}, {2, kHidden});
  const auto strided_input =
      TensorView<const BFloat16, 2>::strided(view_a.data, {2, kHidden}, {kHidden + 1, 1});
  Tensor<BFloat16> host_input(std::vector<BFloat16>(kHidden, __float2bfloat16(0.25f)),
                              {1, kHidden});
  const auto host_view = borrow_tensor_view<2>(host_input).as_const();
  auto oversized = gathered_embeddings(source, std::vector<std::uint32_t>(capacity + 1, 3));
  const auto oversized_view = borrow_tensor_view<2>(oversized).as_const();
  for (const auto invalid :
       {null_input, empty_input, wrong_width, strided_input, host_view, oversized_view}) {
    expect_managed_rejected(
        *a, [&] { a->prefill_embeddings(invalid); }, "Invalid embedding prefill", &hidden_a);
  }
  for (const auto invalid :
       {null_input, empty_input, wrong_width, multiple_rows, strided_input, host_view}) {
    expect_managed_rejected(
        *a, [&] { a->decode_embeddings(invalid, kA.prompt.size()); }, "Invalid embedding decode",
        &hidden_a);
  }
  expect_managed_rejected(
      *a, [&] { a->prefill_embeddings(view_a, kCapacity - 1); },
      "Embedding prefill position exceeds model range", &hidden_a);
  expect_managed_rejected(
      *a, [&] { a->decode_embeddings(continuation_views_a[0], kCapacity); },
      "Embedding decode position exceeds model range", &hidden_a);
  expect_managed_rejected(
      *a,
      [&] {
        a->decode_embeddings(continuation_views_a[0], std::numeric_limits<std::size_t>::max());
      },
      "Embedding position arithmetic overflow", &hidden_a);
  const std::vector<std::uint32_t> invalid_host{kVocabulary};
  expect_managed_rejected(
      *token_a, [&] { token_a->prefill(invalid_host); }, "Host prefill rejects OOV before upload",
      &expected_a);
  expect_managed_rejected(
      *token_a, [&] { token_a->prefill(nullptr, 1); }, "Host prefill rejects null input",
      &expected_a);
  expect_managed_rejected(
      *token_a, [&] { token_a->prefill(kA.prompt.data(), 0); }, "Host prefill rejects empty input",
      &expected_a);
  const std::vector<std::uint32_t> oversized_host(capacity + 1, 3);
  expect_managed_rejected(
      *token_a, [&] { token_a->prefill(oversized_host); }, "Host prefill rejects capacity overflow",
      &expected_a);

  const BFloat16* output_a = nullptr;
  const BFloat16* output_b = nullptr;
  const auto workspace_a = a->decode_workspace_bytes();
  const auto workspace_b = b->decode_workspace_bytes();
  test_alloc::Counts total;
  for (std::size_t step = 0; step < 3; ++step) {
    // The first embedding decode primes launch paths. Later submissions must
    // reuse session storage and preserve the other session's held output.
    test_alloc::Scope probe_a;
    hidden_a = a->decode_embeddings(continuation_views_a[step], kA.prompt.size() + step);
    const auto count_a = probe_a.finish();
    const auto held_hidden_a = download(hidden_a);
    test_alloc::Scope probe_b;
    hidden_b = b->decode_embeddings(continuation_views_b[step], kB.prompt.size() + step);
    const auto count_b = probe_b.finish();
    if (step) {
      for (const auto& count : {count_a, count_b}) {
        total.host_allocations += count.host_allocations;
        total.host_frees += count.host_frees;
        total.device_allocations += count.device_allocations;
        total.device_frees += count.device_frees;
      }
      require(hidden_a.data_ptr() == output_a && hidden_b.data_ptr() == output_b,
              "Embedding decode must reuse each session's fixed hidden output");
    } else {
      output_a = hidden_a.data_ptr();
      output_b = hidden_b.data_ptr();
      require(output_a != output_b, "Embedding sessions must own separate hidden outputs");
    }
    require(hidden_a.shape == std::array<std::size_t, 2>{1, kHidden} &&
                hidden_b.shape == std::array<std::size_t, 2>{1, kHidden} &&
                a->decode_workspace_bytes() == workspace_a &&
                b->decode_workspace_bytes() == workspace_b &&
                a->context_size() == kA.prompt.size() + step + 1 &&
                b->context_size() == kB.prompt.size() + step + 1,
            "Embedding decode must advance private history with fixed workspace capacity");
    equal_values(held_hidden_a, download(hidden_a),
                 "Session B decode modified held A hidden output");
    expected_a = token_a->decode(kA.continuation[step]);
    expected_b = token_b->decode(kB.continuation[step]);
    near_values(download(expected_a), scalar_head(source, hidden_a),
                "Independent scalar head on A embedding decode");
    near_values(download(expected_b), scalar_head(source, hidden_b),
                "Independent scalar head on B embedding decode");
  }
  require(!total.host_allocations && !total.host_frees,
          "Warm embedding decode allocated or freed C++ storage");
#ifdef EDGE_INFER_TEST_CUDA_WRAPPING
  require(!total.device_allocations && !total.device_frees,
          "Warm embedding decode allocated or freed native CUDA storage");
#endif
  equal_values(original_input, download(embedding_a),
               "Embedding execution modified caller input storage");

  // Both directions may share ordinary zero-offset history: embeddings ->
  // token -> embeddings. Continuation parity also detects KV writes on a
  // rejected input above, which cannot be observed through a public cache.
  auto mixed_logits = a->decode(kA.continuation[0]);
  expected_a = token_a->decode(kA.continuation[0]);
  near_values(download(expected_a), download(mixed_logits), "Token decode after embedding history");
  hidden_a = a->decode_embeddings(continuation_views_a[1], a->context_size());
  expected_a = token_a->decode(kA.continuation[1]);
  near_values(download(expected_a), scalar_head(source, hidden_a),
              "Embedding decode after token continuation");
  require(a->context_size() == capacity,
          "Mixed token/embedding history must consume the same physical capacity");
  expect_managed_rejected(
      *a, [&] { a->decode_embeddings(continuation_views_a[0], capacity); },
      "Embedding decode exceeds physical cache capacity", &hidden_a);

  // A logical RoPE offset need not equal the physical cache write offset.
  // Four positions beginning at eight must fit in a four-slot private cache.
  auto shifted_incremental = Session::create(model, 4, graph);
  auto shifted_full = Session::create(model, 4, graph);
  shifted_incremental->prefill_embeddings(view_a, 8);
  auto shifted_last = shifted_incremental->decode_embeddings(continuation_views_a[0], 11);
  auto full_tokens = kA.prompt;
  full_tokens.push_back(kA.continuation[0]);
  auto full_embeddings = gathered_embeddings(source, full_tokens);
  auto full_hidden =
      shifted_full->prefill_embeddings(borrow_tensor_view<2>(full_embeddings).as_const(), 8);
  const auto full_last = full_hidden.checked_subview({3, 0}, {1, kHidden});
  near_values(download(full_last), download(shifted_last),
              "Shifted logical positions with compact physical cache");
  require(shifted_incremental->context_size() == 4 && shifted_full->context_size() == 4,
          "Custom logical positions must consume only the submitted physical rows");

  a->reset();
  expect_managed_rejected(
      *a, [&] { a->decode_embeddings(continuation_views_a[0], 0); },
      "Embedding decode after reset");
  hidden_a = a->prefill_embeddings(view_a);
  expected_a = token_a->prefill(kA.prompt);
  near_values(download(expected_a), scalar_head(source, hidden_a),
              "Repeated embedding prefill after reset");
  // Identical row counts still need different plans when the output switches.
  auto same_session_logits = a->prefill(kA.prompt);
  equal_values(download(expected_a), download(same_session_logits),
               "Token prefill after same-length hidden-output plan");
  require(a->prefill_workspace_bytes() >= a->estimate_prefill_workspace_bytes(kA.prompt.size()),
          "Switching to token output must provide head storage");
  const auto peak_bytes = a->prefill_workspace_bytes();
  hidden_a = a->prefill_embeddings(view_a);
  near_values(download(expected_a), scalar_head(source, hidden_a),
              "Hidden prefill after same-length token-output plan");
  require(a->prefill_workspace_bytes() == peak_bytes,
          "Switching output contracts must retain reusable peak arena capacity");
  std::cout << "Embedding sessions with token graphs " << (graph ? "enabled" : "disabled")
            << ": independent scalar-head parity, compact shifted cache, warm C++ allocations="
            << total.host_allocations;
#ifdef EDGE_INFER_TEST_CUDA_WRAPPING
  std::cout << ", native CUDA allocations=" << total.device_allocations;
#endif
  std::cout << '\n';
}

#ifdef EDGE_INFER_TEST_CUDA_WRAPPING
class InjectedSessionFailure {
 public:
  InjectedSessionFailure(cudaStream_t stream, bool fail_completion) : stream_(stream) {
    cuda_check(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
    test_session_failure::stream = stream;
    test_session_failure::submitted = event_;
    test_session_failure::fail_gemm = true;
    test_session_failure::gemm_failed = false;
    test_session_failure::fail_wait = fail_completion;
    test_session_failure::completion_attempts = 0;
    test_session_failure::actual_completions = 0;
    // Reject output projection after attention has consumed submitted K/V.
    test_session_failure::gemms_until_failure = 4;
  }
  ~InjectedSessionFailure() {
    // Even failed assertions drain real work before borrowed fixtures die.
    test_session_failure::fail_gemm = false;
    test_session_failure::fail_wait = false;
    test_session_failure::gemm_failed = false;
    __real_cudaStreamSynchronize(stream_);
    cudaEventDestroy(event_);
    test_session_failure::stream = nullptr;
    test_session_failure::submitted = nullptr;
  }
  void allow_completion() { test_session_failure::fail_wait = false; }
  cudaEvent_t event() const { return event_; }

 private:
  cudaStream_t stream_;
  cudaEvent_t event_ = nullptr;
};

void execution_failure_completion_test(const Weights& source) {
  const auto model = std::make_shared<const Model>(source, config());
  const auto expected = isolated(source, kA, false);
  for (bool failed_completion : {false, true}) {
    for (int route = 0; route < 3; ++route) {
      auto session = Session::create(model, kCapacity);
      auto sibling = session->new_session(kCapacity);
      auto tokens = input(kA.prompt);
      auto embeddings = gathered_embeddings(source, kA.prompt);
      auto held = session->prefill(borrow_tensor_view<1>(tokens).as_const());
      auto sibling_held = sibling->prefill(kA.prompt);
      const auto sibling_before = download(sibling_held);
      const auto previous = session->context_size();
      const auto decode_bytes = session->decode_workspace_bytes();
      const auto prefill_bytes = session->prefill_workspace_bytes();
      InjectedSessionFailure injection(session->stream(), failed_completion);
      const auto operation = [&] {
        if (route == 0) session->prefill(borrow_tensor_view<1>(tokens).as_const());
        else if (route == 1) session->prefill(kA.prompt);
        else session->prefill_embeddings(borrow_tensor_view<2>(embeddings).as_const());
      };
      std::string original_error;
      test_alloc::Scope allocations;
      try {
        operation();
      } catch (const std::exception& error) {
        original_error = error.what();
      }
      const auto counts = allocations.finish();
      require(original_error == "Direct cuBLAS submission failed: " +
                                    std::to_string(CUBLAS_STATUS_EXECUTION_FAILED),
              "Execution cleanup replaced or missed the original GEMM error");
      require(test_session_failure::completion_attempts > 0,
              "Execution error returned without attempting its own stream completion");
      require(counts.device_frees == 0 && session->decode_workspace_bytes() == decode_bytes &&
                  session->prefill_workspace_bytes() == prefill_bytes,
              "Execution failure released private storage before confirmed completion");
      cudaPointerAttributes attributes{};
      cuda_check(cudaPointerGetAttributes(&attributes, held.data));
      require(attributes.type == cudaMemoryTypeDevice,
              "Failed execution lost its retained output allocation");
      if (failed_completion) {
        require(test_session_failure::actual_completions == 0 && session->context_size() == previous,
                "Failed completion cleared history or claimed successful cleanup");
        const auto rejected = [&](const std::function<void()>& action) {
          bool unusable = false;
          try { action(); }
          catch (const std::exception& error) {
            unusable = std::string(error.what()).find("unusable") != std::string::npos;
          }
          require(unusable, "Poisoned session accepted execution, reset, reconfiguration or fork");
        };
        rejected(operation);
        rejected([&] { session->decode(kA.continuation[0]); });
        rejected([&] { session->reset(); });
        rejected([&] { session->set_graph_enabled(true); });
        rejected([&] { auto child = session->new_session(kCapacity); });
        rejected([&] { auto child = session->fork_executor(); });
        injection.allow_completion();
        session->synchronize();
        cuda_check(cudaEventQuery(injection.event()));
        rejected(operation);  // A later successful wait never revives it.
      } else {
        require(test_session_failure::actual_completions > 0 && session->context_size() == 0,
                "Confirmed execution cleanup did not invalidate partially written history");
        cuda_check(cudaEventQuery(injection.event()));
        auto recovered = session->prefill(kA.prompt);
        equal_values(expected[0].logits, download(recovered), "Recovered session after confirmed cleanup");
      }
      equal_values(sibling_before, download(sibling_held), "Sibling output after execution failure");
      auto continued = sibling->decode(kA.continuation[0]);
      equal_values(expected[1].logits, download(continued), "Sibling continuation after execution failure");
    }
  }
  std::cout << "Controlled execution failures: original error, own-stream drain, sticky poison and sibling isolation\n";
}
#endif

}  // namespace

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::cout << "Qwen3 session test skipped: no CUDA device\n";
    return 77;
  }
  try {
    const auto source = weights();
    model_preparation_test();
    allocation_probe_self_test();
    awq_preparation_test();
    cache_lifetime_test();
    sampling_validation_test();
    isolation_test(source, false);
    validation_test(source);
    isolation_test(source, true);
    cross_mode_test(source);
    embedding_session_test(source, false);
    embedding_session_test(source, true);
#ifdef EDGE_INFER_TEST_CUDA_WRAPPING
    execution_failure_completion_test(source);
#endif
    cuda_check(cudaDeviceSynchronize());
    std::cout << "Qwen3 shared-model session regression passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Qwen3 session regression failed: " << error.what() << '\n';
    return 1;
  }
}
