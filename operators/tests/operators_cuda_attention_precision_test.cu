#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "operators/cuda/execution.hpp"

namespace {

using BF16 = __nv_bfloat16;
constexpr std::size_t width = 128, heads = 4, kv_heads = 2, capacity = 41;
constexpr std::size_t branch_elements = heads * (width + 2);

void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

struct Stream {
  Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
  ~Stream() { cudaStreamDestroy(value); }
  cudaStream_t value = nullptr;
};

template <typename T>
class Buffer {
 public:
  explicit Buffer(std::size_t count) : count_(count) {
    check(cudaMalloc(reinterpret_cast<void**>(&pointer_), count * sizeof(T)));
  }
  ~Buffer() { cudaFree(pointer_); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  void write(const std::vector<T>& values, cudaStream_t stream) {
    if (values.size() != count_) throw std::runtime_error("Test upload extent mismatch");
    check(cudaMemcpyAsync(pointer_, values.data(), count_ * sizeof(T), cudaMemcpyHostToDevice,
                          stream));
    check(cudaStreamSynchronize(stream));
  }
  std::vector<T> read(cudaStream_t stream) const {
    check(cudaStreamSynchronize(stream));
    std::vector<T> result(count_);
    check(cudaMemcpy(result.data(), pointer_, count_ * sizeof(T), cudaMemcpyDeviceToHost));
    return result;
  }
  T* data() const { return pointer_; }

 private:
  T* pointer_ = nullptr;
  std::size_t count_;
};

struct Graph {
  ~Graph() {
    if (executable) cudaGraphExecDestroy(executable);
    if (value) cudaGraphDestroy(value);
  }
  cudaGraph_t value = nullptr;
  cudaGraphExec_t executable = nullptr;
};

double score(const std::vector<BF16>& q, const std::vector<BF16>& k, std::size_t head,
             std::size_t token) {
  double dot = 0;
  const auto kv_head = head / (heads / kv_heads);
  for (std::size_t dim = 0; dim < width; ++dim)
    dot += static_cast<double>(static_cast<float>(q[head * width + dim])) *
           static_cast<double>(static_cast<float>(k[(token * kv_heads + kv_head) * width + dim]));
  return dot / std::sqrt(static_cast<double>(width));
}

// Independent scalar gold: decoded BF16 inputs, stable softmax over the full
// live sequence, and one final BF16 rounding. No branch partition or kernel
// scratch layout participates in the reference calculation.
std::vector<BF16> reference(const std::vector<BF16>& q, const std::vector<BF16>& k,
                            const std::vector<BF16>& v, std::size_t length) {
  std::vector<BF16> output(heads * width);
  for (std::size_t head = 0; head < heads; ++head) {
    std::vector<double> weights(length);
    for (std::size_t token = 0; token < length; ++token) weights[token] = score(q, k, head, token);
    const double maximum = *std::max_element(weights.begin(), weights.end());
    double denominator = 0;
    for (auto& weight : weights) {
      weight = std::exp(weight - maximum);
      denominator += weight;
    }
    const auto kv_head = head / (heads / kv_heads);
    for (std::size_t dim = 0; dim < width; ++dim) {
      double numerator = 0;
      for (std::size_t token = 0; token < length; ++token)
        numerator +=
            weights[token] * static_cast<float>(v[(token * kv_heads + kv_head) * width + dim]);
      output[head * width + dim] = static_cast<BF16>(static_cast<float>(numerator / denominator));
    }
  }
  return output;
}

void compare(const std::vector<BF16>& actual, const std::vector<BF16>& expected, const char* stage,
             std::size_t length) {
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const float a = static_cast<float>(actual[index]), e = static_cast<float>(expected[index]);
    const float bound = 0.000125f + 0.004f * std::max(std::abs(a), std::abs(e));
    if (!std::isfinite(a) || std::abs(a - e) > bound)
      throw std::runtime_error(std::string(stage) + " mismatch at length " +
                               std::to_string(length) + ", element " + std::to_string(index) +
                               ": actual=" + std::to_string(a) + ", expected=" + std::to_string(e));
  }
}

void check_branch_maxima(const std::vector<float>& scratch, const std::vector<BF16>& q,
                         const std::vector<BF16>& k, std::size_t length) {
  const std::size_t branches = std::min<std::size_t>((length + 7) / 8, 5);
  const std::size_t per_branch = (length + branches - 1) / branches;
  for (std::size_t branch = 0; branch < branches; ++branch)
    for (std::size_t head = 0; head < heads; ++head) {
      double maximum = -std::numeric_limits<double>::infinity();
      for (std::size_t token = branch * per_branch;
           token < std::min(length, (branch + 1) * per_branch); ++token)
        maximum = std::max(maximum, score(q, k, head, token));
      const float actual = scratch[branch * branch_elements + head * (width + 2) + width];
      if (!std::isfinite(actual) || std::abs(actual - maximum) > 0.0002)
        throw std::runtime_error("Branch maximum lost FP32 precision at length " +
                                 std::to_string(length));
    }
}

void test_attention() {
  Stream stream;
  const auto context = op::cuda::prepare_execution_context(nullptr, stream.value);
  std::vector<BF16> q(heads * width), k(capacity * kv_heads * width), v(k.size());
  for (std::size_t head = 0; head < heads; ++head)
    for (std::size_t dim = 0; dim < width; ++dim)
      q[head * width + dim] =
          static_cast<BF16>(2.5f * (0.8f + 0.19f * std::sin(dim * 0.17f + head * 0.31f)));
  // High, fractional scores make rounding branch maxima to BF16 observable.
  // V varies independently across tokens, heads and dimensions, so changed
  // branch weights cannot cancel through symmetric or constant-value inputs.
  for (std::size_t token = 0; token < capacity; ++token)
    for (std::size_t head = 0; head < kv_heads; ++head)
      for (std::size_t dim = 0; dim < width; ++dim) {
        const auto index = (token * kv_heads + head) * width + dim;
        k[index] = static_cast<BF16>(1.75f + 0.03f * std::cos(token * 0.73f + head * 0.19f) +
                                     0.05f * std::sin(dim * 0.11f + token * 0.31f));
        v[index] = static_cast<BF16>(0.7f * std::sin(token * 0.47f + dim * 0.13f + head * 0.61f) +
                                     0.02f * (static_cast<int>(token % 7) - 3));
      }
  Buffer<BF16> dq(q.size()), dk(k.size()), dv(v.size()), eager(heads * width),
      graph_output(heads * width);
  Buffer<float> scratch(5 * branch_elements), graph_scratch(3 * branch_elements);
  Buffer<float*> branches(3);
  Buffer<int> lengths(2), pingpong(1);
  dq.write(q, stream.value);
  dk.write(k, stream.value);
  dv.write(v, stream.value);
  branches.write({graph_scratch.data(), graph_scratch.data() + branch_elements,
                  graph_scratch.data() + 2 * branch_elements},
                 stream.value);
  lengths.write({static_cast<int>(capacity), static_cast<int>(capacity)}, stream.value);
  pingpong.write({0}, stream.value);
  check(cudaStreamSynchronize(stream.value));
  const TensorView<const BF16, 3> query{dq.data(), {1, heads, width}, {heads * width, width, 1}};
  const TensorView<const BF16, 3> keys{
      dk.data(), {capacity, kv_heads, width}, {kv_heads * width, width, 1}};
  const TensorView<const BF16, 3> values{
      dv.data(), {capacity, kv_heads, width}, {kv_heads * width, width, 1}};
  const TensorView<BF16, 3> eager_view{eager.data(), {1, heads, width}, {heads * width, width, 1}};
  const TensorView<BF16, 3> graph_view{
      graph_output.data(), {1, heads, width}, {heads * width, width, 1}};
  Graph graph;
  check(cudaStreamBeginCapture(stream.value, cudaStreamCaptureModeThreadLocal));
  op::cuda::attention_graph<BF16>(context, query, keys, values, graph_view, branches.data(),
                                  lengths.data(), pingpong.data());
  check(cudaStreamEndCapture(stream.value, &graph.value));
  check(cudaGraphInstantiate(&graph.executable, graph.value, nullptr, nullptr, 0));
  // Growing across all eager partition thresholds, then shrinking through
  // empty fixed graph branches, checks both numeric and replay-state behavior.
  for (std::size_t length : {1u, 2u, 8u, 9u, 16u, 17u, 24u, 25u, 32u, 33u, 41u, 9u, 2u, 1u}) {
    auto live_keys = keys, live_values = values;
    live_keys.shape[0] = length;
    live_values.shape[0] = length;
    op::cuda::attention_decode<BF16>(context, query, live_keys, live_values, eager_view,
                                     {scratch.data(), {5 * branch_elements}, {1}});
    lengths.write({static_cast<int>(length), static_cast<int>(length)}, stream.value);
    check(cudaGraphLaunch(graph.executable, stream.value));
    const auto expected = reference(q, k, v, length);
    const auto eager_values = eager.read(stream.value),
               graph_values = graph_output.read(stream.value);
    compare(eager_values, expected, "Eager/scalar FP32 gold", length);
    compare(graph_values, expected, "Graph/scalar FP32 gold", length);
    compare(eager_values, graph_values, "Eager/graph partition", length);
    check_branch_maxima(scratch.read(stream.value), q, k, length);
  }
}

}  // namespace

int main() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
    std::cout << "No CUDA device; skipping attention precision test\n";
    return 77;
  }
  try {
    test_attention();
    std::cout << "CUDA attention precision test passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
