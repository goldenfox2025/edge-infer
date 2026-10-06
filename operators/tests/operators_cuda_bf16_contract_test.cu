#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "operators/cuda/execution.hpp"

namespace {
using BF16 = __nv_bfloat16;
constexpr size_t width = 128;

void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
float rounded(float value) {
  return static_cast<float>(static_cast<BF16>(value));
}
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

struct Stream {
  Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
  ~Stream() { cudaStreamDestroy(value); }
  cudaStream_t value = nullptr;
};
template <typename T>
struct Buffer {
  explicit Buffer(size_t count) {
    check(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
  }
  ~Buffer() { cudaFree(data); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  T* data = nullptr;
};
template <typename T>
void upload(Buffer<T>& buffer, const std::vector<T>& values) {
  check(cudaMemcpy(buffer.data, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
}
template <typename T>
std::vector<T> download(Buffer<T>& buffer, size_t count) {
  std::vector<T> values(count);
  check(cudaMemcpy(values.data(), buffer.data, count * sizeof(T), cudaMemcpyDeviceToHost));
  return values;
}

template <typename Function>
void require_rejected(Function&& function, const char* message) {
  try {
    function();
  } catch (const std::runtime_error&) {
    return;
  }
  throw std::runtime_error(message);
}

template <typename T>
void test_silu_multiply() {
  // A non-vector/block extent, offset pointers, mixed signs and cancelling
  // output pairs exercise scalar tails without relying on the old CUDA ops.
  constexpr size_t count = 513;
  constexpr float gate_values[] = {
      -14.0f, -8.75f, -4.8125f, -2.3f, -1.13f, -0.37f, -0.00390625f,
      0.0f, 0.00390625f, 0.37f, 1.13f, 2.3f, 4.8125f, 8.75f, 14.0f,
      1.3125f, 1.3125f};
  constexpr float up_values[] = {
      0.57f, -1.17f, 4.34f, -4.34f, 0.97f, 12.53f, -17.53f,
      -3.34f, 17.53f, -12.53f, -0.97f, 4.34f, -4.34f, 1.17f, -0.57f,
      3.34f, -3.34f};
  constexpr size_t patterns = sizeof(gate_values) / sizeof(gate_values[0]);
  const T guard = static_cast<T>(123.25f);
  std::vector<T> gate(count + 2, guard), up(count + 2, guard), expected(count + 2, guard);
  size_t sensitive = 0;
  for (size_t i = 0; i < count; ++i) {
    gate[i + 1] = static_cast<T>(gate_values[i % patterns]);
    up[i + 1] = static_cast<T>(up_values[i % patterns]);
    const float x = static_cast<float>(gate[i + 1]);
    const float multiplier = static_cast<float>(up[i + 1]);
    // Independent scalar reference: compute SiLU in double, then apply both
    // explicit dtype conversions rather than calling either CUDA operation.
    const float activation = static_cast<float>(
        static_cast<double>(x) / (1.0 + std::exp(-static_cast<double>(x))));
    const T staged = static_cast<T>(activation);
    expected[i + 1] = static_cast<T>(static_cast<float>(staged) * multiplier);
    sensitive += static_cast<float>(expected[i + 1]) !=
                 static_cast<float>(static_cast<T>(activation * multiplier));
  }
  if constexpr (std::is_same_v<T, BF16>)
    require(sensitive > 0, "SiLU multiply fixture must expose intermediate BF16 rounding");
  Buffer<T> g(count + 2), u(count + 2), output(count + 2);
  upload(g, gate);
  upload(u, up);
  upload(output, std::vector<T>(count + 2, guard));
  Stream stream;
  const auto compare = [&](const std::vector<T>& actual) {
    require(std::memcmp(&actual.front(), &guard, sizeof(T)) == 0 &&
                std::memcmp(&actual.back(), &guard, sizeof(T)) == 0,
            "SiLU multiply wrote outside its borrowed extent");
    for (size_t i = 1; i <= count; ++i) {
      if constexpr (std::is_same_v<T, BF16>) {
        require(std::memcmp(&actual[i], &expected[i], sizeof(T)) == 0,
                "SiLU multiply lost intermediate BF16 rounding");
      } else {
        require(std::isfinite(actual[i]) &&
                    std::abs(actual[i] - expected[i]) < 2e-6f * (1.0f + std::abs(expected[i])),
                "FP32 SiLU multiply differs from the scalar reference");
      }
    }
  };
  op::cuda::silu_multiply<T>({g.data + 1, count}, {u.data + 1, count},
                             {output.data + 1, count}, stream.value);
  check(cudaStreamSynchronize(stream.value));
  compare(download(output, count + 2));
  op::cuda::silu_multiply<T>({g.data + 1, count}, {u.data + 1, count},
                             {g.data + 1, count}, stream.value);
  check(cudaStreamSynchronize(stream.value));
  compare(download(g, count + 2));
  require_rejected([&] {
    op::cuda::silu_multiply<T>({g.data, count}, {u.data, count - 1}, {output.data, count}, stream.value);
  }, "SiLU multiply accepted mismatched input extents");
  require_rejected([&] {
    op::cuda::silu_multiply<T>({g.data, count}, {u.data, count}, {output.data, count - 1}, stream.value);
  }, "SiLU multiply accepted a mismatched output extent");
  const size_t too_large = static_cast<size_t>(std::numeric_limits<int>::max()) + 1;
  require_rejected([&] {
    op::cuda::silu_multiply<T>({g.data, too_large}, {u.data, too_large}, {output.data, too_large}, stream.value);
  }, "SiLU multiply accepted an extent exceeding its launch index range");
  op::cuda::silu_multiply<T>({}, {}, {}, stream.value);
  check(cudaGetLastError());
}

template <typename T>
void test_silu_negative_tail() {
  // x=-88 is an observed checkpoint input: its nonzero, normal SiLU result
  // was lost by fast FP32 reciprocal evaluation. Later values exercise true
  // subnormals, including activation products that become normal again. Tiny
  // inputs test activation rounding before multiplication can amplify a loss.
  constexpr float inputs[] = {
      -80.0f, -80.5f, -81.0f, -86.0f, -87.0f, -88.0f, -89.0f,
      -90.0f, -91.0f, -92.0f, -93.0f, -94.0f, -95.0f, -96.0f,
      -97.0f, -98.0f, -99.0f, -100.0f, -0.0f, 0.0f, 88.0f,
      -0x1p-125f, 0x1p-125f, -0x1p-126f, 0x1p-126f,
      -0x1p-127f, 0x1p-127f, -0x1p-133f, 0x1p-133f,
      -0x1p-149f, 0x1p-149f};
  constexpr float multipliers[] = {1.0f, -1.0f, 65536.0f, -65536.0f,
                                    1.0f / 65536.0f, -0.0f, 0.0f};
  constexpr size_t input_count = sizeof(inputs) / sizeof(inputs[0]);
  constexpr size_t multiplier_count = sizeof(multipliers) / sizeof(multipliers[0]);
  constexpr size_t count = input_count * multiplier_count;
  std::vector<T> gate(count), up(count), activations(count), products(count);
  size_t normal_tail = 0, subnormal_tail = 0, amplified_subnormal = 0;
  for (size_t i = 0; i < count; ++i) {
    gate[i] = static_cast<T>(inputs[i / multiplier_count]);
    up[i] = static_cast<T>(multipliers[i % multiplier_count]);
    const double x = static_cast<float>(gate[i]);
    // Independent FP64 oracle uses the original mathematical form. Its
    // denominator remains finite throughout this fixture, unlike FP32 exp.
    const float activation = static_cast<float>(x / (1.0 + std::exp(-x)));
    activations[i] = static_cast<T>(activation);
    const float staged = static_cast<float>(activations[i]);
    const double product = static_cast<double>(staged) * static_cast<float>(up[i]);
    products[i] = static_cast<T>(static_cast<float>(product));
    if (x < -80.0) {
      normal_tail += std::fpclassify(staged) == FP_NORMAL;
      subnormal_tail += std::fpclassify(staged) == FP_SUBNORMAL;
      amplified_subnormal += std::fpclassify(staged) == FP_SUBNORMAL &&
                             std::fpclassify(static_cast<float>(products[i])) == FP_NORMAL;
    }
  }
  require(normal_tail > 0 && subnormal_tail > 0 && amplified_subnormal > 0,
          "SiLU tail fixture must cover normal, subnormal and amplified results");
  const auto compare = [&](const std::vector<T>& actual, const std::vector<T>& expected) {
    for (size_t i = 0; i < count; ++i) {
      const float a = static_cast<float>(actual[i]), e = static_cast<float>(expected[i]);
      require(std::isfinite(a), "SiLU negative tail produced a nonfinite result");
      require(std::signbit(a) == std::signbit(e), "SiLU negative tail lost its output sign");
      if constexpr (std::is_same_v<T, BF16>) {
        if (std::memcmp(&actual[i], &expected[i], sizeof(T)) != 0) {
          unsigned short actual_bits = 0, expected_bits = 0;
          std::memcpy(&actual_bits, &actual[i], sizeof(T));
          std::memcpy(&expected_bits, &expected[i], sizeof(T));
          std::ostringstream message;
          message << "BF16 SiLU tail differs from independent staged FP64 gold at " << i
                  << ": gate=" << std::hexfloat << static_cast<float>(gate[i])
                  << ", up=" << static_cast<float>(up[i]) << ", actual=" << a << ", expected=" << e
                  << ", bits=" << std::hex << actual_bits << "/" << expected_bits;
          throw std::runtime_error(message.str());
        }
      } else {
        // Relative error plus one minimum FP32 step; unlike a unit-scale
        // absolute tolerance, this cannot accept zero for a normal tail value.
        const double bound = 4.0 * std::numeric_limits<float>::epsilon() * std::abs(e) +
                             std::numeric_limits<float>::denorm_min();
        if (std::abs(static_cast<double>(a) - e) > bound) {
          std::ostringstream message;
          message << "FP32 SiLU tail differs from independent FP64 gold at " << i
                  << ": gate=" << std::hexfloat << static_cast<float>(gate[i])
                  << ", up=" << static_cast<float>(up[i]) << ", actual=" << a << ", expected=" << e;
          throw std::runtime_error(message.str());
        }
      }
    }
  };
  Buffer<T> g(count), u(count), output(count), separate(count);
  upload(g, gate);
  upload(u, up);
  Stream stream;
  op::cuda::silu<T>({g.data, count}, {output.data, count}, stream.value);
  check(cudaStreamSynchronize(stream.value));
  compare(download(output, count), activations);
  op::cuda::multiply<T>({output.data, count}, {u.data, count}, {separate.data, count}, stream.value);
  check(cudaStreamSynchronize(stream.value));
  const auto unfused = download(separate, count);
  compare(unfused, products);
  op::cuda::silu_multiply<T>({g.data, count}, {u.data, count}, {output.data, count}, stream.value);
  check(cudaStreamSynchronize(stream.value));
  const auto fused = download(output, count);
  compare(fused, products);
  require(std::memcmp(fused.data(), unfused.data(), count * sizeof(T)) == 0,
          "SiLU fusion changed standalone staged product bits in the negative tail");
  op::cuda::silu_multiply<T>({g.data, count}, {u.data, count}, {g.data, count}, stream.value);
  check(cudaStreamSynchronize(stream.value));
  compare(download(g, count), products);
}

void test_norm() {
  constexpr float eps = 1e-6f;
  std::vector<BF16> input(width), weight(width);
  float sum = 0;
  for (size_t i = 0; i < width; ++i) {
    input[i] = static_cast<BF16>(-0.8f + (i % 7) * 0.25f);
    weight[i] = static_cast<BF16>(0.63f + (i % 5) * 0.17f);
    sum += static_cast<float>(input[i]) * static_cast<float>(input[i]);
  }
  const float inv = 1.0f / std::sqrt(sum / width + eps);
  std::vector<BF16> expected(width);
  size_t sensitive = 0;
  for (size_t i = 0; i < width; ++i) {
    const float x = static_cast<float>(input[i]), w = static_cast<float>(weight[i]);
    expected[i] = static_cast<BF16>(rounded(x * inv) * w);
    sensitive += static_cast<float>(expected[i]) != rounded((x * inv) * w);
  }
  require(sensitive > 0, "RMSNorm fixture must expose the intermediate BF16 rounding");
  Buffer<BF16> x(width), w(width), out(width);
  upload(x, input);
  upload(w, weight);
  op::cuda::rms_norm<BF16>({x.data, width}, {w.data, width}, {out.data, width}, 1, width, eps);
  check(cudaDeviceSynchronize());
  auto actual = download(out, width);
  for (size_t i = 0; i < width; ++i)
    require(static_cast<float>(actual[i]) == static_cast<float>(expected[i]),
            "RMSNorm lost its normalized-activation BF16 rounding");
  // In-place normalization must preserve the same staging.
  op::cuda::rms_norm<BF16>({x.data, width}, {w.data, width}, {x.data, width}, 1, width, eps);
  check(cudaDeviceSynchronize());
  actual = download(x, width);
  for (size_t i = 0; i < width; ++i)
    require(static_cast<float>(actual[i]) == static_cast<float>(expected[i]),
            "In-place BF16 RMSNorm differs");

  std::vector<float> fx(width), fw(width);
  for (size_t i = 0; i < width; ++i) {
    fx[i] = static_cast<float>(input[i]);
    fw[i] = static_cast<float>(weight[i]);
  }
  Buffer<float> dx(width), dw(width), dy(width);
  upload(dx, fx);
  upload(dw, fw);
  op::cuda::rms_norm<float>({dx.data, width}, {dw.data, width}, {dy.data, width}, 1, width, eps);
  check(cudaDeviceSynchronize());
  auto fy = download(dy, width);
  for (size_t i = 0; i < width; ++i)
    require(std::abs(fy[i] - (fx[i] * inv) * fw[i]) < 2e-6f,
            "FP32 RMSNorm must retain FP32 arithmetic");
}

void test_biased_linear() {
  constexpr size_t features = 8, outputs = 2, rows = 2;
  // dot = 1 + 1/256 lies exactly halfway between adjacent BF16 values.
  // Rounding it before adding -1 loses a representable 1/256 residual.
  std::vector<BF16> input(rows * features, BF16{0.0f});
  input[0] = BF16{1.0f}; input[1] = BF16{1.0f / 256};
  input[features] = BF16{-1.0f}; input[features + 1] = BF16{-1.0f / 256};
  std::vector<BF16> physical(outputs * features, BF16{0.0f});
  physical[0] = physical[1] = BF16{1.0f};
  physical[features] = physical[features + 1] = BF16{-1.0f};
  std::vector<BF16> row_major(features * outputs, BF16{0.0f});
  row_major[0] = row_major[2] = BF16{1.0f};
  row_major[1] = row_major[3] = BF16{-1.0f};
  Buffer<BF16> x(input.size()), w(physical.size()), bias(outputs), output(rows * outputs);
  upload(x, input); upload(bias, std::vector<BF16>{BF16{-1.0f}, BF16{1.0f}});
  cublasHandle_t handle = nullptr;
  require(cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS, "Biased linear handle creation failed");
  try {
    const auto ctx = op::cuda::prepare_execution_context(handle, nullptr);
    op::cuda::bind_execution_context(ctx);
    for (bool transposed : {true, false}) {
      upload(w, transposed ? physical : row_major);
      const auto weight = op::cuda::prepare_dense<BF16>(
          {w.data, {features, outputs}, transposed ? std::array<size_t, 2>{1, features}
                                                : std::array<size_t, 2>{outputs, 1}},
          {bias.data, {outputs}, {1}});
      op::cuda::linear<BF16>(ctx, {x.data, {rows, features}, {features, 1}}, weight,
                            {output.data, {rows, outputs}, {outputs, 1}});
      check(cudaDeviceSynchronize());
      const auto actual = download(output, rows * outputs);
      const std::vector<BF16> expected{BF16{1.0f / 256}, BF16{-1.0f / 256}, BF16{-2.0f}, BF16{2.0f}};
      for (size_t i = 0; i < actual.size(); ++i)
        require(static_cast<float>(actual[i]) == static_cast<float>(expected[i]),
                "Dense BF16 linear must add bias before its final rounding");
    }
  } catch (...) {
    cublasDestroy(handle);
    throw;
  }
  cublasDestroy(handle);
}

void test_rope() {
  constexpr size_t position = 3;
  constexpr float theta = 10000.0f;
  std::vector<BF16> input(width), expected(width);
  std::vector<float> cache((position + 1) * width);
  for (size_t i = 0; i < width; ++i) input[i] = static_cast<BF16>(-1.23f + (i % 19) * 0.173f);
  size_t sensitive = 0;
  for (size_t pair = 0; pair < width / 2; ++pair) {
    const float angle = position / std::pow(theta, 2.0f * pair / width);
    const float sine = std::sin(angle), cosine = std::cos(angle);
    cache[position * width + 2 * pair] = sine;
    cache[position * width + 2 * pair + 1] = cosine;
    const float s = rounded(sine), c = rounded(cosine);
    const float x = static_cast<float>(input[pair]),
                y = static_cast<float>(input[pair + width / 2]);
    expected[pair] = static_cast<BF16>(rounded(x * c) - rounded(y * s));
    expected[pair + width / 2] = static_cast<BF16>(rounded(x * s) + rounded(y * c));
    sensitive += static_cast<float>(expected[pair]) != rounded(x * cosine - y * sine);
  }
  require(sensitive > 0, "RoPE fixture must expose staged BF16 products");
  Buffer<BF16> values(width);
  Buffer<float> trig(cache.size());
  Buffer<size_t> offsets(1);
  upload(values, input);
  upload(trig, cache);
  upload(offsets, std::vector<size_t>{position});
  const auto ctx = op::cuda::prepare_execution_context(nullptr, nullptr);
  TensorView<BF16, 3> view{values.data, {1, 1, width}, {width, width, 1}};
  op::cuda::rope_precomputed<BF16>(ctx, view, {trig.data, {position + 1, width}, {width, 1}},
                                 0, offsets.data);
  check(cudaDeviceSynchronize());
  auto actual = download(values, width);
  for (size_t i = 0; i < width; ++i)
    require(static_cast<float>(actual[i]) == static_cast<float>(expected[i]),
            "Cached RoPE lost BF16 trigonometric/product/sum rounding");
  upload(values, input);
  op::cuda::rope<BF16>(ctx, view, position, theta);
  check(cudaDeviceSynchronize());
  actual = download(values, width);
  for (size_t i = 0; i < width; ++i)
    require(static_cast<float>(actual[i]) == static_cast<float>(expected[i]),
            "Eager and cached BF16 RoPE staging differs");
}

void test_small_angle_rope() {
  constexpr size_t positions = 21;
  constexpr float theta = 1000000.0f;
  std::vector<BF16> basis(width, BF16{0.0f});
  for (size_t i = 0; i < width / 2; ++i) basis[i] = BF16{1.0f};
  std::vector<float> table(positions * width);
  for (size_t p = 0; p < positions; ++p)
    for (size_t i = 0; i < width / 2; ++i) {
      const float frequency = 1.0f / std::pow(theta, 2.0f * static_cast<float>(i) / width);
      const float angle = static_cast<float>(p) * frequency;
      table[p * width + 2 * i] = std::sin(angle);
      table[p * width + 2 * i + 1] = std::cos(angle);
    }
  Buffer<BF16> values(width);
  Buffer<float> trig(table.size());
  upload(trig, table);
  const auto ctx = op::cuda::prepare_execution_context(nullptr, nullptr);
  const TensorView<BF16, 3> view{values.data, {1, 1, width}, {width, width, 1}};
  for (size_t p = 0; p < positions; ++p) {
    upload(values, basis);
    op::cuda::rope<BF16>(ctx, view, p, theta);
    check(cudaDeviceSynchronize());
    const auto eager = download(values, width);
    upload(values, basis);
    op::cuda::rope_precomputed<BF16>(ctx, view, {trig.data, {positions, width}, {width, 1}}, p);
    check(cudaDeviceSynchronize());
    const auto cached = download(values, width);
    for (size_t i = 0; i < width / 2; ++i) {
      const double angle = static_cast<double>(p) /
                           std::pow(static_cast<double>(theta), 2.0 * i / width);
      const float sine = rounded(static_cast<float>(std::sin(angle)));
      const float cosine = rounded(static_cast<float>(std::cos(angle)));
      require(static_cast<float>(eager[i]) == cosine &&
                  static_cast<float>(eager[i + width / 2]) == sine &&
                  static_cast<float>(cached[i]) == cosine &&
                  static_cast<float>(cached[i + width / 2]) == sine,
              "RoPE basis must preserve BF16-rounded FP64 trigonometry at small angles");
    }
  }
  bool rejected = false;
  try {
    op::cuda::rope_precomputed<BF16>(ctx, view, {trig.data, {positions, width}, {width, 1}}, positions);
  } catch (const std::out_of_range&) { rejected = true; }
  require(rejected, "Cached RoPE must reject positions beyond its table before launch");
}
}  // namespace

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
  try {
    test_norm();
    test_rope();
    test_small_angle_rope();
    test_biased_linear();
    test_silu_multiply<BF16>();
    test_silu_multiply<float>();
    test_silu_negative_tail<float>();
    test_silu_negative_tail<BF16>();
    std::cout << "BF16 RMSNorm, RoPE and fused SiLU multiply rounding contracts passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
