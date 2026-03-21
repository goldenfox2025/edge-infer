#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "operators/cpu/matmul_cpu.hpp"
#include "operators/unified_operators.hpp"
#include "tensor.hpp"
#include "weight_tensor.hpp"

using GemmFunc =
    std::function<Tensor<float>(const Tensor<float>&, const Tensor<float>&)>;

struct GemmVersion {
  std::string name;
  GemmFunc func;
  std::string description;
};

static std::map<std::string, GemmVersion> gemm_registry;

static void register_gemm(const std::string& name, GemmFunc func,
                          const std::string& description) {
  gemm_registry[name] = {name, func, description};
}

static void verify_matrix_layout(const Tensor<float>& a, const Tensor<float>& b,
                                 size_t M, size_t K, size_t N) {
  const auto& a_shape = a.sizes();
  const auto& b_shape = b.sizes();

  if (a_shape[a_shape.size() - 1] != K || a_shape[a_shape.size() - 2] != M) {
    throw std::runtime_error("Matrix A layout mismatch");
  }
  if (b_shape[b_shape.size() - 1] != K || b_shape[b_shape.size() - 2] != N) {
    throw std::runtime_error("Matrix B layout mismatch");
  }

  std::cout << "Layout verified: A[M=" << M << ",K=" << K << "], B[N=" << N
            << ",K=" << K << "]\n";
}

static Tensor<float> naive_matmul(const Tensor<float>& a,
                                  const Tensor<float>& b) {
  const auto& as = a.sizes();
  const auto& bs = b.sizes();
  if (as.size() < 2 || bs.size() < 2) {
    throw std::runtime_error("naive_matmul requires rank >= 2");
  }

  const size_t M = as[as.size() - 2];
  const size_t K = as[as.size() - 1];
  const size_t N = bs[bs.size() - 2];
  const size_t K2 = bs[bs.size() - 1];
  if (K != K2) {
    throw std::runtime_error("naive_matmul inner dimensions do not match");
  }

  std::vector<size_t> batch_dims(as.begin(), as.end() - 2);
  std::vector<size_t> batch_dims_b(bs.begin(), bs.end() - 2);
  if (batch_dims != batch_dims_b) {
    throw std::runtime_error("naive_matmul batch dimensions do not match");
  }

  size_t batch_size = 1;
  for (auto dim : batch_dims) {
    batch_size *= dim;
  }

  std::vector<size_t> out_shape = batch_dims;
  out_shape.push_back(M);
  out_shape.push_back(N);
  Tensor<float> out(out_shape);

  const float* a_all = a.data_ptr();
  const float* b_all = b.data_ptr();
  float* c_all = out.data_ptr();

  const size_t a_stride = M * K;
  const size_t b_stride = N * K;
  const size_t c_stride = M * N;

  for (size_t batch = 0; batch < batch_size; ++batch) {
    const float* a_ptr = a_all + batch * a_stride;
    const float* b_ptr = b_all + batch * b_stride;
    float* c_ptr = c_all + batch * c_stride;

    for (size_t m = 0; m < M; ++m) {
      for (size_t n = 0; n < N; ++n) {
        float sum = 0.0f;
        for (size_t k = 0; k < K; ++k) {
          sum += a_ptr[m * K + k] * b_ptr[n * K + k];
        }
        c_ptr[m * N + n] = sum;
      }
    }
  }

  return out;
}

static Tensor<float> direct_operator_matmul(const Tensor<float>& a,
                                            const Tensor<float>& b) {
  Tensor<float> input = a;
  Tensor<float> output({a.sizes()[0], a.sizes()[1], b.sizes()[1]});
  op::WeightTensor<float> weight(&b);
  op::MatmulCPUOperator<float> matmul;
  matmul(&output, &input, weight, nullptr);
  return output;
}

static Tensor<float> unified_operator_matmul(const Tensor<float>& a,
                                             const Tensor<float>& b) {
  Tensor<float> input = a;
  Tensor<float> output({a.sizes()[0], a.sizes()[1], b.sizes()[1]});
  op::UnifiedOperators<float> ops(Device::CPU);
  ops.matmul(&output, &input, op::WeightTensor<float>(&b));
  return output;
}

static double elapsed_ms(const std::function<void()>& fn, int iters) {
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) {
    fn();
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> ms = t1 - t0;
  return ms.count() / iters;
}

static Tensor<float> make_random_tensor(std::vector<size_t> shape,
                                        unsigned seed) {
  size_t total = 1;
  for (auto dim : shape) {
    total *= dim;
  }

  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> data(total);
  for (size_t i = 0; i < total; ++i) {
    data[i] = dist(rng);
  }
  return Tensor<float>(std::move(data), shape);
}

static bool compare_and_report(const Tensor<float>& reference,
                               const Tensor<float>& test,
                               const std::string& test_name,
                               double tolerance = 1e-4) {
  if (reference.sizes() != test.sizes()) {
    throw std::runtime_error("Output shapes mismatch");
  }

  const size_t count = reference.numel();
  const float* ref_ptr = reference.data_ptr();
  const float* test_ptr = test.data_ptr();

  double max_abs = 0.0;
  double mae = 0.0;
  double mse = 0.0;
  size_t mismatches = 0;

  for (size_t i = 0; i < count; ++i) {
    const double diff = std::abs(double(ref_ptr[i]) - double(test_ptr[i]));
    max_abs = std::max(max_abs, diff);
    mae += diff;
    mse += diff * diff;
    if (diff > tolerance) {
      mismatches++;
    }
  }

  mae /= std::max<size_t>(1, count);
  mse = std::sqrt(mse / std::max<size_t>(1, count));

  const bool passed = max_abs <= tolerance;
  std::cout << std::fixed << std::setprecision(6)
            << test_name << " accuracy: max_abs=" << max_abs
            << ", mae=" << mae << ", rmse=" << mse << ", mismatches="
            << mismatches << "/" << count << " ["
            << (passed ? "PASS" : "FAIL") << "]\n";
  return passed;
}

static void benchmark_gemm(const std::string& name, const GemmFunc& func,
                           const Tensor<float>& A, const Tensor<float>& B,
                           const Tensor<float>& reference, size_t batch_size,
                           size_t M, size_t K, size_t N, int warmup,
                           int iters) {
  std::cout << "\n=== " << name << " ===\n";
  auto result = func(A, B);
  const bool ok = compare_and_report(reference, result, name);
  if (!ok) {
    std::cout << "Accuracy failed, skipping perf run\n";
    return;
  }

  for (int i = 0; i < warmup; ++i) {
    volatile auto tmp = func(A, B);
    (void)tmp;
  }

  const double time_ms = elapsed_ms([&]() {
    volatile auto tmp = func(A, B);
    (void)tmp;
  }, iters);

  const double flops =
      2.0 * double(batch_size) * double(M) * double(N) * double(K);
  const double gflops = flops / (time_ms * 1e6);

  std::cout << std::fixed << std::setprecision(3)
            << "Performance: " << time_ms << " ms, " << gflops
            << " GFLOP/s\n";
}

int main(int argc, char** argv) {
  size_t B = 1;
  size_t M = 1024;
  size_t K = 1024;
  size_t N = 1024;
  int warmup = 3;
  int iters = 10;

  if (argc >= 5) {
    B = std::stoul(argv[1]);
    M = std::stoul(argv[2]);
    K = std::stoul(argv[3]);
    N = std::stoul(argv[4]);
  }
  if (argc >= 6) {
    warmup = std::stoi(argv[5]);
  }
  if (argc >= 7) {
    iters = std::stoi(argv[6]);
  }

  std::cout << "=== Unified CPU GEMM Benchmark ===\n";
  std::cout << "B=" << B << ", M=" << M << ", K=" << K << ", N=" << N
            << ", warmup=" << warmup << ", iters=" << iters << "\n\n";

  register_gemm("naive_cpu", naive_matmul, "Naive reference");
  register_gemm("direct_cpu_operator", direct_operator_matmul,
                "Direct MatmulCPUOperator call");
  register_gemm("unified_cpu_operator", unified_operator_matmul,
                "UnifiedOperators wrapper call");

  const std::vector<size_t> a_shape = {B, M, K};
  const std::vector<size_t> b_shape = {B, N, K};
  auto A = make_random_tensor(a_shape, 123);
  auto B_ = make_random_tensor(b_shape, 321);

  verify_matrix_layout(A, B_, M, K, N);

  std::cout << "\nGenerating reference result...\n";
  const auto reference = naive_matmul(A, B_);

  std::vector<std::pair<std::string, double>> results;
  for (const auto& [name, version] : gemm_registry) {
    if (name == "naive_cpu") {
      const double time_ms = elapsed_ms([&]() {
        volatile auto tmp = version.func(A, B_);
        (void)tmp;
      }, iters);
      const double flops =
          2.0 * double(B) * double(M) * double(N) * double(K);
      const double gflops = flops / (time_ms * 1e6);
      std::cout << "\n=== " << name << " ===\n";
      std::cout << std::fixed << std::setprecision(3)
                << "Performance: " << time_ms << " ms, " << gflops
                << " GFLOP/s\n";
      results.push_back({name, gflops});
      continue;
    }

    benchmark_gemm(name, version.func, A, B_, reference, B, M, K, N, warmup,
                   iters);
    const double time_ms = elapsed_ms([&]() {
      volatile auto tmp = version.func(A, B_);
      (void)tmp;
    }, iters);
    const double flops = 2.0 * double(B) * double(M) * double(N) * double(K);
    const double gflops = flops / (time_ms * 1e6);
    results.push_back({name, gflops});
  }

  std::cout << "\n=== Performance Summary ===\n";
  double baseline = 0.0;
  for (const auto& [name, gflops] : results) {
    if (name == "naive_cpu") {
      baseline = gflops;
      break;
    }
  }

  for (const auto& [name, gflops] : results) {
    const double speedup = baseline > 0.0 ? gflops / baseline : 1.0;
    std::cout << std::setw(20) << name << ": " << std::setw(8)
              << std::fixed << std::setprecision(3) << gflops
              << " GFLOP/s (speedup: " << speedup << "x)\n";
  }

  std::cout << "\nThe gap between direct CPU operator and UnifiedOperators "
               "shows wrapper overhead.\n";
  return 0;
}
