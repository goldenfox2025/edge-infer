#include <cmath>
#include <cstdlib>
#include <iostream>

#include "operators/unified_operators.hpp"

namespace {

void expect_true(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "test failure: " << message << std::endl;
    std::exit(1);
  }
}

void expect_near(float actual, float expected, float eps,
                 const char* message) {
  if (std::fabs(actual - expected) > eps) {
    std::cerr << "test failure: " << message << ", expected=" << expected
              << ", actual=" << actual << std::endl;
    std::exit(1);
  }
}

}  // namespace

int main() {
  op::UnifiedOperators<float> ops(Device::CPU);

  Tensor<float> a(std::vector<float>{1.0f, 2.0f, 3.0f}, {3});
  Tensor<float> b(std::vector<float>{4.0f, 5.0f, 6.0f}, {3});
  Tensor<float> out({3}, Device::CPU);
  ops.add(&out, &a, &b);
  expect_near(out.data_ptr()[0], 5.0f, 1e-6f, "add[0]");
  expect_near(out.data_ptr()[2], 9.0f, 1e-6f, "add[2]");

  Tensor<float> norm_input(std::vector<float>{1.0f, 3.0f, 0.0f, 0.0f}, {2, 2});
  Tensor<float> norm_weight(std::vector<float>{2.0f, 0.5f}, {2});
  ops.rms_norm(&norm_input, &norm_input, &norm_weight, 1e-6f);
  expect_near(norm_input.data_ptr()[0], 0.8944271f, 1e-5f, "static CPU RMSNorm feature scale");
  expect_near(norm_input.data_ptr()[1], 0.6708203f, 1e-5f, "static CPU RMSNorm second scale");
  expect_near(norm_input.data_ptr()[2], 0.0f, 1e-6f, "static CPU RMSNorm zero row");

  Tensor<float> silu_input(std::vector<float>{1.0f, 2.0f}, {2});
  Tensor<float> mul_input(std::vector<float>{3.0f, 4.0f}, {2});
  Tensor<float> silu_out({2}, Device::CPU);
  ops.silu_multiply(&silu_out, &silu_input, &mul_input);
  expect_true(silu_out.data_ptr()[0] > 2.19f && silu_out.data_ptr()[0] < 2.20f,
              "silu_multiply[0]");
  expect_true(silu_out.data_ptr()[1] > 7.04f && silu_out.data_ptr()[1] < 7.05f,
              "silu_multiply[1]");

  Tensor<float> logits(std::vector<float>{0.0f, 2.0f, 1.0f}, {3});
  const uint32_t greedy = ops.sample_cpu(std::move(logits), 1.0f, 1.0f, 1);
  expect_true(greedy == 1, "sample_cpu greedy");

  Tensor<float> prob_logits(std::vector<float>{0.0f, 2.0f, 1.0f}, {1, 3});
  const float prob = ops.get_token_probability(prob_logits, 0, 1);
  expect_true(prob > 0.66f && prob < 0.67f, "token probability");

  Tensor<float> softmax_in(std::vector<float>{0.0f, 2.0f, 1.0f}, {1, 3});
  Tensor<float> softmax_out({1, 3}, Device::CPU);
  ops.softmax(&softmax_out, &softmax_in, 1);
  const float sum = softmax_out.data_ptr()[0] + softmax_out.data_ptr()[1] +
                    softmax_out.data_ptr()[2];
  expect_near(sum, 1.0f, 1e-5f, "softmax normalization");

  std::cout << "unified_operators_cpu_test passed" << std::endl;
  return 0;
}
