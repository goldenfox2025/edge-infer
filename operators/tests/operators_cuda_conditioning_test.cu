#include "operators/cuda/conditioning.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

void check_cuda(cudaError_t status) {
  if (status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(status));
  }
}

void check_cublas(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("Test cuBLAS error " + std::to_string(status));
  }
}

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

class Stream {
 public:
  Stream() { check_cuda(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
  ~Stream() { cudaStreamDestroy(value); }
  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;
  cudaStream_t value = nullptr;
};

class BlasHandle {
 public:
  BlasHandle() { check_cublas(cublasCreate(&value)); }
  ~BlasHandle() { cublasDestroy(value); }
  BlasHandle(const BlasHandle&) = delete;
  BlasHandle& operator=(const BlasHandle&) = delete;
  cublasHandle_t value = nullptr;
};

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t size) : size_(size) {
    if (size_ != 0) {
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_), size_ * sizeof(T)));
    }
  }
  ~DeviceBuffer() { cudaFree(data_); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  void write(const std::vector<T>& values, cudaStream_t stream) {
    require(values.size() == size_, "Test upload extent mismatch");
    if (size_ != 0) {
      check_cuda(cudaMemcpyAsync(data_, values.data(), size_ * sizeof(T),
                                 cudaMemcpyHostToDevice, stream));
    }
  }

  std::vector<T> read(cudaStream_t stream) const {
    check_cuda(cudaStreamSynchronize(stream));
    std::vector<T> values(size_);
    if (size_ != 0) {
      check_cuda(cudaMemcpy(values.data(), data_, size_ * sizeof(T),
                             cudaMemcpyDeviceToHost));
    }
    return values;
  }

  op::ArrayView<T> view() { return {data_, size_}; }
  op::ArrayView<const T> const_view() const { return {data_, size_}; }

 private:
  T* data_ = nullptr;
  std::size_t size_;
};

template <typename Function>
void expect_throw(Function&& function, const char* message) {
  try {
    function();
  } catch (const std::runtime_error&) {
    return;
  }
  throw std::runtime_error(message);
}

template <typename T>
void expect_near(const std::vector<T>& actual, const std::vector<T>& expected,
                 const char* operation) {
  require(actual.size() == expected.size(), "Test result extent mismatch");
  const float tolerance = std::is_same_v<T, float> ? 2e-5f : 0.008f;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const float observed = static_cast<float>(actual[i]);
    const float reference = static_cast<float>(expected[i]);
    if (!std::isfinite(observed) ||
        std::fabs(observed - reference) > tolerance * (1.0f + std::fabs(reference))) {
      throw std::runtime_error(std::string(operation) + " mismatch at " +
                               std::to_string(i) + ": observed=" +
                               std::to_string(observed) + ", expected=" +
                               std::to_string(reference));
    }
  }
}

template <typename T>
void test_linear(cudaStream_t stream, cublasHandle_t handle,
                 std::size_t rows, std::size_t in, std::size_t out, bool with_bias) {
  std::vector<T> input(rows * in), weights(out * in), bias(out);
  for (std::size_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<T>((static_cast<int>(i % 11) - 5) * 0.125f);
  }
  for (std::size_t i = 0; i < weights.size(); ++i) {
    weights[i] = static_cast<T>((static_cast<int>(i % 13) - 6) * 0.0625f);
  }
  for (std::size_t i = 0; i < bias.size(); ++i) {
    bias[i] = static_cast<T>((static_cast<int>(i % 7) - 3) * 0.25f);
  }
  std::vector<T> expected(rows * out);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t col = 0; col < out; ++col) {
      float value = 0.0f;
      for (std::size_t k = 0; k < in; ++k) {
        value += static_cast<float>(input[row * in + k]) *
                 static_cast<float>(weights[col * in + k]);
      }
      if (with_bias) {
        value += static_cast<float>(bias[col]);
      }
      expected[row * out + col] = static_cast<T>(value);
    }
  }

  DeviceBuffer<T> d_input(input.size()), d_weights(weights.size()),
      d_bias(bias.size()), d_output(expected.size());
  DeviceBuffer<float> scratch(expected.size() + 3);
  d_input.write(input, stream);
  d_weights.write(weights, stream);
  d_bias.write(bias, stream);
  op::cuda::linear<T>(d_input.const_view(), d_weights.const_view(),
                     with_bias ? d_bias.const_view() : op::ArrayView<const T>{},
                     d_output.view(), scratch.view(), rows, in, out, handle, stream);
  cudaStream_t active_stream = nullptr;
  check_cublas(cublasGetStream(handle, &active_stream));
  require(active_stream == stream, "Linear must set the supplied handle stream");
  expect_near(d_output.read(stream), expected, "linear");
}

void test_bf16_bias_rounding(cudaStream_t stream, cublasHandle_t handle) {
  using T = __nv_bfloat16;
  // BF16 ulp at 1 is 1/128. Rounding the GEMM output before adding -1 would
  // erase this half-ulp, whereas biased FP32 accumulation must preserve it.
  const std::vector<T> input = {static_cast<T>(1.0f), static_cast<T>(1.0f)};
  const std::vector<T> weights = {static_cast<T>(1.0f), static_cast<T>(1.0f / 256.0f)};
  const std::vector<T> bias = {static_cast<T>(-1.0f)};
  DeviceBuffer<T> d_input(2), d_weights(2), d_bias(1), output(1);
  DeviceBuffer<float> scratch(1);
  d_input.write(input, stream);
  d_weights.write(weights, stream);
  d_bias.write(bias, stream);
  op::cuda::linear<T>(d_input.const_view(), d_weights.const_view(), d_bias.const_view(),
                     output.view(), scratch.view(), 1, 2, 1, handle, stream);
  require(static_cast<float>(output.read(stream)[0]) == 1.0f / 256.0f,
          "BF16 linear must add bias before rounding the GEMM output");
}

template <typename T>
void test_embedding_sum(cudaStream_t stream, std::size_t frames,
                        std::size_t features, std::size_t books) {
  std::vector<std::vector<T>> host_tables(books);
  std::vector<std::unique_ptr<DeviceBuffer<T>>> buffers;
  std::vector<op::cuda::EmbeddingTable<T>> descriptors;
  std::vector<uint32_t> ids(frames * books);
  for (std::size_t book = 0; book < books; ++book) {
    const std::size_t vocab = book == 0 ? 7 : 5;
    auto& table = host_tables[book];
    table.resize(vocab * features);
    for (std::size_t i = 0; i < table.size(); ++i) {
      table[i] = static_cast<T>((static_cast<int>((i + book * 7) % 23) - 11) *
                                0.015625f);
    }
    buffers.push_back(std::make_unique<DeviceBuffer<T>>(table.size()));
    buffers.back()->write(table, stream);
    descriptors.push_back({buffers.back()->const_view(), vocab});
    for (std::size_t frame = 0; frame < frames; ++frame) {
      ids[frame * books + book] = static_cast<uint32_t>((frame + book) % vocab);
    }
  }

  std::vector<T> expected(frames * features);
  for (std::size_t frame = 0; frame < frames; ++frame) {
    for (std::size_t feature = 0; feature < features; ++feature) {
      float value = 0.0f;
      for (std::size_t book = 0; book < books; ++book) {
        value += static_cast<float>(
            host_tables[book][ids[frame * books + book] * features + feature]);
      }
      expected[frame * features + feature] = static_cast<T>(value);
    }
  }
  DeviceBuffer<T> output(expected.size());
  DeviceBuffer<float> scratch(expected.size() + 5);
  const std::vector<float> unwritten(scratch.view().size,
                                     std::numeric_limits<float>::quiet_NaN());
  scratch.write(unwritten, stream);
  op::cuda::sum_embeddings<T>({descriptors.data(), descriptors.size()},
                              {ids.data(), ids.size()}, output.view(),
                              scratch.view(), frames, features, stream);
  // IDs and descriptors are launch arguments, not asynchronous host operands.
  std::fill(ids.begin(), ids.end(), std::numeric_limits<uint32_t>::max());
  descriptors.clear();
  expect_near(output.read(stream), expected, "sum_embeddings");
}

void test_bf16_sum_rounding(cudaStream_t stream) {
  using T = __nv_bfloat16;
  const std::vector<T> table0 = {static_cast<T>(1.0f)};
  const std::vector<T> table1 = {static_cast<T>(1.0f / 256.0f)};
  const std::vector<T> table2 = {static_cast<T>(-1.0f)};
  DeviceBuffer<T> d0(1), d1(1), d2(1), output(1);
  DeviceBuffer<float> scratch(1);
  d0.write(table0, stream);
  d1.write(table1, stream);
  d2.write(table2, stream);
  const op::cuda::EmbeddingTable<T> tables[] = {
      {d0.const_view(), 1}, {d1.const_view(), 1}, {d2.const_view(), 1}};
  const uint32_t ids[] = {0, 0, 0};
  op::cuda::sum_embeddings<T>({tables, 3}, {ids, 3}, output.view(),
                              scratch.view(), 1, 1, stream);
  require(static_cast<float>(output.read(stream)[0]) == 1.0f / 256.0f,
          "BF16 embedding sum must round once after all codebooks");
}

void test_validation(cudaStream_t stream, cublasHandle_t handle) {
  op::cuda::linear<float>({}, {}, {}, {}, {}, 0, 0, 0, nullptr, stream);
  op::cuda::sum_embeddings<float>({}, {}, {}, {}, 0, 0, stream);
  op::cuda::linear<__nv_bfloat16>({}, {}, {}, {}, {}, 0, 0, 0, nullptr, stream);
  op::cuda::sum_embeddings<__nv_bfloat16>({}, {}, {}, {}, 0, 0, stream);

  const std::vector<float> inputs = {1, 2, 3};
  const std::vector<float> weights = {1, 2, 3, 4, 5, 6};
  const std::vector<float> bias = {1, 2};
  const std::vector<float> sentinel = {-99, -99};
  DeviceBuffer<float> d_input(3), d_weight(6), d_bias(2), output(2), scratch(2);
  d_input.write(inputs, stream);
  d_weight.write(weights, stream);
  d_bias.write(bias, stream);
  output.write(sentinel, stream);
  auto valid_linear = [&] {
    op::cuda::linear<float>(d_input.const_view(), d_weight.const_view(),
                           d_bias.const_view(), output.view(), scratch.view(),
                           1, 3, 2, handle, stream);
  };
  expect_throw([&] {
    op::cuda::linear<float>(d_input.const_view(), {d_weight.const_view().data, 5},
        d_bias.const_view(), output.view(), scratch.view(), 1, 3, 2, handle, stream);
  }, "Linear must reject invalid weight extents");
  expect_throw([&] {
    op::cuda::linear<float>(d_input.const_view(), d_weight.const_view(),
        {d_bias.const_view().data, 1}, output.view(), scratch.view(),
        1, 3, 2, handle, stream);
  }, "Linear must reject invalid bias extents");
  expect_throw([&] {
    op::cuda::linear<float>(d_input.const_view(), d_weight.const_view(),
        d_bias.const_view(), output.view(), {scratch.view().data, 1},
        1, 3, 2, handle, stream);
  }, "Linear must reject undersized float scratch");
  check_cublas(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
  expect_throw(valid_linear, "Linear must reject device pointer mode");
  check_cublas(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
  check_cublas(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
  expect_throw(valid_linear, "Linear must reject reduced-precision math mode");
  check_cublas(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
  require(output.read(stream) == sentinel, "Validation must occur before any writes");

  const op::cuda::EmbeddingTable<float> tables[] = {{d_weight.const_view(), 3}};
  const uint32_t bad_id[] = {3};
  expect_throw([&] {
    op::cuda::sum_embeddings<float>({tables, 1}, {bad_id, 1}, output.view(),
                                    scratch.view(), 1, 2, stream);
  }, "Embedding sum must reject out-of-vocabulary IDs");
  const uint32_t good_id[] = {0};
  expect_throw([&] {
    op::cuda::sum_embeddings<float>({tables, 1}, {good_id, 0}, output.view(),
                                    scratch.view(), 1, 2, stream);
  }, "Embedding sum must reject ID extent mismatch");
  const op::cuda::EmbeddingTable<float> bad_tables[] = {{d_weight.const_view(), 4}};
  expect_throw([&] {
    op::cuda::sum_embeddings<float>({bad_tables, 1}, {good_id, 1}, output.view(),
                                    scratch.view(), 1, 2, stream);
  }, "Embedding sum must reject table extent mismatch");
  require(output.read(stream) == sentinel, "Bad embedding IDs must not launch writes");

  const std::vector<float> scratch_sentinel = {-31, -31};
  scratch.write(scratch_sentinel, stream);
  const op::cuda::EmbeddingTable<float> two_tables[] = {
      {d_weight.const_view(), 3}, {d_weight.const_view(), 3}};
  const uint32_t late_bad_ids[] = {0, 3};
  expect_throw([&] {
    op::cuda::sum_embeddings<float>({two_tables, 2}, {late_bad_ids, 2}, output.view(),
                                    scratch.view(), 1, 2, stream);
  }, "All codebooks must be validated before accumulation begins");
  require(scratch.read(stream) == scratch_sentinel,
          "A later invalid ID must not cause partial scratch writes");

  expect_throw([&] {
    op::cuda::linear<float>({}, {}, {}, {}, {},
        std::numeric_limits<std::size_t>::max(), 2, 1, nullptr, stream);
  }, "Linear must reject dimension multiplication overflow");
  const op::cuda::EmbeddingTable<float> overflow_table[] = {
      {{}, std::numeric_limits<std::size_t>::max()}};
  expect_throw([&] {
    op::cuda::sum_embeddings<float>({overflow_table, 1}, {good_id, 1}, output.view(),
                                    scratch.view(), 1, 2, stream);
  }, "Embedding sum must reject table multiplication overflow");
}

void test_float_scratch_alias(cudaStream_t stream, cublasHandle_t handle) {
  const std::vector<float> input = {2, 3};
  const std::vector<float> weights = {4, 5};
  const std::vector<float> bias = {1};
  DeviceBuffer<float> di(2), dw(2), db(1), output_and_scratch(1);
  di.write(input, stream);
  dw.write(weights, stream);
  db.write(bias, stream);
  op::cuda::linear<float>(di.const_view(), dw.const_view(), db.const_view(),
      output_and_scratch.view(), output_and_scratch.view(), 1, 2, 1, handle, stream);
  require(output_and_scratch.read(stream)[0] == 24.0f,
          "Float output may exactly alias float scratch");
  const op::cuda::EmbeddingTable<float> tables[] = {{dw.const_view(), 2}};
  const uint32_t ids[] = {1};
  op::cuda::sum_embeddings<float>({tables, 1}, {ids, 1},
      output_and_scratch.view(), output_and_scratch.view(), 1, 1, stream);
  require(output_and_scratch.read(stream)[0] == 5.0f,
          "Float embedding output may exactly alias scratch");
}

}  // namespace

int main() {
  try {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
      std::cout << "No CUDA device available; skipping conditioning GPU tests\n";
      return 77;
    }
    Stream stream;
    BlasHandle handle;
    test_linear<float>(stream.value, handle.value, 3, 7, 9, true);
    test_linear<float>(stream.value, handle.value, 5, 18, 17, false);
    test_linear<__nv_bfloat16>(stream.value, handle.value, 3, 7, 9, true);
    test_linear<__nv_bfloat16>(stream.value, handle.value, 5, 18, 17, false);
    test_bf16_bias_rounding(stream.value, handle.value);
    test_embedding_sum<float>(stream.value, 3, 7, 16);
    test_embedding_sum<__nv_bfloat16>(stream.value, 3, 7, 16);
    test_embedding_sum<float>(stream.value, 65, 257, 16);
    test_embedding_sum<__nv_bfloat16>(stream.value, 65, 257, 16);
    test_bf16_sum_rounding(stream.value);
    test_validation(stream.value, handle.value);
    test_float_scratch_alias(stream.value, handle.value);
    std::cout << "CUDA conditioning tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
