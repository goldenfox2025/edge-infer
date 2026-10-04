#pragma once

#include <cublas_v2.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "tensor_view.hpp"
#include "operators/cuda/direct.hpp"

namespace op::cuda {

// Borrowed CUDA resources and launch limits. Preparation queries the current
// device once. The owner serializes handle use and keeps all buffers alive
// until this stream completes. Views carry no ownership or device dispatch.
struct ExecutionContext {
  cublasHandle_t handle = nullptr;
  cudaStream_t stream = nullptr;
  int device = 0;
  int multiprocessors = 1;
  int max_threads_per_block = 1024;
  std::size_t shared_memory_limit = 0;
  const void* attention_padding = nullptr;
};

ExecutionContext prepare_execution_context(cublasHandle_t handle,
                                           cudaStream_t stream);
// Bind once at the start of a decoder submission, including graph capture.
void bind_execution_context(const ExecutionContext& context);

template <typename T>
struct DenseLinearWeight {
  const T* data = nullptr;
  const T* bias = nullptr;
  int input_features = 0;
  int output_features = 0;
  int leading_dimension = 0;
  cublasOperation_t operation = CUBLAS_OP_T;
};

template <typename T>
struct AwqLinearWeight {
  const int32_t* qweight = nullptr;
  const T* scales = nullptr;
  const int32_t* zeros = nullptr;
  const T* bias = nullptr;
  int input_features = 0;
  int output_features = 0;
  int group_size = 0;
  int padded_groups = 0;
};

template <typename T>
DenseLinearWeight<T> prepare_dense(TensorView<const T, 2> weight,
                                  TensorView<const T, 1> bias = {});
template <typename T>
AwqLinearWeight<T> prepare_awq(TensorView<const int32_t, 2> qweight,
                               TensorView<const T, 2> scales,
                               TensorView<const int32_t, 2> zeros,
                               std::size_t group_size,
                               std::size_t input_features,
                               TensorView<const T, 1> bias = {});
template <typename T>
void linear(const ExecutionContext& context, TensorView<const T, 2> input,
             const DenseLinearWeight<T>& weight, TensorView<T, 2> output);
template <typename T>
void awq_linear(const ExecutionContext& context, TensorView<const T, 2> input,
                 const AwqLinearWeight<T>& weight, TensorView<T, 2> output);

template <typename T>
void add(const ExecutionContext& context, ArrayView<const T> a,
          ArrayView<const T> b, ArrayView<T> output);
template <typename T>
inline void multiply(const ExecutionContext& context, ArrayView<const T> a,
                      ArrayView<const T> b, ArrayView<T> output) {
  multiply<T>(a, b, output, context.stream);
}
template <typename T>
inline void silu(const ExecutionContext& context, ArrayView<const T> input,
                  ArrayView<T> output) {
  silu<T>(input, output, context.stream);
}
template <typename T>
inline void rms_norm(const ExecutionContext& context, ArrayView<const T> input,
                      ArrayView<const T> weight, ArrayView<T> output,
                      std::size_t rows, std::size_t features, float epsilon) {
  rms_norm<T>(input, weight, output, rows, features, epsilon, context.stream);
}

template <typename T>
void gather(const ExecutionContext& context, TensorView<const uint32_t, 1> ids,
             TensorView<const T, 2> table, TensorView<T, 2> output);
template <typename T>
void rope(const ExecutionContext& context, TensorView<T, 3> input,
           std::size_t offset, float theta);
template <typename T>
void rope_precomputed(const ExecutionContext& context, TensorView<T, 3> input,
                       TensorView<const float, 2> cache,
                       const std::size_t* offsets,
                       const int* pingpong = nullptr);
template <typename T>
void store_kv(const ExecutionContext& context, TensorView<const T, 2> source,
               TensorView<T, 2> capacity, std::size_t offset,
               const std::size_t* device_offset = nullptr);
template <typename T>
void attention_decode(const ExecutionContext& context,
                       TensorView<const T, 3> q, TensorView<const T, 3> k,
                       TensorView<const T, 3> v, TensorView<T, 3> output,
                       TensorView<T, 1> workspace);
template <typename T>
void attention_prefill(const ExecutionContext& context,
                        TensorView<const T, 3> q, TensorView<const T, 3> k,
                        TensorView<const T, 3> v, TensorView<T, 3> output,
                        std::size_t offset);
// K/V describe fixed cache capacity; lengths selects the live extent on device.
template <typename T>
void attention_graph(const ExecutionContext& context,
                      TensorView<const T, 3> q, TensorView<const T, 3> k,
                      TensorView<const T, 3> v, TensorView<T, 3> output,
                      T** branch_outputs, int* lengths, int* pingpong);

// One row of scratch is reused for batch sampling on the same stream.
// Preparation performs the CUB workspace query; submissions allocate nothing.
struct SamplingPlan {
  std::size_t vocabulary = 0;
  std::size_t scaled_offset = 0;
  std::size_t sorted_offset = 0;
  std::size_t indices_offset = 0;
  std::size_t sorted_indices_offset = 0;
  std::size_t sort_offset = 0;
  std::size_t sort_bytes = 0;
  std::size_t total_bytes = 0;
};

SamplingPlan prepare_sampling(const ExecutionContext& context,
                              std::size_t vocabulary);
// Validate request policy before executing a model that will sample its logits.
void validate_sampling_policy(std::size_t vocabulary, float temperature,
                              float top_p, std::size_t top_k, bool has_rng);
template <typename T>
void sample(const ExecutionContext& context, TensorView<const T, 2> logits,
             TensorView<uint32_t, 1> tokens, TensorView<float, 1> probabilities,
             TensorView<unsigned char, 1> workspace, const SamplingPlan& plan,
             float temperature, float top_p, std::size_t top_k,
             curandState* states);
template <typename T>
void token_probability(const ExecutionContext& context,
                       TensorView<const T, 2> logits, std::size_t row,
                       uint32_t token, float* probability);

}  // namespace op::cuda
