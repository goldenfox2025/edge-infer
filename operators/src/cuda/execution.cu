#include "operators/cuda/execution.hpp"
#include "operators/cuda/execution_kernels.cuh"

#include <cub/cub.cuh>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace op::cuda {
namespace {

void check_cuda(cudaError_t result) {
  if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
void check_blas(cublasStatus_t result) {
  if (result != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("Direct cuBLAS submission failed: " + std::to_string(result));
  }
}
int checked_int(std::size_t value) {
  if (!value || value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("CUDA execution extent must be positive and fit int");
  }
  return static_cast<int>(value);
}
template <typename T, std::size_t Rank>
void require_dense_view(TensorView<T, Rank> view) {
  if (!view.data || !view.numel() || !view.is_contiguous()) {
    throw std::invalid_argument("CUDA execution requires a nonempty contiguous view");
  }
}
template <typename T>
void require_head_view(TensorView<T, 3> view) {
  if (!view.data || !view.numel() || view.stride[2] != 1 ||
      view.stride[1] != view.shape[2] ||
      view.stride[0] < view.shape[1] * view.shape[2]) {
    throw std::invalid_argument("CUDA attention/RoPE requires contiguous heads within each row");
  }
  checked_int(view.shape[0]); checked_int(view.shape[1]); checked_int(view.shape[2]);
}
template <typename T>
bool aligned_attention_view(TensorView<T, 3> view) {
  return reinterpret_cast<std::uintptr_t>(view.data) % 16 == 0 &&
         (view.stride[0] * sizeof(T)) % 16 == 0;
}
template <typename T>
void require_attention(TensorView<const T, 3> q, TensorView<const T, 3> k,
                       TensorView<const T, 3> v, TensorView<T, 3> output) {
  require_head_view(q); require_head_view(k); require_head_view(v); require_head_view(output);
  if (k.shape != v.shape || q.shape != output.shape ||
      q.shape[2] != k.shape[2] || q.shape[1] % k.shape[1] ||
      q.shape[2] > 1024) {
    throw std::invalid_argument("CUDA attention query, KV and output shapes are incompatible");
  }
}

template <typename T>
__global__ void broadcast_bias_kernel(T* output, const T* bias, int rows, int width) {
  const std::size_t count = static_cast<std::size_t>(rows) * width;
  const std::size_t start = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t step = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i = start; i < count; i += step) output[i] = bias[i % width];
}
template <typename T>
__global__ void gather_rows_kernel(const uint32_t* ids, const T* table, T* output,
                                    std::size_t count, std::size_t width,
                                    std::size_t vocabulary) {
  const std::size_t start = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t step = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i = start; i < count; i += step) {
    const uint32_t token = ids[i / width];
    output[i] = token < vocabulary ? table[static_cast<std::size_t>(token) * width + i % width]
                                   : static_cast<T>(0.0f);
  }
}
template <typename T>
__global__ void rotate_kernel(T* data, std::size_t row_stride, std::size_t head_dim,
                               std::size_t offset, float theta, const float* cache,
                               std::size_t cache_stride, const std::size_t* offsets,
                               const int* pingpong) {
  const std::size_t pair = threadIdx.x;
  const std::size_t half = head_dim / 2;
  if (pair >= half) return;
  const std::size_t position = blockIdx.x + (offsets ? offsets[pingpong ? *pingpong : 0] : offset);
  float sine, cosine;
  if (cache) {
    sine = cache[position * cache_stride + pair * 2];
    cosine = cache[position * cache_stride + pair * 2 + 1];
  } else {
    const float exponent = (2.0f * pair) / static_cast<float>(head_dim);
    const float power = static_cast<float>(pow(static_cast<double>(theta),
                                              static_cast<double>(exponent)));
    const float frequency = __fdiv_rn(1.0f, power);
    const float angle = __fmul_rn(static_cast<float>(position), frequency);
    // Fast single-precision sine has an absolute error that loses significant
    // bits for small RoPE angles. Double transcendental calls retain precision
    // under --use_fast_math. Prepared models use their shared table instead.
    sine = static_cast<float>(sin(static_cast<double>(angle)));
    cosine = static_cast<float>(cos(static_cast<double>(angle)));
  }
  T* head = data + static_cast<std::size_t>(blockIdx.x) * row_stride + blockIdx.y * head_dim;
  const float x = static_cast<float>(head[pair]);
  const float y = static_cast<float>(head[pair + half]);
  if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    // BF16 RoPE casts its trigonometric tensors to the activation dtype,
    // rounds each product, then rounds the sum of those BF16 products.
    sine = __bfloat162float(__float2bfloat16_rn(sine));
    cosine = __bfloat162float(__float2bfloat16_rn(cosine));
    const float x_cosine = __bfloat162float(__float2bfloat16_rn(x * cosine));
    const float y_sine = __bfloat162float(__float2bfloat16_rn(y * sine));
    const float x_sine = __bfloat162float(__float2bfloat16_rn(x * sine));
    const float y_cosine = __bfloat162float(__float2bfloat16_rn(y * cosine));
    head[pair] = __float2bfloat16_rn(x_cosine - y_sine);
    head[pair + half] = __float2bfloat16_rn(x_sine + y_cosine);
  } else {
    head[pair] = static_cast<T>(x * cosine - y * sine);
    head[pair + half] = static_cast<T>(x * sine + y * cosine);
  }
}

template <typename T>
__global__ void store_kv_kernel(const T* source, T* cache, std::size_t count,
    std::size_t rows, std::size_t width, std::size_t capacity,
    std::size_t offset, const std::size_t* device_offset) {
  const std::size_t start_row = device_offset ? *device_offset : offset;
  if (start_row > capacity || rows > capacity - start_row) return;
  const std::size_t start = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t step = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i = start; i < count; i += step) cache[start_row * width + i] = source[i];
}

// Stable scalar fallback for FP32 and head widths without a specialized kernel.
// Each block owns one query/head and computes online softmax with FP32 state.
template <typename T>
__global__ void attention_scalar_kernel(const T* q, const T* k, const T* v, T* out,
    int query_heads, int kv_heads, int width, std::size_t q_stride,
    std::size_t k_stride, std::size_t v_stride, std::size_t output_stride,
    int capacity, std::size_t offset, const int* lengths, const int* pingpong) {
  const int dimension = threadIdx.x;
  const int head = blockIdx.y;
  const int query = blockIdx.x;
  const int kv_head = head / (query_heads / kv_heads);
  const int live = lengths ? lengths[pingpong ? *pingpong : 0] : capacity;
  const int end = lengths ? live : min(capacity, static_cast<int>(offset) + query + 1);
  __shared__ float partial[32];
  __shared__ float score;
  float maximum = -INFINITY, sum = 0.0f, value = 0.0f;
  for (int token = 0; token < end && token < capacity; ++token) {
    float product = dimension < width
        ? static_cast<float>(q[query * q_stride + head * width + dimension]) *
          static_cast<float>(k[token * k_stride + kv_head * width + dimension]) : 0.0f;
    for (int distance = 16; distance; distance >>= 1)
      product += __shfl_down_sync(0xffffffff, product, distance);
    if (dimension % 32 == 0) partial[dimension / 32] = product;
    __syncthreads();
    if (dimension < 32) {
      float total = dimension < blockDim.x / 32 ? partial[dimension] : 0.0f;
      for (int distance = 16; distance; distance >>= 1)
        total += __shfl_down_sync(0xffffffff, total, distance);
      if (dimension == 0) score = total * rsqrtf(static_cast<float>(width));
    }
    __syncthreads();
    const float next_maximum = fmaxf(maximum, score);
    const float previous_scale = __expf(maximum - next_maximum);
    const float current_scale = __expf(score - next_maximum);
    sum = sum * previous_scale + current_scale;
    if (dimension < width)
      value = value * previous_scale + current_scale *
              static_cast<float>(v[token * v_stride + kv_head * width + dimension]);
    maximum = next_maximum;
    __syncthreads();
  }
  if (dimension < width)
    out[query * output_stride + head * width + dimension] =
        static_cast<T>(sum > 0.0f ? value / sum : 0.0f);
}
template <typename T>
void launch_scalar_attention(const ExecutionContext& context,
    TensorView<const T, 3> q, TensorView<const T, 3> k,
    TensorView<const T, 3> v, TensorView<T, 3> output,
    std::size_t offset, const int* lengths = nullptr, const int* pingpong = nullptr) {
  const int threads = static_cast<int>((q.shape[2] + 31) / 32 * 32);
  attention_scalar_kernel<T><<<dim3(q.shape[0], q.shape[1]), threads, 0, context.stream>>>(
      q.data, k.data, v.data, output.data, q.shape[1], k.shape[1], q.shape[2],
      q.stride[0], k.stride[0], v.stride[0], output.stride[0], k.shape[0],
      offset, lengths, pingpong);
  check_cuda(cudaGetLastError());
}

template <typename T>
__global__ void scaled_indices_kernel(const T* logits, float* scaled, int* indices,
                                       int vocabulary, float temperature) {
  const std::size_t start = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t step = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i = start; i < static_cast<std::size_t>(vocabulary); i += step) {
    const float value = static_cast<float>(logits[i]);
    scaled[i] = isnan(value) ? -INFINITY : value / temperature;
    indices[i] = static_cast<int>(i);
  }
}
__global__ void sample_sorted_kernel(const float* sorted, const int* indices,
                                      int top_k, float top_p, bool greedy,
                                      curandState* state, uint32_t* token,
                                      float* probability) {
  if (threadIdx.x || blockIdx.x) return;
  if (greedy || !isfinite(sorted[0])) {
    *token = static_cast<uint32_t>(indices[0]);
    if (probability) *probability = 1.0f;
    return;
  }
  float total = 0.0f;
  for (int i = 0; i < top_k; ++i) total += __expf(sorted[i] - sorted[0]);
  const float threshold = top_p * total;
  float retained = 0.0f;
  int candidates = 0;
  do {
    retained += __expf(sorted[candidates] - sorted[0]);
    ++candidates;
  } while (candidates < top_k && retained < threshold);
  curandState current = *state;
  const float target = curand_uniform(&current) * retained;
  float cumulative = 0.0f;
  int selected = candidates - 1;
  for (int i = 0; i < candidates; ++i) {
    cumulative += __expf(sorted[i] - sorted[0]);
    if (cumulative >= target) { selected = i; break; }
  }
  *token = static_cast<uint32_t>(indices[selected]);
  if (probability) *probability = __expf(sorted[selected] - sorted[0]) / retained;
  *state = current;
}
template <typename T>
__global__ void probability_kernel(const T* logits, int vocabulary, uint32_t token,
                                    float* output) {
  if (threadIdx.x || blockIdx.x) return;
  float maximum = -INFINITY;
  for (int i = 0; i < vocabulary; ++i) maximum = fmaxf(maximum, static_cast<float>(logits[i]));
  float denominator = 0.0f;
  for (int i = 0; i < vocabulary; ++i) denominator += __expf(static_cast<float>(logits[i]) - maximum);
  *output = __expf(static_cast<float>(logits[token]) - maximum) / denominator;
}

}  // namespace

ExecutionContext prepare_execution_context(cublasHandle_t handle, cudaStream_t stream) {
  ExecutionContext context;
  context.handle = handle; context.stream = stream;
  check_cuda(cudaGetDevice(&context.device));
  cudaDeviceProp properties{};
  check_cuda(cudaGetDeviceProperties(&properties, context.device));
  context.multiprocessors = properties.multiProcessorCount;
  context.max_threads_per_block = properties.maxThreadsPerBlock;
  context.shared_memory_limit = properties.sharedMemPerBlock;
  context.attention_padding = detail::prepare_attention_padding();
  return context;
}
void bind_execution_context(const ExecutionContext& context) {
  if (context.handle) {
    check_blas(cublasSetPointerMode(context.handle, CUBLAS_POINTER_MODE_HOST));
    check_blas(cublasSetStream(context.handle, context.stream));
  }
}

template <typename T>
DenseLinearWeight<T> prepare_dense(TensorView<const T, 2> weight,
                                    TensorView<const T, 1> bias) {
  const int input = checked_int(weight.shape[0]), output = checked_int(weight.shape[1]);
  if (!weight.data || (bias.data && (bias.shape[0] != weight.shape[1] || !bias.is_contiguous())))
    throw std::invalid_argument("Invalid direct dense linear weight or bias");
  DenseLinearWeight<T> prepared;
  prepared.data = weight.data; prepared.bias = bias.data;
  prepared.input_features = input; prepared.output_features = output;
  if (weight.stride[0] == 1 && weight.stride[1] >= weight.shape[0]) {
    prepared.operation = CUBLAS_OP_T;
    prepared.leading_dimension = checked_int(weight.stride[1]);
  } else if (weight.stride[1] == 1 && weight.stride[0] >= weight.shape[1]) {
    prepared.operation = CUBLAS_OP_N;
    prepared.leading_dimension = checked_int(weight.stride[0]);
  } else throw std::invalid_argument("Direct dense linear supports row-major or transposed matrix views");
  return prepared;
}
template <typename T>
AwqLinearWeight<T> prepare_awq(TensorView<const int32_t, 2> qweight,
    TensorView<const T, 2> scales, TensorView<const int32_t, 2> zeros,
    std::size_t group_size, std::size_t input_features, TensorView<const T, 1> bias) {
  require_dense_view(qweight); require_dense_view(scales); require_dense_view(zeros);
  const int input = checked_int(input_features), output = checked_int(qweight.shape[0]);
  const int group = checked_int(group_size), padded = checked_int(scales.shape[1]);
  if (input_features % group_size || qweight.shape[1] != (input_features + 7) / 8 ||
      scales.shape[0] != qweight.shape[0] || scales.shape[1] < input_features / group_size ||
      zeros.shape[0] != qweight.shape[0] || zeros.shape[1] != (input_features / group_size + 7) / 8 ||
      (bias.data && (bias.shape[0] != qweight.shape[0] || !bias.is_contiguous())))
    throw std::invalid_argument("Direct AWQ linear requires N-major packed weights, scales and zeros");
  return {qweight.data, scales.data, zeros.data, bias.data, input, output, group, padded};
}
template <typename T>
void linear(const ExecutionContext& context, TensorView<const T, 2> input,
              const DenseLinearWeight<T>& weight, TensorView<T, 2> output) {
  require_dense_view(input); require_dense_view(output);
  const int rows = checked_int(input.shape[0]);
  if (!context.handle || !weight.data || input.shape[1] != static_cast<std::size_t>(weight.input_features) ||
      output.shape[0] != input.shape[0] || output.shape[1] != static_cast<std::size_t>(weight.output_features))
    throw std::invalid_argument("Direct dense linear input/output dimensions or handle are invalid");
  const float alpha = 1.0f, beta = weight.bias ? 1.0f : 0.0f;
  constexpr auto dtype = std::is_same_v<T, float> ? CUDA_R_32F : CUDA_R_16BF;
  if (weight.bias) {
    // GEMM accumulates alpha*A*B + beta*C in FP32 before its final dtype cast.
    // Initialize C with the exact bias, preserving cancellation that would be
    // lost by rounding the matrix product to BF16 before adding the bias.
    const auto blocks = std::min<std::size_t>((output.numel() + 255) / 256,
                                             context.multiprocessors * 32);
    broadcast_bias_kernel<T><<<blocks, 256, 0, context.stream>>>(
        output.data, weight.bias, rows, weight.output_features);
    check_cuda(cudaGetLastError());
  }
  check_blas(cublasGemmEx(context.handle, weight.operation, CUBLAS_OP_N,
      weight.output_features, rows, weight.input_features, &alpha,
      weight.data, dtype, weight.leading_dimension, input.data, dtype,
      weight.input_features, &beta, output.data, dtype, weight.output_features,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}
template <typename T>
void awq_linear(const ExecutionContext& context, TensorView<const T, 2> input,
                  const AwqLinearWeight<T>& weight, TensorView<T, 2> output) {
  require_dense_view(input); require_dense_view(output);
  checked_int(input.shape[0]);
  if (!weight.qweight || !weight.scales || !weight.zeros ||
      input.shape[1] != static_cast<std::size_t>(weight.input_features) ||
      output.shape[0] != input.shape[0] || output.shape[1] != static_cast<std::size_t>(weight.output_features))
    throw std::invalid_argument("Direct AWQ linear input/output dimensions are invalid");
  detail::launch_awq(context, input, weight, output);
}

template <typename T>
void gather(const ExecutionContext& context, TensorView<const uint32_t, 1> ids,
              TensorView<const T, 2> table, TensorView<T, 2> output) {
  require_dense_view(ids); require_dense_view(table); require_dense_view(output);
  if (output.shape[0] != ids.shape[0] || output.shape[1] != table.shape[1])
    throw std::invalid_argument("Direct gather input/output shapes differ");
  const auto blocks = std::min<std::size_t>((output.numel() + 255) / 256,
                                           context.multiprocessors * 32);
  gather_rows_kernel<T><<<blocks, 256, 0, context.stream>>>(ids.data, table.data, output.data,
      output.numel(), table.shape[1], table.shape[0]);
  check_cuda(cudaGetLastError());
}
template <typename T>
void rope(const ExecutionContext& context, TensorView<T, 3> input,
            std::size_t offset, float theta) {
  require_head_view(input);
  if (input.shape[2] % 2 || input.shape[2] / 2 > static_cast<std::size_t>(context.max_threads_per_block) ||
      !std::isfinite(theta) || theta <= 0.0f)
    throw std::invalid_argument("Direct RoPE requires a positive theta and supported even head width");
  rotate_kernel<T><<<dim3(input.shape[0], input.shape[1]), input.shape[2] / 2, 0, context.stream>>>(
      input.data, input.stride[0], input.shape[2], offset, theta, nullptr, 0, nullptr, nullptr);
  check_cuda(cudaGetLastError());
}
template <typename T>
void rope_precomputed(const ExecutionContext& context, TensorView<T, 3> input,
    TensorView<const float, 2> cache, std::size_t offset,
    const std::size_t* offsets, const int* pingpong) {
  require_head_view(input); require_dense_view(cache);
  if (cache.shape[1] != input.shape[2] || input.shape[2] % 2 ||
      input.shape[2] / 2 > static_cast<std::size_t>(context.max_threads_per_block))
    throw std::invalid_argument("Direct cached RoPE has incompatible cache or offsets");
  if (!offsets && (offset > cache.shape[0] || input.shape[0] > cache.shape[0] - offset))
    throw std::out_of_range("Direct cached RoPE positions exceed table capacity");
  rotate_kernel<T><<<dim3(input.shape[0], input.shape[1]), input.shape[2] / 2, 0, context.stream>>>(
      input.data, input.stride[0], input.shape[2], offset, 0.0f, cache.data, cache.stride[0], offsets, pingpong);
  check_cuda(cudaGetLastError());
}
template <typename T>
void store_kv(const ExecutionContext& context, TensorView<const T, 2> source,
    TensorView<T, 2> capacity, std::size_t offset, const std::size_t* device_offset) {
  require_dense_view(source); require_dense_view(capacity);
  if (source.shape[1] != capacity.shape[1] || offset > capacity.shape[0] ||
      source.shape[0] > capacity.shape[0] - offset)
    throw std::invalid_argument("Direct KV store source or offset exceeds cache capacity");
  const auto blocks = std::min<std::size_t>((source.numel() + 255) / 256,
                                           context.multiprocessors * 32);
  store_kv_kernel<T><<<blocks, 256, 0, context.stream>>>(source.data, capacity.data,
      source.numel(), source.shape[0], source.shape[1], capacity.shape[0], offset, device_offset);
  check_cuda(cudaGetLastError());
}
template <typename T>
void attention_decode(const ExecutionContext& context, TensorView<const T, 3> q,
    TensorView<const T, 3> k, TensorView<const T, 3> v, TensorView<T, 3> output,
    TensorView<float, 1> workspace) {
  require_attention(q, k, v, output);
  if (q.shape[0] != 1) throw std::invalid_argument("Direct decode attention requires one query row");
  if (std::is_same_v<T, __nv_bfloat16> && q.shape[2] == 128 &&
      q.is_contiguous() && k.is_contiguous() && v.is_contiguous() && output.is_contiguous() &&
      aligned_attention_view(q) && aligned_attention_view(k) && aligned_attention_view(v) && aligned_attention_view(output)) {
    require_dense_view(workspace);
    if (workspace.numel() < 5 * q.shape[1] * (q.shape[2] + 2))
      throw std::invalid_argument("Direct decode attention scratch capacity is insufficient");
    detail::launch_decode_128(context, q, k, v, output, workspace);
  } else launch_scalar_attention(context, q, k, v, output, k.shape[0] - 1);
}
template <typename T>
void attention_prefill(const ExecutionContext& context, TensorView<const T, 3> q,
    TensorView<const T, 3> k, TensorView<const T, 3> v, TensorView<T, 3> output,
    std::size_t offset) {
  require_attention(q, k, v, output);
  if (offset > k.shape[0] || q.shape[0] > k.shape[0] - offset)
    throw std::invalid_argument("Direct prefill attention offset exceeds live KV extent");
  if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    if (q.shape[2] == 128 && q.is_contiguous() && k.is_contiguous() && v.is_contiguous() && output.is_contiguous() &&
        aligned_attention_view(q) && aligned_attention_view(k) && aligned_attention_view(v) && aligned_attention_view(output)) {
      detail::launch_prefill_128(context, q, k, v, output, offset);
      return;
    }
  }
  launch_scalar_attention(context, q, k, v, output, offset);
}
template <typename T>
void attention_graph(const ExecutionContext& context, TensorView<const T, 3> q,
    TensorView<const T, 3> k, TensorView<const T, 3> v, TensorView<T, 3> output,
    float** branch_outputs, int* lengths, int* pingpong) {
  require_attention(q, k, v, output);
  if (q.shape[0] != 1 || !lengths || !pingpong)
    throw std::invalid_argument("Direct graph attention requires one query and device extent state");
  if (std::is_same_v<T, __nv_bfloat16> && q.shape[2] == 128 &&
      q.is_contiguous() && k.is_contiguous() && v.is_contiguous() && output.is_contiguous() &&
      aligned_attention_view(q) && aligned_attention_view(k) && aligned_attention_view(v) && aligned_attention_view(output)) {
    if (!branch_outputs) throw std::invalid_argument("Direct graph attention branch pointers are null");
    detail::launch_graph_128(context, q, k, v, output, branch_outputs, lengths, pingpong);
  } else launch_scalar_attention(context, q, k, v, output, 0, lengths, pingpong);
}

SamplingPlan prepare_sampling(const ExecutionContext& context, std::size_t vocabulary) {
  const int count = checked_int(vocabulary);
  SamplingPlan plan;
  plan.vocabulary = vocabulary;
  constexpr std::size_t alignment = 256;
  const auto reserve = [&](std::size_t bytes, std::size_t& offset) {
    if (bytes > std::numeric_limits<std::size_t>::max() - alignment ||
        plan.total_bytes > std::numeric_limits<std::size_t>::max() - bytes - alignment)
      throw std::overflow_error("Direct sampling scratch extent overflow");
    offset = plan.total_bytes;
    plan.total_bytes += (bytes + alignment - 1) & ~(alignment - 1);
  };
  reserve(vocabulary * sizeof(float), plan.scaled_offset);
  reserve(vocabulary * sizeof(float), plan.sorted_offset);
  reserve(vocabulary * sizeof(int), plan.indices_offset);
  reserve(vocabulary * sizeof(int), plan.sorted_indices_offset);
  check_cuda(cub::DeviceRadixSort::SortPairsDescending(nullptr, plan.sort_bytes,
      static_cast<const float*>(nullptr), static_cast<float*>(nullptr),
      static_cast<const int*>(nullptr), static_cast<int*>(nullptr), count, 0, 32, context.stream));
  reserve(plan.sort_bytes, plan.sort_offset);
  return plan;
}
void validate_sampling_policy(std::size_t vocabulary, float temperature,
                              float top_p, std::size_t top_k, bool has_rng) {
  if (!vocabulary || !top_k || !std::isfinite(temperature) || !std::isfinite(top_p) ||
      top_p <= 0.0f || top_p > 1.0f || std::min(top_k, vocabulary) > 1024)
    throw std::invalid_argument("Direct sampling policy exceeds supported parameters");
  if (temperature > 0.0f && std::min(top_k, vocabulary) > 1 && !has_rng)
    throw std::invalid_argument("Direct stochastic sampling requires owned RNG state");
}
template <typename T>
void sample(const ExecutionContext& context, TensorView<const T, 2> logits,
    TensorView<uint32_t, 1> tokens, TensorView<float, 1> probabilities,
    TensorView<unsigned char, 1> workspace, const SamplingPlan& plan,
    float temperature, float top_p, std::size_t top_k, curandState* states) {
  require_dense_view(logits); require_dense_view(tokens); require_dense_view(workspace);
  validate_sampling_policy(plan.vocabulary, temperature, top_p, top_k, states != nullptr);
  if (tokens.shape[0] != logits.shape[0] || plan.vocabulary != logits.shape[1] ||
      workspace.numel() < plan.total_bytes || !top_k || !std::isfinite(temperature) ||
      !std::isfinite(top_p) || top_p <= 0.0f || top_p > 1.0f ||
      (probabilities.data && (probabilities.shape[0] != logits.shape[0] || !probabilities.is_contiguous())))
    throw std::invalid_argument("Direct sampling parameters, plan or output capacities are invalid");
  top_k = std::min(top_k, plan.vocabulary);
  if (top_k > 1024) throw std::invalid_argument("Direct sampling supports at most 1024 candidates");
  const bool greedy = temperature <= 0.0f || top_k == 1;
  if (!greedy && !states) throw std::invalid_argument("Direct stochastic sampling requires owned RNG state");
  float* scaled = reinterpret_cast<float*>(workspace.data + plan.scaled_offset);
  float* sorted = reinterpret_cast<float*>(workspace.data + plan.sorted_offset);
  int* indices = reinterpret_cast<int*>(workspace.data + plan.indices_offset);
  int* sorted_indices = reinterpret_cast<int*>(workspace.data + plan.sorted_indices_offset);
  const int vocabulary = checked_int(plan.vocabulary);
  const auto blocks = std::min<std::size_t>((plan.vocabulary + 255) / 256,
      static_cast<std::size_t>(context.multiprocessors) * 32);
  for (std::size_t row = 0; row < logits.shape[0]; ++row) {
    scaled_indices_kernel<T><<<blocks, 256, 0, context.stream>>>(
        logits.data + row * logits.stride[0], scaled, indices, vocabulary,
        greedy ? 1.0f : temperature);
    check_cuda(cudaGetLastError());
    std::size_t sort_bytes = plan.sort_bytes;
    check_cuda(cub::DeviceRadixSort::SortPairsDescending(workspace.data + plan.sort_offset,
        sort_bytes, scaled, sorted, indices, sorted_indices, vocabulary, 0, 32, context.stream));
    sample_sorted_kernel<<<1, 1, 0, context.stream>>>(sorted, sorted_indices, top_k, top_p,
        greedy, states, tokens.data + row, probabilities.data ? probabilities.data + row : nullptr);
    check_cuda(cudaGetLastError());
  }
}
template <typename T>
void token_probability(const ExecutionContext& context, TensorView<const T, 2> logits,
                        std::size_t row, uint32_t token, float* probability) {
  require_dense_view(logits);
  if (row >= logits.shape[0] || token >= logits.shape[1] || !probability)
    throw std::invalid_argument("Direct token probability index or output is invalid");
  probability_kernel<T><<<1, 1, 0, context.stream>>>(logits.data + row * logits.stride[0],
      checked_int(logits.shape[1]), token, probability);
  check_cuda(cudaGetLastError());
}

#define INSTANTIATE_EXECUTION(T) \
  template DenseLinearWeight<T> prepare_dense<T>(TensorView<const T, 2>, TensorView<const T, 1>); \
  template AwqLinearWeight<T> prepare_awq<T>(TensorView<const int32_t, 2>, TensorView<const T, 2>, TensorView<const int32_t, 2>, std::size_t, std::size_t, TensorView<const T, 1>); \
  template void linear<T>(const ExecutionContext&, TensorView<const T, 2>, const DenseLinearWeight<T>&, TensorView<T, 2>); \
  template void awq_linear<T>(const ExecutionContext&, TensorView<const T, 2>, const AwqLinearWeight<T>&, TensorView<T, 2>); \
  template void gather<T>(const ExecutionContext&, TensorView<const uint32_t, 1>, TensorView<const T, 2>, TensorView<T, 2>); \
  template void rope<T>(const ExecutionContext&, TensorView<T, 3>, std::size_t, float); \
  template void rope_precomputed<T>(const ExecutionContext&, TensorView<T, 3>, TensorView<const float, 2>, std::size_t, const std::size_t*, const int*); \
  template void store_kv<T>(const ExecutionContext&, TensorView<const T, 2>, TensorView<T, 2>, std::size_t, const std::size_t*); \
  template void attention_decode<T>(const ExecutionContext&, TensorView<const T, 3>, TensorView<const T, 3>, TensorView<const T, 3>, TensorView<T, 3>, TensorView<float, 1>); \
  template void attention_prefill<T>(const ExecutionContext&, TensorView<const T, 3>, TensorView<const T, 3>, TensorView<const T, 3>, TensorView<T, 3>, std::size_t); \
  template void attention_graph<T>(const ExecutionContext&, TensorView<const T, 3>, TensorView<const T, 3>, TensorView<const T, 3>, TensorView<T, 3>, float**, int*, int*); \
  template void sample<T>(const ExecutionContext&, TensorView<const T, 2>, TensorView<uint32_t, 1>, TensorView<float, 1>, TensorView<unsigned char, 1>, const SamplingPlan&, float, float, std::size_t, curandState*); \
  template void token_probability<T>(const ExecutionContext&, TensorView<const T, 2>, std::size_t, uint32_t, float*);

INSTANTIATE_EXECUTION(float)
INSTANTIATE_EXECUTION(__nv_bfloat16)
#undef INSTANTIATE_EXECUTION

}  // namespace op::cuda
