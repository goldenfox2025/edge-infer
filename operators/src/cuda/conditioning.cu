#include "operators/cuda/conditioning.hpp"

#include <algorithm>
#include <climits>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace op::cuda {
namespace {

constexpr unsigned kThreads = 256;
constexpr std::size_t kFramesPerChunk = 64;

void check_cuda(cudaError_t status) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("Conditioning CUDA error: ") +
                             cudaGetErrorString(status));
  }
}

void check_cublas(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("Conditioning cuBLAS error: " +
                             std::to_string(static_cast<int>(status)));
  }
}

std::size_t checked_product(std::size_t a, std::size_t b, const char* name) {
  if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
    throw std::runtime_error(std::string(name) + " extent overflow");
  }
  return a * b;
}

template <typename T>
void check_view(ArrayView<T> view, const char* name) {
  checked_product(view.size, sizeof(T), name);
  if (view.size != 0 && view.data == nullptr) {
    throw std::runtime_error(std::string(name) + " has a null pointer");
  }
}

template <typename T>
void check_extent(ArrayView<T> view, std::size_t expected, const char* name) {
  if (view.size != expected) {
    throw std::runtime_error(std::string(name) + " extent mismatch");
  }
  check_view(view, name);
}

void check_scratch(ArrayView<float> scratch, std::size_t expected) {
  if (scratch.size < expected) {
    throw std::runtime_error("Conditioning float scratch is too small");
  }
  check_view(scratch, "Conditioning float scratch");
}

unsigned blocks_for(std::size_t count) {
  return static_cast<unsigned>(std::min<std::size_t>(
      count / kThreads + (count % kThreads != 0), 4096));
}

template <typename T>
__global__ void finish_linear(const float* accumulation, const T* bias,
                               T* output, std::size_t count,
                               std::size_t out_features) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += stride) {
    float value = accumulation[i];
    if (bias != nullptr) {
      value += static_cast<float>(bias[i % out_features]);
    }
    output[i] = static_cast<T>(value);
  }
}

struct CodeIdsChunk {
  uint32_t ids[kFramesPerChunk];
};

template <typename T>
__global__ void accumulate_embeddings(const T* table, CodeIdsChunk ids,
                                      float* accumulation, std::size_t base_frame,
                                      std::size_t frames, std::size_t features,
                                      bool first_table) {
  const std::size_t count = frames * features;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += stride) {
    const std::size_t frame = i / features;
    const std::size_t feature = i % features;
    const std::size_t output_index = base_frame * features + i;
    const float value = static_cast<float>(
        table[static_cast<std::size_t>(ids.ids[frame]) * features + feature]);
    if (first_table) {
      accumulation[output_index] = value;
    } else {
      accumulation[output_index] += value;
    }
  }
}

template <typename T>
__global__ void finish_embeddings(const float* accumulation, T* output,
                                  std::size_t count) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += stride) {
    output[i] = static_cast<T>(accumulation[i]);
  }
}

}  // namespace

template <typename T>
void linear(ArrayView<const T> input, ArrayView<const T> weight,
            ArrayView<const T> bias, ArrayView<T> output,
            ArrayView<float> accumulation, std::size_t rows,
            std::size_t in_features, std::size_t out_features,
            cublasHandle_t handle, cudaStream_t stream) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, __nv_bfloat16>);
  const auto input_count = checked_product(rows, in_features, "Linear input");
  const auto weight_count = checked_product(out_features, in_features, "Linear weight");
  const auto output_count = checked_product(rows, out_features, "Linear output");
  check_extent(input, input_count, "Linear input");
  check_extent(weight, weight_count, "Linear weight");
  check_extent(output, output_count, "Linear output");
  if (bias.size != 0) {
    check_extent(bias, out_features, "Linear bias");
  }
  check_scratch(accumulation, output_count);
  if (rows == 0) {
    return;
  }
  if (in_features == 0 || out_features == 0) {
    throw std::runtime_error("Nonempty linear requires positive feature dimensions");
  }
  if (rows > INT_MAX || in_features > INT_MAX || out_features > INT_MAX) {
    throw std::runtime_error("Linear dimensions exceed the cuBLAS signed-int limit");
  }
  if (handle == nullptr) {
    throw std::runtime_error("Nonempty linear requires a cuBLAS handle");
  }

  cublasPointerMode_t pointer_mode;
  check_cublas(cublasGetPointerMode(handle, &pointer_mode));
  if (pointer_mode != CUBLAS_POINTER_MODE_HOST) {
    throw std::runtime_error("Conditioning linear requires cuBLAS host pointer mode");
  }
  cublasMath_t math_mode;
  check_cublas(cublasGetMathMode(handle, &math_mode));
  if (math_mode != CUBLAS_DEFAULT_MATH) {
    throw std::runtime_error("Conditioning linear requires CUBLAS_DEFAULT_MATH");
  }
  check_cublas(cublasSetStream(handle, stream));

  const float alpha = 1.0f;
  const float beta = 0.0f;
  constexpr cudaDataType_t operand_type = std::is_same_v<T, float>
                                             ? CUDA_R_32F : CUDA_R_16BF;
  // Column-major C^T = W * X^T over the row-major physical [out,in] weights.
  // Both float and BF16 operands use FP32 accumulation and FP32 GEMM output.
  check_cublas(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                          static_cast<int>(out_features), static_cast<int>(rows),
                          static_cast<int>(in_features), &alpha, weight.data,
                          operand_type, static_cast<int>(in_features), input.data,
                          operand_type, static_cast<int>(in_features), &beta,
                          accumulation.data, CUDA_R_32F,
                          static_cast<int>(out_features), CUBLAS_COMPUTE_32F,
                          CUBLAS_GEMM_DEFAULT));
  finish_linear<T><<<blocks_for(output_count), kThreads, 0, stream>>>(
      accumulation.data, bias.size == 0 ? nullptr : bias.data,
      output.data, output_count, out_features);
  check_cuda(cudaGetLastError());
}

template <typename T>
void sum_embeddings(ArrayView<const EmbeddingTable<T>> tables,
                    ArrayView<const uint32_t> host_ids, ArrayView<T> output,
                    ArrayView<float> accumulation, std::size_t frames,
                    std::size_t features, cudaStream_t stream) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, __nv_bfloat16>);
  const auto id_count = checked_product(frames, tables.size, "Embedding IDs");
  const auto output_count = checked_product(frames, features, "Embedding output");
  check_view(tables, "Embedding tables");
  check_extent(host_ids, id_count, "Embedding IDs");
  check_extent(output, output_count, "Embedding output");
  check_scratch(accumulation, output_count);
  if (frames != 0 && (features == 0 || tables.size == 0)) {
    throw std::runtime_error("Nonempty embedding sum requires tables and positive features");
  }
  for (std::size_t book = 0; book < tables.size; ++book) {
    const auto& table = tables[book];
    const auto table_count = checked_product(table.vocab_size, features, "Embedding table");
    check_extent(table.values, table_count, "Embedding table");
    for (std::size_t frame = 0; frame < frames; ++frame) {
      if (host_ids[frame * tables.size + book] >= table.vocab_size) {
        throw std::runtime_error("Embedding ID out of range at frame " +
                                 std::to_string(frame) + ", codebook " +
                                 std::to_string(book));
      }
    }
  }
  if (frames == 0) {
    return;
  }

  // IDs travel by value in bounded CUDA launch arguments. This avoids borrowed
  // host pointers in device code and any owned device metadata allocation.
  for (std::size_t book = 0; book < tables.size; ++book) {
    for (std::size_t base = 0; base < frames;) {
      const auto chunk_frames = std::min(kFramesPerChunk, frames - base);
      CodeIdsChunk chunk{};
      for (std::size_t frame = 0; frame < chunk_frames; ++frame) {
        chunk.ids[frame] = host_ids[(base + frame) * tables.size + book];
      }
      const auto count = chunk_frames * features;
      accumulate_embeddings<T><<<blocks_for(count), kThreads, 0, stream>>>(
          tables[book].values.data, chunk, accumulation.data, base,
          chunk_frames, features, book == 0);
      check_cuda(cudaGetLastError());
      base += chunk_frames;
    }
  }
  finish_embeddings<T><<<blocks_for(output_count), kThreads, 0, stream>>>(
      accumulation.data, output.data, output_count);
  check_cuda(cudaGetLastError());
}

template void linear<float>(ArrayView<const float>, ArrayView<const float>,
    ArrayView<const float>, ArrayView<float>, ArrayView<float>, std::size_t,
    std::size_t, std::size_t, cublasHandle_t, cudaStream_t);
template void linear<__nv_bfloat16>(ArrayView<const __nv_bfloat16>,
    ArrayView<const __nv_bfloat16>, ArrayView<const __nv_bfloat16>,
    ArrayView<__nv_bfloat16>, ArrayView<float>, std::size_t, std::size_t,
    std::size_t, cublasHandle_t, cudaStream_t);
template void sum_embeddings<float>(ArrayView<const EmbeddingTable<float>>,
    ArrayView<const uint32_t>, ArrayView<float>, ArrayView<float>, std::size_t,
    std::size_t, cudaStream_t);
template void sum_embeddings<__nv_bfloat16>(
    ArrayView<const EmbeddingTable<__nv_bfloat16>>, ArrayView<const uint32_t>,
    ArrayView<__nv_bfloat16>, ArrayView<float>, std::size_t, std::size_t,
    cudaStream_t);

}  // namespace op::cuda
