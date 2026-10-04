#include "speech/qwen_tts_conditioning.hpp"

#include <algorithm>
#include <climits>
#include <limits>
#include <stdexcept>

#include "operators/cuda/direct.hpp"

namespace edge_infer::speech {
namespace {

std::size_t product(std::size_t a, std::size_t b) {
  if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
    throw std::overflow_error("Qwen TTS conditioning extent overflow");
  }
  return a * b;
}

template <typename T>
void require_view(op::ArrayView<T> view, std::size_t expected) {
  product(expected, sizeof(T));
  if (view.size != expected || (expected != 0 && view.data == nullptr)) {
    throw std::invalid_argument("Qwen TTS conditioning tensor extent mismatch");
  }
}

bool overlaps(const void* first, std::size_t first_bytes,
              const void* second, std::size_t second_bytes) {
  if (first_bytes == 0 || second_bytes == 0) return false;
  const auto a = reinterpret_cast<std::uintptr_t>(first);
  const auto b = reinterpret_cast<std::uintptr_t>(second);
  return a <= b ? b - a < first_bytes : a - b < second_bytes;
}

template <typename A, typename B>
void require_disjoint(op::ArrayView<A> first, op::ArrayView<B> second) {
  if (overlaps(first.data, product(first.size, sizeof(A)),
               second.data, product(second.size, sizeof(B)))) {
    throw std::invalid_argument("Qwen TTS projection buffers must not overlap");
  }
}

}  // namespace

template <typename T>
QwenTtsConditioner<T>::QwenTtsConditioner(
    QwenTtsConditioningConfig config, QwenTtsConditioningWeights<T> weights)
    : config_(config), weights_(weights) {
  if (config.text_features == 0 || config.projection_features == 0 ||
      config.hidden_features == 0 || config.codebooks == 0) {
    throw std::invalid_argument("Qwen TTS conditioning dimensions must be positive");
  }
  if (config.text_features > INT_MAX || config.projection_features > INT_MAX ||
      config.hidden_features > INT_MAX) {
    throw std::invalid_argument("Qwen TTS projection dimensions exceed the cuBLAS limit");
  }
  require_view(weights.fc1_weight, product(config.projection_features, config.text_features));
  require_view(weights.fc1_bias, config.projection_features);
  require_view(weights.fc2_weight, product(config.hidden_features, config.projection_features));
  require_view(weights.fc2_bias, config.hidden_features);
  require_view(weights.codec_tables, config.codebooks);
  for (std::size_t group = 0; group < config.codebooks; ++group) {
    const auto& table = weights.codec_tables[group];
    if (table.vocab_size == 0) {
      throw std::invalid_argument("Qwen TTS codec vocabulary must be positive");
    }
    require_view(table.values, product(table.vocab_size, config.hidden_features));
  }
}

template <typename T>
void QwenTtsConditioner<T>::project_text(
    op::ArrayView<const T> input, op::ArrayView<T> output,
    op::ArrayView<T> hidden, op::ArrayView<float> accumulation,
    std::size_t rows, cublasHandle_t handle, cudaStream_t stream) const {
  require_view(input, product(rows, config_.text_features));
  require_view(hidden, product(rows, config_.projection_features));
  require_view(output, product(rows, config_.hidden_features));
  if (rows > INT_MAX || hidden.size > INT_MAX) {
    throw std::invalid_argument("Qwen TTS text projection exceeds the direct SiLU limit");
  }
  const auto needed = product(rows, std::max(config_.projection_features, config_.hidden_features));
  if (accumulation.size < needed || (needed != 0 && accumulation.data == nullptr)) {
    throw std::invalid_argument("Qwen TTS conditioning workspace is too small");
  }
  // Standalone linear may permit output=scratch for float, but this two-layer
  // pipeline must preserve hidden as the second GEMM's input.
  const op::ArrayView<float> used_scratch{accumulation.data, needed};
  require_disjoint(hidden, output);
  require_disjoint(hidden, used_scratch);
  require_disjoint(output, used_scratch);
  const auto reject_write_overlap = [&](auto operand) {
    require_disjoint(hidden, operand);
    require_disjoint(output, operand);
    require_disjoint(used_scratch, operand);
  };
  reject_write_overlap(input);
  reject_write_overlap(weights_.fc1_weight);
  reject_write_overlap(weights_.fc1_bias);
  reject_write_overlap(weights_.fc2_weight);
  reject_write_overlap(weights_.fc2_bias);
  op::cuda::linear<T>(input, weights_.fc1_weight, weights_.fc1_bias, hidden,
                      accumulation, rows, config_.text_features,
                      config_.projection_features, handle, stream);
  op::cuda::silu<T>({hidden.data, hidden.size}, hidden, stream);
  op::cuda::linear<T>({hidden.data, hidden.size}, weights_.fc2_weight,
                      weights_.fc2_bias, output, accumulation, rows,
                      config_.projection_features, config_.hidden_features,
                      handle, stream);
}

template <typename T>
void QwenTtsConditioner<T>::compose_frame_embeddings(
    op::ArrayView<const uint32_t> host_codes, op::ArrayView<const T> aligned_text,
    op::ArrayView<T> output, op::ArrayView<float> accumulation,
    std::size_t frames, cudaStream_t stream) const {
  const auto count = product(frames, config_.hidden_features);
  require_view(host_codes, product(frames, config_.codebooks));
  require_view(output, count);
  if (aligned_text.size != 0) {
    require_view(aligned_text, count);
    if (count > INT_MAX) {
      throw std::invalid_argument("Qwen TTS frame composition exceeds the direct add limit");
    }
    const auto bytes = product(count, sizeof(T));
    if (overlaps(aligned_text.data, bytes, output.data, bytes) ||
        overlaps(aligned_text.data, bytes, accumulation.data, product(count, sizeof(float)))) {
      throw std::invalid_argument("Aligned text must not overlap the composed frame output or workspace");
    }
  }
  op::cuda::sum_embeddings<T>(weights_.codec_tables, host_codes, output,
                              accumulation, frames, config_.hidden_features, stream);
  if (aligned_text.size != 0) {
    op::cuda::add<T>({output.data, output.size}, aligned_text, output, stream);
  }
}

template class QwenTtsConditioner<float>;
template class QwenTtsConditioner<__nv_bfloat16>;

}  // namespace edge_infer::speech
