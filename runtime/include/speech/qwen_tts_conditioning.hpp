#pragma once

#include <cstddef>
#include <cstdint>

#include "operators/cuda/conditioning.hpp"

namespace edge_infer::speech {

struct QwenTtsConditioningConfig {
  std::size_t text_features;
  std::size_t projection_features;
  std::size_t hidden_features;
  std::size_t codebooks;
};

// Weight matrices retain PyTorch Linear's contiguous [output, input] layout.
// All device storage and the host table descriptors are borrowed from the caller.
// Host descriptors remain valid and unchanged while the conditioner is used;
// device storage remains valid until all submitted streams finish.
template <typename T>
struct QwenTtsConditioningWeights {
  op::ArrayView<const T> fc1_weight;
  op::ArrayView<const T> fc1_bias;
  op::ArrayView<const T> fc2_weight;
  op::ArrayView<const T> fc2_bias;
  op::ArrayView<const op::cuda::EmbeddingTable<T>> codec_tables;
};

// Shared conditioning stage for Qwen3-TTS talker models. This is not a speech
// decoder: the caller supplies projected/aligned text and complete audio codes.
// Calls borrow device buffers, workspace, a cuBLAS handle and a CUDA stream.
template <typename T>
class QwenTtsConditioner {
 public:
  QwenTtsConditioner(QwenTtsConditioningConfig config,
                    QwenTtsConditioningWeights<T> weights);

  // Biased Linear -> SiLU -> biased Linear, with rounding at each model boundary.
  // hidden is [rows, projection_features]; accumulation has at least
  // rows * max(projection_features, hidden_features) FP32 elements.
  // Device inputs, weights, hidden, output and workspace must be disjoint.
  void project_text(op::ArrayView<const T> text_embeddings,
                    op::ArrayView<T> output, op::ArrayView<T> hidden,
                    op::ArrayView<float> accumulation, std::size_t rows,
                    cublasHandle_t handle, cudaStream_t stream = nullptr) const;

  // Sum all codebook embeddings in FP32, round once, then add the optional
  // aligned text/tts_pad embedding in the model dtype, matching the talker.
  // host_codes uses [frames, codebooks] order and is validated before launch.
  // Aligned text must not overlap output or the used FP32 workspace region.
  // Table storage must be disjoint from output and workspace.
  void compose_frame_embeddings(op::ArrayView<const uint32_t> host_codes,
                                op::ArrayView<const T> aligned_text,
                                op::ArrayView<T> output,
                                op::ArrayView<float> accumulation,
                                std::size_t frames,
                                cudaStream_t stream = nullptr) const;

  const QwenTtsConditioningConfig& config() const noexcept { return config_; }

 private:
  QwenTtsConditioningConfig config_;
  QwenTtsConditioningWeights<T> weights_;
};

extern template class QwenTtsConditioner<float>;
extern template class QwenTtsConditioner<__nv_bfloat16>;

}  // namespace edge_infer::speech
