#pragma once

#include <algorithm>
#include <functional>
#include <memory>
#include <vector>

#include "base_model.hpp"
#include "execution/cuda_workspace_arena.hpp"
#include "inference.hpp"
#include "operators/cuda/execution.hpp"
#include "speculative_model.hpp"
#include "tensor.hpp"

// Own draft/target caches and prepared sampling storage for one CUDA session.
// Probability-ratio verification is experimental: target replacement sampling
// does not implement the residual distribution required for exact speculation.
template <typename T>
class SpeculativeDecoder : public infer_base {
 public:
  SpeculativeDecoder(std::shared_ptr<BaseModel> target_model,
                     std::shared_ptr<BaseModel> draft_model,
                     size_t spec_length = 6, size_t thread_count = 8);
  ~SpeculativeDecoder();

  void generate_with_callback(const std::vector<uint32_t>& input_ids, size_t max_length,
                              float temperature, float top_p, size_t top_k,
                              std::function<void(uint32_t)> callback) override;
  Device device() const override { return Device::CUDA; }
  void set_use_probability_ratio(bool value) { use_probability_ratio_ = value; }
  bool get_use_probability_ratio() const { return use_probability_ratio_; }
  size_t get_adaptive_spec_length() const { return adaptive_spec_length_; }
  void update_adaptive_spec_length(float acceptance_rate) {
    recent_acceptance_rate_ = 0.7f * recent_acceptance_rate_ + 0.3f * acceptance_rate;
    if (recent_acceptance_rate_ > ACCEPTANCE_THRESHOLD_HIGH)
      adaptive_spec_length_ = std::min(adaptive_spec_length_ + 1, MAX_SPEC_LENGTH);
    else if (recent_acceptance_rate_ < ACCEPTANCE_THRESHOLD_LOW)
      adaptive_spec_length_ = std::max(adaptive_spec_length_ - 1,
                                      std::min(spec_length_, MIN_SPEC_LENGTH));
  }

 private:
  static constexpr size_t MIN_SPEC_LENGTH = 6;
  static constexpr size_t MAX_SPEC_LENGTH = 8;
  static constexpr float ACCEPTANCE_THRESHOLD_HIGH = 0.7f;
  static constexpr float ACCEPTANCE_THRESHOLD_LOW = 0.4f;

  std::shared_ptr<BaseModel> target_model_;
  std::shared_ptr<BaseModel> draft_model_;
  std::shared_ptr<SpeculativeModel<T>> target_spec_model_;
  std::shared_ptr<SpeculativeModel<T>> draft_spec_model_;
  KVCache<T> target_kv_cache_;
  KVCache<T> draft_kv_cache_;
  size_t spec_length_;
  size_t adaptive_spec_length_;
  float recent_acceptance_rate_ = 0.5f;
  bool use_probability_ratio_ = true;

  int cuda_device_id_ = -1;
  cudaStream_t draft_stream_ = nullptr;
  cudaStream_t verify_stream_ = nullptr;
  op::cuda::ExecutionContext draft_context_;
  op::cuda::ExecutionContext verify_context_;
  op::cuda::SamplingPlan draft_sampling_plan_;
  op::cuda::SamplingPlan target_sampling_plan_;
  CudaWorkspaceArena resources_;
  CudaWorkspaceArena prompt_storage_;
  CudaWorkspaceArena draft_sampling_storage_;
  CudaWorkspaceArena target_sampling_storage_;
  TensorView<unsigned char, 1> draft_scratch_;
  TensorView<unsigned char, 1> target_scratch_;
  curandState* draft_states_ = nullptr;
  curandState* target_states_ = nullptr;
  uint32_t* draft_tokens_ = nullptr;
  float* draft_probabilities_ = nullptr;
  float* random_values_ = nullptr;
  uint32_t* target_tokens_ = nullptr;
  float* target_probabilities_ = nullptr;

  // These compatibility descriptors are prepared once, borrow private buffers,
  // and never construct Tensor ownership in draft decode or verification.
  std::vector<Tensor<uint32_t>> token_inputs_;
  std::vector<Tensor<uint32_t>> batch_inputs_;

  void init_cuda_resources();
  void free_cuda_resources() noexcept;
  void generate_draft_tokens(size_t count, bool probabilities,
                             float temperature, float top_p, size_t top_k);
  TensorView<T, 2> verify_logits(size_t count);
  size_t verify_greedy(size_t count, float temperature, float top_p,
                       std::vector<uint32_t>& verified);
  size_t verify_probability_ratio(size_t count, float temperature,
                                  float top_p, size_t top_k,
                                  std::vector<uint32_t>& verified);
};

extern template class SpeculativeDecoder<__nv_bfloat16>;
