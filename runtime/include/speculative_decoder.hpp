#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "base_model.hpp"
#include "execution/cuda_workspace_arena.hpp"
#include "inference.hpp"
#include "operators/cuda/execution.hpp"
#include "speculative_model.hpp"
#include "tensor.hpp"

// Exact greedy speculation over independent target/draft sessions. Each
// generation call supplies a complete prompt and replaces the prior history.
template <typename T>
class SpeculativeDecoder : public infer_base {
 public:
  SpeculativeDecoder(std::shared_ptr<BaseModel> target_model,
                     std::shared_ptr<BaseModel> draft_model,
                     size_t spec_length = 6, size_t thread_count = 8,
                     size_t capacity = 0);
  ~SpeculativeDecoder();

  // Only top_k=1 is supported. Stochastic generation uses InferenceEngine.
  void generate_with_callback(const std::vector<uint32_t>& input_ids, size_t max_length,
                              float temperature, float top_p, size_t top_k,
                              std::function<void(uint32_t)> callback) override;
  Device device() const override {
    require_valid();
    return Device::CUDA;
  }
  void reset() override;
  size_t context_size() const override {
    require_valid();
    return target_kv_cache_.size();
  }
  size_t context_capacity() const override {
    require_valid();
    return target_kv_cache_.get_max_seq_len();
  }
  size_t get_spec_length() const {
    require_valid();
    return spec_length_;
  }

 private:
  void require_valid() const {
    if (!valid_)
      throw std::logic_error("Executor completion failed; construct a new speculative decoder");
  }
  bool valid_ = true;
  static constexpr size_t MAX_SPEC_LENGTH = 8;
  std::shared_ptr<BaseModel> target_model_;
  std::shared_ptr<BaseModel> draft_model_;
  std::shared_ptr<SpeculativeModel<T>> target_spec_model_;
  std::shared_ptr<SpeculativeModel<T>> draft_spec_model_;
  size_t spec_length_;
  KVCache<T> target_kv_cache_;
  KVCache<T> draft_kv_cache_;

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
  uint32_t* draft_tokens_ = nullptr;
  uint32_t* target_tokens_ = nullptr;

  // Compatibility descriptors are prepared once and borrow private buffers.
  std::vector<Tensor<uint32_t>> token_inputs_;
  std::vector<Tensor<uint32_t>> batch_inputs_;

  void init_cuda_resources();
  void complete_execution();
  void free_cuda_resources() noexcept;
  void generate_draft_tokens(size_t count, float temperature, float top_p);
  TensorView<T, 2> verify_logits(size_t count);
  void verify_greedy(size_t count, float temperature, float top_p,
                     std::vector<uint32_t>& verified);
};

extern template class SpeculativeDecoder<__nv_bfloat16>;
