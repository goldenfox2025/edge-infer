#pragma once
#include <memory>
#include <unordered_map>
#include <vector>

#include "base_model.hpp"
#include "execution/decoder.hpp"
#include "speculative_model.hpp"
#include "tensor_view_adapter.hpp"

// Immutable weights are shared. One session owns all mutable resources. Calls
// complete before returning; callers serialize operations on the same session.
template <typename T>
class Qwen3Session : public BaseModel, public SpeculativeModel<T> {
 public:
  explicit Qwen3Session(std::shared_ptr<const Qwen3Model<T>> model);
  Qwen3Session(std::shared_ptr<const Qwen3Model<T>> model, bool enable_graph);
  Qwen3Session(const typename Qwen3Model<T>::Parameters&, const ModelConfig&);
  Qwen3Session(const typename Qwen3Model<T>::Parameters&,
               const typename Qwen3Model<T>::IntegerParameters&,
               const typename Qwen3Model<T>::Parameters&,
               const typename Qwen3Model<T>::IntegerParameters&, const ModelConfig&);
  ~Qwen3Session() override;
  Qwen3Session(const Qwen3Session&) = delete;
  Qwen3Session& operator=(const Qwen3Session&) = delete;
  std::shared_ptr<BaseModel> fork_executor() const override {
    return std::make_shared<Qwen3Session<T>>(model_, use_cuda_graph_);
  }
  static std::unique_ptr<Qwen3Session> create(std::shared_ptr<const Qwen3Model<T>>, size_t capacity,
                                              bool graph = false);
  std::unique_ptr<Qwen3Session> new_session(size_t capacity, bool graph = false) const;
  // Output views borrow session storage until its next operation.
  TensorView<T, 2> prefill(TensorView<const uint32_t, 1>);
  TensorView<T, 2> prefill(const uint32_t* host_tokens, size_t rows);
  TensorView<T, 2> prefill(const std::vector<uint32_t>& tokens) {
    return prefill(tokens.data(), tokens.size());
  }
  TensorView<T, 2> decode(TensorView<const uint32_t, 1>);
  TensorView<T, 2> decode(uint32_t token);
  // Model adapters supply embeddings and consume normalized hidden states.
  // These calls use the same eager backbone, without lookup, LM head or sampling.
  // Logical positions are independent of physical cache slots.
  TensorView<T, 2> prefill_embeddings(TensorView<const T, 2>, size_t position_offset = 0);
  TensorView<T, 2> decode_embeddings(TensorView<const T, 2>, size_t position_offset);
  TensorView<T, 2> prefill(const Tensor<uint32_t>& in) {
    return prefill(borrow_tensor_view<1>(in));
  }
  TensorView<T, 2> decode(const Tensor<uint32_t>& in) { return decode(borrow_tensor_view<1>(in)); }
  void reset();
  size_t context_size() const;
  size_t context_capacity() const;
  const auto& model() const { return model_; }
  cudaStream_t stream() const { return context_.stream; }
  void synchronize() const override;
  bool graph_enabled() const { return use_cuda_graph_; }
  void set_graph_enabled(bool);
  size_t decode_workspace_bytes() const { return decode_workspace_.capacity_bytes(); }
  size_t decode_unaliased_bytes() const { return decode_plan_.unaliased_bytes(); }
  size_t decode_reused_bytes() const { return decode_plan_.reused_bytes(); }
  const WorkspacePlan& decode_workspace_plan() const { return decode_plan_; }
  bool owns_execution_workspaces() const override { return true; }
  size_t estimate_prefill_workspace_bytes(size_t) const override;
  bool verify_params() const override { return model_->verify_params(); }
  void print_model_info() const override;
  Qwen3Session& cuda() override;
  Qwen3Session& cpu() override;
  Device device() const override { return Device::CUDA; }
  size_t get_n_layers() const override { return model_->config().n_layers; }
  size_t get_max_seq_len() const override { return model_->config().max_position_embeddings; }
  size_t get_head_dim() const override { return model_->config().head_dim; }
  size_t get_n_kv_heads() const override { return model_->config().n_kv_heads; }
  uint32_t get_eos_token_id() const override { return model_->config().eos_token_id; }
  size_t get_vocab_size() const override { return model_->config().vocab_size; }
  size_t get_n_heads() const { return model_->config().n_heads; }
  size_t get_hidden_size() const { return model_->config().hidden_size; }
  size_t get_intermediate_size() const { return model_->config().intermediate_size; }
  float get_rms_norm_eps() const { return model_->config().rms_norm_eps; }
  float get_rope_theta() const { return model_->config().rope_theta; }
  int get_quant_type() const { return model_->config().quant_type; }
  const auto& get_params() const { return model_->get_params(); }
  const auto& get_qweight_params() const { return model_->get_qweight_params(); }
  const auto& get_scales_params() const { return model_->get_scales_params(); }
  const auto& get_qzeros_params() const { return model_->get_qzeros_params(); }
  op::WeightTensor<T> get_weight(const std::string& key) const { return model_->get_weight(key); }
  uint32_t* forward(const Tensor<uint32_t>*, ThreadPool&, KVCacheBase*, size_t, float, float,
                    curandState* = nullptr) override;
  uint32_t* prefill(const Tensor<uint32_t>*, ThreadPool&, KVCacheBase*, size_t, float, float,
                    curandState* = nullptr) override;
  // An external cache binds permanently on first use. Its owner keeps the
  // cache and backing storage alive until this session has been destroyed.
  TensorView<T, 2> forward_eager(const Tensor<uint32_t>*, KVCache<T>*);
  TensorView<T, 2> prefill_eager(const Tensor<uint32_t>*, KVCache<T>*);
  TensorView<T, 2> forward_for_graph_logits_only(const Tensor<uint32_t>*, KVCache<T>*);
  TensorView<T, 2> speculative_forward_logits(const Tensor<uint32_t>* in, KVCache<T>* kv) override {
    return forward_eager(in, kv);
  }
  TensorView<T, 2> speculative_prefill_logits(const Tensor<uint32_t>* in, KVCache<T>* kv) override {
    return prefill_eager(in, kv);
  }

 private:
  void initialize();
  void release() noexcept;
  void require_managed() const;
  void validate_and_bind(TensorView<const uint32_t, 1>, KVCache<T>*, bool);
  void validate_cache(KVCache<T>*, size_t, bool);
  void bind_cache(KVCache<T>*);
  TensorView<T, 2> execute_embeddings(TensorView<const T, 2>, size_t, bool);
  TensorView<T, 2> execute(TensorView<const uint32_t, 1>, KVCache<T>&, bool, bool);
  TensorView<T, 2> execute_prevalidated(TensorView<const uint32_t, 1>, KVCache<T>&, bool, bool);
  TensorView<T, 2> execute_host_tokens(const uint32_t*, size_t, bool);
  void run(TensorView<const uint32_t, 1>, KVCache<T>&, DecoderBuffers<T>&, const DecoderStep&);
  void capture_graph(KVCache<T>&);
  uint32_t* sample_logits(TensorView<const T, 2>, float, float, size_t, curandState*);
  std::shared_ptr<const Qwen3Model<T>> model_;
  std::unique_ptr<KVCache<T>> managed_cache_;
  bool history_ready_ = false, use_cuda_graph_ = false;
  op::cuda::ExecutionContext context_;
  WorkspacePlan decode_plan_;
  CudaWorkspaceArena decode_workspace_, prefill_workspace_, graph_storage_, sampling_storage_,
      host_token_storage_;
  DecoderBuffers<T> decode_buffers_{}, prefill_buffers_{};
  size_t prefill_rows_ = 0;
  op::cuda::SamplingPlan sampling_plan_;
  TensorView<unsigned char, 1> sampling_scratch_{};
  TensorView<uint32_t, 1> sampled_token_{}, graph_input_{};
  TensorView<float, 1> sampled_probability_{};
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t graph_exec_ = nullptr;
  size_t* device_offset_ = nullptr;
  int* device_lengths_ = nullptr;
  int* device_pingpong_ = nullptr;
  float** graph_branches_ = nullptr;
  size_t host_offset_ = 0;
  int host_length_ = 0;
  KVCache<T>* bound_cache_ = nullptr;
  size_t bound_capacity_ = 0;
  std::vector<std::pair<const T*, const T*>> bound_bases_;
  std::vector<uint32_t> host_ids_;
};
extern template class Qwen3Session<__nv_bfloat16>;
extern template class Qwen3Session<float>;
