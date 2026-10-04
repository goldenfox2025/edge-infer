#pragma once
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "base_model.hpp"
#include "models/qwen3_model.hpp"
#include "common.hpp"
#include "execution/cuda_workspace_arena.hpp"
#include "cuda_graph_runtime.hpp"
#include "inference.hpp"
#include "operators/unified_operators.hpp"
#include "operators/cuda/dynamic_flash_attention_cuda.cuh"
#include "speculative_model.hpp"
#include "tensor.hpp"
#include "thread_pool.hpp"

template <typename T>
class Qwen3Session : public BaseModel, public SpeculativeModel<T> {
   public:
    explicit Qwen3Session(std::shared_ptr<const Qwen3Model<T>> model);
    Qwen3Session(std::shared_ptr<const Qwen3Model<T>> model, bool enable_graph);
    Qwen3Session(const Qwen3Session&) = delete;
    Qwen3Session& operator=(const Qwen3Session&) = delete;
    Qwen3Session(Qwen3Session&&) = delete;
    Qwen3Session& operator=(Qwen3Session&&) = delete;

    // Managed sessions own a fixed-capacity cache and start with empty history.
    // Creation and execution require the model's CUDA device to be current.
    static std::unique_ptr<Qwen3Session<T>> create(
        std::shared_ptr<const Qwen3Model<T>> model, size_t context_capacity,
        bool enable_graph = false);
    // Share prepared weights with a fresh session; no history or prefix is copied.
    std::unique_ptr<Qwen3Session<T>> new_session(
        size_t context_capacity, bool enable_graph = false) const;
    // Context operations below require a managed session. Logits borrow its storage.
    // Start a new history and return logits for all prompt rows.
    Tensor<T> prefill(const Tensor<uint32_t>& input);
    // Append exactly one token after prefill; the graph flag selects decode mode.
    Tensor<T> decode(const Tensor<uint32_t>& input);
    void reset();
    size_t context_size() const;
    size_t context_capacity() const;

    const std::shared_ptr<const Qwen3Model<T>>& model() const { return model_; }
    cudaStream_t stream() const { return execution_stream_; }
    void synchronize() const;
    bool graph_enabled() const { return use_cuda_graph_; }
    void set_graph_enabled(bool enabled);
    size_t decode_workspace_bytes() const { return decode_workspace_->capacity_bytes(); }
    bool owns_execution_workspaces() const override { return true; }

    // Compatibility constructors prepare a shared model and one session.
    Qwen3Session(const std::unordered_map<std::string, Tensor<T>>& params,
               const ModelConfig& config);

    // Construct an AWQ-weight Qwen3 model and initialize runtime state.
    Qwen3Session(const std::unordered_map<std::string, Tensor<T>>& params,
               const std::unordered_map<std::string, Tensor<int32_t>>& qweight_params,
               const std::unordered_map<std::string, Tensor<T>>& scales_params,
               const std::unordered_map<std::string, Tensor<int32_t>>& qzeros_params,
               const ModelConfig& config);
    // Complete outstanding work and release this session's execution resources.
    ~Qwen3Session() override;

    // Verify that all required model weights are present.
    bool verify_params() const override;
    // Print a concise summary of model dimensions and runtime mode.
    void print_model_info() const override;

    // Run one decode step and sample the next token.
    uint32_t* forward(const Tensor<uint32_t>* input, ThreadPool& thread_pool, KVCacheBase* kv_cache, size_t top_k,
                      float temperature, float top_p, curandState* d_states = nullptr) override;
    // Run prefill on a prompt and sample the next token.
    uint32_t* prefill(const Tensor<uint32_t>* input, ThreadPool& thread_pool, KVCacheBase* kv_cache, size_t top_k,
                      float temperature, float top_p, curandState* d_states = nullptr) override;

    // The shared model is already prepared on CUDA.
    Qwen3Session& cuda() override;
    // CPU execution is unsupported; this leaves the session usable on CUDA.
    Qwen3Session& cpu() override;
    // Return the current device backing this model.
    Device device() const override {
        return Device::CUDA;
    }

    // Return the number of transformer blocks.
    size_t get_n_layers() const override {
        return model_->config().n_layers;
    }
    // Return the configured maximum sequence length.
    size_t get_max_seq_len() const override {
        return model_->config().max_position_embeddings;
    }
    // Return per-head hidden width.
    size_t get_head_dim() const override {
        return model_->config().head_dim;
    }
    // Return the number of KV heads.
    size_t get_n_kv_heads() const override {
        return model_->config().n_kv_heads;
    }
    // Return the EOS token id.
    uint32_t get_eos_token_id() const override {
        return model_->config().eos_token_id;
    }

    // Return the number of attention heads.
    size_t get_n_heads() const {
        return model_->config().n_heads;
    }
    // Return the hidden size.
    size_t get_hidden_size() const {
        return model_->config().hidden_size;
    }
    // Estimate workspace bytes needed for prefill at the given sequence length.
    size_t estimate_prefill_workspace_bytes(size_t seq_len) const override;
    // Return the MLP intermediate size.
    size_t get_intermediate_size() const {
        return model_->config().intermediate_size;
    }
    // Return the RMSNorm epsilon used by this model.
    float get_rms_norm_eps() const {
        return model_->config().rms_norm_eps;
    }
    // Return the RoPE theta used by this model.
    float get_rope_theta() const {
        return model_->config().rope_theta;
    }
    // Return the vocabulary size.
    size_t get_vocab_size() const override {
        return model_->config().vocab_size;
    }
    // Return the quantization mode flag.
    int get_quant_type() const {
        return model_->config().quant_type;
    }
    // Expose dense parameters for debugging or tooling.
    const std::unordered_map<std::string, Tensor<T>>& get_params() const {
        return model_->get_params();
    }
    // Expose AWQ qweight tensors for debugging or tooling.
    const std::unordered_map<std::string, Tensor<int32_t>>& get_qweight_params() const {
        return model_->get_qweight_params();
    }
    // Expose AWQ scale tensors for debugging or tooling.
    const std::unordered_map<std::string, Tensor<T>>& get_scales_params() const {
        return model_->get_scales_params();
    }
    // Expose AWQ zero-point tensors for debugging or tooling.
    const std::unordered_map<std::string, Tensor<int32_t>>& get_qzeros_params() const {
        return model_->get_qzeros_params();
    }

    op::WeightTensor<T> get_weight(const std::string& key) const {
        return model_->get_weight(key);
    }

    // Logits borrow session storage and remain valid until its next operation.
    // Calls complete their GPU work before returning; callers must serialize a session.
    Tensor<T> forward_eager(const Tensor<uint32_t>* input, KVCache<T>* kv_cache);
    // Execute prompt prefill through the same typed decoder sequence.
    Tensor<T> prefill_eager(const Tensor<uint32_t>* input, KVCache<T>* kv_cache);
    // Execute graph decode and return logits, independent of the sampled-forward mode flag.
    Tensor<T> forward_for_graph_logits_only(const Tensor<uint32_t>* input, KVCache<T>* kv_cache);
    // Speculative decode reuses normal eager decode logits.
    Tensor<T> speculative_forward_logits(const Tensor<uint32_t>* input, KVCache<T>* kv_cache) override {
        return forward_eager(input, kv_cache);
    }
    // Speculative prefill reuses normal eager prefill logits.
    Tensor<T> speculative_prefill_logits(const Tensor<uint32_t>* input, KVCache<T>* kv_cache) override {
        return prefill_eager(input, kv_cache);
    }

   private:
    // One bounded set of activations reused by every transformer layer.
    struct DecoderBuffers {
        Tensor<T> residual, hidden, q, k, v, attention;
        Tensor<T> attention_projected, gate, up, ffn, logits;
    };

    void init_runtime_state();
    size_t decoder_workspace_bytes(size_t rows) const;
    DecoderBuffers prepare_buffers(size_t rows, CudaWorkspaceArena& arena);
    void run_decoder(const Tensor<uint32_t>& input, KVCache<T>& cache,
                     DecoderBuffers& buffers, bool prefill);
    void validate_and_bind(const Tensor<uint32_t>* input, KVCache<T>* cache,
                           bool decode);
    void require_managed_cache() const;

    void initialize_graph_fixed_memory();
    void initialize_cuda_graph_with_kv_cache(KVCache<T>* cache);
    void prepare_graph_execution(size_t rope_offset, size_t total_seq_len,
                                 cudaStream_t stream, int pingpong_index = 0);
    Tensor<T> forward_graph_cuda(const Tensor<uint32_t>* input, KVCache<T>* cache,
                                 cudaStream_t stream);
    std::vector<size_t> decode_tensor_shape(const std::string& name) const;
    Tensor<T>& graph_tensor(const std::string& name);
    Tensor<T> run_decode_graph_attention(Tensor<T>& q, const Tensor<T>& k,
                                         const Tensor<T>& v, size_t layer,
                                         cudaStream_t stream);
    void copy_kv_cache(size_t layer, size_t offset, KVCache<T>* cache,
                       const Tensor<T>& k, const Tensor<T>& v,
                       cudaStream_t stream) const;

    std::shared_ptr<const Qwen3Model<T>> model_;
    std::unique_ptr<KVCache<T>> managed_cache_;
    bool managed_history_ready_ = false;
    cudaStream_t execution_stream_ = nullptr;
    cublasHandle_t cublas_handle_ = nullptr;
    std::unique_ptr<op::UnifiedOperators<T>> operators_;
    std::unique_ptr<CudaGraphRuntime<T>> graph_runtime_;
    std::unique_ptr<CudaWorkspaceArena> decode_workspace_;
    CudaWorkspaceArena prefill_workspace_, graph_workspace_;
    CudaWorkspaceArena sampling_workspace_, attention_storage_;
    Tensor<T> attention_workspace_;
    std::unique_ptr<op::DynamicFlashAttentionCUDAOperator<T>> decode_attention_;
    Tensor<uint32_t> sampled_token_;
    DecoderBuffers decode_buffers_, prefill_buffers_;
    size_t prefill_rows_ = 0;
    std::unordered_map<std::string, Tensor<T>> graph_tensors_;
    // The caller keeps this cache and its backing storage alive while bound.
    KVCache<T>* bound_cache_ = nullptr;
    size_t bound_capacity_ = 0;
    std::vector<std::pair<const T*, const T*>> bound_kv_bases_;
    size_t graph_rope_offsets_[2] = {};
    int graph_segment_lengths_[2] = {};
    int graph_pingpong_host_ = 0;
    bool use_cuda_graph_ = false;
};

extern template class Qwen3Session<__nv_bfloat16>;
