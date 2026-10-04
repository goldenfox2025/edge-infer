#pragma once
#include <cuda_bf16.h>  // For __nv_bfloat16 support

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "base_model.hpp"
#include "execution/cuda_workspace_arena.hpp"
#include "cuda_graph_runtime.hpp"
#include "inference.hpp"
#include "operators/unified_operators.hpp"
#include "speculative_model.hpp"
#include "tensor.hpp"
#include "thread_pool.hpp"
#include "execution/workspace_plan.hpp"
#include "weight_tensor.hpp"

// Sampling modes
enum class SampleMode {
    GPU,                    // GPU sampling (default)
    CPU,                    // CPU sampling for tests
    GPU_WITH_ASYNC_PREPARE  // GPU sampling with asynchronous prepare_next for EOS overlap
};

template <typename T>
class QwenModel : public BaseModel, public SpeculativeModel<T> {
   public:
    QwenModel(const std::unordered_map<std::string, Tensor<T>>& params,
              const ModelConfig& config);

    // Construct a model with quantization parameters.
    QwenModel(const std::unordered_map<std::string, Tensor<T>>& params,
              const std::unordered_map<std::string, Tensor<int32_t>>& qweight_params,
              const std::unordered_map<std::string, Tensor<T>>& scales_params,
              const std::unordered_map<std::string, Tensor<int32_t>>& qzeros_params,
              const ModelConfig& config);
    ~QwenModel() override;

    bool verify_params() const override;
    void print_model_info() const override;
    uint32_t* forward(const Tensor<uint32_t>* input, ThreadPool& thread_pool, KVCacheBase* kv_cache, size_t top_k,
                      float temperature, float top_p, curandState* d_states = nullptr) override {
        KVCache<T>* typed_cache = dynamic_cast<KVCache<T>*>(kv_cache);
        if (!typed_cache) {
            throw std::runtime_error("Invalid KV cache type for QwenModel::forward");
        }

        Tensor<T> logits;
        if (device_ == Device::CUDA && use_cuda_graph_) {
            logits = forward_for_graph_logits_only(input, typed_cache);
            apply_prepared_offsets();
        } else {
            logits = forward_logits_only(input, typed_cache);
        }

        if (device_ == Device::CUDA && use_cuda_graph_) {
            return sample_unified(logits, temperature, top_p, top_k, typed_cache, d_states, graph_stream());
        }
        return sample_unified(logits, temperature, top_p, top_k, typed_cache, d_states);
    }
    uint32_t* prefill(const Tensor<uint32_t>* input, ThreadPool& thread_pool, KVCacheBase* kv_cache, size_t top_k,
                      float temperature, float top_p, curandState* d_states = nullptr) override {
        KVCache<T>* typed_cache = dynamic_cast<KVCache<T>*>(kv_cache);
        if (!typed_cache) {
            throw std::runtime_error("Invalid KV cache type for QwenModel::prefill");
        }

        Tensor<T> logits =
            device_ == Device::CUDA ? prefill_cuda(input, typed_cache) : prefill_generic(input, typed_cache);

        if (logits.sizes().size() != 2 || logits.sizes()[0] == 0) {
            throw std::runtime_error("Invalid logits shape returned from Qwen prefill");
        }

        const size_t seq_len = logits.sizes()[0];
        const size_t vocab_size = logits.sizes()[1];
        Tensor<T> last_logits({1, vocab_size}, device_);

        if (device_ == Device::CUDA) {
            cudaMemcpy(last_logits.data_ptr(), logits.data_ptr() + (seq_len - 1) * vocab_size, vocab_size * sizeof(T),
                       cudaMemcpyDeviceToDevice);
        } else {
            std::copy(logits.data_ptr() + (seq_len - 1) * vocab_size, logits.data_ptr() + seq_len * vocab_size,
                      last_logits.data_ptr());
        }

        return sample_unified(last_logits, temperature, top_p, top_k, typed_cache, d_states);
    }

    // Token generation
    std::vector<uint32_t> generate(const std::vector<uint32_t>& input_ids, size_t max_length, float temperature = 1.0f,
                                   float top_p = 0.9f, size_t top_k = 50);

    // Getter methods
    size_t get_n_layers() const override {
        return n_layers_;
    }
    size_t get_max_seq_len() const override {
        return max_position_embeddings_;
    }
    size_t get_head_dim() const override {
        return head_dim_;
    }
    size_t get_n_kv_heads() const override {
        return n_kv_heads_;
    }
    uint32_t get_eos_token_id() const override {
        return eos_token_id_;
    }
    size_t get_hidden_size() const override {
        return hidden_size_;
    }
    size_t estimate_prefill_workspace_bytes(size_t seq_len) const override;

    // Additional getter methods for qwen_decode.cpp
    size_t get_n_heads() const {
        return n_heads_;
    }
    size_t get_intermediate_size() const {
        return intermediate_size_;
    }
    float get_rms_norm_eps() const {
        return rms_norm_eps_;
    }
    float get_rope_theta() const {
        return rope_theta_;
    }
    size_t get_vocab_size() const override {
        return vocab_size_;
    }
    int get_quant_type() const {
        return quant_type_;
    }
    const std::unordered_map<std::string, Tensor<T>>& get_params() const {
        return params_;
    }
    const std::unordered_map<std::string, Tensor<int32_t>>& get_qweight_params() const {
        return qweight_params_;
    }
    const std::unordered_map<std::string, Tensor<T>>& get_scales_params() const {
        return scales_params_;
    }
    const std::unordered_map<std::string, Tensor<int32_t>>& get_qzeros_params() const {
        return qzeros_params_;
    }

    // CUDA versions of forward and prefill.
    // Their implementations can be filled in later (currently as stubs mimicking
    // Llama).
    Tensor<T> forward_cuda(const Tensor<uint32_t>* input, KVCache<T>* kv_cache, const std::string& save_prefix = "");
    Tensor<T> prefill_cuda(const Tensor<uint32_t>* input, KVCache<T>* kv_cache);

    Tensor<T> forward_for_graph(const Tensor<uint32_t>* input, KVCache<T>* kv_cache, cudaStream_t stream = nullptr);

    // CPU sampling helpers
    Tensor<T> forward_logits_only(const Tensor<uint32_t>* input, KVCache<T>* kv_cache);
    Tensor<T> forward_for_graph_logits_only(const Tensor<uint32_t>* input, KVCache<T>* kv_cache);
    Tensor<T> forward_generic(const Tensor<uint32_t>* input, KVCache<T>* kv_cache);
    Tensor<T> prefill_generic(const Tensor<uint32_t>* input, KVCache<T>* kv_cache);
    Tensor<T> speculative_forward_logits(const Tensor<uint32_t>* input, KVCache<T>* kv_cache) override {
        return device_ == Device::CUDA ? forward_cuda(input, kv_cache) : forward_generic(input, kv_cache);
    }
    Tensor<T> speculative_prefill_logits(const Tensor<uint32_t>* input, KVCache<T>* kv_cache) override {
        return device_ == Device::CUDA ? prefill_cuda(input, kv_cache) : prefill_generic(input, kv_cache);
    }
    uint32_t sample_cpu(const Tensor<T>& gpu_logits, float temperature, float top_p, size_t top_k);
    uint32_t* sample_with_metadata_update(const Tensor<T>& logits, float temperature, float top_p, size_t top_k,
                                          KVCache<T>* kv_cache);
    uint32_t* allocate_gpu_result(uint32_t result);

    // Sampling helpers
    void set_sample_mode(SampleMode mode);
    uint32_t* sample_unified(const Tensor<T>& logits, float temperature, float top_p, size_t top_k,
                             KVCache<T>* kv_cache, curandState* d_states = nullptr, cudaStream_t stream = nullptr);
    uint32_t* sample_with_cpu_only(const Tensor<T>& logits, float temperature, float top_p, size_t top_k);

    // Offset preparation helpers
    void compute_next_offsets_async(int offset);
    void apply_prepared_offsets();

    // Access dense or quantized weights.
    op::WeightTensor<T> get_weight(const std::string& key) {
        if (quant_type_ == 1) {
            // Look up quantized weights.
            std::string qweight_key = key + ".qweight";
            std::string scales_key = key + ".scales";
            std::string qzeros_key = key + ".qzeros";

            auto qweight_it = qweight_params_.find(qweight_key);
            auto scales_it = scales_params_.find(scales_key);
            auto qzeros_it = qzeros_params_.find(qzeros_key);

            if (qweight_it != qweight_params_.end() && scales_it != scales_params_.end() &&
                qzeros_it != qzeros_params_.end()) {
                // Return quantized weights.
                return op::WeightTensor<T>(&qweight_it->second, &scales_it->second, &qzeros_it->second, group_size_);
            }
        }

        auto weight_it = params_.find(key);
        if (weight_it != params_.end()) {
            return op::WeightTensor<T>(&weight_it->second);
        }
        // Look up dense weights.
        std::string weight_key = key + ".weight";
        weight_it = params_.find(weight_key);

        if (weight_it != params_.end()) {
            return op::WeightTensor<T>(&weight_it->second);
        }

        // Report a missing weight with its name.
        throw std::runtime_error("Weight not found: " + key +
                                 (quant_type_ == 1 ? " (tried both quantized and regular)" : " (tried regular)"));
    }

    // Device management
    QwenModel& cuda() override;
    QwenModel& cpu() override;
    Device device() const override {
        return device_;
    }

    // CUDA graph helpers
    void initialize_cuda_runtime();
    void cleanup_cuda_runtime();
    void initialize_graph_fixed_memory();                                             // Initialize persistent graph storage
    void cleanup_graph_fixed_memory();                                                // Release persistent graph storage
    void update_rope_offset(size_t offset, cudaStream_t stream, int pongpong_index);  // Update the RoPE offset in persistent storage
    void update_segment_info(size_t total_seq_len, int layer_idx, cudaStream_t stream,
                             int pongpong_index);  // Update flash-attention segment metadata
    void extract_updateable_nodes();               // Collect updateable graph nodes
    void update_graph_kv_addresses(KVCache<T>* kv_cache,
                                   size_t offset);  // Update graph KV-copy destinations
    void update_graph_kv_addresses_async_for_next(KVCache<T>* kv_cache,
                                                  size_t next_offset);  // Prepare graph node parameters asynchronously for the next execution
    void prepare_graph_execution(size_t rope_offset, size_t total_seq_len, cudaStream_t stream,
                                 int pingpong_index = 0);  // Prepare dynamic data before graph execution

    void initialize_cuda_graph_with_kv_cache(KVCache<T>* kv_cache);  // Initialize the CUDA graph using the actual KV cache
    void initialize_decode_workspace();
    WorkspacePlan build_decode_workspace_plan() const;
    WorkspacePlan build_prefill_workspace_plan(size_t seq_len) const;
    Tensor<T> decode_workspace_tensor(const std::string& name, const std::vector<size_t>& shape) const;
    std::vector<size_t> decode_tensor_shape(const std::string& name) const;
    std::string graph_tensor_tag(const std::string& name) const;
    std::string graph_layer_tensor_tag(const std::string& name, size_t layer) const;

    // Layer output diagnostics
    void save_tensor_to_binary(const Tensor<T>& tensor,
                               const std::string& filename);  // Save a tensor to a binary file
    void save_uint32_tensor_to_binary(const Tensor<uint32_t>& tensor,
                                      const std::string& filename);           // Save a uint32 tensor to a binary file
    void save_graph_tensors_after_execution(const std::string& save_prefix);  // Save diagnostic tensors after graph execution

    // RoPE cache helpers
    const Tensor<float>& get_rope_sin_cos_cache() const {
        return rope_sin_cos_cache_;
    }  // Return the RoPE sin/cos cache
    bool has_rope_cache() const {
        return rope_sin_cos_cache_.numel() > 0;
    }  // Check whether the RoPE cache is initialized

    const Tensor<T>* find_optional_param(const std::string& key) const;
    void add_residual_and_norm(Tensor<T>* hidden_states, Tensor<T>* residual, Tensor<T>* update,
                               Tensor<T>* norm_weight, cudaStream_t stream = nullptr);
    CudaGraphRuntime<T>& graph_runtime() {
        return *graph_runtime_;
    }
    const CudaGraphRuntime<T>& graph_runtime() const {
        return *graph_runtime_;
    }
    cudaStream_t graph_stream() const {
        return graph_runtime_ ? graph_runtime_->graph_stream : nullptr;
    }

   private:
    std::array<cudaEvent_t, 3> fa_done_events_;
    size_t vocab_size_;
    size_t n_layers_;
    size_t n_heads_;
    size_t n_kv_heads_;
    size_t hidden_size_;
    size_t head_dim_;
    size_t intermediate_size_;
    size_t max_position_embeddings_;
    uint32_t bos_token_id_;
    uint32_t eos_token_id_;
    float rms_norm_eps_;
    float rope_theta_;

    std::unordered_map<std::string, Tensor<T>> params_;
    std::unordered_map<std::string, Tensor<int32_t>> qweight_params_;
    std::unordered_map<std::string, Tensor<T>> scales_params_;         // Scale factors
    std::unordered_map<std::string, Tensor<int32_t>> qzeros_params_;   // Zero points
    int quant_type_ = 0;                                               // 0: Unquantized, 1: AWQ quantized
    int group_size_ = 128;                                             // Quantization group size
    Device device_;

    std::array<cudaStream_t, kNumStreams> compute_streams_;

    // Unified operator interface
    std::unique_ptr<op::UnifiedOperators<T>> operators_;

    // CPU operator interface for sampling
    std::unique_ptr<op::UnifiedOperators<T>> cpu_operators_;

    // Sampling mode
    SampleMode sample_mode_ = SampleMode::GPU;  // GPU sampling by default

    // Precompute and cache offsets for the next iteration.
    std::vector<int> next_batch_offsets_;
    bool offsets_prepared_ = false;

    // Select CUDA graph or eager execution.
    bool use_cuda_graph_ = true;  // Enable CUDA graph execution

    // Precomputed RoPE sin/cos cache
    Tensor<float> rope_sin_cos_cache_;  // Precomputed sin/cos values with shape [max_seq_len, head_dim]
    std::unique_ptr<CudaGraphRuntime<T>> graph_runtime_;
    std::unique_ptr<CudaWorkspaceArena> decode_workspace_;
    std::unique_ptr<WorkspacePlan> decode_workspace_plan_;
};

// Extern declarations for specializations instantiated elsewhere.
// QwenModel<float> specialization
extern template class QwenModel<float>;
// QwenModel<__nv_bfloat16> specialization
extern template class QwenModel<__nv_bfloat16>;
