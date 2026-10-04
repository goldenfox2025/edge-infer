#pragma once
#include <array>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "base_model.hpp"
#include "common.hpp"
#include "execution/context.hpp"
#include "execution/cuda_workspace_arena.hpp"
#include "cuda_graph_runtime.hpp"
#include "execution/program.hpp"
#include "inference.hpp"
#include "operators/unified_operators.hpp"
#include "speculative_model.hpp"
#include "tensor.hpp"
#include "thread_pool.hpp"
#include "execution/workspace_plan.hpp"

template <typename T>
class Qwen3Model : public BaseModel, public SpeculativeModel<T> {
   public:
    // Construct a dense-weight Qwen3 model and initialize runtime state.
    Qwen3Model(const std::unordered_map<std::string, Tensor<T>>& params,
               const ModelConfig& config);

    // Construct an AWQ-weight Qwen3 model and initialize runtime state.
    Qwen3Model(const std::unordered_map<std::string, Tensor<T>>& params,
               const std::unordered_map<std::string, Tensor<int32_t>>& qweight_params,
               const std::unordered_map<std::string, Tensor<T>>& scales_params,
               const std::unordered_map<std::string, Tensor<int32_t>>& qzeros_params,
               const ModelConfig& config);
    // Release CUDA runtime state, graph state, and auxiliary streams/events.
    ~Qwen3Model() override;

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

    // Move model weights to CUDA and refresh CUDA-only runtime state.
    Qwen3Model& cuda() override;
    // Disable CUDA graph/runtime state and reject CPU execution.
    Qwen3Model& cpu() override;
    // Return the current device backing this model.
    Device device() const override {
        return device_;
    }

    // Return the number of transformer blocks.
    size_t get_n_layers() const override {
        return n_layers_;
    }
    // Return the configured maximum sequence length.
    size_t get_max_seq_len() const override {
        return max_position_embeddings_;
    }
    // Return per-head hidden width.
    size_t get_head_dim() const override {
        return head_dim_;
    }
    // Return the number of KV heads.
    size_t get_n_kv_heads() const override {
        return n_kv_heads_;
    }
    // Return the EOS token id.
    uint32_t get_eos_token_id() const override {
        return eos_token_id_;
    }

    // Return the number of attention heads.
    size_t get_n_heads() const {
        return n_heads_;
    }
    // Return the hidden size.
    size_t get_hidden_size() const {
        return hidden_size_;
    }
    // Estimate workspace bytes needed for prefill at the given sequence length.
    size_t estimate_prefill_workspace_bytes(size_t seq_len) const override;
    // Return the MLP intermediate size.
    size_t get_intermediate_size() const {
        return intermediate_size_;
    }
    // Return the RMSNorm epsilon used by this model.
    float get_rms_norm_eps() const {
        return rms_norm_eps_;
    }
    // Return the RoPE theta used by this model.
    float get_rope_theta() const {
        return rope_theta_;
    }
    // Return the vocabulary size.
    size_t get_vocab_size() const override {
        return vocab_size_;
    }
    // Return the quantization mode flag.
    int get_quant_type() const {
        return quant_type_;
    }
    // Expose dense parameters for debugging or tooling.
    const std::unordered_map<std::string, Tensor<T>>& get_params() const {
        return params_;
    }
    // Expose AWQ qweight tensors for debugging or tooling.
    const std::unordered_map<std::string, Tensor<int32_t>>& get_qweight_params() const {
        return qweight_params_;
    }
    // Expose AWQ scale tensors for debugging or tooling.
    const std::unordered_map<std::string, Tensor<T>>& get_scales_params() const {
        return scales_params_;
    }
    // Expose AWQ zero-point tensors for debugging or tooling.
    const std::unordered_map<std::string, Tensor<int32_t>>& get_qzeros_params() const {
        return qzeros_params_;
    }

    // Return either dense or AWQ weight storage for a logical weight name.
    op::WeightTensor<T> get_weight(const std::string& key) const {
        if (quant_type_ == 1) {
            // Look up quantized weights.
            // Qwen3 weight names do not need a .qweight suffix.
            auto qweight_it = qweight_params_.find(key);
            auto scales_it = scales_params_.find(key);
            auto qzeros_it = qzeros_params_.find(key);

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

        // Report a missing weight with its name.
        throw std::runtime_error("Weight not found: " + key +
                                 (quant_type_ == 1 ? " (tried both quantized and regular)" : " (tried regular)"));
    }

    // Execute one-token decode through prepared nodes.
    Tensor<T> forward_eager(const Tensor<uint32_t>* input, KVCache<T>* kv_cache);
    // Execute prompt prefill through prepared nodes.
    Tensor<T> prefill_eager(const Tensor<uint32_t>* input, KVCache<T>* kv_cache);
    // Execute the decode CUDA graph body.
    Tensor<T> forward_graph_cuda(const Tensor<uint32_t>* input, KVCache<T>* kv_cache,
                                 cudaStream_t stream = nullptr);
    // Select graph or eager decode and return logits only.
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
    enum class AttentionMode : uint8_t {
        Decode,
        Prefill,
    };

    struct AttentionBiases {
        const Tensor<T>* q = nullptr;
        const Tensor<T>* k = nullptr;
        const Tensor<T>* v = nullptr;
        const Tensor<T>* o = nullptr;
    };

    struct MlpBiases {
        const Tensor<T>* gate = nullptr;
        const Tensor<T>* up = nullptr;
        const Tensor<T>* down = nullptr;
    };

    struct DecodeEagerFrame {
        const Tensor<uint32_t>* input = nullptr;
        KVCache<T>* kv_cache = nullptr;
        size_t seq_len = 1;
        size_t offset = 0;
    };

    struct DecodeEagerBuffers {
        std::unordered_map<std::string, Tensor<T>> tensors;

        Tensor<T>& require(const std::string& name) {
            auto it = tensors.find(name);
            if (it == tensors.end()) {
                throw std::runtime_error("Qwen3 decode eager buffer not found: " + name);
            }
            return it->second;
        }

        const Tensor<T>& require(const std::string& name) const {
            auto it = tensors.find(name);
            if (it == tensors.end()) {
                throw std::runtime_error("Qwen3 decode eager buffer not found: " + name);
            }
            return it->second;
        }
    };

    struct DecodePreparedNode {
        std::shared_ptr<op::OperatorBase> op;
        std::vector<std::string> inputs;
        std::vector<std::string> outputs;
        std::vector<op::PackedTensorArg> tensor_args;
        const Tensor<T>* tensor_arg = nullptr;
        std::optional<op::WeightTensor<T>> weight;
        const Tensor<T>* bias = nullptr;
        uint16_t view_heads = 0;
        size_t layer = 0;

        template <typename BufferLookup, typename KvTensorFn>
        void execute(BufferLookup&& buffer_lookup, KvTensorFn&& kv_tensors,
                     const Tensor<uint32_t>* input_tensor, size_t seq_len,
                     size_t offset, size_t n_heads, size_t n_kv_heads,
                     size_t head_dim, float rms_norm_eps, float rope_theta,
                     cudaStream_t stream = nullptr) const {
            if (!op) {
                throw std::runtime_error("Qwen3 prepared node missing operator");
            }

            std::vector<void*> tensors;
            std::vector<void*> statics;
            std::vector<uint64_t> runtime = {
                static_cast<uint64_t>(offset),
                0,
                0,
                static_cast<uint64_t>(n_heads),
                static_cast<uint64_t>(n_kv_heads),
                static_cast<uint64_t>(head_dim),
                static_cast<uint64_t>(seq_len),
                static_cast<uint64_t>(offset + seq_len),
                static_cast<uint64_t>(view_heads),
            };
            std::optional<std::pair<Tensor<T>, Tensor<T>>> kv_cache_pair;
            auto pack_float = [](float value) -> uint64_t {
                uint32_t bits = 0;
                std::memcpy(&bits, &value, sizeof(float));
                return static_cast<uint64_t>(bits);
            };
            runtime[op::kPackedRuntimeThetaBits] = pack_float(rope_theta);
            runtime[op::kPackedRuntimeEpsBits] = pack_float(rms_norm_eps);

            auto require_kv_cache_pair = [&]() -> std::pair<Tensor<T>, Tensor<T>>& {
                if (!kv_cache_pair.has_value()) {
                    kv_cache_pair = kv_tensors(layer);
                }
                return *kv_cache_pair;
            };

            for (const auto& spec : tensor_args) {
                switch (spec.kind) {
                    case op::PackedTensorArgKind::OutputTensor:
                        tensors.push_back(&buffer_lookup(outputs.at(spec.index)));
                        break;
                    case op::PackedTensorArgKind::InputTensor:
                        tensors.push_back(&buffer_lookup(inputs.at(spec.index)));
                        break;
                    case op::PackedTensorArgKind::RuntimeInputTensor:
                        tensors.push_back(const_cast<Tensor<uint32_t>*>(input_tensor));
                        break;
                    case op::PackedTensorArgKind::KvCacheKTensor: {
                        auto& pair = require_kv_cache_pair();
                        tensors.push_back(const_cast<Tensor<T>*>(&pair.first));
                        break;
                    }
                    case op::PackedTensorArgKind::KvCacheVTensor: {
                        auto& pair = require_kv_cache_pair();
                        tensors.push_back(const_cast<Tensor<T>*>(&pair.second));
                        break;
                    }
                }
            }

            if (tensor_arg) {
                statics.push_back(const_cast<Tensor<T>*>(tensor_arg));
            }
            if (weight) {
                statics.push_back(const_cast<op::WeightTensor<T>*>(&*weight));
            }
            if (bias) {
                statics.push_back(
                    const_cast<Tensor<T>*>(static_cast<const Tensor<T>*>(bias)));
            }

            op->execute_packed(tensors, statics, runtime, stream);
        }
    };

    struct DecodeRuntimeArtifacts {
        ExecutionProgram program;
        std::vector<DecodePreparedNode> nodes;
    };

    static const char* attention_semantic(AttentionMode mode) {
        return mode == AttentionMode::Decode ? "decode_attention"
                                             : "prefill_attention";
    }

    // Load scalar model dimensions and token ids from the serialized config.
    void load_config(const ModelConfig& config);
    // Initialize operators, graph state, streams, RoPE cache, and decode workspace.
    void init_runtime_state();
    // Allocate fixed CUDA graph buffers used by decode seq_len=1 replay.
    void initialize_graph_fixed_memory();
    // Precompute RoPE sin/cos tables on CUDA for graph execution.
    void precompute_rope_cache();
    // Update graph-side runtime scalars before replay.
    void prepare_graph_execution(size_t rope_offset, size_t total_seq_len, cudaStream_t stream,
                                 int pingpong_index = 0);
    // Capture the decode CUDA graph and bind its fixed graph tensors.
    void initialize_cuda_graph_with_kv_cache(KVCache<T>* kv_cache);
    // Build and reserve the fixed decode workspace for seq_len=1 eager execution.
    void initialize_decode_workspace();
    // Build decoder IR plus prepared nodes for either decode or prefill mode.
    DecodeRuntimeArtifacts build_decoder_runtime_artifacts(size_t seq_len,
                                                           AttentionMode mode);
    // Materialize one named decode workspace allocation as a Tensor view.
    Tensor<T> decode_workspace_tensor(const std::string& name, const std::vector<size_t>& shape) const;
    // Return the canonical decode-graph tensor shape for legacy fixed-address graph tensors.
    std::vector<size_t> decode_tensor_shape(const std::string& name) const;
    // Materialize all fixed decode workspace tensors by value name.
    DecodeEagerBuffers materialize_decode_eager_buffers() const;
    // Build a stable tag prefix for graph-owned tensors.
    std::string graph_tensor_tag(const std::string& name) const;
    // Build a stable layer-qualified tag for graph-owned tensors.
    std::string graph_layer_tensor_tag(const std::string& name, size_t layer) const;
    // Look up an optional dense tensor parameter and return null when absent.
    const Tensor<T>* find_optional_param(const std::string& key) const;
    // Load optional attention biases for one transformer layer.
    AttentionBiases load_attention_biases(size_t layer) const;
    // Load optional MLP biases for one transformer layer.
    MlpBiases load_mlp_biases(size_t layer) const;
    // Append the correct attention op to execution IR for the selected mode.
    ExecutionValueHandle build_attention_value(ExecutionBuilder<T>& builder,
                                               AttentionMode mode,
                                               const std::string& out_name,
                                               std::vector<ExecutionValueHandle> inputs) const;
    // Append the prepared attention node for the selected mode.
    void append_attention_node(DecodeRuntimeArtifacts& artifacts,
                               AttentionMode mode,
                               const std::string& out_name,
                               std::vector<std::string> inputs,
                               size_t layer) const;
    // Run the fixed-address graph attention helper used only by decode CUDA graph.
    Tensor<T> run_decode_graph_attention(Tensor<T>& q_buf_view,
                                         const Tensor<T>& k_cache_view,
                                         const Tensor<T>& v_cache_view,
                                         size_t layer, cudaStream_t stream);
    // Copy freshly produced K/V tensors into the logical KV cache.
    void copy_kv_cache(size_t layer, size_t offset, KVCache<T>* kv_cache,
                       const Tensor<T>& k_buf_view,
                       const Tensor<T>& v_buf_view,
                       cudaStream_t stream = nullptr) const;
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
    Tensor<float> rope_sin_cos_cache_;
    std::unique_ptr<CudaGraphRuntime<T>> graph_runtime_;
    std::unique_ptr<CudaWorkspaceArena> decode_workspace_;
    std::vector<ExecutionValue> decode_values_;
    std::unique_ptr<WorkspacePlan> decode_workspace_plan_;
    std::vector<DecodePreparedNode> decode_eager_nodes_;
    DecodeEagerFrame decode_eager_frame_;
    bool use_cuda_graph_ = false;
};

// Extern declarations for specializations instantiated elsewhere.
// Qwen3Model<__nv_bfloat16> specialization
extern template class Qwen3Model<__nv_bfloat16>;
