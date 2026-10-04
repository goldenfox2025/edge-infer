#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "qwen3.hpp"

enum class SampleMode { GPU, CPU, GPU_WITH_ASYNC_PREPARE };

// Qwen2/Llama checkpoints use the same prepared decoder as Qwen3, with Q/K
// normalization disabled. FP32 CPU execution is a direct reference path.
template <typename T>
class QwenModel : public BaseModel, public SpeculativeModel<T> {
 public:
    using Parameters = std::unordered_map<std::string, Tensor<T>>;
    using IntegerParameters = std::unordered_map<std::string, Tensor<int32_t>>;

    QwenModel(const Parameters&, const ModelConfig&);
    QwenModel(const Parameters&, const IntegerParameters&, const Parameters&,
              const IntegerParameters&, const ModelConfig&);
    ~QwenModel() override;

    bool verify_params() const override;
    void print_model_info() const override;
    uint32_t* forward(const Tensor<uint32_t>*, ThreadPool&, KVCacheBase*, size_t,
                      float, float, curandState* = nullptr) override;
    uint32_t* prefill(const Tensor<uint32_t>*, ThreadPool&, KVCacheBase*, size_t,
                      float, float, curandState* = nullptr) override;

    QwenModel& cuda() override;
    QwenModel& cpu() override;
    Device device() const override { return device_; }
    std::shared_ptr<BaseModel> fork_executor() const override;
    bool owns_execution_workspaces() const override { return true; }
    size_t estimate_prefill_workspace_bytes(size_t) const override;
    size_t get_n_layers() const override { return config_.n_layers; }
    size_t get_max_seq_len() const override { return config_.max_position_embeddings; }
    size_t get_head_dim() const override { return config_.head_dim; }
    size_t get_n_kv_heads() const override { return config_.n_kv_heads; }
    uint32_t get_eos_token_id() const override { return config_.eos_token_id; }
    size_t get_hidden_size() const override { return config_.hidden_size; }
    size_t get_vocab_size() const override { return config_.vocab_size; }
    size_t get_n_heads() const { return config_.n_heads; }
    size_t get_intermediate_size() const { return config_.intermediate_size; }
    float get_rms_norm_eps() const { return config_.rms_norm_eps; }
    float get_rope_theta() const { return config_.rope_theta; }
    int get_quant_type() const { return config_.quant_type; }
    const Parameters& get_params() const { return params_; }
    const IntegerParameters& get_qweight_params() const { return qweight_params_; }
    const Parameters& get_scales_params() const { return scales_params_; }
    const IntegerParameters& get_qzeros_params() const { return qzeros_params_; }
    op::WeightTensor<T> get_weight(const std::string&) const;

    // Logits borrow model/session storage until the next operation. An external
    // CUDA cache must outlive this adapter and its prepared session.
    TensorView<T, 2> forward_cuda(const Tensor<uint32_t>*, KVCache<T>*, const std::string& = "");
    TensorView<T, 2> prefill_cuda(const Tensor<uint32_t>*, KVCache<T>*);
    TensorView<T, 2> forward_generic(const Tensor<uint32_t>*, KVCache<T>*);
    TensorView<T, 2> prefill_generic(const Tensor<uint32_t>*, KVCache<T>*);
    TensorView<T, 2> forward_logits_only(const Tensor<uint32_t>*, KVCache<T>*);
    TensorView<T, 2> forward_for_graph_logits_only(const Tensor<uint32_t>*, KVCache<T>*);
    TensorView<T, 2> speculative_forward_logits(const Tensor<uint32_t>* in, KVCache<T>* kv) override {
        return forward_logits_only(in, kv);
    }
    TensorView<T, 2> speculative_prefill_logits(const Tensor<uint32_t>* in, KVCache<T>* kv) override {
        return device_ == Device::CUDA ? prefill_cuda(in, kv) : prefill_generic(in, kv);
    }
    void set_sample_mode(SampleMode mode) { sample_mode_ = mode; }
    uint32_t sample_cpu(TensorView<const T, 2>, float, float, size_t);
    uint32_t sample_cpu(const Tensor<T>& logits, float temperature, float top_p, size_t top_k) {
        return sample_cpu(borrow_tensor_view<2>(logits), temperature, top_p, top_k);
    }
    std::vector<uint32_t> generate(const std::vector<uint32_t>&, size_t,
                                   float = 1.0f, float = 0.9f, size_t = 50);

 private:
    struct ForkExecutorTag {};
    QwenModel(const QwenModel& prototype, ForkExecutorTag);

    struct CpuLinear {
        TensorView<const T, 2> weight;
        TensorView<const T, 1> bias;
    };
    struct CpuLayer {
        TensorView<const T, 1> attention_norm, ffn_norm;
        CpuLinear q, k, v, o, gate, up, down;
    };
    struct CpuWorkspace {
        WorkspacePlan plan;
        std::vector<T> storage;
        DecoderBuffers<T> buffers{};
        size_t rows = 0;
    };

    void prepare_cuda();
    void prepare_cpu();
    void reserve_cpu_workspace(CpuWorkspace&, size_t);
    TensorView<T, 2> execute_cpu(const Tensor<uint32_t>*, KVCache<T>*, bool);
    static void cpu_linear(TensorView<const T, 2>, const CpuLinear&, TensorView<T, 2>);
    void cpu_norm(TensorView<const T, 2>, TensorView<const T, 1>, TensorView<T, 2>) const;
    void cpu_rope(TensorView<T, 2>, size_t, size_t) const;
    uint32_t* cpu_sample_result(TensorView<const T, 2>, float, float, size_t);

    ModelConfig source_config_;
    Qwen3Config config_{};
    Device device_ = Device::CPU;
    SampleMode sample_mode_ = SampleMode::CPU;
    Parameters params_, scales_params_;
    IntegerParameters qweight_params_, qzeros_params_;
    std::shared_ptr<Qwen3Model<T>> prepared_model_;
    std::unique_ptr<Qwen3Session<T>> cuda_session_;
    CudaWorkspaceArena cpu_sample_storage_;
    std::vector<CpuLayer> cpu_layers_;
    TensorView<const T, 2> cpu_embedding_{};
    TensorView<const T, 1> cpu_output_norm_{};
    CpuLinear cpu_output_{};
    CpuWorkspace cpu_decode_, cpu_prefill_;
    std::vector<T> sample_download_;
    std::vector<std::pair<float, uint32_t>> sample_candidates_;
    uint64_t sample_rng_ = 0x2545f4914f6cdd1dULL;
};

extern template class QwenModel<float>;
extern template class QwenModel<__nv_bfloat16>;
