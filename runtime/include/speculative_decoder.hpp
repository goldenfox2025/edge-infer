#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "CudaMemoryPool.hpp"
#include "base_model.hpp"
#include "inference.hpp"
#include "speculative_model.hpp"
#include "tensor.hpp"
#include "thread_pool.hpp"

namespace op {
template <typename T>
class UnifiedOperators;
}

// Coordinate draft generation and target-model verification.
template <typename T>
class SpeculativeDecoder : public infer_base {
   public:
    // Construct from target and draft models.
    SpeculativeDecoder(std::shared_ptr<BaseModel> target_model, std::shared_ptr<BaseModel> draft_model,
                       size_t spec_length = 6,   // Default speculative length: 6
                       size_t thread_count = 8);  // Worker count (8 by default)

    // Release CUDA resources.
    ~SpeculativeDecoder();

    // Deliver generated tokens through the callback.
    void generate_with_callback(const std::vector<uint32_t>& input_ids, size_t max_length, float temperature,
                                float top_p, size_t top_k, std::function<void(uint32_t)> callback) override;
    Device device() const override {
        return device_;
    }

    // Select probability-ratio verification.
    void set_use_probability_ratio(bool use_ratio) {
        use_probability_ratio_ = use_ratio;
    }

    // Return whether probability-ratio verification is enabled.
    bool get_use_probability_ratio() const {
        return use_probability_ratio_;
    }

    // Return the adaptive speculative length.
    size_t get_adaptive_spec_length() const {
        return adaptive_spec_length_;
    }

    // Update the adaptive speculative length.
    void update_adaptive_spec_length(float acceptance_rate) {
        // Update the acceptance-rate exponential moving average.
        recent_acceptance_rate_ = 0.7f * recent_acceptance_rate_ + 0.3f * acceptance_rate;

        if (recent_acceptance_rate_ > ACCEPTANCE_THRESHOLD_HIGH) {
            // Increase the speculative length at high acceptance rates.
            adaptive_spec_length_ = std::min(adaptive_spec_length_ + 1, MAX_SPEC_LENGTH);
        } else if (recent_acceptance_rate_ < ACCEPTANCE_THRESHOLD_LOW) {
            // Decrease the speculative length at low acceptance rates.
            adaptive_spec_length_ = std::max(adaptive_spec_length_ - 1, MIN_SPEC_LENGTH);
        }
    }

   private:
    // Target model
    std::shared_ptr<BaseModel> target_model_;
    // Draft model
    std::shared_ptr<BaseModel> draft_model_;
    std::shared_ptr<SpeculativeModel<T>> target_spec_model_;
    std::shared_ptr<SpeculativeModel<T>> draft_spec_model_;
    // Target-model KV cache
    KVCache<T> target_kv_cache_;
    // Draft-model KV cache
    KVCache<T> draft_kv_cache_;

    ThreadPool thread_pool_;
    // CUDA random state
    curandState* d_states;
    std::unique_ptr<op::UnifiedOperators<T>> operators_;
    // Reuse token storage to avoid repeated allocations.
    uint32_t* d_reuse_token;
    // Persistent GPU storage for draft tokens
    uint32_t* d_draft_tokens;
    // Persistent GPU storage for draft-token probabilities
    float* d_draft_probs;
    // GPU random-number storage
    float* d_random_values;

    // Device type
    Device device_;
    // Tokens proposed in each speculative iteration
    size_t spec_length_;
    // Enable probability-ratio verification
    bool use_probability_ratio_ = true;

    // Adaptive speculative-length parameters
    float recent_acceptance_rate_ = 0.5f;  // Recent acceptance rate
    size_t adaptive_spec_length_;          // Adaptive speculative length
    static constexpr size_t MIN_SPEC_LENGTH = 6;   // Minimum speculative length
    static constexpr size_t MAX_SPEC_LENGTH = 8;   // Maximum speculative length
    static constexpr float ACCEPTANCE_THRESHOLD_HIGH = 0.7f;  // High acceptance-rate threshold
    static constexpr float ACCEPTANCE_THRESHOLD_LOW = 0.4f;   // Low acceptance-rate threshold

    // CUDA streams for asynchronous operations
    cudaStream_t main_stream_;    // Main execution stream
    cudaStream_t draft_stream_;   // Draft-model stream
    cudaStream_t verify_stream_;  // Verification stream

    static constexpr const char* kReuseTokenTag = "spec_reuse_token";
    static constexpr const char* kDraftTokensTag = "spec_draft_tokens";
    static constexpr const char* kDraftProbsTag = "spec_draft_probs";
    static constexpr const char* kRandomValuesTag = "spec_random_values";

    // Initialize CUDA resources.
    void init_cuda_resources();
    // Release CUDA resources.
    void free_cuda_resources();

    // Verify draft tokens greedily by comparing token IDs.
    size_t verify_draft_tokens_greedy(const std::vector<uint32_t>& prefix_tokens,
                                      std::vector<uint32_t*>& draft_tokens_gpu, float temperature, float top_p,
                                      size_t top_k, std::vector<uint32_t>& verified_tokens, cudaStream_t stream);

    // Verify draft tokens using probability ratios.
    size_t verify_draft_tokens_prob_ratio(const std::vector<uint32_t>& prefix_tokens,
                                          std::vector<uint32_t*>& draft_tokens_gpu, float temperature, float top_p,
                                          size_t top_k, std::vector<uint32_t>& verified_tokens, cudaStream_t stream);

    // Dispatch to the selected draft-token verification method.
    size_t verify_draft_tokens_gpu(const std::vector<uint32_t>& prefix_tokens, std::vector<uint32_t*>& draft_tokens_gpu,
                                   float temperature, float top_p, size_t top_k, std::vector<uint32_t>& verified_tokens,
                                   cudaStream_t stream) {
        if (use_probability_ratio_) {
            return verify_draft_tokens_prob_ratio(prefix_tokens, draft_tokens_gpu, temperature, top_p, top_k,
                                                  verified_tokens, stream);
        } else {
            return verify_draft_tokens_greedy(prefix_tokens, draft_tokens_gpu, temperature, top_p, top_k,
                                              verified_tokens, stream);
        }
    }

    // Generate draft tokens and return their GPU pointers.
    std::vector<uint32_t*> generate_draft_tokens_gpu(uint32_t* input_token, size_t num_tokens, float temperature,
                                                     float top_p, size_t top_k);

    // Generate draft tokens and probabilities, returning GPU pointers.
    std::pair<std::vector<uint32_t*>, std::vector<float*>> generate_draft_tokens_with_probs_gpu(
        uint32_t* input_token, size_t num_tokens, float temperature, float top_p, size_t top_k);

    // Combine draft-token GPU values into a tensor.
    Tensor<uint32_t> combine_draft_tokens(std::vector<uint32_t*>& draft_tokens_gpu);
};

// Explicit template declarations
extern template class SpeculativeDecoder<__nv_bfloat16>;
