#include "speculative_decoder.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common.hpp"
#include "operators/unified_operators.hpp"

template <typename T>
SpeculativeDecoder<T>::SpeculativeDecoder(std::shared_ptr<BaseModel> target_model,
                                          std::shared_ptr<BaseModel> draft_model, size_t spec_length,
                                          size_t thread_count)
    : target_model_(target_model),
      draft_model_(draft_model),
      target_kv_cache_(target_model->get_n_layers(), target_model->get_max_seq_len(),
                       target_model->get_head_dim() * target_model->get_n_kv_heads(), Device::CUDA),
      draft_kv_cache_(draft_model->get_n_layers(), draft_model->get_max_seq_len(),
                      draft_model->get_head_dim() * draft_model->get_n_kv_heads(), Device::CUDA),
      thread_pool_(thread_count),
      device_(Device::CUDA),
      spec_length_(spec_length),
      adaptive_spec_length_(spec_length),
      d_states(nullptr),
      operators_(nullptr),
      d_reuse_token(nullptr),
      main_stream_(nullptr),
      draft_stream_(nullptr),
      verify_stream_(nullptr) {

    if (target_model_->device() != Device::CUDA) {
        target_model_->cuda();
    }
    if (draft_model_->device() != Device::CUDA) {
        draft_model_->cuda();
    }
    target_spec_model_ = std::dynamic_pointer_cast<SpeculativeModel<T>>(target_model_);
    draft_spec_model_ = std::dynamic_pointer_cast<SpeculativeModel<T>>(draft_model_);
    if (!target_spec_model_ || !draft_spec_model_) {
        throw std::runtime_error("Speculative decoding requires models that implement SpeculativeModel");
    }
    operators_ = std::make_unique<op::UnifiedOperators<T>>(Device::CUDA);
    draft_kv_cache_.clear();
    target_kv_cache_.clear();

    init_cuda_resources();

}

template <typename T>
SpeculativeDecoder<T>::~SpeculativeDecoder() {

    free_cuda_resources();
}

template <typename T>
void SpeculativeDecoder<T>::init_cuda_resources() {

    cudaMalloc(&d_states, sizeof(curandState));
    int seed = std::chrono::system_clock::now().time_since_epoch().count();
    operators_->init_curand(d_states, seed, 0, nullptr);

    cudaStreamCreate(&main_stream_);
    cudaStreamCreate(&draft_stream_);
    cudaStreamCreate(&verify_stream_);

    // Reuse tagged storage to avoid allocation churn between decode steps.
    if (GlobalCudaMemoryPool::has_tag(kReuseTokenTag)) {
        d_reuse_token = static_cast<uint32_t*>(GlobalCudaMemoryPool::get_tagged_memory(kReuseTokenTag));
    } else {
        d_reuse_token = static_cast<uint32_t*>(GlobalCudaMemoryPool::allocate_tagged(kReuseTokenTag, sizeof(uint32_t)));
    }

    // Fall back to a direct CUDA allocation if tagged allocation fails.
    if (d_reuse_token == nullptr) {
        (cudaMalloc(&d_reuse_token, sizeof(uint32_t)));
    }

    if (GlobalCudaMemoryPool::has_tag(kDraftTokensTag)) {
        d_draft_tokens = static_cast<uint32_t*>(GlobalCudaMemoryPool::get_tagged_memory(kDraftTokensTag));
    } else {
        d_draft_tokens = static_cast<uint32_t*>(
            GlobalCudaMemoryPool::allocate_tagged(kDraftTokensTag,
                                                  sizeof(uint32_t) * (spec_length_ + 1)));  // One extra slot holds the input token.
    }

    // Fall back to a direct CUDA allocation if tagged allocation fails.
    if (d_draft_tokens == nullptr) {
        (cudaMalloc(&d_draft_tokens, sizeof(uint32_t) * (spec_length_ + 1)));
    }

    if (GlobalCudaMemoryPool::has_tag(kDraftProbsTag)) {
        d_draft_probs = static_cast<float*>(GlobalCudaMemoryPool::get_tagged_memory(kDraftProbsTag));
    } else {
        d_draft_probs = static_cast<float*>(
            GlobalCudaMemoryPool::allocate_tagged(kDraftProbsTag,
                                                  sizeof(float) * (spec_length_ + 1)));  // One extra slot holds the input token.
    }

    // Fall back to a direct CUDA allocation if tagged allocation fails.
    if (d_draft_probs == nullptr) {
        (cudaMalloc(&d_draft_probs, sizeof(float) * (spec_length_ + 1)));
    }

    if (GlobalCudaMemoryPool::has_tag(kRandomValuesTag)) {
        d_random_values = static_cast<float*>(GlobalCudaMemoryPool::get_tagged_memory(kRandomValuesTag));
    } else {
        d_random_values =
            static_cast<float*>(GlobalCudaMemoryPool::allocate_tagged(kRandomValuesTag, sizeof(float) * spec_length_));
    }

    // Fall back to a direct CUDA allocation if tagged allocation fails.
    if (d_random_values == nullptr) {
        (cudaMalloc(&d_random_values, sizeof(float) * spec_length_));
    }
}

template <typename T>
void SpeculativeDecoder<T>::free_cuda_resources() {
    // Wait for work on every owned stream before releasing resources.
    if (main_stream_) {
        cudaStreamSynchronize(main_stream_);
        cudaStreamDestroy(main_stream_);
        main_stream_ = nullptr;
    }

    if (draft_stream_) {
        cudaStreamSynchronize(draft_stream_);
        cudaStreamDestroy(draft_stream_);
        draft_stream_ = nullptr;
    }

    if (verify_stream_) {
        cudaStreamSynchronize(verify_stream_);
        cudaStreamDestroy(verify_stream_);
        verify_stream_ = nullptr;
    }

    if (d_states != nullptr) {
        cudaDeviceSynchronize();
        cudaFree(d_states);
        d_states = nullptr;
    }

    // The pool retains ownership of tagged allocations.
    if (d_reuse_token != nullptr && !GlobalCudaMemoryPool::has_tag(kReuseTokenTag)) {
        cudaFree(d_reuse_token);
        d_reuse_token = nullptr;
    }

    if (d_draft_tokens != nullptr && !GlobalCudaMemoryPool::has_tag(kDraftTokensTag)) {
        cudaFree(d_draft_tokens);
        d_draft_tokens = nullptr;
    }

    if (d_draft_probs != nullptr && !GlobalCudaMemoryPool::has_tag(kDraftProbsTag)) {
        cudaFree(d_draft_probs);
        d_draft_probs = nullptr;
    }

    if (d_random_values != nullptr && !GlobalCudaMemoryPool::has_tag(kRandomValuesTag)) {
        cudaFree(d_random_values);
        d_random_values = nullptr;
    }
}

// Generate draft tokens and return pointers to their device storage.
template <typename T>
std::vector<uint32_t*> SpeculativeDecoder<T>::generate_draft_tokens_gpu(uint32_t* input_token, size_t num_tokens,
                                                                        float temperature, float top_p, size_t top_k) {

    std::vector<uint32_t*> gpu_tokens;

    GpuTimer draft_total_timer;
    draft_total_timer.start();

    try {

        cudaMemcpyAsync(d_draft_tokens, input_token, sizeof(uint32_t), cudaMemcpyDeviceToDevice, draft_stream_);

        gpu_tokens.push_back(d_draft_tokens);

        std::cout << "[Draft] Generating " << num_tokens << " draft tokens" << std::endl;
        float total_forward_time = 0.0f;
        float total_sample_time = 0.0f;

        for (size_t i = 0; i < num_tokens; i++) {

            uint32_t* d_current_token = d_draft_tokens + i;
            Tensor<uint32_t> input(d_current_token, {1}, device_);

            draft_kv_cache_.resize(draft_kv_cache_.size() + 1);

            uint32_t* next_token_ptr = d_draft_tokens + i + 1;

            GpuTimer forward_timer;
            forward_timer.start();

            Tensor<T> logits = draft_spec_model_->speculative_forward_logits(&input, &draft_kv_cache_);

            forward_timer.stop();
            float forward_time = forward_timer.milliseconds();
            total_forward_time += forward_time;

            GpuTimer sample_timer;
            sample_timer.start();

            operators_->sample_to_fixed(std::move(logits), next_token_ptr, temperature, top_p, 1, d_states,
                                        draft_stream_);

            sample_timer.stop();
            float sample_time = sample_timer.milliseconds();
            total_sample_time += sample_time;

            // Retain device pointers to avoid per-token transfers to the host.
            gpu_tokens.push_back(next_token_ptr);

            // Check EOS only on the final draft step to avoid repeated synchronization.
            if (i == num_tokens - 1) {

                uint32_t token_value;
                cudaMemcpyAsync(&token_value, next_token_ptr, sizeof(uint32_t), cudaMemcpyDeviceToHost, draft_stream_);
                cudaStreamSynchronize(draft_stream_);

                if (token_value == draft_model_->get_eos_token_id()) {
                    break;
                }
            }
        }

        draft_total_timer.stop();
        float draft_total_time = draft_total_timer.milliseconds();

        std::cout << "[Draft] Generated " << gpu_tokens.size() - 1 << " draft tokens" << std::endl;

        std::vector<uint32_t> host_draft_tokens(gpu_tokens.size() - 1);
        for (size_t i = 1; i < gpu_tokens.size(); i++) {
            uint32_t token_value;
            cudaMemcpyAsync(&token_value, gpu_tokens[i], sizeof(uint32_t), cudaMemcpyDeviceToHost, draft_stream_);
            host_draft_tokens[i - 1] = token_value;
        }
        cudaStreamSynchronize(draft_stream_);

        std::cout << "[Draft tokens] ";
        for (size_t i = 0; i < host_draft_tokens.size(); i++) {
            std::cout << host_draft_tokens[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "[Timing] Draft model forward: " << total_forward_time
                  << "ms (average: " << (total_forward_time / (gpu_tokens.size() - 1)) << "ms/token), "
                  << "Sampling: " << total_sample_time << "ms (average: " << (total_sample_time / (gpu_tokens.size() - 1))
                  << "ms/token), "
                  << "Total time: " << draft_total_time << "ms" << std::endl;

    } catch (const std::exception& e) {

        std::cout << "[Draft] Token generation failed: " << e.what() << std::endl;
    }

    return gpu_tokens;
}

// Generate draft tokens and probabilities in device storage.
template <typename T>
std::pair<std::vector<uint32_t*>, std::vector<float*>> SpeculativeDecoder<T>::generate_draft_tokens_with_probs_gpu(
    uint32_t* input_token, size_t num_tokens, float temperature, float top_p, size_t top_k) {

    std::vector<uint32_t*> gpu_tokens;
    std::vector<float*> gpu_probs;

    GpuTimer draft_total_timer;
    draft_total_timer.start();

    try {

        cudaMemcpyAsync(d_draft_tokens, input_token, sizeof(uint32_t), cudaMemcpyDeviceToDevice, draft_stream_);

        gpu_tokens.push_back(d_draft_tokens);
        gpu_probs.push_back(d_draft_probs);  // Keep a placeholder probability slot so token and probability indices match.

        std::cout << "[Draft] Generating " << num_tokens << " draft tokens with probabilities" << std::endl;
        float total_forward_time = 0.0f;
        float total_sample_time = 0.0f;

        for (size_t i = 0; i < num_tokens; i++) {

            uint32_t* d_current_token = d_draft_tokens + i;
            Tensor<uint32_t> input(d_current_token, {1}, device_);

            draft_kv_cache_.resize(draft_kv_cache_.size() + 1);

            uint32_t* next_token_ptr = d_draft_tokens + i + 1;
            float* next_prob_ptr = d_draft_probs + i + 1;

            GpuTimer forward_timer;
            forward_timer.start();

            Tensor<T> logits = draft_spec_model_->speculative_forward_logits(&input, &draft_kv_cache_);

            forward_timer.stop();
            float forward_time = forward_timer.milliseconds();
            total_forward_time += forward_time;

            GpuTimer sample_timer;
            sample_timer.start();

            operators_->sample_to_fixed_with_prob(std::move(logits), next_token_ptr, next_prob_ptr, temperature,
                                                  top_p, top_k, d_states, draft_stream_);

            sample_timer.stop();
            float sample_time = sample_timer.milliseconds();
            total_sample_time += sample_time;

            // Retain device pointers to avoid per-token transfers to the host.
            gpu_tokens.push_back(next_token_ptr);
            gpu_probs.push_back(next_prob_ptr);

            if (i >= num_tokens - 2) {
                uint32_t token_value;
                cudaMemcpyAsync(&token_value, next_token_ptr, sizeof(uint32_t), cudaMemcpyDeviceToHost, draft_stream_);
                cudaStreamSynchronize(draft_stream_);

                if (token_value == draft_model_->get_eos_token_id()) {
                    break;
                }
            }
        }

        cudaStreamSynchronize(draft_stream_);

        draft_total_timer.stop();
        float draft_total_time = draft_total_timer.milliseconds();

        std::cout << "[Draft] Generated " << gpu_tokens.size() - 1 << " draft tokens with probabilities" << std::endl;

        std::vector<uint32_t> host_draft_tokens(gpu_tokens.size() - 1);
        std::vector<float> host_draft_probs(gpu_probs.size() - 1);
        for (size_t i = 1; i < gpu_tokens.size(); i++) {
            uint32_t token_value;
            float prob_value;
            cudaMemcpyAsync(&token_value, gpu_tokens[i], sizeof(uint32_t), cudaMemcpyDeviceToHost, draft_stream_);
            cudaMemcpyAsync(&prob_value, gpu_probs[i], sizeof(float), cudaMemcpyDeviceToHost, draft_stream_);
            host_draft_tokens[i - 1] = token_value;
            host_draft_probs[i - 1] = prob_value;
        }
        cudaStreamSynchronize(draft_stream_);

        std::cout << "[Draft tokens and probabilities]" << std::endl;
        for (size_t i = 0; i < host_draft_tokens.size(); i++) {
            std::cout << host_draft_tokens[i] << "(" << host_draft_probs[i] << ") ";
        }
        std::cout << std::endl;

        std::cout << "[Timing] Draft model forward: " << total_forward_time
                  << "ms (average: " << (total_forward_time / (gpu_tokens.size() - 1)) << "ms/token), "
                  << "Sampling: " << total_sample_time << "ms (average: " << (total_sample_time / (gpu_tokens.size() - 1))
                  << "ms/token), "
                  << "Total time: " << draft_total_time << "ms" << std::endl;

    } catch (const std::exception& e) {

        std::cout << "[Draft] Token generation failed: " << e.what() << std::endl;
    }

    return {gpu_tokens, gpu_probs};
}

template <typename T>
void SpeculativeDecoder<T>::generate_with_callback(const std::vector<uint32_t>& input_ids, size_t max_length,
                                                   float temperature, float top_p, size_t top_k,
                                                   std::function<void(uint32_t)> callback) {
    try {
        std::vector<uint32_t> current_ids;
        current_ids.reserve(max_length);
        current_ids = input_ids;

        if (current_ids.empty()) {
            std::cout << "Error: input sequence is empty" << std::endl;
            return;
        }

        std::cout << "[Initialization] Input sequence length: " << current_ids.size() << std::endl;
        std::cout << "[Initialization] Using " << (use_probability_ratio_ ? "probability-ratio " : "greedy ") << "speculative sampling"
                  << std::endl;

        uint32_t first_target_token_value = -1;

        {

            GpuTimer target_init_timer;
            target_init_timer.start();

            // Reserve KV space for the complete prompt.
            // The caller must extend the logical KV length before prefill or decode.

            target_kv_cache_.resize(target_kv_cache_.size() + current_ids.size());

            Tensor<uint32_t> input_tensor(std::vector<uint32_t>(current_ids.begin(), current_ids.end()),
                                         {current_ids.size()}, device_);
            uint32_t* first_token = target_model_->prefill(&input_tensor, thread_pool_, &target_kv_cache_, top_k,
                                                           temperature, top_p, d_states);

            if (first_token == nullptr) {
                std::cout << "Warning: target model initial prefill returned a null pointer" << std::endl;
                return;
            }

            cudaMemcpyAsync(&first_target_token_value, first_token, sizeof(uint32_t), cudaMemcpyDeviceToHost,
                            main_stream_);
            cudaStreamSynchronize(main_stream_);

            if (first_target_token_value >= target_model_->get_vocab_size()) {
                throw std::runtime_error("Target model prefill returned an out-of-range token: " +
                                         std::to_string(first_target_token_value));
            }

            target_init_timer.stop();
            float target_init_time = target_init_timer.milliseconds();

            std::cout << "[Initialization] Target model initialized in: " << target_init_time << "ms" << std::endl;

            callback(first_target_token_value);

            if (first_target_token_value == target_model_->get_eos_token_id()) {
                std::cout << "[Initialization] Target model generated EOS; stopping generation" << std::endl;
                return;
            }
        }

        {

            GpuTimer draft_init_timer;
            draft_init_timer.start();
            draft_kv_cache_.resize(draft_kv_cache_.size() + current_ids.size());

            Tensor<uint32_t> draft_input_tensor(std::vector<uint32_t>(current_ids.begin(), current_ids.end()),
                                                {current_ids.size()}, device_);
            uint32_t* draft_first_token = draft_model_->prefill(&draft_input_tensor, thread_pool_, &draft_kv_cache_,
                                                                top_k, temperature, top_p, d_states);

            if (draft_first_token == nullptr) {
                throw std::runtime_error("Draft model initial prefill returned a null pointer");
            }

            draft_init_timer.stop();
            float draft_init_time = draft_init_timer.milliseconds();

            std::cout << "[Initialization] Draft model initialized in: " << draft_init_time << "ms" << std::endl;
        }

        current_ids.push_back(first_target_token_value);

        size_t max_iterations = max_length - current_ids.size();
        size_t iteration = 0;

        std::cout << "[Decode] Starting speculative decoding; maximum iterations: " << max_iterations << std::endl;

        GpuTimer total_spec_timer;
        total_spec_timer.start();

        while (current_ids.size() < max_length && iteration < max_iterations) {
            try {

                GpuTimer iteration_timer;
                iteration_timer.start();

                std::cout << "[Decode] Iteration " << iteration + 1 << "/" << max_iterations << std::endl;

                std::cout << "[KV cache] Target model: " << target_kv_cache_.size()
                          << ", draft model: " << draft_kv_cache_.size() << std::endl;

                std::cout << "[Token sequence] ";
                for (size_t i = 0; i < std::min(current_ids.size(), size_t(10)); i++) {
                    std::cout << current_ids[i] << " ";
                }
                if (current_ids.size() > 10) {
                    std::cout << "... (total: " << current_ids.size() << " tokens)";
                }
                std::cout << std::endl;

                const uint32_t last_token = current_ids.back();
                std::cout << "[Current token] " << last_token << std::endl;
                cudaMemcpyAsync(d_reuse_token, &last_token, sizeof(uint32_t), cudaMemcpyHostToDevice, main_stream_);
                cudaStreamSynchronize(main_stream_);

                std::vector<uint32_t*> gpu_tokens;
                std::vector<float*> gpu_probs;

                GpuTimer draft_gen_timer;
                draft_gen_timer.start();

                if (use_probability_ratio_) {

                    size_t current_spec_length = adaptive_spec_length_;

                    auto [tokens, probs] =
                        generate_draft_tokens_with_probs_gpu(d_reuse_token, current_spec_length, temperature, top_p, top_k);
                    gpu_tokens = tokens;
                    gpu_probs = probs;

                    std::cout << "[Adaptive] Current speculative length: " << current_spec_length << ", recent acceptance rate: " << recent_acceptance_rate_ << std::endl;
                } else {

                    gpu_tokens = generate_draft_tokens_gpu(d_reuse_token, adaptive_spec_length_, temperature, top_p, top_k);
                }

                draft_gen_timer.stop();
                float draft_gen_time = draft_gen_timer.milliseconds();

                std::vector<uint32_t> verified_tokens;
                verified_tokens.reserve(gpu_tokens.size());

                GpuTimer verify_timer;
                verify_timer.start();

                size_t match_length = verify_draft_tokens_gpu(current_ids, gpu_tokens, temperature, top_p, top_k,
                                                              verified_tokens, main_stream_);

                size_t num_draft_tokens = gpu_tokens.size() - 1;  // Exclude the input token.
                float acceptance_rate = (num_draft_tokens > 0) ? (float)match_length / num_draft_tokens : 0.0f;
                update_adaptive_spec_length(acceptance_rate);

                verify_timer.stop();
                float verify_time = verify_timer.milliseconds();

                // Verification yields an accepted prefix or a replacement from the target model.

                bool found_eos = false;

                size_t eos_pos = verified_tokens.size();
                for (size_t i = 0; i < verified_tokens.size(); i++) {
                    if (verified_tokens[i] == target_model_->get_eos_token_id()) {
                        eos_pos = i;
                        found_eos = true;
                        break;
                    }
                }

                // Append tokens only through the first EOS, if present.
                size_t tokens_to_add = found_eos ? eos_pos + 1 : verified_tokens.size();
                current_ids.insert(current_ids.end(), verified_tokens.begin(), verified_tokens.begin() + tokens_to_add);

                for (size_t i = 0; i < tokens_to_add; i++) {
                    if (verified_tokens[i] >= target_model_->get_vocab_size()) {
                        throw std::runtime_error("Speculative verification returned an out-of-range token: " +
                                                 std::to_string(verified_tokens[i]));
                    }
                    callback(verified_tokens[i]);
                }

                iteration_timer.stop();
                float iteration_time = iteration_timer.milliseconds();

                std::cout << "[Decode] Iteration " << iteration + 1 << " completed; generated " << tokens_to_add << " tokens"
                          << std::endl;

                std::cout << "[Verified tokens] ";
                for (size_t i = 0; i < std::min(verified_tokens.size(), size_t(10)); i++) {
                    std::cout << verified_tokens[i] << " ";
                }
                if (verified_tokens.size() > 10) {
                    std::cout << "... (total: " << verified_tokens.size() << " tokens)";
                }
                std::cout << std::endl;

                std::cout << "[KV cache update] Target model: " << target_kv_cache_.size()
                          << ", draft model: " << draft_kv_cache_.size() << std::endl;

                std::cout << "[Timing] Draft generation: " << draft_gen_time << "ms, "
                          << "Verification: " << verify_time << "ms, "
                          << "Total iteration time: " << iteration_time << "ms" << std::endl;

                if (found_eos) {
                    std::cout << "[Decode] EOS detected; stopping generation" << std::endl;
                    break;
                }

                iteration++;
            } catch (const std::exception& e) {
                std::cout << "Speculative decoding iteration failed: " << e.what() << std::endl;
            }
        }

        total_spec_timer.stop();
        float total_spec_time = total_spec_timer.milliseconds();

        std::cout << "[Decode] Speculative decoding completed; generated " << (current_ids.size() - input_ids.size())
                  << " tokens in: " << total_spec_time << "ms" << std::endl;
        if (iteration > 0) {
            std::cout << "[Performance] Average tokens per iteration: " << ((current_ids.size() - input_ids.size()) / (float)iteration)
                      << "; average iteration time: " << (total_spec_time / iteration) << "ms" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cout << "Speculative decoding failed: " << e.what() << std::endl;
    }
}

// Verify a draft batch greedily by comparing token IDs.
template <typename T>
size_t SpeculativeDecoder<T>::verify_draft_tokens_greedy(const std::vector<uint32_t>& prefix_tokens,
                                                         std::vector<uint32_t*>& draft_tokens_gpu, float temperature,
                                                         float top_p, size_t top_k,
                                                         std::vector<uint32_t>& verified_tokens, cudaStream_t stream) {

    int max_match_length = 0;
    verified_tokens.clear();

    if (draft_tokens_gpu.empty() || draft_tokens_gpu.size() == 1) {
        return 0;  // There are no draft tokens to verify.
    }

    try {

        cudaStream_t verify_stream = verify_stream_;

        size_t original_cache_size = target_kv_cache_.size();

        // Drop the trailing prediction; batch inputs stop one token before their outputs.
        draft_tokens_gpu.pop_back();

        size_t num_draft_tokens = draft_tokens_gpu.size() - 1;  // Exclude the input token.

        std::cout << "[Verification] Tokens to verify: " << num_draft_tokens << std::endl;

        GpuTimer combine_timer;
        combine_timer.start();

        // Verify the entire draft in one target-model prefill.
        // Pack draft inputs into a [sequence_length] tensor.
        Tensor<uint32_t> combined_tokens = combine_draft_tokens(draft_tokens_gpu);

        combine_timer.stop();
        float combine_time = combine_timer.milliseconds();

        target_kv_cache_.resize(original_cache_size + combined_tokens.numel());

        static const std::string kTargetTokensTag = "spec_target_tokens";
        uint32_t* target_tokens = nullptr;

        if (GlobalCudaMemoryPool::has_tag(kTargetTokensTag)) {
            target_tokens = static_cast<uint32_t*>(GlobalCudaMemoryPool::get_tagged_memory(kTargetTokensTag));
        }

        // Allocate tagged storage if no active allocation exists.
        if (target_tokens == nullptr) {
            target_tokens = static_cast<uint32_t*>(
                GlobalCudaMemoryPool::allocate_tagged(kTargetTokensTag, sizeof(uint32_t) * combined_tokens.numel()));
        }

        GpuTimer prefill_timer;
        prefill_timer.start();

        Tensor<T> logits_tensor = target_spec_model_->speculative_prefill_logits(&combined_tokens, &target_kv_cache_);

        size_t base_position = original_cache_size;  // KV position at the start of verification.

        prefill_timer.stop();
        float prefill_time = prefill_timer.milliseconds();

        GpuTimer sample_timer;
        sample_timer.start();

        operators_->sample_batch_to_fixed(std::move(logits_tensor), target_tokens, temperature, top_p, top_k,
                                          d_states, verify_stream);

        sample_timer.stop();
        float sample_time = sample_timer.milliseconds();

        GpuTimer compare_timer;
        compare_timer.start();

        int found_mismatch = -1;
        uint32_t last_target_token_value = 0;

        std::vector<uint32_t> host_target_tokens(num_draft_tokens);
        std::vector<uint32_t> host_draft_tokens(num_draft_tokens);

        cudaMemcpyAsync(host_target_tokens.data(), target_tokens, sizeof(uint32_t) * num_draft_tokens,
                        cudaMemcpyDeviceToHost, verify_stream);

        cudaMemcpyAsync(host_draft_tokens.data(), d_draft_tokens + 1, sizeof(uint32_t) * num_draft_tokens,
                        cudaMemcpyDeviceToHost, verify_stream);

        // Wait for host transfers before inspecting their results.
        cudaStreamSynchronize(verify_stream);

        for (size_t i = 0; i < num_draft_tokens; i++) {
            if (host_target_tokens[i] != host_draft_tokens[i]) {
                found_mismatch = i;
                last_target_token_value = host_target_tokens[i];

                break;
            }

            verified_tokens.push_back(host_target_tokens[i]);
        }

        compare_timer.stop();
        float compare_time = compare_timer.milliseconds();

        if (found_mismatch == -1 && num_draft_tokens > 0) {
            last_target_token_value = host_target_tokens[num_draft_tokens - 1];
            max_match_length = num_draft_tokens;
        } else {
            max_match_length = found_mismatch;
        }

        // On a mismatch, roll back both KV caches and append the target replacement.
        if (found_mismatch != -1 && last_target_token_value >= 0) {
            draft_kv_cache_.resize(draft_kv_cache_.size() - num_draft_tokens + found_mismatch);
            target_kv_cache_.resize(original_cache_size + found_mismatch + 1);
            verified_tokens.push_back(last_target_token_value);
        }

        std::cout << "[Verification] Matching tokens: " << max_match_length << "/" << num_draft_tokens << " ("
                  << (max_match_length * 100.0 / num_draft_tokens) << "%)" << std::endl;

        std::cout << "[Token comparison]" << std::endl;
        std::cout << "  Draft model: ";
        for (size_t i = 0; i < num_draft_tokens; i++) {
            std::cout << host_draft_tokens[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "  Target model: ";
        for (size_t i = 0; i < num_draft_tokens; i++) {
            std::cout << host_target_tokens[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "  Matches: ";
        for (size_t i = 0; i < num_draft_tokens; i++) {
            if (i < max_match_length) {
                std::cout << "✓ ";
            } else {
                std::cout << "✗ ";
            }
        }
        std::cout << std::endl;

        // The pool retains ownership of tagged allocations.

        return max_match_length;
    } catch (const std::exception& e) {
        std::cout << "[Verification] Failed: " << e.what() << std::endl;
        return 0;
    }
}

// Verify a draft batch using target-to-draft probability ratios.
template <typename T>
size_t SpeculativeDecoder<T>::verify_draft_tokens_prob_ratio(const std::vector<uint32_t>& prefix_tokens,
                                                             std::vector<uint32_t*>& draft_tokens_gpu,
                                                             float temperature, float top_p, size_t top_k,
                                                             std::vector<uint32_t>& verified_tokens,
                                                             cudaStream_t stream) {

    int accepted_length = 0;
    verified_tokens.clear();

    if (draft_tokens_gpu.empty() || draft_tokens_gpu.size() == 1) {
        return 0;  // There are no draft tokens to verify.
    }

    try {

        cudaStream_t verify_stream = verify_stream_;

        size_t original_cache_size = target_kv_cache_.size();

        // Drop the trailing prediction; batch inputs stop one token before their outputs.
        draft_tokens_gpu.pop_back();

        size_t num_draft_tokens = draft_tokens_gpu.size() - 1;  // Exclude the input token.

        std::cout << "[Verification] Tokens to verify: " << num_draft_tokens << std::endl;

        GpuTimer combine_timer;
        combine_timer.start();

        // Verify the entire draft in one target-model prefill.
        // Pack draft inputs into a [sequence_length] tensor.
        Tensor<uint32_t> combined_tokens = combine_draft_tokens(draft_tokens_gpu);

        combine_timer.stop();
        float combine_time = combine_timer.milliseconds();

        target_kv_cache_.resize(original_cache_size + combined_tokens.numel());

        GpuTimer prefill_timer;
        prefill_timer.start();

        Tensor<T> target_logits = target_spec_model_->speculative_prefill_logits(&combined_tokens, &target_kv_cache_);

        size_t base_position = original_cache_size;  // KV position at the start of verification.

        prefill_timer.stop();
        float prefill_time = prefill_timer.milliseconds();

        GpuTimer random_timer;
        random_timer.start();

        // Generate random values for acceptance decisions.
        operators_->generate_random_values(d_random_values, num_draft_tokens, d_states, verify_stream);

        random_timer.stop();
        float random_time = random_timer.milliseconds();

        GpuTimer compare_timer;
        compare_timer.start();

        // Transfer tokens and random values in batches.
        std::vector<uint32_t> host_draft_tokens(num_draft_tokens);
        std::vector<float> host_random_values(num_draft_tokens);
        std::vector<float> host_draft_probs(num_draft_tokens);

        cudaMemcpyAsync(host_draft_tokens.data(), d_draft_tokens + 1, sizeof(uint32_t) * num_draft_tokens,
                        cudaMemcpyDeviceToHost, verify_stream);
        cudaMemcpyAsync(host_random_values.data(), d_random_values, sizeof(float) * num_draft_tokens,
                        cudaMemcpyDeviceToHost, verify_stream);
        cudaMemcpyAsync(host_draft_probs.data(), d_draft_probs + 1, sizeof(float) * num_draft_tokens,
                        cudaMemcpyDeviceToHost, verify_stream);

        // Wait for host transfers before inspecting their results.
        cudaStreamSynchronize(verify_stream);

        bool rejected = false;
        uint32_t rejected_token = 0;

        for (size_t i = 0; i < num_draft_tokens && !rejected; i++) {
            uint32_t draft_token = host_draft_tokens[i];

            float target_prob = operators_->get_token_probability(target_logits, i, draft_token, verify_stream);

            float draft_prob = host_draft_probs[i];

            float prob_ratio = target_prob / (draft_prob + 1e-10);
            float random_value = host_random_values[i];

            std::cout << "  Token " << i << ": " << draft_token << " (target: " << target_prob
                      << ", draft: " << draft_prob << ", ratio: " << prob_ratio << ", random: " << random_value << ") ";

            if (random_value < std::min(1.0f, prob_ratio)) {
                verified_tokens.push_back(draft_token);
                accepted_length++;
                std::cout << "✓ accepted" << std::endl;
            } else {
                rejected = true;

                // Sample a replacement from the target model.
                Tensor<T> current_logits = target_logits.slice({i, 0}, {i + 1, target_logits.sizes()[1]});
                uint32_t* target_token_ptr;
                cudaMalloc(&target_token_ptr, sizeof(uint32_t));

                operators_->sample_to_fixed(std::move(current_logits), target_token_ptr, temperature, top_p, top_k,
                                            d_states, verify_stream);

                cudaMemcpyAsync(&rejected_token, target_token_ptr, sizeof(uint32_t), cudaMemcpyDeviceToHost, verify_stream);
                cudaStreamSynchronize(verify_stream);

                verified_tokens.push_back(rejected_token);
                cudaFree(target_token_ptr);

                std::cout << "✗ rejected; replacement: " << rejected_token << std::endl;
                break;
            }
        }

        compare_timer.stop();
        float compare_time = compare_timer.milliseconds();

        if (rejected) {
            draft_kv_cache_.resize(draft_kv_cache_.size() - num_draft_tokens + accepted_length);
            target_kv_cache_.resize(original_cache_size + accepted_length + 1);
        }

        std::cout << "[Verification] Accepted tokens: " << accepted_length << "/" << num_draft_tokens << " ("
                  << (accepted_length * 100.0 / num_draft_tokens) << "%)" << std::endl;

        return accepted_length;
    } catch (const std::exception& e) {
        std::cout << "[Verification] Failed: " << e.what() << std::endl;
        return 0;
    }
}

// Pack draft-token device pointers into a tensor.
template <typename T>
Tensor<uint32_t> SpeculativeDecoder<T>::combine_draft_tokens(std::vector<uint32_t*>& draft_tokens_gpu) {

    Tensor<uint32_t> combined_tokens = Tensor<uint32_t>::combine_gpu_ptrs(draft_tokens_gpu, device_);
    return combined_tokens;
}

template class SpeculativeDecoder<__nv_bfloat16>;
