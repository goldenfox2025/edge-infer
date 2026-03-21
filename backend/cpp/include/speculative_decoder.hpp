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

// 投机解码器类，用于实现投机解码功能
template <typename T>
class SpeculativeDecoder : public infer_base {
   public:
    // 构造函数，接收目标模型和草稿模型
    SpeculativeDecoder(std::shared_ptr<BaseModel> target_model, std::shared_ptr<BaseModel> draft_model,
                       size_t spec_length = 6,   // 投机长度默认为12
                       size_t thread_count = 8);  // 线程池大小参数，默认8线程

    // 析构函数，释放CUDA资源
    ~SpeculativeDecoder();

    // 生成文本，通过回调函数返回每个token
    void generate_with_callback(const std::vector<uint32_t>& input_ids, size_t max_length, float temperature,
                                float top_p, size_t top_k, std::function<void(uint32_t)> callback) override;
    Device device() const override {
        return device_;
    }

    // 设置是否使用基于概率比值的投机采样
    void set_use_probability_ratio(bool use_ratio) {
        use_probability_ratio_ = use_ratio;
    }

    // 获取当前是否使用基于概率比值的投机采样
    bool get_use_probability_ratio() const {
        return use_probability_ratio_;
    }
    
    // 获取当前自适应投机长度
    size_t get_adaptive_spec_length() const {
        return adaptive_spec_length_;
    }
    
    // 更新自适应投机长度
    void update_adaptive_spec_length(float acceptance_rate) {
        // 使用指数移动平均更新最近接受率
        recent_acceptance_rate_ = 0.7f * recent_acceptance_rate_ + 0.3f * acceptance_rate;
        
        if (recent_acceptance_rate_ > ACCEPTANCE_THRESHOLD_HIGH) {
            // 高接受率，增加投机长度
            adaptive_spec_length_ = std::min(adaptive_spec_length_ + 1, MAX_SPEC_LENGTH);
        } else if (recent_acceptance_rate_ < ACCEPTANCE_THRESHOLD_LOW) {
            // 低接受率，减少投机长度
            adaptive_spec_length_ = std::max(adaptive_spec_length_ - 1, MIN_SPEC_LENGTH);
        }
    }

   private:
    // 目标模型（大模型）
    std::shared_ptr<BaseModel> target_model_;
    // 草稿模型（小模型）
    std::shared_ptr<BaseModel> draft_model_;
    std::shared_ptr<SpeculativeModel<T>> target_spec_model_;
    std::shared_ptr<SpeculativeModel<T>> draft_spec_model_;
    // 目标模型KV缓存
    KVCache<T> target_kv_cache_;
    // 草稿模型KV缓存
    KVCache<T> draft_kv_cache_;
    // 线程池
    ThreadPool thread_pool_;
    // CUDA随机状态
    curandState* d_states;
    std::unique_ptr<op::UnifiedOperators<T>> operators_;
    // 用于重用的token内存，避免频繁的分配和释放
    uint32_t* d_reuse_token;
    // 用于存储草稿模型生成的tokens的固定GPU内存
    uint32_t* d_draft_tokens;
    // 用于存储草稿模型生成的token概率的固定GPU内存
    float* d_draft_probs;
    // 用于存储随机数的GPU内存
    float* d_random_values;
    
    // 设备类型
    Device device_;
    // 投机长度（一次生成多少个token）
    size_t spec_length_;
    // 是否使用基于概率比值的投机采样
    bool use_probability_ratio_ = true;  
    
    // 自适应投机长度相关参数
    float recent_acceptance_rate_ = 0.5f;  // 最近的接受率
    size_t adaptive_spec_length_;          // 自适应投机长度
    static constexpr size_t MIN_SPEC_LENGTH = 6;   // 最小投机长度
    static constexpr size_t MAX_SPEC_LENGTH = 8;   // 最大投机长度
    static constexpr float ACCEPTANCE_THRESHOLD_HIGH = 0.7f;  // 高接受率阈值
    static constexpr float ACCEPTANCE_THRESHOLD_LOW = 0.4f;   // 低接受率阈值

    // CUDA流，用于异步操作
    cudaStream_t main_stream_;    // 主流，用于主要操作
    cudaStream_t draft_stream_;   // 草稿模型流
    cudaStream_t verify_stream_;  // 验证流

    // 内存标签，用于固定内存分配
    static constexpr const char* kReuseTokenTag = "spec_reuse_token";
    static constexpr const char* kDraftTokensTag = "spec_draft_tokens";
    static constexpr const char* kDraftProbsTag = "spec_draft_probs";
    static constexpr const char* kRandomValuesTag = "spec_random_values";

    // 初始化CUDA资源
    void init_cuda_resources();
    // 释放CUDA资源
    void free_cuda_resources();

    // 批量验证草稿模型生成的token - 贪心方法（比较token ID是否相同）
    size_t verify_draft_tokens_greedy(const std::vector<uint32_t>& prefix_tokens,
                                      std::vector<uint32_t*>& draft_tokens_gpu, float temperature, float top_p,
                                      size_t top_k, std::vector<uint32_t>& verified_tokens, cudaStream_t stream);

    // 批量验证草稿模型生成的token - 基于概率比值的方法
    size_t verify_draft_tokens_prob_ratio(const std::vector<uint32_t>& prefix_tokens,
                                          std::vector<uint32_t*>& draft_tokens_gpu, float temperature, float top_p,
                                          size_t top_k, std::vector<uint32_t>& verified_tokens, cudaStream_t stream);

    // 批量验证草稿模型生成的token - 根据设置选择验证方法
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

    // 使用草稿模型生成多个token，直接返回GPU指针数组
    std::vector<uint32_t*> generate_draft_tokens_gpu(uint32_t* input_token, size_t num_tokens, float temperature,
                                                     float top_p, size_t top_k);

    // 使用草稿模型生成多个token及其概率，直接返回GPU指针数组
    std::pair<std::vector<uint32_t*>, std::vector<float*>> generate_draft_tokens_with_probs_gpu(
        uint32_t* input_token, size_t num_tokens, float temperature, float top_p, size_t top_k);

    // 将草稿token GPU指针组合成一个tensor
    Tensor<uint32_t> combine_draft_tokens(std::vector<uint32_t*>& draft_tokens_gpu);
};

// 显式声明模板类
extern template class SpeculativeDecoder<__nv_bfloat16>;
