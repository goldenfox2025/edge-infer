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

// 构造函数实现
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
      thread_pool_(thread_count),  // 使用传入的线程数
      device_(Device::CUDA),
      spec_length_(spec_length),
      adaptive_spec_length_(spec_length),  // 初始化自适应投机长度
      d_states(nullptr),
      operators_(nullptr),
      d_reuse_token(nullptr),
      main_stream_(nullptr),
      draft_stream_(nullptr),
      verify_stream_(nullptr) {
    // 确保两个模型都在CUDA上
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
    // 初始化CUDA资源
    init_cuda_resources();

    // std::cout << "【投机解码器初始化完成】" << std::endl;
    // std::cout << "目标模型: " << target_model_->get_n_layers() << " 层" <<
    // std::endl; std::cout << "草稿模型: " << draft_model_->get_n_layers() << "
    // 层" << std::endl; std::cout << "投机长度: " << spec_length_ << std::endl;
    // std::cout << "线程池大小: " << thread_count << std::endl;
}

// 析构函数实现
template <typename T>
SpeculativeDecoder<T>::~SpeculativeDecoder() {
    // 释放CUDA资源
    free_cuda_resources();
}

// 初始化CUDA资源
template <typename T>
void SpeculativeDecoder<T>::init_cuda_resources() {
    // 分配CUDA随机状态
    cudaMalloc(&d_states, sizeof(curandState));
    int seed = std::chrono::system_clock::now().time_since_epoch().count();
    operators_->init_curand(d_states, seed, 0, nullptr);

    // 创建CUDA流，用于异步操作
    cudaStreamCreate(&main_stream_);
    cudaStreamCreate(&draft_stream_);
    cudaStreamCreate(&verify_stream_);

    // 使用内存池分配固定内存，避免频繁的分配和释放
    if (GlobalCudaMemoryPool::has_tag(kReuseTokenTag)) {
        d_reuse_token = static_cast<uint32_t*>(GlobalCudaMemoryPool::get_tagged_memory(kReuseTokenTag));
    } else {
        d_reuse_token = static_cast<uint32_t*>(GlobalCudaMemoryPool::allocate_tagged(kReuseTokenTag, sizeof(uint32_t)));
    }

    // 如果内存池分配失败，则使用普通的CUDA内存分配
    if (d_reuse_token == nullptr) {
        (cudaMalloc(&d_reuse_token, sizeof(uint32_t)));
    }

    // 分配草稿模型tokens的固定内存
    // 为spec_length_个token分配内存
    if (GlobalCudaMemoryPool::has_tag(kDraftTokensTag)) {
        d_draft_tokens = static_cast<uint32_t*>(GlobalCudaMemoryPool::get_tagged_memory(kDraftTokensTag));
    } else {
        d_draft_tokens = static_cast<uint32_t*>(
            GlobalCudaMemoryPool::allocate_tagged(kDraftTokensTag,
                                                  sizeof(uint32_t) * (spec_length_ + 1)));  // +1 为输入token
    }

    // 如果内存池分配失败，则使用普通的CUDA内存分配
    if (d_draft_tokens == nullptr) {
        (cudaMalloc(&d_draft_tokens, sizeof(uint32_t) * (spec_length_ + 1)));
    }

    // 分配草稿模型token概率的固定内存
    if (GlobalCudaMemoryPool::has_tag(kDraftProbsTag)) {
        d_draft_probs = static_cast<float*>(GlobalCudaMemoryPool::get_tagged_memory(kDraftProbsTag));
    } else {
        d_draft_probs = static_cast<float*>(
            GlobalCudaMemoryPool::allocate_tagged(kDraftProbsTag,
                                                  sizeof(float) * (spec_length_ + 1)));  // +1 为输入token
    }

    // 如果内存池分配失败，则使用普通的CUDA内存分配
    if (d_draft_probs == nullptr) {
        (cudaMalloc(&d_draft_probs, sizeof(float) * (spec_length_ + 1)));
    }

    // 分配随机数的固定内存
    if (GlobalCudaMemoryPool::has_tag(kRandomValuesTag)) {
        d_random_values = static_cast<float*>(GlobalCudaMemoryPool::get_tagged_memory(kRandomValuesTag));
    } else {
        d_random_values =
            static_cast<float*>(GlobalCudaMemoryPool::allocate_tagged(kRandomValuesTag, sizeof(float) * spec_length_));
    }

    // 如果内存池分配失败，则使用普通的CUDA内存分配
    if (d_random_values == nullptr) {
        (cudaMalloc(&d_random_values, sizeof(float) * spec_length_));
    }
}

// 释放CUDA资源
template <typename T>
void SpeculativeDecoder<T>::free_cuda_resources() {
    // 同步所有流，确保所有操作完成
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

    // 不需要释放标记内存，由内存池管理
    if (d_reuse_token != nullptr && !GlobalCudaMemoryPool::has_tag(kReuseTokenTag)) {
        cudaFree(d_reuse_token);
        d_reuse_token = nullptr;
    }

    // 释放草稿模型tokens的固定内存
    if (d_draft_tokens != nullptr && !GlobalCudaMemoryPool::has_tag(kDraftTokensTag)) {
        cudaFree(d_draft_tokens);
        d_draft_tokens = nullptr;
    }

    // 释放草稿模型token概率的固定内存
    if (d_draft_probs != nullptr && !GlobalCudaMemoryPool::has_tag(kDraftProbsTag)) {
        cudaFree(d_draft_probs);
        d_draft_probs = nullptr;
    }

    // 释放随机数的固定内存
    if (d_random_values != nullptr && !GlobalCudaMemoryPool::has_tag(kRandomValuesTag)) {
        cudaFree(d_random_values);
        d_random_values = nullptr;
    }
}

// 使用草稿模型生成多个token，直接返回GPU指针数组
template <typename T>
std::vector<uint32_t*> SpeculativeDecoder<T>::generate_draft_tokens_gpu(uint32_t* input_token, size_t num_tokens,
                                                                        float temperature, float top_p, size_t top_k) {
    // 预分配GPU指针数组，避免频繁扩容
    std::vector<uint32_t*> gpu_tokens;

    // 计时开始 - 整个草稿生成过程
    GpuTimer draft_total_timer;
    draft_total_timer.start();

    try {
        // 将输入token复制到固定内存的第一个位置
        cudaMemcpyAsync(d_draft_tokens, input_token, sizeof(uint32_t), cudaMemcpyDeviceToDevice, draft_stream_);

        // 添加输入token指针到结果数组
        gpu_tokens.push_back(d_draft_tokens);

        std::cout << "【草稿】开始生成 " << num_tokens << " 个草稿token" << std::endl;
        float total_forward_time = 0.0f;
        float total_sample_time = 0.0f;

        // 生成剩余token，使用forward_cuda进行自回归生成
        for (size_t i = 0; i < num_tokens; i++) {
            // 构造输入张量，使用当前token位置
            uint32_t* d_current_token = d_draft_tokens + i;
            Tensor<uint32_t> input(d_current_token, {1}, device_);

            // 扩展KV缓存
            draft_kv_cache_.resize(draft_kv_cache_.size() + 1);

            // 使用sample_to_fixed直接将结果写入固定内存的下一个位置
            uint32_t* next_token_ptr = d_draft_tokens + i + 1;

            // 计时开始 - forward
            GpuTimer forward_timer;
            forward_timer.start();

            // 使用forward_cuda获取logits并直接采样到固定内存位置
            Tensor<T> logits = draft_spec_model_->speculative_forward_logits(&input, &draft_kv_cache_);

            forward_timer.stop();
            float forward_time = forward_timer.milliseconds();
            total_forward_time += forward_time;

            // 计时开始 - 采样
            GpuTimer sample_timer;
            sample_timer.start();

            operators_->sample_to_fixed(std::move(logits), next_token_ptr, temperature, top_p, 1, d_states,
                                        draft_stream_);

            sample_timer.stop();
            float sample_time = sample_timer.milliseconds();
            total_sample_time += sample_time;

            // 保存GPU指针，避免不必要的GPU-CPU数据传输
            gpu_tokens.push_back(next_token_ptr);

            // 只在最后一次迭代检查EOS token，避免频繁的同步
            if (i == num_tokens - 1) {
                // 检查是否需要提前停止生成（例如EOS token）
                uint32_t token_value;
                cudaMemcpyAsync(&token_value, next_token_ptr, sizeof(uint32_t), cudaMemcpyDeviceToHost, draft_stream_);
                cudaStreamSynchronize(draft_stream_);

                // 如果是EOS token，提前结束生成
                if (token_value == draft_model_->get_eos_token_id()) {
                    break;
                }
            }
        }

        draft_total_timer.stop();
        float draft_total_time = draft_total_timer.milliseconds();

        // 打印草稿生成的耗时统计
        std::cout << "【草稿】生成了 " << gpu_tokens.size() - 1 << " 个草稿token" << std::endl;

        // 打印生成的草稿token
        std::vector<uint32_t> host_draft_tokens(gpu_tokens.size() - 1);
        for (size_t i = 1; i < gpu_tokens.size(); i++) {
            uint32_t token_value;
            cudaMemcpyAsync(&token_value, gpu_tokens[i], sizeof(uint32_t), cudaMemcpyDeviceToHost, draft_stream_);
            host_draft_tokens[i - 1] = token_value;
        }
        cudaStreamSynchronize(draft_stream_);

        std::cout << "【草稿Token】";
        for (size_t i = 0; i < host_draft_tokens.size(); i++) {
            std::cout << host_draft_tokens[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "【耗时】草稿模型forward: " << total_forward_time
                  << "ms (平均: " << (total_forward_time / (gpu_tokens.size() - 1)) << "ms/token), "
                  << "采样: " << total_sample_time << "ms (平均: " << (total_sample_time / (gpu_tokens.size() - 1))
                  << "ms/token), "
                  << "总耗时: " << draft_total_time << "ms" << std::endl;

    } catch (const std::exception& e) {
        // 记录错误并返回已生成的token
        std::cout << "【草稿】生成token时出错: " << e.what() << std::endl;
    }

    return gpu_tokens;
}

// 使用草稿模型生成多个token及其概率，直接返回GPU指针数组
template <typename T>
std::pair<std::vector<uint32_t*>, std::vector<float*>> SpeculativeDecoder<T>::generate_draft_tokens_with_probs_gpu(
    uint32_t* input_token, size_t num_tokens, float temperature, float top_p, size_t top_k) {
    // 预分配GPU指针数组，避免频繁扩容
    std::vector<uint32_t*> gpu_tokens;
    std::vector<float*> gpu_probs;

    // 计时开始 - 整个草稿生成过程
    GpuTimer draft_total_timer;
    draft_total_timer.start();

    try {
        // 将输入token复制到固定内存的第一个位置
        cudaMemcpyAsync(d_draft_tokens, input_token, sizeof(uint32_t), cudaMemcpyDeviceToDevice, draft_stream_);

        // 添加输入token指针到结果数组
        gpu_tokens.push_back(d_draft_tokens);
        gpu_probs.push_back(d_draft_probs);  // 输入token没有概率值，但为了保持索引一致，仍添加

        // 开始生成草稿token
        std::cout << "【草稿】开始生成 " << num_tokens << " 个草稿token（带概率）" << std::endl;
        float total_forward_time = 0.0f;
        float total_sample_time = 0.0f;

        // 生成剩余token，使用forward_cuda进行自回归生成
        for (size_t i = 0; i < num_tokens; i++) {
            // 构造输入张量，使用当前token位置
            uint32_t* d_current_token = d_draft_tokens + i;
            Tensor<uint32_t> input(d_current_token, {1}, device_);

            // 扩展KV缓存
            draft_kv_cache_.resize(draft_kv_cache_.size() + 1);

            // 使用sample_to_fixed_with_prob直接将结果写入固定内存的下一个位置
            uint32_t* next_token_ptr = d_draft_tokens + i + 1;
            float* next_prob_ptr = d_draft_probs + i + 1;

            // 计时开始 - forward
            GpuTimer forward_timer;
            forward_timer.start();

            // 使用forward_cuda获取logits
            Tensor<T> logits = draft_spec_model_->speculative_forward_logits(&input, &draft_kv_cache_);

            forward_timer.stop();
            float forward_time = forward_timer.milliseconds();
            total_forward_time += forward_time;

            // 计时开始 - 采样
            GpuTimer sample_timer;
            sample_timer.start();

            // 使用sample_to_fixed_with_prob将token和概率写入固定内存位置
            operators_->sample_to_fixed_with_prob(std::move(logits), next_token_ptr, next_prob_ptr, temperature,
                                                  top_p, top_k, d_states, draft_stream_);

            sample_timer.stop();
            float sample_time = sample_timer.milliseconds();
            total_sample_time += sample_time;

            // 保存GPU指针，避免不必要的GPU-CPU数据传输
            gpu_tokens.push_back(next_token_ptr);
            gpu_probs.push_back(next_prob_ptr);

            // 检查EOS token
            if (i >= num_tokens - 2) {
                uint32_t token_value;
                cudaMemcpyAsync(&token_value, next_token_ptr, sizeof(uint32_t), cudaMemcpyDeviceToHost, draft_stream_);
                cudaStreamSynchronize(draft_stream_);

                if (token_value == draft_model_->get_eos_token_id()) {
                    break;
                }
            }
        }

        // 确保所有操作完成
        cudaStreamSynchronize(draft_stream_);

        draft_total_timer.stop();
        float draft_total_time = draft_total_timer.milliseconds();

        // 打印草稿生成的耗时统计
        std::cout << "【草稿】生成了 " << gpu_tokens.size() - 1 << " 个草稿token（带概率）" << std::endl;

        // 打印生成的草稿token和概率
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

        std::cout << "【草稿Token及概率】" << std::endl;
        for (size_t i = 0; i < host_draft_tokens.size(); i++) {
            std::cout << host_draft_tokens[i] << "(" << host_draft_probs[i] << ") ";
        }
        std::cout << std::endl;

        std::cout << "【耗时】草稿模型forward: " << total_forward_time
                  << "ms (平均: " << (total_forward_time / (gpu_tokens.size() - 1)) << "ms/token), "
                  << "采样: " << total_sample_time << "ms (平均: " << (total_sample_time / (gpu_tokens.size() - 1))
                  << "ms/token), "
                  << "总耗时: " << draft_total_time << "ms" << std::endl;

    } catch (const std::exception& e) {
        // 记录错误并返回已生成的token
        std::cout << "【草稿】生成token时出错: " << e.what() << std::endl;
    }

    return {gpu_tokens, gpu_probs};
}

// 批处理批量从GPU读取token值
template <typename T>
void SpeculativeDecoder<T>::generate_with_callback(const std::vector<uint32_t>& input_ids, size_t max_length,
                                                   float temperature, float top_p, size_t top_k,
                                                   std::function<void(uint32_t)> callback) {
    try {
        std::vector<uint32_t> current_ids;
        current_ids.reserve(max_length);  // 提前分配最大可能的大小
        current_ids = input_ids;

        // 确保输入序列非空
        if (current_ids.empty()) {
            std::cout << "错误: 输入序列为空" << std::endl;
            return;
        }

        std::cout << "【初始化】输入序列长度: " << current_ids.size() << std::endl;
        std::cout << "【初始化】使用" << (use_probability_ratio_ ? "基于概率比值的" : "贪心") << "投机采样"
                  << std::endl;

        uint32_t first_target_token_value = -1;
        // 1. 目标模型初始化 - 一次性预分配足够的GPU内存
        {
            // 计时开始 - 目标模型初始化
            GpuTimer target_init_timer;
            target_init_timer.start();

            // 为整个提示词序列预分配足够的内存，避免后续频繁调整大小
            // 进入prefill或forward时，需要手动扩展出当前输入的长度
            // 已有长度基础上，扩展输入长度
            target_kv_cache_.resize(target_kv_cache_.size() + current_ids.size());

            // 首次使用prefill处理整个提示词序列 - 直接创建张量而不是移动
            Tensor<uint32_t> input_tensor(std::vector<uint32_t>(current_ids.begin(), current_ids.end()),
                                         {current_ids.size()}, device_);
            uint32_t* first_token = target_model_->prefill(&input_tensor, thread_pool_, &target_kv_cache_, top_k,
                                                           temperature, top_p, d_states);

            if (first_token == nullptr) {
                std::cout << "警告: 目标模型初始预填充返回空指针" << std::endl;
                return;
            }

            // 从GPU读取第一个token值 - 使用异步拷贝
            cudaMemcpyAsync(&first_target_token_value, first_token, sizeof(uint32_t), cudaMemcpyDeviceToHost,
                            main_stream_);
            cudaStreamSynchronize(main_stream_);  // 确保拷贝完成

            if (first_target_token_value >= target_model_->get_vocab_size()) {
                throw std::runtime_error("目标模型prefill返回了越界token: " +
                                         std::to_string(first_target_token_value));
            }

            target_init_timer.stop();
            float target_init_time = target_init_timer.milliseconds();

            std::cout << "【初始化】目标模型初始化完成，耗时: " << target_init_time << "ms" << std::endl;

            // 回调函数处理第一个token
            callback(first_target_token_value);

            // 检查是否为EOS
            if (first_target_token_value == target_model_->get_eos_token_id()) {
                std::cout << "【初始化】目标模型生成EOS，提前结束生成" << std::endl;
                return;
            }
        }

        // 2. 草稿模型初始化 - 同样预分配内存并使用批量处理
        {
            // 计时开始 - 草稿模型初始化
            GpuTimer draft_init_timer;
            draft_init_timer.start();
            draft_kv_cache_.resize(draft_kv_cache_.size() + current_ids.size());

            // 使用prefill为草稿模型准备KV缓存 - 直接传递数据不创建副本
            Tensor<uint32_t> draft_input_tensor(std::vector<uint32_t>(current_ids.begin(), current_ids.end()),
                                                {current_ids.size()}, device_);
            uint32_t* draft_first_token = draft_model_->prefill(&draft_input_tensor, thread_pool_, &draft_kv_cache_,
                                                                top_k, temperature, top_p, d_states);

            if (draft_first_token == nullptr) {
                throw std::runtime_error("草稿模型初始预填充返回空指针");
            }

            draft_init_timer.stop();
            float draft_init_time = draft_init_timer.milliseconds();

            std::cout << "【初始化】草稿模型初始化完成，耗时: " << draft_init_time << "ms" << std::endl;
        }

        current_ids.push_back(first_target_token_value);
        // 3. 不需要额外分配GPU内存，直接使用预分配的重用内存
        // 将最大迭代次数限制为剩余的生成长度
        size_t max_iterations = max_length - current_ids.size();
        size_t iteration = 0;

        // 主循环：投机解码
        std::cout << "【主循环】开始投机解码，最大迭代次数: " << max_iterations << std::endl;

        // 计时开始 - 整个投机解码过程
        GpuTimer total_spec_timer;
        total_spec_timer.start();

        while (current_ids.size() < max_length && iteration < max_iterations) {
            try {
                // 计时开始 - 当前迭代
                GpuTimer iteration_timer;
                iteration_timer.start();

                std::cout << "【主循环】迭代 " << iteration + 1 << "/" << max_iterations << std::endl;

                // 打印当前KV缓存大小
                std::cout << "【KV缓存】目标模型: " << target_kv_cache_.size()
                          << ", 草稿模型: " << draft_kv_cache_.size() << std::endl;

                // 打印当前token序列
                std::cout << "【Token序列】";
                for (size_t i = 0; i < std::min(current_ids.size(), size_t(10)); i++) {
                    std::cout << current_ids[i] << " ";
                }
                if (current_ids.size() > 10) {
                    std::cout << "... (共" << current_ids.size() << "个token)";
                }
                std::cout << std::endl;

                // 获得目标模型最后推理出的token
                const uint32_t last_token = current_ids.back();
                std::cout << "【当前Token】" << last_token << std::endl;
                cudaMemcpyAsync(d_reuse_token, &last_token, sizeof(uint32_t), cudaMemcpyHostToDevice, main_stream_);
                cudaStreamSynchronize(main_stream_);  // 确保拷贝完成

                std::vector<uint32_t*> gpu_tokens;
                std::vector<float*> gpu_probs;

                // 计时开始 - 草稿生成
                GpuTimer draft_gen_timer;
                draft_gen_timer.start();

                if (use_probability_ratio_) {
                    // 使用自适应投机长度
                    size_t current_spec_length = adaptive_spec_length_;
                    
                    // 使用带概率的草稿生成
                    auto [tokens, probs] =
                        generate_draft_tokens_with_probs_gpu(d_reuse_token, current_spec_length, temperature, top_p, top_k);
                    gpu_tokens = tokens;
                    gpu_probs = probs;
                    
                    std::cout << "【自适应】当前投机长度: " << current_spec_length << ", 最近接受率: " << recent_acceptance_rate_ << std::endl;
                } else {
                    // 使用普通的草稿生成
                    gpu_tokens = generate_draft_tokens_gpu(d_reuse_token, adaptive_spec_length_, temperature, top_p, top_k);
                }

                draft_gen_timer.stop();
                float draft_gen_time = draft_gen_timer.milliseconds();

                // 验证草稿模型生成的token
                std::vector<uint32_t> verified_tokens;
                verified_tokens.reserve(gpu_tokens.size());

                // 计时开始 - 验证过程
                GpuTimer verify_timer;
                verify_timer.start();

                // 使用验证函数
                size_t match_length = verify_draft_tokens_gpu(current_ids, gpu_tokens, temperature, top_p, top_k,
                                                              verified_tokens, main_stream_);

                // 计算接受率并更新自适应投机长度
                size_t num_draft_tokens = gpu_tokens.size() - 1;  // 减去输入token
                float acceptance_rate = (num_draft_tokens > 0) ? (float)match_length / num_draft_tokens : 0.0f;
                update_adaptive_spec_length(acceptance_rate);

                verify_timer.stop();
                float verify_time = verify_timer.milliseconds();

                // verified_tokens一定会包含至少一个token(要么是匹配的token，要么是目标模型生成的替代token)

                // 批量添加验证通过的token - 避免逐个添加的开销
                bool found_eos = false;

                // 检查是否有EOS token并找到位置
                size_t eos_pos = verified_tokens.size();
                for (size_t i = 0; i < verified_tokens.size(); i++) {
                    if (verified_tokens[i] == target_model_->get_eos_token_id()) {
                        eos_pos = i;
                        found_eos = true;
                        break;
                    }
                }

                // 只添加到EOS标记处（如果有）
                size_t tokens_to_add = found_eos ? eos_pos + 1 : verified_tokens.size();
                current_ids.insert(current_ids.end(), verified_tokens.begin(), verified_tokens.begin() + tokens_to_add);

                // 批量调用回调函数
                for (size_t i = 0; i < tokens_to_add; i++) {
                    if (verified_tokens[i] >= target_model_->get_vocab_size()) {
                        throw std::runtime_error("投机解码验证阶段返回了越界token: " +
                                                 std::to_string(verified_tokens[i]));
                    }
                    callback(verified_tokens[i]);
                }

                iteration_timer.stop();
                float iteration_time = iteration_timer.milliseconds();

                // 打印当前迭代的耗时统计
                std::cout << "【主循环】迭代 " << iteration + 1 << " 完成，生成了 " << tokens_to_add << " 个token"
                          << std::endl;

                // 打印验证后的token
                std::cout << "【验证后Token】";
                for (size_t i = 0; i < std::min(verified_tokens.size(), size_t(10)); i++) {
                    std::cout << verified_tokens[i] << " ";
                }
                if (verified_tokens.size() > 10) {
                    std::cout << "... (共" << verified_tokens.size() << "个token)";
                }
                std::cout << std::endl;

                // 打印KV缓存变化
                std::cout << "【KV缓存更新】目标模型: " << target_kv_cache_.size()
                          << ", 草稿模型: " << draft_kv_cache_.size() << std::endl;

                std::cout << "【耗时】草稿生成: " << draft_gen_time << "ms, "
                          << "验证过程: " << verify_time << "ms, "
                          << "迭代总耗时: " << iteration_time << "ms" << std::endl;

                // 如果找到了EOS，提前结束生成
                if (found_eos) {
                    std::cout << "【主循环】检测到EOS，提前结束生成" << std::endl;
                    break;
                }

                iteration++;
            } catch (const std::exception& e) {
                std::cout << "投机解码迭代中出错: " << e.what() << std::endl;
            }
        }

        total_spec_timer.stop();
        float total_spec_time = total_spec_timer.milliseconds();

        // 打印整个投机解码过程的耗时统计
        std::cout << "【主循环】投机解码完成，共生成 " << (current_ids.size() - input_ids.size())
                  << " 个token，耗时: " << total_spec_time << "ms" << std::endl;
        if (iteration > 0) {
            std::cout << "【性能】平均每个迭代生成 " << ((current_ids.size() - input_ids.size()) / (float)iteration)
                      << " 个token，平均每个迭代耗时: " << (total_spec_time / iteration) << "ms" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cout << "投机解码过程中出错: " << e.what() << std::endl;
    }
}

// 批量验证草稿模型生成的token - 贪心方法（比较token ID是否相同）
template <typename T>
size_t SpeculativeDecoder<T>::verify_draft_tokens_greedy(const std::vector<uint32_t>& prefix_tokens,
                                                         std::vector<uint32_t*>& draft_tokens_gpu, float temperature,
                                                         float top_p, size_t top_k,
                                                         std::vector<uint32_t>& verified_tokens, cudaStream_t stream) {
    // 记录最长匹配长度
    int max_match_length = 0;
    verified_tokens.clear();

    // 检查输入有效性
    if (draft_tokens_gpu.empty() || draft_tokens_gpu.size() == 1) {
        return 0;  // 如果没有token或只有一个token（输入token），直接返回
    }

    try {
        // 使用验证专用流
        cudaStream_t verify_stream = verify_stream_;

        // 保存目标模型当前的KV缓存大小
        size_t original_cache_size = target_kv_cache_.size();

        // 移除最后一个token（这是下一个要生成的token，不是草稿token）
        draft_tokens_gpu.pop_back();

        // 获取草稿token数量
        size_t num_draft_tokens = draft_tokens_gpu.size() - 1;  // 减去输入token

        // 打印需要验证的总长度
        std::cout << "【验证】需要验证的总长度: " << num_draft_tokens << std::endl;

        // 计时开始 - 组合tokens
        GpuTimer combine_timer;
        combine_timer.start();

        // 一次性验证所有草稿token
        // 将草稿token组合成一个tensor，形状为[seqlen]
        Tensor<uint32_t> combined_tokens = combine_draft_tokens(draft_tokens_gpu);

        combine_timer.stop();
        float combine_time = combine_timer.milliseconds();

        // 扩展KV缓存以容纳新的token
        target_kv_cache_.resize(original_cache_size + combined_tokens.numel());

        // 使用内存池分配固定内存用于存储目标模型的采样结果
        static const std::string kTargetTokensTag = "spec_target_tokens";
        uint32_t* target_tokens = nullptr;

        // 检查是否已经有标记内存
        if (GlobalCudaMemoryPool::has_tag(kTargetTokensTag)) {
            target_tokens = static_cast<uint32_t*>(GlobalCudaMemoryPool::get_tagged_memory(kTargetTokensTag));
        }

        // 如果没有标记内存或者标记内存不活跃，重新分配
        if (target_tokens == nullptr) {
            target_tokens = static_cast<uint32_t*>(
                GlobalCudaMemoryPool::allocate_tagged(kTargetTokensTag, sizeof(uint32_t) * combined_tokens.numel()));
        }

        // 计时开始 - 目标模型prefill
        GpuTimer prefill_timer;
        prefill_timer.start();

        // 使用prefill_cuda获取logits并直接采样到固定内存
        Tensor<T> logits_tensor = target_spec_model_->speculative_prefill_logits(&combined_tokens, &target_kv_cache_);

        // 保存目标模型的logits到文件
        size_t base_position = original_cache_size;  // 验证开始时的KV缓存位置
        // for (size_t i = 0; i < combined_tokens.numel(); i++) {
        //     Tensor<T> single_logits = logits_tensor.slice({i, 0}, {i + 1, logits_tensor.sizes()[1]});
        //     // 计算token在整个序列中的绝对位置
        //     size_t absolute_position = base_position + i;
        //     std::string logits_file = "./logits_data/target/logits_" + std::to_string(absolute_position) + ".bin";
        //     saveTensorToFile(single_logits, logits_file);
        //     std::cout << "已保存目标模型logits到: " << logits_file << std::endl;
        // }

        prefill_timer.stop();
        float prefill_time = prefill_timer.milliseconds();

        // 计时开始 - 采样
        GpuTimer sample_timer;
        sample_timer.start();

        // 使用sample_batch_to_fixed直接将结果写入固定内存
        operators_->sample_batch_to_fixed(std::move(logits_tensor), target_tokens, temperature, top_p, top_k,
                                          d_states, verify_stream);

        sample_timer.stop();
        float sample_time = sample_timer.milliseconds();

        // 计时开始 - 比较token
        GpuTimer compare_timer;
        compare_timer.start();

        // 比较目标模型和草稿模型的token
        int found_mismatch = -1;
        uint32_t last_target_token_value = 0;

        // 分配主机内存用于批量复制token
        std::vector<uint32_t> host_target_tokens(num_draft_tokens);
        std::vector<uint32_t> host_draft_tokens(num_draft_tokens);

        // 批量复制目标模型生成的token到主机内存
        cudaMemcpyAsync(host_target_tokens.data(), target_tokens, sizeof(uint32_t) * num_draft_tokens,
                        cudaMemcpyDeviceToHost, verify_stream);

        cudaMemcpyAsync(host_draft_tokens.data(), d_draft_tokens + 1, sizeof(uint32_t) * num_draft_tokens,
                        cudaMemcpyDeviceToHost, verify_stream);

        // 等待所有复制完成
        cudaStreamSynchronize(verify_stream);

        for (size_t i = 0; i < num_draft_tokens; i++) {
            if (host_target_tokens[i] != host_draft_tokens[i]) {
                found_mismatch = i;
                last_target_token_value = host_target_tokens[i];
    
                break;
            }

            // 添加到验证通过的token列表
            verified_tokens.push_back(host_target_tokens[i]);
        }

        compare_timer.stop();
        float compare_time = compare_timer.milliseconds();

        // 如果没有找到不匹配的token，设置最后一个token值
        if (found_mismatch == -1 && num_draft_tokens > 0) {
            last_target_token_value = host_target_tokens[num_draft_tokens - 1];
            max_match_length = num_draft_tokens;
        } else {
            max_match_length = found_mismatch;
        }

        // 如果找到不匹配的token，调整KV缓存大小并添加目标模型生成的token
        if (found_mismatch != -1 && last_target_token_value >= 0) {
            draft_kv_cache_.resize(draft_kv_cache_.size() - num_draft_tokens + found_mismatch);
            target_kv_cache_.resize(original_cache_size + found_mismatch + 1);
            verified_tokens.push_back(last_target_token_value);
        }

        // 打印验证成功的长度和各部分耗时
        std::cout << "【验证】验证成功的长度: " << max_match_length << "/" << num_draft_tokens << " ("
                  << (max_match_length * 100.0 / num_draft_tokens) << "%)" << std::endl;

        // 打印草稿token和目标token的对比
        std::cout << "【Token对比】" << std::endl;
        std::cout << "  草稿模型: ";
        for (size_t i = 0; i < num_draft_tokens; i++) {
            std::cout << host_draft_tokens[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "  目标模型: ";
        for (size_t i = 0; i < num_draft_tokens; i++) {
            std::cout << host_target_tokens[i] << " ";
        }
        std::cout << std::endl;

        // 打印匹配情况
        std::cout << "  匹配情况: ";
        for (size_t i = 0; i < num_draft_tokens; i++) {
            if (i < max_match_length) {
                std::cout << "✓ ";  // 匹配
            } else {
                std::cout << "✗ ";  // 不匹配
            }
        }
        std::cout << std::endl;

        // // 打印KV缓存调整情况
        // if (found_mismatch != -1) {
        //     std::cout << "【KV缓存调整】目标模型: " << target_kv_cache_.size()
        //               << ", 草稿模型: " << draft_kv_cache_.size() << std::endl;
        // } else {
        //     std::cout << "【KV缓存调整】全部匹配，无需调整" << std::endl;
        // }

        // std::cout << "【耗时】组合tokens: " << combine_time << "ms, "
        //           << "目标模型prefill: " << prefill_time << "ms, "
        //           << "采样: " << sample_time << "ms, "
        //           << "比较token: " << compare_time << "ms, "
        //           << "总耗时: " << (combine_time + prefill_time + sample_time + compare_time) << "ms" << std::endl;

        // 不需要释放标记内存，由内存池管理

        return max_match_length;
    } catch (const std::exception& e) {
        std::cout << "【验证】验证过程中出错: " << e.what() << std::endl;
        return 0;
    }
}

// 批量验证草稿模型生成的token - 基于概率比值的方法
template <typename T>
size_t SpeculativeDecoder<T>::verify_draft_tokens_prob_ratio(const std::vector<uint32_t>& prefix_tokens,
                                                             std::vector<uint32_t*>& draft_tokens_gpu,
                                                             float temperature, float top_p, size_t top_k,
                                                             std::vector<uint32_t>& verified_tokens,
                                                             cudaStream_t stream) {
    // 记录最长匹配长度
    int accepted_length = 0;
    verified_tokens.clear();

    // 检查输入有效性
    if (draft_tokens_gpu.empty() || draft_tokens_gpu.size() == 1) {
        return 0;  // 如果没有token或只有一个token（输入token），直接返回
    }

    try {
        // 使用验证专用流
        cudaStream_t verify_stream = verify_stream_;

        // 保存目标模型当前的KV缓存大小
        size_t original_cache_size = target_kv_cache_.size();

        // 移除最后一个token（这是下一个要生成的token，不是草稿token）
        draft_tokens_gpu.pop_back();

        // 获取草稿token数量
        size_t num_draft_tokens = draft_tokens_gpu.size() - 1;  // 减去输入token

        // 打印需要验证的总长度
        std::cout << "【验证】需要验证的总长度: " << num_draft_tokens << std::endl;

        // 计时开始 - 组合tokens
        GpuTimer combine_timer;
        combine_timer.start();

        // 一次性验证所有草稿token
        // 将草稿token组合成一个tensor，形状为[seqlen]
        Tensor<uint32_t> combined_tokens = combine_draft_tokens(draft_tokens_gpu);

        combine_timer.stop();
        float combine_time = combine_timer.milliseconds();

        // 扩展KV缓存以容纳新的token
        target_kv_cache_.resize(original_cache_size + combined_tokens.numel());

        // 计时开始 - 目标模型prefill
        GpuTimer prefill_timer;
        prefill_timer.start();

        // 使用prefill_cuda获取logits - 移除第三个参数false
        Tensor<T> target_logits = target_spec_model_->speculative_prefill_logits(&combined_tokens, &target_kv_cache_);

        // 保存目标模型的logits到文件
        // 注意：combined_tokens中可能有多个token，每个token对应一个logits输出
        // 我们需要分别保存每个token位置的logits
        size_t base_position = original_cache_size;  // 验证开始时的KV缓存位置
        // for (size_t i = 0; i < combined_tokens.numel(); i++) {
        //     Tensor<T> single_logits = target_logits.slice({i, 0}, {i + 1, target_logits.sizes()[1]});
        //     // 计算token在整个序列中的绝对位置
        //     size_t absolute_position = base_position + i;
        //     std::string logits_file = "./logits_data/target/logits_" + std::to_string(absolute_position) + ".bin";
        //     saveTensorToFile(single_logits, logits_file);
        //     std::cout << "已保存目标模型logits到: " << logits_file << std::endl;
        // }

        prefill_timer.stop();
        float prefill_time = prefill_timer.milliseconds();

        // 计时开始 - 生成随机数
        GpuTimer random_timer;
        random_timer.start();

        // 生成随机数用于接受/拒绝决策
        operators_->generate_random_values(d_random_values, num_draft_tokens, d_states, verify_stream);

        random_timer.stop();
        float random_time = random_timer.milliseconds();

        // 计时开始 - 比较概率
        GpuTimer compare_timer;
        compare_timer.start();

        // 【优化9】批量复制token和随机数到主机内存
        std::vector<uint32_t> host_draft_tokens(num_draft_tokens);
        std::vector<float> host_random_values(num_draft_tokens);
        std::vector<float> host_draft_probs(num_draft_tokens);
        
        // 批量复制数据到主机内存
        cudaMemcpyAsync(host_draft_tokens.data(), d_draft_tokens + 1, sizeof(uint32_t) * num_draft_tokens,
                        cudaMemcpyDeviceToHost, verify_stream);
        cudaMemcpyAsync(host_random_values.data(), d_random_values, sizeof(float) * num_draft_tokens,
                        cudaMemcpyDeviceToHost, verify_stream);
        cudaMemcpyAsync(host_draft_probs.data(), d_draft_probs + 1, sizeof(float) * num_draft_tokens,
                        cudaMemcpyDeviceToHost, verify_stream);

        // 等待所有复制完成
        cudaStreamSynchronize(verify_stream);

        // 逐个验证草稿token
        bool rejected = false;
        uint32_t rejected_token = 0;

        for (size_t i = 0; i < num_draft_tokens && !rejected; i++) {
            uint32_t draft_token = host_draft_tokens[i];
            
            // 获取目标模型对该token的概率
            float target_prob = operators_->get_token_probability(target_logits, i, draft_token, verify_stream);
            
            // 获取草稿模型对该token的概率
            float draft_prob = host_draft_probs[i];
            
            // 计算概率比值
            float prob_ratio = target_prob / (draft_prob + 1e-10);
            float random_value = host_random_values[i];

            std::cout << "  Token " << i << ": " << draft_token << " (目标: " << target_prob
                      << ", 草稿: " << draft_prob << ", 比值: " << prob_ratio << ", 随机: " << random_value << ") ";

            if (random_value < std::min(1.0f, prob_ratio)) {
                verified_tokens.push_back(draft_token);
                accepted_length++;
                std::cout << "✓ 接受" << std::endl;
            } else {
                rejected = true;
                
         
                
                // 使用目标模型采样替代token
                Tensor<T> current_logits = target_logits.slice({i, 0}, {i + 1, target_logits.sizes()[1]});
                uint32_t* target_token_ptr;
                cudaMalloc(&target_token_ptr, sizeof(uint32_t));
                
                operators_->sample_to_fixed(std::move(current_logits), target_token_ptr, temperature, top_p, top_k,
                                            d_states, verify_stream);
                
                cudaMemcpyAsync(&rejected_token, target_token_ptr, sizeof(uint32_t), cudaMemcpyDeviceToHost, verify_stream);
                cudaStreamSynchronize(verify_stream);
                
                verified_tokens.push_back(rejected_token);
                cudaFree(target_token_ptr);
                
                std::cout << "✗ 拒绝，替换为: " << rejected_token << std::endl;
                break;
            }
        }

        compare_timer.stop();
        float compare_time = compare_timer.milliseconds();

        // 调整KV缓存大小
        if (rejected) {
            draft_kv_cache_.resize(draft_kv_cache_.size() - num_draft_tokens + accepted_length);
            target_kv_cache_.resize(original_cache_size + accepted_length + 1);
        }

        // 打印验证成功的长度和各部分耗时
        std::cout << "【验证】接受的长度: " << accepted_length << "/" << num_draft_tokens << " ("
                  << (accepted_length * 100.0 / num_draft_tokens) << "%)" << std::endl;

        // 打印KV缓存调整情况
        // if (rejected) {
        //     std::cout << "【KV缓存调整】目标模型: " << target_kv_cache_.size()
        //               << ", 草稿模型: " << draft_kv_cache_.size() << std::endl;
        // } else {
        //     std::cout << "【KV缓存调整】全部接受，无需调整" << std::endl;
        // }

        // std::cout << "【耗时】组合tokens: " << combine_time << "ms, "
        //           << "目标模型prefill: " << prefill_time << "ms, "
        //           << "生成随机数: " << random_time << "ms, "
        //           << "比较概率: " << compare_time << "ms, "
        //           << "总耗时: " << (combine_time + prefill_time + random_time + compare_time) << "ms" << std::endl;

        return accepted_length;
    } catch (const std::exception& e) {
        std::cout << "【验证】验证过程中出错: " << e.what() << std::endl;
        return 0;
    }
}

// 将草稿token GPU指针组合成一个tensor
template <typename T>
Tensor<uint32_t> SpeculativeDecoder<T>::combine_draft_tokens(std::vector<uint32_t*>& draft_tokens_gpu) {
    // 使用Tensor类的静态方法将GPU指针组合成一个tensor
    Tensor<uint32_t> combined_tokens = Tensor<uint32_t>::combine_gpu_ptrs(draft_tokens_gpu, device_);
    // std::cout << "【组合】已将 " << draft_tokens_gpu.size()
    // << " 个草稿token组合成tensor，形状为[" << combined_tokens.sizes()[0]
    // << "]" << std::endl;
    // debugPrintTensor(combined_tokens, "combined_tokens");
    return combined_tokens;
}

// 显式实例化模板类
template class SpeculativeDecoder<__nv_bfloat16>;
