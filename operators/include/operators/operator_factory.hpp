#pragma once

#include <memory>

#include "operators/cpu/add_cpu.hpp"
#include "operators/cpu/attention_output_prefill_cpu.hpp"
#include "operators/cpu/attention_scores_prefill_cpu.hpp"
#include "operators/cpu/gather_cpu.hpp"
#include "operators/cpu/kv_cache_write_cpu.hpp"
#include "operators/cpu/matmul_cpu.hpp"
#include "operators/cpu/multiply_cpu.hpp"
#include "operators/cpu/rms_norm_cpu.hpp"
#include "operators/cpu/rope_cpu.hpp"
#include "operators/cpu/sample_cpu.hpp"
#include "operators/cpu/silu_cpu.hpp"
#include "operators/cpu/softmax_cpu.hpp"
#include "operators/cuda/add_cuda.cuh"
#include "operators/cuda/attention_output_prefill_cuda.cuh"
#include "operators/cuda/attention_scores_prefill_cuda.cuh"
#include "operators/cuda/dynamic_flash_attention_cuda.cuh"
#include "operators/cuda/flash_attention_prefill_cuda.cuh"
#include "operators/cuda/gather_cuda.cuh"
#include "operators/cuda/kv_cache_write_cuda.cuh"
#include "operators/cuda/matmul/matmul_cuda.cuh"
#include "operators/cuda/matmul/matmul_selector.hpp"
#include "operators/cuda/multiply_cuda.cuh"
#include "operators/cuda/rms_norm_cuda.cuh"
#include "operators/cuda/rope_cuda.cuh"
#include "operators/cuda/sample_cuda.cuh"
#include "operators/cuda/silu_cuda.cuh"
#include "operators/cuda/softmax_cuda.cuh"
#include "operators/operator_base.hpp"
#include "weight_tensor.hpp"

namespace op {

// 算子工厂类
template <typename T>
class OperatorFactory {
 public:
  // 创建并注册所有CPU算子
  static void registerCPUOperators() {
    auto& registry = OperatorRegistry<T>::instance();

    auto kv_cache_write = std::make_shared<KvCacheWriteCPUOperator<T>>();
    registry.registerOperator(OperatorType::KV_CACHE_WRITE,
                              OperatorPlatform::CPU, kv_cache_write);

    // 注册RoPE CPU算子
    auto rope_cpu = std::make_shared<RopeCPUOperator<T>>();
    registry.registerOperator(OperatorType::ROPE, OperatorPlatform::CPU,
                              rope_cpu);

    // 注册RMS Norm CPU算子
    auto rms_norm_cpu = std::make_shared<RmsNormCPUOperator<T>>();
    registry.registerOperator(OperatorType::RMS_NORM, OperatorPlatform::CPU,
                              rms_norm_cpu);

    // 注册Multiply CPU算子
    auto multiply_cpu = std::make_shared<MultiplyCPUOperator<T>>();
    registry.registerOperator(OperatorType::MULTIPLY, OperatorPlatform::CPU,
                              multiply_cpu);

    // 注册SiLU CPU算子
    auto silu_cpu = std::make_shared<SiluCPUOperator<T>>();
    registry.registerOperator(OperatorType::SILU, OperatorPlatform::CPU,
                              silu_cpu);

    // 注册Add CPU算子
    auto add_cpu = std::make_shared<AddCPUOperator<T>>();
    registry.registerOperator(OperatorType::ADD, OperatorPlatform::CPU,
                              add_cpu);

    // 注册MatMul CPU算子
    auto matmul_cpu = std::make_shared<MatmulCPUOperator<T>>();
    registry.registerOperator(OperatorType::MATMUL, OperatorPlatform::CPU,
                              matmul_cpu);

    auto gather_cpu = std::make_shared<GatherCPUOperator<T>>();
    registry.registerOperator(OperatorType::GATHER, OperatorPlatform::CPU,
                              gather_cpu);

    // 注册Sample CPU算子
    auto sample_cpu = std::make_shared<SampleCPUOperator<T>>();
    registry.registerOperator(OperatorType::SAMPLE, OperatorPlatform::CPU,
                              sample_cpu);

    auto softmax_cpu = std::make_shared<SoftmaxCPUOperator<T>>();
    registry.registerOperator(OperatorType::SOFTMAX, OperatorPlatform::CPU,
                              softmax_cpu);

    auto attention_scores_prefill_cpu =
        std::make_shared<AttentionScoresPrefillCPUOperator<T>>();
    registry.registerOperator(OperatorType::ATTENTION_SCORES_PREFILL,
                              OperatorPlatform::CPU,
                              attention_scores_prefill_cpu);

    auto attention_output_prefill_cpu =
        std::make_shared<AttentionOutputPrefillCPUOperator<T>>();
    registry.registerOperator(OperatorType::ATTENTION_OUTPUT_PREFILL,
                              OperatorPlatform::CPU,
                              attention_output_prefill_cpu);
  }

  // 创建并注册所有CUDA算子
  static void registerCUDAOperators() {
    auto& registry = OperatorRegistry<T>::instance();

    auto kv_cache_write = std::make_shared<KvCacheWriteCUDAOperator<T>>();
    registry.registerOperator(OperatorType::KV_CACHE_WRITE,
                              OperatorPlatform::CUDA, kv_cache_write);

    // 注册RoPE CUDA算子
    auto rope_cuda = std::make_shared<RopeCUDAOperator<T>>();
    registry.registerOperator(OperatorType::ROPE, OperatorPlatform::CUDA,
                              rope_cuda);

    // 注册RMS Norm CUDA算子
    auto rms_norm_cuda = std::make_shared<RmsNormCUDAOperator<T>>();
    registry.registerOperator(OperatorType::RMS_NORM, OperatorPlatform::CUDA,
                              rms_norm_cuda);

    // 注册Multiply CUDA算子
    auto multiply_cuda = std::make_shared<MultiplyCUDAOperator<T>>();
    registry.registerOperator(OperatorType::MULTIPLY, OperatorPlatform::CUDA,
                              multiply_cuda);

    // 注册SiLU CUDA算子
    auto silu_cuda = std::make_shared<SiluCUDAOperator<T>>();
    registry.registerOperator(OperatorType::SILU, OperatorPlatform::CUDA,
                              silu_cuda);

    // 注册Add CUDA算子
    auto add_cuda = std::make_shared<AddCUDAOperator<T>>();
    registry.registerOperator(OperatorType::ADD, OperatorPlatform::CUDA,
                              add_cuda);

    // 注册MatMul CUDA算子
    auto matmul_cuda = std::make_shared<MatmulCUDAOperator<T>>();
    registry.registerOperator(OperatorType::MATMUL, OperatorPlatform::CUDA,
                              matmul_cuda);

    // 注册Softmax CUDA算子
    auto softmax_cuda = std::make_shared<SoftmaxCUDAOperator<T>>();
    registry.registerOperator(OperatorType::SOFTMAX, OperatorPlatform::CUDA,
                              softmax_cuda);

    auto gather_cuda = std::make_shared<GatherCUDAOperator<T>>();
    registry.registerOperator(OperatorType::GATHER, OperatorPlatform::CUDA,
                              gather_cuda);

    auto sample_cuda = std::make_shared<SampleCUDAOperator<T>>();
    registry.registerOperator(OperatorType::SAMPLE, OperatorPlatform::CUDA,
                              sample_cuda);

    auto dynamic_flash_attention_cuda =
        std::make_shared<DynamicFlashAttentionCUDAOperator<T>>();
    registry.registerOperator(OperatorType::DYNAMIC_FLASH_ATTENTION,
                              OperatorPlatform::CUDA,
                              dynamic_flash_attention_cuda);

    auto flash_attention_prefill_cuda =
        std::make_shared<FlashAttentionPrefillCUDAOperator<T>>();
    registry.registerOperator(OperatorType::FLASH_ATTENTION_PREFILL,
                              OperatorPlatform::CUDA,
                              flash_attention_prefill_cuda);

    auto attention_scores_prefill_cuda =
        std::make_shared<AttentionScoresPrefillCUDAOperator<T>>();
    registry.registerOperator(OperatorType::ATTENTION_SCORES_PREFILL,
                              OperatorPlatform::CUDA,
                              attention_scores_prefill_cuda);

    auto attention_output_prefill_cuda =
        std::make_shared<AttentionOutputPrefillCUDAOperator<T>>();
    registry.registerOperator(OperatorType::ATTENTION_OUTPUT_PREFILL,
                              OperatorPlatform::CUDA,
                              attention_output_prefill_cuda);

    // 注册其他CUDA算子...
  }

  // 获取算子
  static std::shared_ptr<RopeOperator<T>> getRopeOperator(
      OperatorPlatform platform) {
    auto& registry = OperatorRegistry<T>::instance();
    return registry.template getOperator<RopeOperator<T>>(OperatorType::ROPE,
                                                          platform);
  }

  // 获取RMS Norm算子
  static std::shared_ptr<RmsNormOperator<T>> getRmsNormOperator(
      OperatorPlatform platform) {
    auto& registry = OperatorRegistry<T>::instance();
    return registry.template getOperator<RmsNormOperator<T>>(
        OperatorType::RMS_NORM, platform);
  }

  // 获取Multiply算子
  static std::shared_ptr<MultiplyOperator<T>> getMultiplyOperator(
      OperatorPlatform platform) {
    auto& registry = OperatorRegistry<T>::instance();
    return registry.template getOperator<MultiplyOperator<T>>(
        OperatorType::MULTIPLY, platform);
  }

  // 获取SiLU算子
  static std::shared_ptr<SiluOperator<T>> getSiluOperator(
      OperatorPlatform platform) {
    auto& registry = OperatorRegistry<T>::instance();
    return registry.template getOperator<SiluOperator<T>>(OperatorType::SILU,
                                                          platform);
  }

  // 获取Add算子
  static std::shared_ptr<AddOperator<T>> getAddOperator(
      OperatorPlatform platform) {
    auto& registry = OperatorRegistry<T>::instance();
    return registry.template getOperator<AddOperator<T>>(OperatorType::ADD,
                                                         platform);
  }

  // 获取MatMul算子
  static std::shared_ptr<MatmulOperator<T>> getMatmulOperator(
      OperatorPlatform platform) {
    auto& registry = OperatorRegistry<T>::instance();
    return registry.template getOperator<MatmulOperator<T>>(
        OperatorType::MATMUL, platform);
  }

  // 获取Gather算子
  static std::shared_ptr<GatherOperator<T>> getGatherOperator(
      OperatorPlatform platform) {
    auto& registry = OperatorRegistry<T>::instance();
    return registry.template getOperator<GatherOperator<T>>(
        OperatorType::GATHER, platform);
  }

  // 获取Sample算子
  static std::shared_ptr<SampleOperator<T>> getSampleOperator(
      OperatorPlatform platform) {
    auto& registry = OperatorRegistry<T>::instance();
    return registry.template getOperator<SampleOperator<T>>(
        OperatorType::SAMPLE, platform);
  }

  // 获取DynamicFlashAttention算子
  static std::shared_ptr<DynamicFlashAttentionOperator<T>>
  getDynamicFlashAttentionOperator(OperatorPlatform platform) {
    auto& registry = OperatorRegistry<T>::instance();
    return registry.template getOperator<DynamicFlashAttentionOperator<T>>(
        OperatorType::DYNAMIC_FLASH_ATTENTION, platform);
  }

  // 获取AttentionScoresPrefill算子
  static std::shared_ptr<AttentionScoresPrefillOperator<T>>
  getAttentionScoresPrefillOperator(OperatorPlatform platform) {
    auto& registry = OperatorRegistry<T>::instance();
    return registry.template getOperator<AttentionScoresPrefillOperator<T>>(
        OperatorType::ATTENTION_SCORES_PREFILL, platform);
  }

  // 获取AttentionOutputPrefill算子
  static std::shared_ptr<AttentionOutputPrefillOperator<T>>
  getAttentionOutputPrefillOperator(OperatorPlatform platform) {
    auto& registry = OperatorRegistry<T>::instance();
    return registry.template getOperator<AttentionOutputPrefillOperator<T>>(
        OperatorType::ATTENTION_OUTPUT_PREFILL, platform);
  }

  static std::shared_ptr<FlashAttentionPrefillOperator<T>>
  getFlashAttentionPrefillOperator(OperatorPlatform platform) {
    auto& registry = OperatorRegistry<T>::instance();
    return registry.template getOperator<FlashAttentionPrefillOperator<T>>(
        OperatorType::FLASH_ATTENTION_PREFILL, platform);
  }

  // 获取Softmax算子
  static std::shared_ptr<SoftmaxOperator<T>> getSoftmaxOperator(
      OperatorPlatform platform) {
    auto& registry = OperatorRegistry<T>::instance();
    return registry.template getOperator<SoftmaxOperator<T>>(
        OperatorType::SOFTMAX, platform);
  }

  static std::shared_ptr<OperatorBase> getOperatorBaseByName(
      const std::string& name, OperatorPlatform platform) {
    auto& registry = OperatorRegistry<T>::instance();
    return registry.getOperatorByName(name, platform);
  }
};

}  // namespace op
