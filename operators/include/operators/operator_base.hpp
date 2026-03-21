#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>  // 添加这个头文件以声明 curandState 类型

#include <cstdint>
#include <memory>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensor.hpp"

namespace op {

// 前向声明 WeightTensor
template <typename T>
class WeightTensor;

// MatMul实现类型枚举
enum class MatmulType { DEFAULT, CUBLAS, CUTLASS, AWQ };

// 算子类型枚举
enum class OperatorType {
  ROPE,
  RMS_NORM,
  KV_CACHE_WRITE,
  MATMUL,
  GATHER,
  SOFTMAX,
  ADD,
  MULTIPLY,
  SILU,
  SAMPLE,
  DYNAMIC_FLASH_ATTENTION,
  FLASH_ATTENTION_PREFILL,
  ATTENTION_SCORES_PREFILL,
  ATTENTION_OUTPUT_PREFILL,
  // 添加更多算子类型...
};

// 算子平台枚举
enum class OperatorPlatform {
  CPU,
  CUDA,
};

struct OperatorBehavior {
  bool supports_inplace = false;
  bool has_side_effect = false;
  size_t output_count = 1;
};

using OperatorShape = std::vector<size_t>;
using OperatorShapeList = std::vector<OperatorShape>;

enum class PackedTensorArgKind : uint8_t {
  OutputTensor,
  InputTensor,
  RuntimeInputTensor,
  KvCacheKTensor,
  KvCacheVTensor,
};

struct PackedTensorArg {
  PackedTensorArgKind kind;
  uint16_t index = 0;
  uint16_t aux = 0;
};

constexpr size_t kPackedRuntimeOffset = 0;
constexpr size_t kPackedRuntimeThetaBits = 1;
constexpr size_t kPackedRuntimeEpsBits = 2;
constexpr size_t kPackedRuntimeNHeads = 3;
constexpr size_t kPackedRuntimeNKvHeads = 4;
constexpr size_t kPackedRuntimeHeadDim = 5;
constexpr size_t kPackedRuntimeSeqLen = 6;
constexpr size_t kPackedRuntimeTotalSeqLen = 7;
constexpr size_t kPackedRuntimeViewHeads = 8;

inline float unpack_packed_float(uint64_t bits) {
  uint32_t value = static_cast<uint32_t>(bits);
  float result = 0.0f;
  std::memcpy(&result, &value, sizeof(float));
  return result;
}

template <typename T>
inline Tensor<T> packed_view(Tensor<T>* tensor, const std::vector<size_t>& shape) {
  return static_cast<const Tensor<T>&>(*tensor).view(shape);
}

// 基础算子接口
class OperatorBase {
 public:
  virtual ~OperatorBase() = default;

  // 获取算子类型
  virtual OperatorType type() const = 0;

  // 获取算子平台
  virtual OperatorPlatform platform() const = 0;

  // 获取算子名称
  virtual std::string name() const = 0;

  // 算子行为描述，供构图/内存规划层查询。
  virtual OperatorBehavior behavior() const { return {}; }

  virtual OperatorShapeList infer_output_shapes(
      const OperatorShapeList&,
      const OperatorShapeList& hinted_outputs) const {
    return hinted_outputs;
  }

  virtual size_t infer_workspace_bytes(const OperatorShapeList&,
                                       const OperatorShapeList&) const {
    return 0;
  }

  virtual size_t workspace_alignment() const { return 256; }

  virtual std::vector<PackedTensorArg> packed_tensor_args(
      bool /*inplace*/ = false, uint16_t /*view_heads*/ = 0) const {
    return {};
  }

  virtual void execute_packed(const std::vector<void*>&,
                              const std::vector<void*>&,
                              const std::vector<uint64_t>&,
                              cudaStream_t = nullptr) {
    throw std::runtime_error("Packed execution not implemented for operator: " +
                             name());
  }
};

// RoPE算子接口
template <typename T>
class RopeOperator : public OperatorBase {
 public:
  virtual ~RopeOperator() = default;

  // RoPE算子实现 - 使用一重指针
  virtual void operator()(Tensor<T>* tensor, size_t offset, float theta,
                          cudaStream_t stream = nullptr) = 0;

  // 获取算子类型
  OperatorType type() const override { return OperatorType::ROPE; }

  // 获取算子名称
  std::string name() const override { return "rope"; }

  OperatorBehavior behavior() const override {
    OperatorBehavior info;
    info.supports_inplace = true;
    return info;
  }

  OperatorShapeList infer_output_shapes(
      const OperatorShapeList& inputs,
      const OperatorShapeList& hinted_outputs) const override {
    return !inputs.empty() ? OperatorShapeList{inputs.front()} : hinted_outputs;
  }

  std::vector<PackedTensorArg> packed_tensor_args(
      bool, uint16_t view_heads) const override {
    return {{PackedTensorArgKind::OutputTensor, 0, view_heads}};
  }

  void execute_packed(const std::vector<void*>& tensors,
                      const std::vector<void*>&,
                      const std::vector<uint64_t>& runtime,
                      cudaStream_t stream = nullptr) override {
    if (tensors.size() != 1) {
      throw std::runtime_error("rope packed args mismatch");
    }
    auto* tensor = static_cast<Tensor<T>*>(tensors[0]);
    const size_t seq_len = static_cast<size_t>(runtime[kPackedRuntimeSeqLen]);
    const size_t view_heads =
        static_cast<size_t>(runtime[kPackedRuntimeViewHeads]);
    const size_t head_dim = static_cast<size_t>(runtime[kPackedRuntimeHeadDim]);
    const size_t offset = static_cast<size_t>(runtime[kPackedRuntimeOffset]);
    const float theta = unpack_packed_float(runtime[kPackedRuntimeThetaBits]);
    auto view = packed_view(tensor, {seq_len, view_heads, head_dim});
    (*this)(&view, offset, theta, stream);
  }
};

// RMS Norm算子接口
template <typename T>
class RmsNormOperator : public OperatorBase {
 public:
  virtual ~RmsNormOperator() = default;

  // RMS Norm算子实现 - 使用一重指针
  virtual void operator()(Tensor<T>* output, Tensor<T>* input,
                          Tensor<T>* weight, float eps,
                          cudaStream_t stream = nullptr) = 0;

  // 获取算子类型
  OperatorType type() const override { return OperatorType::RMS_NORM; }

  // 获取算子名称
  std::string name() const override { return "rms_norm"; }

  OperatorBehavior behavior() const override {
    OperatorBehavior info;
    info.supports_inplace = true;
    return info;
  }

  OperatorShapeList infer_output_shapes(
      const OperatorShapeList& inputs,
      const OperatorShapeList& hinted_outputs) const override {
    return !inputs.empty() ? OperatorShapeList{inputs.front()} : hinted_outputs;
  }

  std::vector<PackedTensorArg> packed_tensor_args(
      bool inplace, uint16_t view_heads) const override {
    return {
        {PackedTensorArgKind::OutputTensor, 0, 0},
        {PackedTensorArgKind::InputTensor, 0, 0},
    };
  }

  void execute_packed(const std::vector<void*>& tensors,
                      const std::vector<void*>& statics,
                      const std::vector<uint64_t>& runtime,
                      cudaStream_t stream = nullptr) override {
    if (tensors.size() != 2 || statics.size() != 1) {
      throw std::runtime_error("rms_norm packed args mismatch");
    }
    auto* output = static_cast<Tensor<T>*>(tensors[0]);
    auto* input = static_cast<Tensor<T>*>(tensors[1]);
    auto* weight = static_cast<Tensor<T>*>(statics[0]);
    const auto eps = unpack_packed_float(runtime[kPackedRuntimeEpsBits]);
    const size_t view_heads =
        static_cast<size_t>(runtime[kPackedRuntimeViewHeads]);
    if (view_heads != 0) {
      const size_t seq_len = static_cast<size_t>(runtime[kPackedRuntimeSeqLen]);
      const size_t head_dim =
          static_cast<size_t>(runtime[kPackedRuntimeHeadDim]);
      auto output_view = packed_view(output, {seq_len, view_heads, head_dim});
      auto input_view = packed_view(input, {seq_len, view_heads, head_dim});
      (*this)(&output_view, &input_view, weight, eps, stream);
      return;
    }
    (*this)(output, input, weight, eps, stream);
  }
};

template <typename T>
class KvCacheWriteOperator : public OperatorBase {
 public:
  ~KvCacheWriteOperator() override = default;

  virtual void operator()(const Tensor<T>* src_k, const Tensor<T>* src_v,
                          Tensor<T>* dst_k_cache, Tensor<T>* dst_v_cache,
                          size_t offset, cudaStream_t stream = nullptr) = 0;

  OperatorType type() const override { return OperatorType::KV_CACHE_WRITE; }
  std::string name() const override { return "kv_cache_write"; }
  OperatorBehavior behavior() const override {
    OperatorBehavior info;
    info.has_side_effect = true;
    info.output_count = 0;
    return info;
  }

  OperatorShapeList infer_output_shapes(
      const OperatorShapeList&,
      const OperatorShapeList&) const override {
    return {};
  }

  std::vector<PackedTensorArg> packed_tensor_args(
      bool, uint16_t view_heads) const override {
    return {
        {PackedTensorArgKind::InputTensor, 0, view_heads},
        {PackedTensorArgKind::InputTensor, 1, view_heads},
        {PackedTensorArgKind::KvCacheKTensor, 0, 0},
        {PackedTensorArgKind::KvCacheVTensor, 0, 0},
    };
  }

  void execute_packed(const std::vector<void*>& tensors,
                      const std::vector<void*>&,
                      const std::vector<uint64_t>& runtime,
                      cudaStream_t stream = nullptr) override {
    if (tensors.size() != 4) {
      throw std::runtime_error("kv_cache_write packed args mismatch");
    }
    auto* src_k_base = static_cast<Tensor<T>*>(tensors[0]);
    auto* src_v_base = static_cast<Tensor<T>*>(tensors[1]);
    auto* dst_k_cache = static_cast<Tensor<T>*>(tensors[2]);
    auto* dst_v_cache = static_cast<Tensor<T>*>(tensors[3]);
    const size_t seq_len = static_cast<size_t>(runtime[kPackedRuntimeSeqLen]);
    const size_t view_heads =
        static_cast<size_t>(runtime[kPackedRuntimeViewHeads]);
    const size_t head_dim = static_cast<size_t>(runtime[kPackedRuntimeHeadDim]);
    const size_t offset = static_cast<size_t>(runtime[kPackedRuntimeOffset]);
    auto src_k = packed_view(src_k_base, {seq_len, view_heads, head_dim});
    auto src_v = packed_view(src_v_base, {seq_len, view_heads, head_dim});
    (*this)(&src_k, &src_v, dst_k_cache, dst_v_cache, offset, stream);
  }
};

// Add算子接口
template <typename T>
class AddOperator : public OperatorBase {
 public:
  virtual ~AddOperator() = default;

  // Add算子实现 - 使用一重指针
  virtual void operator()(Tensor<T>* output, Tensor<T>* input_a,
                          Tensor<T>* input_b,
                          cudaStream_t stream = nullptr) = 0;

  // 获取算子类型
  OperatorType type() const override { return OperatorType::ADD; }

  // 获取算子名称
  std::string name() const override { return "add"; }

  OperatorShapeList infer_output_shapes(
      const OperatorShapeList& inputs,
      const OperatorShapeList& hinted_outputs) const override {
    return !inputs.empty() ? OperatorShapeList{inputs.front()} : hinted_outputs;
  }

  std::vector<PackedTensorArg> packed_tensor_args(
      bool, uint16_t) const override {
    return {
        {PackedTensorArgKind::OutputTensor, 0, 0},
        {PackedTensorArgKind::InputTensor, 0, 0},
        {PackedTensorArgKind::InputTensor, 1, 0},
    };
  }

  void execute_packed(const std::vector<void*>& tensors,
                      const std::vector<void*>&,
                      const std::vector<uint64_t>&,
                      cudaStream_t stream = nullptr) override {
    if (tensors.size() != 3) {
      throw std::runtime_error("add packed args mismatch");
    }
    auto* output = static_cast<Tensor<T>*>(tensors[0]);
    auto* input_a = static_cast<Tensor<T>*>(tensors[1]);
    auto* input_b = static_cast<Tensor<T>*>(tensors[2]);
    (*this)(output, input_a, input_b, stream);
  }
};

// Multiply算子接口
template <typename T>
class MultiplyOperator : public OperatorBase {
 public:
  virtual ~MultiplyOperator() = default;

  // Multiply算子实现 - 使用一重指针
  virtual void operator()(Tensor<T>* output, Tensor<T>* input_a,
                          Tensor<T>* input_b,
                          cudaStream_t stream = nullptr) = 0;

  // 获取算子类型
  OperatorType type() const override { return OperatorType::MULTIPLY; }

  // 获取算子名称
  std::string name() const override { return "multiply"; }

  OperatorBehavior behavior() const override {
    OperatorBehavior info;
    info.supports_inplace = true;
    return info;
  }

  OperatorShapeList infer_output_shapes(
      const OperatorShapeList& inputs,
      const OperatorShapeList& hinted_outputs) const override {
    return !inputs.empty() ? OperatorShapeList{inputs.front()} : hinted_outputs;
  }

  std::vector<PackedTensorArg> packed_tensor_args(
      bool, uint16_t) const override {
    return {
        {PackedTensorArgKind::OutputTensor, 0, 0},
        {PackedTensorArgKind::InputTensor, 0, 0},
        {PackedTensorArgKind::InputTensor, 1, 0},
    };
  }

  void execute_packed(const std::vector<void*>& tensors,
                      const std::vector<void*>&,
                      const std::vector<uint64_t>&,
                      cudaStream_t stream = nullptr) override {
    if (tensors.size() != 3) {
      throw std::runtime_error("multiply packed args mismatch");
    }
    auto* output = static_cast<Tensor<T>*>(tensors[0]);
    auto* input_a = static_cast<Tensor<T>*>(tensors[1]);
    auto* input_b = static_cast<Tensor<T>*>(tensors[2]);
    (*this)(output, input_a, input_b, stream);
  }
};

// SiLU算子接口
template <typename T>
class SiluOperator : public OperatorBase {
 public:
  virtual ~SiluOperator() = default;

  // SiLU算子实现 - 使用一重指针
  virtual void operator()(Tensor<T>* output, Tensor<T>* input,
                          cudaStream_t stream = nullptr) = 0;

  // 获取算子类型
  OperatorType type() const override { return OperatorType::SILU; }

  // 获取算子名称
  std::string name() const override { return "silu"; }

  OperatorBehavior behavior() const override {
    OperatorBehavior info;
    info.supports_inplace = true;
    return info;
  }

  OperatorShapeList infer_output_shapes(
      const OperatorShapeList& inputs,
      const OperatorShapeList& hinted_outputs) const override {
    return !inputs.empty() ? OperatorShapeList{inputs.front()} : hinted_outputs;
  }

  std::vector<PackedTensorArg> packed_tensor_args(
      bool, uint16_t) const override {
    return {
        {PackedTensorArgKind::OutputTensor, 0, 0},
        {PackedTensorArgKind::InputTensor, 0, 0},
    };
  }

  void execute_packed(const std::vector<void*>& tensors,
                      const std::vector<void*>&,
                      const std::vector<uint64_t>&,
                      cudaStream_t stream = nullptr) override {
    if (tensors.size() != 2) {
      throw std::runtime_error("silu packed args mismatch");
    }
    auto* output = static_cast<Tensor<T>*>(tensors[0]);
    auto* input = static_cast<Tensor<T>*>(tensors[1]);
    (*this)(output, input, stream);
  }
};

// MatMul算子接口
template <typename T>
class MatmulOperator : public OperatorBase {
 public:
  virtual ~MatmulOperator() = default;

  // MatMul算子实现 - 使用WeightTensor作为参数
  virtual void operator()(Tensor<T>* output, Tensor<T>* input,
                          const WeightTensor<T>& weight,
                          const Tensor<T>* bias = nullptr,
                          cudaStream_t stream = nullptr) = 0;

  // 获取算子类型
  OperatorType type() const override { return OperatorType::MATMUL; }

  // 获取算子名称
  std::string name() const override { return "matmul"; }

  std::vector<PackedTensorArg> packed_tensor_args(
      bool, uint16_t) const override {
    return {
        {PackedTensorArgKind::OutputTensor, 0, 0},
        {PackedTensorArgKind::InputTensor, 0, 0},
    };
  }

  void execute_packed(const std::vector<void*>& tensors,
                      const std::vector<void*>& statics,
                      const std::vector<uint64_t>&,
                      cudaStream_t stream = nullptr) override {
    if (tensors.size() != 2 || (statics.size() != 1 && statics.size() != 2)) {
      throw std::runtime_error("matmul packed args mismatch");
    }
    auto* output = static_cast<Tensor<T>*>(tensors[0]);
    auto* input = static_cast<Tensor<T>*>(tensors[1]);
    auto* weight = static_cast<const WeightTensor<T>*>(statics[0]);
    const Tensor<T>* bias =
        statics.size() == 2 ? static_cast<const Tensor<T>*>(statics[1]) : nullptr;
    (*this)(output, input, *weight, bias, stream);
  }
};

// MatMul算子实现的基类
template <typename T>
class MatmulOperatorImpl : public MatmulOperator<T> {
 public:
  virtual ~MatmulOperatorImpl() = default;

  // 获取MatMul实现类型
  virtual MatmulType impl_type() const = 0;
};

// Gather算子接口
template <typename T>
class GatherOperator : public OperatorBase {
 public:
  virtual ~GatherOperator() = default;

  // Gather算子实现
  virtual void operator()(Tensor<T>* output, const Tensor<uint32_t>* input,
                          const Tensor<T>* embedding_table,
                          cudaStream_t stream = nullptr) = 0;

  // 获取算子类型
  OperatorType type() const override { return OperatorType::GATHER; }

  // 获取算子名称
  std::string name() const override { return "gather"; }

  std::vector<PackedTensorArg> packed_tensor_args(
      bool, uint16_t) const override {
    return {
        {PackedTensorArgKind::OutputTensor, 0, 0},
        {PackedTensorArgKind::RuntimeInputTensor, 0, 0},
    };
  }

  void execute_packed(const std::vector<void*>& tensors,
                      const std::vector<void*>& statics,
                      const std::vector<uint64_t>&,
                      cudaStream_t stream = nullptr) override {
    if (tensors.size() != 2 || statics.size() != 1) {
      throw std::runtime_error("gather packed args mismatch");
    }
    auto* output = static_cast<Tensor<T>*>(tensors[0]);
    auto* input = static_cast<const Tensor<uint32_t>*>(tensors[1]);
    auto* embedding = static_cast<const Tensor<T>*>(statics[0]);
    (*this)(output, input, embedding, stream);
  }
};

// Sample算子接口
template <typename T>
class SampleOperator : public OperatorBase {
 public:
  virtual ~SampleOperator() = default;

  // Sample算子实现
  virtual uint32_t* operator()(Tensor<T>&& logits, float temperature,
                               float top_p, size_t top_k, curandState* d_states,
                               cudaStream_t stream = nullptr) = 0;

  // 获取算子类型
  OperatorType type() const override { return OperatorType::SAMPLE; }

  // 获取算子名称
  std::string name() const override { return "sample"; }

  OperatorBehavior behavior() const override {
    OperatorBehavior info;
    info.has_side_effect = true;
    info.output_count = 0;
    return info;
  }
};

// DynamicFlashAttention算子接口
template <typename T>
class DynamicFlashAttentionOperator : public OperatorBase {
 public:
  virtual ~DynamicFlashAttentionOperator() = default;

  // DynamicFlashAttention算子实现
  virtual void operator()(Tensor<T>& Q, const Tensor<T>& K, const Tensor<T>& V,
                          Tensor<T>& output, int n_kv_heads,
                          cudaStream_t stream = nullptr) = 0;

  // 获取算子类型
  OperatorType type() const override {
    return OperatorType::DYNAMIC_FLASH_ATTENTION;
  }

  // 获取算子名称
  std::string name() const override { return "dynamic_flash_attention"; }

  OperatorShapeList infer_output_shapes(
      const OperatorShapeList& inputs,
      const OperatorShapeList& hinted_outputs) const override {
    return !inputs.empty() ? OperatorShapeList{inputs.front()} : hinted_outputs;
  }

  std::vector<PackedTensorArg> packed_tensor_args(
      bool, uint16_t view_heads) const override {
    return {
        {PackedTensorArgKind::InputTensor, 0, view_heads},
        {PackedTensorArgKind::KvCacheKTensor, 0, 0},
        {PackedTensorArgKind::KvCacheVTensor, 0, 0},
        {PackedTensorArgKind::OutputTensor, 0, view_heads},
    };
  }

  void execute_packed(const std::vector<void*>& tensors,
                      const std::vector<void*>&,
                      const std::vector<uint64_t>& runtime,
                      cudaStream_t stream = nullptr) override {
    if (tensors.size() != 4) {
      throw std::runtime_error("dynamic_flash_attention packed args mismatch");
    }
    auto* q_base = static_cast<Tensor<T>*>(tensors[0]);
    auto* k_cache = static_cast<Tensor<T>*>(tensors[1]);
    auto* v_cache = static_cast<Tensor<T>*>(tensors[2]);
    auto* output_base = static_cast<Tensor<T>*>(tensors[3]);
    const int n_kv_heads =
        static_cast<int>(runtime[kPackedRuntimeNKvHeads]);
    const size_t seq_len = static_cast<size_t>(runtime[kPackedRuntimeSeqLen]);
    const size_t total_seq_len =
        static_cast<size_t>(runtime[kPackedRuntimeTotalSeqLen]);
    const size_t view_heads =
        static_cast<size_t>(runtime[kPackedRuntimeViewHeads]);
    const size_t head_dim = static_cast<size_t>(runtime[kPackedRuntimeHeadDim]);
    auto q = packed_view(q_base, {seq_len, view_heads, head_dim});
    auto k = packed_view(k_cache,
        {total_seq_len, static_cast<size_t>(n_kv_heads), head_dim});
    auto v = packed_view(v_cache,
        {total_seq_len, static_cast<size_t>(n_kv_heads), head_dim});
    Tensor<T> output =
        seq_len == 1 ? packed_view(output_base, {view_heads, head_dim})
                     : packed_view(output_base, {seq_len, view_heads, head_dim});
    (*this)(q, k, v, output, n_kv_heads, stream);
  }
};

// FlashAttentionPrefill算子接口
template <typename T>
class FlashAttentionPrefillOperator : public OperatorBase {
 public:
  virtual ~FlashAttentionPrefillOperator() = default;

  virtual void operator()(const Tensor<T>& Q, const Tensor<T>& K,
                          const Tensor<T>& V, Tensor<T>& output, int n_heads,
                          int n_kv_heads, int head_dim, int seq_len,
                          int total_seq_len, int offset,
                          cudaStream_t stream = nullptr) = 0;

  OperatorType type() const override {
    return OperatorType::FLASH_ATTENTION_PREFILL;
  }

  std::string name() const override { return "flash_attention_prefill"; }

  OperatorShapeList infer_output_shapes(
      const OperatorShapeList& inputs,
      const OperatorShapeList& hinted_outputs) const override {
    return !inputs.empty() ? OperatorShapeList{inputs.front()} : hinted_outputs;
  }

  std::vector<PackedTensorArg> packed_tensor_args(
      bool, uint16_t view_heads) const override {
    return {
        {PackedTensorArgKind::InputTensor, 0, view_heads},
        {PackedTensorArgKind::KvCacheKTensor, 0, 0},
        {PackedTensorArgKind::KvCacheVTensor, 0, 0},
        {PackedTensorArgKind::OutputTensor, 0, view_heads},
    };
  }

  void execute_packed(const std::vector<void*>& tensors,
                      const std::vector<void*>&,
                      const std::vector<uint64_t>& runtime,
                      cudaStream_t stream = nullptr) override {
    if (tensors.size() != 4) {
      throw std::runtime_error("flash_attention_prefill packed args mismatch");
    }
    auto* q_base = static_cast<Tensor<T>*>(tensors[0]);
    auto* k_cache = static_cast<Tensor<T>*>(tensors[1]);
    auto* v_cache = static_cast<Tensor<T>*>(tensors[2]);
    auto* output_base = static_cast<Tensor<T>*>(tensors[3]);
    const int n_heads = static_cast<int>(runtime[kPackedRuntimeNHeads]);
    const int n_kv_heads =
        static_cast<int>(runtime[kPackedRuntimeNKvHeads]);
    const int head_dim = static_cast<int>(runtime[kPackedRuntimeHeadDim]);
    const int seq_len = static_cast<int>(runtime[kPackedRuntimeSeqLen]);
    const int total_seq_len =
        static_cast<int>(runtime[kPackedRuntimeTotalSeqLen]);
    const int offset = static_cast<int>(runtime[kPackedRuntimeOffset]);
    const size_t view_heads =
        static_cast<size_t>(runtime[kPackedRuntimeViewHeads]);
    auto q = packed_view(q_base, {static_cast<size_t>(seq_len), view_heads,
                           static_cast<size_t>(head_dim)});
    auto k = packed_view(k_cache, {static_cast<size_t>(total_seq_len),
                            static_cast<size_t>(n_kv_heads),
                            static_cast<size_t>(head_dim)});
    auto v = packed_view(v_cache, {static_cast<size_t>(total_seq_len),
                            static_cast<size_t>(n_kv_heads),
                            static_cast<size_t>(head_dim)});
    auto output = packed_view(output_base, {static_cast<size_t>(seq_len), view_heads,
                                     static_cast<size_t>(head_dim)});
    (*this)(q, k, v, output, n_heads, n_kv_heads, head_dim, seq_len,
            total_seq_len, offset, stream);
  }
};

// AttentionScoresPrefill算子接口
template <typename T>
class AttentionScoresPrefillOperator : public OperatorBase {
 public:
  virtual ~AttentionScoresPrefillOperator() = default;

  // AttentionScoresPrefill算子实现
  virtual void operator()(const Tensor<T>& Q, const Tensor<T>& K,
                          Tensor<T>& att_scores, size_t head_dim,
                          cudaStream_t stream = nullptr) = 0;

  // 获取算子类型
  OperatorType type() const override {
    return OperatorType::ATTENTION_SCORES_PREFILL;
  }

  // 获取算子名称
  std::string name() const override { return "attention_scores_prefill"; }
};

// AttentionOutputPrefill算子接口
template <typename T>
class AttentionOutputPrefillOperator : public OperatorBase {
 public:
  virtual ~AttentionOutputPrefillOperator() = default;

  // AttentionOutputPrefill算子实现
  virtual void operator()(const Tensor<T>& att_scores, const Tensor<T>& V,
                          Tensor<T>& att_output, size_t n_heads,
                          size_t head_dim, size_t total_seq_len,
                          size_t n_kv_heads, cudaStream_t stream = nullptr) = 0;

  // 获取算子类型
  OperatorType type() const override {
    return OperatorType::ATTENTION_OUTPUT_PREFILL;
  }

  // 获取算子名称
  std::string name() const override { return "attention_output_prefill"; }
};

// Softmax算子接口
template <typename T>
class SoftmaxOperator : public OperatorBase {
 public:
  virtual ~SoftmaxOperator() = default;

  // Softmax算子实现
  virtual void operator()(Tensor<T>* output, const Tensor<T>* input, int dim,
                          bool mask = false, int offset = 0,
                          cudaStream_t stream = nullptr) = 0;

  // 获取算子类型
  OperatorType type() const override { return OperatorType::SOFTMAX; }

  // 获取算子名称
  std::string name() const override { return "softmax"; }
};

// 其他算子接口可以在这里添加...

// 算子管理器 - 单例模式
template <typename T>
class OperatorRegistry {
 public:
  static OperatorRegistry<T>& instance() {
    static OperatorRegistry<T> instance;
    return instance;
  }

  // 注册算子
  void registerOperator(OperatorType type, OperatorPlatform platform,
                        std::shared_ptr<OperatorBase> op) {
    std::string key = getKey(type, platform);
    if (operators_.find(key) != operators_.end()) {
      return;
    }
    operators_[key] = op;
    if (op) {
      operators_by_name_[getNameKey(op->name(), platform)] = op;
      if (operators_any_platform_.find(op->name()) == operators_any_platform_.end()) {
        operators_any_platform_[op->name()] = op;
      }
    }
  }

  // 获取算子
  template <typename OpType>
  std::shared_ptr<OpType> getOperator(OperatorType type,
                                      OperatorPlatform platform) {
    std::string key = getKey(type, platform);
    if (operators_.find(key) == operators_.end()) {
      return nullptr;
    }
    return std::dynamic_pointer_cast<OpType>(operators_[key]);
  }

  // 检查算子是否已注册
  bool hasOperator(OperatorType type, OperatorPlatform platform) {
    std::string key = getKey(type, platform);
    return operators_.find(key) != operators_.end();
  }

  std::shared_ptr<OperatorBase> getOperatorByName(const std::string& name) {
    auto it = operators_any_platform_.find(name);
    return it == operators_any_platform_.end() ? nullptr : it->second;
  }

  std::shared_ptr<OperatorBase> getOperatorByName(const std::string& name,
                                                  OperatorPlatform platform) {
    auto it = operators_by_name_.find(getNameKey(name, platform));
    return it == operators_by_name_.end() ? nullptr : it->second;
  }

 private:
  OperatorRegistry() = default;
  ~OperatorRegistry() = default;

  // 禁止拷贝和赋值
  OperatorRegistry(const OperatorRegistry&) = delete;
  OperatorRegistry& operator=(const OperatorRegistry&) = delete;

  // 生成算子键
  std::string getKey(OperatorType type, OperatorPlatform platform) {
    return std::to_string(static_cast<int>(type)) + "_" +
           std::to_string(static_cast<int>(platform));
  }

  std::string getNameKey(const std::string& name, OperatorPlatform platform) {
    return name + "_" + std::to_string(static_cast<int>(platform));
  }

  // 存储算子的映射表
  std::unordered_map<std::string, std::shared_ptr<OperatorBase>> operators_;
  std::unordered_map<std::string, std::shared_ptr<OperatorBase>> operators_by_name_;
  std::unordered_map<std::string, std::shared_ptr<OperatorBase>> operators_any_platform_;
};

}  // namespace op
