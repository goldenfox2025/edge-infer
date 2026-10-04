#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cstdint>
#include <memory>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensor.hpp"

namespace op {

template <typename T>
class WeightTensor;

enum class MatmulType { DEFAULT, CUBLAS, CUTLASS, AWQ };

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

};

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

class OperatorBase {
 public:
  virtual ~OperatorBase() = default;

  virtual OperatorType type() const = 0;

  virtual OperatorPlatform platform() const = 0;

  virtual std::string name() const = 0;

  // Behavior metadata for graph construction and workspace planning.
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

template <typename T>
class RopeOperator : public OperatorBase {
 public:
  virtual ~RopeOperator() = default;

  virtual void operator()(Tensor<T>* tensor, size_t offset, float theta,
                          cudaStream_t stream = nullptr) = 0;

  OperatorType type() const override { return OperatorType::ROPE; }

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

template <typename T>
class RmsNormOperator : public OperatorBase {
 public:
  virtual ~RmsNormOperator() = default;

  virtual void operator()(Tensor<T>* output, Tensor<T>* input,
                          Tensor<T>* weight, float eps,
                          cudaStream_t stream = nullptr) = 0;

  OperatorType type() const override { return OperatorType::RMS_NORM; }

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

template <typename T>
class AddOperator : public OperatorBase {
 public:
  virtual ~AddOperator() = default;

  virtual void operator()(Tensor<T>* output, Tensor<T>* input_a,
                          Tensor<T>* input_b,
                          cudaStream_t stream = nullptr) = 0;

  OperatorType type() const override { return OperatorType::ADD; }

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

template <typename T>
class MultiplyOperator : public OperatorBase {
 public:
  virtual ~MultiplyOperator() = default;

  virtual void operator()(Tensor<T>* output, Tensor<T>* input_a,
                          Tensor<T>* input_b,
                          cudaStream_t stream = nullptr) = 0;

  OperatorType type() const override { return OperatorType::MULTIPLY; }

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

template <typename T>
class SiluOperator : public OperatorBase {
 public:
  virtual ~SiluOperator() = default;

  virtual void operator()(Tensor<T>* output, Tensor<T>* input,
                          cudaStream_t stream = nullptr) = 0;

  OperatorType type() const override { return OperatorType::SILU; }

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

template <typename T>
class MatmulOperator : public OperatorBase {
 public:
  virtual ~MatmulOperator() = default;

  virtual void operator()(Tensor<T>* output, Tensor<T>* input,
                          const WeightTensor<T>& weight,
                          const Tensor<T>* bias = nullptr,
                          cudaStream_t stream = nullptr) = 0;

  OperatorType type() const override { return OperatorType::MATMUL; }

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

template <typename T>
class MatmulOperatorImpl : public MatmulOperator<T> {
 public:
  virtual ~MatmulOperatorImpl() = default;

  virtual MatmulType impl_type() const = 0;
};

template <typename T>
class GatherOperator : public OperatorBase {
 public:
  virtual ~GatherOperator() = default;

  virtual void operator()(Tensor<T>* output, const Tensor<uint32_t>* input,
                          const Tensor<T>* embedding_table,
                          cudaStream_t stream = nullptr) = 0;

  OperatorType type() const override { return OperatorType::GATHER; }

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

template <typename T>
class SampleOperator : public OperatorBase {
 public:
  virtual ~SampleOperator() = default;

  virtual uint32_t* operator()(Tensor<T>&& logits, float temperature,
                               float top_p, size_t top_k, curandState* d_states,
                               cudaStream_t stream = nullptr) = 0;

  OperatorType type() const override { return OperatorType::SAMPLE; }

  std::string name() const override { return "sample"; }

  OperatorBehavior behavior() const override {
    OperatorBehavior info;
    info.has_side_effect = true;
    info.output_count = 0;
    return info;
  }
};

template <typename T>
class DynamicFlashAttentionOperator : public OperatorBase {
 public:
  virtual ~DynamicFlashAttentionOperator() = default;

  virtual void operator()(Tensor<T>& Q, const Tensor<T>& K, const Tensor<T>& V,
                          Tensor<T>& output, int n_kv_heads,
                          cudaStream_t stream = nullptr) = 0;

  OperatorType type() const override {
    return OperatorType::DYNAMIC_FLASH_ATTENTION;
  }

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

template <typename T>
class AttentionScoresPrefillOperator : public OperatorBase {
 public:
  virtual ~AttentionScoresPrefillOperator() = default;

  virtual void operator()(const Tensor<T>& Q, const Tensor<T>& K,
                          Tensor<T>& att_scores, size_t head_dim,
                          cudaStream_t stream = nullptr) = 0;

  OperatorType type() const override {
    return OperatorType::ATTENTION_SCORES_PREFILL;
  }

  std::string name() const override { return "attention_scores_prefill"; }
};

template <typename T>
class AttentionOutputPrefillOperator : public OperatorBase {
 public:
  virtual ~AttentionOutputPrefillOperator() = default;

  virtual void operator()(const Tensor<T>& att_scores, const Tensor<T>& V,
                          Tensor<T>& att_output, size_t n_heads,
                          size_t head_dim, size_t total_seq_len,
                          size_t n_kv_heads, cudaStream_t stream = nullptr) = 0;

  OperatorType type() const override {
    return OperatorType::ATTENTION_OUTPUT_PREFILL;
  }

  std::string name() const override { return "attention_output_prefill"; }
};

template <typename T>
class SoftmaxOperator : public OperatorBase {
 public:
  virtual ~SoftmaxOperator() = default;

  virtual void operator()(Tensor<T>* output, const Tensor<T>* input, int dim,
                          bool mask = false, int offset = 0,
                          cudaStream_t stream = nullptr) = 0;

  OperatorType type() const override { return OperatorType::SOFTMAX; }

  std::string name() const override { return "softmax"; }
};

template <typename T>
class OperatorRegistry {
 public:
  static OperatorRegistry<T>& instance() {
    static OperatorRegistry<T> instance;
    return instance;
  }

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

  template <typename OpType>
  std::shared_ptr<OpType> getOperator(OperatorType type,
                                      OperatorPlatform platform) {
    std::string key = getKey(type, platform);
    if (operators_.find(key) == operators_.end()) {
      return nullptr;
    }
    return std::dynamic_pointer_cast<OpType>(operators_[key]);
  }

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

  OperatorRegistry(const OperatorRegistry&) = delete;
  OperatorRegistry& operator=(const OperatorRegistry&) = delete;

  std::string getKey(OperatorType type, OperatorPlatform platform) {
    return std::to_string(static_cast<int>(type)) + "_" +
           std::to_string(static_cast<int>(platform));
  }

  std::string getNameKey(const std::string& name, OperatorPlatform platform) {
    return name + "_" + std::to_string(static_cast<int>(platform));
  }

  std::unordered_map<std::string, std::shared_ptr<OperatorBase>> operators_;
  std::unordered_map<std::string, std::shared_ptr<OperatorBase>> operators_by_name_;
  std::unordered_map<std::string, std::shared_ptr<OperatorBase>> operators_any_platform_;
};

}  // namespace op
