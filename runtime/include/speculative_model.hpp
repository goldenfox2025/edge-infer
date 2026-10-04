#pragma once

#include "kvcache_base.hpp"
#include "tensor.hpp"
#include "tensor_view.hpp"

template <typename T>
class KVCache;

template <typename T>
class SpeculativeModel {
 public:
  virtual ~SpeculativeModel() = default;

  virtual TensorView<T, 2> speculative_forward_logits(const Tensor<uint32_t>* input,
                                               KVCache<T>* kv_cache) = 0;
  virtual TensorView<T, 2> speculative_prefill_logits(const Tensor<uint32_t>* input,
                                               KVCache<T>* kv_cache) = 0;
};
