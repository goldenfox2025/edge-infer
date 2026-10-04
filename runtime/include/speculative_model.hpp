#pragma once

#include "kvcache_base.hpp"
#include "tensor.hpp"

template <typename T>
class KVCache;

template <typename T>
class SpeculativeModel {
 public:
  virtual ~SpeculativeModel() = default;

  virtual Tensor<T> speculative_forward_logits(const Tensor<uint32_t>* input,
                                               KVCache<T>* kv_cache) = 0;
  virtual Tensor<T> speculative_prefill_logits(const Tensor<uint32_t>* input,
                                               KVCache<T>* kv_cache) = 0;
};
