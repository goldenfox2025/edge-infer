#pragma once
#include "tensor.hpp"

class KVCacheBase {
 public:
  virtual ~KVCacheBase() = default;

  virtual void resize(size_t new_size) = 0;
  virtual void clear() = 0;
  virtual size_t size() const = 0;
  virtual Device device() const = 0;

  virtual size_t get_n_layers() const = 0;
  virtual size_t get_head_dim() const = 0;
  virtual size_t get_max_seq_len() const = 0;
};
