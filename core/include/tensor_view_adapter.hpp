#pragma once

#include "tensor.hpp"
#include "tensor_view.hpp"

// Cold compatibility boundary. The resulting view never retains ownership.
template <std::size_t Rank, typename T>
TensorView<T, Rank> borrow_tensor_view(Tensor<T>& tensor) {
  if (tensor.sizes().size() != Rank || tensor.strides().size() != Rank) {
    throw std::invalid_argument("TensorView adapter rank mismatch");
  }
  std::array<std::size_t, Rank> shape{}, stride{};
  for (std::size_t axis = 0; axis < Rank; ++axis) {
    shape[axis] = tensor.sizes()[axis];
    stride[axis] = tensor.strides()[axis];
  }
  return TensorView<T, Rank>::strided(tensor.data_ptr(), shape, stride);
}

template <std::size_t Rank, typename T>
TensorView<const T, Rank> borrow_tensor_view(const Tensor<T>& tensor) {
  if (tensor.sizes().size() != Rank || tensor.strides().size() != Rank) {
    throw std::invalid_argument("TensorView adapter rank mismatch");
  }
  std::array<std::size_t, Rank> shape{}, stride{};
  for (std::size_t axis = 0; axis < Rank; ++axis) {
    shape[axis] = tensor.sizes()[axis];
    stride[axis] = tensor.strides()[axis];
  }
  return TensorView<const T, Rank>::strided(tensor.data_ptr(), shape, stride);
}
