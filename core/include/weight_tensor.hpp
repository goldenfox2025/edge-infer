#pragma once

#include <cuda_bf16.h>

#include <memory>
#include <stdexcept>
#include <string>

#include "tensor.hpp"

namespace op {

// Wrap either dense weights or quantized weights behind one interface.
template <typename T>
class WeightTensor {
   public:
    // Construct from dense weights.
    WeightTensor(const Tensor<T>* tensor) : tensor_(tensor), is_quantized_(false) {
    }

    // Construct from quantized weights.
    WeightTensor(const Tensor<int32_t>* qweight, const Tensor<T>* scales, const Tensor<int32_t>* qzeros, int group_size)
        : qweight_(qweight), scales_(scales), qzeros_(qzeros), group_size_(group_size), is_quantized_(true) {
    }

    // Return whether these weights are quantized.
    bool is_quantized() const {
        return is_quantized_;
    }

    // Access dense weights.
    const Tensor<T>* tensor() const {
        if (is_quantized_)
            throw std::runtime_error("Not a regular tensor");
        return tensor_;
    }

    // Access the quantized weight parameters.
    const Tensor<int32_t>* qweight() const {
        if (!is_quantized_)
            throw std::runtime_error("Not a quantized tensor");
        return qweight_;
    }

    const Tensor<T>* scales() const {
        if (!is_quantized_)
            throw std::runtime_error("Not a quantized tensor");
        return scales_;
    }

    const Tensor<int32_t>* qzeros() const {
        if (!is_quantized_)
            throw std::runtime_error("Not a quantized tensor");
        return qzeros_;
    }

    int group_size() const {
        if (!is_quantized_)
            throw std::runtime_error("Not a quantized tensor");
        return group_size_;
    }

   private:
    // Dense weights
    const Tensor<T>* tensor_ = nullptr;

    // Quantized weights
    const Tensor<int32_t>* qweight_ = nullptr;
    const Tensor<T>* scales_ = nullptr;
    const Tensor<int32_t>* qzeros_ = nullptr;
    int group_size_ = 0;

    // Whether the weights are quantized
    bool is_quantized_ = false;
};

}  // namespace op
