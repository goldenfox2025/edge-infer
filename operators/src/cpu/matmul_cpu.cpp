#include "operators/cpu/matmul_cpu.hpp"

#include <cuda_bf16.h>

#include <cmath>
#include <stdexcept>

namespace op {

template <typename T>
void MatmulCPUOperator<T>::operator()(Tensor<T>* output, Tensor<T>* input, const WeightTensor<T>& weight,
                                      const Tensor<T>* bias, cudaStream_t stream) {

    if (weight.is_quantized()) {
        throw std::runtime_error("CPU MatMul does not support quantized weights");
    }

    // These compatibility adapters do not support BF16 on the CPU.
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        throw std::runtime_error("MatMul operator for __nv_bfloat16 not supported on CPU platform");
    } else {
        const auto& input_shape = input->sizes();
        const auto& weight_shape = weight.tensor()->sizes();
        const auto& input_strides = input->strides();
        const auto& weight_strides = weight.tensor()->strides();
        const auto& output_strides = output->strides();

        if (input_shape.size() != 2 || weight_shape.size() != 2 || output->sizes().size() != 2) {
            throw std::runtime_error("CPU MatMul currently expects rank-2 tensors");
        }

        const size_t M = input_shape[0];
        const size_t K = input_shape[1];

        bool weight_is_kn = false;
        size_t N = 0;

        if (weight_shape[0] == K &&
            (weight_shape[1] != K || weight_strides[0] < weight_strides[1])) {
            weight_is_kn = true;
            N = weight_shape[1];
        } else if (weight_shape[1] == K) {
            N = weight_shape[0];
        } else {
            throw std::runtime_error("Input and weight dimensions mismatch for MatMul");
        }

        if (output->sizes()[0] != M || output->sizes()[1] != N) {
            throw std::runtime_error("Output tensor shape mismatch for MatMul");
        }

        for (size_t m = 0; m < M; ++m) {
            for (size_t n = 0; n < N; ++n) {
                float sum = 0.0f;

                for (size_t k = 0; k < K; ++k) {
                    const size_t input_idx = m * input_strides[0] + k * input_strides[1];
                    const size_t weight_idx =
                        weight_is_kn ? (k * weight_strides[0] + n * weight_strides[1])
                                     : (n * weight_strides[0] + k * weight_strides[1]);
                    sum += static_cast<float>(input->data_ptr()[input_idx]) *
                           static_cast<float>(weight.tensor()->data_ptr()[weight_idx]);
                }

                if (bias) {
                    sum += static_cast<float>(bias->data_ptr()[n]);
                }

                output->data_ptr()[m * output_strides[0] + n * output_strides[1]] =
                    static_cast<T>(sum);
            }
        }
    }
}

template class MatmulCPUOperator<float>;
template class MatmulCPUOperator<__nv_bfloat16>;

}  // namespace op
