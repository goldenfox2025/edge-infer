#pragma once

#include "operators/cuda/execution.hpp"

namespace op::cuda::detail {

const void* prepare_attention_padding();
template <typename T>
void launch_awq(const ExecutionContext&, TensorView<const T, 2>,
                const AwqLinearWeight<T>&, TensorView<T, 2>);
template <typename T>
void launch_decode_128(const ExecutionContext&, TensorView<const T, 3>,
                       TensorView<const T, 3>, TensorView<const T, 3>,
                       TensorView<T, 3>, TensorView<float, 1>);
template <typename T>
void launch_attention_gather(const ExecutionContext&, const float*, const float*,
                             const float*, const float*, const float*, int,
                             TensorView<T, 3>);
void launch_prefill_128(const ExecutionContext&, TensorView<const __nv_bfloat16, 3>,
                        TensorView<const __nv_bfloat16, 3>,
                        TensorView<const __nv_bfloat16, 3>,
                        TensorView<__nv_bfloat16, 3>, std::size_t);
template <typename T>
void launch_graph_128(const ExecutionContext&, TensorView<const T, 3>,
                      TensorView<const T, 3>, TensorView<const T, 3>,
                      TensorView<T, 3>, float**, int*, int*);
template <typename T>
void launch_graph_gather(const ExecutionContext&, float**, TensorView<T, 3>, int*);

}  // namespace op::cuda::detail
