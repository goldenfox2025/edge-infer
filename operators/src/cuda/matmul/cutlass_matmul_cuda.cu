#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <stdexcept>

#include "operators/cuda/matmul/cutlass_matmul_cuda.cuh"

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

namespace op {

inline void checkCutlassStatus(cutlass::Status status, const char* file,
                               int line) {
    if (status != cutlass::Status::kSuccess) {
        char errorMsg[256];
        snprintf(errorMsg, sizeof(errorMsg), "CUTLASS error %d at %s:%d",
                 static_cast<int>(status), file, line);
        throw std::runtime_error(errorMsg);
    }
}

#define CHECK_CUTLASS(call) checkCutlassStatus((call), __FILE__, __LINE__)

template <typename T>
struct to_cutlass_type {
    using type = T;
};

template <>
struct to_cutlass_type<__nv_bfloat16> {
    using type = cutlass::bfloat16_t;
};

template <typename ElementA, typename ElementB, typename ElementOutput, typename LayoutA = cutlass::layout::RowMajor,
          typename LayoutB = cutlass::layout::ColumnMajor, typename LayoutOutput = cutlass::layout::RowMajor,
          typename ElementAccumulator = float, typename ElementComputeEpilogue = ElementAccumulator>
cutlass::Status run_cutlass_gemm(int m, int n, int k, ElementA const *d_a, ElementB const *d_b,
                                 ElementOutput const *d_bias, ElementOutput *d_d, cudaStream_t stream = 0,
                                 ElementComputeEpilogue alpha = ElementComputeEpilogue(1), int split_k_slices = 1) {

    using ElementA_t = typename to_cutlass_type<ElementA>::type;
    using ElementB_t = typename to_cutlass_type<ElementB>::type;
    using ElementOutput_t = typename to_cutlass_type<ElementOutput>::type;

    // Use Tensor Core MMA operations with float accumulation.
    using MMAOp = cutlass::arch::OpClassTensorOp;
    using SmArch = cutlass::arch::Sm80;  // This kernel configuration targets SM80.

    // Threadblock, warp, and instruction tile shapes.
    using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;
    using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;
    using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;

    using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

    int const NumStages = 2;

    // Apply the linear-combination epilogue with a broadcast bias.
    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput_t, 128 / cutlass::sizeof_bits<ElementOutput_t>::value, ElementAccumulator, ElementComputeEpilogue,
        cutlass::epilogue::thread::ScaleType::NoBetaScaling>;

    using Gemm = cutlass::gemm::device::Gemm<ElementA_t, LayoutA, ElementB_t, LayoutB, ElementOutput_t, LayoutOutput,
                                             ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp,
                                             ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages, 8, 8>;

    cutlass::gemm::GemmCoord problem_size(m, n, k);

    cutlass::TensorRef<ElementA_t, LayoutA> ref_A(const_cast<ElementA_t *>(reinterpret_cast<const ElementA_t *>(d_a)),
                                                  LayoutA(k));
    cutlass::TensorRef<ElementB_t, LayoutB> ref_B(const_cast<ElementB_t *>(reinterpret_cast<const ElementB_t *>(d_b)),
                                                  LayoutB(n));
    cutlass::TensorRef<ElementOutput_t, LayoutOutput> ref_D(reinterpret_cast<ElementOutput_t *>(d_d), LayoutOutput(n));

    typename Gemm::Arguments arguments{
        problem_size, ref_A,   ref_B,         {reinterpret_cast<const ElementOutput_t *>(d_bias), 0},
        ref_D,        {alpha}, split_k_slices};

    // Allocate the workspace required by this CUTLASS configuration.
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    Gemm gemm_op;
    cutlass::Status status = gemm_op.can_implement(arguments);
    CHECK_CUTLASS(status);

    status = gemm_op.initialize(arguments, workspace.get());
    CHECK_CUTLASS(status);

    status = gemm_op(stream);
    CHECK_CUTLASS(status);

    return status;
}

template <typename T>
void CutlassMatmulCUDAOperator<T>::operator()(Tensor<T> *output, Tensor<T> *input, const WeightTensor<T> &weight,
                                              const Tensor<T> *bias, cudaStream_t stream) {

    if (weight.is_quantized()) {
        throw std::runtime_error("CUTLASS MatMul does not support quantized weights");
    }

    const std::vector<size_t> &A_shape = input->sizes();
    const std::vector<size_t> &B_shape = weight.tensor()->sizes();

    // Row-major inputs: A [M, K], B [N, K], producing C = A * B^T.
    size_t M = A_shape[0];
    size_t K = A_shape[1];
    size_t N = B_shape[0];

    cutlass::Status status =
        run_cutlass_gemm<T, T, T>(M, N, K, input->data_ptr(), weight.tensor()->data_ptr(),
                                  bias != nullptr ? bias->data_ptr() : nullptr, output->data_ptr(), stream);

    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS matrix multiplication failed");
    }
}

template class CutlassMatmulCUDAOperator<float>;
template class CutlassMatmulCUDAOperator<__nv_bfloat16>;

}  // namespace op

#undef CHECK_CUTLASS
