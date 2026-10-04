#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/device_memory.h>
#include <cutlass/util/host_tensor.h>

#include <cstdio>
#include <stdexcept>
#include <string>

#include "cutlass_c_api.h"


#define CUTLASS_CHECK(status)                                                                                 \
    {                                                                                                         \
        cutlass::Status error = status;                                                                       \
        if (error != cutlass::Status::kSuccess) {                                                             \
            throw std::runtime_error("CUTLASS operation failed: " + std::to_string(static_cast<int>(error))); \
        }                                                                                                     \
    }

//---------------------------------------------------------

//---------------------------------------------------------
using ElementA = cutlass::bfloat16_t;
using ElementB = cutlass::bfloat16_t;
using ElementC = cutlass::bfloat16_t;
using ElementAccumulator = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;


using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementC, 1, ElementAccumulator, ElementAccumulator>;


using GemmKernel = cutlass::gemm::device::GemmUniversal<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2>;                                                            // Pipeline stages

//---------------------------------------------------------

//---------------------------------------------------------
static cutlass::Status run_bf16_gemm(int m, int n, int k, void const* d_A, void const* d_B, void const* d_C, void* d_D,
                                     cudaStream_t stream) {

    cutlass::gemm::GemmCoord problem_size(m, n, k);


    ElementAccumulator alpha = ElementAccumulator(1.0f);
    ElementAccumulator beta =
        d_C ? ElementAccumulator(1.0f) : ElementAccumulator(0.0f);


    typename GemmKernel::Arguments arguments{

        cutlass::gemm::GemmUniversalMode::kGemm,
        problem_size,
        1,
        {alpha, beta},


        static_cast<ElementA const*>(d_A),
        static_cast<ElementB const*>(d_B),
        static_cast<ElementC const*>(d_C),
        static_cast<ElementC*>(d_D),


        problem_size.mk().product(),
        problem_size.nk().product(),
        problem_size.mn().product(),
        problem_size.mn().product(),


        k,
        k,
        n,
        n
    };


    GemmKernel gemm_op;


    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        return status;
    }


    size_t workspace_size = GemmKernel::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);


    status = gemm_op.initialize(arguments, workspace.get(), stream);
    if (status != cutlass::Status::kSuccess) {
        return status;
    }


    return gemm_op.run(stream);
}

//---------------------------------------------------------

//---------------------------------------------------------
extern "C" void cutlass_gemm_c_api(int m, int n, int k, my_cutlass_dtype_t dtype, void const* ptr_a, void const* ptr_b,
                                   void const* ptr_bias, void* ptr_d, cudaStream_t stream) {
    try {

        if (dtype != MY_CUTLASS_DTYPE_BF16) {
            std::fprintf(stderr, "[cutlass_gemm_c_api] Only BF16 supported\n");
            return;
        }


        cutlass::Status status = run_bf16_gemm(m, n, k, ptr_a, ptr_b, ptr_bias, ptr_d, stream);
        if (status != cutlass::Status::kSuccess) {
            std::fprintf(stderr, "[cutlass_gemm_c_api] CUTLASS failed: %d\n", static_cast<int>(status));
        }
    } catch (std::exception const& e) {
        std::fprintf(stderr, "[cutlass_gemm_c_api] Exception: %s\n", e.what());
    }
}

extern "C" const char* cutlass_status_to_string(my_cutlass_status_t st) {
    switch (st) {
        case MY_CUTLASS_STATUS_SUCCESS:
            return "Success";
        case MY_CUTLASS_STATUS_ERROR_INVALID_PROBLEM:
            return "Invalid Problem";
        case MY_CUTLASS_STATUS_ERROR_NOT_SUPPORTED:
            return "Not Supported";
        default:
            return "Internal Error";
    }
}
