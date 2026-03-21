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

#include "cutlass_c_api.h"  // ← 你的枚举 / 错误码声明保持不变

// 定义CUTLASS_CHECK宏
#define CUTLASS_CHECK(status)                                                                                 \
    {                                                                                                         \
        cutlass::Status error = status;                                                                       \
        if (error != cutlass::Status::kSuccess) {                                                             \
            throw std::runtime_error("CUTLASS operation failed: " + std::to_string(static_cast<int>(error))); \
        }                                                                                                     \
    }

//---------------------------------------------------------
// CUTLASS类型定义 - 专门针对BF16的GEMM配置
//---------------------------------------------------------
using ElementA = cutlass::bfloat16_t;  // 矩阵A的元素类型：BF16
using ElementB = cutlass::bfloat16_t;  // 矩阵B的元素类型：BF16
using ElementC = cutlass::bfloat16_t;  // 矩阵C/D的元素类型：BF16
using ElementAccumulator = float;      // 累加器类型：FP32（更高精度）

using LayoutA = cutlass::layout::RowMajor;     // 矩阵A布局：行主序
using LayoutB = cutlass::layout::ColumnMajor;  // 矩阵B布局：列主序（为了更好的内存访问模式）
using LayoutC = cutlass::layout::RowMajor;     // 矩阵C/D布局：行主序

// Epilogue操作：线性组合 alpha * (A*B) + beta * C
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementC, 1, ElementAccumulator, ElementAccumulator>;

// GemmUniversal kernel配置
using GemmKernel = cutlass::gemm::device::GemmUniversal<
    ElementA, LayoutA,                                             // 矩阵A类型和布局
    ElementB, LayoutB,                                             // 矩阵B类型和布局
    ElementC, LayoutC,                                             // 矩阵C/D类型和布局
    ElementAccumulator,                                            // 累加器类型
    cutlass::arch::OpClassTensorOp,                                // 使用Tensor Core操作
    cutlass::arch::Sm80,                                           // 目标架构：Ampere SM80
    cutlass::gemm::GemmShape<128, 128, 32>,                        // Threadblock tile形状：M=128, N=128, K=32
    cutlass::gemm::GemmShape<64, 64, 32>,                          // Warp tile形状：M=64, N=64, K=32
    cutlass::gemm::GemmShape<16, 8, 16>,                           // Instruction形状：M=16, N=8, K=16 (Tensor Core)
    EpilogueOp,                                                    // Epilogue操作
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,  // 线程块调度策略
    2>;                                                            // Pipeline stages

//---------------------------------------------------------
// 核心GEMM实现函数
//---------------------------------------------------------
static cutlass::Status run_bf16_gemm(int m, int n, int k, void const* d_A, void const* d_B, void const* d_C, void* d_D,
                                     cudaStream_t stream) {
    // 构造GEMM问题规模：M x N x K
    cutlass::gemm::GemmCoord problem_size(m, n, k);

    // 设置epilogue参数
    ElementAccumulator alpha = ElementAccumulator(1.0f);  // 缩放因子α=1.0
    ElementAccumulator beta =
        d_C ? ElementAccumulator(1.0f) : ElementAccumulator(0.0f);  // 如果有bias则β=1.0，否则β=0.0

    /***********************************************************

     * - 使用16参数的完整格式
     * - 显式提供所有必需的参数
     ***********************************************************/
    typename GemmKernel::Arguments arguments{
        // === 必需参数（前4个）===
        cutlass::gemm::GemmUniversalMode::kGemm,  // 1. 运行模式
        problem_size,                             // 2. 问题规模：M x N x K
        1,                                        // 3. 批次数量：1（单批次）
        {alpha, beta},                            // 4. Epilogue参数：{α, β}

        // === 数据指针（4个）===
        static_cast<ElementA const*>(d_A),  // 5. 矩阵A指针
        static_cast<ElementB const*>(d_B),  // 6. 矩阵B指针
        static_cast<ElementC const*>(d_C),  // 7. 矩阵C指针（bias，可为nullptr）
        static_cast<ElementC*>(d_D),        // 8. 矩阵D指针（输出）

        // === 批次步长（4个）- 单批次时全为0 ===
        problem_size.mk().product(),  // 9. batch_stride_A：A矩阵批次间步长（单批次时实际不使用）
        problem_size.nk().product(),  // 10. batch_stride_B：B矩阵批次间步长
        problem_size.mn().product(),  // 11. batch_stride_C：C矩阵批次间步长
        problem_size.mn().product(),  // 12. batch_stride_D：D矩阵批次间步长

        // === 内存步长/Leading Dimension（4个）===
        k,  // 13. stride_a：A矩阵的leading dimension（行主序，每行K个元素）
        k,  // 14. stride_b：B矩阵的leading dimension（列主序，每列K个元素）
        n,  // 15. stride_c：C矩阵的leading dimension（行主序，每行N个元素）
        n   // 16. stride_d：D矩阵的leading dimension（行主序，每行N个元素）
    };

    // 执行GEMM计算
    GemmKernel gemm_op;

    // 1. 检查是否可以实现这个配置
    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        return status;  // 返回错误状态而不抛出异常
    }

    // 2. 获取并分配工作空间内存
    size_t workspace_size = GemmKernel::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    // 3. 初始化kernel
    status = gemm_op.initialize(arguments, workspace.get(), stream);
    if (status != cutlass::Status::kSuccess) {
        return status;
    }

    // 4. 运行kernel
    return gemm_op.run(stream);
}

//---------------------------------------------------------
// C-API 封装层 - 完全隔离CUTLASS类型与用户Tensor类型
//---------------------------------------------------------
extern "C" void cutlass_gemm_c_api(int m, int n, int k, my_cutlass_dtype_t dtype, void const* ptr_a, void const* ptr_b,
                                   void const* ptr_bias, void* ptr_d, cudaStream_t stream) {
    try {
        // 类型检查：当前只支持BF16
        if (dtype != MY_CUTLASS_DTYPE_BF16) {
            std::fprintf(stderr, "[cutlass_gemm_c_api] Only BF16 supported\n");
            return;
        }

        // 调用实际的GEMM实现
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
