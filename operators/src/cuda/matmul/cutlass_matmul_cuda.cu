#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <stdexcept>

#include "operators/cuda/matmul/cutlass_matmul_cuda.cuh"

// CUTLASS相关头文件
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

// 定义类型转换 traits
template <typename T>
struct to_cutlass_type {
    using type = T;
};

template <>
struct to_cutlass_type<__nv_bfloat16> {
    using type = cutlass::bfloat16_t;
};

// CUTLASS GEMM实现函数
template <typename ElementA, typename ElementB, typename ElementOutput, typename LayoutA = cutlass::layout::RowMajor,
          typename LayoutB = cutlass::layout::ColumnMajor, typename LayoutOutput = cutlass::layout::RowMajor,
          typename ElementAccumulator = float, typename ElementComputeEpilogue = ElementAccumulator>
cutlass::Status run_cutlass_gemm(int m, int n, int k, ElementA const *d_a, ElementB const *d_b,
                                 ElementOutput const *d_bias, ElementOutput *d_d, cudaStream_t stream = 0,
                                 ElementComputeEpilogue alpha = ElementComputeEpilogue(1), int split_k_slices = 1) {
    // 使用 to_cutlass_type 对输入数据类型做转换
    using ElementA_t = typename to_cutlass_type<ElementA>::type;
    using ElementB_t = typename to_cutlass_type<ElementB>::type;
    using ElementOutput_t = typename to_cutlass_type<ElementOutput>::type;

    // 使用TensorOp作为MMA操作类型
    using MMAOp = cutlass::arch::OpClassTensorOp;
    using SmArch = cutlass::arch::Sm80;  // 可以根据实际GPU架构调整

    // 定义线程块和warp的形状
    using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;
    using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;
    using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;

    // 使用标准swizzle
    using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

    // 定义流水线阶段数
    int const NumStages = 2;

    // 定义 epilogue 操作
    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput_t, 128 / cutlass::sizeof_bits<ElementOutput_t>::value, ElementAccumulator, ElementComputeEpilogue,
        cutlass::epilogue::thread::ScaleType::NoBetaScaling>;

    // 定义 GEMM 操作类型
    using Gemm = cutlass::gemm::device::Gemm<ElementA_t, LayoutA, ElementB_t, LayoutB, ElementOutput_t, LayoutOutput,
                                             ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp,
                                             ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages, 8, 8>;

    // 构造问题尺寸
    cutlass::gemm::GemmCoord problem_size(m, n, k);

    // 创建TensorRef对象
    cutlass::TensorRef<ElementA_t, LayoutA> ref_A(const_cast<ElementA_t *>(reinterpret_cast<const ElementA_t *>(d_a)),
                                                  LayoutA(k));
    cutlass::TensorRef<ElementB_t, LayoutB> ref_B(const_cast<ElementB_t *>(reinterpret_cast<const ElementB_t *>(d_b)),
                                                  LayoutB(n));
    cutlass::TensorRef<ElementOutput_t, LayoutOutput> ref_D(reinterpret_cast<ElementOutput_t *>(d_d), LayoutOutput(n));

    // 构造 Gemm kernel 参数
    typename Gemm::Arguments arguments{
        problem_size, ref_A,   ref_B,         {reinterpret_cast<const ElementOutput_t *>(d_bias), 0},
        ref_D,        {alpha}, split_k_slices};

    // 查询 workspace 内存大小并分配
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    // 实例化 GEMM 对象，并检查问题是否可实现
    Gemm gemm_op;
    cutlass::Status status = gemm_op.can_implement(arguments);
    CHECK_CUTLASS(status);

    // 初始化 GEMM 操作
    status = gemm_op.initialize(arguments, workspace.get());
    CHECK_CUTLASS(status);

    // 调用 CUTLASS GEMM 内核
    status = gemm_op(stream);
    CHECK_CUTLASS(status);

    return status;
}

// 实现CutlassMatmulCUDAOperator的operator()方法
template <typename T>
void CutlassMatmulCUDAOperator<T>::operator()(Tensor<T> *output, Tensor<T> *input, const WeightTensor<T> &weight,
                                              const Tensor<T> *bias, cudaStream_t stream) {
    // 确保权重不是量化的
    if (weight.is_quantized()) {
        throw std::runtime_error("CUTLASS MatMul does not support quantized weights");
    }

    // 获取输入尺寸
    const std::vector<size_t> &A_shape = input->sizes();
    const std::vector<size_t> &B_shape = weight.tensor()->sizes();

    // A: [M, K], B: [N, K]（保证 A 的第二维与 B 的第二维一致）
    size_t M = A_shape[0];
    size_t K = A_shape[1];
    size_t N = B_shape[0];

    // 调用CUTLASS的GEMM实现
    cutlass::Status status =
        run_cutlass_gemm<T, T, T>(M, N, K, input->data_ptr(), weight.tensor()->data_ptr(),
                                  bias != nullptr ? bias->data_ptr() : nullptr, output->data_ptr(), stream);

    // 检查CUTLASS操作状态
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS matrix multiplication failed");
    }
}

// 显式模板实例化
template class CutlassMatmulCUDAOperator<float>;
template class CutlassMatmulCUDAOperator<__nv_bfloat16>;

}  // namespace op

#undef CHECK_CUTLASS
