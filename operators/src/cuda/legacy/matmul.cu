#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>  // printf
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "cuda/legacy/legacy_cuda_api.cuh"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_splitk_parallel.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass_c_api.h"

#define WARP_SIZE 32

inline void checkCublasStatus(cublasStatus_t status, const char *file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        char errorMsg[256];
        // Note: cublasGetErrorString is not a standard function.
        // Provide a basic message.
        snprintf(errorMsg, sizeof(errorMsg), "cuBLAS error %d at %s:%d", static_cast<int>(status), file, line);
        fprintf(stderr, "%s\n", errorMsg);
        throw std::runtime_error(errorMsg);
    }
}
#define CHECK_CUBLAS(call) checkCublasStatus(call, __FILE__, __LINE__)

namespace cuda_OP {
// === Warp Reduce Sum 模板函数 ===
template <typename T, const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ T warp_reduce_sum(T val) {
#pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, mask);
    }
    return val;
}

template <typename T>
__global__ void gemv_with_bias_kernel(const T *A, const T *B, const T *bias, T *C, int M, int K, int N) {
    // A: [1, K] (行主序), B: [N, K] (列主序存储！), C: [1, N]
    // 计算: C[n] = sum(A[k] * B[n, k]) + bias[n] for k in [0, K)
    // 注意：B是列主序存储，所以 B[n, k] = B[n * K + k]
    // 每个warp负责一个输出元素
    int tx = threadIdx.x;          // 0~31
    int ty = threadIdx.y;          // 0~blockDim.y
    int bx = blockIdx.x;           // 0~(N-1)/blockDim.y
    int lane = tx % WARP_SIZE;     // 0~31
    int n = bx * blockDim.y + ty;  // 输出元素索引

    if (n < N) {
        T sum = T(0);

        // 计算需要的warp数量来覆盖K维度
        int NUM_WARPS = (K + WARP_SIZE - 1) / WARP_SIZE;

#pragma unroll
        for (int w = 0; w < NUM_WARPS; ++w) {
            int k = w * WARP_SIZE + lane;
            if (k < K) {
                // A[k] * B[n, k]，B是列主序存储，所以 B[n, k] = B[n * K + k]
                sum += A[k] * B[n * K + k];
            }
        }

        // warp内规约
        sum = warp_reduce_sum<T, WARP_SIZE>(sum);

        // 只有第一个线程写结果，并加上bias
        if (lane == 0) {
            C[n] = sum + bias[n];
        }
    }
}

template <typename T>
__global__ void gemv_kernel(const T *A, const T *B, T *C, int M, int K, int N) {
    // A: [1, K] (行主序), B: [N, K] (列主序存储！), C: [1, N]
    // 计算: C[n] = sum(A[k] * B[n, k]) for k in [0, K)
    // 注意：B是列主序存储，所以 B[n, k] = B[n * K + k]
    // 每个warp负责一个输出元素
    int tx = threadIdx.x;          // 0~31
    int ty = threadIdx.y;          // 0~blockDim.y
    int bx = blockIdx.x;           // 0~(N-1)/blockDim.y
    int lane = tx % WARP_SIZE;     // 0~31
    int n = bx * blockDim.y + ty;  // 输出元素索引

    if (n < N) {
        T sum = T(0);

        // 计算需要的warp数量来覆盖K维度
        int NUM_WARPS = (K + WARP_SIZE - 1) / WARP_SIZE;

#pragma unroll
        for (int w = 0; w < NUM_WARPS; ++w) {
            int k = w * WARP_SIZE + lane;
            if (k < K) {
                // A[k] * B[n, k]，B是列主序存储，所以 B[n, k] = B[n * K + k]
                sum += A[k] * B[n * K + k];
            }
        }

        // warp内规约
        sum = warp_reduce_sum<T, WARP_SIZE>(sum);

        // 只有第一个线程写结果
        if (lane == 0) {
            C[n] = sum;
        }
    }
}

// === 向量化版本的 GEMV kernel (适用于 K 是 4 的倍数) ===
// 计算 C = A * B^T + bias，其中 A: [1, K], B: [N, K], C: [1, N]
template <typename T>
__global__ void gemv_with_bias_vectorized_kernel(const T *A, const T *B, const T *bias, T *C, int M, int K, int N) {
    // 每个线程处理4个K元素来提高内存带宽利用率
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int lane = tx % WARP_SIZE;
    int n = bx * blockDim.y + ty;

    if (n < N) {
        T sum = T(0);

        // 每个warp处理 4*WARP_SIZE 个K元素
        int NUM_WARPS = (((K + WARP_SIZE - 1) / WARP_SIZE) + 4 - 1) / 4;

#pragma unroll
        for (int w = 0; w < NUM_WARPS; ++w) {
            int k_base = (w * WARP_SIZE + lane) * 4;
            // 实际就是4
            constexpr int VEC_UNIT = sizeof(float2) / sizeof(T);
            Vec_2<T, VEC_UNIT> va, vb;

            va.f2 = *reinterpret_cast<const float2 *>(&A[k_base]);
            vb.f2 = *reinterpret_cast<const float2 *>(&B[n * K + k_base]);
            // 向量化加载和计算
            for (int i = 0; i < VEC_UNIT; ++i) {
                sum += static_cast<float>(va.t[i]) * static_cast<float>(vb.t[i]);
            }
        }

        sum = warp_reduce_sum<float, WARP_SIZE>(sum);

        if (lane == 0) {
            C[n] = static_cast<T>(sum) + bias[n];
        }
    }
}

// 计算 C = A * B^T，其中 A: [1, K], B: [N, K], C: [1, N]
template <typename T>
__global__ void gemv_vectorized_kernel(const T *A, const T *B, T *C, int M, int K, int N) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int lane = tx % WARP_SIZE;
    int n = bx * blockDim.y + ty;

    if (n < N) {
        float sum = 0.f;
        constexpr int VEC_UNIT = sizeof(float4) / sizeof(T);
        // 每个warp处理 4*WARP_SIZE 个K元素
        int NUM_WARPS = (((K + WARP_SIZE - 1) / WARP_SIZE) + VEC_UNIT - 1) / VEC_UNIT;
#pragma unroll
        for (int w = 0; w < NUM_WARPS; ++w) {
            int k_base = (w * WARP_SIZE + lane) * VEC_UNIT;
            // 实际就是4
            Vec<T, VEC_UNIT> va, vb;

            va.f4 = *reinterpret_cast<const float4 *>(&A[k_base]);
            vb.f4 = *reinterpret_cast<const float4 *>(&B[n * K + k_base]);
            // 向量化加载和计算
            for (int i = 0; i < VEC_UNIT; ++i) {
                sum += static_cast<float>(va.t[i]) * static_cast<float>(vb.t[i]);
            }
        }

        sum = warp_reduce_sum<float, WARP_SIZE>(sum);

        if (lane == 0) {
            C[n] = static_cast<T>(sum);
        }
    }
}

template <typename T>
struct to_cutlass_type {
    using type = T;
};
template <>
struct to_cutlass_type<__nv_bfloat16> {
    using type = cutlass::bfloat16_t;  // 专门处理 bfloat16 类型
};

template <typename ElementA, typename ElementB, typename ElementOutput, typename LayoutA, typename LayoutB,
          typename LayoutOutput, typename ElementAccumulator = float,
          typename ElementComputeEpilogue = ElementAccumulator, typename MMAOp = cutlass::arch::OpClassTensorOp,
          typename SmArch = cutlass::arch::Sm80, typename ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>,
          typename ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>,
          typename ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>,
          typename SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, int NumStages = 2>
cutlass::Status run_cutlass_gemm_raw_templated(int m, int n, int k, ElementA const *d_a, ElementB const *d_b,
                                               ElementOutput const *d_bias, ElementOutput *d_d, cudaStream_t stream = 0,
                                               ElementComputeEpilogue alpha = ElementComputeEpilogue(1),
                                               int split_k_slices = 1) {
    // 1. 类型转换: 使用 to_cutlass_type 将用户类型映射为 Cutlass 支持类型
    using ElementA_t = typename to_cutlass_type<ElementA>::type;
    using ElementB_t = typename to_cutlass_type<ElementB>::type;
    using ElementOutput_t = typename to_cutlass_type<ElementOutput>::type;

    // 2. 定义 Epilogue 操作: alpha * (A*B) + bias, 不启用 Beta 缩放
    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput_t, 128 / cutlass::sizeof_bits<ElementOutput_t>::value, ElementAccumulator, ElementComputeEpilogue,
        cutlass::epilogue::thread::ScaleType::NoBetaScaling>;

    // 3. 定义 GEMM 类型: 指定所有核心模板参数
    using Gemm =
        cutlass::gemm::device::Gemm<ElementA_t, LayoutA, ElementB_t, LayoutB, ElementOutput_t, LayoutOutput,
                                    ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp,
                                    EpilogueOp, SwizzleThreadBlock, NumStages, 8, 8  // 可选的线程副本分区参数
                                    >;

    // 4. 构造问题规模
    cutlass::gemm::GemmCoord problem_size(m, n, k);

    // 5. 构造 TensorRef: 将原始指针和布局转换为 Cutlass 张量引用
    cutlass::TensorRef<ElementA_t, LayoutA> ref_A(const_cast<ElementA_t *>(reinterpret_cast<const ElementA_t *>(d_a)),
                                                  LayoutA(k)  // leading dimension = k
    );
    cutlass::TensorRef<ElementB_t, LayoutB> ref_B(const_cast<ElementB_t *>(reinterpret_cast<const ElementB_t *>(d_b)),
                                                  LayoutB(n));
    cutlass::TensorRef<ElementOutput_t, LayoutOutput> ref_D(reinterpret_cast<ElementOutput_t *>(d_d), LayoutOutput(n));

    // 6. 构造参数对象: 包含输入、输出、bias、alpha、split-K 切片等
    typename Gemm::Arguments arguments{
        problem_size,  ref_A,   ref_B, {reinterpret_cast<const ElementOutput_t *>(d_bias), 0},  // bias ptr + stride
        ref_D,         {alpha},                                                                 // epilogue 参数
        split_k_slices                                                                          // split-K 切片数
    };

    // 7. 分配内部 workspace
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    // 8. 实例化运算对象，并检查是否可实现
    Gemm gemm_op;
    cutlass::Status status = gemm_op.can_implement(arguments);
    CUTLASS_CHECK(status);

    // 9. 初始化并执行
    status = gemm_op.initialize(arguments, workspace.get());
    CUTLASS_CHECK(status);
    status = gemm_op(stream);  // 在指定的 CUDA 流上执行
    CUTLASS_CHECK(status);

    return status;
}

// SplitK工作空间管理器
class SplitKWorkspaceManager {
   private:
    void *workspace_ptr;
    size_t current_size;
    std::mutex mtx;
    static SplitKWorkspaceManager &instance() {
        static SplitKWorkspaceManager instance;
        return instance;
    }

    // 私有构造函数
    SplitKWorkspaceManager() : workspace_ptr(nullptr), current_size(0) {
    }

   public:
    // 禁止拷贝和赋值
    SplitKWorkspaceManager(const SplitKWorkspaceManager &) = delete;
    SplitKWorkspaceManager &operator=(const SplitKWorkspaceManager &) = delete;

    // 获取工作空间
    static void *get_workspace(size_t required_size) {
        return instance().get_workspace_internal(required_size);
    }

    // 清理资源
    static void cleanup() {
        instance().cleanup_internal();
    }

   private:
    void *get_workspace_internal(size_t required_size) {
        std::lock_guard<std::mutex> lock(mtx);

        if (required_size > current_size) {
            // 释放旧的
            if (workspace_ptr) {
                cudaFree(workspace_ptr);
                workspace_ptr = nullptr;
            }
            // 分配新的
            cudaMalloc(&workspace_ptr, required_size);
            current_size = required_size;
        }
        return workspace_ptr;
    }

    void cleanup_internal() {
        std::lock_guard<std::mutex> lock(mtx);
        if (workspace_ptr) {
            cudaFree(workspace_ptr);
            workspace_ptr = nullptr;
        }
        current_size = 0;
    }

    ~SplitKWorkspaceManager() {
        cleanup_internal();
    }
};

template <typename T>
cutlass::Status run_cutlass_splitk_gemm_with_bias(int m, int n, int k, const T *d_a, const T *d_b, const T *d_bias,
                                                  T *d_d, cudaStream_t stream = 0, int split_k_slices = 2) {
    // 先检查类型，如果是float就跳过CUTLASS
    if constexpr (std::is_same_v<T, float>) {
        printf("Float type not supported in CUTLASS split-K, skipping...\n");
        return cutlass::Status::kErrorNotSupported;
    }

    using ElementType = typename to_cutlass_type<T>::type;
    using ElementAccumulator = float;
    using ElementComputeEpilogue = float;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutOutput = cutlass::layout::RowMajor;

    using EpilogueOp =
        cutlass::epilogue::thread::LinearCombination<ElementType, 128 / cutlass::sizeof_bits<ElementType>::value,
                                                     ElementAccumulator, ElementComputeEpilogue>;

    using GemmSplitK = cutlass::gemm::device::GemmSplitKParallel<
        ElementType, LayoutA, ElementType, LayoutB, ElementType, LayoutOutput, ElementAccumulator,
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75, cutlass::gemm::GemmShape<16, 128, 64>,
        cutlass::gemm::GemmShape<16, 64, 32>, cutlass::gemm::GemmShape<16, 8, 16>, EpilogueOp>;

    cutlass::gemm::GemmCoord problem_size(m, n, k);

    cutlass::TensorRef<ElementType const, LayoutA> tensor_a(reinterpret_cast<const ElementType *>(d_a), LayoutA(k));
    cutlass::TensorRef<ElementType const, LayoutB> tensor_b(reinterpret_cast<const ElementType *>(d_b), LayoutB(n));
    cutlass::TensorRef<ElementType, LayoutOutput> tensor_d(reinterpret_cast<ElementType *>(d_d), LayoutOutput(n));

    ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
    ElementComputeEpilogue beta = ElementComputeEpilogue(0);

    cutlass::TensorRef<ElementType const, LayoutOutput> tensor_c;
    if (d_bias != nullptr) {
        tensor_c = cutlass::TensorRef<ElementType const, LayoutOutput>(reinterpret_cast<const ElementType *>(d_bias),
                                                                       LayoutOutput(0));
        beta = ElementComputeEpilogue(1);
    } else {
        tensor_c = cutlass::TensorRef<ElementType const, LayoutOutput>(nullptr, LayoutOutput(0));
    }

    typename GemmSplitK::Arguments arguments{problem_size, tensor_a,      tensor_b,      tensor_c,
                                             tensor_d,     {alpha, beta}, split_k_slices};

    GemmSplitK gemm_op;
    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        printf("CUTLASS GemmSplitKParallel can_implement failed: %d (M=%d, N=%d, K=%d, split_k=%d)\n",
               static_cast<int>(status), m, n, k, split_k_slices);
        return status;
    }

    // 获取所需workspace大小
    size_t workspace_size = GemmSplitK::get_workspace_size(arguments);

    // 从管理器获取workspace
    void *workspace_ptr = SplitKWorkspaceManager::get_workspace(workspace_size);

    // 初始化并执行
    status = gemm_op.initialize(arguments, workspace_ptr);
    if (status != cutlass::Status::kSuccess) {
        printf("CUTLASS GemmSplitKParallel initialize failed: %d\n", static_cast<int>(status));
        return status;
    }

    status = gemm_op(stream);
    if (status != cutlass::Status::kSuccess) {
        printf("CUTLASS GemmSplitKParallel execution failed: %d\n", static_cast<int>(status));
    }

    return status;
}

template <typename T>
__global__ void matmul_kernel(const T *A, const T *B, T *C, int M, int K, int N) {
    __shared__ T As[16][16];
    __shared__ T Bs[16][16];
    int row = blockIdx.y * 16 + threadIdx.y;
    int col = blockIdx.x * 16 + threadIdx.x;
    T sum = T(0);
    // 计算需要的 tile 数量
    int numTiles = (K + 16 - 1) / 16;
    for (int t = 0; t < numTiles; ++t) {
        int A_col = t * 16 + threadIdx.x;
        if (row < M && A_col < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + A_col];
        } else {
            As[threadIdx.y][threadIdx.x] = T(0);
        }
        int B_row = t * 16 + threadIdx.y;
        if (col < N && B_row < K) {
            Bs[threadIdx.y][threadIdx.x] = B[col * K + B_row];
        } else {
            Bs[threadIdx.y][threadIdx.x] = T(0);
        }
        __syncthreads();
        for (int k = 0; k < 16; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
        // Use code with caution.
    }
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
template <typename T>
__global__ void add_bias_kernel(T *C, const T *bias, int M, int N, int ldc) {
    // 计算当前线程负责处理的全局行索引 (row) 和列索引 (col)
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // 边界检查，确保线程在矩阵的有效范围内
    if (row < M && col < N) {
        // 计算 C 矩阵中元素的线性索引 (行主序)
        int c_index = row * ldc + col;

        // 执行加法: C[row][col] = C[row][col] + bias[col]
        C[c_index] = C[c_index] + bias[col];
    }
}
// --------------------------------------------------
// 带偏置的矩阵乘法kernel
// --------------------------------------------------
template <typename T>
__global__ void matmul_with_bias_kernel(const T *A, const T *B, const T *bias, T *C, int M, int K, int N) {
    __shared__ T As[16][16];
    __shared__ T Bs[16][16];
    int row = blockIdx.y * 16 + threadIdx.y;
    int col = blockIdx.x * 16 + threadIdx.x;
    T sum = T(0);
    // 计算需要的 tile 数量
    int numTiles = (K + 16 - 1) / 16;
    for (int t = 0; t < numTiles; ++t) {
        int A_col = t * 16 + threadIdx.x;
        if (row < M && A_col < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + A_col];
        } else {
            As[threadIdx.y][threadIdx.x] = T(0);
        }
        int B_row = t * 16 + threadIdx.y;
        if (col < N && B_row < K) {
            Bs[threadIdx.y][threadIdx.x] = B[col * K + B_row];
        } else {
            Bs[threadIdx.y][threadIdx.x] = T(0);
        }
        __syncthreads();
        for (int k = 0; k < 16; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
        // Use code with caution.
    }
    if (row < M && col < N) {
        // 在结果中加上偏置
        C[row * N + col] = sum + bias[col];
    }
}

/**
 * @brief 基于模板的 cuBLAS GEMM 包装函数，输入和输出类型相同。
 *
 * 执行操作：C = alpha * op(A) * op(B) + beta * C
 * 输入矩阵 A、B 以及输出矩阵 C 的类型都由模板参数 InputType 决定
 * (必须是 float 或 nv_bfloat16)。
 * 计算通常在内部使用 FP32 精度执行（由 compute_type
 * 控制），以获得更好的精度和性能。 结果直接写入 d_C 指向的设备内存中。
 *
 * @tparam InputType   输入矩阵 A, B 和输出矩阵 C 的数据类型 (float 或
 * nv_bfloat16)。
 * @param handle        cuBLAS 库句柄。
 * @param transa        指定操作 op(A)：CUBLAS_OP_N 或 CUBLAS_OP_T。
 * @param transb        指定操作 op(B)：CUBLAS_OP_N 或 CUBLAS_OP_T。
 * @param m             矩阵 op(A) 和矩阵 C 的行数。
 * @param n             矩阵 op(B) 和矩阵 C 的列数。
 * @param k             矩阵 op(A) 的列数和矩阵 op(B) 的行数。
 * @param alpha         用于 op(A) * op(B) 的标量乘子 (指向 const float
 * 的主机指针)。
 * @param d_A           指向设备内存中矩阵 A 的指针 (const InputType*)。
 * @param lda           矩阵 A 的主维度。
 * @param d_B           指向设备内存中矩阵 B 的指针 (const InputType*)。
 * @param ldb           矩阵 B 的主维度。
 * @param beta          用于矩阵 C 的标量乘子 (指向 const float 的主机指针)。
 * @param d_C           指向设备内存中矩阵 C 的指针
 * (InputType*)。结果将写入此处，类型与输入匹配。
 * @param ldc           矩阵 C 的主维度。
 */
template <typename InputType>
void cublas_matmul_wrapper(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
                           int k,
                           const float *alpha,  // 改回 const float*
                           const InputType *d_A, int lda, const InputType *d_B, int ldb,
                           const float *beta,  // 改回 const float*
                           InputType *d_C,     // *** 输出类型为 InputType* ***
                           int ldc) {
    cudaDataType_t cuda_data_type_A;
    cudaDataType_t cuda_data_type_B;
    cudaDataType_t cuda_data_type_C;

    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    // printf("启用 TF32 计算类型\n");

    // cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    // printf("使用 FP32 计算类型\n");

    // 根据模板类型 InputType 确定 A, B, C 的 CUDA 数据类型
    if constexpr (std::is_same_v<InputType, nv_bfloat16>) {
        cuda_data_type_A = CUDA_R_16BF;
        cuda_data_type_B = CUDA_R_16BF;
        cuda_data_type_C = CUDA_R_16BF;
        // printf("数据类型: A=BF16, B=BF16, C=BF16\n");
    } else if constexpr (std::is_same_v<InputType, float>) {
        cuda_data_type_A = CUDA_R_32F;
        cuda_data_type_B = CUDA_R_32F;
        cuda_data_type_C = CUDA_R_32F;
        // printf("数据类型: A=FP32, B=FP32, C=FP32\n");
        // 可选: Ampere+ 可考虑 TF32 计算
        // compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
    } else {
        static_assert(std::is_same_v<InputType, nv_bfloat16> || std::is_same_v<InputType, float>,
                      "cublas_matmul_wrapper 只支持 nv_bfloat16 和 float "
                      "输入/输出类型。");
        return;  // 或者抛出异常
    }

    cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;  // 让 cuBLAS 选择

    cublasStatus_t status = cublasGemmEx(handle, transa, transb, m, n, k,
                                         alpha,             // 标量 alpha (主机)
                                         d_A,               // 矩阵 A (设备)
                                         cuda_data_type_A,  // A 的类型
                                         lda,
                                         d_B,               // 矩阵 B (设备)
                                         cuda_data_type_B,  // B 的类型
                                         ldb,
                                         beta,              // 标量 beta (主机)
                                         d_C,               // 矩阵 C (设备) - 类型现在是 InputType*
                                         cuda_data_type_C,  // C 的类型现在根据 InputType 确定
                                         ldc,
                                         compute_type,  // 内部计算精度 (推荐保持 FP32)
                                         algo);

    // 使用 CHECK_CUBLAS 宏来检查返回状态
    CHECK_CUBLAS(status);
}

template <typename T>
void matmul(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> *C, cudaStream_t stream, const Tensor<T> *bias,
            int use_) {
    // 如果选择使用 cublas 计算，则调用 cublas 的包装函数接口

    const std::vector<size_t> &A_shape = A.sizes();
    const std::vector<size_t> &B_shape = B.sizes();

    // A: [M, K], B: [N, K]（保证 A 的第二维与 B 的第二维一致）
    size_t M = A_shape[0];
    size_t K = A_shape[1];
    size_t N = B_shape[1];

    if (M == 1) {
        // printf("使用 GEMV 优化分支 (M=1)\n");
        // 使用优化的GEMV kernel
        constexpr int ROWS_PER_BLOCK = 4;   // 每个block处理4个输出元素
        dim3 blockDim(32, ROWS_PER_BLOCK);  // 32线程构成一个warp，4个warp处理4个输出
        dim3 gridDim((N + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, 1);

        if (bias != nullptr) {
            // 检查偏置形状
            const std::vector<size_t> &bias_shape = bias->sizes();
            if (bias_shape.size() != 1) {
                throw std::runtime_error("Bias must be a 1D tensor");
            }
            if (bias_shape[0] != N) {
                throw std::runtime_error("Bias size must match output column dimension");
            }

            // printf("使用带bias的GEMV kernel\n");
            // 根据K的大小选择不同的优化策略
            if (K % 4 == 0 && K >= 128) {
                // K较大且是4的倍数，使用向量化版本
                gemv_with_bias_vectorized_kernel<T>
                    <<<gridDim, blockDim, 0, stream>>>(A.data_ptr(), B.data_ptr(), bias->data_ptr(), C->data_ptr(),
                                                       static_cast<int>(M), static_cast<int>(K), static_cast<int>(N));
            } else {
                // 使用标准版本
                gemv_with_bias_kernel<T><<<gridDim, blockDim, 0, stream>>>(A.data_ptr(), B.data_ptr(), bias->data_ptr(),
                                                                           C->data_ptr(), static_cast<int>(M),
                                                                           static_cast<int>(K), static_cast<int>(N));
            }
        } else {
            // printf("使用无bias的GEMV kernel\n");
            // 根据K的大小选择不同的优化策略
            if (K % 4 == 0 && K >= 128) {
                // K较大且是4的倍数，使用向量化版本
                gemv_vectorized_kernel<T><<<gridDim, blockDim, 0, stream>>>(A.data_ptr(), B.data_ptr(), C->data_ptr(),
                                                                            static_cast<int>(M), static_cast<int>(K),
                                                                            static_cast<int>(N));
            } else {
                // 使用标准版本
                gemv_kernel<T><<<gridDim, blockDim, 0, stream>>>(A.data_ptr(), B.data_ptr(), C->data_ptr(),
                                                                 static_cast<int>(M), static_cast<int>(K),
                                                                 static_cast<int>(N));
            }
        }

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA GEMV kernel launch failed: " + std::string(cudaGetErrorString(err)));
        }
        return;
    }

    if (bias == nullptr && use_ == 2) {
        use_ = 1;
    }

    if (use_ == 2) {
        //        cutlass::Status status = run_cutlass_gemm_raw_templated<T,                             // ElementA
        //                                                     T,                             // ElementB
        //                                                     T,                             // ElementOutput
        //                                                     cutlass::layout::RowMajor,     // LayoutA
        //                                                     cutlass::layout::ColumnMajor,  // LayoutB
        //                                                     cutlass::layout::RowMajor,     // LayoutOutput
        //                                                     float,                         // ElementAccumulator
        //                                                     float,                         // ElementComputeEpilogue
        //                                                     cutlass::arch::OpClassTensorOp>(
        // M, N, K, A.data_ptr(), B.data_ptr(), bias->data_ptr(), C->data_ptr(), stream);
        // cutlass::Status status = run_cutlass_splitk_gemm_with_bias<T>(
        //     static_cast<int>(M), static_cast<int>(N), static_cast<int>(K), A.data_ptr(), B.data_ptr(),
        //     bias ? bias->data_ptr() : nullptr, C->data_ptr(), stream);

        const int m = A.sizes()[0];  // !!! 请使用您 Tensor 的实际接口
        const int k = A.sizes()[1];
        const int n = B.sizes()[1];

        const void *ptr_a = A.data_ptr();  // !!! 请使用您 Tensor 的实际接口
        const void *ptr_b = B.data_ptr();
        void *ptr_d = C->data_ptr();
        const void *ptr_bias = (bias ? bias->data_ptr() : nullptr);

        my_cutlass_dtype_t dtype;
        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            dtype = my_cutlass_dtype_t::MY_CUTLASS_DTYPE_BF16;
        } else {
            throw std::runtime_error("Unsupported template type T for cuda_OP::matmul -> cutlass bridge.");
        }

        // 2. 直接调用简单、干净的 C 接口
        cutlass_gemm_c_api(m, n, k, dtype, ptr_a, ptr_b, ptr_bias, ptr_d, stream);
    } else if (use_ == 1) {
        // 这是直接CUDA算子库的实现，与统一算子库不同，它使用自己的static cublas句柄
        // 这是一个独立的实现，可以直接通过Tensor的matmul函数调用
        static cublasHandle_t handle = nullptr;
        // 使用静态标志和互斥锁确保线程安全的单次初始化
        // 虽然本质上不需要
        // 这里没有多线程操作，不会出现多线程重复初始化一个句柄
        static std::once_flag init_flag;
        static std::mutex handle_mutex;  // 保护对 handle 的并发使用 (如果需要)
        // 确保 cublasCreate 只被调用一次，且线程安全
        std::call_once(init_flag, [&]() {
            std::lock_guard<std::mutex> lock(handle_mutex);  // 锁定以进行创建
            // printf("--- Initializing static cublasHandle ---\n"); // 调试信息
            cublasStatus_t status = cublasCreate(&handle);
            if (status != CUBLAS_STATUS_SUCCESS) {
                fprintf(stderr, "FATAL ERROR: cublasCreate failed in static init: %d\n", status);
                // 在这里可能需要做更健壮的错误处理，比如抛出异常或终止程序
                handle = nullptr;  // 确保句柄无效
            } else {
                // cudaStream_t stream = nullptr; // 或获取一个全局流
                // cublasSetStream(handle, stream);
                // printf("--- Static cublasHandle initialized: %p ---\n",
                // (void*)handle); // 调试信息

                // 注册程序退出时调用的清理函数
                std::atexit([]() {
                    std::lock_guard<std::mutex> lock(handle_mutex);  // 同样锁定以进行销毁
                    if (handle != nullptr) {
                        // printf("--- Destroying static cublasHandle: %p ---\n",
                        // (void*)handle); // 调试信息
                        cublasDestroy(handle);
                        handle = nullptr;  // 标记为已销毁
                    }
                });
            }
        });

        // 检查初始化是否成功 (如果在 call_once 中处理失败)
        if (handle == nullptr) {
            fprintf(stderr, "Error: cuBLAS handle was not initialized correctly.\n");
            return;  // 或者抛出异常
        }
        // 原始数据（均按行主序存储）：
        // A: M×K, 每行有 K 个元素  -> lda = K
        // B: N×K, 每行有 K 个元素  -> ldb = K
        // C: M×N, 每行有 N 个元素  -> ldc = N
        int lda = K;  // A 每行有 K 个元素
        int ldb = K;  // B 每行有 K 个元素
        int ldc = N;  // C 每行有 N 个元素
        const float alpha = 1.0f;
        const float beta = 0.0f;
        {  // 引入作用域方便 lock_guard 管理 虽然实际上并不十分需要（本项目暂不支持多线程操作）
            std::lock_guard<std::mutex> lock(handle_mutex);
            // 将 cuBLAS 操作与传入的 stream 关联
            CHECK_CUBLAS(cublasSetStream(handle, stream));  // 确保 handle 使用正确的流

            // 目标计算： C = A * B^T
            // 利用转换： C^T = B * A^T
            // GEMM 调用参数：
            //    m = N, n = M, k = K
            // 同时对 A 和 B 使用转置操作：
            cublas_matmul_wrapper<T>(handle, CUBLAS_OP_T, CUBLAS_OP_N, int(N), int(M), int(K), &alpha,
                                     B.data_ptr(),  // 原始 B
                                     ldb,           // ldb = K
                                     A.data_ptr(),  // 原始 A
                                     lda,           // lda = K
                                     &beta,
                                     C->data_ptr(),  // 输出 C
                                     ldc);           // ldc = N
            // cudaStreamSynchronize(stream);
        }
        // std::cout << "cublas_matmul_wrapper<T> 调用成功" << std::endl;
        if (bias != nullptr) {
            dim3 blockDim(16, 16);
            // 计算网格大小，确保覆盖所有元素
            dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);
            add_bias_kernel<T><<<gridDim, blockDim, 0, stream>>>(C->data_ptr(), bias->data_ptr(), static_cast<int>(M),
                                                                 static_cast<int>(N), ldc);
        }
        // std::cout << "add_bias_kernel 调用成功" << std::endl;
        return;
    }

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    if (bias == nullptr) {
        // 使用无偏置版本的kernel
        matmul_kernel<T><<<numBlocks, threadsPerBlock, 0, stream>>>(A.data_ptr(), B.data_ptr(), C->data_ptr(), M, K, N);
    } else {
        // 检查偏置形状
        const std::vector<size_t> &bias_shape = bias->sizes();
        if (bias_shape.size() != 1) {
            throw std::runtime_error("Bias must be a 1D tensor");
        }
        if (bias_shape[0] != N) {
            throw std::runtime_error("Bias size must match output column dimension");
        }
        // 使用带偏置版本的kernel
        matmul_with_bias_kernel<T><<<numBlocks, threadsPerBlock, 0, stream>>>(A.data_ptr(), B.data_ptr(),
                                                                              bias->data_ptr(), C->data_ptr(), M, K, N);
        // Use code with caution.
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }

    return;
}

template void matmul<float>(const Tensor<float> &, const Tensor<float> &, Tensor<float> *, cudaStream_t,
                            const Tensor<float> *, int);
template void matmul<__nv_bfloat16>(const Tensor<__nv_bfloat16> &, const Tensor<__nv_bfloat16> &,
                                    Tensor<__nv_bfloat16> *, cudaStream_t, const Tensor<__nv_bfloat16> *, int);

}  // namespace cuda_OP
