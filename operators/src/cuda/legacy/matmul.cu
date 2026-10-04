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


    int tx = threadIdx.x;          // 0~31
    int ty = threadIdx.y;          // 0~blockDim.y
    int bx = blockIdx.x;           // 0~(N-1)/blockDim.y
    int lane = tx % WARP_SIZE;     // 0~31
    int n = bx * blockDim.y + ty;

    if (n < N) {
        T sum = T(0);


        int NUM_WARPS = (K + WARP_SIZE - 1) / WARP_SIZE;

#pragma unroll
        for (int w = 0; w < NUM_WARPS; ++w) {
            int k = w * WARP_SIZE + lane;
            if (k < K) {

                sum += A[k] * B[n * K + k];
            }
        }


        sum = warp_reduce_sum<T, WARP_SIZE>(sum);


        if (lane == 0) {
            C[n] = sum + bias[n];
        }
    }
}

template <typename T>
__global__ void gemv_kernel(const T *A, const T *B, T *C, int M, int K, int N) {


    int tx = threadIdx.x;          // 0~31
    int ty = threadIdx.y;          // 0~blockDim.y
    int bx = blockIdx.x;           // 0~(N-1)/blockDim.y
    int lane = tx % WARP_SIZE;     // 0~31
    int n = bx * blockDim.y + ty;

    if (n < N) {
        T sum = T(0);


        int NUM_WARPS = (K + WARP_SIZE - 1) / WARP_SIZE;

#pragma unroll
        for (int w = 0; w < NUM_WARPS; ++w) {
            int k = w * WARP_SIZE + lane;
            if (k < K) {

                sum += A[k] * B[n * K + k];
            }
        }


        sum = warp_reduce_sum<T, WARP_SIZE>(sum);


        if (lane == 0) {
            C[n] = sum;
        }
    }
}


template <typename T>
__global__ void gemv_with_bias_vectorized_kernel(const T *A, const T *B, const T *bias, T *C, int M, int K, int N) {

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int lane = tx % WARP_SIZE;
    int n = bx * blockDim.y + ty;

    if (n < N) {
        T sum = T(0);


        int NUM_WARPS = (((K + WARP_SIZE - 1) / WARP_SIZE) + 4 - 1) / 4;

#pragma unroll
        for (int w = 0; w < NUM_WARPS; ++w) {
            int k_base = (w * WARP_SIZE + lane) * 4;

            constexpr int VEC_UNIT = sizeof(float2) / sizeof(T);
            Vec_2<T, VEC_UNIT> va, vb;

            va.f2 = *reinterpret_cast<const float2 *>(&A[k_base]);
            vb.f2 = *reinterpret_cast<const float2 *>(&B[n * K + k_base]);

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

        int NUM_WARPS = (((K + WARP_SIZE - 1) / WARP_SIZE) + VEC_UNIT - 1) / VEC_UNIT;
#pragma unroll
        for (int w = 0; w < NUM_WARPS; ++w) {
            int k_base = (w * WARP_SIZE + lane) * VEC_UNIT;

            Vec<T, VEC_UNIT> va, vb;

            va.f4 = *reinterpret_cast<const float4 *>(&A[k_base]);
            vb.f4 = *reinterpret_cast<const float4 *>(&B[n * K + k_base]);

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
    using type = cutlass::bfloat16_t;
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

    using ElementA_t = typename to_cutlass_type<ElementA>::type;
    using ElementB_t = typename to_cutlass_type<ElementB>::type;
    using ElementOutput_t = typename to_cutlass_type<ElementOutput>::type;


    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput_t, 128 / cutlass::sizeof_bits<ElementOutput_t>::value, ElementAccumulator, ElementComputeEpilogue,
        cutlass::epilogue::thread::ScaleType::NoBetaScaling>;


    using Gemm =
        cutlass::gemm::device::Gemm<ElementA_t, LayoutA, ElementB_t, LayoutB, ElementOutput_t, LayoutOutput,
                                    ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp,
                                    EpilogueOp, SwizzleThreadBlock, NumStages, 8, 8
                                    >;


    cutlass::gemm::GemmCoord problem_size(m, n, k);


    cutlass::TensorRef<ElementA_t, LayoutA> ref_A(const_cast<ElementA_t *>(reinterpret_cast<const ElementA_t *>(d_a)),
                                                  LayoutA(k)  // leading dimension = k
    );
    cutlass::TensorRef<ElementB_t, LayoutB> ref_B(const_cast<ElementB_t *>(reinterpret_cast<const ElementB_t *>(d_b)),
                                                  LayoutB(n));
    cutlass::TensorRef<ElementOutput_t, LayoutOutput> ref_D(reinterpret_cast<ElementOutput_t *>(d_d), LayoutOutput(n));


    typename Gemm::Arguments arguments{
        problem_size,  ref_A,   ref_B, {reinterpret_cast<const ElementOutput_t *>(d_bias), 0},  // bias ptr + stride
        ref_D,         {alpha},
        split_k_slices
    };


    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);


    Gemm gemm_op;
    cutlass::Status status = gemm_op.can_implement(arguments);
    CUTLASS_CHECK(status);


    status = gemm_op.initialize(arguments, workspace.get());
    CUTLASS_CHECK(status);
    status = gemm_op(stream);
    CUTLASS_CHECK(status);

    return status;
}


class SplitKWorkspaceManager {
   private:
    void *workspace_ptr;
    size_t current_size;
    std::mutex mtx;
    static SplitKWorkspaceManager &instance() {
        static SplitKWorkspaceManager instance;
        return instance;
    }


    SplitKWorkspaceManager() : workspace_ptr(nullptr), current_size(0) {
    }

   public:

    SplitKWorkspaceManager(const SplitKWorkspaceManager &) = delete;
    SplitKWorkspaceManager &operator=(const SplitKWorkspaceManager &) = delete;


    static void *get_workspace(size_t required_size) {
        return instance().get_workspace_internal(required_size);
    }


    static void cleanup() {
        instance().cleanup_internal();
    }

   private:
    void *get_workspace_internal(size_t required_size) {
        std::lock_guard<std::mutex> lock(mtx);

        if (required_size > current_size) {

            if (workspace_ptr) {
                cudaFree(workspace_ptr);
                workspace_ptr = nullptr;
            }

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


    size_t workspace_size = GemmSplitK::get_workspace_size(arguments);


    void *workspace_ptr = SplitKWorkspaceManager::get_workspace(workspace_size);


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

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;


    if (row < M && col < N) {

        int c_index = row * ldc + col;


        C[c_index] = C[c_index] + bias[col];
    }
}
// --------------------------------------------------

// --------------------------------------------------
template <typename T>
__global__ void matmul_with_bias_kernel(const T *A, const T *B, const T *bias, T *C, int M, int K, int N) {
    __shared__ T As[16][16];
    __shared__ T Bs[16][16];
    int row = blockIdx.y * 16 + threadIdx.y;
    int col = blockIdx.x * 16 + threadIdx.x;
    T sum = T(0);

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

        C[row * N + col] = sum + bias[col];
    }
}


template <typename InputType>
void cublas_matmul_wrapper(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
                           int k,
                           const float *alpha,
                           const InputType *d_A, int lda, const InputType *d_B, int ldb,
                           const float *beta,
                           InputType *d_C,
                           int ldc) {
    cudaDataType_t cuda_data_type_A;
    cudaDataType_t cuda_data_type_B;
    cudaDataType_t cuda_data_type_C;

    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;


    // cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;


    if constexpr (std::is_same_v<InputType, nv_bfloat16>) {
        cuda_data_type_A = CUDA_R_16BF;
        cuda_data_type_B = CUDA_R_16BF;
        cuda_data_type_C = CUDA_R_16BF;

    } else if constexpr (std::is_same_v<InputType, float>) {
        cuda_data_type_A = CUDA_R_32F;
        cuda_data_type_B = CUDA_R_32F;
        cuda_data_type_C = CUDA_R_32F;


        // compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
    } else {
        static_assert(std::is_same_v<InputType, nv_bfloat16> || std::is_same_v<InputType, float>,
                      "cublas_matmul_wrapper supports only nv_bfloat16 and float "
                      "input/output types");
        return;
    }

    cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;

    cublasStatus_t status = cublasGemmEx(handle, transa, transb, m, n, k,
                                         alpha,
                                         d_A,
                                         cuda_data_type_A,
                                         lda,
                                         d_B,
                                         cuda_data_type_B,
                                         ldb,
                                         beta,
                                         d_C,
                                         cuda_data_type_C,
                                         ldc,
                                         compute_type,
                                         algo);


    CHECK_CUBLAS(status);
}

template <typename T>
void matmul(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> *C, cudaStream_t stream, const Tensor<T> *bias,
            int use_) {


    const std::vector<size_t> &A_shape = A.sizes();
    const std::vector<size_t> &B_shape = B.sizes();


    size_t M = A_shape[0];
    size_t K = A_shape[1];
    size_t N = B_shape[1];

    if (M == 1) {


        constexpr int ROWS_PER_BLOCK = 4;
        dim3 blockDim(32, ROWS_PER_BLOCK);
        dim3 gridDim((N + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, 1);

        if (bias != nullptr) {

            const std::vector<size_t> &bias_shape = bias->sizes();
            if (bias_shape.size() != 1) {
                throw std::runtime_error("Bias must be a 1D tensor");
            }
            if (bias_shape[0] != N) {
                throw std::runtime_error("Bias size must match output column dimension");
            }


            if (K % 4 == 0 && K >= 128) {

                gemv_with_bias_vectorized_kernel<T>
                    <<<gridDim, blockDim, 0, stream>>>(A.data_ptr(), B.data_ptr(), bias->data_ptr(), C->data_ptr(),
                                                       static_cast<int>(M), static_cast<int>(K), static_cast<int>(N));
            } else {

                gemv_with_bias_kernel<T><<<gridDim, blockDim, 0, stream>>>(A.data_ptr(), B.data_ptr(), bias->data_ptr(),
                                                                           C->data_ptr(), static_cast<int>(M),
                                                                           static_cast<int>(K), static_cast<int>(N));
            }
        } else {


            if (K % 4 == 0 && K >= 128) {

                gemv_vectorized_kernel<T><<<gridDim, blockDim, 0, stream>>>(A.data_ptr(), B.data_ptr(), C->data_ptr(),
                                                                            static_cast<int>(M), static_cast<int>(K),
                                                                            static_cast<int>(N));
            } else {

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

        const int m = A.sizes()[0];
        const int k = A.sizes()[1];
        const int n = B.sizes()[1];

        const void *ptr_a = A.data_ptr();
        const void *ptr_b = B.data_ptr();
        void *ptr_d = C->data_ptr();
        const void *ptr_bias = (bias ? bias->data_ptr() : nullptr);

        my_cutlass_dtype_t dtype;
        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            dtype = my_cutlass_dtype_t::MY_CUTLASS_DTYPE_BF16;
        } else {
            throw std::runtime_error("Unsupported template type T for cuda_OP::matmul -> cutlass bridge.");
        }


        cutlass_gemm_c_api(m, n, k, dtype, ptr_a, ptr_b, ptr_bias, ptr_d, stream);
    } else if (use_ == 1) {


        static cublasHandle_t handle = nullptr;


        static std::once_flag init_flag;
        static std::mutex handle_mutex;

        std::call_once(init_flag, [&]() {
            std::lock_guard<std::mutex> lock(handle_mutex);

            cublasStatus_t status = cublasCreate(&handle);
            if (status != CUBLAS_STATUS_SUCCESS) {
                fprintf(stderr, "FATAL ERROR: cublasCreate failed in static init: %d\n", status);

                handle = nullptr;
            } else {

                // cublasSetStream(handle, stream);
                // printf("--- Static cublasHandle initialized: %p ---\n",


                std::atexit([]() {
                    std::lock_guard<std::mutex> lock(handle_mutex);
                    if (handle != nullptr) {
                        // printf("--- Destroying static cublasHandle: %p ---\n",

                        cublasDestroy(handle);
                        handle = nullptr;
                    }
                });
            }
        });


        if (handle == nullptr) {
            fprintf(stderr, "Error: cuBLAS handle was not initialized correctly.\n");
            return;
        }


        int lda = K;
        int ldb = K;
        int ldc = N;
        const float alpha = 1.0f;
        const float beta = 0.0f;
        {
            std::lock_guard<std::mutex> lock(handle_mutex);

            CHECK_CUBLAS(cublasSetStream(handle, stream));


            //    m = N, n = M, k = K

            cublas_matmul_wrapper<T>(handle, CUBLAS_OP_T, CUBLAS_OP_N, int(N), int(M), int(K), &alpha,
                                     B.data_ptr(),
                                     ldb,           // ldb = K
                                     A.data_ptr(),
                                     lda,           // lda = K
                                     &beta,
                                     C->data_ptr(),
                                     ldc);           // ldc = N
            // cudaStreamSynchronize(stream);
        }

        if (bias != nullptr) {
            dim3 blockDim(16, 16);

            dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);
            add_bias_kernel<T><<<gridDim, blockDim, 0, stream>>>(C->data_ptr(), bias->data_ptr(), static_cast<int>(M),
                                                                 static_cast<int>(N), ldc);
        }

        return;
    }

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    if (bias == nullptr) {

        matmul_kernel<T><<<numBlocks, threadsPerBlock, 0, stream>>>(A.data_ptr(), B.data_ptr(), C->data_ptr(), M, K, N);
    } else {

        const std::vector<size_t> &bias_shape = bias->sizes();
        if (bias_shape.size() != 1) {
            throw std::runtime_error("Bias must be a 1D tensor");
        }
        if (bias_shape[0] != N) {
            throw std::runtime_error("Bias size must match output column dimension");
        }

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
