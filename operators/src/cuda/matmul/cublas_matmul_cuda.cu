#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <mutex>
#include <stdexcept>

#include "operators/cuda/cuda_resource_manager.cuh"
#include "operators/cuda/matmul/cublas_matmul_cuda.cuh"

inline void checkCublasStatus(cublasStatus_t status, const char *file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        char errorMsg[256];
        snprintf(errorMsg, sizeof(errorMsg), "cuBLAS error %d at %s:%d", static_cast<int>(status), file, line);
        fprintf(stderr, "%s\n", errorMsg);
        throw std::runtime_error(errorMsg);
    }
}
#define CHECK_CUBLAS(call) checkCublasStatus(call, __FILE__, __LINE__)

namespace op {

template <typename T>
CublasMatmulCUDAOperator<T>::CublasMatmulCUDAOperator() : initialized_(false) {
    // CUDAResourceManager owns and lazily initializes the shared cuBLAS handle.
}

template <typename T>
CublasMatmulCUDAOperator<T>::~CublasMatmulCUDAOperator() {
    // The shared handle is released by CUDAResourceManager.
}

// Compatibility method; CUDAResourceManager owns the actual handle.
template <typename T>
void CublasMatmulCUDAOperator<T>::initialize() {
    initialized_ = true;
}

// Compatibility method; CUDAResourceManager owns the actual handle.
template <typename T>
void CublasMatmulCUDAOperator<T>::destroy() {
    initialized_ = false;
}

template <typename InputType>
void cublas_matmul_wrapper(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
                           int k, const float *alpha, const InputType *d_A, int lda, const InputType *d_B, int ldb,
                           const float *beta, InputType *d_C, int ldc) {
    cudaDataType_t cuda_data_type_A;
    cudaDataType_t cuda_data_type_B;
    cudaDataType_t cuda_data_type_C;
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;

    if constexpr (std::is_same_v<InputType, __nv_bfloat16>) {
        cuda_data_type_A = CUDA_R_16BF;
        cuda_data_type_B = CUDA_R_16BF;
        cuda_data_type_C = CUDA_R_16BF;
    } else if constexpr (std::is_same_v<InputType, float>) {
        cuda_data_type_A = CUDA_R_32F;
        cuda_data_type_B = CUDA_R_32F;
        cuda_data_type_C = CUDA_R_32F;
    } else {
        static_assert(std::is_same_v<InputType, __nv_bfloat16> || std::is_same_v<InputType, float>,
                      "cublas_matmul_wrapper supports only __nv_bfloat16 or float input/output types");
        return;
    }

    cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;

    cublasStatus_t status = cublasGemmEx(handle, transa, transb, m, n, k, alpha, d_A, cuda_data_type_A, lda, d_B,
                                         cuda_data_type_B, ldb, beta, d_C, cuda_data_type_C, ldc, compute_type, algo);

    CHECK_CUBLAS(status);
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

template <typename T>
void CublasMatmulCUDAOperator<T>::operator()(Tensor<T> *output, Tensor<T> *input, const WeightTensor<T> &weight,
                                             const Tensor<T> *bias, cudaStream_t stream) {

    if (weight.is_quantized()) {
        throw std::runtime_error("cuBLAS MatMul does not support quantized weights");
    }

    initialized_ = true;

    cublasHandle_t handle = CUDAResourceManager::instance().getCublasHandle();

    if (stream) {
        CUDAResourceManager::instance().setCublasStream(stream);
    }

    const std::vector<size_t> &A_shape = input->sizes();
    const std::vector<size_t> &B_shape = weight.tensor()->sizes();

    // A is [M, K]. The loader provides logical weight views [K, N]
    // over physical row-major [N, K] storage, producing C [M, N].
    size_t M = A_shape[0];
    size_t K = A_shape[1];
    size_t N = B_shape[1];

    int lda = K;
    int ldb = K;
    int ldc = N;

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // cuBLAS uses column-major views: C^T = B * A^T.
    cublas_matmul_wrapper<T>(handle,
                             CUBLAS_OP_T,
                             CUBLAS_OP_N,
                             int(N),                       // m = N
                             int(M),                       // n = M
                             int(K),                       // k = K
                             &alpha,                       // alpha = 1.0
                             weight.tensor()->data_ptr(),
                             ldb,                          // ldb = K
                             input->data_ptr(),
                             lda,                          // lda = K
                             &beta,                        // beta = 0.0
                             output->data_ptr(),
                             ldc                           // ldc = N
    );

    if (bias != nullptr) {
        dim3 blockDim(16, 16);
        dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

        add_bias_kernel<T><<<gridDim, blockDim, 0, stream>>>(output->data_ptr(), bias->data_ptr(), static_cast<int>(M),
                                                             static_cast<int>(N), ldc);
    }
}

template class CublasMatmulCUDAOperator<float>;
template class CublasMatmulCUDAOperator<__nv_bfloat16>;

}  // namespace op
