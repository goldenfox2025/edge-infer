#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <mutex>
#include <stdexcept>

#include "operators/cuda/cuda_resource_manager.cuh"
#include "operators/cuda/matmul/cublas_matmul_cuda.cuh"

// 检查cuBLAS状态的辅助函数
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

// CublasMatmulCUDAOperator构造函数
template <typename T>
CublasMatmulCUDAOperator<T>::CublasMatmulCUDAOperator() : initialized_(false) {
    // 不再在这里初始化句柄，将使用CUDAResourceManager
}

// CublasMatmulCUDAOperator析构函数
template <typename T>
CublasMatmulCUDAOperator<T>::~CublasMatmulCUDAOperator() {
    // 不再在这里释放句柄，由CUDAResourceManager负责释放
}

// 初始化cuBLAS句柄 - 这个方法将不再使用，保留以满足头文件要求
template <typename T>
void CublasMatmulCUDAOperator<T>::initialize() {
    initialized_ = true;  // 直接标记为已初始化，实际句柄由ResourceManager管理
}

// 销毁cuBLAS句柄 - 这个方法将不再使用，保留以满足头文件要求
template <typename T>
void CublasMatmulCUDAOperator<T>::destroy() {
    initialized_ = false;  // 直接标记为未初始化，实际句柄由ResourceManager管理
}

// cuBLAS矩阵乘法包装函数
template <typename InputType>
void cublas_matmul_wrapper(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
                           int k, const float *alpha, const InputType *d_A, int lda, const InputType *d_B, int ldb,
                           const float *beta, InputType *d_C, int ldc) {
    cudaDataType_t cuda_data_type_A;
    cudaDataType_t cuda_data_type_B;
    cudaDataType_t cuda_data_type_C;
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;

    // 根据模板类型设置CUDA数据类型
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
                      "cublas_matmul_wrapper 只支持 __nv_bfloat16 和 float 输入/输出类型");
        return;
    }

    // 选择算法
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;

    // 执行GEMM操作
    cublasStatus_t status = cublasGemmEx(handle, transa, transb, m, n, k, alpha, d_A, cuda_data_type_A, lda, d_B,
                                         cuda_data_type_B, ldb, beta, d_C, cuda_data_type_C, ldc, compute_type, algo);

    CHECK_CUBLAS(status);
}

// 添加偏置的CUDA核函数
template <typename T>
__global__ void add_bias_kernel(T *C, const T *bias, int M, int N, int ldc) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        int c_index = row * ldc + col;
        C[c_index] = C[c_index] + bias[col];
    }
}

// 实现CublasMatmulCUDAOperator的operator()方法
template <typename T>
void CublasMatmulCUDAOperator<T>::operator()(Tensor<T> *output, Tensor<T> *input, const WeightTensor<T> &weight,
                                             const Tensor<T> *bias, cudaStream_t stream) {
    // 确保权重不是量化的
    if (weight.is_quantized()) {
        throw std::runtime_error("cuBLAS MatMul does not support quantized weights");
    }

    initialized_ = true;  // 标记为已初始化

    // 从资源管理器获取cuBLAS句柄
    cublasHandle_t handle = CUDAResourceManager::instance().getCublasHandle();

    // 设置流
    if (stream) {
        CUDAResourceManager::instance().setCublasStream(stream);
    }

    // 获取输入尺寸
    const std::vector<size_t> &A_shape = input->sizes();
    const std::vector<size_t> &B_shape = weight.tensor()->sizes();

    // A: [M, K], B: [N, K]（保证 A 的第二维与 B 的第二维一致）
    size_t M = A_shape[0];
    size_t K = A_shape[1];
    size_t N = B_shape[1];

    // 设置布局和步长
    int lda = K;  // A 每行有 K 个元素
    int ldb = K;  // B 每行有 K 个元素
    int ldc = N;  // C 每行有 N 个元素

    // 设置alpha和beta值
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // 调用cuBLAS的GEMM操作
    // 计算: C = A * B^T (或表示为 C^T = B * A^T)
    cublas_matmul_wrapper<T>(handle,
                             CUBLAS_OP_T,                  // 转置B
                             CUBLAS_OP_N,                  // 不转置A
                             int(N),                       // m = N
                             int(M),                       // n = M
                             int(K),                       // k = K
                             &alpha,                       // alpha = 1.0
                             weight.tensor()->data_ptr(),  // B矩阵
                             ldb,                          // ldb = K
                             input->data_ptr(),            // A矩阵
                             lda,                          // lda = K
                             &beta,                        // beta = 0.0
                             output->data_ptr(),           // C矩阵
                             ldc                           // ldc = N
    );

    // 如果提供了偏置，添加偏置
    if (bias != nullptr) {
        dim3 blockDim(16, 16);
        dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

        add_bias_kernel<T><<<gridDim, blockDim, 0, stream>>>(output->data_ptr(), bias->data_ptr(), static_cast<int>(M),
                                                             static_cast<int>(N), ldc);
    }
}

// 显式模板实例化
template class CublasMatmulCUDAOperator<float>;
template class CublasMatmulCUDAOperator<__nv_bfloat16>;

}  // namespace op