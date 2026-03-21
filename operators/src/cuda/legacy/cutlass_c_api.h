#pragma once
#include <cuda_runtime.h>

typedef enum { MY_CUTLASS_DTYPE_BF16, MY_CUTLASS_DTYPE_FLOAT32 } my_cutlass_dtype_t;

typedef enum {
    MY_CUTLASS_STATUS_SUCCESS = 0,
    MY_CUTLASS_STATUS_ERROR_INVALID_PROBLEM,
    MY_CUTLASS_STATUS_ERROR_NOT_SUPPORTED,
    MY_CUTLASS_STATUS_ERROR_INTERNAL
} my_cutlass_status_t;

#ifdef __cplusplus
extern "C" {
#endif

void cutlass_gemm_c_api(int m, int n, int k, my_cutlass_dtype_t dtype, const void* ptr_a, const void* ptr_b,
                        const void* ptr_bias, void* ptr_d, cudaStream_t stream);

// 辅助函数：将状态码转换为字符串
const char* cutlass_status_to_string(my_cutlass_status_t status);

#ifdef __cplusplus
}  // extern "C"
#endif