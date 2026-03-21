#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>

#include <cmath>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "cuda/legacy/legacy_cuda_api.cuh"
#define WARP_SIZE 32
// 因为要支持多维度 所以暂时不考虑向量化加载
// 但高维softmax似乎也可以
// a1 b1 c1 d1
// a2 b2 c2 d2
// 相当于一次性加载4个数据，然后每个线程处理列方向上
// 上面是二维的情况。
// 在三维甚至四维时候，需要额外考虑。
namespace cuda_OP {

// 在这里，尝试一种模板注入的新写法
// 练习罢了

__align__(8) struct ml {
    __device__ ml(float m, float l) : m(m), l(l) {
    }
    __device__ ml() : m(-1e9f), l(0.0f) {
    }
    float m;
    float l;
};

struct online_softmax_op {
    __device__ __forceinline__ ml operator()(const ml &a, const ml &b) const {
        ml mi;
        ml ma;
        ml ret;
        if (a.m < b.m) {
            mi = a;
            ma = b;
        } else {
            mi = b;
            ma = a;
        }
        mi.l *= expf(mi.m - ma.m);
        ret.l = mi.l + ma.l;
        ret.m = ma.m;
        return ret;
    }
};

__device__ __forceinline__ ml warp_all_reduce_for_ml(ml val) {
    unsigned int active_mask = __activemask();  // 获取当前 warp 中活跃线程的掩码
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float m = val.m;
        float l = val.l;
        float m_ = __shfl_xor_sync(active_mask, m, offset);
        float l_ = __shfl_xor_sync(active_mask, l, offset);
        val = online_softmax_op()(val, ml(m_, l_));
    }
    return val;
}
// 一次处理sum和max
// 暂时不想考虑步长，因为本项目里没有用到，很麻烦
// 先写完onlinesoftmax再考虑步长的加入
template <typename T>
__global__ void online_softmax_kernel_for_last_dim(T *output, const T *data, int seq_len, int n_heads,
                                                   int total_seq_len, int offset) {
    int idx = blockIdx.x;
    int seq_id = idx / n_heads;
    int head_id = idx % n_heads;
    int tid = threadIdx.x;
    if (seq_id >= seq_len || head_id >= n_heads)
        return;

    int start_idx = seq_id * (n_heads * total_seq_len) + head_id * total_seq_len;
    int valid_length = offset + seq_id + 1;

    // 在这里，我们仍然强制线程不超过1024
    // 这样就可以一次warp归约 + 一次共享内存读取 + 一次warp归约解决问题
    // 如果超过1024的话 一个普适的方法是第二次warp归约改成共享内存归约
    // 可以作为另一个实现（在total_seq_len相当长的时候）
    // 或者直接分块softmax 反正这个算法支持
    // --问题：速度会如何？--
    __shared__ ml sdata[32];
    ml thread_ml;
    for (int i = tid; i < total_seq_len; i += blockDim.x) {
        float val = (i >= valid_length) ? float(-1e9) : static_cast<float>(data[start_idx + i]);
        thread_ml = online_softmax_op()(thread_ml, ml(val, 1.0f));
    }
    // 所有线程的任务结束

    thread_ml = warp_all_reduce_for_ml(thread_ml);

    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int warp_num = blockDim.x / 32;

    if (lane_id == 0) {
        sdata[warp_id] = thread_ml;
    }
    __syncthreads();

    if (warp_id == 0) {
        ml warp_ml = lane_id < warp_num ? sdata[lane_id] : ml();
        warp_ml = warp_all_reduce_for_ml(warp_ml);
        if (lane_id == 0) {
            sdata[0] = warp_ml;
        }
    }
    __syncthreads();
    float max_val = sdata[0].m;
    float sum_val = sdata[0].l;

    // 得到了最终的ml

    for (int i = tid; i < total_seq_len; i += blockDim.x) {
        float val = (i >= valid_length) ? float(-1e9) : static_cast<float>(data[start_idx + i]);
        float exp_val = __expf(val - max_val);
        output[start_idx + i] = static_cast<T>(exp_val / sum_val);
    }
}
// 仅支持最后一个维度 因果mask
template <typename T>
__global__ void softmax_kernel_for_last_dim_v0(T *output, const T *data, int seq_len, int n_heads, int total_seq_len,
                                               int offset) {
    int idx = blockIdx.x;
    int seq_id = idx / n_heads;
    int head_id = idx % n_heads;

    if (seq_id >= seq_len || head_id >= n_heads)
        return;

    int start_idx = seq_id * (n_heads * total_seq_len) + head_id * total_seq_len;
    int valid_length = offset + seq_id + 1;

    __shared__ float sdata[32];
    int tid = threadIdx.x;
    int warp_num = blockDim.x / 32;
    // find max
    float thread_max = -1e9f;
    for (int i = tid; i < total_seq_len; i += blockDim.x) {
        float val = (i >= valid_length) ? float(-1e9) : static_cast<float>(data[start_idx + i]);
        thread_max = fmaxf(thread_max, val);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_max = fmaxf(thread_max, __shfl_down_sync(0xFFFFFFFF, thread_max, offset));
    }
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    if (lane_id == 0)
        sdata[warp_id] = thread_max;
    __syncthreads();
    float max_val = -1e9f;
    if (warp_id == 0) {
        max_val = lane_id < warp_num ? sdata[lane_id] : -1e9f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            max_val = fmaxf(max_val, __shfl_down_sync(0xFFFFFFFF, max_val, offset));
        }
    }
    if (warp_id == 0 && lane_id == 0)
        sdata[0] = max_val;
    __syncthreads();

    // compute sum
    max_val = sdata[0];
    float thread_sum = 0.0f;
    for (int i = tid; i < total_seq_len; i += blockDim.x) {
        float val = (i >= valid_length) ? float(-1e9) : static_cast<float>(data[start_idx + i]);
        float exp_val = __expf(val - max_val);
        output[start_idx + i] = static_cast<T>(exp_val);
        thread_sum += exp_val;
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, offset);
    }

    if (lane_id == 0) {
        sdata[warp_id] = thread_sum;
    }
    __syncthreads();

    float warp_sum = 0.0f;
    if (warp_id == 0) {
        warp_sum = lane_id < warp_num ? sdata[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, offset);
        }
    }
    if (warp_id == 0 && lane_id == 0)
        sdata[0] = warp_sum;
    __syncthreads();

    // normalize
    float block_sum = sdata[0];
    for (int i = tid; i < total_seq_len; i += blockDim.x) {
        output[start_idx + i] /= static_cast<T>(block_sum);
    }
}

template <typename T>
void softmax(Tensor<T> *output, const Tensor<T> *input, int dim, bool mask, int offset, cudaStream_t stream) {
    const std::vector<size_t> &shape = input->sizes();
    if (shape.size() == 3 && dim == 2) {
        int seq_len = shape[0];
        int n_heads = shape[1];
        int total_seq_len = shape[2];
        int total_rows = seq_len * n_heads;
        int THREADS_PER_BLOCK = 512;  // 增加线程数以处理更长的序列
        // 计算每个block需要的共享内存大小：每个warp一个float
        // int sharedMemSize = (THREADS_PER_BLOCK / 32 + 1) * sizeof(float);
        online_softmax_kernel_for_last_dim<T><<<total_rows, THREADS_PER_BLOCK, 0, stream>>>(
            output->data_ptr(), input->data_ptr(), seq_len, n_heads, total_seq_len, offset);
    } else if (shape.size() == 2 && dim == 1) {
        int seq_len = 1;
        int n_heads = shape[0];
        int total_seq_len = shape[1];
        int total_rows = seq_len * n_heads;
        int THREADS_PER_BLOCK = 512;  // 增加线程数以处理更长的序列
        // 计算每个block需要的共享内存大小：每个warp一个float
        // int sharedMemSize = (THREADS_PER_BLOCK / 32 + 1) * sizeof(float);
        online_softmax_kernel_for_last_dim<T><<<total_rows, THREADS_PER_BLOCK, 0, stream>>>(
            output->data_ptr(), input->data_ptr(), seq_len, n_heads, total_seq_len, offset);
    } else {
        throw std::runtime_error("softmax: Unsupported tensor dimension or dim value");
    }
    checkCudaError(cudaGetLastError());
    // checkCudaError(cudaDeviceSynchronize());
}

template void softmax<nvbf16>(Tensor<nvbf16> *, const Tensor<nvbf16> *, int, bool, int, cudaStream_t);

template void softmax<float>(Tensor<float> *, const Tensor<float> *, int, bool, int, cudaStream_t);

}  // namespace cuda_OP
