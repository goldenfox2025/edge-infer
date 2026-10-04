#pragma once

#include <cuda_runtime.h>

#include <vector>

#include "tensor.hpp"

template <typename T>
struct CudaGraphRuntime {
    int pingpong_index = 0;

    cudaGraph_t cuda_graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    cudaStream_t graph_stream = nullptr;
    bool graph_initialized = false;

    Tensor<uint32_t> graph_input_tensor;
    Tensor<T> graph_output_tensor;

    size_t* d_rope_offset = nullptr;
    int* d_offset_array = nullptr;

    std::vector<Tensor<T>> fixed_k_buffers;
    std::vector<Tensor<T>> fixed_v_buffers;
    std::vector<cudaGraphNode_t> kv_copy_nodes;

    Tensor<int> segment_info_tensor;
    int* d_segment_info = nullptr;
    Tensor<T*> output_ptrs_tensor;
    T** d_output_ptrs = nullptr;
    std::vector<Tensor<T>> fixed_fa_outputs;
    int* pingpong = nullptr;

    cudaStream_t prep_stream = nullptr;
    size_t last_kv_cache_size = 0;

    void reset_state() {
        pingpong_index = 0;
        graph_initialized = false;
        last_kv_cache_size = 0;
    }

    void release_graph_objects() {
        synchronize_streams();
        if (graph_exec) {
            cudaGraphExecDestroy(graph_exec);
            graph_exec = nullptr;
        }
        if (cuda_graph) {
            cudaGraphDestroy(cuda_graph);
            cuda_graph = nullptr;
        }
        graph_initialized = false;
        kv_copy_nodes.clear();
    }

    void release_streams() {
        if (graph_stream) {
            cudaStreamSynchronize(graph_stream);
            cudaStreamDestroy(graph_stream);
            graph_stream = nullptr;
        }
        if (prep_stream) {
            cudaStreamSynchronize(prep_stream);
            cudaStreamDestroy(prep_stream);
            prep_stream = nullptr;
        }
    }

    void release_fixed_memory() {
        synchronize_streams();
        if (d_rope_offset) {
            cudaFree(d_rope_offset);
            d_rope_offset = nullptr;
        }
        if (d_offset_array) {
            cudaFree(d_offset_array);
            d_offset_array = nullptr;
        }

        kv_copy_nodes.clear();
        d_segment_info = nullptr;
        d_output_ptrs = nullptr;
        fixed_k_buffers.clear();
        fixed_v_buffers.clear();
        fixed_fa_outputs.clear();
    }

    void release_pingpong() {
        synchronize_streams();
        if (pingpong) {
            cudaFree(pingpong);
            pingpong = nullptr;
        }
    }

    void synchronize_streams() noexcept {
        if (graph_stream) cudaStreamSynchronize(graph_stream);
        if (prep_stream) cudaStreamSynchronize(prep_stream);
    }
};
