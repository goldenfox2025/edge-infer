#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "cuda_graph_runtime.hpp"
#include "inference.hpp"

template <typename T>
class GraphRunner {
 public:
  static void extract_kv_copy_nodes(CudaGraphRuntime<T>& runtime,
                                    size_t n_layers, size_t n_kv_heads,
                                    size_t head_dim) {
    if (!runtime.cuda_graph) {
      return;
    }

    size_t num_nodes = 0;
    check(cudaGraphGetNodes(runtime.cuda_graph, nullptr, &num_nodes));

    std::vector<cudaGraphNode_t> nodes(num_nodes);
    check(cudaGraphGetNodes(runtime.cuda_graph, nodes.data(), &num_nodes));

    // CUDA node enumeration has no layer order. Identify each copy by its
    // fixed source address and store deterministic [K0, V0, K1, V1, ...] slots.
    runtime.kv_copy_nodes.assign(n_layers * 2, nullptr);
    for (size_t i = 0; i < num_nodes; ++i) {
      cudaGraphNodeType node_type;
      check(cudaGraphNodeGetType(nodes[i], &node_type));
      if (node_type != cudaGraphNodeTypeMemcpy) {
        continue;
      }

      cudaMemcpy3DParms params;
      check(cudaGraphMemcpyNodeGetParams(nodes[i], &params));

      for (size_t layer = 0; layer < n_layers; ++layer) {
        Tensor<T> k_buf =
            static_cast<const Tensor<T>&>(runtime.fixed_k_buffers[layer]).view({1, n_kv_heads, head_dim});
        Tensor<T> v_buf =
            static_cast<const Tensor<T>&>(runtime.fixed_v_buffers[layer]).view({1, n_kv_heads, head_dim});
        if (params.srcPtr.ptr == k_buf.data_ptr()) {
          runtime.kv_copy_nodes[2 * layer] = nodes[i];
          break;
        }
        if (params.srcPtr.ptr == v_buf.data_ptr()) {
          runtime.kv_copy_nodes[2 * layer + 1] = nodes[i];
          break;
        }
      }
    }
    for (auto node : runtime.kv_copy_nodes) {
      if (!node) throw std::runtime_error("Missing CUDA graph KV copy node");
    }
  }

  static void update_kv_copy_nodes(CudaGraphRuntime<T>& runtime,
                                   KVCache<T>* kv_cache, size_t offset,
                                   size_t n_layers) {
    if (!runtime.graph_initialized || !runtime.graph_exec || !kv_cache ||
        runtime.kv_copy_nodes.empty()) {
      return;
    }

    size_t node_idx = 0;
    for (size_t layer = 0;
         layer < n_layers && node_idx < runtime.kv_copy_nodes.size();
         ++layer) {
      if (node_idx < runtime.kv_copy_nodes.size()) {
        update_memcpy_node(runtime, runtime.kv_copy_nodes[node_idx],
                           kv_cache->k_cache(layer, offset));
        ++node_idx;
      }
      if (node_idx < runtime.kv_copy_nodes.size()) {
        update_memcpy_node(runtime, runtime.kv_copy_nodes[node_idx],
                           kv_cache->v_cache(layer, offset));
        ++node_idx;
      }
    }
  }

  template <typename WarmupFn, typename CaptureFn>
  static void initialize(CudaGraphRuntime<T>& runtime, const std::string& label,
                         WarmupFn&& warmup_fn, CaptureFn&& capture_fn) {
    if (runtime.graph_initialized) {
      return;
    }

    warmup_fn();

    cudaError_t result = cudaStreamBeginCapture(runtime.graph_stream,
                                                cudaStreamCaptureModeGlobal);
    if (result != cudaSuccess) {
      throw std::runtime_error("Failed to begin " + label +
                               " CUDA graph capture: " +
                               std::string(cudaGetErrorString(result)));
    }

    bool capture_active = true;
    try {
      runtime.graph_output_tensor = capture_fn();

      result = cudaStreamEndCapture(runtime.graph_stream, &runtime.cuda_graph);
      capture_active = false;
      if (result != cudaSuccess) {
        throw std::runtime_error("Failed to end " + label +
                                 " CUDA graph capture: " +
                                 std::string(cudaGetErrorString(result)));
      }

      result = cudaGraphInstantiate(&runtime.graph_exec, runtime.cuda_graph,
                                    nullptr, nullptr, 0);
      if (result != cudaSuccess) {
        throw std::runtime_error("Failed to instantiate " + label +
                                 " CUDA graph: " +
                                 std::string(cudaGetErrorString(result)));
      }

      runtime.graph_initialized = true;
    } catch (...) {
      if (capture_active) {
        cudaStreamEndCapture(runtime.graph_stream, &runtime.cuda_graph);
      }
      runtime.release_graph_objects();
      throw;
    }
  }

  static void launch(CudaGraphRuntime<T>& runtime, const std::string& label) {
    cudaError_t result = cudaGraphLaunch(runtime.graph_exec, runtime.graph_stream);
    if (result != cudaSuccess) {
      throw std::runtime_error("Failed to launch " + label + " CUDA graph: " +
                               std::string(cudaGetErrorString(result)));
    }
    runtime.pingpong_index = 1 - runtime.pingpong_index;
  }

 private:
  static void check(cudaError_t result) {
    if (result != cudaSuccess) {
      throw std::runtime_error(std::string("CUDA graph operation failed: ") +
                               cudaGetErrorString(result));
    }
  }

  static void update_memcpy_node(CudaGraphRuntime<T>& runtime,
                                 cudaGraphNode_t node,
                                 Tensor<T>& target_slice) {
    cudaMemcpy3DParms params;
    cudaError_t get_param_err = cudaGraphMemcpyNodeGetParams(node, &params);
    check(get_param_err);

    params.dstPtr.ptr = target_slice.data_ptr();
    check(cudaGraphExecMemcpyNodeSetParams(runtime.graph_exec, node, &params));
  }
};
