#pragma once

#include <cuda_bf16.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "backend/cpp/include/tensor.hpp"
#include "weight_processor_utils.hpp"

namespace py = pybind11;

namespace llama_weight_processor {

// 将 Llama 权重标准化到 Qwen2.5 的命名风格，便于复用同一个解码器实现。
inline void process_global_weights(const py::dict& weights,
                                   std::unordered_map<std::string, Tensor<float>>& cpp_weights) {
    const std::unordered_map<std::string, std::string> key_mapping = {
        {"model.embed_tokens.weight", "token_embeddings.weight"},
        {"model.norm.weight", "norm.weight"},
        {"lm_head.weight", "lm_head"}};

    // 如果没有 embedding，则使用 lm_head 作为 embedding。
    if (!weights.contains("model.embed_tokens.weight") && weights.contains("lm_head.weight")) {
        py::array_t<float> np_array = weights["lm_head.weight"].cast<py::array_t<float>>();
        std::vector<size_t> shape;
        for (int i = 0; i < np_array.ndim(); i++) {
            shape.push_back(np_array.shape(i));
        }
        std::cout << "No embedding_table found, using lm_head as embedding" << std::endl;
        std::vector<float> data(np_array.data(), np_array.data() + np_array.size());
        cpp_weights.emplace("token_embeddings.weight", Tensor<float>(std::move(data), shape));
    }

    // 处理全局权重
    for (const auto& [src_key, dst_key] : key_mapping) {
        if (weights.contains(src_key)) {
            weight_processor_utils::print_processing_info(src_key, dst_key);
            py::array_t<float> np_array = weights[src_key.c_str()].cast<py::array_t<float>>();
            std::vector<size_t> shape;
            for (int i = 0; i < np_array.ndim(); i++) {
                shape.push_back(np_array.shape(i));
            }
            std::vector<float> data(np_array.data(), np_array.data() + np_array.size());
            if (dst_key == "lm_head") {
                cpp_weights.emplace(dst_key, Tensor<float>(std::move(data), shape).transpose(-1, -2));
            } else {
                cpp_weights.emplace(dst_key, Tensor<float>(std::move(data), shape));
            }
        }
    }
}

// 处理层级权重
inline void process_layer_weights(const py::dict& weights,
                                  std::unordered_map<std::string, Tensor<float>>& cpp_weights) {
    const std::vector<std::pair<std::string, std::string>> layer_key_mapping = {
        {"input_layernorm.weight", "input_layernorm.weight"},
        {"post_attention_layernorm.weight", "post_attention_layernorm.weight"},
        {"self_attn.q_proj.weight", "self_attn.q_proj.weight"},
        {"self_attn.k_proj.weight", "self_attn.k_proj.weight"},
        {"self_attn.v_proj.weight", "self_attn.v_proj.weight"},
        {"self_attn.o_proj.weight", "self_attn.o_proj.weight"},
        {"mlp.up_proj.weight", "mlp.up_proj.weight"},
        {"mlp.down_proj.weight", "mlp.down_proj.weight"},
        {"mlp.gate_proj.weight", "mlp.gate_proj.weight"}};

    for (auto item : weights) {
        std::string key = py::str(item.first).cast<std::string>();
        if (key.find("model.layers.") == 0) {
            for (const auto& [src_suffix, dst_prefix] : layer_key_mapping) {
                std::string pattern = "." + src_suffix;
                if (key.find(pattern) != std::string::npos) {
                    // 提取层索引，构造目标键名
                    size_t start = std::string("model.layers.").size();
                    size_t end = key.find('.', start);
                    std::string layer_str = key.substr(start, end - start);
                    int layer = std::stoi(layer_str);
                    std::string dst_key = "layers." + std::to_string(layer) + "." + dst_prefix;

                    weight_processor_utils::print_processing_info(key, dst_key);

                    py::array_t<float> np_array = item.second.cast<py::array_t<float>>();
                    std::vector<size_t> shape;
                    for (int i = 0; i < np_array.ndim(); i++) {
                        shape.push_back(np_array.shape(i));
                    }
                    std::vector<float> data(np_array.data(), np_array.data() + np_array.size());
                    if (src_suffix.find("proj.weight") != std::string::npos) {
                        cpp_weights.emplace(dst_key, Tensor<float>(std::move(data), shape).transpose(-1, -2));
                    } else {
                        cpp_weights.emplace(dst_key, Tensor<float>(std::move(data), shape));
                    }
                }
            }
        }
    }
}

// 处理 Llama 模型权重（FP32）
inline std::unordered_map<std::string, Tensor<float>> process_weights(const py::dict& weights) {
    std::unordered_map<std::string, Tensor<float>> cpp_weights;

    // 初始化进度条
    size_t total_weights = weights.size();
    weight_processor_utils::init_progress(total_weights, "Llama");

    // 处理全局权重
    process_global_weights(weights, cpp_weights);

    // 处理层级权重
    process_layer_weights(weights, cpp_weights);

    // 完成进度条
    weight_processor_utils::finish_progress();

    return cpp_weights;
}

}  // namespace llama_weight_processor
