#pragma once

#include <cuda_bf16.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <iostream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "tensor.hpp"
#include "weight_processor_utils.hpp"

namespace py = pybind11;

namespace qwen_weight_processor {

inline void process_global_weights_fp32(const py::dict& weights,
                                        std::unordered_map<std::string, Tensor<float>>& cpp_weights) {

    const std::unordered_map<std::string, std::string> qwen_key_mapping = {
        {"model.embed_tokens.weight", "token_embeddings.weight"},
        {"model.norm.weight", "norm.weight"},
        {"lm_head.weight", "lm_head"},
        {"transformer.wte.weight", "token_embeddings.weight"},
        {"transformer.ln_f.weight", "norm.weight"}};

    for (const auto& [src_key, dst_key] : qwen_key_mapping) {
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

    // Process lm_head ( special handling ) - and AWQ and BF16 keep consistent
    if (weights.contains("lm_head.weight")) {
        weight_processor_utils::print_processing_info("lm_head.weight", "lm_head");
        py::array_t<float> np_array = weights["lm_head.weight"].cast<py::array_t<float>>();
        std::vector<size_t> shape;
        for (int i = 0; i < np_array.ndim(); i++) {
            shape.push_back(np_array.shape(i));
        }
        std::vector<float> data(np_array.data(), np_array.data() + np_array.size());
        cpp_weights.emplace("lm_head", Tensor<float>(std::move(data), shape).transpose(-1, -2));
    }
}

inline void process_layer_weights_fp32(const py::dict& weights,
                                       std::unordered_map<std::string, Tensor<float>>& cpp_weights) {

    const std::vector<std::pair<std::string, std::string>> qwen_layer_key_mapping = {
        {"input_layernorm.weight", "input_layernorm.weight"},
        {"post_attention_layernorm.weight", "post_attention_layernorm.weight"},
        {"self_attn.q_proj.weight", "self_attn.q_proj.weight"},
        {"self_attn.k_proj.weight", "self_attn.k_proj.weight"},
        {"self_attn.v_proj.weight", "self_attn.v_proj.weight"},
        {"self_attn.o_proj.weight", "self_attn.o_proj.weight"},
        {"mlp.gate_proj.weight", "mlp.gate_proj.weight"},
        {"mlp.up_proj.weight", "mlp.up_proj.weight"},
        {"mlp.down_proj.weight", "mlp.down_proj.weight"}};

    const std::vector<std::pair<std::string, std::string>> qwen_layer_bias_mapping = {
        {"self_attn.q_proj.bias", "self_attn.q_proj.bias"},
        {"self_attn.k_proj.bias", "self_attn.k_proj.bias"},
        {"self_attn.v_proj.bias", "self_attn.v_proj.bias"},
        {"self_attn.o_proj.bias", "self_attn.o_proj.bias"}};

    for (auto item : weights) {
        std::string key = py::str(item.first).cast<std::string>();
        if (key.find("model.layers.") == 0) {
            for (const auto& [src_suffix, dst_suffix] : qwen_layer_key_mapping) {
                std::string pattern = "." + src_suffix;
                if (key.find(pattern) != std::string::npos) {
                    size_t start = std::string("model.layers.").size();
                    size_t end = key.find('.', start);
                    std::string layer_str = key.substr(start, end - start);
                    int layer = std::stoi(layer_str);
                    std::string dst_key = "layers." + std::to_string(layer) + "." + dst_suffix;
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
            for (const auto& [src_suffix, dst_suffix] : qwen_layer_bias_mapping) {
                std::string pattern = "." + src_suffix;
                if (key.find(pattern) != std::string::npos) {
                    size_t start = std::string("model.layers.").size();
                    size_t end = key.find('.', start);
                    std::string layer_str = key.substr(start, end - start);
                    int layer = std::stoi(layer_str);
                    std::string dst_key = "layers." + std::to_string(layer) + "." + dst_suffix;
                    weight_processor_utils::print_processing_info(key, dst_key);
                    py::array_t<float> np_array = item.second.cast<py::array_t<float>>();
                    std::vector<size_t> shape;
                    for (int i = 0; i < np_array.ndim(); i++) {
                        shape.push_back(np_array.shape(i));
                    }
                    std::vector<float> data(np_array.data(), np_array.data() + np_array.size());
                    cpp_weights.emplace(dst_key, Tensor<float>(std::move(data), shape));
                }
            }
        }
    }
}

inline std::unordered_map<std::string, Tensor<float>> process_weights_fp32(const py::dict& weights) {
    std::unordered_map<std::string, Tensor<float>> cpp_weights;

    size_t total_weights = weights.size();
    weight_processor_utils::init_progress(total_weights, "Qwen FP32");

    process_global_weights_fp32(weights, cpp_weights);

    process_layer_weights_fp32(weights, cpp_weights);

    // Complete the progress bar - move here, and AWQ and BF16 keep consistent
    weight_processor_utils::finish_progress();

    if (cpp_weights.find("lm_head") == cpp_weights.end()) {
        std::cout << "\nWarning: lm_head not found in weights dict, creating from "
                     "token_embeddings.weight"
                  << std::endl;
        if (cpp_weights.find("token_embeddings.weight") != cpp_weights.end()) {
            try {
                Tensor<float> lm_head = cpp_weights.at("token_embeddings.weight").transpose(-1, -2);
                cpp_weights.emplace("lm_head", std::move(lm_head));
            } catch (const std::exception& e) {
                std::cerr << "Error creating lm_head: " << e.what() << std::endl;
                throw;
            }
        }
    }

    return cpp_weights;
}

inline void process_global_weights_bf16(const py::dict& weights,
                                        std::unordered_map<std::string, Tensor<__nv_bfloat16>>& cpp_weights) {

    const std::unordered_map<std::string, std::string> global_weights_map = {
        {"model.embed_tokens.weight", "token_embeddings.weight"},
        {"model.norm.weight", "norm.weight"},
        {"lm_head.weight", "lm_head"},
        {"model.lm_head.weight", "lm_head"}};

    for (const auto& [src_key, dst_key] : global_weights_map) {
        if (weights.contains(src_key)) {
            weight_processor_utils::print_processing_info(src_key, dst_key);
            py::object tensor = weights[src_key.c_str()];
            Tensor<__nv_bfloat16> bf16_tensor = weight_processor_utils::convert_bf16_tensor(tensor);
            if (dst_key == "lm_head") {
                cpp_weights.emplace(dst_key, bf16_tensor.transpose(-1, -2));
            } else {
                cpp_weights.emplace(dst_key, std::move(bf16_tensor));
            }
        }
    }

    // Process lm_head ( special handling ) - and AWQ keep consistent
    if (weights.contains("lm_head.weight")) {
        weight_processor_utils::print_processing_info("lm_head.weight", "lm_head");
        py::object tensor = weights["lm_head.weight"];
        Tensor<__nv_bfloat16> bf16_tensor = weight_processor_utils::convert_bf16_tensor(tensor);
        cpp_weights.emplace("lm_head", bf16_tensor.transpose(-1, -2));
    }
}

inline void process_layer_weights_bf16(const py::dict& weights,
                                       std::unordered_map<std::string, Tensor<__nv_bfloat16>>& cpp_weights) {

    const std::vector<std::pair<std::string, std::string>> qwen_layer_key_mapping = {
        {"input_layernorm.weight", "input_layernorm.weight"},
        {"post_attention_layernorm.weight", "post_attention_layernorm.weight"},
        {"self_attn.q_proj.weight", "self_attn.q_proj.weight"},
        {"self_attn.k_proj.weight", "self_attn.k_proj.weight"},
        {"self_attn.v_proj.weight", "self_attn.v_proj.weight"},
        {"self_attn.o_proj.weight", "self_attn.o_proj.weight"},
        {"mlp.gate_proj.weight", "mlp.gate_proj.weight"},
        {"mlp.up_proj.weight", "mlp.up_proj.weight"},
        {"mlp.down_proj.weight", "mlp.down_proj.weight"}};

    const std::vector<std::pair<std::string, std::string>> qwen_layer_bias_mapping = {
        {"self_attn.q_proj.bias", "self_attn.q_proj.bias"},
        {"self_attn.k_proj.bias", "self_attn.k_proj.bias"},
        {"self_attn.v_proj.bias", "self_attn.v_proj.bias"},
        {"self_attn.o_proj.bias", "self_attn.o_proj.bias"}};

    for (auto item : weights) {
        std::string key = py::str(item.first).cast<std::string>();
        if (key.find("model.layers.") == 0) {
            for (const auto& [src_suffix, dst_suffix] : qwen_layer_key_mapping) {
                std::string pattern = "." + src_suffix;
                if (key.find(pattern) != std::string::npos) {
                    size_t start = std::string("model.layers.").size();
                    size_t end = key.find('.', start);
                    std::string layer_str = key.substr(start, end - start);
                    int layer = std::stoi(layer_str);
                    std::string dst_key = "layers." + std::to_string(layer) + "." + dst_suffix;
                    weight_processor_utils::print_processing_info(key, dst_key);
                    py::object tensor = weights[key.c_str()];
                    Tensor<__nv_bfloat16> bf16_tensor = weight_processor_utils::convert_bf16_tensor(tensor);
                    if (src_suffix.find("proj.weight") != std::string::npos ||
                        src_suffix.find("gate_proj.weight") != std::string::npos ||
                        src_suffix.find("up_proj.weight") != std::string::npos ||
                        src_suffix.find("down_proj.weight") != std::string::npos) {
                        cpp_weights.emplace(dst_key, bf16_tensor.transpose(-1, -2));
                    } else {
                        cpp_weights.emplace(dst_key, std::move(bf16_tensor));
                    }
                }
            }
            for (const auto& [src_suffix, dst_suffix] : qwen_layer_bias_mapping) {
                std::string pattern = "." + src_suffix;
                if (key.find(pattern) != std::string::npos) {
                    size_t start = std::string("model.layers.").size();
                    size_t end = key.find('.', start);
                    std::string layer_str = key.substr(start, end - start);
                    int layer = std::stoi(layer_str);
                    std::string dst_key = "layers." + std::to_string(layer) + "." + dst_suffix;
                    weight_processor_utils::print_processing_info(key, dst_key);
                    py::object tensor = weights[key.c_str()];
                    Tensor<__nv_bfloat16> bf16_tensor = weight_processor_utils::convert_bf16_tensor(tensor);
                    cpp_weights.emplace(dst_key, std::move(bf16_tensor));
                }
            }
        }
    }
}

inline std::unordered_map<std::string, Tensor<__nv_bfloat16>> process_weights_bf16(const py::dict& weights) {
    std::unordered_map<std::string, Tensor<__nv_bfloat16>> cpp_weights;

    size_t total_weights = weights.size();
    weight_processor_utils::init_progress(total_weights, "Qwen BF16");

    process_global_weights_bf16(weights, cpp_weights);

    process_layer_weights_bf16(weights, cpp_weights);

    // Complete the progress bar - move here, and AWQ keep consistent
    weight_processor_utils::finish_progress();

    if (cpp_weights.find("lm_head") == cpp_weights.end()) {
        std::cout << "\nWarning: lm_head not found in BF16 weights, creating from "
                     "token_embeddings.weight"
                  << std::endl;
        if (cpp_weights.find("token_embeddings.weight") != cpp_weights.end()) {
            Tensor<__nv_bfloat16> lm_head = cpp_weights.at("token_embeddings.weight").transpose(-1, -2);
            cpp_weights.emplace("lm_head", std::move(lm_head));
        } else {
            std::cerr << "Error: token_embeddings.weight not found in BF16 weights" << std::endl;
        }
    }

    return cpp_weights;
}

inline void process_global_weights_awq(const py::dict& weights,
                                       std::unordered_map<std::string, Tensor<__nv_bfloat16>>& cpp_weights) {

    const std::unordered_map<std::string, std::string> global_weights_map = {
        {"model.embed_tokens.weight", "token_embeddings.weight"},
        {"model.norm.weight", "norm.weight"},
        {"lm_head.weight", "lm_head"},
        {"model.lm_head.weight", "lm_head"}};

    for (const auto& [src_key, dst_key] : global_weights_map) {
        if (weights.contains(src_key)) {
            weight_processor_utils::print_processing_info(src_key, dst_key);
            py::object tensor = weights[src_key.c_str()];

            size_t params_count = weight_processor_utils::calculate_params_count(tensor);

            Tensor<__nv_bfloat16> bf16_tensor = weight_processor_utils::convert_bf16_tensor(tensor);
            if (dst_key == "lm_head") {
                cpp_weights.emplace(dst_key, bf16_tensor.transpose(-1, -2));
            } else {
                cpp_weights.emplace(dst_key, std::move(bf16_tensor));
            }
        }
    }

    if (weights.contains("lm_head.weight")) {
        weight_processor_utils::print_processing_info("lm_head.weight", "lm_head");
        py::object tensor = weights["lm_head.weight"];

        size_t params_count = weight_processor_utils::calculate_params_count(tensor);

        Tensor<__nv_bfloat16> bf16_tensor = weight_processor_utils::convert_bf16_tensor(tensor);
        cpp_weights.emplace("lm_head", bf16_tensor.transpose(-1, -2));
    }
}

inline void process_quantized_weights_awq(const py::dict& weights,
                                          std::unordered_map<std::string, Tensor<__nv_bfloat16>>& cpp_weights,
                                          std::unordered_map<std::string, Tensor<int32_t>>& cpp_qweight_params,
                                          std::unordered_map<std::string, Tensor<__nv_bfloat16>>& cpp_scales_params,
                                          std::unordered_map<std::string, Tensor<int32_t>>& cpp_qzeros_params) {
    for (auto item : weights) {
        std::string key = py::str(item.first).cast<std::string>();

        if (key.find(".qweight") != std::string::npos) {
            std::string base_key = key.substr(0, key.find(".qweight"));
            std::string dst_key = base_key;

            if (base_key.find("model.layers.") == 0) {
                size_t start = std::string("model.layers.").size();
                size_t end = base_key.find('.', start);
                std::string layer_str = base_key.substr(start, end - start);
                int layer = std::stoi(layer_str);

                dst_key = "layers." + std::to_string(layer) + base_key.substr(end);
            }

            std::string full_dst_key = dst_key + ".qweight";
            weight_processor_utils::print_processing_info(key, full_dst_key);

            py::object tensor_obj = py::reinterpret_borrow<py::object>(item.second);

            size_t params_count = weight_processor_utils::calculate_params_count(tensor_obj) * 8;

            py::array_t<int32_t> np_array = tensor_obj.cast<py::array_t<int32_t>>();
            std::vector<size_t> shape;
            for (int i = 0; i < np_array.ndim(); i++) {
                shape.push_back(np_array.shape(i));
            }
            std::vector<int32_t> data(np_array.data(), np_array.data() + np_array.size());
            cpp_qweight_params.emplace(full_dst_key, Tensor<int32_t>(std::move(data), shape));
        }

        else if (key.find(".scales") != std::string::npos) {
            std::string base_key = key.substr(0, key.find(".scales"));
            std::string dst_key = base_key;

            if (base_key.find("model.layers.") == 0) {
                size_t start = std::string("model.layers.").size();
                size_t end = base_key.find('.', start);
                std::string layer_str = base_key.substr(start, end - start);
                int layer = std::stoi(layer_str);

                dst_key = "layers." + std::to_string(layer) + base_key.substr(end);
            }

            std::string full_dst_key = dst_key + ".scales";
            weight_processor_utils::print_processing_info(key, full_dst_key);

            py::object tensor_obj = py::reinterpret_borrow<py::object>(item.second);

            size_t params_count = weight_processor_utils::calculate_params_count(tensor_obj);

            py::array_t<float> np_array = tensor_obj.cast<py::array_t<float>>();
            std::vector<size_t> shape;
            for (int i = 0; i < np_array.ndim(); i++) {
                shape.push_back(np_array.shape(i));
            }

            std::vector<float> float_data(np_array.data(), np_array.data() + np_array.size());

            std::vector<__nv_bfloat16> bf16_data;
            bf16_data.reserve(float_data.size());
            for (const auto& val : float_data) {
                bf16_data.push_back(static_cast<__nv_bfloat16>(val));
            }

            cpp_scales_params.emplace(full_dst_key, Tensor<__nv_bfloat16>(std::move(bf16_data), shape));
        }

        else if (key.find(".qzeros") != std::string::npos) {
            std::string base_key = key.substr(0, key.find(".qzeros"));
            std::string dst_key = base_key;

            if (base_key.find("model.layers.") == 0) {
                size_t start = std::string("model.layers.").size();
                size_t end = base_key.find('.', start);
                std::string layer_str = base_key.substr(start, end - start);
                int layer = std::stoi(layer_str);

                dst_key = "layers." + std::to_string(layer) + base_key.substr(end);
            }

            std::string full_dst_key = dst_key + ".qzeros";
            weight_processor_utils::print_processing_info(key, full_dst_key);

            py::object tensor_obj = py::reinterpret_borrow<py::object>(item.second);

            size_t params_count = weight_processor_utils::calculate_params_count(tensor_obj);

            py::array_t<int32_t> np_array = tensor_obj.cast<py::array_t<int32_t>>();
            std::vector<size_t> shape;
            for (int i = 0; i < np_array.ndim(); i++) {
                shape.push_back(np_array.shape(i));
            }
            std::vector<int32_t> data(np_array.data(), np_array.data() + np_array.size());
            cpp_qzeros_params.emplace(full_dst_key, Tensor<int32_t>(std::move(data), shape));
        }

        else if (key.find("model.layers.") == 0 && key.find(".qweight") == std::string::npos &&
                 key.find(".scales") == std::string::npos && key.find(".qzeros") == std::string::npos) {

            const std::vector<std::pair<std::string, std::string>> qwen_layer_key_mapping = {
                {"input_layernorm.weight", "input_layernorm.weight"},
                {"post_attention_layernorm.weight", "post_attention_layernorm.weight"},
                {"self_attn.q_proj.bias", "self_attn.q_proj.bias"},
                {"self_attn.k_proj.bias", "self_attn.k_proj.bias"},
                {"self_attn.v_proj.bias", "self_attn.v_proj.bias"},
                {"self_attn.o_proj.bias", "self_attn.o_proj.bias"}};

            for (const auto& [src_suffix, dst_suffix] : qwen_layer_key_mapping) {
                std::string pattern = "." + src_suffix;
                if (key.find(pattern) != std::string::npos) {
                    size_t start = std::string("model.layers.").size();
                    size_t end = key.find('.', start);
                    std::string layer_str = key.substr(start, end - start);
                    int layer = std::stoi(layer_str);
                    std::string dst_key = "layers." + std::to_string(layer) + "." + dst_suffix;
                    weight_processor_utils::print_processing_info(key, dst_key);
                    py::object tensor = weights[key.c_str()];

                    size_t params_count = weight_processor_utils::calculate_params_count(tensor);

                    Tensor<__nv_bfloat16> bf16_tensor = weight_processor_utils::convert_bf16_tensor(tensor);
                    cpp_weights.emplace(dst_key, std::move(bf16_tensor));
                    break;
                }
            }
        }
    }
}

inline std::tuple<
    std::unordered_map<std::string, Tensor<__nv_bfloat16>>, std::unordered_map<std::string, Tensor<int32_t>>,
    std::unordered_map<std::string, Tensor<__nv_bfloat16>>, std::unordered_map<std::string, Tensor<int32_t>>>
process_weights_awq(const py::dict& weights) {
    std::unordered_map<std::string, Tensor<__nv_bfloat16>> cpp_weights;
    std::unordered_map<std::string, Tensor<int32_t>> cpp_qweight_params;
    std::unordered_map<std::string, Tensor<__nv_bfloat16>> cpp_scales_params;
    std::unordered_map<std::string, Tensor<int32_t>> cpp_qzeros_params;

    size_t total_weights = weights.size();
    weight_processor_utils::init_progress(total_weights, "Qwen AWQ");

    process_global_weights_awq(weights, cpp_weights);

    process_quantized_weights_awq(weights, cpp_weights, cpp_qweight_params, cpp_scales_params, cpp_qzeros_params);

    weight_processor_utils::finish_progress();

    if (cpp_weights.find("lm_head") == cpp_weights.end()) {
        std::cout << "\nWarning: lm_head not found in AWQ weights, creating from "
                     "token_embeddings.weight"
                  << std::endl;
        if (cpp_weights.find("token_embeddings.weight") != cpp_weights.end()) {
            Tensor<__nv_bfloat16> lm_head = cpp_weights.at("token_embeddings.weight").transpose(-1, -2);
            cpp_weights.emplace("lm_head", std::move(lm_head));
        } else {
            std::cerr << "Error: token_embeddings.weight not found in AWQ weights" << std::endl;
        }
    }

    return {cpp_weights, cpp_qweight_params, cpp_scales_params, cpp_qzeros_params};
}

}  // namespace qwen_weight_processor
