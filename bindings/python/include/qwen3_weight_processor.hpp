#pragma once

#include <cstdlib>
#include <iostream>
#include <set>
#include <tuple>

#include "weight_processor_utils.hpp"

namespace qwen3_weight_processor {

inline bool verbose_awq_debug() {
    const char* value = std::getenv("EDGE_INFER_VERBOSE_WEIGHTS");
    return value != nullptr && std::string(value) == "1";
}

inline void process_global_weights_bf16(const py::dict& weights,
                                        std::unordered_map<std::string, Tensor<__nv_bfloat16>>& cpp_weights);

inline void process_layer_weights_bf16(const py::dict& weights,
                                       std::unordered_map<std::string, Tensor<__nv_bfloat16>>& cpp_weights);

inline void process_global_weights_awq(const py::dict& weights,
                                       std::unordered_map<std::string, Tensor<__nv_bfloat16>>& cpp_weights);

inline void process_quantized_weights_awq(const py::dict& weights,
                                          std::unordered_map<std::string, Tensor<__nv_bfloat16>>& cpp_weights,
                                          std::unordered_map<std::string, Tensor<int32_t>>& cpp_qweight_params,
                                          std::unordered_map<std::string, Tensor<__nv_bfloat16>>& cpp_scales_params,
                                          std::unordered_map<std::string, Tensor<int32_t>>& cpp_qzeros_params);

inline void process_global_weights_bf16(const py::dict& weights,
                                        std::unordered_map<std::string, Tensor<__nv_bfloat16>>& cpp_weights) {

    const std::unordered_map<std::string, std::string> global_weights_map = {
        {"model.embed_tokens.weight", "token_embeddings.weight"},
        {"model.norm.weight", "rms_out_w"},
        {"lm_head.weight", "lm_head"}};

    for (const auto& [src_key, dst_key] : global_weights_map) {
        if (weights.contains(src_key)) {
            weight_processor_utils::print_processing_info(src_key, dst_key);
            py::object tensor = weights[src_key.c_str()];

            size_t params_count = weight_processor_utils::calculate_params_count(tensor);

            Tensor<__nv_bfloat16> bf16_tensor = weight_processor_utils::convert_bf16_tensor(tensor);
            if (dst_key == "lm_head") {
                // lm_head weights require a logical transpose, to match matrix multiplication
                auto shape_obj = tensor.attr("shape");
                auto shape_tuple = shape_obj.cast<py::tuple>();
                std::vector<size_t> shape;
                for (size_t i = 0; i < shape_tuple.size(); i++) {
                    shape.push_back(shape_tuple[i].cast<size_t>());
                }

                if (shape.size() >= 2) {

                    // std::cout << " for lm_head Perform a logical transpose " << std::endl;
                    // for lm_head perform a logical transpose, keep KN format but viewed as NK format
                    cpp_weights.emplace(dst_key, bf16_tensor.transpose(-1, -2));
                } else {

                    cpp_weights.emplace(dst_key, bf16_tensor.transpose(-1, -2));
                }
            } else {
                cpp_weights.emplace(dst_key, std::move(bf16_tensor));
            }
        }
    }
}

inline void process_layer_weights_bf16(const py::dict& weights,
                                       std::unordered_map<std::string, Tensor<__nv_bfloat16>>& cpp_weights) {

    const std::vector<std::pair<std::string, std::string>> layer_key_mapping = {
        {"input_layernorm.weight", "rms_att_w"}, {"post_attention_layernorm.weight", "rms_ffn_w"},
        {"self_attn.q_proj.weight", "wq"},       {"self_attn.k_proj.weight", "wk"},
        {"self_attn.v_proj.weight", "wv"},       {"self_attn.o_proj.weight", "wo"},
        {"self_attn.q_norm.weight", "q_norm"},   {"self_attn.k_norm.weight", "k_norm"},
        {"mlp.gate_proj.weight", "w_gate"},      {"mlp.up_proj.weight", "w_up"},
        {"mlp.down_proj.weight", "w_down"}};

    // Linear layers requiring transposition ( perform only a logical transpose, without physically transposing data )
    const std::unordered_set<std::string> transpose_layers = {"wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down"};

    for (auto item : weights) {
        std::string key = py::str(item.first).cast<std::string>();
        if (key.find("model.layers.") == 0) {
            for (const auto& [src_suffix, dst_prefix] : layer_key_mapping) {
                std::string pattern = "." + src_suffix;
                if (key.find(pattern) != std::string::npos) {

                    size_t start = std::string("model.layers.").size();
                    size_t end = key.find('.', start);
                    std::string layer_str = key.substr(start, end - start);
                    int layer = std::stoi(layer_str);
                    std::string dst_key = dst_prefix + std::to_string(layer);

                    weight_processor_utils::print_processing_info(key, dst_key);

                    py::object tensor = py::reinterpret_borrow<py::object>(item.second);
                    size_t params_count = weight_processor_utils::calculate_params_count(tensor);

                    Tensor<__nv_bfloat16> bf16_tensor = weight_processor_utils::convert_bf16_tensor(tensor);

                    // Check whether a transpose is needed ( from KN convert to NK format )- Only the logical layout is transposed here, without moving the physical data
                    if (transpose_layers.find(dst_prefix) != transpose_layers.end()) {
                        // std::cout << " Logically transpose the weights: " << dst_key << std::endl;
                        cpp_weights.emplace(dst_key, bf16_tensor.transpose(-1, -2));
                    } else {
                        cpp_weights.emplace(dst_key, std::move(bf16_tensor));
                    }
                    break;
                }
            }
        }
    }
}

inline void process_global_weights_awq(const py::dict& weights,
                                       std::unordered_map<std::string, Tensor<__nv_bfloat16>>& cpp_weights) {

    const std::unordered_map<std::string, std::string> global_weights_map = {
        {"model.embed_tokens.weight", "token_embeddings.weight"},
        {"model.norm.weight", "rms_out_w"},
        {"lm_head.weight", "lm_head"}};

    for (const auto& [src_key, dst_key] : global_weights_map) {
        if (weights.contains(src_key)) {
            weight_processor_utils::print_processing_info(src_key, dst_key);
            py::object tensor = weights[src_key.c_str()];

            size_t params_count = weight_processor_utils::calculate_params_count(tensor);

            Tensor<__nv_bfloat16> bf16_tensor = weight_processor_utils::convert_bf16_tensor(tensor);
            if (dst_key == "lm_head") {
                // lm_head weights require a logical transpose, to match matrix multiplication
                auto shape_obj = tensor.attr("shape");
                auto shape_tuple = shape_obj.cast<py::tuple>();
                std::vector<size_t> shape;
                for (size_t i = 0; i < shape_tuple.size(); i++) {
                    shape.push_back(shape_tuple[i].cast<size_t>());
                }

                if (shape.size() >= 2) {

                    // std::cout << " for AWQ lm_head Perform a logical transpose " << std::endl;
                    // for lm_head perform a logical transpose
                    cpp_weights.emplace(dst_key, bf16_tensor.transpose(-1, -2));
                } else {

                    cpp_weights.emplace(dst_key, bf16_tensor.transpose(-1, -2));
                }
            } else {
                cpp_weights.emplace(dst_key, std::move(bf16_tensor));
            }
        }
    }
}

inline void process_quantized_weights_awq(const py::dict& weights,
                                          std::unordered_map<std::string, Tensor<__nv_bfloat16>>& cpp_weights,
                                          std::unordered_map<std::string, Tensor<int32_t>>& cpp_qweight_params,
                                          std::unordered_map<std::string, Tensor<__nv_bfloat16>>& cpp_scales_params,
                                          std::unordered_map<std::string, Tensor<int32_t>>& cpp_qzeros_params) {

    std::unordered_set<std::string> weights_to_transpose;
    std::unordered_map<std::string, std::vector<size_t>> original_shapes;

    std::set<int> all_layers;
    std::set<std::string> all_weight_types = {"wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down"};

    std::set<std::string> found_qweight_keys;
    std::set<std::string> found_scales_keys;
    std::set<std::string> found_qzeros_keys;

    if (verbose_awq_debug()) {
        std::cout << "\n=== AWQ Weight processing started ===" << std::endl;
        std::cout << " All weight keys:" << std::endl;
        for (auto item : weights) {
            std::string key = py::str(item.first).cast<std::string>();
            std::cout << "  " << key << std::endl;
        }
        std::cout << "=== Processing weights ===" << std::endl;
    }

    for (auto item : weights) {
        std::string key = py::str(item.first).cast<std::string>();

        if (key.find("model.layers.") == 0) {

            size_t start = std::string("model.layers.").size();
            size_t end = key.find('.', start);
            if (end != std::string::npos) {
                std::string layer_str = key.substr(start, end - start);
                try {
                    int layer = std::stoi(layer_str);
                    all_layers.insert(layer);
                } catch (const std::exception& e) {
                    std::cerr << " Warning: Cannot parse the layer index: " << layer_str << std::endl;
                }
            }

            if (key.find(".qweight") != std::string::npos) {

                size_t start = std::string("model.layers.").size();
                size_t end = key.find('.', start);
                std::string layer_str = key.substr(start, end - start);
                int layer = std::stoi(layer_str);

                std::string weight_type = key.substr(end + 1);
                weight_type = weight_type.substr(0, weight_type.find(".qweight"));

                std::string dst_prefix;
                if (weight_type == "self_attn.q_proj") {
                    dst_prefix = "wq";
                } else if (weight_type == "self_attn.k_proj") {
                    dst_prefix = "wk";
                } else if (weight_type == "self_attn.v_proj") {
                    dst_prefix = "wv";
                } else if (weight_type == "self_attn.o_proj") {
                    dst_prefix = "wo";
                } else if (weight_type == "mlp.gate_proj") {
                    dst_prefix = "w_gate";
                } else if (weight_type == "mlp.up_proj") {
                    dst_prefix = "w_up";
                } else if (weight_type == "mlp.down_proj") {
                    dst_prefix = "w_down";
                } else {
                    continue;
                }

                std::string dst_key = dst_prefix + std::to_string(layer);
                weight_processor_utils::print_processing_info(key, dst_key);

                found_qweight_keys.insert(dst_key);

                py::object tensor = py::reinterpret_borrow<py::object>(item.second);
                size_t params_count = weight_processor_utils::calculate_params_count(tensor);

                auto np_array = weight_processor_utils::as_contiguous_array<int32_t>(tensor);
                std::vector<size_t> shape;
                for (int i = 0; i < np_array.ndim(); i++) {
                    shape.push_back(np_array.shape(i));
                }

                original_shapes[dst_key] = shape;

                // Mark this weight for transposition ( all linear-layer weights must be transposed from KN convert to NK)
                weights_to_transpose.insert(dst_key);

                std::vector<int32_t> data(np_array.data(), np_array.data() + np_array.size());
                cpp_qweight_params.emplace(dst_key, Tensor<int32_t>(std::move(data), shape));
            }

            else if (key.find(".scales") != std::string::npos) {

                size_t start = std::string("model.layers.").size();
                size_t end = key.find('.', start);
                std::string layer_str = key.substr(start, end - start);
                int layer = std::stoi(layer_str);

                std::string weight_type = key.substr(end + 1);
                weight_type = weight_type.substr(0, weight_type.find(".scales"));

                std::string dst_prefix;
                if (weight_type == "self_attn.q_proj") {
                    dst_prefix = "wq";
                } else if (weight_type == "self_attn.k_proj") {
                    dst_prefix = "wk";
                } else if (weight_type == "self_attn.v_proj") {
                    dst_prefix = "wv";
                } else if (weight_type == "self_attn.o_proj") {
                    dst_prefix = "wo";
                } else if (weight_type == "mlp.gate_proj") {
                    dst_prefix = "w_gate";
                } else if (weight_type == "mlp.up_proj") {
                    dst_prefix = "w_up";
                } else if (weight_type == "mlp.down_proj") {
                    dst_prefix = "w_down";
                } else {
                    continue;
                }

                std::string dst_key = dst_prefix + std::to_string(layer);
                weight_processor_utils::print_processing_info(key, dst_key);

                found_scales_keys.insert(dst_key);

                py::object tensor = py::reinterpret_borrow<py::object>(item.second);
                size_t params_count = weight_processor_utils::calculate_params_count(tensor);

                Tensor<__nv_bfloat16> bf16_tensor = weight_processor_utils::convert_bf16_tensor(tensor);
                cpp_scales_params.emplace(dst_key, std::move(bf16_tensor));
            }

            else if (key.find(".qzeros") != std::string::npos) {

                size_t start = std::string("model.layers.").size();
                size_t end = key.find('.', start);
                std::string layer_str = key.substr(start, end - start);
                int layer = std::stoi(layer_str);

                std::string weight_type = key.substr(end + 1);
                weight_type = weight_type.substr(0, weight_type.find(".qzeros"));

                std::string dst_prefix;
                if (weight_type == "self_attn.q_proj") {
                    dst_prefix = "wq";
                } else if (weight_type == "self_attn.k_proj") {
                    dst_prefix = "wk";
                } else if (weight_type == "self_attn.v_proj") {
                    dst_prefix = "wv";
                } else if (weight_type == "self_attn.o_proj") {
                    dst_prefix = "wo";
                } else if (weight_type == "mlp.gate_proj") {
                    dst_prefix = "w_gate";
                } else if (weight_type == "mlp.up_proj") {
                    dst_prefix = "w_up";
                } else if (weight_type == "mlp.down_proj") {
                    dst_prefix = "w_down";
                } else {
                    continue;
                }

                std::string dst_key = dst_prefix + std::to_string(layer);
                weight_processor_utils::print_processing_info(key, dst_key);

                found_qzeros_keys.insert(dst_key);

                py::object tensor = py::reinterpret_borrow<py::object>(item.second);
                size_t params_count = weight_processor_utils::calculate_params_count(tensor);

                auto np_array = weight_processor_utils::as_contiguous_array<int32_t>(tensor);
                std::vector<size_t> shape;
                for (int i = 0; i < np_array.ndim(); i++) {
                    shape.push_back(np_array.shape(i));
                }
                std::vector<int32_t> data(np_array.data(), np_array.data() + np_array.size());
                cpp_qzeros_params.emplace(dst_key, Tensor<int32_t>(std::move(data), shape));
            }

            else if (key.find("input_layernorm.weight") != std::string::npos ||
                     key.find("post_attention_layernorm.weight") != std::string::npos ||
                     key.find("self_attn.q_norm.weight") != std::string::npos ||
                     key.find("self_attn.k_norm.weight") != std::string::npos) {

                size_t start = std::string("model.layers.").size();
                size_t end = key.find('.', start);
                std::string layer_str = key.substr(start, end - start);
                int layer = std::stoi(layer_str);

                std::string dst_key;
                if (key.find("input_layernorm.weight") != std::string::npos) {
                    dst_key = "rms_att_w" + std::to_string(layer);
                } else if (key.find("post_attention_layernorm.weight") != std::string::npos) {
                    dst_key = "rms_ffn_w" + std::to_string(layer);
                } else if (key.find("self_attn.q_norm.weight") != std::string::npos) {
                    dst_key = "q_norm" + std::to_string(layer);
                } else if (key.find("self_attn.k_norm.weight") != std::string::npos) {
                    dst_key = "k_norm" + std::to_string(layer);
                }

                weight_processor_utils::print_processing_info(key, dst_key);

                py::object tensor = py::reinterpret_borrow<py::object>(item.second);
                size_t params_count = weight_processor_utils::calculate_params_count(tensor);

                Tensor<__nv_bfloat16> bf16_tensor = weight_processor_utils::convert_bf16_tensor(tensor);
                cpp_weights.emplace(dst_key, std::move(bf16_tensor));
            }
        }
    }

    if (verbose_awq_debug()) {
        std::cout << "\n=== Check AWQ Weight completeness ===" << std::endl;
    }

    bool missing_weights = false;
    for (const int& layer : all_layers) {
        for (const std::string& weight_type : all_weight_types) {
            std::string key = weight_type + std::to_string(layer);

            if (found_qweight_keys.find(key) == found_qweight_keys.end()) {
                std::cerr << " Error: Missing required weight " << key << ".qweight" << std::endl;
                missing_weights = true;
            }

            if (found_scales_keys.find(key) == found_scales_keys.end()) {
                std::cerr << " Error: Missing required weight " << key << ".scales" << std::endl;
                missing_weights = true;
            }

            if (found_qzeros_keys.find(key) == found_qzeros_keys.end()) {
                std::cerr << " Error: Missing required weight " << key << ".qzeros" << std::endl;
                missing_weights = true;
            }
        }
    }

    if (missing_weights) {
        std::cerr << "AWQ Weight validation failed: Missing required weights, cannot continue " << std::endl;
        throw std::runtime_error("AWQ Model weights are incomplete, Missing required weights ");
    }

    if (verbose_awq_debug()) {
        std::cout << "\n=== AWQ Weight-processing results ===" << std::endl;
        std::cout << "cpp_qweight_params Key count: " << cpp_qweight_params.size() << std::endl;
        for (const auto& [key, _] : cpp_qweight_params) {
            std::cout << " qweight key: " << key << std::endl;
        }

        std::cout << "cpp_scales_params Key count: " << cpp_scales_params.size() << std::endl;
        for (const auto& [key, _] : cpp_scales_params) {
            std::cout << " scales key: " << key << std::endl;
        }

        std::cout << "cpp_qzeros_params Key count: " << cpp_qzeros_params.size() << std::endl;
        for (const auto& [key, _] : cpp_qzeros_params) {
            std::cout << " qzeros key: " << key << std::endl;
        }

        std::cout << "=== AWQ Weight completeness check complete ===" << std::endl;
    }
}

inline std::unordered_map<std::string, Tensor<__nv_bfloat16>> process_weights_bf16(const py::dict& weights) {
    std::unordered_map<std::string, Tensor<__nv_bfloat16>> cpp_weights;

    size_t total_weights = weights.size();
    weight_processor_utils::init_progress(total_weights, "Qwen3 BF16");

    process_global_weights_bf16(weights, cpp_weights);

    process_layer_weights_bf16(weights, cpp_weights);

    weight_processor_utils::finish_progress();

    if (cpp_weights.find("lm_head") == cpp_weights.end()) {
        std::cout << "\nWarning: lm_head not found in BF16 weights, creating from "
                     "token_embeddings.weight"
                  << std::endl;
        if (cpp_weights.find("token_embeddings.weight") != cpp_weights.end()) {
            Tensor<__nv_bfloat16> lm_head = cpp_weights.at("token_embeddings.weight").transpose(-1, -2);
            cpp_weights.emplace("lm_head", std::move(lm_head));
        }
    }

    return cpp_weights;
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
    weight_processor_utils::init_progress(total_weights, "Qwen3 AWQ");

    process_global_weights_awq(weights, cpp_weights);

    process_quantized_weights_awq(weights, cpp_weights, cpp_qweight_params, cpp_scales_params, cpp_qzeros_params);

    int num_layers = 0;

    for (const auto& [key, _] : cpp_qweight_params) {
        if (key.substr(0, 2) == "wq") {
            int layer = std::stoi(key.substr(2));
            num_layers = std::max(num_layers, layer + 1);
        }
    }

    if (verbose_awq_debug()) {
        std::cout << "\n Detected Qwen3 Model layer count: " << num_layers << std::endl;
    }

    weight_processor_utils::finish_progress();

    if (cpp_weights.find("lm_head") == cpp_weights.end()) {
        std::cout << "\nWarning: lm_head not found in AWQ found in the weights, Trying from token_embeddings.weight Create " << std::endl;
        if (cpp_weights.find("token_embeddings.weight") != cpp_weights.end()) {
            try {
                Tensor<__nv_bfloat16> lm_head = cpp_weights.at("token_embeddings.weight").transpose(-1, -2);
                cpp_weights.emplace("lm_head", std::move(lm_head));
                std::cout << " Successfully from token_embeddings.weight Create lm_head" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << " Create lm_head failed: " << e.what() << std::endl;
            }
        } else {
            std::cerr << "Error: token_embeddings.weight in AWQ also not found in the weights, Cannot create lm_head" << std::endl;
        }
    }

    return {cpp_weights, cpp_qweight_params, cpp_scales_params, cpp_qzeros_params};
}

}  // namespace qwen3_weight_processor
