#pragma once

#include <cstdlib>
#include <iostream>
#include <set>
#include <tuple>

#include "weight_processor_utils.hpp"

// Qwen3 模型权重处理器命名空间
namespace qwen3_weight_processor {

inline bool verbose_awq_debug() {
    const char* value = std::getenv("LLM_INFER_VERBOSE_WEIGHTS");
    return value != nullptr && std::string(value) == "1";
}

// 前向声明所有的辅助函数
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

// 处理 BF16 全局权重
inline void process_global_weights_bf16(const py::dict& weights,
                                        std::unordered_map<std::string, Tensor<__nv_bfloat16>>& cpp_weights) {
    // 全局权重映射
    const std::unordered_map<std::string, std::string> global_weights_map = {
        {"model.embed_tokens.weight", "token_embeddings.weight"},
        {"model.norm.weight", "rms_out_w"},
        {"lm_head.weight", "lm_head"}};

    // 处理非量化的全局权重
    for (const auto& [src_key, dst_key] : global_weights_map) {
        if (weights.contains(src_key)) {
            weight_processor_utils::print_processing_info(src_key, dst_key);
            py::object tensor = weights[src_key.c_str()];

            // 计算参数数量并累加到总数
            size_t params_count = weight_processor_utils::calculate_params_count(tensor);

            Tensor<__nv_bfloat16> bf16_tensor = weight_processor_utils::convert_bf16_tensor(tensor);
            if (dst_key == "lm_head") {
                // lm_head 权重需要逻辑转置，与矩阵乘法操作匹配
                auto shape_obj = tensor.attr("shape");
                auto shape_tuple = shape_obj.cast<py::tuple>();
                std::vector<size_t> shape;
                for (size_t i = 0; i < shape_tuple.size(); i++) {
                    shape.push_back(shape_tuple[i].cast<size_t>());
                }

                if (shape.size() >= 2) {
                    // std::cout << "  lm_head 原始形状: [" << shape[0] << ", " << shape[1] << "]" << std::endl;
                    // std::cout << "  对lm_head执行逻辑转置" << std::endl;
                    // 对lm_head进行逻辑转置，保持KN格式但在访问时被视为NK格式
                    cpp_weights.emplace(dst_key, bf16_tensor.transpose(-1, -2));
                } else {
                    // 如果形状不是2D，也进行转置处理
                    // std::cout << "  lm_head 不是2D张量，执行默认转置" << std::endl;
                    cpp_weights.emplace(dst_key, bf16_tensor.transpose(-1, -2));
                }
            } else {
                cpp_weights.emplace(dst_key, std::move(bf16_tensor));
            }
        }
    }
}

// 处理 BF16 层级权重
inline void process_layer_weights_bf16(const py::dict& weights,
                                       std::unordered_map<std::string, Tensor<__nv_bfloat16>>& cpp_weights) {
    // 层级权重映射
    const std::vector<std::pair<std::string, std::string>> layer_key_mapping = {
        {"input_layernorm.weight", "rms_att_w"}, {"post_attention_layernorm.weight", "rms_ffn_w"},
        {"self_attn.q_proj.weight", "wq"},       {"self_attn.k_proj.weight", "wk"},
        {"self_attn.v_proj.weight", "wv"},       {"self_attn.o_proj.weight", "wo"},
        {"self_attn.q_norm.weight", "q_norm"},   {"self_attn.k_norm.weight", "k_norm"},
        {"mlp.gate_proj.weight", "w_gate"},      {"mlp.up_proj.weight", "w_up"},
        {"mlp.down_proj.weight", "w_down"}};

    // 需要转置的线性层列表（仅进行逻辑转置，不进行物理转置）
    const std::unordered_set<std::string> transpose_layers = {"wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down"};

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
                    std::string dst_key = dst_prefix + std::to_string(layer);

                    weight_processor_utils::print_processing_info(key, dst_key);

                    // 计算参数数量并累加到总数
                    py::object tensor = py::reinterpret_borrow<py::object>(item.second);
                    size_t params_count = weight_processor_utils::calculate_params_count(tensor);

                    // 转换为bf16张量
                    Tensor<__nv_bfloat16> bf16_tensor = weight_processor_utils::convert_bf16_tensor(tensor);

                    // 检查是否需要转置（从KN转为NK格式）- 这里只进行逻辑转置，没有物理数据移动
                    if (transpose_layers.find(dst_prefix) != transpose_layers.end()) {
                        // std::cout << "  对权重进行逻辑转置: " << dst_key << std::endl;
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

// 处理 AWQ 全局权重
inline void process_global_weights_awq(const py::dict& weights,
                                       std::unordered_map<std::string, Tensor<__nv_bfloat16>>& cpp_weights) {
    // 全局权重映射
    const std::unordered_map<std::string, std::string> global_weights_map = {
        {"model.embed_tokens.weight", "token_embeddings.weight"},
        {"model.norm.weight", "rms_out_w"},
        {"lm_head.weight", "lm_head"}};

    // 处理非量化的全局权重
    for (const auto& [src_key, dst_key] : global_weights_map) {
        if (weights.contains(src_key)) {
            weight_processor_utils::print_processing_info(src_key, dst_key);
            py::object tensor = weights[src_key.c_str()];

            // 计算参数数量并累加到总数
            size_t params_count = weight_processor_utils::calculate_params_count(tensor);

            Tensor<__nv_bfloat16> bf16_tensor = weight_processor_utils::convert_bf16_tensor(tensor);
            if (dst_key == "lm_head") {
                // lm_head 权重需要逻辑转置，与矩阵乘法操作匹配
                auto shape_obj = tensor.attr("shape");
                auto shape_tuple = shape_obj.cast<py::tuple>();
                std::vector<size_t> shape;
                for (size_t i = 0; i < shape_tuple.size(); i++) {
                    shape.push_back(shape_tuple[i].cast<size_t>());
                }

                // 检查并显示形状信息
                if (shape.size() >= 2) {
                    // std::cout << "  AWQ lm_head 原始形状: [" << shape[0] << ", " << shape[1] << "]" << std::endl;
                    // std::cout << "  对AWQ lm_head执行逻辑转置" << std::endl;
                    // 对lm_head进行逻辑转置
                    cpp_weights.emplace(dst_key, bf16_tensor.transpose(-1, -2));
                } else {
                    // 如果形状不是2D，也进行转置处理
                    // std::cout << "  AWQ lm_head 不是2D张量，执行默认转置" << std::endl;
                    cpp_weights.emplace(dst_key, bf16_tensor.transpose(-1, -2));
                }
            } else {
                cpp_weights.emplace(dst_key, std::move(bf16_tensor));
            }
        }
    }
}

// 处理 AWQ 量化权重
inline void process_quantized_weights_awq(const py::dict& weights,
                                          std::unordered_map<std::string, Tensor<__nv_bfloat16>>& cpp_weights,
                                          std::unordered_map<std::string, Tensor<int32_t>>& cpp_qweight_params,
                                          std::unordered_map<std::string, Tensor<__nv_bfloat16>>& cpp_scales_params,
                                          std::unordered_map<std::string, Tensor<int32_t>>& cpp_qzeros_params) {
    // 添加处理变量：标记哪些权重需要转置
    // 这里我们先收集所有权重，处理完后再进行转置
    std::unordered_set<std::string> weights_to_transpose;
    std::unordered_map<std::string, std::vector<size_t>> original_shapes;

    // 收集所有层索引和权重类型，用于后续检查缺失的权重
    std::set<int> all_layers;
    std::set<std::string> all_weight_types = {"wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down"};

    // 收集所有已找到的权重键，用于检测缺失的权重
    std::set<std::string> found_qweight_keys;
    std::set<std::string> found_scales_keys;
    std::set<std::string> found_qzeros_keys;

    // 调试信息：列出所有权重键
    if (verbose_awq_debug()) {
        std::cout << "\n=== AWQ权重处理开始 ===" << std::endl;
        std::cout << "所有权重键名：" << std::endl;
        for (auto item : weights) {
            std::string key = py::str(item.first).cast<std::string>();
            std::cout << "  " << key << std::endl;
        }
        std::cout << "=== 开始处理权重 ===" << std::endl;
    }

    for (auto item : weights) {
        std::string key = py::str(item.first).cast<std::string>();

        // 处理层级权重
        if (key.find("model.layers.") == 0) {
            // 提取层索引，不论权重类型
            size_t start = std::string("model.layers.").size();
            size_t end = key.find('.', start);
            if (end != std::string::npos) {
                std::string layer_str = key.substr(start, end - start);
                try {
                    int layer = std::stoi(layer_str);
                    all_layers.insert(layer);
                } catch (const std::exception& e) {
                    std::cerr << "警告: 无法解析层索引: " << layer_str << std::endl;
                }
            }

            // 处理量化权重
            if (key.find(".qweight") != std::string::npos) {
                // 提取层索引和权重名称
                size_t start = std::string("model.layers.").size();
                size_t end = key.find('.', start);
                std::string layer_str = key.substr(start, end - start);
                int layer = std::stoi(layer_str);

                // 提取权重类型
                std::string weight_type = key.substr(end + 1);
                weight_type = weight_type.substr(0, weight_type.find(".qweight"));

                // 映射权重类型到目标键名前缀
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
                    continue;  // 跳过未知权重类型
                }

                std::string dst_key = dst_prefix + std::to_string(layer);
                weight_processor_utils::print_processing_info(key, dst_key);

                // 记录已找到的qweight键
                found_qweight_keys.insert(dst_key);

                // 处理量化权重
                py::object tensor = py::reinterpret_borrow<py::object>(item.second);
                size_t params_count = weight_processor_utils::calculate_params_count(tensor);

                // 转换为int32张量
                py::array_t<int32_t> np_array = tensor.cast<py::array_t<int32_t>>();
                std::vector<size_t> shape;
                for (int i = 0; i < np_array.ndim(); i++) {
                    shape.push_back(np_array.shape(i));
                }

                // 记录原始形状，稍后用于转置
                original_shapes[dst_key] = shape;

                // 标记此权重需要转置（所有线性层权重都需要从KN转为NK）
                weights_to_transpose.insert(dst_key);

                std::vector<int32_t> data(np_array.data(), np_array.data() + np_array.size());
                cpp_qweight_params.emplace(dst_key, Tensor<int32_t>(std::move(data), shape));
            }

            // 处理缩放因子
            else if (key.find(".scales") != std::string::npos) {
                // 提取层索引和权重名称
                size_t start = std::string("model.layers.").size();
                size_t end = key.find('.', start);
                std::string layer_str = key.substr(start, end - start);
                int layer = std::stoi(layer_str);

                // 提取权重类型
                std::string weight_type = key.substr(end + 1);
                weight_type = weight_type.substr(0, weight_type.find(".scales"));

                // 映射权重类型到目标键名前缀
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
                    continue;  // 跳过未知权重类型
                }

                std::string dst_key = dst_prefix + std::to_string(layer);
                weight_processor_utils::print_processing_info(key, dst_key);

                // 记录已找到的scales键
                found_scales_keys.insert(dst_key);

                // 处理缩放因子
                py::object tensor = py::reinterpret_borrow<py::object>(item.second);
                size_t params_count = weight_processor_utils::calculate_params_count(tensor);

                // 转换为bf16张量
                Tensor<__nv_bfloat16> bf16_tensor = weight_processor_utils::convert_bf16_tensor(tensor);
                cpp_scales_params.emplace(dst_key, std::move(bf16_tensor));
            }

            // 处理量化零点
            else if (key.find(".qzeros") != std::string::npos) {
                // 提取层索引和权重名称
                size_t start = std::string("model.layers.").size();
                size_t end = key.find('.', start);
                std::string layer_str = key.substr(start, end - start);
                int layer = std::stoi(layer_str);

                // 提取权重类型
                std::string weight_type = key.substr(end + 1);
                weight_type = weight_type.substr(0, weight_type.find(".qzeros"));

                // 映射权重类型到目标键名前缀
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
                    continue;  // 跳过未知权重类型
                }

                std::string dst_key = dst_prefix + std::to_string(layer);
                weight_processor_utils::print_processing_info(key, dst_key);

                // 记录已找到的qzeros键
                found_qzeros_keys.insert(dst_key);

                // 处理量化零点
                py::object tensor = py::reinterpret_borrow<py::object>(item.second);
                size_t params_count = weight_processor_utils::calculate_params_count(tensor);

                // 转换为int32张量
                py::array_t<int32_t> np_array = tensor.cast<py::array_t<int32_t>>();
                std::vector<size_t> shape;
                for (int i = 0; i < np_array.ndim(); i++) {
                    shape.push_back(np_array.shape(i));
                }
                std::vector<int32_t> data(np_array.data(), np_array.data() + np_array.size());
                cpp_qzeros_params.emplace(dst_key, Tensor<int32_t>(std::move(data), shape));
            }

            // 处理非量化层级权重
            else if (key.find("input_layernorm.weight") != std::string::npos ||
                     key.find("post_attention_layernorm.weight") != std::string::npos ||
                     key.find("self_attn.q_norm.weight") != std::string::npos ||
                     key.find("self_attn.k_norm.weight") != std::string::npos) {
                // 提取层索引
                size_t start = std::string("model.layers.").size();
                size_t end = key.find('.', start);
                std::string layer_str = key.substr(start, end - start);
                int layer = std::stoi(layer_str);

                // 构造目标键名
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

                // 处理非量化层级权重
                py::object tensor = py::reinterpret_borrow<py::object>(item.second);
                size_t params_count = weight_processor_utils::calculate_params_count(tensor);

                // 转换为bf16张量
                Tensor<__nv_bfloat16> bf16_tensor = weight_processor_utils::convert_bf16_tensor(tensor);
                cpp_weights.emplace(dst_key, std::move(bf16_tensor));
            }
        }
    }

    // 检查并处理缺失的权重
    if (verbose_awq_debug()) {
        std::cout << "\n=== 检查AWQ权重完整性 ===" << std::endl;
    }

    // 检查所有层是否存在所有必要的权重
    bool missing_weights = false;
    for (const int& layer : all_layers) {
        for (const std::string& weight_type : all_weight_types) {
            std::string key = weight_type + std::to_string(layer);

            if (found_qweight_keys.find(key) == found_qweight_keys.end()) {
                std::cerr << "错误: 缺少关键权重 " << key << ".qweight" << std::endl;
                missing_weights = true;
            }

            if (found_scales_keys.find(key) == found_scales_keys.end()) {
                std::cerr << "错误: 缺少关键权重 " << key << ".scales" << std::endl;
                missing_weights = true;
            }

            if (found_qzeros_keys.find(key) == found_qzeros_keys.end()) {
                std::cerr << "错误: 缺少关键权重 " << key << ".qzeros" << std::endl;
                missing_weights = true;
            }
        }
    }

    if (missing_weights) {
        std::cerr << "AWQ权重检查失败: 缺少必要权重，无法继续" << std::endl;
        throw std::runtime_error("AWQ模型权重不完整，缺少必要权重");
    }

    // 添加处理权重结果的调试信息
    if (verbose_awq_debug()) {
        std::cout << "\n=== AWQ权重处理后信息 ===" << std::endl;
        std::cout << "cpp_qweight_params 键名数量: " << cpp_qweight_params.size() << std::endl;
        for (const auto& [key, _] : cpp_qweight_params) {
            std::cout << "  qweight键: " << key << std::endl;
        }

        std::cout << "cpp_scales_params 键名数量: " << cpp_scales_params.size() << std::endl;
        for (const auto& [key, _] : cpp_scales_params) {
            std::cout << "  scales键: " << key << std::endl;
        }

        std::cout << "cpp_qzeros_params 键名数量: " << cpp_qzeros_params.size() << std::endl;
        for (const auto& [key, _] : cpp_qzeros_params) {
            std::cout << "  qzeros键: " << key << std::endl;
        }

        std::cout << "=== AWQ权重完整性检查完成 ===" << std::endl;
    }
}

// 处理 Qwen3 模型权重（BF16）
inline std::unordered_map<std::string, Tensor<__nv_bfloat16>> process_weights_bf16(const py::dict& weights) {
    std::unordered_map<std::string, Tensor<__nv_bfloat16>> cpp_weights;

    // 初始化进度条
    size_t total_weights = weights.size();
    weight_processor_utils::init_progress(total_weights, "Qwen3 BF16");

    // 处理全局权重
    process_global_weights_bf16(weights, cpp_weights);

    // 处理层级权重
    process_layer_weights_bf16(weights, cpp_weights);

    // 完成进度条
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

// 处理 Qwen3 AWQ 量化权重
inline std::tuple<
    std::unordered_map<std::string, Tensor<__nv_bfloat16>>, std::unordered_map<std::string, Tensor<int32_t>>,
    std::unordered_map<std::string, Tensor<__nv_bfloat16>>, std::unordered_map<std::string, Tensor<int32_t>>>
process_weights_awq(const py::dict& weights) {
    std::unordered_map<std::string, Tensor<__nv_bfloat16>> cpp_weights;
    std::unordered_map<std::string, Tensor<int32_t>> cpp_qweight_params;
    std::unordered_map<std::string, Tensor<__nv_bfloat16>> cpp_scales_params;
    std::unordered_map<std::string, Tensor<int32_t>> cpp_qzeros_params;

    // 初始化进度条
    size_t total_weights = weights.size();
    weight_processor_utils::init_progress(total_weights, "Qwen3 AWQ");

    // 处理全局权重
    process_global_weights_awq(weights, cpp_weights);

    // 处理量化权重
    process_quantized_weights_awq(weights, cpp_weights, cpp_qweight_params, cpp_scales_params, cpp_qzeros_params);

    // 检查是否所有层都有wq、wk、wv、wo等权重
    int num_layers = 0;
    // 寻找最大层号
    for (const auto& [key, _] : cpp_qweight_params) {
        if (key.substr(0, 2) == "wq") {
            int layer = std::stoi(key.substr(2));
            num_layers = std::max(num_layers, layer + 1);
        }
    }

    if (verbose_awq_debug()) {
        std::cout << "\n检测到Qwen3模型层数: " << num_layers << std::endl;
    }

    // 完成进度条
    weight_processor_utils::finish_progress();

    // 如果 lm_head 缺失，则尝试从 token_embeddings.weight 转置生成
    if (cpp_weights.find("lm_head") == cpp_weights.end()) {
        std::cout << "\nWarning: lm_head 未在AWQ权重中找到，尝试从token_embeddings.weight创建" << std::endl;
        if (cpp_weights.find("token_embeddings.weight") != cpp_weights.end()) {
            try {
                Tensor<__nv_bfloat16> lm_head = cpp_weights.at("token_embeddings.weight").transpose(-1, -2);
                cpp_weights.emplace("lm_head", std::move(lm_head));
                std::cout << "成功从token_embeddings.weight创建lm_head" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "创建lm_head时出错: " << e.what() << std::endl;
            }
        } else {
            std::cerr << "Error: token_embeddings.weight在AWQ权重中也未找到，无法创建lm_head" << std::endl;
        }
    }

    return {cpp_weights, cpp_qweight_params, cpp_scales_params, cpp_qzeros_params};
}

}  // namespace qwen3_weight_processor
