#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "base_model.hpp"
#include "qwen.hpp"
#include "qwen3.hpp"

enum class ModelType {
    LLAMA,
    QWEN,
    QWEN_BF16,
    QWEN_AWQ,
    QWEN3_BF16,
    QWEN3_AWQ,
};

inline ModelType model_type_from_string(const std::string& type_str) {
    if (type_str == "llama") {
        return ModelType::LLAMA;
    } else if (type_str == "qwen") {
        return ModelType::QWEN;
    } else if (type_str == "qwen_bf16") {
        return ModelType::QWEN_BF16;
    } else if (type_str == "qwen_awq") {
        return ModelType::QWEN_AWQ;
    } else if (type_str == "qwen3_bf16") {
        return ModelType::QWEN3_BF16;
    } else if (type_str == "qwen3_awq") {
        return ModelType::QWEN3_AWQ;
    } else {
        throw std::runtime_error("Unsupported model type: " + type_str);
    }
}

class ModelFactory {
   public:

    static std::shared_ptr<BaseModel> create_model(ModelType type,
                                                   const std::unordered_map<std::string, Tensor<float>>& weights,
                                                   const ModelConfig& config) {
        switch (type) {
            case ModelType::LLAMA:
            case ModelType::QWEN: {
                auto model = std::make_shared<QwenModel<float>>(weights, config);
                model->print_model_info();
                if (!model->verify_params()) {
                    throw std::runtime_error("Model parameter verification failed");
                }
                return model;
            }
            default:
                throw std::runtime_error("Unsupported model type for FP32 weights in create_model");
        }
    }

    static std::shared_ptr<BaseModel> create_model_bf16(
        ModelType type, const std::unordered_map<std::string, Tensor<__nv_bfloat16>>& weights,
        const ModelConfig& config) {
        switch (type) {
            case ModelType::QWEN_BF16: {
                auto model = std::make_shared<QwenModel<__nv_bfloat16>>(weights, config);

                model->print_model_info();
                if (!model->verify_params()) {
                    throw std::runtime_error("Model parameter verification failed");
                }
                return model;
            }
            case ModelType::QWEN3_BF16: {
                auto model = std::make_shared<Qwen3Session<__nv_bfloat16>>(weights, config);
                model->print_model_info();
                if (!model->verify_params()) {
                    throw std::runtime_error("Model parameter verification failed");
                }
                return model;
            }
            default:
                throw std::runtime_error("Unsupported model type for BF16 weights in create_model_bf16");
        }
    }

    static std::shared_ptr<BaseModel> create_model_quantized(
        ModelType type, const std::unordered_map<std::string, Tensor<__nv_bfloat16>>& weights,
        const std::unordered_map<std::string, Tensor<int32_t>>& qweight_params,
        const std::unordered_map<std::string, Tensor<__nv_bfloat16>>& scales_params,
        const std::unordered_map<std::string, Tensor<int32_t>>& qzeros_params,
        const ModelConfig& config) {
        switch (type) {
            case ModelType::QWEN_AWQ: {
                auto model = std::make_shared<QwenModel<__nv_bfloat16>>(weights, qweight_params, scales_params,
                                                                        qzeros_params, config);
                model->print_model_info();
                if (!model->verify_params()) {
                    throw std::runtime_error("Model parameter verification failed");
                }
                return model;
            }
            case ModelType::QWEN3_AWQ: {
                auto model = std::make_shared<Qwen3Session<__nv_bfloat16>>(weights, qweight_params, scales_params,
                                                                         qzeros_params, config);
                model->print_model_info();
                if (!model->verify_params()) {
                    throw std::runtime_error("Model parameter verification failed");
                }
                return model;
            }
            default:
                throw std::runtime_error(
                    "Unsupported model type for quantized weights in "
                    "create_model_quantized");
        }
    }
};
