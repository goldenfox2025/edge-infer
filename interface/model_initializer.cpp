#include "model_initializer.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include "CudaMemoryPool.hpp"
#include "device_manager.hpp"
#include "include/weight_processor.hpp"
#include "model_factory.hpp"

namespace {

void populate_qwen_family_config(ModelConfig& cpp_config,
                                 py::dict config, bool include_head_dim = false) {
    cpp_config["n_layers"] = config["num_hidden_layers"].cast<int>();
    cpp_config["n_heads"] = config["num_attention_heads"].cast<int>();
    cpp_config["n_kv_heads"] = config["num_key_value_heads"].cast<int>();
    cpp_config["intermediate_size"] = config["intermediate_size"].cast<int>();
    cpp_config["rms_norm_eps"] = config["rms_norm_eps"].cast<double>();
    cpp_config["rope_theta"] = config["rope_theta"].cast<double>();

    if (include_head_dim && config.contains("head_dim")) {
        cpp_config["head_dim"] = config["head_dim"].cast<int>();
        std::cout << "使用配置中的head_dim: " << cpp_config["head_dim"] << std::endl;
    }
}

Device resolve_device_for_cuda_only_model(const char* model_name) {
    Device default_device = DeviceManager::instance().getDefaultDevice();
    if (default_device == Device::CPU) {
        std::cerr << "Warning: " << model_name
                  << " currently only supports CUDA execution. "
                  << "Forcing CUDA device despite CPU being requested." << std::endl;
        return Device::CUDA;
    }
    return default_device;
}

int resolve_group_size(py::dict config, int fallback = 128) {
    return config.contains("group_size") ? config["group_size"].cast<int>() : fallback;
}

ModelConfig build_decoder_family_config(py::dict config,
                                        bool include_head_dim = false) {
    ModelConfig cpp_config = ModelInitializer::build_base_config(config);
    populate_qwen_family_config(cpp_config, config, include_head_dim);
    return cpp_config;
}

void apply_default_device(std::shared_ptr<BaseModel>& model, Device device) {
    if (device == Device::CUDA) {
        model->cuda();
    } else {
        model->cpu();
    }
}

template <typename WeightLoader>
bool init_fp32_decoder_family_model(py::dict config, py::dict weights, ModelType type, const char* error_label,
                                    WeightLoader&& weight_loader, std::shared_ptr<BaseModel>& model,
                                    std::unique_ptr<infer_base>& engine) {
    try {
        ModelConfig cpp_config = build_decoder_family_config(config);
        auto cpp_weights_fp32 = weight_loader(weights);

        Device default_device = DeviceManager::instance().getDefaultDevice();
        model = ModelFactory::create_model(type, cpp_weights_fp32, cpp_config);
        apply_default_device(model, default_device);
        engine = std::make_unique<InferenceEngine<float>>(model, default_device);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error initializing " << error_label << ": " << e.what() << std::endl;
        return false;
    }
}

template <typename WeightLoader>
bool init_bf16_cuda_only_model(py::dict config, py::dict weights, ModelType type, const char* model_name,
                               const char* error_label, WeightLoader&& weight_loader,
                               std::shared_ptr<BaseModel>& model, std::unique_ptr<infer_base>& engine,
                               bool include_head_dim = false) {
    try {
        ModelConfig cpp_config = build_decoder_family_config(config, include_head_dim);
        auto cpp_weights_bf16 = weight_loader(weights);

        Device default_device = resolve_device_for_cuda_only_model(model_name);
        model = ModelFactory::create_model_bf16(type, cpp_weights_bf16, cpp_config);
        engine = std::make_unique<InferenceEngine<__nv_bfloat16>>(model, default_device);
        model->cuda();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error initializing " << error_label << ": " << e.what() << std::endl;
        return false;
    }
}

template <typename WeightLoader>
bool init_awq_cuda_only_model(py::dict config, py::dict weights, ModelType type, const char* model_name,
                              const char* error_label, WeightLoader&& weight_loader,
                              std::shared_ptr<BaseModel>& model, std::unique_ptr<infer_base>& engine,
                              bool include_head_dim = false) {
    try {
        ModelConfig cpp_config = build_decoder_family_config(config, include_head_dim);
        cpp_config["quant_type"] = 1;
        cpp_config["group_size"] = resolve_group_size(config);

        auto [bf16_weights, qweight_params, scales_params, qzeros_params] =
            weight_loader(weights);

        Device default_device = resolve_device_for_cuda_only_model(model_name);
        model = ModelFactory::create_model_quantized(type, bf16_weights, qweight_params, scales_params,
                                                     qzeros_params, cpp_config);
        engine = std::make_unique<InferenceEngine<__nv_bfloat16>>(model, default_device);
        model->cuda();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error initializing " << error_label << ": " << e.what() << std::endl;
        return false;
    }
}

}  // namespace

// 打印配置和权重信息
void ModelInitializer::print_config_and_weights_info(py::dict config, py::dict weights) {
    const char* verbose = std::getenv("LLM_INFER_VERBOSE_WEIGHTS");
    if (!(verbose != nullptr && std::string(verbose) == "1")) {
        return;
    }

    std::cout << "\n===== Configuration Items =====" << std::endl;
    for (const auto& item : config) {
        std::string key = py::str(item.first).cast<std::string>();
        std::string type_str = py::str(item.second.get_type()).cast<std::string>();
        std::cout << "Config key: " << key << ", type: " << type_str << std::endl;
    }
    std::cout << "\n===== Weight Items =====" << std::endl;
    for (const auto& item : weights) {
        std::string key = py::str(item.first).cast<std::string>();
        std::string type_str = py::str(item.second.get_type()).cast<std::string>();
        std::cout << "Weight key: " << key << ", type: " << type_str << std::endl;
    }
}

// 构建基础配置
ModelConfig ModelInitializer::build_base_config(py::dict config) {
    ModelConfig cpp_config;
    cpp_config["vocab_size"] = config["vocab_size"].cast<int>();
    cpp_config["hidden_size"] = config["hidden_size"].cast<int>();
    cpp_config["max_position_embeddings"] = config["max_position_embeddings"].cast<int>();
    cpp_config["bos_token_id"] = config["bos_token_id"].cast<int>();
    cpp_config["eos_token_id"] = config["eos_token_id"].cast<int>();
    return cpp_config;
}

// 初始化 Llama 模型
bool ModelInitializer::init_llama_model(py::dict config, py::dict weights, std::shared_ptr<BaseModel>& model,
                                        std::unique_ptr<infer_base>& engine) {
    return init_fp32_decoder_family_model(config, weights, ModelType::LLAMA, "Llama model",
                                          weight_processor::process_llama_weights, model, engine);
}

// 初始化 Qwen FP32 模型
bool ModelInitializer::init_qwen_fp32_model(py::dict config, py::dict weights, std::shared_ptr<BaseModel>& model,
                                            std::unique_ptr<infer_base>& engine) {
    return init_fp32_decoder_family_model(config, weights, ModelType::QWEN, "Qwen FP32 model",
                                          weight_processor::process_qwen_weights_fp32, model, engine);
}

// 初始化 Qwen BF16 模型
bool ModelInitializer::init_qwen_bf16_model(py::dict config, py::dict weights, std::shared_ptr<BaseModel>& model,
                                            std::unique_ptr<infer_base>& engine) {
    return init_bf16_cuda_only_model(config, weights, ModelType::QWEN_BF16, "Qwen BF16 model",
                                     "Qwen BF16 model", weight_processor::process_qwen_weights_bf16, model, engine);
}

// 初始化 Qwen AWQ 模型
bool ModelInitializer::init_qwen_awq_model(py::dict config, py::dict weights, std::shared_ptr<BaseModel>& model,
                                           std::unique_ptr<infer_base>& engine) {
    return init_awq_cuda_only_model(config, weights, ModelType::QWEN_AWQ, "Qwen AWQ model", "Qwen AWQ model",
                                    weight_processor::process_qwen_weights_awq, model, engine);
}

// 初始化 Qwen3 BF16 模型
bool ModelInitializer::init_qwen3_bf16_model(py::dict config, py::dict weights, std::shared_ptr<BaseModel>& model,
                                             std::unique_ptr<infer_base>& engine) {
    return init_bf16_cuda_only_model(config, weights, ModelType::QWEN3_BF16, "Qwen3 BF16 model",
                                     "Qwen3 BF16 model", weight_processor::process_qwen3_weights_bf16, model, engine,
                                     true);
}

// 初始化 Qwen3 AWQ 模型
bool ModelInitializer::init_qwen3_awq_model(py::dict config, py::dict weights, std::shared_ptr<BaseModel>& model,
                                            std::unique_ptr<infer_base>& engine) {
    return init_awq_cuda_only_model(config, weights, ModelType::QWEN3_AWQ, "Qwen3 AWQ model", "Qwen3 AWQ model",
                                    weight_processor::process_qwen3_weights_awq, model, engine, true);
}

// 初始化 CUDA 内存池
bool ModelInitializer::init_cuda_memory_pool(const ModelConfig& config) {
    try {
        // 获取模型配置参数
        size_t hidden_dim = config.at("hidden_size");

        // 获取最大序列长度，如果配置中有的话
        size_t seq_len = 32;  // 默认序列长度
        if (config.find("max_position_embeddings") != config.end()) {
            seq_len = std::min(seq_len, static_cast<size_t>(config.at("max_position_embeddings")));
        }

        // 检查GPU内存状态
        size_t free_memory = 0, total_memory = 0;
        cudaError_t err = cudaMemGetInfo(&free_memory, &total_memory);
        if (err == cudaSuccess) {
            std::cout << "Current GPU memory: " << (free_memory / (1024 * 1024)) << " MB free, "
                      << (total_memory / (1024 * 1024)) << " MB total" << std::endl;

            if (free_memory > 1024 * 1024 * 1024) {
                const size_t prefill_max_size =
                    std::min(static_cast<size_t>(1024 * 1024 * 1024), free_memory * 3 / 4);
                const size_t prefill_size =
                    std::min(prefill_max_size, static_cast<size_t>(256 * 1024 * 1024));

                // 开启prefill模式
                std::cout << "Enabling prefill mode with initial buffer size: "
                          << (prefill_size / (1024 * 1024)) << " MB, max reserved size: "
                          << (prefill_max_size / (1024 * 1024)) << " MB" << std::endl;
                GlobalCudaMemoryPool::enable_prefill_mode(prefill_size, prefill_max_size);
            } else {
                std::cout << "Skipping prefill mode due to low available memory" << std::endl;
            }
        }
        return true;
    } catch (const std::exception& e) {
        // 如果prefill模式初始化失败，只打印警告，不影响模型加载
        std::cerr << "Warning: Failed to initialize prefill mode: " << e.what() << std::endl;
        return false;
    }
}
