#include "model_initializer.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include "device_manager.hpp"
#include "include/weight_processor.hpp"

namespace {

ModelConfig build_decoder_config(py::dict config) {
  if (config.contains("hidden_act") &&
      config["hidden_act"].cast<std::string>() != "silu") {
    throw std::invalid_argument("Only SiLU-gated decoder MLPs are supported");
  }
  if (config.contains("rope_scaling") && !config["rope_scaling"].is_none()) {
    const py::object scaling = config["rope_scaling"];
    if (!py::isinstance<py::dict>(scaling) || py::len(scaling) != 0) {
      throw std::invalid_argument("Scaled or multi-axis RoPE is not supported by text models");
    }
  }
  const bool sliding_enabled =
      config.contains("use_sliding_window") &&
      config["use_sliding_window"].cast<bool>();
  const bool implicit_sliding =
      !config.contains("use_sliding_window") &&
      config.contains("sliding_window") &&
      !config["sliding_window"].is_none() &&
      config["sliding_window"].cast<long long>() > 0;
  if (sliding_enabled || implicit_sliding) {
    throw std::invalid_argument("Sliding-window attention is not supported");
  }
  if (config.contains("layer_types") && !config["layer_types"].is_none()) {
    for (const auto& layer : config["layer_types"]) {
      if (py::cast<std::string>(layer) != "full_attention") {
        throw std::invalid_argument("Only full-attention decoder layers are supported");
      }
    }
  }
  ModelConfig result = ModelInitializer::build_base_config(config);
  result["n_layers"] = config["num_hidden_layers"].cast<int>();
  result["n_heads"] = config["num_attention_heads"].cast<int>();
  result["n_kv_heads"] = config["num_key_value_heads"].cast<int>();
  result["intermediate_size"] = config["intermediate_size"].cast<int>();
  result["rms_norm_eps"] = config["rms_norm_eps"].cast<double>();
  result["rope_theta"] = config["rope_theta"].cast<double>();
  if (config.contains("head_dim")) {
    result["head_dim"] = config["head_dim"].cast<int>();
  }
  return result;
}

}  // namespace

std::shared_ptr<BaseModel> ModelInitializer::prepare_model(
    py::dict config, py::dict weights, ModelType type, Device device) {
  // Resolve declared model semantics once, without mutating the caller's dict.
  // Weight processors only map names, casts and layouts after this admission.
  auto cpp_config = build_decoder_config(config);
  const auto n_layers = config["num_hidden_layers"].cast<size_t>();
  weights = decoder_weight_mapper::admit(config, weights);
  const bool awq = type == ModelType::QWEN_AWQ || type == ModelType::QWEN3_AWQ;
  if (awq) {
    cpp_config["quant_type"] = 1;
    cpp_config["group_size"] = decoder_weight_mapper::awq_group_size(config);
  }
  if (device == Device::CUDA && !DeviceManager::instance().isCudaAvailable()) {
    throw std::runtime_error("CUDA requested but no CUDA device is available");
  }
  const bool fp32 = type == ModelType::LLAMA || type == ModelType::QWEN;
  if (!fp32 && device != Device::CUDA) {
    throw std::invalid_argument("BF16 and AWQ models require CUDA");
  }

  std::shared_ptr<BaseModel> model;
  switch (type) {
    case ModelType::LLAMA:
      model = ModelFactory::create_model(
          type, weight_processor::process_llama_weights(weights, n_layers), cpp_config);
      break;
    case ModelType::QWEN:
      model = ModelFactory::create_model(
          type, weight_processor::process_qwen_weights_fp32(weights, n_layers), cpp_config);
      break;
    case ModelType::QWEN_BF16:
      model = ModelFactory::create_model_bf16(
          type, weight_processor::process_qwen_weights_bf16(weights, n_layers), cpp_config);
      break;
    case ModelType::QWEN3_BF16:
      model = ModelFactory::create_model_bf16(
          type, weight_processor::process_qwen3_weights_bf16(weights, n_layers), cpp_config);
      break;
    case ModelType::QWEN_AWQ:
    case ModelType::QWEN3_AWQ: {
      auto [bf16, quantized, scales, zeros] =
          type == ModelType::QWEN3_AWQ
              ? weight_processor::process_qwen3_weights_awq(weights, n_layers)
              : weight_processor::process_qwen_weights_awq(weights, n_layers);
      model = ModelFactory::create_model_quantized(
          type, bf16, quantized, scales, zeros, cpp_config);
      break;
    }
    default:
      throw std::invalid_argument("Unsupported model type");
  }
  if (device == Device::CUDA) {
    model->cuda();
  } else {
    model->cpu();
  }
  return model;
}

void ModelInitializer::print_config_and_weights_info(py::dict config,
                                                    py::dict weights) {
  const char* verbose = std::getenv("EDGE_INFER_VERBOSE_WEIGHTS");
  if (!verbose || std::string(verbose) != "1") return;
  std::cout << "\n===== Configuration Items =====" << std::endl;
  for (const auto& item : config) {
    std::cout << "Config key: " << py::str(item.first).cast<std::string>()
              << std::endl;
  }
  std::cout << "\n===== Weight Items =====" << std::endl;
  for (const auto& item : weights) {
    std::cout << "Weight key: " << py::str(item.first).cast<std::string>()
              << std::endl;
  }
}

ModelConfig ModelInitializer::build_base_config(py::dict config) {
  ModelConfig result;
  result["vocab_size"] = config["vocab_size"].cast<int>();
  result["hidden_size"] = config["hidden_size"].cast<int>();
  result["max_position_embeddings"] =
      config["max_position_embeddings"].cast<int>();
  result["bos_token_id"] = config["bos_token_id"].cast<int>();
  result["eos_token_id"] = config["eos_token_id"].cast<int>();
  return result;
}
