#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "base_model.hpp"
#include "execution/cuda_workspace_arena.hpp"
#include "weight_tensor.hpp"

struct Qwen3Config {
    size_t vocab_size;
    size_t n_layers;
    size_t n_heads;
    size_t n_kv_heads;
    size_t hidden_size;
    size_t head_dim;
    size_t intermediate_size;
    size_t max_position_embeddings;
    uint32_t bos_token_id;
    uint32_t eos_token_id;
    float rms_norm_eps;
    float rope_theta;
    int quant_type;
    int group_size;
};

// A prepared CUDA model. Execution state belongs to Qwen3Session, never here.
// Weight views refer to private owned storage, copied once during construction.
template <typename T>
class Qwen3Model {
 public:
    using Parameters = std::unordered_map<std::string, Tensor<T>>;
    using IntegerParameters = std::unordered_map<std::string, Tensor<int32_t>>;

    struct Layer {
        op::WeightTensor<T> q, k, v, o, gate, up, down;
        const Tensor<T>* attention_norm;
        const Tensor<T>* ffn_norm;
        const Tensor<T>* q_norm;
        const Tensor<T>* k_norm;
        const Tensor<T>* q_bias;
        const Tensor<T>* k_bias;
        const Tensor<T>* v_bias;
        const Tensor<T>* o_bias;
        const Tensor<T>* gate_bias;
        const Tensor<T>* up_bias;
        const Tensor<T>* down_bias;
    };

    Qwen3Model(const Parameters& params, const ModelConfig& config)
        : Qwen3Model(params, {}, {}, {}, config, false) {}

    Qwen3Model(const Parameters& params, const IntegerParameters& qweights,
               const Parameters& scales, const IntegerParameters& qzeros,
               const ModelConfig& config)
        : Qwen3Model(params, qweights, scales, qzeros, config, true) {}

    Qwen3Model(const Qwen3Model&) = delete;
    Qwen3Model& operator=(const Qwen3Model&) = delete;
    Qwen3Model(Qwen3Model&&) = delete;
    Qwen3Model& operator=(Qwen3Model&&) = delete;
    ~Qwen3Model() {
        int previous = cuda_device_id_;
        cudaGetDevice(&previous);
        if (previous != cuda_device_id_) cudaSetDevice(cuda_device_id_);
        weight_storage_.release();
        rope_storage_.release();
        if (previous != cuda_device_id_) cudaSetDevice(previous);
    }

    const Qwen3Config& config() const { return config_; }
    Device device() const { return Device::CUDA; }
    int cuda_device_id() const { return cuda_device_id_; }
    const Parameters& get_params() const { return params_; }
    const IntegerParameters& get_qweight_params() const { return qweight_params_; }
    const Parameters& get_scales_params() const { return scales_params_; }
    const IntegerParameters& get_qzeros_params() const { return qzeros_params_; }
    const Tensor<float>& rope_cache() const { return rope_sin_cos_cache_; }
    const std::vector<Layer>& layers() const { return layers_; }
    const Tensor<T>& embedding() const { return params_.at("token_embeddings.weight"); }
    const Tensor<T>& output_norm() const { return params_.at("rms_out_w"); }
    const op::WeightTensor<T>& output_weight() const { return *output_weight_; }

    op::WeightTensor<T> get_weight(const std::string& key) const {
        if (config_.quant_type == 1) {
            auto q = qweight_params_.find(key);
            auto s = scales_params_.find(key);
            auto z = qzeros_params_.find(key);
            if (q != qweight_params_.end() && s != scales_params_.end() &&
                z != qzeros_params_.end()) {
                return op::WeightTensor<T>(&q->second, &s->second, &z->second,
                                           config_.group_size);
            }
        }
        auto dense = params_.find(key);
        if (dense != params_.end()) return op::WeightTensor<T>(&dense->second);
        throw std::runtime_error("Qwen3 weight not found: " + key);
    }

    bool verify_params() const {
        if (!params_.count("token_embeddings.weight") || !params_.count("rms_out_w")) {
            return false;
        }
        try {
            (void)get_weight("lm_head");
            for (size_t layer = 0; layer < config_.n_layers; ++layer) {
                const auto suffix = std::to_string(layer);
                for (const char* prefix : {"rms_att_w", "rms_ffn_w", "q_norm", "k_norm"}) {
                    if (!params_.count(std::string(prefix) + suffix)) return false;
                }
                for (const char* prefix : {"wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down"}) {
                    (void)get_weight(std::string(prefix) + suffix);
                }
            }
        } catch (const std::runtime_error&) {
            return false;
        }
        return true;
    }

 private:
    static size_t positive_size(const ModelConfig& config, const char* key) {
        const double value = config.at(key);
        if (!std::isfinite(value) || value < 1 || std::floor(value) != value ||
            value >= static_cast<double>(std::numeric_limits<size_t>::max())) {
            throw std::invalid_argument(std::string("Invalid Qwen3 config: ") + key);
        }
        return static_cast<size_t>(value);
    }

    static uint32_t token_id(const ModelConfig& config, const char* key) {
        const double value = config.at(key);
        if (!std::isfinite(value) || value < 0 || std::floor(value) != value ||
            value > std::numeric_limits<uint32_t>::max()) {
            throw std::invalid_argument(std::string("Invalid Qwen3 token id: ") + key);
        }
        return static_cast<uint32_t>(value);
    }

    static Qwen3Config parse_config(const ModelConfig& source, bool quantized) {
        Qwen3Config result{};
        result.vocab_size = positive_size(source, "vocab_size");
        result.n_layers = positive_size(source, "n_layers");
        result.n_heads = positive_size(source, "n_heads");
        result.n_kv_heads = positive_size(source, "n_kv_heads");
        result.hidden_size = positive_size(source, "hidden_size");
        result.intermediate_size = positive_size(source, "intermediate_size");
        result.max_position_embeddings = positive_size(source, "max_position_embeddings");
        if (source.count("head_dim")) {
            result.head_dim = positive_size(source, "head_dim");
        } else {
            if (result.hidden_size % result.n_heads) {
                throw std::invalid_argument("Qwen3 hidden_size must be divisible by n_heads when head_dim is absent");
            }
            result.head_dim = result.hidden_size / result.n_heads;
        }
        result.bos_token_id = token_id(source, "bos_token_id");
        result.eos_token_id = token_id(source, "eos_token_id");
        result.rms_norm_eps = static_cast<float>(source.at("rms_norm_eps"));
        result.rope_theta = static_cast<float>(source.at("rope_theta"));
        result.quant_type = quantized ? 1 : 0;
        const size_t group_size = source.count("group_size")
                                      ? positive_size(source, "group_size") : 128;
        if (group_size > static_cast<size_t>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument("Qwen3 group_size exceeds the supported integer range");
        }
        result.group_size = static_cast<int>(group_size);
        if (result.n_heads % result.n_kv_heads || result.head_dim % 2 ||
            result.n_heads > std::numeric_limits<uint16_t>::max() ||
            result.n_kv_heads > std::numeric_limits<size_t>::max() / result.head_dim ||
            result.n_heads > std::numeric_limits<size_t>::max() / result.head_dim ||
            !std::isfinite(result.rms_norm_eps) || result.rms_norm_eps <= 0 ||
            !std::isfinite(result.rope_theta) || result.rope_theta <= 0) {
            throw std::invalid_argument("Invalid Qwen3 attention or normalization config");
        }
        const auto maximum = static_cast<size_t>(std::numeric_limits<int>::max());
        for (size_t value : {result.vocab_size, result.n_layers, result.hidden_size,
                             result.intermediate_size, result.max_position_embeddings,
                             result.n_heads * result.head_dim,
                             result.n_kv_heads * result.head_dim}) {
            if (value > maximum) throw std::invalid_argument("Qwen3 dimensions exceed the supported integer range");
        }
        if (result.hidden_size > 10240 || result.head_dim > 10240) {
            throw std::invalid_argument("Qwen3 normalization width exceeds the supported kernel range");
        }
        if (result.hidden_size % 8) {
            throw std::invalid_argument("Qwen3 embedding width must be divisible by 8 for CUDA gather");
        }
        return result;
    }

    static size_t aligned_bytes(size_t bytes) {
        constexpr size_t alignment = 256;
        if (bytes > std::numeric_limits<size_t>::max() - alignment + 1) {
            throw std::overflow_error("Qwen3 model storage size overflow");
        }
        return (bytes + alignment - 1) & ~(alignment - 1);
    }

    template <typename U>
    static size_t parameter_bytes(const Tensor<U>& tensor) {
        if (!tensor.numel() || tensor.numel() > std::numeric_limits<size_t>::max() / sizeof(U)) {
            throw std::invalid_argument("Empty or oversized Qwen3 weight");
        }
        const auto& shape = tensor.sizes();
        const auto& strides = tensor.strides();
        const bool transposed = shape.size() == 2 && strides.size() == 2 &&
                                strides[0] == 1 && strides[1] == shape[0];
        if (!tensor.is_contiguous() && !transposed) {
            throw std::invalid_argument("Qwen3 preparation supports contiguous or two-dimensional transposed weights");
        }
        return tensor.numel() * sizeof(U);
    }

    static int current_cuda_device() {
        int device_id = 0;
        const auto status = cudaGetDevice(&device_id);
        if (status != cudaSuccess) throw std::runtime_error("Cannot determine Qwen3 CUDA device");
        return device_id;
    }

    template <typename U>
    static void add_storage_bytes(const std::unordered_map<std::string, Tensor<U>>& params,
                                  size_t& total) {
        for (const auto& entry : params) {
            const size_t bytes = aligned_bytes(parameter_bytes(entry.second));
            if (bytes > std::numeric_limits<size_t>::max() - total) {
                throw std::overflow_error("Qwen3 model storage size overflow");
            }
            total += bytes;
        }
    }

    template <typename U>
    void copy_parameters(const std::unordered_map<std::string, Tensor<U>>& source,
                         std::unordered_map<std::string, Tensor<U>>& destination,
                         size_t& offset) {
        destination.reserve(source.size());
        for (const auto& entry : source) {
            const auto& input = entry.second;
            const size_t bytes = parameter_bytes(input);
            auto* output = weight_storage_.template ptr_at<U>(offset);
            const auto kind = input.device() == Device::CUDA
                                  ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
            const auto status = cudaMemcpy(output, input.data_ptr(), bytes, kind);
            if (status != cudaSuccess) {
                throw std::runtime_error("Failed to prepare Qwen3 weight " + entry.first +
                                         ": " + cudaGetErrorString(status));
            }
            const auto& shape = input.sizes();
            const auto& strides = input.strides();
            if (shape.size() == 2 && strides.size() == 2 &&
                strides[0] == 1 && strides[1] == shape[0]) {
                destination.emplace(entry.first, Tensor<U>::from_external_buffer(
                    output, {shape[1], shape[0]}, Device::CUDA).transpose(0, 1));
            } else {
                destination.emplace(entry.first,
                    Tensor<U>::from_external_buffer(output, shape, Device::CUDA));
            }
            offset += aligned_bytes(bytes);
        }
    }

    Qwen3Model(const Parameters& params, const IntegerParameters& qweights,
               const Parameters& scales, const IntegerParameters& qzeros,
               const ModelConfig& source, bool quantized)
        : config_(parse_config(source, quantized)) {
        size_t bytes = 0;
        add_storage_bytes(params, bytes);
        add_storage_bytes(qweights, bytes);
        add_storage_bytes(scales, bytes);
        add_storage_bytes(qzeros, bytes);
        if (!bytes) throw std::invalid_argument("Qwen3 model has no weights");
        weight_storage_.reserve(bytes);
        size_t offset = 0;
        copy_parameters(params, params_, offset);
        copy_parameters(qweights, qweight_params_, offset);
        copy_parameters(scales, scales_params_, offset);
        copy_parameters(qzeros, qzeros_params_, offset);
        if (!verify_params()) throw std::invalid_argument("Qwen3 model parameter verification failed");
        prepare_layers();
        initialize_rope();
    }

    const Tensor<T>* optional_parameter(const std::string& name, size_t width) const {
        auto found = params_.find(name);
        if (found == params_.end()) return nullptr;
        require_shape(found->second, {width}, name);
        return &found->second;
    }

    template <typename U>
    static void require_shape(const Tensor<U>& tensor, const std::vector<size_t>& shape,
                              const std::string& name) {
        if (tensor.sizes() != shape) throw std::invalid_argument("Qwen3 weight shape mismatch: " + name);
    }

    template <typename U>
    static void require_contiguous(const Tensor<U>& tensor, const std::string& name) {
        if (!tensor.is_contiguous()) {
            throw std::invalid_argument("Qwen3 weight must be contiguous: " + name);
        }
    }

    op::WeightTensor<T> prepared_linear(const std::string& name, size_t input, size_t output) const {
        auto weight = get_weight(name);
        if (!weight.is_quantized()) {
            require_shape(*weight.tensor(), {input, output}, name);
        } else {
            if (input % config_.group_size) {
                throw std::invalid_argument("Qwen3 AWQ input width must be divisible by group_size: " + name);
            }
            // The executing AWQ kernel consumes contiguous N-major packing.
            const size_t groups = input / config_.group_size;
            require_shape(*weight.qweight(), {output, (input + 7) / 8}, name + ".qweight");
            require_shape(*weight.qzeros(), {output, (groups + 7) / 8}, name + ".qzeros");
            const auto& scale_shape = weight.scales()->sizes();
            if (scale_shape.size() != 2 || scale_shape[0] != output || scale_shape[1] < groups) {
                throw std::invalid_argument("Qwen3 AWQ scale shape mismatch: " + name + ".scales");
            }
            require_contiguous(*weight.qweight(), name + ".qweight");
            require_contiguous(*weight.scales(), name + ".scales");
            require_contiguous(*weight.qzeros(), name + ".qzeros");
        }
        return weight;
    }

    void prepare_layers() {
        require_shape(embedding(), {config_.vocab_size, config_.hidden_size}, "token_embeddings.weight");
        require_contiguous(embedding(), "token_embeddings.weight");
        require_shape(output_norm(), {config_.hidden_size}, "rms_out_w");
        output_weight_ = prepared_linear("lm_head", config_.hidden_size, config_.vocab_size);
        layers_.reserve(config_.n_layers);
        const size_t q_width = config_.n_heads * config_.head_dim;
        const size_t kv_width = config_.n_kv_heads * config_.head_dim;
        for (size_t index = 0; index < config_.n_layers; ++index) {
            const auto suffix = std::to_string(index);
            const auto prefix = "layers." + suffix + ".";
            auto norm = [&](const char* name, size_t width) -> const Tensor<T>* {
                const auto key = std::string(name) + suffix;
                const auto& tensor = params_.at(key);
                require_shape(tensor, {width}, key);
                return &tensor;
            };
            layers_.push_back({
                prepared_linear("wq" + suffix, config_.hidden_size, q_width),
                prepared_linear("wk" + suffix, config_.hidden_size, kv_width),
                prepared_linear("wv" + suffix, config_.hidden_size, kv_width),
                prepared_linear("wo" + suffix, q_width, config_.hidden_size),
                prepared_linear("w_gate" + suffix, config_.hidden_size, config_.intermediate_size),
                prepared_linear("w_up" + suffix, config_.hidden_size, config_.intermediate_size),
                prepared_linear("w_down" + suffix, config_.intermediate_size, config_.hidden_size),
                norm("rms_att_w", config_.hidden_size), norm("rms_ffn_w", config_.hidden_size),
                norm("q_norm", config_.head_dim), norm("k_norm", config_.head_dim),
                optional_parameter(prefix + "self_attn.q_proj.bias", q_width),
                optional_parameter(prefix + "self_attn.k_proj.bias", kv_width),
                optional_parameter(prefix + "self_attn.v_proj.bias", kv_width),
                optional_parameter(prefix + "self_attn.o_proj.bias", config_.hidden_size),
                optional_parameter(prefix + "mlp.gate_proj.bias", config_.intermediate_size),
                optional_parameter(prefix + "mlp.up_proj.bias", config_.intermediate_size),
                optional_parameter(prefix + "mlp.down_proj.bias", config_.hidden_size)
            });
        }
    }

    void initialize_rope() {
        if (config_.max_position_embeddings > std::numeric_limits<size_t>::max() / config_.head_dim ||
            config_.max_position_embeddings * config_.head_dim >
                std::numeric_limits<size_t>::max() / sizeof(float)) {
            throw std::overflow_error("Qwen3 RoPE storage size overflow");
        }
        std::vector<float> values(config_.max_position_embeddings * config_.head_dim);
        for (size_t pos = 0; pos < config_.max_position_embeddings; ++pos) {
            for (size_t i = 0; i < config_.head_dim / 2; ++i) {
                const float frequency = 1.0f / std::pow(config_.rope_theta,
                    2.0f * static_cast<float>(i) / static_cast<float>(config_.head_dim));
                const float angle = static_cast<float>(pos) * frequency;
                const size_t index = pos * config_.head_dim + 2 * i;
                values[index] = std::sin(angle);
                values[index + 1] = std::cos(angle);
            }
        }
        rope_storage_.reserve(values.size() * sizeof(float));
        auto* output = rope_storage_.template ptr_at<float>(0);
        const auto status = cudaMemcpy(output, values.data(), values.size() * sizeof(float),
                                       cudaMemcpyHostToDevice);
        if (status != cudaSuccess) throw std::runtime_error("Failed to prepare Qwen3 RoPE cache");
        rope_sin_cos_cache_ = Tensor<float>::from_external_buffer(output,
            {config_.max_position_embeddings, config_.head_dim}, Device::CUDA);
    }

    const Qwen3Config config_;
    const int cuda_device_id_ = current_cuda_device();
    CudaWorkspaceArena weight_storage_;
    CudaWorkspaceArena rope_storage_;
    Parameters params_;
    IntegerParameters qweight_params_;
    Parameters scales_params_;
    IntegerParameters qzeros_params_;
    Tensor<float> rope_sin_cos_cache_;
    std::vector<Layer> layers_;
    std::optional<op::WeightTensor<T>> output_weight_;
};
