#include "qwen.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <type_traits>

#include "common.hpp"

namespace {
struct NormalizedKey {
    std::string name;
    bool transpose = false;
};

NormalizedKey normalize_key(std::string key) {
    const bool raw = key.rfind("model.", 0) == 0;
    if (raw) key.erase(0, 6);
    if (key == "embed_tokens.weight" || key == "transformer.wte.weight")
        return {"token_embeddings.weight"};
    if (key == "norm.weight" || key == "transformer.ln_f.weight") return {"rms_out_w"};
    if (key == "lm_head.weight") return {"lm_head", true};
    if (key.rfind("layers.", 0) == 0) {
        const auto end = key.find('.', 7);
        if (end == std::string::npos) throw std::invalid_argument("Malformed Qwen layer parameter: " + key);
        const auto index = key.substr(7, end - 7);
        if (index.empty() || index.find_first_not_of("0123456789") != std::string::npos)
            throw std::invalid_argument("Malformed Qwen layer index: " + key);
        const auto suffix = key.substr(end + 1);
        if (suffix == "input_layernorm.weight") return {"rms_att_w" + index};
        if (suffix == "post_attention_layernorm.weight") return {"rms_ffn_w" + index};
        const std::pair<const char*, const char*> projections[] = {
            {"self_attn.q_proj", "wq"}, {"self_attn.k_proj", "wk"},
            {"self_attn.v_proj", "wv"}, {"self_attn.o_proj", "wo"},
            {"mlp.gate_proj", "w_gate"}, {"mlp.up_proj", "w_up"},
            {"mlp.down_proj", "w_down"}
        };
        for (const auto& projection : projections) {
            const std::string base(projection.first);
            if (suffix == base || suffix == base + ".weight")
                return {std::string(projection.second) + index, raw};
            for (const char* quant_suffix : {".qweight", ".scales", ".qzeros"})
                if (suffix == base + quant_suffix) return {std::string(projection.second) + index};
        }
    }
    for (const char* suffix : {".qweight", ".scales", ".qzeros"}) {
        const auto length = std::char_traits<char>::length(suffix);
        if (key.size() > length && key.compare(key.size() - length, length, suffix) == 0)
            return {key.substr(0, key.size() - length)};
    }
    return {std::move(key)};
}

template <typename U>
std::unordered_map<std::string, Tensor<U>> normalize_parameters(
    const std::unordered_map<std::string, Tensor<U>>& source, bool packed = false,
    size_t hidden = 0, size_t vocabulary = 0) {
    std::unordered_map<std::string, Tensor<U>> result;
    result.reserve(source.size());
    for (const auto& item : source) {
        const auto key = normalize_key(item.first);
        bool transpose = key.transpose && !packed;
        if (transpose && key.name == "lm_head" && hidden && vocabulary) {
            const auto& shape = item.second.sizes();
            if (shape.size() != 2 ||
                (shape != std::vector<size_t>{hidden, vocabulary} &&
                 shape != std::vector<size_t>{vocabulary, hidden}))
                throw std::invalid_argument("Qwen output head shape must match hidden size and vocabulary");
            const bool already_logical = shape[0] == hidden && shape[1] == vocabulary;
            // Square heads are ambiguous by shape. Preserve the raw .weight
            // convention; an already-logical square head uses the lm_head key.
            transpose = !already_logical || hidden == vocabulary;
        }
        auto tensor = transpose ? item.second.transpose(0, 1) : item.second;
        if (!result.emplace(key.name, std::move(tensor)).second)
            throw std::invalid_argument("Duplicate normalized Qwen parameter: " + key.name);
    }
    return result;
}

// Copy compact physical storage and materialize the logical order once. This
// owns CPU weights even when the caller subsequently mutates its checkpoint.
template <typename U>
std::unordered_map<std::string, Tensor<U>> copy_cpu_parameters(
    const std::unordered_map<std::string, Tensor<U>>& source) {
    std::unordered_map<std::string, Tensor<U>> result;
    result.reserve(source.size());
    for (const auto& item : source) {
        const auto& tensor = item.second;
        const auto& shape = tensor.sizes();
        const auto& stride = tensor.strides();
        const bool transposed = shape.size() == 2 && stride.size() == 2 &&
                                stride[0] == 1 && stride[1] == shape[0];
        if (!tensor.numel() || (!tensor.is_contiguous() && !transposed))
            throw std::invalid_argument("Qwen requires nonempty contiguous or transposed weights: " + item.first);
        if (tensor.numel() > std::numeric_limits<size_t>::max() / sizeof(U))
            throw std::overflow_error("Qwen weight byte extent overflow");
        std::vector<U> physical(tensor.numel());
        if (tensor.device() == Device::CUDA)
            CUDA_CHECK(cudaMemcpy(physical.data(), tensor.data_ptr(), physical.size() * sizeof(U), cudaMemcpyDeviceToHost));
        else std::copy_n(tensor.data_ptr(), physical.size(), physical.data());
        if (transposed && !tensor.is_contiguous()) {
            std::vector<U> logical(physical.size());
            for (size_t row = 0; row < shape[0]; ++row)
                for (size_t column = 0; column < shape[1]; ++column)
                    logical[row * shape[1] + column] = physical[row * stride[0] + column * stride[1]];
            physical.swap(logical);
        }
        result.emplace(item.first, Tensor<U>(std::move(physical), shape));
    }
    return result;
}

ModelConfig decoder_configuration(ModelConfig config) {
    config["qk_norm"] = 0;
    return config;
}

void validate_cpu_sampling_policy(float temperature, float top_p, size_t top_k) {
    if (!std::isfinite(temperature) || !std::isfinite(top_p) ||
        top_p <= 0.0f || top_p > 1.0f || !top_k)
        throw std::invalid_argument("Invalid Qwen CPU sampling policy");
}
}

template <typename T>
QwenModel<T>::QwenModel(const Parameters& parameters, const ModelConfig& source)
    : source_config_(decoder_configuration(source)),
      config_(Qwen3Model<T>::parse_configuration(source_config_)),
      params_(normalize_parameters(parameters, false, config_.hidden_size, config_.vocab_size)) {
    if (source.count("quant_type") && source.at("quant_type") != 0)
        throw std::invalid_argument("Quantized Qwen construction requires packed parameter maps");
    if (params_.empty()) throw std::invalid_argument("Qwen requires model parameters");
    if constexpr (std::is_same_v<T, __nv_bfloat16>) device_ = Device::CUDA;
    else device_ = params_.begin()->second.device();
    if (device_ == Device::CUDA) prepare_cuda();
    else {
        params_ = copy_cpu_parameters(params_);
        prepare_cpu();
    }
}

template <typename T>
QwenModel<T>::QwenModel(const Parameters& parameters, const IntegerParameters& qweights,
                       const Parameters& scales, const IntegerParameters& qzeros, const ModelConfig& source)
    : source_config_(decoder_configuration(source)),
      config_(Qwen3Model<T>::parse_configuration(source_config_, true)),
      device_(Device::CUDA),
      params_(normalize_parameters(parameters, false, config_.hidden_size, config_.vocab_size)),
      scales_params_(normalize_parameters(scales, true)),
      qweight_params_(normalize_parameters(qweights, true)),
      qzeros_params_(normalize_parameters(qzeros, true)) {
    prepare_cuda();
}

template <typename T> QwenModel<T>::~QwenModel() = default;

template <typename T>
QwenModel<T>::QwenModel(const QwenModel& prototype, ForkExecutorTag)
    : source_config_(prototype.source_config_), config_(prototype.config_),
      device_(prototype.device_), sample_mode_(prototype.sample_mode_),
      params_(prototype.params_), scales_params_(prototype.scales_params_),
      qweight_params_(prototype.qweight_params_), qzeros_params_(prototype.qzeros_params_),
      prepared_model_(prototype.prepared_model_) {
    if (device_ == Device::CPU) {
        // Tensor descriptors retain the initial model's private CPU storage.
        // Preparation only resolves read-only views and creates this executor's
        // mutable buffers; it never snapshots the same weights a second time.
        prepare_cpu();
        sample_mode_ = prototype.sample_mode_;
        return;
    }
    if (device_ != Device::CUDA || !prepared_model_ || !prototype.cuda_session_)
        throw std::logic_error("A shared Qwen executor fork requires a prepared CUDA prototype");
    // Only execution resources are new. The retained prepared model owns all
    // shallow-copied weight descriptors independently of the prototype.
    cuda_session_ = std::make_unique<Qwen3Session<T>>(prepared_model_, prototype.cuda_session_->graph_enabled());
    cpu_sample_storage_.reserve(sizeof(uint32_t));
    sample_download_.resize(config_.vocab_size);
    sample_candidates_.resize(config_.vocab_size);
}

template <typename T>
std::shared_ptr<BaseModel> QwenModel<T>::fork_executor() const {
    return std::shared_ptr<BaseModel>(new QwenModel(*this, ForkExecutorTag{}));
}

template <typename T> void QwenModel<T>::prepare_cuda() {
    auto model = config_.quant_type
        ? std::make_shared<Qwen3Model<T>>(params_, qweight_params_, scales_params_, qzeros_params_, source_config_)
        : std::make_shared<Qwen3Model<T>>(params_, source_config_);
    auto session = std::make_unique<Qwen3Session<T>>(model);
    auto parameters = model->get_params();
    auto qweights = model->get_qweight_params();
    auto scales = model->get_scales_params();
    auto qzeros = model->get_qzeros_params();
    cpu_sample_storage_.reserve(sizeof(uint32_t));
    sample_download_.resize(config_.vocab_size);
    sample_candidates_.resize(config_.vocab_size);
    params_.swap(parameters);
    qweight_params_.swap(qweights);
    scales_params_.swap(scales);
    qzeros_params_.swap(qzeros);
    cuda_session_ = std::move(session);
    prepared_model_ = std::move(model);
    cpu_layers_.clear();
    device_ = Device::CUDA;
    sample_mode_ = SampleMode::GPU;
}

template <typename T> void QwenModel<T>::prepare_cpu() {
    if constexpr (!std::is_same_v<T, float>)
        throw std::invalid_argument("Qwen CPU reference requires FP32 weights");
    if (config_.quant_type) throw std::invalid_argument("Qwen CPU reference does not support AWQ weights");
    auto norm = [&](const std::string& name) {
        const auto& tensor = params_.at(name);
        auto view = borrow_tensor_view<1>(tensor);
        if (tensor.device() != Device::CPU || view.shape[0] != config_.hidden_size || !view.is_contiguous())
            throw std::invalid_argument("Invalid Qwen CPU norm: " + name);
        return view;
    };
    auto linear = [&](const std::string& name, size_t input, size_t output, const std::string& bias_name) {
        const auto& tensor = params_.at(name);
        CpuLinear result{borrow_tensor_view<2>(tensor), {}};
        if (tensor.device() != Device::CPU || result.weight.shape != std::array<size_t, 2>{input, output})
            throw std::invalid_argument("Invalid Qwen CPU projection: " + name);
        const auto bias = params_.find(bias_name);
        if (bias != params_.end()) {
            result.bias = borrow_tensor_view<1>(bias->second).as_const();
            if (bias->second.device() != Device::CPU || result.bias.shape[0] != output || !result.bias.is_contiguous())
                throw std::invalid_argument("Invalid Qwen CPU bias: " + bias_name);
        }
        return result;
    };
    const auto& embedding = params_.at("token_embeddings.weight");
    cpu_embedding_ = borrow_tensor_view<2>(embedding);
    if (embedding.device() != Device::CPU ||
        cpu_embedding_.shape != std::array<size_t, 2>{config_.vocab_size, config_.hidden_size})
        throw std::invalid_argument("Invalid Qwen CPU embedding shape");
    cpu_output_norm_ = norm("rms_out_w");
    cpu_output_ = linear("lm_head", config_.hidden_size, config_.vocab_size, "lm_head.bias");
    cpu_layers_.clear();
    cpu_layers_.reserve(config_.n_layers);
    const size_t q_width = config_.n_heads * config_.head_dim;
    const size_t kv_width = config_.n_kv_heads * config_.head_dim;
    for (size_t layer = 0; layer < config_.n_layers; ++layer) {
        const auto suffix = std::to_string(layer);
        const auto prefix = "layers." + suffix + ".";
        cpu_layers_.push_back({norm("rms_att_w" + suffix), norm("rms_ffn_w" + suffix),
            linear("wq" + suffix, config_.hidden_size, q_width, prefix + "self_attn.q_proj.bias"),
            linear("wk" + suffix, config_.hidden_size, kv_width, prefix + "self_attn.k_proj.bias"),
            linear("wv" + suffix, config_.hidden_size, kv_width, prefix + "self_attn.v_proj.bias"),
            linear("wo" + suffix, q_width, config_.hidden_size, prefix + "self_attn.o_proj.bias"),
            linear("w_gate" + suffix, config_.hidden_size, config_.intermediate_size, prefix + "mlp.gate_proj.bias"),
            linear("w_up" + suffix, config_.hidden_size, config_.intermediate_size, prefix + "mlp.up_proj.bias"),
            linear("w_down" + suffix, config_.intermediate_size, config_.hidden_size, prefix + "mlp.down_proj.bias")});
    }
    reserve_cpu_workspace(cpu_decode_, 1);
    sample_download_.resize(config_.vocab_size);
    sample_candidates_.resize(config_.vocab_size);
    sample_mode_ = SampleMode::CPU;
}

template <typename T>
void QwenModel<T>::reserve_cpu_workspace(CpuWorkspace& workspace, size_t rows) {
    if (rows <= workspace.rows) {
        auto& b = workspace.buffers;
        for (auto* view : {&b.residual, &b.attention_norm, &b.ffn_norm, &b.final_norm,
                           &b.q, &b.k, &b.v, &b.attention, &b.projected, &b.gate,
                           &b.up, &b.ffn, &b.logits}) view->shape[0] = rows;
        return;
    }
    auto plan = plan_decoder_workspace<T>(config_, rows);
    const size_t elements = plan.total_bytes() / sizeof(T);
    if (workspace.storage.size() < elements) workspace.storage.resize(elements);
    auto take = [&](const char* name, size_t width) -> TensorView<T, 2> {
        return {workspace.storage.data() + plan.at(name).offset / sizeof(T), {rows, width}, {width, 1}};
    };
    workspace.buffers = {
        take("residual", config_.hidden_size), take("attention_norm", config_.hidden_size),
        take("ffn_norm", config_.hidden_size), take("final_norm", config_.hidden_size),
        take("q", config_.n_heads * config_.head_dim), take("k", config_.n_kv_heads * config_.head_dim),
        take("v", config_.n_kv_heads * config_.head_dim), take("attention", config_.n_heads * config_.head_dim),
        take("projected", config_.hidden_size), take("gate", config_.intermediate_size),
        take("up", config_.intermediate_size), take("ffn", config_.hidden_size), take("logits", config_.vocab_size), {}
    };
    workspace.plan = std::move(plan);
    workspace.rows = rows;
}

template <typename T>
void QwenModel<T>::cpu_linear(TensorView<const T, 2> input, const CpuLinear& weight, TensorView<T, 2> output) {
    for (size_t row = 0; row < input.shape[0]; ++row)
        for (size_t column = 0; column < output.shape[1]; ++column) {
            float value = weight.bias.data ? static_cast<float>(weight.bias(column)) : 0.0f;
            for (size_t index = 0; index < input.shape[1]; ++index)
                value += static_cast<float>(input(row, index)) * static_cast<float>(weight.weight(index, column));
            output(row, column) = static_cast<T>(value);
        }
}

template <typename T>
void QwenModel<T>::cpu_norm(TensorView<const T, 2> input, TensorView<const T, 1> weight,
                          TensorView<T, 2> output) const {
    for (size_t row = 0; row < input.shape[0]; ++row) {
        float sum = 0.0f;
        for (size_t column = 0; column < input.shape[1]; ++column) {
            const float value = static_cast<float>(input(row, column));
            sum += value * value;
        }
        const float inverse = 1.0f / std::sqrt(sum / input.shape[1] + config_.rms_norm_eps);
        for (size_t column = 0; column < input.shape[1]; ++column)
            output(row, column) = static_cast<T>(static_cast<float>(input(row, column)) * inverse *
                                                static_cast<float>(weight(column)));
    }
}

template <typename T> void QwenModel<T>::cpu_rope(TensorView<T, 2> input, size_t heads, size_t offset) const {
    const size_t half = config_.head_dim / 2;
    for (size_t row = 0; row < input.shape[0]; ++row)
        for (size_t head = 0; head < heads; ++head)
            for (size_t index = 0; index < half; ++index) {
                const float angle = static_cast<float>(offset + row) *
                    std::pow(config_.rope_theta, -2.0f * static_cast<float>(index) / config_.head_dim);
                const float sine = std::sin(angle), cosine = std::cos(angle);
                const size_t first = head * config_.head_dim + index, second = first + half;
                const float x = static_cast<float>(input(row, first)), y = static_cast<float>(input(row, second));
                input(row, first) = static_cast<T>(x * cosine - y * sine);
                input(row, second) = static_cast<T>(y * cosine + x * sine);
            }
}

template <typename T>
TensorView<T, 2> QwenModel<T>::execute_cpu(const Tensor<uint32_t>* input, KVCache<T>* cache, bool decode) {
    if (device_ != Device::CPU || !input || !cache || input->device() != Device::CPU ||
        cache->device() != Device::CPU || input->sizes().size() != 1 || !input->is_contiguous())
        throw std::invalid_argument("Qwen CPU reference requires contiguous rank-one CPU input and cache");
    const auto tokens = borrow_tensor_view<1>(*input);
    const size_t rows = tokens.shape[0];
    if (!rows || (decode && rows != 1) || cache->get_n_layers() != config_.n_layers ||
        cache->get_head_dim() != config_.n_kv_heads * config_.head_dim ||
        !cache->get_max_seq_len() || cache->get_max_seq_len() > config_.max_position_embeddings ||
        cache->size() < rows || cache->size() > cache->get_max_seq_len() ||
        cache->get_max_seq_len() != cache->k_capacity_view(0).shape[0] ||
        cache->size() > cache->k_capacity_view(0).shape[0])
        throw std::invalid_argument("Qwen CPU cache geometry or token extent is incompatible");
    for (size_t row = 0; row < rows; ++row)
        if (tokens(row) >= config_.vocab_size) throw std::invalid_argument("Qwen CPU token exceeds vocabulary");
    // Validation precedes any activation/cache write. Growing prefill prepares
    // offsets once; each operator below uses only resolved borrowed views.
    auto& workspace = decode ? cpu_decode_ : cpu_prefill_;
    reserve_cpu_workspace(workspace, rows);
    auto& b = workspace.buffers;
    const size_t offset = cache->size() - rows;
    for (size_t row = 0; row < rows; ++row)
        for (size_t column = 0; column < config_.hidden_size; ++column)
            b.residual(row, column) = cpu_embedding_(tokens(row), column);
    const size_t kv_width = config_.n_kv_heads * config_.head_dim;
    const size_t groups = config_.n_heads / config_.n_kv_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(config_.head_dim));
    for (size_t layer = 0; layer < cpu_layers_.size(); ++layer) {
        const auto& weights = cpu_layers_[layer];
        cpu_norm(b.residual.as_const(), weights.attention_norm, b.attention_norm);
        cpu_linear(b.attention_norm.as_const(), weights.q, b.q);
        cpu_linear(b.attention_norm.as_const(), weights.k, b.k);
        cpu_linear(b.attention_norm.as_const(), weights.v, b.v);
        cpu_rope(b.q, config_.n_heads, offset);
        cpu_rope(b.k, config_.n_kv_heads, offset);
        for (size_t row = 0; row < rows; ++row) {
            auto key = cache->k_token(layer, offset + row), value = cache->v_token(layer, offset + row);
            for (size_t column = 0; column < kv_width; ++column) {
                key(column) = b.k(row, column);
                value(column) = b.v(row, column);
            }
        }
        auto keys = cache->k_view(layer), values = cache->v_view(layer);
        for (size_t row = 0; row < rows; ++row)
            for (size_t head = 0; head < config_.n_heads; ++head) {
                const size_t query_base = head * config_.head_dim;
                const size_t key_base = (head / groups) * config_.head_dim;
                float maximum = -std::numeric_limits<float>::infinity(), denominator = 0.0f;
                for (size_t column = 0; column < config_.head_dim; ++column)
                    b.attention(row, query_base + column) = static_cast<T>(0.0f);
                for (size_t position = 0; position <= offset + row; ++position) {
                    float score = 0.0f;
                    for (size_t column = 0; column < config_.head_dim; ++column)
                        score += static_cast<float>(b.q(row, query_base + column)) *
                                 static_cast<float>(keys(position, key_base + column));
                    score *= scale;
                    const float next_maximum = std::max(maximum, score);
                    const float previous_scale = std::exp(maximum - next_maximum);
                    const float probability = std::exp(score - next_maximum);
                    denominator = denominator * previous_scale + probability;
                    for (size_t column = 0; column < config_.head_dim; ++column)
                        b.attention(row, query_base + column) = static_cast<T>(
                            static_cast<float>(b.attention(row, query_base + column)) * previous_scale +
                            probability * static_cast<float>(values(position, key_base + column)));
                    maximum = next_maximum;
                }
                for (size_t column = 0; column < config_.head_dim; ++column)
                    b.attention(row, query_base + column) = static_cast<T>(
                        static_cast<float>(b.attention(row, query_base + column)) / denominator);
            }
        cpu_linear(b.attention.as_const(), weights.o, b.projected);
        for (size_t index = 0; index < b.residual.numel(); ++index)
            b.residual.data[index] = static_cast<T>(static_cast<float>(b.residual.data[index]) +
                                                    static_cast<float>(b.projected.data[index]));
        cpu_norm(b.residual.as_const(), weights.ffn_norm, b.ffn_norm);
        cpu_linear(b.ffn_norm.as_const(), weights.gate, b.gate);
        cpu_linear(b.ffn_norm.as_const(), weights.up, b.up);
        for (size_t index = 0; index < b.gate.numel(); ++index) {
            const float gate = static_cast<float>(b.gate.data[index]);
            b.gate.data[index] = static_cast<T>((gate / (1.0f + std::exp(-gate))) * static_cast<float>(b.up.data[index]));
        }
        cpu_linear(b.gate.as_const(), weights.down, b.ffn);
        for (size_t index = 0; index < b.residual.numel(); ++index)
            b.residual.data[index] = static_cast<T>(static_cast<float>(b.residual.data[index]) +
                                                    static_cast<float>(b.ffn.data[index]));
    }
    cpu_norm(b.residual.as_const(), cpu_output_norm_, b.final_norm);
    cpu_linear(b.final_norm.as_const(), cpu_output_, b.logits);
    return b.logits;
}

template <typename T>
TensorView<T, 2> QwenModel<T>::forward_cuda(const Tensor<uint32_t>* in, KVCache<T>* kv, const std::string& diagnostic) {
    if (!diagnostic.empty()) throw std::invalid_argument("Direct Qwen execution does not serialize intermediate tensors");
    if (!cuda_session_ || device_ != Device::CUDA) throw std::logic_error("Qwen CUDA session is not prepared");
    return cuda_session_->forward_eager(in, kv);
}
template <typename T> TensorView<T, 2> QwenModel<T>::prefill_cuda(const Tensor<uint32_t>* in, KVCache<T>* kv) {
    if (!cuda_session_ || device_ != Device::CUDA) throw std::logic_error("Qwen CUDA session is not prepared");
    return cuda_session_->prefill_eager(in, kv);
}
template <typename T> TensorView<T, 2> QwenModel<T>::forward_generic(const Tensor<uint32_t>* in, KVCache<T>* kv) {
    return execute_cpu(in, kv, true);
}
template <typename T> TensorView<T, 2> QwenModel<T>::prefill_generic(const Tensor<uint32_t>* in, KVCache<T>* kv) {
    return execute_cpu(in, kv, false);
}
template <typename T> TensorView<T, 2> QwenModel<T>::forward_logits_only(const Tensor<uint32_t>* in, KVCache<T>* kv) {
    return device_ == Device::CUDA ? forward_cuda(in, kv) : forward_generic(in, kv);
}
template <typename T> TensorView<T, 2> QwenModel<T>::forward_for_graph_logits_only(const Tensor<uint32_t>* in, KVCache<T>* kv) {
    if (!cuda_session_ || device_ != Device::CUDA) throw std::logic_error("Qwen CUDA session is not prepared");
    return cuda_session_->forward_for_graph_logits_only(in, kv);
}

template <typename T>
uint32_t QwenModel<T>::sample_cpu(TensorView<const T, 2> logits, float temperature, float top_p, size_t top_k) {
    if (!logits.data || !logits.shape[0] || logits.shape[1] != config_.vocab_size || !logits.is_contiguous() ||
        !std::isfinite(temperature) || !std::isfinite(top_p) || top_p <= 0.0f || top_p > 1.0f || !top_k)
        throw std::invalid_argument("Invalid Qwen CPU sampling input or policy");
    auto row = logits.template select<0>(logits.shape[0] - 1);
    if (device_ == Device::CUDA) {
        CUDA_CHECK(cudaMemcpy(sample_download_.data(), row.data, row.numel() * sizeof(T), cudaMemcpyDeviceToHost));
        row = {sample_download_.data(), row.shape, {1}};
    }
    top_k = std::min(top_k, config_.vocab_size);
    const bool greedy = temperature <= 0.0f || top_k == 1;
    for (size_t index = 0; index < config_.vocab_size; ++index) {
        float value = static_cast<float>(row(index));
        if (std::isnan(value)) value = -std::numeric_limits<float>::infinity();
        sample_candidates_[index] = {greedy ? value : value / temperature, static_cast<uint32_t>(index)};
    }
    auto order = [](const auto& left, const auto& right) {
        return left.first > right.first || (left.first == right.first && left.second < right.second);
    };
    if (greedy) return std::min_element(sample_candidates_.begin(), sample_candidates_.end(), order)->second;
    std::partial_sort(sample_candidates_.begin(), sample_candidates_.begin() + top_k, sample_candidates_.end(), order);
    const float maximum = sample_candidates_[0].first;
    if (!std::isfinite(maximum)) return sample_candidates_[0].second;
    float total = 0.0f;
    for (size_t index = 0; index < top_k; ++index) {
        sample_candidates_[index].first = std::exp(sample_candidates_[index].first - maximum);
        total += sample_candidates_[index].first;
    }
    float retained = 0.0f;
    size_t count = 0;
    do { retained += sample_candidates_[count++].first; } while (count < top_k && retained < top_p * total);
    sample_rng_ ^= sample_rng_ >> 12;
    sample_rng_ ^= sample_rng_ << 25;
    sample_rng_ ^= sample_rng_ >> 27;
    const uint64_t random = sample_rng_ * 0x2545f4914f6cdd1dULL;
    float target = static_cast<float>((random >> 11) * (1.0 / 9007199254740992.0)) * retained;
    for (size_t index = 0; index < count; ++index) {
        target -= sample_candidates_[index].first;
        if (target <= 0.0f) return sample_candidates_[index].second;
    }
    return sample_candidates_[count - 1].second;
}

template <typename T>
uint32_t* QwenModel<T>::cpu_sample_result(TensorView<const T, 2> logits, float temperature, float top_p, size_t top_k) {
    const auto token = sample_cpu(logits, temperature, top_p, top_k);
    if (device_ == Device::CPU) return new uint32_t(token); // BaseModel caller owns this compatibility result.
    auto* result = cpu_sample_storage_.template ptr_at<uint32_t>(0);
    CUDA_CHECK(cudaMemcpy(result, &token, sizeof(token), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
    return result;
}

template <typename T>
uint32_t* QwenModel<T>::forward(const Tensor<uint32_t>* input, ThreadPool& pool, KVCacheBase* base,
                              size_t top_k, float temperature, float top_p, curandState* states) {
    if (device_ == Device::CUDA && sample_mode_ != SampleMode::CPU)
        return cuda_session_->forward(input, pool, base, top_k, temperature, top_p, states);
    validate_cpu_sampling_policy(temperature, top_p, top_k);
    auto* cache = dynamic_cast<KVCache<T>*>(base);
    auto logits = device_ == Device::CUDA && cuda_session_->graph_enabled()
        ? forward_for_graph_logits_only(input, cache) : forward_logits_only(input, cache);
    return cpu_sample_result(logits.as_const(), temperature, top_p, top_k);
}
template <typename T>
uint32_t* QwenModel<T>::prefill(const Tensor<uint32_t>* input, ThreadPool& pool, KVCacheBase* base,
                              size_t top_k, float temperature, float top_p, curandState* states) {
    if (device_ == Device::CUDA && sample_mode_ != SampleMode::CPU)
        return cuda_session_->prefill(input, pool, base, top_k, temperature, top_p, states);
    validate_cpu_sampling_policy(temperature, top_p, top_k);
    auto* cache = dynamic_cast<KVCache<T>*>(base);
    auto logits = device_ == Device::CUDA ? prefill_cuda(input, cache) : prefill_generic(input, cache);
    return cpu_sample_result(logits.as_const(), temperature, top_p, top_k);
}

template <typename T> QwenModel<T>& QwenModel<T>::cuda() {
    if (device_ != Device::CUDA) prepare_cuda();
    return *this;
}
template <typename T> void QwenModel<T>::synchronize() const {
    if (device_ == Device::CUDA) {
        if (!cuda_session_) throw std::logic_error("Qwen CUDA session is not prepared");
        cuda_session_->synchronize();
    }
}
template <typename T> QwenModel<T>& QwenModel<T>::cpu() {
    if (device_ == Device::CPU) return *this;
    if constexpr (!std::is_same_v<T, float>) throw std::invalid_argument("Qwen CPU reference requires FP32 weights");
    if (config_.quant_type) throw std::invalid_argument("Qwen CPU reference does not support AWQ weights");
    auto parameters = copy_cpu_parameters(prepared_model_->get_params());
    auto previous_parameters = std::move(params_);
    params_ = std::move(parameters);
    try {
        prepare_cpu();
    } catch (...) {
        params_ = std::move(previous_parameters);
        cpu_layers_.clear();
        throw;
    }
    cuda_session_.reset();
    cpu_sample_storage_.release();
    prepared_model_.reset();
    device_ = Device::CPU;
    return *this;
}
template <typename T> bool QwenModel<T>::verify_params() const {
    return device_ == Device::CUDA ? prepared_model_ && prepared_model_->verify_params()
                                  : cpu_layers_.size() == config_.n_layers && cpu_embedding_.data && cpu_output_.weight.data;
}
template <typename T> void QwenModel<T>::print_model_info() const {
    std::cout << "Qwen2/Llama decoder: " << config_.n_layers << " layers, " << config_.hidden_size
              << " hidden, " << config_.vocab_size << " vocabulary; "
              << (device_ == Device::CUDA ? "prepared CUDA session" : "FP32 CPU reference") << '\n';
}
template <typename T> size_t QwenModel<T>::estimate_prefill_workspace_bytes(size_t rows) const {
    return rows ? plan_decoder_workspace<T>(config_, rows).total_bytes() : 0;
}
template <typename T> op::WeightTensor<T> QwenModel<T>::get_weight(const std::string& name) const {
    const auto key = normalize_key(name);
    if (config_.quant_type) {
        auto q = qweight_params_.find(key.name), z = qzeros_params_.find(key.name);
        auto s = scales_params_.find(key.name);
        if (q != qweight_params_.end() && z != qzeros_params_.end() && s != scales_params_.end())
            return op::WeightTensor<T>(&q->second, &s->second, &z->second, config_.group_size);
    }
    auto dense = params_.find(key.name);
    if (dense != params_.end()) return op::WeightTensor<T>(&dense->second);
    throw std::runtime_error("Qwen weight not found: " + name);
}
template class QwenModel<float>;
template class QwenModel<__nv_bfloat16>;
