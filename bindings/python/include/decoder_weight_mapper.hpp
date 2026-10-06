#pragma once

#include <limits>
#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "weight_processor_utils.hpp"

// Resolve the common Hugging Face dense-decoder schema at model preparation.
// This mapper has no role in execution or operator dispatch.
namespace decoder_weight_mapper {

enum class Kind { Dense, Norm, Bias, Packed, Scale, Zero };
struct Entry { std::string name; Kind kind; bool transpose = false; };

inline std::string lowercase(py::handle value, const char* name) {
    if (!py::isinstance<py::str>(value))
        throw std::invalid_argument(std::string("AWQ ") + name + " must be a string");
    auto text = py::cast<std::string>(value);
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return text;
}

inline int positive_integer(py::handle value, const char* name) {
    if (!py::isinstance<py::int_>(value) || py::isinstance<py::bool_>(value))
        throw std::invalid_argument(std::string("AWQ ") + name + " must be a positive integer");
    const auto number = py::cast<long long>(value);
    if (number <= 0 || number > std::numeric_limits<int>::max())
        throw std::invalid_argument(std::string("AWQ ") + name + " exceeds the supported integer range");
    return static_cast<int>(number);
}

// Raw AWQ dictionaries use the fixed native 4-bit asymmetric GEMV layout.
// Declared metadata must describe that layout; it cannot select another packer.
inline int awq_group_size(const py::dict& config) {
    py::dict quantization;
    if (config.contains("quantization_config")) {
        const auto metadata = config["quantization_config"];
        if (!py::isinstance<py::dict>(metadata))
            throw std::invalid_argument("quantization_config must be a dictionary");
        quantization = py::reinterpret_borrow<py::dict>(metadata);
    }
    int group = 0;
    for (const auto& settings : {config, quantization}) {
        if (settings.contains("quant_method") && lowercase(settings["quant_method"], "quant_method") != "awq")
            throw std::invalid_argument("Only AWQ quant_method is supported");
        if (settings.contains("version") && lowercase(settings["version"], "version") != "gemv")
            throw std::invalid_argument("Only AWQ GEMV packing is supported");
        for (const char* key : {"bits", "w_bit"}) {
            if (settings.contains(key) && positive_integer(settings[key], key) != 4)
                throw std::invalid_argument("Only 4-bit AWQ packing is supported");
        }
        if (settings.contains("zero_point")) {
            const auto value = settings["zero_point"];
            if (!py::isinstance<py::bool_>(value) || !py::cast<bool>(value))
                throw std::invalid_argument("AWQ zero_point must be true for asymmetric packing");
        }
        for (const char* key : {"group_size", "q_group_size"}) {
            if (!settings.contains(key)) continue;
            const int value = positive_integer(settings[key], key);
            if (group && group != value)
                throw std::invalid_argument("Contradictory AWQ group sizes");
            group = value;
        }
    }
    return group ? group : 128;
}

inline py::dict admit(const py::dict& config, const py::dict& source) {
    bool tied = false;
    if (config.contains("tie_word_embeddings")) {
        const auto value = config["tie_word_embeddings"];
        if (!py::isinstance<py::bool_>(value))
            throw std::invalid_argument("tie_word_embeddings must be a boolean");
        tied = py::cast<bool>(value);
    }
    if (!source.contains("model.embed_tokens.weight"))
        throw std::invalid_argument("Decoder checkpoint requires model.embed_tokens.weight");
    bool has_head = false;
    for (const char* key : {"lm_head.weight", "model.lm_head.weight", "lm_head.qweight",
                           "model.lm_head.qweight", "lm_head.scales", "model.lm_head.scales",
                           "lm_head.qzeros", "model.lm_head.qzeros"})
        has_head = has_head || source.contains(key);
    if (!has_head && !tied)
        throw std::invalid_argument("Untied decoder checkpoint requires an explicit lm_head weight");
    py::dict weights;
    for (const auto& item : source) weights[item.first] = item.second;
    if (!has_head) weights["lm_head.weight"] = source["model.embed_tokens.weight"];
    return weights;
}

inline Entry projection(const std::string& source, const std::string& name,
                        const std::string& bias_name, const std::string& suffix) {
    if (suffix == "weight") return {name, Kind::Dense, true};
    if (suffix == "bias") return {bias_name, Kind::Bias};
    if (suffix == "qweight") return {name, Kind::Packed};
    if (suffix == "scales") return {name, Kind::Scale};
    if (suffix == "qzeros") return {name, Kind::Zero};
    throw std::invalid_argument("Unsupported decoder weight key: " + source);
}

inline Entry parse(const std::string& source, bool qk_norm, size_t n_layers = 0) {
    if (source == "model.embed_tokens.weight") return {"token_embeddings.weight", Kind::Dense};
    if (source == "model.norm.weight") return {"rms_out_w", Kind::Norm};
    for (const char* head_prefix : {"lm_head.", "model.lm_head."}) {
        const std::string prefix(head_prefix);
        if (source.compare(0, prefix.size(), prefix) == 0)
            return projection(source, "lm_head", "lm_head.bias", source.substr(prefix.size()));
    }
    const std::string prefix = "model.layers.";
    if (source.compare(0, prefix.size(), prefix) != 0)
        throw std::invalid_argument("Unsupported decoder weight key: " + source);
    const auto end = source.find('.', prefix.size());
    const auto index = source.substr(prefix.size(), end - prefix.size());
    if (end == std::string::npos || index.empty() ||
        index.find_first_not_of("0123456789") != std::string::npos ||
        (index.size() > 1 && index.front() == '0'))
        throw std::invalid_argument("Malformed decoder layer index: " + source);
    size_t layer = 0;
    for (const char digit : index) {
        const size_t value = static_cast<size_t>(digit - '0');
        if (layer > (static_cast<size_t>(std::numeric_limits<int>::max()) - value) / 10)
            throw std::invalid_argument("Decoder layer index exceeds the supported range: " + source);
        layer = layer * 10 + value;
    }
    if (n_layers && layer >= n_layers)
        throw std::invalid_argument("Decoder weight layer exceeds configuration: " + source);
    const auto suffix = source.substr(end + 1);
    const auto number = std::to_string(layer);
    if (suffix == "input_layernorm.weight") return {"rms_att_w" + number, Kind::Norm};
    if (suffix == "post_attention_layernorm.weight") return {"rms_ffn_w" + number, Kind::Norm};
    if (qk_norm && suffix == "self_attn.q_norm.weight") return {"q_norm" + number, Kind::Norm};
    if (qk_norm && suffix == "self_attn.k_norm.weight") return {"k_norm" + number, Kind::Norm};
    const auto extension = suffix.rfind('.');
    if (extension != std::string::npos) {
        const auto base = suffix.substr(0, extension), format = suffix.substr(extension + 1);
        const std::pair<const char*, const char*> projections[] = {
            {"self_attn.q_proj", "wq"}, {"self_attn.k_proj", "wk"},
            {"self_attn.v_proj", "wv"}, {"self_attn.o_proj", "wo"},
            {"mlp.gate_proj", "w_gate"}, {"mlp.up_proj", "w_up"},
            {"mlp.down_proj", "w_down"}
        };
        for (const auto& item : projections) {
            if (base == item.first)
                return projection(source, std::string(item.second) + number,
                                  "layers." + number + "." + base + ".bias", format);
        }
    }
    throw std::invalid_argument("Unsupported decoder weight key: " + source);
}

template <typename T> using Parameters = std::unordered_map<std::string, Tensor<T>>;

template <typename T>
inline void insert(Parameters<T>& target, const Entry& entry, Tensor<T> tensor) {
    const size_t rank = entry.kind == Kind::Norm || entry.kind == Kind::Bias ? 1 : 2;
    if (tensor.sizes().size() != rank)
        throw std::invalid_argument("Decoder checkpoint weight rank mismatch: " + entry.name);
    if (entry.transpose) tensor = tensor.transpose(0, 1);
    if (!target.emplace(entry.name, std::move(tensor)).second)
        throw std::invalid_argument("Duplicate normalized decoder weight: " + entry.name);
}

template <typename T>
inline Tensor<T> convert(py::handle source) {
    const auto object = py::reinterpret_borrow<py::object>(source);
    if constexpr (std::is_same_v<T, float>) {
        return weight_processor_utils::convert_float_tensor(object);
    } else {
        static_assert(std::is_same_v<T, __nv_bfloat16>);
        return weight_processor_utils::convert_bf16_tensor(object);
    }
}

template <typename T>
inline Parameters<T> dense(const py::dict& weights, bool qk_norm, size_t n_layers = 0) {
    Parameters<T> result;
    result.reserve(weights.size());
    for (const auto& item : weights) {
        if (!py::isinstance<py::str>(item.first))
            throw std::invalid_argument("Decoder checkpoint weight names must be strings");
        const auto entry = parse(py::cast<std::string>(item.first), qk_norm, n_layers);
        if (entry.kind == Kind::Packed || entry.kind == Kind::Scale || entry.kind == Kind::Zero)
            throw std::invalid_argument("Packed decoder weights require an AWQ model: " + entry.name);
        insert(result, entry, convert<T>(item.second));
    }
    return result;
}

using AwqParameters = std::tuple<Parameters<__nv_bfloat16>, Parameters<int32_t>,
                                 Parameters<__nv_bfloat16>, Parameters<int32_t>>;

inline weight_processor_utils::ContiguousArray<int32_t> packed_array(py::handle value) {
    const auto source = py::reinterpret_borrow<py::object>(value);
    if (py::hasattr(source, "detach")) {
        const auto torch = py::module::import("torch");
        if (!source.attr("dtype").is(torch.attr("int32")))
            throw std::invalid_argument("AWQ packed weights and zeros must have int32 dtype");
    } else {
        const auto array = py::array::ensure(source);
        if (!array || array.dtype().kind() != 'i' || array.itemsize() != sizeof(int32_t))
            throw std::invalid_argument("AWQ packed weights and zeros must have int32 dtype");
    }
    return weight_processor_utils::as_contiguous_array<int32_t>(source);
}

inline AwqParameters awq(const py::dict& weights, bool qk_norm, size_t n_layers = 0) {
    AwqParameters result;
    auto& [dense_weights, packed, scales, zeros] = result;
    for (const auto& item : weights) {
        if (!py::isinstance<py::str>(item.first))
            throw std::invalid_argument("Decoder checkpoint weight names must be strings");
        const auto entry = parse(py::cast<std::string>(item.first), qk_norm, n_layers);
        if (entry.kind == Kind::Packed || entry.kind == Kind::Zero) {
            const auto array = packed_array(item.second);
            std::vector<size_t> shape;
            for (int axis = 0; axis < array.ndim(); ++axis) shape.push_back(array.shape(axis));
            std::vector<int32_t> values(array.size());
            if (!values.empty()) std::memcpy(values.data(), array.data(), values.size() * sizeof(int32_t));
            insert(entry.kind == Kind::Packed ? packed : zeros, entry,
                   Tensor<int32_t>(std::move(values), shape));
        } else if (entry.kind == Kind::Scale) {
            insert(scales, entry, convert<__nv_bfloat16>(item.second));
        } else {
            insert(dense_weights, entry, convert<__nv_bfloat16>(item.second));
        }
    }
    for (const auto& item : packed) {
        if (!scales.count(item.first) || !zeros.count(item.first))
            throw std::invalid_argument("Incomplete AWQ decoder weight: " + item.first);
        if (dense_weights.count(item.first))
            throw std::invalid_argument("Dense and AWQ weights overlap: " + item.first);
    }
    for (const auto& item : scales) {
        if (!packed.count(item.first)) throw std::invalid_argument("AWQ scale has no packed weight: " + item.first);
    }
    for (const auto& item : zeros) {
        if (!packed.count(item.first)) throw std::invalid_argument("AWQ zero has no packed weight: " + item.first);
    }
    return result;
}

}  // namespace decoder_weight_mapper
