#include "qwen.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t kHidden = 8, kIntermediate = 16, kVocabulary = 16;
constexpr float kEpsilon = 1.0e-5f, kTheta = 10000.0f;
using Parameters = Qwen3Model<float>::Parameters;

void expect(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

Tensor<float> matrix(std::size_t input, std::size_t output, std::size_t seed) {
    std::vector<float> data(input * output);
    for (std::size_t row = 0; row < output; ++row) {
        for (std::size_t column = 0; column < input; ++column) {
            const auto residue = static_cast<int>((row * 7 + column * 3 + seed * 11) % 17) - 8;
            data[row * input + column] = (static_cast<float>(residue) + 0.25f) * 0.002f +
                (column == row % input ? 0.15f : 0.0f);
        }
    }
    return Tensor<float>(std::move(data), {output, input}).transpose(0, 1);
}

Parameters weights() {
    Parameters result;
    std::vector<float> embeddings(kVocabulary * kHidden);
    for (std::size_t token = 0; token < kVocabulary; ++token) {
        for (std::size_t feature = 0; feature < kHidden; ++feature) {
            const auto residue = static_cast<int>((token * 13 + feature * 7 + token * feature) % 23) - 11;
            embeddings[token * kHidden + feature] = (static_cast<float>(residue) + 0.5f) * 0.03f;
        }
    }
    result.emplace("token_embeddings.weight", Tensor<float>(std::move(embeddings), {kVocabulary, kHidden}));
    for (const char* name : {"rms_att_w0", "rms_ffn_w0", "rms_out_w"}) {
        result.emplace(name, Tensor<float>(std::vector<float>(kHidden, 1.0f), {kHidden}));
    }
    result.emplace("wq0", matrix(kHidden, kHidden, 1));
    result.emplace("wk0", matrix(kHidden, kHidden, 2));
    result.emplace("wv0", matrix(kHidden, kHidden, 3));
    result.emplace("wo0", matrix(kHidden, kHidden, 4));
    result.emplace("w_gate0", matrix(kHidden, kIntermediate, 5));
    result.emplace("w_up0", matrix(kHidden, kIntermediate, 6));
    result.emplace("w_down0", matrix(kIntermediate, kHidden, 7));
    result.emplace("lm_head", matrix(kHidden, kVocabulary, 8));
    for (const auto& name : {"q", "k", "v"}) {
        std::vector<float> bias(kHidden);
        const float offset = name[0] == 'q' ? 0.01f : name[0] == 'k' ? -0.015f : 0.02f;
        for (std::size_t feature = 0; feature < kHidden; ++feature) bias[feature] = offset * (feature + 1);
        result.emplace(std::string("layers.0.self_attn.") + name + "_proj.bias",
                       Tensor<float>(std::move(bias), {kHidden}));
    }
    // Qwen2 disables Q/K normalization and supplies no corresponding tensors.
    return result;
}

ModelConfig configuration() {
    return {{"vocab_size", kVocabulary}, {"n_layers", 1}, {"n_heads", 1},
            {"n_kv_heads", 1}, {"hidden_size", kHidden}, {"head_dim", kHidden},
            {"intermediate_size", kIntermediate}, {"max_position_embeddings", 8},
            {"bos_token_id", 1}, {"eos_token_id", 2}, {"rms_norm_eps", kEpsilon},
            {"rope_theta", kTheta}, {"qk_norm", 0}};
}

std::vector<float> normalize(const std::vector<float>& input) {
    double square_sum = 0;
    for (float value : input) square_sum += static_cast<double>(value) * value;
    const auto inverse = 1.0 / std::sqrt(square_sum / input.size() + kEpsilon);
    std::vector<float> output(input.size());
    for (std::size_t feature = 0; feature < input.size(); ++feature) output[feature] = input[feature] * inverse;
    return output;
}

std::vector<float> linear(const Parameters& source, const char* name,
                          const std::vector<float>& input, const char* bias_name = nullptr) {
    const auto& weight = source.at(name);
    const auto output_width = weight.sizes()[1];
    std::vector<float> output(output_width);
    for (std::size_t row = 0; row < output_width; ++row) {
        const auto bias = bias_name ? source.find(bias_name) : source.end();
        double sum = bias != source.end() ? bias->second.data_ptr()[row] : 0;
        for (std::size_t feature = 0; feature < input.size(); ++feature) {
            sum += static_cast<double>(input[feature]) * weight.data_ptr()[row * input.size() + feature];
        }
        output[row] = static_cast<float>(sum);
    }
    return output;
}

void rotate(std::vector<float>& value, std::size_t position) {
    constexpr auto half = kHidden / 2;
    for (std::size_t pair = 0; pair < half; ++pair) {
        const auto frequency = 1.0 / std::pow(static_cast<double>(kTheta), 2.0 * pair / kHidden);
        const auto angle = position * frequency;
        const float first = value[pair], second = value[pair + half];
        value[pair] = first * std::cos(angle) - second * std::sin(angle);
        value[pair + half] = first * std::sin(angle) + second * std::cos(angle);
    }
}

struct Reference {
    std::vector<float> logits, keys, values;
};

Reference cpu_reference(const Parameters& source, const std::vector<uint32_t>& history) {
    Reference result;
    for (std::size_t position = 0; position < history.size(); ++position) {
        const auto* embedding = source.at("token_embeddings.weight").data_ptr() + history[position] * kHidden;
        std::vector<float> residual(embedding, embedding + kHidden);
        const auto hidden = normalize(residual);
        auto query = linear(source, "wq0", hidden, "layers.0.self_attn.q_proj.bias");
        auto key = linear(source, "wk0", hidden, "layers.0.self_attn.k_proj.bias");
        const auto value = linear(source, "wv0", hidden, "layers.0.self_attn.v_proj.bias");
        rotate(query, position);
        rotate(key, position);
        result.keys.insert(result.keys.end(), key.begin(), key.end());
        result.values.insert(result.values.end(), value.begin(), value.end());
        std::vector<double> scores(position + 1);
        double maximum = -INFINITY;
        for (std::size_t token = 0; token <= position; ++token) {
            double score = 0;
            for (std::size_t feature = 0; feature < kHidden; ++feature) {
                score += static_cast<double>(query[feature]) * result.keys[token * kHidden + feature];
            }
            scores[token] = score / std::sqrt(static_cast<double>(kHidden));
            maximum = std::max(maximum, scores[token]);
        }
        double denominator = 0;
        for (double& score : scores) { score = std::exp(score - maximum); denominator += score; }
        std::vector<float> attention(kHidden);
        for (std::size_t feature = 0; feature < kHidden; ++feature) {
            double sum = 0;
            for (std::size_t token = 0; token <= position; ++token) {
                sum += scores[token] / denominator * result.values[token * kHidden + feature];
            }
            attention[feature] = static_cast<float>(sum);
        }
        const auto projected = linear(source, "wo0", attention);
        for (std::size_t feature = 0; feature < kHidden; ++feature) residual[feature] += projected[feature];
        const auto ffn_hidden = normalize(residual);
        auto gate = linear(source, "w_gate0", ffn_hidden);
        const auto up = linear(source, "w_up0", ffn_hidden);
        for (std::size_t feature = 0; feature < kIntermediate; ++feature) {
            gate[feature] = gate[feature] / (1.0f + std::exp(-gate[feature])) * up[feature];
        }
        const auto ffn = linear(source, "w_down0", gate);
        for (std::size_t feature = 0; feature < kHidden; ++feature) residual[feature] += ffn[feature];
        const auto logits = linear(source, "lm_head", normalize(residual));
        result.logits.insert(result.logits.end(), logits.begin(), logits.end());
    }
    return result;
}

template <typename T, std::size_t Rank>
std::vector<float> download(TensorView<T, Rank> view) {
    expect(view.is_contiguous(), "float smoke download requires contiguous views");
    std::vector<float> data(view.numel());
    check(cudaMemcpy(data.data(), view.data_ptr(), data.size() * sizeof(float), cudaMemcpyDeviceToHost));
    return data;
}

template <typename T, std::size_t Rank>
std::vector<float> snapshot(TensorView<T, Rank> view, Device device) {
    if (device == Device::CUDA) return download(view);
    expect(view.is_contiguous(), "CPU reference snapshot requires contiguous views");
    return {view.data_ptr(), view.data_ptr() + view.numel()};
}

Parameters clone_checkpoint(const Parameters& source) {
    Parameters result;
    for (const auto& item : source) {
        const auto& tensor = item.second;
        std::vector<float> physical(tensor.data_ptr(), tensor.data_ptr() + tensor.numel());
        if (tensor.is_contiguous()) {
            result.emplace(item.first, Tensor<float>(std::move(physical), tensor.sizes()));
        } else {
            expect(tensor.sizes().size() == 2 && tensor.strides()[0] == 1 &&
                       tensor.strides()[1] == tensor.sizes()[0],
                   "synthetic checkpoint expects compact transposed matrices");
            result.emplace(item.first,
                Tensor<float>(std::move(physical), {tensor.sizes()[1], tensor.sizes()[0]}).transpose(0, 1));
        }
    }
    return result;
}

Parameters huggingface_checkpoint(const Parameters& source) {
    const std::unordered_map<std::string, std::string> names{
        {"token_embeddings.weight", "model.embed_tokens.weight"},
        {"rms_att_w0", "model.layers.0.input_layernorm.weight"},
        {"rms_ffn_w0", "model.layers.0.post_attention_layernorm.weight"},
        {"rms_out_w", "model.norm.weight"},
        {"wq0", "model.layers.0.self_attn.q_proj.weight"},
        {"wk0", "model.layers.0.self_attn.k_proj.weight"},
        {"wv0", "model.layers.0.self_attn.v_proj.weight"},
        {"wo0", "model.layers.0.self_attn.o_proj.weight"},
        {"w_gate0", "model.layers.0.mlp.gate_proj.weight"},
        {"w_up0", "model.layers.0.mlp.up_proj.weight"},
        {"w_down0", "model.layers.0.mlp.down_proj.weight"},
        {"lm_head", "lm_head.weight"}};
    Parameters result;
    for (const auto& item : source) {
        const auto found = names.find(item.first);
        const auto name = found == names.end() ? "model." + item.first : found->second;
        const bool projection = item.first == "lm_head" ||
            (item.first.size() > 1 && item.first[0] == 'w');
        result.emplace(name, projection ? item.second.transpose(0, 1) : item.second);
    }
    return result;
}

float compare(const std::vector<float>& expected, const std::vector<float>& actual, const char* label) {
    expect(expected.size() == actual.size(), std::string(label) + " extent differs");
    float maximum = 0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const auto error = std::fabs(expected[index] - actual[index]);
        maximum = std::max(maximum, error);
        expect(std::isfinite(actual[index]) && error <= 2.0e-5f + 2.0e-5f * std::fabs(expected[index]),
               std::string(label) + " differs at " + std::to_string(index));
    }
    return maximum;
}

void smoke(const Parameters& source, bool graph) {
    const std::vector<uint32_t> prompt{3, 5, 7};
    std::vector<uint32_t> history = prompt;
    auto model = std::make_shared<Qwen3Model<float>>(source, configuration());
    expect(!model->config().qk_norm && !model->layers()[0].q_norm.data &&
               !model->layers()[0].k_norm.data,
           "Qwen2 shared decoder must disable Q/K normalization");
    KVCache<float> cache(1, 8, kHidden, Device::CUDA);
    Qwen3Session<float> session(model, graph);
    cache.resize(prompt.size());
    Tensor<uint32_t> prompt_tensor(std::vector<uint32_t>(prompt), {prompt.size()}, Device::CUDA);
    auto logits = session.prefill_eager(&prompt_tensor, &cache);
    auto reference = cpu_reference(source, history);
    float maximum_logits = compare(reference.logits, download(logits), "Qwen2 prefill logits");
    float maximum_keys = compare(reference.keys, download(cache.k_view(0)), "Qwen2 prefill keys");
    float maximum_values = compare(reference.values, download(cache.v_view(0)), "Qwen2 prefill values");
    for (uint32_t next : {uint32_t{9}, uint32_t{11}}) {
        history.push_back(next);
        Tensor<uint32_t> token(std::vector<uint32_t>{next}, {1}, Device::CUDA);
        cache.resize(history.size());
        logits = graph ? session.forward_for_graph_logits_only(&token, &cache)
                       : session.forward_eager(&token, &cache);
        reference = cpu_reference(source, history);
        const auto last = reference.logits.end() - kVocabulary;
        maximum_logits = std::max(maximum_logits,
            compare(std::vector<float>(last, reference.logits.end()), download(logits), "Qwen2 decode logits"));
        maximum_keys = std::max(maximum_keys,
            compare(reference.keys, download(cache.k_view(0)), "Qwen2 decode keys"));
        maximum_values = std::max(maximum_values,
            compare(reference.values, download(cache.v_view(0)), "Qwen2 decode values"));
    }
    std::cout << "Qwen2 FP32 " << (graph ? "graph" : "eager")
              << ": CPU-reference max_abs_logits=" << maximum_logits
              << ", max_abs_K=" << maximum_keys << ", max_abs_V=" << maximum_values << '\n';
}

void adapter_smoke(const Parameters& source, Device device, bool graph,
                   bool huggingface, const char* label) {
    const std::vector<uint32_t> prompt{3, 5, 7};
    std::vector<uint32_t> history = prompt;
    // External storage outlives the adapter and its captured graph.
    KVCache<float> cache(1, 8, kHidden, device);
    for (auto entry : {std::make_pair(cache.k_capacity_view(0), -17.0f),
                       std::make_pair(cache.v_capacity_view(0), -19.0f)}) {
        if (device == Device::CPU) std::fill_n(entry.first.data_ptr(), entry.first.numel(), entry.second);
        else {
            const std::vector<float> sentinel(entry.first.numel(), entry.second);
            check(cudaMemcpy(entry.first.data_ptr(), sentinel.data(), sentinel.size() * sizeof(float),
                             cudaMemcpyHostToDevice));
        }
    }
    auto checkpoint = clone_checkpoint(source);
    if (huggingface) checkpoint = huggingface_checkpoint(checkpoint);
    auto config = configuration();
    config["qk_norm"] = 1; // The Qwen2/Llama adapter must override this Qwen3 option.
    QwenModel<float> model(checkpoint, config);
    expect(model.device() == Device::CPU && model.verify_params(),
           "Qwen compatibility adapter must prepare its FP32 CPU path");
    if (device == Device::CUDA) model.cuda();
    expect(model.device() == device && model.verify_params(),
           "Qwen compatibility adapter device preparation failed");
    // Preparation must retain private weights before caller storage is reused.
    for (auto& item : checkpoint) {
        std::fill_n(item.second.data_ptr(), item.second.numel(), -100.0f);
    }
    cache.resize(prompt.size());
    Tensor<uint32_t> prompt_tensor(std::vector<uint32_t>(prompt), {prompt.size()}, device);
    auto logits = device == Device::CUDA ? model.prefill_cuda(&prompt_tensor, &cache)
                                        : model.prefill_generic(&prompt_tensor, &cache);
    auto reference = cpu_reference(source, history);
    float maximum_logits = compare(reference.logits, snapshot(logits, device), "Adapter prefill logits");
    float maximum_keys = compare(reference.keys, snapshot(cache.k_view(0), device), "Adapter prefill keys");
    float maximum_values = compare(reference.values, snapshot(cache.v_view(0), device), "Adapter prefill values");
    const float* fixed_decode_output = nullptr;
    for (uint32_t next : {uint32_t{9}, uint32_t{11}}) {
        history.push_back(next);
        Tensor<uint32_t> token(std::vector<uint32_t>{next}, {1}, device);
        cache.resize(history.size());
        logits = graph ? model.forward_for_graph_logits_only(&token, &cache)
                       : model.forward_logits_only(&token, &cache);
        expect(!fixed_decode_output || logits.data_ptr() == fixed_decode_output,
               "Qwen compatibility decode output must retain fixed storage");
        fixed_decode_output = logits.data_ptr();
        reference = cpu_reference(source, history);
        maximum_logits = std::max(maximum_logits,
            compare(std::vector<float>(reference.logits.end() - kVocabulary, reference.logits.end()),
                    snapshot(logits, device), "Adapter decode logits"));
        maximum_keys = std::max(maximum_keys,
            compare(reference.keys, snapshot(cache.k_view(0), device), "Adapter decode keys"));
        maximum_values = std::max(maximum_values,
            compare(reference.values, snapshot(cache.v_view(0), device), "Adapter decode values"));
    }
    const auto held_logits = snapshot(logits, device);
    const auto held_keys = snapshot(cache.k_capacity_view(0), device);
    const auto held_values = snapshot(cache.v_capacity_view(0), device);
    cache.resize(history.size() + 1);
    Tensor<uint32_t> invalid(std::vector<uint32_t>{kVocabulary}, {1}, device);
    bool rejected = false;
    try {
        if (graph) model.forward_for_graph_logits_only(&invalid, &cache);
        else model.forward_logits_only(&invalid, &cache);
    } catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected && snapshot(logits, device) == held_logits &&
               snapshot(cache.k_capacity_view(0), device) == held_keys &&
               snapshot(cache.v_capacity_view(0), device) == held_values,
           "Qwen compatibility invalid token must preserve output and cache storage");
    std::cout << label << " FP32 adapter " << (device == Device::CPU ? "CPU" : graph ? "graph" : "eager")
              << (huggingface ? " (Hugging Face weights)" : " (processed weights)")
              << ": CPU-reference max_abs_logits=" << maximum_logits
              << ", max_abs_K=" << maximum_keys << ", max_abs_V=" << maximum_values << '\n';
}

void cpu_adapter_isolation_test(const Parameters& source) {
    KVCache<float> cache_a(1, 8, kHidden, Device::CPU);
    KVCache<float> cache_b(1, 8, kHidden, Device::CPU);
    QwenModel<float> a(source, configuration()), b(source, configuration());
    std::vector<uint32_t> history_a{3, 5, 7}, history_b{2, 4};
    Tensor<uint32_t> prompt_a(std::vector<uint32_t>(history_a), {history_a.size()});
    Tensor<uint32_t> prompt_b(std::vector<uint32_t>(history_b), {history_b.size()});
    cache_a.resize(history_a.size());
    cache_b.resize(history_b.size());
    auto logits_a = a.prefill_generic(&prompt_a, &cache_a);
    const auto held_prefill = snapshot(logits_a, Device::CPU);
    auto logits_b = b.prefill_generic(&prompt_b, &cache_b);
    expect(snapshot(logits_a, Device::CPU) == held_prefill && logits_a.data_ptr() != logits_b.data_ptr(),
           "CPU compatibility adapters must preserve private held prefill outputs");
    const std::array<uint32_t, 2> next_a{9, 11}, next_b{6, 8};
    float* fixed_a = nullptr;
    float* fixed_b = nullptr;
    for (std::size_t step = 0; step < next_a.size(); ++step) {
        history_a.push_back(next_a[step]);
        history_b.push_back(next_b[step]);
        Tensor<uint32_t> token_a(std::vector<uint32_t>{next_a[step]}, {1});
        Tensor<uint32_t> token_b(std::vector<uint32_t>{next_b[step]}, {1});
        cache_a.resize(history_a.size());
        cache_b.resize(history_b.size());
        logits_a = a.forward_generic(&token_a, &cache_a);
        const auto held_a = snapshot(logits_a, Device::CPU);
        const auto held_a_keys = snapshot(cache_a.k_view(0), Device::CPU);
        logits_b = b.forward_generic(&token_b, &cache_b);
        expect(snapshot(logits_a, Device::CPU) == held_a &&
                   snapshot(cache_a.k_view(0), Device::CPU) == held_a_keys &&
                   logits_a.data_ptr() != logits_b.data_ptr() &&
                   (!fixed_a || fixed_a == logits_a.data_ptr()) &&
                   (!fixed_b || fixed_b == logits_b.data_ptr()),
               "Interleaved CPU adapters must preserve private fixed decode storage");
        fixed_a = logits_a.data_ptr();
        fixed_b = logits_b.data_ptr();
        for (const auto& entry : {std::make_pair(&history_a, &cache_a), std::make_pair(&history_b, &cache_b)}) {
            const auto reference = cpu_reference(source, *entry.first);
            const auto current_logits = entry.first == &history_a ? logits_a : logits_b;
            compare(std::vector<float>(reference.logits.end() - kVocabulary, reference.logits.end()),
                    snapshot(current_logits, Device::CPU), "Interleaved CPU adapter logits");
            compare(reference.keys, snapshot(entry.second->k_view(0), Device::CPU), "Interleaved CPU adapter keys");
            compare(reference.values, snapshot(entry.second->v_view(0), Device::CPU), "Interleaved CPU adapter values");
        }
    }
    std::cout << "Qwen2 CPU adapters: interleaved histories preserve private outputs and KV\n";
}

void adapter_migration_test(const Parameters& source) {
    KVCache<float> cache_cpu_a(1, 8, kHidden, Device::CPU);
    KVCache<float> cache_cuda_a(1, 8, kHidden, Device::CUDA);
    KVCache<float> cache_cpu_b(1, 8, kHidden, Device::CPU);
    KVCache<float> cache_cuda_b(1, 8, kHidden, Device::CUDA);
    QwenModel<float> model(source, configuration());
    const std::vector<uint32_t> prompt{3, 5, 7};
    const auto reference = cpu_reference(source, prompt);
    Tensor<uint32_t> input_cpu(std::vector<uint32_t>(prompt), {prompt.size()});
    Tensor<uint32_t> input_cuda(std::vector<uint32_t>(prompt), {prompt.size()}, Device::CUDA);
    cache_cpu_a.resize(prompt.size());
    compare(reference.logits, snapshot(model.prefill_generic(&input_cpu, &cache_cpu_a), Device::CPU),
            "Initial CPU migration logits");
    model.cuda();
    cache_cuda_a.resize(prompt.size());
    compare(reference.logits, snapshot(model.prefill_cuda(&input_cuda, &cache_cuda_a), Device::CUDA),
            "Initial CUDA migration logits");
    Tensor<uint32_t> token(std::vector<uint32_t>{9}, {1}, Device::CUDA);
    cache_cuda_a.resize(prompt.size() + 1);
    model.forward_for_graph_logits_only(&token, &cache_cuda_a);
    model.cpu();
    cache_cpu_b.resize(prompt.size());
    compare(reference.logits, snapshot(model.prefill_generic(&input_cpu, &cache_cpu_b), Device::CPU),
            "Returned CPU migration logits");
    model.cuda();
    cache_cuda_b.resize(prompt.size());
    compare(reference.logits, snapshot(model.prefill_cuda(&input_cuda, &cache_cuda_b), Device::CUDA),
            "Reprepared CUDA migration logits");
    cache_cuda_b.resize(prompt.size() + 1);
    auto logits = model.forward_for_graph_logits_only(&token, &cache_cuda_b);
    auto history = prompt;
    history.push_back(9);
    const auto after = cpu_reference(source, history);
    compare(std::vector<float>(after.logits.end() - kVocabulary, after.logits.end()),
            snapshot(logits, Device::CUDA), "Reprepared graph migration logits");
    compare(after.keys, snapshot(cache_cuda_b.k_view(0), Device::CUDA), "Reprepared graph migration keys");
    compare(after.values, snapshot(cache_cuda_b.v_view(0), Device::CUDA), "Reprepared graph migration values");
    std::cout << "Qwen2 FP32 adapter: CPU/CUDA/CPU/CUDA migration preserves weights and fresh graph binding\n";
}

}  // namespace

int main() {
    try {
        const auto source = weights();
        auto llama = clone_checkpoint(source);
        for (const char* projection : {"q", "k", "v"}) {
            llama.erase(std::string("layers.0.self_attn.") + projection + "_proj.bias");
        }
        adapter_smoke(source, Device::CPU, false, false, "Qwen2");
        adapter_smoke(source, Device::CPU, false, true, "Qwen2");
        adapter_smoke(llama, Device::CPU, false, true, "Llama-style");
        cpu_adapter_isolation_test(source);
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) {
            std::cout << "qwen2_float_test GPU checks skipped: no CUDA device; CPU reference checks passed\n";
            return 77;
        }
        smoke(source, false);
        smoke(source, true);
        adapter_smoke(source, Device::CUDA, false, false, "Qwen2");
        adapter_smoke(source, Device::CUDA, true, true, "Qwen2");
        adapter_smoke(llama, Device::CUDA, true, true, "Llama-style");
        adapter_migration_test(source);
        std::cout << "qwen2_float_test passed\n";
    } catch (const std::exception& error) {
        std::cerr << "qwen2_float_test failed: " << error.what() << '\n';
        return 1;
    }
}
