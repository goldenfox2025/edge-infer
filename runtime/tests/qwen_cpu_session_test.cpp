#include "qwen.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <future>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using Model = QwenModel<float>;
using Parameters = Model::Parameters;
constexpr size_t kLayers = 2, kHidden = 8, kHead = 4, kIntermediate = 16;
constexpr size_t kVocabulary = 16, kCapacity = 12;

void expect(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

Tensor<float> matrix(size_t input, size_t output, size_t seed) {
    std::vector<float> values(input * output);
    for (size_t row = 0; row < input; ++row) {
        for (size_t column = 0; column < output; ++column) {
            const auto residue = static_cast<int>((row * 7 + column * 11 + seed * 3) % 19) - 9;
            values[row * output + column] = residue * 0.0125f +
                (row == column % input ? 0.15f : 0.0f);
        }
    }
    return Tensor<float>(std::move(values), {input, output});
}

Parameters checkpoint() {
    Parameters result;
    std::vector<float> embedding(kVocabulary * kHidden);
    for (size_t token = 0; token < kVocabulary; ++token) {
        for (size_t feature = 0; feature < kHidden; ++feature) {
            const auto residue = static_cast<int>((token * 13 + feature * 7 + token * feature) % 23) - 11;
            embedding[token * kHidden + feature] = (residue + 0.5f) * 0.03f;
        }
    }
    result.emplace("token_embeddings.weight", Tensor<float>(std::move(embedding), {kVocabulary, kHidden}));
    result.emplace("rms_out_w", Tensor<float>(std::vector<float>(kHidden, 1.0f), {kHidden}));
    result.emplace("lm_head", matrix(kHidden, kVocabulary, 100));
    for (size_t layer = 0; layer < kLayers; ++layer) {
        const auto suffix = std::to_string(layer);
        for (const char* prefix : {"rms_att_w", "rms_ffn_w"}) {
            result.emplace(std::string(prefix) + suffix,
                           Tensor<float>(std::vector<float>(kHidden, 1.0f), {kHidden}));
        }
        result.emplace("wq" + suffix, matrix(kHidden, kHidden, 10 * layer + 1));
        result.emplace("wk" + suffix, matrix(kHidden, kHead, 10 * layer + 2));
        result.emplace("wv" + suffix, matrix(kHidden, kHead, 10 * layer + 3));
        result.emplace("wo" + suffix, matrix(kHidden, kHidden, 10 * layer + 4));
        result.emplace("w_gate" + suffix, matrix(kHidden, kIntermediate, 10 * layer + 5));
        result.emplace("w_up" + suffix, matrix(kHidden, kIntermediate, 10 * layer + 6));
        result.emplace("w_down" + suffix, matrix(kIntermediate, kHidden, 10 * layer + 7));
    }
    return result;
}

ModelConfig configuration() {
    return {{"vocab_size", kVocabulary}, {"n_layers", kLayers}, {"n_heads", 2},
            {"n_kv_heads", 1}, {"hidden_size", kHidden}, {"head_dim", kHead},
            {"intermediate_size", kIntermediate}, {"max_position_embeddings", kCapacity},
            {"bos_token_id", 1}, {"eos_token_id", 15}, {"rms_norm_eps", 1.0e-6},
            {"rope_theta", 10000}};
}

std::shared_ptr<Model> fork(const Model& prototype) {
    auto result = std::dynamic_pointer_cast<Model>(prototype.fork_executor());
    expect(result && result->device() == Device::CPU && result->verify_params(),
           "CPU executor fork must retain a prepared FP32 model");
    return result;
}

void expect_shared_weights(const Model& prototype, const Model& child) {
    const auto& source = prototype.get_params();
    const auto& target = child.get_params();
    expect(source.size() == target.size(), "CPU fork changed weight count");
    for (const auto& entry : source) {
        const auto& weight = target.at(entry.first);
        expect(weight.device() == Device::CPU && weight.sizes() == entry.second.sizes() &&
                   weight.strides() == entry.second.strides() &&
                   weight.data_ptr() == entry.second.data_ptr(),
               "CPU fork copied or changed private weight " + entry.first);
    }
}

std::vector<float> values(TensorView<const float, 2> view) {
    std::vector<float> result;
    result.reserve(view.numel());
    for (size_t row = 0; row < view.shape[0]; ++row) {
        for (size_t column = 0; column < view.shape[1]; ++column) {
            const float value = view(row, column);
            expect(std::isfinite(value), "CPU output must be finite");
            result.push_back(value);
        }
    }
    return result;
}

struct State {
    std::vector<float> logits, keys, cache_values;
};

State snapshot(TensorView<const float, 2> logits, const KVCache<float>& cache) {
    State result{values(logits), {}, {}};
    for (size_t layer = 0; layer < kLayers; ++layer) {
        const auto keys = values(cache.k_view(layer));
        const auto cached_values = values(cache.v_view(layer));
        result.keys.insert(result.keys.end(), keys.begin(), keys.end());
        result.cache_values.insert(result.cache_values.end(), cached_values.begin(), cached_values.end());
    }
    return result;
}

void expect_state(const State& expected, const State& actual, const char* label) {
    expect(expected.logits == actual.logits && expected.keys == actual.keys &&
               expected.cache_values == actual.cache_values,
           std::string(label) + " changed logits or active KV storage");
}

Tensor<uint32_t> token_tensor(const std::vector<uint32_t>& tokens) {
    return Tensor<uint32_t>(std::vector<uint32_t>(tokens), {tokens.size()}, Device::CPU);
}

std::vector<State> sequence(Model& executor, const std::vector<uint32_t>& prompt,
                            const std::array<uint32_t, 3>& continuation) {
    KVCache<float> cache(kLayers, kCapacity, kHead, Device::CPU);
    cache.resize(prompt.size());
    auto input = token_tensor(prompt);
    auto logits = executor.prefill_generic(&input, &cache);
    std::vector<State> result{snapshot(logits.as_const(), cache)};
    for (const auto token : continuation) {
        auto next = token_tensor({token});
        cache.resize(cache.size() + 1);
        logits = executor.forward_generic(&next, &cache);
        result.push_back(snapshot(logits.as_const(), cache));
    }
    return result;
}

void test_shared_weights_and_private_execution() {
    auto source = checkpoint();
    auto prototype = std::make_shared<Model>(source, configuration());
    for (const auto& entry : source) {
        expect(prototype->get_params().at(entry.first).data_ptr() != entry.second.data_ptr(),
               "Initial CPU preparation must snapshot caller-owned weight " + entry.first);
    }
    auto first = fork(*prototype), second = fork(*prototype);
    expect_shared_weights(*prototype, *first);
    expect_shared_weights(*prototype, *second);

    const std::vector<uint32_t> prompt_a{3, 5, 7}, prompt_b{2, 4};
    const std::array<uint32_t, 3> next_a{9, 11, 1}, next_b{6, 8, 10};
    auto reference_a = fork(*prototype), reference_b = fork(*prototype);
    const auto expected_a = sequence(*reference_a, prompt_a, next_a);
    const auto expected_b = sequence(*reference_b, prompt_b, next_b);
    expect(expected_a.back().logits != expected_b.back().logits,
           "Distinct histories must exercise distinguishable CPU outputs");
    reference_a.reset();
    reference_b.reset();

    // Caller checkpoint mutation must not affect the private model snapshot or
    // any fork. Forked Tensor descriptors keep that snapshot alive themselves.
    for (auto& entry : source) std::fill_n(entry.second.data_ptr(), entry.second.numel(), -100.0f);
    source.clear();
    prototype.reset();

    KVCache<float> cache_a(kLayers, kCapacity, kHead, Device::CPU);
    KVCache<float> cache_b(kLayers, kCapacity, kHead, Device::CPU);
    auto input_a = token_tensor(prompt_a), input_b = token_tensor(prompt_b);
    cache_a.resize(prompt_a.size());
    cache_b.resize(prompt_b.size());
    auto logits_a = first->prefill_generic(&input_a, &cache_a);
    const auto held_prefill = snapshot(logits_a.as_const(), cache_a);
    auto logits_b = second->prefill_generic(&input_b, &cache_b);
    expect(logits_a.data != logits_b.data, "CPU forks must own different prefill output storage");
    expect_state(expected_a[0], snapshot(logits_a.as_const(), cache_a), "First prefill");
    expect_state(expected_b[0], snapshot(logits_b.as_const(), cache_b), "Second prefill");
    expect_state(held_prefill, snapshot(logits_a.as_const(), cache_a), "Held first prefill");
    const float* fixed_a = nullptr;
    const float* fixed_b = nullptr;
    for (size_t step = 0; step < next_a.size(); ++step) {
        auto token_a = token_tensor({next_a[step]}), token_b = token_tensor({next_b[step]});
        cache_a.resize(cache_a.size() + 1);
        cache_b.resize(cache_b.size() + 1);
        logits_a = first->forward_generic(&token_a, &cache_a);
        const auto held_a = snapshot(logits_a.as_const(), cache_a);
        logits_b = second->forward_generic(&token_b, &cache_b);
        expect(logits_a.data != logits_b.data && (!fixed_a || fixed_a == logits_a.data) &&
                   (!fixed_b || fixed_b == logits_b.data),
               "CPU forks must own separate stable decode storage");
        fixed_a = logits_a.data;
        fixed_b = logits_b.data;
        expect_state(expected_a[step + 1], snapshot(logits_a.as_const(), cache_a), "First decode");
        expect_state(expected_b[step + 1], snapshot(logits_b.as_const(), cache_b), "Second decode");
        expect_state(held_a, snapshot(logits_a.as_const(), cache_a), "Held first decode");
    }

    auto retained = fork(*first);
    expect_shared_weights(*first, *retained);
    first.reset();
    auto concurrent_a = std::async(std::launch::async, [&] { return sequence(*retained, prompt_a, next_a); });
    auto concurrent_b = std::async(std::launch::async, [&] { return sequence(*second, prompt_b, next_b); });
    const auto actual_a = concurrent_a.get(), actual_b = concurrent_b.get();
    for (size_t step = 0; step < expected_a.size(); ++step) {
        expect_state(expected_a[step], actual_a[step], "Retained independent CPU executor");
        expect_state(expected_b[step], actual_b[step], "Concurrent independent CPU executor");
    }
}

void test_private_sampling_state() {
    const auto source = checkpoint();
    Model prototype(source, configuration());
    auto first = fork(prototype), second = fork(prototype);
    auto expected_first = fork(prototype), expected_second = fork(prototype);
    std::vector<float> scores(kVocabulary);
    for (size_t index = 0; index < scores.size(); ++index) scores[index] = index * 0.125f;
    Tensor<float> logits(std::move(scores), {1, kVocabulary});
    const auto unchanged = values(borrow_tensor_view<2>(static_cast<const Tensor<float>&>(logits)));
    for (size_t iteration = 0; iteration < 16; ++iteration) {
        expect(first->sample_cpu(logits, 0.8f, 0.75f, 8) ==
                   expected_first->sample_cpu(logits, 0.8f, 0.75f, 8),
               "Other CPU executor changed the first sampling sequence");
        for (size_t extra = 0; extra < 3; ++extra) {
            expect(second->sample_cpu(logits, 0.8f, 0.75f, 8) ==
                       expected_second->sample_cpu(logits, 0.8f, 0.75f, 8),
                   "CPU forks must own independent RNG and sampling scratch");
        }
    }
    expect(unchanged == values(borrow_tensor_view<2>(static_cast<const Tensor<float>&>(logits))),
           "CPU sampling must preserve borrowed logits");
}

}  // namespace

int main() {
    try {
        test_shared_weights_and_private_execution();
        test_private_sampling_state();
        std::cout << "CPU sessions share private weights, retain their lifetime, and isolate execution and sampling\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "CPU session test failed: " << error.what() << '\n';
        return 1;
    }
}
