#include "qwen3.hpp"
#include "speculative_decoder.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using Scalar = __nv_bfloat16;
using Model = Qwen3Model<Scalar>;
using Session = Qwen3Session<Scalar>;
using Engine = InferenceEngine<Scalar>;
using Speculation = SpeculativeDecoder<Scalar>;
using Parameters = Model::Parameters;

constexpr std::size_t kHidden = 128, kIntermediate = 256, kVocabulary = 64, kCapacity = 16;

void expect(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

ModelConfig configuration() {
    // An unreachable EOS sentinel ensures every requested verification window
    // executes. EOS callback policy is tested separately from this fixture.
    return {{"vocab_size", kVocabulary}, {"n_layers", 1}, {"n_heads", 1}, {"n_kv_heads", 1},
            {"hidden_size", kHidden}, {"head_dim", kHidden}, {"intermediate_size", kIntermediate},
            {"max_position_embeddings", kCapacity}, {"bos_token_id", 1},
            {"eos_token_id", std::numeric_limits<uint32_t>::max()},
            {"rms_norm_eps", 1.0e-6}, {"rope_theta", 10000}};
}

Tensor<Scalar> norm(std::size_t width) {
    return Tensor<Scalar>(std::vector<Scalar>(width, __float2bfloat16(1.0f)), {width});
}

Tensor<Scalar> matrix(std::size_t input, std::size_t output, std::size_t seed, float diagonal) {
    std::vector<Scalar> data(input * output);
    for (std::size_t row = 0; row < output; ++row) {
        for (std::size_t feature = 0; feature < input; ++feature) {
            const int residue = static_cast<int>((row * 37 + feature * 19 + seed * 11) % 23) - 11;
            const float value = (residue + 0.25f) * 0.001f + (feature == row % input ? diagonal : 0.0f);
            data[row * input + feature] = __float2bfloat16(value);
        }
    }
    return Tensor<Scalar>(std::move(data), {output, input}).transpose(0, 1);
}

// Reuse the nonzero fixture definition from qwen3_session_test: transposed
// checkpoint matrices, deterministic embeddings, and a complete decoder block.
Parameters weights() {
    Parameters result;
    std::vector<Scalar> embeddings(kVocabulary * kHidden);
    for (std::size_t token = 0; token < kVocabulary; ++token) {
        for (std::size_t feature = 0; feature < kHidden; ++feature) {
            const int residue = static_cast<int>((token * 29 + feature * 13 + token * feature * 3) % 71) - 35;
            embeddings[token * kHidden + feature] = __float2bfloat16((residue + 0.5f) * 0.02f);
        }
    }
    result.emplace("token_embeddings.weight", Tensor<Scalar>(std::move(embeddings), {kVocabulary, kHidden}));
    for (const char* name : {"rms_out_w", "rms_att_w0", "rms_ffn_w0", "q_norm0", "k_norm0"}) {
        result.emplace(name, norm(kHidden));
    }
    result.emplace("wq0", matrix(kHidden, kHidden, 1, 0.125f));
    result.emplace("wk0", matrix(kHidden, kHidden, 2, 0.125f));
    result.emplace("wv0", matrix(kHidden, kHidden, 3, 0.125f));
    result.emplace("wo0", matrix(kHidden, kHidden, 4, 0.125f));
    result.emplace("w_gate0", matrix(kHidden, kIntermediate, 5, 0.08f));
    result.emplace("w_up0", matrix(kHidden, kIntermediate, 6, 0.08f));
    result.emplace("w_down0", matrix(kIntermediate, kHidden, 7, 0.08f));
    result.emplace("lm_head", matrix(kHidden, kVocabulary, 8, 0.25f));
    return result;
}

uint32_t argmax(TensorView<Scalar, 2> logits) {
    const auto row = logits.select<0>(logits.shape[0] - 1);
    std::vector<Scalar> values(row.numel());
    check(cudaMemcpy(values.data(), row.data, values.size() * sizeof(Scalar), cudaMemcpyDeviceToHost));
    return static_cast<uint32_t>(std::max_element(values.begin(), values.end(), [](Scalar a, Scalar b) {
        return __bfloat162float(a) < __bfloat162float(b);
    }) - values.begin());
}

std::vector<uint32_t> reference(const std::shared_ptr<const Model>& model,
                                const std::vector<uint32_t>& prompt, std::size_t maximum) {
    if (maximum <= prompt.size()) return {};
    auto session = Session::create(model, kCapacity, false);
    Tensor<uint32_t> input(std::vector<uint32_t>(prompt), {prompt.size()}, Device::CUDA);
    auto logits = session->prefill(input);
    std::vector<uint32_t> output;
    while (prompt.size() + output.size() < maximum) {
        const auto token = argmax(logits);
        output.push_back(token);
        if (prompt.size() + output.size() < maximum) logits = session->decode(token);
    }
    return output;
}

std::vector<uint32_t> collect(infer_base& frontend, const std::vector<uint32_t>& prompt,
                              std::size_t maximum) {
    std::vector<uint32_t> output;
    frontend.generate_with_callback(prompt, maximum, 1.0f, 1.0f, 1,
        [&](uint32_t token) {
            expect(token < kVocabulary, "Generation callback token is outside vocabulary");
            output.push_back(token);
        });
    return output;
}

void same(const std::vector<uint32_t>& actual, const std::vector<uint32_t>& expected,
          const std::string& label) {
    expect(actual == expected, label + " greedy callback sequence differs from the ordinary decoder");
}

void generation_test() {
    auto immutable = std::make_shared<Model>(weights(), configuration());
    auto source = std::make_shared<Session>(immutable, false);
    auto draft = std::make_shared<Session>(immutable, false);
    const std::vector<uint32_t> prompt_a{3, 7, 11}, prompt_b{17, 23, 29, 31};
    constexpr std::size_t maximum = 14;
    const auto expected_a = reference(immutable, prompt_a, maximum);
    const auto expected_b = reference(immutable, prompt_b, maximum);
    expect(expected_a.size() == maximum - prompt_a.size() &&
               expected_b.size() == maximum - prompt_b.size(),
           "Synthetic reference must reach full requested length");

    Engine ordinary(source, Device::CUDA);
    expect(!ordinary.has_warmed_up_, "New engine must own a fresh warmup state");
    same(collect(ordinary, prompt_a, maximum), expected_a, "Initial ordinary engine");
    expect(ordinary.has_warmed_up_, "First ordinary request must prepare this engine");
    same(collect(ordinary, prompt_b, maximum), expected_b, "Ordinary complete prompt replaces prior history");
    same(collect(ordinary, prompt_a, maximum), expected_a, "Ordinary prompt replay without explicit reset");
    ordinary.reset();
    same(collect(ordinary, prompt_b, maximum), expected_b, "Ordinary reset/new prompt");
    ordinary.reset();
    same(collect(ordinary, prompt_a, prompt_a.size()), {}, "Ordinary max_length equal to prompt");
    same(collect(ordinary, prompt_a, prompt_a.size() - 1), {}, "Ordinary max_length below prompt");

    for (const std::size_t window : {std::size_t{1}, std::size_t{6}, std::size_t{8}}) {
        Speculation speculative(source, draft, window, 1);
        same(collect(speculative, prompt_a, maximum), expected_a, "Speculation window " + std::to_string(window));
        same(collect(speculative, prompt_b, maximum), expected_b, "Repeated speculative request");
        same(collect(speculative, prompt_a, prompt_a.size()), {}, "Speculative max_length equal to prompt");
        same(collect(speculative, prompt_a, prompt_a.size() - 1), {}, "Speculative max_length below prompt");
        const auto short_expected = reference(immutable, prompt_a, prompt_a.size() + 3);
        same(collect(speculative, prompt_a, prompt_a.size() + 3), short_expected, "Short final speculative segment");
        same(collect(speculative, prompt_a, maximum), expected_a, "Speculative prompt replay");
        const auto held_extent = speculative.context_size();
        for (std::size_t unsupported : {std::size_t{0}, std::size_t{2}, std::size_t{50}}) {
            bool rejected = false;
            try {
                speculative.generate_with_callback(prompt_b, maximum, 1.0f, 1.0f, unsupported,
                    [](uint32_t) { throw std::runtime_error("Rejected sampling policy invoked callback"); });
            } catch (const std::invalid_argument&) { rejected = true; }
            expect(rejected && speculative.context_size() == held_extent,
                   "Non-greedy speculative policy must reject before changing request state");
        }
        ordinary.reset();
        same(collect(ordinary, prompt_a, maximum), expected_a, "Ordinary engine after speculation");
        std::cout << "Greedy speculative initial window=" << window
                  << ": full/repeated/short/replayed sequences match ordinary decoding\n";
    }

    // Reverse output vocabulary rows in a private draft model. With an even
    // vocabulary every uniquely maximal proposal differs from the target,
    // exercising replacement and removal of rejected draft cache tails.
    auto rejected_weights = weights();
    auto* head = rejected_weights.at("lm_head").data_ptr();
    for (std::size_t row = 0; row < kVocabulary / 2; ++row) {
        for (std::size_t feature = 0; feature < kHidden; ++feature) {
            std::swap(head[row * kHidden + feature], head[(kVocabulary - 1 - row) * kHidden + feature]);
        }
    }
    auto rejected_model = std::make_shared<Model>(rejected_weights, configuration());
    auto rejected_draft = std::make_shared<Session>(rejected_model, false);
    {
        Speculation speculative(source, rejected_draft, 8, 1);
        same(collect(speculative, prompt_a, maximum), expected_a, "Rejected draft proposals");
        same(collect(speculative, prompt_b, maximum), expected_b, "Rejected draft repeated request");
    }
    ordinary.reset();
    same(collect(ordinary, prompt_b, maximum), expected_b, "Ordinary after rejected speculation destruction");

    {
        Engine independent(source, Device::CUDA);
        expect(!independent.has_warmed_up_, "Engine warmup must not leak across instances");
        same(collect(independent, prompt_b, maximum), expected_b, "Independent frontend sharing source");
        ordinary.reset();
        same(collect(ordinary, prompt_a, maximum), expected_a, "Original frontend after independent execution");
    }
    {
        Engine recreated(source, Device::CUDA);
        same(collect(recreated, prompt_a, maximum), expected_a, "Recreated frontend after prior destruction");
    }
    {
        constexpr std::size_t capacity = 8;
        Engine bounded(source, Device::CUDA, capacity);
        Speculation speculative(source, draft, 6, 1, capacity);
        const auto expected = reference(immutable, prompt_a, capacity);
        expect(bounded.context_capacity() == capacity && speculative.context_capacity() == capacity,
               "Bounded frontends must reserve the requested context capacity");
        same(collect(bounded, prompt_a, maximum), expected, "Bounded ordinary generation");
        same(collect(speculative, prompt_a, maximum), expected, "Bounded speculative generation");
        bounded.reset(); speculative.reset();
        expect(!bounded.context_size() && !speculative.context_size(), "Reset must clear bounded frontend histories");
        bool rejected = false;
        try { Engine invalid(source, Device::CUDA, kCapacity + 1); }
        catch (const std::invalid_argument&) { rejected = true; }
        expect(rejected, "Ordinary capacity beyond model limit must reject");
        rejected = false;
        try { Speculation invalid(source, draft, 6, 1, kCapacity + 1); }
        catch (const std::invalid_argument&) { rejected = true; }
        expect(rejected, "Speculative capacity beyond a model limit must reject");
    }
    {
        auto eos_configuration = configuration();
        eos_configuration["eos_token_id"] = expected_a.front();
        auto eos_model = std::make_shared<Model>(weights(), eos_configuration);
        auto eos_source = std::make_shared<Session>(eos_model, false);
        Engine ordinary_eos(eos_source, Device::CUDA);
        Speculation speculative_eos(eos_source, eos_source, 6, 1);
        same(collect(ordinary_eos, prompt_a, maximum), {}, "Ordinary initial EOS is not emitted");
        same(collect(speculative_eos, prompt_a, maximum), {}, "Speculative initial EOS is not emitted");
    }
    std::cout << "Engine/speculative executors: shared immutable weights survive independent cache binding and destruction\n";
}

}  // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) {
        std::cout << "inference_generation_test skipped: no CUDA device\n";
        return 77;
    }
    try {
        generation_test();
        std::cout << "inference_generation_test passed\n";
    } catch (const std::exception& error) {
        std::cerr << "inference_generation_test failed: " << error.what() << '\n';
        return 1;
    }
}
