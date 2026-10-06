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

class WarmupError : public std::runtime_error {
 public:
    WarmupError() : std::runtime_error("injected warmup prefill failure") {}
};

class OrdinaryDecodeError : public std::runtime_error {
 public:
    OrdinaryDecodeError() : std::runtime_error("injected ordinary decode failure") {}
};

// No model kernels are needed to exercise the CUDA-only warmup boundary.
// Keep a real device token for a successful retry after recoverable failure.
class WarmupModel final : public BaseModel {
 public:
    explicit WarmupModel(bool fail_completion_on_prefill)
        : fail_completion_on_prefill_(fail_completion_on_prefill),
          token_(std::vector<uint32_t>{2}, {1}, Device::CUDA) {}

    uint32_t* prefill(const Tensor<uint32_t>*, ThreadPool&, KVCacheBase* cache,
                      size_t, float, float, curandState*) override {
        ++prefill_calls;
        last_cache = static_cast<KVCache<Scalar>*>(cache);
        k_backing = last_cache->k_view(0).data;
        v_backing = last_cache->v_view(0).data;
        if (fail_prefill) {
            fail_completion = fail_completion_on_prefill_;
            throw WarmupError{};
        }
        return token_.data_ptr();
    }
    uint32_t* forward(const Tensor<uint32_t>*, ThreadPool&, KVCacheBase* cache,
                      size_t, float, float, curandState*) override {
        ++decode_calls;
        last_cache = static_cast<KVCache<Scalar>*>(cache);
        if (fail_decode) {
            // Real submitted work can alter old KV before an executor rejects
            // the remaining operation. It may also discard its logical cache.
            check(cudaMemsetAsync(last_cache->k_capacity_view(0).data, 0, sizeof(Scalar)));
            if (clear_failed_decode_cache) last_cache->clear();
            fail_completion = fail_completion_on_decode;
            throw OrdinaryDecodeError{};
        }
        return token_.data_ptr();
    }
    void synchronize() const override {
        ++synchronize_calls;
        synchronized_cache_size = last_cache ? last_cache->size() : 0;
        if (fail_completion || synchronize_calls == fail_once_completion_call)
            throw std::runtime_error("injected warmup completion failure");
        check(cudaDeviceSynchronize());
    }
    bool verify_params() const override { return true; }
    void print_model_info() const override {}
    BaseModel& cuda() override { return *this; }
    BaseModel& cpu() override { throw std::runtime_error("CUDA-only warmup fixture"); }
    Device device() const override { return Device::CUDA; }
    size_t get_n_layers() const override { return 1; }
    size_t get_max_seq_len() const override { return 4; }
    size_t get_head_dim() const override { return 1; }
    size_t get_n_kv_heads() const override { return 1; }
    size_t get_vocab_size() const override { return 4; }
    uint32_t get_eos_token_id() const override { return 3; }
    size_t get_hidden_size() const override { return 1; }

    bool fail_prefill = true, fail_completion = false;
    bool fail_decode = false, clear_failed_decode_cache = false, fail_completion_on_decode = false;
    int fail_once_completion_call = 0;
    int prefill_calls = 0, decode_calls = 0;
    mutable int synchronize_calls = 0;
    mutable size_t synchronized_cache_size = 0;
    KVCache<Scalar>* last_cache = nullptr;
    Scalar *k_backing = nullptr, *v_backing = nullptr;

 private:
    bool fail_completion_on_prefill_;
    Tensor<uint32_t> token_;
};

void warmup_failure_test() {
    for (bool automatic : {false, true}) {
        for (bool failed_completion : {false, true}) {
            auto model = std::make_shared<WarmupModel>(failed_completion);
            Engine engine(model, Device::CUDA, 4);
            int callbacks = 0;
            bool caught = false;
            try {
                if (automatic) {
                    engine.generate_with_callback({1}, 4, 1.0f, 1.0f, 1,
                                                  [&](uint32_t) { ++callbacks; });
                } else {
                    engine.warmup(2, false, 1.0f, 1.0f, 1);
                }
            } catch (const WarmupError& error) {
                caught = std::string(error.what()) == "injected warmup prefill failure";
            }
            expect(caught && model->prefill_calls == 1 && model->decode_calls == 0 &&
                       callbacks == 0 && !engine.has_warmed_up_,
                   "Direct and automatic warmup must preserve the original prefill error");
            expect(model->synchronize_calls == (automatic ? 2 : 1),
                   "Warmup failure must attempt executor completion exactly once");
            model->fail_prefill = false;
            model->fail_completion = false;
            if (failed_completion) {
                expect(model->last_cache->size() == (automatic ? 4 : 2) &&
                           model->last_cache->k_view(0).data == model->k_backing &&
                           model->last_cache->v_view(0).data == model->v_backing,
                       "Failed warmup completion must retain cache extent and backing");
                const int completions = model->synchronize_calls;
                const auto rejects = [](auto operation) {
                    bool rejected = false;
                    try { operation(); }
                    catch (const std::logic_error& error) {
                        rejected = std::string(error.what()).find("construct a new inference engine") !=
                                   std::string::npos;
                    }
                    expect(rejected, "Failed warmup completion must invalidate the engine");
                };
                rejects([&] { engine.reset(); });
                rejects([&] { engine.warmup(2, false, 1.0f, 1.0f, 1); });
                rejects([&] { engine.generate_with_callback({1}, 4, 1.0f, 1.0f, 1,
                                                            [&](uint32_t) { ++callbacks; }); });
                rejects([&] { engine.context_size(); });
                rejects([&] { engine.context_capacity(); });
                rejects([&] { engine.device(); });
                expect(model->synchronize_calls == completions && model->prefill_calls == 1 &&
                           model->decode_calls == 0 && callbacks == 0,
                       "Invalid warmup engine must reject before touching its executor");
            } else {
                expect(engine.context_size() == 0,
                       "Successful warmup failure cleanup must clear logical cache");
                std::vector<uint32_t> output;
                engine.generate_with_callback({1}, 4, 1.0f, 1.0f, 1,
                                              [&](uint32_t token) { output.push_back(token); });
                expect(output == std::vector<uint32_t>({2, 2, 2}) && engine.has_warmed_up_ &&
                           engine.context_size() == 3,
                       "Successful warmup failure cleanup must allow the same engine to retry");
            }
        }
    }
}

void ordinary_direct_failure_test() {
    Tensor<uint32_t> token(std::vector<uint32_t>{1}, {1}, Device::CUDA);
    ThreadPool unused(0);
    for (bool failed_completion : {false, true}) {
        for (bool executor_cleared_cache : {false, true}) {
            auto model = std::make_shared<WarmupModel>(false);
            model->fail_prefill = false;
            Engine engine(model, Device::CUDA, 4);
            engine.generate_with_callback({1}, 4, 1.0f, 1.0f, 1, [](uint32_t) {});
            expect(engine.context_size() == 3, "Direct decode fixture must have committed history");
            const auto* k_backing = model->k_backing;
            const auto* v_backing = model->v_backing;
            const int completions = model->synchronize_calls;
            model->fail_decode = true;
            model->clear_failed_decode_cache = executor_cleared_cache;
            model->fail_completion_on_decode = failed_completion;
            bool original_error = false;
            try { engine.generate_next_token(unused, token.data_ptr(), 1.0f, 1.0f, 1); }
            catch (const OrdinaryDecodeError&) { original_error = true; }
            const size_t submitted_size = executor_cleared_cache ? 0 : 4;
            expect(original_error && model->synchronize_calls == completions + 1 &&
                       model->synchronized_cache_size == submitted_size,
                   "Direct decode must preserve its error and drain before changing cache extent");
            expect(model->last_cache->size() == (failed_completion ? submitted_size : 0) &&
                       model->last_cache->k_capacity_view(0).data == k_backing &&
                       model->last_cache->v_capacity_view(0).data == v_backing,
                   "Direct failure must never restore discarded history or release cache backing");
            model->fail_decode = false;
            model->fail_completion = false;
            if (failed_completion) {
                // A later successful backend wait cannot revive this engine.
                model->synchronize();
                const auto rejects = [](auto operation) {
                    bool invalid = false;
                    try { operation(); }
                    catch (const std::logic_error&) { invalid = true; }
                    expect(invalid, "Failed direct completion must permanently invalidate the engine");
                };
                rejects([&] { engine.context_size(); });
                rejects([&] { engine.reset(); });
                rejects([&] { engine.generate_next_token(unused, token.data_ptr(), 1.0f, 1.0f, 1); });
            } else {
                expect(engine.context_size() == 0,
                       "Completed direct failure cleanup must leave an empty request");
                int callbacks = 0;
                engine.generate_with_callback({1}, 4, 1.0f, 1.0f, 1,
                                              [&](uint32_t) { ++callbacks; });
                expect(callbacks == 3 && engine.context_size() == 3,
                       "Completed direct failure cleanup must permit a fresh request");
                engine.generate_next_token(unused, token.data_ptr(), 1.0f, 1.0f, 1);
                const int decoded = model->decode_calls;
                bool full = false;
                try { engine.generate_next_token(unused, token.data_ptr(), 1.0f, 1.0f, 1); }
                catch (const std::length_error&) { full = true; }
                expect(full && engine.context_size() == 4 && model->decode_calls == decoded,
                       "Capacity rejection must preserve completed history and avoid execution");
            }
        }
    }
}

void ordinary_completion_invalidation_test() {
    for (bool warmup : {false, true}) {
        auto model = std::make_shared<WarmupModel>(false);
        model->fail_prefill = false;
        Engine engine(model, Device::CUDA, 4);
        if (!warmup)
            engine.generate_with_callback({1}, 4, 1.0f, 1.0f, 1, [](uint32_t) {});
        model->fail_once_completion_call = model->synchronize_calls + 1;
        bool original_error = false;
        try {
            if (warmup) engine.warmup(2, false, 1.0f, 1.0f, 1);
            else engine.reset();
        } catch (const std::runtime_error& error) {
            original_error = std::string(error.what()) == "injected warmup completion failure";
        }
        expect(original_error && model->last_cache->size() == 3 &&
                   model->last_cache->k_capacity_view(0).data == model->k_backing &&
                   model->last_cache->v_capacity_view(0).data == model->v_backing,
               "Reset and warmup completion failure must retain extent, backing and original error");
        if (warmup) expect(!engine.has_warmed_up_, "Failed warmup completion cannot mark preparation complete");
        model->synchronize();
        const auto rejects = [](auto operation) {
            bool invalid = false;
            try { operation(); }
            catch (const std::logic_error&) { invalid = true; }
            expect(invalid, "A successful cleanup retry cannot revive a failed engine");
        };
        rejects([&] { engine.context_size(); });
        rejects([&] { engine.context_capacity(); });
        rejects([&] { engine.device(); });
        rejects([&] { engine.reset(); });
        rejects([&] { engine.warmup(); });
        rejects([&] { engine.generate_with_callback({1}, 4, 1.0f, 1.0f, 1, [](uint32_t) {}); });
    }
}

class SpeculativePrefillError : public std::runtime_error {
 public:
    SpeculativePrefillError() : std::runtime_error("injected speculative prefill failure") {}
};

class SpeculativeCompletionError : public std::runtime_error {
 public:
    explicit SpeculativeCompletionError(const std::string& owner)
        : std::runtime_error(owner + " completion failed") {}
};

class SpeculativeFixture final : public BaseModel, public SpeculativeModel<Scalar> {
 public:
    explicit SpeculativeFixture(std::string owner)
        : owner_(std::move(owner)), logits_(fixture_logits(), {8, 4}, Device::CUDA) {
        check(cudaGetDevice(&device_));
    }
    TensorView<Scalar, 2> speculative_prefill_logits(const Tensor<uint32_t>* input,
                                                   KVCache<Scalar>* cache) override {
        require_device();
        ++prefill_calls;
        bind(cache);
        if (fail_prefill) {
            if (on_prefill_failure) on_prefill_failure();
            throw SpeculativePrefillError{};
        }
        return {logits_.data_ptr(), {input->numel(), 4}, {4, 1}};
    }
    TensorView<Scalar, 2> speculative_forward_logits(const Tensor<uint32_t>*,
                                                   KVCache<Scalar>* cache) override {
        require_device();
        ++decode_calls;
        bind(cache);
        return {logits_.data_ptr(), {1, 4}, {4, 1}};
    }
    uint32_t* prefill(const Tensor<uint32_t>*, ThreadPool&, KVCacheBase*,
                      size_t, float, float, curandState*) override {
        throw std::logic_error("Speculative fixture exposes logits only");
    }
    uint32_t* forward(const Tensor<uint32_t>*, ThreadPool&, KVCacheBase*,
                      size_t, float, float, curandState*) override {
        throw std::logic_error("Speculative fixture exposes logits only");
    }
    void synchronize() const override {
        require_device();
        ++synchronize_calls;
        if (fail_completion) throw SpeculativeCompletionError(owner_);
        check(cudaDeviceSynchronize());
    }
    bool verify_params() const override { return true; }
    void print_model_info() const override {}
    BaseModel& cuda() override { return *this; }
    BaseModel& cpu() override { throw std::runtime_error("CUDA-only speculative fixture"); }
    Device device() const override { return Device::CUDA; }
    size_t get_n_layers() const override { return 1; }
    size_t get_max_seq_len() const override { return 8; }
    size_t get_head_dim() const override { return 1; }
    size_t get_n_kv_heads() const override { return 1; }
    size_t get_vocab_size() const override { return 4; }
    uint32_t get_eos_token_id() const override { return 3; }
    size_t get_hidden_size() const override { return 1; }

    bool fail_prefill = false, fail_completion = false;
    std::function<void()> on_prefill_failure;
    int prefill_calls = 0, decode_calls = 0;
    mutable int synchronize_calls = 0;
    KVCache<Scalar>* last_cache = nullptr;
    Scalar *k_backing = nullptr, *v_backing = nullptr;

 private:
    static std::vector<Scalar> fixture_logits() {
        std::vector<Scalar> values(8 * 4, __float2bfloat16(0.0f));
        for (size_t row = 0; row < 8; ++row) values[row * 4 + 2] = __float2bfloat16(4.0f);
        return values;
    }
    void require_device() const {
        int selected = -1;
        check(cudaGetDevice(&selected));
        expect(selected == device_, "Speculative native phases must select their own CUDA device");
    }
    void bind(KVCache<Scalar>* cache) {
        last_cache = cache;
        k_backing = cache->k_view(0).data;
        v_backing = cache->v_view(0).data;
    }
    std::string owner_;
    Tensor<Scalar> logits_;
    int device_ = -1;
};

void expect_invalid_speculation(Speculation& decoder, int& callbacks) {
    const auto rejects = [](auto operation) {
        bool rejected = false;
        try { operation(); }
        catch (const std::logic_error& error) {
            rejected = std::string(error.what()).find("construct a new speculative decoder") != std::string::npos;
        }
        expect(rejected, "Failed executor completion must invalidate every public speculative operation");
    };
    rejects([&] { decoder.reset(); });
    rejects([&] { decoder.generate_with_callback({}, 0, 1.0f, 1.0f, 2,
                                                [&](uint32_t) { ++callbacks; }); });
    rejects([&] { decoder.context_size(); });
    rejects([&] { decoder.context_capacity(); });
    rejects([&] { decoder.get_spec_length(); });
    rejects([&] { decoder.device(); });
}

void expect_retained_speculative_cache(const SpeculativeFixture& model, size_t extent) {
    expect(model.last_cache && model.last_cache->size() == extent &&
               model.last_cache->k_view(0).data == model.k_backing &&
               model.last_cache->v_view(0).data == model.v_backing,
           "Failed speculative completion must retain cache extent and backing");
}

void speculative_completion_test() {
    // Completion fails in neither executor, in the target, or in the draft.
    // A draft prefill failure occurs after both caches have acquired history.
    for (int failed_executor : {0, 1, 2}) {
        auto target = std::make_shared<SpeculativeFixture>("target");
        auto draft = std::make_shared<SpeculativeFixture>("draft");
        draft->fail_prefill = true;
        draft->on_prefill_failure = [&] {
            target->fail_completion = failed_executor == 1;
            draft->fail_completion = failed_executor == 2;
        };
        Speculation decoder(target, draft, 3, 0, 8);
        std::vector<uint32_t> output;
        bool caught = false;
        try {
            decoder.generate_with_callback({1}, 6, 1.0f, 1.0f, 1,
                                           [&](uint32_t token) { output.push_back(token); });
        } catch (const SpeculativePrefillError& error) {
            caught = std::string(error.what()) == "injected speculative prefill failure";
        }
        expect(caught && output == std::vector<uint32_t>{2} && target->prefill_calls == 1 &&
                   draft->prefill_calls == 1 && target->decode_calls == 0 && draft->decode_calls == 0,
               "Speculative cleanup must preserve the original prefill exception and emitted prefix");
        expect(target->synchronize_calls == 2 && draft->synchronize_calls == 2,
               "Speculative reset and failure cleanup must complete both executors even if one fails");
        target->fail_completion = draft->fail_completion = false;
        draft->fail_prefill = false;
        if (failed_executor) {
            expect_retained_speculative_cache(*target, 1);
            expect_retained_speculative_cache(*draft, 1);
            int callbacks = 0;
            expect_invalid_speculation(decoder, callbacks);
            expect(callbacks == 0 && target->prefill_calls == 1 && draft->prefill_calls == 1 &&
                       target->synchronize_calls == 2 && draft->synchronize_calls == 2,
                   "Invalid speculative decoder must reject before touching either executor");
        } else {
            expect(decoder.context_size() == 0 && draft->last_cache->size() == 0,
                   "Successful speculative cleanup must clear both logical caches");
            output.clear();
            decoder.generate_with_callback({1}, 6, 1.0f, 1.0f, 1,
                                           [&](uint32_t token) { output.push_back(token); });
            expect(output == std::vector<uint32_t>({2, 2, 2, 2, 2}) && decoder.context_size() == 5 &&
                       draft->last_cache->size() == 5,
                   "Successful speculative failure cleanup must allow the same decoder to retry");
        }
    }

    for (int failed_executor : {1, 2}) {
        auto target = std::make_shared<SpeculativeFixture>("target");
        auto draft = std::make_shared<SpeculativeFixture>("draft");
        Speculation decoder(target, draft, 3, 0, 8);
        decoder.generate_with_callback({1}, 6, 1.0f, 1.0f, 1, [](uint32_t) {});
        const int target_completions = target->synchronize_calls;
        const int draft_completions = draft->synchronize_calls;
        target->fail_completion = failed_executor == 1;
        draft->fail_completion = failed_executor == 2;
        bool caught = false;
        try { decoder.reset(); }
        catch (const SpeculativeCompletionError& error) {
            caught = std::string(error.what()) ==
                     (failed_executor == 1 ? "target completion failed" : "draft completion failed");
        }
        expect(caught && target->synchronize_calls == target_completions + 1 &&
                   draft->synchronize_calls == draft_completions + 1,
               "Speculative reset must preserve its completion error while attempting both executors");
        expect_retained_speculative_cache(*target, 5);
        expect_retained_speculative_cache(*draft, 5);
        target->fail_completion = draft->fail_completion = false;
        int callbacks = 0;
        expect_invalid_speculation(decoder, callbacks);
        expect(callbacks == 0, "Failed speculative reset must prevent callback execution");
    }
}

void speculative_callback_device_test(int devices) {
    if (devices < 2) return;
    int original = -1;
    check(cudaGetDevice(&original));
    const int selected = (original + 1) % devices;
    auto target = std::make_shared<SpeculativeFixture>("target");
    auto draft = std::make_shared<SpeculativeFixture>("draft");
    Speculation decoder(target, draft, 3, 0, 8);
    std::vector<uint32_t> output;
    try {
        decoder.generate_with_callback({1}, 6, 1.0f, 1.0f, 1, [&](uint32_t token) {
            output.push_back(token);
            check(cudaSetDevice(selected));
        });
        int after = -1;
        check(cudaGetDevice(&after));
        expect(output == std::vector<uint32_t>({2, 2, 2, 2, 2}) && after == selected,
               "Speculative callbacks must preserve their CUDA device selection across later native phases");
    } catch (...) {
        cudaSetDevice(original);
        throw;
    }
    check(cudaSetDevice(original));
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
        warmup_failure_test();
        ordinary_direct_failure_test();
        ordinary_completion_invalidation_test();
        speculative_completion_test();
        speculative_callback_device_test(devices);
        generation_test();
        std::cout << "inference_generation_test passed\n";
    } catch (const std::exception& error) {
        std::cerr << "inference_generation_test failed: " << error.what() << '\n';
        return 1;
    }
}
