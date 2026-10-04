#include "operators/cuda/execution.hpp"
#include "../../runtime/tests/allocation_probe.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

struct Context {
    cudaStream_t stream = nullptr;
    op::cuda::ExecutionContext execution;
    Context() {
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        execution = op::cuda::prepare_execution_context(nullptr, stream);
    }
    ~Context() { cudaStreamSynchronize(stream); cudaStreamDestroy(stream); }
};

template <typename T>
struct Buffer {
    T* data = nullptr;
    std::size_t count;
    explicit Buffer(std::size_t size) : count(size) {
        check(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
    }
    ~Buffer() { cudaFree(data); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    TensorView<T, 1> view() { return TensorView<T, 1>::contiguous(data, {count}); }
    TensorView<const T, 2> rows(std::size_t width) const {
        return TensorView<const T, 2>::contiguous(data, {count / width, width});
    }
    void upload(const std::vector<T>& values, cudaStream_t stream) {
        expect(values.size() == count, "sampling upload extent differs");
        check(cudaMemcpyAsync(data, values.data(), count * sizeof(T), cudaMemcpyHostToDevice, stream));
        check(cudaStreamSynchronize(stream));
    }
    std::vector<T> download(cudaStream_t stream) const {
        std::vector<T> values(count);
        check(cudaMemcpyAsync(values.data(), data, count * sizeof(T), cudaMemcpyDeviceToHost, stream));
        check(cudaStreamSynchronize(stream));
        return values;
    }
};

__global__ void initialize_random_state(curandState* state, unsigned long long seed) {
    if (!threadIdx.x && !blockIdx.x) curand_init(seed, 0, 0, state);
}

constexpr std::size_t kVocabulary = 4;
const std::array<float, kVocabulary> kLogits{
    std::log(0.4f), std::log(0.3f), std::log(0.2f), std::log(0.1f)};

std::array<double, kVocabulary> reference_distribution(float temperature,
                                                       float top_p,
                                                       std::size_t top_k) {
    std::array<std::size_t, kVocabulary> order{0, 1, 2, 3};
    std::sort(order.begin(), order.end(), [&](auto left, auto right) {
        return kLogits[left] > kLogits[right];
    });
    std::array<double, kVocabulary> probability{};
    if (temperature <= 0 || top_k == 1) {
        probability[order[0]] = 1;
        return probability;
    }
    const auto candidates = std::min(top_k, kVocabulary);
    double total = 0;
    for (std::size_t index = 0; index < candidates; ++index) {
        probability[order[index]] = std::exp(
            (static_cast<double>(kLogits[order[index]]) - kLogits[order[0]]) / temperature);
        total += probability[order[index]];
    }
    std::size_t retained = 0;
    double cumulative = 0;
    do {
        cumulative += probability[order[retained++]];
    } while (retained < candidates && cumulative < top_p * total);
    for (std::size_t index = retained; index < kVocabulary; ++index) probability[order[index]] = 0;
    for (double& value : probability) value /= cumulative;
    return probability;
}

void expect_no_allocations(const test_alloc::Counts& count) {
    expect(count.host_allocations == 0 && count.host_frees == 0,
           "prepared direct sampling allocated/freed C++ heap storage");
#ifdef EDGE_INFER_TEST_CUDA_WRAPPING
    expect(count.device_allocations == 0 && count.device_frees == 0,
           "prepared direct sampling allocated/freed native device storage");
#endif
}

template <typename T>
T encode(float value) { return static_cast<T>(value); }
template <>
__nv_bfloat16 encode(float value) { return __float2bfloat16(value); }

template <typename T>
void greedy_test() {
    Context context;
    constexpr std::size_t rows = 3;
    const std::array<float, rows * kVocabulary> values{
        2, 1, 0, -1, -3, 5, 1, 0, 0, -1, 4, 2};
    std::vector<T> converted;
    for (float value : values) converted.push_back(encode<T>(value));
    Buffer<T> logits(values.size());
    Buffer<uint32_t> tokens(rows);
    Buffer<float> probabilities(rows);
    const auto plan = op::cuda::prepare_sampling(context.execution, kVocabulary);
    Buffer<unsigned char> scratch(plan.total_bytes);
    logits.upload(converted, context.stream);
    for (const auto parameters : std::array<std::pair<float, std::size_t>, 2>{{{1.0f, 1}, {0.0f, 4}}}) {
        op::cuda::sample(context.execution, logits.rows(kVocabulary), tokens.view(),
            probabilities.view(), scratch.view(), plan, parameters.first, 1.0f,
            parameters.second, nullptr);
        check(cudaStreamSynchronize(context.stream));
        const auto actual = tokens.download(context.stream);
        const auto probability = probabilities.download(context.stream);
        expect(actual == std::vector<uint32_t>{0, 1, 2}, "greedy sampling must match row argmax");
        expect(std::all_of(probability.begin(), probability.end(), [](float value) { return value == 1; }),
               "greedy selected probability must be one");
    }
}

void distribution_and_private_scratch_test() {
    constexpr std::size_t samples = 2048;
    Context a, b;
    Buffer<float> logits(samples * kVocabulary);
    std::vector<float> values(logits.count);
    for (std::size_t row = 0; row < samples; ++row) {
        std::copy(kLogits.begin(), kLogits.end(), values.begin() + row * kVocabulary);
    }
    logits.upload(values, a.stream);
    Buffer<uint32_t> tokens_a(samples), tokens_b(samples);
    Buffer<float> probabilities_a(samples), probabilities_b(samples);
    const auto plan_a = op::cuda::prepare_sampling(a.execution, kVocabulary);
    const auto plan_b = op::cuda::prepare_sampling(b.execution, kVocabulary);
    Buffer<unsigned char> scratch_a(plan_a.total_bytes), scratch_b(plan_b.total_bytes);
    Buffer<curandState> state_a(1), state_b(1);
    initialize_random_state<<<1, 1, 0, a.stream>>>(state_a.data, 1234567);
    initialize_random_state<<<1, 1, 0, b.stream>>>(state_b.data, 7654321);
    check(cudaGetLastError());
    check(cudaStreamSynchronize(a.stream));
    check(cudaStreamSynchronize(b.stream));
    struct Case { float temperature, top_p; std::size_t top_k; };
    const std::array<Case, 3> cases{{{1.0f, 1.0f, 2}, {1.0f, 0.65f, 4}, {0.75f, 0.95f, 3}}};
    // Prime CUB/runtime submission once before measuring steady sampling.
    op::cuda::sample(a.execution, logits.rows(kVocabulary), tokens_a.view(),
        probabilities_a.view(), scratch_a.view(), plan_a, 1.0f, 1.0f, 1, nullptr);
    op::cuda::sample(b.execution, logits.rows(kVocabulary), tokens_b.view(),
        probabilities_b.view(), scratch_b.view(), plan_b, 1.0f, 1.0f, 1, nullptr);
    check(cudaStreamSynchronize(a.stream));
    check(cudaStreamSynchronize(b.stream));
    for (const auto& parameters : cases) {
        const auto expected = reference_distribution(parameters.temperature,
                                                       parameters.top_p, parameters.top_k);
        test_alloc::Scope probe;
        op::cuda::sample(a.execution, logits.rows(kVocabulary), tokens_a.view(),
            probabilities_a.view(), scratch_a.view(), plan_a, parameters.temperature,
            parameters.top_p, parameters.top_k, state_a.data);
        check(cudaStreamSynchronize(a.stream));
        const auto allocation_count = probe.finish();
        expect_no_allocations(allocation_count);
        const auto actual = tokens_a.download(a.stream);
        const auto selected_probability = probabilities_a.download(a.stream);
        std::array<std::size_t, kVocabulary> counts{};
        for (std::size_t row = 0; row < samples; ++row) {
            expect(actual[row] < kVocabulary && expected[actual[row]] > 0,
                   "sampling produced a token outside top-k/top-p support");
            ++counts[actual[row]];
            expect(std::fabs(selected_probability[row] - expected[actual[row]]) < 2.0e-5,
                   "selected probability differs from CPU filtered softmax");
        }
        double maximum_frequency_error = 0;
        for (std::size_t token = 0; token < kVocabulary; ++token) {
            const double observed = static_cast<double>(counts[token]) / samples;
            maximum_frequency_error = std::max(maximum_frequency_error, std::fabs(observed - expected[token]));
        }
        expect(maximum_frequency_error < 0.05, "seeded empirical sampling distribution differs from CPU reference");
        // B uses a distinct plan/scratch/output/state. Its execution must not
        // change held A outputs or probabilities.
        op::cuda::sample(b.execution, logits.rows(kVocabulary), tokens_b.view(),
            probabilities_b.view(), scratch_b.view(), plan_b, 1.0f, 1.0f, 1, nullptr);
        check(cudaStreamSynchronize(b.stream));
        expect(tokens_a.download(a.stream) == actual &&
                   probabilities_a.download(a.stream) == selected_probability,
               "private sampler context B modified held A output");
        std::cout << "sampling top_k=" << parameters.top_k << ", top_p=" << parameters.top_p
                  << ": max_frequency_error=" << maximum_frequency_error
                  << ", C++ allocations=" << allocation_count.host_allocations;
#ifdef EDGE_INFER_TEST_CUDA_WRAPPING
        std::cout << ", native CUDA allocations=" << allocation_count.device_allocations;
#else
        std::cout << ", CUDA allocation tracing unavailable in this build";
#endif
        std::cout << '\n';
    }
    Buffer<float> full_probability(kVocabulary);
    for (std::size_t token = 0; token < kVocabulary; ++token) {
        op::cuda::token_probability(a.execution, logits.rows(kVocabulary), 0,
            static_cast<uint32_t>(token), full_probability.data + token);
    }
    const auto actual = full_probability.download(a.stream);
    for (std::size_t token = 0; token < kVocabulary; ++token) {
        const float expected = std::array<float, 4>{0.4f, 0.3f, 0.2f, 0.1f}[token];
        expect(std::fabs(actual[token] - expected) < 2.0e-6,
               "token_probability differs from full softmax");
    }
}

void invalid_sampling_preserves_output_test() {
    Context context;
    Buffer<float> logits(2048);
    logits.upload(std::vector<float>(2048, 0.25f), context.stream);
    Buffer<uint32_t> tokens(1);
    Buffer<float> probabilities(1);
    const auto plan = op::cuda::prepare_sampling(context.execution, 2048);
    Buffer<unsigned char> scratch(plan.total_bytes);
    constexpr uint32_t sentinel = 0xa5c39e71U;
    for (std::size_t top_k : {std::size_t{0}, std::size_t{1025}}) {
        tokens.upload({sentinel}, context.stream);
        probabilities.upload({-17.0f}, context.stream);
        bool rejected = false;
        try {
            op::cuda::sample(context.execution, logits.rows(2048), tokens.view(),
                probabilities.view(), scratch.view(), plan, 1.0f, 1.0f, top_k, nullptr);
        } catch (const std::invalid_argument&) { rejected = true; }
        expect(rejected, "direct sampling must reject invalid top_k before launches");
        check(cudaStreamSynchronize(context.stream));
        expect(tokens.download(context.stream)[0] == sentinel &&
                   probabilities.download(context.stream)[0] == -17.0f,
               "invalid direct sampling modified fixed outputs");
    }
}

}  // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) {
        std::cout << "operators_cuda_sampling_test skipped: no CUDA device\n";
        return 77;
    }
    try {
        greedy_test<float>();
        greedy_test<__nv_bfloat16>();
        distribution_and_private_scratch_test();
        invalid_sampling_preserves_output_test();
        std::cout << "operators_cuda_sampling_test passed\n";
    } catch (const std::exception& error) {
        std::cerr << "operators_cuda_sampling_test failed: " << error.what() << '\n';
        return 1;
    }
}
