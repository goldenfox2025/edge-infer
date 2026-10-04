#include "operators/cuda/execution.hpp"
#include "../../runtime/tests/allocation_probe.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

void check(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("AWQ test cuBLAS setup failed");
}

struct Context {
    cudaStream_t stream = nullptr;
    cublasHandle_t handle = nullptr;
    op::cuda::ExecutionContext execution;
    Context() {
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        check(cublasCreate(&handle));
        execution = op::cuda::prepare_execution_context(handle, stream);
        op::cuda::bind_execution_context(execution);
    }
    ~Context() {
        cudaStreamSynchronize(stream);
        cublasDestroy(handle);
        cudaStreamDestroy(stream);
    }
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
    TensorView<T, 1> vector() { return TensorView<T, 1>::contiguous(data, {count}); }
    TensorView<T, 2> matrix(std::size_t rows, std::size_t columns) {
        expect(rows * columns == count, "AWQ test matrix extent differs");
        return TensorView<T, 2>::contiguous(data, {rows, columns});
    }
    void upload(const std::vector<T>& values, cudaStream_t stream) {
        expect(values.size() == count, "AWQ test upload extent differs");
        check(cudaMemcpyAsync(data, values.data(), count * sizeof(T), cudaMemcpyHostToDevice, stream));
    }
    std::vector<T> download(cudaStream_t stream) const {
        check(cudaStreamSynchronize(stream));
        std::vector<T> values(count);
        check(cudaMemcpy(values.data(), data, count * sizeof(T), cudaMemcpyDeviceToHost));
        return values;
    }
};

template <typename T>
T encode(float value) { return static_cast<T>(value); }

template <>
__nv_bfloat16 encode(float value) { return __float2bfloat16(value); }

template <typename T>
float decode(T value) { return static_cast<float>(value); }

template <>
float decode(__nv_bfloat16 value) { return __bfloat162float(value); }

struct Shape {
    std::size_t input, output, group;
};

// The final two cases reach BF16 MMA GEMM with a partial output tile and
// three live rows. Other cases cover GEMV and scalar fallback dispatch.
constexpr std::array<Shape, 6> kShapes{{
    {24, 16, 4}, {24, 16, 8}, {24, 16, 12}, {256, 16, 128},
    {32, 17, 4}, {1024, 17, 128}}};

template <typename T>
struct Fixture {
    std::vector<T> input, scales, bias, dense;
    std::vector<int32_t> packed_weights, packed_zeros;
    std::vector<float> reference;
    std::size_t groups, padded_groups;

    Fixture(Shape shape, std::size_t rows)
        : input(rows * shape.input), bias(shape.output), dense(shape.input * shape.output),
          packed_weights(shape.output * ((shape.input + 7) / 8), 0),
          packed_zeros(shape.output * ((shape.input / shape.group + 7) / 8), 0),
          reference(rows * shape.output), groups(shape.input / shape.group),
          padded_groups(groups + 3) {
        scales.assign(shape.output * padded_groups, encode<T>(-73.0f));
        const auto weight_stride = (shape.input + 7) / 8;
        const auto zero_stride = (groups + 7) / 8;
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t feature = 0; feature < shape.input; ++feature) {
                const auto residue = static_cast<int>((feature * 3 + feature / 8 + row * 5) % 17) - 8;
                input[row * shape.input + feature] = encode<T>(residue * 0.0625f + (row + 1) * 0.03125f);
            }
        }
        for (std::size_t output = 0; output < shape.output; ++output) {
            const float sign = output % 2 ? -1.0f : 1.0f;
            bias[output] = encode<T>(sign * (output % 5 + 1) * 0.015625f);
            for (std::size_t group = 0; group < groups; ++group) {
                // Binary-exact BF16 scales and weights isolate packing and
                // group-boundary errors from dequantization-rounding policy.
                scales[output * padded_groups + group] = encode<T>(
                    (1 + output % 2 + group % 4) * 0.03125f);
                const uint32_t zero = static_cast<uint32_t>(
                    (output * 2 + group * 3 + 3) % 13 + 1);
                auto& packed = packed_zeros[output * zero_stride + group / 8];
                packed = static_cast<int32_t>(static_cast<uint32_t>(packed) |
                    (zero << ((group % 8) * 4)));
            }
            for (std::size_t feature = 0; feature < shape.input; ++feature) {
                const uint32_t quantized = static_cast<uint32_t>(
                    (feature * 5 + output * 3 + shape.input / 8) % 16);
                auto& packed = packed_weights[output * weight_stride + feature / 8];
                packed = static_cast<int32_t>(static_cast<uint32_t>(packed) |
                    (quantized << ((feature % 8) * 4)));
                const auto group = feature / shape.group;
                // Compute dense weights from the unpacked fixture definition,
                // independently of the packed words passed to CUDA.
                const uint32_t zero = static_cast<uint32_t>(
                    (output * 2 + group * 3 + 3) % 13 + 1);
                const float scale = decode(scales[output * padded_groups + group]);
                const float weight = (static_cast<float>(quantized) - zero) * scale;
                dense[output * shape.input + feature] = encode<T>(weight);
                expect(decode(dense[output * shape.input + feature]) == weight,
                       "AWQ fixture dequantized weight must be exactly representable");
            }
        }
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t output = 0; output < shape.output; ++output) {
                double result = decode(bias[output]);
                for (std::size_t feature = 0; feature < shape.input; ++feature) {
                    result += static_cast<double>(decode(input[row * shape.input + feature])) *
                        decode(dense[output * shape.input + feature]);
                }
                reference[row * shape.output + output] = decode(encode<T>(static_cast<float>(result)));
            }
        }
    }
};

template <typename T>
float compare(const std::vector<T>& actual, const std::vector<float>& reference,
              const char* operation) {
    expect(actual.size() == reference.size(), "AWQ result extent differs");
    float maximum = 0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const float value = decode(actual[index]);
        const float error = std::fabs(value - reference[index]);
        maximum = std::max(maximum, error);
        const float tolerance = std::is_same_v<T, float>
            ? 2.0e-5f + 2.0e-5f * std::fabs(reference[index])
            : 0.015625f + 0.01f * std::fabs(reference[index]);
        if (!std::isfinite(value) || error > tolerance) {
            throw std::runtime_error(std::string(operation) + " differs at " + std::to_string(index) +
                ": actual=" + std::to_string(value) + ", expected=" + std::to_string(reference[index]));
        }
    }
    return maximum;
}

template <typename T>
void run(Context& context, Shape shape, std::size_t rows) {
    Fixture<T> fixture(shape, rows);
    Buffer<T> input(fixture.input.size()), scales(fixture.scales.size()), bias(fixture.bias.size()),
        dense(fixture.dense.size()), output(fixture.reference.size()), dense_output(fixture.reference.size());
    Buffer<int32_t> packed(fixture.packed_weights.size()), zeros(fixture.packed_zeros.size());
    input.upload(fixture.input, context.stream);
    scales.upload(fixture.scales, context.stream);
    bias.upload(fixture.bias, context.stream);
    dense.upload(fixture.dense, context.stream);
    packed.upload(fixture.packed_weights, context.stream);
    zeros.upload(fixture.packed_zeros, context.stream);
    check(cudaStreamSynchronize(context.stream));
    const auto awq = op::cuda::prepare_awq<T>(
        packed.matrix(shape.output, (shape.input + 7) / 8).as_const(),
        scales.matrix(shape.output, fixture.padded_groups).as_const(),
        zeros.matrix(shape.output, (fixture.groups + 7) / 8).as_const(),
        shape.group, shape.input, bias.vector().as_const());
    const auto weight = op::cuda::prepare_dense<T>(
        TensorView<const T, 2>::strided(dense.data, {shape.input, shape.output}, {1, shape.input}),
        bias.vector().as_const());
    const auto inputs = input.matrix(rows, shape.input).as_const();
    const auto outputs = output.matrix(rows, shape.output);
    op::cuda::awq_linear(context.execution, inputs, awq, outputs);
    op::cuda::linear(context.execution, inputs, weight, dense_output.matrix(rows, shape.output));
    check(cudaStreamSynchronize(context.stream));
    test_alloc::Scope probe;
    op::cuda::awq_linear(context.execution, inputs, awq, outputs);
    check(cudaStreamSynchronize(context.stream));
    const auto count = probe.finish();
    expect(count.host_allocations == 0 && count.host_frees == 0,
           "Prepared AWQ allocated/freed C++ storage");
#ifdef EDGE_INFER_TEST_CUDA_WRAPPING
    expect(count.device_allocations == 0 && count.device_frees == 0,
           "Prepared AWQ allocated/freed native CUDA storage");
#endif
    const auto actual = output.download(context.stream);
    const auto dense_actual = dense_output.download(context.stream);
    const float maximum_awq = compare(actual, fixture.reference, "AWQ against independent CPU dense reference");
    const float maximum_dense = compare(dense_actual, fixture.reference, "Prepared dense against CPU reference");
    std::vector<float> dense_values(dense_actual.size());
    for (std::size_t index = 0; index < dense_actual.size(); ++index) dense_values[index] = decode(dense_actual[index]);
    const float maximum_pair = compare(actual, dense_values, "AWQ against prepared dense");
    std::cout << "AWQ " << (std::is_same_v<T, float> ? "FP32" : "BF16")
              << " K=" << shape.input << ", N=" << shape.output << ", group=" << shape.group
              << ", rows=" << rows << ": max_abs_CPU=" << maximum_awq
              << ", max_abs_dense_CPU=" << maximum_dense << ", max_abs_AWQ_dense=" << maximum_pair
              << ", C++ allocations=" << count.host_allocations;
#ifdef EDGE_INFER_TEST_CUDA_WRAPPING
    std::cout << ", native CUDA allocations=" << count.device_allocations;
#else
    std::cout << ", CUDA allocation tracing unavailable in this build";
#endif
    std::cout << '\n';
}

}  // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) {
        std::cout << "operators_cuda_awq_test skipped: no CUDA device\n";
        return 77;
    }
    try {
        Context context;
        for (const auto shape : kShapes) {
            for (const std::size_t rows : {std::size_t{1}, std::size_t{3}}) {
                run<float>(context, shape, rows);
                run<__nv_bfloat16>(context, shape, rows);
            }
        }
        std::cout << "operators_cuda_awq_test passed\n";
    } catch (const std::exception& error) {
        std::cerr << "operators_cuda_awq_test failed: " << error.what() << '\n';
        return 1;
    }
}
