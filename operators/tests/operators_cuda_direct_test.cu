#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "operators/core/cpu_reference.hpp"
#include "operators/cuda/direct.hpp"

namespace {

void check_cuda(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

class Stream {
public:
    Stream() { check_cuda(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(value); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    cudaStream_t value = nullptr;
};

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t size) : size_(size) {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_), size * sizeof(T)));
    }
    ~DeviceBuffer() { cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void write(const std::vector<T>& values, cudaStream_t stream) {
        if (values.size() != size_) {
            throw std::runtime_error("test buffer extent mismatch");
        }
        check_cuda(cudaMemcpyAsync(data_, values.data(), size_ * sizeof(T),
                                   cudaMemcpyHostToDevice, stream));
    }
    std::vector<T> read(cudaStream_t stream) const {
        check_cuda(cudaStreamSynchronize(stream));
        std::vector<T> result(size_);
        check_cuda(cudaMemcpy(result.data(), data_, size_ * sizeof(T), cudaMemcpyDeviceToHost));
        return result;
    }
    op::ArrayView<T> view() { return {data_, size_}; }
    op::ArrayView<const T> const_view() const { return {data_, size_}; }

private:
    T* data_ = nullptr;
    size_t size_ = 0;
};

template <typename T>
void expect_near(const std::vector<T>& actual, const std::vector<T>& expected,
                 const char* operation) {
    const float tolerance = std::is_same_v<T, float> ? 5e-5f : 0.02f;
    for (size_t i = 0; i < actual.size(); ++i) {
        const float a = static_cast<float>(actual[i]);
        const float e = static_cast<float>(expected[i]);
        if (!std::isfinite(a) || std::fabs(a - e) > tolerance * (1.0f + std::fabs(e))) {
            throw std::runtime_error(std::string(operation) +
                                     (std::is_same_v<T, float> ? " float" : " BF16") +
                                     ", size=" + std::to_string(actual.size()) +
                                     " mismatch at element " + std::to_string(i) +
                                     ": actual=" + std::to_string(a) + ", expected=" + std::to_string(e));
        }
    }
}

template <typename Function>
void expect_throw(Function&& function, const char* message) {
    try {
        function();
    } catch (const std::runtime_error&) {
        return;
    }
    throw std::runtime_error(message);
}

template <typename T>
void test_elementwise(cudaStream_t stream, size_t size) {
    std::vector<T> a(size), b(size), expected(size);
    std::vector<float> reference_a(size), reference_b(size), reference_output(size);
    for (size_t i = 0; i < size; ++i) {
        a[i] = static_cast<T>((static_cast<int>(i % 7) - 3) * 0.25f);
        b[i] = static_cast<T>((static_cast<int>(i % 5) - 2) * 0.5f);
        reference_a[i] = static_cast<float>(a[i]);
        reference_b[i] = static_cast<float>(b[i]);
    }
    // CUDA toolkit host BF16 arithmetic overloads may be device-only. Use the
    // decoded inputs with float CPU references, then round the expected output.
    auto round_expected = [&] {
        for (size_t i = 0; i < size; ++i) {
            expected[i] = static_cast<T>(reference_output[i]);
        }
    };
    DeviceBuffer<T> da(size), db(size), output(size);
    da.write(a, stream);
    db.write(b, stream);

    op::cpu::add<float>({reference_a.data(), size}, {reference_b.data(), size},
                        {reference_output.data(), size});
    round_expected();
    op::cuda::add<T>(da.const_view(), db.const_view(), output.view(), stream);
    expect_near(output.read(stream), expected, "add");
    op::cuda::add<T>(da.const_view(), db.const_view(), da.view(), stream);
    expect_near(da.read(stream), expected, "in-place add");

    da.write(a, stream);
    op::cpu::multiply<float>({reference_a.data(), size}, {reference_b.data(), size},
                             {reference_output.data(), size});
    round_expected();
    op::cuda::multiply<T>(da.const_view(), db.const_view(), output.view(), stream);
    expect_near(output.read(stream), expected, "multiply tail");
    op::cuda::multiply<T>(da.const_view(), db.const_view(), db.view(), stream);
    expect_near(db.read(stream), expected, "in-place multiply second input");

    op::cpu::silu<float>({reference_a.data(), size}, {reference_output.data(), size});
    round_expected();
    op::cuda::silu<T>(da.const_view(), output.view(), stream);
    expect_near(output.read(stream), expected, "SiLU");
    op::cuda::silu<T>(da.const_view(), da.view(), stream);
    expect_near(da.read(stream), expected, "in-place SiLU");

    auto short_output = output.view();
    --short_output.size;
    expect_throw([&] { op::cuda::add<T>(da.const_view(), db.const_view(), short_output, stream); },
                 "add accepted mismatched extents");
    expect_throw([&] { op::cuda::multiply<T>(da.const_view(), db.const_view(), short_output, stream); },
                 "multiply accepted mismatched extents");
    expect_throw([&] { op::cuda::silu<T>(da.const_view(), short_output, stream); },
                 "SiLU accepted mismatched extents");
}

template <typename T>
void test_rms_norm(cudaStream_t stream, size_t feature_dim, size_t rows) {
    const size_t size = feature_dim * rows;
    std::vector<T> values(size), weights(feature_dim), expected(size);
    std::vector<float> reference_values(size), reference_weights(feature_dim), reference_output(size);
    for (size_t i = 0; i < size; ++i) {
        const float value = i < feature_dim ? 0.0f : (static_cast<int>(i % 11) - 5) * 0.25f;
        values[i] = static_cast<T>(value);
        reference_values[i] = static_cast<float>(values[i]);
    }
    for (size_t i = 0; i < feature_dim; ++i) {
        weights[i] = static_cast<T>(0.5f + (i % 3) * 0.25f);
        reference_weights[i] = static_cast<float>(weights[i]);
    }
    DeviceBuffer<T> input(size), output(size), weight(feature_dim);
    input.write(values, stream);
    weight.write(weights, stream);
    op::cpu::rms_norm<float>({reference_values.data(), size},
                            {reference_weights.data(), feature_dim},
                            {reference_output.data(), size}, rows, feature_dim, 1e-6f);
    for (size_t i = 0; i < size; ++i) {
        expected[i] = static_cast<T>(reference_output[i]);
    }
    op::cuda::rms_norm<T>(input.const_view(), weight.const_view(), output.view(),
                          rows, feature_dim, 1e-6f, stream);
    expect_near(output.read(stream), expected, "RMSNorm rows");
    op::cuda::rms_norm<T>(input.const_view(), weight.const_view(), input.view(),
                          rows, feature_dim, 1e-6f, stream);
    expect_near(input.read(stream), expected, "in-place RMSNorm");
    expect_throw([&] {
        op::cuda::rms_norm<T>(input.const_view(), weight.const_view(), output.view(),
                              rows + 1, feature_dim, 1e-6f, stream);
    }, "RMSNorm accepted a mismatched row count");
}

template <typename T>
void test_empty_and_rms_guard(cudaStream_t stream) {
    op::cuda::add<T>({}, {}, {}, stream);
    op::cuda::multiply<T>({}, {}, {}, stream);
    op::cuda::silu<T>({}, {}, stream);
    op::cuda::rms_norm<T>({}, {}, {}, 0, 0, 1e-6f, stream);
    check_cuda(cudaGetLastError());

    DeviceBuffer<T> input(10241), output(10241), weights(10241);
    expect_throw([&] {
        op::cuda::rms_norm<T>(input.const_view(), weights.const_view(), output.view(),
                              1, 10241, 1e-6f, stream);
    }, "RMSNorm accepted a row exceeding its register cache");
}

template <typename T>
void test_type(cudaStream_t stream) {
    test_empty_and_rms_guard<T>(stream);
    test_elementwise<T>(stream, 17);
    test_elementwise<T>(stream, 18);
    test_elementwise<T>(stream, 4099);
    test_elementwise<T>(stream, 4098);
    test_rms_norm<T>(stream, 7, 3);
    test_rms_norm<T>(stream, 10240, 2);
}

}  // namespace

int main() {
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess || device_count == 0) {
        std::cout << "Skipping CUDA direct tests: " << cudaGetErrorString(status) << "\n";
        return 77;
    }
    try {
        Stream stream;
        test_type<float>(stream.value);
        test_type<__nv_bfloat16>(stream.value);
        std::cout << "operators_cuda_direct_test passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
