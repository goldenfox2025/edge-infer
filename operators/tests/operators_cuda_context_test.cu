#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include "operators/cuda/cuda_resource_manager.cuh"
#include "operators/cuda/matmul/cublas_matmul_cuda.cuh"
#include "operators/unified_operators.hpp"

namespace {

void check_cuda(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

void check_cublas(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS status " + std::to_string(static_cast<int>(status)));
    }
}

class Stream {
public:
    Stream() { check_cuda(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() {
        cudaStreamSynchronize(value);
        cudaStreamDestroy(value);
    }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    cudaStream_t value = nullptr;
};

class Handle {
public:
    Handle() { check_cublas(cublasCreate(&value)); }
    ~Handle() { cublasDestroy(value); }
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
    cublasHandle_t value = nullptr;
};

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(const std::vector<T>& values) : DeviceBuffer(values.size()) {
        check_cuda(cudaMemcpy(data_, values.data(), size_ * sizeof(T), cudaMemcpyHostToDevice));
    }
    explicit DeviceBuffer(size_t size) : size_(size) {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_), size_ * sizeof(T)));
    }
    ~DeviceBuffer() { cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    Tensor<T> tensor(const std::vector<size_t>& shape) {
        return Tensor<T>::from_external_buffer(data_, shape, Device::CUDA);
    }
    std::vector<T> read() const {
        std::vector<T> values(size_);
        check_cuda(cudaMemcpy(values.data(), data_, size_ * sizeof(T), cudaMemcpyDeviceToHost));
        return values;
    }

private:
    T* data_ = nullptr;
    size_t size_;
};

void expect_stream(cublasHandle_t handle, cudaStream_t expected) {
    cudaStream_t actual = nullptr;
    check_cublas(cublasGetStream(handle, &actual));
    if (actual != expected) {
        throw std::runtime_error("Matmul used a different cuBLAS stream");
    }
}

template <typename Function>
void expect_invalid_argument(Function&& function) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error("Invalid borrowed-handle construction succeeded");
}

template <typename T>
std::vector<T> reference(const std::vector<T>& input, const std::vector<T>& physical_weight,
                         const std::vector<T>* bias, size_t rows, size_t features,
                         size_t outputs) {
    std::vector<T> result(rows * outputs);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t out = 0; out < outputs; ++out) {
            float sum = 0.0f;
            for (size_t feature = 0; feature < features; ++feature) {
                sum += static_cast<float>(input[row * features + feature]) *
                       static_cast<float>(physical_weight[out * features + feature]);
            }
            // Preserve the existing GEMM output rounding followed by a separate
            // elementwise bias addition, including its second BF16 rounding.
            T value = static_cast<T>(sum);
            if (bias) {
                value = static_cast<T>(static_cast<float>(value) + static_cast<float>((*bias)[out]));
            }
            result[row * outputs + out] = value;
        }
    }
    return result;
}

template <typename T>
void expect_values(const std::vector<T>& actual, const std::vector<T>& expected) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error("Matmul output size mismatch");
    }
    // Dyadic test inputs are exact in TF32; BF16 checks the rounded result exactly.
    const float tolerance = std::is_same_v<T, float> ? 1e-6f : 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
        const float value = static_cast<float>(actual[i]);
        const float target = static_cast<float>(expected[i]);
        if (!std::isfinite(value) || std::fabs(value - target) > tolerance) {
            throw std::runtime_error(std::string(std::is_same_v<T, float> ? "float" : "BF16") +
                                     " matmul mismatch at " + std::to_string(i) +
                                     ": actual=" + std::to_string(value) +
                                     ", expected=" + std::to_string(target));
        }
    }
}

template <typename T>
void run(cublasHandle_t legacy_handle, cudaStream_t legacy_stream) {
    constexpr size_t rows = 3, features = 7, outputs = 5;
    std::vector<T> first_input(rows * features), second_input(rows * features);
    std::vector<T> physical_weight(outputs * features), host_bias(outputs);
    for (size_t i = 0; i < first_input.size(); ++i) {
        first_input[i] = static_cast<T>((static_cast<int>(i % 11) - 5) / 16.0f);
        second_input[i] = static_cast<T>((static_cast<int>((i * 3) % 13) - 6) / 8.0f);
    }
    for (size_t i = 0; i < physical_weight.size(); ++i) {
        physical_weight[i] = static_cast<T>((static_cast<int>((i * 5) % 17) - 8) / 16.0f);
    }
    for (size_t i = 0; i < host_bias.size(); ++i) {
        host_bias[i] = static_cast<T>((static_cast<int>(i) - 2) / 32.0f);
    }

    Stream first_stream, second_stream;
    Handle first_handle, second_handle;
    op::UnifiedOperators<T> first(Device::CUDA, first_handle.value);
    op::UnifiedOperators<T> second(Device::CUDA, second_handle.value);
    auto first_prepared = first.get_operator_base("matmul");
    auto second_prepared = second.get_operator_base("matmul");
    if (!first_prepared || !second_prepared || first_prepared == second_prepared ||
        first_prepared->type() != op::OperatorType::MATMUL ||
        first_prepared->platform() != op::OperatorPlatform::CUDA) {
        throw std::runtime_error("Borrowed facades did not expose independent matmul operators");
    }
    expect_stream(legacy_handle, legacy_stream);

    expect_invalid_argument([&] { op::UnifiedOperators<T> invalid(Device::CUDA, nullptr); });
    expect_invalid_argument([&] { op::UnifiedOperators<T> invalid(Device::CPU, first_handle.value); });

    DeviceBuffer<T> first_data(first_input), second_data(second_input);
    DeviceBuffer<T> weights(physical_weight), biases(host_bias);
    DeviceBuffer<T> first_result(rows * outputs), second_result(rows * outputs);
    auto first_tensor = first_data.tensor({rows, features});
    auto second_tensor = second_data.tensor({rows, features});
    // The loader uses a logical [K,N] view of physical row-major [N,K] weights.
    auto weight_tensor = weights.tensor({features, outputs});
    auto bias_tensor = biases.tensor({outputs});
    auto first_output = first_result.tensor({rows, outputs});
    auto second_output = second_result.tensor({rows, outputs});
    const op::WeightTensor<T> weight(&weight_tensor);

    int device = 0;
    check_cuda(cudaGetDevice(&device));
    auto submit = [&](op::UnifiedOperators<T>& facade,
                      const std::shared_ptr<op::OperatorBase>& prepared,
                      Tensor<T>& input, Tensor<T>& output,
                      const Tensor<T>* bias, cudaStream_t stream) {
        check_cuda(cudaSetDevice(device));
        std::vector<void*> tensors{&output, &input};
        std::vector<void*> statics{const_cast<op::WeightTensor<T>*>(&weight)};
        if (bias) {
            statics.push_back(const_cast<Tensor<T>*>(bias));
        }
        for (int iteration = 0; iteration < 16; ++iteration) {
            if (iteration % 2 == 0) {
                facade.matmul(&output, &input, weight, bias, stream);
            } else {
                prepared->execute_packed(tensors, statics, {}, stream);
            }
        }
    };

    // Enqueue from two host threads onto independent nonblocking streams;
    // both callers share immutable weights and use separate input/output storage.
    for (bool first_has_bias : {true, false}) {
        std::exception_ptr failures[2];
        const Tensor<T>* first_bias = first_has_bias ? &bias_tensor : nullptr;
        const Tensor<T>* second_bias = first_has_bias ? nullptr : &bias_tensor;
        std::thread first_worker([&] {
            try {
                submit(first, first_prepared, first_tensor, first_output, first_bias, first_stream.value);
            } catch (...) {
                failures[0] = std::current_exception();
            }
        });
        std::thread second_worker([&] {
            try {
                submit(second, second_prepared, second_tensor, second_output, second_bias, second_stream.value);
            } catch (...) {
                failures[1] = std::current_exception();
            }
        });
        first_worker.join();
        second_worker.join();
        // Complete both streams before an error can unwind caller-owned storage.
        check_cuda(cudaStreamSynchronize(first_stream.value));
        check_cuda(cudaStreamSynchronize(second_stream.value));
        for (const auto& failure : failures) {
            if (failure) std::rethrow_exception(failure);
        }
        expect_stream(first_handle.value, first_stream.value);
        expect_stream(second_handle.value, second_stream.value);
        expect_stream(legacy_handle, legacy_stream);
        expect_values(first_result.read(), reference(first_input, physical_weight,
                      first_has_bias ? &host_bias : nullptr, rows, features, outputs));
        expect_values(second_result.read(), reference(second_input, physical_weight,
                      first_has_bias ? nullptr : &host_bias, rows, features, outputs));
    }

    // A null stream must reset a handle after use on a nonblocking stream.
    first.matmul(&first_output, &first_tensor, weight, &bias_tensor, nullptr);
    second_prepared->execute_packed({&second_output, &second_tensor},
                                   {const_cast<op::WeightTensor<T>*>(&weight)}, {}, nullptr);
    expect_stream(first_handle.value, nullptr);
    expect_stream(second_handle.value, nullptr);
    check_cuda(cudaStreamSynchronize(nullptr));
    expect_values(first_result.read(), reference(first_input, physical_weight, &host_bias,
                  rows, features, outputs));
    expect_values(second_result.read(), reference(second_input, physical_weight,
                  static_cast<const std::vector<T>*>(nullptr), rows, features, outputs));
    expect_stream(legacy_handle, legacy_stream);

    // Prepared operators may outlive the facade, but still borrow the caller's handle.
    std::shared_ptr<op::OperatorBase> retained;
    {
        op::UnifiedOperators<T> temporary(Device::CUDA, first_handle.value);
        retained = temporary.get_operator_base("matmul");
    }
    retained->execute_packed({&first_output, &first_tensor},
                            {const_cast<op::WeightTensor<T>*>(&weight), &bias_tensor}, {}, first_stream.value);
    check_cuda(cudaStreamSynchronize(first_stream.value));
    expect_values(first_result.read(), reference(first_input, physical_weight, &host_bias,
                  rows, features, outputs));
    expect_stream(legacy_handle, legacy_stream);

    // The compatibility path also resets its previous stream when passed nullptr.
    op::CublasMatmulCUDAOperator<T> legacy;
    legacy(&first_output, &first_tensor, weight, &bias_tensor, nullptr);
    expect_stream(legacy_handle, nullptr);
    check_cuda(cudaStreamSynchronize(nullptr));
    expect_values(first_result.read(), reference(first_input, physical_weight, &host_bias,
                  rows, features, outputs));
    op::CUDAResourceManager::instance().setCublasStream(legacy_stream);
}

}  // namespace

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::cout << "SKIP: CUDA device unavailable\n";
        return 77;
    }
    try {
        Stream legacy_stream;
        // Exercise first-use setCublasStream before any getCublasHandle call.
        auto& resources = op::CUDAResourceManager::instance();
        resources.setCublasStream(legacy_stream.value);
        const auto legacy_handle = resources.getCublasHandle();
        run<float>(legacy_handle, legacy_stream.value);
        run<__nv_bfloat16>(legacy_handle, legacy_stream.value);
        resources.setCublasStream(nullptr);
        std::cout << "Borrowed CUDA matmul contexts passed for float and BF16\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
