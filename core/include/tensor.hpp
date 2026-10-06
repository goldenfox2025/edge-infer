#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <initializer_list>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

enum class Device { CPU, CUDA };

// Owning checkpoint/boundary storage. Copies and slices share ownership;
// fixed TensorView descriptors are used for execution. There is no global pool.
template <typename T>
class Tensor {
  template <typename>
  friend class Tensor;
  static void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
  }
  static size_t count(const std::vector<size_t>& shape) {
    size_t value = 1;
    for (auto n : shape) {
      if (n && value > std::numeric_limits<size_t>::max() / n)
        throw std::overflow_error("Tensor element count overflow");
      value *= n;
    }
    if (value > std::numeric_limits<size_t>::max() / sizeof(T))
      throw std::overflow_error("Tensor byte extent overflow");
    return value;
  }
  static std::vector<size_t> dense_strides(const std::vector<size_t>& shape) {
    std::vector<size_t> strides(shape.size());
    size_t value = 1;
    for (size_t i = shape.size(); i-- > 0;) {
      strides[i] = value;
      if (shape[i] && value > std::numeric_limits<size_t>::max() / shape[i])
        throw std::overflow_error("Tensor stride overflow");
      value *= shape[i];
    }
    return strides;
  }
  static std::shared_ptr<T> own(T* pointer, int device) {
    return std::shared_ptr<T>(pointer, [device](T* p) noexcept {
      int previous = device;
      cudaGetDevice(&previous);
      if (previous != device) cudaSetDevice(device);
      cudaFree(p);
      if (previous != device) cudaSetDevice(previous);
    });
  }
  static std::shared_ptr<T> adopt(T* pointer) {
    if (!pointer) return {};
    cudaPointerAttributes attributes{};
    check(cudaPointerGetAttributes(&attributes, pointer));
    if (attributes.type != cudaMemoryTypeDevice)
      throw std::invalid_argument("Tensor ownership requires CUDA device storage");
    return own(pointer, attributes.device);
  }
  static std::shared_ptr<T> allocate(size_t elements) {
    if (!elements) return {};
    int device = 0;
    check(cudaGetDevice(&device));
    T* pointer = nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&pointer), elements * sizeof(T)));
    // shared_ptr invokes its deleter if control-block allocation throws.
    return own(pointer, device);
  }
  std::vector<T> host_values() const {
    std::vector<T> result(length_);
    if (!length_) return result;
    if (is_contiguous()) {
      if (device_ == Device::CPU)
        std::copy_n(data_ptr(), length_, result.data());
      else
        check(cudaMemcpy(result.data(), data_ptr(), nbytes(), cudaMemcpyDeviceToHost));
      return result;
    }
    // Migration materializes a strided view in logical row-major order.
    size_t span = 1;
    for (size_t i = 0; i < shape_.size(); ++i) span += (shape_[i] - 1) * strides_[i];
    std::vector<T> physical;
    const T* source = data_ptr();
    if (device_ == Device::CUDA) {
      physical.resize(span);
      check(cudaMemcpy(physical.data(), source, span * sizeof(T), cudaMemcpyDeviceToHost));
      source = physical.data();
    }
    for (size_t flat = 0; flat < length_; ++flat) {
      size_t rest = flat, offset = 0;
      for (size_t i = shape_.size(); i-- > 0;) {
        offset += (rest % shape_[i]) * strides_[i];
        rest /= shape_[i];
      }
      result[flat] = source[offset];
    }
    return result;
  }

 public:
  Tensor() = default;
  explicit Tensor(const std::vector<size_t>& shape, Device device = Device::CPU)
      : shape_(shape), strides_(dense_strides(shape)), length_(count(shape)), device_(device) {
    if (device == Device::CPU)
      data_ = std::make_shared<std::vector<T>>(length_);
    else if (device == Device::CUDA)
      gpu_data_ = allocate(length_);
    else
      throw std::invalid_argument("Invalid Tensor device");
  }
  Tensor(std::initializer_list<size_t> shape, Device device = Device::CPU)
      : Tensor(std::vector<size_t>(shape), device) {}
  Tensor(std::shared_ptr<std::vector<T>> data, const std::vector<size_t>& shape)
      : data_(std::move(data)),
        shape_(shape),
        strides_(dense_strides(shape)),
        length_(count(shape)) {
    if (!data_ || data_->size() < length_)
      throw std::invalid_argument("Tensor data is shorter than shape");
  }
  Tensor(std::vector<T>&& data, const std::vector<size_t>& shape, Device device = Device::CPU)
      : Tensor(std::make_shared<std::vector<T>>(std::move(data)), shape) {
    if (device == Device::CUDA)
      cuda();
    else if (device != Device::CPU)
      throw std::invalid_argument("Invalid Tensor device");
  }
  // Transfer ownership of storage allocated by cudaMalloc. Borrowing is explicit.
  Tensor(T* pointer, const std::vector<size_t>& shape, Device device)
      : shape_(shape), strides_(dense_strides(shape)), length_(count(shape)), device_(device) {
    if (device != Device::CUDA || (!pointer && length_))
      throw std::invalid_argument("Tensor adoption requires CUDA storage");
    gpu_data_ = adopt(pointer);
  }
  static Tensor from_external_buffer(T* pointer, const std::vector<size_t>& shape, Device device) {
    Tensor result;
    result.shape_ = shape;
    result.strides_ = dense_strides(shape);
    result.length_ = count(shape);
    result.device_ = device;
    if (device != Device::CUDA || (!pointer && result.length_))
      throw std::invalid_argument("Tensor borrowing requires CUDA storage");
    result.gpu_data_ = std::shared_ptr<T>(pointer, [](T*) {});
    return result;
  }
  template <typename U>
  static Tensor combine_gpu_ptrs(const std::vector<U*>& pointers, Device device = Device::CUDA) {
    static_assert(sizeof(U) == sizeof(T), "Token pointer element size mismatch");
    if (pointers.empty() || device != Device::CUDA)
      throw std::invalid_argument("Combining pointers requires nonempty CUDA input");
    Tensor result({pointers.size()}, device);
    for (size_t i = 0; i < pointers.size(); ++i) {
      if (!pointers[i]) throw std::invalid_argument("Null combined CUDA pointer");
      check(cudaMemcpy(result.data_ptr() + i, pointers[i], sizeof(T), cudaMemcpyDeviceToDevice));
    }
    check(cudaStreamSynchronize(nullptr));
    return result;
  }
  Tensor(const Tensor&) = default;
  Tensor& operator=(const Tensor&) = default;
  Tensor(Tensor&&) noexcept = default;
  Tensor& operator=(Tensor&&) noexcept = default;

  size_t nbytes() const noexcept { return length_ * sizeof(T); }
  size_t numel() const noexcept { return length_; }
  Device device() const noexcept { return device_; }
  size_t offset() const noexcept { return offset_; }
  const std::vector<size_t>& sizes() const noexcept { return shape_; }
  const std::vector<size_t>& strides() const noexcept { return strides_; }
  T* data_ptr() noexcept {
    T* base = device_ == Device::CPU ? (data_ ? data_->data() : nullptr) : gpu_data_.get();
    return offset_ ? base + offset_ : base;
  }
  const T* data_ptr() const noexcept {
    const T* base = device_ == Device::CPU ? (data_ ? data_->data() : nullptr) : gpu_data_.get();
    return offset_ ? base + offset_ : base;
  }
  bool is_contiguous() const noexcept {
    if (!length_) return true;
    size_t expected = 1;
    for (size_t i = shape_.size(); i-- > 0;) {
      if (shape_[i] != 1 && strides_[i] != expected) return false;
      expected *= shape_[i];
    }
    return true;
  }
  Tensor& view(const std::vector<size_t>& shape) & {
    if (count(shape) != length_)
      throw std::invalid_argument("Tensor reshape changes element count");
    if (shape == shape_) return *this;
    if (!is_contiguous()) throw std::invalid_argument("Tensor reshape requires contiguous storage");
    auto strides = dense_strides(shape);
    shape_ = shape;
    strides_ = std::move(strides);
    return *this;
  }
  Tensor view(const std::vector<size_t>& shape) const& {
    auto result = *this;
    result.view(shape);
    return result;
  }
  Tensor view(const std::vector<size_t>& shape) && {
    view(shape);
    return std::move(*this);
  }
  Tensor transpose(int first, int second) const {
    const int rank = static_cast<int>(shape_.size());
    if (first < 0) first += rank;
    if (second < 0) second += rank;
    if (first < 0 || second < 0 || first >= rank || second >= rank)
      throw std::out_of_range("Tensor transpose axis out of range");
    Tensor result = *this;
    std::swap(result.shape_[first], result.shape_[second]);
    std::swap(result.strides_[first], result.strides_[second]);
    return result;
  }
  Tensor slice(const std::vector<size_t>& first, const std::vector<size_t>& end) const {
    if (first.size() != shape_.size() || end.size() != shape_.size())
      throw std::invalid_argument("Tensor slice rank mismatch");
    auto shape = shape_;
    for (size_t i = 0; i < shape.size(); ++i) {
      if (first[i] > end[i] || end[i] > shape[i])
        throw std::out_of_range("Tensor slice exceeds extent");
      shape[i] = end[i] - first[i];
    }
    Tensor result = *this;
    result.shape_ = std::move(shape);
    result.length_ = shape_.empty() && !length_ ? 0 : count(result.shape_);
    if (result.length_) {
      for (size_t i = 0; i < shape_.size(); ++i) result.offset_ += first[i] * strides_[i];
    }
    return result;
  }
  Tensor squeeze(size_t axis) {
    if (axis >= shape_.size()) throw std::out_of_range("Tensor squeeze axis out of range");
    if (shape_[axis] != 1) throw std::invalid_argument("Tensor squeeze requires a singleton axis");
    shape_.erase(shape_.begin() + axis);
    strides_.erase(strides_.begin() + axis);
    return *this;
  }
  Tensor& cuda() {
    if (device_ == Device::CUDA) return *this;
    auto values = host_values();
    auto storage = allocate(length_);
    if (length_) {
      check(cudaMemcpy(storage.get(), values.data(), nbytes(), cudaMemcpyHostToDevice));
      check(cudaStreamSynchronize(nullptr));
    }
    gpu_data_ = std::move(storage);
    data_.reset();
    offset_ = 0;
    strides_ = dense_strides(shape_);
    device_ = Device::CUDA;
    return *this;
  }
  Tensor& cpu() {
    if (device_ == Device::CPU) return *this;
    auto storage = std::make_shared<std::vector<T>>(host_values());
    data_ = std::move(storage);
    gpu_data_.reset();
    offset_ = 0;
    strides_ = dense_strides(shape_);
    device_ = Device::CPU;
    return *this;
  }

 private:
  std::shared_ptr<std::vector<T>> data_;
  std::shared_ptr<T> gpu_data_;
  std::vector<size_t> shape_, strides_;
  size_t offset_ = 0, length_ = 0;
  Device device_ = Device::CPU;
};

template <typename From, typename To>
Tensor<To> tensor_convert(const Tensor<From>& source) {
  if (!source.is_contiguous())
    throw std::invalid_argument("Tensor conversion requires contiguous storage");
  std::vector<From> input(source.numel());
  if (source.device() == Device::CPU)
    std::copy_n(source.data_ptr(), input.size(), input.data());
  else if (!input.empty()) {
    const auto status =
        cudaMemcpy(input.data(), source.data_ptr(), source.nbytes(), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
  }
  std::vector<To> output(input.size());
  std::transform(input.begin(), input.end(), output.begin(),
                 [](From value) { return static_cast<To>(value); });
  return Tensor<To>(std::move(output), source.sizes(), source.device());
}
