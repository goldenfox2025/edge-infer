#pragma once
#include <cuda_runtime.h>

#include <initializer_list>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "CudaMemoryPool.hpp"  // CUDA allocation and shared pool access

enum class Device { CPU, CUDA };

// Forward declaration for template friend declarations.
template <typename T>
class Tensor;

// Conversion helper for Tensor<__nv_bfloat16> to Tensor<float>.
template <typename FromType, typename ToType>
Tensor<ToType> tensor_convert(const Tensor<FromType>& src);

template <typename T>
class Tensor {
   private:
    // Static CUDA error checking is also usable from const members.
    static inline void checkCudaError(cudaError_t error) {
        if (error != cudaSuccess) {
            throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(error)));
        }
    }

    static size_t compute_numel(const std::vector<size_t>& shape) {
        size_t numel = 1;
        for (size_t dim : shape) {
            numel *= dim;
        }
        return numel;
    }

    static std::shared_ptr<T> make_gpu_owner(T* gpu_ptr) {
        return std::shared_ptr<T>(gpu_ptr, [](T* ptr) { GlobalCudaMemoryPool::instance().free(ptr); });
    }

   public:
    // Allow Tensor<U> specializations to access Tensor<T> internals.
    template <typename U>
    friend class Tensor;

    // Grant the conversion helper access to tensor internals.
    template <typename FromType, typename ToType>
    friend Tensor<ToType> tensor_convert(const Tensor<FromType>& src);

    // Copy individual GPU values into one tensor.
    // Parameters:
    // - gpu_ptrs: GPU pointer array
    // - device: Device type (CUDA by default)
    // Return a tensor containing all pointed-to values, with shape [gpu_ptrs.size()].
    template <typename PtrType>
    static Tensor<T> combine_gpu_ptrs(const std::vector<PtrType*>& gpu_ptrs, Device device = Device::CUDA) {
        if (gpu_ptrs.empty()) {
            throw std::runtime_error("Cannot combine empty GPU pointers array");
        }

        // Create the result with shape [gpu_ptrs.size()].
        size_t seq_len = gpu_ptrs.size();
        Tensor<T> result({seq_len}, device);

        if (device != Device::CUDA) {
            throw std::runtime_error("combine_gpu_ptrs only supports CUDA device");
        }

        // Copy the value from each input pointer.
        for (size_t i = 0; i < seq_len; ++i) {
            // Copy each GPU value into its corresponding result element.
            checkCudaError(cudaMemcpy(result.data_ptr() + i,    // Destination element
                                      gpu_ptrs[i],              // Source GPU pointer
                                      sizeof(T),                // Element size
                                      cudaMemcpyDeviceToDevice  // Device-to-device copy
                                      ));
        }

        return result;
    }

    static Tensor<T> from_external_buffer(T* ptr, const std::vector<size_t>& shape, Device device) {
        if (ptr == nullptr) {
            throw std::runtime_error("from_external_buffer requires a non-null pointer");
        }
        if (device != Device::CUDA) {
            throw std::runtime_error("from_external_buffer currently only supports CUDA buffers");
        }

        Tensor<T> result;
        result.shape_ = shape;
        result.strides_ = compute_strides(shape);
        result.offset_ = 0;
        result.length_ = compute_numel(shape);
        result.device_ = device;
        result.data_.reset();
        result.gpu_data_ = std::shared_ptr<T>(ptr, [](T* /*unused*/) {});
        result.tag_.clear();
        return result;
    }

    // Construct an empty CPU tensor.
    Tensor()
        : data_(std::make_shared<std::vector<T>>()),
          offset_(0),
          length_(0),
          device_(Device::CPU),
          gpu_data_(nullptr, [](T* ptr) { /* no-op */ }) {
    }

    // Construct a CPU tensor from data covering every element of the shape.
    Tensor(std::shared_ptr<std::vector<T>> data, const std::vector<size_t>& shape)
        : data_(data),
          shape_(shape),
          offset_(0),
          length_(compute_numel(shape)),
          device_(Device::CPU),
          gpu_data_(nullptr, [](T* ptr) { /* no-op */ }) {
        if (length_ > data_->size()) {
            throw std::runtime_error("Data size does not match tensor shape");
        }
        strides_ = compute_strides(shape_);
    }

    // Allocate storage for the supplied shape and device.
    // is_prefill: Whether this is a prefill allocation
    // tag: Tag for persistent allocations
    Tensor(std::initializer_list<size_t> shape, Device device = Device::CPU, bool is_prefill = false,
           const std::string& tag = "")
        : shape_(shape),
          offset_(0),
          length_(compute_numel(shape_)),
          device_(device),
          gpu_data_(nullptr, [](T* ptr) { /* no-op */ }),
          tag_(tag) {
        strides_ = compute_strides(shape_);
        if (device_ == Device::CPU) {
            data_ = std::make_shared<std::vector<T>>(length_);
            gpu_data_.reset();
        } else if (device_ == Device::CUDA) {
            data_.reset();
            // Allocate GPU storage through the memory pool.
            T* gpu_ptr = nullptr;
            if (!tag_.empty()) {
                // Use a tagged persistent allocation.
                gpu_ptr = static_cast<T*>(GlobalCudaMemoryPool::allocate_tagged(tag_, length_ * sizeof(T), is_prefill));
            } else {
                // Ordinary allocation
                gpu_ptr = static_cast<T*>(GlobalCudaMemoryPool::instance().allocate(length_ * sizeof(T), is_prefill));
            }
            gpu_data_ = make_gpu_owner(gpu_ptr);
        } else {
            throw std::runtime_error("Invalid device specified");
        }
    }

    // Wrap an already allocated GPU pointer returned by sampling.
    Tensor(T* gpu_ptr, const std::vector<size_t>& shape, Device device)
        : shape_(shape),
          offset_(0),
          length_(compute_numel(shape_)),
          device_(Device::CUDA),
          data_(nullptr),
          gpu_data_(make_gpu_owner(gpu_ptr)) {
        if (device == Device::CPU) {
            throw std::runtime_error("Invalid device specified in Tensor constructor");
        }
        strides_ = compute_strides(shape_);
    }
    // Move vector data into a CPU tensor.
    Tensor(std::vector<T>&& data, const std::vector<size_t>& shape)
        : data_(std::make_shared<std::vector<T>>(std::move(data))),
          shape_(shape),
          offset_(0),
          length_(compute_numel(shape_)),
          device_(Device::CPU),
          gpu_data_(nullptr, [](T* ptr) { /* no-op */ }) {
        if (length_ > data_->size()) {
            throw std::runtime_error("Data size does not match tensor shape");
        }
        strides_ = compute_strides(shape_);
    }

    // Construct from moved vector data, shape, and device.
    Tensor(std::vector<T>&& data, const std::vector<size_t>& shape, Device device)
        : shape_(shape), offset_(0), length_(compute_numel(shape_)), device_(device), gpu_data_(nullptr, [](T* ptr) { /* no-op */ }) {
        strides_ = compute_strides(shape_);
        if (device_ == Device::CPU) {
            // Keep the vector data directly on the CPU.
            data_ = std::make_shared<std::vector<T>>(std::move(data));
        } else if (device_ == Device::CUDA) {
            // Allocate GPU storage through the pool and copy synchronously.
            // Do not use cudaMemcpyAsync: the source vector is destroyed when the constructor returns.
            data_.reset();
            T* gpu_ptr = static_cast<T*>(GlobalCudaMemoryPool::instance().allocate(length_ * sizeof(T)));
            checkCudaError(cudaMemcpy(gpu_ptr, data.data(), length_ * sizeof(T), cudaMemcpyHostToDevice));
            gpu_data_ = make_gpu_owner(gpu_ptr);
        } else {
            throw std::runtime_error("Invalid device specified");
        }
    }

    // Allocate a tensor for the shape (CPU by default).
    Tensor(const std::vector<size_t>& shape)
        : shape_(shape), offset_(0), length_(compute_numel(shape_)), device_(Device::CPU), gpu_data_(nullptr, [](T* ptr) { /* no-op */ }) {
        data_ = std::make_shared<std::vector<T>>(length_);
        strides_ = compute_strides(shape_);
    }

    // Allocate a tensor for the shape and device.
    // is_prefill: Whether this is a prefill allocation
    // tag: Tag for persistent allocations
    Tensor(const std::vector<size_t>& shape, Device device, bool is_prefill = false, const std::string& tag = "")
        : shape_(shape),
          offset_(0),
          length_(compute_numel(shape_)),
          device_(device),
          gpu_data_(nullptr, [](T* ptr) { /* no-op */ }),
          tag_(tag) {
        strides_ = compute_strides(shape_);
        if (device_ == Device::CPU) {
            data_ = std::make_shared<std::vector<T>>(length_);
            gpu_data_.reset();
        } else if (device_ == Device::CUDA) {
            data_.reset();
            // Allocate GPU storage through the memory pool.
            T* gpu_ptr = nullptr;
            if (!tag_.empty()) {
                // Use a tagged persistent allocation.
                gpu_ptr = static_cast<T*>(GlobalCudaMemoryPool::allocate_tagged(tag_, length_ * sizeof(T), is_prefill));
            } else {
                // Ordinary allocation
                gpu_ptr = static_cast<T*>(GlobalCudaMemoryPool::instance().allocate(length_ * sizeof(T), is_prefill));
            }
            gpu_data_ = make_gpu_owner(gpu_ptr);
        } else {
            throw std::runtime_error("Invalid device specified");
        }
    }

    // CUDA copies share gpu_data_ rather than copying device storage.
    Tensor(const Tensor& other)
        : data_(other.data_),
          shape_(other.shape_),
          strides_(other.strides_),
          offset_(other.offset_),
          length_(other.length_),
          device_(other.device_),
          tag_(other.tag_) {
        if (device_ == Device::CUDA) {
            gpu_data_ = other.gpu_data_;
        } else {
            gpu_data_.reset();
        }
    }

    // Assignment shares CUDA storage.
    Tensor& operator=(const Tensor& other) {
        if (this != &other) {
            shape_ = other.shape_;
            strides_ = other.strides_;
            offset_ = other.offset_;
            length_ = other.length_;
            device_ = other.device_;
            tag_ = other.tag_;
            if (device_ == Device::CUDA) {
                gpu_data_ = other.gpu_data_;
                data_.reset();
            } else {
                data_ = other.data_;
                gpu_data_.reset();
            }
        }
        return *this;
    }

    int nbytes() const {
        return sizeof(T) * length_;
    }
    // Return a const data pointer.
    const T* data_ptr() const {
        if (device_ == Device::CPU) {
            return data_->data() + offset_;
        } else {
            return gpu_data_.get() + offset_;
        }
    }
    // Return a mutable data pointer.
    T* data_ptr() {
        if (device_ == Device::CPU) {
            return data_->data() + offset_;
        } else {
            return gpu_data_.get() + offset_;
        }
    }

    // Return the tensor shape.
    const std::vector<size_t>& sizes() const {
        return shape_;
    }

    // Return the element count.
    size_t numel() const {
        return length_;
    }

    // Return the strides.
    const std::vector<size_t>& strides() const {
        return strides_;
    }  // Tensor member

    // Check whether the tensor is contiguous.
    bool is_contiguous() const {
        if (strides_.empty() || shape_.empty())
            return true;
        size_t expected_stride = 1;
        for (int i = shape_.size() - 1; i >= 0; --i) {
            // Singleton dimensions do not constrain contiguity because their strides are arbitrary.
            if (shape_[i] != 1) {
                if (strides_[i] != expected_stride) {
                    return false;
                }
            }
            expected_stride *= shape_[i];
        }
        return true;
    }

    // Reshape a tensor view in place.
    Tensor<T>& view(const std::vector<size_t>& new_shape) & {
        // 1. Require the same total element count.
        size_t new_numel = 1;
        for (size_t dim : new_shape) {
            new_numel *= dim;
        }
        if (new_numel != length_) {
            throw std::runtime_error("view: New shape's number of elements must match original");
        }

        // 2. Fast path for a fully contiguous tensor.
        if (this->is_contiguous()) {
            shape_ = new_shape;
            strides_ = compute_strides(new_shape);
            return *this;
        }

        // Check whether only a contiguous suffix is being reshaped.
        // The innermost stride must be one.
        if (strides_.empty() || strides_.back() != 1) {
            throw std::runtime_error(
                "view failed: a view can only be created for tensors that are contiguous "
                "or have a stride of 1 for the last dimension.");
        }

        // Find the first dimension d where the shapes differ.
        int d = 0;
        while (d < shape_.size() && d < new_shape.size() && shape_[d] == new_shape[d]) {
            d++;
        }

        // Require matching element counts in the old and new suffixes starting at d.
        size_t old_tail_numel = 1;
        for (size_t i = d; i < shape_.size(); ++i)
            old_tail_numel *= shape_[i];

        size_t new_tail_numel = 1;
        for (size_t i = d; i < new_shape.size(); ++i)
            new_tail_numel *= new_shape[i];

        if (old_tail_numel == new_tail_numel) {
            // The suffix can be reshaped without copying.
            std::vector<size_t> final_strides(new_shape.size());

            // Preserve strides in the common prefix.
            for (int i = 0; i < d; ++i) {
                final_strides[i] = strides_[i];
            }

            // Compute strides for the reshaped suffix.
            // strides_.back() == 1 establishes a C-contiguous suffix here.
            // Compute suffix strides backwards, starting from one.
            size_t current_stride = 1;
            for (int i = new_shape.size() - 1; i >= d; --i) {
                final_strides[i] = current_stride;
                if (new_shape[i] > 0) {  // Avoid multiplication by zero
                    current_stride *= new_shape[i];
                }
            }

            shape_ = new_shape;
            strides_ = final_strides;
            return *this;
        }

        // 4. Reject layouts that cannot be reshaped safely without copying.
        throw std::runtime_error(
            "view failed: cannot view this non-contiguous tensor in this way without copying data.");
    }

    // Const lvalue overload
    Tensor<T> view(const std::vector<size_t>& new_shape) const& {
        Tensor result = *this;   // Copy view metadata
        result.view(new_shape);  // Reshape through the mutable lvalue overload
        return result;
    }

    // Rvalue overload
    Tensor<T> view(const std::vector<size_t>& new_shape) && {
        this->view(new_shape);  // Reshape the temporary in place
        return std::move(*this);
    }
    // transpose: swap two dimensions
    Tensor<T> transpose(int dim0, int dim1) const {
        if (dim0 < 0)
            dim0 += shape_.size();
        if (dim1 < 0)
            dim1 += shape_.size();
        if (dim0 >= shape_.size() || dim1 >= shape_.size()) {
            throw std::runtime_error("transpose: dimension index out of range");
        }
        Tensor<T> result(*this);
        std::swap(result.shape_[dim0], result.shape_[dim1]);
        std::swap(result.strides_[dim0], result.strides_[dim1]);
        return result;
    }

    // slice: create a view sharing the underlying storage
    Tensor<T> slice(const std::vector<size_t>& start, const std::vector<size_t>& end) const {
        if (start.size() != shape_.size() || end.size() != shape_.size()) {
            throw std::runtime_error("slice: start and end must have same dimensions as tensor");
        }
        std::vector<size_t> new_shape(shape_.size());
        for (size_t i = 0; i < shape_.size(); i++) {
            if (start[i] > shape_[i] || end[i] > shape_[i] || start[i] > end[i]) {
                throw std::runtime_error("slice: invalid start or end indices");
            }
            new_shape[i] = end[i] - start[i];
        }
        size_t new_offset = offset_;
        for (size_t i = 0; i < shape_.size(); i++) {
            new_offset += start[i] * strides_[i];
        }
        size_t new_length = 1;
        for (size_t dim : new_shape) {
            new_length *= dim;
        }
        Tensor<T> result;
        result.shape_ = new_shape;
        result.strides_ = strides_;
        result.offset_ = new_offset;
        result.length_ = new_length;
        result.device_ = device_;
        if (device_ == Device::CPU) {
            result.data_ = data_;
            result.gpu_data_.reset();
        } else {
            result.data_.reset();
            result.gpu_data_ = gpu_data_;
        }
        return result;
    }

    Tensor<T> squeeze(size_t dim) {
        if (dim >= shape_.size()) {
            std::cerr << "Dimension " << dim << " is out of range." << std::endl;
            return *this;
        }
        if (shape_[dim] != 1) {
            std::cout << "Cannot squeeze dimension " << dim << " because its size is " << shape_[dim] << " (not 1)."
                      << std::endl;
            return *this;
        }
        shape_.erase(shape_.begin() + dim);
        strides_.erase(strides_.begin() + dim);
        return *this;
    }

    // Copy CPU data into GPU storage allocated through the pool.
    // is_prefill: Whether this is a prefill allocation
    // tag: Tag for persistent allocations
    Tensor<T>& cuda(bool is_prefill = false, const std::string& tag = "") {
        if (device_ == Device::CUDA)
            return *this;

        // Use the new allocation tag when supplied.
        if (!tag.empty()) {
            tag_ = tag;
        }

        // Allocate GPU storage through the memory pool.
        T* gpu_ptr = nullptr;
        if (!tag_.empty()) {
            // Use a tagged persistent allocation.
            gpu_ptr = static_cast<T*>(GlobalCudaMemoryPool::allocate_tagged(tag_, length_ * sizeof(T), is_prefill));
        } else {
            // Ordinary allocation
            gpu_ptr = static_cast<T*>(GlobalCudaMemoryPool::instance().allocate(length_ * sizeof(T), is_prefill));
        }

        checkCudaError(cudaMemcpy(gpu_ptr, data_ptr(), length_ * sizeof(T), cudaMemcpyHostToDevice));
        data_.reset();
        gpu_data_ = make_gpu_owner(gpu_ptr);
        device_ = Device::CUDA;
        return *this;
    }

    // Copy GPU data to CPU storage.
    Tensor<T>& cpu() {
        if (device_ == Device::CPU)
            return *this;
        data_ = std::make_shared<std::vector<T>>(length_);
        checkCudaError(cudaMemcpy(data_->data(), gpu_data_.get(), length_ * sizeof(T), cudaMemcpyDeviceToHost));
        gpu_data_.reset();
        device_ = Device::CPU;
        return *this;
    }

    // Return the current device.
    Device device() const {
        return device_;
    }
    size_t offset() const {
        return offset_;
    }

    // Return the allocation tag.
    const std::string& tag() const {
        return tag_;
    }

   private:
    // Compute the strides.
    static std::vector<size_t> compute_strides(const std::vector<size_t>& shape) {
        std::vector<size_t> strides(shape.size());
        size_t stride = 1;
        for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    }

    std::shared_ptr<std::vector<T>> data_;  // CPU storage
    std::shared_ptr<T> gpu_data_;           // GPU storage owned by shared_ptr with a custom deleter
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    size_t offset_;
    size_t length_;
    Device device_;
    std::string tag_;  // Tag for persistent allocations
};

// Tensor conversion helper
template <typename FromType, typename ToType>
Tensor<ToType> tensor_convert(const Tensor<FromType>& src) {
    // Only reinterpret contiguous layouts without copying; reject non-contiguous views to preserve offsets and strides.
    if (!src.is_contiguous()) {
        throw std::runtime_error("tensor_convert currently requires a contiguous tensor");
    }

    Tensor<ToType> result(src.shape_, src.device_);
    result.offset_ = 0;
    result.length_ = src.length_;
    result.strides_ = result.compute_strides(src.shape_);
    result.tag_.clear();

    if (src.device_ == Device::CPU) {
        auto new_data = std::make_shared<std::vector<ToType>>(src.length_);
        const FromType* src_ptr = src.data_ptr();
        for (size_t i = 0; i < src.length_; ++i) {
            (*new_data)[i] = static_cast<ToType>(src_ptr[i]);
        }
        result.data_ = new_data;
        result.gpu_data_.reset();
    } else {
        std::vector<FromType> host_data(src.length_);
        const FromType* src_ptr = src.data_ptr();
        cudaError_t err = cudaMemcpy(host_data.data(), src_ptr, src.length_ * sizeof(FromType), cudaMemcpyDeviceToHost);
        result.checkCudaError(err);
        std::vector<ToType> converted(src.length_);
        for (size_t i = 0; i < src.length_; ++i) {
            converted[i] = static_cast<ToType>(host_data[i]);
        }
        err = cudaMemcpy(result.data_ptr(), converted.data(), src.length_ * sizeof(ToType), cudaMemcpyHostToDevice);
        result.checkCudaError(err);
    }
    return result;
}
