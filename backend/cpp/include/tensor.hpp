#pragma once
#include <cuda_runtime.h>

#include <initializer_list>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "CudaMemoryPool.hpp"  // 该头中定义了 CudaMemoryPool，提供静态方法 allocate/free

enum class Device { CPU, CUDA };

// 前向声明 Tensor 类模板（用于友元声明）
template <typename T>
class Tensor;

// 声明类型转换辅助函数，专用于 Tensor<__nv_bfloat16> 到 Tensor<float> 的转换
template <typename FromType, typename ToType>
Tensor<ToType> tensor_convert(const Tensor<FromType>& src);

template <typename T>
class Tensor {
   private:
    // 检查 CUDA 错误的静态内联函数（可在 const 成员中调用）
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
    // 友元声明，允许任意 Tensor<U> 访问 Tensor<T> 的私有成员
    template <typename U>
    friend class Tensor;

    // 将类型转换辅助函数声明为友元
    template <typename FromType, typename ToType>
    friend Tensor<ToType> tensor_convert(const Tensor<FromType>& src);

    // 静态方法：将多个GPU指针组合成一个tensor
    // 参数：
    // - gpu_ptrs: GPU指针数组
    // - device: 设备类型（默认为CUDA）
    // 返回：包含所有指针数据的tensor，形状为[gpu_ptrs.size()]
    template <typename PtrType>
    static Tensor<T> combine_gpu_ptrs(const std::vector<PtrType*>& gpu_ptrs, Device device = Device::CUDA) {
        if (gpu_ptrs.empty()) {
            throw std::runtime_error("Cannot combine empty GPU pointers array");
        }

        // 创建结果tensor，形状为[gpu_ptrs.size()]
        size_t seq_len = gpu_ptrs.size();
        Tensor<T> result({seq_len}, device);

        if (device != Device::CUDA) {
            throw std::runtime_error("combine_gpu_ptrs only supports CUDA device");
        }

        // 为每个指针分配临时内存并复制数据
        for (size_t i = 0; i < seq_len; ++i) {
            // 从GPU指针复制到结果tensor的对应位置
            checkCudaError(cudaMemcpy(result.data_ptr() + i,    // 目标位置
                                      gpu_ptrs[i],              // 源GPU指针
                                      sizeof(T),                // 每个元素的大小
                                      cudaMemcpyDeviceToDevice  // 设备到设备的复制
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

    // 默认构造函数（CPU 模式，空 tensor）
    Tensor()
        : data_(std::make_shared<std::vector<T>>()),
          offset_(0),
          length_(0),
          device_(Device::CPU),
          gpu_data_(nullptr, [](T* ptr) { /* no-op */ }) {
    }

    // 从已有数据和形状构造（要求数据至少包含所有元素，CPU 模式）
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

    // 从 initializer_list 构造，分配新的数据，并指定设备
    // is_prefill: 是否是prefill阶段的内存分配
    // tag: 内存标签，用于固定内存分配
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
            // 通过内存池申请 GPU 内存
            T* gpu_ptr = nullptr;
            if (!tag_.empty()) {
                // 使用标签分配固定内存
                gpu_ptr = static_cast<T*>(GlobalCudaMemoryPool::allocate_tagged(tag_, length_ * sizeof(T), is_prefill));
            } else {
                // 常规内存分配
                gpu_ptr = static_cast<T*>(GlobalCudaMemoryPool::instance().allocate(length_ * sizeof(T), is_prefill));
            }
            gpu_data_ = make_gpu_owner(gpu_ptr);
        } else {
            throw std::runtime_error("Invalid device specified");
        }
    }

    // 这个仅限gpu_ptr已经被分配了内存，用于sample返回gpu指针的从尝试版本
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
    // 从右值 vector 数据和形状构造（CPU 模式）
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

    // 从右值 vector 数据、形状和 device 构造
    Tensor(std::vector<T>&& data, const std::vector<size_t>& shape, Device device)
        : shape_(shape), offset_(0), length_(compute_numel(shape_)), device_(device), gpu_data_(nullptr, [](T* ptr) { /* no-op */ }) {
        strides_ = compute_strides(shape_);
        if (device_ == Device::CPU) {
            // CPU 模式：直接保存数据
            data_ = std::make_shared<std::vector<T>>(std::move(data));
        } else if (device_ == Device::CUDA) {
            // GPU 模式：通过内存池申请 GPU 内存并同步拷贝数据。
            // 这里不能用 cudaMemcpyAsync，因为源 vector 在构造函数返回后就会析构。
            data_.reset();
            T* gpu_ptr = static_cast<T*>(GlobalCudaMemoryPool::instance().allocate(length_ * sizeof(T)));
            checkCudaError(cudaMemcpy(gpu_ptr, data.data(), length_ * sizeof(T), cudaMemcpyHostToDevice));
            gpu_data_ = make_gpu_owner(gpu_ptr);
        } else {
            throw std::runtime_error("Invalid device specified");
        }
    }

    // 从 shape 构造，分配新的数据（CPU 模式缺省）
    Tensor(const std::vector<size_t>& shape)
        : shape_(shape), offset_(0), length_(compute_numel(shape_)), device_(Device::CPU), gpu_data_(nullptr, [](T* ptr) { /* no-op */ }) {
        data_ = std::make_shared<std::vector<T>>(length_);
        strides_ = compute_strides(shape_);
    }

    // 从 shape 和 device 构造
    // is_prefill: 是否是prefill阶段的内存分配
    // tag: 内存标签，用于固定内存分配
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
            // 通过内存池申请 GPU 内存
            T* gpu_ptr = nullptr;
            if (!tag_.empty()) {
                // 使用标签分配固定内存
                gpu_ptr = static_cast<T*>(GlobalCudaMemoryPool::allocate_tagged(tag_, length_ * sizeof(T), is_prefill));
            } else {
                // 常规内存分配
                gpu_ptr = static_cast<T*>(GlobalCudaMemoryPool::instance().allocate(length_ * sizeof(T), is_prefill));
            }
            gpu_data_ = make_gpu_owner(gpu_ptr);
        } else {
            throw std::runtime_error("Invalid device specified");
        }
    }

    // 拷贝构造（CUDA 下采用浅拷贝，即共享 gpu_data_）
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

    // 赋值运算符（CUDA 下采用浅拷贝）
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
    // 返回数据指针（const 版本）
    const T* data_ptr() const {
        if (device_ == Device::CPU) {
            return data_->data() + offset_;
        } else {
            return gpu_data_.get() + offset_;
        }
    }
    // 返回数据指针（非 const 版本）
    T* data_ptr() {
        if (device_ == Device::CPU) {
            return data_->data() + offset_;
        } else {
            return gpu_data_.get() + offset_;
        }
    }

    // 返回张量尺寸
    const std::vector<size_t>& sizes() const {
        return shape_;
    }

    // 返回元素总数
    size_t numel() const {
        return length_;
    }

    // 返回 strides
    const std::vector<size_t>& strides() const {
        return strides_;
    }  // 在 Tensor class 中

    // 检查张量是否连续的辅助函数
    bool is_contiguous() const {
        if (strides_.empty() || shape_.empty())
            return true;
        size_t expected_stride = 1;
        for (int i = shape_.size() - 1; i >= 0; --i) {
            // 尺寸为1的维度不参与连续性判断，因为它的stride可以是任意值
            if (shape_[i] != 1) {
                if (strides_[i] != expected_stride) {
                    return false;
                }
            }
            expected_stride *= shape_[i];
        }
        return true;
    }

    // 智能 view 函数 - 左值引用版本
    Tensor<T>& view(const std::vector<size_t>& new_shape) & {
        // 1. 检查总元素数量是否匹配 (不变)
        size_t new_numel = 1;
        for (size_t dim : new_shape) {
            new_numel *= dim;
        }
        if (new_numel != length_) {
            throw std::runtime_error("view: New shape's number of elements must match original");
        }

        // 2. 快速路径：如果整个张量是连续的 (不变)
        if (this->is_contiguous()) {
            shape_ = new_shape;
            strides_ = compute_strides(new_shape);
            return *this;
        }

        // 检查是否仅在对一个连续的尾部进行变形
        // 一个简单且有效的判断条件是：张量最内维度的 stride 必须为 1
        if (strides_.empty() || strides_.back() != 1) {
            throw std::runtime_error(
                "view failed: a view can only be created for tensors that are contiguous "
                "or have a stride of 1 for the last dimension.");
        }

        // 找到新旧 shape 开始不同的第一个维度 d
        int d = 0;
        while (d < shape_.size() && d < new_shape.size() && shape_[d] == new_shape[d]) {
            d++;
        }

        // 检查从 d 开始的旧尾部元素总数是否与新尾部匹配
        size_t old_tail_numel = 1;
        for (size_t i = d; i < shape_.size(); ++i)
            old_tail_numel *= shape_[i];

        size_t new_tail_numel = 1;
        for (size_t i = d; i < new_shape.size(); ++i)
            new_tail_numel *= new_shape[i];

        if (old_tail_numel == new_tail_numel) {
            // 条件满足，可以执行智能 view
            std::vector<size_t> final_strides(new_shape.size());

            // 拷贝公共前缀部分的 stride
            for (int i = 0; i < d; ++i) {
                final_strides[i] = strides_[i];
            }

            // 为变形后的尾部计算新的 stride
            // 因为我们已经确认 strides_.back() == 1，所以尾部是“C风格”连续的
            // 我们可以从1开始，反向计算这部分的 stride
            size_t current_stride = 1;
            for (int i = new_shape.size() - 1; i >= d; --i) {
                final_strides[i] = current_stride;
                if (new_shape[i] > 0) {  // 避免乘以0
                    current_stride *= new_shape[i];
                }
            }

            shape_ = new_shape;
            strides_ = final_strides;
            return *this;
        }

        // 4. 如果所有路径都失败，说明这是一个无法安全 view 的情况
        throw std::runtime_error(
            "view failed: cannot view this non-contiguous tensor in this way without copying data.");
    }

    // const 左值版本 (不变)
    Tensor<T> view(const std::vector<size_t>& new_shape) const& {
        Tensor result = *this;   // 浅拷贝元信息
        result.view(new_shape);  // 调用上面的左值版本进行智能修改
        return result;
    }

    // 右值版本 (不变)
    Tensor<T> view(const std::vector<size_t>& new_shape) && {
        this->view(new_shape);  // 对将亡值自身进行修改
        return std::move(*this);
    }
    // transpose：交换两个维度
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

    // slice：提取张量的一部分，仍共享底层数据
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

    // 转换到 CUDA：将数据从 CPU 拷贝到 GPU，通过内存池申请 GPU 内存
    // is_prefill: 是否是prefill阶段的内存分配
    // tag: 内存标签，用于固定内存分配
    Tensor<T>& cuda(bool is_prefill = false, const std::string& tag = "") {
        if (device_ == Device::CUDA)
            return *this;

        // 如果提供了新标签，则使用新标签
        if (!tag.empty()) {
            tag_ = tag;
        }

        // 通过内存池申请 GPU 内存
        T* gpu_ptr = nullptr;
        if (!tag_.empty()) {
            // 使用标签分配固定内存
            gpu_ptr = static_cast<T*>(GlobalCudaMemoryPool::allocate_tagged(tag_, length_ * sizeof(T), is_prefill));
        } else {
            // 常规内存分配
            gpu_ptr = static_cast<T*>(GlobalCudaMemoryPool::instance().allocate(length_ * sizeof(T), is_prefill));
        }

        checkCudaError(cudaMemcpy(gpu_ptr, data_ptr(), length_ * sizeof(T), cudaMemcpyHostToDevice));
        data_.reset();
        gpu_data_ = make_gpu_owner(gpu_ptr);
        device_ = Device::CUDA;
        return *this;
    }

    // 转换到 CPU：将数据从 GPU 拷贝到 CPU
    Tensor<T>& cpu() {
        if (device_ == Device::CPU)
            return *this;
        data_ = std::make_shared<std::vector<T>>(length_);
        checkCudaError(cudaMemcpy(data_->data(), gpu_data_.get(), length_ * sizeof(T), cudaMemcpyDeviceToHost));
        gpu_data_.reset();
        device_ = Device::CPU;
        return *this;
    }

    // 返回当前设备
    Device device() const {
        return device_;
    }
    size_t offset() const {
        return offset_;
    }

    // 获取内存标签
    const std::string& tag() const {
        return tag_;
    }

   private:
    // 计算 strides
    static std::vector<size_t> compute_strides(const std::vector<size_t>& shape) {
        std::vector<size_t> strides(shape.size());
        size_t stride = 1;
        for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    }

    std::shared_ptr<std::vector<T>> data_;  // CPU 数据
    std::shared_ptr<T> gpu_data_;           // GPU 数据（使用 shared_ptr 管理，并传入自定义删除器）
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    size_t offset_;
    size_t length_;
    Device device_;
    std::string tag_;  // 内存标签，用于固定内存分配
};

// 实现类型转换辅助函数
template <typename FromType, typename ToType>
Tensor<ToType> tensor_convert(const Tensor<FromType>& src) {
    // 仅对连续张量执行无复制布局转换；非连续视图直接拒绝，避免制造错误 offset/stride。
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
