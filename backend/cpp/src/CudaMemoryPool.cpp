#include "CudaMemoryPool.hpp"

// 此文件为 GlobalCudaMemoryPool 类中声明的静态成员变量提供定义。
// 这是C++的要求，确保这些变量在整个程序中有且仅有一个实例。

CudaMemoryPool* GlobalCudaMemoryPool::pool_instance_ptr = nullptr;

std::mutex GlobalCudaMemoryPool::init_mutex_;

// 注意：
// 所有方法（包括 instance()）的实现都已在 CudaMemoryPool.hpp 头文件中完成。
// 这样做使得大部分实现变为内联（inline），并且简化了项目结构。
// 此处不再需要任何其他代码。
