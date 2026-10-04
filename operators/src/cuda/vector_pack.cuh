#pragma once
#include <cuda_runtime.h>
template <typename T, int N>
union Vec { float4 f4; T t[N]; };
template <typename T, int N>
union Vec_2 { float2 f2; T t[N]; };
