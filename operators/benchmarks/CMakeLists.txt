# Optional reference benchmark, excluded from the default engine build.
add_executable(avx_matmul_bench EXCLUDE_FROM_ALL avx_matmul_bench.cpp)
target_link_libraries(avx_matmul_bench PRIVATE unified_operators)
# The historical name is retained. No AVX-only instructions are assumed.
