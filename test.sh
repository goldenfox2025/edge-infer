#!/bin/bash

set -e

echo "🚀 AVX GEMM Benchmark Test Suite"
echo "================================"

# Build the test directly
echo "🛠️  Building AVX matmul benchmark..."
g++ -O3 -mavx2 -mfma -fopenmp -I./backend/cpp/include -I./operators/include -I./SICore -std=c++17 operators/benchmarks/avx_matmul_bench.cpp -o test_matmul

# Check if the executable exists
if [ ! -f "test_matmul" ]; then
    echo "❌ Build failed: test_matmul executable not found"
    exit 1
fi

echo "✅ Build successful"
echo ""

# Test parameters
SIZES=(
    "1 512 512 512"    # Small
    "1 1024 1024 1024" # Medium  
    "1 2048 2048 2048" # Large
)

echo "🧪 Running AVX GEMM accuracy and performance tests..."
echo ""

for size in "${SIZES[@]}"; do
    echo "┌─────────────────────────────────────────┐"
    echo "│ Testing size: B M K N = $size      │"
    echo "└─────────────────────────────────────────┘"
    
    # Set OpenMP threads and run benchmark
    # Format: B M K N [warmup] [iters]
    export OMP_NUM_THREADS=4
    ./test_matmul $size 3 5
    
    echo ""
done

echo "🏁 All tests completed!"
echo ""
echo "💡 Tips for adding new GEMM versions:"
echo "   1. Implement your function in operators/src/cpu/matmul_cpu.cpp or a new operator source"
echo "   2. Add registration in operators/benchmarks/avx_matmul_bench.cpp main()"
echo "   3. Run this script again to test automatically"
