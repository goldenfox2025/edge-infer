# These tests use only the public core headers, with no engine, CUDA or Python.
foreach(test_name IN ITEMS decode_workspace_plan_test workspace_liveness_test operators_core_test)
  add_executable(${test_name} ${test_name}.cpp)
  target_link_libraries(${test_name} PRIVATE operators_core)
  add_test(NAME ${test_name} COMMAND ${test_name})
  set_tests_properties(${test_name} PROPERTIES LABELS "core;cpu" TIMEOUT 10)
endforeach()

add_executable(tensor_view_test tensor_view_test.cpp ../../runtime/tests/allocation_probe.cpp)
target_link_libraries(tensor_view_test PRIVATE operators_core)
add_test(NAME tensor_view_test COMMAND tensor_view_test)
set_tests_properties(tensor_view_test PROPERTIES LABELS "core;cpu" TIMEOUT 10)

# Direct CUDA tests link computation only; historical adapter tests are optional.
if(TARGET operators_cuda)
  add_executable(operators_cuda_awq_test operators_cuda_awq_test.cu ../../runtime/tests/allocation_probe.cpp)
  target_link_libraries(operators_cuda_awq_test PRIVATE EdgeInfer::operators_cuda)
  set_target_properties(operators_cuda_awq_test PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND UNIX)
    target_compile_definitions(operators_cuda_awq_test PRIVATE EDGE_INFER_TEST_CUDA_WRAPPING=1)
    target_link_options(operators_cuda_awq_test PRIVATE
      LINKER:--wrap=cudaMalloc LINKER:--wrap=cudaMallocAsync
      LINKER:--wrap=cudaFree LINKER:--wrap=cudaFreeAsync)
  endif()
  add_test(NAME operators_cuda_awq_test COMMAND operators_cuda_awq_test)
  set_tests_properties(operators_cuda_awq_test PROPERTIES LABELS "operators;cuda;gpu" SKIP_RETURN_CODE 77 TIMEOUT 120)
  add_executable(operators_cuda_sampling_test operators_cuda_sampling_test.cu ../../runtime/tests/allocation_probe.cpp)
  target_link_libraries(operators_cuda_sampling_test PRIVATE EdgeInfer::operators_cuda)
  set_target_properties(operators_cuda_sampling_test PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND UNIX)
    target_compile_definitions(operators_cuda_sampling_test PRIVATE EDGE_INFER_TEST_CUDA_WRAPPING=1)
    target_link_options(operators_cuda_sampling_test PRIVATE
      LINKER:--wrap=cudaMalloc LINKER:--wrap=cudaMallocAsync
      LINKER:--wrap=cudaFree LINKER:--wrap=cudaFreeAsync)
  endif()
  add_test(NAME operators_cuda_sampling_test COMMAND operators_cuda_sampling_test)
  set_tests_properties(operators_cuda_sampling_test PROPERTIES LABELS "operators;cuda;gpu" SKIP_RETURN_CODE 77 TIMEOUT 120)
  if(TARGET unified_operators)
  find_package(Threads REQUIRED)
  add_executable(operators_cuda_context_test operators_cuda_context_test.cu)
  target_link_libraries(operators_cuda_context_test PRIVATE unified_operators Threads::Threads)
  set_target_properties(operators_cuda_context_test PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
  add_test(NAME operators_cuda_context_test COMMAND operators_cuda_context_test)
  set_tests_properties(operators_cuda_context_test PROPERTIES
    LABELS "operators;cuda;gpu" SKIP_RETURN_CODE 77 TIMEOUT 60)
  endif()

  add_executable(operators_cuda_conditioning_test operators_cuda_conditioning_test.cu)
  target_link_libraries(operators_cuda_conditioning_test PRIVATE operators_cuda)
  set_target_properties(operators_cuda_conditioning_test PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
  add_test(NAME operators_cuda_conditioning_test COMMAND operators_cuda_conditioning_test)
  set_tests_properties(operators_cuda_conditioning_test PROPERTIES
    LABELS "operators;cuda;gpu" SKIP_RETURN_CODE 77 TIMEOUT 60)

  add_executable(operators_cuda_direct_test operators_cuda_direct_test.cu)
  target_link_libraries(operators_cuda_direct_test PRIVATE operators_cuda)
  set_target_properties(operators_cuda_direct_test PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
  )
  add_test(NAME operators_cuda_direct_test COMMAND operators_cuda_direct_test)
  set_tests_properties(operators_cuda_direct_test PROPERTIES
    LABELS "operators;cuda;gpu"
    SKIP_RETURN_CODE 77
    TIMEOUT 60
  )

  if(TARGET unified_operators)
  add_executable(unified_operators_cpu_test
    unified_operators_cpu_test.cpp
  )
  target_link_libraries(unified_operators_cpu_test PRIVATE
    unified_operators
  )
  add_test(NAME unified_operators_cpu_test COMMAND unified_operators_cpu_test)
  set_tests_properties(unified_operators_cpu_test PROPERTIES LABELS "operators;cpu" TIMEOUT 30)
  endif()
endif()
