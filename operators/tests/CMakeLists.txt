# These tests use only the public core headers, with no engine, CUDA or Python.
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../../core/tests/support")
foreach(test_name IN ITEMS decode_workspace_plan_test workspace_liveness_test operators_core_test)
  add_executable(${test_name} ${test_name}.cpp)
  target_link_libraries(${test_name} PRIVATE operators_core)
  add_test(NAME ${test_name} COMMAND ${test_name})
  set_tests_properties(${test_name} PROPERTIES LABELS "core;cpu" TIMEOUT 10)
endforeach()

add_executable(tensor_view_test tensor_view_test.cpp ../../core/tests/support/allocation_probe.cpp)
target_link_libraries(tensor_view_test PRIVATE operators_core)
add_test(NAME tensor_view_test COMMAND tensor_view_test)
set_tests_properties(tensor_view_test PROPERTIES LABELS "core;cpu" TIMEOUT 10)

# Direct CUDA tests link computation only.
if(TARGET operators_cuda)
  foreach(test_name IN ITEMS operators_cuda_attention_precision_test operators_cuda_bf16_contract_test)
    add_executable(${test_name} ${test_name}.cu)
    target_link_libraries(${test_name} PRIVATE EdgeInfer::operators_cuda)
    set_target_properties(${test_name} PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    add_test(NAME ${test_name} COMMAND ${test_name})
    set_tests_properties(${test_name} PROPERTIES LABELS "operators;cuda;gpu" SKIP_RETURN_CODE 77 TIMEOUT 120)
  endforeach()
  add_executable(operators_cuda_awq_test operators_cuda_awq_test.cu ../../core/tests/support/allocation_probe.cpp)
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
  add_executable(operators_cuda_sampling_test operators_cuda_sampling_test.cu ../../core/tests/support/allocation_probe.cpp)
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

endif()
