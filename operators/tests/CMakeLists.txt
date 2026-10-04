# These tests use only the public core headers, with no engine, CUDA or Python.
foreach(test_name IN ITEMS decode_workspace_plan_test workspace_liveness_test operators_core_test)
  add_executable(${test_name} ${test_name}.cpp)
  target_link_libraries(${test_name} PRIVATE operators_core)
  add_test(NAME ${test_name} COMMAND ${test_name})
  set_tests_properties(${test_name} PROPERTIES LABELS "core;cpu" TIMEOUT 10)
endforeach()

# The current Tensor/operator API still depends on the CUDA toolkit even
# for CPU tensors. Build this test with the CUDA compatibility library.
if(TARGET unified_operators)
  add_executable(operators_cuda_conditioning_test operators_cuda_conditioning_test.cu)
  target_link_libraries(operators_cuda_conditioning_test PRIVATE unified_operators)
  set_target_properties(operators_cuda_conditioning_test PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
  add_test(NAME operators_cuda_conditioning_test COMMAND operators_cuda_conditioning_test)
  set_tests_properties(operators_cuda_conditioning_test PROPERTIES
    LABELS "operators;cuda;gpu" SKIP_RETURN_CODE 77 TIMEOUT 60)

  add_executable(operators_cuda_direct_test operators_cuda_direct_test.cu)
  target_link_libraries(operators_cuda_direct_test PRIVATE unified_operators)
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

  add_executable(unified_operators_cpu_test
    unified_operators_cpu_test.cpp
  )
  target_link_libraries(unified_operators_cpu_test PRIVATE
    unified_operators
  )
  add_test(NAME unified_operators_cpu_test COMMAND unified_operators_cpu_test)
  set_tests_properties(unified_operators_cpu_test PROPERTIES LABELS "operators;cpu" TIMEOUT 30)
endif()
