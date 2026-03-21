cmake_minimum_required(VERSION 3.15)

add_executable(unified_operators_cpu_test
    unified_operators_cpu_test.cpp
    ${CMAKE_SOURCE_DIR}/backend/cpp/src/CudaMemoryPool.cpp
)

target_link_libraries(unified_operators_cpu_test PRIVATE unified_operators)
target_link_libraries(unified_operators_cpu_test PRIVATE CUDA::cuda_driver)

target_include_directories(unified_operators_cpu_test PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/backend/cpp/include
)

add_test(NAME unified_operators_cpu_test COMMAND unified_operators_cpu_test)

add_executable(decode_workspace_plan_test
    decode_workspace_plan_test.cpp
)

target_include_directories(decode_workspace_plan_test PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/backend/cpp/include
)

add_test(NAME decode_workspace_plan_test COMMAND decode_workspace_plan_test)

add_executable(workspace_liveness_test
    workspace_liveness_test.cpp
)

target_include_directories(workspace_liveness_test PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/backend/cpp/include
)

add_test(NAME workspace_liveness_test COMMAND workspace_liveness_test)
