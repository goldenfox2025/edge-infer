cmake_minimum_required(VERSION 3.20)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  project(edge_infer_python_callback_tests LANGUAGES CXX)
  include(CTest)
  set(PYBIND11_FINDPYTHON ON)
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(pybind11 CONFIG REQUIRED)
endif()

find_package(Threads REQUIRED)
pybind11_add_module(_edge_callback_boundary_test MODULE callback_boundary_module.cpp)
target_compile_features(_edge_callback_boundary_test PRIVATE cxx_std_17)
target_include_directories(_edge_callback_boundary_test PRIVATE
                           "${CMAKE_CURRENT_SOURCE_DIR}/../include")
target_link_libraries(_edge_callback_boundary_test PRIVATE Threads::Threads)

if(BUILD_TESTING)
  add_test(NAME python_callback_boundary_test
    COMMAND "${Python_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/callback_boundary_test.py"
            "$<TARGET_FILE_DIR:_edge_callback_boundary_test>"
  )
  set_tests_properties(python_callback_boundary_test PROPERTIES
                       LABELS "bindings;python;cpu" TIMEOUT 15)
  if(TARGET model_bridge)
    pybind11_add_module(_edge_qwen3_logits_test MODULE qwen3_logits_module.cpp ../model_initializer.cpp)
    target_compile_features(_edge_qwen3_logits_test PRIVATE cxx_std_17)
    target_include_directories(_edge_qwen3_logits_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/..")
    target_link_libraries(_edge_qwen3_logits_test PRIVATE EdgeInfer::runtime)
    pybind11_add_module(_edge_weight_conversion_test MODULE weight_conversion_module.cpp)
    target_compile_features(_edge_weight_conversion_test PRIVATE cxx_std_17)
    target_include_directories(_edge_weight_conversion_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../include")
    target_link_libraries(_edge_weight_conversion_test PRIVATE EdgeInfer::core_cuda)
    add_test(NAME python_weight_conversion_test
      COMMAND "${Python_EXECUTABLE}"
              "${CMAKE_CURRENT_SOURCE_DIR}/weight_conversion_test.py"
              "$<TARGET_FILE_DIR:_edge_weight_conversion_test>"
    )
    set_tests_properties(python_weight_conversion_test PROPERTIES
                         LABELS "bindings;python;cpu" TIMEOUT 30)
    add_test(NAME python_model_session_test
      COMMAND "${Python_EXECUTABLE}"
              "${CMAKE_CURRENT_SOURCE_DIR}/model_session_test.py"
              "$<TARGET_FILE_DIR:model_bridge>"
    )
    set_tests_properties(python_model_session_test PROPERTIES
                         LABELS "bindings;python;cpu" TIMEOUT 30)
  endif()
endif()
