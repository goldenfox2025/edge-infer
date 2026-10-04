pybind11_add_module(model_bridge MODULE model_bridge.cpp model_initializer.cpp)
target_link_libraries(model_bridge PRIVATE EdgeInfer::runtime)
target_include_directories(model_bridge PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(model_bridge PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
foreach(config IN LISTS CMAKE_CONFIGURATION_TYPES)
  string(TOUPPER "${config}" config_upper)
  set_target_properties(model_bridge PROPERTIES
    "LIBRARY_OUTPUT_DIRECTORY_${config_upper}" "${CMAKE_BINARY_DIR}"
    "RUNTIME_OUTPUT_DIRECTORY_${config_upper}" "${CMAKE_BINARY_DIR}"
  )
endforeach()

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
