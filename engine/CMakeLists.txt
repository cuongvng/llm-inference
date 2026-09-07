add_library(llm_engine
  src/device_info.cu
  kernels/vector_add.cu
)

target_include_directories(llm_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(llm_engine PUBLIC CUDA::cudart)
target_compile_definitions(llm_engine PUBLIC LLM_BUILD_SM=${LLM_CUDA_ARCH})
set_target_properties(llm_engine PROPERTIES
  CUDA_ARCHITECTURES ${LLM_CUDA_ARCH}
  CUDA_SEPARABLE_COMPILATION OFF
  POSITION_INDEPENDENT_CODE ON
)

add_library(llm::engine ALIAS llm_engine)

# One test executable per source file; no test framework by design (
# dependency policy) -- a test is a main() that returns non-zero on failure.
set(LLM_TESTS
  test_vector_add
  test_device_baseline
)

foreach(test_name IN LISTS LLM_TESTS)
  add_executable(${test_name} tests/${test_name}.cu)
  target_link_libraries(${test_name} PRIVATE llm_engine)
  set_target_properties(${test_name} PROPERTIES CUDA_ARCHITECTURES ${LLM_CUDA_ARCH})
  add_test(NAME ${test_name} COMMAND ${test_name})
  # A hung GPU test should fail the suite, not the CI job's wall clock.
  set_tests_properties(${test_name} PROPERTIES TIMEOUT 120)
endforeach()
