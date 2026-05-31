cmake_minimum_required(VERSION 3.10)
project(ZanaFlow C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Library
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "src/*.c")

add_library(zanaflow STATIC ${SOURCES})
target_include_directories(zanaflow PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Math lib (safe for UNIX + MinGW; harmless elsewhere)
if(UNIX OR MINGW)
    target_link_libraries(zanaflow PUBLIC m)
endif()

# Tests
include(CTest)
enable_testing()

# Helper for adding tests (prevents copy/paste mistakes)
function(zf_add_test target_name test_source)
    add_executable(${target_name} ${test_source})
    target_link_libraries(${target_name} PRIVATE zanaflow)
    add_test(NAME ${target_name} COMMAND ${target_name})
endfunction()

# ---- Enabled tests ----
zf_add_test(test_adam           tests/test_adam_sanity.c)
zf_add_test(test_adam_exact_step tests/test_adam_exact_step.c)

# ---- Other tests (enable when ready) ----
zf_add_test(test_losses tests/test_losses.c)
zf_add_test(test_autograd tests/test_autograd.c)
zf_add_test(test_linalg tests/test_linalg.c)
zf_add_test(test_activations tests/test_activations.c)
zf_add_test(test_dense tests/test_dense.c)
zf_add_test(test_ops tests/test_ops.c)
zf_add_test(test_sgd tests/test_sgd.c)
zf_add_test(test_tensors tests/test_tensors.c)
zf_add_test(test_tensor_advanced tests/test_tensor_advanced.c)
zf_add_test(test_dense_init tests/test_dense_init.c)
