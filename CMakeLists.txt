cmake_minimum_required(VERSION 3.10)
project(ZanaFlow C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

include_directories(include)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "src/*.c")

add_library(zanaflow STATIC ${SOURCES})
target_include_directories(zanaflow PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(UNIX)
    target_link_libraries(zanaflow PUBLIC m)
endif()

include(CTest)
enable_testing()

add_executable(test_losses tests/test_losses.c)
target_link_libraries(test_losses PRIVATE zanaflow)
add_test(NAME test_losses COMMAND test_losses)

add_executable(test_autograd tests/test_autograd.c)
target_link_libraries(test_autograd PRIVATE zanaflow)
add_test(NAME test_autograd COMMAND test_autograd)

add_executable(test_linalg tests/test_linalg.c)
target_link_libraries(test_linalg PRIVATE zanaflow)
add_test(NAME test_linalg COMMAND test_linalg)

# بقیه تست‌ها
add_executable(test_activations tests/test_activations.c)
target_link_libraries(test_activations PRIVATE zanaflow)
add_test(NAME test_activations COMMAND test_activations)


add_executable(test_dense tests/test_dense.c)
target_link_libraries(test_dense PRIVATE zanaflow)
add_test(NAME test_dense COMMAND test_dense)

add_executable(test_ops tests/test_ops.c)
target_link_libraries(test_ops PRIVATE zanaflow)
add_test(NAME test_ops COMMAND test_ops)

add_executable(test_sgd tests/test_sgd.c)
target_link_libraries(test_sgd PRIVATE zanaflow)
add_test(NAME test_sgd COMMAND test_sgd)

add_executable(test_tensors tests/test_tensors.c)
target_link_libraries(test_tensors PRIVATE zanaflow)
add_test(NAME test_tensors COMMAND test_tensors)

add_executable(test_tensor_advanced tests/test_tensor_advanced.c)
target_link_libraries(test_tensor_advanced PRIVATE zanaflow)
add_test(NAME test_tensor_advanced COMMAND test_tensor_advanced)

add_executable(test_dense_init tests/test_dense_init.c)
target_link_libraries(test_dense_init PRIVATE zanaflow)
add_test(NAME test_dense_init COMMAND test_dense_init)
