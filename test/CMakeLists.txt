cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(buildit_conv_test)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -fopenmp")
set(SOURCES "tests.cpp")

add_executable(buildit_conv_test ${SOURCES})
target_link_libraries(buildit_conv_test "${TORCH_LIBRARIES}")
target_include_directories(buildit_conv_test PUBLIC "../runtime" "../generated_code")
set_property(TARGET buildit_conv_test PROPERTY CXX_STANDARD 14)
