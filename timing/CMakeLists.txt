cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(conv_performance)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(SOURCES "experiments.cpp")

add_executable(conv_performance ${SOURCES})
target_link_libraries(conv_performance "${TORCH_LIBRARIES}")
target_include_directories(conv_performance PUBLIC "../runtime" "../generated_code")
set_property(TARGET conv_performance PROPERTY CXX_STANDARD 14)
