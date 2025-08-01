#
# NVIDIA INTERNAL, DO NOT DISTRIBUTE
#

# CMAKE (some fatbin related)
cmake_minimum_required(VERSION 3.22)

# PROJECT
project(green_context LANGUAGES C CXX CUDA)

# VERBOSE COMPILE
set(CMAKE_VERBOSE_MAKEFILE ON)

# COMPILE AND LINKING
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# CTK
find_package(CUDAToolkit REQUIRED)

# Python and pybind11 for Python bindings
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Find pybind11
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYBIND11_FOUND
)

if(NOT PYBIND11_FOUND EQUAL 0)
    message(WARNING "pybind11 not found. Python bindings will be disabled.")
    set(BUILD_PYTHON_BINDINGS OFF)
else()
    find_package(pybind11 REQUIRED PATHS ${PYBIND11_CMAKE_DIR})
    set(BUILD_PYTHON_BINDINGS ON)
endif()

#
# SET FLAGS
#

# CUDA Flag
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --keep -lineinfo")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G")
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CXX FLAG
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ARCH
set(CMAKE_CUDA_ARCHITECTURES 90a 100a 120a)

# Debug/Release
set(CMAKE_BUILD_TYPE "Release")

#
# UTILITY HEADER
#
find_path(GCFO_UTILS_INCLUDE_DIR . REQUIRED)
message(GCFO_UTILS_INCLUDE_DIR="${GCFO_UTILS_INCLUDE_DIR}")

# INCLUDE DIRECTORY FOR ALL
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

#
# PYTHON EXTENSION MODULE
#
if(BUILD_PYTHON_BINDINGS)
    message(STATUS "Building Python extension module")
    
    # Create Python extension module
    pybind11_add_module(gtx gtx.cpp)
    
    # Link libraries
    target_link_libraries(gtx PRIVATE 
        CUDA::cuda_driver
        CUDA::cudart
    )
    
    # Set properties
    set_target_properties(gtx PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    
    message(STATUS "Python extension will be built as: gtx")
else()
    message(STATUS "Skipping Python bindings (pybind11 not found)")
endif()

#
# OPTIONAL: C++ LIBRARY (without Python bindings)
#
# Uncomment below if you want to build a standalone C++ library
# add_library(green_context_lib STATIC gtx.cpp)
# target_link_libraries(green_context_lib PUBLIC CUDA::cuda_driver CUDA::cudart)

# Display build information
message(STATUS "=== Build Configuration ===")
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "Python executable: ${Python_EXECUTABLE}")
message(STATUS "CUDA version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build Python bindings: ${BUILD_PYTHON_BINDINGS}")
message(STATUS "==========================")