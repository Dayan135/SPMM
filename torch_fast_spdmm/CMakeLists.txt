cmake_minimum_required(VERSION 3.26 FATAL_ERROR)
project(${SKBUILD_PROJECT_NAME}
        VERSION ${SKBUILD_PROJECT_VERSION}
        LANGUAGES CXX CUDA)

if(DEFINED ENV{CONDA_PREFIX})
  set(CUDA_TOOLKIT_ROOT_DIR "$ENV{CONDA_PREFIX}" CACHE PATH "CUDA toolkit root")
  set(CUDA_TOOLKIT_INCLUDE "$ENV{CONDA_PREFIX}/include")
  set(CUDA_TOOLKIT_LIB_DIR "$ENV{CONDA_PREFIX}/lib")
  list(APPEND CMAKE_LIBRARY_PATH "${CUDA_TOOLKIT_LIB_DIR}")
  list(APPEND CMAKE_INCLUDE_PATH "${CUDA_TOOLKIT_INCLUDE}")
endif()

# ---- Toolchain ----
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

# ---- Python & PyTorch ----
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(Torch CONFIG REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

find_package(CUDAToolkit REQUIRED)

set(CPP_SOURCES bindings/bindings.cpp pbr_matrix/pbr_matrix.cpp)

set(CU_SOURCES bindings/bindings_kernels.cu bindings/bindings_cusparse.cu bindings/bindings_boo_rowexcl.cu kernels/boo_rowexcl_kernel.cu kernels/csr_kernels.cu kernels/pbr_kernel.cu)
add_definitions(-DCUDA_ENABLED=1)

add_library(torch_fast_spdmm_core MODULE
  ${CPP_SOURCES}
  ${CU_SOURCES}
)

target_include_directories(torch_fast_spdmm_core PRIVATE
  ${TORCH_INCLUDE_DIRS}
  ${Python_INCLUDE_DIRS}
  ${CUDAToolkit_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/kernels
  ${CMAKE_CURRENT_SOURCE_DIR}/pbr_matrix
)

target_compile_options(torch_fast_spdmm_core PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -w>
  $<$<COMPILE_LANGUAGE:CUDA>:-std=c++20>
)

target_link_libraries(torch_fast_spdmm_core PRIVATE
  ${TORCH_LIBRARIES}
  pybind11::module
  CUDA::cusparse
)

set_target_properties(torch_fast_spdmm_core PROPERTIES
  PREFIX "${PYTHON_MODULE_PREFIX}"
  SUFFIX "${PYTHON_MODULE_EXTENSION}"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build"
)
