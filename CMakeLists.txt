cmake_minimum_required(VERSION 3.20)
project(Atlas LANGUAGES CXX)

# ============================================================
# Global Settings
# ============================================================

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ============================================================
# Atlas Memory Library
# ============================================================

add_library(atlas_memory STATIC
  atlas_memory/src/layout.cpp
  atlas_memory/src/workspace.cpp
  atlas_memory/src/packing_a.cpp
  atlas_memory/src/packing_b.cpp
)

target_include_directories(atlas_memory PUBLIC
  atlas_memory/include
)

# ============================================================
# GEMM Kernels Library
# ============================================================

add_library(gemm_kernels STATIC
  gemm/v0_naive.cpp
  gemm/v1_loop_reorder.cpp
  gemm/v2_blocked.cpp
  gemm/v3_scalar_tile.cpp
  gemm/v4_neon_4x4.cpp
  gemm/v4_neon_8x8.cpp
  gemm/v5_packed.cpp
  gemm/v6_parallel.cpp
)

target_include_directories(gemm_kernels PUBLIC
  gemm
)

target_link_libraries(gemm_kernels PUBLIC
  atlas_memory
)

# ============================================================
# Compile / Link Flags (Apple M2 tuned)
# ============================================================

foreach(t atlas_memory gemm_kernels)
  target_compile_options(${t} PUBLIC
    $<$<CONFIG:Release>:-O3 -mcpu=apple-m2>
    $<$<CONFIG:Debug>:-O1 -g -fsanitize=address,undefined>
  )

  target_link_options(${t} PUBLIC
    $<$<CONFIG:Debug>:-fsanitize=address,undefined>
  )
endforeach()

# ============================================================
# OpenBLAS support (for test_vs_blas)
# ============================================================

# Homebrew default path
set(OPENBLAS_HOME /opt/homebrew/opt/openblas)

find_path(OPENBLAS_INCLUDE_DIR cblas.h
    HINTS ${OPENBLAS_HOME}/include
)

find_library(OPENBLAS_LIB NAMES openblas
    HINTS ${OPENBLAS_HOME}/lib
)

if(NOT OPENBLAS_INCLUDE_DIR OR NOT OPENBLAS_LIB)
    message(FATAL_ERROR "OpenBLAS not found. Please install via 'brew install openblas'")
endif()

# ============================================================
# Testing + Benchmarks
# ============================================================

enable_testing()

function(add_test_executable name)
  add_executable(${name} tests/${name}.cpp)

  # Default targets
  target_link_libraries(${name}
    PRIVATE gemm_kernels atlas_memory
  )

  # Special case for test_vs_blas
  if(${name} STREQUAL "test_vs_blas")
    target_include_directories(${name} PRIVATE ${OPENBLAS_INCLUDE_DIR})
    target_link_libraries(${name} PRIVATE ${OPENBLAS_LIB} pthread)
  endif()

  target_compile_options(${name} PRIVATE
    $<$<CONFIG:Release>:-O3 -mcpu=apple-m2>
    $<$<CONFIG:Debug>:-O1 -g -fsanitize=address,undefined>
  )

  target_link_options(${name} PRIVATE
    $<$<CONFIG:Debug>:-fsanitize=address,undefined>
  )

  add_test(NAME ${name} COMMAND ${name})
endfunction()

# ============================================================
# Benchmarks
# ============================================================

add_test_executable(benchmark_gemm_v0)
add_test_executable(benchmark_gemm_v1)
add_test_executable(benchmark_gemm_v2)
add_test_executable(benchmark_gemm_v3)
add_test_executable(benchmark_gemm_v4_4x4)
add_test_executable(benchmark_gemm_v4_8x8)
add_test_executable(benchmark_gemm_v5)
add_test_executable(benchmark_gemm_v6)
add_test_executable(benchmark_naive_vs_packed)

# ============================================================
# Correctness + Memory Tests
# ============================================================

add_test_executable(test_basic_blocked_gemm)
add_test_executable(test_gemm_correctness)
add_test_executable(test_layout_and_alignment)
add_test_executable(test_layout_math)
add_test_executable(test_multiple_block_configs)
add_test_executable(test_packing_correctness)
add_test_executable(test_reset_behavior)
add_test_executable(test_stress_allocation)

# ============================================================
# OpenBLAS comparison
# ============================================================

add_test_executable(test_vs_blas)

