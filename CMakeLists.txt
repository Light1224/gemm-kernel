cmake_minimum_required(VERSION 3.20)
project(Atlas LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ==============================
# Atlas Memory
# ==============================

add_library(atlas_memory STATIC
  atlas_memory/src/layout.cpp
  atlas_memory/src/workspace.cpp
  atlas_memory/src/packing_a.cpp
  atlas_memory/src/packing_b.cpp
)

target_include_directories(atlas_memory PUBLIC atlas_memory/include)

# ==============================
# GEMM Kernels
# ==============================

add_library(gemm_kernels STATIC
  gemm/v0_naive.cpp
  gemm/v1_loop_reorder.cpp
  gemm/v2_blocked.cpp
  gemm/v3_scalar_tile.cpp
)

target_include_directories(gemm_kernels PUBLIC gemm)

# ==============================
# Compile / Link Flags
# ==============================

foreach(t atlas_memory gemm_kernels)
  target_compile_options(${t} PUBLIC
    $<$<CONFIG:Release>:-O3 -mcpu=apple-m2>
    $<$<CONFIG:Debug>:-O1 -g -fsanitize=address,undefined>
  )

  target_link_options(${t} PUBLIC
    $<$<CONFIG:Debug>:-fsanitize=address,undefined>
  )
endforeach()

# ==============================
# Benchmarks
# ==============================

enable_testing()

function(add_gemm_bench name)
  add_executable(${name} tests/${name}.cpp)

  target_link_libraries(${name}
    PRIVATE gemm_kernels atlas_memory
  )

  target_compile_options(${name} PRIVATE
    $<$<CONFIG:Release>:-O3 -mcpu=apple-m2>
    $<$<CONFIG:Debug>:-O1 -g -fsanitize=address,undefined>
  )

  target_link_options(${name} PRIVATE
    $<$<CONFIG:Debug>:-fsanitize=address,undefined>
  )

  add_test(NAME ${name} COMMAND ${name})
endfunction()

add_gemm_bench(benchmark_gemm_v0)
add_gemm_bench(benchmark_gemm_v1)
add_gemm_bench(benchmark_gemm_v2)
add_gemm_bench(benchmark_gemm_v3)
add_gemm_bench(test_gemm_correctness) 

