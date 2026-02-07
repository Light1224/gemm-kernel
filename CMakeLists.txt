cmake_minimum_required(VERSION 3.20)
project(Atlas LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ==============================
# Atlas Memory Library
# ==============================

add_library(atlas_memory STATIC
    atlas_memory/src/layout.cpp
    atlas_memory/src/workspace.cpp
    atlas_memory/src/packing_a.cpp
    atlas_memory/src/packing_b.cpp
)

target_include_directories(atlas_memory
    PUBLIC atlas_memory/include
)

target_compile_options(atlas_memory PRIVATE
    $<$<CONFIG:Release>:-O3 -mcpu=apple-m2>
    $<$<CONFIG:Debug>:-O1 -g -fsanitize=address,undefined>
)

target_link_options(atlas_memory PRIVATE
    $<$<CONFIG:Debug>:-fsanitize=address,undefined>
)

# ==============================
# Tests
# ==============================

enable_testing()

function(add_atlas_test name)
    add_executable(${name} tests/${name}.cpp)
    target_link_libraries(${name} PRIVATE atlas_memory)

    target_compile_options(${name} PRIVATE
        $<$<CONFIG:Release>:-O3 -mcpu=apple-m2>
        $<$<CONFIG:Debug>:-O1 -g -fsanitize=address,undefined>
    )

    target_link_options(${name} PRIVATE
        $<$<CONFIG:Debug>:-fsanitize=address,undefined>
    )

    add_test(NAME ${name} COMMAND ${name})
endfunction()

add_atlas_test(test_layout_and_alignment)
add_atlas_test(test_layout_math)
add_atlas_test(test_reset_behavior)
add_atlas_test(test_packing_correctness)
add_atlas_test(test_multiple_block_configs)
add_atlas_test(test_stress_allocation)
add_atlas_test(test_basic_blocked_gemm)
add_atlas_test(benchmark_naive_vs_packed)



