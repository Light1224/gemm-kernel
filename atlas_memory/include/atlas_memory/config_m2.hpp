#pragma once
#include <cstddef>

namespace atlas_memory::config {

constexpr std::size_t CACHE_LINE = 64;
constexpr std::size_t SIMD_ALIGNMENT = 128;
constexpr std::size_t PAGE_SIZE = 16 * 1024;

constexpr std::size_t DEFAULT_BM = 256;
constexpr std::size_t DEFAULT_BN = 256;
constexpr std::size_t DEFAULT_BK = 256;

constexpr std::size_t MR = 8;
constexpr std::size_t NR = 8;

constexpr std::size_t MAX_WORKSPACE_BYTES = 512ull * 1024 * 1024;

} // namespace atlas_memory::config
