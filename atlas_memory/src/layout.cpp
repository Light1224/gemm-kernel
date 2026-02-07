#include "../include/atlas_memory/layout.hpp"
#include "../include/atlas_memory/config_m2.hpp"
#include <cassert>

namespace atlas_memory {

static std::size_t align_up(std::size_t x, std::size_t align) {
  return (x + align - 1) & ~(align - 1);
}

Layout compute_layout(std::size_t BM, std::size_t BN, std::size_t BK,
                      std::size_t MR, std::size_t NR) {
  Layout l{};

  std::size_t offset = 0;

  l.a.offset = align_up(offset, config::SIMD_ALIGNMENT);
  l.a.bytes = BM * BK * sizeof(float);
  offset = l.a.offset + l.a.bytes;

  l.b.offset = align_up(offset, config::SIMD_ALIGNMENT);
  l.b.bytes = BK * BN * sizeof(float);
  offset = l.b.offset + l.b.bytes;

  l.accum.offset = align_up(offset, config::SIMD_ALIGNMENT);
  l.accum.bytes = MR * NR * sizeof(float);
  offset = l.accum.offset + l.accum.bytes;

  l.total_bytes = align_up(offset, config::SIMD_ALIGNMENT);

  assert(l.total_bytes < config::MAX_WORKSPACE_BYTES);

  return l;
}

} // namespace atlas_memory
