#pragma once
#include <cstddef>

namespace atlas_memory {

struct Region {
  std::size_t offset;
  std::size_t bytes;
};

struct Layout {
  Region a;
  Region b;
  Region accum;
  std::size_t total_bytes;
};

Layout compute_layout(std::size_t BM, std::size_t BN, std::size_t BK,
                      std::size_t MR, std::size_t NR);

} // namespace atlas_memory
