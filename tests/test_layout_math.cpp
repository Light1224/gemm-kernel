#include "../atlas_memory/include/atlas_memory/config_m2.hpp"
#include "../atlas_memory/include/atlas_memory/layout.hpp"

#include <cassert>
#include <iostream>

using namespace atlas_memory;

int main() {
  std::cout << "\n=== TEST: Layout Math ===\n";

  constexpr std::size_t BM = 256;
  constexpr std::size_t BN = 256;
  constexpr std::size_t BK = 256;
  constexpr std::size_t MR = 8;
  constexpr std::size_t NR = 8;

  auto layout = compute_layout(BM, BN, BK, MR, NR);

  std::size_t expected_A = BM * BK * sizeof(float);
  std::size_t expected_B = BK * BN * sizeof(float);
  std::size_t expected_C = MR * NR * sizeof(float);

  assert(layout.a.bytes == expected_A);
  assert(layout.b.bytes == expected_B);
  assert(layout.accum.bytes == expected_C);

  std::cout << "Layout math PASSED\n";
}
