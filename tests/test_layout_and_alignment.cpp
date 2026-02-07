#include "../atlas_memory/include/atlas_memory/config_m2.hpp"
#include "../atlas_memory/include/atlas_memory/layout.hpp"
#include "../atlas_memory/include/atlas_memory/workspace.hpp"

#include <cassert>
#include <cstdint>
#include <iostream>

using namespace atlas_memory;

static bool aligned(void *ptr, std::size_t align) {
  return reinterpret_cast<std::uintptr_t>(ptr) % align == 0;
}

int main() {
  std::cout << "\n=== TEST: Layout & Alignment ===\n";

  constexpr std::size_t BM = 128;
  constexpr std::size_t BN = 192;
  constexpr std::size_t BK = 64;
  constexpr std::size_t MR = 8;
  constexpr std::size_t NR = 8;

  Workspace ws(BM, BN, BK, MR, NR);

  auto *A = ws.packA();
  auto *B = ws.packB();
  auto *C = ws.accum();

  std::cout << "Total capacity: " << ws.total_capacity() << "\n";
  std::cout << "A capacity: " << ws.packA_capacity() << "\n";
  std::cout << "B capacity: " << ws.packB_capacity() << "\n";
  std::cout << "Accum capacity: " << ws.accum_capacity() << "\n";

  assert(aligned(A, config::SIMD_ALIGNMENT));
  assert(aligned(B, config::SIMD_ALIGNMENT));
  assert(aligned(C, config::SIMD_ALIGNMENT));

  assert(A != B);
  assert(A != C);
  assert(B != C);

  std::cout << "Layout & Alignment PASSED\n";
}
