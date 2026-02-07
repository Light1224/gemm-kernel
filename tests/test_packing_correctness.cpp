#include "../atlas_memory/include/atlas_memory/packing.hpp"

#include <cassert>
#include <iostream>
#include <vector>

using namespace atlas_memory;

int main() {
  std::cout << "\n=== TEST: Packing Correctness ===\n";

  const int rows = 4;
  const int cols = 5;
  const int ld = 7;

  std::vector<float> src(rows * ld);
  std::vector<float> dstA(rows * cols);
  std::vector<float> dstB(rows * cols);

  for (int i = 0; i < rows * ld; ++i)
    src[i] = static_cast<float>(i);

  pack_A(dstA.data(), src.data(), rows, cols, ld);

  for (int i = 0; i < rows; ++i)
    for (int k = 0; k < cols; ++k)
      assert(dstA[i * cols + k] == src[i * ld + k]);

  pack_B(dstB.data(), src.data(), rows, cols, ld);

  for (int k = 0; k < rows; ++k)
    for (int j = 0; j < cols; ++j)
      assert(dstB[k * cols + j] == src[k * ld + j]);

  std::cout << "Packing PASSED\n";
}
