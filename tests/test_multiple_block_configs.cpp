#include "../atlas_memory/include/atlas_memory/workspace.hpp"
#include <iostream>

using namespace atlas_memory;

int main() {
  std::cout << "\n=== TEST: Multiple Block Configurations ===\n";

  const int configs[][3] = {
      {64, 64, 64}, {128, 128, 64}, {256, 128, 128}, {192, 256, 128}};

  for (auto &c : configs) {
    Workspace ws(c[0], c[1], c[2], 8, 8);
    std::cout << "Config BM=" << c[0] << " BN=" << c[1] << " BK=" << c[2]
              << " OK\n";
  }

  std::cout << "Multiple configs PASSED\n";
}
