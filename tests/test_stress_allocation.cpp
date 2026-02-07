#include "../atlas_memory/include/atlas_memory/workspace.hpp"
#include <iostream>

using namespace atlas_memory;

int main() {
  std::cout << "\n=== TEST: Stress Allocation ===\n";

  for (int i = 0; i < 2000; ++i) {
    Workspace ws(128, 128, 128, 8, 8);
    if (i % 250 == 0)
      std::cout << "Iteration " << i << "\n";
  }

  std::cout << "Stress PASSED\n";
}
