#include "../atlas_memory/include/atlas_memory/workspace.hpp"

#include <cassert>
#include <iostream>

using namespace atlas_memory;

int main() {
  std::cout << "\n=== TEST: Reset Behavior ===\n";

  Workspace ws(128, 128, 128, 8, 8);

  float *acc = ws.accum();
  std::size_t elements = ws.accum_capacity() / sizeof(float);

  for (std::size_t i = 0; i < elements; ++i)
    acc[i] = 123.456f;

  ws.reset();

  for (std::size_t i = 0; i < elements; ++i)
    assert(acc[i] == 0.0f);

  std::cout << "Reset PASSED\n";
}
