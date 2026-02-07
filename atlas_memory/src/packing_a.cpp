#include "../include/atlas_memory/packing.hpp"
namespace atlas_memory {

void pack_A(float *dst, const float *src, int rows, int cols, int ld) {
  for (int i = 0; i < rows; ++i) {
    const float *s = src + i * ld;
    float *d = dst + i * cols;
    for (int k = 0; k < cols; ++k)
      d[k] = s[k];
  }
}

} // namespace atlas_memory
