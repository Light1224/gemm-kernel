#include "../include/atlas_memory/packing.hpp"
namespace atlas_memory {

void pack_B(float *dst, const float *src, int rows, int cols, int ld) {
  for (int k = 0; k < rows; ++k) {
    for (int j = 0; j < cols; ++j) {
      dst[k * cols + j] = src[k * ld + j];
    }
  }
}

} // namespace atlas_memory
