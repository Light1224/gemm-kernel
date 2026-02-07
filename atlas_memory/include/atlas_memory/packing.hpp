#pragma once

namespace atlas_memory {

void pack_A(float *dst, const float *src, int rows, int cols, int ld);

void pack_B(float *dst, const float *src, int rows, int cols, int ld);

} // namespace atlas_memory
