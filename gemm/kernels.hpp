#pragma once
#include "kernel_config.hpp"

namespace gemm {

// v0 — naive triple loop
void gemm_v0_naive(const float *A, const float *B, float *C,
                   const GemmConfig &cfg);

// v1 — loop reordered (i-k-j)
void gemm_v1_loop_reorder(const float *A, const float *B, float *C,
                          const GemmConfig &cfg);

// v2 — cache blocked
void gemm_v2_blocked(const float *A, const float *B, float *C,
                     const GemmConfig &cfg);

// v3 — scalar micro-tiled (MR x NR)
void gemm_v3_scalar_tile(const float *A, const float *B, float *C,
                         const GemmConfig &cfg);

// v4 — NEON microkernels
void gemm_v4_neon_4x4(const float *A, const float *B, float *C,
                      const GemmConfig &cfg);

void gemm_v4_neon_8x8(const float *A, const float *B, float *C,
                      const GemmConfig &cfg);

// v5 — packed + NEON + custom workspace
void gemm_v5_packed_neon(const float *A, const float *B, float *C,
                         const GemmConfig &cfg);

// v6 — parallel packed + NEON (std::thread based)
void gemm_v6_parallel(const float *A, const float *B, float *C,
                      const GemmConfig &cfg);

} // namespace gemm
