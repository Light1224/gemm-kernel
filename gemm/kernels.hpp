#pragma once
#include "kernel_config.hpp"

namespace gemm {

void gemm_v0_naive(const float *A, const float *B, float *C,
                   const GemmConfig &cfg);

void gemm_v1_loop_reorder(const float *A, const float *B, float *C,
                          const GemmConfig &cfg);

void gemm_v2_blocked(const float *A, const float *B, float *C,
                     const GemmConfig &cfg);

void gemm_v3_scalar_tile(const float *A, const float *B, float *C,
                         const GemmConfig &cfg);

} // namespace gemm
