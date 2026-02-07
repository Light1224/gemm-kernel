#include "kernel_config.hpp"

namespace gemm {

void gemm_v0_naive(const float *A, const float *B, float *C,
                   const GemmConfig &cfg) {
  for (index_t i = 0; i < cfg.M; ++i) {
    for (index_t j = 0; j < cfg.N; ++j) {
      float sum = 0.0f;
      for (index_t k = 0; k < cfg.K; ++k) {
        sum += A[i * cfg.lda + k] * B[k * cfg.ldb + j];
      }
      C[i * cfg.ldc + j] = sum;
    }
  }
}

} // namespace gemm
