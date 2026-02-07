#include "kernel_config.hpp"

namespace gemm {

void gemm_v1_loop_reorder(const float *A, const float *B, float *C,
                          const GemmConfig &cfg) {
  for (index_t i = 0; i < cfg.M; ++i)
    for (index_t j = 0; j < cfg.N; ++j)
      C[i * cfg.ldc + j] = 0.0f;

  for (index_t i = 0; i < cfg.M; ++i) {
    for (index_t k = 0; k < cfg.K; ++k) {
      float a = A[i * cfg.lda + k];
      const float *b = B + k * cfg.ldb;
      float *c = C + i * cfg.ldc;
      for (index_t j = 0; j < cfg.N; ++j)
        c[j] += a * b[j];
    }
  }
}

} // namespace gemm
