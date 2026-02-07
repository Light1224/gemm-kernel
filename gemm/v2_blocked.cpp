#include "kernel_config.hpp"
#include <algorithm>

namespace gemm {

static constexpr index_t BM = 64;
static constexpr index_t BN = 64;
static constexpr index_t BK = 64;

void gemm_v2_blocked(const float *A, const float *B, float *C,
                     const GemmConfig &cfg) {
  for (index_t i = 0; i < cfg.M; ++i)
    for (index_t j = 0; j < cfg.N; ++j)
      C[i * cfg.ldc + j] = 0.0f;

  for (index_t ii = 0; ii < cfg.M; ii += BM) {
    for (index_t kk = 0; kk < cfg.K; kk += BK) {
      for (index_t jj = 0; jj < cfg.N; jj += BN) {

        index_t i_max = std::min(ii + BM, cfg.M);
        index_t k_max = std::min(kk + BK, cfg.K);
        index_t j_max = std::min(jj + BN, cfg.N);

        for (index_t i = ii; i < i_max; ++i) {
          for (index_t k = kk; k < k_max; ++k) {
            float a = A[i * cfg.lda + k];
            const float *b = B + k * cfg.ldb;
            float *c = C + i * cfg.ldc;
            for (index_t j = jj; j < j_max; ++j)
              c[j] += a * b[j];
          }
        }
      }
    }
  }
}

} // namespace gemm
