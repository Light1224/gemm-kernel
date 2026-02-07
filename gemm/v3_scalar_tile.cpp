#include "kernel_config.hpp"
#include <algorithm>

namespace gemm {

static constexpr index_t BM = 64;
static constexpr index_t BN = 64;
static constexpr index_t BK = 64;
static constexpr index_t MR = 4;
static constexpr index_t NR = 4;

void gemm_v3_scalar_tile(const float *A, const float *B, float *C,
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

        for (index_t i = ii; i < i_max; i += MR) {
          for (index_t j = jj; j < j_max; j += NR) {

            float acc[MR][NR] = {};

            index_t im = std::min(MR, i_max - i);
            index_t jm = std::min(NR, j_max - j);

            for (index_t k = kk; k < k_max; ++k) {
              for (index_t ii2 = 0; ii2 < im; ++ii2) {
                float a = A[(i + ii2) * cfg.lda + k];
                const float *b = B + k * cfg.ldb + j;
                for (index_t jj2 = 0; jj2 < jm; ++jj2)
                  acc[ii2][jj2] += a * b[jj2];
              }
            }

            for (index_t ii2 = 0; ii2 < im; ++ii2) {
              float *c = C + (i + ii2) * cfg.ldc + j;
              for (index_t jj2 = 0; jj2 < jm; ++jj2)
                c[jj2] += acc[ii2][jj2];
            }
          }
        }
      }
    }
  }
}

} // namespace gemm
