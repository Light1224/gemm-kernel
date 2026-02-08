#include "kernel_config.hpp"
#include <algorithm>
#include <arm_neon.h>

namespace gemm {

static constexpr index_t MR = 8;
static constexpr index_t NR = 8;

static inline void microkernel_8x8(const float *A, const float *B, float *C,
                                   index_t lda, index_t ldb, index_t ldc,
                                   index_t K) {
  float32x4_t c[8][2];

  for (int i = 0; i < 8; ++i) {
    c[i][0] = vdupq_n_f32(0.0f);
    c[i][1] = vdupq_n_f32(0.0f);
  }

  for (index_t k = 0; k < K; ++k) {

    float32x4_t b0 = vld1q_f32(B + k * ldb);
    float32x4_t b1 = vld1q_f32(B + k * ldb + 4);

    for (int i = 0; i < 8; ++i) {
      float32x4_t a = vdupq_n_f32(A[i * lda + k]);
      c[i][0] = vfmaq_f32(c[i][0], a, b0);
      c[i][1] = vfmaq_f32(c[i][1], a, b1);
    }
  }

  for (int i = 0; i < 8; ++i) {
    vst1q_f32(C + i * ldc, c[i][0]);
    vst1q_f32(C + i * ldc + 4, c[i][1]);
  }
}

void gemm_v4_neon_8x8(const float *A, const float *B, float *C,
                      const GemmConfig &cfg) {
  for (index_t i = 0; i < cfg.M; i += MR) {
    for (index_t j = 0; j < cfg.N; j += NR) {

      index_t im = std::min(MR, cfg.M - i);
      index_t jm = std::min(NR, cfg.N - j);

      if (im == 8 && jm == 8) {
        microkernel_8x8(A + i * cfg.lda, B + j, C + i * cfg.ldc + j, cfg.lda,
                        cfg.ldb, cfg.ldc, cfg.K);
      } else {
        for (index_t ii = 0; ii < im; ++ii)
          for (index_t jj = 0; jj < jm; ++jj) {
            float sum = 0;
            for (index_t k = 0; k < cfg.K; ++k)
              sum += A[(i + ii) * cfg.lda + k] * B[k * cfg.ldb + (j + jj)];
            C[(i + ii) * cfg.ldc + (j + jj)] = sum;
          }
      }
    }
  }
}

} // namespace gemm
