#include "kernel_config.hpp"
#include <algorithm>
#include <arm_neon.h>

namespace gemm {

static constexpr index_t MR = 4;
static constexpr index_t NR = 4;

static inline void microkernel_4x4(const float *A, const float *B, float *C,
                                   index_t lda, index_t ldb, index_t ldc,
                                   index_t K) {
  float32x4_t c0 = vdupq_n_f32(0.0f);
  float32x4_t c1 = vdupq_n_f32(0.0f);
  float32x4_t c2 = vdupq_n_f32(0.0f);
  float32x4_t c3 = vdupq_n_f32(0.0f);

  for (index_t k = 0; k < K; ++k) {

    float32x4_t b = vld1q_f32(B + k * ldb);

    float32x4_t a0 = vdupq_n_f32(A[0 * lda + k]);
    float32x4_t a1 = vdupq_n_f32(A[1 * lda + k]);
    float32x4_t a2 = vdupq_n_f32(A[2 * lda + k]);
    float32x4_t a3 = vdupq_n_f32(A[3 * lda + k]);

    c0 = vfmaq_f32(c0, a0, b);
    c1 = vfmaq_f32(c1, a1, b);
    c2 = vfmaq_f32(c2, a2, b);
    c3 = vfmaq_f32(c3, a3, b);
  }

  vst1q_f32(C + 0 * ldc, c0);
  vst1q_f32(C + 1 * ldc, c1);
  vst1q_f32(C + 2 * ldc, c2);
  vst1q_f32(C + 3 * ldc, c3);
}

void gemm_v4_neon_4x4(const float *A, const float *B, float *C,
                      const GemmConfig &cfg) {
  for (index_t i = 0; i < cfg.M; i += MR) {
    for (index_t j = 0; j < cfg.N; j += NR) {

      index_t im = std::min(MR, cfg.M - i);
      index_t jm = std::min(NR, cfg.N - j);

      if (im == 4 && jm == 4) {
        microkernel_4x4(A + i * cfg.lda, B + j, C + i * cfg.ldc + j, cfg.lda,
                        cfg.ldb, cfg.ldc, cfg.K);
      } else {
        // fallback scalar
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
