#include "../atlas_memory/include/atlas_memory/config_m2.hpp"
#include "../atlas_memory/include/atlas_memory/packing.hpp"
#include "../atlas_memory/include/atlas_memory/workspace.hpp"
#include "kernel_config.hpp"

#include <algorithm>
#include <arm_neon.h>
#include <atomic>
#include <thread>
#include <vector>

namespace gemm {

using namespace atlas_memory;
using index_t = std::size_t;

// ================================================================
// 8x8 NEON microkernel (self-contained, correct)
// ================================================================
static inline void microkernel_8x8(const float *A, const float *B, float *C,
                                   index_t K, index_t Nb, index_t ldc) {
  float32x4_t c[8][2];

  // Load C tile
  for (int i = 0; i < 8; ++i) {
    c[i][0] = vld1q_f32(C + i * ldc);
    c[i][1] = vld1q_f32(C + i * ldc + 4);
  }

  for (index_t k = 0; k < K; ++k) {

    float32x4_t b0 = vld1q_f32(B + k * Nb);
    float32x4_t b1 = vld1q_f32(B + k * Nb + 4);

    for (int i = 0; i < 8; ++i) {
      float32x4_t a = vdupq_n_f32(A[i * K + k]);
      c[i][0] = vfmaq_f32(c[i][0], b0, a);
      c[i][1] = vfmaq_f32(c[i][1], b1, a);
    }
  }

  // Store back
  for (int i = 0; i < 8; ++i) {
    vst1q_f32(C + i * ldc, c[i][0]);
    vst1q_f32(C + i * ldc + 4, c[i][1]);
  }
}

// ================================================================
// Worker: dynamic tile scheduling
// ================================================================
static void worker(const float *A, const float *B, float *C,
                   const GemmConfig &cfg, std::atomic<index_t> &tile_counter,
                   index_t total_tiles) {
  constexpr index_t BM = config::DEFAULT_BM;
  constexpr index_t BN = config::DEFAULT_BN;
  constexpr index_t BK = config::DEFAULT_BK;
  constexpr index_t MR = config::MR;
  constexpr index_t NR = config::NR;

  Workspace ws(BM, BN, BK, MR, NR);

  while (true) {

    index_t tile_id = tile_counter.fetch_add(1);
    if (tile_id >= total_tiles)
      break;

    index_t tiles_per_row = (cfg.N + BN - 1) / BN;

    index_t ii = (tile_id / tiles_per_row) * BM;
    index_t jj = (tile_id % tiles_per_row) * BN;

    index_t Mb = std::min(BM, cfg.M - ii);
    index_t Nb = std::min(BN, cfg.N - jj);

    for (index_t kk = 0; kk < cfg.K; kk += BK) {

      index_t Kb = std::min(BK, cfg.K - kk);

      pack_A(ws.packA(), A + ii * cfg.lda + kk, Mb, Kb, cfg.lda);
      pack_B(ws.packB(), B + kk * cfg.ldb + jj, Kb, Nb, cfg.ldb);

      for (index_t i = 0; i < Mb; i += MR) {
        for (index_t j = 0; j < Nb; j += NR) {

          index_t mr = std::min(MR, Mb - i);
          index_t nr = std::min(NR, Nb - j);

          float *cptr = C + (ii + i) * cfg.ldc + (jj + j);
          const float *aptr = ws.packA() + i * Kb;
          const float *bptr = ws.packB() + j;

          if (mr == 8 && nr == 8) {
            microkernel_8x8(aptr, bptr, cptr, Kb, Nb, cfg.ldc);
          } else {
            // scalar cleanup
            for (index_t ii2 = 0; ii2 < mr; ++ii2) {
              for (index_t jj2 = 0; jj2 < nr; ++jj2) {

                float sum = 0.f;

                for (index_t k = 0; k < Kb; ++k) {
                  sum += aptr[ii2 * Kb + k] * ws.packB()[k * Nb + j + jj2];
                }

                cptr[ii2 * cfg.ldc + jj2] += sum;
              }
            }
          }
        }
      }
    }
  }
}

// ================================================================
// Public API
// ================================================================
void gemm_v6_parallel(const float *A, const float *B, float *C,
                      const GemmConfig &cfg) {
  unsigned num_threads = std::thread::hardware_concurrency();
  if (num_threads == 0)
    num_threads = 4;

  constexpr index_t BM = config::DEFAULT_BM;
  constexpr index_t BN = config::DEFAULT_BN;

  index_t tiles_m = (cfg.M + BM - 1) / BM;
  index_t tiles_n = (cfg.N + BN - 1) / BN;

  index_t total_tiles = tiles_m * tiles_n;

  std::atomic<index_t> tile_counter(0);

  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  for (unsigned t = 0; t < num_threads; ++t) {
    threads.emplace_back(worker, A, B, C, std::cref(cfg),
                         std::ref(tile_counter), total_tiles);
  }

  for (auto &th : threads)
    th.join();
}

} // namespace gemm
