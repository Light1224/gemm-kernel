#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#include "../atlas_memory/include/atlas_memory/packing.hpp"
#include "../atlas_memory/include/atlas_memory/workspace.hpp"

using namespace atlas_memory;
using clock_type = std::chrono::high_resolution_clock;

// ------------------------------------------------------------
// Naive GEMM
// ------------------------------------------------------------
static void naive_gemm(const float *A, const float *B, float *C, size_t M,
                       size_t N, size_t K) {
  for (size_t i = 0; i < M; ++i)
    for (size_t j = 0; j < N; ++j)
      for (size_t k = 0; k < K; ++k)
        C[i * N + j] += A[i * K + k] * B[k * N + j];
}

// ------------------------------------------------------------
// Blocked Packed GEMM (edge-safe)
// ------------------------------------------------------------
static void blocked_packed_gemm(const float *A, const float *B, float *C,
                                size_t M, size_t N, size_t K, size_t BM,
                                size_t BN, size_t BK, size_t MR, size_t NR) {

  Workspace ws(BM, BN, BK, MR, NR);
  float *packA = ws.packA();
  float *packB = ws.packB();

  for (size_t ii = 0; ii < M; ii += BM) {
    size_t curBM = std::min(BM, M - ii);

    for (size_t jj = 0; jj < N; jj += BN) {
      size_t curBN = std::min(BN, N - jj);

      for (size_t kk = 0; kk < K; kk += BK) {
        size_t curBK = std::min(BK, K - kk);

        pack_A(packA, A + ii * K + kk, (int)curBM, (int)curBK, (int)K);

        pack_B(packB, B + kk * N + jj, (int)curBK, (int)curBN, (int)N);

        for (size_t i = 0; i < curBM; ++i)
          for (size_t j = 0; j < curBN; ++j)
            for (size_t k = 0; k < curBK; ++k)
              C[(ii + i) * N + (jj + j)] +=
                  packA[i * curBK + k] * packB[k * curBN + j];
      }
    }
  }
}

// ------------------------------------------------------------
// Run single benchmark
// ------------------------------------------------------------
void run_test(size_t M, size_t N, size_t K) {

  constexpr size_t BM = 64;
  constexpr size_t BN = 64;
  constexpr size_t BK = 32;

  constexpr size_t MR = 4;
  constexpr size_t NR = 4;

  constexpr int ITERS = 3;

  std::vector<float> A(M * K);
  std::vector<float> B(K * N);
  std::vector<float> C1(M * N, 0.0f);
  std::vector<float> C2(M * N, 0.0f);

  // Deterministic fill
  for (size_t i = 0; i < A.size(); ++i)
    A[i] = static_cast<float>((i % 17) * 0.1f);

  for (size_t i = 0; i < B.size(); ++i)
    B[i] = static_cast<float>((i % 13) * 0.1f);

  double best_naive = 1e9;
  double best_packed = 1e9;

  // ---------------- Naive ----------------
  for (int i = 0; i < ITERS; ++i) {
    std::fill(C1.begin(), C1.end(), 0.0f);

    auto t1 = clock_type::now();
    naive_gemm(A.data(), B.data(), C1.data(), M, N, K);
    auto t2 = clock_type::now();

    best_naive =
        std::min(best_naive, std::chrono::duration<double>(t2 - t1).count());
  }

  // ---------------- Packed ----------------
  for (int i = 0; i < ITERS; ++i) {
    std::fill(C2.begin(), C2.end(), 0.0f);

    auto t1 = clock_type::now();
    blocked_packed_gemm(A.data(), B.data(), C2.data(), M, N, K, BM, BN, BK, MR,
                        NR);
    auto t2 = clock_type::now();

    best_packed =
        std::min(best_packed, std::chrono::duration<double>(t2 - t1).count());
  }

  // ---------------- Correctness (FIXED) ----------------
  double max_error = 0.0;

  for (size_t i = 0; i < M * N; ++i) {
    double diff =
        std::abs(static_cast<double>(C1[i]) - static_cast<double>(C2[i]));

    max_error = std::max(max_error, diff);
  }

  // ---------------- Metrics ----------------
  double flops = 2.0 * M * N * K;
  double gflops_naive = flops / best_naive / 1e9;
  double gflops_packed = flops / best_packed / 1e9;

  double speedup = best_naive / best_packed;

  std::cout << std::setw(6) << M << "x" << std::setw(6) << N << "x"
            << std::setw(6) << K << " | " << std::setw(8) << std::fixed
            << std::setprecision(2) << gflops_naive << " | " << std::setw(8)
            << gflops_packed << " | " << std::setw(6) << speedup << "x | "
            << (max_error < 1e-4 ? "OK" : "ERR") << "\n";
}

// ------------------------------------------------------------

int main() {

  std::cout << "\n=== PACKING vs NAIVE SCALING TEST ===\n\n";

  std::cout << "   Size        |  Naive GF  | Packed GF | Speed  | Check\n";
  std::cout
      << "--------------------------------------------------------------\n";

  std::vector<size_t> sizes = {8,   16,  32,  64,  96,  128, 192,
                               256, 320, 400, 512, 640, 1024};

  for (auto s : sizes)
    run_test(s, s, s);

  std::cout << "\n(Observe when speedup crosses 1.0x)\n\n";

  return 0;
}
