#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "../atlas_memory/include/atlas_memory/packing.hpp"
#include "../atlas_memory/include/atlas_memory/workspace.hpp"

using namespace atlas_memory;

static void naive_gemm(const float *A, const float *B, float *C, size_t M,
                       size_t N, size_t K) {
  for (size_t i = 0; i < M; ++i)
    for (size_t j = 0; j < N; ++j)
      for (size_t k = 0; k < K; ++k)
        C[i * N + j] += A[i * K + k] * B[k * N + j];
}

int main() {
  std::cout << "\n=== TEST: Basic Blocked GEMM Using Atlas Memory ===\n";

  constexpr size_t M = 64;
  constexpr size_t N = 64;
  constexpr size_t K = 64;

  constexpr size_t BM = 64;
  constexpr size_t BN = 64;
  constexpr size_t BK = 64;

  constexpr size_t MR = 4;
  constexpr size_t NR = 4;

  std::vector<float> A(M * K);
  std::vector<float> B(K * N);
  std::vector<float> C(M * N, 0.0f);
  std::vector<float> C_ref(M * N, 0.0f);

  for (size_t i = 0; i < M * K; ++i)
    A[i] = static_cast<float>((i % 7) - 3);

  for (size_t i = 0; i < K * N; ++i)
    B[i] = static_cast<float>((i % 5) - 2);

  Workspace ws(BM, BN, BK, MR, NR);

  float *packA = ws.packA();
  float *packB = ws.packB();

  pack_A(A.data(), packA, M, K, K);
  pack_B(B.data(), packB, K, N, N);

  // Simple blocked compute using packed panels
  for (size_t i = 0; i < BM; ++i)
    for (size_t j = 0; j < BN; ++j)
      for (size_t k = 0; k < BK; ++k)
        C[i * N + j] += packA[i * BK + k] * packB[k * BN + j];

  naive_gemm(A.data(), B.data(), C_ref.data(), M, N, K);

  float max_error = 0.0f;

  for (size_t i = 0; i < M * N; ++i) {
    float diff = std::abs(C[i] - C_ref[i]);
    if (diff > max_error)
      max_error = diff;
  }

  std::cout << "Max error: " << max_error << "\n";

  assert(max_error < 1e-4f);

  std::cout << "Basic Blocked GEMM PASSED\n";

  return 0;
}
