#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "../gemm/kernels.hpp"

using namespace gemm;

static float max_abs_diff(const std::vector<float> &a,
                          const std::vector<float> &b) {
  float m = 0.0f;
  for (size_t i = 0; i < a.size(); ++i)
    m = std::max(m, std::abs(a[i] - b[i]));
  return m;
}

static void fill_random(std::vector<float> &x) {
  static std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (float &v : x)
    v = dist(rng);
}

int main() {
  constexpr float eps = 1e-4f;

  std::vector<int> sizes = {8, 16, 32, 64, 96, 128, 192, 256};

  std::cout << "\n=== GEMM v6 Parallel Correctness Check ===\n";
  std::cout << std::setw(6) << "N" << std::setw(22) << "max |v6 - naive|"
            << "\n";
  std::cout << std::string(30, '-') << "\n";

  for (int n : sizes) {
    GemmConfig cfg;
    cfg.M = cfg.N = cfg.K = n;
    cfg.lda = cfg.ldb = cfg.ldc = n;

    std::vector<float> A(n * n);
    std::vector<float> B(n * n);
    std::vector<float> C_ref(n * n, 0.0f);
    std::vector<float> C_v6(n * n, 0.0f);

    fill_random(A);
    fill_random(B);

    // Reference
    gemm_v0_naive(A.data(), B.data(), C_ref.data(), cfg);

    // v6 parallel packed
    gemm_v6_parallel(A.data(), B.data(), C_v6.data(), cfg);

    float err = max_abs_diff(C_ref, C_v6);

    std::cout << std::setw(6) << n << std::setw(22) << err << "\n";

    if (err > eps) {
      std::cerr << "\n❌ GEMM v6 FAILED at N=" << n << " (error = " << err
                << ")\n";
      return 1;
    }
  }

  std::cout << "\n✅ GEMM v6 passed all correctness checks.\n";
  return 0;
}
