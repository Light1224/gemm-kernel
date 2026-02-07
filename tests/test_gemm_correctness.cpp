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
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (float &v : x)
    v = dist(rng);
}

int main() {
  constexpr float eps = 1e-4f;

  std::vector<int> sizes = {8, 16, 32, 64, 96, 128, 192, 256};

  std::cout << "\n=== GEMM correctness check ===\n";
  std::cout << std::setw(6) << "N" << std::setw(15) << "v1 error"
            << std::setw(15) << "v2 error" << std::setw(15) << "v3 error"
            << "\n";
  std::cout << std::string(51, '-') << "\n";

  for (int n : sizes) {
    GemmConfig cfg;
    cfg.M = cfg.N = cfg.K = n;
    cfg.lda = cfg.ldb = cfg.ldc = n;

    std::vector<float> A(n * n);
    std::vector<float> B(n * n);
    std::vector<float> C_ref(n * n);
    std::vector<float> C1(n * n);
    std::vector<float> C2(n * n);
    std::vector<float> C3(n * n);

    fill_random(A);
    fill_random(B);

    gemm_v0_naive(A.data(), B.data(), C_ref.data(), cfg);
    gemm_v1_loop_reorder(A.data(), B.data(), C1.data(), cfg);
    gemm_v2_blocked(A.data(), B.data(), C2.data(), cfg);
    gemm_v3_scalar_tile(A.data(), B.data(), C3.data(), cfg);

    float e1 = max_abs_diff(C_ref, C1);
    float e2 = max_abs_diff(C_ref, C2);
    float e3 = max_abs_diff(C_ref, C3);

    std::cout << std::setw(6) << n << std::setw(15) << e1 << std::setw(15) << e2
              << std::setw(15) << e3 << "\n";

    if (e1 > eps || e2 > eps || e3 > eps) {
      std::cerr << "\n❌ GEMM correctness FAILED at N=" << n << "\n";
      return 1;
    }
  }

  std::cout << "\n✅ All GEMM versions passed correctness checks.\n";
  return 0;
}
