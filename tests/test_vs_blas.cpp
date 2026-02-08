#include "benchmark_common.hpp"
#include <algorithm>
#include <cblas.h>
#include <cstdlib>

namespace bench {

// Optional: compute max absolute difference between two matrices
inline float max_abs_diff(const std::vector<float> &X,
                          const std::vector<float> &Y) {
  float max_diff = 0.0f;
  for (size_t i = 0; i < X.size(); ++i)
    max_diff = std::max(max_diff, std::abs(X[i] - Y[i]));
  return max_diff;
}

} // namespace bench

int main() {
  using namespace bench;

  // Force OpenBLAS to single-thread for consistent results
  setenv("OPENBLAS_NUM_THREADS", "1", 1);

  print_header("OpenBLAS SGEMM");

  auto run = [&](size_t M, size_t N, size_t K) {
    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C(M * N);

    fill_matrix(A);
    fill_matrix(B);
    std::fill(C.begin(), C.end(), 0.0f);

    // Warmup
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f,
                A.data(), K, B.data(), N, 0.0f, C.data(), N);

    std::fill(C.begin(), C.end(), 0.0f);

    // Measure time
    auto t0 = clock::now();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f,
                A.data(), K, B.data(), N, 0.0f, C.data(), N);
    auto t1 = clock::now();

    print_row(M, N, K, seconds(t0, t1));
  };

  // Same loops as GEMM v6
  print_scenario("Square matrices");
  for (size_t n = 8; n <= 1024; n *= 2)
    run(n, n, n);

  print_scenario("Tall-skinny (B reuse)");
  for (size_t m : {128, 256, 512, 1024})
    run(m, 64, 1024);

  print_scenario("Wide (A reuse)");
  for (size_t n : {128, 256, 512, 1024})
    run(64, n, 1024);

  print_scenario("Large-K regime");
  for (size_t k : {128, 256, 512, 1024, 2048})
    run(256, 256, k);

  print_scenario("Cache-stress");
  for (size_t n : {384, 512, 768})
    run(n, n, n);

  return 0;
}
