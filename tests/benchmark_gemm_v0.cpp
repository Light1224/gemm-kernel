#include "benchmark_common.hpp"

int main() {
  using namespace gemm;
  using namespace bench;

  print_header("GEMM v0 naive");

  auto run = [&](size_t M, size_t N, size_t K) {
    std::vector<float> A(M * K), B(K * N), C(M * N);
    fill_matrix(A);
    fill_matrix(B);

    GemmConfig cfg{M, N, K, K, N, N};

    gemm_v0_naive(A.data(), B.data(), C.data(), cfg); // warm-up

    auto t0 = clock::now();
    gemm_v0_naive(A.data(), B.data(), C.data(), cfg);
    auto t1 = clock::now();

    print_row(M, N, K, seconds(t0, t1));
  };

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
}
