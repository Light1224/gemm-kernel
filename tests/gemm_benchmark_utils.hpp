#pragma once
#include <chrono>
#include <cstddef>

inline double seconds_now() {
  using clock = std::chrono::high_resolution_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

inline double gflops(std::size_t M, std::size_t N, std::size_t K,
                     double seconds) {
  double flops = 2.0 * M * N * K;
  return flops / seconds / 1e9;
}
