#pragma once
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "../gemm/kernels.hpp"

namespace bench {

using clock = std::chrono::high_resolution_clock;

inline double seconds(clock::time_point a, clock::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

inline void fill_matrix(std::vector<float> &x) {
  for (size_t i = 0; i < x.size(); ++i)
    x[i] = float((i * 1315423911u) & 0xFF) / 255.0f;
}

inline double gflops(size_t M, size_t N, size_t K, double t) {
  double flops = 2.0 * M * N * K;
  return flops / (t * 1e9);
}

// arithmetic intensity: FLOPs / bytes
inline double arithmetic_intensity(size_t M, size_t N, size_t K) {
  double flops = 2.0 * M * N * K;
  double bytes = 4.0 * (M * K + K * N + M * N);
  return flops / bytes;
}

inline void print_header(const std::string &name) {
  std::cout << "\n=== " << name << " ===\n";
  std::cout << std::setw(8) << "M" << std::setw(8) << "N" << std::setw(8) << "K"
            << std::setw(12) << "Time(s)" << std::setw(12) << "GFLOP/s"
            << std::setw(10) << "AI"
            << "\n";
  std::cout << std::string(58, '-') << "\n";
}

inline void print_scenario(const std::string &name) {
  std::cout << "\n-- " << name << " --\n";
}

inline void print_row(size_t M, size_t N, size_t K, double t) {
  std::cout << std::setw(8) << M << std::setw(8) << N << std::setw(8) << K
            << std::setw(12) << std::fixed << std::setprecision(6) << t
            << std::setw(12) << std::setprecision(2) << gflops(M, N, K, t)
            << std::setw(10) << std::setprecision(2)
            << arithmetic_intensity(M, N, K) << "\n";
}

} // namespace bench
