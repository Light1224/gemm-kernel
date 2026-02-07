#pragma once
#include <cstddef>
#include <iterator>

namespace gemm {

using index_t = std::size_t;

struct GemmConfig {
  index_t M;
  index_t N;
  index_t K;
  index_t lda;
  index_t ldb;
  index_t ldc;
};

} // namespace gemm
