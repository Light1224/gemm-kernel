#include "../include/atlas_memory/workspace.hpp"
#include "../include/atlas_memory/config_m2.hpp"
#include "../include/atlas_memory/layout.hpp"
#include <cassert>
#include <cstdlib>
#include <cstring>

namespace atlas_memory {

Workspace::Workspace(std::size_t BM, std::size_t BN, std::size_t BK,
                     std::size_t MR, std::size_t NR)
    : BM_(BM), BN_(BN), BK_(BK), MR_(MR), NR_(NR) {
  auto layout = compute_layout(BM_, BN_, BK_, MR_, NR_);

  total_bytes_ = layout.total_bytes;
  a_bytes_ = layout.a.bytes;
  b_bytes_ = layout.b.bytes;
  accum_bytes_ = layout.accum.bytes;

  allocate(total_bytes_);
  pre_touch();

  a_ptr_ = reinterpret_cast<float *>(reinterpret_cast<char *>(base_) +
                                     layout.a.offset);

  b_ptr_ = reinterpret_cast<float *>(reinterpret_cast<char *>(base_) +
                                     layout.b.offset);

  accum_ptr_ = reinterpret_cast<float *>(reinterpret_cast<char *>(base_) +
                                         layout.accum.offset);
}

Workspace::~Workspace() {
  if (base_)
    std::free(base_);
}

void Workspace::allocate(std::size_t bytes) {
  int res = posix_memalign(&base_, config::SIMD_ALIGNMENT, bytes);
  assert(res == 0);
}

void Workspace::pre_touch() {
  char *ptr = reinterpret_cast<char *>(base_);
  for (std::size_t i = 0; i < total_bytes_; i += config::PAGE_SIZE)
    ptr[i] = 0;
}

float *Workspace::packA() noexcept { return a_ptr_; }
float *Workspace::packB() noexcept { return b_ptr_; }
float *Workspace::accum() noexcept { return accum_ptr_; }

std::size_t Workspace::packA_capacity() const noexcept { return a_bytes_; }
std::size_t Workspace::packB_capacity() const noexcept { return b_bytes_; }
std::size_t Workspace::accum_capacity() const noexcept { return accum_bytes_; }
std::size_t Workspace::total_capacity() const noexcept { return total_bytes_; }

void Workspace::reset() noexcept { std::memset(accum_ptr_, 0, accum_bytes_); }

} // namespace atlas_memory
