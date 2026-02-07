#pragma once
#include <cstddef>

namespace atlas_memory {

class Workspace {
public:
  Workspace(std::size_t BM, std::size_t BN, std::size_t BK, std::size_t MR,
            std::size_t NR);

  ~Workspace();

  Workspace(const Workspace &) = delete;
  Workspace &operator=(const Workspace &) = delete;

  float *packA() noexcept;
  float *packB() noexcept;
  float *accum() noexcept;

  std::size_t packA_capacity() const noexcept;
  std::size_t packB_capacity() const noexcept;
  std::size_t accum_capacity() const noexcept;

  std::size_t total_capacity() const noexcept;

  void reset() noexcept;

private:
  void allocate(std::size_t bytes);
  void pre_touch();

private:
  void *base_{nullptr};
  std::size_t total_bytes_{0};

  std::size_t BM_, BN_, BK_, MR_, NR_;

  std::size_t a_bytes_{0};
  std::size_t b_bytes_{0};
  std::size_t accum_bytes_{0};

  float *a_ptr_{nullptr};
  float *b_ptr_{nullptr};
  float *accum_ptr_{nullptr};
};

} // namespace atlas_memory
