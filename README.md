# GEMM-Kernel: High-Performance Matrix Multiplication for ARM NEON

A progressive optimization study of GEMM (General Matrix Multiplication) kernels, demonstrating the journey from naive implementations to highly tuned parallel SIMD code optimized for Apple Silicon (M2).

## Overview

This repository implements the fundamental matrix multiplication operation `C = A × B` with progressively sophisticated optimizations:

- **Naive baselines** → **Cache-aware blocking** → **Register tiling** → **SIMD vectorization** → **Memory packing** → **Multi-threading**
- Target architecture: **ARM 64-bit NEON** (Apple Silicon M2)
- Language: **C++20** with ARM NEON intrinsics
- Build system: **CMake 3.20+**

## Project Structure

```
gemm-kernel/
├── gemm/                    # GEMM kernel implementations (v0-v7)
│   ├── kernels.hpp          # Kernel interface declarations
│   ├── kernel_config.hpp    # Configuration structures
│   ├── v0_naive.cpp         # Naive triple loop
│   ├── v1_loop_reorder.cpp  # Loop reordering (i-k-j)
│   ├── v2_blocked.cpp       # Cache blocking (64×64×64)
│   ├── v3_scalar_tile.cpp   # Scalar micro-tiling (4×4)
│   ├── v4_neon_4x4.cpp      # NEON 4×4 microkernel
│   ├── v4_neon_8x8.cpp      # NEON 8×8 microkernel
│   ├── v5_packed.cpp        # Packed + NEON (8×8)
│   ├── v6_parallel.cpp      # Multi-threaded packed NEON
│   └── v7_tuned.cpp         # Future: Auto-tuned parameters
│
├── atlas_memory/            # Memory management library
│   ├── include/atlas_memory/
│   │   ├── config_m2.hpp    # M2-specific config (BM=BN=BK=256)
│   │   ├── workspace.hpp    # Pre-allocated aligned buffers
│   │   ├── packing.hpp      # Matrix packing functions
│   │   └── layout.hpp       # Memory layout utilities
│   └── src/                 # Implementation files
│
├── tests/                   # Tests and per-version benchmarks
│   ├── test_gemm_correctness.cpp      # Numerical correctness
│   ├── test_vs_blas.cpp               # OpenBLAS comparison
│   ├── test_packing_correctness.cpp   # Packing validation
│   ├── test_layout_and_alignment.cpp  # Memory layout tests
│   ├── benchmark_gemm_v0.cpp          # v0 performance
│   ├── benchmark_gemm_v1.cpp          # v1 performance
│   └── ...                             # v2-v6 benchmarks
│
├── benchmarks/              # Advanced benchmarking scripts
│   ├── benchmark_gemm.cpp          # Comprehensive benchmark suite
│   ├── benchmark_block_sizes.cpp   # Block size optimization
│   ├── benchmark_packing.cpp       # Packing overhead analysis
│   ├── benchmark_scaling.cpp       # Multi-threaded scaling
│   └── plot_results.py             # (Placeholder) Result visualization
│
├── profiling/               # Performance profiling documentation
│   ├── perf_notes.md        # Linux perf usage
│   ├── flamegraph_notes.md  # Flamegraph generation
│   ├── tlb_notes.md         # TLB profiling
│   └── roofline_plot.py     # (Placeholder) Roofline plotting
│
├── docs/                    # Design documentation
│   ├── optimisation_phases.md     # Optimization strategy
│   ├── m2_architecture_notes.md   # M2 architecture specifics
│   ├── atlas_memory_design.md     # Memory library design
│   ├── microkernel_design.md      # Microkernel implementation
│   ├── blocking_strategy.md       # Cache blocking approach
│   └── roofline_analysis.md       # Roofline model analysis
│
├── ci/                      # CI/CD configuration
│   └── performance_regression.yml  # (Placeholder) CI config
│
└── CMakeLists.txt           # Build configuration
```

## GEMM Kernel Versions

### Version Progression

| Version | Name | Optimization | Performance Target | Key Techniques |
|---------|------|--------------|-------------------|----------------|
| **v0** | Naive | Triple loop (i-j-k) | Baseline (~0.5 GFLOPS) | Basic matrix indexing |
| **v1** | Loop Reorder | Better loop order (i-k-j) | ~2× improvement | Improved memory access pattern |
| **v2** | Blocked | Cache blocking | ~5-8× improvement | 64×64×64 blocks, L1/L2 cache reuse |
| **v3** | Scalar Tile | Register micro-tiling | ~10-15× improvement | 4×4 tiles with register accumulation |
| **v4** | NEON | SIMD vectorization | ~30-50× improvement | ARM NEON intrinsics (4×4 or 8×8 tiles) |
| **v5** | Packed | Memory packing + NEON | ~50-80× improvement | Contiguous memory layout, 8×8 NEON |
| **v6** | Parallel | Multi-threaded | ~200-400× improvement | `std::thread` parallelism, work stealing |
| **v7** | Tuned | Auto-tuned parameters | (Not yet implemented) | Runtime parameter optimization |

### Version Details

#### v0: Naive Implementation
```cpp
for (i = 0; i < M; ++i)
  for (j = 0; j < N; ++j)
    for (k = 0; k < K; ++k)
      C[i][j] += A[i][k] * B[k][j];
```
- **Issues**: Poor cache locality, strided B access
- **GFLOPS**: ~0.5 (baseline)

#### v1: Loop Reordering
```cpp
for (i = 0; i < M; ++i)
  for (k = 0; k < K; ++k)
    for (j = 0; j < N; ++j)
      C[i][j] += A[i][k] * B[k][j];
```
- **Improvement**: Sequential access to B[k][j]
- **GFLOPS**: ~1-2× faster

#### v2: Cache Blocking
- **Block sizes**: BM=64, BN=64, BK=64
- **Benefit**: Fits working set in L1/L2 cache
- **Loop structure**: Blocked outer loops (ii, kk, jj) + inner loops (i, k, j)

#### v3: Scalar Micro-Tiling
- **Tile size**: MR=4, NR=4
- **Benefit**: Accumulates 16 values in registers
- **Technique**: Manual unrolling and register reuse

#### v4: NEON Intrinsics
- **4×4 variant**: `vfmaq_f32` on 4 float32x4_t accumulators
- **8×8 variant**: 16 vector registers (8×2), processes 64 elements per microkernel
- **Key instruction**: `vfmaq_f32` (fused multiply-add)

```cpp
// 8×8 NEON microkernel example
float32x4_t c[8][2];  // 8 rows × 2 vectors (8 columns)
for (k = 0; k < K; ++k) {
  float32x4_t b0 = vld1q_f32(B + k*ldb);
  float32x4_t b1 = vld1q_f32(B + k*ldb + 4);
  for (i = 0; i < 8; ++i) {
    float32x4_t a = vdupq_n_f32(A[i*lda + k]);
    c[i][0] = vfmaq_f32(c[i][0], a, b0);
    c[i][1] = vfmaq_f32(c[i][1], a, b1);
  }
}
```

#### v5: Packed Layout
- **Packing**: Converts A and B to contiguous panel layout
- **Workspace**: Pre-allocated aligned buffers (128-byte alignment)
- **Benefit**: Eliminates strided access, maximizes cache line utilization

#### v6: Parallelization
- **Threading**: `std::thread` pool with dynamic work stealing
- **Granularity**: Distributes BM×BN blocks across threads
- **Scalability**: Near-linear scaling up to 8-10 cores

## Atlas Memory Library

The **atlas_memory** library provides optimized memory management for GEMM operations:

### Components

1. **Workspace Class** (`workspace.hpp`)
   - Pre-allocates aligned memory buffers
   - Methods:
     - `packA()`: Returns buffer for packed A blocks
     - `packB()`: Returns buffer for packed B blocks
     - `accum()`: Returns accumulation buffer
   - Alignment: 128 bytes (SIMD), 64 bytes (cache line)

2. **Packing Functions** (`packing.hpp`)
   - `pack_A()`: Converts A matrix to packed panel layout
   - `pack_B()`: Converts B matrix to packed panel layout
   - Layout: Contiguous MR×K and K×NR panels

3. **Configuration** (`config_m2.hpp`)
   ```cpp
   constexpr size_t CACHE_LINE = 64;
   constexpr size_t SIMD_ALIGNMENT = 128;
   constexpr size_t DEFAULT_BM = 256;  // L2 cache blocking
   constexpr size_t DEFAULT_BN = 256;
   constexpr size_t DEFAULT_BK = 256;
   constexpr size_t MR = 8;  // Microkernel rows
   constexpr size_t NR = 8;  // Microkernel columns
   ```

4. **Layout Utilities** (`layout.hpp`)
   - Stride calculations
   - Alignment helpers
   - Padding logic

## Building the Project

### Prerequisites
- **OS**: macOS (Apple Silicon) or ARM Linux
- **Compiler**: Clang 14+ or GCC 11+ with ARM NEON support
- **CMake**: 3.20 or later
- **OpenBLAS**: Required for `test_vs_blas` (install via `brew install openblas` on macOS)

### Build Instructions

```bash
# Clone the repository
git clone https://github.com/Light1224/gemm-kernel.git
cd gemm-kernel

# Create build directory
mkdir -p build && cd build

# Configure (Release mode with M2 optimizations)
cmake -DCMAKE_BUILD_TYPE=Release ..

# Build all targets
cmake --build . -j$(nproc)

# Run tests
ctest --output-on-failure
```

### Build Modes

- **Release** (default): `-O3 -mcpu=apple-m2`
- **Debug**: `-O1 -g -fsanitize=address,undefined`

### Build Targets

```bash
# Build specific targets
cmake --build . --target gemm_kernels        # GEMM library only
cmake --build . --target atlas_memory        # Memory library only
cmake --build . --target benchmark_gemm_v6   # v6 benchmark
cmake --build . --target test_gemm_correctness  # Correctness test
```

## Running Tests and Benchmarks

### Correctness Tests

```bash
# Run all tests
ctest

# Run specific correctness tests
./test_gemm_correctness
./test_packing_correctness
./test_vs_blas  # Compare against OpenBLAS
```

### Performance Benchmarks

```bash
# Single-version benchmarks
./benchmark_gemm_v0  # Baseline (naive)
./benchmark_gemm_v4_8x8  # NEON 8×8
./benchmark_gemm_v6  # Parallel packed

# Comprehensive benchmark suite
./benchmark_gemm

# Specialized benchmarks
./benchmark_block_sizes  # Find optimal BM/BN/BK
./benchmark_packing      # Measure packing overhead
./benchmark_scaling      # Multi-threading scaling
```

### Benchmark Output

```
=== GEMM v6 Parallel Packed + NEON ===
Configuration                 M       N       K      Time(s)    GFLOPS
Square matrices
                             8       8       8    0.000001       1.024
                            16      16      16    0.000002       4.096
                            ...
                          1024    1024    1024    0.015234     140.8
```

**Metrics**:
- **Time**: Wall-clock time for single `C = A × B` operation
- **GFLOPS**: Billion floating-point operations per second
  - Formula: `(2 × M × N × K) / (time × 10^9)`

## Testing Infrastructure

### Test Categories

1. **Correctness Tests** (`test_*.cpp`)
   - Validate against naive reference implementation
   - Test edge cases (non-multiple-of-block sizes)
   - Compare against OpenBLAS

2. **Memory Tests**
   - `test_layout_and_alignment.cpp`: Alignment validation
   - `test_stress_allocation.cpp`: Memory stress testing
   - `test_reset_behavior.cpp`: Workspace reset correctness

3. **Per-Version Benchmarks** (`benchmark_gemm_v*.cpp`)
   - Consistent benchmark harness across versions
   - Tests multiple matrix sizes and shapes:
     - Square: 8×8 to 1024×1024
     - Tall-skinny: High M, low N (B reuse)
     - Wide: Low M, high N (A reuse)
     - Large-K: Fixed M/N, varying K

## Performance Analysis Tools

### Profiling Scripts (Placeholders)

- **`profiling/roofline_plot.py`**: Generate roofline model plots
- **`benchmarks/plot_results.py`**: Visualize benchmark results

### Profiling Documentation

The `profiling/` directory contains documentation for:

1. **`perf_notes.md`**: Using Linux `perf` for CPU profiling
2. **`flamegraph_notes.md`**: Generating flamegraphs
3. **`tlb_notes.md`**: TLB (Translation Lookaside Buffer) profiling

### Example Profiling Workflow

```bash
# Linux perf profiling
perf record -g ./benchmark_gemm_v6
perf report

# Generate flamegraph (Linux)
perf script | stackcollapse-perf.pl | flamegraph.pl > flamegraph.svg

# macOS Instruments
instruments -t "Time Profiler" -D trace.trace ./benchmark_gemm_v6
```

## Design Documentation

The `docs/` directory contains detailed design notes:

1. **`optimisation_phases.md`**: Step-by-step optimization strategy
2. **`m2_architecture_notes.md`**: Apple M2 architecture specifics
   - NEON vector units
   - Cache hierarchy (L1: 128KB data, L2: 12MB)
   - Memory bandwidth
3. **`atlas_memory_design.md`**: Memory library architecture
4. **`microkernel_design.md`**: NEON microkernel implementation details
5. **`blocking_strategy.md`**: Cache blocking parameter selection
6. **`roofline_analysis.md`**: Roofline model performance analysis

## Expected Performance (Apple M2)

| Version | Matrix Size | GFLOPS | Speedup vs v0 |
|---------|-------------|--------|---------------|
| v0 (naive) | 1024×1024 | ~0.5 | 1× (baseline) |
| v1 (reorder) | 1024×1024 | ~1.0 | 2× |
| v2 (blocked) | 1024×1024 | ~4.0 | 8× |
| v3 (scalar tile) | 1024×1024 | ~8.0 | 16× |
| v4 (NEON 8×8) | 1024×1024 | ~25.0 | 50× |
| v5 (packed) | 1024×1024 | ~40.0 | 80× |
| v6 (parallel, 8 cores) | 1024×1024 | ~200+ | 400× |
| OpenBLAS (reference) | 1024×1024 | ~300+ | - |

*Note: Actual performance varies based on system load, memory bandwidth, and thermal throttling.*

## Hardware Requirements

### Target: Apple Silicon M2

- **Architecture**: ARMv8.5-A (64-bit)
- **SIMD**: ARM NEON (128-bit vectors, 4×float32 per instruction)
- **FMA**: Fused multiply-add (`vfmaq_f32`)
- **Cores**: 8 (4 performance + 4 efficiency)
- **L1 Cache**: 128 KB data + 192 KB instruction per P-core
- **L2 Cache**: 12 MB shared (performance cores)
- **Memory**: Unified memory (up to 24 GB, ~100 GB/s bandwidth)

### Other ARM Platforms

Should work on any ARMv8-A platform with NEON support:
- Raspberry Pi 4/5
- AWS Graviton instances
- NVIDIA Jetson series
- Android devices (requires Android NDK)

## CI/CD (Placeholder)

The `ci/performance_regression.yml` file is a placeholder for future GitHub Actions workflows:

```yaml
# Planned CI features:
# - Build verification (Debug + Release)
# - Correctness tests (ctest)
# - Performance regression detection
# - Benchmark result archiving
```

## Contributing

This is a learning/demonstration repository. Key areas for contribution:

1. **v7_tuned.cpp**: Implement auto-tuning for BM/BN/BK/MR/NR
2. **Profiling scripts**: Complete `roofline_plot.py` and `plot_results.py`
3. **Documentation**: Expand design docs with diagrams and analysis
4. **Portability**: Test and optimize for other ARM platforms
5. **Advanced techniques**:
   - SVE (Scalable Vector Extension) support
   - GPU offloading (Metal/OpenCL)
   - Mixed-precision (FP16/BF16)

## License

See `LICENSE` file for details.

## References

- **BLIS Framework**: [https://github.com/flame/blis](https://github.com/flame/blis)
- **Goto Algorithm**: Kazushige Goto and Robert van de Geijn. *High-performance implementation of the level-3 BLAS*. ACM TOMS, 2008.
- **ARM NEON**: [ARM NEON Intrinsics Reference](https://developer.arm.com/architectures/instruction-sets/simd-isas/neon/intrinsics)
- **Apple M2**: [Apple Silicon Documentation](https://developer.apple.com/documentation/apple-silicon)

---

**Author**: Light1224  
**Repository**: [https://github.com/Light1224/gemm-kernel](https://github.com/Light1224/gemm-kernel)