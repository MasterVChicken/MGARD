/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 */

#ifndef MGARD_X_UTILITIES_KERNEL_FUSION_H
#define MGARD_X_UTILITIES_KERNEL_FUSION_H

#include <cstdlib>

namespace mgard_x {

// The hybrid (BlockMGARD) local stage fuses the block-local decompose and
// recompose kernels with quantization and dequantization, so coefficients
// never round-trip through global memory as T. The unfused two-pass
// implementation is still there and still exercised by the test suite; these
// switches select it.
//
// This is purely a performance choice. On a deterministic backend the fused
// and unfused paths produce byte-identical output, and on every backend they
// reconstruct identically, so a file compressed either way decompresses either
// way and nothing about the choice is recorded in the file header. That is why
// it is an environment variable rather than a compression parameter: it does
// not belong in the Config that describes the data.
//
// Set either variable to any value to force the corresponding direction onto
// the separate-pass path:
//
//   MGARD_X_DISABLE_FUSED_DECOMPOSE_QUANTIZE    (compression)
//   MGARD_X_DISABLE_FUSED_DEQUANTIZE_RECOMPOSE  (decompression)
//
// Read on each call rather than cached in a function-local static, so an
// embedding application can flip them at runtime and so a single test process
// can exercise both paths. The cost is one getenv per Compress()/Decompress()
// call -- not per element -- which is nothing next to the work those do.

inline bool FuseDecomposeQuantizeEnabled() {
  return std::getenv("MGARD_X_DISABLE_FUSED_DECOMPOSE_QUANTIZE") == nullptr;
}

inline bool FuseDequantizeRecomposeEnabled() {
  return std::getenv("MGARD_X_DISABLE_FUSED_DEQUANTIZE_RECOMPOSE") == nullptr;
}

} // namespace mgard_x

#endif
