/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_FUSED_DEQUANTIZE_RECOMPOSE_HPP
#define MGARD_X_FUSED_DEQUANTIZE_RECOMPOSE_HPP

#include <cstdlib>
#include <type_traits>
#include <utility>

namespace mgard_x {

// Detects whether a compressor provides the fused DequantizeRecompose path
// (currently only HybridHierarchyCompressor).
template <DIM D, typename T, typename DeviceType, typename CompressorType,
          typename = void>
struct HasFusedDequantizeRecompose : std::false_type {};

template <DIM D, typename T, typename DeviceType, typename CompressorType>
struct HasFusedDequantizeRecompose<
    D, T, DeviceType, CompressorType,
    std::void_t<decltype(std::declval<CompressorType&>().DequantizeRecompose(
        std::declval<Array<D, T, DeviceType>&>(),
        std::declval<enum error_bound_type>(), std::declval<T>(),
        std::declval<T>(), std::declval<T>(), 0))>> : std::true_type {};

// Dequantize+Recompose step of the decompression pipelines: runs the fused
// single-pass implementation when the compressor provides one and the
// configuration supports it, the two-step path otherwise.
template <DIM D, typename T, typename DeviceType, typename CompressorType>
void DequantizeRecomposeStep(CompressorType& compressor,
                             Array<D, T, DeviceType>& decompressed_data,
                             enum error_bound_type ebtype, T tol, T s, T norm,
                             int queue_idx) {
  if constexpr (HasFusedDequantizeRecompose<D, T, DeviceType,
                                            CompressorType>::value) {
    // Escape hatch for A/B benchmarking and debugging: set
    // MGARD_X_DISABLE_FUSED_DEQUANTIZE_RECOMPOSE to force the unfused path.
    static const bool disable_fused =
        std::getenv("MGARD_X_DISABLE_FUSED_DEQUANTIZE_RECOMPOSE") != nullptr;
    if (!disable_fused && compressor.hybrid_quantizer.CanFuseQuantize(s)) {
      compressor.DequantizeRecompose(decompressed_data, ebtype, tol, s, norm,
                                     queue_idx);
      return;
    }
  }
  compressor.Dequantize(decompressed_data, ebtype, tol, s, norm, queue_idx);
  compressor.Recompose(decompressed_data, true, queue_idx);
}

}  // namespace mgard_x

#endif
