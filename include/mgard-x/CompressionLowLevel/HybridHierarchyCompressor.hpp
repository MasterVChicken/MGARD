/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#include "../Config/Config.h"
#include "../Hierarchy/Hierarchy.h"
#include "../RuntimeX/RuntimeX.h"
#include "../Utilities/Types.h"
#include "CompressorCache.hpp"
#include "HybridHierarchyCompressor.h"

#ifndef MGARD_X_HYBRID_HIERARCHY_COMPRESSOR_HPP
#define MGARD_X_HYBRID_HIERARCHY_COMPRESSOR_HPP

namespace mgard_x {

static bool debug_print_compression = true;

template <DIM D, typename T, typename DeviceType>
HybridHierarchyCompressor<D, T, DeviceType>::HybridHierarchyCompressor()
    : initialized(false) {}

template <DIM D, typename T, typename DeviceType>
HybridHierarchyCompressor<D, T, DeviceType>::HybridHierarchyCompressor(
    Hierarchy<D, T, DeviceType> &hierarchy, Config config)
    : initialized(true),
      hierarchy(&hierarchy),
      config(config),
      local_refactor(hierarchy, config),
      lossless_compressor(calculate_padded_size(hierarchy), config),
      local_quantizer(hierarchy, config) {
  norm_array = Array<1, T, DeviceType>({1});
  norm_tmp_array = Array<1, T, DeviceType>({hierarchy.total_num_elems()},
                                           (T *)local_refactor.w_array.data());
                                          
  SIZE total_num_elems_1D = calculate_padded_size(hierarchy);
  
  // Reuse workspace. Warning:
  // if space is enough
  if (sizeof(QUANTIZED_INT) <= sizeof(T)) {
    local_quantized_array = Array<1, QUANTIZED_INT, DeviceType>(
        {total_num_elems_1D}, (QUANTIZED_INT *)local_refactor.w_array.data());
  } else {
    // if space is not enough
    local_quantized_array =
        Array<1, QUANTIZED_INT, DeviceType>({total_num_elems_1D});
  }
}

template <DIM D, typename T, typename DeviceType>
void HybridHierarchyCompressor<D, T, DeviceType>::Adapt(
    Hierarchy<D, T, DeviceType> &hierarchy, Config config, int queue_idx) {
  this->initialized = true;
  this->hierarchy = &hierarchy;
  this->config = config;
  local_refactor.Adapt(hierarchy, config, queue_idx);
  lossless_compressor.Adapt(calculate_padded_size(hierarchy), config, queue_idx);

  local_quantizer.Adapt(hierarchy, config, queue_idx);
  norm_array.resize({1}, queue_idx);
  norm_tmp_array = Array<1, T, DeviceType>({hierarchy.total_num_elems()},
                                           (T *)local_refactor.w_array.data());

  SIZE total_num_elems_1D = calculate_padded_size(hierarchy);
  // log::info("total_num_elems_1D: " + std::to_string(total_num_elems_1D));
  
  // Reuse workspace. Warning:
  // if space is enough
  if (sizeof(QUANTIZED_INT) <= sizeof(T)) {
    local_quantized_array = Array<1, QUANTIZED_INT, DeviceType>(
        {total_num_elems_1D}, (QUANTIZED_INT *)local_refactor.w_array.data());
  } else {
    // if space is not enough
    local_quantized_array.resize({total_num_elems_1D}, queue_idx);
  }
}

// Need further calculation
template <DIM D, typename T, typename DeviceType>
size_t HybridHierarchyCompressor<D, T, DeviceType>::EstimateMemoryFootprint(
    std::vector<SIZE> shape, Config config) {
  Hierarchy<D, T, DeviceType> hierarchy;
  hierarchy.EstimateMemoryFootprint(shape);
  size_t size = 0;
  size += BlockLocalHierarchyDataRefactorType::EstimateMemoryFootprint(shape);
  // log::info(
  //     "Data refactor space: " +
  //     std::to_string(
  //         (double)(BlockLocalHierarchyDataRefactorType::EstimateMemoryFootprint(
  //             shape)) /
  //         1e9) +
  //     " GB");
  size += LocalQuantizerType::EstimateMemoryFootprint(shape);
  // log::info(
  //     "Quantizer space: " +
  //     std::to_string(
  //         (double)(LocalQuantizerType::EstimateMemoryFootprint(shape)) / 1e9) +
  //     " GB");
  size += LosslessCompressorType::EstimateMemoryFootprint(
      calculate_padded_size(hierarchy), config);
  // log::info(
  //     "Lossless space: " +
  //     std::to_string((double)(LosslessCompressorType::EstimateMemoryFootprint(
  //                        hierarchy.total_num_elems(), config)) /
  //                    1e9) +
  //     " GB");
  size += sizeof(T);
  if (sizeof(QUANTIZED_INT) > sizeof(T)) {
    size += sizeof(T) * calculate_padded_size(hierarchy);
    size += sizeof(QUANTIZED_INT) * calculate_padded_size(hierarchy);
  }
  return size;
}

template <DIM D, typename T, typename DeviceType>
void HybridHierarchyCompressor<D, T, DeviceType>::CalculateNorm(
    Array<D, T, DeviceType> &original_data, enum error_bound_type ebtype, T s,
    T &norm, int queue_idx) {
  if (ebtype == error_bound_type::REL) {
    norm =
        norm_calculator(original_data, SubArray(norm_tmp_array),
                        SubArray(norm_array), s, config.normalize_coordinates);
  }
}

template <DIM D, typename T, typename DeviceType>
void HybridHierarchyCompressor<D, T, DeviceType>::Decompose(
    Array<D, T, DeviceType> &original_data, int queue_idx) {
  // hybrid_refactor.Decompose(original_data, decomposed_array, queue_idx);
  local_refactor.Decompose(SubArray(original_data), queue_idx);
}

template <DIM D, typename T, typename DeviceType>
void HybridHierarchyCompressor<D, T, DeviceType>::Quantize(
    Array<D, T, DeviceType> &original_data, enum error_bound_type ebtype, T tol,
    T s, T norm, int queue_idx) {
  std::vector<SIZE> original_shape =
      hierarchy->level_shape(hierarchy->l_target());
  SIZE total_num_elems_1D = 1;
  for (DIM d = 0; d < original_shape.size(); d++) {
    SIZE cur_dim_shape = ((original_shape[d] - 1) / 8 + 1) * 8;
    total_num_elems_1D *= cur_dim_shape;
  }

  SubArray<1, T, DeviceType> data_subarray({total_num_elems_1D},
                                           original_data.data());
  local_quantizer.Quantize(data_subarray, ebtype, tol, s, norm,
                           local_quantized_array, lossless_compressor,
                           queue_idx);
}

template <DIM D, typename T, typename DeviceType>
void HybridHierarchyCompressor<D, T, DeviceType>::LosslessCompress(
    Array<1, Byte, DeviceType> &compressed_data, int queue_idx) {
    lossless_compressor.Compress(local_quantized_array, compressed_data,
                                 queue_idx);
}

template <DIM D, typename T, typename DeviceType>
void HybridHierarchyCompressor<D, T, DeviceType>::Serialize(
    Array<1, Byte, DeviceType> &compressed_data, int queue_idx) {
  lossless_compressor.Serialize(compressed_data, queue_idx);
}

template <DIM D, typename T, typename DeviceType>
void HybridHierarchyCompressor<D, T, DeviceType>::Deserialize(
    Array<1, Byte, DeviceType> &compressed_data, int queue_idx) {
  // lossless_compressor.Deserialize(compressed_data, queue_idx);
}

template <DIM D, typename T, typename DeviceType>
void HybridHierarchyCompressor<D, T, DeviceType>::Recompose(
    Array<D, T, DeviceType> &decompressed_data, int queue_idx) {
  local_refactor.Recompose(SubArray(decompressed_data), queue_idx);
}

template <DIM D, typename T, typename DeviceType>
void HybridHierarchyCompressor<D, T, DeviceType>::Dequantize(
    Array<D, T, DeviceType> &decompressed_data, enum error_bound_type ebtype,
    T tol, T s, T norm, int queue_idx) {
  std::vector<SIZE> original_shape =
      hierarchy->level_shape(hierarchy->l_target());
  SIZE total_num_elems_1D = 1;
  for (int d = 0; d < original_shape.size(); d++) {
    total_num_elems_1D *= (((original_shape[d] - 1) / 8 + 1) * 8);
  }
  SubArray<1, T, DeviceType> decompressed_data_subarray(
      {total_num_elems_1D}, decompressed_data.data());
  // Direct calculation
  local_quantizer.Dequantize(decompressed_data_subarray, ebtype, tol, s, norm,
                             local_quantized_array, lossless_compressor,
                             queue_idx);
}

template <DIM D, typename T, typename DeviceType>
void HybridHierarchyCompressor<D, T, DeviceType>::LosslessDecompress(
    Array<1, Byte, DeviceType> &compressed_data, int queue_idx) {
  lossless_compressor.Decompress(compressed_data, local_quantized_array,
                                 queue_idx);
}


template <DIM D, typename T, typename DeviceType>
void HybridHierarchyCompressor<D, T, DeviceType>::Compress(
    Array<D, T, DeviceType> &original_data, enum error_bound_type ebtype, T tol,
    T s, T &norm, Array<1, Byte, DeviceType> &compressed_data, int queue_idx) {
  config.apply();

  DeviceRuntime<DeviceType>::SelectDevice(config.dev_id);
  log::info("Select device: " + DeviceRuntime<DeviceType>::GetDeviceName());
  Timer timer_total;
  for (int d = D - 1; d >= 0; d--) {
    if (hierarchy->level_shape(hierarchy->l_target(), d) !=
        original_data.shape(d)) {
      log::err(
          "The shape of input array does not match the shape initilized "
          "in hierarchy!");
      return;
    }
  }

  if (log::level & log::TIME) timer_total.start();

  CalculateNorm(original_data, ebtype, s, norm, queue_idx);
  // log::info(std::to_string(original_data.totalNumElems()));
  // PrintSubarray("Original before decompose", SubArray(original_data));
  log::info("Before decompose()");
  Decompose(original_data, queue_idx);
  log::info("After decompose()");
  // log::info(std::to_string(original_data.totalNumElems()));
  // PrintSubarray("Original after decompose", SubArray(original_data));
  // // PrintSubarray("Decomposed", SubArray(decomposed_array));
  log::info("Before quantize");
  Quantize(original_data, ebtype, tol, s, norm, queue_idx);
  log::info("After quantize");
  // log::info("Num of Original data after quantization:");
  // log::info(std::to_string(original_data.totalNumElems()));
  // PrintSubarray("Quantized", SubArray(local_quantized_array));
  // PrintSubarray("Compressed data before lossless",SubArray(compressed_data));
  log::info("Before lossless");
  LosslessCompress(compressed_data, queue_idx);
  log::info("After lossless");
  // PrintSubarray("Quantized data before lossless",SubArray(local_quantized_array));
  // PrintSubarray("Compressed data after lossless",SubArray(compressed_data));
  // From printing result, we found lossless didn't do anything to compressed_data
  if (config.compress_with_dryrun) {
    Dequantize(original_data, ebtype, tol, s, norm, queue_idx);
    // PrintSubarray("Original data after dequantization", SubArray(original_data));
    Recompose(original_data, queue_idx);
    // PrintSubarray("Original data after recompose", SubArray(original_data));
  }

  if (log::level & log::TIME) {
    DeviceRuntime<DeviceType>::SyncQueue(0);
    timer_total.end();
    timer_total.print("Low-level compression");
    log::time(
        "Low-level compression throughput: " +
        std::to_string((double)(hierarchy->total_num_elems() * sizeof(T)) /
                       timer_total.get() / 1e9) +
        " GB/s");
    timer_total.clear();
  }
}

template <DIM D, typename T, typename DeviceType>
void HybridHierarchyCompressor<D, T, DeviceType>::Decompress(
    Array<1, Byte, DeviceType> &compressed_data, enum error_bound_type ebtype,
    T tol, T s, T &norm, Array<D, T, DeviceType> &decompressed_data,
    int queue_idx) {
  config.apply();
  log::info("Have we ever in Decompress?");

  DeviceRuntime<DeviceType>::SelectDevice(config.dev_id);
  log::info("Select device: " + DeviceRuntime<DeviceType>::GetDeviceName());
  Timer timer_total, timer_each;

  if (log::level & log::TIME) timer_total.start();

  decompressed_data.resize(hierarchy->level_shape(hierarchy->l_target()));
  // LosslessDecompress(compressed_data, queue_idx);
  Dequantize(decompressed_data, ebtype, tol, s, norm, queue_idx);
  // PrintSubarray("Dequantized", SubArray(hybrid_quantized_array));
  Recompose(decompressed_data, queue_idx);
  // PrintSubarray("Recomposed", SubArray(decompressed_data));

  if (log::level & log::TIME) {
    DeviceRuntime<DeviceType>::SyncQueue(0);
    timer_total.end();
    timer_total.print("Low-level decompression");
    log::time(
        "Low-level decompression throughput: " +
        std::to_string((double)(hierarchy->total_num_elems() * sizeof(T)) /
                       timer_total.get() / 1e9) +
        " GB/s");
    timer_total.clear();
  }
}

template <DIM D, typename T, typename DeviceType>
SIZE HybridHierarchyCompressor<D, T, DeviceType>::calculate_padded_size
(Hierarchy<D, T, DeviceType> &hierarchy) {
    std::vector<SIZE> original_shape =
        hierarchy.level_shape(hierarchy.l_target());
    SIZE total_num_elems_1D = 1;
    for (int d = 0; d < original_shape.size(); d++) {
      // 这个公式的作用是将维度向上取整到最接近的8的倍数
      total_num_elems_1D *= (((original_shape[d] - 1) / 8 + 1) * 8);
    }
    return total_num_elems_1D;
  }

}  // namespace mgard_x

#endif