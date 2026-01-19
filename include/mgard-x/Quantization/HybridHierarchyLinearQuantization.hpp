/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_HYBRID_HIERARCHY_LINEAR_QUANTIZATION_TEMPLATE
#define MGARD_X_HYBRID_HIERARCHY_LINEAR_QUANTIZATION_TEMPLATE

#include "../RuntimeX/RuntimeX.h"
#include "LinearQuantization.hpp"
#include "LocalQuantization.hpp"
#include "QuantizationInterface.hpp"

namespace mgard_x {

#define MGARDX_QUANTIZE 1
#define MGARDX_DEQUANTIZE 2

template <DIM D, typename T, typename Q, typename DeviceType>
class HybridHierarchyQuantizer
    : public QuantizationInterface<D, T, Q, DeviceType> {
 public:
  HybridHierarchyQuantizer() : initialized(false) {}

  HybridHierarchyQuantizer(Hierarchy<D, T, DeviceType>& hierarchy,
                           Hierarchy<D, T, DeviceType>& global_hierarchy,
                           Config config)
      : initialized(true),
        hierarchy(&hierarchy),
        global_hierarchy(&global_hierarchy),
        config(config) {
    this->L = config.num_local_refactoring_level;
    this->M = config.num_global_refactoring_level;
    ComputeLocalShapes();
  }

  void Adapt(Hierarchy<D, T, DeviceType>& hierarchy,
             Hierarchy<D, T, DeviceType>& global_hierarchy, Config config,
             int queue_idx) {
    this->initialized = true;
    this->hierarchy = &hierarchy;
    this->global_hierarchy = &global_hierarchy;
    this->config = config;
    this->L = config.num_local_refactoring_level;
    this->M = config.num_global_refactoring_level;

    local_quantizer.Adapt(hierarchy, config, queue_idx);
    global_quantizer.Adapt(global_hierarchy, config, queue_idx);

    ComputeLocalShapes();
  }

  // Set block-level tolerances according to ROI table
  void SetBlockTolerances(const std::vector<T>& initial_block_tolerances) {
    BuildROIToleranceMap(initial_block_tolerances);
  }

  // Return the error budget for global quantization
  T ErrorBudgetAllocation(T tol) {
    if (!roi_tolerance_map.empty()) {
      return GetMinToleranceForGlobal();
    }
  }

  void Quantize(SubArray<D, T, DeviceType> original_data,
                enum error_bound_type ebtype, T tol, T s, T norm,
                SubArray<D, Q, DeviceType> quantized_data, int queue_idx) {}

  void Dequantize(SubArray<D, T, DeviceType> original_data,
                  enum error_bound_type ebtype, T tol, T s, T norm,
                  SubArray<D, Q, DeviceType> quantized_data, int queue_idx) {}

  // Here we take in a ROI MAP
  template <typename LosslessCompressorType>
  void Quantize(SubArray<1, T, DeviceType> original_data,
                enum error_bound_type ebtype, T tol, T s, T norm,
                SubArray<1, Q, DeviceType> quantized_data,
                LosslessCompressorType& lossless, int queue_idx) {
    // Global quantization
    T global_tol = ErrorBudgetAllocation(tol);
    SIZE global_q_size = (this->M > 0) ? (global_hierarchy->l_target() + 1) : 0;
    std::vector<SIZE> global_shape =
        global_hierarchy->level_shape(global_hierarchy->l_target());
    SubArray<D, T, DeviceType> global_data_v(global_shape,
                                             original_data.data());
    SubArray<D, Q, DeviceType> global_data_q(global_shape,
                                             quantized_data.data());

    global_quantizer.Quantize(global_data_v, ebtype, global_tol, s, norm,
                              global_data_q, lossless, queue_idx);

    // Local quantization
    SIZE local_offset = original_data.shape(0) - global_q_size;
    SubArray<1, T, DeviceType> local_data_v({local_offset},
                                            original_data(global_q_size));
    SubArray<1, Q, DeviceType> local_data_q({local_offset},
                                            quantized_data(global_q_size));
    // For local quantizer, we take a calculated tolerance map
    local_quantizer.Quantize(local_data_v, ebtype, global_tol, s, norm,
                             local_data_q, lossless, queue_idx);
  }

  // Here we take in a ROI MAP
  template <typename LosslessCompressorType>
  void Dequantize(SubArray<1, T, DeviceType> original_data,
                  enum error_bound_type ebtype, T tol, T s, T norm,
                  SubArray<1, Q, DeviceType> quantized_data,
                  LosslessCompressorType& lossless, int queue_idx) {
    // Global dequantization
    T global_tol = ErrorBudgetAllocation(tol);
    SIZE global_q_size = (this->M > 0) ? (global_hierarchy->l_target() + 1) : 0;
    std::vector<SIZE> global_shape =
        global_hierarchy->level_shape(global_hierarchy->l_target());
    SubArray<D, T, DeviceType> global_data_v(global_shape,
                                             original_data.data());
    SubArray<D, Q, DeviceType> global_data_q(global_shape,
                                             quantized_data.data());

    global_quantizer.Dequantize(global_data_v, ebtype, global_tol, s, norm,
                                global_data_q, lossless, queue_idx);

    // Local dequantization
    SIZE local_offset = original_data.shape(0) - global_q_size;
    SubArray<1, T, DeviceType> local_data_v({local_offset},
                                            original_data(global_q_size));
    SubArray<1, Q, DeviceType> local_data_q({local_offset},
                                            quantized_data(global_q_size));
    // For local quantizer, we take a calculated tolerance map
    local_quantizer.Dequantize(local_data_v, ebtype, global_tol, s, norm,
                               local_data_q, lossless, queue_idx);
  }

  // Compute fine and coarse shapes for all local levels
  void ComputeLocalShapes() {
    fine_shapes.clear();
    coarse_shapes.clear();

    // Get original shape from hierarchy
    std::vector<SIZE> coarse_shape =
        hierarchy->level_shape(hierarchy->l_target());

    for (SIZE l = 0; l < this->L; ++l) {
      std::vector<SIZE> fine_shape(D);
      for (DIM d = 0; d < D; ++d) {
        // Round up to multiple of 8 for fine shape
        coarse_shape[d] = ((coarse_shape[d] - 1) / 8 + 1) * 8;
        fine_shape[d] = coarse_shape[d];
        // Compute next coarse shape (8->5 mapping)
        coarse_shape[d] = ((coarse_shape[d] - 1) / 8 + 1) * 5;
      }
      fine_shapes.push_back(fine_shape);
      coarse_shapes.push_back(coarse_shape);
    }
  }

  // Build ROI tolerance map for all local levels (stored as 1D array)
  void BuildROIToleranceMap(const std::vector<T>& initial_tolerances) {
    roi_tolerance_map.clear();
    level_offsets.clear();
    level_block_counts.clear();

    // Level 0: use initial tolerances directly
    level_offsets.push_back(0);
    level_block_counts.push_back(initial_tolerances.size());
    roi_tolerance_map.insert(roi_tolerance_map.end(),
                             initial_tolerances.begin(),
                             initial_tolerances.end());

    // Propagate tolerances through local levels
    for (SIZE level = 1; level < this->L; ++level) {
      SIZE prev_offset = level_offsets[level - 1];
      SIZE prev_count = level_block_counts[level - 1];

      std::vector<T> prev_level_tolerances(
          roi_tolerance_map.begin() + prev_offset,
          roi_tolerance_map.begin() + prev_offset + prev_count);

      std::vector<T> next_level_tolerances =
          PropagateTolerances(prev_level_tolerances, level - 1, level);

      level_offsets.push_back(roi_tolerance_map.size());
      level_block_counts.push_back(next_level_tolerances.size());
      roi_tolerance_map.insert(roi_tolerance_map.end(),
                               next_level_tolerances.begin(),
                               next_level_tolerances.end());
    }
  }

  // Propagate error to next level
  std::vector<T> PropagateTolerances(const std::vector<T>& current_tolerances,
                                     SIZE curr_level, SIZE next_level) {
    // Get current and next level block dimensions from computed shapes
    std::vector<SIZE> curr_blocks = GetBlockDimensions(curr_level);
    std::vector<SIZE> next_blocks = GetBlockDimensions(next_level);

    SIZE next_size = 1;
    for (DIM d = 0; d < D; ++d) {
      next_size *= next_blocks[d];
    }

    // Initialize tolerance list for next block
    std::vector<T> next_tolerances(next_size, std::numeric_limits<T>::max());

    // For each block in next level, find minimum tolerance from contributing blocks
    for (SIZE idx = 0; idx < next_size; ++idx) {
      std::vector<SIZE> next_coord = LinearToCoord(idx, next_blocks);
      T min_tol = std::numeric_limits<T>::max();

      // Find all contributing blocks from current level
      std::vector<SIZE> contributing_blocks =
          GetContributingBlocks(next_coord, curr_blocks);

      for (SIZE contrib_idx : contributing_blocks) {
        if (contrib_idx < current_tolerances.size()) {
          min_tol = std::min(min_tol, current_tolerances[contrib_idx]);
        }
      }

      next_tolerances[idx] = min_tol;
    }

    return next_tolerances;
  }

  // Contributing tables:
  // NEXT BLOCK       CUR BLOCK
  //     0               0,1
  //     1              1,2,3
  //     2               3,4
  //     3              4,5,6
  //     4               6,7
  // Get contributing block indices from previous level for a given next-level block
  std::vector<SIZE> GetContributingBlocks(
      const std::vector<SIZE>& next_coord,
      const std::vector<SIZE>& curr_blocks) {
    std::vector<std::vector<SIZE>> contrib_per_dim(D);

    // For each dimension, apply the 5->8 inverse mapping
    for (DIM d = 0; d < D; ++d) {
      SIZE next_idx = next_coord[d];
      SIZE group = next_idx / 5;
      SIZE offset = next_idx % 5;
      SIZE base = group * 8;

      // Apply the propagation pattern
      switch (offset) {
        case 0:  // min(0, 1)
          contrib_per_dim[d] = {base + 0, base + 1};
          break;
        case 1:  // min(1, 2, 3)
          contrib_per_dim[d] = {base + 1, base + 2, base + 3};
          break;
        case 2:  // min(3, 4)
          contrib_per_dim[d] = {base + 3, base + 4};
          break;
        case 3:  // min(4, 5, 6)
          contrib_per_dim[d] = {base + 4, base + 5, base + 6};
          break;
        case 4:  // min(6, 7)
          contrib_per_dim[d] = {base + 6, base + 7};
          break;
      }

      // Filter out-of-bounds indices
      std::vector<SIZE> valid;
      for (SIZE idx : contrib_per_dim[d]) {
        if (idx < curr_blocks[d]) {
          valid.push_back(idx);
        }
      }
      contrib_per_dim[d] = valid;
    }

    // Generate all combinations (Cartesian product)
    return CartesianProduct(contrib_per_dim, curr_blocks);
  }

  // Cartesian product of contributing indices across dimensions
  std::vector<SIZE> CartesianProduct(
      const std::vector<std::vector<SIZE>>& indices_per_dim,
      const std::vector<SIZE>& blocks) {
    std::vector<SIZE> result;
    std::vector<SIZE> coord(D);
    CartesianProductHelper(indices_per_dim, blocks, 0, coord, result);
    return result;
  }

  void CartesianProductHelper(
      const std::vector<std::vector<SIZE>>& indices_per_dim,
      const std::vector<SIZE>& blocks, DIM dim, std::vector<SIZE>& coord,
      std::vector<SIZE>& result) {
    if (dim == D) {
      result.push_back(CoordToLinear(coord, blocks));
      return;
    }

    for (SIZE idx : indices_per_dim[dim]) {
      coord[dim] = idx;
      CartesianProductHelper(indices_per_dim, blocks, dim + 1, coord, result);
    }
  }

  // Convert linear index to coordinate
  std::vector<SIZE> LinearToCoord(SIZE linear_idx,
                                  const std::vector<SIZE>& dims) {
    std::vector<SIZE> coord(D);
    for (int d = D - 1; d >= 0; --d) {
      coord[d] = linear_idx % dims[d];
      linear_idx /= dims[d];
    }
    return coord;
  }

  // Convert coordinate to linear index
  SIZE CoordToLinear(const std::vector<SIZE>& coord,
                     const std::vector<SIZE>& dims) {
    SIZE linear = 0;
    SIZE stride = 1;
    for (int d = D - 1; d >= 0; --d) {
      linear += coord[d] * stride;
      stride *= dims[d];
    }
    return linear;
  }

  // Get block dimensions at a specific level
  std::vector<SIZE> GetBlockDimensions(SIZE level) {
    // Use the fine shape for this level (before decomposition)
    const std::vector<SIZE>& fine_shape = fine_shapes[level];

    // Calculate block size (8x8x8 for local decomposition)
    const SIZE BLOCK_SIZE = 8;

    // Calculate number of blocks in each dimension
    std::vector<SIZE> block_dims(D);
    for (DIM d = 0; d < D; ++d) {
      block_dims[d] = (fine_shape[d] + BLOCK_SIZE - 1) / BLOCK_SIZE;
    }

    return block_dims;
  }

  // Get minimum tolerance for global quantization
  T GetMinToleranceForGlobal() {
    if (roi_tolerance_map.empty() || level_offsets.empty()) {
      return std::numeric_limits<T>::max();
    }

    // Get the last level
    SIZE last_level_idx = level_offsets.size() - 1;
    SIZE last_offset = level_offsets[last_level_idx];
    SIZE last_count = level_block_counts[last_level_idx];

    T min_tol = std::numeric_limits<T>::max();
    for (SIZE i = 0; i < last_count; ++i) {
      min_tol = std::min(min_tol, roi_tolerance_map[last_offset + i]);
    }

    return min_tol;
  }

  bool initialized;
  SIZE L;  // Number of local levels
  SIZE M;  // Number of global levels

  Hierarchy<D, T, DeviceType>* hierarchy;
  Hierarchy<D, T, DeviceType>* global_hierarchy;
  Config config;

  LocalQuantizer<D, T, Q, DeviceType> local_quantizer;
  LinearQuantizer<D, T, Q, DeviceType> global_quantizer;

  // 1D ROI tolerance map: all levels stored consecutively
  std::vector<T> roi_tolerance_map;

  // Offset for each level in the 1D tolerance map
  std::vector<SIZE> level_offsets;

  // Number of blocks at each level
  std::vector<SIZE> level_block_counts;

  // Fine and coarse shapes for each local level
  std::vector<std::vector<SIZE>> fine_shapes;
  std::vector<std::vector<SIZE>> coarse_shapes;
};

}  // namespace mgard_x

#endif