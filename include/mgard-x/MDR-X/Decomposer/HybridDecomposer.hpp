/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: September 21, 2026
 */

#ifndef _MDR_HYBRID_DECOMPOSER_HPP
#define _MDR_HYBRID_DECOMPOSER_HPP

#include "../../DataRefactoring/DataRefactor.hpp"
#include "../../DataRefactoring/InCacheBlock/DataRefactoring.h"
#include "../../DataRefactoring/MultiDimension/DataRefactoring.h"
#include "../../RuntimeX/Utilities/Exceptions.h"
#include "../Interleaver/DirectInterleaver.hpp"

#include <limits>

namespace mgard_x {
namespace MDR {

// MDR level structure. MDR level 0 is the coarsest; the last level is the
// finest.
//
// Global (MultiDim) mode: the levels of the multigrid hierarchy over the whole
// domain.
//
// Hybrid (BlockMGARD) mode: L block-local levels of the in-cache 8^D -> 5^D
// decomposition, followed by G global multigrid levels over the (small)
// coarsest block-local region:
//   MDR levels [0, G]           global levels over the coarsest region
//   MDR level  G + 1 + j        block-local level L - 1 - j (j = 0 is the
//                               coarsest block-local level)
// Block-local coefficients are already linear (block-major) in the transform
// output, so they need no interleaving; only the global part does.
struct MDRLevelLayout {
  bool hybrid = false;
  int L = 0; // block-local levels
  int G = 0; // l_target of the global hierarchy (MultiDim mode: whole domain)
  std::vector<SIZE> level_num_elems;

  // Hybrid only
  std::vector<std::vector<SIZE>> local_fine_shapes;   // level 0 = finest
  std::vector<std::vector<SIZE>> local_coarse_shapes; // level 0 = finest
  std::vector<SIZE> local_coeff_size;                 // level 0 = finest
  std::vector<SIZE> global_shape; // shape of the global region
  std::vector<std::vector<SIZE>> global_level_shapes;

  SIZE num_levels() const { return level_num_elems.size(); }
  SIZE max_level_num_elems() const {
    SIZE n = 0;
    for (SIZE e : level_num_elems)
      n = std::max(n, e);
    return n;
  }
  // MDR level that holds the coefficients of block-local level l
  int mdr_level_of_local(int l) const { return G + L - l; }
};

// Level shapes of the MGARD-X multigrid hierarchy, same rule as
// Hierarchy::init: each dimension coarsens n -> n / 2 + 1 down to 2, the
// level count is limited by the shortest dimension and by max_level.
inline std::vector<std::vector<SIZE>>
global_level_shapes(std::vector<SIZE> shape, SIZE max_level) {
  DIM D = shape.size();
  std::vector<std::vector<SIZE>> shape_level(D);
  for (DIM d = 0; d < D; d++) {
    SIZE n = shape[d];
    while (n > 2) {
      shape_level[d].push_back(n);
      n = n / 2 + 1;
    }
    shape_level[d].push_back(2);
  }
  SIZE nlevel = shape_level[0].size();
  for (DIM d = 1; d < D; d++) {
    nlevel = std::min(nlevel, (SIZE)shape_level[d].size());
  }
  SIZE l_target = std::min(nlevel - 1, max_level);
  std::vector<std::vector<SIZE>> level_shapes(l_target + 1,
                                              std::vector<SIZE>(D));
  for (SIZE l = 0; l <= l_target; l++) {
    for (DIM d = 0; d < D; d++) {
      level_shapes[l][d] = shape_level[d][l_target - l];
    }
  }
  return level_shapes;
}

inline bool mdr_use_hybrid(DIM D, const Config &config) {
  if (config.decomposition != decomposition_type::Hybrid) {
    return false;
  }
  if (D > 3) {
    throw ProcessingException(
        "MDR-X: the hybrid (block-local) decomposition supports 1D-3D only.");
  }
  if (config.num_local_refactoring_level < 1) {
    throw ProcessingException(
        "MDR-X: the hybrid decomposition needs at least one local level.");
  }
  return true;
}

inline SIZE product(const std::vector<SIZE> &shape) {
  SIZE n = 1;
  for (SIZE s : shape)
    n *= s;
  return n;
}

template <DIM D, typename T, typename DeviceType>
std::vector<SIZE> shape_of(const SubArray<D, T, DeviceType> &v) {
  std::vector<SIZE> shape(D);
  for (DIM d = 0; d < D; d++)
    shape[d] = v.shape(d);
  return shape;
}

// shape: the full (finest) shape of the domain being refactored.
inline MDRLevelLayout build_level_layout(std::vector<SIZE> shape,
                                         const Config &config) {
  DIM D = shape.size();
  MDRLevelLayout layout;
  layout.hybrid = mdr_use_hybrid(D, config);

  if (!layout.hybrid) {
    std::vector<std::vector<SIZE>> level_shapes =
        global_level_shapes(shape, config.max_larget_level);
    layout.G = level_shapes.size() - 1;
    SIZE prev = 0;
    for (int l = 0; l <= layout.G; l++) {
      SIZE curr = product(level_shapes[l]);
      layout.level_num_elems.push_back(curr - prev);
      prev = curr;
    }
    return layout;
  }

  layout.L = config.num_local_refactoring_level;
  std::vector<SIZE> coarse_shape = shape;
  for (int l = 0; l < layout.L; l++) {
    std::vector<SIZE> fine_shape(D);
    for (DIM d = 0; d < D; d++) {
      fine_shape[d] =
          ((coarse_shape[d] - 1) / MGARDX_HYBRID_LOCAL_BLOCK_SIZE + 1) *
          MGARDX_HYBRID_LOCAL_BLOCK_SIZE;
      coarse_shape[d] = fine_shape[d] / MGARDX_HYBRID_LOCAL_BLOCK_SIZE *
                        MGARDX_HYBRID_LOCAL_COARSE_SIZE;
    }
    layout.local_fine_shapes.push_back(fine_shape);
    layout.local_coarse_shapes.push_back(coarse_shape);
    layout.local_coeff_size.push_back(product(fine_shape) -
                                      product(coarse_shape));
  }
  layout.global_shape = coarse_shape;

  // Negative = as many global levels as the coarsest region allows.
  SIZE max_global = config.num_global_refactoring_level < 0
                        ? std::numeric_limits<SIZE>::max()
                        : (SIZE)config.num_global_refactoring_level;
  layout.global_level_shapes = global_level_shapes(coarse_shape, max_global);
  layout.G = layout.global_level_shapes.size() - 1;

  SIZE prev = 0;
  for (int l = 0; l <= layout.G; l++) {
    SIZE curr = product(layout.global_level_shapes[l]);
    layout.level_num_elems.push_back(curr - prev);
    prev = curr;
  }
  for (int l = layout.L - 1; l >= 0; l--) {
    layout.level_num_elems.push_back(layout.local_coeff_size[l]);
  }
  return layout;
}

// Hybrid (BlockMGARD) decomposer for MDR-X: block-local in-cache levels plus
// global levels over the coarsest region. decompose() writes the coefficients
// of every MDR level straight into that level's linear buffer, so it replaces
// both the MultiDim decomposer and the interleaver.
template <DIM D, typename T, typename DeviceType> class HybridDecomposer {
public:
  HybridDecomposer() : initialized(false) {}

  void Adapt(const MDRLevelLayout &layout, bool orthogonal_projection,
             int queue_idx) {
    // The pipelines re-Adapt for every subdomain, inside their timed loops.
    // Rebuilding the global hierarchy each time costs several ms, so keep it
    // unless the global region changed.
    bool global_changed = !initialized || layout.G != this->layout.G ||
                          layout.global_shape != this->layout.global_shape;
    this->initialized = true;
    this->layout = layout;
    this->orthogonal_projection = orthogonal_projection;
    for (int i = 0; i < 2; i++) {
      coarse_buffers[i].resize(layout.local_fine_shapes[0], queue_idx);
    }
    coarsest_buffer.resize({product(layout.global_shape)}, queue_idx);
    if (layout.G > 0 && global_changed) {
      // A fresh Config: DataRefactor only runs the MultiDim kernels when
      // config.decomposition is MultiDim (Recompose ignores Hybrid).
      Config global_config;
      global_config.max_larget_level = layout.G;
      global_hierarchy =
          Hierarchy<D, T, DeviceType>(layout.global_shape, global_config);
      global_refactor.Adapt(global_hierarchy, global_config, queue_idx);
      global_interleaver.Adapt(global_hierarchy, queue_idx);
    }
  }

  static size_t EstimateMemoryFootprint(const MDRLevelLayout &layout) {
    size_t size = 2 * product(layout.local_fine_shapes[0]) * sizeof(T);
    size += product(layout.global_shape) * sizeof(T);
    if (layout.G > 0) {
      Hierarchy<D, T, DeviceType> hierarchy;
      size += hierarchy.EstimateMemoryFootprint(layout.global_shape);
      size += data_refactoring::DataRefactor<
          D, T, DeviceType>::EstimateMemoryFootprint(layout.global_shape);
      size += DirectInterleaver<D, T, DeviceType>::EstimateMemoryFootprint(
          layout.global_shape);
    }
    return size;
  }

  // data: the full-resolution input (not modified).
  // level_data[i]: buffer of MDR level i; the first level_num_elems[i] entries
  // are written.
  void decompose(SubArray<D, T, DeviceType> data,
                 std::vector<SubArray<1, T, DeviceType>> &level_data,
                 int queue_idx) {
    const int L = layout.L;
    Timer timer;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }

    // Level-0 fine buffer: the input padded up to a multiple of the block
    // size. The padding must be zero, CopyND only writes the input extent.
    SubArray<D, T, DeviceType> fine =
        padded_view(layout.local_fine_shapes[0], coarse_buffers[1].data());
    if (product(layout.local_fine_shapes[0]) != product(shape_of(data))) {
      coarse_buffers[1].memset(0, queue_idx);
    }
    data_refactoring::multi_dimension::CopyND(data, fine, queue_idx);

    for (int l = 0; l < L; l++) {
      int buffer_idx = l % 2;
      // Zero so that the padding of the next level's fine grid is zero.
      coarse_buffers[buffer_idx].memset(0, queue_idx);
      SubArray<D, T, DeviceType> coarse = padded_view(
          layout.local_coarse_shapes[l], coarse_buffers[buffer_idx].data());
      SubArray<1, T, DeviceType> coeff(
          {layout.local_coeff_size[l]},
          level_data[layout.mdr_level_of_local(l)].data());
      data_refactoring::in_cache_block::decompose<D, T, DeviceType>(
          fine, coarse, coeff, orthogonal_projection, queue_idx);
      if (l < L - 1) {
        fine = padded_view(layout.local_fine_shapes[l + 1],
                           coarse_buffers[buffer_idx].data());
      }
    }

    // Compact the coarsest block-local region. Without global levels it is
    // MDR level 0 itself and goes straight to its buffer.
    SubArray<D, T, DeviceType> coarsest =
        padded_view(layout.global_shape, coarse_buffers[(L - 1) % 2].data());
    SubArray<D, T, DeviceType> global_data(layout.global_shape,
                                           layout.G > 0 ? coarsest_buffer.data()
                                                        : level_data[0].data());
    global_data.project(D - 3, D - 2, D - 1);
    data_refactoring::multi_dimension::CopyND(coarsest, global_data, queue_idx);

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("Local Decomposition", product(shape_of(data)) * sizeof(T));
      timer.clear();
    }

    if (layout.G > 0) {
      global_refactor.Decompose(global_data, orthogonal_projection, queue_idx);
      global_interleaver.interleave(global_data, level_data, layout.G,
                                    queue_idx);
    }
  }

  // Inverse of decompose(). level_data is read only; output (full resolution)
  // is overwritten.
  void recompose(std::vector<SubArray<1, T, DeviceType>> &level_data,
                 SubArray<D, T, DeviceType> output, int queue_idx) {
    const int L = layout.L;
    SubArray<D, T, DeviceType> coarse(layout.global_shape,
                                      layout.G > 0 ? coarsest_buffer.data()
                                                   : level_data[0].data());
    coarse.project(D - 3, D - 2, D - 1);
    if (layout.G > 0) {
      global_interleaver.reposition(level_data, coarse, layout.G, queue_idx);
      global_refactor.Recompose(coarse, orthogonal_projection, queue_idx);
    }

    Timer timer;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }
    // Every block writes all of its 8^D fine nodes, so the ping-pong buffers
    // need no zeroing: each value read at level l was written at level l + 1.
    SubArray<D, T, DeviceType> fine;
    for (int i = 0; i < L; i++) {
      int l = L - 1 - i;
      int buffer_idx = i % 2;
      fine = padded_view(layout.local_fine_shapes[l],
                         coarse_buffers[buffer_idx].data());
      SubArray<1, T, DeviceType> coeff(
          {layout.local_coeff_size[l]},
          level_data[layout.mdr_level_of_local(l)].data());
      data_refactoring::in_cache_block::recompose<D, T, DeviceType>(
          fine, coarse, coeff, orthogonal_projection, queue_idx);
      if (l > 0) {
        coarse = padded_view(layout.local_coarse_shapes[l - 1],
                             coarse_buffers[buffer_idx].data());
      }
    }
    // Drop the block padding
    SubArray<D, T, DeviceType> result =
        padded_view(shape_of(output), fine.data());
    data_refactoring::multi_dimension::CopyND(result, output, queue_idx);
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("Local Recomposition", product(shape_of(output)) * sizeof(T));
      timer.clear();
    }
  }

  void print() const {
    std::cout << "Hybrid (block-local + global) decomposer" << std::endl;
  }

private:
  // View of `shape` inside a ping-pong buffer laid out densely with the
  // level-0 fine shape as leading dimensions.
  SubArray<D, T, DeviceType> padded_view(std::vector<SIZE> shape, T *ptr) {
    SubArray<D, T, DeviceType> view(shape, ptr);
    for (DIM d = 0; d < D; d++) {
      view.setLd(d, layout.local_fine_shapes[0][d]);
    }
    view.project(D - 3, D - 2, D - 1);
    return view;
  }

  bool initialized;
  bool orthogonal_projection = false;
  MDRLevelLayout layout;
  Array<D, T, DeviceType> coarse_buffers[2];
  Array<1, T, DeviceType> coarsest_buffer;
  Hierarchy<D, T, DeviceType> global_hierarchy;
  data_refactoring::DataRefactor<D, T, DeviceType> global_refactor;
  DirectInterleaver<D, T, DeviceType> global_interleaver;
};

} // namespace MDR
} // namespace mgard_x
#endif
