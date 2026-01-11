/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include "BlockLocalHierarchyDataRefactor.hpp"
#include "DataRefactor.hpp"
#include "HybridHierarchyDataRefactorInterface.hpp"
#include "InCacheBlock/DataRefactoring.h"
#include "MultiDimension/DataRefactoring.h"
#include "SingleDimension/DataRefactoring.h"

#ifndef MGARD_X_HYBRID_HIERARCHY_DATA_REFACTOROR_HPP
#define MGARD_X_HYBRID_HIERARCHY_DATA_REFACTOROR_HPP
namespace mgard_x {

namespace data_refactoring {

template <DIM D, typename T, typename DeviceType>
class HybridHierarchyDataRefactor
    : public HybridHierarchyDataRefactorInterface<D, T, DeviceType> {
 public:
  HybridHierarchyDataRefactor() : initialized(false) {}
  HybridHierarchyDataRefactor(Hierarchy<D, T, DeviceType>& hierarchy,
                              Config config)
      : initialized(true), hierarchy(&hierarchy), config(config) {
    this->L = config.num_local_refactoring_level;
    this->M = config.num_global_refactoring_level;
  }

  void Adapt(Hierarchy<D, T, DeviceType>& hierarchy, Config config,
             int queue_idx) {
    this->initialized = true;
    this->hierarchy = &hierarchy;
    this->config = config;

    this->L = config.num_local_refactoring_level;
    this->M = config.num_global_refactoring_level;

    // We have to set global hierarchy and global config here
    local_refactor.Adapt(hierarchy, config, queue_idx);

    std::vector<SIZE> global_hierarchy_shape =
        local_refactor.coarse_shapes[this->L - 1];
    Config global_config;
    global_config.max_larget_level = this->M;
    this->global_hierarchy =
        Hierarchy<D, T, DeviceType>(global_hierarchy_shape, global_config);
    global_refactor.Adapt(global_hierarchy, global_config, queue_idx);
  }

  // Need to add memory for local later
  static size_t EstimateMemoryFootprint(std::vector<SIZE> shape) {
    size_t size = 0;
    // Global memory size

    // Local size (double check needed)
    return size;
  }

  size_t DecomposedDataSize() { return local_refactor.DecomposedDataSize(); }

  void Decompose(SubArray<D, T, DeviceType> data,
                 SubArray<1, T, DeviceType> decomposed_data, int queue_idx) {
    // Local decomposition
    local_refactor.Decompose(data, decomposed_data, queue_idx);

    std::vector<SIZE> local_coarest_shape =
        local_refactor.coarse_shapes[this->L - 1];
    SubArray<D, T, DeviceType> global_input_data({local_coarest_shape},
                                             decomposed_data.data());
    for (DIM d = 0; d < D; d++) {
      global_input_data.setLd(d, local_coarest_shape[d]);
    }
    global_input_data.project(0, 1, 2);

    // Global decomposition
    global_refactor.Decompose(global_input_data, true, queue_idx);
  }

  void Recompose(SubArray<D, T, DeviceType> data,
                 SubArray<1, T, DeviceType> decomposed_data, int queue_idx) {
    std::vector<SIZE> local_coarest_shape =
        local_refactor.coarse_shapes[this->L - 1];
    SubArray<D, T, DeviceType> global_input_data({local_coarest_shape},
                                             decomposed_data.data());
    for (DIM d = 0; d < D; d++) {
      global_input_data.setLd(d, local_coarest_shape[d]);
    }
    global_input_data.project(0, 1, 2);

    // Global recomposition
    global_refactor.Recompose(global_input_data, true, queue_idx);

    // Local recomposition
    local_refactor.Recompose(data, decomposed_data, queue_idx);
  }

  bool initialized;
  Hierarchy<D, T, DeviceType>* hierarchy;
  Hierarchy<D, T, DeviceType> global_hierarchy;
  Config config;

  SIZE L;  // Number of local levels
  SIZE M;  // Number of global levels

  BlockLocalHierarchyDataRefactor<D, T, DeviceType> local_refactor;
  DataRefactor<D, T, DeviceType> global_refactor;
};

}  // namespace data_refactoring

}  // namespace mgard_x

#endif