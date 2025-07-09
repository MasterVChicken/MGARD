#include "InCacheBlock/DataRefactoring.h"
#include "MultiDimension/DataRefactoring.h"

#ifndef MGARD_X_BLOCK_LOCAL_HIERARCHY_DATA_REFACTOR_HPP
#define MGARD_X_BLOCK_LOCAL_HIERARCHY_DATA_REFACTOR_HPP

namespace mgard_x {

namespace data_refactoring {

// Add temp space for further reuse

template <DIM D, typename T, typename DeviceType>
class BlockLocalHierarchyDataRefactor{
 public:
  BlockLocalHierarchyDataRefactor() : initialized(false) {}
  BlockLocalHierarchyDataRefactor(Hierarchy<D, T, DeviceType> &hierarchy,
                                  Config config)
      : initialized(true), hierarchy(&hierarchy), config(config) {
    // coarse_shape is intialized as the original data size
    coarse_shape = hierarchy.level_shape(hierarchy.l_target());
    if (config.num_local_refactoring_level > 0) {
      for (int l = 0; l < config.num_local_refactoring_level; l++) {
        SIZE last_level_size = 1;
        SIZE cur_level_size = 1;
        for (DIM d = 0; d < D; d++) {
          coarse_shape[d] = ((coarse_shape[d] - 1) / 8 + 1) * 8;
          last_level_size *= coarse_shape[d];
          coarse_shape[d] = ((coarse_shape[d] - 1) / 8 + 1) * 5;
          cur_level_size *= coarse_shape[d];
        }
        coarse_shapes.push_back(coarse_shape);
        coarse_num_elems.push_back(last_level_size);
        // Initialize the first coarse data shape
        if (l == 0) {
          w_array = Array<D, T, DeviceType>(coarse_shape);
        }
        local_coeff_size.push_back(last_level_size - cur_level_size);
      }
    }
  }

  void Adapt(Hierarchy<D, T, DeviceType> &hierarchy, Config config,
             int queue_idx) {
    this->initialized = true;
    this->hierarchy = &hierarchy;
    this->config = config;
    coarse_shape = hierarchy.level_shape(hierarchy.l_target());
    coarse_shapes.clear();
    coarse_num_elems.clear();
    local_coeff_size.clear();

    if (config.num_local_refactoring_level > 0) {
      for (int l = 0; l < config.num_local_refactoring_level; l++) {
        SIZE last_level_size = 1;
        SIZE cur_level_size = 1;
        for (DIM d = 0; d < D; d++) {
          coarse_shape[d] = ((coarse_shape[d] - 1) / 8 + 1) * 8;
          last_level_size *= coarse_shape[d];
          coarse_shape[d] = ((coarse_shape[d] - 1) / 8 + 1) * 5;
          cur_level_size *= coarse_shape[d];
        }
        coarse_shapes.push_back(coarse_shape);
        coarse_num_elems.push_back(last_level_size);
        // Initialize the first coarse data shape
        if (l == 0) {
          w_array = Array<D, T, DeviceType>(coarse_shape);
        }
        local_coeff_size.push_back(last_level_size - cur_level_size);
      }
    }
  }

  static size_t EstimateMemoryFootprint(std::vector<SIZE> shape) {
    size_t size = 0;
    return size;
  }

  size_t DecomposedDataSize() {
    size_t coeff_size = 0;

    for (int l = 0; l < config.num_local_refactoring_level; l++) {
      coeff_size += local_coeff_size[l];
    }

    size_t coarse_data_num = 1;
    for (DIM d = 0; d < D; d++) {
      coarse_data_num *=
          coarse_shapes[config.num_local_refactoring_level - 1][d];
    }
    coeff_size += coarse_data_num;

    return coeff_size;
  }

  void Decompose(SubArray<D, T, DeviceType> data,
                 SubArray<1, T, DeviceType> decomposed_data, int queue_idx) {
    SubArray<D, T, DeviceType> w_subarray(w_array);
    SIZE accumulated_local_coeff_size = 0;
    if (config.num_local_refactoring_level > 0) {
      // Here we initially process num_local_refactoring_level = 1
      for (SIZE l = 0; l < config.num_local_refactoring_level; l++) {
        accumulated_local_coeff_size += local_coeff_size[l];
        SubArray<1, T, DeviceType> local_coeff(
            {local_coeff_size[l]}, decomposed_data(decomposed_data.shape(0) -
                                                 accumulated_local_coeff_size));

        in_cache_block::decompose<D, T, DeviceType>(data, w_subarray,
                                                    local_coeff, queue_idx);

        SubArray<D, T, DeviceType> tmp = w_subarray;
        if (l + 1 < config.num_local_refactoring_level) {
          w_subarray =
              SubArray<D, T, DeviceType>(coarse_shapes[l + 1], data.data());
        }
        data = tmp;
      }
    }

    SubArray<D, T, DeviceType> out_coarse(
        coarse_shapes[config.num_local_refactoring_level - 1],
        decomposed_data((IDX)0));
    multi_dimension::CopyND(data, out_coarse, queue_idx);

    multi_dimension::CopyND(out_coarse, data, queue_idx);
  }

  void Recompose(SubArray<D, T, DeviceType> data,
                 SubArray<1, T, DeviceType> decomposed_data, int queue_idx) {
    SubArray<D, T, DeviceType> in_coarse(
        {coarse_shapes[config.num_local_refactoring_level - 1]},
        decomposed_data((IDX)0));
    multi_dimension::CopyND(in_coarse, data, queue_idx);
    SubArray<D, T, DeviceType> w_subarray(data);
    SubArray<D, T, DeviceType> data_subarray(data);
    SIZE coarse_offset = 1;
    for (SIZE d = 0; d < D; d++) {
      coarse_offset *= coarse_shapes[config.num_local_refactoring_level - 1][d];
    }
    if (config.num_local_refactoring_level > 0) {
      for (SIZE l = 0; l < config.num_local_refactoring_level; l++) {
        SIZE sz = local_coeff_size[config.num_local_refactoring_level - l - 1];
        SubArray<1, T, DeviceType> local_coeff(
            {sz}, decomposed_data((IDX)coarse_offset));
        in_cache_block::recompose<D, T, DeviceType>(data_subarray, w_subarray,
                                                    local_coeff, queue_idx);

        w_subarray = data_subarray;
        coarse_offset +=
            local_coeff_size[config.num_local_refactoring_level - l - 1];
      }
    }
  }

  bool initialized;
  Hierarchy<D, T, DeviceType> *hierarchy;
  Config config;
  std::vector<SIZE> coarse_shape;
  std::vector<SIZE> coarse_num_elems;
  std::vector<std::vector<SIZE>> coarse_shapes;
  std::vector<SIZE> local_coeff_size;
  Array<D, T, DeviceType> w_array;
};

}  // namespace data_refactoring
}  // namespace mgard_x

#endif