#include "InCacheBlock/DataRefactoring.h"
#include "MultiDimension/DataRefactoring.h"

#ifndef MGARD_X_BLOCK_LOCAL_HIERARCHY_DATA_REFACTOR_HPP
#define MGARD_X_BLOCK_LOCAL_HIERARCHY_DATA_REFACTOR_HPP

namespace mgard_x {

namespace data_refactoring {

template <DIM D, typename T, typename DeviceType>
class BlockLocalHierarchyDataRefactor {
 public:
  BlockLocalHierarchyDataRefactor() : initialized(false) {}
  BlockLocalHierarchyDataRefactor(Hierarchy<D, T, DeviceType> &hierarchy,
                                  Config config)
      : initialized(true), hierarchy(&hierarchy), config(config) {
    this->L = config.num_local_refactoring_level;
    compute_local_ranges();
    prepare_layers();

    w_array = Array<1, T, DeviceType>({fine_num_elems[0]});
  }

  void Adapt(Hierarchy<D, T, DeviceType> &hierarchy, Config config,
             int queue_idx) {
    this->initialized = true;
    this->hierarchy = &hierarchy;
    this->config = config;
    this->L = config.num_local_refactoring_level;
    compute_local_ranges();
    layer_len.clear();
    layer_off.clear();
    prepare_layers();

    w_array.resize({fine_num_elems[0]}, queue_idx);
  }

  static size_t EstimateMemoryFootprint(std::vector<SIZE> shape) {
    size_t size = 1;
    for (DIM d = 0; d < shape.size(); d++) {
      int dim8 = ((shape[d] - 1) / 8 + 1) * 8;
      size *= dim8;
    }
    return size * sizeof(T);
  }

  size_t DecomposedDataSize() {
    return layer_off[this->L] + layer_len[this->L];
  }

  void compute_local_ranges() {
    // Get original shape from hierarchy
    coarse_shape = hierarchy->level_shape(hierarchy->l_target());
    fine_num_elems.clear();
    coarse_num_elems.clear();
    local_coeff_size.clear();
    coarse_shapes.clear();
    fine_shapes.clear();

    for (int l = 0; l < this->L; ++l) {
      SIZE last_level_size = 1, curr_level_size = 1;
      std::vector<SIZE> fine_shape(D);
      for (DIM d = 0; d < D; ++d) {
        coarse_shape[d] = ((coarse_shape[d] - 1) / 8 + 1) * 8;
        last_level_size *= coarse_shape[d];
        coarse_shape[d] = ((coarse_shape[d] - 1) / 8 + 1) * 5;
        curr_level_size *= coarse_shape[d];
      }
      for (DIM d = 0; d < D; ++d) {
        fine_shape[d] = coarse_shape[d];
        fine_shape[d] = ((fine_shape[d] - 1) / 5 + 1) * 8;
      }
      fine_num_elems.push_back(last_level_size);
      coarse_num_elems.push_back(curr_level_size);
      local_coeff_size.push_back(last_level_size - curr_level_size);
      coarse_shapes.push_back(coarse_shape);
      fine_shapes.push_back(fine_shape);
    }
  }

  void prepare_layers() {
    layer_len.assign(this->L + 1, 0);
    layer_off.assign(this->L + 1, 0);

    // The length of coarsest layer
    layer_len[0] = coarse_num_elems[this->L - 1];
    layer_off[0] = 0;

    SIZE accum = layer_len[0];

    for (SIZE l = 1; l <= this->L; ++l) {
      layer_len[l] = local_coeff_size[this->L - l];
      layer_off[l] = accum;
      accum += layer_len[l];
    }
  }

  void Decompose(SubArray<D, T, DeviceType> data, int queue_idx) {
    SubArray<1, T, DeviceType> decomposed_data({fine_num_elems[0]},
                                               w_array.data());
    log::info("Fine num" + std::to_string(fine_num_elems[0]));
    // Create a copy for data
    SubArray<D, T, DeviceType> data_sub(fine_shapes[0], data.data());
    for (DIM d = 0; d < D; d++) {
      log::info("Dim" + std::to_string(d) + " : " +
                std::to_string(fine_shapes[0][d]));
    }

    if (this->L > 0) {
      accumulated_local_coeff_size = 0;
      // Here we initially process num_local_refactoring_level = 1
      for (SIZE l = 0; l < this->L; l++) {
        accumulated_local_coeff_size += local_coeff_size[l];
        SubArray<1, T, DeviceType> local_coeff(
            {local_coeff_size[l]},
            decomposed_data(decomposed_data.shape(0) -
                            accumulated_local_coeff_size));

        SubArray<D, T, DeviceType> coarse(coarse_shapes[l],
                                          decomposed_data.data());

        // The params sequence here is org, coarse, coeff, queue_idx
        in_cache_block::decompose<D, T, DeviceType>(data_sub, coarse,
                                                    local_coeff, queue_idx);
        // PrintSubarray("Original:", data_sub);
        // PrintSubarray("Coarse subarray after decompose()", coarse);
        // PrintSubarray("Coeff subarray after decompose()", local_coeff);

        SubArray<D, T, DeviceType> tmp = coarse;
        if (l + 1 < this->L) {
          coarse = SubArray<D, T, DeviceType>(coarse_shapes[l + 1],
                                              decomposed_data.data());
        }
        data_sub = tmp;
      }
    }
    // PrintSubarray("Whole decomposed data after decompose()",
    // decomposed_data);

    // Needs copy back
    // But I think here we should reshape data to shape of 1D and perform copy
    SubArray<1, T, DeviceType> data_1D({fine_num_elems[0]}, data.data());
    multi_dimension::CopyND(decomposed_data, data_1D, queue_idx);
    // PrintSubarray("Data 1D after CopyND()", data_1D);
  }

  void Recompose(SubArray<D, T, DeviceType> data, int queue_idx) {
    // data contains:
    // [0, coarse_num_elems[L-1]): coarse data
    // [coarse_num_elems[L-1], fine_num_elems[0]): coeff

    if (this->L > 0) {
      // use w_array as buffer
      SubArray<D, T, DeviceType> output_array(fine_shapes[0], w_array.data());

      for (SIZE l = 0; l < this->L; l++) {
        SIZE level_idx = this->L - 1 - l;

        if (l == 0) {
          // for first run, directly read from input
          SubArray<D, T, DeviceType> coarser(coarse_shapes[level_idx],
                                             data.data());
          SubArray<1, T, DeviceType> local_coeff({layer_len[l + 1]},
                                                 data((IDX)layer_off[l + 1]));

          in_cache_block::recompose<D, T, DeviceType>(output_array, coarser,
                                                      local_coeff, queue_idx);
        } else {
          // for L > 1: read from output_array and write back
          SubArray<D, T, DeviceType> coarser(coarse_shapes[level_idx],
                                             output_array.data());
          SubArray<1, T, DeviceType> local_coeff(
              {layer_len[l + 1]},
              data((IDX)layer_off[l + 1]));  // remains reading coeff from original
          SubArray<D, T, DeviceType> finer(fine_shapes[level_idx],
                                           output_array.data());

          in_cache_block::recompose<D, T, DeviceType>(finer, coarser,
                                                      local_coeff, queue_idx);
        }
      }

      // copy back
      SubArray<1, T, DeviceType> src({fine_num_elems[0]}, w_array.data());
      SubArray<1, T, DeviceType> dst({fine_num_elems[0]}, data.data());
      multi_dimension::CopyND(src, dst, queue_idx);
    }
  }

  std::vector<SIZE> coarse_shape;
  SIZE accumulated_local_coeff_size = 0;
  bool initialized;
  SIZE L;
  Hierarchy<D, T, DeviceType> *hierarchy;
  Config config;
  std::vector<SIZE> layer_len;
  // change off to offset
  std::vector<SIZE> layer_off;

  std::vector<SIZE> fine_num_elems;
  std::vector<SIZE> coarse_num_elems;
  std::vector<SIZE> local_coeff_size;
  std::vector<std::vector<SIZE>> coarse_shapes;
  std::vector<std::vector<SIZE>> fine_shapes;

  Array<1, T, DeviceType> w_array;
};

}  // namespace data_refactoring
}  // namespace mgard_x

#endif