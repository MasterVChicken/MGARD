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
    coarse_shape = hierarchy->level_shape(hierarchy->l_target());
    // for (int d = 0; d < coarse_shape.size(); d++) {
    //   log::info("Dim " + std::to_string(d) + " : " +
    //             std::to_string(coarse_shape[d]));
    // }

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
      // log::info("L = " + std::to_string(l) +
      //           ", fine_num_elems = " + std::to_string(fine_num_elems[l])
      //           +
      //           ", local_coeff_size = " +
      //           std::to_string(local_coeff_size[l]));
    }
  }

  void prepare_layers() {
    layer_len.assign(this->L + 1, 0);
    layer_off.assign(this->L + 1, 0);

    // The length of coarsest layer
    layer_len[0] = coarse_num_elems[this->L-1];
    layer_off[0] = 0;

    SIZE accum = layer_len[0];

    for (SIZE l = 1; l <= this->L; ++l) {
      layer_len[l] = local_coeff_size[this->L - l];
      layer_off[l] = accum;
      accum += layer_len[l];
    }
  }

  void Decompose(SubArray<D, T, DeviceType> data, int queue_idx) {
    // log::info("Size of fine_num_elems[0]: " +
    // std::to_string(fine_num_elems[0]));
    SubArray<1, T, DeviceType> decomposed_data({fine_num_elems[0]},
                                               w_array.data());
    // Create a copy for data
    SubArray<D, T, DeviceType> data_sub(fine_shapes[0], data.data());
    for (int d = 0; d < D; ++d) {
      data_sub.setLd(d, data.ld(d));
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
                                          decomposed_data((IDX)0));
        // The params sequence here is org, coarse, coeff, queue_idx
        in_cache_block::decompose<D, T, DeviceType>(data_sub, coarse,
                                                    local_coeff, queue_idx);
        // PrintSubarray("Data Sub: ", data_sub);
        // PrintSubarray("Coarse: ", coarse);
        // PrintSubarray("Local coeff: ", local_coeff);

        SubArray<D, T, DeviceType> tmp = coarse;
        if (l + 1 < this->L) {
          coarse = SubArray<D, T, DeviceType>(coarse_shapes[l + 1],
                                              decomposed_data((IDX)0));
        }
        data_sub = tmp;
      }
    }

    // Needs copy back
    SubArray<D, T, DeviceType> decomposed_data_ND(fine_shapes[0],
                                          decomposed_data((IDX)0));
    multi_dimension::CopyND(decomposed_data_ND, data, queue_idx);
    // PrintSubarray("data after CopyND:", data);
  }

  void Recompose(SubArray<D, T, DeviceType> data, int queue_idx) {
    SubArray<D, T, DeviceType> decomposed_array(fine_shapes[0], data.data());
    SubArray<D, T, DeviceType> recomposed_array(fine_shapes[0], w_array.data());

    if (this->L > 0) {
      SubArray<D, T, DeviceType> coarser(coarse_shapes[this->L-1], decomposed_array.data());
      SubArray<D, T, DeviceType> finer(fine_shapes[this->L - 1],
                                       decomposed_array.data());
      for (SIZE l = 0; l < this->L; l++) {
        SubArray<1, T, DeviceType> local_coeff(
            {layer_len[l + 1]}, decomposed_array((IDX)layer_off[l + 1]));

        in_cache_block::recompose<D, T, DeviceType>(finer, coarser, local_coeff,
                                                    queue_idx);
        coarser = finer;
        if (l + 1 < this->L) {
          finer = SubArray<D, T, DeviceType>(fine_shapes[l + 1],
                                             decomposed_array((IDX)0));
        }
      }
    }
    // PrintSubarray("Decomposed Array in Recompose():", decomposed_array);
    // multi_dimension::CopyND(recomposed_array, decomposed_array, queue_idx);
    // PrintSubarray("Decomposed Array in Recompose() after:", decomposed_array);
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