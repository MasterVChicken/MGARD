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
  BlockLocalHierarchyDataRefactor(Hierarchy<D, T, DeviceType>& hierarchy,
                                  Config config)
      : initialized(true), hierarchy(&hierarchy), config(config) {
    this->L = config.num_local_refactoring_level;
    compute_local_ranges();

    w_array = Array<1, T, DeviceType>({DecomposedCoeffSize()});
    temp_coarest = Array<1, T, DeviceType>({coarse_num_elems[this->L - 1]});
    if (this->L > 0) {
      coarse_buffers.resize(2);
      coarse_buffers[0] = Array<D, T, DeviceType>(fine_shapes[0]);
      coarse_buffers[1] = Array<D, T, DeviceType>(fine_shapes[0]);
    }
  }

  void Adapt(Hierarchy<D, T, DeviceType>& hierarchy, Config config,
             int queue_idx) {
    this->initialized = true;
    this->hierarchy = &hierarchy;
    this->config = config;
    this->L = config.num_local_refactoring_level;
    compute_local_ranges();

    w_array.resize({DecomposedCoeffSize()}, queue_idx);
    temp_coarest.resize({coarse_num_elems[this->L - 1]}, queue_idx);
    if (this->L > 0) {
      coarse_buffers.resize(2);
      coarse_buffers[0].resize(fine_shapes[0], queue_idx);
      coarse_buffers[1].resize(fine_shapes[0], queue_idx);
    }
  }

  // Should be carefully re-write
  static size_t EstimateMemoryFootprint(std::vector<SIZE> shape) {
    size_t size = 3;
    for (DIM d = 0; d < shape.size(); d++) {
      int dim8 = ((shape[d] - 1) / 8 + 1) * 8;
      size *= dim8;
    }
    return size * sizeof(T);
  }

  size_t DecomposedDataSize() {
    SIZE decomposed_size = coarse_num_elems[this->L - 1];
    for (SIZE l = 0; l < this->L; l++) {
      decomposed_size += local_coeff_size[l];
    }
    return decomposed_size;
  }

  size_t DecomposedCoeffSize() {
    SIZE decomposed_coeff_size = 0;
    for (SIZE l = 0; l < this->L; l++) {
      decomposed_coeff_size += local_coeff_size[l];
    }
    return decomposed_coeff_size;
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
        fine_shape[d] = coarse_shape[d];
        coarse_shape[d] = ((coarse_shape[d] - 1) / 8 + 1) * 5;
        curr_level_size *= coarse_shape[d];
        // log::info("L: " + std::to_string(l) + ", DIM: " + std::to_string(d) +
        //           ", Fine: " + std::to_string(fine_shape[d]) +
        //           ", Coarse: " + std::to_string(coarse_shape[d]));
      }
      fine_num_elems.push_back(last_level_size);
      coarse_num_elems.push_back(curr_level_size);
      local_coeff_size.push_back(last_level_size - curr_level_size);
      // log::info("L: " + std::to_string(l) + ", Local coeff_size: " +
      //           std::to_string(last_level_size - curr_level_size));
      coarse_shapes.push_back(coarse_shape);
      fine_shapes.push_back(fine_shape);
    }
  }

  void Decompose(SubArray<D, T, DeviceType> data,
                 SubArray<1, T, DeviceType> output_decomposed, int queue_idx) {
    Timer timer;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }

    SubArray<D, T, DeviceType> fine(coarse_buffers[1]);
    SubArray<D, T, DeviceType> coarse;
    // CopyND follows the shape of 1st param
    multi_dimension::CopyND(data, fine, queue_idx);
    SubArray<1, T, DeviceType> decomposed_coeff(w_array);

    if (this->L > 0) {
      // Will be reused between decompose and recompose
      accumulated_local_coeff_size = 0;
      for (SIZE l = 0; l < this->L; l++) {
        accumulated_local_coeff_size += local_coeff_size[l];
        // Think about a way to change this local_coeff
        SubArray<1, T, DeviceType> local_coeff(
            {local_coeff_size[l]},
            decomposed_coeff(decomposed_coeff.shape(0) -
                             accumulated_local_coeff_size));

        int buffer_idx = l % 2;
        coarse_buffers[buffer_idx].memset(0, queue_idx);
        coarse = SubArray<D, T, DeviceType>(coarse_shapes[l],
                                          coarse_buffers[buffer_idx].data());
        for (DIM d = 0; d < D; d++) {
          coarse.setLd(d, fine_shapes[0][d]);
        }
        coarse.project(0, 1, 2);

        in_cache_block::decompose<D, T, DeviceType>(fine, coarse, local_coeff,
                                                    queue_idx);
        if (l < this->L - 1) {
          fine = SubArray<D, T, DeviceType>(fine_shapes[l + 1],
                                            coarse_buffers[buffer_idx].data());
          for (DIM d = 0; d < D; d++) {
            fine.setLd(d, fine_shapes[0][d]);
          }
          fine.project(0, 1, 2);
        }
      }
    }

    int final_buffer_id = (this->L - 1) % 2;
    SubArray<D, T, DeviceType> coarsest(coarse_shapes[this->L - 1],
                                        coarse_buffers[final_buffer_id].data());
    for (DIM d = 0; d < D; d++) {
      coarsest.setLd(d, fine_shapes[0][d]);
    }
    coarsest.project(0, 1, 2);
    // log::info("Find read buffer idx: " + std::to_string(final_buffer_id));

    SubArray<D, T, DeviceType> temp_coarest_subarray(coarse_shapes[this->L - 1],
                                                     temp_coarest.data());
    multi_dimension::CopyND(coarsest, temp_coarest_subarray, queue_idx);
    multi_dimension::CopyND(SubArray(temp_coarest), output_decomposed,
                            queue_idx);

    SubArray<1, T, DeviceType> data_coeff(
        {DecomposedCoeffSize()},
        output_decomposed.data() + coarse_num_elems[this->L - 1]);
    multi_dimension::CopyND(decomposed_coeff, data_coeff, queue_idx);

    // PrintSubarray("Temp in decompose:",SubArray(temp_coarest));

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("Decomposition", hierarchy->total_num_elems() * sizeof(T));
      timer.clear();
    }
  }

  void Recompose(SubArray<D, T, DeviceType> data,
                 SubArray<1, T, DeviceType> input_decomposed, int queue_idx) {
    Timer timer;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }

    // PrintSubarray("Temp in recompose:",SubArray(temp_coarest));

    coarse_buffers[0].memset(0, queue_idx);
    coarse_buffers[1].memset(0, queue_idx);

    SubArray<D, T, DeviceType> coarse(coarse_shapes[this->L - 1],
                                      temp_coarest.data());

    if (this->L > 0) {
      for (SIZE l = 0; l < this->L; l++) {
        SIZE level_idx = this->L - l - 1;

        SubArray<1, T, DeviceType> local_coeff(
            {local_coeff_size[level_idx]},
            input_decomposed(input_decomposed.shape(0) -
                             accumulated_local_coeff_size));

        int buffer_idx = l % 2;

        SubArray<D, T, DeviceType> fine(fine_shapes[level_idx],
                                        coarse_buffers[buffer_idx].data());
        for (DIM d = 0; d < D; d++) {
          fine.setLd(d, fine_shapes[0][d]);
        }
        fine.project(0, 1, 2);
        // log::info("Buffer idx for fine buffer: " +
        // std::to_string(buffer_idx));

        in_cache_block::recompose<D, T, DeviceType>(fine, coarse, local_coeff,
                                                    queue_idx);

        // Implementation 1
        // coarse = fine;

        // Implementation 2
        if (l < this->L - 1) {
          coarse = SubArray<D, T, DeviceType>(coarse_shapes[level_idx - 1],
                                            coarse_buffers[buffer_idx].data());
          for (DIM d = 0; d < D; d++) {
            coarse.setLd(d, fine_shapes[0][d]);
          }
          coarse.project(0, 1, 2);
        }

        // Implementation 3
        // if (l < this->L - 1) {
        //   SubArray<D, T, DeviceType> coarse_temp = fine;
        //   coarse = SubArray<D, T, DeviceType>(coarse_shapes[level_idx - 1],
        //                                       coarse_temp.data());

        //   for (DIM d = 0; d < D; d++) {
        //     coarse.setLd(d, fine_shapes[0][d]);
        //   }
        //   coarse.project(0, 1, 2);
        // }

        accumulated_local_coeff_size -= local_coeff_size[level_idx];
      }

      // copy back, using ND
      SubArray<D, T, DeviceType> src(
          hierarchy->level_shape(hierarchy->l_target()),
          coarse_buffers[(this->L - 1) % 2].data());

      for (DIM d = 0; d < D; d++) {
        src.setLd(d, fine_shapes[0][d]);
      }
      src.project(0, 1, 2);

      SubArray<D, T, DeviceType> dst(
          hierarchy->level_shape(hierarchy->l_target()), data.data());

      multi_dimension::CopyND(src, dst, queue_idx);

      if (log::level & log::TIME) {
        DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
        timer.end();
        timer.print("Recomposition", hierarchy->total_num_elems() * sizeof(T));
        timer.clear();
      }
    }
  }

  std::vector<SIZE> coarse_shape;
  SIZE accumulated_local_coeff_size = 0;
  bool initialized;
  SIZE L;
  Hierarchy<D, T, DeviceType>* hierarchy;
  Config config;

  std::vector<SIZE> fine_num_elems;
  std::vector<SIZE> coarse_num_elems;
  std::vector<SIZE> local_coeff_size;
  std::vector<std::vector<SIZE>> coarse_shapes;
  std::vector<std::vector<SIZE>> fine_shapes;
  std::vector<SIZE> original_input_shape;
  std::vector<SIZE> padded_input_shape;

  std::vector<Array<D, T, DeviceType>> coarse_buffers;

  Array<1, T, DeviceType> w_array;
  Array<1, T, DeviceType> temp_coarest;
};

}  // namespace data_refactoring
}  // namespace mgard_x

#endif