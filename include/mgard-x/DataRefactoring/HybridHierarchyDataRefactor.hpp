/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

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

    ComputeLocalShapes();
    SetupGlobalHierarchy();
    InitializeBuffers();
  }

  void Adapt(Hierarchy<D, T, DeviceType>& hierarchy, Config config,
             int queue_idx) {
    this->initialized = true;
    this->hierarchy = &hierarchy;
    this->config = config;

    this->L = config.num_local_refactoring_level;
    this->M = config.num_global_refactoring_level;

    ComputeLocalShapes();
    SetupGlobalHierarchy();
    InitializeBuffers(queue_idx);
  }

  void ComputeLocalShapes() {
    coarse_shape = hierarchy->level_shape(hierarchy->l_target());
    coarse_shapes.clear();
    fine_shapes.clear();
    coarse_num_elems.clear();
    fine_num_elems.clear();
    local_coeff_size.clear();

    for (int l = 0; l < this->L; ++l) {
      SIZE last_level_size = 1, curr_level_size = 1;
      std::vector<SIZE> fine_shape(D);

      for (DIM d = 0; d < D; ++d) {
        // 8 padding
        coarse_shape[d] = ((coarse_shape[d] - 1) / 8 + 1) * 8;
        last_level_size *= coarse_shape[d];
        fine_shape[d] = coarse_shape[d];

        // 8 -> 5
        coarse_shape[d] = ((coarse_shape[d] - 1) / 8 + 1) * 5;
        curr_level_size *= coarse_shape[d];
      }

      fine_shapes.push_back(fine_shape);
      coarse_shapes.push_back(coarse_shape);
      fine_num_elems.push_back(last_level_size);
      coarse_num_elems.push_back(curr_level_size);
      local_coeff_size.push_back(last_level_size - curr_level_size);

      // log::info("Local level " + std::to_string(l) +
      //          ": fine=" + std::to_string(last_level_size) +
      //          ", coarse=" + std::to_string(curr_level_size) +
      //          ", coeffs=" + std::to_string(last_level_size - curr_level_size));
    }
  }

  void SetupGlobalHierarchy() {
    // Inherent the hierarchy from the results of local
    if (this->M > 0 && this->L > 0) {
      global_hierarchy = Hierarchy<D, T, DeviceType>(coarse_shape, config);
      global_refactor =
          DataRefactor<D, T, DeviceType>(global_hierarchy, config);
    }
  }

  void InitializeBuffers(int queue_idx = 0) {
    // Allocate buffers for local decomposition
    if (this->L > 0) {
      coarse_buffers.resize(2);
      coarse_buffers[0] = Array<D, T, DeviceType>(fine_shapes[0]);
      coarse_buffers[1] = Array<D, T, DeviceType>(fine_shapes[0]);

      // Buffer for local coefficients
      size_t total_local_coeffs = 0;
      for (int l = 0; l < this->L; ++l) {
        total_local_coeffs += local_coeff_size[l];
      }
      // TODO: Check if we need this one here
      local_coeff_array = Array<1, T, DeviceType>({total_local_coeffs});

      // Temporary buffer for coarsest local data
      temp_coarsest = Array<1, T, DeviceType>({coarse_num_elems[this->L - 1]});
    }

    // Buffer for global coefficients (if M > 0)
    if (this->M > 0 && this->L > 0) {
      coarse_array = Array<D, T, DeviceType>(coarse_shape);
    }
  }

  // Need to check if this func needs re-write
  static size_t EstimateMemoryFootprint(std::vector<SIZE> shape) {
    size_t size = 0;
    return size;
  }

  size_t DecomposedDataSize() {
    size_t total_size = 0;

    // local
    for (int l = 0; l < this->L; l++) {
      total_size += local_coeff_size[l];
    }

    // global

    // Wondering how this two statements differ here
    // Think global_hierarchy.total_num_elems() should be same with
    // coarse_num_elems[this->L - 1] Maybe we can optimize by remove this
    // if-else if statement Just leave it here and ask mentor about it
    if (this->M > 0) {
      total_size += global_hierarchy.total_num_elems();
    } else if (this->L > 0) {
      // If we have no global but have local
      total_size += coarse_num_elems[this->L - 1];
    }

    return total_size;
  }

  size_t LocalCoeffSize() {
    size_t total = 0;
    for (int l = 0; l < this->L; ++l) {
      total += local_coeff_size[l];
    }
    return total;
  }

  void Decompose(SubArray<D, T, DeviceType> data,
                 SubArray<1, T, DeviceType> decomposed_data, int queue_idx) {
    Timer timer;

    // Local decomposition
    if (this->L > 0) {
      if (log::level & log::TIME) {
        DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
        timer.start();
      }

      LocalDecompose(data, decomposed_data, queue_idx);

      if (log::level & log::TIME) {
        DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
        timer.end();
        timer.print("Local Decomposition (L=" + std::to_string(this->L) + ")",
                    hierarchy->total_num_elems() * sizeof(T));
        timer.clear();
      }
    }

    // Global decomposition
    if (this->M > 0) {
      if (log::level & log::TIME) {
        DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
        timer.start();
      }

      GlobalDecompose(decomposed_data, queue_idx);

      if (log::level & log::TIME) {
        DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
        timer.end();
        timer.print("Global Decomposition (M=" + std::to_string(this->M) + ")",
                    coarse_num_elems[this->L - 1] * sizeof(T));
        timer.clear();
      }
    }

    // if (log::level & log::TIME) {
    //   DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    //   timer.end();
    //   timer.print("Decomposition");
    //   log::time(
    //       "Decomposition throughput: " +
    //       std::to_string((double)(hierarchy->total_num_elems() * sizeof(T)) /
    //                      timer.get() / 1e9) +
    //       " GB/s");
    //   timer.clear();
    // }
  }

  void LocalDecompose(SubArray<D, T, DeviceType> data,
                      SubArray<1, T, DeviceType> decomposed_data,
                      int queue_idx) {
    // Initialize fine buffer with input data
    SubArray<D, T, DeviceType> fine(coarse_buffers[1]);
    multi_dimension::CopyND(data, fine, queue_idx);

    SubArray<1, T, DeviceType> local_coeff_subarray(local_coeff_array);
    accumulated_local_coeff_size = 0;

    for (SIZE l = 0; l < this->L; ++l) {
      accumulated_local_coeff_size += local_coeff_size[l];

      // Get coefficient output location (stored from back to front)
      SubArray<1, T, DeviceType> local_coeff(
          {local_coeff_size[l]},
          local_coeff_subarray(local_coeff_subarray.shape(0) -
                               accumulated_local_coeff_size));

      int buffer_idx = l % 2;
      coarse_buffers[buffer_idx].memset(0, queue_idx);

      SubArray<D, T, DeviceType> coarse(coarse_shapes[l],
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

    // Copy coarsest data to temp buffer
    int final_buffer_idx = (this->L - 1) % 2;
    SubArray<D, T, DeviceType> coarsest(
        coarse_shapes[this->L - 1], coarse_buffers[final_buffer_idx].data());
    for (DIM d = 0; d < D; d++) {
      coarsest.setLd(d, fine_shapes[0][d]);
    }
    coarsest.project(0, 1, 2);

    SubArray<D, T, DeviceType> temp_coarsest_subarray(
        coarse_shapes[this->L - 1], temp_coarsest.data());
    multi_dimension::CopyND(coarsest, temp_coarsest_subarray, queue_idx);

    // Copy local coefficients to output
    // Layout: [global_data | local_coeffs]
    SIZE global_data_size = (this->M > 0) ? global_hierarchy.total_num_elems()
                                          : coarse_num_elems[this->L - 1];

    if (this->M == 0) {
      // No global decomposition, copy coarsest directly
      multi_dimension::CopyND(SubArray(temp_coarsest), decomposed_data,
                              queue_idx);
    }

    // Copy local coefficients after global data position
    SubArray<1, T, DeviceType> output_local_coeff(
        {LocalCoeffSize()}, decomposed_data.data() + global_data_size);
    multi_dimension::CopyND(local_coeff_subarray, output_local_coeff,
                            queue_idx);
  }

  void GlobalDecompose(SubArray<1, T, DeviceType> decomposed_data,
                       int queue_idx) {
    // Copy coarsest local data to coarse_array for global processing
    SubArray<D, T, DeviceType> coarse_data(coarse_shape, coarse_array.data());
    SubArray<D, T, DeviceType> temp_coarsest_subarray(
        coarse_shapes[this->L - 1], temp_coarsest.data());
    multi_dimension::CopyND(temp_coarsest_subarray, coarse_data, queue_idx);

    // Perform global decomposition in-place
    global_refactor.Decompose(coarse_data, true, queue_idx);

    // Copy result to output (beginning of decomposed_data)
    SubArray<D, T, DeviceType> global_coeff_output(
        global_hierarchy.level_shape(global_hierarchy.l_target()),
        decomposed_data.data());
    multi_dimension::CopyND(coarse_data, global_coeff_output, queue_idx);
  }

  void Recompose(SubArray<D, T, DeviceType> data,
                 SubArray<1, T, DeviceType> decomposed_data, int queue_idx) {
    Timer timer;

    SIZE global_data_size = (this->M > 0) ? global_hierarchy.total_num_elems()
                                          : coarse_num_elems[this->L - 1];

    // Global Recomposition
    if (this->M > 0) {
      if (log::level & log::TIME) {
        DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
        timer.start();
      }

      GlobalRecompose(decomposed_data, queue_idx);

      if (log::level & log::TIME) {
        DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
        timer.end();
        timer.print("Global Recomposition (M=" + std::to_string(this->M) + ")",
                    global_data_size * sizeof(T));
        timer.clear();
      }
    } else if (this->L > 0) {
      // Copy coarsest directly to temp buffer
      multi_dimension::CopyND(decomposed_data, SubArray(temp_coarsest),
                              queue_idx);
    }

    // Local Recomposition
    if (this->L > 0) {
      if (log::level & log::TIME) {
        DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
        timer.start();
      }

      LocalRecompose(data, decomposed_data, queue_idx);

      if (log::level & log::TIME) {
        DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
        timer.end();
        timer.print("Local Recomposition (L=" + std::to_string(this->L) + ")",
                    hierarchy->total_num_elems() * sizeof(T));
        timer.clear();
      }
    }

    // if (log::level & log::TIME) {
    //   DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    //   timer.end();
    //   timer.print("Recomposition");
    //   log::time(
    //       "Recomposition throughput: " +
    //       std::to_string((double)(hierarchy->total_num_elems() * sizeof(T)) /
    //                      timer.get() / 1e9) +
    //       " GB/s");
    //   timer.clear();
    // }
  }

  void GlobalRecompose(SubArray<1, T, DeviceType> decomposed_data,
                       int queue_idx) {
    // Copy global coefficients to coarse_array
    SubArray<D, T, DeviceType> coarse_data(coarse_shape, coarse_array.data());
    SubArray<D, T, DeviceType> global_coeff_input(
        global_hierarchy.level_shape(global_hierarchy.l_target()),
        decomposed_data.data());
    multi_dimension::CopyND(global_coeff_input, coarse_data, queue_idx);

    // Perform global recomposition in-place
    global_refactor.Recompose(coarse_data, true, queue_idx);

    // Copy result to temp_coarsest for local recomposition
    SubArray<D, T, DeviceType> temp_coarsest_subarray(
        coarse_shapes[this->L - 1], temp_coarsest.data());
    multi_dimension::CopyND(coarse_data, temp_coarsest_subarray, queue_idx);
  }

  void LocalRecompose(SubArray<D, T, DeviceType> data,
                      SubArray<1, T, DeviceType> decomposed_data,
                      int queue_idx) {
    // Clear buffers
    coarse_buffers[0].memset(0, queue_idx);
    coarse_buffers[1].memset(0, queue_idx);

    // Start with coarsest data
    SubArray<D, T, DeviceType> coarse(coarse_shapes[this->L - 1],
                                      temp_coarsest.data());

    SIZE global_data_size = (this->M > 0) ? global_hierarchy.total_num_elems()
                                          : coarse_num_elems[this->L - 1];

    // Process levels in reverse order (L-1 down to 0)
    for (SIZE l = 0; l < this->L; ++l) {
      SIZE level_idx = this->L - l - 1;

      // Get coefficient input location
      SubArray<1, T, DeviceType> local_coeff(
          {local_coeff_size[level_idx]},
          decomposed_data.data() + global_data_size +
              (LocalCoeffSize() - accumulated_local_coeff_size));

      // Setup fine buffer
      int buffer_idx = l % 2;

      SubArray<D, T, DeviceType> fine(fine_shapes[level_idx],
                                      coarse_buffers[buffer_idx].data());
      for (DIM d = 0; d < D; d++) {
        fine.setLd(d, fine_shapes[0][d]);
      }
      fine.project(0, 1, 2);

      // Perform recomposition
      in_cache_block::recompose<D, T, DeviceType>(fine, coarse, local_coeff,
                                                  queue_idx);

      // Update coarse for next iteration
      if (l < this->L - 1) {
        coarse = SubArray<D, T, DeviceType>(coarse_shapes[level_idx - 1],
                                            coarse_buffers[buffer_idx].data());
        for (DIM d = 0; d < D; d++) {
          coarse.setLd(d, fine_shapes[0][d]);
        }
        coarse.project(0, 1, 2);
      }

      accumulated_local_coeff_size -= local_coeff_size[level_idx];
    }

    // Copy final result to output
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
  }

  bool initialized;
  Hierarchy<D, T, DeviceType>* hierarchy;
  Hierarchy<D, T, DeviceType> global_hierarchy;
  Config config;

  SIZE L;  // Number of local levels
  SIZE M;  // Number of global levels

  std::vector<SIZE> coarse_shape;
  std::vector<std::vector<SIZE>> coarse_shapes;
  std::vector<std::vector<SIZE>> fine_shapes;
  std::vector<SIZE> coarse_num_elems;
  std::vector<SIZE> fine_num_elems;
  std::vector<SIZE> local_coeff_size;
  SIZE accumulated_local_coeff_size = 0;

  DataRefactor<D, T, DeviceType> global_refactor;
  Array<D, T, DeviceType> coarse_array;
  std::vector<Array<D, T, DeviceType>> coarse_buffers;
  Array<1, T, DeviceType> local_coeff_array;
  Array<1, T, DeviceType> temp_coarsest;
};

}  // namespace data_refactoring

}  // namespace mgard_x

#endif