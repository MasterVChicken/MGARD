/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: September 21, 2026
 */

#ifndef MGARD_X_PARALLEL_RLE_COUNT_RUNS_HPP
#define MGARD_X_PARALLEL_RLE_COUNT_RUNS_HPP

#include "../../RuntimeX/RuntimeX.h"

namespace mgard_x {

namespace parallel_rle {

// Number of runs RunLengthEncoding would produce, without materializing the
// start marks: the same run starts as StartMarksFunctor (index 0, every
// MAX_RUN-th index, and every change of symbol), counted per thread.
template <typename T_symbol, typename C_run, typename DeviceType>
class CountRunsFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT CountRunsFunctor() {}
  MGARDX_CONT CountRunsFunctor(SubArray<1, T_symbol, DeviceType> data,
                               SubArray<1, SIZE, DeviceType> partial_counts)
      : data(data), partial_counts(partial_counts) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    IDX start = FunctorBase<DeviceType>::GetBlockIdX() *
                    FunctorBase<DeviceType>::GetBlockDimX() +
                FunctorBase<DeviceType>::GetThreadIdX();
    IDX n = data.shape(0);
    IDX grid_size = FunctorBase<DeviceType>::GetGridDimX() *
                    FunctorBase<DeviceType>::GetBlockDimX();
    IDX MAX_RUN = (IDX)1 << (sizeof(C_run) * 8);
    SIZE count = 0;
    for (IDX i = start; i < n; i += grid_size) {
      if (i == 0 || i % MAX_RUN == 0 || *data(i) != *data(i - 1)) {
        count++;
      }
    }
    *partial_counts(start) = count;
  }

  MGARDX_CONT size_t shared_memory_size() { return 0; }

private:
  SubArray<1, T_symbol, DeviceType> data;
  SubArray<1, SIZE, DeviceType> partial_counts;
};

template <typename T_symbol, typename C_run, typename DeviceType>
class CountRunsKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "rle count runs";
  static constexpr SIZE BLOCK = 256;
  static constexpr SIZE GRID = 132 * 8;
  static constexpr SIZE NUM_PARTIALS = BLOCK * GRID;

  MGARDX_CONT CountRunsKernel(SubArray<1, T_symbol, DeviceType> data,
                              SubArray<1, SIZE, DeviceType> partial_counts)
      : data(data), partial_counts(partial_counts) {}

  MGARDX_CONT Task<CountRunsFunctor<T_symbol, C_run, DeviceType>>
  GenTask(int queue_idx) {
    using FunctorType = CountRunsFunctor<T_symbol, C_run, DeviceType>;
    FunctorType functor(data, partial_counts);
    return Task(functor, 1, 1, GRID, 1, 1, BLOCK, functor.shared_memory_size(),
                queue_idx, std::string(Name));
  }

private:
  SubArray<1, T_symbol, DeviceType> data;
  SubArray<1, SIZE, DeviceType> partial_counts;
};

} // namespace parallel_rle

} // namespace mgard_x

#endif
