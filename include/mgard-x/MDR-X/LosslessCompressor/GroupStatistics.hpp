/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: October 7, 2026
 */

#ifndef MGARD_X_MDR_GROUP_STATISTICS_HPP
#define MGARD_X_MDR_GROUP_STATISTICS_HPP

#include "../../RuntimeX/RuntimeX.h"

namespace mgard_x {
namespace MDR {

// Statistics that decide how each merged bitplane group of a level is
// compressed, for all groups of the level in one pass: the number of runs
// RunLengthEncoding would produce (a run starts at a group's first byte, at
// every MAX_RUN-th byte of the group, and at every change of byte) and the
// byte histogram Huffman needs. A group is rows [first_row, first_row + rows)
// of the encoded bitplanes, read as bytes; blockIdx.y selects the group and
// blockIdx.x a chunk of it, so each block keeps one shared histogram.
template <typename T_bitplane, typename DeviceType>
class GroupStatisticsFunctor : public Functor<DeviceType> {
public:
  static constexpr SIZE BLOCK = 256;
  static constexpr SIZE WORDS_PER_THREAD = 16;
  static constexpr SIZE WORDS_PER_BLOCK = BLOCK * WORDS_PER_THREAD;
  static constexpr int REPLICAS = 8;
  static constexpr int BINS = 256;

  MGARDX_CONT GroupStatisticsFunctor() {}
  MGARDX_CONT GroupStatisticsFunctor(SubArray<2, T_bitplane, DeviceType> data,
                                     SIZE num_bitplanes, SIZE group_size,
                                     SIZE sign_rows, SIZE max_run,
                                     SubArray<1, unsigned int, DeviceType> runs,
                                     SubArray<1, unsigned int, DeviceType> freq)
      : data(data), num_bitplanes(num_bitplanes), group_size(group_size),
        sign_rows(sign_rows), max_run(max_run), runs(runs), freq(freq) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    hist = (unsigned int *)FunctorBase<DeviceType>::GetSharedMemory();
    block_counts = hist + REPLICAS * BINS;
    tid = FunctorBase<DeviceType>::GetThreadIdX();
    for (SIZE i = tid; i < REPLICAS * BINS; i += BLOCK) {
      hist[i] = 0;
    }
    if (tid < 2) {
      block_counts[tid] = 0;
    }
    group = FunctorBase<DeviceType>::GetBlockIdY();
    SIZE b = group * group_size;
    SIZE first_row = b == 0 ? 0 : b + sign_rows;
    SIZE rows =
        (group_size < num_bitplanes - b ? group_size : num_bitplanes - b) +
        (b == 0 ? sign_rows : 0);
    SIZE words_per_row = data.shape(1);
    group_words = rows * words_per_row;
    base = (T_bitplane *)data(first_row, 0);
  }

  MGARDX_EXEC void Operation2() {
    unsigned int zeros = 0, run_starts = 0;
    unsigned int *my_hist = hist + (tid % REPLICAS) * BINS;
    SIZE chunk = FunctorBase<DeviceType>::GetBlockIdX() * WORDS_PER_BLOCK;
    for (SIZE k = 0; k < WORDS_PER_THREAD; k++) {
      SIZE w = chunk + k * BLOCK + tid;
      if (w >= group_words) {
        break;
      }
      T_bitplane word = base[w];
      unsigned int prev =
          w > 0 ? (unsigned int)(base[w - 1] >> (8 * (sizeof(T_bitplane) - 1)))
                : 0;
      for (int j = 0; j < (int)sizeof(T_bitplane); j++) {
        unsigned int cur = (unsigned int)((word >> (8 * j)) & 0xff);
        SIZE i = w * sizeof(T_bitplane) + j;
        if (i == 0 || i % max_run == 0 || cur != prev) {
          run_starts++;
        }
        if (cur == 0) {
          zeros++;
        } else {
          Atomic<unsigned int, AtomicSharedMemory, AtomicDeviceScope,
                 DeviceType>::Add(&my_hist[cur], 1u);
        }
        prev = cur;
      }
    }
    if (zeros) {
      Atomic<unsigned int, AtomicSharedMemory, AtomicDeviceScope,
             DeviceType>::Add(&block_counts[0], zeros);
    }
    if (run_starts) {
      Atomic<unsigned int, AtomicSharedMemory, AtomicDeviceScope,
             DeviceType>::Add(&block_counts[1], run_starts);
    }
  }

  MGARDX_EXEC void Operation3() {
    for (SIZE s = tid; s < BINS; s += BLOCK) {
      unsigned int sum = s == 0 ? block_counts[0] : 0;
      for (int r = 0; r < REPLICAS; r++) {
        sum += hist[r * BINS + s];
      }
      if (sum) {
        Atomic<unsigned int, AtomicGlobalMemory, AtomicDeviceScope,
               DeviceType>::Add(freq(group * BINS + s), sum);
      }
    }
    if (tid == 0 && block_counts[1]) {
      Atomic<unsigned int, AtomicGlobalMemory, AtomicDeviceScope,
             DeviceType>::Add(runs(group), block_counts[1]);
    }
  }

  MGARDX_CONT size_t shared_memory_size() {
    return (REPLICAS * BINS + 2) * sizeof(unsigned int);
  }

private:
  SubArray<2, T_bitplane, DeviceType> data;
  SIZE num_bitplanes, group_size, sign_rows, max_run;
  SubArray<1, unsigned int, DeviceType> runs, freq;
  unsigned int *hist, *block_counts;
  SIZE tid, group, group_words;
  T_bitplane *base;
};

template <typename T_bitplane, typename DeviceType>
class GroupStatisticsKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "mdr group statistics";
  using FunctorType = GroupStatisticsFunctor<T_bitplane, DeviceType>;

  MGARDX_CONT GroupStatisticsKernel(SubArray<2, T_bitplane, DeviceType> data,
                                    SIZE num_bitplanes, SIZE group_size,
                                    SIZE sign_rows, SIZE max_run,
                                    SIZE num_groups,
                                    SubArray<1, unsigned int, DeviceType> runs,
                                    SubArray<1, unsigned int, DeviceType> freq)
      : data(data), num_bitplanes(num_bitplanes), group_size(group_size),
        sign_rows(sign_rows), max_run(max_run), num_groups(num_groups),
        runs(runs), freq(freq) {}

  MGARDX_CONT Task<FunctorType> GenTask(int queue_idx) {
    FunctorType functor(data, num_bitplanes, group_size, sign_rows, max_run,
                        runs, freq);
    // The largest group: group 0 with the sign rows, or a full group.
    SIZE max_rows =
        (group_size < num_bitplanes ? group_size : num_bitplanes) + sign_rows;
    SIZE max_words = max_rows * data.shape(1);
    SIZE gridx = (max_words + FunctorType::WORDS_PER_BLOCK - 1) /
                 FunctorType::WORDS_PER_BLOCK;
    return Task(functor, 1, num_groups, gridx, 1, 1, FunctorType::BLOCK,
                functor.shared_memory_size(), queue_idx, std::string(Name));
  }

private:
  SubArray<2, T_bitplane, DeviceType> data;
  SIZE num_bitplanes, group_size, sign_rows, max_run, num_groups;
  SubArray<1, unsigned int, DeviceType> runs, freq;
};

} // namespace MDR
} // namespace mgard_x

#endif
