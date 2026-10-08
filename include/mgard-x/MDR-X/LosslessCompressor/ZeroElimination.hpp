/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: October 8, 2026
 */

#ifndef MGARD_X_MDR_ZERO_ELIMINATION_HPP
#define MGARD_X_MDR_ZERO_ELIMINATION_HPP

#include "../../RuntimeX/RuntimeX.h"
#include <cstring>

namespace mgard_x {
namespace MDR {

// Zero elimination of encoded bitplane rows: a single-pass, data-parallel
// alternative to Huffman/RLE. A row of 32-bit words is split into chunks of
// 32 words; each chunk stores a 32-bit bitmap of its nonzero words (bit j for
// word j) and those words, packed. A row is stored raw when that is not
// smaller.
//
// Merged group stream (all fields uint32 unless noted, 4-byte aligned):
//   signature "MGXZELM" + 1 pad byte, num_rows, num_words,
//   count[num_rows]            nonzero words of each row, RAW_ROW if raw
//   then for each row: raw:     words[num_words]
//                      packed:  super_offsets[num_super] (first nonzero word
//                               of each super-chunk), bitmaps[num_chunks],
//                               words[count]
// A super-chunk is SUPER chunks (one thread block); its offset lets every
// block decode independently.
namespace zero_elimination {

static constexpr SIZE CHUNK = 32;  // words per bitmap
static constexpr SIZE SUPER = 256; // chunks per super-chunk (= block size)
static constexpr uint32_t RAW_ROW = 0xffffffffu;
static constexpr SIZE SIGNATURE_BYTES = 8;
inline const Byte *signature() {
  static const Byte s[SIGNATURE_BYTES] = {'M', 'G', 'X', 'Z', 'E', 'L', 'M', 0};
  return s;
}

MGARDX_CONT_EXEC uint32_t popcount32(uint32_t x) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return __popc(x);
#else
  x = x - ((x >> 1) & 0x55555555u);
  x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
  return (((x + (x >> 4)) & 0x0F0F0F0Fu) * 0x01010101u) >> 24;
#endif
}

MGARDX_CONT_EXEC SIZE num_chunks(SIZE num_words) {
  return (num_words + CHUNK - 1) / CHUNK;
}
MGARDX_CONT_EXEC SIZE num_super(SIZE num_words) {
  return (num_chunks(num_words) + SUPER - 1) / SUPER;
}
// Size in bytes of a row's section.
inline SIZE row_bytes(uint32_t count, SIZE num_words) {
  if (count == RAW_ROW) {
    return num_words * sizeof(uint32_t);
  }
  return (num_super(num_words) + num_chunks(num_words) + count) *
         sizeof(uint32_t);
}
inline SIZE header_bytes(SIZE num_rows) {
  return SIGNATURE_BYTES + 2 * sizeof(uint32_t) + num_rows * sizeof(uint32_t);
}

// Per-row description used by the write and decode kernels.
struct RowTask {
  uint64_t stream;  // device address of the row's group stream
  uint64_t section; // byte offset of the row's section in the stream
  uint32_t row;     // row of the encoded bitplanes
  uint32_t count;   // nonzero words, or RAW_ROW
  uint32_t index;   // row index within the group
  uint32_t group_rows;
};

// Nonzero words of each chunk and of each super-chunk of rows
// [0, num_rows) of the encoded bitplanes.
template <typename T_bitplane, typename DeviceType>
class CountFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT CountFunctor() {}
  MGARDX_CONT CountFunctor(SubArray<2, T_bitplane, DeviceType> rows,
                           SubArray<1, uint32_t, DeviceType> chunk_counts,
                           SubArray<1, uint32_t, DeviceType> super_counts)
      : rows(rows), chunk_counts(chunk_counts), super_counts(super_counts) {
    Functor<DeviceType>();
  }
  MGARDX_EXEC void Operation1() {
    sum = (uint32_t *)FunctorBase<DeviceType>::GetSharedMemory();
    tid = FunctorBase<DeviceType>::GetThreadIdX();
    if (tid == 0) {
      *sum = 0;
    }
  }
  MGARDX_EXEC void Operation2() {
    SIZE num_words = rows.shape(1);
    SIZE row = FunctorBase<DeviceType>::GetBlockIdY();
    SIZE chunk = FunctorBase<DeviceType>::GetBlockIdX() * SUPER + tid;
    if (chunk >= num_chunks(num_words)) {
      return;
    }
    const T_bitplane *w = rows(row, chunk * CHUNK);
    SIZE end = num_words - chunk * CHUNK < CHUNK ? num_words - chunk * CHUNK
                                                 : CHUNK;
    uint32_t count = 0;
    for (SIZE j = 0; j < end; j++) {
      count += w[j] != 0;
    }
    *chunk_counts(row * num_chunks(num_words) + chunk) = count;
    if (count) {
      Atomic<uint32_t, AtomicSharedMemory, AtomicDeviceScope, DeviceType>::Add(
          sum, count);
    }
  }
  MGARDX_EXEC void Operation3() {
    if (tid == 0) {
      SIZE row = FunctorBase<DeviceType>::GetBlockIdY();
      *super_counts(row * num_super(rows.shape(1)) +
                    FunctorBase<DeviceType>::GetBlockIdX()) = *sum;
    }
  }
  MGARDX_CONT size_t shared_memory_size() { return sizeof(uint32_t); }

private:
  SubArray<2, T_bitplane, DeviceType> rows;
  SubArray<1, uint32_t, DeviceType> chunk_counts, super_counts;
  uint32_t *sum;
  SIZE tid;
};

template <typename T_bitplane, typename DeviceType>
class CountKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "zero elimination count";
  using FunctorType = CountFunctor<T_bitplane, DeviceType>;
  MGARDX_CONT CountKernel(SubArray<2, T_bitplane, DeviceType> rows,
                          SIZE num_rows,
                          SubArray<1, uint32_t, DeviceType> chunk_counts,
                          SubArray<1, uint32_t, DeviceType> super_counts)
      : rows(rows), num_rows(num_rows), chunk_counts(chunk_counts),
        super_counts(super_counts) {}
  MGARDX_CONT Task<FunctorType> GenTask(int queue_idx) {
    FunctorType functor(rows, chunk_counts, super_counts);
    return Task(functor, 1, num_rows, num_super(rows.shape(1)), 1, 1, SUPER,
                functor.shared_memory_size(), queue_idx, std::string(Name));
  }

private:
  SubArray<2, T_bitplane, DeviceType> rows;
  SIZE num_rows;
  SubArray<1, uint32_t, DeviceType> chunk_counts, super_counts;
};

// Exclusive scan of the super-chunk counts of each row, and row totals.
template <typename DeviceType> class ScanFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT ScanFunctor() {}
  MGARDX_CONT ScanFunctor(SIZE num_super,
                          SubArray<1, uint32_t, DeviceType> super_counts,
                          SubArray<1, uint32_t, DeviceType> totals)
      : num_super(num_super), super_counts(super_counts), totals(totals) {
    Functor<DeviceType>();
  }
  MGARDX_EXEC void Operation1() {
    if (FunctorBase<DeviceType>::GetThreadIdX() != 0) {
      return;
    }
    SIZE row = FunctorBase<DeviceType>::GetBlockIdX();
    uint32_t sum = 0;
    for (SIZE s = 0; s < num_super; s++) {
      uint32_t c = *super_counts(row * num_super + s);
      *super_counts(row * num_super + s) = sum;
      sum += c;
    }
    *totals(row) = sum;
  }
  MGARDX_CONT size_t shared_memory_size() { return 0; }

private:
  SIZE num_super;
  SubArray<1, uint32_t, DeviceType> super_counts, totals;
};

template <typename DeviceType> class ScanKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "zero elimination scan";
  using FunctorType = ScanFunctor<DeviceType>;
  MGARDX_CONT ScanKernel(SIZE num_rows, SIZE num_super,
                         SubArray<1, uint32_t, DeviceType> super_counts,
                         SubArray<1, uint32_t, DeviceType> totals)
      : num_rows(num_rows), num_super(num_super), super_counts(super_counts),
        totals(totals) {}
  MGARDX_CONT Task<FunctorType> GenTask(int queue_idx) {
    FunctorType functor(num_super, super_counts, totals);
    return Task(functor, 1, 1, num_rows, 1, 1, 32, 0, queue_idx,
                std::string(Name));
  }

private:
  SIZE num_rows, num_super;
  SubArray<1, uint32_t, DeviceType> super_counts, totals;
};

// Writes each row's section (and the group headers) into the group streams.
template <typename T_bitplane, typename DeviceType>
class WriteFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT WriteFunctor() {}
  MGARDX_CONT WriteFunctor(SubArray<2, T_bitplane, DeviceType> rows,
                           SubArray<1, RowTask, DeviceType> tasks,
                           SubArray<1, uint32_t, DeviceType> chunk_counts,
                           SubArray<1, uint32_t, DeviceType> super_offsets)
      : rows(rows), tasks(tasks), chunk_counts(chunk_counts),
        super_offsets(super_offsets) {
    Functor<DeviceType>();
  }
  MGARDX_EXEC void Operation1() {
    prefix = (uint32_t *)FunctorBase<DeviceType>::GetSharedMemory();
    tid = FunctorBase<DeviceType>::GetThreadIdX();
    task = *tasks(FunctorBase<DeviceType>::GetBlockIdY());
    num_words = rows.shape(1);
    nchunks = num_chunks(num_words);
    nsuper = num_super(num_words);
    chunk = FunctorBase<DeviceType>::GetBlockIdX() * SUPER + tid;
    stream = (Byte *)task.stream;
    if (FunctorBase<DeviceType>::GetBlockIdX() == 0 && tid == 0) {
      uint32_t *header = (uint32_t *)(stream + SIGNATURE_BYTES);
      if (task.index == 0) {
        const Byte sig[SIGNATURE_BYTES] = {'M', 'G', 'X', 'Z', 'E', 'L', 'M', 0};
        for (int i = 0; i < (int)SIGNATURE_BYTES; i++) {
          stream[i] = sig[i];
        }
        header[0] = task.group_rows;
        header[1] = (uint32_t)num_words;
      }
      header[2 + task.index] = task.count;
    }
    if (task.count != RAW_ROW) {
      prefix[tid] = chunk < nchunks
                        ? *chunk_counts(task.row * nchunks + chunk)
                        : 0;
    }
  }
  MGARDX_EXEC void Operation2() {
    // Exclusive prefix of the chunk counts of this super-chunk.
    if (task.count != RAW_ROW && tid == 0) {
      uint32_t sum = 0;
      for (SIZE i = 0; i < SUPER; i++) {
        uint32_t c = prefix[i];
        prefix[i] = sum;
        sum += c;
      }
    }
  }
  MGARDX_EXEC void Operation3() {
    if (chunk >= nchunks) {
      return;
    }
    const T_bitplane *w = rows(task.row, chunk * CHUNK);
    SIZE end = num_words - chunk * CHUNK < CHUNK ? num_words - chunk * CHUNK
                                                 : CHUNK;
    uint32_t *section = (uint32_t *)(stream + task.section);
    if (task.count == RAW_ROW) {
      for (SIZE j = 0; j < end; j++) {
        section[chunk * CHUNK + j] = (uint32_t)w[j];
      }
      return;
    }
    uint32_t super_offset =
        *super_offsets(task.row * nsuper + FunctorBase<DeviceType>::GetBlockIdX());
    if (tid == 0) {
      section[FunctorBase<DeviceType>::GetBlockIdX()] = super_offset;
    }
    uint32_t *bitmaps = section + nsuper;
    uint32_t *words = bitmaps + nchunks + super_offset + prefix[tid];
    uint32_t bitmap = 0, k = 0;
    for (SIZE j = 0; j < end; j++) {
      if (w[j] != 0) {
        bitmap |= 1u << j;
        words[k++] = (uint32_t)w[j];
      }
    }
    bitmaps[chunk] = bitmap;
  }
  MGARDX_CONT size_t shared_memory_size() { return SUPER * sizeof(uint32_t); }

private:
  SubArray<2, T_bitplane, DeviceType> rows;
  SubArray<1, RowTask, DeviceType> tasks;
  SubArray<1, uint32_t, DeviceType> chunk_counts, super_offsets;
  uint32_t *prefix;
  RowTask task;
  Byte *stream;
  SIZE tid, num_words, nchunks, nsuper, chunk;
};

template <typename T_bitplane, typename DeviceType>
class WriteKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "zero elimination write";
  using FunctorType = WriteFunctor<T_bitplane, DeviceType>;
  MGARDX_CONT WriteKernel(SubArray<2, T_bitplane, DeviceType> rows,
                          SubArray<1, RowTask, DeviceType> tasks,
                          SIZE num_tasks,
                          SubArray<1, uint32_t, DeviceType> chunk_counts,
                          SubArray<1, uint32_t, DeviceType> super_offsets)
      : rows(rows), tasks(tasks), num_tasks(num_tasks),
        chunk_counts(chunk_counts), super_offsets(super_offsets) {}
  MGARDX_CONT Task<FunctorType> GenTask(int queue_idx) {
    FunctorType functor(rows, tasks, chunk_counts, super_offsets);
    return Task(functor, 1, num_tasks, num_super(rows.shape(1)), 1, 1, SUPER,
                functor.shared_memory_size(), queue_idx, std::string(Name));
  }

private:
  SubArray<2, T_bitplane, DeviceType> rows;
  SubArray<1, RowTask, DeviceType> tasks;
  SIZE num_tasks;
  SubArray<1, uint32_t, DeviceType> chunk_counts, super_offsets;
};

// Decodes row sections back into rows of the encoded bitplanes.
template <typename T_bitplane, typename DeviceType>
class DecodeFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT DecodeFunctor() {}
  MGARDX_CONT DecodeFunctor(SubArray<2, T_bitplane, DeviceType> rows,
                            SubArray<1, RowTask, DeviceType> tasks)
      : rows(rows), tasks(tasks) {
    Functor<DeviceType>();
  }
  MGARDX_EXEC void Operation1() {
    prefix = (uint32_t *)FunctorBase<DeviceType>::GetSharedMemory();
    tid = FunctorBase<DeviceType>::GetThreadIdX();
    task = *tasks(FunctorBase<DeviceType>::GetBlockIdY());
    num_words = rows.shape(1);
    nchunks = num_chunks(num_words);
    nsuper = num_super(num_words);
    chunk = FunctorBase<DeviceType>::GetBlockIdX() * SUPER + tid;
    section = (const uint32_t *)((const Byte *)task.stream + task.section);
    bitmap = 0;
    if (task.count != RAW_ROW) {
      bitmap = chunk < nchunks ? section[nsuper + chunk] : 0;
      prefix[tid] = popcount32(bitmap);
    }
  }
  MGARDX_EXEC void Operation2() {
    if (task.count != RAW_ROW && tid == 0) {
      uint32_t sum = 0;
      for (SIZE i = 0; i < SUPER; i++) {
        uint32_t c = prefix[i];
        prefix[i] = sum;
        sum += c;
      }
    }
  }
  MGARDX_EXEC void Operation3() {
    if (chunk >= nchunks) {
      return;
    }
    T_bitplane *out = rows(task.row, chunk * CHUNK);
    SIZE end = num_words - chunk * CHUNK < CHUNK ? num_words - chunk * CHUNK
                                                 : CHUNK;
    if (task.count == RAW_ROW) {
      for (SIZE j = 0; j < end; j++) {
        out[j] = section[chunk * CHUNK + j];
      }
      return;
    }
    const uint32_t *words = section + nsuper + nchunks +
                            section[FunctorBase<DeviceType>::GetBlockIdX()] +
                            prefix[tid];
    uint32_t k = 0;
    for (SIZE j = 0; j < end; j++) {
      out[j] = (bitmap >> j) & 1u ? words[k++] : 0;
    }
  }
  MGARDX_CONT size_t shared_memory_size() { return SUPER * sizeof(uint32_t); }

private:
  SubArray<2, T_bitplane, DeviceType> rows;
  SubArray<1, RowTask, DeviceType> tasks;
  uint32_t *prefix;
  RowTask task;
  const uint32_t *section;
  uint32_t bitmap;
  SIZE tid, num_words, nchunks, nsuper, chunk;
};

template <typename T_bitplane, typename DeviceType>
class DecodeKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "zero elimination decode";
  using FunctorType = DecodeFunctor<T_bitplane, DeviceType>;
  MGARDX_CONT DecodeKernel(SubArray<2, T_bitplane, DeviceType> rows,
                           SubArray<1, RowTask, DeviceType> tasks,
                           SIZE num_tasks)
      : rows(rows), tasks(tasks), num_tasks(num_tasks) {}
  MGARDX_CONT Task<FunctorType> GenTask(int queue_idx) {
    FunctorType functor(rows, tasks);
    return Task(functor, 1, num_tasks, num_super(rows.shape(1)), 1, 1, SUPER,
                functor.shared_memory_size(), queue_idx, std::string(Name));
  }

private:
  SubArray<2, T_bitplane, DeviceType> rows;
  SubArray<1, RowTask, DeviceType> tasks;
  SIZE num_tasks;
};

// Warp-cooperative kernels (32-lane sub-groups, CUDA): each warp of a
// super-chunk block handles 32 consecutive chunks, one word per lane, so that
// row loads and stores are coalesced; a chunk bitmap is one ballot. Same
// stream format as the kernels above.
static constexpr SIZE WARPS = SUPER / CHUNK; // warps per super-chunk block

// Inclusive prefix sum of c over the 32 lanes.
template <typename DeviceType>
MGARDX_EXEC uint32_t warp_inclusive_scan(SubGroup<DeviceType> &sg, int lane,
                                         uint32_t c) {
#pragma unroll
  for (int d = 1; d < (int)CHUNK; d *= 2) {
    uint32_t t = sg.shfl(c, lane >= d ? lane - d : lane);
    if (lane >= d) {
      c += t;
    }
  }
  return c;
}

// Chunk bitmaps of rows [0, num_rows) and nonzero words of each
// super-chunk.
template <typename T_bitplane, typename DeviceType>
class BitmapFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT BitmapFunctor() {}
  MGARDX_CONT BitmapFunctor(SubArray<2, T_bitplane, DeviceType> rows,
                            SubArray<1, uint32_t, DeviceType> bitmaps,
                            SubArray<1, uint32_t, DeviceType> super_counts)
      : rows(rows), bitmaps(bitmaps), super_counts(super_counts) {
    Functor<DeviceType>();
  }
  MGARDX_EXEC void Operation1() {
    SubGroup<DeviceType> sg;
    const int lane = sg.lane();
    sums = (uint32_t *)FunctorBase<DeviceType>::GetSharedMemory();
    SIZE warp = FunctorBase<DeviceType>::GetThreadIdX() / CHUNK;
    SIZE row = FunctorBase<DeviceType>::GetBlockIdY();
    SIZE num_words = rows.shape(1), nchunks = num_chunks(num_words);
    SIZE first = FunctorBase<DeviceType>::GetBlockIdX() * SUPER + warp * CHUNK;
    uint32_t mine = 0;
#pragma unroll 8
    for (int k = 0; k < (int)CHUNK; k++) {
      if (first + k >= nchunks) {
        break;
      }
      SIZE idx = (first + k) * CHUNK + lane;
      uint32_t bitmap = sg.ballot(idx < num_words && *rows(row, idx) != 0);
      if (lane == k) {
        mine = bitmap;
      }
    }
    if (first + lane < nchunks) {
      *bitmaps(row * nchunks + first + lane) = mine;
    }
    uint32_t c = popcount32(mine);
    for (int offset = 16; offset > 0; offset /= 2) {
      c += sg.shfl(c, lane ^ offset);
    }
    if (lane == 0) {
      sums[warp] = c;
    }
  }
  MGARDX_EXEC void Operation2() {
    if (FunctorBase<DeviceType>::GetThreadIdX() == 0) {
      uint32_t c = 0;
      for (SIZE w = 0; w < WARPS; w++) {
        c += sums[w];
      }
      *super_counts(FunctorBase<DeviceType>::GetBlockIdY() *
                        num_super(rows.shape(1)) +
                    FunctorBase<DeviceType>::GetBlockIdX()) = c;
    }
  }
  MGARDX_CONT size_t shared_memory_size() { return WARPS * sizeof(uint32_t); }

private:
  SubArray<2, T_bitplane, DeviceType> rows;
  SubArray<1, uint32_t, DeviceType> bitmaps, super_counts;
  uint32_t *sums;
};

template <typename T_bitplane, typename DeviceType>
class BitmapKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "zero elimination bitmap";
  using FunctorType = BitmapFunctor<T_bitplane, DeviceType>;
  MGARDX_CONT BitmapKernel(SubArray<2, T_bitplane, DeviceType> rows,
                           SIZE num_rows,
                           SubArray<1, uint32_t, DeviceType> bitmaps,
                           SubArray<1, uint32_t, DeviceType> super_counts)
      : rows(rows), num_rows(num_rows), bitmaps(bitmaps),
        super_counts(super_counts) {}
  MGARDX_CONT Task<FunctorType> GenTask(int queue_idx) {
    FunctorType functor(rows, bitmaps, super_counts);
    return Task(functor, 1, num_rows, num_super(rows.shape(1)), 1, 1, SUPER,
                functor.shared_memory_size(), queue_idx, std::string(Name));
  }

private:
  SubArray<2, T_bitplane, DeviceType> rows;
  SIZE num_rows;
  SubArray<1, uint32_t, DeviceType> bitmaps, super_counts;
};

// Nonzero words of each super-chunk from given chunk bitmaps (thread per
// chunk).
template <typename DeviceType>
class SuperCountFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT SuperCountFunctor() {}
  MGARDX_CONT SuperCountFunctor(SIZE nchunks,
                                SubArray<1, uint32_t, DeviceType> bitmaps,
                                SubArray<1, uint32_t, DeviceType> super_counts)
      : nchunks(nchunks), bitmaps(bitmaps), super_counts(super_counts) {
    Functor<DeviceType>();
  }
  MGARDX_EXEC void Operation1() {
    SubGroup<DeviceType> sg;
    sums = (uint32_t *)FunctorBase<DeviceType>::GetSharedMemory();
    SIZE tid = FunctorBase<DeviceType>::GetThreadIdX();
    SIZE row = FunctorBase<DeviceType>::GetBlockIdY();
    SIZE chunk = FunctorBase<DeviceType>::GetBlockIdX() * SUPER + tid;
    uint32_t c =
        chunk < nchunks ? popcount32(*bitmaps(row * nchunks + chunk)) : 0;
    for (int offset = 16; offset > 0; offset /= 2) {
      c += sg.shfl(c, sg.lane() ^ offset);
    }
    if (sg.lane() == 0) {
      sums[tid / CHUNK] = c;
    }
  }
  MGARDX_EXEC void Operation2() {
    if (FunctorBase<DeviceType>::GetThreadIdX() == 0) {
      uint32_t c = 0;
      for (SIZE w = 0; w < WARPS; w++) {
        c += sums[w];
      }
      SIZE nsuper = (nchunks + SUPER - 1) / SUPER;
      *super_counts(FunctorBase<DeviceType>::GetBlockIdY() * nsuper +
                    FunctorBase<DeviceType>::GetBlockIdX()) = c;
    }
  }
  MGARDX_CONT size_t shared_memory_size() { return WARPS * sizeof(uint32_t); }

private:
  SIZE nchunks;
  SubArray<1, uint32_t, DeviceType> bitmaps, super_counts;
  uint32_t *sums;
};

template <typename DeviceType> class SuperCountKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "zero elimination super count";
  using FunctorType = SuperCountFunctor<DeviceType>;
  MGARDX_CONT SuperCountKernel(SIZE num_rows, SIZE nchunks,
                               SubArray<1, uint32_t, DeviceType> bitmaps,
                               SubArray<1, uint32_t, DeviceType> super_counts)
      : num_rows(num_rows), nchunks(nchunks), bitmaps(bitmaps),
        super_counts(super_counts) {}
  MGARDX_CONT Task<FunctorType> GenTask(int queue_idx) {
    FunctorType functor(nchunks, bitmaps, super_counts);
    return Task(functor, 1, num_rows, (nchunks + SUPER - 1) / SUPER, 1, 1,
                SUPER, functor.shared_memory_size(), queue_idx,
                std::string(Name));
  }

private:
  SIZE num_rows, nchunks;
  SubArray<1, uint32_t, DeviceType> bitmaps, super_counts;
};

// Shared part of the warp write and decode functors: thread t of a block
// holds the bitmap of chunk t of the super-chunk and, after Operation2, the
// offset of its first nonzero word within the super-chunk.
template <typename DeviceType> struct WarpChunkScan {
  uint32_t *warp_totals;
  uint32_t bitmap, offset;
  MGARDX_EXEC void begin(SubGroup<DeviceType> &sg, int lane, SIZE warp,
                         uint32_t chunk_bitmap) {
    bitmap = chunk_bitmap;
    uint32_t c = popcount32(bitmap);
    uint32_t inclusive = warp_inclusive_scan(sg, lane, c);
    offset = inclusive - c;
    if (lane == (int)CHUNK - 1) {
      warp_totals[warp] = inclusive;
    }
  }
  MGARDX_EXEC void end(SIZE warp) {
    for (SIZE w = 0; w < warp; w++) {
      offset += warp_totals[w];
    }
  }
};

template <typename T_bitplane, typename DeviceType>
class WriteWarpFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT WriteWarpFunctor() {}
  MGARDX_CONT WriteWarpFunctor(SubArray<2, T_bitplane, DeviceType> rows,
                               SubArray<1, RowTask, DeviceType> tasks,
                               SubArray<1, uint32_t, DeviceType> bitmaps,
                               SubArray<1, uint32_t, DeviceType> super_offsets)
      : rows(rows), tasks(tasks), bitmaps(bitmaps),
        super_offsets(super_offsets) {
    Functor<DeviceType>();
  }
  MGARDX_EXEC void Operation1() {
    SubGroup<DeviceType> sg;
    lane = sg.lane();
    tid = FunctorBase<DeviceType>::GetThreadIdX();
    warp = tid / CHUNK;
    scan.warp_totals = (uint32_t *)FunctorBase<DeviceType>::GetSharedMemory();
    task = *tasks(FunctorBase<DeviceType>::GetBlockIdY());
    num_words = rows.shape(1);
    nchunks = num_chunks(num_words);
    nsuper = num_super(num_words);
    chunk = FunctorBase<DeviceType>::GetBlockIdX() * SUPER + tid;
    stream = (Byte *)task.stream;
    if (FunctorBase<DeviceType>::GetBlockIdX() == 0 && tid == 0) {
      uint32_t *header = (uint32_t *)(stream + SIGNATURE_BYTES);
      if (task.index == 0) {
        const Byte sig[SIGNATURE_BYTES] = {'M', 'G', 'X', 'Z', 'E', 'L', 'M', 0};
        for (int i = 0; i < (int)SIGNATURE_BYTES; i++) {
          stream[i] = sig[i];
        }
        header[0] = task.group_rows;
        header[1] = (uint32_t)num_words;
      }
      header[2 + task.index] = task.count;
    }
    if (task.count != RAW_ROW) {
      scan.begin(sg, lane, warp,
                 chunk < nchunks ? *bitmaps(task.row * nchunks + chunk) : 0);
    }
  }
  MGARDX_EXEC void Operation2() {
    SubGroup<DeviceType> sg;
    uint32_t *section = (uint32_t *)(stream + task.section);
    SIZE first = FunctorBase<DeviceType>::GetBlockIdX() * SUPER + warp * CHUNK;
    if (task.count == RAW_ROW) {
      for (SIZE k = 0; k < CHUNK && first + k < nchunks; k++) {
        SIZE idx = (first + k) * CHUNK + lane;
        if (idx < num_words) {
          section[idx] = (uint32_t)*rows(task.row, idx);
        }
      }
      return;
    }
    scan.end(warp);
    uint32_t super_offset = *super_offsets(
        task.row * nsuper + FunctorBase<DeviceType>::GetBlockIdX());
    if (tid == 0) {
      section[FunctorBase<DeviceType>::GetBlockIdX()] = super_offset;
    }
    uint32_t *bitmap_out = section + nsuper;
    if (chunk < nchunks) {
      bitmap_out[chunk] = scan.bitmap;
    }
    uint32_t *words = bitmap_out + nchunks + super_offset;
    const uint32_t below = (1u << lane) - 1;
    for (SIZE k = 0; k < CHUNK && first + k < nchunks; k++) {
      uint32_t bitmap = sg.shfl(scan.bitmap, (int)k);
      uint32_t offset = sg.shfl(scan.offset, (int)k);
      if ((bitmap >> lane) & 1u) {
        words[offset + popcount32(bitmap & below)] =
            (uint32_t)*rows(task.row, (first + k) * CHUNK + lane);
      }
    }
  }
  MGARDX_CONT size_t shared_memory_size() { return WARPS * sizeof(uint32_t); }

private:
  SubArray<2, T_bitplane, DeviceType> rows;
  SubArray<1, RowTask, DeviceType> tasks;
  SubArray<1, uint32_t, DeviceType> bitmaps, super_offsets;
  WarpChunkScan<DeviceType> scan;
  RowTask task;
  Byte *stream;
  int lane;
  SIZE tid, warp, num_words, nchunks, nsuper, chunk;
};

template <typename T_bitplane, typename DeviceType>
class WriteWarpKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "zero elimination write";
  using FunctorType = WriteWarpFunctor<T_bitplane, DeviceType>;
  MGARDX_CONT WriteWarpKernel(SubArray<2, T_bitplane, DeviceType> rows,
                              SubArray<1, RowTask, DeviceType> tasks,
                              SIZE num_tasks,
                              SubArray<1, uint32_t, DeviceType> bitmaps,
                              SubArray<1, uint32_t, DeviceType> super_offsets)
      : rows(rows), tasks(tasks), num_tasks(num_tasks), bitmaps(bitmaps),
        super_offsets(super_offsets) {}
  MGARDX_CONT Task<FunctorType> GenTask(int queue_idx) {
    FunctorType functor(rows, tasks, bitmaps, super_offsets);
    return Task(functor, 1, num_tasks, num_super(rows.shape(1)), 1, 1, SUPER,
                functor.shared_memory_size(), queue_idx, std::string(Name));
  }

private:
  SubArray<2, T_bitplane, DeviceType> rows;
  SubArray<1, RowTask, DeviceType> tasks;
  SIZE num_tasks;
  SubArray<1, uint32_t, DeviceType> bitmaps, super_offsets;
};

template <typename T_bitplane, typename DeviceType>
class DecodeWarpFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT DecodeWarpFunctor() {}
  MGARDX_CONT DecodeWarpFunctor(SubArray<2, T_bitplane, DeviceType> rows,
                                SubArray<1, RowTask, DeviceType> tasks)
      : rows(rows), tasks(tasks) {
    Functor<DeviceType>();
  }
  MGARDX_EXEC void Operation1() {
    SubGroup<DeviceType> sg;
    lane = sg.lane();
    tid = FunctorBase<DeviceType>::GetThreadIdX();
    warp = tid / CHUNK;
    scan.warp_totals = (uint32_t *)FunctorBase<DeviceType>::GetSharedMemory();
    task = *tasks(FunctorBase<DeviceType>::GetBlockIdY());
    num_words = rows.shape(1);
    nchunks = num_chunks(num_words);
    nsuper = num_super(num_words);
    chunk = FunctorBase<DeviceType>::GetBlockIdX() * SUPER + tid;
    section = (const uint32_t *)((const Byte *)task.stream + task.section);
    if (task.count != RAW_ROW) {
      scan.begin(sg, lane, warp, chunk < nchunks ? section[nsuper + chunk] : 0);
    }
  }
  MGARDX_EXEC void Operation2() {
    SubGroup<DeviceType> sg;
    SIZE first = FunctorBase<DeviceType>::GetBlockIdX() * SUPER + warp * CHUNK;
    if (task.count == RAW_ROW) {
      for (SIZE k = 0; k < CHUNK && first + k < nchunks; k++) {
        SIZE idx = (first + k) * CHUNK + lane;
        if (idx < num_words) {
          *rows(task.row, idx) = section[idx];
        }
      }
      return;
    }
    scan.end(warp);
    const uint32_t *words = section + nsuper + nchunks +
                            section[FunctorBase<DeviceType>::GetBlockIdX()];
    const uint32_t below = (1u << lane) - 1;
    for (SIZE k = 0; k < CHUNK && first + k < nchunks; k++) {
      uint32_t bitmap = sg.shfl(scan.bitmap, (int)k);
      uint32_t offset = sg.shfl(scan.offset, (int)k);
      SIZE idx = (first + k) * CHUNK + lane;
      if (idx < num_words) {
        *rows(task.row, idx) = (bitmap >> lane) & 1u
                                   ? words[offset + popcount32(bitmap & below)]
                                   : 0;
      }
    }
  }
  MGARDX_CONT size_t shared_memory_size() { return WARPS * sizeof(uint32_t); }

private:
  SubArray<2, T_bitplane, DeviceType> rows;
  SubArray<1, RowTask, DeviceType> tasks;
  WarpChunkScan<DeviceType> scan;
  RowTask task;
  const uint32_t *section;
  int lane;
  SIZE tid, warp, num_words, nchunks, nsuper, chunk;
};

template <typename T_bitplane, typename DeviceType>
class DecodeWarpKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "zero elimination decode";
  using FunctorType = DecodeWarpFunctor<T_bitplane, DeviceType>;
  MGARDX_CONT DecodeWarpKernel(SubArray<2, T_bitplane, DeviceType> rows,
                               SubArray<1, RowTask, DeviceType> tasks,
                               SIZE num_tasks)
      : rows(rows), tasks(tasks), num_tasks(num_tasks) {}
  MGARDX_CONT Task<FunctorType> GenTask(int queue_idx) {
    FunctorType functor(rows, tasks);
    return Task(functor, 1, num_tasks, num_super(rows.shape(1)), 1, 1, SUPER,
                functor.shared_memory_size(), queue_idx, std::string(Name));
  }

private:
  SubArray<2, T_bitplane, DeviceType> rows;
  SubArray<1, RowTask, DeviceType> tasks;
  SIZE num_tasks;
};

} // namespace zero_elimination
} // namespace MDR
} // namespace mgard_x

#endif
