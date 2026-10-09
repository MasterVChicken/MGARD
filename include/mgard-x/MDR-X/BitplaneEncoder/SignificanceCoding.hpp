/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: October 8, 2026
 */

#ifndef MGARD_X_MDR_SIGNIFICANCE_CODING_HPP
#define MGARD_X_MDR_SIGNIFICANCE_CODING_HPP

#include "../../RuntimeX/RuntimeX.h"
#include "../LosslessCompressor/ZeroElimination.hpp"

namespace mgard_x {
namespace MDR {

// Significance-coded signs (format version 4; binary encoding, contiguous
// words, zero elimination). Instead of a sign row in the first bitplane
// group, the sign of a coefficient is stored in the group where it becomes
// significant (the group of its first nonzero bit), so a retrieval reads the
// signs of the coefficients it reconstructs as nonzero and no others.
//
// A tile is TILE words (1024 coefficients, one zero-elimination chunk) and a
// segment SEGMENT tiles. The sign section that ends the stream of group g
// holds one bit per coefficient that becomes significant in g, in
// coefficient order:
//   offsets[num_segments]   first word of each segment's bits (in words[])
//   words[]                 the bits; each segment starts at a word boundary
// so that segments decode independently.
//
// Encoder output: row 0 of the encoded bitplanes (the sign row otherwise)
// holds, in the words of each tile, the tile's sign bits group by group and
// in coefficient order within a group; counts(g, tile) is the number of bits
// of groups 0 to g in the tile (where group g's bits end) and
// segment_bits(g, segment) the number of bits of group g in a segment.
//
// Reconstruction state (level_signs, which has a byte per coefficient): for
// the word of coefficients [32 w, 32 w + 32), 32-bit word 8 w holds those
// that are nonzero in the bitplanes decoded so far (bit c for coefficient
// 32 w + c), and word 8 w + 1 their signs.
namespace significance {

static constexpr SIZE TILE = 32;    // words per tile
static constexpr SIZE SEGMENT = 64; // tiles per segment
static constexpr SIZE MAX_GROUPS = 64;
static constexpr SIZE STATE_STRIDE = 8; // state words per bitplane word

MGARDX_CONT_EXEC SIZE num_tiles(SIZE num_words) {
  return (num_words + TILE - 1) / TILE;
}
MGARDX_CONT_EXEC SIZE num_segments(SIZE num_words) {
  return (num_tiles(num_words) + SEGMENT - 1) / SEGMENT;
}
MGARDX_CONT_EXEC SIZE num_groups(SIZE num_bitplanes, SIZE group_size) {
  return (num_bitplanes + group_size - 1) / group_size;
}

MGARDX_CONT_EXEC int ctz32(uint32_t x) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return __ffs(x) - 1;
#else
  return __builtin_ctz(x);
#endif
}

// k (1-32) bits of words[] starting at bit pos.
MGARDX_CONT_EXEC uint32_t read_bits(const uint32_t *words, SIZE pos, int k) {
  SIZE w = pos / 32;
  int s = (int)(pos % 32);
  uint32_t x = words[w] >> s;
  if (s > 0 && s + k > 32) {
    x |= words[w + 1] << (32 - s);
  }
  return k == 32 ? x : x & ((1u << k) - 1);
}

// Description of a group's sign section for the sign writer.
struct SignTask {
  uint64_t stream;  // device address of the group stream
  uint64_t section; // byte offset of the sign section in the stream
  uint32_t header_index; // header word that records the section's words
  uint32_t words;        // words[] of the section
};

// Packs the signs of each tile from the encoded rows (row 0: signs, row
// b + 1: bitplane b) in place into row 0, and counts them (portable; thread
// per tile). segment_bits must be zero on entry.
template <typename T_bitplane, typename DeviceType>
class SignPackFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT SignPackFunctor() {}
  MGARDX_CONT SignPackFunctor(SIZE num_bitplanes, SIZE group_size,
                              SubArray<2, T_bitplane, DeviceType> rows,
                              SubArray<1, uint32_t, DeviceType> counts,
                              SubArray<1, uint32_t, DeviceType> segment_bits)
      : num_bitplanes(num_bitplanes), group_size(group_size), rows(rows),
        counts(counts), segment_bits(segment_bits) {
    Functor<DeviceType>();
  }
  MGARDX_EXEC void Operation1() {
    SIZE tile = FunctorBase<DeviceType>::GetBlockIdX() *
                    FunctorBase<DeviceType>::GetBlockDimX() +
                FunctorBase<DeviceType>::GetThreadIdX();
    SIZE num_words = rows.shape(1);
    SIZE ntiles = num_tiles(num_words), nseg = num_segments(num_words);
    if (tile >= ntiles) {
      return;
    }
    SIZE first = tile * TILE;
    SIZE end = num_words - first < TILE ? num_words - first : TILE;
    SIZE G = num_groups(num_bitplanes, group_size);
    uint32_t signs[TILE], out[TILE];
    uint32_t hist[MAX_GROUPS], pos[MAX_GROUPS];
    for (SIZE w = 0; w < TILE; w++) {
      signs[w] = w < end ? (uint32_t)*rows(0, first + w) : 0;
      out[w] = 0;
    }
    for (SIZE g = 0; g < G; g++) {
      hist[g] = 0;
    }
    for (int pass = 0; pass < 2; pass++) {
      for (SIZE w = 0; w < end; w++) {
        uint32_t sig = 0;
        for (SIZE g = 0; g < G; g++) {
          uint32_t orr = 0;
          for (SIZE b = g * group_size;
               b < (g + 1) * group_size && b < num_bitplanes; b++) {
            orr |= (uint32_t)*rows(b + 1, first + w);
          }
          uint32_t fresh = orr & ~sig;
          sig |= orr;
          if (pass == 0) {
            hist[g] += zero_elimination::popcount32(fresh);
            continue;
          }
          while (fresh) {
            int c = ctz32(fresh);
            out[pos[g] / 32] |= ((signs[w] >> c) & 1u) << (pos[g] % 32);
            pos[g]++;
            fresh &= fresh - 1;
          }
        }
      }
      if (pass == 0) {
        uint32_t sum = 0;
        for (SIZE g = 0; g < G; g++) {
          pos[g] = sum;
          sum += hist[g];
        }
      }
    }
    for (SIZE w = 0; w < end; w++) {
      *rows(0, first + w) = out[w];
    }
    for (SIZE g = 0; g < G; g++) {
      *counts(g * ntiles + tile) = pos[g];
      if (hist[g]) {
        Atomic<uint32_t, AtomicGlobalMemory, AtomicDeviceScope,
               DeviceType>::Add(segment_bits(g * nseg + tile / SEGMENT),
                                hist[g]);
      }
    }
  }
  MGARDX_CONT size_t shared_memory_size() { return 0; }

private:
  SIZE num_bitplanes, group_size;
  SubArray<2, T_bitplane, DeviceType> rows;
  SubArray<1, uint32_t, DeviceType> counts, segment_bits;
};

template <typename T_bitplane, typename DeviceType>
class SignPackKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "significance sign pack";
  using FunctorType = SignPackFunctor<T_bitplane, DeviceType>;
  MGARDX_CONT SignPackKernel(SIZE num_bitplanes, SIZE group_size,
                             SubArray<2, T_bitplane, DeviceType> rows,
                             SubArray<1, uint32_t, DeviceType> counts,
                             SubArray<1, uint32_t, DeviceType> segment_bits)
      : num_bitplanes(num_bitplanes), group_size(group_size), rows(rows),
        counts(counts), segment_bits(segment_bits) {}
  MGARDX_CONT Task<FunctorType> GenTask(int queue_idx) {
    FunctorType functor(num_bitplanes, group_size, rows, counts, segment_bits);
    SIZE tbx = 64;
    SIZE gridx = (num_tiles(rows.shape(1)) + tbx - 1) / tbx;
    return Task(functor, 1, 1, gridx, 1, 1, tbx, 0, queue_idx,
                std::string(Name));
  }

private:
  SIZE num_bitplanes, group_size;
  SubArray<2, T_bitplane, DeviceType> rows;
  SubArray<1, uint32_t, DeviceType> counts, segment_bits;
};

// Inclusive prefix sum of c over the lanes of a sub-group.
template <typename DeviceType>
MGARDX_EXEC uint32_t subgroup_inclusive_scan(SubGroup<DeviceType> &sg, int lane,
                                             uint32_t c) {
  for (int d = 1; d < sg.size(); d *= 2) {
    uint32_t t = sg.shfl(c, lane >= d ? lane - d : lane);
    if (lane >= d) {
      c += t;
    }
  }
  return c;
}

// Word offsets of the segments in each group's words[] (exclusive scan of
// the segments' words) and the words of each group: block per group, thread
// per run of consecutive segments, block scan of the runs' sums.
template <typename DeviceType>
class SignScanFunctor : public Functor<DeviceType> {
public:
  static constexpr SIZE THREADS = 256;
  MGARDX_CONT SignScanFunctor() {}
  MGARDX_CONT SignScanFunctor(SIZE nseg,
                              SubArray<1, uint32_t, DeviceType> segment_bits,
                              SubArray<1, uint32_t, DeviceType> offsets,
                              SubArray<1, uint32_t, DeviceType> totals)
      : nseg(nseg), segment_bits(segment_bits), offsets(offsets),
        totals(totals) {
    Functor<DeviceType>();
  }
  MGARDX_EXEC uint32_t words(SIZE s) {
    return (*segment_bits(g * nseg + s) + 31) / 32;
  }
  MGARDX_EXEC void Operation1() {
    partial = (uint32_t *)FunctorBase<DeviceType>::GetSharedMemory();
    tid = FunctorBase<DeviceType>::GetThreadIdX();
    g = FunctorBase<DeviceType>::GetBlockIdX();
    SIZE per = (nseg + THREADS - 1) / THREADS;
    begin = tid * per < nseg ? tid * per : nseg;
    end = begin + per < nseg ? begin + per : nseg;
    sum = 0;
    for (SIZE s = begin; s < end; s++) {
      sum += words(s);
    }
  }
  MGARDX_EXEC void Operation2() {
    SubGroup<DeviceType> sg;
    const int lane = sg.lane();
    inclusive = subgroup_inclusive_scan(sg, lane, sum);
    if (lane == sg.size() - 1) {
      partial[tid / sg.size()] = inclusive;
    }
  }
  MGARDX_EXEC void Operation3() {
    if (tid == 0) {
      SubGroup<DeviceType> sg;
      uint32_t running = 0;
      for (SIZE w = 0; w < THREADS / sg.size(); w++) {
        uint32_t t = partial[w];
        partial[w] = running;
        running += t;
      }
      partial[THREADS] = running;
    }
  }
  MGARDX_EXEC void Operation4() {
    SubGroup<DeviceType> sg;
    uint32_t base = partial[tid / sg.size()] + inclusive - sum;
    for (SIZE s = begin; s < end; s++) {
      *offsets(g * nseg + s) = base;
      base += words(s);
    }
    if (tid == 0) {
      *totals(g) = partial[THREADS];
    }
  }
  MGARDX_CONT size_t shared_memory_size() {
    return (THREADS + 1) * sizeof(uint32_t);
  }

private:
  SIZE nseg;
  SubArray<1, uint32_t, DeviceType> segment_bits, offsets, totals;
  uint32_t *partial;
  SIZE tid, g, begin, end;
  uint32_t sum, inclusive;
};

template <typename DeviceType> class SignScanKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "significance sign scan";
  using FunctorType = SignScanFunctor<DeviceType>;
  MGARDX_CONT SignScanKernel(SIZE num_groups, SIZE nseg,
                             SubArray<1, uint32_t, DeviceType> segment_bits,
                             SubArray<1, uint32_t, DeviceType> offsets,
                             SubArray<1, uint32_t, DeviceType> totals)
      : num_groups(num_groups), nseg(nseg), segment_bits(segment_bits),
        offsets(offsets), totals(totals) {}
  MGARDX_CONT Task<FunctorType> GenTask(int queue_idx) {
    FunctorType functor(nseg, segment_bits, offsets, totals);
    return Task(functor, 1, 1, num_groups, 1, 1, FunctorType::THREADS,
                functor.shared_memory_size(), queue_idx, std::string(Name));
  }

private:
  SIZE num_groups, nseg;
  SubArray<1, uint32_t, DeviceType> segment_bits, offsets, totals;
};

// Writes the sign sections: block (segment, group), thread per tile.
template <typename T_bitplane, typename DeviceType>
class SignWriteFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT SignWriteFunctor() {}
  MGARDX_CONT SignWriteFunctor(SubArray<2, T_bitplane, DeviceType> rows,
                               SubArray<1, uint32_t, DeviceType> counts,
                               SubArray<1, uint32_t, DeviceType> offsets,
                               SubArray<1, SignTask, DeviceType> tasks)
      : rows(rows), counts(counts), offsets(offsets), tasks(tasks) {
    Functor<DeviceType>();
  }
  MGARDX_EXEC void Operation1() {
    prefix = (uint32_t *)FunctorBase<DeviceType>::GetSharedMemory();
    buffer = prefix + SEGMENT + 1;
    tid = FunctorBase<DeviceType>::GetThreadIdX();
    SIZE seg = FunctorBase<DeviceType>::GetBlockIdX();
    g = FunctorBase<DeviceType>::GetBlockIdY();
    num_words = rows.shape(1);
    ntiles = num_tiles(num_words);
    tile = seg * SEGMENT + tid;
    count = 0;
    start = 0;
    if (tile < ntiles) {
      start = g > 0 ? *counts((g - 1) * ntiles + tile) : 0;
      count = *counts(g * ntiles + tile) - start;
    }
  }
  // Exclusive scan of the tiles' counts: sub-group scans, then their totals.
  MGARDX_EXEC void Operation2() {
    SubGroup<DeviceType> sg;
    const int lane = sg.lane();
    inclusive = subgroup_inclusive_scan(sg, lane, count);
    if (lane == sg.size() - 1) {
      prefix[tid / sg.size()] = inclusive;
    }
  }
  MGARDX_EXEC void Operation3() {
    if (tid == 0) {
      SubGroup<DeviceType> sg;
      uint32_t running = 0;
      for (SIZE w = 0; w < SEGMENT / sg.size(); w++) {
        uint32_t t = prefix[w];
        prefix[w] = running;
        running += t;
      }
      prefix[SEGMENT] = running;
    }
  }
  MGARDX_EXEC void Operation4() {
    for (SIZE i = tid; i < (prefix[SEGMENT] + 31) / 32; i += SEGMENT) {
      buffer[i] = 0;
    }
  }
  MGARDX_EXEC void Operation5() {
    if (count == 0) {
      return;
    }
    SubGroup<DeviceType> sg;
    const T_bitplane *src = rows(0, tile * TILE);
    SIZE dst = prefix[tid / sg.size()] + inclusive - count;
    for (uint32_t k = 0; k < count; k += 32) {
      int take = count - k < 32 ? (int)(count - k) : 32;
      uint32_t bits = read_bits((const uint32_t *)src, start + k, take);
      SIZE d = dst + k;
      int s = (int)(d % 32);
      Atomic<uint32_t, AtomicSharedMemory, AtomicDeviceScope, DeviceType>::Or(
          buffer + d / 32, bits << s);
      if (s > 0 && s + take > 32) {
        Atomic<uint32_t, AtomicSharedMemory, AtomicDeviceScope,
               DeviceType>::Or(buffer + d / 32 + 1, bits >> (32 - s));
      }
    }
  }
  MGARDX_EXEC void Operation6() {
    SIZE seg = FunctorBase<DeviceType>::GetBlockIdX();
    SIZE nseg = num_segments(num_words);
    SignTask task = *tasks(g);
    Byte *stream = (Byte *)task.stream;
    uint32_t *section = (uint32_t *)(stream + task.section);
    uint32_t offset = *offsets(g * nseg + seg);
    SIZE words = (prefix[SEGMENT] + 31) / 32;
    for (SIZE i = tid; i < words; i += SEGMENT) {
      section[nseg + offset + i] = buffer[i];
    }
    if (tid == 0) {
      section[seg] = offset;
      if (seg == 0) {
        ((uint32_t *)(stream +
                      zero_elimination::SIGNATURE_BYTES))[task.header_index] =
            task.words;
      }
    }
  }
  MGARDX_CONT size_t shared_memory_size() {
    return (SEGMENT + 1 + SEGMENT * TILE) * sizeof(uint32_t);
  }

private:
  SubArray<2, T_bitplane, DeviceType> rows;
  SubArray<1, uint32_t, DeviceType> counts, offsets;
  SubArray<1, SignTask, DeviceType> tasks;
  uint32_t *prefix, *buffer;
  SIZE tid, g, num_words, ntiles, tile;
  uint32_t count, start, inclusive;
};

template <typename T_bitplane, typename DeviceType>
class SignWriteKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "significance sign write";
  using FunctorType = SignWriteFunctor<T_bitplane, DeviceType>;
  MGARDX_CONT SignWriteKernel(SIZE num_groups,
                              SubArray<2, T_bitplane, DeviceType> rows,
                              SubArray<1, uint32_t, DeviceType> counts,
                              SubArray<1, uint32_t, DeviceType> offsets,
                              SubArray<1, SignTask, DeviceType> tasks)
      : num_groups(num_groups), rows(rows), counts(counts), offsets(offsets),
        tasks(tasks) {}
  MGARDX_CONT Task<FunctorType> GenTask(int queue_idx) {
    FunctorType functor(rows, counts, offsets, tasks);
    return Task(functor, 1, num_groups, num_segments(rows.shape(1)), 1, 1,
                SEGMENT, functor.shared_memory_size(), queue_idx,
                std::string(Name));
  }

private:
  SIZE num_groups;
  SubArray<2, T_bitplane, DeviceType> rows;
  SubArray<1, uint32_t, DeviceType> counts, offsets;
  SubArray<1, SignTask, DeviceType> tasks;
};

// Reconstruction (portable; thread per segment): updates the state (signs)
// of the coefficients with the groups of bitplanes
// [starting_bitplane, starting_bitplane + num_bitplanes) (rows
// starting_bitplane + 1 + b), whose sign sections are sections[i] for the
// i-th group. The decoder then reads every sign from the state.
template <typename T_bitplane, typename DeviceType>
class SignResolveFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT SignResolveFunctor() {}
  MGARDX_CONT SignResolveFunctor(SIZE n, int starting_bitplane,
                                 int num_bitplanes, SIZE group_size,
                                 SubArray<2, T_bitplane, DeviceType> rows,
                                 SubArray<1, bool, DeviceType> signs,
                                 SubArray<1, uint64_t, DeviceType> sections)
      : n(n), starting_bitplane(starting_bitplane),
        num_bitplanes(num_bitplanes), group_size(group_size), rows(rows),
        signs(signs), sections(sections) {
    Functor<DeviceType>();
  }
  MGARDX_EXEC void Operation1() {
    SIZE seg = FunctorBase<DeviceType>::GetBlockIdX() *
                   FunctorBase<DeviceType>::GetBlockDimX() +
               FunctorBase<DeviceType>::GetThreadIdX();
    SIZE num_words = n / 32;
    SIZE nseg = num_segments(num_words);
    if (seg >= nseg) {
      return;
    }
    SIZE G = num_groups(num_bitplanes, group_size);
    SIZE pos[MAX_GROUPS];
    for (SIZE i = 0; i < G; i++) {
      const uint32_t *offsets = (const uint32_t *)*sections(i);
      pos[i] = (SIZE)offsets[seg] * 32;
    }
    SIZE end = (seg + 1) * SEGMENT * TILE;
    end = end < num_words ? end : num_words;
    uint32_t *state = (uint32_t *)signs.data();
    for (SIZE w = seg * SEGMENT * TILE; w < end; w++) {
      uint32_t sig = 0, sgn = 0;
      if (starting_bitplane > 0) {
        sig = state[STATE_STRIDE * w];
        sgn = state[STATE_STRIDE * w + 1];
      }
      uint32_t old_sig = sig;
      for (SIZE i = 0; i < G; i++) {
        uint32_t orr = 0;
        for (SIZE b = i * group_size;
             b < (i + 1) * group_size && b < (SIZE)num_bitplanes; b++) {
          orr |= (uint32_t)*rows(starting_bitplane + 1 + b, w);
        }
        uint32_t fresh = orr & ~sig;
        sig |= orr;
        const uint32_t *words = (const uint32_t *)*sections(i) + nseg;
        while (fresh) {
          int c = ctz32(fresh);
          sgn |= read_bits(words, pos[i], 1) << c;
          pos[i]++;
          fresh &= fresh - 1;
        }
      }
      if (starting_bitplane == 0 || sig != old_sig) {
        state[STATE_STRIDE * w] = sig;
        state[STATE_STRIDE * w + 1] = sgn;
      }
    }
  }
  MGARDX_CONT size_t shared_memory_size() { return 0; }

private:
  SIZE n;
  int starting_bitplane, num_bitplanes;
  SIZE group_size;
  SubArray<2, T_bitplane, DeviceType> rows;
  SubArray<1, bool, DeviceType> signs;
  SubArray<1, uint64_t, DeviceType> sections;
};

template <typename T_bitplane, typename DeviceType>
class SignResolveKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "significance sign resolve";
  using FunctorType = SignResolveFunctor<T_bitplane, DeviceType>;
  MGARDX_CONT SignResolveKernel(SIZE n, int starting_bitplane,
                                int num_bitplanes, SIZE group_size,
                                SubArray<2, T_bitplane, DeviceType> rows,
                                SubArray<1, bool, DeviceType> signs,
                                SubArray<1, uint64_t, DeviceType> sections)
      : n(n), starting_bitplane(starting_bitplane),
        num_bitplanes(num_bitplanes), group_size(group_size), rows(rows),
        signs(signs), sections(sections) {}
  MGARDX_CONT Task<FunctorType> GenTask(int queue_idx) {
    FunctorType functor(n, starting_bitplane, num_bitplanes, group_size, rows,
                        signs, sections);
    SIZE tbx = 64;
    SIZE gridx = (num_segments(n / 32) + tbx - 1) / tbx;
    return Task(functor, 1, 1, gridx, 1, 1, tbx, 0, queue_idx,
                std::string(Name));
  }

private:
  SIZE n;
  int starting_bitplane, num_bitplanes;
  SIZE group_size;
  SubArray<2, T_bitplane, DeviceType> rows;
  SubArray<1, bool, DeviceType> signs;
  SubArray<1, uint64_t, DeviceType> sections;
};

} // namespace significance
} // namespace MDR
} // namespace mgard_x

#endif
