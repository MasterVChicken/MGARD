/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: October 8, 2026
 */

#ifndef MGARD_X_MDR_ZERO_ELIMINATION_HPP
#define MGARD_X_MDR_ZERO_ELIMINATION_HPP

#include "../../RuntimeX/RuntimeX.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace mgard_x {
namespace MDR {

// MGARDX_MDR_PORTABLE_KERNELS=1: the portable zero-elimination and
// significance-coding kernels also on CUDA (for testing; same results).
inline bool portable_kernels() {
  static const bool portable = [] {
    const char *e = std::getenv("MGARDX_MDR_PORTABLE_KERNELS");
    return e != nullptr && std::string(e) == "1";
  }();
  return portable;
}

// Zero elimination of encoded bitplane rows: a single-pass, data-parallel
// alternative to Huffman/RLE. A row of 32-bit words is split into chunks of
// 32 words; each chunk has a 32-bit bitmap of its nonzero words (bit j for
// word j), and only the nonzero words are stored. Two-level rows also mark
// the chunks with a nonzero bitmap (one bit per chunk) and store only those
// bitmaps, so that sparse rows cost little more than their nonzero words. A
// row is stored raw when that is not smaller.
//
// Sparse words (groups of kind SPARSE): a two-level row may instead store
// each nonzero word as a 3-bit code and a payload: k <= SPARSE_MAX_ONES one
// bits as k 5-bit positions (ascending), more as the 32 bits (RAW_CODE). The
// codes and payloads of a super-chunk form its region, word aligned: the
// codes of its stored words in order, then their payloads in order. A row is
// sparse when that is smaller (payload > 0: its region words).
//
// Merged group stream (all fields uint32 unless noted, 4-byte aligned):
//   signature (8 bytes), num_rows, num_words,
//   count[num_rows]            nonzero words of each row, RAW_ROW if raw
//   chunks[num_rows]           (two-level) chunks with a nonzero bitmap
//   payload[num_rows]          (sparse words) region words, 0: not sparse
//   sign_words                 (with signs) words of the sign section
//   then for each row, raw:    words[num_words]
//                  one-level:  word_offsets[num_super], bitmaps[num_chunks],
//                              words[count]
//                  two-level:  word_offsets[num_super],
//                              bitmap_offsets[num_super],
//                              marks[num_level2] (bit c of mark i: chunk
//                              32 i + c has a nonzero bitmap),
//                              bitmaps[chunks], words[count]
//                  sparse:     word_offsets[num_super],
//                              bitmap_offsets[num_super],
//                              region_offsets[num_super], marks[num_level2],
//                              bitmaps[chunks], regions[payload]
//   then (with signs) the sign section (SignificanceCoding.hpp).
// A super-chunk is SUPER chunks (one thread block); its offsets (first
// nonzero word, first stored bitmap, region) let every block decode
// independently. Signatures: "MGXZELM" one-level, "MGXZESG" one-level with
// signs (versions 3-4, read only), "MGXZ2LM" two-level, "MGXZ2SG" two-level
// with signs, "MGXZ3LM" / "MGXZ3SG" the same with sparse words.
namespace zero_elimination {

static constexpr SIZE CHUNK = 32;  // words per bitmap
static constexpr SIZE SUPER = 256; // chunks per super-chunk (= block size)
static constexpr uint32_t RAW_ROW = 0xffffffffu;
static constexpr SIZE SIGNATURE_BYTES = 8;

// Group kinds (flags; 0: not a zero-elimination group). SPARSE only with
// TWO_LEVEL.
static constexpr uint32_t ZE = 1, SIGNS = 2, TWO_LEVEL = 4, SPARSE = 8;

// Sparse words: codes and payload bits.
static constexpr uint32_t SPARSE_MAX_ONES = 6, RAW_CODE = 7, CODE_BITS = 3;
// Region words of a super-chunk at most (every word stored raw).
static constexpr SIZE MAX_REGION =
    (CODE_BITS * SUPER * CHUNK + 32 * SUPER * CHUNK) / 32;

MGARDX_CONT_EXEC void write_signature(Byte *stream, uint32_t kind) {
  const bool signs = kind & SIGNS, two = kind & TWO_LEVEL,
             sparse = kind & SPARSE;
  const Byte sig[SIGNATURE_BYTES] = {
      'M',
      'G',
      'X',
      'Z',
      sparse ? (Byte)'3' : (two ? (Byte)'2' : (Byte)'E'),
      signs ? (Byte)'S' : (Byte)'L',
      signs ? (Byte)'G' : (Byte)'M',
      0};
  for (int i = 0; i < (int)SIGNATURE_BYTES; i++) {
    stream[i] = sig[i];
  }
}
inline uint32_t group_kind(const Byte *header) {
  for (uint32_t kind = ZE; kind <= (ZE | SIGNS | TWO_LEVEL | SPARSE);
       kind += 2) {
    if ((kind & SPARSE) && !(kind & TWO_LEVEL)) {
      continue;
    }
    Byte sig[SIGNATURE_BYTES];
    write_signature(sig, kind);
    if (std::memcmp(header, sig, SIGNATURE_BYTES) == 0) {
      return kind;
    }
  }
  return 0;
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
MGARDX_CONT_EXEC SIZE num_level2(SIZE num_words) {
  return (num_chunks(num_words) + 31) / 32;
}

// Sparse words: the code of a nonzero word, the payload bits of a code, and
// the payload bits of a word (0 for a zero word, which is not stored).
MGARDX_CONT_EXEC uint32_t sparse_code(uint32_t x) {
  uint32_t k = popcount32(x);
  return k <= SPARSE_MAX_ONES ? k : RAW_CODE;
}
MGARDX_CONT_EXEC uint32_t code_bits(uint32_t code) {
  return code == RAW_CODE ? 32 : 5 * code;
}
MGARDX_CONT_EXEC uint32_t sparse_bits(uint32_t x) {
  return x ? code_bits(sparse_code(x)) : 0;
}
// Region words of a super-chunk with count stored words of payload bits.
MGARDX_CONT_EXEC uint32_t region_words(uint32_t count, uint32_t bits) {
  return (CODE_BITS * count + bits + 31) / 32;
}
// Index of the lowest one bit of x (x != 0).
MGARDX_CONT_EXEC uint32_t lowest_bit(uint32_t x) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return __ffs(x) - 1;
#else
  return popcount32((x & (0u - x)) - 1);
#endif
}
// Bits [pos, pos + k) of a little-endian bit stream of words (k <= 32); P:
// an unsigned position type (32 bits within a region).
template <typename P>
MGARDX_CONT_EXEC uint32_t get_bits(const uint32_t *words, P pos, int k) {
  P w = pos / 32;
  int s = (int)(pos % 32);
  uint32_t x = words[w] >> s;
  if (s > 0 && s + k > 32) {
    x |= words[w + 1] << (32 - s);
  }
  return k == 32 ? x : x & ((1u << k) - 1);
}

// Size in bytes of a row's section (payload: region words of a sparse row,
// 0 otherwise).
inline SIZE row_bytes(uint32_t count, uint32_t chunks, SIZE num_words,
                      bool two_level, uint32_t payload = 0) {
  if (count == RAW_ROW) {
    return num_words * sizeof(uint32_t);
  }
  if (two_level && payload) {
    return (3 * num_super(num_words) + num_level2(num_words) + chunks +
            payload) *
           sizeof(uint32_t);
  }
  if (two_level) {
    return (2 * num_super(num_words) + num_level2(num_words) + chunks +
            count) *
           sizeof(uint32_t);
  }
  return (num_super(num_words) + num_chunks(num_words) + count) *
         sizeof(uint32_t);
}
// Header words: count[] from word 2, then chunks[] (two-level), payload[]
// (sparse words), then the sign words (with signs).
MGARDX_CONT_EXEC SIZE header_words(SIZE num_rows, uint32_t kind) {
  return 2 +
         num_rows * (1 + ((kind & TWO_LEVEL) ? 1 : 0) +
                     ((kind & SPARSE) ? 1 : 0)) +
         ((kind & SIGNS) ? 1 : 0);
}
inline SIZE header_bytes(SIZE num_rows, uint32_t kind) {
  return SIGNATURE_BYTES + header_words(num_rows, kind) * sizeof(uint32_t);
}

// Per-row description used by the write and decode kernels.
struct RowTask {
  uint64_t stream;  // device address of the row's group stream
  uint64_t section; // byte offset of the row's section in the stream
  uint32_t row;     // row of the encoded bitplanes
  uint32_t count;   // nonzero words, or RAW_ROW
  uint32_t chunks;  // (two-level) chunks with a nonzero bitmap
  uint32_t index;   // row index within the group
  uint32_t group_rows;
  uint32_t kind;    // of the group
  uint32_t payload; // sparse row: region words; 0 otherwise
};

// Words before the marks of a two-level row's section (super-chunk offsets).
MGARDX_CONT_EXEC SIZE offset_words(const RowTask &task, SIZE nsuper) {
  return (task.payload ? 3 : 2) * nsuper;
}

// Writes the group header fields of a task's row (and the group's signature
// and sizes with its first row).
MGARDX_EXEC void write_header(const RowTask &task, SIZE num_words) {
  Byte *stream = (Byte *)task.stream;
  uint32_t *header = (uint32_t *)(stream + SIGNATURE_BYTES);
  if (task.index == 0) {
    write_signature(stream, task.kind);
    header[0] = task.group_rows;
    header[1] = (uint32_t)num_words;
  }
  header[2 + task.index] = task.count;
  if (task.kind & TWO_LEVEL) {
    header[2 + task.group_rows + task.index] = task.chunks;
  }
  if (task.kind & SPARSE) {
    header[2 + 2 * task.group_rows + task.index] = task.payload;
  }
}

// Sets bits [pos, pos + k) (zero on entry) of a region in shared memory to x.
template <typename DeviceType>
MGARDX_EXEC void put_bits(uint32_t *region, SIZE pos, uint32_t x, int k) {
  int s = (int)(pos % 32);
  Atomic<uint32_t, AtomicSharedMemory, AtomicDeviceScope, DeviceType>::Or(
      region + pos / 32, x << s);
  if (s > 0 && s + k > 32) {
    Atomic<uint32_t, AtomicSharedMemory, AtomicDeviceScope, DeviceType>::Or(
        region + pos / 32 + 1, x >> (32 - s));
  }
}

// Writes the code and the payload of a stored word: the code of the word of
// index `index` among the super-chunk's stored words, the payload at bit
// `pos` of the region.
template <typename DeviceType>
MGARDX_EXEC void put_sparse_word(uint32_t *region, uint32_t index, SIZE pos,
                                 uint32_t x) {
  uint32_t code = sparse_code(x);
  put_bits<DeviceType>(region, CODE_BITS * index, code, CODE_BITS);
  // The positions as one value (at most 30 bits).
  uint32_t payload = x;
  if (code != RAW_CODE) {
    payload = 0;
    int shift = 0;
    for (uint32_t m = x; m; m &= m - 1, shift += 5) {
      payload |= lowest_bit(m) << shift;
    }
  }
  put_bits<DeviceType>(region, pos, payload, (int)code_bits(code));
}

// Decodes a stored word of a region: code given, payload at bit pos.
template <typename P>
MGARDX_EXEC uint32_t get_sparse_word(const uint32_t *region, P pos,
                                     uint32_t code) {
  uint32_t payload = get_bits(region, pos, (int)code_bits(code));
  if (code == RAW_CODE) {
    return payload;
  }
  uint32_t x = 0;
#pragma unroll
  for (uint32_t i = 0; i < SPARSE_MAX_ONES; i++) {
    if (i < code) {
      x |= 1u << ((payload >> (5 * i)) & 31u);
    }
  }
  return x;
}

// Nonzero words and payload bits of each chunk, and nonzero words, nonzero
// chunks and region words of each super-chunk, of rows [0, num_rows) of the
// encoded bitplanes: super_counts(r, s), super_counts(num_rows + r, s) and
// super_counts(2 num_rows + r, s).
template <typename T_bitplane, typename DeviceType>
class CountFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT CountFunctor() {}
  MGARDX_CONT CountFunctor(SIZE num_rows, SubArray<2, T_bitplane, DeviceType> rows,
                           SubArray<1, uint32_t, DeviceType> chunk_counts,
                           SubArray<1, uint32_t, DeviceType> chunk_bits,
                           SubArray<1, uint32_t, DeviceType> super_counts)
      : num_rows(num_rows), rows(rows), chunk_counts(chunk_counts),
        chunk_bits(chunk_bits), super_counts(super_counts) {
    Functor<DeviceType>();
  }
  MGARDX_EXEC void Operation1() {
    sum = (uint32_t *)FunctorBase<DeviceType>::GetSharedMemory();
    tid = FunctorBase<DeviceType>::GetThreadIdX();
    if (tid == 0) {
      sum[0] = 0;
      sum[1] = 0;
      sum[2] = 0;
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
    uint32_t count = 0, bits = 0;
    for (SIZE j = 0; j < end; j++) {
      count += w[j] != 0;
      bits += sparse_bits((uint32_t)w[j]);
    }
    *chunk_counts(row * num_chunks(num_words) + chunk) = count;
    if (chunk_bits.data() != nullptr) {
      *chunk_bits(row * num_chunks(num_words) + chunk) = bits;
    }
    if (count) {
      Atomic<uint32_t, AtomicSharedMemory, AtomicDeviceScope, DeviceType>::Add(
          sum, count);
      Atomic<uint32_t, AtomicSharedMemory, AtomicDeviceScope, DeviceType>::Add(
          sum + 1, 1u);
      Atomic<uint32_t, AtomicSharedMemory, AtomicDeviceScope, DeviceType>::Add(
          sum + 2, bits);
    }
  }
  MGARDX_EXEC void Operation3() {
    if (tid == 0) {
      SIZE row = FunctorBase<DeviceType>::GetBlockIdY();
      SIZE nsuper = num_super(rows.shape(1));
      SIZE s = FunctorBase<DeviceType>::GetBlockIdX();
      *super_counts(row * nsuper + s) = sum[0];
      *super_counts((num_rows + row) * nsuper + s) = sum[1];
      *super_counts((2 * num_rows + row) * nsuper + s) =
          chunk_bits.data() != nullptr ? region_words(sum[0], sum[2]) : 0;
    }
  }
  MGARDX_CONT size_t shared_memory_size() { return 3 * sizeof(uint32_t); }

private:
  SIZE num_rows;
  SubArray<2, T_bitplane, DeviceType> rows;
  SubArray<1, uint32_t, DeviceType> chunk_counts, chunk_bits, super_counts;
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
                          SubArray<1, uint32_t, DeviceType> chunk_bits,
                          SubArray<1, uint32_t, DeviceType> super_counts)
      : rows(rows), num_rows(num_rows), chunk_counts(chunk_counts),
        chunk_bits(chunk_bits), super_counts(super_counts) {}
  MGARDX_CONT Task<FunctorType> GenTask(int queue_idx) {
    FunctorType functor(num_rows, rows, chunk_counts, chunk_bits,
                        super_counts);
    return Task(functor, 1, num_rows, num_super(rows.shape(1)), 1, 1, SUPER,
                functor.shared_memory_size(), queue_idx, std::string(Name));
  }

private:
  SubArray<2, T_bitplane, DeviceType> rows;
  SIZE num_rows;
  SubArray<1, uint32_t, DeviceType> chunk_counts, chunk_bits, super_counts;
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

// Region of super-chunk s of a sparse row: offset from the row's first and
// words, from the scanned region counts (super_offsets(2 num_rows + r, s)).
template <typename DeviceType>
MGARDX_EXEC void region_extent(SubArray<1, uint32_t, DeviceType> &super_offsets,
                               SIZE num_rows, const RowTask &task, SIZE nsuper,
                               SIZE s, uint32_t &offset, uint32_t &words) {
  SIZE base = (2 * num_rows + task.row) * nsuper;
  offset = *super_offsets(base + s);
  words = (s + 1 < nsuper ? *super_offsets(base + s + 1) : task.payload) -
          offset;
}

// Stored words of super-chunk s of a row, from its word offsets.
MGARDX_EXEC uint32_t super_count(const uint32_t *section, const RowTask &task,
                                 SIZE nsuper, SIZE s) {
  return (s + 1 < nsuper ? section[s + 1] : task.count) - section[s];
}

// Writes each row's two-level section (and the group headers) into the group
// streams (portable; thread per chunk). super_offsets: the scanned counts of
// CountKernel (word offsets of rows [0, num_rows), then bitmap offsets, then
// region offsets); chunk_bits: the payload bits of each chunk (sparse rows).
// A sparse super-chunk's region is assembled in shared memory.
template <typename T_bitplane, typename DeviceType>
class WriteFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT WriteFunctor() {}
  MGARDX_CONT WriteFunctor(SIZE num_rows,
                           SubArray<2, T_bitplane, DeviceType> rows,
                           SubArray<1, RowTask, DeviceType> tasks,
                           SubArray<1, uint32_t, DeviceType> chunk_counts,
                           SubArray<1, uint32_t, DeviceType> chunk_bits,
                           SubArray<1, uint32_t, DeviceType> super_offsets)
      : num_rows(num_rows), rows(rows), tasks(tasks),
        chunk_counts(chunk_counts), chunk_bits(chunk_bits),
        super_offsets(super_offsets) {
    Functor<DeviceType>();
  }
  MGARDX_EXEC void Operation1() {
    prefix = (uint32_t *)FunctorBase<DeviceType>::GetSharedMemory();
    chunk_prefix = prefix + SUPER;
    bit_prefix = chunk_prefix + SUPER;
    region = bit_prefix + SUPER;
    tid = FunctorBase<DeviceType>::GetThreadIdX();
    task = *tasks(FunctorBase<DeviceType>::GetBlockIdY());
    num_words = rows.shape(1);
    nchunks = num_chunks(num_words);
    nsuper = num_super(num_words);
    s = FunctorBase<DeviceType>::GetBlockIdX();
    chunk = s * SUPER + tid;
    if (s == 0 && tid == 0) {
      write_header(task, num_words);
    }
    count = 0;
    uint32_t bits = 0;
    if (task.count != RAW_ROW && chunk < nchunks) {
      count = *chunk_counts(task.row * nchunks + chunk);
      if (task.payload) {
        bits = *chunk_bits(task.row * nchunks + chunk);
      }
    }
    prefix[tid] = count;
    chunk_prefix[tid] = count != 0;
    bit_prefix[tid] = bits;
    region_offset = region_size = 0;
    if (task.count != RAW_ROW && task.payload) {
      region_extent(super_offsets, num_rows, task, nsuper, s, region_offset,
                    region_size);
      for (SIZE i = tid; i < region_size; i += SUPER) {
        region[i] = 0;
      }
    }
  }
  MGARDX_EXEC void Operation2() {
    // Exclusive prefixes of the chunk counts, of the nonzero chunks and of
    // the payload bits of this super-chunk.
    if (task.count != RAW_ROW && tid == 0) {
      uint32_t sum = 0, nonzero = 0, bits = 0;
      for (SIZE i = 0; i < SUPER; i++) {
        uint32_t c = prefix[i], z = chunk_prefix[i], b = bit_prefix[i];
        prefix[i] = sum;
        chunk_prefix[i] = nonzero;
        bit_prefix[i] = bits;
        sum += c;
        nonzero += z;
        bits += b;
      }
    }
  }
  MGARDX_EXEC void Operation3() {
    if (chunk >= nchunks) {
      return;
    }
    Byte *stream = (Byte *)task.stream;
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
    uint32_t word_offset = *super_offsets(task.row * nsuper + s);
    uint32_t bitmap_offset = *super_offsets((num_rows + task.row) * nsuper + s);
    if (tid == 0) {
      section[s] = word_offset;
      section[nsuper + s] = bitmap_offset;
      if (task.payload) {
        section[2 * nsuper + s] = region_offset;
      }
    }
    uint32_t *marks = section + offset_words(task, nsuper);
    if (chunk % 32 == 0) {
      // Thread of the first chunk of a mark word builds it.
      uint32_t mark = 0;
      for (SIZE c = 0; c < 32 && chunk + c < nchunks; c++) {
        mark |= (uint32_t)(*chunk_counts(task.row * nchunks + chunk + c) != 0)
                << c;
      }
      marks[chunk / 32] = mark;
    }
    if (count == 0) {
      return;
    }
    uint32_t *bitmaps = marks + num_level2(num_words);
    uint32_t bitmap = 0, k = 0;
    if (task.payload) {
      // Codes, then payloads, of the super-chunk's stored words.
      uint32_t super_words =
          (s + 1 < nsuper ? *super_offsets(task.row * nsuper + s + 1)
                          : task.count) -
          word_offset;
      SIZE pos = CODE_BITS * super_words + bit_prefix[tid];
      for (SIZE j = 0; j < end; j++) {
        uint32_t x = (uint32_t)w[j];
        if (x != 0) {
          bitmap |= 1u << j;
          put_sparse_word<DeviceType>(region, prefix[tid] + k++, pos, x);
          pos += sparse_bits(x);
        }
      }
    } else {
      uint32_t *words = bitmaps + task.chunks + word_offset + prefix[tid];
      for (SIZE j = 0; j < end; j++) {
        if (w[j] != 0) {
          bitmap |= 1u << j;
          words[k++] = (uint32_t)w[j];
        }
      }
    }
    bitmaps[bitmap_offset + chunk_prefix[tid]] = bitmap;
  }
  MGARDX_EXEC void Operation4() {
    if (task.count == RAW_ROW || !task.payload) {
      return;
    }
    uint32_t *out = (uint32_t *)((Byte *)task.stream + task.section) +
                    3 * nsuper + num_level2(num_words) + task.chunks +
                    region_offset;
    for (SIZE i = tid; i < region_size; i += SUPER) {
      out[i] = region[i];
    }
  }
  // Shared region words (sparse tasks: their largest super-chunk region).
  SIZE region_capacity = MAX_REGION;
  MGARDX_CONT size_t shared_memory_size() {
    return (3 * SUPER + region_capacity) * sizeof(uint32_t);
  }

private:
  SIZE num_rows;
  SubArray<2, T_bitplane, DeviceType> rows;
  SubArray<1, RowTask, DeviceType> tasks;
  SubArray<1, uint32_t, DeviceType> chunk_counts, chunk_bits, super_offsets;
  uint32_t *prefix, *chunk_prefix, *bit_prefix, *region;
  RowTask task;
  uint32_t count, region_offset, region_size;
  SIZE tid, num_words, nchunks, nsuper, s, chunk;
};

template <typename T_bitplane, typename DeviceType>
class WriteKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "zero elimination write";
  using FunctorType = WriteFunctor<T_bitplane, DeviceType>;
  MGARDX_CONT WriteKernel(SIZE num_rows,
                          SubArray<2, T_bitplane, DeviceType> rows,
                          SubArray<1, RowTask, DeviceType> tasks,
                          SIZE num_tasks,
                          SubArray<1, uint32_t, DeviceType> chunk_counts,
                          SubArray<1, uint32_t, DeviceType> chunk_bits,
                          SubArray<1, uint32_t, DeviceType> super_offsets,
                          SIZE region_capacity = MAX_REGION)
      : num_rows(num_rows), rows(rows), tasks(tasks), num_tasks(num_tasks),
        chunk_counts(chunk_counts), chunk_bits(chunk_bits),
        super_offsets(super_offsets), region_capacity(region_capacity) {}
  MGARDX_CONT Task<FunctorType> GenTask(int queue_idx) {
    FunctorType functor(num_rows, rows, tasks, chunk_counts, chunk_bits,
                        super_offsets);
    functor.region_capacity = region_capacity;
    return Task(functor, 1, num_tasks, num_super(rows.shape(1)), 1, 1, SUPER,
                functor.shared_memory_size(), queue_idx, std::string(Name));
  }

private:
  SIZE num_rows;
  SubArray<2, T_bitplane, DeviceType> rows;
  SubArray<1, RowTask, DeviceType> tasks;
  SIZE num_tasks;
  SubArray<1, uint32_t, DeviceType> chunk_counts, chunk_bits, super_offsets;
  SIZE region_capacity;
};

// Decodes row sections (one- or two-level, sparse) back into rows of the
// encoded bitplanes (portable; thread per chunk).
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
    bit_prefix = prefix + SUPER;
    tid = FunctorBase<DeviceType>::GetThreadIdX();
    task = *tasks(FunctorBase<DeviceType>::GetBlockIdY());
    num_words = rows.shape(1);
    nchunks = num_chunks(num_words);
    nsuper = num_super(num_words);
    chunk = FunctorBase<DeviceType>::GetBlockIdX() * SUPER + tid;
    section = (const uint32_t *)((const Byte *)task.stream + task.section);
    two_level = task.kind & TWO_LEVEL;
    bitmap = 0;
    marked = false;
    if (task.count != RAW_ROW) {
      if (two_level) {
        const uint32_t *marks = section + offset_words(task, nsuper);
        marked = chunk < nchunks && ((marks[chunk / 32] >> (chunk % 32)) & 1u);
        prefix[tid] = marked;
      } else {
        bitmap = chunk < nchunks ? section[nsuper + chunk] : 0;
        prefix[tid] = popcount32(bitmap);
      }
    }
  }
  MGARDX_EXEC void Operation2() {
    if (task.count != RAW_ROW && two_level && tid == 0) {
      exclusive_scan(prefix);
    }
  }
  MGARDX_EXEC void Operation3() {
    if (task.count != RAW_ROW && two_level) {
      const uint32_t *bitmaps =
          section + offset_words(task, nsuper) + num_level2(num_words);
      bitmap = marked ? bitmaps[section[nsuper +
                                        FunctorBase<DeviceType>::GetBlockIdX()] +
                                prefix[tid]]
                      : 0;
    }
  }
  MGARDX_EXEC void Operation4() {
    if (task.count != RAW_ROW && two_level) {
      prefix[tid] = popcount32(bitmap);
    }
  }
  MGARDX_EXEC void Operation5() {
    if (task.count != RAW_ROW && tid == 0) {
      exclusive_scan(prefix);
    }
  }
  MGARDX_EXEC void Operation6() {
    // Sparse rows: payload bits of the chunk, from the codes of its words.
    if (task.count == RAW_ROW || !task.payload) {
      return;
    }
    SIZE s = FunctorBase<DeviceType>::GetBlockIdX();
    region = section + 3 * nsuper + num_level2(num_words) + task.chunks +
             section[2 * nsuper + s];
    payload_start = CODE_BITS * super_count(section, task, nsuper, s);
    uint32_t bits = 0, k = popcount32(bitmap);
    for (uint32_t i = 0; i < k; i++) {
      bits += code_bits(get_bits(region, CODE_BITS * (prefix[tid] + i),
                                 CODE_BITS));
    }
    bit_prefix[tid] = bits;
  }
  MGARDX_EXEC void Operation7() {
    if (task.count != RAW_ROW && task.payload && tid == 0) {
      exclusive_scan(bit_prefix);
    }
  }
  MGARDX_EXEC void Operation8() {
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
    if (task.payload) {
      SIZE pos = payload_start + bit_prefix[tid];
      uint32_t k = prefix[tid];
      for (SIZE j = 0; j < end; j++) {
        uint32_t x = 0;
        if ((bitmap >> j) & 1u) {
          uint32_t code = get_bits(region, CODE_BITS * k++, CODE_BITS);
          x = get_sparse_word(region, pos, code);
          pos += code_bits(code);
        }
        out[j] = x;
      }
      return;
    }
    const uint32_t *words =
        two_level
            ? section + 2 * nsuper + num_level2(num_words) + task.chunks
            : section + nsuper + nchunks;
    words += section[FunctorBase<DeviceType>::GetBlockIdX()] + prefix[tid];
    uint32_t k = 0;
    for (SIZE j = 0; j < end; j++) {
      out[j] = (bitmap >> j) & 1u ? words[k++] : 0;
    }
  }
  MGARDX_CONT size_t shared_memory_size() {
    return 2 * SUPER * sizeof(uint32_t);
  }

private:
  MGARDX_EXEC void exclusive_scan(uint32_t *v) {
    uint32_t sum = 0;
    for (SIZE i = 0; i < SUPER; i++) {
      uint32_t c = v[i];
      v[i] = sum;
      sum += c;
    }
  }
  SubArray<2, T_bitplane, DeviceType> rows;
  SubArray<1, RowTask, DeviceType> tasks;
  uint32_t *prefix, *bit_prefix;
  RowTask task;
  const uint32_t *section, *region;
  uint32_t bitmap;
  SIZE payload_start;
  bool two_level, marked;
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
// row loads and stores are coalesced; a chunk bitmap is one ballot, and the
// mark word of a warp's 32 chunks one more. Same stream format as the
// kernels above.
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

// Sum over the warp.
template <typename DeviceType>
MGARDX_EXEC uint32_t warp_sum(SubGroup<DeviceType> &sg, int lane, uint32_t c) {
  for (int offset = 16; offset > 0; offset /= 2) {
    c += sg.shfl(c, lane ^ offset);
  }
  return c;
}

// Chunk bitmaps (and, given chunk_bits, payload bits) of rows [0, num_rows),
// and nonzero words, nonzero chunks and region words of each super-chunk (as
// CountKernel).
template <typename T_bitplane, typename DeviceType>
class BitmapFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT BitmapFunctor() {}
  MGARDX_CONT BitmapFunctor(SIZE num_rows,
                            SubArray<2, T_bitplane, DeviceType> rows,
                            SubArray<1, uint32_t, DeviceType> bitmaps,
                            SubArray<1, uint32_t, DeviceType> chunk_bits,
                            SubArray<1, uint32_t, DeviceType> super_counts)
      : num_rows(num_rows), rows(rows), bitmaps(bitmaps),
        chunk_bits(chunk_bits), super_counts(super_counts) {
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
    const bool bits = chunk_bits.data() != nullptr;
    uint32_t mine = 0, mine_bits = 0;
#pragma unroll 8
    for (int k = 0; k < (int)CHUNK; k++) {
      if (first + k >= nchunks) {
        break;
      }
      SIZE idx = (first + k) * CHUNK + lane;
      uint32_t x = idx < num_words ? (uint32_t)*rows(row, idx) : 0;
      uint32_t bitmap = sg.ballot(x != 0);
      uint32_t b = bits && bitmap ? warp_sum(sg, lane, sparse_bits(x)) : 0;
      if (lane == k) {
        mine = bitmap;
        mine_bits = b;
      }
    }
    if (first + lane < nchunks) {
      *bitmaps(row * nchunks + first + lane) = mine;
      if (bits) {
        *chunk_bits(row * nchunks + first + lane) = mine_bits;
      }
    }
    uint32_t c = warp_sum(sg, lane, popcount32(mine));
    uint32_t z = popcount32(sg.ballot(mine != 0));
    uint32_t b = warp_sum(sg, lane, mine_bits);
    if (lane == 0) {
      sums[warp] = c;
      sums[WARPS + warp] = z;
      sums[2 * WARPS + warp] = b;
    }
  }
  MGARDX_EXEC void Operation2() {
    if (FunctorBase<DeviceType>::GetThreadIdX() == 0) {
      uint32_t c = 0, z = 0, b = 0;
      for (SIZE w = 0; w < WARPS; w++) {
        c += sums[w];
        z += sums[WARPS + w];
        b += sums[2 * WARPS + w];
      }
      SIZE nsuper = num_super(rows.shape(1));
      SIZE row = FunctorBase<DeviceType>::GetBlockIdY();
      SIZE s = FunctorBase<DeviceType>::GetBlockIdX();
      *super_counts(row * nsuper + s) = c;
      *super_counts((num_rows + row) * nsuper + s) = z;
      *super_counts((2 * num_rows + row) * nsuper + s) =
          chunk_bits.data() != nullptr ? region_words(c, b) : 0;
    }
  }
  MGARDX_CONT size_t shared_memory_size() {
    return 3 * WARPS * sizeof(uint32_t);
  }

private:
  SIZE num_rows;
  SubArray<2, T_bitplane, DeviceType> rows;
  SubArray<1, uint32_t, DeviceType> bitmaps, chunk_bits, super_counts;
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
                           SubArray<1, uint32_t, DeviceType> chunk_bits,
                           SubArray<1, uint32_t, DeviceType> super_counts)
      : rows(rows), num_rows(num_rows), bitmaps(bitmaps),
        chunk_bits(chunk_bits), super_counts(super_counts) {}
  MGARDX_CONT Task<FunctorType> GenTask(int queue_idx) {
    FunctorType functor(num_rows, rows, bitmaps, chunk_bits, super_counts);
    return Task(functor, 1, num_rows, num_super(rows.shape(1)), 1, 1, SUPER,
                functor.shared_memory_size(), queue_idx, std::string(Name));
  }

private:
  SubArray<2, T_bitplane, DeviceType> rows;
  SIZE num_rows;
  SubArray<1, uint32_t, DeviceType> bitmaps, chunk_bits, super_counts;
};

// Nonzero words, nonzero chunks and region words of each super-chunk from
// given chunk bitmaps and payload bits (thread per chunk; as CountKernel;
// region words 0 without chunk_bits).
template <typename DeviceType>
class SuperCountFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT SuperCountFunctor() {}
  MGARDX_CONT SuperCountFunctor(SIZE num_rows, SIZE nchunks,
                                SubArray<1, uint32_t, DeviceType> bitmaps,
                                SubArray<1, uint32_t, DeviceType> chunk_bits,
                                SubArray<1, uint32_t, DeviceType> super_counts)
      : num_rows(num_rows), nchunks(nchunks), bitmaps(bitmaps),
        chunk_bits(chunk_bits), super_counts(super_counts) {
    Functor<DeviceType>();
  }
  MGARDX_EXEC void Operation1() {
    SubGroup<DeviceType> sg;
    const int lane = sg.lane();
    sums = (uint32_t *)FunctorBase<DeviceType>::GetSharedMemory();
    SIZE tid = FunctorBase<DeviceType>::GetThreadIdX();
    SIZE row = FunctorBase<DeviceType>::GetBlockIdY();
    SIZE chunk = FunctorBase<DeviceType>::GetBlockIdX() * SUPER + tid;
    uint32_t bitmap = chunk < nchunks ? *bitmaps(row * nchunks + chunk) : 0;
    uint32_t bits = chunk < nchunks && chunk_bits.data() != nullptr
                        ? *chunk_bits(row * nchunks + chunk)
                        : 0;
    uint32_t c = warp_sum(sg, lane, popcount32(bitmap));
    uint32_t z = popcount32(sg.ballot(bitmap != 0));
    uint32_t b = warp_sum(sg, lane, bits);
    if (lane == 0) {
      sums[tid / CHUNK] = c;
      sums[WARPS + tid / CHUNK] = z;
      sums[2 * WARPS + tid / CHUNK] = b;
    }
  }
  MGARDX_EXEC void Operation2() {
    if (FunctorBase<DeviceType>::GetThreadIdX() == 0) {
      uint32_t c = 0, z = 0, b = 0;
      for (SIZE w = 0; w < WARPS; w++) {
        c += sums[w];
        z += sums[WARPS + w];
        b += sums[2 * WARPS + w];
      }
      SIZE nsuper = (nchunks + SUPER - 1) / SUPER;
      SIZE row = FunctorBase<DeviceType>::GetBlockIdY();
      SIZE s = FunctorBase<DeviceType>::GetBlockIdX();
      *super_counts(row * nsuper + s) = c;
      *super_counts((num_rows + row) * nsuper + s) = z;
      *super_counts((2 * num_rows + row) * nsuper + s) =
          chunk_bits.data() != nullptr ? region_words(c, b) : 0;
    }
  }
  MGARDX_CONT size_t shared_memory_size() {
    return 3 * WARPS * sizeof(uint32_t);
  }

private:
  SIZE num_rows, nchunks;
  SubArray<1, uint32_t, DeviceType> bitmaps, chunk_bits, super_counts;
  uint32_t *sums;
};

template <typename DeviceType> class SuperCountKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "zero elimination super count";
  using FunctorType = SuperCountFunctor<DeviceType>;
  MGARDX_CONT SuperCountKernel(SIZE num_rows, SIZE nchunks,
                               SubArray<1, uint32_t, DeviceType> bitmaps,
                               SubArray<1, uint32_t, DeviceType> chunk_bits,
                               SubArray<1, uint32_t, DeviceType> super_counts)
      : num_rows(num_rows), nchunks(nchunks), bitmaps(bitmaps),
        chunk_bits(chunk_bits), super_counts(super_counts) {}
  MGARDX_CONT Task<FunctorType> GenTask(int queue_idx) {
    FunctorType functor(num_rows, nchunks, bitmaps, chunk_bits, super_counts);
    return Task(functor, 1, num_rows, (nchunks + SUPER - 1) / SUPER, 1, 1,
                SUPER, functor.shared_memory_size(), queue_idx,
                std::string(Name));
  }

private:
  SIZE num_rows, nchunks;
  SubArray<1, uint32_t, DeviceType> bitmaps, chunk_bits, super_counts;
};

// Shared part of the warp write and decode functors: thread t of a block
// holds the bitmap of chunk t of the super-chunk and, after end(), the
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

// Inclusive prefix over the lanes of the payload bits of their codes (0: no
// stored word), and their total: 5 k bits per sparse word and 32 per raw
// one, from four ballots.
template <typename DeviceType>
MGARDX_EXEC uint32_t payload_prefix(SubGroup<DeviceType> &sg, int lane,
                                    uint32_t code, uint32_t &total) {
  uint32_t k = code == RAW_CODE ? 0 : code;
  uint32_t b0 = sg.ballot((k & 1u) != 0), b1 = sg.ballot((k & 2u) != 0),
           b2 = sg.ballot((k & 4u) != 0), raw = sg.ballot(code == RAW_CODE);
  const uint32_t upto = (2u << lane) - 1;
  total = 5 * (popcount32(b0) + 2 * popcount32(b1) + 4 * popcount32(b2)) +
          32 * popcount32(raw);
  return 5 * (popcount32(b0 & upto) + 2 * popcount32(b1 & upto) +
              4 * popcount32(b2 & upto)) +
         32 * popcount32(raw & upto);
}

// The same for a value per chunk (payload bits): after end(), thread t has
// the sum of the values of the chunks before chunk t in the super-chunk.
template <typename DeviceType> struct WarpSumScan {
  uint32_t *warp_totals;
  uint32_t offset;
  MGARDX_EXEC void begin(SubGroup<DeviceType> &sg, int lane, SIZE warp,
                         uint32_t value) {
    uint32_t inclusive = warp_inclusive_scan(sg, lane, value);
    offset = inclusive - value;
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

// Position of a warp's marked chunks among the super-chunk's stored bitmaps:
// thread t of a block marks chunk t, and after end() has the index of its
// bitmap from the super-chunk's first.
template <typename DeviceType> struct WarpMarkScan {
  uint32_t *warp_totals;
  uint32_t mark, index;
  MGARDX_EXEC void begin(int lane, SIZE warp, uint32_t warp_mark) {
    mark = warp_mark;
    index = popcount32(mark & ((1u << lane) - 1));
    if (lane == 0) {
      warp_totals[warp] = popcount32(mark);
    }
  }
  MGARDX_EXEC bool marked(int lane) const { return (mark >> lane) & 1u; }
  MGARDX_EXEC void end(SIZE warp) {
    for (SIZE w = 0; w < warp; w++) {
      index += warp_totals[w];
    }
  }
};

// Sparse rows: the region of a super-chunk is assembled in shared memory
// (codes and payloads at the positions given by the scans of the chunk
// counts and payload bits), then stored coalesced.
template <typename T_bitplane, typename DeviceType>
class WriteWarpFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT WriteWarpFunctor() {}
  MGARDX_CONT WriteWarpFunctor(SIZE num_rows,
                               SubArray<2, T_bitplane, DeviceType> rows,
                               SubArray<1, RowTask, DeviceType> tasks,
                               SubArray<1, uint32_t, DeviceType> bitmaps,
                               SubArray<1, uint32_t, DeviceType> chunk_bits,
                               SubArray<1, uint32_t, DeviceType> super_offsets)
      : num_rows(num_rows), rows(rows), tasks(tasks), bitmaps(bitmaps),
        chunk_bits(chunk_bits), super_offsets(super_offsets) {
    Functor<DeviceType>();
  }
  MGARDX_EXEC void Operation1() {
    SubGroup<DeviceType> sg;
    lane = sg.lane();
    tid = FunctorBase<DeviceType>::GetThreadIdX();
    warp = tid / CHUNK;
    scan.warp_totals = (uint32_t *)FunctorBase<DeviceType>::GetSharedMemory();
    marks.warp_totals = scan.warp_totals + WARPS;
    bit_scan.warp_totals = scan.warp_totals + 2 * WARPS;
    region = scan.warp_totals + 3 * WARPS;
    task = *tasks(FunctorBase<DeviceType>::GetBlockIdY());
    num_words = rows.shape(1);
    nchunks = num_chunks(num_words);
    nsuper = num_super(num_words);
    s = FunctorBase<DeviceType>::GetBlockIdX();
    chunk = s * SUPER + tid;
    if (s == 0 && tid == 0) {
      write_header(task, num_words);
    }
    region_offset = region_size = 0;
    if (task.count != RAW_ROW) {
      uint32_t bitmap =
          chunk < nchunks ? *bitmaps(task.row * nchunks + chunk) : 0;
      scan.begin(sg, lane, warp, bitmap);
      marks.begin(lane, warp, sg.ballot(bitmap != 0));
      if (task.payload) {
        bit_scan.begin(sg, lane, warp,
                       chunk < nchunks ? *chunk_bits(task.row * nchunks + chunk)
                                       : 0);
        region_extent(super_offsets, num_rows, task, nsuper, s, region_offset,
                      region_size);
        for (SIZE i = tid; i < region_size; i += SUPER) {
          region[i] = 0;
        }
      }
    }
  }
  MGARDX_EXEC void Operation2() {
    SubGroup<DeviceType> sg;
    uint32_t *section = (uint32_t *)((Byte *)task.stream + task.section);
    SIZE first = s * SUPER + warp * CHUNK;
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
    marks.end(warp);
    uint32_t word_offset = *super_offsets(task.row * nsuper + s);
    uint32_t bitmap_offset = *super_offsets((num_rows + task.row) * nsuper + s);
    if (tid == 0) {
      section[s] = word_offset;
      section[nsuper + s] = bitmap_offset;
      if (task.payload) {
        section[2 * nsuper + s] = region_offset;
      }
    }
    uint32_t *mark_out = section + offset_words(task, nsuper);
    if (lane == 0 && first < nchunks) {
      mark_out[first / CHUNK] = marks.mark;
    }
    uint32_t *bitmap_out = mark_out + num_level2(num_words);
    if (marks.marked(lane)) {
      bitmap_out[bitmap_offset + marks.index] = scan.bitmap;
    }
    const uint32_t below = (1u << lane) - 1;
    const SIZE nk = first < nchunks ? (nchunks - first < CHUNK ? nchunks - first
                                                               : CHUNK)
                                    : 0;
    if (task.payload) {
      // The warp's chunks, each lane's word of the next one loaded ahead.
      auto load = [&](SIZE k, uint32_t bitmap) -> uint32_t {
        return (bitmap >> lane) & 1u
                   ? (uint32_t)*rows(task.row, (first + k) * CHUNK + lane)
                   : 0;
      };
      uint32_t bitmap = nk > 0 ? sg.shfl(scan.bitmap, 0) : 0;
      uint32_t x = load(0, bitmap);
      bit_scan.end(warp);
      uint32_t super_words =
          (s + 1 < nsuper ? *super_offsets(task.row * nsuper + s + 1)
                          : task.count) -
          word_offset;
      SIZE payload_start = CODE_BITS * super_words;
      for (SIZE k = 0; k < nk; k++) {
        uint32_t next_bitmap = k + 1 < nk ? sg.shfl(scan.bitmap, (int)k + 1) : 0;
        uint32_t next_x = load(k + 1, next_bitmap);
        if (bitmap != 0) {
          uint32_t offset = sg.shfl(scan.offset, (int)k);
          uint32_t bit_offset = sg.shfl(bit_scan.offset, (int)k);
          bool stored = (bitmap >> lane) & 1u;
          uint32_t code = stored ? sparse_code(x) : 0, total;
          uint32_t len = stored ? code_bits(code) : 0;
          uint32_t inclusive = payload_prefix(sg, lane, code, total);
          if (stored) {
            put_sparse_word<DeviceType>(
                region, offset + popcount32(bitmap & below),
                payload_start + bit_offset + inclusive - len, x);
          }
        }
        bitmap = next_bitmap;
        x = next_x;
      }
      return;
    }
    uint32_t *words = bitmap_out + task.chunks + word_offset;
    for (SIZE k = 0; k < nk; k++) {
      uint32_t bitmap = sg.shfl(scan.bitmap, (int)k);
      uint32_t offset = sg.shfl(scan.offset, (int)k);
      if ((bitmap >> lane) & 1u) {
        words[offset + popcount32(bitmap & below)] =
            (uint32_t)*rows(task.row, (first + k) * CHUNK + lane);
      }
    }
  }
  MGARDX_EXEC void Operation3() {
    if (task.count == RAW_ROW || !task.payload) {
      return;
    }
    uint32_t *out = (uint32_t *)((Byte *)task.stream + task.section) +
                    3 * nsuper + num_level2(num_words) + task.chunks +
                    region_offset;
    for (SIZE i = tid; i < region_size; i += SUPER) {
      out[i] = region[i];
    }
  }
  // Shared region words (sparse tasks: their largest super-chunk region).
  SIZE region_capacity = MAX_REGION;
  MGARDX_CONT size_t shared_memory_size() {
    return (3 * WARPS + region_capacity) * sizeof(uint32_t);
  }

private:
  SIZE num_rows;
  SubArray<2, T_bitplane, DeviceType> rows;
  SubArray<1, RowTask, DeviceType> tasks;
  SubArray<1, uint32_t, DeviceType> bitmaps, chunk_bits, super_offsets;
  WarpChunkScan<DeviceType> scan;
  WarpMarkScan<DeviceType> marks;
  WarpSumScan<DeviceType> bit_scan;
  uint32_t *region;
  RowTask task;
  uint32_t region_offset, region_size;
  int lane;
  SIZE tid, warp, num_words, nchunks, nsuper, s, chunk;
};

template <typename T_bitplane, typename DeviceType>
class WriteWarpKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "zero elimination write";
  using FunctorType = WriteWarpFunctor<T_bitplane, DeviceType>;
  MGARDX_CONT WriteWarpKernel(SIZE num_rows,
                              SubArray<2, T_bitplane, DeviceType> rows,
                              SubArray<1, RowTask, DeviceType> tasks,
                              SIZE num_tasks,
                              SubArray<1, uint32_t, DeviceType> bitmaps,
                              SubArray<1, uint32_t, DeviceType> chunk_bits,
                              SubArray<1, uint32_t, DeviceType> super_offsets,
                              SIZE region_capacity = MAX_REGION)
      : num_rows(num_rows), rows(rows), tasks(tasks), num_tasks(num_tasks),
        bitmaps(bitmaps), chunk_bits(chunk_bits), super_offsets(super_offsets),
        region_capacity(region_capacity) {}
  MGARDX_CONT Task<FunctorType> GenTask(int queue_idx) {
    FunctorType functor(num_rows, rows, tasks, bitmaps, chunk_bits,
                        super_offsets);
    functor.region_capacity = region_capacity;
    return Task(functor, 1, num_tasks, num_super(rows.shape(1)), 1, 1, SUPER,
                functor.shared_memory_size(), queue_idx, std::string(Name));
  }

private:
  SIZE num_rows;
  SubArray<2, T_bitplane, DeviceType> rows;
  SubArray<1, RowTask, DeviceType> tasks;
  SIZE num_tasks;
  SubArray<1, uint32_t, DeviceType> bitmaps, chunk_bits, super_offsets;
  SIZE region_capacity;
};

// Sparse rows: the super-chunk's region is copied (coalesced) into shared
// memory; each warp sums the payload bits of its chunks' codes, and then
// decodes its chunks in order from the warps' totals and a running sum.
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
    marks.warp_totals = scan.warp_totals + WARPS;
    warp_bits = scan.warp_totals + 2 * WARPS;
    region = warp_bits + WARPS;
    task = *tasks(FunctorBase<DeviceType>::GetBlockIdY());
    num_words = rows.shape(1);
    nchunks = num_chunks(num_words);
    nsuper = num_super(num_words);
    s = FunctorBase<DeviceType>::GetBlockIdX();
    chunk = s * SUPER + tid;
    section = (const uint32_t *)((const Byte *)task.stream + task.section);
    two_level = task.kind & TWO_LEVEL;
    if (task.count == RAW_ROW) {
      return;
    }
    if (two_level) {
      SIZE first = s * SUPER + warp * CHUNK;
      marks.begin(lane, warp,
                  first < nchunks
                      ? section[offset_words(task, nsuper) + first / CHUNK]
                      : 0);
      if (task.payload) {
        // The region into shared memory (independent of the marks).
        const uint32_t *src =
            section + 3 * nsuper + num_level2(num_words) + task.chunks;
        uint32_t offset = section[2 * nsuper + s];
        uint32_t end =
            s + 1 < nsuper ? section[2 * nsuper + s + 1] : task.payload;
        for (uint32_t i = (uint32_t)tid; i < end - offset;
             i += (uint32_t)SUPER) {
          region[i] = src[offset + i];
        }
        payload_start = CODE_BITS * super_count(section, task, nsuper, s);
      }
    } else {
      scan.begin(sg, lane, warp, chunk < nchunks ? section[nsuper + chunk] : 0);
    }
  }
  MGARDX_EXEC void Operation2() {
    if (task.count == RAW_ROW || !two_level) {
      return;
    }
    SubGroup<DeviceType> sg;
    marks.end(warp);
    const uint32_t *bitmaps =
        section + offset_words(task, nsuper) + num_level2(num_words);
    uint32_t first_bitmap = section[nsuper + s];
    scan.begin(sg, lane, warp,
               marks.marked(lane) ? bitmaps[first_bitmap + marks.index] : 0);
  }
  // The warp's chunks first + k, k < warp_chunks(first), have their word lane
  // at out[k * CHUNK] (warp_out), valid while k * CHUNK + lane is below
  // warp_limit(first); loops use these 32-bit indices.
  MGARDX_EXEC int warp_chunks(SIZE first) const {
    return first < nchunks ? (int)(nchunks - first < CHUNK ? nchunks - first
                                                           : CHUNK)
                           : 0;
  }
  MGARDX_EXEC uint32_t warp_limit(SIZE first) const {
    SIZE begin = first * CHUNK;
    return begin < num_words ? (uint32_t)(num_words - begin < CHUNK * CHUNK
                                              ? num_words - begin
                                              : CHUNK * CHUNK)
                             : 0;
  }
  MGARDX_EXEC T_bitplane *warp_out(SIZE first) {
    return rows(task.row, 0) + first * CHUNK + lane;
  }
  MGARDX_EXEC void Operation3() {
    SubGroup<DeviceType> sg;
    const SIZE first = s * SUPER + warp * CHUNK;
    const int chunks = warp_chunks(first);
    const uint32_t limit = warp_limit(first);
    T_bitplane *out = warp_out(first);
    if (task.count == RAW_ROW) {
      const uint32_t *in = section + first * CHUNK + lane;
      for (int k = 0; k < chunks; k++) {
        if ((uint32_t)k * 32u + lane < limit) {
          out[k * 32] = in[k * 32];
        }
      }
      return;
    }
    scan.end(warp);
    const uint32_t below = (1u << lane) - 1;
    if (task.payload) {
      // Payload bits of the warp's stored words (consecutive codes).
      uint32_t base = sg.shfl(scan.offset, 0);
      uint32_t count = warp_sum(sg, lane, popcount32(scan.bitmap));
      uint32_t bits = 0;
      for (uint32_t i = lane; i < count; i += 32u) {
        bits += code_bits(get_bits(region, CODE_BITS * (base + i), CODE_BITS));
      }
      bits = warp_sum(sg, lane, bits);
      if (lane == 0) {
        warp_bits[warp] = bits;
      }
      return;
    }
    const uint32_t *words =
        two_level
            ? section + 2 * nsuper + num_level2(num_words) + task.chunks
            : section + nsuper + nchunks;
    words += section[s];
    for (int k = 0; k < chunks; k++) {
      uint32_t bitmap = sg.shfl(scan.bitmap, k);
      uint32_t offset = sg.shfl(scan.offset, k);
      if ((uint32_t)k * 32u + lane < limit) {
        out[k * 32] = (bitmap >> lane) & 1u
                          ? words[offset + popcount32(bitmap & below)]
                          : 0;
      }
    }
  }
  MGARDX_EXEC void Operation4() {
    if (task.count == RAW_ROW || !task.payload) {
      return;
    }
    SubGroup<DeviceType> sg;
    const SIZE first = s * SUPER + warp * CHUNK;
    const int chunks = warp_chunks(first);
    const uint32_t limit = warp_limit(first);
    T_bitplane *out = warp_out(first);
    const uint32_t below = (1u << lane) - 1;
    // Bit positions in the region (32 bits suffice).
    uint32_t pos = (uint32_t)payload_start;
    for (SIZE w = 0; w < warp; w++) {
      pos += warp_bits[w];
    }
    for (int k = 0; k < chunks; k++) {
      uint32_t bitmap = sg.shfl(scan.bitmap, k);
      uint32_t x = 0;
      if (bitmap != 0) {
        uint32_t offset = sg.shfl(scan.offset, k);
        bool stored = (bitmap >> lane) & 1u;
        uint32_t code =
            stored ? get_bits(region,
                              CODE_BITS * (offset + popcount32(bitmap & below)),
                              CODE_BITS)
                   : 0;
        uint32_t len = stored ? code_bits(code) : 0, total;
        uint32_t inclusive = payload_prefix(sg, lane, code, total);
        if (stored) {
          x = get_sparse_word(region, pos + inclusive - len, code);
        }
        pos += total;
      }
      if ((uint32_t)k * 32u + lane < limit) {
        out[k * 32] = x;
      }
    }
  }
  // Shared region words (sparse tasks: their largest super-chunk region).
  SIZE region_capacity = MAX_REGION;
  MGARDX_CONT size_t shared_memory_size() {
    return (3 * WARPS + region_capacity) * sizeof(uint32_t);
  }

private:
  SubArray<2, T_bitplane, DeviceType> rows;
  SubArray<1, RowTask, DeviceType> tasks;
  WarpChunkScan<DeviceType> scan;
  WarpMarkScan<DeviceType> marks;
  RowTask task;
  const uint32_t *section;
  uint32_t *warp_bits, *region;
  SIZE payload_start;
  bool two_level;
  int lane;
  SIZE tid, warp, num_words, nchunks, nsuper, s, chunk;
};

template <typename T_bitplane, typename DeviceType>
class DecodeWarpKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "zero elimination decode";
  using FunctorType = DecodeWarpFunctor<T_bitplane, DeviceType>;
  MGARDX_CONT DecodeWarpKernel(SubArray<2, T_bitplane, DeviceType> rows,
                               SubArray<1, RowTask, DeviceType> tasks,
                               SIZE num_tasks,
                               SIZE region_capacity = MAX_REGION)
      : rows(rows), tasks(tasks), num_tasks(num_tasks),
        region_capacity(region_capacity) {}
  MGARDX_CONT Task<FunctorType> GenTask(int queue_idx) {
    FunctorType functor(rows, tasks);
    functor.region_capacity = region_capacity;
    return Task(functor, 1, num_tasks, num_super(rows.shape(1)), 1, 1, SUPER,
                functor.shared_memory_size(), queue_idx, std::string(Name));
  }

private:
  SubArray<2, T_bitplane, DeviceType> rows;
  SubArray<1, RowTask, DeviceType> tasks;
  SIZE num_tasks;
  SIZE region_capacity;
};

// The largest region a super-chunk of the sparse rows of tasks [first, last)
// can have (0 without sparse rows): the shared region words that a writer or
// decoder launch over these tasks needs.
inline SIZE region_capacity(const std::vector<RowTask> &tasks, SIZE first,
                            SIZE last) {
  SIZE capacity = 0;
  for (SIZE i = first; i < last; i++) {
    if (tasks[i].count != RAW_ROW) {
      capacity = std::max(capacity, std::min((SIZE)tasks[i].payload, MAX_REGION));
    }
  }
  return capacity;
}

} // namespace zero_elimination
} // namespace MDR
} // namespace mgard_x

#endif
