/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: October 8, 2026
 */

#ifndef MGARD_X_MDR_BP_ENCODER_WARP_TRANSPOSE_HPP
#define MGARD_X_MDR_BP_ENCODER_WARP_TRANSPOSE_HPP

#include "../../RuntimeX/RuntimeX.h"
#include "../LosslessCompressor/ZeroElimination.hpp"
#include "BPEncoderRegisterBlock.hpp"
#include "BitplaneEncoderInterface.hpp"
#include "SignificanceCoding.hpp"
#include <type_traits>
#include <utility>

namespace mgard_x {
namespace MDR {

// The MDR-X bitplane encoder (BPEncoderWarpTranspose, at the end of this
// file). It writes the Locality Block layout of HP-MDR: word w of a bitplane
// row holds the bits of the 32 consecutive coefficients [32 w, 32 w + 32),
// here coefficient 32 w + d at bit d, so that the leading bitplanes of smooth
// blocks are zero words. On CUDA a warp encodes a tile of 32 words: each lane
// loads one coefficient (coalesced, the thread mapping of HP-MDR's Register
// Shuffle encoder) and a 32 x 32 bit transpose in five shuffle stages gives
// each lane one word, which shared memory then stores coalesced. Other
// backends, and 64-bit words, use portable kernels with one thread per word.
// The Register Block layout of HP-MDR (words of strided coefficients; mdr-x
// -sw, and always with the negabinary encoding) is written with the kernels
// of BPEncoderRegisterBlock.hpp.

// Transposes a 32 x 32 bit matrix held one row per lane: on return, bit c of
// lane l is bit l of lane c on entry.
template <typename DeviceType>
MGARDX_EXEC uint32_t warp_transpose32(SubGroup<DeviceType> &sg, int lane,
                                      uint32_t x) {
  const uint32_t masks[5] = {0x0000ffffu, 0x00ff00ffu, 0x0f0f0f0fu,
                             0x33333333u, 0x55555555u};
#pragma unroll
  for (int i = 0; i < 5; i++) {
    const int s = 16 >> i;
    const uint32_t m = masks[i];
    uint32_t o = sg.shfl(x, lane ^ s);
    x = (lane & s) ? (x & ~m) | ((o >> s) & m) : (x & m) | ((o << s) & ~m);
  }
  return x;
}

// One stage of warp_transpose_reduce32: v[0..S) += the other half's v[S..2S)
// of lane ^ S. S is a template parameter so that the loops unroll and v stays
// in registers (the compiler kept the loop over s rolled, indexing v).
template <int S, typename T, typename DeviceType>
MGARDX_EXEC void transpose_reduce_stage(SubGroup<DeviceType> &sg, int lane,
                                        T *v) {
  const bool upper = lane & S;
#pragma unroll
  for (int i = 0; i < S; i++) {
    // Selects of values, not of v[i] / v[i + S] (address selects put v in
    // local memory).
    const T lo = v[i], hi = v[i + S];
    T send = upper ? lo : hi;
    T keep = upper ? hi : lo;
    v[i] = keep + sg.shfl(send, lane ^ S);
  }
  if constexpr (S > 1) {
    transpose_reduce_stage<S / 2>(sg, lane, v);
  }
}

// Sums v[0..32) over the 32 lanes: lane l returns the sum of v[l]
// (31 exchanges instead of 32 butterfly reductions).
template <typename T, typename DeviceType>
MGARDX_EXEC T warp_transpose_reduce32(SubGroup<DeviceType> &sg, int lane,
                                      T *v) {
  transpose_reduce_stage<16>(sg, lane, v);
  return v[0];
}

// Squared errors of the single-precision sums of BPEncoderWarpTransposeFunctor
// (ControlL2) are within a relative 2^-17 of exact: each lane adds at most 32
// terms, the warp the lanes' sums in 5 more steps, and the remainders are
// exact for float data and within 17 roundings for double data (leaving out
// underflow, far below the last bitplane).
// ComposedRefactor::StoreMetadata scales the collected errors by this, so
// that they bound the exact ones.
constexpr double L2_ERROR_MARGIN = 1 + 0x1p-16;

// Lanes of the warp holding the same value (CUDA, sm_70+).
MGARDX_EXEC uint32_t warp_match_any(uint32_t x) {
#if defined(__CUDA_ARCH__)
  return __match_any_sync(0xffffffffu, x);
#else
  return 1u;
#endif
}

// Sum of x over the warp.
MGARDX_EXEC uint32_t warp_reduce_add(uint32_t x) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return __reduce_add_sync(0xffffffffu, x);
#elif defined(__CUDA_ARCH__)
  for (int offset = 16; offset > 0; offset /= 2) {
    x += __shfl_xor_sync(0xffffffffu, x, offset);
  }
  return x;
#else
  return x;
#endif
}

MGARDX_EXEC int count_leading_zeros(uint32_t x) {
#if defined(__CUDA_ARCH__)
  return __clz(x);
#else
  return x ? __builtin_clz(x) : 32;
#endif
}
MGARDX_EXEC int count_leading_zeros(uint64_t x) {
#if defined(__CUDA_ARCH__)
  return __clzll(x);
#else
  return x ? __builtin_clzll(x) : 64;
#endif
}

MGARDX_EXEC void sync_block() {
#if defined(__CUDA_ARCH__)
  __syncthreads();
#endif
}

// Bits 0, 8, 16, 24 of x as bits 0-3.
MGARDX_EXEC uint32_t gather_byte_bits(uint32_t x) {
  return ((x & 0x01010101u) * 0x01020408u) >> 24 & 0xfu;
}

// Binary bitplane encoding in the Locality Block layout (word w of a row
// holds the bits of coefficients [32 w, 32 w + 32), coefficient 32 w + d at
// bit d) with one lane per coefficient: a 32-lane sub-group loads 32
// consecutive coefficients (coalesced) and a bit transpose turns their
// fixed-point values into one bitplane word per lane. A tile of 32 words is
// handled by NUM_BITPLANES / 32 sub-groups, sub-group h taking bits
// [32 h, 32 h + 32) (the top one also the signs); the words are staged in
// shared memory so that rows are stored coalesced. The per-bitplane squared
// errors are summed per lane in single precision (collect_errors), then
// reduced per thread block into level_errors_workspace(b, block);
// BPErrorSumKernel adds the blocks up. Same rows and signs as
// BPEncoderWarpTransposePortableFunctor, and the same errors up to rounding
// (L2_ERROR_MARGIN).
// With ze_bitmaps given, a tile being a zero-elimination chunk, it also
// writes the chunk bitmaps of every row (zero_elimination::SuperCountKernel
// then gives the super-chunk counts), and with ze_bits the chunks' sparse-word
// payload bits. With sign_group_size > 0
// (significance-coded signs, SignificanceCoding.hpp), row 0 gets the tile's
// packed signs instead of the sign row, and sign_counts, sign_segment_bits
// (zero on entry) their counts: the same as SignPackKernel.
template <typename T_data, typename T_fp, typename T_bitplane, typename T_error,
          int NUM_BITPLANES, bool ControlL2, typename DeviceType>
class BPEncoderWarpTransposeFunctor : public Functor<DeviceType> {
public:
  static constexpr SIZE WORDS = 32;
  static constexpr int HALVES = NUM_BITPLANES / 32; // sub-groups per tile
  static constexpr SIZE ROWS = 33;  // rows staged per sub-group (+ sign)
  static constexpr SIZE PITCH = WORDS + 1; // conflict-free transposed stores
  static constexpr SIZE WARPS = 4;
  static constexpr uint32_t NO_GROUP = 0xffu;
  // Per warp, significance-coded signs: the count of each group, the nonzero
  // coefficients of each word, the slot of each nonzero coefficient
  // (group << 10 | position in the group), and a byte per packed sign.
  static constexpr SIZE SIGN_WORDS = significance::MAX_GROUPS + WORDS +
                                     WORDS * WORDS / 2 + WORDS * WORDS / 4;
  static_assert(NUM_BITPLANES % 32 == 0 && WARPS % HALVES == 0);
  MGARDX_CONT BPEncoderWarpTransposeFunctor() {}
  MGARDX_CONT
  BPEncoderWarpTransposeFunctor(
      SIZE n, SubArray<1, T_data, DeviceType> abs_max,
      SubArray<1, T_data, DeviceType> v,
      SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
      SubArray<2, T_error, DeviceType> level_errors_workspace,
      SubArray<1, uint32_t, DeviceType> ze_bitmaps,
      SubArray<1, uint32_t, DeviceType> ze_bits, int sign_group_size,
      SubArray<1, uint32_t, DeviceType> sign_counts,
      SubArray<1, uint32_t, DeviceType> sign_segment_bits)
      : n(n), abs_max(abs_max), v(v), encoded_bitplanes(encoded_bitplanes),
        level_errors_workspace(level_errors_workspace), ze_bitmaps(ze_bitmaps),
        ze_bits(ze_bits), sign_group_size(sign_group_size),
        sign_counts(sign_counts), sign_segment_bits(sign_segment_bits) {
    Functor<DeviceType>();
  }

  static MGARDX_CONT_EXEC size_t shared_memory_bytes(bool signs = false) {
    return WARPS * ROWS * PITCH * sizeof(uint32_t) +
           (ControlL2 ? WARPS * ROWS * sizeof(T_error) : 0) +
           (signs ? WARPS * SIGN_WORDS * sizeof(uint32_t) : 0);
  }

  // Slot of the coefficient of word j (top sub-group): its group g (the
  // group of its first nonzero bit) and its position among the tile's
  // coefficients of group g.
  MGARDX_EXEC void add_slot(SubGroup<DeviceType> &sg, int lane, int j,
                            uint32_t g, uint32_t *count, uint32_t *nonzero,
                            uint16_t *slot) {
    uint32_t mask = sg.ballot(g != NO_GROUP);
    if (lane == 0) {
      nonzero[j] = mask;
    }
    if (mask == 0) {
      return;
    }
    uint32_t same = warp_match_any(g);
    uint32_t base = g != NO_GROUP ? count[g] : 0;
    sg.sync();
    if (g != NO_GROUP) {
      if (lane == sg.ffs(same) - 1) {
        count[g] = base + zero_elimination::popcount32(same);
      }
      slot[j * WORDS + lane] =
          (uint16_t)(g << 10 |
                     (base + zero_elimination::popcount32(
                                 same & ((1u << lane) - 1))));
    }
    sg.sync();
  }

  // Packs the signs of the tile into staging row 0 (top sub-group; staging
  // row 0 holds the sign words) and records the counts: each sign goes to the
  // byte of its packed position, and the bytes are then gathered into words.
  MGARDX_EXEC void pack_signs(SubGroup<DeviceType> &sg, int lane,
                              uint32_t *staging, uint32_t *count,
                              uint32_t *nonzero, uint16_t *slot,
                              uint8_t *packed, SIZE tile,
                              SIZE words_in_tile) {
    const SIZE num_words = n / WORDS;
    const SIZE ntiles = significance::num_tiles(num_words);
    const SIZE nseg = significance::num_segments(num_words);
    const int G = (int)significance::num_groups(NUM_BITPLANES,
                                                sign_group_size);
    // Group ends: inclusive scan of the counts (two per lane).
    uint32_t c0 = count[lane], c1 = count[lane + 32];
    uint32_t s0 = zero_elimination::warp_inclusive_scan(sg, lane, c0);
    uint32_t s1 = zero_elimination::warp_inclusive_scan(sg, lane, c1) +
                  sg.shfl(s0, 31);
    for (int k = 0; k < 2; k++) {
      int g = lane + 32 * k;
      uint32_t c = k == 0 ? c0 : c1;
      if (g < G) {
        // Where group g's signs end in the tile.
        *sign_counts(g * ntiles + tile) = k == 0 ? s0 : s1;
        if (c) {
          Atomic<uint32_t, AtomicGlobalMemory, AtomicDeviceScope,
                 DeviceType>::Add(sign_segment_bits(g * nseg +
                                                    tile / significance::SEGMENT),
                                  c);
        }
      }
    }
    // Group starts.
    uint32_t total = sg.shfl(s1, 31);
    sg.sync();
    count[lane] = s0 - c0;
    count[lane + 32] = s1 - c1;
    sg.sync();
    for (int j = 0; j < (int)words_in_tile; j++) {
      if ((nonzero[j] >> lane) & 1u) {
        uint32_t s = slot[j * WORDS + lane];
        packed[count[s >> 10] + (s & 1023u)] = (staging[j] >> lane) & 1u;
      }
    }
    sg.sync();
    const uint32_t *bytes = (const uint32_t *)packed + 8 * lane;
    uint32_t x = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) {
      x |= gather_byte_bits(bytes[i]) << (4 * i);
    }
    // Bytes past the signs were not written.
    uint32_t first = 32 * lane;
    if (total < first + 32) {
      x = total > first ? x & ((1u << (total - first)) - 1) : 0;
    }
    staging[lane] = x;
    sg.sync();
  }

  // Squared errors of sub-group H in single precision (a = |shifted|, fp its
  // integer part): e[32 - k] += (a mod 2^(32 H + k))^2 / 2^(64 H) for k in
  // [0, 32), with a mod 2^b built bit by bit in two independent chains (the
  // 2^-32 scaling keeps the upper sub-group of doubles in float range). For
  // float data the remainders are exact: they are parts of a's significand.
  template <int H>
  MGARDX_EXEC void collect_errors(float *e, T_fp fp, T_data a) {
    const uint32_t x = (uint32_t)(fp >> (32 * H));
    float d0, d1;
    if constexpr (std::is_same<T_data, float>::value) {
      d0 = a - (float)fp;
      d1 = a - (float)(fp & ~(T_fp)0xffff);
    } else {
      constexpr T_error UNIT = H == 0 ? 1.0 : 0x1p-32;
      const T_error mantissa = a - (T_error)fp;
      d0 = (float)(((T_error)(fp & (((T_fp)1 << (32 * H)) - 1)) + mantissa) *
                   UNIT);
      d1 = (float)(((T_error)(fp & (((T_fp)1 << (32 * H + 16)) - 1)) +
                    mantissa) *
                   UNIT);
    }
    collect_steps(e, x, d0, d1, std::make_integer_sequence<int, 16>{});
  }

  template <int... K>
  MGARDX_EXEC void collect_steps(float *e, uint32_t x, float &d0, float &d1,
                                 std::integer_sequence<int, K...>) {
    (collect_step<K>(e, x, d0, d1), ...);
  }

  template <int K>
  MGARDX_EXEC void collect_step(float *e, uint32_t x, float &d0, float &d1) {
    e[32 - K] = fmaf(d0, d0, e[32 - K]);
    e[16 - K] = fmaf(d1, d1, e[16 - K]);
    if (x & (1u << K)) {
      d0 += (float)(1u << K);
    }
    if (x & (1u << (16 + K))) {
      d1 += (float)(1u << (16 + K));
    }
  }

  MGARDX_EXEC void Operation1() {
    SubGroup<DeviceType> sg;
    const int lane = sg.lane();
    const SIZE warp = FunctorBase<DeviceType>::GetThreadIdX() / WORDS;
    const int h = (int)(warp % HALVES);
    const bool top = h == HALVES - 1;
    uint32_t *staging =
        (uint32_t *)FunctorBase<DeviceType>::GetSharedMemory() +
        warp * ROWS * PITCH;
    T_error *block_errors = staged_errors();
    const int exp = abs_max_exponent();

    // Squared errors: row 0's (no bitplane) in T_error, the others per
    // sub-group in float (collect_errors).
    T_error error0 = 0;
    float errors[ROWS];
    if constexpr (ControlL2) {
#pragma unroll
      for (int b = 0; b < (int)ROWS; b++) {
        errors[b] = 0;
      }
    }
    SIZE tile = (FunctorBase<DeviceType>::GetBlockIdX() * WARPS + warp) /
                HALVES;
    SIZE num_words = n / WORDS;
    SIZE first = tile * WORDS;
    // Significance-coded signs (top sub-group).
    const bool signs = top && sign_group_size > 0;
    uint32_t *group_count =
        (uint32_t *)((Byte *)FunctorBase<DeviceType>::GetSharedMemory() +
                     shared_memory_bytes(false)) +
        warp * SIGN_WORDS;
    uint32_t *nonzero = group_count + significance::MAX_GROUPS;
    uint16_t *slot = (uint16_t *)(nonzero + WORDS);
    uint8_t *packed = (uint8_t *)(slot + WORDS * WORDS);
    // Group of bitplane b: b * inverse >> 16 (exact for b <= 64).
    const uint32_t inverse =
        signs ? (65536u + sign_group_size - 1) / sign_group_size : 0;
    if (signs) {
      group_count[lane] = 0;
      group_count[lane + 32] = 0;
      sg.sync();
    }
    if (first < num_words) {
      SIZE words_in_tile =
          num_words - first < WORDS ? num_words - first : WORDS;
      T_data scale = exp > 0 ? (T_data)((T_fp)1 << (NUM_BITPLANES - exp))
                             : (T_data)pow(2, NUM_BITPLANES - exp);
      // Zero-elimination bitmaps and payload bits: lane l builds those of
      // staged row 32 - l, and the sign row's (top sub-group) in
      // sign_bitmap, sign_bits.
      uint32_t row_bitmap = 0, sign_bitmap = 0, row_bits = 0, sign_bits = 0;
      T_data next = *v(first * WORDS + lane);
      for (int j = 0; j < (int)words_in_tile; j++) {
        T_data shifted = next * scale;
        if (j + 1 < (int)words_in_tile) {
          next = *v((first + j + 1) * WORDS + lane);
        }
        T_fp fp = (T_fp)fabs(shifted);
        if (top) {
          T_bitplane s = sg.ballot(signbit(shifted) != 0);
          if (lane == 0) {
            staging[j] = s;
          }
          sign_bitmap |= (uint32_t)(s != 0) << j;
          sign_bits += zero_elimination::sparse_bits(s);
        }
        if (signs) {
          // The group of the coefficient's first nonzero bit.
          uint32_t g = fp != 0 ? (uint32_t)count_leading_zeros(fp) * inverse >>
                                     16
                               : NO_GROUP;
          add_slot(sg, lane, j, g, group_count, nonzero, slot);
        }
        // Bit 32 h + l of fp goes to staged row 32 - l.
        uint32_t x = warp_transpose32(sg, lane, (uint32_t)(fp >> (32 * h)));
        staging[(32 - lane) * PITCH + j] = x;
        row_bitmap |= (uint32_t)(x != 0) << j;
        row_bits += zero_elimination::sparse_bits(x);
        if constexpr (ControlL2) {
          if constexpr (HALVES == 1) {
            collect_errors<0>(errors, fp, fabs(shifted));
          } else {
            if (h == 0) {
              collect_errors<0>(errors, fp, fabs(shifted));
            } else {
              collect_errors<HALVES - 1>(errors, fp, fabs(shifted));
            }
          }
          if (top) {
            error0 += shifted * shifted;
          }
        }
      }
      sg.sync();
      if (signs) {
        pack_signs(sg, lane, staging, group_count, nonzero, slot, packed,
                   tile, words_in_tile);
      }
      // Staged row r is row r + 32 (HALVES - 1 - h) of the encoded bitplanes.
      const SIZE row_offset = 32 * (HALVES - 1 - h);
      if (lane < (int)words_in_tile) {
#pragma unroll 8
        for (int r = top ? 0 : 1; r < (int)ROWS; r++) {
          *encoded_bitplanes(r + row_offset, first + lane) =
              staging[r * PITCH + lane];
        }
      }
      if (ze_bitmaps.data() != nullptr) {
        const SIZE nchunks = zero_elimination::num_chunks(num_words);
        *ze_bitmaps((32 - lane + row_offset) * nchunks + tile) = row_bitmap;
        if (top && lane == 0) {
          *ze_bitmaps(row_offset * nchunks + tile) = sign_bitmap;
        }
        if (ze_bits.data() != nullptr) {
          *ze_bits((32 - lane + row_offset) * nchunks + tile) = row_bits;
          if (top && lane == 0) {
            *ze_bits(row_offset * nchunks + tile) = sign_bits;
          }
        }
      }
    }
    if constexpr (ControlL2) {
      // The sub-group's sums (lane l: row l), back in T_error without the
      // 2^-64 scaling of the upper sub-group.
      const T_error unit = HALVES > 1 && top ? 0x1p64 : 1.0;
      float sum = warp_transpose_reduce32(sg, lane, errors);
      float sum32 = errors[32];
      for (int offset = 16; offset > 0; offset /= 2) {
        sum32 += sg.shfl(sum32, lane ^ offset);
        error0 += sg.shfl(error0, lane ^ offset);
      }
      block_errors[warp * ROWS + lane] =
          lane == 0 ? error0 : (T_error)sum * unit;
      if (lane == 0) {
        block_errors[warp * ROWS + 32] = (T_error)sum32 * unit;
      }
    }
  }

  MGARDX_EXEC void Operation2() {
    if constexpr (ControlL2) {
      const T_error *block_errors = staged_errors();
      const int exp = abs_max_exponent();
      for (SIZE b = FunctorBase<DeviceType>::GetThreadIdX();
           b < NUM_BITPLANES + 1; b += FunctorBase<DeviceType>::GetBlockDimX()) {
        // Error b is staged entry b - 32 (HALVES - 1 - h) of sub-group h.
        int h = HALVES - 1 - (b == 0 ? 0 : (int)(b - 1) / 32);
        SIZE local = b - 32 * (HALVES - 1 - h);
        T_error e = 0;
        for (SIZE w = h; w < WARPS; w += HALVES) {
          e += block_errors[w * ROWS + local];
        }
        *level_errors_workspace(b, FunctorBase<DeviceType>::GetBlockIdX()) =
            ldexp(e, 2 * (-NUM_BITPLANES + exp));
      }
    }
  }

  MGARDX_CONT size_t shared_memory_size() {
    return shared_memory_bytes(sign_group_size > 0);
  }

private:
  // Recomputed in each Operation instead of kept in members between them
  // (functor members written in device code can put the functor in local
  // memory).
  MGARDX_EXEC T_error *staged_errors() {
    return (T_error *)((uint32_t *)FunctorBase<DeviceType>::GetSharedMemory() +
                       WARPS * ROWS * PITCH);
  }
  MGARDX_EXEC int abs_max_exponent() {
    int e;
    frexp(*abs_max((IDX)0), &e);
    return e;
  }

  SIZE n;
  SubArray<1, T_data, DeviceType> abs_max;
  SubArray<1, T_data, DeviceType> v;
  SubArray<2, T_bitplane, DeviceType> encoded_bitplanes;
  SubArray<2, T_error, DeviceType> level_errors_workspace;
  SubArray<1, uint32_t, DeviceType> ze_bitmaps, ze_bits;
  int sign_group_size;
  SubArray<1, uint32_t, DeviceType> sign_counts, sign_segment_bits;
};

template <typename T_data, typename T_fp, typename T_bitplane, typename T_error,
          int NUM_BITPLANES, bool ControlL2, typename DeviceType>
class BPEncoderWarpTransposeKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "warp transpose bp encoder";
  using FunctorType =
      BPEncoderWarpTransposeFunctor<T_data, T_fp, T_bitplane, T_error,
                                    NUM_BITPLANES, ControlL2, DeviceType>;
  // Thread blocks (= partial error sums) for n coefficients.
  static SIZE num_blocks(SIZE n) {
    constexpr SIZE tiles_per_block = FunctorType::WARPS / FunctorType::HALVES;
    SIZE tiles = (n / 32 + 31) / 32;
    return std::max((SIZE)1, (tiles + tiles_per_block - 1) / tiles_per_block);
  }
  MGARDX_CONT
  BPEncoderWarpTransposeKernel(
      SIZE n, SubArray<1, T_data, DeviceType> abs_max,
      SubArray<1, T_data, DeviceType> v,
      SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
      SubArray<2, T_error, DeviceType> level_errors_workspace,
      SubArray<1, uint32_t, DeviceType> ze_bitmaps = {},
      SubArray<1, uint32_t, DeviceType> ze_bits = {}, int sign_group_size = 0,
      SubArray<1, uint32_t, DeviceType> sign_counts = {},
      SubArray<1, uint32_t, DeviceType> sign_segment_bits = {})
      : n(n), abs_max(abs_max), v(v), encoded_bitplanes(encoded_bitplanes),
        level_errors_workspace(level_errors_workspace), ze_bitmaps(ze_bitmaps),
        ze_bits(ze_bits), sign_group_size(sign_group_size),
        sign_counts(sign_counts), sign_segment_bits(sign_segment_bits) {}
  MGARDX_CONT Task<FunctorType> GenTask(int queue_idx) {
    FunctorType functor(n, abs_max, v, encoded_bitplanes,
                        level_errors_workspace, ze_bitmaps, ze_bits,
                        sign_group_size, sign_counts, sign_segment_bits);
    return Task(functor, 1, 1, num_blocks(n), 1, 1, FunctorType::WARPS * 32,
                functor.shared_memory_size(), queue_idx, std::string(Name));
  }

private:
  SIZE n;
  SubArray<1, T_data, DeviceType> abs_max;
  SubArray<1, T_data, DeviceType> v;
  SubArray<2, T_bitplane, DeviceType> encoded_bitplanes;
  SubArray<2, T_error, DeviceType> level_errors_workspace;
  SubArray<1, uint32_t, DeviceType> ze_bitmaps, ze_bits;
  int sign_group_size;
  SubArray<1, uint32_t, DeviceType> sign_counts, sign_segment_bits;
};

// level_errors(b) = sum of workspace(b, 0 .. num_partials), one block per
// row, in a fixed order.
template <typename T_error, typename DeviceType>
class BPErrorSumFunctor : public Functor<DeviceType> {
public:
  static constexpr SIZE THREADS = 256;
  MGARDX_CONT BPErrorSumFunctor() {}
  MGARDX_CONT BPErrorSumFunctor(SIZE num_partials,
                                SubArray<2, T_error, DeviceType> workspace,
                                SubArray<1, T_error, DeviceType> level_errors)
      : num_partials(num_partials), workspace(workspace),
        level_errors(level_errors) {
    Functor<DeviceType>();
  }
  MGARDX_EXEC void Operation1() {
    SubGroup<DeviceType> sg;
    sums = (T_error *)FunctorBase<DeviceType>::GetSharedMemory();
    SIZE tid = FunctorBase<DeviceType>::GetThreadIdX();
    SIZE row = FunctorBase<DeviceType>::GetBlockIdX();
    T_error e = 0;
    for (SIZE i = tid; i < num_partials; i += THREADS) {
      e += *workspace(row, i);
    }
    for (int offset = sg.size() / 2; offset > 0; offset /= 2) {
      e += sg.shfl(e, sg.lane() ^ offset);
    }
    if (sg.lane() == 0) {
      sums[tid / sg.size()] = e;
    }
  }
  MGARDX_EXEC void Operation2() {
    if (FunctorBase<DeviceType>::GetThreadIdX() == 0) {
      SubGroup<DeviceType> sg;
      T_error e = 0;
      for (SIZE w = 0; w < THREADS / sg.size(); w++) {
        e += sums[w];
      }
      *level_errors(FunctorBase<DeviceType>::GetBlockIdX()) = e;
    }
  }
  MGARDX_CONT size_t shared_memory_size() { return THREADS * sizeof(T_error); }

private:
  SIZE num_partials;
  SubArray<2, T_error, DeviceType> workspace;
  SubArray<1, T_error, DeviceType> level_errors;
  T_error *sums;
};

template <typename T_error, typename DeviceType>
class BPErrorSumKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "bp error sum";
  using FunctorType = BPErrorSumFunctor<T_error, DeviceType>;
  MGARDX_CONT BPErrorSumKernel(SIZE num_rows, SIZE num_partials,
                               SubArray<2, T_error, DeviceType> workspace,
                               SubArray<1, T_error, DeviceType> level_errors)
      : num_rows(num_rows), num_partials(num_partials), workspace(workspace),
        level_errors(level_errors) {}
  MGARDX_CONT Task<FunctorType> GenTask(int queue_idx) {
    FunctorType functor(num_partials, workspace, level_errors);
    return Task(functor, 1, 1, num_rows, 1, 1, FunctorType::THREADS,
                functor.shared_memory_size(), queue_idx, std::string(Name));
  }

private:
  SIZE num_rows, num_partials;
  SubArray<2, T_error, DeviceType> workspace;
  SubArray<1, T_error, DeviceType> level_errors;
};

// In-register transpose of a 32 x 32 bit matrix: on return, bit c of m[r] is
// bit r of m[c] on entry.
MGARDX_EXEC void transpose32(uint32_t *m) {
  const uint32_t masks[5] = {0x0000ffffu, 0x00ff00ffu, 0x0f0f0f0fu,
                             0x33333333u, 0x55555555u};
#pragma unroll
  for (int i = 0; i < 5; i++) {
    const int s = 16 >> i;
#pragma unroll
    for (int k = 0; k < 32; k++) {
      if ((k & s) == 0) {
        uint32_t t = ((m[k] >> s) ^ m[k + s]) & masks[i];
        m[k + s] ^= t;
        m[k] ^= t << s;
      }
    }
  }
}

// Decoders use tile_values from this many bitplanes on (fewer: a broadcast
// of every word per bitplane is cheaper).
static constexpr int TRANSPOSED_DECODE = 4;

// The fixed-point values of a tile's coefficients from their bitplane words
// (lane j holds words[b], word j of bitplane row b): on return lo[j] (and
// hi[j], with more than 32 bitplanes) holds the low (high) 32 bits of the
// value of coefficient (j, lane), bit NUM_BITPLANES - 1 - b from row b. A
// cross-lane transpose of each row, then an in-register transpose, instead
// of NUM_BITPLANES broadcasts per word.
template <int NUM_BITPLANES, typename DeviceType>
MGARDX_EXEC void tile_values(SubGroup<DeviceType> &sg, int lane,
                             uint32_t *words, uint32_t *lo, uint32_t *hi) {
#pragma unroll
  for (int b = 0; b < NUM_BITPLANES; b++) {
    // Bit j: bit b of coefficient (j, lane).
    words[b] = warp_transpose32(sg, lane, words[b]);
  }
#pragma unroll
  for (int r = 0; r < 32; r++) {
    lo[r] = r < NUM_BITPLANES ? words[NUM_BITPLANES - 1 - r] : 0;
  }
  transpose32(lo);
  if constexpr (NUM_BITPLANES > 32) {
#pragma unroll
    for (int r = 0; r < 32; r++) {
      hi[r] = 32 + r < NUM_BITPLANES ? words[NUM_BITPLANES - 33 - r] : 0;
    }
    transpose32(hi);
  }
}

// Decoding counterpart: lane j loads word j of a tile for each row; lane l
// then assembles coefficient l of every word (tile_values, or a broadcast of
// every word for few bitplanes).
template <typename T_data, typename T_fp, typename T_bitplane,
          int NUM_BITPLANES, typename DeviceType>
class BPDecoderWarpTransposeFunctor : public Functor<DeviceType> {
public:
  static constexpr SIZE WORDS = 32;
  MGARDX_CONT BPDecoderWarpTransposeFunctor() {}
  MGARDX_CONT
  BPDecoderWarpTransposeFunctor(
      SIZE n, int starting_bitplane, SubArray<1, T_data, DeviceType> abs_max,
      SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
      SubArray<1, bool, DeviceType> signs, SubArray<1, T_data, DeviceType> v)
      : n(n), starting_bitplane(starting_bitplane), abs_max(abs_max),
        encoded_bitplanes(encoded_bitplanes), signs(signs), v(v) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    SubGroup<DeviceType> sg;
    const int lane = sg.lane();
    SIZE tile = (FunctorBase<DeviceType>::GetBlockIdX() *
                     FunctorBase<DeviceType>::GetBlockDimX() +
                 FunctorBase<DeviceType>::GetThreadIdX()) /
                WORDS;
    SIZE num_words = n / WORDS;
    SIZE first = tile * WORDS;
    if (first >= num_words) {
      return;
    }
    SIZE words_in_tile = num_words - first < WORDS ? num_words - first : WORDS;
    bool valid = lane < (int)words_in_tile;

    int exp;
    frexp(*abs_max((IDX)0), &exp);
    int ending_bitplane = starting_bitplane + NUM_BITPLANES;
    T_data scale = pow(2, -ending_bitplane + exp);

    T_bitplane words[NUM_BITPLANES];
#pragma unroll
    for (int b = 0; b < NUM_BITPLANES; b++) {
      words[b] =
          valid ? *encoded_bitplanes(starting_bitplane + b + 1, first + lane)
                : 0;
    }
    T_bitplane sign_word =
        starting_bitplane == 0 && valid ? *encoded_bitplanes(0, first + lane)
                                        : 0;
    auto put = [&](int j, T_fp fp, bool row_sign) {
      SIZE idx = (first + j) * WORDS + lane;
      bool sign;
      if (starting_bitplane == 0) {
        sign = row_sign;
        *signs(idx) = sign;
      } else {
        sign = *signs(idx);
      }
      T_data data = (T_data)fp * scale;
      *v(idx) = sign ? -data : data;
    };
    if constexpr (NUM_BITPLANES >= TRANSPOSED_DECODE &&
                  sizeof(T_bitplane) == 4) {
      uint32_t lo[32], hi[NUM_BITPLANES > 32 ? 32 : 1];
      tile_values<NUM_BITPLANES>(sg, lane, words, lo, hi);
      // Bit j: the sign of coefficient (j, lane).
      uint32_t sign_bits = warp_transpose32(sg, lane, sign_word);
#pragma unroll
      for (int j = 0; j < (int)WORDS; j++) {
        if (j < (int)words_in_tile) {
          T_fp fp = lo[j];
          if constexpr (NUM_BITPLANES > 32) {
            fp |= (T_fp)hi[j] << 32;
          }
          put(j, fp, (sign_bits >> j) & 1u);
        }
      }
    } else {
      for (int j = 0; j < (int)words_in_tile; j++) {
        T_fp fp = 0;
#pragma unroll
        for (int b = 0; b < NUM_BITPLANES; b++) {
          T_bitplane w = sg.shfl(words[b], j);
          fp |= (T_fp)((w >> lane) & 1) << (NUM_BITPLANES - 1 - b);
        }
        put(j, fp, (sg.shfl(sign_word, j) >> lane) & 1);
      }
    }
  }

  MGARDX_CONT size_t shared_memory_size() { return 0; }

private:
  SIZE n;
  int starting_bitplane;
  SubArray<1, T_data, DeviceType> abs_max;
  SubArray<2, T_bitplane, DeviceType> encoded_bitplanes;
  SubArray<1, bool, DeviceType> signs;
  SubArray<1, T_data, DeviceType> v;
};

template <typename T_data, typename T_fp, typename T_bitplane,
          int NUM_BITPLANES, typename DeviceType>
class BPDecoderWarpTransposeKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "warp transpose bp decoder";
  using FunctorType = BPDecoderWarpTransposeFunctor<T_data, T_fp, T_bitplane,
                                                    NUM_BITPLANES, DeviceType>;
  MGARDX_CONT
  BPDecoderWarpTransposeKernel(
      SIZE n, int starting_bitplane, SubArray<1, T_data, DeviceType> abs_max,
      SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
      SubArray<1, bool, DeviceType> signs, SubArray<1, T_data, DeviceType> v)
      : n(n), starting_bitplane(starting_bitplane), abs_max(abs_max),
        encoded_bitplanes(encoded_bitplanes), signs(signs), v(v) {}
  MGARDX_CONT Task<FunctorType> GenTask(int queue_idx) {
    FunctorType functor(n, starting_bitplane, abs_max, encoded_bitplanes,
                        signs, v);
    SIZE tiles = (n / 32 + 31) / 32;
    SIZE tbx = 256;
    SIZE gridx = std::max((SIZE)1, (tiles * 32 + tbx - 1) / tbx);
    return Task(functor, 1, 1, gridx, 1, 1, tbx, 0, queue_idx,
                std::string(Name));
  }

private:
  SIZE n;
  int starting_bitplane;
  SubArray<1, T_data, DeviceType> abs_max;
  SubArray<2, T_bitplane, DeviceType> encoded_bitplanes;
  SubArray<1, bool, DeviceType> signs;
  SubArray<1, T_data, DeviceType> v;
};

// Publication of a value with an epoch (one 64-bit word), for the chained
// scan of BPDecoderWarpTransposeSignFunctor (CUDA).
MGARDX_EXEC void publish_epoch(uint64_t *p, uint32_t epoch, uint32_t value) {
#if defined(__CUDA_ARCH__)
  *(volatile unsigned long long *)p =
      ((unsigned long long)epoch << 32) | value;
#endif
}
MGARDX_EXEC uint32_t wait_epoch(const uint64_t *p, uint32_t epoch) {
#if defined(__CUDA_ARCH__)
  unsigned long long x;
  while (((x = *(volatile const unsigned long long *)p) >> 32) != epoch) {
    __nanosleep(32);
  }
  return (uint32_t)x;
#else
  return 0;
#endif
}
// A status word as published, without waiting (CUDA).
MGARDX_EXEC uint64_t peek_epoch(const uint64_t *p) {
#if defined(__CUDA_ARCH__)
  return *(volatile const unsigned long long *)p;
#else
  return 0;
#endif
}
// Prefetch of the cache line holding *p into L1 or L2 (CUDA).
MGARDX_EXEC void prefetch_l1(const void *p) {
#if defined(__CUDA_ARCH__)
  asm volatile("prefetch.L1 [%0];" ::"l"(p));
#endif
}
MGARDX_EXEC void prefetch_l2(const void *p) {
#if defined(__CUDA_ARCH__)
  asm volatile("prefetch.L2 [%0];" ::"l"(p));
#endif
}

// Decoding with significance-coded signs (SignificanceCoding.hpp; same
// results as SignResolveKernel followed by the decoder): warp per tile as
// BPDecoderWarpTransposeFunctor, WARPS tiles per thread block. Each warp counts
// the coefficients of its tile that become nonzero in each group, from the
// words it holds; the position of the tile's signs in a group's section is the
// segment's offset plus the counts of the tiles before it in the segment:
// those of the block (shared memory) and those of the blocks before it in the
// segment, which every block publishes (status(block, group), with this
// launch's epoch) before waiting for its predecessors' (a thread per group and
// predecessor). sections(i): the sign section of the i-th group decoded.
template <typename T_data, typename T_fp, typename T_bitplane,
          int NUM_BITPLANES, typename DeviceType>
class BPDecoderWarpTransposeSignFunctor : public Functor<DeviceType> {
public:
  static constexpr SIZE WORDS = 32;
  static constexpr SIZE WARPS = 8;
  static constexpr SIZE BLOCKS_PER_SEGMENT = significance::SEGMENT / WARPS;
  static constexpr SIZE MAX_GROUPS = NUM_BITPLANES;
  // Threads per group in the block's scan: one per warp and per block of the
  // segment.
  static constexpr SIZE LANES =
      WARPS > BLOCKS_PER_SEGMENT ? WARPS : BLOCKS_PER_SEGMENT;
  static_assert(significance::SEGMENT % WARPS == 0);
  MGARDX_CONT BPDecoderWarpTransposeSignFunctor() {}
  MGARDX_CONT
  BPDecoderWarpTransposeSignFunctor(
      SIZE n, int starting_bitplane, int group_size,
      SubArray<1, T_data, DeviceType> abs_max,
      SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
      SubArray<1, bool, DeviceType> signs,
      SubArray<1, uint64_t, DeviceType> sections,
      SubArray<1, uint64_t, DeviceType> status, uint32_t epoch,
      SubArray<1, T_data, DeviceType> v)
      : n(n), starting_bitplane(starting_bitplane), group_size(group_size),
        abs_max(abs_max), encoded_bitplanes(encoded_bitplanes), signs(signs),
        sections(sections), status(status), epoch(epoch), v(v) {
    Functor<DeviceType>();
    // Bitplanes that end a group (requests start at a group boundary).
    ends = 0;
    for (int b = 0; b < NUM_BITPLANES; b++) {
      if ((b + 1) % group_size == 0 || b == NUM_BITPLANES - 1) {
        ends |= (uint64_t)1 << b;
      }
    }
    num_groups = (NUM_BITPLANES + group_size - 1) / group_size;
  }

  MGARDX_EXEC void Operation1() {
    SubGroup<DeviceType> sg;
    const int lane = sg.lane();
    const SIZE tid = FunctorBase<DeviceType>::GetThreadIdX();
    const SIZE warp = tid / WORDS;
    const SIZE block = FunctorBase<DeviceType>::GetBlockIdX();
    // counts[WARPS][MAX_GROUPS], before[WARPS][MAX_GROUPS] (the warps'
    // exclusive prefixes in the block), then the position of the block's first
    // tile in each group's section.
    uint32_t *counts = (uint32_t *)FunctorBase<DeviceType>::GetSharedMemory();
    uint32_t *before = counts + WARPS * MAX_GROUPS;
    uint32_t *position = before + WARPS * MAX_GROUPS;
    const SIZE num_words = n / WORDS;
    const SIZE nseg = significance::num_segments(num_words);
    const SIZE G = num_groups;
    const SIZE in_segment = block % BLOCKS_PER_SEGMENT;
    int exp;
    frexp(*abs_max((IDX)0), &exp);
    T_data scale = pow(2, -(starting_bitplane + NUM_BITPLANES) + exp);
    {
      SIZE tile = block * WARPS + warp;
      SIZE first = tile * WORDS;
      SIZE words_in_tile =
          first >= num_words ? 0
                             : (num_words - first < WORDS ? num_words - first
                                                          : WORDS);
      bool valid = lane < (int)words_in_tile;
      T_bitplane words[NUM_BITPLANES];
#pragma unroll
      for (int b = 0; b < NUM_BITPLANES; b++) {
        words[b] =
            valid ? *encoded_bitplanes(starting_bitplane + b + 1, first + lane)
                  : 0;
      }
      // The segment's offset in each group's section (independent of the
      // words).
      if (tid < G) {
        position[tid] =
            ((const uint32_t *)*sections(tid))[block / BLOCKS_PER_SEGMENT] * 32;
      }
      // The predecessors' counts of the look-back below (thread tid's first
      // pair), loaded now: mostly published already past the first wave.
      uint64_t early = 0;
      if (tid < G * LANES && tid % LANES < in_segment) {
        early = peek_epoch(
            status((block - in_segment + tid % LANES) * G + tid / LANES));
      }
      // The segment's signs into L2: a tile's signs take at most one line
      // (32 words), and the block prefetches the lines at its tiles' indices
      // in the segment, about where its own signs are (not in the last
      // segment, whose end is not known here).
      const SIZE segment = block / BLOCKS_PER_SEGMENT;
      for (SIZE t = tid; segment + 1 < nseg && t < G * WARPS;
           t += WARPS * WORDS) {
        const uint32_t *offsets = (const uint32_t *)*sections(t / WARPS);
        const uint32_t word =
            offsets[segment] + (uint32_t)(in_segment * WARPS + t % WARPS) * 32;
        if (word < offsets[segment + 1]) {
          prefetch_l2(offsets + nseg + word);
        }
      }
      // State of the word's 32 coefficients: nonzero ones and their signs.
      uint32_t *state = (uint32_t *)signs.data() +
                        (first + lane) * significance::STATE_STRIDE;
      uint32_t sig0 = 0, sgn0 = 0;
      if (valid && starting_bitplane > 0) {
        sig0 = state[0];
        sgn0 = state[1];
      }
      {
        uint32_t sig = sig0, orr = 0;
        int g = 0;
#pragma unroll
        for (int b = 0; b < NUM_BITPLANES; b++) {
          orr |= words[b];
          if ((ends >> b) & 1) {
            uint32_t c =
                warp_reduce_add(zero_elimination::popcount32(orr & ~sig));
            sig |= orr;
            orr = 0;
            if (lane == 0) {
              counts[warp * MAX_GROUPS + g] = c;
            }
            g++;
          }
        }
      }
      sync_block();
      // Thread (g, p): the prefix of warp p in the block, the block's count
      // (published by p = 0, as soon as counted), and the count of the p-th
      // block of the segment if it is before this one.
      for (SIZE t = tid; t < G * LANES; t += WARPS * WORDS) {
        const SIZE g = t / LANES, p = t % LANES;
        uint32_t prefix = 0, aggregate = 0;
        for (SIZE w = 0; w < WARPS; w++) {
          uint32_t c = counts[w * MAX_GROUPS + g];
          prefix += w < p ? c : 0;
          aggregate += c;
        }
        if (p < WARPS) {
          before[p * MAX_GROUPS + g] = prefix;
        }
        if (p == 0 && in_segment + 1 < BLOCKS_PER_SEGMENT) {
          publish_epoch(status(block * G + g), epoch, aggregate);
        }
        if (p < in_segment) {
          const uint64_t x = t == tid ? early : 0;
          uint32_t c =
              x >> 32 == epoch
                  ? (uint32_t)x
                  : wait_epoch(status((block - in_segment + p) * G + g), epoch);
          if (c) {
            Atomic<uint32_t, AtomicSharedMemory, AtomicDeviceScope,
                   DeviceType>::Add(position + g, c);
          }
        }
      }
      sync_block();
      // The warp's signs of a group are at most 1024 bits (two cache lines):
      // prefetched for the reads below.
      for (SIZE g = lane; g < G; g += WORDS) {
        const uint32_t c = counts[warp * MAX_GROUPS + g];
        if (c) {
          const uint32_t start = position[g] + before[warp * MAX_GROUPS + g];
          const uint32_t *section = (const uint32_t *)*sections(g) + nseg;
          prefetch_l1(section + start / 32);
          prefetch_l1(section + (start + c - 1) / 32);
        }
      }
      uint32_t sig = sig0, sgn = sgn0, orr = 0;
      int g = 0;
#pragma unroll
      for (int b = 0; b < NUM_BITPLANES; b++) {
        orr |= words[b];
        if ((ends >> b) & 1) {
          uint32_t fresh = orr & ~sig;
          sig |= orr;
          orr = 0;
          if (sg.ballot(fresh != 0) == 0) {
            g++;
            continue;
          }
          uint32_t k = zero_elimination::popcount32(fresh);
          uint32_t inclusive = zero_elimination::warp_inclusive_scan(sg, lane, k);
          SIZE pos =
              position[g] + before[warp * MAX_GROUPS + g] + inclusive - k;
          if (k) {
            const uint32_t *section = (const uint32_t *)*sections(g) + nseg;
            uint32_t bits = significance::read_bits(section, pos, (int)k);
            // Bit i of bits to the i-th coefficient of fresh, four at a time.
            for (uint32_t m = fresh; m; bits >>= 4) {
#pragma unroll
              for (int i = 0; i < 4; i++) {
                const uint32_t low = m & (0u - m);
                m ^= low;
                sgn |= low & (0u - ((bits >> i) & 1u));
              }
            }
          }
          g++;
        }
      }
      if constexpr (NUM_BITPLANES >= TRANSPOSED_DECODE &&
                    sizeof(T_bitplane) == 4) {
        uint32_t lo[32], hi[NUM_BITPLANES > 32 ? 32 : 1];
        tile_values<NUM_BITPLANES>(sg, lane, words, lo, hi);
        // Bit j: the sign of coefficient (j, lane).
        uint32_t sign_bits = warp_transpose32(sg, lane, sgn);
#pragma unroll
        for (int j = 0; j < (int)WORDS; j++) {
          if (j < (int)words_in_tile) {
            T_fp fp = lo[j];
            if constexpr (NUM_BITPLANES > 32) {
              fp |= (T_fp)hi[j] << 32;
            }
            T_data data = (T_data)fp * scale;
            *v((first + j) * WORDS + lane) =
                (sign_bits >> j) & 1u ? -data : data;
          }
        }
      } else {
        for (int j = 0; j < (int)words_in_tile; j++) {
          T_fp fp = 0;
#pragma unroll
          for (int b = 0; b < NUM_BITPLANES; b++) {
            T_bitplane w = sg.shfl(words[b], j);
            fp |= (T_fp)((w >> lane) & 1) << (NUM_BITPLANES - 1 - b);
          }
          bool sign = (sg.shfl(sgn, j) >> lane) & 1u;
          T_data data = (T_data)fp * scale;
          *v((first + j) * WORDS + lane) = sign ? -data : data;
        }
      }
      if (valid && (starting_bitplane == 0 || sig != sig0)) {
        state[0] = sig;
        state[1] = sgn;
      }
    }
  }

  MGARDX_CONT size_t shared_memory_size() {
    return (2 * WARPS + 1) * MAX_GROUPS * sizeof(uint32_t);
  }

private:
  SIZE n;
  int starting_bitplane, group_size;
  SubArray<1, T_data, DeviceType> abs_max;
  SubArray<2, T_bitplane, DeviceType> encoded_bitplanes;
  SubArray<1, bool, DeviceType> signs;
  SubArray<1, uint64_t, DeviceType> sections, status;
  uint32_t epoch;
  SubArray<1, T_data, DeviceType> v;
  uint64_t ends;
  SIZE num_groups;
};

template <typename T_data, typename T_fp, typename T_bitplane,
          int NUM_BITPLANES, typename DeviceType>
class BPDecoderWarpTransposeSignKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "warp transpose bp decoder (signs)";
  using FunctorType =
      BPDecoderWarpTransposeSignFunctor<T_data, T_fp, T_bitplane, NUM_BITPLANES,
                                        DeviceType>;
  // Thread blocks, and status words per group (one per block).
  static SIZE num_blocks(SIZE n) {
    return (significance::num_tiles(n / 32) + FunctorType::WARPS - 1) /
           FunctorType::WARPS;
  }
  MGARDX_CONT
  BPDecoderWarpTransposeSignKernel(
      SIZE n, int starting_bitplane, int group_size,
      SubArray<1, T_data, DeviceType> abs_max,
      SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
      SubArray<1, bool, DeviceType> signs,
      SubArray<1, uint64_t, DeviceType> sections,
      SubArray<1, uint64_t, DeviceType> status, uint32_t epoch,
      SubArray<1, T_data, DeviceType> v)
      : n(n), starting_bitplane(starting_bitplane), group_size(group_size),
        abs_max(abs_max), encoded_bitplanes(encoded_bitplanes), signs(signs),
        sections(sections), status(status), epoch(epoch), v(v) {}
  MGARDX_CONT Task<FunctorType> GenTask(int queue_idx) {
    FunctorType functor(n, starting_bitplane, group_size, abs_max,
                        encoded_bitplanes, signs, sections, status, epoch, v);
    return Task(functor, 1, 1, num_blocks(n), 1, 1, FunctorType::WARPS * 32,
                functor.shared_memory_size(), queue_idx, std::string(Name));
  }

private:
  SIZE n;
  int starting_bitplane, group_size;
  SubArray<1, T_data, DeviceType> abs_max;
  SubArray<2, T_bitplane, DeviceType> encoded_bitplanes;
  SubArray<1, bool, DeviceType> signs;
  SubArray<1, uint64_t, DeviceType> sections, status;
  uint32_t epoch;
  SubArray<1, T_data, DeviceType> v;
};

// Portable kernels of BPEncoderWarpTranspose (backends other than CUDA, and
// 64-bit words): one thread per word, which loads the 32 consecutive
// coefficients of its word (the thread mapping of HP-MDR's Locality Block
// encoder). Same rows and signs as BPEncoderWarpTransposeFunctor.
template <typename T_data, typename T_fp, typename T_bitplane, typename T_error,
          int NUM_BITPLANES, bool ControlL2, typename DeviceType>
class BPEncoderWarpTransposePortableFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT
  BPEncoderWarpTransposePortableFunctor() {}
  MGARDX_CONT
  BPEncoderWarpTransposePortableFunctor(
      SIZE n, SubArray<1, T_data, DeviceType> abs_max,
      SubArray<1, T_data, DeviceType> v,
      SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
      SubArray<2, T_error, DeviceType> level_errors_workspace)
      : n(n), abs_max(abs_max), encoded_bitplanes(encoded_bitplanes), v(v),
        level_errors_workspace(level_errors_workspace) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void encode_batch(T_fp *v, T_bitplane *encoded) {

#pragma unroll
    for (int bp_idx = 0; bp_idx < NUM_BITPLANES; bp_idx++) {
      T_bitplane buffer = 0;
      for (int data_idx = 0; data_idx < BATCH_SIZE; data_idx++) {
        T_bitplane bit =
            (v[data_idx] >> (NUM_BITPLANES - 1 - bp_idx)) & (T_bitplane)1;
        buffer |= bit << data_idx;
      }
      encoded[bp_idx] = buffer;
    }
  }

  MGARDX_EXEC void error_collect(T_data *shifted_data, T_error *errors,
                                 int exp) {
    for (int bp_idx = 0; bp_idx < NUM_BITPLANES; bp_idx++) {
      for (int data_idx = 0; data_idx < BATCH_SIZE; data_idx++) {
        T_data data = shifted_data[data_idx];
        T_fp fp_data = (T_fp)fabs(data);
        T_error mantissa = fabs(data) - fp_data;
        T_fp mask = ((T_fp)1 << bp_idx) - 1;
        T_error diff = (T_error)(fp_data & mask) + mantissa;
        errors[NUM_BITPLANES - bp_idx] += diff * diff;
      }
    }
    for (int data_idx = 0; data_idx < BATCH_SIZE; data_idx++) {
      T_data data = shifted_data[data_idx];
      errors[0] += data * data;
    }

    for (int bp_idx = 0; bp_idx < NUM_BITPLANES + 1; bp_idx++) {
      errors[bp_idx] = ldexp(errors[bp_idx], 2 * (-NUM_BITPLANES + exp));
    }
  }

  MGARDX_EXEC void Operation1() {
    SIZE batch_idx = FunctorBase<DeviceType>::GetBlockIdX() *
                         FunctorBase<DeviceType>::GetBlockDimX() +
                     FunctorBase<DeviceType>::GetThreadIdX();

    SIZE num_full_batches = n / BATCH_SIZE;

    T_data shifted_data[BATCH_SIZE];
    T_fp fp_data[BATCH_SIZE];
    T_bitplane encoded_data[NUM_BITPLANES];
    T_bitplane encoded_sign = 0;
    T_error errors[NUM_BITPLANES + 1];

    int exp;
    frexp(*abs_max((IDX)0), &exp);

    if (batch_idx >= num_full_batches) {
      return;
    }

    if (exp > 0) {
#pragma unroll
      for (int data_idx = 0; data_idx < BATCH_SIZE; data_idx++) {
        T_data data = *v(batch_idx * BATCH_SIZE + data_idx);
        // this can cause overflow
        shifted_data[data_idx] = data * ((T_fp)1 << NUM_BITPLANES - exp);
        // ldexp without constant argument is slow
        fp_data[data_idx] = (T_fp)fabs(shifted_data[data_idx]);
      }
    } else {
#pragma unroll
      for (int data_idx = 0; data_idx < BATCH_SIZE; data_idx++) {
        T_data data = *v(batch_idx * BATCH_SIZE + data_idx);
        shifted_data[data_idx] = data * pow(2, NUM_BITPLANES - exp);
        fp_data[data_idx] = (T_fp)fabs(shifted_data[data_idx]);
      }
    }

    // encode sign
    // Shift amount runs up to BATCH_SIZE - 1 (bits of T_bitplane), so the
    // value being shifted must be T_bitplane, not T_fp: when T_bitplane is
    // wider than T_fp (e.g. uint64_t bitplanes with float data, T_fp =
    // uint32_t), shifting a T_fp by >= 32 is undefined behavior.
    for (int data_idx = 0; data_idx < BATCH_SIZE; data_idx++) {
      encoded_sign += (T_bitplane)(signbit(shifted_data[data_idx]) == 0 ? 0 : 1)
                      << data_idx;
    }
    // encode data
    encode_batch(fp_data, encoded_data);
// store data: row 0 holds the signs, bitplane b is row b + 1
#pragma unroll
    for (int bp_idx = 0; bp_idx < NUM_BITPLANES; bp_idx++) {
      *encoded_bitplanes(bp_idx + 1, batch_idx) = encoded_data[bp_idx];
    }
    // store sign
    *encoded_bitplanes(0, batch_idx) = encoded_sign;
    if constexpr (ControlL2) {
      for (int bp_idx = 0; bp_idx < NUM_BITPLANES + 1; bp_idx++) {
        errors[bp_idx] = 0;
      }
      error_collect(shifted_data, errors, exp);
      for (int bp_idx = 0; bp_idx < NUM_BITPLANES + 1; bp_idx++) {
        *level_errors_workspace(bp_idx, batch_idx) = errors[bp_idx];
      }
    }
  }

  MGARDX_CONT size_t shared_memory_size() { return 0; }

private:
  SIZE n;
  SubArray<1, T_data, DeviceType> abs_max;
  SubArray<1, T_data, DeviceType> v;
  SubArray<2, T_bitplane, DeviceType> encoded_bitplanes;
  SubArray<2, T_error, DeviceType> level_errors_workspace;
  static constexpr int BATCH_SIZE = sizeof(T_bitplane) * 8;
};

template <typename T_data, typename T_fp, typename T_bitplane, typename T_error,
          int NUM_BITPLANES, bool ControlL2, typename DeviceType>
class BPEncoderWarpTransposePortableKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name =
      "warp transpose bp encoder (portable)";
  static constexpr int BATCH_SIZE = sizeof(T_bitplane) * 8;
  MGARDX_CONT
  BPEncoderWarpTransposePortableKernel(
      SIZE n, SubArray<1, T_data, DeviceType> abs_max,
      SubArray<1, T_data, DeviceType> v,
      SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
      SubArray<2, T_error, DeviceType> level_errors_workspace)
      : n(n), abs_max(abs_max), encoded_bitplanes(encoded_bitplanes), v(v),
        level_errors_workspace(level_errors_workspace) {}

  using FunctorType = BPEncoderWarpTransposePortableFunctor<
      T_data, T_fp, T_bitplane, T_error, NUM_BITPLANES, ControlL2, DeviceType>;
  using TaskType = Task<FunctorType>;

  MGARDX_CONT TaskType GenTask(int queue_idx) {
    FunctorType functor(n, abs_max, v, encoded_bitplanes,
                        level_errors_workspace);
    SIZE total_thread = std::max((SIZE)1, n / BATCH_SIZE);
    return Task(functor, 1, 1, (total_thread - 1) / 256 + 1, 1, 1, 256,
                functor.shared_memory_size(), queue_idx, std::string(Name));
  }

private:
  SIZE n;
  SubArray<1, T_data, DeviceType> abs_max;
  SubArray<1, T_data, DeviceType> v;
  SubArray<2, T_bitplane, DeviceType> encoded_bitplanes;
  SubArray<2, T_error, DeviceType> level_errors_workspace;
};

// Decoding counterpart of BPEncoderWarpTransposePortableFunctor. SignMasks:
// every sign is in the sign masks of the significance-coded state
// (SignificanceCoding.hpp); there is no sign row.
template <typename T_data, typename T_fp, typename T_bitplane,
          int NUM_BITPLANES, typename DeviceType, bool SignMasks = false>
class BPDecoderWarpTransposePortableFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT
  BPDecoderWarpTransposePortableFunctor() {}
  MGARDX_CONT
  BPDecoderWarpTransposePortableFunctor(
      SIZE n, int starting_bitplane, SubArray<1, T_data, DeviceType> abs_max,
      SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
      SubArray<1, bool, DeviceType> signs, SubArray<1, T_data, DeviceType> v)
      : n(n), starting_bitplane(starting_bitplane), abs_max(abs_max),
        encoded_bitplanes(encoded_bitplanes), signs(signs), v(v) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void decode_batch(T_fp *v, T_bitplane *encoded) {
#pragma unroll
    for (int data_idx = 0; data_idx < BATCH_SIZE; data_idx++) {
      T_fp buffer = 0;
      for (int bp_idx = 0; bp_idx < NUM_BITPLANES; bp_idx++) {
        T_fp bit = (encoded[bp_idx] >> data_idx) & (T_fp)1;
        buffer += bit << (NUM_BITPLANES - 1 - bp_idx);
      }
      v[data_idx] = buffer;
    }
  }

  MGARDX_EXEC void Operation1() {
    SIZE batch_idx = FunctorBase<DeviceType>::GetBlockIdX() *
                         FunctorBase<DeviceType>::GetBlockDimX() +
                     FunctorBase<DeviceType>::GetThreadIdX();

    SIZE num_full_batches = n / BATCH_SIZE;

    T_data shifted_data[BATCH_SIZE];
    T_fp fp_data[BATCH_SIZE];
    T_fp fp_sign[BATCH_SIZE];
    T_bitplane encoded_data[NUM_BITPLANES];
    T_bitplane encoded_sign;

    int exp;
    frexp(*abs_max((IDX)0), &exp);

    if (batch_idx >= num_full_batches) {
      return;
    }

    int ending_bitplane = starting_bitplane + NUM_BITPLANES;

#pragma unroll
    for (int bp_idx = 0; bp_idx < NUM_BITPLANES; bp_idx++) {
      encoded_data[bp_idx] =
          *encoded_bitplanes(starting_bitplane + bp_idx + 1, batch_idx);
    }
    // decode data
    decode_batch(fp_data, encoded_data);

    if constexpr (SignMasks) {
      const uint32_t *state = (const uint32_t *)signs.data();
#pragma unroll
      for (int data_idx = 0; data_idx < BATCH_SIZE; data_idx++) {
        SIZE i = batch_idx * BATCH_SIZE + data_idx;
        fp_sign[data_idx] =
            (state[significance::STATE_STRIDE * (i / 32) + 1] >> (i % 32)) & 1u;
      }
    } else if (starting_bitplane == 0) {
      // decode sign
      encoded_sign = *encoded_bitplanes(0, batch_idx);
#pragma unroll
      for (int data_idx = 0; data_idx < BATCH_SIZE; data_idx++) {
        fp_sign[data_idx] = (encoded_sign >> data_idx) & (T_fp)1;
        *signs(batch_idx * BATCH_SIZE + data_idx) = fp_sign[data_idx];
      }
    } else {
#pragma unroll
      for (int data_idx = 0; data_idx < BATCH_SIZE; data_idx++) {
        fp_sign[data_idx] = *signs(batch_idx * BATCH_SIZE + data_idx);
      }
    }
#pragma unroll
    for (int data_idx = 0; data_idx < BATCH_SIZE; data_idx++) {
      shifted_data[data_idx] = (T_data)fp_data[data_idx];
      // It is beneficial to use pow instead of ldexp
      T_data data = shifted_data[data_idx] * pow(2, -ending_bitplane + exp);
      data = fp_sign[data_idx] ? -data : data;
      *v(batch_idx * BATCH_SIZE + data_idx) = data;
    }
  }

  MGARDX_CONT size_t shared_memory_size() { return 0; }

private:
  SIZE n;
  int starting_bitplane;
  SubArray<1, T_data, DeviceType> abs_max;
  SubArray<2, T_bitplane, DeviceType> encoded_bitplanes;
  SubArray<1, bool, DeviceType> signs;
  SubArray<1, T_data, DeviceType> v;
  static constexpr int BATCH_SIZE = sizeof(T_bitplane) * 8;
};

template <typename T_data, typename T_fp, typename T_bitplane,
          int NUM_BITPLANES, typename DeviceType, bool SignMasks = false>
class BPDecoderWarpTransposePortableKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name =
      "warp transpose bp decoder (portable)";
  static constexpr SIZE BATCH_SIZE = sizeof(T_bitplane) * 8;
  MGARDX_CONT
  BPDecoderWarpTransposePortableKernel(
      SIZE n, int starting_bitplane, SubArray<1, T_data, DeviceType> abs_max,
      SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
      SubArray<1, bool, DeviceType> signs, SubArray<1, T_data, DeviceType> v)
      : n(n), starting_bitplane(starting_bitplane), abs_max(abs_max),
        encoded_bitplanes(encoded_bitplanes), signs(signs), v(v) {}

  using FunctorType = BPDecoderWarpTransposePortableFunctor<
      T_data, T_fp, T_bitplane, NUM_BITPLANES, DeviceType, SignMasks>;
  using TaskType = Task<FunctorType>;

  MGARDX_CONT TaskType GenTask(int queue_idx) {
    FunctorType functor(n, starting_bitplane, abs_max, encoded_bitplanes, signs,
                        v);
    SIZE total_thread = std::max((SIZE)1, n / BATCH_SIZE);
    return Task(functor, 1, 1, (total_thread - 1) / 256 + 1, 1, 1, 256,
                functor.shared_memory_size(), queue_idx, std::string(Name));
  }

private:
  SIZE n;
  int starting_bitplane;
  SubArray<1, T_data, DeviceType> abs_max;
  SubArray<2, T_bitplane, DeviceType> encoded_bitplanes;
  SubArray<1, bool, DeviceType> signs;
  SubArray<1, T_data, DeviceType> v;
};

// The MDR-X bitplane encoder (see the top of this file). Kernels by layout
// (SetLayout) and backend:
//   Locality Block, CUDA with 32-bit words: BPEncoderWarpTransposeKernel,
//     BPDecoderWarpTransposeKernel / BPDecoderWarpTransposeSignKernel;
//   Locality Block, otherwise: BPEncoderWarpTransposePortableKernel,
//     BPDecoderWarpTransposePortableKernel (after SignResolveKernel with
//     significance-coded signs);
//   Register Block (always with NegaBinary): BPEncoderRegisterBlockKernel,
//     BPDecoderRegisterBlockKernel.
template <DIM D, typename T_data, typename T_bitplane, typename T_error,
          bool NegaBinary, bool ControlL2, typename DeviceType>
class BPEncoderWarpTranspose
    : public concepts::BitplaneEncoderInterface<D, T_data, T_bitplane, T_error,
                                                ControlL2, DeviceType> {
public:
  static constexpr SIZE BATCH_SIZE = sizeof(T_bitplane) * 8;
  static constexpr int MAX_BITPLANES = sizeof(T_data) * 8;
  using T_sfp = typename std::conditional<std::is_same<T_data, double>::value,
                                          int64_t, int32_t>::type;
  using T_fp = typename std::conditional<std::is_same<T_data, double>::value,
                                         uint64_t, uint32_t>::type;

  BPEncoderWarpTranspose() : initialized(false) {
    static_assert(
        std::is_floating_point<T_data>::value,
        "BPEncoderWarpTranspose: input data must be floating points.");
    static_assert(!std::is_same<T_data, long double>::value,
                  "BPEncoderWarpTranspose: long double is not supported.");
    static_assert(std::is_unsigned<T_bitplane>::value,
                  "BPEncoderWarpTranspose: streams must be unsigned integers.");
    static_assert(std::is_integral<T_bitplane>::value,
                  "BPEncoderWarpTranspose: streams must be unsigned integers.");
  }
  BPEncoderWarpTranspose(Hierarchy<D, T_data, DeviceType> &hierarchy)
      : BPEncoderWarpTranspose() {
    Adapt(hierarchy, 0);
    DeviceRuntime<DeviceType>::SyncQueue(0);
  }

  // Layout of the encoded bitplanes: NUM_ROWS rows of bitplane_length(n)
  // words. With the binary (sign-magnitude) encoding row 0 holds the signs
  // (or, with significance-coded signs, the packed signs) and bitplane b is
  // row b + 1; the negabinary encoding has no sign row.
  static constexpr int SIGN_ROWS = NegaBinary ? 0 : 1;
  static constexpr int NUM_ROWS = MAX_BITPLANES + SIGN_ROWS;

  static SIZE bitplane_length(SIZE n) { return num_blocks(n); }

  static SIZE num_blocks(SIZE n) {
    const SIZE batch_size = sizeof(T_bitplane) * 8;
    SIZE num_blocks = (n - 1) / batch_size + 1;
    return num_blocks;
  }

  void Adapt(Hierarchy<D, T_data, DeviceType> &hierarchy, int queue_idx) {
    Adapt(hierarchy, hierarchy.level_num_elems(hierarchy.l_target()),
          queue_idx);
  }

  // max_level_num_elems: element count of the largest level to be encoded
  // (with the hybrid decomposition the levels do not follow the hierarchy).
  void Adapt(Hierarchy<D, T_data, DeviceType> &hierarchy,
             SIZE max_level_num_elems, int queue_idx) {
    this->initialized = true;
    this->hierarchy = &hierarchy;
    max_level_num_elems = round_up(max_level_num_elems, BATCH_SIZE);

    level_errors_work_array.resize(
        {MAX_BITPLANES + 1, num_blocks(max_level_num_elems)}, queue_idx);
    DeviceCollective<DeviceType>::Sum(
        num_blocks(max_level_num_elems), SubArray<1, T_error, DeviceType>(),
        SubArray<1, T_error, DeviceType>(), level_error_sum_work_array, false,
        queue_idx);
    if constexpr (std::is_same<DeviceType, CUDA>::value &&
                  sizeof(T_bitplane) == 4) {
      // Status words of the sign decoder's chained scan for the largest
      // level and any group size (allocating them at first use would put the
      // allocation in the decoding of each level).
      using SignKernel =
          BPDecoderWarpTransposeSignKernel<T_data, T_fp, T_bitplane, 1,
                                           DeviceType>;
      SIZE status_size = SignKernel::num_blocks(max_level_num_elems) *
                         significance::num_groups(MAX_BITPLANES, 1);
      if (status_size > sign_status_size) {
        sign_status.resize({status_size}, queue_idx);
        sign_status_size = status_size;
        sign_status.memset(0, queue_idx);
        sign_epoch = 0;
      }
    }
  }

  static size_t EstimateMemoryFootprint(std::vector<SIZE> shape) {
    Hierarchy<D, T_data, DeviceType> hierarchy(shape, Config());
    SIZE max_level_num_elems = hierarchy.level_num_elems(hierarchy.l_target());
    size_t size = 0;
    size += hierarchy.EstimateMemoryFootprint(shape);
    size +=
        (MAX_BITPLANES + 1) * num_blocks(max_level_num_elems) * sizeof(T_error);
    for (int level_idx = 0; level_idx < hierarchy.l_target() + 1; level_idx++) {
      size += hierarchy.level_num_elems(level_idx) * sizeof(bool);
    }
    return size;
  }

  // Layout of the bitplane words (Config::mdr_bitplane_layout, recorded in
  // the metadata). The negabinary encoding always uses the Register Block
  // layout.
  void SetLayout(bitplane_layout_type layout) { this->layout = layout; }
  bitplane_layout_type Layout() const {
    return NegaBinary ? bitplane_layout_type::RegisterBlock : layout;
  }

  // Significance-coded signs with groups of group_size bitplanes (0: a sign
  // row). Binary encoding in the Locality Block layout only.
  void SetSignCoding(int group_size) {
    sign_group_size =
        Layout() == bitplane_layout_type::LocalityBlock ? group_size : 0;
  }
  bool SignCoding() const { return sign_group_size > 0; }

  static bool PortableSigns() { return portable_kernels(); }

  // TODO: remove num_bitplanes in the future
  void encode(SIZE n, int num_bitplanes,
              SubArray<1, T_data, DeviceType> abs_max,
              SubArray<1, T_data, DeviceType> v,
              SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
              SubArray<1, T_error, DeviceType> level_errors, int queue_idx) {
    encode(n, num_bitplanes, abs_max, v, encoded_bitplanes, level_errors,
           queue_idx, SubArray<1, uint32_t, DeviceType>());
  }

  // ze_bitmaps (may be empty): zero-elimination chunk bitmaps of every row,
  // written along with the rows when the encoder supports it, and ze_bits
  // (may be empty) their sparse-word payload bits. Returns whether they were
  // written. With significance-coded signs (SetSignCoding), row 0 gets the
  // packed signs and sign_counts, sign_segment_bits their counts
  // (SignificanceCoding.hpp).
  bool encode(SIZE n, int num_bitplanes,
              SubArray<1, T_data, DeviceType> abs_max,
              SubArray<1, T_data, DeviceType> v,
              SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
              SubArray<1, T_error, DeviceType> level_errors, int queue_idx,
              SubArray<1, uint32_t, DeviceType> ze_bitmaps,
              SubArray<1, uint32_t, DeviceType> ze_bits = {},
              SubArray<1, uint32_t, DeviceType> sign_counts = {},
              SubArray<1, uint32_t, DeviceType> sign_segment_bits = {}) {
    if (SignCoding()) {
      MemoryManager<DeviceType>::Memset1D(
          sign_segment_bits.data(),
          significance::num_groups(MAX_BITPLANES, sign_group_size) *
              significance::num_segments(encoded_bitplanes.shape(1)),
          0, queue_idx);
    }
    bool signs_packed = false;
    bool bitmaps_written = encode_rows(
        n, abs_max, v, encoded_bitplanes, level_errors, queue_idx, ze_bitmaps,
        ze_bits, sign_counts, sign_segment_bits, signs_packed);
    if (SignCoding() && !signs_packed) {
      DeviceLauncher<DeviceType>::Execute(
          significance::SignPackKernel<T_bitplane, DeviceType>(
              MAX_BITPLANES, sign_group_size, encoded_bitplanes, sign_counts,
              sign_segment_bits),
          queue_idx);
    }
    return bitmaps_written;
  }

private:
  bool encode_rows(SIZE n, SubArray<1, T_data, DeviceType> abs_max,
                   SubArray<1, T_data, DeviceType> v,
                   SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
                   SubArray<1, T_error, DeviceType> level_errors, int queue_idx,
                   SubArray<1, uint32_t, DeviceType> ze_bitmaps,
                   SubArray<1, uint32_t, DeviceType> ze_bits,
                   SubArray<1, uint32_t, DeviceType> sign_counts,
                   SubArray<1, uint32_t, DeviceType> sign_segment_bits,
                   bool &signs_packed) {

    if (n % BATCH_SIZE != 0) {
      throw std::runtime_error(
          "BPEncoderWarpTranspose: n is not a multiple of BATCH_SIZE");
    }
    SubArray<2, T_error, DeviceType> level_errors_work(level_errors_work_array);

    if (Layout() == bitplane_layout_type::RegisterBlock) {
      DeviceLauncher<DeviceType>::Execute(
          BPEncoderRegisterBlockKernel<T_data, T_fp, T_sfp, T_bitplane, T_error,
                                       MAX_BITPLANES, NegaBinary, ControlL2,
                                       DeviceType>(
              n, abs_max, v, encoded_bitplanes, level_errors_work),
          queue_idx);
    } else if constexpr (std::is_same<DeviceType, CUDA>::value &&
                         sizeof(T_bitplane) == 4) {
      using WarpKernel =
          BPEncoderWarpTransposeKernel<T_data, T_fp, T_bitplane, T_error,
                                       MAX_BITPLANES, ControlL2, DeviceType>;
      signs_packed = SignCoding() && !PortableSigns();
      DeviceLauncher<DeviceType>::Execute(
          WarpKernel(n, abs_max, v, encoded_bitplanes, level_errors_work,
                     ze_bitmaps, ze_bits, signs_packed ? sign_group_size : 0,
                     sign_counts, sign_segment_bits),
          queue_idx);
      if constexpr (ControlL2) {
        DeviceLauncher<DeviceType>::Execute(
            BPErrorSumKernel<T_error, DeviceType>(
                MAX_BITPLANES + 1, WarpKernel::num_blocks(n), level_errors_work,
                level_errors),
            queue_idx);
      }
      return ze_bitmaps.data() != nullptr;
    } else {
      DeviceLauncher<DeviceType>::Execute(
          BPEncoderWarpTransposePortableKernel<T_data, T_fp, T_bitplane,
                                               T_error, MAX_BITPLANES,
                                               ControlL2, DeviceType>(
              n, abs_max, v, encoded_bitplanes, level_errors_work),
          queue_idx);
    }

    if constexpr (ControlL2) {
      SIZE reduce_size = num_blocks(n);
      for (int i = 0; i < MAX_BITPLANES + 1; i++) {
        SubArray<1, T_error, DeviceType> curr_errors({reduce_size},
                                                     level_errors_work(i, 0));
        SubArray<1, T_error, DeviceType> sum_error({1}, level_errors(i));
        DeviceCollective<DeviceType>::Sum(reduce_size, curr_errors, sum_error,
                                          level_error_sum_work_array, true,
                                          queue_idx);
      }
    }
    return false;
  }

public:
  void decode(SIZE n, int num_bitplanes,
              SubArray<1, T_data, DeviceType> abs_max,
              SubArray<2, T_bitplane, DeviceType> encoded_bitplanes, int level,
              SubArray<1, T_data, DeviceType> v, int queue_idx) {}

  void progressive_decode(SIZE n, int starting_bitplane, int num_bitplanes,
                          SubArray<1, T_data, DeviceType> abs_max,
                          SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
                          SubArray<1, bool, DeviceType> level_signs, int level,
                          SubArray<1, T_data, DeviceType> v, int queue_idx) {
    progressive_decode(n, starting_bitplane, num_bitplanes, abs_max,
                       encoded_bitplanes, level_signs, level, v, queue_idx,
                       SubArray<1, uint64_t, DeviceType>());
  }

  // decode the data and record necessary information for progressiveness.
  // sign_sections: with significance-coded signs, the device addresses of
  // the sign sections of the groups decoded.
  void progressive_decode(SIZE n, int starting_bitplane, int num_bitplanes,
                          SubArray<1, T_data, DeviceType> abs_max,
                          SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
                          SubArray<1, bool, DeviceType> level_signs, int level,
                          SubArray<1, T_data, DeviceType> v, int queue_idx,
                          SubArray<1, uint64_t, DeviceType> sign_sections) {

#define WT_DECODE(NUM_BITPLANES)                                               \
  if (num_bitplanes == NUM_BITPLANES) {                                        \
    decode_with<NUM_BITPLANES>(n, starting_bitplane, abs_max,                  \
                               encoded_bitplanes, level_signs, v, queue_idx,   \
                               sign_sections);                                 \
  }
    WT_DECODE(1);
    WT_DECODE(2);
    WT_DECODE(3);
    WT_DECODE(4);
    WT_DECODE(5);
    WT_DECODE(6);
    WT_DECODE(7);
    WT_DECODE(8);
    WT_DECODE(9);
    WT_DECODE(10);
    WT_DECODE(11);
    WT_DECODE(12);
    WT_DECODE(13);
    WT_DECODE(14);
    WT_DECODE(15);
    WT_DECODE(16);
    WT_DECODE(17);
    WT_DECODE(18);
    WT_DECODE(19);
    WT_DECODE(20);
    WT_DECODE(21);
    WT_DECODE(22);
    WT_DECODE(23);
    WT_DECODE(24);
    WT_DECODE(25);
    WT_DECODE(26);
    WT_DECODE(27);
    WT_DECODE(28);
    WT_DECODE(29);
    WT_DECODE(30);
    WT_DECODE(31);
    WT_DECODE(32);
    WT_DECODE(33);
    WT_DECODE(34);
    WT_DECODE(35);
    WT_DECODE(36);
    WT_DECODE(37);
    WT_DECODE(38);
    WT_DECODE(39);
    WT_DECODE(40);
    WT_DECODE(41);
    WT_DECODE(42);
    WT_DECODE(43);
    WT_DECODE(44);
    WT_DECODE(45);
    WT_DECODE(46);
    WT_DECODE(47);
    WT_DECODE(48);
    WT_DECODE(49);
    WT_DECODE(50);
    WT_DECODE(51);
    WT_DECODE(52);
    WT_DECODE(53);
    WT_DECODE(54);
    WT_DECODE(55);
    WT_DECODE(56);
    WT_DECODE(57);
    WT_DECODE(58);
    WT_DECODE(59);
    WT_DECODE(60);
    WT_DECODE(61);
    WT_DECODE(62);
    WT_DECODE(63);
    WT_DECODE(64);
#undef WT_DECODE
  }

  void print() const {
    std::cout << "Warp transpose bitplane encoder" << std::endl;
  }

private:
  template <int NUM_BITPLANES>
  void decode_with(SIZE n, int starting_bitplane,
                   SubArray<1, T_data, DeviceType> abs_max,
                   SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
                   SubArray<1, bool, DeviceType> level_signs,
                   SubArray<1, T_data, DeviceType> v, int queue_idx,
                   SubArray<1, uint64_t, DeviceType> sign_sections) {
    if constexpr (std::is_same<DeviceType, CUDA>::value &&
                  sizeof(T_bitplane) == 4) {
      if (SignCoding() && !PortableSigns()) {
        using SignKernel =
            BPDecoderWarpTransposeSignKernel<T_data, T_fp, T_bitplane,
                                             NUM_BITPLANES, DeviceType>;
        // Status words of the chained scan, tagged with an epoch per launch
        // (zeroed when allocated or when the epoch wraps around).
        SIZE status_size =
            SignKernel::num_blocks(n) *
            significance::num_groups(NUM_BITPLANES, sign_group_size);
        if (++sign_epoch == 0 || status_size > sign_status_size) {
          if (status_size > sign_status_size) {
            sign_status.resize({status_size}, queue_idx);
            sign_status_size = status_size;
          }
          sign_status.memset(0, queue_idx);
          sign_epoch = 1;
        }
        DeviceLauncher<DeviceType>::Execute(
            SignKernel(n, starting_bitplane, sign_group_size, abs_max,
                       encoded_bitplanes, level_signs, sign_sections,
                       SubArray(sign_status), sign_epoch, v),
            queue_idx);
        return;
      }
    }
    if (SignCoding()) {
      // The signs of the coefficients that become nonzero, into the state;
      // then decoded with every sign from the state.
      DeviceLauncher<DeviceType>::Execute(
          significance::SignResolveKernel<T_bitplane, DeviceType>(
              n, starting_bitplane, NUM_BITPLANES, sign_group_size,
              encoded_bitplanes, level_signs, sign_sections),
          queue_idx);
      DeviceLauncher<DeviceType>::Execute(
          BPDecoderWarpTransposePortableKernel<T_data, T_fp, T_bitplane,
                                               NUM_BITPLANES, DeviceType, true>(
              n, starting_bitplane, abs_max, encoded_bitplanes, level_signs, v),
          queue_idx);
    } else if (Layout() == bitplane_layout_type::RegisterBlock) {
      DeviceLauncher<DeviceType>::Execute(
          BPDecoderRegisterBlockKernel<T_data, T_fp, T_sfp, T_bitplane,
                                       NUM_BITPLANES, NegaBinary, DeviceType>(
              n, starting_bitplane, abs_max, encoded_bitplanes, level_signs, v),
          queue_idx);
    } else if constexpr (std::is_same<DeviceType, CUDA>::value &&
                         sizeof(T_bitplane) == 4) {
      DeviceLauncher<DeviceType>::Execute(
          BPDecoderWarpTransposeKernel<T_data, T_fp, T_bitplane, NUM_BITPLANES,
                                       DeviceType>(
              n, starting_bitplane, abs_max, encoded_bitplanes, level_signs, v),
          queue_idx);
    } else {
      DeviceLauncher<DeviceType>::Execute(
          BPDecoderWarpTransposePortableKernel<T_data, T_fp, T_bitplane,
                                               NUM_BITPLANES, DeviceType>(
              n, starting_bitplane, abs_max, encoded_bitplanes, level_signs, v),
          queue_idx);
    }
  }

  bitplane_layout_type layout = bitplane_layout_type::LocalityBlock;
  int sign_group_size = 0;
  Array<1, uint64_t, DeviceType> sign_status;
  SIZE sign_status_size = 0;
  uint32_t sign_epoch = 0;
  bool initialized;
  Hierarchy<D, T_data, DeviceType> *hierarchy;
  Array<2, T_error, DeviceType> level_errors_work_array;
  Array<1, Byte, DeviceType> level_error_sum_work_array;
};

} // namespace MDR
} // namespace mgard_x

#endif
