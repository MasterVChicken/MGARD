/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: October 8, 2026
 */

#ifndef MGARD_X_MDR_BP_ENCODER_WARP_HPP
#define MGARD_X_MDR_BP_ENCODER_WARP_HPP

#include "../../RuntimeX/RuntimeX.h"
#include "../LosslessCompressor/ZeroElimination.hpp"
#include "SignificanceCoding.hpp"
#include <type_traits>
#include <utility>

namespace mgard_x {
namespace MDR {

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

// Sums v[0..32) over the 32 lanes: lane l returns the sum of v[l]
// (31 exchanges instead of 32 butterfly reductions).
template <typename T, typename DeviceType>
MGARDX_EXEC T warp_transpose_reduce32(SubGroup<DeviceType> &sg, int lane,
                                      T *v) {
#pragma unroll
  for (int s = 16; s >= 1; s /= 2) {
    const bool upper = lane & s;
#pragma unroll
    for (int i = 0; i < s; i++) {
      T send = upper ? v[i] : v[i + s];
      T keep = upper ? v[i + s] : v[i];
      v[i] = keep + sg.shfl(send, lane ^ s);
    }
  }
  return v[0];
}

// d += bit ? 2^E : 0. On CUDA a predicated add with an immediate operand
// (the compiler otherwise adds unconditionally and selects).
template <int E> MGARDX_EXEC void add_pow2_if(double &d, uint32_t bit);
#if defined(__CUDA_ARCH__)
#define MGARDX_BP_ADD_POW2(E, HEX)                                             \
  template <>                                                                  \
  MGARDX_EXEC void add_pow2_if<E>(double &d, uint32_t bit) {                   \
    asm("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %1, 0;\n\t"                      \
        "@p add.rn.f64 %0, %0, 0d" HEX "0000000000000;\n\t}"                   \
        : "+d"(d)                                                              \
        : "r"(bit));                                                           \
  }
#else
#define MGARDX_BP_ADD_POW2(E, HEX)                                             \
  template <>                                                                  \
  MGARDX_EXEC void add_pow2_if<E>(double &d, uint32_t bit) {                   \
    if (bit) {                                                                 \
      d += (double)((uint64_t)1 << E);                                         \
    }                                                                          \
  }
#endif
MGARDX_BP_ADD_POW2(0, "3FF")
MGARDX_BP_ADD_POW2(1, "400")
MGARDX_BP_ADD_POW2(2, "401")
MGARDX_BP_ADD_POW2(3, "402")
MGARDX_BP_ADD_POW2(4, "403")
MGARDX_BP_ADD_POW2(5, "404")
MGARDX_BP_ADD_POW2(6, "405")
MGARDX_BP_ADD_POW2(7, "406")
MGARDX_BP_ADD_POW2(8, "407")
MGARDX_BP_ADD_POW2(9, "408")
MGARDX_BP_ADD_POW2(10, "409")
MGARDX_BP_ADD_POW2(11, "40A")
MGARDX_BP_ADD_POW2(12, "40B")
MGARDX_BP_ADD_POW2(13, "40C")
MGARDX_BP_ADD_POW2(14, "40D")
MGARDX_BP_ADD_POW2(15, "40E")
MGARDX_BP_ADD_POW2(16, "40F")
MGARDX_BP_ADD_POW2(17, "410")
MGARDX_BP_ADD_POW2(18, "411")
MGARDX_BP_ADD_POW2(19, "412")
MGARDX_BP_ADD_POW2(20, "413")
MGARDX_BP_ADD_POW2(21, "414")
MGARDX_BP_ADD_POW2(22, "415")
MGARDX_BP_ADD_POW2(23, "416")
MGARDX_BP_ADD_POW2(24, "417")
MGARDX_BP_ADD_POW2(25, "418")
MGARDX_BP_ADD_POW2(26, "419")
MGARDX_BP_ADD_POW2(27, "41A")
MGARDX_BP_ADD_POW2(28, "41B")
MGARDX_BP_ADD_POW2(29, "41C")
MGARDX_BP_ADD_POW2(30, "41D")
MGARDX_BP_ADD_POW2(31, "41E")
MGARDX_BP_ADD_POW2(32, "41F")
MGARDX_BP_ADD_POW2(33, "420")
MGARDX_BP_ADD_POW2(34, "421")
MGARDX_BP_ADD_POW2(35, "422")
MGARDX_BP_ADD_POW2(36, "423")
MGARDX_BP_ADD_POW2(37, "424")
MGARDX_BP_ADD_POW2(38, "425")
MGARDX_BP_ADD_POW2(39, "426")
MGARDX_BP_ADD_POW2(40, "427")
MGARDX_BP_ADD_POW2(41, "428")
MGARDX_BP_ADD_POW2(42, "429")
MGARDX_BP_ADD_POW2(43, "42A")
MGARDX_BP_ADD_POW2(44, "42B")
MGARDX_BP_ADD_POW2(45, "42C")
MGARDX_BP_ADD_POW2(46, "42D")
MGARDX_BP_ADD_POW2(47, "42E")
MGARDX_BP_ADD_POW2(48, "42F")
MGARDX_BP_ADD_POW2(49, "430")
MGARDX_BP_ADD_POW2(50, "431")
MGARDX_BP_ADD_POW2(51, "432")
MGARDX_BP_ADD_POW2(52, "433")
MGARDX_BP_ADD_POW2(53, "434")
MGARDX_BP_ADD_POW2(54, "435")
MGARDX_BP_ADD_POW2(55, "436")
MGARDX_BP_ADD_POW2(56, "437")
MGARDX_BP_ADD_POW2(57, "438")
MGARDX_BP_ADD_POW2(58, "439")
MGARDX_BP_ADD_POW2(59, "43A")
MGARDX_BP_ADD_POW2(60, "43B")
MGARDX_BP_ADD_POW2(61, "43C")
MGARDX_BP_ADD_POW2(62, "43D")
MGARDX_BP_ADD_POW2(63, "43E")
#undef MGARDX_BP_ADD_POW2

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

// Binary bitplane encoding in the contiguous word layout (word w of a row
// holds the bits of coefficients [32 w, 32 w + 32), coefficient 32 w + d at
// bit d) with one lane per coefficient: a 32-lane sub-group loads 32
// consecutive coefficients (coalesced) and a bit transpose turns their
// fixed-point values into one bitplane word per lane. A tile of 32 words is
// handled by NUM_BITPLANES / 32 sub-groups, sub-group h taking bits
// [32 h, 32 h + 32) (the top one also the signs); the words are staged in
// shared memory so that rows are stored coalesced. The per-bitplane squared
// errors are reduced per thread block into level_errors_workspace(b, block);
// BPErrorSumKernel adds the blocks up. Same rows, signs and errors (up to
// summation order) as BPEncoderRegisterBlockFunctor<..., Contiguous = true>.
// With ze_bitmaps given, a tile being a zero-elimination chunk, it also
// writes the chunk bitmaps of every row (zero_elimination::SuperCountKernel
// then gives the super-chunk counts), and with ze_bits the chunks' sparse-word
// payload bits. With sign_group_size > 0
// (significance-coded signs, SignificanceCoding.hpp), row 0 gets the tile's
// packed signs instead of the sign row, and sign_counts, sign_segment_bits
// (zero on entry) their counts: the same as SignPackKernel.
template <typename T_data, typename T_fp, typename T_bitplane, typename T_error,
          int NUM_BITPLANES, bool ControlL2, typename DeviceType>
class BPEncoderWarpFunctor : public Functor<DeviceType> {
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
  MGARDX_CONT BPEncoderWarpFunctor() {}
  MGARDX_CONT
  BPEncoderWarpFunctor(SIZE n, SubArray<1, T_data, DeviceType> abs_max,
                       SubArray<1, T_data, DeviceType> v,
                       SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
                       SubArray<2, T_error, DeviceType> level_errors_workspace,
                       SubArray<1, uint32_t, DeviceType> ze_bitmaps,
                       SubArray<1, uint32_t, DeviceType> ze_bits,
                       int sign_group_size,
                       SubArray<1, uint32_t, DeviceType> sign_counts,
                       SubArray<1, uint32_t, DeviceType> sign_segment_bits)
      : n(n), abs_max(abs_max), v(v), encoded_bitplanes(encoded_bitplanes),
        level_errors_workspace(level_errors_workspace), ze_bitmaps(ze_bitmaps),
        ze_bits(ze_bits), sign_group_size(sign_group_size),
        sign_counts(sign_counts),
        sign_segment_bits(sign_segment_bits) {
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

  // Squared errors of sub-group H: e[32 - k] += (|shifted| mod 2^(32 H + k))^2
  // for k in [0, 32), with |shifted| mod 2^b built bit by bit (exact) in two
  // independent chains.
  template <int H>
  MGARDX_EXEC void collect_errors(T_error *e, T_fp fp, T_error mantissa) {
    const uint32_t x = (uint32_t)(fp >> (32 * H));
    T_error d0 = H == 0 ? mantissa
                        : (T_error)(fp & (((T_fp)1 << (32 * H)) - 1)) +
                              mantissa;
    T_error d1 = (T_error)(fp & (((T_fp)1 << (32 * H + 16)) - 1)) + mantissa;
    collect_steps<H>(e, x, d0, d1, std::make_integer_sequence<int, 16>{});
  }

  template <int H, int... K>
  MGARDX_EXEC void collect_steps(T_error *e, uint32_t x, T_error &d0,
                                 T_error &d1, std::integer_sequence<int, K...>) {
    (collect_step<H, K>(e, x, d0, d1), ...);
  }

  template <int H, int K>
  MGARDX_EXEC void collect_step(T_error *e, uint32_t x, T_error &d0,
                                T_error &d1) {
    // Fused (the compiler does not contract these for double data).
    e[32 - K] = fma(d0, d0, e[32 - K]);
    e[16 - K] = fma(d1, d1, e[16 - K]);
    if constexpr (std::is_same<T_error, double>::value) {
      add_pow2_if<32 * H + K>(d0, x & (1u << K));
      add_pow2_if<32 * H + 16 + K>(d1, x & (1u << (16 + K)));
    } else {
      if (x & (1u << K)) {
        d0 += (T_error)((T_fp)1 << (32 * H + K));
      }
      if (x & (1u << (16 + K))) {
        d1 += (T_error)((T_fp)1 << (32 * H + 16 + K));
      }
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
    block_errors = (T_error *)((uint32_t *)FunctorBase<
                                   DeviceType>::GetSharedMemory() +
                               WARPS * ROWS * PITCH);
    frexp(*abs_max((IDX)0), &exp);

    T_error errors[ROWS];
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
          T_error mantissa = fabs(shifted) - fp;
          if constexpr (HALVES == 1) {
            collect_errors<0>(errors, fp, mantissa);
          } else {
            if (h == 0) {
              collect_errors<0>(errors, fp, mantissa);
            } else {
              collect_errors<HALVES - 1>(errors, fp, mantissa);
            }
          }
          if (top) {
            errors[0] += shifted * shifted;
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
      block_errors[warp * ROWS + lane] =
          warp_transpose_reduce32(sg, lane, errors);
      T_error e = errors[32];
      for (int offset = 16; offset > 0; offset /= 2) {
        e += sg.shfl(e, lane ^ offset);
      }
      if (lane == 0) {
        block_errors[warp * ROWS + 32] = e;
      }
    }
  }

  MGARDX_EXEC void Operation2() {
    if constexpr (ControlL2) {
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
  SIZE n;
  SubArray<1, T_data, DeviceType> abs_max;
  SubArray<1, T_data, DeviceType> v;
  SubArray<2, T_bitplane, DeviceType> encoded_bitplanes;
  SubArray<2, T_error, DeviceType> level_errors_workspace;
  SubArray<1, uint32_t, DeviceType> ze_bitmaps, ze_bits;
  int sign_group_size;
  SubArray<1, uint32_t, DeviceType> sign_counts, sign_segment_bits;
  T_error *block_errors;
  int exp;
};

template <typename T_data, typename T_fp, typename T_bitplane, typename T_error,
          int NUM_BITPLANES, bool ControlL2, typename DeviceType>
class BPEncoderWarpKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "warp bp encoder";
  using FunctorType =
      BPEncoderWarpFunctor<T_data, T_fp, T_bitplane, T_error, NUM_BITPLANES,
                           ControlL2, DeviceType>;
  // Thread blocks (= partial error sums) for n coefficients.
  static SIZE num_blocks(SIZE n) {
    constexpr SIZE tiles_per_block = FunctorType::WARPS / FunctorType::HALVES;
    SIZE tiles = (n / 32 + 31) / 32;
    return std::max((SIZE)1, (tiles + tiles_per_block - 1) / tiles_per_block);
  }
  MGARDX_CONT
  BPEncoderWarpKernel(SIZE n, SubArray<1, T_data, DeviceType> abs_max,
                      SubArray<1, T_data, DeviceType> v,
                      SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
                      SubArray<2, T_error, DeviceType> level_errors_workspace,
                      SubArray<1, uint32_t, DeviceType> ze_bitmaps = {},
                      SubArray<1, uint32_t, DeviceType> ze_bits = {},
                      int sign_group_size = 0,
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
class BPDecoderWarpFunctor : public Functor<DeviceType> {
public:
  static constexpr SIZE WORDS = 32;
  MGARDX_CONT BPDecoderWarpFunctor() {}
  MGARDX_CONT
  BPDecoderWarpFunctor(SIZE n, int starting_bitplane,
                       SubArray<1, T_data, DeviceType> abs_max,
                       SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
                       SubArray<1, bool, DeviceType> signs,
                       SubArray<1, T_data, DeviceType> v)
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
class BPDecoderWarpKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "warp bp decoder";
  using FunctorType = BPDecoderWarpFunctor<T_data, T_fp, T_bitplane,
                                           NUM_BITPLANES, DeviceType>;
  MGARDX_CONT
  BPDecoderWarpKernel(SIZE n, int starting_bitplane,
                      SubArray<1, T_data, DeviceType> abs_max,
                      SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
                      SubArray<1, bool, DeviceType> signs,
                      SubArray<1, T_data, DeviceType> v)
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
// scan of BPDecoderSignWarpFunctor (CUDA).
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
// Prefetch of the cache line holding *p into L1 (CUDA).
MGARDX_EXEC void prefetch_l1(const void *p) {
#if defined(__CUDA_ARCH__)
  asm volatile("prefetch.L1 [%0];" ::"l"(p));
#endif
}

// Decoding with significance-coded signs (SignificanceCoding.hpp; same
// results as SignResolveKernel followed by the decoder): warp per tile as
// BPDecoderWarpFunctor, WARPS tiles per thread block. Each warp counts the
// coefficients of its tile that become nonzero in each group, from the words
// it holds; the position of the tile's signs in a group's section is the
// segment's offset plus the counts of the tiles before it in the segment:
// those of the block (shared memory) and those of the blocks before it in the
// segment, which every block publishes (status(block, group), with this
// launch's epoch) before waiting for its predecessors' (a thread per group and
// predecessor). sections(i): the sign section of the i-th group decoded.
template <typename T_data, typename T_fp, typename T_bitplane,
          int NUM_BITPLANES, typename DeviceType>
class BPDecoderSignWarpFunctor : public Functor<DeviceType> {
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
  MGARDX_CONT BPDecoderSignWarpFunctor() {}
  MGARDX_CONT
  BPDecoderSignWarpFunctor(SIZE n, int starting_bitplane, int group_size,
                           SubArray<1, T_data, DeviceType> abs_max,
                           SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
                           SubArray<1, bool, DeviceType> signs,
                           SubArray<1, uint64_t, DeviceType> sections,
                           SubArray<1, uint64_t, DeviceType> status,
                           uint32_t epoch, SubArray<1, T_data, DeviceType> v)
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
          uint32_t c =
              wait_epoch(status((block - in_segment + p) * G + g), epoch);
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
            for (uint32_t m = fresh; m; m &= m - 1) {
              sgn |= (bits & 1u) << (sg.ffs(m) - 1);
              bits >>= 1;
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
class BPDecoderSignWarpKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "warp bp decoder (signs)";
  using FunctorType = BPDecoderSignWarpFunctor<T_data, T_fp, T_bitplane,
                                               NUM_BITPLANES, DeviceType>;
  // Thread blocks, and status words per group (one per block).
  static SIZE num_blocks(SIZE n) {
    return (significance::num_tiles(n / 32) + FunctorType::WARPS - 1) /
           FunctorType::WARPS;
  }
  MGARDX_CONT
  BPDecoderSignWarpKernel(SIZE n, int starting_bitplane, int group_size,
                          SubArray<1, T_data, DeviceType> abs_max,
                          SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
                          SubArray<1, bool, DeviceType> signs,
                          SubArray<1, uint64_t, DeviceType> sections,
                          SubArray<1, uint64_t, DeviceType> status,
                          uint32_t epoch, SubArray<1, T_data, DeviceType> v)
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

} // namespace MDR
} // namespace mgard_x

#endif
