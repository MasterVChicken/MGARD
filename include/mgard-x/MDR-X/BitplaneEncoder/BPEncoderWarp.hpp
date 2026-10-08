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
// then gives the super-chunk counts).
template <typename T_data, typename T_fp, typename T_bitplane, typename T_error,
          int NUM_BITPLANES, bool ControlL2, typename DeviceType>
class BPEncoderWarpFunctor : public Functor<DeviceType> {
public:
  static constexpr SIZE WORDS = 32;
  static constexpr int HALVES = NUM_BITPLANES / 32; // sub-groups per tile
  static constexpr SIZE ROWS = 33;  // rows staged per sub-group (+ sign)
  static constexpr SIZE PITCH = WORDS + 1; // conflict-free transposed stores
  static constexpr SIZE WARPS = 4;
  static_assert(NUM_BITPLANES % 32 == 0 && WARPS % HALVES == 0);
  MGARDX_CONT BPEncoderWarpFunctor() {}
  MGARDX_CONT
  BPEncoderWarpFunctor(SIZE n, SubArray<1, T_data, DeviceType> abs_max,
                       SubArray<1, T_data, DeviceType> v,
                       SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
                       SubArray<2, T_error, DeviceType> level_errors_workspace,
                       SubArray<1, uint32_t, DeviceType> ze_bitmaps)
      : n(n), abs_max(abs_max), v(v), encoded_bitplanes(encoded_bitplanes),
        level_errors_workspace(level_errors_workspace), ze_bitmaps(ze_bitmaps) {
    Functor<DeviceType>();
  }

  static MGARDX_CONT_EXEC size_t shared_memory_bytes() {
    return WARPS * ROWS * PITCH * sizeof(uint32_t) +
           (ControlL2 ? WARPS * ROWS * sizeof(T_error) : 0);
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
    if (first < num_words) {
      SIZE words_in_tile =
          num_words - first < WORDS ? num_words - first : WORDS;
      T_data scale = exp > 0 ? (T_data)((T_fp)1 << (NUM_BITPLANES - exp))
                             : (T_data)pow(2, NUM_BITPLANES - exp);
      // Zero-elimination bitmaps: lane l builds the one of staged row
      // 32 - l, and the sign row's (top sub-group) in sign_bitmap.
      uint32_t row_bitmap = 0, sign_bitmap = 0;
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
        }
        // Bit 32 h + l of fp goes to staged row 32 - l.
        uint32_t x = warp_transpose32(sg, lane, (uint32_t)(fp >> (32 * h)));
        staging[(32 - lane) * PITCH + j] = x;
        row_bitmap |= (uint32_t)(x != 0) << j;
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

  MGARDX_CONT size_t shared_memory_size() { return shared_memory_bytes(); }

private:
  SIZE n;
  SubArray<1, T_data, DeviceType> abs_max;
  SubArray<1, T_data, DeviceType> v;
  SubArray<2, T_bitplane, DeviceType> encoded_bitplanes;
  SubArray<2, T_error, DeviceType> level_errors_workspace;
  SubArray<1, uint32_t, DeviceType> ze_bitmaps;
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
                      SubArray<1, uint32_t, DeviceType> ze_bitmaps = {})
      : n(n), abs_max(abs_max), v(v), encoded_bitplanes(encoded_bitplanes),
        level_errors_workspace(level_errors_workspace), ze_bitmaps(ze_bitmaps) {}
  MGARDX_CONT Task<FunctorType> GenTask(int queue_idx) {
    FunctorType functor(n, abs_max, v, encoded_bitplanes,
                        level_errors_workspace, ze_bitmaps);
    return Task(functor, 1, 1, num_blocks(n), 1, 1, FunctorType::WARPS * 32,
                functor.shared_memory_size(), queue_idx, std::string(Name));
  }

private:
  SIZE n;
  SubArray<1, T_data, DeviceType> abs_max;
  SubArray<1, T_data, DeviceType> v;
  SubArray<2, T_bitplane, DeviceType> encoded_bitplanes;
  SubArray<2, T_error, DeviceType> level_errors_workspace;
  SubArray<1, uint32_t, DeviceType> ze_bitmaps;
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

// Decoding counterpart: lane j loads word j of a tile for each row, and every
// word is broadcast (shfl) so that each lane assembles its own coefficient.
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
    for (int j = 0; j < (int)words_in_tile; j++) {
      T_fp fp = 0;
#pragma unroll
      for (int b = 0; b < NUM_BITPLANES; b++) {
        T_bitplane w = sg.shfl(words[b], j);
        fp |= (T_fp)((w >> lane) & 1) << (NUM_BITPLANES - 1 - b);
      }
      SIZE idx = (first + j) * WORDS + lane;
      bool sign;
      if (starting_bitplane == 0) {
        sign = (sg.shfl(sign_word, j) >> lane) & 1;
        *signs(idx) = sign;
      } else {
        sign = *signs(idx);
      }
      T_data data = (T_data)fp * scale;
      *v(idx) = sign ? -data : data;
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

} // namespace MDR
} // namespace mgard_x

#endif
