#ifndef _MDR_BP_ENCODER_REGISTER_BLOCK_HPP
#define _MDR_BP_ENCODER_REGISTER_BLOCK_HPP

#include "../../RuntimeX/RuntimeX.h"

#include "BPEncoderWarp.hpp"
#include "BitplaneEncoderInterface.hpp"
#include "SignificanceCoding.hpp"
#include <string.h>

namespace mgard_x {
namespace MDR {

// Contiguous: word w of a row holds the bits of coefficients
// [32 w, 32 w + 32), coefficient 32 w + d at bit d. Otherwise (the original
// layout) word w holds coefficients w, w + n / 32, ..., coefficient
// d * n / 32 + w at bit 31 - d.
template <typename T_data, typename T_fp, typename T_sfp, typename T_bitplane,
          typename T_error, int NUM_BITPLANES, bool NegaBinary, bool ControlL2,
          typename DeviceType, bool Contiguous = false>
class BPEncoderRegisterBlockFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT
  BPEncoderRegisterBlockFunctor() {}
  MGARDX_CONT
  BPEncoderRegisterBlockFunctor(
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
        buffer |= bit << (Contiguous ? data_idx : BATCH_SIZE - 1 - data_idx);
      }
      encoded[bp_idx] = buffer;
    }
  }

  MGARDX_EXEC void error_collect_binary(T_data *shifted_data, T_error *errors,
                                        int exp) {

    int batch_idx = FunctorBase<DeviceType>::GetBlockIdX() *
                        FunctorBase<DeviceType>::GetBlockDimX() +
                    FunctorBase<DeviceType>::GetThreadIdX();

    for (int bp_idx = 0; bp_idx < NUM_BITPLANES; bp_idx++) {
      for (int data_idx = 0; data_idx < BATCH_SIZE; data_idx++) {
        T_data data = shifted_data[data_idx];
        T_fp fp_data = (T_fp)fabs(data);
        T_error mantissa = fabs(data) - fp_data;
        T_fp mask = ((T_fp)1 << bp_idx) - 1;
        T_error diff = (T_error)(fp_data & mask) + mantissa;
        // if (bp_idx == 31 && batch_idx == 0) {
        //   printf(
        //       "data: %f  fp_data: %llu  fps_data: %lld  mask: %llu  diff:
        //       %f\n", data, fp_data, sfp_data, mask, diff);
        // }
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

  MGARDX_EXEC void error_collect_negabinary(T_data *shifted_data,
                                            T_error *errors, int exp) {

    int batch_idx = FunctorBase<DeviceType>::GetBlockIdX() *
                        FunctorBase<DeviceType>::GetBlockDimX() +
                    FunctorBase<DeviceType>::GetThreadIdX();

    for (int bp_idx = 0; bp_idx < NUM_BITPLANES; bp_idx++) {
      for (int data_idx = 0; data_idx < BATCH_SIZE; data_idx++) {
        T_data data = shifted_data[data_idx];
        T_fp fp_data = (T_fp)fabs(data);
        T_error mantissa = fabs(data) - fp_data;
        T_fp mask = ((T_fp)1 << bp_idx) - 1;
        T_fp ngb_data = Math<DeviceType>::binary2negabinary((T_sfp)data);
        T_error diff =
            (T_error)Math<DeviceType>::negabinary2binary(ngb_data & mask) +
            mantissa;
        // if (bp_idx == 31 && batch_idx == 0) {
        //   printf(
        //       "data: %f  fp_data: %llu  fps_data: %lld  mask: %llu  diff:
        //       %f\n", data, fp_data, sfp_data, mask, diff);
        // }
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

  MGARDX_EXEC void EncodeBinary() {
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
        T_data data = *v(Contiguous ? batch_idx * BATCH_SIZE + data_idx
                                    : data_idx * num_full_batches + batch_idx);
        // this can cause overflow
        shifted_data[data_idx] = data * ((T_fp)1 << NUM_BITPLANES - exp);
        // ldexp without constant argument is slow
        // shifted_data[data_idx] = ldexp(data, NUM_BITPLANES - exp);
        fp_data[data_idx] = (T_fp)fabs(shifted_data[data_idx]);

        // if (num_full_batches == 1) printf("data: %f * %d %d, shifted_data: %f
        // fp_data: %llu \n", data, NUM_BITPLANES, exp, shifted_data[data_idx],
        // fp_data[data_idx]);
      }
    } else {
#pragma unroll
      for (int data_idx = 0; data_idx < BATCH_SIZE; data_idx++) {
        T_data data = *v(Contiguous ? batch_idx * BATCH_SIZE + data_idx
                                    : data_idx * num_full_batches + batch_idx);
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
                      << (Contiguous ? data_idx : BATCH_SIZE - 1 - data_idx);
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
      // errors[] is uninitialized stack memory; error_collect_binary
      // accumulates into it with +=, so it must be zeroed first.
      for (int bp_idx = 0; bp_idx < NUM_BITPLANES + 1; bp_idx++) {
        errors[bp_idx] = 0;
      }
      error_collect_binary(shifted_data, errors, exp);
      for (int bp_idx = 0; bp_idx < NUM_BITPLANES + 1; bp_idx++) {
        *level_errors_workspace(bp_idx, batch_idx) = errors[bp_idx];
      }
    }
  }

  MGARDX_EXEC void EncodeNegaBinary() {
    SIZE batch_idx = FunctorBase<DeviceType>::GetBlockIdX() *
                         FunctorBase<DeviceType>::GetBlockDimX() +
                     FunctorBase<DeviceType>::GetThreadIdX();

    SIZE num_full_batches = n / BATCH_SIZE;

    T_data shifted_data[BATCH_SIZE];
    T_fp fp_data[BATCH_SIZE];
    T_bitplane encoded_data[NUM_BITPLANES];
    T_error errors[NUM_BITPLANES + 1];

    int exp;
    frexp(*abs_max((IDX)0), &exp);
    exp += 2;

    if (batch_idx >= num_full_batches) {
      return;
    }

    if (exp > 0) {
#pragma unroll
      for (int data_idx = 0; data_idx < BATCH_SIZE; data_idx++) {
        T_data data = 0;
        data = *v(data_idx * num_full_batches + batch_idx);
        // This can cause overflow
        shifted_data[data_idx] = data * ((T_fp)1 << NUM_BITPLANES - exp);
        // ldexp without constant argument is slow
        // shifted_data[data_idx] = ldexp(data, NUM_BITPLANES - exp);
        fp_data[data_idx] =
            Math<DeviceType>::binary2negabinary((T_sfp)shifted_data[data_idx]);
      }
    } else {
#pragma unroll
      for (int data_idx = 0; data_idx < BATCH_SIZE; data_idx++) {
        T_data data = 0;
        data = *v(data_idx * num_full_batches + batch_idx);
        shifted_data[data_idx] = data * pow(2, NUM_BITPLANES - exp);
        // ldexp without constant argument is slow
        // shifted_data[data_idx] = ldexp(data, NUM_BITPLANES - exp);
        fp_data[data_idx] =
            Math<DeviceType>::binary2negabinary((T_sfp)shifted_data[data_idx]);
      }
    }

    // encode data
    encode_batch(fp_data, encoded_data);
// store data
#pragma unroll
    for (int bp_idx = 0; bp_idx < NUM_BITPLANES; bp_idx++) {
      *encoded_bitplanes(bp_idx, batch_idx) = encoded_data[bp_idx];
    }

    if constexpr (ControlL2) {
      // errors[] is uninitialized stack memory; error_collect_negabinary
      // accumulates into it with +=, so it must be zeroed first.
#pragma unroll
      for (int bp_idx = 0; bp_idx < NUM_BITPLANES + 1; bp_idx++) {
        errors[bp_idx] = 0;
      }
      error_collect_negabinary(shifted_data, errors, exp);
#pragma unroll
      for (int bp_idx = 0; bp_idx < NUM_BITPLANES + 1; bp_idx++) {
        *level_errors_workspace(bp_idx, batch_idx) = errors[bp_idx];
      }
    }
  }

  MGARDX_EXEC void Operation1() {
    if constexpr (NegaBinary) {
      EncodeNegaBinary();
    } else {
      EncodeBinary();
    }
  }

  MGARDX_CONT size_t shared_memory_size() {
    size_t size = 0;
    return size;
  }

private:
  // parameters
  SIZE n;
  SubArray<1, T_data, DeviceType> abs_max;
  SubArray<1, T_data, DeviceType> v;
  SubArray<2, T_bitplane, DeviceType> encoded_bitplanes;
  SubArray<2, T_error, DeviceType> level_errors_workspace;
  static constexpr int BATCH_SIZE = sizeof(T_bitplane) * 8;
};

template <typename T_data, typename T_fp, typename T_sfp, typename T_bitplane,
          typename T_error, int NUM_BITPLANES, bool NegaBinary, bool ControlL2,
          typename DeviceType, bool Contiguous = false>
class BPEncoderRegisterBlockKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "grouped bp encoder";
  static constexpr int BATCH_SIZE = sizeof(T_bitplane) * 8;
  MGARDX_CONT
  BPEncoderRegisterBlockKernel(
      SIZE n, SubArray<1, T_data, DeviceType> abs_max,
      SubArray<1, T_data, DeviceType> v,
      SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
      SubArray<2, T_error, DeviceType> level_errors_workspace)
      : n(n), abs_max(abs_max), encoded_bitplanes(encoded_bitplanes), v(v),
        level_errors_workspace(level_errors_workspace) {}

  using FunctorType =
      BPEncoderRegisterBlockFunctor<T_data, T_fp, T_sfp, T_bitplane, T_error,
                                    NUM_BITPLANES, NegaBinary, ControlL2,
                                    DeviceType, Contiguous>;
  using TaskType = Task<FunctorType>;

  MGARDX_CONT TaskType GenTask(int queue_idx) {
    FunctorType functor(n, abs_max, v, encoded_bitplanes,
                        level_errors_workspace);
    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    SIZE total_thread = std::max((SIZE)1, n / BATCH_SIZE);
    tbz = 1;
    tby = 1;
    tbx = 256;
    gridz = 1;
    gridy = 1;
    gridx = (total_thread - 1) / tbx + 1;
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                std::string(Name));
  }

private:
  SIZE n;
  SubArray<1, T_data, DeviceType> abs_max;
  SubArray<1, T_data, DeviceType> v;
  SubArray<2, T_bitplane, DeviceType> encoded_bitplanes;
  SubArray<2, T_error, DeviceType> level_errors_workspace;
};

// SignMasks: every sign is in the sign masks of the significance-coded state
// (SignificanceCoding.hpp); there is no sign row.
template <typename T_data, typename T_fp, typename T_sfp, typename T_bitplane,
          int NUM_BITPLANES, bool NegaBinary, typename DeviceType,
          bool Contiguous = false, bool SignMasks = false>
class BPDecoderRegisterBlockFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT
  BPDecoderRegisterBlockFunctor() {}
  MGARDX_CONT
  BPDecoderRegisterBlockFunctor(
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
        T_fp bit = (encoded[bp_idx] >>
                    (Contiguous ? data_idx : BATCH_SIZE - 1 - data_idx)) &
                   (T_fp)1;
        buffer += bit << (NUM_BITPLANES - 1 - bp_idx);
        // printf("bit: %llu, buffer: %llu\n", bit, buffer);
      }
      v[data_idx] = buffer;
    }
  }

  MGARDX_EXEC void DecodeBinary() {
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
      // if (num_full_batches == 1) printf("encoded_data: %u\n",
      // encoded_data[bp_idx]);
    }
    // decode data
    decode_batch(fp_data, encoded_data);

    if constexpr (SignMasks) {
      const uint32_t *state = (const uint32_t *)signs.data();
#pragma unroll
      for (int data_idx = 0; data_idx < BATCH_SIZE; data_idx++) {
        SIZE i = Contiguous ? batch_idx * BATCH_SIZE + data_idx
                            : data_idx * num_full_batches + batch_idx;
        fp_sign[data_idx] =
            (state[significance::STATE_STRIDE * (i / 32) + 1] >> (i % 32)) &
            1u;
      }
    } else if (starting_bitplane == 0) {
      // decode sign
      encoded_sign = *encoded_bitplanes(0, batch_idx);
#pragma unroll
      for (int data_idx = 0; data_idx < BATCH_SIZE; data_idx++) {
        fp_sign[data_idx] =
            (encoded_sign >>
             (Contiguous ? data_idx : BATCH_SIZE - 1 - data_idx)) &
            (T_fp)1;
        *signs(Contiguous ? batch_idx * BATCH_SIZE + data_idx
                          : data_idx * num_full_batches + batch_idx) =
            fp_sign[data_idx];
      }
    } else {
#pragma unroll
      for (int data_idx = 0; data_idx < BATCH_SIZE; data_idx++) {
        fp_sign[data_idx] =
            *signs(Contiguous ? batch_idx * BATCH_SIZE + data_idx
                              : data_idx * num_full_batches + batch_idx);
      }
    }
#pragma unroll
    for (int data_idx = 0; data_idx < BATCH_SIZE; data_idx++) {
      shifted_data[data_idx] = (T_data)fp_data[data_idx];
      // It is beneficial to use pow instead of ldexp
      T_data data = shifted_data[data_idx] * pow(2, -ending_bitplane + exp);
      // T_data data = ldexp(shifted_data[data_idx], -ending_bitplane + exp);
      data = fp_sign[data_idx] ? -data : data;
      *v(Contiguous ? batch_idx * BATCH_SIZE + data_idx
                    : data_idx * num_full_batches + batch_idx) = data;

      // if (num_full_batches == 1) printf("%llu %f %f\n", fp_data[data_idx],
      // shifted_data[data_idx], data);
    }
  }

  MGARDX_EXEC void DecodeNegaBinary() {
    SIZE batch_idx = FunctorBase<DeviceType>::GetBlockIdX() *
                         FunctorBase<DeviceType>::GetBlockDimX() +
                     FunctorBase<DeviceType>::GetThreadIdX();

    SIZE num_full_batches = n / BATCH_SIZE;

    T_data shifted_data[BATCH_SIZE];
    T_fp fp_data[BATCH_SIZE];
    T_bitplane encoded_data[NUM_BITPLANES];

    int exp;
    frexp(*abs_max((IDX)0), &exp);
    exp += 2;

    if (batch_idx >= num_full_batches) {
      return;
    }

    int ending_bitplane = starting_bitplane + NUM_BITPLANES;

// load bitplanes
#pragma unroll
    for (int bp_idx = 0; bp_idx < NUM_BITPLANES; bp_idx++) {
      encoded_data[bp_idx] =
          *encoded_bitplanes(starting_bitplane + bp_idx, batch_idx);
      // print_bits(encoded_data[bp_idx], batch_size);
    }
    // decode data
    decode_batch(fp_data, encoded_data);

// store data
#pragma unroll
    for (int data_idx = 0; data_idx < BATCH_SIZE; data_idx++) {
      shifted_data[data_idx] =
          Math<DeviceType>::negabinary2binary(fp_data[data_idx]);
      // No noticing difference between the two
      T_data data = shifted_data[data_idx] * pow(2, -ending_bitplane + exp);
      // T_data data = ldexp(shifted_data[data_idx], -ending_bitplane + exp);
      data = ending_bitplane % 2 != 0 ? -data : data;
      *v(data_idx * num_full_batches + batch_idx) = data;
      // printf("%f: ", data); print_bits(fp_data[data_idx], b);
    }
  }

  MGARDX_EXEC void Operation1() {
    if constexpr (NegaBinary) {
      DecodeNegaBinary();
    } else {
      DecodeBinary();
    }
  }

  MGARDX_CONT size_t shared_memory_size() {
    size_t size = 0;
    return size;
  }

private:
  // parameters
  SIZE n;
  int starting_bitplane;
  SubArray<1, T_data, DeviceType> abs_max;
  SubArray<2, T_bitplane, DeviceType> encoded_bitplanes;
  SubArray<1, bool, DeviceType> signs;
  SubArray<1, T_data, DeviceType> v;
  static constexpr int BATCH_SIZE = sizeof(T_bitplane) * 8;
  static constexpr int MAX_BITPLANES = sizeof(T_data) * 8;
};

template <typename T_data, typename T_fp, typename T_sfp, typename T_bitplane,
          int NUM_BITPLANES, bool NegaBinary, typename DeviceType,
          bool Contiguous = false, bool SignMasks = false>
class BPDecoderRegisterBlockKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "grouped bp decoder";
  static constexpr SIZE BATCH_SIZE = sizeof(T_bitplane) * 8;
  static constexpr int MAX_BITPLANES = sizeof(T_data) * 8;
  MGARDX_CONT
  BPDecoderRegisterBlockKernel(
      SIZE n, int starting_bitplane, SubArray<1, T_data, DeviceType> abs_max,
      SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
      SubArray<1, bool, DeviceType> signs, SubArray<1, T_data, DeviceType> v)
      : n(n), starting_bitplane(starting_bitplane), abs_max(abs_max),
        encoded_bitplanes(encoded_bitplanes), signs(signs), v(v) {}

  using FunctorType =
      BPDecoderRegisterBlockFunctor<T_data, T_fp, T_sfp, T_bitplane,
                                    NUM_BITPLANES, NegaBinary, DeviceType,
                                    Contiguous, SignMasks>;
  using TaskType = Task<FunctorType>;

  MGARDX_CONT TaskType GenTask(int queue_idx) {

    FunctorType functor(n, starting_bitplane, abs_max, encoded_bitplanes, signs,
                        v);
    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    SIZE total_thread = std::max((SIZE)1, n / BATCH_SIZE);
    tbz = 1;
    tby = 1;
    tbx = 256;
    gridz = 1;
    gridy = 1;
    gridx = (total_thread - 1) / tbx + 1;
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
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

// general bitplane encoder that encodes data by block using T_stream type
// buffer
template <DIM D, typename T_data, typename T_bitplane, typename T_error,
          bool NegaBinary, bool ControlL2, typename DeviceType>
class BPEncoderRegisterBlock
    : public concepts::BitplaneEncoderInterface<D, T_data, T_bitplane, T_error,
                                                ControlL2, DeviceType> {
public:
  static constexpr SIZE BATCH_SIZE = sizeof(T_bitplane) * 8;
  static constexpr int MAX_BITPLANES = sizeof(T_data) * 8;
  using T_sfp = typename std::conditional<std::is_same<T_data, double>::value,
                                          int64_t, int32_t>::type;
  using T_fp = typename std::conditional<std::is_same<T_data, double>::value,
                                         uint64_t, uint32_t>::type;

  BPEncoderRegisterBlock() : initialized(false) {
    static_assert(std::is_floating_point<T_data>::value,
                  "GeneralBPEncoder: input data must be floating points.");
    static_assert(!std::is_same<T_data, long double>::value,
                  "GeneralBPEncoder: long double is not supported.");
    static_assert(std::is_unsigned<T_bitplane>::value,
                  "GroupedBPBlockEncoder: streams must be unsigned integers.");
    static_assert(std::is_integral<T_bitplane>::value,
                  "GroupedBPBlockEncoder: streams must be unsigned integers.");
  }
  BPEncoderRegisterBlock(Hierarchy<D, T_data, DeviceType> &hierarchy) {
    static_assert(std::is_floating_point<T_data>::value,
                  "GeneralBPEncoder: input data must be floating points.");
    static_assert(!std::is_same<T_data, long double>::value,
                  "GeneralBPEncoder: long double is not supported.");
    static_assert(std::is_unsigned<T_bitplane>::value,
                  "GroupedBPBlockEncoder: streams must be unsigned integers.");
    static_assert(std::is_integral<T_bitplane>::value,
                  "GroupedBPBlockEncoder: streams must be unsigned integers.");
    Adapt(hierarchy, 0);
    DeviceRuntime<DeviceType>::SyncQueue(0);
  }

  // Layout of the encoded bitplanes: NUM_ROWS rows of bitplane_length(n)
  // words. With the binary (sign-magnitude) encoding row 0 holds the signs
  // and bitplane b is row b + 1; the negabinary encoding has no sign row.
  // (The signs used to share row 0 with bitplane 0 in a row twice as long,
  // which left the second half of every other row as zero padding.)
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
      using SignKernel = BPDecoderSignWarpKernel<T_data, T_fp, T_bitplane, 1,
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
    bool bitmaps_written =
        encode_rows(n, abs_max, v, encoded_bitplanes, level_errors, queue_idx,
                    ze_bitmaps, ze_bits, sign_counts, sign_segment_bits,
                    signs_packed);
    if (SignCoding() && !signs_packed) {
      DeviceLauncher<DeviceType>::Execute(
          significance::SignPackKernel<T_bitplane, DeviceType>(
              MAX_BITPLANES, sign_group_size, encoded_bitplanes, sign_counts,
              sign_segment_bits),
          queue_idx);
    }
    return bitmaps_written;
  }

  // Significance-coded signs with groups of group_size bitplanes (0: a sign
  // row). Binary encoding with contiguous words only.
  void SetSignCoding(int group_size) {
    sign_group_size = (NegaBinary || !contiguous) ? 0 : group_size;
  }
  bool SignCoding() const { return sign_group_size > 0; }

  static bool PortableSigns() { return portable_kernels(); }

private:
  bool encode_rows(SIZE n, SubArray<1, T_data, DeviceType> abs_max,
                   SubArray<1, T_data, DeviceType> v,
                   SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
                   SubArray<1, T_error, DeviceType> level_errors,
                   int queue_idx,
                   SubArray<1, uint32_t, DeviceType> ze_bitmaps,
                   SubArray<1, uint32_t, DeviceType> ze_bits,
                   SubArray<1, uint32_t, DeviceType> sign_counts,
                   SubArray<1, uint32_t, DeviceType> sign_segment_bits,
                   bool &signs_packed) {

    if (n % BATCH_SIZE != 0) {
      throw std::runtime_error(
          "BPEncoderV1b: n is not a multiple of BATCH_SIZE");
    }
    SubArray<2, T_error, DeviceType> level_errors_work(level_errors_work_array);

    if (!contiguous || NegaBinary) {
      DeviceLauncher<DeviceType>::Execute(
          BPEncoderRegisterBlockKernel<T_data, T_fp, T_sfp, T_bitplane, T_error,
                                       MAX_BITPLANES, NegaBinary, ControlL2,
                                       DeviceType>(
              n, abs_max, v, encoded_bitplanes, level_errors_work),
          queue_idx);
    } else if constexpr (std::is_same<DeviceType, CUDA>::value &&
                         sizeof(T_bitplane) == 4) {
      using WarpKernel = BPEncoderWarpKernel<T_data, T_fp, T_bitplane, T_error,
                                             MAX_BITPLANES, ControlL2,
                                             DeviceType>;
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
          BPEncoderRegisterBlockKernel<T_data, T_fp, T_sfp, T_bitplane, T_error,
                                       MAX_BITPLANES, NegaBinary, ControlL2,
                                       DeviceType, true>(
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

    // if (num_bitplanes > 0) {
    //   DeviceLauncher<DeviceType>::Execute(
    //       BPDecoderRegisterBlockKernel<T_data, T_fp, T_sfp, T_bitplane,
    //       NegaBinary,
    //                            DeviceType>(n, starting_bitplane,
    //                            num_bitplanes,
    //                                        abs_max, encoded_bitplanes,
    //                                        level_signs, v),
    //       queue_idx);
    // }

#define V1B_DECODE(NUM_BITPLANES)                                              \
  if (num_bitplanes == NUM_BITPLANES) {                                        \
    decode_with<NUM_BITPLANES>(n, starting_bitplane, abs_max,                  \
                               encoded_bitplanes, level_signs, v, queue_idx,   \
                               sign_sections);                                 \
  }
    V1B_DECODE(1);
    V1B_DECODE(2);
    V1B_DECODE(3);
    V1B_DECODE(4);
    V1B_DECODE(5);
    V1B_DECODE(6);
    V1B_DECODE(7);
    V1B_DECODE(8);
    V1B_DECODE(9);
    V1B_DECODE(10);
    V1B_DECODE(11);
    V1B_DECODE(12);
    V1B_DECODE(13);
    V1B_DECODE(14);
    V1B_DECODE(15);
    V1B_DECODE(16);
    V1B_DECODE(17);
    V1B_DECODE(18);
    V1B_DECODE(19);
    V1B_DECODE(20);
    V1B_DECODE(21);
    V1B_DECODE(22);
    V1B_DECODE(23);
    V1B_DECODE(24);
    V1B_DECODE(25);
    V1B_DECODE(26);
    V1B_DECODE(27);
    V1B_DECODE(28);
    V1B_DECODE(29);
    V1B_DECODE(30);
    V1B_DECODE(31);
    V1B_DECODE(32);
    V1B_DECODE(33);
    V1B_DECODE(34);
    V1B_DECODE(35);
    V1B_DECODE(36);
    V1B_DECODE(37);
    V1B_DECODE(38);
    V1B_DECODE(39);
    V1B_DECODE(40);
    V1B_DECODE(41);
    V1B_DECODE(42);
    V1B_DECODE(43);
    V1B_DECODE(44);
    V1B_DECODE(45);
    V1B_DECODE(46);
    V1B_DECODE(47);
    V1B_DECODE(48);
    V1B_DECODE(49);
    V1B_DECODE(50);
    V1B_DECODE(51);
    V1B_DECODE(52);
    V1B_DECODE(53);
    V1B_DECODE(54);
    V1B_DECODE(55);
    V1B_DECODE(56);
    V1B_DECODE(57);
    V1B_DECODE(58);
    V1B_DECODE(59);
    V1B_DECODE(60);
    V1B_DECODE(61);
    V1B_DECODE(62);
    V1B_DECODE(63);
    V1B_DECODE(64);
  }

  void print() const { std::cout << "Grouped bitplane encoder" << std::endl; }

  // Word layout of the rows (see BPEncoderRegisterBlockFunctor): contiguous
  // words hold 32 consecutive coefficients. NegaBinary is always strided.
  void SetWordOrder(bool contiguous) { this->contiguous = contiguous; }
  bool Contiguous() const { return contiguous && !NegaBinary; }

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
        using SignKernel = BPDecoderSignWarpKernel<T_data, T_fp, T_bitplane,
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
          BPDecoderRegisterBlockKernel<T_data, T_fp, T_sfp, T_bitplane,
                                       NUM_BITPLANES, NegaBinary, DeviceType,
                                       true, true>(n, starting_bitplane,
                                                   abs_max, encoded_bitplanes,
                                                   level_signs, v),
          queue_idx);
    } else if (!contiguous || NegaBinary) {
      DeviceLauncher<DeviceType>::Execute(
          BPDecoderRegisterBlockKernel<T_data, T_fp, T_sfp, T_bitplane,
                                       NUM_BITPLANES, NegaBinary, DeviceType>(
              n, starting_bitplane, abs_max, encoded_bitplanes, level_signs, v),
          queue_idx);
    } else if constexpr (std::is_same<DeviceType, CUDA>::value &&
                         sizeof(T_bitplane) == 4) {
      DeviceLauncher<DeviceType>::Execute(
          BPDecoderWarpKernel<T_data, T_fp, T_bitplane, NUM_BITPLANES,
                              DeviceType>(n, starting_bitplane, abs_max,
                                          encoded_bitplanes, level_signs, v),
          queue_idx);
    } else {
      DeviceLauncher<DeviceType>::Execute(
          BPDecoderRegisterBlockKernel<T_data, T_fp, T_sfp, T_bitplane,
                                       NUM_BITPLANES, NegaBinary, DeviceType,
                                       true>(n, starting_bitplane, abs_max,
                                             encoded_bitplanes, level_signs, v),
          queue_idx);
    }
  }

  bool contiguous = false;
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
