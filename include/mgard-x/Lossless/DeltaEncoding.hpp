/*
 * Copyright 2026, University of Oregon.
 * BlockMGARD: Accelerating Adaptive Scientific Data Reduction with
 * Region-of-Interest Error Control on GPUs
 * Author: Yanliang Li (leonli@uoregon.edu)
 * Date: April 28, 2026 (redesigned May 2026)
 *
 * GPU-first delta + fixed-rate bit-plane lossless encoding.
 * Inspired by cuSZp (https://github.com/szcompressor/cuSZp).
 * Reference: cuSZp_2_3D.cu (warp-pack, outlier, lookback)
 *            cuSZp_3_3D.cu (3D predictor, fixed/plain/outlier modes)
 *
 * Block size: 32 elements = 1 GPU warp.
 *   GPU: __ballot_sync for O(1) bit-plane packing per plane.
 *        __shfl_xor_sync warp-reduce for per-block max.
 *   CPU: sequential fallback (lane 0 does all work).
 *
 * Three encoding modes:
 *   Fixed   – uniform bit-rate across all blocks (no per-block metadata).
 *   Plain   – per-block variable bit-rate (standard delta coding).
 *   Outlier – lane-0 encoded separately; remaining 31 lanes use fewer bits.
 *
 * Compressed data layout:
 *   [DeltaHeader 32B]  n, global_size, local_block_stride, mode, fixed_rate
 *   Mode 0 (Fixed):
 *     [num_blocks × (4 + fixed_rate*4) B]  block data at fixed stride
 *   Mode 1 (Plain):
 *     [uint8_t rates[num_blocks], padded↑4B]
 *     per block (rate>0): [4B sign_mask][rate×4B bit-planes]
 *   Mode 2 (Outlier):
 *     [uint8_t rate_bytes[num_blocks], padded↑4B]
 *     plain block (rate>0):  [4B sign][rate×4B planes]
 *     outlier block:         [1B info][1/2/4/8B abs_outlier]
 *                            [4B sign31][rate31×4B planes31]
 *
 * rate_byte encoding (Mode 2):
 *   bit 7 = 0 → plain;  bits 6:0 = rate (0–64)
 *   bit 7 = 1 → outlier; bits 6:5 = size_enc (0→1B,1→2B,2→4B,3→8B)
 *                         bits 4:0 = rate_no_outlier (0–31)
 *   outlier info byte (first byte in block data):
 *     bit 7 = sign, bits 1:0 = size_enc (redundant, for decoder convenience)
 *
 * Segment boundaries (delta resets before each):
 *   [0, global_size)                   – global wavelet segment
 *   [global_size + k*local_block_stride, …) – each local spatial block
 */

#ifndef MGARD_X_DELTA_ENCODING_TEMPLATE_HPP
#define MGARD_X_DELTA_ENCODING_TEMPLATE_HPP

#include <algorithm>
#include <cstdint>
#include <cstring>

#include "../RuntimeX/RuntimeX.h"

namespace mgard_x {

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────
static constexpr SIZE DELTA_BLOCK_SIZE = 32;   // elements per block = 1 warp

enum class DeltaMode : uint8_t { Fixed = 0, Plain = 1, Outlier = 2 };

// 32-byte header at the start of every compressed buffer
struct DeltaHeader {
  size_t  n;
  size_t  global_size;
  size_t  local_block_stride;
  uint8_t mode;          // DeltaMode value
  uint8_t fixed_rate;    // used only in mode Fixed
  uint8_t _pad[6];
};
static_assert(sizeof(DeltaHeader) == 32, "DeltaHeader must be 32 bytes");

// ─────────────────────────────────────────────────────────────────────────────
// Scalar helpers (host + device)
// ─────────────────────────────────────────────────────────────────────────────
MGARDX_CONT_EXEC
static int delta_bit_num(uint64_t x) {
  if (x == 0) return 0;
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return 64 - __clzll(static_cast<unsigned long long>(x));
#elif defined(__GNUC__) || defined(__clang__)
  return 64 - __builtin_clzll(x);
#else
  int b = 1; while (x >>= 1) ++b; return b;
#endif
}

MGARDX_CONT_EXEC
static bool delta_is_reset(SIZE idx, SIZE global_size,
                                  SIZE local_block_stride) {
  if (idx == 0) return true;
  if (global_size > 0 && idx == global_size) return true;
  if (global_size > 0 && local_block_stride > 0 && idx > global_size &&
      (idx - global_size) % local_block_stride == 0)
    return true;
  return false;
}

// Bytes needed to store a uint64_t (1, 2, 4, or 8)
MGARDX_CONT_EXEC
static int outlier_byte_count(uint64_t abs_d) {
  int bits = delta_bit_num(abs_d);
  if (bits <= 8)  return 1;
  if (bits <= 16) return 2;
  if (bits <= 32) return 4;
  return 8;
}

// Encode byte count as 2-bit field: 1→0, 2→1, 4→2, 8→3
MGARDX_CONT_EXEC
static int outlier_size_enc(int bytes) {
  return (bytes == 1) ? 0 : (bytes == 2) ? 1 : (bytes == 4) ? 2 : 3;
}

// Decode 2-bit field → byte count
MGARDX_CONT_EXEC
static int outlier_size_dec(int enc) {
  return 1 << enc;   // 0→1, 1→2, 2→4, 3→8
}

// ─────────────────────────────────────────────────────────────────────────────
// Warp-level primitives (GPU inline; no-op stubs on CPU)
// ─────────────────────────────────────────────────────────────────────────────

// Warp-reduce: return max(val) at every lane
MGARDX_EXEC
static uint64_t warp_reduce_max_u64(uint64_t val) {
#if defined(__CUDA_ARCH__)
  for (int i = 16; i >= 1; i >>= 1) {
    unsigned long long other =
        __shfl_xor_sync(0xffffffff, static_cast<unsigned long long>(val), i);
    if (other > val) val = static_cast<uint64_t>(other);
  }
#elif defined(__HIP_DEVICE_COMPILE__)
  for (int i = 16; i >= 1; i >>= 1) {
    unsigned long long other =
        __shfl_xor(static_cast<unsigned long long>(val), i);
    if (other > val) val = static_cast<uint64_t>(other);
  }
#endif
  return val;
}

// Ballot: bit lane_i set iff pred of lane_i is true
MGARDX_EXEC
static uint32_t warp_ballot(bool pred) {
#if defined(__CUDA_ARCH__)
  return __ballot_sync(0xffffffff, pred);
#elif defined(__HIP_DEVICE_COMPILE__)
  return __ballot(pred);
#else
  return 0u;
#endif
}

// Broadcast a 32-bit value from lane 0 to all lanes
MGARDX_EXEC
static uint32_t warp_shfl_i32(uint32_t val, int src_lane) {
#if defined(__CUDA_ARCH__)
  return __shfl_sync(0xffffffff, val, src_lane);
#elif defined(__HIP_DEVICE_COMPILE__)
  return __shfl(val, src_lane);
#else
  (void)src_lane; return val;
#endif
}

// Broadcast a uint8_t from lane 0
MGARDX_EXEC
static uint8_t warp_shfl_u8(uint8_t val, int src_lane) {
  return static_cast<uint8_t>(warp_shfl_i32(static_cast<uint32_t>(val), src_lane));
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: compute delta and |delta| for a given element index
// ─────────────────────────────────────────────────────────────────────────────
template <typename T, typename DeviceType>
MGARDX_EXEC
void elem_delta(SIZE idx, SIZE n,
                               SIZE global_size, SIZE local_block_stride,
                               SubArray<1, T, DeviceType> v,
                               int64_t &d, uint64_t &abs_d) {
  if (idx >= n) { d = 0; abs_d = 0; return; }
  const int64_t cur  = static_cast<int64_t>(*v(idx));
  const int64_t prev = delta_is_reset(idx, global_size, local_block_stride)
                       ? 0LL
                       : static_cast<int64_t>(*v(idx - 1));
  d     = cur - prev;
  abs_d = (d < 0) ? static_cast<uint64_t>(-d) : static_cast<uint64_t>(d);
}

// ══════════════════════════════════════════════════════════════════════════════
// KERNEL 1 — PlainRateFunctor
//   1 warp (32 threads) per 32-element block.
//   Computes per-block bit-rate needed for variable-rate (plain) encoding.
//   Also used internally by the Fixed-rate mode.
// ══════════════════════════════════════════════════════════════════════════════
template <typename T, typename DeviceType>
class PlainRateFunctor : public Functor<DeviceType> {
 public:
  MGARDX_EXEC PlainRateFunctor() {}
  MGARDX_EXEC PlainRateFunctor(SIZE n, SIZE global_size,
                               SIZE local_block_stride,
                               SubArray<1, T, DeviceType> v,
                               SubArray<1, uint8_t, DeviceType> rates)
      : n(n), global_size(global_size),
        local_block_stride(local_block_stride), v(v), rates(rates) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    const SIZE blk  = FunctorBase<DeviceType>::GetBlockIdX();
    const int  lane = static_cast<int>(FunctorBase<DeviceType>::GetThreadIdX());
    const SIZE num_blocks = (n + DELTA_BLOCK_SIZE - 1) / DELTA_BLOCK_SIZE;
    if (blk >= num_blocks) return;

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    // ── GPU warp-parallel path ────────────────────────────────────────────
    int64_t d; uint64_t abs_d;
    elem_delta(blk * DELTA_BLOCK_SIZE + lane, n,
               global_size, local_block_stride, v, d, abs_d);

    uint64_t max_abs = warp_reduce_max_u64(abs_d);  // same at every lane
    if (lane == 0)
      *rates(blk) = static_cast<uint8_t>(delta_bit_num(max_abs));
#else
    // ── CPU sequential fallback (lane 0 only) ────────────────────────────
    if (lane != 0) return;
    uint64_t max_abs = 0;
    for (int i = 0; i < static_cast<int>(DELTA_BLOCK_SIZE); ++i) {
      int64_t d; uint64_t abs_d;
      elem_delta(blk * DELTA_BLOCK_SIZE + i, n,
                 global_size, local_block_stride, v, d, abs_d);
      if (abs_d > max_abs) max_abs = abs_d;
    }
    *rates(blk) = static_cast<uint8_t>(delta_bit_num(max_abs));
#endif
  }
  MGARDX_CONT size_t shared_memory_size() { return 0; }

 private:
  SIZE n, global_size, local_block_stride;
  SubArray<1, T, DeviceType>       v;
  SubArray<1, uint8_t, DeviceType> rates;
};

template <typename T, typename DeviceType>
class PlainRateKernel : public Kernel {
 public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "delta_plain_rate";

  MGARDX_CONT PlainRateKernel(SIZE n, SIZE global_size,
                              SIZE local_block_stride,
                              SubArray<1, T, DeviceType> v,
                              SubArray<1, uint8_t, DeviceType> rates)
      : n(n), global_size(global_size),
        local_block_stride(local_block_stride), v(v), rates(rates) {}

  MGARDX_CONT Task<PlainRateFunctor<T, DeviceType>> GenTask(int queue_idx) {
    using F = PlainRateFunctor<T, DeviceType>;
    F functor(n, global_size, local_block_stride, v, rates);
    const SIZE num_blocks = (n + DELTA_BLOCK_SIZE - 1) / DELTA_BLOCK_SIZE;
    return Task(functor, 1, 1, num_blocks, 1, 1, DELTA_BLOCK_SIZE, 0,
                queue_idx, std::string(Name));
  }

 private:
  SIZE n, global_size, local_block_stride;
  SubArray<1, T, DeviceType>       v;
  SubArray<1, uint8_t, DeviceType> rates;
};

// ══════════════════════════════════════════════════════════════════════════════
// KERNEL 2 — OutlierRateFunctor
//   1 warp per block.  Evaluates whether treating lane-0 as an outlier saves
//   bytes. Writes a rate_byte[] that encodes both the mode decision and rate.
// ══════════════════════════════════════════════════════════════════════════════
template <typename T, typename DeviceType>
class OutlierRateFunctor : public Functor<DeviceType> {
 public:
  MGARDX_EXEC OutlierRateFunctor() {}
  MGARDX_EXEC OutlierRateFunctor(SIZE n, SIZE global_size,
                                 SIZE local_block_stride,
                                 SubArray<1, T, DeviceType> v,
                                 SubArray<1, uint8_t, DeviceType> rate_bytes,
                                 SubArray<1, SIZE, DeviceType>   byte_sizes)
      : n(n), global_size(global_size),
        local_block_stride(local_block_stride), v(v),
        rate_bytes(rate_bytes), byte_sizes(byte_sizes) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    const SIZE blk  = FunctorBase<DeviceType>::GetBlockIdX();
    const int  lane = static_cast<int>(FunctorBase<DeviceType>::GetThreadIdX());
    const SIZE num_blocks = (n + DELTA_BLOCK_SIZE - 1) / DELTA_BLOCK_SIZE;
    if (blk >= num_blocks) return;

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    // ── GPU warp-parallel path ────────────────────────────────────────────
    int64_t d; uint64_t abs_d;
    elem_delta(blk * DELTA_BLOCK_SIZE + lane, n,
               global_size, local_block_stride, v, d, abs_d);

    // max of ALL 32 elements
    uint64_t max_abs_all = warp_reduce_max_u64(abs_d);
    // max of lanes 1–31 (outlier candidate is lane 0)
    uint64_t abs_d_31 = (lane == 0) ? 0ULL : abs_d;
    uint64_t max_abs_31 = warp_reduce_max_u64(abs_d_31);

    if (lane == 0) {
      decide_and_store(blk, abs_d, max_abs_all, max_abs_31);
    }
#else
    // ── CPU sequential fallback ───────────────────────────────────────────
    if (lane != 0) return;
    uint64_t abs_d_lane0 = 0, max_abs_all = 0, max_abs_31 = 0;
    for (int i = 0; i < static_cast<int>(DELTA_BLOCK_SIZE); ++i) {
      int64_t di; uint64_t ai;
      elem_delta(blk * DELTA_BLOCK_SIZE + i, n,
                 global_size, local_block_stride, v, di, ai);
      if (i == 0) abs_d_lane0 = ai;
      if (ai > max_abs_all) max_abs_all = ai;
      if (i != 0 && ai > max_abs_31) max_abs_31 = ai;
    }
    decide_and_store(blk, abs_d_lane0, max_abs_all, max_abs_31);
#endif
  }

  MGARDX_CONT size_t shared_memory_size() { return 0; }

 private:
  MGARDX_EXEC void decide_and_store(SIZE blk, uint64_t abs_d0,
                                    uint64_t max_abs_all,
                                    uint64_t max_abs_31) {
    const int rate_all       = delta_bit_num(max_abs_all);
    const int rate_no_outlier= delta_bit_num(max_abs_31);
    const int ob             = outlier_byte_count(abs_d0);
    const int se             = outlier_size_enc(ob);

    // Plain cost: 4B sign + rate_all × 4B planes  (0 if rate_all == 0)
    const SIZE cost_plain = (rate_all > 0)
        ? static_cast<SIZE>(4 + rate_all * 4) : 0;
    // Outlier cost: 1B info + ob B magnitude + 4B sign31 + rate_no_outlier×4B
    //   (if abs_d0 == 0 and rate_no_outlier == 0 the block is all-zero anyway)
    const SIZE cost_outlier = static_cast<SIZE>(1 + ob + 4 + rate_no_outlier * 4);

    // Use outlier mode only if: rate_no_outlier fits in 5 bits AND it saves bytes
    const bool use_outlier = (rate_no_outlier <= 31) &&
                             (cost_outlier < cost_plain);

    if (use_outlier) {
      *rate_bytes(blk) = static_cast<uint8_t>(
          0x80u | (static_cast<unsigned>(se) << 5) |
          static_cast<unsigned>(rate_no_outlier));
      *byte_sizes(blk) = cost_outlier;
    } else {
      // plain block: bit 7 = 0, bits 6:0 = rate_all (0–64)
      *rate_bytes(blk) = static_cast<uint8_t>(rate_all & 0x7F);
      *byte_sizes(blk) = cost_plain;
    }
  }

  SIZE n, global_size, local_block_stride;
  SubArray<1, T, DeviceType>      v;
  SubArray<1, uint8_t, DeviceType> rate_bytes;
  SubArray<1, SIZE, DeviceType>    byte_sizes;
};

template <typename T, typename DeviceType>
class OutlierRateKernel : public Kernel {
 public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "delta_outlier_rate";

  MGARDX_CONT OutlierRateKernel(SIZE n, SIZE global_size,
                                SIZE local_block_stride,
                                SubArray<1, T, DeviceType> v,
                                SubArray<1, uint8_t, DeviceType> rate_bytes,
                                SubArray<1, SIZE, DeviceType>   byte_sizes)
      : n(n), global_size(global_size),
        local_block_stride(local_block_stride), v(v),
        rate_bytes(rate_bytes), byte_sizes(byte_sizes) {}

  MGARDX_CONT Task<OutlierRateFunctor<T, DeviceType>> GenTask(int queue_idx) {
    using F = OutlierRateFunctor<T, DeviceType>;
    F functor(n, global_size, local_block_stride, v, rate_bytes, byte_sizes);
    const SIZE num_blocks = (n + DELTA_BLOCK_SIZE - 1) / DELTA_BLOCK_SIZE;
    return Task(functor, 1, 1, num_blocks, 1, 1, DELTA_BLOCK_SIZE, 0,
                queue_idx, std::string(Name));
  }

 private:
  SIZE n, global_size, local_block_stride;
  SubArray<1, T, DeviceType>      v;
  SubArray<1, uint8_t, DeviceType> rate_bytes;
  SubArray<1, SIZE, DeviceType>    byte_sizes;
};

// ══════════════════════════════════════════════════════════════════════════════
// KERNEL 3 — FixedPackFunctor
//   1 warp per block. All blocks use the same fixed_rate.
//   Output offset = blk * (4 + fixed_rate*4)  (no offset array needed).
// ══════════════════════════════════════════════════════════════════════════════
template <typename T, typename DeviceType>
class FixedPackFunctor : public Functor<DeviceType> {
 public:
  MGARDX_EXEC FixedPackFunctor() {}
  MGARDX_EXEC FixedPackFunctor(SIZE n, SIZE num_blocks,
                               SIZE global_size, SIZE local_block_stride,
                               uint8_t fixed_rate,
                               SubArray<1, T, DeviceType>       v,
                               SubArray<1, uint8_t, DeviceType> out_data)
      : n(n), num_blocks(num_blocks),
        global_size(global_size), local_block_stride(local_block_stride),
        fixed_rate(fixed_rate), v(v), out_data(out_data) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    const SIZE blk  = FunctorBase<DeviceType>::GetBlockIdX();
    const int  lane = static_cast<int>(FunctorBase<DeviceType>::GetThreadIdX());
    if (blk >= num_blocks || fixed_rate == 0) return;

    const SIZE block_bytes = static_cast<SIZE>(4 + fixed_rate * 4);
    uint8_t *out = out_data((IDX)(blk * block_bytes));

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    // ── GPU warp path ─────────────────────────────────────────────────────
    int64_t d; uint64_t abs_d;
    elem_delta(blk * DELTA_BLOCK_SIZE + lane, n,
               global_size, local_block_stride, v, d, abs_d);

    // sign: bit lane set if element is negative
    uint32_t sign_mask = warp_ballot(d < 0 &&
                         (blk * DELTA_BLOCK_SIZE + lane) < n);
    if (lane == 0) reinterpret_cast<uint32_t *>(out)[0] = sign_mask;
    out += 4;

    for (int p = 0; p < static_cast<int>(fixed_rate); ++p) {
      uint32_t plane = warp_ballot(static_cast<bool>((abs_d >> p) & 1u));
      if (lane == 0) reinterpret_cast<uint32_t *>(out)[0] = plane;
      out += 4;
    }
#else
    // ── CPU sequential fallback ───────────────────────────────────────────
    if (lane != 0) return;
    uint32_t sign_mask = 0;
    uint64_t abs_ds[DELTA_BLOCK_SIZE] = {};
    for (int i = 0; i < static_cast<int>(DELTA_BLOCK_SIZE); ++i) {
      int64_t d; uint64_t abs_d;
      elem_delta(blk * DELTA_BLOCK_SIZE + i, n,
                 global_size, local_block_stride, v, d, abs_d);
      abs_ds[i] = abs_d;
      if (d < 0 && (blk * DELTA_BLOCK_SIZE + i) < n)
        sign_mask |= (1u << i);
    }
    reinterpret_cast<uint32_t *>(out)[0] = sign_mask;
    out += 4;
    for (int p = 0; p < static_cast<int>(fixed_rate); ++p) {
      uint32_t plane = 0;
      for (int i = 0; i < static_cast<int>(DELTA_BLOCK_SIZE); ++i)
        if ((abs_ds[i] >> p) & 1u) plane |= (1u << i);
      reinterpret_cast<uint32_t *>(out)[0] = plane;
      out += 4;
    }
#endif
  }
  MGARDX_CONT size_t shared_memory_size() { return 0; }

 private:
  SIZE n, num_blocks, global_size, local_block_stride;
  uint8_t fixed_rate;
  SubArray<1, T, DeviceType>       v;
  SubArray<1, uint8_t, DeviceType> out_data;
};

template <typename T, typename DeviceType>
class FixedPackKernel : public Kernel {
 public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "delta_fixed_pack";

  MGARDX_CONT FixedPackKernel(SIZE n, SIZE num_blocks,
                              SIZE global_size, SIZE local_block_stride,
                              uint8_t fixed_rate,
                              SubArray<1, T, DeviceType>       v,
                              SubArray<1, uint8_t, DeviceType> out_data)
      : n(n), num_blocks(num_blocks),
        global_size(global_size), local_block_stride(local_block_stride),
        fixed_rate(fixed_rate), v(v), out_data(out_data) {}

  MGARDX_CONT Task<FixedPackFunctor<T, DeviceType>> GenTask(int queue_idx) {
    using F = FixedPackFunctor<T, DeviceType>;
    F functor(n, num_blocks, global_size, local_block_stride,
              fixed_rate, v, out_data);
    return Task(functor, 1, 1, num_blocks, 1, 1, DELTA_BLOCK_SIZE, 0,
                queue_idx, std::string(Name));
  }

 private:
  SIZE n, num_blocks, global_size, local_block_stride;
  uint8_t fixed_rate;
  SubArray<1, T, DeviceType>       v;
  SubArray<1, uint8_t, DeviceType> out_data;
};

// ══════════════════════════════════════════════════════════════════════════════
// KERNEL 4 — PlainPackFunctor
//   Variable-rate packing. 1 warp per block.
//   byte_offsets[blk] gives the output start for that block's data.
// ══════════════════════════════════════════════════════════════════════════════
template <typename T, typename DeviceType>
class PlainPackFunctor : public Functor<DeviceType> {
 public:
  MGARDX_EXEC PlainPackFunctor() {}
  MGARDX_EXEC PlainPackFunctor(SIZE n, SIZE num_blocks,
                               SIZE global_size, SIZE local_block_stride,
                               SubArray<1, T, DeviceType>       v,
                               SubArray<1, uint8_t, DeviceType> rates,
                               SubArray<1, SIZE, DeviceType>    byte_offsets,
                               SubArray<1, uint8_t, DeviceType> out_data)
      : n(n), num_blocks(num_blocks),
        global_size(global_size), local_block_stride(local_block_stride),
        v(v), rates(rates), byte_offsets(byte_offsets), out_data(out_data) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    const SIZE blk  = FunctorBase<DeviceType>::GetBlockIdX();
    const int  lane = static_cast<int>(FunctorBase<DeviceType>::GetThreadIdX());
    if (blk >= num_blocks) return;

    const uint8_t rate = *rates(blk);
    if (!rate) return;

    uint8_t *out = out_data((IDX)(*byte_offsets(blk)));

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    // ── GPU warp path ─────────────────────────────────────────────────────
    int64_t d; uint64_t abs_d;
    elem_delta(blk * DELTA_BLOCK_SIZE + lane, n,
               global_size, local_block_stride, v, d, abs_d);

    uint32_t sign_mask = warp_ballot(d < 0 &&
                         (blk * DELTA_BLOCK_SIZE + lane) < n);
    if (lane == 0) reinterpret_cast<uint32_t *>(out)[0] = sign_mask;
    out += 4;

    for (int p = 0; p < static_cast<int>(rate); ++p) {
      uint32_t plane = warp_ballot(static_cast<bool>((abs_d >> p) & 1u));
      if (lane == 0) reinterpret_cast<uint32_t *>(out)[0] = plane;
      out += 4;
    }
#else
    // ── CPU sequential fallback ───────────────────────────────────────────
    if (lane != 0) return;
    uint32_t sign_mask = 0;
    uint64_t abs_ds[DELTA_BLOCK_SIZE] = {};
    for (int i = 0; i < static_cast<int>(DELTA_BLOCK_SIZE); ++i) {
      int64_t d; uint64_t abs_d;
      elem_delta(blk * DELTA_BLOCK_SIZE + i, n,
                 global_size, local_block_stride, v, d, abs_d);
      abs_ds[i] = abs_d;
      if (d < 0 && (blk * DELTA_BLOCK_SIZE + i) < n)
        sign_mask |= (1u << i);
    }
    reinterpret_cast<uint32_t *>(out)[0] = sign_mask;
    out += 4;
    for (int p = 0; p < static_cast<int>(rate); ++p) {
      uint32_t plane = 0;
      for (int i = 0; i < static_cast<int>(DELTA_BLOCK_SIZE); ++i)
        if ((abs_ds[i] >> p) & 1u) plane |= (1u << i);
      reinterpret_cast<uint32_t *>(out)[0] = plane;
      out += 4;
    }
#endif
  }
  MGARDX_CONT size_t shared_memory_size() { return 0; }

 private:
  SIZE n, num_blocks, global_size, local_block_stride;
  SubArray<1, T, DeviceType>       v;
  SubArray<1, uint8_t, DeviceType> rates;
  SubArray<1, SIZE, DeviceType>    byte_offsets;
  SubArray<1, uint8_t, DeviceType> out_data;
};

template <typename T, typename DeviceType>
class PlainPackKernel : public Kernel {
 public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "delta_plain_pack";

  MGARDX_CONT PlainPackKernel(SIZE n, SIZE num_blocks,
                              SIZE global_size, SIZE local_block_stride,
                              SubArray<1, T, DeviceType>       v,
                              SubArray<1, uint8_t, DeviceType> rates,
                              SubArray<1, SIZE, DeviceType>    byte_offsets,
                              SubArray<1, uint8_t, DeviceType> out_data)
      : n(n), num_blocks(num_blocks),
        global_size(global_size), local_block_stride(local_block_stride),
        v(v), rates(rates), byte_offsets(byte_offsets), out_data(out_data) {}

  MGARDX_CONT Task<PlainPackFunctor<T, DeviceType>> GenTask(int queue_idx) {
    using F = PlainPackFunctor<T, DeviceType>;
    F functor(n, num_blocks, global_size, local_block_stride,
              v, rates, byte_offsets, out_data);
    return Task(functor, 1, 1, num_blocks, 1, 1, DELTA_BLOCK_SIZE, 0,
                queue_idx, std::string(Name));
  }

 private:
  SIZE n, num_blocks, global_size, local_block_stride;
  SubArray<1, T, DeviceType>       v;
  SubArray<1, uint8_t, DeviceType> rates;
  SubArray<1, SIZE, DeviceType>    byte_offsets;
  SubArray<1, uint8_t, DeviceType> out_data;
};

// ══════════════════════════════════════════════════════════════════════════════
// KERNEL 5 — OutlierPackFunctor
//   Outlier-aware packing. Lane 0 is the outlier candidate.
//   rate_byte bit 7=1 → outlier block; bit 7=0 → fall through to plain pack.
// ══════════════════════════════════════════════════════════════════════════════
template <typename T, typename DeviceType>
class OutlierPackFunctor : public Functor<DeviceType> {
 public:
  MGARDX_EXEC OutlierPackFunctor() {}
  MGARDX_EXEC OutlierPackFunctor(SIZE n, SIZE num_blocks,
                                 SIZE global_size, SIZE local_block_stride,
                                 SubArray<1, T, DeviceType>       v,
                                 SubArray<1, uint8_t, DeviceType> rate_bytes,
                                 SubArray<1, SIZE, DeviceType>    byte_offsets,
                                 SubArray<1, uint8_t, DeviceType> out_data)
      : n(n), num_blocks(num_blocks),
        global_size(global_size), local_block_stride(local_block_stride),
        v(v), rate_bytes(rate_bytes), byte_offsets(byte_offsets),
        out_data(out_data) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    const SIZE blk  = FunctorBase<DeviceType>::GetBlockIdX();
    const int  lane = static_cast<int>(FunctorBase<DeviceType>::GetThreadIdX());
    if (blk >= num_blocks) return;

    const uint8_t rb = *rate_bytes(blk);
    const bool is_outlier_block = (rb >> 7) & 1u;

    uint8_t *out = out_data((IDX)(*byte_offsets(blk)));

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    // ── GPU warp path ─────────────────────────────────────────────────────
    int64_t d; uint64_t abs_d;
    elem_delta(blk * DELTA_BLOCK_SIZE + lane, n,
               global_size, local_block_stride, v, d, abs_d);

    if (!is_outlier_block) {
      // Plain block (bit 7 = 0)
      const uint8_t rate = rb & 0x7Fu;
      if (!rate) return;
      uint32_t sign_mask = warp_ballot(d < 0 &&
                           (blk * DELTA_BLOCK_SIZE + lane) < n);
      if (lane == 0) reinterpret_cast<uint32_t *>(out)[0] = sign_mask;
      out += 4;
      for (int p = 0; p < static_cast<int>(rate); ++p) {
        uint32_t plane = warp_ballot(static_cast<bool>((abs_d >> p) & 1u));
        if (lane == 0) reinterpret_cast<uint32_t *>(out)[0] = plane;
        out += 4;
      }
    } else {
      // Outlier block: lane 0 is the outlier
      const int se   = (rb >> 5) & 0x3;
      const int ob   = outlier_size_dec(se);
      const uint8_t rate31 = rb & 0x1Fu;

      // Lane 0 writes outlier info byte + magnitude
      if (lane == 0) {
        // info byte: bit 7 = sign, bits 1:0 = size_enc
        uint8_t info = static_cast<uint8_t>((d < 0 ? 0x80u : 0x00u) |
                                             static_cast<unsigned>(se));
        out[0] = info;
        for (int b = 0; b < ob; ++b)
          out[1 + b] = static_cast<uint8_t>((abs_d >> (8 * b)) & 0xFFu);
      }
      // All lanes advance past the outlier header (local pointer, no mem write)
      out += 1 + ob;

      // Sign mask for lanes 1–31 (lane 0 excluded → contributes false)
      uint32_t sign_mask = warp_ballot(lane != 0 && d < 0 &&
                           (blk * DELTA_BLOCK_SIZE + lane) < n);
      if (lane == 0) reinterpret_cast<uint32_t *>(out)[0] = sign_mask;
      out += 4;

      // Bit planes for lanes 1–31; lane 0 contributes 0 (excluded)
      const uint64_t eff_abs_d = (lane == 0) ? 0ULL : abs_d;
      for (int p = 0; p < static_cast<int>(rate31); ++p) {
        uint32_t plane = warp_ballot(static_cast<bool>((eff_abs_d >> p) & 1u));
        if (lane == 0) reinterpret_cast<uint32_t *>(out)[0] = plane;
        out += 4;
      }
    }
#else
    // ── CPU sequential fallback ───────────────────────────────────────────
    if (lane != 0) return;
    if (!is_outlier_block) {
      const uint8_t rate = rb & 0x7Fu;
      if (!rate) return;
      uint32_t sign_mask = 0;
      uint64_t abs_ds[DELTA_BLOCK_SIZE] = {};
      for (int i = 0; i < static_cast<int>(DELTA_BLOCK_SIZE); ++i) {
        int64_t di; uint64_t ai;
        elem_delta(blk * DELTA_BLOCK_SIZE + i, n,
                   global_size, local_block_stride, v, di, ai);
        abs_ds[i] = ai;
        if (di < 0 && (blk * DELTA_BLOCK_SIZE + i) < n) sign_mask |= (1u<<i);
      }
      reinterpret_cast<uint32_t *>(out)[0] = sign_mask;
      out += 4;
      for (int p = 0; p < static_cast<int>(rate); ++p) {
        uint32_t plane = 0;
        for (int i = 0; i < static_cast<int>(DELTA_BLOCK_SIZE); ++i)
          if ((abs_ds[i] >> p) & 1u) plane |= (1u << i);
        reinterpret_cast<uint32_t *>(out)[0] = plane;
        out += 4;
      }
    } else {
      const int se     = (rb >> 5) & 0x3;
      const int ob     = outlier_size_dec(se);
      const uint8_t rate31 = rb & 0x1Fu;
      // lane-0 delta (outlier)
      int64_t d0; uint64_t abs_d0;
      elem_delta(blk * DELTA_BLOCK_SIZE + 0, n,
                 global_size, local_block_stride, v, d0, abs_d0);
      out[0] = static_cast<uint8_t>((d0 < 0 ? 0x80u : 0x00u) |
                                     static_cast<unsigned>(se));
      for (int b = 0; b < ob; ++b)
        out[1 + b] = static_cast<uint8_t>((abs_d0 >> (8 * b)) & 0xFFu);
      out += 1 + ob;
      // lanes 1–31
      uint32_t sign_mask = 0;
      uint64_t abs_ds[DELTA_BLOCK_SIZE] = {};
      for (int i = 1; i < static_cast<int>(DELTA_BLOCK_SIZE); ++i) {
        int64_t di; uint64_t ai;
        elem_delta(blk * DELTA_BLOCK_SIZE + i, n,
                   global_size, local_block_stride, v, di, ai);
        abs_ds[i] = ai;
        if (di < 0 && (blk * DELTA_BLOCK_SIZE + i) < n) sign_mask |= (1u<<i);
      }
      reinterpret_cast<uint32_t *>(out)[0] = sign_mask;
      out += 4;
      for (int p = 0; p < static_cast<int>(rate31); ++p) {
        uint32_t plane = 0;
        for (int i = 1; i < static_cast<int>(DELTA_BLOCK_SIZE); ++i)
          if ((abs_ds[i] >> p) & 1u) plane |= (1u << i);
        reinterpret_cast<uint32_t *>(out)[0] = plane;
        out += 4;
      }
    }
#endif
  }
  MGARDX_CONT size_t shared_memory_size() { return 0; }

 private:
  SIZE n, num_blocks, global_size, local_block_stride;
  SubArray<1, T, DeviceType>       v;
  SubArray<1, uint8_t, DeviceType> rate_bytes;
  SubArray<1, SIZE, DeviceType>    byte_offsets;
  SubArray<1, uint8_t, DeviceType> out_data;
};

template <typename T, typename DeviceType>
class OutlierPackKernel : public Kernel {
 public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "delta_outlier_pack";

  MGARDX_CONT OutlierPackKernel(SIZE n, SIZE num_blocks,
                                SIZE global_size, SIZE local_block_stride,
                                SubArray<1, T, DeviceType>       v,
                                SubArray<1, uint8_t, DeviceType> rate_bytes,
                                SubArray<1, SIZE, DeviceType>    byte_offsets,
                                SubArray<1, uint8_t, DeviceType> out_data)
      : n(n), num_blocks(num_blocks),
        global_size(global_size), local_block_stride(local_block_stride),
        v(v), rate_bytes(rate_bytes), byte_offsets(byte_offsets),
        out_data(out_data) {}

  MGARDX_CONT Task<OutlierPackFunctor<T, DeviceType>> GenTask(int queue_idx) {
    using F = OutlierPackFunctor<T, DeviceType>;
    F functor(n, num_blocks, global_size, local_block_stride,
              v, rate_bytes, byte_offsets, out_data);
    return Task(functor, 1, 1, num_blocks, 1, 1, DELTA_BLOCK_SIZE, 0,
                queue_idx, std::string(Name));
  }

 private:
  SIZE n, num_blocks, global_size, local_block_stride;
  SubArray<1, T, DeviceType>       v;
  SubArray<1, uint8_t, DeviceType> rate_bytes;
  SubArray<1, SIZE, DeviceType>    byte_offsets;
  SubArray<1, uint8_t, DeviceType> out_data;
};

// ══════════════════════════════════════════════════════════════════════════════
// KERNEL 6 — FixedUnpackFunctor
//   Decompress fixed-rate blocks. Writes signed int64_t deltas.
// ══════════════════════════════════════════════════════════════════════════════
template <typename DeviceType>
class FixedUnpackFunctor : public Functor<DeviceType> {
 public:
  MGARDX_EXEC FixedUnpackFunctor() {}
  MGARDX_EXEC FixedUnpackFunctor(SIZE n, SIZE num_blocks, uint8_t fixed_rate,
                                 SubArray<1, uint8_t, DeviceType> in_data,
                                 SubArray<1, int64_t, DeviceType> deltas)
      : n(n), num_blocks(num_blocks), fixed_rate(fixed_rate),
        in_data(in_data), deltas(deltas) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    const SIZE blk  = FunctorBase<DeviceType>::GetBlockIdX();
    const int  lane = static_cast<int>(FunctorBase<DeviceType>::GetThreadIdX());
    if (blk >= num_blocks) return;

    const SIZE base = blk * DELTA_BLOCK_SIZE;

    if (!fixed_rate) return;  // d_deltas already zero-initialized

    const SIZE block_bytes = static_cast<SIZE>(4 + fixed_rate * 4);
    const uint8_t *in = in_data((IDX)(blk * block_bytes));

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    // ── GPU warp path ─────────────────────────────────────────────────────
    const uint32_t sign_mask =
        reinterpret_cast<const uint32_t *>(in)[0];
    in += 4;
    uint64_t abs_d = 0;
    for (int p = 0; p < static_cast<int>(fixed_rate); ++p) {
      uint32_t plane = reinterpret_cast<const uint32_t *>(in)[0];
      in += 4;
      if ((plane >> lane) & 1u) abs_d |= static_cast<uint64_t>(1) << p;
    }
    const SIZE idx = base + lane;
    if (idx < n) {
      bool neg = (sign_mask >> lane) & 1u;
      *deltas(idx) = neg ? -static_cast<int64_t>(abs_d)
                         :  static_cast<int64_t>(abs_d);
    }
#else
    // ── CPU sequential fallback ───────────────────────────────────────────
    if (lane != 0) return;
    const uint32_t sign_mask = reinterpret_cast<const uint32_t *>(in)[0];
    in += 4;
    uint64_t abs_ds[DELTA_BLOCK_SIZE] = {};
    for (int p = 0; p < static_cast<int>(fixed_rate); ++p) {
      uint32_t plane = reinterpret_cast<const uint32_t *>(in)[0];
      in += 4;
      for (int i = 0; i < static_cast<int>(DELTA_BLOCK_SIZE); ++i)
        if ((plane >> i) & 1u) abs_ds[i] |= static_cast<uint64_t>(1) << p;
    }
    for (int i = 0; i < static_cast<int>(DELTA_BLOCK_SIZE); ++i) {
      SIZE idx = base + i;
      if (idx >= n) break;
      bool neg = (sign_mask >> i) & 1u;
      *deltas(idx) = neg ? -static_cast<int64_t>(abs_ds[i])
                         :  static_cast<int64_t>(abs_ds[i]);
    }
#endif
  }
  MGARDX_CONT size_t shared_memory_size() { return 0; }

 private:
  SIZE n, num_blocks;
  uint8_t fixed_rate;
  SubArray<1, uint8_t, DeviceType> in_data;
  SubArray<1, int64_t, DeviceType> deltas;
};

template <typename DeviceType>
class FixedUnpackKernel : public Kernel {
 public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "delta_fixed_unpack";

  MGARDX_CONT FixedUnpackKernel(SIZE n, SIZE num_blocks, uint8_t fixed_rate,
                                SubArray<1, uint8_t, DeviceType> in_data,
                                SubArray<1, int64_t, DeviceType> deltas)
      : n(n), num_blocks(num_blocks), fixed_rate(fixed_rate),
        in_data(in_data), deltas(deltas) {}

  MGARDX_CONT Task<FixedUnpackFunctor<DeviceType>> GenTask(int queue_idx) {
    using F = FixedUnpackFunctor<DeviceType>;
    F functor(n, num_blocks, fixed_rate, in_data, deltas);
    return Task(functor, 1, 1, num_blocks, 1, 1, DELTA_BLOCK_SIZE, 0,
                queue_idx, std::string(Name));
  }

 private:
  SIZE n, num_blocks;
  uint8_t fixed_rate;
  SubArray<1, uint8_t, DeviceType> in_data;
  SubArray<1, int64_t, DeviceType> deltas;
};

// ══════════════════════════════════════════════════════════════════════════════
// KERNEL 7 — PlainUnpackFunctor
//   Decompress variable-rate (plain) blocks. Writes int64_t deltas.
// ══════════════════════════════════════════════════════════════════════════════
template <typename DeviceType>
class PlainUnpackFunctor : public Functor<DeviceType> {
 public:
  MGARDX_EXEC PlainUnpackFunctor() {}
  MGARDX_EXEC PlainUnpackFunctor(SIZE n, SIZE num_blocks,
                                 SubArray<1, uint8_t, DeviceType> in_data,
                                 SubArray<1, uint8_t, DeviceType> rates,
                                 SubArray<1, SIZE, DeviceType>    byte_offsets,
                                 SubArray<1, int64_t, DeviceType> deltas)
      : n(n), num_blocks(num_blocks), in_data(in_data), rates(rates),
        byte_offsets(byte_offsets), deltas(deltas) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    const SIZE blk  = FunctorBase<DeviceType>::GetBlockIdX();
    const int  lane = static_cast<int>(FunctorBase<DeviceType>::GetThreadIdX());
    if (blk >= num_blocks) return;

    const SIZE base      = blk * DELTA_BLOCK_SIZE;
    const uint8_t rate   = *rates(blk);

    if (!rate) return;  // d_deltas already zero-initialized

    const uint8_t *in = in_data((IDX)(*byte_offsets(blk)));

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    // ── GPU warp path ─────────────────────────────────────────────────────
    const uint32_t sign_mask = reinterpret_cast<const uint32_t *>(in)[0];
    in += 4;
    uint64_t abs_d = 0;
    for (int p = 0; p < static_cast<int>(rate); ++p) {
      uint32_t plane = reinterpret_cast<const uint32_t *>(in)[0];
      in += 4;
      if ((plane >> lane) & 1u) abs_d |= static_cast<uint64_t>(1) << p;
    }
    const SIZE idx = base + lane;
    if (idx < n) {
      bool neg = (sign_mask >> lane) & 1u;
      *deltas(idx) = neg ? -static_cast<int64_t>(abs_d)
                         :  static_cast<int64_t>(abs_d);
    }
#else
    // ── CPU sequential fallback ───────────────────────────────────────────
    if (lane != 0) return;
    const uint32_t sign_mask = reinterpret_cast<const uint32_t *>(in)[0];
    in += 4;
    uint64_t abs_ds[DELTA_BLOCK_SIZE] = {};
    for (int p = 0; p < static_cast<int>(rate); ++p) {
      uint32_t plane = reinterpret_cast<const uint32_t *>(in)[0];
      in += 4;
      for (int i = 0; i < static_cast<int>(DELTA_BLOCK_SIZE); ++i)
        if ((plane >> i) & 1u) abs_ds[i] |= static_cast<uint64_t>(1) << p;
    }
    for (int i = 0; i < static_cast<int>(DELTA_BLOCK_SIZE); ++i) {
      SIZE idx = base + i;
      if (idx >= n) break;
      bool neg = (sign_mask >> i) & 1u;
      *deltas(idx) = neg ? -static_cast<int64_t>(abs_ds[i])
                         :  static_cast<int64_t>(abs_ds[i]);
    }
#endif
  }
  MGARDX_CONT size_t shared_memory_size() { return 0; }

 private:
  SIZE n, num_blocks;
  SubArray<1, uint8_t, DeviceType> in_data;
  SubArray<1, uint8_t, DeviceType> rates;
  SubArray<1, SIZE, DeviceType>    byte_offsets;
  SubArray<1, int64_t, DeviceType> deltas;
};

template <typename DeviceType>
class PlainUnpackKernel : public Kernel {
 public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "delta_plain_unpack";

  MGARDX_CONT PlainUnpackKernel(SIZE n, SIZE num_blocks,
                                SubArray<1, uint8_t, DeviceType> in_data,
                                SubArray<1, uint8_t, DeviceType> rates,
                                SubArray<1, SIZE, DeviceType>    byte_offsets,
                                SubArray<1, int64_t, DeviceType> deltas)
      : n(n), num_blocks(num_blocks), in_data(in_data), rates(rates),
        byte_offsets(byte_offsets), deltas(deltas) {}

  MGARDX_CONT Task<PlainUnpackFunctor<DeviceType>> GenTask(int queue_idx) {
    using F = PlainUnpackFunctor<DeviceType>;
    F functor(n, num_blocks, in_data, rates, byte_offsets, deltas);
    return Task(functor, 1, 1, num_blocks, 1, 1, DELTA_BLOCK_SIZE, 0,
                queue_idx, std::string(Name));
  }

 private:
  SIZE n, num_blocks;
  SubArray<1, uint8_t, DeviceType> in_data;
  SubArray<1, uint8_t, DeviceType> rates;
  SubArray<1, SIZE, DeviceType>    byte_offsets;
  SubArray<1, int64_t, DeviceType> deltas;
};

// ══════════════════════════════════════════════════════════════════════════════
// KERNEL 8 — OutlierUnpackFunctor
//   Handles both plain blocks (rate_byte bit 7=0) and outlier blocks.
// ══════════════════════════════════════════════════════════════════════════════
template <typename DeviceType>
class OutlierUnpackFunctor : public Functor<DeviceType> {
 public:
  MGARDX_EXEC OutlierUnpackFunctor() {}
  MGARDX_EXEC OutlierUnpackFunctor(SIZE n, SIZE num_blocks,
                                   SubArray<1, uint8_t, DeviceType> in_data,
                                   SubArray<1, uint8_t, DeviceType> rate_bytes,
                                   SubArray<1, SIZE, DeviceType>    byte_offsets,
                                   SubArray<1, int64_t, DeviceType> deltas)
      : n(n), num_blocks(num_blocks), in_data(in_data),
        rate_bytes(rate_bytes), byte_offsets(byte_offsets), deltas(deltas) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    const SIZE blk  = FunctorBase<DeviceType>::GetBlockIdX();
    const int  lane = static_cast<int>(FunctorBase<DeviceType>::GetThreadIdX());
    if (blk >= num_blocks) return;

    const SIZE base    = blk * DELTA_BLOCK_SIZE;
    const uint8_t rb   = *rate_bytes(blk);
    const bool is_out  = (rb >> 7) & 1u;

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    // ── GPU warp path ─────────────────────────────────────────────────────
    if (!is_out) {
      // plain block
      const uint8_t rate = rb & 0x7Fu;
      if (!rate) return;  // d_deltas already zero-initialized
      const uint8_t *in = in_data((IDX)(*byte_offsets(blk)));
      const uint32_t sign_mask = reinterpret_cast<const uint32_t *>(in)[0];
      in += 4;
      uint64_t abs_d = 0;
      for (int p = 0; p < static_cast<int>(rate); ++p) {
        uint32_t plane = reinterpret_cast<const uint32_t *>(in)[0];
        in += 4;
        if ((plane >> lane) & 1u) abs_d |= static_cast<uint64_t>(1) << p;
      }
      const SIZE idx = base + lane;
      if (idx < n) {
        bool neg = (sign_mask >> lane) & 1u;
        *deltas(idx) = neg ? -static_cast<int64_t>(abs_d)
                           :  static_cast<int64_t>(abs_d);
      }
    } else {
      // outlier block: lane 0 is the outlier
      const int se     = (rb >> 5) & 0x3;
      const int ob     = outlier_size_dec(se);
      const uint8_t rate31 = rb & 0x1Fu;

      const uint8_t *in = in_data((IDX)(*byte_offsets(blk)));

      // Read outlier info (all lanes: lane 0 will use it, broadcast later)
      uint8_t info_byte = in[0];
      bool outlier_neg  = (info_byte >> 7) & 1u;
      uint64_t abs_out  = 0;
      for (int b = 0; b < ob; ++b)
        abs_out |= static_cast<uint64_t>(in[1 + b]) << (8 * b);
      in += 1 + ob;

      // Sign and planes for lanes 1–31
      const uint32_t sign_mask = reinterpret_cast<const uint32_t *>(in)[0];
      in += 4;
      uint64_t abs_d = 0;
      for (int p = 0; p < static_cast<int>(rate31); ++p) {
        uint32_t plane = reinterpret_cast<const uint32_t *>(in)[0];
        in += 4;
        if ((plane >> lane) & 1u) abs_d |= static_cast<uint64_t>(1) << p;
      }

      if (lane == 0) {
        // Outlier delta
        const SIZE idx = base + 0;
        if (idx < n)
          *deltas(idx) = outlier_neg ? -static_cast<int64_t>(abs_out)
                                     :  static_cast<int64_t>(abs_out);
      } else {
        const SIZE idx = base + lane;
        if (idx < n) {
          bool neg = (sign_mask >> lane) & 1u;
          *deltas(idx) = neg ? -static_cast<int64_t>(abs_d)
                             :  static_cast<int64_t>(abs_d);
        }
      }
    }
#else
    // ── CPU sequential fallback ───────────────────────────────────────────
    if (lane != 0) return;
    if (!is_out) {
      const uint8_t rate = rb & 0x7Fu;
      if (!rate) return;  // d_deltas already zero-initialized
      const uint8_t *in = in_data((IDX)(*byte_offsets(blk)));
      const uint32_t sign_mask = reinterpret_cast<const uint32_t *>(in)[0];
      in += 4;
      uint64_t abs_ds[DELTA_BLOCK_SIZE] = {};
      for (int p = 0; p < static_cast<int>(rate); ++p) {
        uint32_t plane = reinterpret_cast<const uint32_t *>(in)[0];
        in += 4;
        for (int i = 0; i < static_cast<int>(DELTA_BLOCK_SIZE); ++i)
          if ((plane >> i) & 1u) abs_ds[i] |= static_cast<uint64_t>(1) << p;
      }
      for (int i = 0; i < static_cast<int>(DELTA_BLOCK_SIZE); ++i) {
        SIZE idx = base + i;
        if (idx >= n) break;
        bool neg = (sign_mask >> i) & 1u;
        *deltas(idx) = neg ? -static_cast<int64_t>(abs_ds[i])
                           :  static_cast<int64_t>(abs_ds[i]);
      }
    } else {
      const int se     = (rb >> 5) & 0x3;
      const int ob     = outlier_size_dec(se);
      const uint8_t rate31 = rb & 0x1Fu;
      const uint8_t *in = in_data((IDX)(*byte_offsets(blk)));
      uint8_t info_byte = in[0];
      bool outlier_neg  = (info_byte >> 7) & 1u;
      uint64_t abs_out  = 0;
      for (int b = 0; b < ob; ++b)
        abs_out |= static_cast<uint64_t>(in[1 + b]) << (8 * b);
      in += 1 + ob;
      // lane 0 is outlier
      if (base + 0 < n)
        *deltas(base + 0) = outlier_neg ? -static_cast<int64_t>(abs_out)
                                        :  static_cast<int64_t>(abs_out);
      const uint32_t sign_mask = reinterpret_cast<const uint32_t *>(in)[0];
      in += 4;
      uint64_t abs_ds[DELTA_BLOCK_SIZE] = {};
      for (int p = 0; p < static_cast<int>(rate31); ++p) {
        uint32_t plane = reinterpret_cast<const uint32_t *>(in)[0];
        in += 4;
        for (int i = 1; i < static_cast<int>(DELTA_BLOCK_SIZE); ++i)
          if ((plane >> i) & 1u) abs_ds[i] |= static_cast<uint64_t>(1) << p;
      }
      for (int i = 1; i < static_cast<int>(DELTA_BLOCK_SIZE); ++i) {
        SIZE idx = base + i;
        if (idx >= n) break;
        bool neg = (sign_mask >> i) & 1u;
        *deltas(idx) = neg ? -static_cast<int64_t>(abs_ds[i])
                           :  static_cast<int64_t>(abs_ds[i]);
      }
    }
#endif
  }
  MGARDX_CONT size_t shared_memory_size() { return 0; }

 private:
  SIZE n, num_blocks;
  SubArray<1, uint8_t, DeviceType> in_data;
  SubArray<1, uint8_t, DeviceType> rate_bytes;
  SubArray<1, SIZE, DeviceType>    byte_offsets;
  SubArray<1, int64_t, DeviceType> deltas;
};

template <typename DeviceType>
class OutlierUnpackKernel : public Kernel {
 public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "delta_outlier_unpack";

  MGARDX_CONT OutlierUnpackKernel(SIZE n, SIZE num_blocks,
                                  SubArray<1, uint8_t, DeviceType> in_data,
                                  SubArray<1, uint8_t, DeviceType> rate_bytes,
                                  SubArray<1, SIZE, DeviceType>    byte_offsets,
                                  SubArray<1, int64_t, DeviceType> deltas)
      : n(n), num_blocks(num_blocks), in_data(in_data),
        rate_bytes(rate_bytes), byte_offsets(byte_offsets), deltas(deltas) {}

  MGARDX_CONT Task<OutlierUnpackFunctor<DeviceType>> GenTask(int queue_idx) {
    using F = OutlierUnpackFunctor<DeviceType>;
    F functor(n, num_blocks, in_data, rate_bytes, byte_offsets, deltas);
    return Task(functor, 1, 1, num_blocks, 1, 1, DELTA_BLOCK_SIZE, 0,
                queue_idx, std::string(Name));
  }

 private:
  SIZE n, num_blocks;
  SubArray<1, uint8_t, DeviceType> in_data;
  SubArray<1, uint8_t, DeviceType> rate_bytes;
  SubArray<1, SIZE, DeviceType>    byte_offsets;
  SubArray<1, int64_t, DeviceType> deltas;
};

// ══════════════════════════════════════════════════════════════════════════════
// KERNEL 9 — ReconstructLocalFunctor
//   1 thread per local spatial block.
//   Sequentially prefix-sums deltas[] in-place within each block,
//   writing reconstructed quantized values to output[].
// ══════════════════════════════════════════════════════════════════════════════
template <typename T, typename DeviceType>
class ReconstructLocalFunctor : public Functor<DeviceType> {
 public:
  MGARDX_EXEC ReconstructLocalFunctor() {}
  MGARDX_EXEC ReconstructLocalFunctor(SIZE n, SIZE global_size,
                                      SIZE local_block_stride,
                                      SubArray<1, int64_t, DeviceType> deltas,
                                      SubArray<1, T, DeviceType>       output)
      : n(n), global_size(global_size),
        local_block_stride(local_block_stride), deltas(deltas),
        output(output) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    const SIZE local_n = (global_size < n) ? n - global_size : 0;
    if (!local_n || !local_block_stride) return;
    const SIZE num_local_blks =
        (local_n + local_block_stride - 1) / local_block_stride;
    const SIZE lb = FunctorBase<DeviceType>::GetBlockIdX() *
                        FunctorBase<DeviceType>::GetBlockDimX() +
                    FunctorBase<DeviceType>::GetThreadIdX();
    if (lb >= num_local_blks) return;

    const SIZE start = global_size + lb * local_block_stride;
    const SIZE end   = (start + local_block_stride < n)
                       ? start + local_block_stride : n;
    int64_t running = 0;
    for (SIZE idx = start; idx < end; ++idx) {
      running += *deltas(idx);
      *output(idx) = static_cast<T>(running);
    }
  }
  MGARDX_CONT size_t shared_memory_size() { return 0; }

 private:
  SIZE n, global_size, local_block_stride;
  SubArray<1, int64_t, DeviceType> deltas;
  SubArray<1, T, DeviceType>       output;
};

template <typename T, typename DeviceType>
class ReconstructLocalKernel : public Kernel {
 public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "delta_reconstruct_local";

  MGARDX_CONT ReconstructLocalKernel(SIZE n, SIZE global_size,
                                     SIZE local_block_stride,
                                     SubArray<1, int64_t, DeviceType> deltas,
                                     SubArray<1, T, DeviceType>       output)
      : n(n), global_size(global_size),
        local_block_stride(local_block_stride), deltas(deltas),
        output(output) {}

  MGARDX_CONT Task<ReconstructLocalFunctor<T, DeviceType>> GenTask(
      int queue_idx) {
    using F = ReconstructLocalFunctor<T, DeviceType>;
    F functor(n, global_size, local_block_stride, deltas, output);
    const SIZE local_n = (global_size < n) ? n - global_size : 0;
    const SIZE num_local_blks = (!local_n || !local_block_stride)
        ? 1 : (local_n + local_block_stride - 1) / local_block_stride;
    const SIZE tbx   = 256;
    const SIZE gridx = (num_local_blks + tbx - 1) / tbx;
    return Task(functor, 1, 1, gridx, 1, 1, tbx, 0, queue_idx,
                std::string(Name));
  }

 private:
  SIZE n, global_size, local_block_stride;
  SubArray<1, int64_t, DeviceType> deltas;
  SubArray<1, T, DeviceType>       output;
};

// KERNEL 10 — CopyGlobalFunctor: copy int64_t → T for global segment
template <typename T, typename DeviceType>
class CopyGlobalFunctor : public Functor<DeviceType> {
 public:
  MGARDX_EXEC CopyGlobalFunctor() {}
  MGARDX_EXEC CopyGlobalFunctor(SIZE global_size,
                                SubArray<1, int64_t, DeviceType> values,
                                SubArray<1, T, DeviceType>       output)
      : global_size(global_size), values(values), output(output) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    const SIZE idx = FunctorBase<DeviceType>::GetBlockIdX() *
                         FunctorBase<DeviceType>::GetBlockDimX() +
                     FunctorBase<DeviceType>::GetThreadIdX();
    if (idx >= global_size) return;
    *output(idx) = static_cast<T>(*values(idx));
  }
  MGARDX_CONT size_t shared_memory_size() { return 0; }

 private:
  SIZE global_size;
  SubArray<1, int64_t, DeviceType> values;
  SubArray<1, T, DeviceType>       output;
};

template <typename T, typename DeviceType>
class CopyGlobalKernel : public Kernel {
 public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "delta_copy_global";

  MGARDX_CONT CopyGlobalKernel(SIZE global_size,
                               SubArray<1, int64_t, DeviceType> values,
                               SubArray<1, T, DeviceType>       output)
      : global_size(global_size), values(values), output(output) {}

  MGARDX_CONT Task<CopyGlobalFunctor<T, DeviceType>> GenTask(int queue_idx) {
    using F = CopyGlobalFunctor<T, DeviceType>;
    F functor(global_size, values, output);
    const SIZE tbx   = 256;
    const SIZE gridx = global_size ? (global_size + tbx - 1) / tbx : 1;
    return Task(functor, 1, 1, gridx, 1, 1, tbx, 0, queue_idx,
                std::string(Name));
  }

 private:
  SIZE global_size;
  SubArray<1, int64_t, DeviceType> values;
  SubArray<1, T, DeviceType>       output;
};

// ══════════════════════════════════════════════════════════════════════════════
// Internal helper: run scan + return total bytes; allocate + fill rate metadata
// ══════════════════════════════════════════════════════════════════════════════
template <typename DeviceType>
static SIZE compute_offsets_and_total(
    SIZE num_blocks,
    SubArray<1, SIZE, DeviceType>    byte_sizes,
    Array<1, SIZE, DeviceType>      &byte_offsets_arr,
    int queue_idx) {
  byte_offsets_arr = Array<1, SIZE, DeviceType>({num_blocks + 1});
  Array<1, Byte, DeviceType> scan_ws;
  DeviceCollective<DeviceType>::ScanSumExtended(
      num_blocks, byte_sizes, SubArray(byte_offsets_arr),
      scan_ws, false, queue_idx);

  SIZE total = 0;
  MemoryManager<DeviceType>::Copy1D(
      &total,
      byte_offsets_arr.data() + num_blocks,
      1, queue_idx);
  DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
  return total;
}

// ══════════════════════════════════════════════════════════════════════════════
// Internal helper: run shared reconstruct after any unpack
// ══════════════════════════════════════════════════════════════════════════════
template <typename T, typename DeviceType>
static void reconstruct_from_deltas(
    SIZE n, SIZE global_size, SIZE local_block_stride,
    Array<1, int64_t, DeviceType>  &d_deltas,
    Array<1, T, DeviceType>        &output,
    int queue_idx) {
  // Global segment: inclusive prefix sum of deltas[0..global_size)
  if (global_size > 0) {
    Array<1, int64_t, DeviceType> global_scan({global_size});
    SubArray<1, int64_t, DeviceType> gsub({global_size}, d_deltas.data());
    Array<1, Byte, DeviceType> ws;
    DeviceCollective<DeviceType>::ScanSumInclusive(
        global_size, gsub, SubArray(global_scan), ws, false, queue_idx);
    MemoryManager<DeviceType>::Copy1D(
        d_deltas.data(), global_scan.data(), global_size, queue_idx);
    DeviceLauncher<DeviceType>::Execute(
        CopyGlobalKernel<T, DeviceType>(
            global_size, SubArray(d_deltas), SubArray(output)),
        queue_idx);
  }
  // Local segments: sequential prefix sum per block
  if (global_size < n) {
    DeviceLauncher<DeviceType>::Execute(
        ReconstructLocalKernel<T, DeviceType>(
            n, global_size, local_block_stride,
            SubArray(d_deltas), SubArray(output)),
        queue_idx);
  }
}

// ══════════════════════════════════════════════════════════════════════════════
// MEMORY FOOTPRINT ESTIMATES
//   Conservative upper-bounds on peak device memory across compress+decompress.
//   Call with sizeof(T) matching the quantized-integer type (e.g. sizeof(int)).
// ══════════════════════════════════════════════════════════════════════════════

// Fixed-rate: every block uses rate_max bits (worst case rate=64 for uint64_t)
template <typename T>
static size_t DeltaEncodingEstimateMemoryFootprintFixed(SIZE n) {
  const SIZE num_blocks = (n + DELTA_BLOCK_SIZE - 1) / DELTA_BLOCK_SIZE;
  // compress: d_rates + worst-case output (rate=64 → 4+64*4=260 bytes/block)
  const size_t compress_peak =
      static_cast<size_t>(num_blocks) * sizeof(uint8_t) +
      sizeof(DeltaHeader) +
      static_cast<size_t>(num_blocks) * (4 + 64 * 4);
  // decompress: output + d_deltas + d_rate_bytes + d_byte_offsets
  const size_t decompress_peak =
      static_cast<size_t>(n) * sizeof(T) +
      static_cast<size_t>(n) * sizeof(int64_t) +
      static_cast<size_t>(num_blocks) * sizeof(uint8_t) +
      static_cast<size_t>(num_blocks + 1) * sizeof(SIZE);
  return std::max(compress_peak, decompress_peak);
}

// Plain (variable-rate): per-block rate stored; data bytes vary by block
template <typename T>
static size_t DeltaEncodingEstimateMemoryFootprintPlain(SIZE n) {
  const SIZE num_blocks = (n + DELTA_BLOCK_SIZE - 1) / DELTA_BLOCK_SIZE;
  const SIZE rate_arr_bytes = (num_blocks + 3) / 4 * 4;
  // compress: d_rates + d_byte_offsets + worst-case output
  const size_t compress_peak =
      static_cast<size_t>(num_blocks) * sizeof(uint8_t) +
      static_cast<size_t>(num_blocks + 1) * sizeof(SIZE) +
      sizeof(DeltaHeader) + rate_arr_bytes +
      static_cast<size_t>(num_blocks) * (4 + 64 * 4);
  // decompress: output + d_deltas + d_rate_bytes + d_byte_offsets
  const size_t decompress_peak =
      static_cast<size_t>(n) * sizeof(T) +
      static_cast<size_t>(n) * sizeof(int64_t) +
      static_cast<size_t>(num_blocks) * sizeof(uint8_t) +
      static_cast<size_t>(num_blocks + 1) * sizeof(SIZE);
  return std::max(compress_peak, decompress_peak);
}

// Outlier-aware: rate byte encodes outlier flag + outlier size + residual rate
template <typename T>
static size_t DeltaEncodingEstimateMemoryFootprintOutlier(SIZE n) {
  const SIZE num_blocks = (n + DELTA_BLOCK_SIZE - 1) / DELTA_BLOCK_SIZE;
  const SIZE rate_arr_bytes = (num_blocks + 3) / 4 * 4;
  // compress: d_rate_bytes + d_byte_sizes + d_byte_offsets + worst-case output
  // worst-case outlier block: 1(flag) + 8(outlier_val) + 4(signs) + 31*4(planes) = 137 bytes
  const size_t compress_peak =
      static_cast<size_t>(num_blocks) * sizeof(uint8_t) +
      static_cast<size_t>(num_blocks) * sizeof(SIZE) +   // d_byte_sizes
      static_cast<size_t>(num_blocks + 1) * sizeof(SIZE) + // d_byte_offsets
      sizeof(DeltaHeader) + rate_arr_bytes +
      static_cast<size_t>(num_blocks) * (1 + 8 + 4 + 31 * 4);
  // decompress: output + d_deltas + d_rate_bytes + d_byte_offsets
  const size_t decompress_peak =
      static_cast<size_t>(n) * sizeof(T) +
      static_cast<size_t>(n) * sizeof(int64_t) +
      static_cast<size_t>(num_blocks) * sizeof(uint8_t) +
      static_cast<size_t>(num_blocks + 1) * sizeof(SIZE);
  return std::max(compress_peak, decompress_peak);
}

// ══════════════════════════════════════════════════════════════════════════════
// PUBLIC API — DeltaEncodingCompressFixed
// ══════════════════════════════════════════════════════════════════════════════
template <typename T, typename DeviceType>
Array<1, Byte, DeviceType>
DeltaEncodingCompressFixed(SubArray<1, T, DeviceType> &input_data,
                           SIZE global_size, SIZE local_block_stride,
                           int queue_idx) {
  Timer timer;
  if (log::level & log::TIME) timer.start();

  const SIZE n          = input_data.shape(0);
  const SIZE num_blocks = (n + DELTA_BLOCK_SIZE - 1) / DELTA_BLOCK_SIZE;

  // Step 1: per-block rates
  Array<1, uint8_t, DeviceType> d_rates({num_blocks});
  d_rates.memset(0);
  DeviceLauncher<DeviceType>::Execute(
      PlainRateKernel<T, DeviceType>(
          n, global_size, local_block_stride, input_data, SubArray(d_rates)),
      queue_idx);

  // Step 2: find max rate on host → fixed_rate
  std::vector<uint8_t> h_rates(num_blocks);
  MemoryManager<DeviceType>::Copy1D(
      h_rates.data(), d_rates.data(), num_blocks, queue_idx);
  DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
  const uint8_t fixed_rate = *std::max_element(h_rates.begin(), h_rates.end());

  // Step 3: build output
  const SIZE block_bytes = fixed_rate ? static_cast<SIZE>(4 + fixed_rate * 4) : 0;
  const SIZE data_bytes  = num_blocks * block_bytes;
  const SIZE total_size  = sizeof(DeltaHeader) + data_bytes;

  Array<1, Byte, DeviceType> output({static_cast<SIZE>(total_size)});
  output.memset(0);

  DeltaHeader hdr{};
  hdr.n                  = static_cast<size_t>(n);
  hdr.global_size        = static_cast<size_t>(global_size);
  hdr.local_block_stride = static_cast<size_t>(local_block_stride);
  hdr.mode               = static_cast<uint8_t>(DeltaMode::Fixed);
  hdr.fixed_rate         = fixed_rate;
  MemoryManager<DeviceType>::Copy1D(
      reinterpret_cast<Byte *>(output.data()),
      reinterpret_cast<Byte *>(&hdr),
      sizeof(DeltaHeader), queue_idx);

  // Step 4: pack
  if (data_bytes > 0 && fixed_rate > 0) {
    SubArray<1, uint8_t, DeviceType> out_data(
        {data_bytes},
        reinterpret_cast<uint8_t *>(output.data()) + sizeof(DeltaHeader));
    DeviceLauncher<DeviceType>::Execute(
        FixedPackKernel<T, DeviceType>(
            n, num_blocks, global_size, local_block_stride,
            fixed_rate, input_data, out_data),
        queue_idx);
  }

  log::info("DeltaEncoding Fixed: " + std::to_string(n * sizeof(T)) + "/" +
            std::to_string(total_size) + " (" +
            std::to_string(static_cast<double>(n * sizeof(T)) / total_size) +
            "x)");
  if (log::level & log::TIME) {
    DeviceRuntime<DeviceType>::SyncDevice();
    timer.end();
    timer.print("DeltaEncoding Fixed compress");
    timer.clear();
  }
  return output;
}

// ══════════════════════════════════════════════════════════════════════════════
// PUBLIC API — DeltaEncodingCompressPlain
// ══════════════════════════════════════════════════════════════════════════════
template <typename T, typename DeviceType>
Array<1, Byte, DeviceType>
DeltaEncodingCompressPlain(SubArray<1, T, DeviceType> &input_data,
                           SIZE global_size, SIZE local_block_stride,
                           int queue_idx) {
  Timer timer;
  if (log::level & log::TIME) timer.start();

  const SIZE n          = input_data.shape(0);
  const SIZE num_blocks = (n + DELTA_BLOCK_SIZE - 1) / DELTA_BLOCK_SIZE;

  // Step 1: per-block rates
  Array<1, uint8_t, DeviceType> d_rates({num_blocks});
  d_rates.memset(0);
  DeviceLauncher<DeviceType>::Execute(
      PlainRateKernel<T, DeviceType>(
          n, global_size, local_block_stride, input_data, SubArray(d_rates)),
      queue_idx);

  // Step 2: byte offsets via host prefix sum (rates already on host)
  std::vector<uint8_t> h_rates(num_blocks);
  MemoryManager<DeviceType>::Copy1D(
      h_rates.data(), d_rates.data(), num_blocks, queue_idx);
  DeviceRuntime<DeviceType>::SyncQueue(queue_idx);

  std::vector<SIZE> h_byte_offsets(num_blocks + 1);
  h_byte_offsets[0] = 0;
  for (SIZE b = 0; b < num_blocks; ++b) {
    const SIZE bs = h_rates[b]
        ? static_cast<SIZE>(4 + h_rates[b] * 4) : 0;
    h_byte_offsets[b + 1] = h_byte_offsets[b] + bs;
  }
  const SIZE data_bytes = h_byte_offsets[num_blocks];

  Array<1, SIZE, DeviceType> d_byte_offsets({num_blocks + 1});
  MemoryManager<DeviceType>::Copy1D(
      d_byte_offsets.data(), h_byte_offsets.data(), num_blocks + 1, queue_idx);

  // Step 4: build output buffer
  const SIZE rate_arr_bytes = (num_blocks + 3) / 4 * 4;
  const SIZE total_size     = sizeof(DeltaHeader) + rate_arr_bytes + data_bytes;

  Array<1, Byte, DeviceType> output({static_cast<SIZE>(total_size)});
  output.memset(0);

  DeltaHeader hdr{};
  hdr.n                  = static_cast<size_t>(n);
  hdr.global_size        = static_cast<size_t>(global_size);
  hdr.local_block_stride = static_cast<size_t>(local_block_stride);
  hdr.mode               = static_cast<uint8_t>(DeltaMode::Plain);
  hdr.fixed_rate         = 0;
  MemoryManager<DeviceType>::Copy1D(
      reinterpret_cast<Byte *>(output.data()),
      reinterpret_cast<Byte *>(&hdr), sizeof(DeltaHeader), queue_idx);

  // Copy rates array after header
  MemoryManager<DeviceType>::Copy1D(
      reinterpret_cast<Byte *>(output.data()) + sizeof(DeltaHeader),
      reinterpret_cast<Byte *>(d_rates.data()),
      num_blocks, queue_idx);

  // Step 5: pack sign + bit-planes
  if (data_bytes > 0) {
    SubArray<1, uint8_t, DeviceType> out_data(
        {data_bytes},
        reinterpret_cast<uint8_t *>(output.data()) +
        sizeof(DeltaHeader) + rate_arr_bytes);
    DeviceLauncher<DeviceType>::Execute(
        PlainPackKernel<T, DeviceType>(
            n, num_blocks, global_size, local_block_stride,
            input_data, SubArray(d_rates),
            SubArray(d_byte_offsets), out_data),
        queue_idx);
  }

  log::info("DeltaEncoding Plain: " + std::to_string(n * sizeof(T)) + "/" +
            std::to_string(total_size) + " (" +
            std::to_string(static_cast<double>(n * sizeof(T)) / total_size) +
            "x)");
  if (log::level & log::TIME) {
    DeviceRuntime<DeviceType>::SyncDevice();
    timer.end();
    timer.print("DeltaEncoding Plain compress");
    timer.clear();
  }
  return output;
}

// ══════════════════════════════════════════════════════════════════════════════
// PUBLIC API — DeltaEncodingCompressOutlier
// ══════════════════════════════════════════════════════════════════════════════
template <typename T, typename DeviceType>
Array<1, Byte, DeviceType>
DeltaEncodingCompressOutlier(SubArray<1, T, DeviceType> &input_data,
                             SIZE global_size, SIZE local_block_stride,
                             int queue_idx) {
  Timer timer;
  if (log::level & log::TIME) timer.start();

  const SIZE n          = input_data.shape(0);
  const SIZE num_blocks = (n + DELTA_BLOCK_SIZE - 1) / DELTA_BLOCK_SIZE;

  // Step 1: outlier-aware rate decision
  Array<1, uint8_t, DeviceType> d_rate_bytes({num_blocks});
  Array<1, SIZE, DeviceType>    d_byte_sizes({num_blocks});
  d_rate_bytes.memset(0);
  d_byte_sizes.memset(0);
  DeviceLauncher<DeviceType>::Execute(
      OutlierRateKernel<T, DeviceType>(
          n, global_size, local_block_stride, input_data,
          SubArray(d_rate_bytes), SubArray(d_byte_sizes)),
      queue_idx);

  // Step 2: exclusive scan → byte offsets + total
  Array<1, SIZE, DeviceType> d_byte_offsets;
  const SIZE data_bytes = compute_offsets_and_total<DeviceType>(
      num_blocks, SubArray(d_byte_sizes), d_byte_offsets, queue_idx);

  // Step 3: build output
  const SIZE rate_arr_bytes = (num_blocks + 3) / 4 * 4;
  const SIZE total_size     = sizeof(DeltaHeader) + rate_arr_bytes + data_bytes;

  Array<1, Byte, DeviceType> output({static_cast<SIZE>(total_size)});
  output.memset(0);

  DeltaHeader hdr{};
  hdr.n                  = static_cast<size_t>(n);
  hdr.global_size        = static_cast<size_t>(global_size);
  hdr.local_block_stride = static_cast<size_t>(local_block_stride);
  hdr.mode               = static_cast<uint8_t>(DeltaMode::Outlier);
  hdr.fixed_rate         = 0;
  MemoryManager<DeviceType>::Copy1D(
      reinterpret_cast<Byte *>(output.data()),
      reinterpret_cast<Byte *>(&hdr), sizeof(DeltaHeader), queue_idx);

  MemoryManager<DeviceType>::Copy1D(
      reinterpret_cast<Byte *>(output.data()) + sizeof(DeltaHeader),
      reinterpret_cast<Byte *>(d_rate_bytes.data()),
      num_blocks, queue_idx);

  // Step 4: pack
  if (data_bytes > 0) {
    SubArray<1, uint8_t, DeviceType> out_data(
        {data_bytes},
        reinterpret_cast<uint8_t *>(output.data()) +
        sizeof(DeltaHeader) + rate_arr_bytes);
    DeviceLauncher<DeviceType>::Execute(
        OutlierPackKernel<T, DeviceType>(
            n, num_blocks, global_size, local_block_stride,
            input_data, SubArray(d_rate_bytes),
            SubArray(d_byte_offsets), out_data),
        queue_idx);
  }

  log::info("DeltaEncoding Outlier: " + std::to_string(n * sizeof(T)) + "/" +
            std::to_string(total_size) + " (" +
            std::to_string(static_cast<double>(n * sizeof(T)) / total_size) +
            "x)");
  if (log::level & log::TIME) {
    DeviceRuntime<DeviceType>::SyncDevice();
    timer.end();
    timer.print("DeltaEncoding Outlier compress");
    timer.clear();
  }
  return output;
}

// ══════════════════════════════════════════════════════════════════════════════
// PUBLIC API — DeltaEncodingDecompress (unified; reads mode from header)
// ══════════════════════════════════════════════════════════════════════════════
template <typename T, typename DeviceType>
Array<1, T, DeviceType>
DeltaEncodingDecompress(SubArray<1, Byte, DeviceType> &input_data,
                        int queue_idx) {
  Timer timer;
  if (log::level & log::TIME) timer.start();

  // Read header
  DeltaHeader hdr{};
  MemoryManager<DeviceType>::Copy1D(
      reinterpret_cast<Byte *>(&hdr),
      input_data.data(),
      sizeof(DeltaHeader), queue_idx);
  DeviceRuntime<DeviceType>::SyncQueue(queue_idx);

  const SIZE n                  = static_cast<SIZE>(hdr.n);
  const SIZE global_size        = static_cast<SIZE>(hdr.global_size);
  const SIZE local_block_stride = static_cast<SIZE>(hdr.local_block_stride);
  const DeltaMode mode          = static_cast<DeltaMode>(hdr.mode);
  const uint8_t fixed_rate      = hdr.fixed_rate;
  const SIZE num_blocks         = (n + DELTA_BLOCK_SIZE - 1) / DELTA_BLOCK_SIZE;

  // Output and delta arrays
  Array<1, T, DeviceType>      output({n});
  Array<1, int64_t, DeviceType> d_deltas({n});
  d_deltas.memset(0);

  if (mode == DeltaMode::Fixed) {
    // Block data starts right after header; fixed stride
    const SIZE data_offset = sizeof(DeltaHeader);
    SubArray<1, uint8_t, DeviceType> in_data(
        {input_data.shape(0) - data_offset},
        reinterpret_cast<uint8_t *>(input_data.data()) + data_offset);

    DeviceLauncher<DeviceType>::Execute(
        FixedUnpackKernel<DeviceType>(
            n, num_blocks, fixed_rate, in_data, SubArray(d_deltas)),
        queue_idx);
  } else {
    // Variable-rate (Plain or Outlier): read rate metadata, scan, unpack
    const SIZE rate_arr_bytes  = (num_blocks + 3) / 4 * 4;
    const SIZE rates_offset    = sizeof(DeltaHeader);
    const SIZE data_offset     = rates_offset + rate_arr_bytes;

    Array<1, uint8_t, DeviceType> d_rate_bytes({num_blocks});
    MemoryManager<DeviceType>::Copy1D(
        d_rate_bytes.data(),
        reinterpret_cast<uint8_t *>(input_data.data()) + rates_offset,
        num_blocks, queue_idx);

    // Recompute byte sizes from rate bytes
    std::vector<uint8_t> h_rate_bytes(num_blocks);
    MemoryManager<DeviceType>::Copy1D(
        h_rate_bytes.data(), d_rate_bytes.data(), num_blocks, queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);

    std::vector<SIZE> h_byte_sizes(num_blocks);
    for (SIZE b = 0; b < num_blocks; ++b) {
      const uint8_t rb = h_rate_bytes[b];
      if (mode == DeltaMode::Plain) {
        const uint8_t rate = rb & 0x7Fu;
        h_byte_sizes[b] = rate ? static_cast<SIZE>(4 + rate * 4) : 0;
      } else {  // Outlier
        const bool is_out = (rb >> 7) & 1u;
        if (!is_out) {
          const uint8_t rate = rb & 0x7Fu;
          h_byte_sizes[b] = rate ? static_cast<SIZE>(4 + rate * 4) : 0;
        } else {
          const int se   = (rb >> 5) & 0x3;
          const int ob   = outlier_size_dec(se);
          const uint8_t r31 = rb & 0x1Fu;
          h_byte_sizes[b] = static_cast<SIZE>(1 + ob + 4 + r31 * 4);
        }
      }
    }

    std::vector<SIZE> h_byte_offsets(num_blocks + 1);
    h_byte_offsets[0] = 0;
    for (SIZE b = 0; b < num_blocks; ++b)
      h_byte_offsets[b + 1] = h_byte_offsets[b] + h_byte_sizes[b];

    Array<1, SIZE, DeviceType> d_byte_offsets({num_blocks + 1});
    MemoryManager<DeviceType>::Copy1D(
        d_byte_offsets.data(), h_byte_offsets.data(), num_blocks + 1,
        queue_idx);

    SubArray<1, uint8_t, DeviceType> in_body(
        {input_data.shape(0) - data_offset},
        reinterpret_cast<uint8_t *>(input_data.data()) + data_offset);

    if (mode == DeltaMode::Plain) {
      DeviceLauncher<DeviceType>::Execute(
          PlainUnpackKernel<DeviceType>(
              n, num_blocks, in_body,
              SubArray(d_rate_bytes), SubArray(d_byte_offsets),
              SubArray(d_deltas)),
          queue_idx);
    } else {
      DeviceLauncher<DeviceType>::Execute(
          OutlierUnpackKernel<DeviceType>(
              n, num_blocks, in_body,
              SubArray(d_rate_bytes), SubArray(d_byte_offsets),
              SubArray(d_deltas)),
          queue_idx);
    }
  }

  // Reconstruct: prefix-sum deltas → original quantized values
  reconstruct_from_deltas<T, DeviceType>(
      n, global_size, local_block_stride, d_deltas, output, queue_idx);

  if (log::level & log::TIME) {
    DeviceRuntime<DeviceType>::SyncDevice();
    timer.end();
    timer.print("DeltaEncoding decompress");
    timer.clear();
  }
  return output;
}

} // namespace mgard_x

#endif // MGARD_X_DELTA_ENCODING_TEMPLATE_HPP
