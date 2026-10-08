/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: October 8, 2026
 */

#ifndef MGARD_X_HIERARCHICAL_BLOCK_KERNEL_TEMPLATE
#define MGARD_X_HIERARCHICAL_BLOCK_KERNEL_TEMPLATE

#include "../../RuntimeX/RuntimeX.h"
#include <cstring>
#include <type_traits>

namespace mgard_x {

namespace data_refactoring {

namespace in_cache_block {

// Hierarchical-basis (no orthogonal projection) 8x8x8 and 8x8 block
// transforms: the same coarse values and coefficients, bit for bit, as
// Decompose8x8x8Functor / Recompose8x8x8Functor (Decompose8x8Functor /
// Recompose8x8Functor) with orthogonal_projection = false, which then reduce
// to a stencil: every coefficient is its node minus the multilinear
// interpolation of the coarse corners around it. A thread block takes BLOCKS
// consecutive blocks along x, one thread per (y, x) node (column of 8 nodes
// in 3D), so that the input/output rows and the per-block coefficient runs
// are accessed contiguously.
//
// Per dimension, nodes {0, 2, 4, 6, 7} are coarse and {1, 3, 5} carry
// coefficients. A block's coefficients (387 in 3D, 39 in 2D) are stored in
// lexicographic order of their nodes, its coarse values at the block's
// position of the coarse grid.
namespace hierarchical_block {

static constexpr int BLOCKS = 4;          // 8x8x8 blocks per thread block
static constexpr int WIDTH = 8 * BLOCKS;  // nodes along x per thread block
static constexpr int COEFFS = 387;        // coefficients per 3D block
static constexpr int COEFFS_2D = 39;      // coefficients per 2D block
static constexpr int CWIDTH = 5 * BLOCKS; // coarse nodes along x

MGARDX_EXEC constexpr bool is_coarse(int p) { return p % 2 == 0 || p == 7; }
// Coarse nodes before p (its coarse index when p is coarse).
MGARDX_EXEC constexpr int coarse_before(int p) { return (p + 1) >> 1; }
// Position of the coefficient of node (z, y, x) within its block.
MGARDX_EXEC constexpr int coeff_rank(int z, int y, int x) {
  return z * 64 + y * 8 + x -
         (coarse_before(z) * 25 +
          (is_coarse(z) ? coarse_before(y) * 5 +
                              (is_coarse(y) ? coarse_before(x) : 0)
                        : 0));
}
// Staged coarse grid of a thread block: [5][5][CWIDTH].
MGARDX_EXEC constexpr int coarse_slot(int z, int y, int b, int x) {
  return coarse_before(z) * 5 * CWIDTH + coarse_before(y) * CWIDTH + b * 5 +
         coarse_before(x);
}

// 2D counterparts (a 2D block is the z = 0 plane of a 3D one).
MGARDX_EXEC constexpr int coeff_rank_2d(int y, int x) {
  return y * 8 + x - (coarse_before(y) * 5 + (is_coarse(y) ? coarse_before(x) : 0));
}
MGARDX_EXEC constexpr int coarse_slot_2d(int y, int b, int x) {
  return coarse_before(y) * CWIDTH + b * 5 + coarse_before(x);
}

// Interpolation at coefficient node (z, y, x) from the coarse values around
// it, V(dz, dy, dx); the coefficient is the node minus it. Same operations,
// in the same order, as the reference functors.
template <typename T, typename Value>
MGARDX_EXEC T interpolation(int z, int y, int x, Value V) {
  const bool zf = !is_coarse(z), yf = !is_coarse(y), xf = !is_coarse(x);
  const int n = zf + yf + xf;
  if (n == 1) {
    T left = zf ? V(-1, 0, 0) : yf ? V(0, -1, 0) : V(0, 0, -1);
    T right = zf ? V(1, 0, 0) : yf ? V(0, 1, 0) : V(0, 0, 1);
    return (left + right) * (T)0.5;
  } else if (n == 2) {
    // Corner (sa, sb) over the coefficient dimensions (a, b) = (z, y),
    // (z, x) or (y, x).
    auto corner = [&](int sa, int sb) {
      return zf ? (yf ? V(sa, sb, 0) : V(sa, 0, sb)) : V(0, sa, sb);
    };
    T c00 = corner(-1, -1), c02 = corner(-1, 1), c20 = corner(1, -1),
      c22 = corner(1, 1);
    return (c00 + c02 + c20 + c22) / 4;
  } else {
    T c000 = V(-1, -1, -1), c002 = V(-1, -1, 1), c020 = V(-1, 1, -1),
      c022 = V(-1, 1, 1), c200 = V(1, -1, -1), c202 = V(1, -1, 1),
      c220 = V(1, 1, -1), c222 = V(1, 1, 1);
    return (c000 + c002 + c020 + c022 + c200 + c202 + c220 + c222) / 8;
  }
}

// Folds a block's per-thread maxima of |coefficient| into *abs_max with one
// atomic per thread block (as bits: non-negative floats order like their bit
// patterns). Part A: warp maxima to sm_max; part B: thread 0.
template <typename T, typename DeviceType>
MGARDX_EXEC void block_abs_max_a(T m, T *sm_max, SIZE tid) {
  SubGroup<DeviceType> sg;
  for (int offset = sg.size() / 2; offset > 0; offset /= 2) {
    T o = sg.shfl(m, sg.lane() ^ offset);
    m = o > m ? o : m;
  }
  if (sg.lane() == 0) {
    sm_max[tid / sg.size()] = m;
  }
}
template <typename T, typename DeviceType>
MGARDX_EXEC void block_abs_max_b(T *sm_max, SIZE threads, T *abs_max) {
  SubGroup<DeviceType> sg;
  T m = 0;
  for (SIZE w = 0; w < threads / sg.size(); w++) {
    m = sm_max[w] > m ? sm_max[w] : m;
  }
  using U = std::conditional_t<sizeof(T) == 4, unsigned int,
                               unsigned long long>;
  U bits;
  memcpy(&bits, &m, sizeof(T));
  Atomic<U, AtomicGlobalMemory, AtomicDeviceScope, DeviceType>::Max(
      (U *)abs_max, bits);
}

} // namespace hierarchical_block

template <DIM D, typename T, typename DeviceType>
class HierarchicalDecompose8x8x8Functor : public Functor<DeviceType> {
public:
  MGARDX_CONT HierarchicalDecompose8x8x8Functor() {}
  MGARDX_CONT
  HierarchicalDecompose8x8x8Functor(SubArray<D, T, DeviceType> v,
                                    SubArray<D, T, DeviceType> coarse,
                                    SubArray<1, T, DeviceType> coeff,
                                    SubArray<1, T, DeviceType> abs_max)
      : v(v), coarse(coarse), coeff(coeff), abs_max(abs_max) {
    Functor<DeviceType>();
  }

  // Load the 8 x 8 x WIDTH nodes.
  MGARDX_EXEC void Operation1() {
    using namespace hierarchical_block;
    tx = FunctorBase<DeviceType>::GetThreadIdX();
    ty = FunctorBase<DeviceType>::GetThreadIdY();
    nbx = v.shape(D - 1) / 8;
    first_block = FunctorBase<DeviceType>::GetBlockIdX() * BLOCKS;
    blocks = nbx - first_block < BLOCKS ? nbx - first_block : BLOCKS;
    sm = (T *)FunctorBase<DeviceType>::GetSharedMemory();
    stage_coeff = sm + 8 * 8 * WIDTH;
    stage_coarse = stage_coeff + BLOCKS * COEFFS;
    valid = tx / 8 < blocks;
    SIZE z0 = FunctorBase<DeviceType>::GetBlockIdZ() * 8;
    SIZE y = FunctorBase<DeviceType>::GetBlockIdY() * 8 + ty;
    SIZE x = first_block * 8 + tx;
#pragma unroll
    for (int z = 0; z < 8; z++) {
      col[z] = valid ? *v(z0 + z, y, x) : (T)0;
      sm[(z * 8 + ty) * WIDTH + tx] = col[z];
    }
  }

  // Coefficients and coarse values, staged in their output order.
  MGARDX_EXEC void Operation2() {
    using namespace hierarchical_block;
    if (!valid) {
      return;
    }
    const int b = tx / 8, x = tx % 8, y = ty;
#pragma unroll
    for (int z = 0; z < 8; z++) {
      if (is_coarse(z) && is_coarse(y) && is_coarse(x)) {
        stage_coarse[coarse_slot(z, y, b, x)] = col[z];
      } else {
        auto V = [&](int dz, int dy, int dx) {
          return dy == 0 && dx == 0
                     ? col[z + dz]
                     : sm[((z + dz) * 8 + y + dy) * WIDTH + tx + dx];
        };
        stage_coeff[b * COEFFS + coeff_rank(z, y, x)] =
            col[z] - interpolation<T>(z, y, x, V);
      }
    }
  }

  // The blocks' coefficients are contiguous; the coarse grid rows are
  // 5 * blocks long.
  MGARDX_EXEC void Operation3() {
    using namespace hierarchical_block;
    SIZE tid = ty * WIDTH + tx;
    SIZE bz = FunctorBase<DeviceType>::GetBlockIdZ();
    SIZE by = FunctorBase<DeviceType>::GetBlockIdY();
    SIZE nby = v.shape(D - 2) / 8;
    SIZE first_bid = (bz * nby + by) * nbx + first_block;
    T *out = coeff((IDX)(first_bid * COEFFS));
    T m = 0;
    for (SIZE i = tid; i < (SIZE)blocks * COEFFS; i += 8 * WIDTH) {
      T c = stage_coeff[i];
      out[i] = c;
      m = fabs(c) > m ? fabs(c) : m;
    }
    SIZE width = 5 * blocks;
    for (SIZE i = tid; i < 25 * width; i += 8 * WIDTH) {
      SIZE row = i / width, cx = i % width;
      *coarse(bz * 5 + row / 5, by * 5 + row % 5, first_block * 5 + cx) =
          stage_coarse[row * CWIDTH + cx];
    }
    if (abs_max.data() != nullptr) {
      block_abs_max_a<T, DeviceType>(m, sm, tid);
    }
  }

  MGARDX_EXEC void Operation4() {
    using namespace hierarchical_block;
    if (abs_max.data() != nullptr && ty == 0 && tx == 0) {
      block_abs_max_b<T, DeviceType>(sm, 8 * WIDTH, abs_max.data());
    }
  }

  MGARDX_CONT size_t shared_memory_size() {
    using namespace hierarchical_block;
    return (8 * 8 * WIDTH + BLOCKS * COEFFS + 25 * CWIDTH) * sizeof(T);
  }

private:
  SubArray<D, T, DeviceType> v;
  SubArray<D, T, DeviceType> coarse;
  SubArray<1, T, DeviceType> coeff;
  SubArray<1, T, DeviceType> abs_max; // optional: max |coefficient|
  T *sm, *stage_coeff, *stage_coarse;
  T col[8];
  int tx, ty, blocks;
  bool valid;
  SIZE nbx, first_block;
};

template <DIM D, typename T, typename DeviceType>
class HierarchicalDecompose8x8x8Kernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "hierarchical decompose 8x8x8";
  using FunctorType = HierarchicalDecompose8x8x8Functor<D, T, DeviceType>;
  MGARDX_CONT
  HierarchicalDecompose8x8x8Kernel(SubArray<D, T, DeviceType> v,
                                   SubArray<D, T, DeviceType> coarse,
                                   SubArray<1, T, DeviceType> coeff,
                                   SubArray<1, T, DeviceType> abs_max = {})
      : v(v), coarse(coarse), coeff(coeff), abs_max(abs_max) {}
  MGARDX_CONT Task<FunctorType> GenTask(int queue_idx) {
    using namespace hierarchical_block;
    FunctorType functor(v, coarse, coeff, abs_max);
    SIZE nbz = v.shape(D - 3) / 8, nby = v.shape(D - 2) / 8;
    SIZE nbx = v.shape(D - 1) / 8;
    return Task(functor, nbz, nby, (nbx + BLOCKS - 1) / BLOCKS, 1, 8, WIDTH,
                functor.shared_memory_size(), queue_idx, std::string(Name));
  }

private:
  SubArray<D, T, DeviceType> v;
  SubArray<D, T, DeviceType> coarse;
  SubArray<1, T, DeviceType> coeff;
  SubArray<1, T, DeviceType> abs_max;
};

template <DIM D, typename T, typename DeviceType>
class HierarchicalRecompose8x8x8Functor : public Functor<DeviceType> {
public:
  MGARDX_CONT HierarchicalRecompose8x8x8Functor() {}
  MGARDX_CONT
  HierarchicalRecompose8x8x8Functor(SubArray<D, T, DeviceType> v,
                                    SubArray<D, T, DeviceType> coarse,
                                    SubArray<1, T, DeviceType> coeff)
      : v(v), coarse(coarse), coeff(coeff) {
    Functor<DeviceType>();
  }

  // Stage the blocks' coefficients and coarse values.
  MGARDX_EXEC void Operation1() {
    using namespace hierarchical_block;
    tx = FunctorBase<DeviceType>::GetThreadIdX();
    ty = FunctorBase<DeviceType>::GetThreadIdY();
    nbx = v.shape(D - 1) / 8;
    first_block = FunctorBase<DeviceType>::GetBlockIdX() * BLOCKS;
    blocks = nbx - first_block < BLOCKS ? nbx - first_block : BLOCKS;
    stage_coeff = (T *)FunctorBase<DeviceType>::GetSharedMemory();
    stage_coarse = stage_coeff + BLOCKS * COEFFS;
    SIZE tid = ty * WIDTH + tx;
    SIZE bz = FunctorBase<DeviceType>::GetBlockIdZ();
    SIZE by = FunctorBase<DeviceType>::GetBlockIdY();
    SIZE nby = v.shape(D - 2) / 8;
    SIZE first_bid = (bz * nby + by) * nbx + first_block;
    T *in = coeff((IDX)(first_bid * COEFFS));
    for (SIZE i = tid; i < (SIZE)blocks * COEFFS; i += 8 * WIDTH) {
      stage_coeff[i] = in[i];
    }
    SIZE width = 5 * blocks;
    for (SIZE i = tid; i < 25 * width; i += 8 * WIDTH) {
      SIZE row = i / width, cx = i % width;
      stage_coarse[row * CWIDTH + cx] =
          *coarse(bz * 5 + row / 5, by * 5 + row % 5, first_block * 5 + cx);
    }
  }

  // Every node from its coefficient and the coarse values around it.
  MGARDX_EXEC void Operation2() {
    using namespace hierarchical_block;
    if (tx / 8 >= blocks) {
      return;
    }
    const int b = tx / 8, x = tx % 8, y = ty;
    SIZE z0 = FunctorBase<DeviceType>::GetBlockIdZ() * 8;
    SIZE y_gl = FunctorBase<DeviceType>::GetBlockIdY() * 8 + ty;
    SIZE x_gl = first_block * 8 + tx;
#pragma unroll
    for (int z = 0; z < 8; z++) {
      T value;
      if (is_coarse(z) && is_coarse(y) && is_coarse(x)) {
        value = stage_coarse[coarse_slot(z, y, b, x)];
      } else {
        // The neighbors of a coefficient node are coarse.
        auto V = [&](int dz, int dy, int dx) {
          return stage_coarse[coarse_slot(z + dz, y + dy, b, x + dx)];
        };
        value = stage_coeff[b * COEFFS + coeff_rank(z, y, x)] +
                interpolation<T>(z, y, x, V);
      }
      *v(z0 + z, y_gl, x_gl) = value;
    }
  }

  MGARDX_CONT size_t shared_memory_size() {
    using namespace hierarchical_block;
    return (BLOCKS * COEFFS + 25 * CWIDTH) * sizeof(T);
  }

private:
  SubArray<D, T, DeviceType> v;
  SubArray<D, T, DeviceType> coarse;
  SubArray<1, T, DeviceType> coeff;
  T *stage_coeff, *stage_coarse;
  int tx, ty, blocks;
  SIZE nbx, first_block;
};

template <DIM D, typename T, typename DeviceType>
class HierarchicalRecompose8x8x8Kernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "hierarchical recompose 8x8x8";
  using FunctorType = HierarchicalRecompose8x8x8Functor<D, T, DeviceType>;
  MGARDX_CONT
  HierarchicalRecompose8x8x8Kernel(SubArray<D, T, DeviceType> v,
                                   SubArray<D, T, DeviceType> coarse,
                                   SubArray<1, T, DeviceType> coeff)
      : v(v), coarse(coarse), coeff(coeff) {}
  MGARDX_CONT Task<FunctorType> GenTask(int queue_idx) {
    using namespace hierarchical_block;
    FunctorType functor(v, coarse, coeff);
    SIZE nbz = v.shape(D - 3) / 8, nby = v.shape(D - 2) / 8;
    SIZE nbx = v.shape(D - 1) / 8;
    return Task(functor, nbz, nby, (nbx + BLOCKS - 1) / BLOCKS, 1, 8, WIDTH,
                functor.shared_memory_size(), queue_idx, std::string(Name));
  }

private:
  SubArray<D, T, DeviceType> v;
  SubArray<D, T, DeviceType> coarse;
  SubArray<1, T, DeviceType> coeff;
};

template <DIM D, typename T, typename DeviceType>
class HierarchicalDecompose8x8Functor : public Functor<DeviceType> {
public:
  MGARDX_CONT HierarchicalDecompose8x8Functor() {}
  MGARDX_CONT
  HierarchicalDecompose8x8Functor(SubArray<D, T, DeviceType> v,
                                  SubArray<D, T, DeviceType> coarse,
                                  SubArray<1, T, DeviceType> coeff,
                                  SubArray<1, T, DeviceType> abs_max)
      : v(v), coarse(coarse), coeff(coeff), abs_max(abs_max) {
    Functor<DeviceType>();
  }

  // Load the 8 x WIDTH nodes.
  MGARDX_EXEC void Operation1() {
    using namespace hierarchical_block;
    tx = FunctorBase<DeviceType>::GetThreadIdX();
    ty = FunctorBase<DeviceType>::GetThreadIdY();
    nbx = v.shape(D - 1) / 8;
    first_block = FunctorBase<DeviceType>::GetBlockIdX() * BLOCKS;
    blocks = nbx - first_block < BLOCKS ? nbx - first_block : BLOCKS;
    sm = (T *)FunctorBase<DeviceType>::GetSharedMemory();
    stage_coeff = sm + 8 * WIDTH;
    stage_coarse = stage_coeff + BLOCKS * COEFFS_2D;
    valid = tx / 8 < blocks;
    m = valid ? *v(FunctorBase<DeviceType>::GetBlockIdY() * 8 + ty,
                   first_block * 8 + tx)
              : (T)0;
    sm[ty * WIDTH + tx] = m;
  }

  MGARDX_EXEC void Operation2() {
    using namespace hierarchical_block;
    if (!valid) {
      return;
    }
    const int b = tx / 8, x = tx % 8, y = ty;
    if (is_coarse(y) && is_coarse(x)) {
      stage_coarse[coarse_slot_2d(y, b, x)] = m;
    } else {
      auto V = [&](int, int dy, int dx) {
        return sm[(y + dy) * WIDTH + tx + dx];
      };
      stage_coeff[b * COEFFS_2D + coeff_rank_2d(y, x)] =
          m - interpolation<T>(0, y, x, V);
    }
  }

  MGARDX_EXEC void Operation3() {
    using namespace hierarchical_block;
    SIZE tid = ty * WIDTH + tx;
    SIZE by = FunctorBase<DeviceType>::GetBlockIdY();
    T *out = coeff((IDX)((by * nbx + first_block) * COEFFS_2D));
    T mx = 0;
    for (SIZE i = tid; i < (SIZE)blocks * COEFFS_2D; i += 8 * WIDTH) {
      T c = stage_coeff[i];
      out[i] = c;
      mx = fabs(c) > mx ? fabs(c) : mx;
    }
    SIZE width = 5 * blocks;
    for (SIZE i = tid; i < 5 * width; i += 8 * WIDTH) {
      SIZE row = i / width, cx = i % width;
      *coarse(by * 5 + row, first_block * 5 + cx) =
          stage_coarse[row * CWIDTH + cx];
    }
    if (abs_max.data() != nullptr) {
      block_abs_max_a<T, DeviceType>(mx, sm, tid);
    }
  }

  MGARDX_EXEC void Operation4() {
    using namespace hierarchical_block;
    if (abs_max.data() != nullptr && ty == 0 && tx == 0) {
      block_abs_max_b<T, DeviceType>(sm, 8 * WIDTH, abs_max.data());
    }
  }

  MGARDX_CONT size_t shared_memory_size() {
    using namespace hierarchical_block;
    return (8 * WIDTH + BLOCKS * COEFFS_2D + 5 * CWIDTH) * sizeof(T);
  }

private:
  SubArray<D, T, DeviceType> v;
  SubArray<D, T, DeviceType> coarse;
  SubArray<1, T, DeviceType> coeff;
  SubArray<1, T, DeviceType> abs_max; // optional: max |coefficient|
  T *sm, *stage_coeff, *stage_coarse;
  T m;
  int tx, ty, blocks;
  bool valid;
  SIZE nbx, first_block;
};

template <DIM D, typename T, typename DeviceType>
class HierarchicalDecompose8x8Kernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "hierarchical decompose 8x8";
  using FunctorType = HierarchicalDecompose8x8Functor<D, T, DeviceType>;
  MGARDX_CONT
  HierarchicalDecompose8x8Kernel(SubArray<D, T, DeviceType> v,
                                 SubArray<D, T, DeviceType> coarse,
                                 SubArray<1, T, DeviceType> coeff,
                                 SubArray<1, T, DeviceType> abs_max = {})
      : v(v), coarse(coarse), coeff(coeff), abs_max(abs_max) {}
  MGARDX_CONT Task<FunctorType> GenTask(int queue_idx) {
    using namespace hierarchical_block;
    FunctorType functor(v, coarse, coeff, abs_max);
    SIZE nby = v.shape(D - 2) / 8, nbx = v.shape(D - 1) / 8;
    return Task(functor, 1, nby, (nbx + BLOCKS - 1) / BLOCKS, 1, 8, WIDTH,
                functor.shared_memory_size(), queue_idx, std::string(Name));
  }

private:
  SubArray<D, T, DeviceType> v;
  SubArray<D, T, DeviceType> coarse;
  SubArray<1, T, DeviceType> coeff;
  SubArray<1, T, DeviceType> abs_max;
};

template <DIM D, typename T, typename DeviceType>
class HierarchicalRecompose8x8Functor : public Functor<DeviceType> {
public:
  MGARDX_CONT HierarchicalRecompose8x8Functor() {}
  MGARDX_CONT
  HierarchicalRecompose8x8Functor(SubArray<D, T, DeviceType> v,
                                  SubArray<D, T, DeviceType> coarse,
                                  SubArray<1, T, DeviceType> coeff)
      : v(v), coarse(coarse), coeff(coeff) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    using namespace hierarchical_block;
    tx = FunctorBase<DeviceType>::GetThreadIdX();
    ty = FunctorBase<DeviceType>::GetThreadIdY();
    nbx = v.shape(D - 1) / 8;
    first_block = FunctorBase<DeviceType>::GetBlockIdX() * BLOCKS;
    blocks = nbx - first_block < BLOCKS ? nbx - first_block : BLOCKS;
    stage_coeff = (T *)FunctorBase<DeviceType>::GetSharedMemory();
    stage_coarse = stage_coeff + BLOCKS * COEFFS_2D;
    SIZE tid = ty * WIDTH + tx;
    SIZE by = FunctorBase<DeviceType>::GetBlockIdY();
    T *in = coeff((IDX)((by * nbx + first_block) * COEFFS_2D));
    for (SIZE i = tid; i < (SIZE)blocks * COEFFS_2D; i += 8 * WIDTH) {
      stage_coeff[i] = in[i];
    }
    SIZE width = 5 * blocks;
    for (SIZE i = tid; i < 5 * width; i += 8 * WIDTH) {
      SIZE row = i / width, cx = i % width;
      stage_coarse[row * CWIDTH + cx] =
          *coarse(by * 5 + row, first_block * 5 + cx);
    }
  }

  MGARDX_EXEC void Operation2() {
    using namespace hierarchical_block;
    if (tx / 8 >= blocks) {
      return;
    }
    const int b = tx / 8, x = tx % 8, y = ty;
    T value;
    if (is_coarse(y) && is_coarse(x)) {
      value = stage_coarse[coarse_slot_2d(y, b, x)];
    } else {
      auto V = [&](int, int dy, int dx) {
        return stage_coarse[coarse_slot_2d(y + dy, b, x + dx)];
      };
      value = stage_coeff[b * COEFFS_2D + coeff_rank_2d(y, x)] +
              interpolation<T>(0, y, x, V);
    }
    *v(FunctorBase<DeviceType>::GetBlockIdY() * 8 + ty, first_block * 8 + tx) =
        value;
  }

  MGARDX_CONT size_t shared_memory_size() {
    using namespace hierarchical_block;
    return (BLOCKS * COEFFS_2D + 5 * CWIDTH) * sizeof(T);
  }

private:
  SubArray<D, T, DeviceType> v;
  SubArray<D, T, DeviceType> coarse;
  SubArray<1, T, DeviceType> coeff;
  T *stage_coeff, *stage_coarse;
  int tx, ty, blocks;
  SIZE nbx, first_block;
};

template <DIM D, typename T, typename DeviceType>
class HierarchicalRecompose8x8Kernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "hierarchical recompose 8x8";
  using FunctorType = HierarchicalRecompose8x8Functor<D, T, DeviceType>;
  MGARDX_CONT
  HierarchicalRecompose8x8Kernel(SubArray<D, T, DeviceType> v,
                                 SubArray<D, T, DeviceType> coarse,
                                 SubArray<1, T, DeviceType> coeff)
      : v(v), coarse(coarse), coeff(coeff) {}
  MGARDX_CONT Task<FunctorType> GenTask(int queue_idx) {
    using namespace hierarchical_block;
    FunctorType functor(v, coarse, coeff);
    SIZE nby = v.shape(D - 2) / 8, nbx = v.shape(D - 1) / 8;
    return Task(functor, 1, nby, (nbx + BLOCKS - 1) / BLOCKS, 1, 8, WIDTH,
                functor.shared_memory_size(), queue_idx, std::string(Name));
  }

private:
  SubArray<D, T, DeviceType> v;
  SubArray<D, T, DeviceType> coarse;
  SubArray<1, T, DeviceType> coeff;
};

} // namespace in_cache_block

} // namespace data_refactoring

} // namespace mgard_x

#endif
