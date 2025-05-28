/*
 * Copyright 2023, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: Jan. 15, 2023
 */

#ifndef MGARD_X_RECOMP_MULTI_DIMENSION_8x8x8_KERNEL_TEMPLATE
#define MGARD_X_RECOMP_MULTI_DIMENSION_8x8x8_KERNEL_TEMPLATE

#include "../../RuntimeX/RuntimeX.h"
#include "../MultiDimension/Correction/IPKFunctor.h"
#include "../MultiDimension/Correction/LPKFunctor.h"
#include "IndexTable3x3x3.hpp"
#include "IndexTable5x5x5.hpp"
#include "IndexTable8x8x8.hpp"

#define DECOMPOSE 0
#define RECOMPOSE 1

namespace mgard_x {

namespace data_refactoring {

namespace in_cache_block {

/*

v           x           y           z           c                total
8*8*8(512)  5*8*8(320)  5*5*8(200)  5*5*5(125)  0                1157
5*5*5(125)  3*5*5(75)   3*3*5(45)   3*3*3(27)   8*8*8-5*5*5(387) 659
3*3*3(27)   2*3*3(18)   2*2*3(12)   2*2*2(8)    8*8*8-3*3*3(485) 550

 v(512)  x(320)  y(200)  z(125)
c8(512)  v(125)  x( 75)  y( 45)  z(27)
c8(512) c5( 98)  x( 18)  y( 12)  z( 8)
c8(512) c5( 98)  c3(19)  c2( 8)
*/

template <DIM D, typename T, SIZE Z, SIZE Y, SIZE X, OPTION OP,
          typename DeviceType>
class RecompMultiDimension8x8x8Functor : public Functor<DeviceType> {
 public:
  MGARDX_CONT RecompMultiDimension8x8x8Functor() {}
  MGARDX_CONT RecompMultiDimension8x8x8Functor(
      SubArray<D, T, DeviceType> v, SubArray<D, T, DeviceType> coarse,
      SubArray<1, T, DeviceType> coeff)
      : v(v), coarse(coarse), coeff(coeff) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void initialize_sm_8x8x8() {
    sm_v = (T *)FunctorBase<DeviceType>::GetSharedMemory();
    sm_x = sm_v + 8 * 8 * 8;
    sm_y = sm_x + 5 * 8 * 8;
    sm_z = sm_y + 5 * 5 * 8;
  }

  MGARDX_EXEC void initialize_sm_5x5x5() {
    sm_c8 = (T *)FunctorBase<DeviceType>::GetSharedMemory();
    sm_v = sm_c8 + 8 * 8 * 8;
    sm_x = sm_v + 5 * 5 * 5;
    sm_y = sm_x + 5 * 5 * 3;
    sm_z = sm_y + 5 * 3 * 3;
  }

  MGARDX_EXEC void initialize_sm_3x3x3() {
    sm_c8 = (T *)FunctorBase<DeviceType>::GetSharedMemory();
    sm_c5 = sm_c8 + 8 * 8 * 8;
    sm_v = sm_c5 + 5 * 5 * 5;
    sm_x = sm_v + 3 * 3 * 3;
    sm_y = sm_x + 3 * 3 * 2;
    sm_z = sm_y + 3 * 2 * 2;
  }

  MGARDX_EXEC void initialize_sm_2x2x2() {
    sm_c8 = (T *)FunctorBase<DeviceType>::GetSharedMemory();
    sm_c5 = sm_c8 + 8 * 8 * 8;
    sm_c3 = sm_c5 + 5 * 5 * 5;
    sm_c2 = sm_c3 + 3 * 3 * 3;
  }

  // Interpolation
  MGARDX_EXEC void Operation1() {
    initialize_sm_8x8x8();
    x = FunctorBase<DeviceType>::GetThreadIdX();
    y = FunctorBase<DeviceType>::GetThreadIdY();
    z = FunctorBase<DeviceType>::GetThreadIdZ();
    x_tb = FunctorBase<DeviceType>::GetBlockIdX();
    y_tb = FunctorBase<DeviceType>::GetBlockIdY();
    z_tb = FunctorBase<DeviceType>::GetBlockIdZ();
    x_gl = X * x_tb + x;
    y_gl = Y * y_tb + y;
    z_gl = Z * z_tb + z;

    tid = z * X * Y + y * X + x;
    bid = z_tb * FunctorBase<DeviceType>::GetGridDimX() *
              FunctorBase<DeviceType>::GetGridDimY() +
          y_tb * FunctorBase<DeviceType>::GetGridDimX() + x_tb;
    if (z == 0 && y == 0 && x == 0) sm_v[zero_const_offset] = (T)0;

    offset = get_idx(ld1, ld2, z, y, x);
    sm_v[offset] = 0.0;
  }

  MGARDX_EXEC void Operation2() {
    if (tid < 125) {
      int const *index = Coarse_Reorder_8x8x8(tid);
      sm_v[Coarse_Offset_8x8x8(tid)] = *coarse(
          z_tb * 5 + index[0], y_tb * 5 + index[1], x_tb * 5 + index[2]);
    } else {
      int op_tid = tid - 125;
      sm_v[Coeff_Offset_8x8x8(op_tid)] = *coeff(bid * 387 + op_tid);
    }
  }

  // MassTransX
  MGARDX_EXEC void Operation3() {
    if (tid < 320) {
      int const *index = MassTrans_X_Offset_8x8x8(tid);
      T a = sm_v[index[0]];
      T b = sm_v[index[1]];
      T c = sm_v[index[2]];
      T d = sm_v[index[3]];
      T e = sm_v[index[4]];
      T const *dist = MassTrans_Weights_8x8x8<T>(index[6]);
      sm_x[index[5]] =
          a * dist[0] + b * dist[1] + c * dist[2] + d * dist[3] + e * dist[4];
    }
  }

  // MassTransY
  MGARDX_EXEC void Operation4() {
    if (tid < 200) {
      int const *index = MassTrans_Y_Offset_8x8x8(tid);
      T a = sm_x[index[0]];
      T b = sm_x[index[1]];
      T c = sm_x[index[2]];
      T d = sm_x[index[3]];
      T e = sm_x[index[4]];
      T const *dist = MassTrans_Weights_8x8x8<T>(index[6]);
      sm_y[index[5]] =
          a * dist[0] + b * dist[1] + c * dist[2] + d * dist[3] + e * dist[4];
    }
  }

  // MassTransZ
  MGARDX_EXEC void Operation5() {
    if (tid < 125) {
      int const *index = MassTrans_Z_Offset_8x8x8(tid);
      T a = sm_y[index[0]];
      T b = sm_y[index[1]];
      T c = sm_y[index[2]];
      T d = sm_y[index[3]];
      T e = sm_y[index[4]];
      T const *dist = MassTrans_Weights_8x8x8<T>(index[6]);
      sm_z[index[5]] =
          a * dist[0] + b * dist[1] + c * dist[2] + d * dist[3] + e * dist[4];
    }
  }

  // TriadiagX
  MGARDX_EXEC void Operation6() {
    if (tid < 25) {
      int const *index = TriDiag_X_Offset_8x8x8(tid);
      T a = sm_z[index[0]];
      T b = sm_z[index[1]];
      T c = sm_z[index[2]];
      T d = sm_z[index[3]];
      T e = sm_z[index[4]];

      a += 0 * amxbm_8x8x8<T>(0);
      b += a * amxbm_8x8x8<T>(1);
      c += b * amxbm_8x8x8<T>(2);
      d += c * amxbm_8x8x8<T>(3);
      e += d * amxbm_8x8x8<T>(4);

      e = (e + am_8x8x8<T>(5) * 0) * bm_8x8x8<T>(5);
      d = (d + am_8x8x8<T>(4) * e) * bm_8x8x8<T>(4);
      c = (c + am_8x8x8<T>(3) * d) * bm_8x8x8<T>(3);
      b = (b + am_8x8x8<T>(2) * c) * bm_8x8x8<T>(2);
      a = (a + am_8x8x8<T>(1) * b) * bm_8x8x8<T>(1);

      sm_z[index[0]] = a;
      sm_z[index[1]] = b;
      sm_z[index[2]] = c;
      sm_z[index[3]] = d;
      sm_z[index[4]] = e;
    }
  }

  // TriadiagY
  MGARDX_EXEC void Operation7() {
    if (tid < 25) {
      int const *index = TriDiag_Y_Offset_8x8x8(tid);
      T a = sm_z[index[0]];
      T b = sm_z[index[1]];
      T c = sm_z[index[2]];
      T d = sm_z[index[3]];
      T e = sm_z[index[4]];

      a += 0 * amxbm_8x8x8<T>(0);
      b += a * amxbm_8x8x8<T>(1);
      c += b * amxbm_8x8x8<T>(2);
      d += c * amxbm_8x8x8<T>(3);
      e += d * amxbm_8x8x8<T>(4);

      e = (e + am_8x8x8<T>(5) * 0) * bm_8x8x8<T>(5);
      d = (d + am_8x8x8<T>(4) * e) * bm_8x8x8<T>(4);
      c = (c + am_8x8x8<T>(3) * d) * bm_8x8x8<T>(3);
      b = (b + am_8x8x8<T>(2) * c) * bm_8x8x8<T>(2);
      a = (a + am_8x8x8<T>(1) * b) * bm_8x8x8<T>(1);

      sm_z[index[0]] = a;
      sm_z[index[1]] = b;
      sm_z[index[2]] = c;
      sm_z[index[3]] = d;
      sm_z[index[4]] = e;
    }
  }

  // TriadiagZ
  MGARDX_EXEC void Operation8() {
    if (tid < 25) {
      int const *index = TriDiag_Z_Offset_8x8x8(tid);
      T a = sm_z[index[0]];
      T b = sm_z[index[1]];
      T c = sm_z[index[2]];
      T d = sm_z[index[3]];
      T e = sm_z[index[4]];

      a += 0 * amxbm_8x8x8<T>(0);
      b += a * amxbm_8x8x8<T>(1);
      c += b * amxbm_8x8x8<T>(2);
      d += c * amxbm_8x8x8<T>(3);
      e += d * amxbm_8x8x8<T>(4);

      e = (e + am_8x8x8<T>(5) * 0) * bm_8x8x8<T>(5);
      d = (d + am_8x8x8<T>(4) * e) * bm_8x8x8<T>(4);
      c = (c + am_8x8x8<T>(3) * d) * bm_8x8x8<T>(3);
      b = (b + am_8x8x8<T>(2) * c) * bm_8x8x8<T>(2);
      a = (a + am_8x8x8<T>(1) * b) * bm_8x8x8<T>(1);

      sm_z[index[0]] = a;
      sm_z[index[1]] = b;
      sm_z[index[2]] = c;
      sm_z[index[3]] = d;
      sm_z[index[4]] = e;
    }
  }

  // Deapply Correction
  MGARDX_EXEC void Operation9() {
    if (tid < 125) {
      sm_v[Coarse_Offset_8x8x8(tid)] -= sm_z[tid];
    } 
  }

  MGARDX_EXEC void Operation10(){
    op_tid = tid;
    if (tid < 225) {
      left = sm_v[Coeff1D_L_Offset_8x8x8(op_tid)];
      right = sm_v[Coeff1D_R_Offset_8x8x8(op_tid)];
      middle = sm_v[Coeff1D_M_Offset_8x8x8(op_tid)];
      middle = middle + (left + right) * (T)0.5;
      sm_v[Coeff1D_M_Offset_8x8x8(op_tid)] = middle;
    } else if (tid >= 256 && tid < 256 + 135) {
      op_tid -= 256;
      T c00 = sm_v[Coeff2D_LL_Offset_8x8x8(op_tid)];
      T c02 = sm_v[Coeff2D_LR_Offset_8x8x8(op_tid)];
      T c20 = sm_v[Coeff2D_RL_Offset_8x8x8(op_tid)];
      T c22 = sm_v[Coeff2D_RR_Offset_8x8x8(op_tid)];
      T c11 = sm_v[Coeff2D_MM_Offset_8x8x8(op_tid)];
      c11 += (c00 + c02 + c20 + c22) / 4;
      sm_v[Coeff2D_MM_Offset_8x8x8(op_tid)] = c11;
    } else if (tid >= 416 && tid < 416 + 27) {
      op_tid -= 416;
      T c000 = sm_v[Coeff3D_LLL_Offset_8x8x8(op_tid)];
      T c002 = sm_v[Coeff3D_LLR_Offset_8x8x8(op_tid)];
      T c020 = sm_v[Coeff3D_LRL_Offset_8x8x8(op_tid)];
      T c022 = sm_v[Coeff3D_LRR_Offset_8x8x8(op_tid)];
      T c200 = sm_v[Coeff3D_RLL_Offset_8x8x8(op_tid)];
      T c202 = sm_v[Coeff3D_RLR_Offset_8x8x8(op_tid)];
      T c220 = sm_v[Coeff3D_RRL_Offset_8x8x8(op_tid)];
      T c222 = sm_v[Coeff3D_RRR_Offset_8x8x8(op_tid)];
      T c111 = sm_v[Coeff3D_MMM_Offset_8x8x8(op_tid)];
      c111 += (c000 + c002 + c020 + c022 + c200 + c202 + c220 + c222) / 8;
      sm_v[Coeff3D_MMM_Offset_8x8x8(op_tid)] = c111;
    }
  }

  MGARDX_EXEC void Operation11(){
    if (z_gl < v.shape(D - 3) && y_gl < v.shape(D - 2) &&
        x_gl < v.shape(D - 1)) {
       *v(z_gl, y_gl, x_gl) = sm_v[offset];
    }
  }

  MGARDX_CONT size_t shared_memory_size() {
    size_t size = (Z * Y * X) + Z * Y * (X / 2 + 1) +
                  Z * (Y / 2 + 1) * (X / 2 + 1) +
                  (Z / 2 + 1) * (Y / 2 + 1) * (X / 2 + 1) + 1;
    return size * sizeof(T);
  }

 private:
  SubArray<D, T, DeviceType> v;
  SubArray<D, T, DeviceType> coarse;
  SubArray<1, T, DeviceType> coeff;
  T *sm_v, *sm_x, *sm_y, *sm_z, *sm_c8, *sm_c5, *sm_c3, *sm_c2;
  int ld1 = X;
  int ld2 = Y;
  int z, y, x, z_tb, y_tb, x_tb, z_gl, y_gl, x_gl;
  int tid, bid, op_tid;
  T left, right, middle;
  int offset;
  int zero_const_offset = (Z * Y * X) + Z * Y * (X / 2 + 1) +
                          Z * (Y / 2 + 1) * (X / 2 + 1) +
                          (Z / 2 + 1) * (Y / 2 + 1) * (X / 2 + 1);
};

template <DIM D, typename T, OPTION OP, typename DeviceType>
class RecompMultiDimension8x8x8Kernel : public Kernel {
 public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "lwpk";
  MGARDX_CONT
  RecompMultiDimension8x8x8Kernel(SubArray<D, T, DeviceType> v,
                                  SubArray<D, T, DeviceType> coarse,
                                  SubArray<1, T, DeviceType> coeff)
      : v(v), coarse(coarse), coeff(coeff) {}

  MGARDX_CONT
      Task<RecompMultiDimension8x8x8Functor<D, T, 8, 8, 8, OP, DeviceType>>
      GenTask(int queue_idx) {
    using FunctorType =
        RecompMultiDimension8x8x8Functor<D, T, 8, 8, 8, OP, DeviceType>;
    FunctorType functor(v, coarse, coeff);

    SIZE total_thread_z = v.shape(D - 3);
    SIZE total_thread_y = v.shape(D - 2);
    SIZE total_thread_x = v.shape(D - 1);

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbz = 8;
    tby = 8;
    tbx = 8;
    gridz = ceil((double)total_thread_z / tbz);
    gridy = ceil((double)total_thread_y / tby);
    gridx = ceil((double)total_thread_x / tbx);

    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                std::string(Name));
  }

 private:
  SubArray<D, T, DeviceType> v;
  SubArray<D, T, DeviceType> coarse;
  SubArray<1, T, DeviceType> coeff;
};

}  // namespace in_cache_block

}  // namespace data_refactoring

}  // namespace mgard_x

#endif