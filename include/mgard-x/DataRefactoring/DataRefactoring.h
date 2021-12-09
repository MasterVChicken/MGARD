/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#ifndef MGARD_X_DATA_REFACTORING
#define MGARD_X_DATA_REFACTORING

// #include "Common.h"
#include "../Handle.h"
#include "../RuntimeX/RuntimeXPublic.h"


namespace mgard_x {

// template <DIM D, typename T, typename DeviceType>
// void calc_coeff_pointers(Handle<D, T> &handle, DIM curr_dims[3], DIM l, SubArray<D, T> doutput,
//                          SubArray<D, T> &dcoarse,
//                          SubArray<D, T> &dcoeff_f,
//                          SubArray<D, T> &dcoeff_c,
//                          SubArray<D, T> &dcoeff_r,
//                          SubArray<D, T> &dcoeff_cf,
//                          SubArray<D, T> &dcoeff_rf,
//                          SubArray<D, T> &dcoeff_rc,
//                          SubArray<D, T> &dcoeff_rcf);

// template <DIM D, typename T, typename DeviceType>
// void calc_coefficients_3d(Handle<D, T> &handle, SubArray<D, T> dinput, 
//                         SubArray<D, T> &doutput, SIZE l, int queue_idx);

// template <DIM D, typename T, typename DeviceType>
// void coefficients_restore_3d(Handle<D, T> &handle, SubArray<D, T> dinput, 
//                         SubArray<D, T> &doutput, SIZE l, int queue_idx);

// template <DIM D, typename T, typename DeviceType>
// void calc_correction_3d(Handle<D, T> &handle, SubArray<D, T> dcoeff, 
//                         SubArray<D, T> &dcorrection, SIZE l, int queue_idx);

// template <DIM D, typename T, typename DeviceType>
// void calc_coefficients_nd(Handle<D, T> &handle, SubArray<D, T> dinput1, 
//                           SubArray<D, T> dinput2, 
//                         SubArray<D, T> &doutput, SIZE l, int queue_idx);

// template <DIM D, typename T, typename DeviceType>
// void coefficients_restore_nd(Handle<D, T> &handle, SubArray<D, T> dinput1, 
//                              SubArray<D, T> dinput2, 
//                              SubArray<D, T> &doutput, SIZE l, int queue_idx);

// template <DIM D, typename T, typename DeviceType>
// void calc_correction_nd(Handle<D, T> &handle, SubArray<D, T> dcoeff, 
//                         SubArray<D, T> &dcorrection, SIZE l, int queue_idx);

template <DIM D, typename T, typename DeviceType>
void decompose(Handle<D, T, DeviceType> &handle, SubArray<D, T, DeviceType>& v, 
               SIZE l_target, int queue_idx);

template <DIM D, typename T, typename DeviceType>
void recompose(Handle<D, T, DeviceType> &handle, SubArray<D, T, DeviceType>& v, 
               SIZE l_target, int queue_idx);

} // namespace mgard_x

#endif