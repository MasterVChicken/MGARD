/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */




#include "RuntimeXPublic.h"


#ifndef MGARD_X_RUNTIME_X_H
#define MGARD_X_RUNTIME_X_H

namespace mgard_x {



// reduction operations
// #define SUM 0
// #define MAX 1

}

#endif


#include "DataTypes.h"
#include "AutoTuners/AutoTuner.h"
#include "Tasks/Task.h"

#ifdef MGARDX_COMPILE_SERIAL
#include "DeviceAdapters/DeviceAdapterSerial.h"
#endif

#ifdef MGARDX_COMPILE_CUDA
#include "DeviceAdapters/DeviceAdapterCuda.h"
#endif

#ifdef MGARDX_COMPILE_HIP
#include "DeviceAdapters/DeviceAdapterHip.h"
#endif

#ifdef MGARDX_COMPILE_KOKKOS
#include "DeviceAdapters/DeviceAdapterKokkos.h"
#endif

#include "Utilities/CheckShape.hpp"
#include "Utilities/OffsetCalculators.hpp"

#include "DataStructures/Array.hpp"
#include "DataStructures/SubArray.hpp"
#include "DataStructures/SubArrayCopy.hpp"
#include "Utilities/SubArrayPrinter.hpp"