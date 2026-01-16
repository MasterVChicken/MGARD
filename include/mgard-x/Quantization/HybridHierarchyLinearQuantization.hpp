/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_HYBRID_HIERARCHY_LINEAR_QUANTIZATION_TEMPLATE
#define MGARD_X_HYBRID_HIERARCHY_LINEAR_QUANTIZATION_TEMPLATE

#include "../RuntimeX/RuntimeX.h"
#include "LinearQuantization.hpp"
#include "LocalQuantization.hpp"
#include "QuantizationInterface.hpp"

namespace mgard_x {

#define MGARDX_QUANTIZE 1
#define MGARDX_DEQUANTIZE 2

// TODO: Results shows the error is too strict, fix this bug

template <DIM D, typename T, typename Q, typename DeviceType>
class HybridHierarchyQuantizer
    : public QuantizationInterface<D, T, Q, DeviceType> {
 public:
  HybridHierarchyQuantizer() : initialized(false) {}

  HybridHierarchyQuantizer(Hierarchy<D, T, DeviceType>& hierarchy,
                           Hierarchy<D, T, DeviceType>& global_hierarchy,
                           Config config)
      : initialized(true),
        hierarchy(&hierarchy),
        global_hierarchy(&global_hierarchy),
        config(config) {}

  // Think about how can we construct a index table for ROIs

  void Adapt(Hierarchy<D, T, DeviceType>& hierarchy,
             Hierarchy<D, T, DeviceType>& global_hierarchy, Config config,
             int queue_idx) {
    this->initialized = true;
    this->hierarchy = &hierarchy;
    this->global_hierarchy = &global_hierarchy;
    this->config = config;

    local_quantizer.Adapt(hierarchy, config, queue_idx);
    global_quantizer.Adapt(global_hierarchy, config, queue_idx);
  }

  // Return the error budget for global quantization
  T ErrorBudgetAllocation(T tol) { return tol / pow(2, this->L); }

  static size_t EstimateMemoryFootprint(std::vector<SIZE> shape,
                                        Config config) {
    size_t size = 0;
    return size;
  }

  void Quantize(SubArray<D, T, DeviceType> original_data,
                enum error_bound_type ebtype, T tol, T s, T norm,
                SubArray<D, Q, DeviceType> quantized_data, int queue_idx) {}

  void Dequantize(SubArray<D, T, DeviceType> original_data,
                  enum error_bound_type ebtype, T tol, T s, T norm,
                  SubArray<D, Q, DeviceType> quantized_data, int queue_idx) {}

  template <typename LosslessCompressorType>
  void Quantize(SubArray<1, T, DeviceType> original_data,
                enum error_bound_type ebtype, T tol, T s, T norm,
                SubArray<1, Q, DeviceType> quantized_data,
                LosslessCompressorType& lossless, int queue_idx) {
    // Global quantization
    T global_tol = ErrorBudgetAllocation(tol);
    SIZE global_q_size = (this->M > 0) ? (global_hierarchy->l_target() + 1) : 0;
    std::vector<SIZE> global_shape =
        global_hierarchy->level_shape(global_hierarchy->l_target());
    SubArray<D, T, DeviceType> global_data_v(global_shape,
                                             original_data.data());
    SubArray<D, Q, DeviceType> global_data_q(global_shape,
                                             quantized_data.data());

    global_quantizer.Quantize(global_data_v, ebtype, global_tol, s, norm,
                              global_data_q, lossless, queue_idx);

    // Local quantization
    SIZE local_offset = original_data.shape(0) - global_q_size;
    SubArray<1, T, DeviceType> local_data_v({local_offset},
                                            original_data(global_q_size));
    SubArray<1, Q, DeviceType> local_data_q({local_offset},
                                            quantized_data(global_q_size));
    local_quantizer.Quantize(local_data_v, ebtype, global_tol, s, norm,
                             local_data_q, lossless, queue_idx);
  }

  template <typename LosslessCompressorType>
  void Dequantize(SubArray<1, T, DeviceType> original_data,
                  enum error_bound_type ebtype, T tol, T s, T norm,
                  SubArray<1, Q, DeviceType> quantized_data,
                  LosslessCompressorType& lossless, int queue_idx) {
    // Global dequantization
    T global_tol = ErrorBudgetAllocation(tol);
    SIZE global_q_size = (this->M > 0) ? (global_hierarchy->l_target() + 1) : 0;
    std::vector<SIZE> global_shape =
        global_hierarchy->level_shape(global_hierarchy->l_target());
    SubArray<D, T, DeviceType> global_data_v(global_shape,
                                             original_data.data());
    SubArray<D, Q, DeviceType> global_data_q(global_shape,
                                             quantized_data.data());

    global_quantizer.Dequantize(global_data_v, ebtype, global_tol, s, norm,
                                global_data_q, lossless, queue_idx);

    // Local dequantization
    SIZE local_offset = original_data.shape(0) - global_q_size;
    SubArray<1, T, DeviceType> local_data_v({local_offset},
                                            original_data(global_q_size));
    SubArray<1, Q, DeviceType> local_data_q({local_offset},
                                            quantized_data(global_q_size));
    local_quantizer.Dequantize(local_data_v, ebtype, global_tol, s, norm,
                               local_data_q, lossless, queue_idx);
  }

  bool initialized;
  SIZE L;  // Number of local levels
  SIZE M;  // Number of global levels

  Hierarchy<D, T, DeviceType>* hierarchy;
  Hierarchy<D, T, DeviceType>* global_hierarchy;
  Config config;

  LocalQuantizer<D, T, Q, DeviceType> local_quantizer;
  LinearQuantizer<D, T, Q, DeviceType> global_quantizer;
};

}  // namespace mgard_x

#endif