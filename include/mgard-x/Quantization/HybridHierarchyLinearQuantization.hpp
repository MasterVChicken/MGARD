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

template <DIM D, typename T, typename Q, OPTION OP, typename DeviceType>
class HybridQuantizeFunctor : public Functor<DeviceType> {
 public:
  MGARDX_CONT HybridQuantizeFunctor() {}
  MGARDX_CONT HybridQuantizeFunctor(T quantizer, SubArray<1, T, DeviceType> v,
                                    SubArray<1, Q, DeviceType> quantized_v)
      : quantizer(quantizer), v(v), quantized_v(quantized_v) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    SIZE idx = FunctorBase<DeviceType>::GetBlockIdX() *
                   FunctorBase<DeviceType>::GetBlockDimX() +
               FunctorBase<DeviceType>::GetThreadIdX();

    if (idx < v.shape(0)) {
      T t = *v(idx);
      Q quantized_data;

      if constexpr (OP == MGARDX_QUANTIZE) {
        if constexpr (sizeof(T) == sizeof(double)) {
          quantized_data = copysign((T)0.5 + fabs(t * quantizer), t);
        } else if constexpr (sizeof(T) == sizeof(float)) {
          quantized_data = copysign((T)0.5 + fabsf(t * quantizer), t);
        }
        *quantized_v(idx) = quantized_data;
      } else if constexpr (OP == MGARDX_DEQUANTIZE) {
        quantized_data = *quantized_v(idx);
        *v(idx) = (quantizer) * (T)quantized_data;
      }
    }
  }
  MGARDX_CONT size_t shared_memory_size() {
    size_t size = 0;
    return size;
  }

 private:
  T quantizer;
  SubArray<1, T, DeviceType> v;
  SubArray<1, Q, DeviceType> quantized_v;
};

template <DIM D, typename T, typename Q, OPTION OP, typename DeviceType>
class HybridQuantizeKernel : public Kernel {
 public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "hyb_qk";
  MGARDX_CONT HybridQuantizeKernel(T quantizer, SubArray<1, T, DeviceType> v,
                                   SubArray<1, Q, DeviceType> quantized_v)
      : quantizer(quantizer), v(v), quantized_v(quantized_v) {}

  MGARDX_CONT Task<HybridQuantizeFunctor<D, T, Q, OP, DeviceType>> GenTask(
      int queue_idx) {
    using FunctorType = HybridQuantizeFunctor<D, T, Q, OP, DeviceType>;
    FunctorType functor(quantizer, v, quantized_v);

    SIZE tbx = 256, tby = 1, tbz = 1;
    SIZE gridx = ceil((double)v.shape(0) / tbx);
    SIZE gridy = 1, gridz = 1;
    size_t sm_size = 0;

    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                std::string(Name));
  }

 private:
  T quantizer;
  SubArray<1, T, DeviceType> v;
  SubArray<1, Q, DeviceType> quantized_v;
};

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
        config(config) {
    Initialize();
  }

  void Adapt(Hierarchy<D, T, DeviceType>& hierarchy,
             Hierarchy<D, T, DeviceType>& global_hierarchy, Config config,
             int queue_idx) {
    this->initialized = true;
    this->hierarchy = &hierarchy;
    this->global_hierarchy = &global_hierarchy;
    this->config = config;
    Initialize();
  }

  void Initialize() {
    this->L = config.num_local_refactoring_level;
    this->M = config.num_global_refactoring_level;

    if (this->M < 0 && global_hierarchy != nullptr) {
      this->M = global_hierarchy->l_target();
    }

    if (this->M > 0 && global_hierarchy == nullptr) {
      log::err("HybridHierarchyQuantizer: M > 0 but global_hierarchy is null");
      this->M = 0;
    }

    ComputeLocalShapes();
    ComputeGlobalSizes();
    PrepareLayerOffsets();

    // log::info("HybridHierarchyQuantizer initialized: L=" +
    //          std::to_string(this->L) + ", M=" + std::to_string(this->M) +
    //          ", total_layers=" + std::to_string(layer_len.size()));
  }

  void ComputeLocalShapes() {
    coarse_shape = hierarchy->level_shape(hierarchy->l_target());
    local_coeff_size.clear();
    coarse_num_elems.clear();

    for (int l = 0; l < this->L; ++l) {
      SIZE last_level_size = 1, curr_level_size = 1;
      for (DIM d = 0; d < D; ++d) {
        coarse_shape[d] = ((coarse_shape[d] - 1) / 8 + 1) * 8;
        last_level_size *= coarse_shape[d];
        coarse_shape[d] = ((coarse_shape[d] - 1) / 8 + 1) * 5;
        curr_level_size *= coarse_shape[d];
      }
      coarse_num_elems.push_back(curr_level_size);
      local_coeff_size.push_back(last_level_size - curr_level_size);
    }
  }

  void ComputeGlobalSizes() {
    if (this->M > 0 && global_hierarchy != nullptr) {
      if (this->M > global_hierarchy->l_target()) {
        this->M = global_hierarchy->l_target();
      }
      // Global Total Size 等于 Local 也就是 Coarsest 的大小
      global_total_size = (this->L > 0) ? coarse_num_elems[this->L - 1]
                                        : hierarchy->total_num_elems();
    } else {
      global_total_size = (this->L > 0) ? coarse_num_elems[this->L - 1]
                                        : hierarchy->total_num_elems();
    }
  }

  void PrepareLayerOffsets() {
    // 这里我们只管理 Local Layers 的 Offset
    // Global 部分的数据被视为一整块，由 MGARD 原始 Kernel 处理
    layer_len.clear();
    layer_off.clear();

    // 如果没有 Global Refactor (M=0)，Local Coarsest 是第一层
    SIZE offset = 0;
    if (this->M == 0) {
      layer_len.push_back(global_total_size);
      layer_off.push_back(0);
      offset += global_total_size;
    } else {
      // 如果有 Global Refactor，Local Layers 紧跟在 Global Data 之后
      offset = global_total_size;
    }

    // Local coefficient layers (stored back to front: L-1, L-2, ..., 0)
    for (int l = this->L - 1; l >= 0; --l) {
      layer_len.push_back(local_coeff_size[l]);
      layer_off.push_back(offset);
      offset += local_coeff_size[l];
    }
  }

  // ===========================================================================
  // Error Budget Allocation
  // ===========================================================================
  void CalcQuantizers(T* quantizers, enum error_bound_type type, T tol, T s,
                      T norm, bool reciprocal) {
    if (s == std::numeric_limits<T>::infinity()) {
      double abs_tol = tol;
      if (type == error_bound_type::REL) {
        abs_tol *= norm;
      }
      abs_tol *= 2;

      double C = (1 + std::pow(3, D));

      // 1. Fill Global Quantizers (Indices 0 to M)
      // MGARD standard kernel accesses quantizers by level index (0 is finest,
      // l_target is coarsest)
      if (this->M > 0) {
        // Global Coarsest (Level 0 in MGARD logic usually, check level_marks
        // definition) Usually: Level 0 = Finest, Level l_target = Coarsest
        // Let's assume standard MGARD order: 0...l_target

        // 我们需要填充 global_hierarchy->l_target() + 1 个 entries
        for (int m = 0; m <= global_hierarchy->l_target(); ++m) {
          // Propagation depth calculation
          // Coarsest (m = l_target) has deepest propagation
          // Finest (m = 0) has shallowest in global, but sits on top of Local

          // Note: This logic depends on how you want to distribute error.
          // Current assumption: Simple Uniform for safety, or Depth based.
          // Let's use Depth based.

          // Depth of Global Level 'm':
          // Distance from finest global (0) to m is m.
          // Distance from m to coarsest global is (M - m).
          // Plus L local levels underneath.

          // Strictness should increase with depth (m increasing towards
          // coarsest) Depth = (global_hierarchy->l_target() - m) + L + 1 ?? NO,
          // typically Coarsest needs highest accuracy.

          // Let's stick to a safe Uniform distribution weighted by Total Layers
          // for now to ensure bound is met, then you can tune.
          SIZE total_depth = this->M + 1 + this->L;
          quantizers[m] = abs_tol / (total_depth * C);
        }
      } else if (this->L > 0) {
        // M=0, Index 0 is Local Coarsest
        quantizers[0] = abs_tol / ((this->L + 1) * C);
      }

      // 2. Fill Local Quantizers
      // Offset in quantizers array
      SIZE q_offset = (this->M > 0) ? (global_hierarchy->l_target() + 1) : 1;

      // Local coeffs processed L-1 down to 0
      for (int l = this->L - 1; l >= 0; --l) {
        SIZE depth = (this->L - l);  // 1 to L
        // Adjust for global layers on top if any? No, local is bottom.
        SIZE total_depth_factor = (this->L + 1 + this->M);

        quantizers[q_offset] = abs_tol / (total_depth_factor * C);
        q_offset++;
      }

      if (reciprocal) {
        SIZE total_entries = q_offset;
        for (SIZE i = 0; i < total_entries; ++i) {
          quantizers[i] = 1.0 / quantizers[i];
        }
      }
    } else {
      // L2 norm: different error propagation (quadratic accumulation)
      log::err(
          "L2 norm (s != inf) not yet supported in HybridHierarchyQuantizer");
      exit(-1);
    }
  }

  static size_t EstimateMemoryFootprint(std::vector<SIZE> shape,
                                        Config config) {
    size_t size = 0;
    // Quantizer array storage
    SIZE L = config.num_local_refactoring_level;
    SIZE M = config.num_global_refactoring_level;
    size += sizeof(T) * (L + M + 2);
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
    
    // Allocation for quantizers
    SIZE global_q_size = (this->M > 0) ? (global_hierarchy->l_target() + 1) : 0;
    SIZE total_q_size = global_q_size + this->L + (this->M == 0 ? 1 : 0);
    T* host_quantizers = new T[total_q_size];

    CalcQuantizers(host_quantizers, ebtype, tol, s, norm, true);

    Timer timer;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }

    // --- PART 1: GLOBAL QUANTIZATION (Spatial / Interleaved) ---
    if (this->M > 0) {
        // Cast the linear start of the array to N-Dim Global Shape
        std::vector<SIZE> global_shape = global_hierarchy->level_shape(global_hierarchy->l_target());
        SubArray<D, T, DeviceType> global_data_v(global_shape, original_data.data());
        SubArray<D, Q, DeviceType> global_data_q(global_shape, quantized_data.data());

        // Prepare metadata for standard MGARD Kernel
        SubArray<2, SIZE, DeviceType> level_ranges = global_hierarchy->level_ranges();
        SubArray<2, int, DeviceType> level_marks = global_hierarchy->level_marks();
        SubArray<3, T, DeviceType> level_volumes = global_hierarchy->level_volumes(false);
        
        // Copy Global Quantizers to GPU
        Array<1, T, DeviceType> global_quantizers_arr({global_q_size});
        MemoryManager<DeviceType>::Copy1D(global_quantizers_arr.data(), 
                                          host_quantizers, 
                                          global_q_size, queue_idx);
        SubArray<1, T, DeviceType> global_quantizers_sub(global_quantizers_arr);

        bool calc_vol = (s != std::numeric_limits<T>::infinity());

        // Launch standard Spatial Kernel
        DeviceLauncher<DeviceType>::Execute(
            LevelwiseLinearQuantizerKernel<D, T, MGARDX_QUANTIZE, DeviceType>(
                level_ranges, level_marks, global_hierarchy->l_target(),
                global_quantizers_sub, level_volumes, calc_vol,
                global_data_v, global_data_q),
            queue_idx);
    } 

    // --- PART 2: LOCAL QUANTIZATION (Linear) ---
    // Identify where Local processing starts
    // If M=0, layer 0 is Coarsest (handled as linear). 
    // If M>0, layers start after Global Data.
    SIZE local_start_idx = (this->M > 0) ? 0 : 0; 
    SIZE quantizer_start_idx = (this->M > 0) ? global_q_size : 0;

    for (SIZE i = local_start_idx; i < layer_len.size(); ++i) {
        T q = host_quantizers[quantizer_start_idx + i];
        
        SubArray<1, T, DeviceType> v_in({layer_len[i]}, original_data.data() + layer_off[i]);
        SubArray<1, Q, DeviceType> qv({layer_len[i]}, quantized_data.data() + layer_off[i]);

        DeviceLauncher<DeviceType>::Execute(
            HybridQuantizeKernel<D, T, Q, MGARDX_QUANTIZE, DeviceType>(q, v_in, qv),
            queue_idx);
    }

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("Hybrid Quantization",
                  hierarchy->total_num_elems() * sizeof(T));
      timer.clear();
    }

    delete[] host_quantizers;
  }

  template <typename LosslessCompressorType>
  void Dequantize(SubArray<1, T, DeviceType> original_data,
                  enum error_bound_type ebtype, T tol, T s, T norm,
                  SubArray<1, Q, DeviceType> quantized_data,
                  LosslessCompressorType& lossless, int queue_idx) {
    
    SIZE global_q_size = (this->M > 0) ? (global_hierarchy->l_target() + 1) : 0;
    SIZE total_q_size = global_q_size + this->L + (this->M == 0 ? 1 : 0);
    T* host_quantizers = new T[total_q_size];

    CalcQuantizers(host_quantizers, ebtype, tol, s, norm, false);

    Timer timer;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }

    // --- PART 1: GLOBAL DEQUANTIZATION ---
    if (this->M > 0) {
        std::vector<SIZE> global_shape = global_hierarchy->level_shape(global_hierarchy->l_target());
        SubArray<D, T, DeviceType> global_data_v(global_shape, original_data.data());
        SubArray<D, Q, DeviceType> global_data_q(global_shape, quantized_data.data());

        SubArray<2, SIZE, DeviceType> level_ranges = global_hierarchy->level_ranges();
        SubArray<2, int, DeviceType> level_marks = global_hierarchy->level_marks();
        SubArray<3, T, DeviceType> level_volumes = global_hierarchy->level_volumes(true); // true for dequantize (usually)
        
        Array<1, T, DeviceType> global_quantizers_arr({global_q_size});
        MemoryManager<DeviceType>::Copy1D(global_quantizers_arr.data(), 
                                          host_quantizers, 
                                          global_q_size, queue_idx);
        SubArray<1, T, DeviceType> global_quantizers_sub(global_quantizers_arr);

        bool calc_vol = (s != std::numeric_limits<T>::infinity());

        DeviceLauncher<DeviceType>::Execute(
            LevelwiseLinearQuantizerKernel<D, T, MGARDX_DEQUANTIZE, DeviceType>(
                level_ranges, level_marks, global_hierarchy->l_target(),
                global_quantizers_sub, level_volumes, calc_vol,
                global_data_v, global_data_q),
            queue_idx);
    }

    // --- PART 2: LOCAL DEQUANTIZATION ---
    SIZE local_start_idx = (this->M > 0) ? 0 : 0; 
    SIZE quantizer_start_idx = (this->M > 0) ? global_q_size : 0;

    for (SIZE i = local_start_idx; i < layer_len.size(); ++i) {
        T q = host_quantizers[quantizer_start_idx + i];
        SubArray<1, T, DeviceType> v_in({layer_len[i]}, original_data.data() + layer_off[i]);
        SubArray<1, Q, DeviceType> qv({layer_len[i]}, quantized_data.data() + layer_off[i]);

        DeviceLauncher<DeviceType>::Execute(
            HybridQuantizeKernel<D, T, Q, MGARDX_DEQUANTIZE, DeviceType>(q, v_in, qv),
            queue_idx);
    }

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("Hybrid Dequantization",
                  hierarchy->total_num_elems() * sizeof(T));
      timer.clear();
    }

    delete[] host_quantizers;
  }

  bool initialized;
  SIZE L;  // Number of local levels
  SIZE M;  // Number of global levels

  Hierarchy<D, T, DeviceType>* hierarchy;
  Hierarchy<D, T, DeviceType>* global_hierarchy;
  Config config;

  // Local level info
  std::vector<SIZE> coarse_shape;
  std::vector<SIZE> coarse_num_elems;
  std::vector<SIZE> local_coeff_size;

  // Global level info
  SIZE global_total_size;

  std::vector<SIZE> layer_len;
  std::vector<SIZE> layer_off;
};

}  // namespace mgard_x

#endif