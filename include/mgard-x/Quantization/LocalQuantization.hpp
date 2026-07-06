#ifndef MGARD_X_LOCAL_QUANTIZATION_TEMPLATE
#define MGARD_X_LOCAL_QUANTIZATION_TEMPLATE

#include "../RuntimeX/RuntimeX.h"
#include "QuantizationInterface.hpp"

namespace mgard_x {

#define MGARDX_QUANTIZE 1
#define MGARDX_DEQUANTIZE 2

// Non-ROI Version
template <typename T, typename Q, OPTION OP, typename DeviceType>
class QuantizeLocalLevelFunctor : public Functor<DeviceType> {
 public:
  MGARDX_EXEC QuantizeLocalLevelFunctor() {}
  MGARDX_EXEC QuantizeLocalLevelFunctor(T quantizer,
                                        SubArray<1, T, DeviceType> v,
                                        SubArray<1, Q, DeviceType> quantized_v,
                                        bool prep_huffman, SIZE dict_size)
      : quantizer(quantizer), v(v), quantized_v(quantized_v),
        prep_huffman(prep_huffman), dict_size(dict_size) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    idx = FunctorBase<DeviceType>::GetBlockIdX() *
              FunctorBase<DeviceType>::GetBlockDimX() +
          FunctorBase<DeviceType>::GetThreadIdX();

    if (idx < v.shape(0)) {
      T t = *v(idx);
      Q quantized_data;
      T volume = 1;

      if constexpr (OP == MGARDX_QUANTIZE) {
        if constexpr (sizeof(T) == sizeof(double)) {
          quantized_data = copysign((T)0.5 + fabs(t * quantizer * volume), t);
        } else if constexpr (sizeof(T) == sizeof(float)) {
          quantized_data = copysign((T)0.5 + fabsf(t * quantizer * volume), t);
        }
        // Fold the Huffman dictionary shift into quantization (mirrors
        // LevelwiseLinearQuantizerNDFunctor): the lossless stage expects
        // values in the non-negative dictionary range and separates
        // out-of-range entries as outliers.
        if (prep_huffman) {
          quantized_data += dict_size / 2;
        }
        *quantized_v(idx) = quantized_data;
      } else if constexpr (OP == MGARDX_DEQUANTIZE) {
        quantized_data = *quantized_v(idx);
        if (prep_huffman) {
          quantized_data -= dict_size / 2;
        }
        *v(idx) = (quantizer * volume) * (T)quantized_data;
      }
    }
  }

  MGARDX_CONT size_t shared_memory_size() { return 0; }

 private:
  SIZE idx;
  T quantizer;
  SubArray<1, T, DeviceType> v;
  SubArray<1, Q, DeviceType> quantized_v;
  bool prep_huffman;
  SIZE dict_size;
};

template <typename T, typename Q, OPTION OP, typename DeviceType>
class QuantizeLocalLevelKernel : public Kernel {
 public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "lvl_qk";

  MGARDX_CONT
  QuantizeLocalLevelKernel(T quantizer, SubArray<1, T, DeviceType> v,
                           SubArray<1, Q, DeviceType> quantized_v,
                           bool prep_huffman, SIZE dict_size)
      : quantizer(quantizer), v(v), quantized_v(quantized_v),
        prep_huffman(prep_huffman), dict_size(dict_size) {}

  MGARDX_CONT Task<QuantizeLocalLevelFunctor<T, Q, OP, DeviceType>> GenTask(
      int queue_idx) {
    using FunctorType = QuantizeLocalLevelFunctor<T, Q, OP, DeviceType>;
    FunctorType functor(quantizer, v, quantized_v, prep_huffman, dict_size);

    SIZE tbx = 256, tby = 1, tbz = 1;
    SIZE gridx = (v.shape(0) + tbx - 1) / tbx;
    SIZE gridy = 1, gridz = 1;

    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, 0, queue_idx,
                std::string(Name));
  }

 private:
  T quantizer;
  SubArray<1, T, DeviceType> v;
  SubArray<1, Q, DeviceType> quantized_v;
  bool prep_huffman;
  SIZE dict_size;
};

// ROI Version
template <typename T, typename Q, OPTION OP, typename DeviceType>
class QuantizeLocalLevelROIFunctor : public Functor<DeviceType> {
 public:
  MGARDX_EXEC QuantizeLocalLevelROIFunctor() {}
  MGARDX_EXEC QuantizeLocalLevelROIFunctor(
      SubArray<1, T, DeviceType> quantizers, SubArray<1, T, DeviceType> v,
      SubArray<1, Q, DeviceType> quantized_v, bool prep_huffman,
      SIZE dict_size)
      : quantizers(quantizers), v(v), quantized_v(quantized_v),
        prep_huffman(prep_huffman), dict_size(dict_size) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    idx = FunctorBase<DeviceType>::GetBlockIdX() *
              FunctorBase<DeviceType>::GetBlockDimX() +
          FunctorBase<DeviceType>::GetThreadIdX();

    if (idx < v.shape(0)) {
      SIZE block_idx = idx / 387;

      T quantizer = *quantizers(block_idx);

      T t = *v(idx);
      Q quantized_data;
      T volume = 1;

      if constexpr (OP == MGARDX_QUANTIZE) {
        if constexpr (sizeof(T) == sizeof(double)) {
          quantized_data = copysign((T)0.5 + fabs(t * quantizer * volume), t);
        } else if constexpr (sizeof(T) == sizeof(float)) {
          quantized_data = copysign((T)0.5 + fabsf(t * quantizer * volume), t);
        }
        if (prep_huffman) {
          quantized_data += dict_size / 2;
        }
        *quantized_v(idx) = quantized_data;
      } else if constexpr (OP == MGARDX_DEQUANTIZE) {
        quantized_data = *quantized_v(idx);
        if (prep_huffman) {
          quantized_data -= dict_size / 2;
        }
        *v(idx) = (quantizer * volume) * (T)quantized_data;
      }
    }
  }

  MGARDX_CONT size_t shared_memory_size() { return 0; }

 private:
  SIZE idx;
  SubArray<1, T, DeviceType> quantizers;
  SubArray<1, T, DeviceType> v;
  SubArray<1, Q, DeviceType> quantized_v;
  bool prep_huffman;
  SIZE dict_size;
};

template <typename T, typename Q, OPTION OP, typename DeviceType>
class QuantizeLocalLevelROIKernel : public Kernel {
 public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "lvl_qk_roi";

  MGARDX_CONT
  QuantizeLocalLevelROIKernel(SubArray<1, T, DeviceType> quantizers,
                              SubArray<1, T, DeviceType> v,
                              SubArray<1, Q, DeviceType> quantized_v,
                              bool prep_huffman, SIZE dict_size)
      : quantizers(quantizers), v(v), quantized_v(quantized_v),
        prep_huffman(prep_huffman), dict_size(dict_size) {}

  MGARDX_CONT Task<QuantizeLocalLevelROIFunctor<T, Q, OP, DeviceType>> GenTask(
      int queue_idx) {
    using FunctorType = QuantizeLocalLevelROIFunctor<T, Q, OP, DeviceType>;
    FunctorType functor(quantizers, v, quantized_v, prep_huffman, dict_size);

    SIZE tbx = 256, tby = 1, tbz = 1;
    SIZE gridx = (v.shape(0) + tbx - 1) / tbx;
    SIZE gridy = 1, gridz = 1;

    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, 0, queue_idx,
                std::string(Name));
  }

 private:
  SubArray<1, T, DeviceType> quantizers;
  SubArray<1, T, DeviceType> v;
  SubArray<1, Q, DeviceType> quantized_v;
  bool prep_huffman;
  SIZE dict_size;
};

template <DIM D, typename T, typename Q, typename DeviceType>
class LocalQuantizer : public QuantizationInterface<D, T, Q, DeviceType> {
 public:
  LocalQuantizer() : initialized(false) {}
  LocalQuantizer(Hierarchy<D, T, DeviceType>& hierarchy, Config config)
      : initialized(true), hierarchy(&hierarchy), config(config) {
    this->L = config.num_local_refactoring_level;
    this->M = config.num_global_refactoring_level;
    compute_local_ranges();
    prepare_layers();
  }

  // Add logic to determine if roi or not
  void Adapt(Hierarchy<D, T, DeviceType>& hierarchy, Config config,
             int queue_idx) {
    this->initialized = true;
    this->hierarchy = &hierarchy;
    this->config = config;
    this->L = config.num_local_refactoring_level;
    this->M = config.num_global_refactoring_level;
    compute_local_ranges();
    prepare_layers();
  }

  static size_t EstimateMemoryFootprint(std::vector<SIZE> shape) {
    size_t size = 0;
    return size;
  }

  void compute_local_ranges() {
    coarse_shape = hierarchy->level_shape(hierarchy->l_target());

    fine_num_elems.clear();
    coarse_num_elems.clear();
    local_coeff_size.clear();

    for (int l = 0; l < this->L; ++l) {
      SIZE last_level_size = 1, curr_level_size = 1;
      for (DIM d = 0; d < D; ++d) {
        coarse_shape[d] = ((coarse_shape[d] - 1) / 8 + 1) * 8;
        last_level_size *= coarse_shape[d];
        coarse_shape[d] = ((coarse_shape[d] - 1) / 8 + 1) * 5;
        curr_level_size *= coarse_shape[d];
      }
      fine_num_elems.push_back(last_level_size);
      coarse_num_elems.push_back(curr_level_size);
      local_coeff_size.push_back(last_level_size - curr_level_size);
    }
  }

  void prepare_layers() {
    if (this->L == 0) {
      layer_len.clear();
      layer_off.clear();
      return;
    }

    layer_len.assign(this->L + 1, 0);
    layer_off.assign(this->L + 1, 0);

    // The length of coarsest layer (level 0)
    layer_len[0] = coarse_num_elems[this->L-1];
    layer_off[0] = 0;

    SIZE accum = layer_len[0];

    for (SIZE l = 1; l <= this->L; ++l) {
      layer_len[l] = local_coeff_size[this->L - l];
      layer_off[l] = accum;
      accum += layer_len[l];
    }
  }

  // Calculate quantizers between levels(Used in Non-ROI)
  void CalcQuantizers(size_t dof, T* quantizers, enum error_bound_type type,
                      T tol, T s, T norm, SIZE l_target,
                      enum decomposition_type decomposition, bool reciprocal) {
    double abs_tol = tol;
    if (type == error_bound_type::REL) {
      abs_tol *= norm;
    }
    abs_tol *= 2;

    if (s == std::numeric_limits<T>::infinity()) {
      double C = (1 + std::pow(3, D));

      for (int l = 0; l <= l_target; l++) {
        // Modified here
        quantizers[l] = (abs_tol) / (std::pow(2, l + 1) * C);

        if (reciprocal) {
          quantizers[l] = 1.0f / quantizers[l];
        }
      }
    } else {
      throw ProcessingException("Only L-inf supported");
    }
  }

  void Quantize(SubArray<D, T, DeviceType> original_data,
                enum error_bound_type ebtype, T tol, T s, T norm,
                SubArray<D, Q, DeviceType> quantized_data, int queue_idx) {}

  void Dequantize(SubArray<D, T, DeviceType> original_data,
                  enum error_bound_type ebtype, T tol, T s, T norm,
                  SubArray<D, Q, DeviceType> quantized_data, int queue_idx) {}

  // Non-ROI
  template <typename LosslessCompressorType>
  void Quantize(SubArray<1, T, DeviceType> original_data,
                enum error_bound_type ebtype, T tol, T s, T norm,
                SubArray<1, Q, DeviceType> quantized_data,
                LosslessCompressorType& lossless, int queue_idx) {
    T* host_quantizers = new T[this->L + 1];
    CalcQuantizers(hierarchy->total_num_elems(), host_quantizers, ebtype, tol,
                   s, norm, this->L, config.decomposition, true);
    bool prep_huffman = config.lossless != lossless_type::CPU_Lossless &&
                        config.lossless != lossless_type::BlockDelta &&
                        config.lossless != lossless_type::LZ4;
    SIZE huff_dict_size = config.huff_dict_size;

    Timer timer;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }

    SIZE start_level = (this->M > 0) ? 1 : 0;
    SIZE offset_adjustment = (this->M > 0) ? layer_off[1] : 0;

    for (SIZE l = start_level; l <= this->L; ++l) {
      SIZE adjusted_off = layer_off[l] - offset_adjustment;
      SubArray<1, T, DeviceType> v_in({layer_len[l]},
                                      original_data((IDX)adjusted_off));
      SubArray<1, Q, DeviceType> qv({layer_len[l]},
                                    quantized_data((IDX)adjusted_off));
      // Launch
      T quantizer = host_quantizers[l];
      DeviceLauncher<DeviceType>::Execute(
          QuantizeLocalLevelKernel<T, Q, MGARDX_QUANTIZE, DeviceType>(
              quantizer, v_in, qv, prep_huffman, huff_dict_size),
          queue_idx);
    }

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("Local Quantization",
                  hierarchy->total_num_elems() * sizeof(T));
      timer.clear();
    }

    delete[] host_quantizers;
  }

  // Non-ROI
  template <typename LosslessCompressorType>
  void Dequantize(SubArray<1, T, DeviceType> original_data,
                  enum error_bound_type ebtype, T tol, T s, T norm,
                  SubArray<1, Q, DeviceType> quantized_data,
                  LosslessCompressorType& lossless, int queue_idx) {
    T* host_quantizers = new T[this->L + 1];
    CalcQuantizers(hierarchy->total_num_elems(), host_quantizers, ebtype, tol,
                   s, norm, this->L, config.decomposition, false);
    bool prep_huffman = config.lossless != lossless_type::CPU_Lossless &&
                        config.lossless != lossless_type::BlockDelta &&
                        config.lossless != lossless_type::LZ4;
    SIZE huff_dict_size = config.huff_dict_size;

    Timer timer;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }

    SIZE start_level = (this->M > 0) ? 1 : 0;
    SIZE offset_adjustment = (this->M > 0) ? layer_off[1] : 0;

    for (SIZE l = start_level; l <= this->L; ++l) {
      SIZE adjusted_off = layer_off[l] - offset_adjustment;
      SubArray<1, T, DeviceType> v_in({layer_len[l]},
                                      original_data((IDX)adjusted_off));
      SubArray<1, Q, DeviceType> qv({layer_len[l]},
                                    quantized_data((IDX)adjusted_off));
      // Launch
      T quantizer = host_quantizers[l];
      DeviceLauncher<DeviceType>::Execute(
          QuantizeLocalLevelKernel<T, Q, MGARDX_DEQUANTIZE, DeviceType>(
              quantizer, v_in, qv, prep_huffman, huff_dict_size),
          queue_idx);
    }

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("Local Dequantization",
                  hierarchy->total_num_elems() * sizeof(T));
      timer.clear();
    }

    delete[] host_quantizers;
  }

  // With ROI
  template <typename LosslessCompressorType>
  void Quantize(SubArray<1, T, DeviceType> original_data,
                enum error_bound_type ebtype, double tol, T s, T norm,
                SubArray<1, Q, DeviceType> quantized_data,
                const std::vector<double>& roi_tolerance_map,
                const std::vector<SIZE>& level_offsets,
                const std::vector<SIZE>& level_block_counts,
                LosslessCompressorType& lossless, int queue_idx) {
    if (s != std::numeric_limits<T>::infinity()) {
      throw ProcessingException("Only L-inf supported");
    }

    double C = (1 + std::pow(3, D));
    bool prep_huffman = config.lossless != lossless_type::CPU_Lossless &&
                        config.lossless != lossless_type::BlockDelta &&
                        config.lossless != lossless_type::LZ4;
    SIZE huff_dict_size = config.huff_dict_size;

    Timer timer;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }

    SIZE accumulated_coeff_size = 0;
    // Process Layer 1 to Layer L with ROI tolerances
    for (SIZE l = 0; l < this->L; ++l) {
      SIZE roi_level = l;
      SIZE level_offset = level_offsets[roi_level];
      SIZE num_blocks = level_block_counts[roi_level];

      // Pre-compute quantizers for all blocks in this layer
      std::vector<T> host_quantizers(num_blocks);
      for (SIZE b = 0; b < num_blocks; ++b) {
        double block_tol = roi_tolerance_map[level_offset + b];
        if (ebtype == error_bound_type::REL) {
          block_tol *= norm;
        }
        block_tol *= 2;

        // l=0 is finest coefficients (laid out at the end of the data array),
        // which maps to non-ROI layer L. The correct exponent is (L - l + 1).
        T block_quantizer = block_tol / (std::pow(2, this->L - l + 1) * C);

        // reciprocal for quantization
        host_quantizers[b] = 1.0 / block_quantizer;
      }

      // Copy to device
      Array<1, T, DeviceType> device_quantizers({num_blocks});
      device_quantizers.load(host_quantizers.data(), 0, queue_idx);

      accumulated_coeff_size += local_coeff_size[l];
      SubArray<1, T, DeviceType> v_in(
          {local_coeff_size[l]},
          original_data(original_data.shape(0) - accumulated_coeff_size));
      SubArray<1, Q, DeviceType> qv(
          {local_coeff_size[l]},
          quantized_data(quantized_data.shape(0) - accumulated_coeff_size));

      DeviceLauncher<DeviceType>::Execute(
          QuantizeLocalLevelROIKernel<T, Q, MGARDX_QUANTIZE, DeviceType>(
              SubArray<1, T, DeviceType>(device_quantizers), v_in, qv,
              prep_huffman, huff_dict_size),
          queue_idx);
    }

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("Local Quantization with ROI",
                  hierarchy->total_num_elems() * sizeof(T));
      timer.clear();
    }
  }

  // With ROI
  template <typename LosslessCompressorType>
  void Dequantize(SubArray<1, T, DeviceType> original_data,
                  enum error_bound_type ebtype, double tol, T s, T norm,
                  SubArray<1, Q, DeviceType> quantized_data,
                  const std::vector<double>& roi_tolerance_map,
                  const std::vector<SIZE>& level_offsets,
                  const std::vector<SIZE>& level_block_counts,
                  LosslessCompressorType& lossless, int queue_idx) {
    if (s != std::numeric_limits<T>::infinity()) {
      throw ProcessingException("Only L-inf supported");
    }

    double C = (1 + std::pow(3, D));
    bool prep_huffman = config.lossless != lossless_type::CPU_Lossless &&
                        config.lossless != lossless_type::BlockDelta &&
                        config.lossless != lossless_type::LZ4;
    SIZE huff_dict_size = config.huff_dict_size;

    Timer timer;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }

    SIZE accumulated_coeff_size = 0;
    // Process Layer 1 to Layer L with ROI tolerances
    for (SIZE l = 0; l < this->L; ++l) {
      SIZE roi_level = l;
      SIZE level_offset = level_offsets[roi_level];
      SIZE num_blocks = level_block_counts[roi_level];

      // Pre-compute quantizers for all blocks in this layer
      std::vector<T> host_quantizers(num_blocks);
      for (SIZE b = 0; b < num_blocks; ++b) {
        double block_tol = roi_tolerance_map[level_offset + b];
        if (ebtype == error_bound_type::REL) {
          block_tol *= norm;
        }
        block_tol *= 2;

        // l=0 is finest coefficients (laid out at the end of the data array),
        // which maps to non-ROI layer L. The correct exponent is (L - l + 1).
        T block_quantizer = block_tol / (std::pow(2, this->L - l + 1) * C);

        // no reciprocal for dequantization
        host_quantizers[b] = block_quantizer;
      }

      // Copy to device
      Array<1, T, DeviceType> device_quantizers({num_blocks});
      device_quantizers.load(host_quantizers.data(), 0, queue_idx);

      accumulated_coeff_size += local_coeff_size[l];
      SubArray<1, T, DeviceType> v_in(
          {local_coeff_size[l]},
          original_data(original_data.shape(0) - accumulated_coeff_size));
      SubArray<1, Q, DeviceType> qv(
          {local_coeff_size[l]},
          quantized_data(quantized_data.shape(0) - accumulated_coeff_size));

      DeviceLauncher<DeviceType>::Execute(
          QuantizeLocalLevelROIKernel<T, Q, MGARDX_DEQUANTIZE, DeviceType>(
              SubArray<1, T, DeviceType>(device_quantizers), v_in, qv,
              prep_huffman, huff_dict_size),
          queue_idx);
    }

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("Local Dequantization with ROI",
                  hierarchy->total_num_elems() * sizeof(T));
      timer.clear();
    }
  }

  bool initialized;
  SIZE L;
  SIZE M;
  Hierarchy<D, T, DeviceType>* hierarchy;
  Config config;

  // For Non-ROI
  std::vector<SIZE> layer_len;
  std::vector<SIZE> layer_off;

  // For ROI
  std::vector<double> tol_table;

  std::vector<SIZE> fine_num_elems;
  std::vector<SIZE> coarse_num_elems;
  std::vector<SIZE> local_coeff_size;
  std::vector<SIZE> coarse_shape;
};

}  // namespace mgard_x

#endif