#ifndef MGARD_X_LOCAL_QUANTIZATION_TEMPLATE
#define MGARD_X_LOCAL_QUANTIZATION_TEMPLATE

#include "../RuntimeX/RuntimeX.h"
#include "QuantizationInterface.hpp"

namespace mgard_x {

#define MGARDX_QUANTIZE 1
#define MGARDX_DEQUANTIZE 2

template <typename T, typename Q, OPTION OP, typename DeviceType>
class QuantizeLocalLevelFunctor : public Functor<DeviceType> {
 public:
  MGARDX_EXEC QuantizeLocalLevelFunctor() {}
  MGARDX_EXEC QuantizeLocalLevelFunctor(T quantizer,
                                        SubArray<1, T, DeviceType> v,
                                        SubArray<1, Q, DeviceType> quantized_v)
      : quantizer(quantizer), v(v), quantized_v(quantized_v) {
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
        *quantized_v(idx) = quantized_data;
      } else if constexpr (OP == MGARDX_DEQUANTIZE) {
        quantized_data = *quantized_v(idx);
        *v(idx) = (quantizer * volume) * (T)quantized_data;
      }
    }
  }

  MGARDX_CONT size_t shared_memory_size() {
    size_t size = 0;
    return size;
  }

 private:
  SIZE idx;
  T quantizer;
  SubArray<1, T, DeviceType> v;
  SubArray<1, Q, DeviceType> quantized_v;
};

template <typename T, typename Q, OPTION OP, typename DeviceType>
class QuantizeLocalLevelKernel : public Kernel {
 public:
  // Not sure if needed for auto-tuning
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "lvl_qk";
  MGARDX_CONT
  QuantizeLocalLevelKernel(T quantizer, SubArray<1, T, DeviceType> v,
                           SubArray<1, Q, DeviceType> quantized_v)
      : quantizer(quantizer), v(v), quantized_v(quantized_v) {}
  MGARDX_CONT Task<QuantizeLocalLevelFunctor<T, Q, OP, DeviceType>> GenTask(
      int queue_idx) {
    using FunctorType = QuantizeLocalLevelFunctor<T, Q, OP, DeviceType>;
    FunctorType functor(quantizer, v, quantized_v);

    SIZE total_thread_z = 1;
    SIZE total_thread_y = 1;
    SIZE total_thread_x = v.shape(0);

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbz = 1;
    tby = 1;
    tbx = 256;
    gridz = ceil((double)total_thread_z / tbz);
    gridy = ceil((double)total_thread_y / tby);
    gridx = ceil((double)total_thread_x / tbx);

    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                std::string(Name));
  }

 private:
  T quantizer;
  SubArray<1, T, DeviceType> v;
  SubArray<1, Q, DeviceType> quantized_v;
};

template <DIM D, typename T, typename Q, typename DeviceType>
class LocalQuantizer : public QuantizationInterface<D, T, Q, DeviceType> {
 public:
  LocalQuantizer() : initialized(false) {}
  LocalQuantizer(Hierarchy<D, T, DeviceType>& hierarchy, Config config)
      : initialized(true), hierarchy(&hierarchy), config(config) {
    compute_local_ranges();
    prepare_layers();
  }

  void Adapt(Hierarchy<D, T, DeviceType>& hierarchy, Config config,
             int queue_idx) {
    this->initialized = true;
    this->hierarchy = &hierarchy;
    this->config = config;
    compute_local_ranges();
    layer_len.clear();
    layer_off.clear();
    prepare_layers();
  }

  static size_t EstimateMemoryFootprint(std::vector<SIZE> shape) {
    size_t size = 0;
    return size;
  }

  void compute_local_ranges() {
    SIZE L = config.num_local_refactoring_level;
    coarse_shape = hierarchy->level_shape(hierarchy->l_target());

    coarse_num_elems.clear();
    local_coeff_size.clear();

    for (int l = 0; l < L; ++l) {
      SIZE last_level_size = 1, curr_level_size = 1;
      for (DIM d = 0; d < D; ++d) {
        last_level_size *= coarse_shape[d];
        coarse_shape[d] = ((coarse_shape[d] - 1) / 8 + 1) * 5;
        curr_level_size *= coarse_shape[d];
      }
      coarse_num_elems.push_back(last_level_size);
      local_coeff_size.push_back(last_level_size - curr_level_size);
    }
  }

  void prepare_layers() {
    SIZE L = config.num_local_refactoring_level;
    layer_len.assign(L + 1, 0);
    layer_off.assign(L + 1, 0);

    // The length of coarsest layer
    layer_len[0] = coarse_num_elems.back();
    layer_off[0] = 0;

    SIZE accum = layer_len[0];

    for (SIZE l = 1; l <= L; ++l) {
      layer_len[l] = local_coeff_size[l - 1];
      layer_off[l] = accum;
      accum += layer_len[l];
    }
  }

  void CalcQuantizers(size_t dof, T* quantizers, enum error_bound_type type,
                      T tol, T s, T norm, SIZE l_target,
                      enum decomposition_type decomposition, bool reciprocal) {
    double abs_tol = tol;
    if (type == error_bound_type::REL) {
      abs_tol *= norm;
    }
    abs_tol *= 2;
    if (s == std::numeric_limits<T>::infinity()) {
      // Use ben's quantizer for now
      for (int l = 0; l < l_target + 1; l++) {
        quantizers[l] = (abs_tol) / (l_target + 1) * (1 + std::pow(3, D));
        if (reciprocal) {
          quantizers[l] = 1.0f / quantizers[l];
        }
      }
    } else {
      // warning for un-inf
      
    }
  }

  void Quantize(SubArray<D, T, DeviceType> original_data,
                enum error_bound_type ebtype, T tol, T s, T norm,
                SubArray<D, Q, DeviceType> quantized_data,
                int queue_idx){}

  void Dequantize(SubArray<D, T, DeviceType> original_data,
                  enum error_bound_type ebtype, T tol, T s, T norm,
                  SubArray<D, Q, DeviceType> quantized_data,
                  int queue_idx){}

  template <typename LosslessCompressorType>
  void Quantize(SubArray<1, T, DeviceType> original_data,
                enum error_bound_type ebtype, T tol, T s, T norm,
                SubArray<1, Q, DeviceType> quantized_data,
                LosslessCompressorType& lossless, int queue_idx){
    SIZE L = hierarchy->l_target();
    std::vector<T> quantizers_buf(L + 1);
    CalcQuantizers(hierarchy->total_num_elems(), quantizers_buf.data(), ebtype,
                   tol, s, norm, L, config.decomposition, true);
    SubArray<1,T,DeviceType> quantizers_array({L+1},quantizers_buf.data());

    for (SIZE l = 0; l <= L; ++l) {
      SubArray<1, T, DeviceType> v_in({layer_len[l]},
                                      original_data.data() + layer_off[l]);
      SubArray<1, Q, DeviceType> v_out = quantized_data;
      SubArray<1, QUANTIZED_INT, DeviceType> qv = quantized_data;
      // Launch
      T quantizer = *quantizers_array(l);
      DeviceLauncher<DeviceType>::Execute(
          QuantizeLocalLevelKernel<T, Q, MGARDX_QUANTIZE, DeviceType>(
              quantizer, v_in, qv),
          queue_idx);
    }
  }

  template <typename LosslessCompressorType>
  void Dequantize(SubArray<1, T, DeviceType> original_data,
                  enum error_bound_type ebtype, T tol, T s, T norm,
                  SubArray<1, Q, DeviceType> quantized_data,
                  LosslessCompressorType& lossless, int queue_idx){
    SIZE L = hierarchy->l_target();
    std::vector<T> quantizers_buf(L + 1);
    CalcQuantizers(hierarchy->total_num_elems(), quantizers_buf.data(), ebtype,
                   tol, s, norm, L, config.decomposition, true);
    SubArray<1,T,DeviceType> quantizers_array({L+1},quantizers_buf.data());
    

    for (SIZE l = 0; l <= L; ++l) {
      SubArray<1, T, DeviceType> v_in({layer_len[l]},
                                      original_data.data() + layer_off[l]);
      SubArray<1, QUANTIZED_INT, DeviceType> qv = quantized_data;
      // Launch
      T quantizer = *quantizers_array(l);
      DeviceLauncher<DeviceType>::Execute(
          QuantizeLocalLevelKernel<T, Q, MGARDX_DEQUANTIZE, DeviceType>(
              quantizer, v_in, qv),
          queue_idx);
    }
  }

  bool initialized;
  Hierarchy<D, T, DeviceType>* hierarchy;
  Config config;
  std::vector<SIZE> layer_len;
  // change off to offset
  std::vector<SIZE> layer_off;

  std::vector<SIZE> coarse_num_elems;
  std::vector<SIZE> local_coeff_size;
  std::vector<SIZE> coarse_shape;
};

}  // namespace mgard_x

#endif