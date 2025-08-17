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
    this->L = config.num_local_refactoring_level;
    compute_local_ranges();
    prepare_layers();
  }

  void Adapt(Hierarchy<D, T, DeviceType>& hierarchy, Config config,
             int queue_idx) {
    this->initialized = true;
    this->hierarchy = &hierarchy;
    this->config = config;
    this->L = config.num_local_refactoring_level;
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
    coarse_shape = hierarchy->level_shape(hierarchy->l_target());
    // for (int d = 0; d < coarse_shape.size(); d++) {
    //   log::info("Dim " + std::to_string(d) + " : " +
    //             std::to_string(coarse_shape[d]));
    // }

    fine_num_elems.clear();
    coarse_num_elems.clear();
    local_coeff_size.clear();

    // In that way we can have coarse_shape[0] store transformed 8x8x8 original
    // data
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
      // log::info("L = " + std::to_string(l) +
      //           ", fine_num_elems = " + std::to_string(fine_num_elems[l])
      //           +
      //           ", local_coeff_size = " +
      //           std::to_string(local_coeff_size[l]));
    }
  }

  void prepare_layers() {
    layer_len.assign(this->L + 1, 0);
    layer_off.assign(this->L + 1, 0);

    // The length of coarsest layer
    layer_len[0] = coarse_num_elems.back();
    layer_off[0] = 0;

    SIZE accum = layer_len[0];

    for (SIZE l = 1; l <= this->L; ++l) {
      layer_len[l] = local_coeff_size[this->L - l];
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
      // ben
      for (int l = 0; l < l_target + 1; l++) {
        quantizers[l] = (abs_tol) / ((l_target + 1) * (1 + std::pow(3, D)));
        // Debug info
        // log::info("Abs Tol: " + std::to_string(abs_tol));
        // log::info("l_target: " + std::to_string(l_target));
        // log::info("D: " + std::to_string(D));
        if (reciprocal) {
          quantizers[l] = 1.0f / quantizers[l];
        }
      }
    } else {
      // warning for un-inf
      log::err("Only L-inf supported");
      exit(-1);
    }
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
    T* host_quantizers = new T[this->L + 1];
    CalcQuantizers(hierarchy->total_num_elems(), host_quantizers, ebtype, tol,
                   s, norm, this->L, config.decomposition, true);

    // Debug for quantizers
    // for (int i = 0; i <= this->L; i++) {
    //   log::info("Quantizer[" + std::to_string(i) +
    //             "]: " + std::to_string(host_quantizers[i]));
    // }

    // log::info("=== LocalQuantizer Debug ===");
    // log::info("original_data.shape(0): " +
    //           std::to_string(original_data.shape(0)));
    // log::info("L: " + std::to_string(this->L));

    for (SIZE l = 0; l <= this->L; ++l) {
      // log::info("Layer " + std::to_string(l) + ":");
      // log::info("  layer_len[" + std::to_string(l) +
      //           "]: " + std::to_string(layer_len[l]));
      // log::info("  layer_off[" + std::to_string(l) +
      //           "]: " + std::to_string(layer_off[l]));
      // log::info("  access range: " + std::to_string(layer_off[l]) + " to " +
      //           std::to_string(layer_off[l] + layer_len[l] - 1));

      if (layer_off[l] + layer_len[l] > original_data.shape(0)) {
        log::err("*** BOUNDARY VIOLATION ***");
        log::err("Trying to access beyond array bounds!");
        log::err("Array size: " + std::to_string(original_data.shape(0)));
        log::err("Access end: " + std::to_string(layer_off[l] + layer_len[l]));
        return;
      }
      SubArray<1, T, DeviceType> v_in({layer_len[l]},
                                      original_data((IDX)layer_off[l]));
      SubArray<1, Q, DeviceType> qv({layer_len[l]},
                                    quantized_data((IDX)layer_off[l]));
      // Launch
      T quantizer = host_quantizers[l];
      DeviceLauncher<DeviceType>::Execute(
          QuantizeLocalLevelKernel<T, Q, MGARDX_QUANTIZE, DeviceType>(quantizer,
                                                                      v_in, qv),
          queue_idx);
      // PrintSubarray("Oringal data before quantizer:", v_in);
      // log::info("Quantizer: " + std::to_string(quantizer));
      // PrintSubarray("Quantized Array: ", qv);
    }
  }

  template <typename LosslessCompressorType>
  void Dequantize(SubArray<1, T, DeviceType> original_data,
                  enum error_bound_type ebtype, T tol, T s, T norm,
                  SubArray<1, Q, DeviceType> quantized_data,
                  LosslessCompressorType& lossless, int queue_idx) {
    T* host_quantizers = new T[this->L + 1];
    CalcQuantizers(hierarchy->total_num_elems(), host_quantizers, ebtype, tol,
                   s, norm, this->L, config.decomposition, false);

    for (SIZE l = 0; l <= this->L; ++l) {
      // log::info("Layer " + std::to_string(l) + ":");
      // log::info("  layer_len[" + std::to_string(l) +
      //           "]: " + std::to_string(layer_len[l]));
      // log::info("  layer_off[" + std::to_string(l) +
      //           "]: " + std::to_string(layer_off[l]));
      // log::info("  access range: " + std::to_string(layer_off[l]) + " to " +
      //           std::to_string(layer_off[l] + layer_len[l] - 1));

      if (layer_off[l] + layer_len[l] > original_data.shape(0)) {
        log::err("*** BOUNDARY VIOLATION ***");
        log::err("Trying to access beyond array bounds!");
        log::err("Array size: " + std::to_string(original_data.shape(0)));
        log::err("Access end: " + std::to_string(layer_off[l] + layer_len[l]));
        return;
      }
      SubArray<1, T, DeviceType> v_in({layer_len[l]},
                                      original_data((IDX)layer_off[l]));
      SubArray<1, Q, DeviceType> qv({layer_len[l]},
                                    quantized_data((IDX)layer_off[l]));
      // Launch
      T quantizer = host_quantizers[l];
      DeviceLauncher<DeviceType>::Execute(
          QuantizeLocalLevelKernel<T, Q, MGARDX_DEQUANTIZE, DeviceType>(
              quantizer, v_in, qv),
          queue_idx);
      // PrintSubarray("Quantized data before dequantization:", qv);
      // log::info("Quantizer: " + std::to_string(quantizer));
      // PrintSubarray("Dequantized Array: ", v_in);
    }
  }

  bool initialized;
  SIZE L;
  Hierarchy<D, T, DeviceType>* hierarchy;
  Config config;
  std::vector<SIZE> layer_len;
  // change off to offset
  std::vector<SIZE> layer_off;

  std::vector<SIZE> fine_num_elems;
  std::vector<SIZE> coarse_num_elems;
  std::vector<SIZE> local_coeff_size;
  std::vector<SIZE> coarse_shape;
};

}  // namespace mgard_x

#endif