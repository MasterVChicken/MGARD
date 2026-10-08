#ifndef _MDR_COMPOSED_REFACTOR_HPP
#define _MDR_COMPOSED_REFACTOR_HPP

#include "../BitplaneEncoder/BitplaneEncoder.hpp"
#include "../Decomposer/Decomposer.hpp"
#include "../ErrorCollector/ErrorCollector.hpp"
#include "../Interleaver/Interleaver.hpp"
#include "../LosslessCompressor/LevelCompressor.hpp"
// #include "../RefactorUtils.hpp"
#include "../Writer/Writer.hpp"
#include "RefactorInterface.hpp"
// #include "../DataStructures/MDRData.hpp"
#include <algorithm>
#include <iostream>
namespace mgard_x {
namespace MDR {
// a decomposition-based scientific data refactor: compose a refactor using
// decomposer, interleaver, encoder, and error collector
// ControlL2: collect the per-level squared errors for every bitplane count.
// They are stored in the metadata and needed for L2 (s != inf) requests.
template <DIM D, typename T_data, typename DeviceType, bool ControlL2 = true,
          typename Basis = Hierarchical, bool NegaBinary = false>
class ComposedRefactor
    : public concepts::RefactorInterface<D, T_data, DeviceType> {
public:
  using HierarchyType = Hierarchy<D, T_data, DeviceType>;
  using T_bitplane = uint32_t;
  using T_error = double;
  using Decomposer = MGARDDecomposer<D, T_data, Basis, DeviceType>;
  using Interleaver = DirectInterleaver<D, T_data, DeviceType>;
  using LocalDecomposer = HybridDecomposer<D, T_data, DeviceType>;
  static constexpr bool OrthogonalBasis =
      std::is_same<Basis, Orthogonal>::value;

  constexpr static bool ProfileBPEncoder = false;
  // using Encoder = GroupedBPEncoder<D, T_data, T_bitplane, T_error,
  //                                ControlL2, DeviceType>;
  // using Encoder = BPEncoderLocalityBlock<D, T_data, T_bitplane, T_error,
  // NegaBinary,
  //                                ControlL2, DeviceType>;
  using Encoder = BPEncoderRegisterBlock<D, T_data, T_bitplane, T_error,
                                         NegaBinary, ControlL2, DeviceType>;
  // using Encoder = BPEncoderRegisterShift<D, T_data, T_bitplane, T_error,
  // NegaBinary,
  //                              ControlL2, DeviceType>;
  // using Encoder = BPEncoderRegisterBallot<D, T_data, T_bitplane, T_error,
  // NegaBinary,
  //                              ControlL2, DeviceType>;
  // using Encoder = BPEncoderRegisterReduceAll<D, T_data, T_bitplane, T_error,
  // NegaBinary,
  //                              ControlL2, DeviceType>;
  // using Encoder = BPEncoderRegisterMatchAny<D, T_data, T_bitplane, T_error,
  // NegaBinary,
  //                              ControlL2, DeviceType>;

  // using Compressor = DefaultLevelCompressor<T_bitplane, HUFFMAN, DeviceType>;
  // using Compressor = DefaultLevelCompressor<T_bitplane, RLE, DeviceType>;
  using Compressor = HybridLevelCompressor<T_bitplane, DeviceType>;
  // using Compressor = NullLevelCompressor<T_bitplane, DeviceType>;

  static constexpr SIZE BATCH_SIZE = sizeof(T_bitplane) * 8;
  static constexpr SIZE MAX_BITPLANES = sizeof(T_data) * 8;

  ComposedRefactor() : initialized(false) {}

  ComposedRefactor(Hierarchy<D, T_data, DeviceType> &hierarchy, Config config) {
    Adapt(hierarchy, config, 0);
    DeviceRuntime<DeviceType>::SyncQueue(0);
  }

  static SIZE MaxOutputDataSize(std::vector<SIZE> shape, Config config) {
    MDRLevelLayout layout = build_level_layout(shape, config);
    SIZE size = 0;
    for (int level_idx = 0; level_idx < layout.num_levels(); level_idx++) {
      size += Encoder::NUM_ROWS *
              Encoder::bitplane_length(layout.level_num_elems[level_idx]) *
              sizeof(T_bitplane);
    }
    return size;
  }

  ~ComposedRefactor() {}

  void Adapt(Hierarchy<D, T_data, DeviceType> &hierarchy, Config config,
             int queue_idx) {
    this->initialized = true;
    this->hierarchy = &hierarchy;
    this->layout =
        build_level_layout(hierarchy.level_shape(hierarchy.l_target()), config);
    if (layout.hybrid) {
      local_decomposer.Adapt(layout, OrthogonalBasis, queue_idx);
    } else {
      decomposer.Adapt(hierarchy, config, queue_idx);
      interleaver.Adapt(hierarchy, queue_idx);
    }
    encoder.Adapt(hierarchy, layout.max_level_num_elems(), queue_idx);
    encoder.SetWordOrder(config.mdr_contiguous_words);
    // batched_encoder.Adapt(hierarchy, queue_idx);
    compressor.Adapt(encoder.bitplane_length(layout.max_level_num_elems()),
                     Encoder::MAX_BITPLANES, config, queue_idx);

    level_data_array.resize(layout.num_levels());
    level_data_subarray.resize(layout.num_levels());
    abs_max_array.resize(layout.num_levels());
    for (int level_idx = 0; level_idx < layout.num_levels(); level_idx++) {
      level_data_array[level_idx].resize(
          {round_up(layout.level_num_elems[level_idx], BATCH_SIZE)}, queue_idx);
      // interleave() only ever writes the level's real elements; the
      // round-up padding above (needed so the encoder's batch-aligned
      // kernels can run) is otherwise left as whatever cudaMalloc/pool
      // memory previously held. AbsMax and encode() below both operate
      // over the full padded length, so uninitialized padding pollutes
      // the level-wide abs_max scale factor -- harmless while the pool
      // memory happens to be zero, but corrupts every real element's
      // encoding once some larger, unrelated allocation has left large
      // leftover values in that memory. Zero it once here; interleave()
      // never touches it again for the lifetime of this object.
      level_data_array[level_idx].memset(0, queue_idx);
      level_data_subarray[level_idx] =
          SubArray<1, T_data, DeviceType>(level_data_array[level_idx]);
      abs_max_array[level_idx].resize({1}, queue_idx);
      abs_max_array[level_idx].hostAllocate(false, queue_idx);
    }

    DeviceCollective<DeviceType>::AbsMax(
        layout.max_level_num_elems(), SubArray<1, T_data, DeviceType>(),
        SubArray<1, T_data, DeviceType>(), abs_max_workspace, false, 0);
    encoded_bitplanes_array.resize(layout.num_levels());
    encoded_bitplanes_subarray.resize(layout.num_levels());
    level_num_elems.resize(layout.num_levels());
    level_errors_array.resize(layout.num_levels());
    level_errors_subarray.resize(layout.num_levels());
    exp.resize(layout.num_levels());
    for (int level_idx = 0; level_idx < layout.num_levels(); level_idx++) {
      encoded_bitplanes_array[level_idx].resize(
          {(SIZE)Encoder::NUM_ROWS,
           encoder.bitplane_length(layout.level_num_elems[level_idx])},
          queue_idx);
      encoded_bitplanes_subarray[level_idx] =
          SubArray<2, T_bitplane, DeviceType>(
              encoded_bitplanes_array[level_idx]);
      level_num_elems[level_idx] = layout.level_num_elems[level_idx];
      level_errors_array[level_idx].resize({(SIZE)Encoder::MAX_BITPLANES + 1},
                                           queue_idx);
      level_errors_subarray[level_idx] =
          SubArray<1, T_error, DeviceType>(level_errors_array[level_idx]);
    }
    // Zero elimination: chunk bitmaps of each level, written by the encoder
    // when it can (encode() returns whether it did).
    use_zero_elimination = config.mdr_zero_elimination;
    ze_bitmaps_array.resize(use_zero_elimination ? layout.num_levels() : 0);
    ze_given.assign(layout.num_levels(), false);
    for (int level_idx = 0;
         use_zero_elimination && level_idx < layout.num_levels(); level_idx++) {
      SIZE words = encoder.bitplane_length(layout.level_num_elems[level_idx]);
      ze_bitmaps_array[level_idx].resize(
          {Encoder::NUM_ROWS * zero_elimination::num_chunks(words)}, queue_idx);
    }
  }

  static size_t EstimateMemoryFootprint(std::vector<SIZE> shape,
                                        Config config) {
    MDRLevelLayout layout = build_level_layout(shape, config);
    Hierarchy<D, T_data, DeviceType> hierarchy;
    size_t size = 0;
    size += hierarchy.EstimateMemoryFootprint(shape);
    for (int level_idx = 0; level_idx < layout.num_levels(); level_idx++) {
      size += round_up(layout.level_num_elems[level_idx], BATCH_SIZE) *
              sizeof(T_data);
    }
    size += sizeof(T_data);
    Array<1, Byte, DeviceType> tmp;
    DeviceCollective<DeviceType>::AbsMax(
        layout.max_level_num_elems(), SubArray<1, T_data, DeviceType>(),
        SubArray<1, T_data, DeviceType>(), tmp, false, 0);
    size += tmp.shape(0);
    for (int level_idx = 0; level_idx < layout.num_levels(); level_idx++) {
      size += Encoder::NUM_ROWS *
              Encoder::bitplane_length(layout.level_num_elems[level_idx]) *
              sizeof(T_bitplane);
      size += sizeof(T_error) * (Encoder::MAX_BITPLANES + 1);
    }

    SIZE max_n = Encoder::bitplane_length(layout.max_level_num_elems());

    size += (Encoder::MAX_BITPLANES + 1) * sizeof(T_error);
    if (layout.hybrid) {
      size += LocalDecomposer::EstimateMemoryFootprint(layout);
    } else {
      size += Decomposer::EstimateMemoryFootprint(shape);
      size += Interleaver::EstimateMemoryFootprint(shape);
    }
    size += Encoder::EstimateMemoryFootprint(shape);
    size += Compressor::EstimateMemoryFootprint(max_n, config);
    return size;
  }

  static std::vector<std::vector<SIZE>>
  EstimateMaxBitplaneSizes(std::vector<SIZE> shape, Config config) {
    return EstimateMaxBitplaneSizes(build_level_layout(shape, config),
                                    config.mdr_bitplane_group_size);
  }

  std::vector<std::vector<SIZE>> EstimateMaxBitplaneSizes() const {
    return EstimateMaxBitplaneSizes(layout, compressor.num_merged_bitplanes);
  }

  const std::vector<SIZE> &LevelNumElems() const {
    return layout.level_num_elems;
  }

  static std::vector<std::vector<SIZE>>
  EstimateMaxBitplaneSizes(const MDRLevelLayout &layout, int group_size) {
    std::vector<std::vector<SIZE>> estimation;
    estimation.resize(layout.num_levels());
    for (int level_idx = 0; level_idx < layout.num_levels(); level_idx++) {
      estimation[level_idx].resize(Encoder::MAX_BITPLANES);
      for (int bitplane_idx = 0; bitplane_idx < Encoder::MAX_BITPLANES;
           bitplane_idx++) {
        if (bitplane_idx % group_size == 0) {
          estimation[level_idx][bitplane_idx] =
              Encoder::bitplane_length(layout.level_num_elems[level_idx]) *
              sizeof(T_bitplane) *
              (group_size + (bitplane_idx == 0 ? Encoder::SIGN_ROWS : 0));
          // A group is stored compressed only when that at least halves it
          // (HybridLevelCompressor checks the ratio before writing), so its
          // size is at most the raw size plus a few KB of headers.
          estimation[level_idx][bitplane_idx] += 4096;
        } else {
          estimation[level_idx][bitplane_idx] = 1;
        }
      }
    }
    return estimation;
  }

  void Refactor(Array<D, T_data, DeviceType> &data_array,
                MDRMetadata &mdr_metadata, MDRData<DeviceType> &mdr_data,
                int queue_idx) {
    mdr_metadata.Initialize(layout.num_levels(), Encoder::MAX_BITPLANES);
    mdr_data.Resize(*this, *hierarchy, queue_idx);

    SubArray<D, T_data, DeviceType> data(data_array);

    Timer timer, timer_all;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer_all.start();
    }
    if (layout.hybrid) {
      // Writes every level's coefficients directly into level_data_subarray
      // (no separate interleave pass for the block-local levels).
      local_decomposer.decompose(data, level_data_subarray, queue_idx);
    } else {
      decomposer.decompose(data_array, hierarchy->l_target(), 0, queue_idx);

      if (log::level & log::TIME) {
        DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
        timer.start();
      }
      interleaver.interleave(data, level_data_subarray, hierarchy->l_target(),
                             queue_idx);
      if (log::level & log::TIME) {
        DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
        timer.end();
        timer.print("Interleave",
                    hierarchy->total_num_elems() * sizeof(T_data));
        timer.clear();
      }
    }

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }

    for (int level_idx = 0; level_idx < layout.num_levels(); level_idx++) {
      DeviceCollective<DeviceType>::AbsMax(
          level_data_subarray[level_idx].shape(0),
          level_data_subarray[level_idx], SubArray(abs_max_array[level_idx]),
          abs_max_workspace, true, queue_idx);

      {
        // DumpSubArray("level_"+std::to_string(level_idx),
        // level_data_subarray[level_idx]); for (SIZE i = 0; i <
        // level_data_subarray[level_idx].shape(0); i += 1e6) {
        // // for (SIZE i = 0; i <  10; i += 10) {
        //   SIZE n = std::min(level_data_subarray[level_idx].shape(0) - i,
        //   (SIZE)1e6); SubArray<1, T_data, DeviceType> data_block({n},
        //   level_data_subarray[level_idx](i));
        //   // PrintSubarray("data_block", data_block);
        //   T_data * ddd = new T_data[n];
        //   MemoryManager<DeviceType>::Copy1D(ddd, data_block.data(), n,
        //   queue_idx); DeviceRuntime<DeviceType>::SyncQueue(queue_idx);

        //   T_data min = fabs(ddd[0]);
        //   T_data max = fabs(ddd[0]);
        //   for (SIZE j = 0; j < n; j++) {
        //     min = std::min(min, fabs(ddd[j]));
        //     max = std::max(max, fabs(ddd[j]));
        //   }

        //   int c = 0;
        //   for (SIZE j = 0; j < n; j++) {
        //     if (fabs(ddd[i]) > max * 0.001) {
        //       c++;
        //     }
        //   }
        //   std::cout << "cpu: [" << n << "] " << max << " - "<< min  << " c: "
        //   << c << std::endl;

        //   DeviceCollective<DeviceType>::AbsMax(
        //       n,
        //       data_block, SubArray(abs_max_array[level_idx]),
        //       abs_max_workspace, true, queue_idx);
        //   DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
        //   abs_max_array[level_idx].hostCopy(false, queue_idx);
        //   DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
        //   T_data abs_max = abs_max_array[level_idx].dataHost()[0];

        //   DeviceCollective<DeviceType>::AbsMin(
        //       n,
        //       data_block, SubArray(abs_max_array[level_idx]),
        //       abs_max_workspace, true, queue_idx);
        //   DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
        //   abs_max_array[level_idx].hostCopy(false, queue_idx);
        //   DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
        //   T_data abs_min = abs_max_array[level_idx].dataHost()[0];

        //   std::cout << "abs: " << abs_max << " - "<< abs_min << std::endl;
        // }
      }

      encoded_bitplanes_array[level_idx].resize(
          {(SIZE)Encoder::NUM_ROWS,
           encoder.bitplane_length(layout.level_num_elems[level_idx])},
          queue_idx);
      encoded_bitplanes_subarray[level_idx] =
          SubArray<2, T_bitplane, DeviceType>(
              encoded_bitplanes_array[level_idx]);

      Timer timer_iter;
      if constexpr (ProfileBPEncoder) {
        DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
        timer_iter.start();
      }
      SubArray<1, uint32_t, DeviceType> ze_bitmaps;
      if (use_zero_elimination) {
        ze_bitmaps = SubArray(ze_bitmaps_array[level_idx]);
      }
      ze_given[level_idx] = encoder.encode(
          level_data_subarray[level_idx].shape(0), Encoder::MAX_BITPLANES,
          SubArray(abs_max_array[level_idx]), level_data_subarray[level_idx],
          encoded_bitplanes_subarray[level_idx],
          level_errors_subarray[level_idx], queue_idx, ze_bitmaps);
      if constexpr (ProfileBPEncoder) {
        DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
        timer_iter.end();
        timer_iter.print(
            "Encoding level (# of coefficients: " +
                std::to_string(level_data_subarray[level_idx].shape(0)) + ")",
            level_data_subarray[level_idx].shape(0) * sizeof(T_data), true);
      }
    }

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("Encoding", hierarchy->total_num_elems() * sizeof(T_data));
      timer.clear();
    }

    // if (log::level & log::TIME) {
    //   DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    //   timer.start();
    // }

    // for (int level_idx = 0; level_idx < layout.num_levels();
    //      level_idx++) {
    //   compressor.compress_level(encoded_bitplanes_subarray[level_idx],
    //                             mdr_data.compressed_bitplanes[level_idx],
    //                             level_idx, queue_idx);
    //   for (int bitplane_idx = 0; bitplane_idx < Encoder::MAX_BITPLANES;
    //        bitplane_idx++) {
    //     mdr_metadata.level_sizes[level_idx][bitplane_idx] +=
    //         mdr_data.compressed_bitplanes[level_idx][bitplane_idx].shape(0);
    //   }
    // }
    // if (log::level & log::TIME) {
    //   DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    //   timer.end();
    //   timer.print("Lossless", hierarchy->total_num_elems() * sizeof(T_data));
    //   timer.clear();
    // }

    // Compress(mdr_metadata, mdr_data, queue_idx);
    // StoreMetadata(mdr_metadata, mdr_data, queue_idx);
    // for (int level_idx = 0; level_idx < layout.num_levels();
    //      level_idx++) {
    //   abs_max_array[level_idx].hostCopy(false, queue_idx);
    //   DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    //   T_data level_max_error = abs_max_array[level_idx].dataHost()[0];
    //   mdr_metadata.level_error_bounds[level_idx] = level_max_error;
    //   mdr_metadata.level_num_elems[level_idx] =
    //   layout.level_num_elems[level_idx]; std::vector<T_error>
    //   squared_error(Encoder::MAX_BITPLANES + 1);
    //   MemoryManager<DeviceType>::Copy1D(squared_error.data(),
    //                                     level_errors_array[level_idx].data(),
    //                                     Encoder::MAX_BITPLANES + 1,
    //                                     queue_idx);
    //   mdr_metadata.level_squared_errors[level_idx] = squared_error;
    //   // PrintSubarray("level_errors", level_errors_subarray[level_idx]);
    // }

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer_all.end();
      timer_all.print(layout.hybrid ? "Hybrid Decompose + Encoding"
                                    : "Decompose + Interleave + Encoding",
                      hierarchy->total_num_elems() * sizeof(T_data));
      timer_all.clear();
    }
  }

  void Compress(MDRMetadata &mdr_metadata, MDRData<DeviceType> &mdr_data,
                int queue_idx) {
    Timer timer;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }
    // With every level's chunk bitmaps given by the encoder, all levels are
    // zero-eliminated at once.
    bool all_given = use_zero_elimination &&
                     std::all_of(ze_given.begin(), ze_given.end(),
                                 [](bool given) { return given; });
    if constexpr (std::is_same<DeviceType, CUDA>::value) {
      if (all_given) {
        std::vector<SubArray<1, uint32_t, DeviceType>> ze_bitmaps;
        for (int level_idx = 0; level_idx < layout.num_levels(); level_idx++) {
          ze_bitmaps.push_back(SubArray(ze_bitmaps_array[level_idx]));
        }
        compressor.compress_levels_zero_elimination(
            encoded_bitplanes_subarray, mdr_data.compressed_bitplanes,
            ze_bitmaps, queue_idx, Encoder::SIGN_ROWS);
      }
    } else {
      all_given = false;
    }
    for (int level_idx = 0; !all_given && level_idx < layout.num_levels();
         level_idx++) {
      SubArray<1, uint32_t, DeviceType> ze_bitmaps;
      if (ze_given[level_idx]) {
        ze_bitmaps = SubArray(ze_bitmaps_array[level_idx]);
      }
      compressor.compress_level(encoded_bitplanes_subarray[level_idx],
                                mdr_data.compressed_bitplanes[level_idx],
                                level_idx, queue_idx, Encoder::SIGN_ROWS,
                                ze_bitmaps);
    }
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("Lossless", hierarchy->total_num_elems() * sizeof(T_data));
      timer.clear();
    }
  }

  void StoreMetadata(MDRMetadata &mdr_metadata, MDRData<DeviceType> &mdr_data,
                     int queue_idx) {
    mdr_metadata.group_size = compressor.num_merged_bitplanes;
    mdr_metadata.word_order = encoder.Contiguous() ? 1 : 0;
    for (int level_idx = 0; level_idx < layout.num_levels(); level_idx++) {
      abs_max_array[level_idx].hostCopy(false, queue_idx);
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      T_data level_max_error = abs_max_array[level_idx].dataHost()[0];
      mdr_metadata.level_error_bounds[level_idx] = level_max_error;
      mdr_metadata.level_num_elems[level_idx] =
          layout.level_num_elems[level_idx];
      std::vector<T_error> squared_error(Encoder::MAX_BITPLANES + 1);
      MemoryManager<DeviceType>::Copy1D(squared_error.data(),
                                        level_errors_array[level_idx].data(),
                                        Encoder::MAX_BITPLANES + 1, queue_idx);
      mdr_metadata.level_squared_errors[level_idx] = squared_error;
      for (int bitplane_idx = 0; bitplane_idx < Encoder::MAX_BITPLANES;
           bitplane_idx++) {
        mdr_metadata.level_sizes[level_idx][bitplane_idx] +=
            mdr_data.compressed_bitplanes[level_idx][bitplane_idx].shape(0);
      }
      // PrintSubarray("level_errors", level_errors_subarray[level_idx]);
    }
  }

  void print() const {
    std::cout << "Composed refactor with the following components."
              << std::endl;
    std::cout << "Decomposer: ";
    if (layout.hybrid) {
      local_decomposer.print();
    } else {
      decomposer.print();
      std::cout << "Interleaver: ";
      interleaver.print();
    }
    std::cout << "Encoder: ";
    encoder.print();
  }

  bool initialized = false;

private:
  Hierarchy<D, T_data, DeviceType> *hierarchy;
  MDRLevelLayout layout;
  Decomposer decomposer;
  Interleaver interleaver;
  LocalDecomposer local_decomposer;
  Encoder encoder;
  // BatchedEncoder batched_encoder;
  Compressor compressor;

  std::vector<Array<1, T_data, DeviceType>> level_data_array;
  std::vector<SubArray<1, T_data, DeviceType>> level_data_subarray;

  std::vector<Array<1, T_data, DeviceType>> abs_max_array;
  Array<1, Byte, DeviceType> abs_max_workspace;

  std::vector<Array<2, T_bitplane, DeviceType>> encoded_bitplanes_array;
  std::vector<SubArray<2, T_bitplane, DeviceType>> encoded_bitplanes_subarray;

  std::vector<Array<1, T_error, DeviceType>> level_errors_array;
  std::vector<SubArray<1, T_error, DeviceType>> level_errors_subarray;

  std::vector<SIZE> level_num_elems;
  std::vector<int32_t> exp;

  bool use_zero_elimination = false;
  std::vector<Array<1, uint32_t, DeviceType>> ze_bitmaps_array;
  std::vector<bool> ze_given;
};
} // namespace MDR
} // namespace mgard_x
#endif
