#ifndef _MDR_HYBRID_LEVEL_COMPRESSOR_HPP
#define _MDR_HYBRID_LEVEL_COMPRESSOR_HPP

#include "../../Lossless/ParallelHuffman/Huffman.hpp"
#include "../../Lossless/ParallelRLE/RunLengthEncoding.hpp"
#include "../../Lossless/Zstd.hpp"
// #include "../RefactorUtils.hpp"
#include "../BitplaneEncoder/SignificanceCoding.hpp"
#include "GroupStatistics.hpp"
#include "ZeroElimination.hpp"
#include "LevelCompressorInterface.hpp"
#include "LosslessCompressor.hpp"
#include <cstring>

namespace mgard_x {
namespace MDR {

// interface for lossless compressor
template <typename T_bitplane, typename DeviceType>
class HybridLevelCompressor
    : public concepts::LevelCompressorInterface<T_bitplane, DeviceType> {
public:
  using T_compress = u_int8_t;
  // using T_compress = u_int16_t;

  static constexpr int byte_ratio = sizeof(T_bitplane) / sizeof(T_compress);
  static constexpr int _huff_dict_size = 256;
  static constexpr int _huff_block_size = 1024;
  // Bitplanes per merged group (1 to MAX_GROUP_SIZE): the unit of
  // compression and of retrieval. Set from the config when refactoring and
  // from the metadata when reconstructing.
  static constexpr int MAX_GROUP_SIZE = 4;
  // Queues for concurrent Huffman decoding (the MDR pipelines use 0-2).
  static constexpr int FIRST_DECODE_QUEUE = 3;
  static constexpr int DECODE_QUEUES = 8;
  static_assert(FIRST_DECODE_QUEUE + DECODE_QUEUES <= MGARDX_NUM_QUEUES);
  int num_merged_bitplanes = MAX_GROUP_SIZE;
  void SetGroupSize(int group_size) {
    if (group_size < 1 || group_size > MAX_GROUP_SIZE) {
      throw std::runtime_error("MDR-X: bitplane group size must be 1 to " +
                               std::to_string(MAX_GROUP_SIZE) + ", got " +
                               std::to_string(group_size));
    }
    num_merged_bitplanes = group_size;
  }

  SIZE size_threshold = 1e6;
  float cr_threshold = 2.0;

  HybridLevelCompressor() : initialized(false) {}
  HybridLevelCompressor(SIZE max_n, Config config) {
    this->initialized = true;
    Adapt(max_n * byte_ratio, config, 0);
    DeviceRuntime<DeviceType>::SyncQueue(0);
  }
  ~HybridLevelCompressor(){};

  // A merged group spans num_merged_bitplanes rows of encoded bitplanes;
  // group 0 also carries the encoder's sign rows (sign_rows, 0 or 1), so a
  // group has at most MAX_GROUP_ROWS rows.
  static constexpr int MAX_GROUP_ROWS = MAX_GROUP_SIZE + 1;
  static SIZE group_first_row(SIZE bitplane_idx, int sign_rows) {
    return bitplane_idx == 0 ? 0 : bitplane_idx + sign_rows;
  }
  SIZE group_num_rows(SIZE bitplane_idx, SIZE num_bitplanes,
                      int sign_rows) const {
    return std::min((SIZE)num_merged_bitplanes, num_bitplanes - bitplane_idx) +
           (bitplane_idx == 0 ? sign_rows : 0);
  }

  // Significance-coded signs (zero elimination only, SignificanceCoding.hpp):
  // the rows of a group are its bitplanes (row b + 1 for bitplane b; row 0
  // holds the encoder's packed signs) and its stream ends with its signs.
  bool sign_coding = false;
  void SetSignCoding(bool sign_coding) { this->sign_coding = sign_coding; }
  SIZE ze_first_row(SIZE bitplane_idx, int sign_rows) const {
    return sign_coding ? bitplane_idx + 1
                       : group_first_row(bitplane_idx, sign_rows);
  }
  SIZE ze_num_rows(SIZE bitplane_idx, SIZE num_bitplanes,
                   int sign_rows) const {
    return sign_coding ? std::min((SIZE)num_merged_bitplanes,
                                  num_bitplanes - bitplane_idx)
                       : group_num_rows(bitplane_idx, num_bitplanes, sign_rows);
  }
  SIZE num_groups(SIZE num_bitplanes) const {
    return significance::num_groups(num_bitplanes, num_merged_bitplanes);
  }

  // Sparse words (zero elimination, Config::mdr_sparse_words): rows may store
  // their nonzero words as codes and one-bit positions (ZeroElimination.hpp).
  bool sparse_words = false;
  void SetSparseWords(bool sparse_words) { this->sparse_words = sparse_words; }

  void Adapt(SIZE max_n, SIZE max_bitplanes, Config config, int queue_idx) {
    this->initialized = true;
    this->config = config;
    SetGroupSize(config.mdr_bitplane_group_size);
    huffman.Resize(max_n * byte_ratio * MAX_GROUP_ROWS, _huff_dict_size,
                   _huff_block_size, config.estimate_outlier_ratio, queue_idx);
    rle.Resize(max_n * byte_ratio * MAX_GROUP_ROWS, queue_idx);
    zstd.Resize(max_n * sizeof(T_bitplane), config.zstd_compress_level,
                queue_idx);
    // Sized up front (all levels at once: up to ZE_MAX_LEVELS levels whose
    // sizes sum to at most 3 times the largest): growing them while
    // compressing reallocates, and the frees synchronize the device.
    SIZE max_rows = max_bitplanes + 1;
    ze_tasks.resize({max_rows * ZE_MAX_LEVELS}, queue_idx);
    // Per level: nonzero words, nonzero chunks and region words of each row,
    // then the sign words of each group.
    ze_totals.resize({4 * max_rows * ZE_MAX_LEVELS}, queue_idx);
    SetSparseWords(config.mdr_sparse_words);
    if (config.mdr_zero_elimination) {
      ze_chunk_counts.resize(
          {max_rows * zero_elimination::num_chunks(max_n)}, queue_idx);
      if (sparse_words) {
        ze_chunk_bits.resize(
            {max_rows * zero_elimination::num_chunks(max_n)}, queue_idx);
      }
      ze_super_counts.resize(
          {3 * max_rows *
           (3 * zero_elimination::num_super(max_n) + ZE_MAX_LEVELS)},
          queue_idx);
      sign_offsets.resize(
          {max_rows *
           (3 * significance::num_segments(max_n) + ZE_MAX_LEVELS)},
          queue_idx);
      sign_tasks.resize({max_rows * ZE_MAX_LEVELS}, queue_idx);
    }
  }
  static constexpr SIZE ZE_MAX_LEVELS = 64;
  static size_t EstimateMemoryFootprint(SIZE max_n, Config config) {
    size_t size = 0;
    size += Huffman<T_bitplane, T_bitplane, HUFFMAN_CODE, DeviceType>::
        EstimateMemoryFootprint(max_n * byte_ratio * MAX_GROUP_ROWS,
                                _huff_dict_size, _huff_block_size,
                                config.estimate_outlier_ratio);
    size += parallel_rle::RunLengthEncoding<
        T_compress, u_int32_t, u_int32_t,
        DeviceType>::EstimateMemoryFootprint(max_n * byte_ratio *
                                             MAX_GROUP_ROWS);
    size +=
        Zstd<DeviceType>::EstimateMemoryFootprint(max_n * sizeof(T_bitplane));
    return size;
  }

  void
  compress_level(SubArray<2, T_bitplane, DeviceType> &encoded_bitplanes,
                 std::vector<Array<1, Byte, DeviceType>> &compressed_bitplanes,
                 int level_idx, int queue_idx, int sign_rows,
                 SubArray<1, uint32_t, DeviceType> ze_bitmaps = {},
                 SubArray<1, uint32_t, DeviceType> ze_bits = {},
                 SubArray<1, uint32_t, DeviceType> sign_counts = {},
                 SubArray<1, uint32_t, DeviceType> sign_segment_bits = {}) {
    if (config.mdr_zero_elimination) {
      compress_level_zero_elimination(encoded_bitplanes, compressed_bitplanes,
                                      queue_idx, sign_rows, ze_bitmaps, ze_bits,
                                      sign_counts, sign_segment_bits);
      return;
    }

    std::vector<float> cr, time;
    bool huffman_success, rle_success, zstd_success;
    SIZE num_bitplanes = encoded_bitplanes.shape(0) - sign_rows;
    // What size_threshold is compared with: a full group of 4 bitplanes in
    // the format version 0 layout, where encoders with a sign row instead
    // reserved a sign slot in every row and so had rows twice as long. It
    // depends only on the level, so every group size compresses the same
    // levels as before (with the halved rows, groups of 1-2M coefficients,
    // e.g. the finest level of a 128^3 subdomain, would be stored raw).
    SIZE threshold_size = encoded_bitplanes.shape(1) * byte_ratio *
                          MAX_GROUP_SIZE * (sign_rows > 0 ? 2 : 1);
    bool try_rle_huffman = threshold_size > size_threshold &&
                           config.lossless != lossless_type::Huffman_Zstd;
    if (try_rle_huffman) {
      group_statistics(encoded_bitplanes, num_bitplanes, sign_rows, queue_idx);
    }
    for (SIZE bitplane_idx = 0; bitplane_idx < num_bitplanes; bitplane_idx++) {
      if (bitplane_idx % num_merged_bitplanes == 0) {
        SIZE merged_bitplane_size =
            encoded_bitplanes.shape(1) * byte_ratio *
            group_num_rows(bitplane_idx, num_bitplanes, sign_rows);
        Timer timer;
        timer.start();
        T_compress *bitplane = (T_compress *)encoded_bitplanes(
            group_first_row(bitplane_idx, sign_rows), 0);

        Array<1, T_compress, DeviceType> encoded_bitplane(
            {merged_bitplane_size}, bitplane);
        int old_log_level = log::level;
        log::level = 0;
        huffman_success = false;
        rle_success = false;
        zstd_success = false;
        // cr_threshold = 2.0;
        if (threshold_size > size_threshold &&
            config.lossless == lossless_type::Huffman_Zstd) {
          zstd_success =
              compress_zstd((Byte *)bitplane, merged_bitplane_size,
                            compressed_bitplanes[bitplane_idx], queue_idx);
        } else if (try_rle_huffman) {
          // Decide from the group's statistics before compressing: RLE only
          // when its own estimate passes (same formula, from the run count),
          // Huffman only when the entropy bound can reach the target. Skips
          // exactly the attempts that would fail.
          SIZE group = bitplane_idx / num_merged_bitplanes;
          const unsigned int *freq = &group_freqs[group * _huff_dict_size];
          if (rle.CRFromRuns(merged_bitplane_size, group_runs[group]) >=
              cr_threshold) {
            rle.Compress(encoded_bitplane, compressed_bitplanes[bitplane_idx],
                         (SIZE)group_runs[group], queue_idx);
            rle_success = true;
          }
          if (rle_success) {
            rle.Serialize(compressed_bitplanes[bitplane_idx], queue_idx);
          } else if (huffman_may_reach(freq, merged_bitplane_size,
                                       cr_threshold)) {
            // Primary data only: no outliers.
            huffman.outlier_count = 0;
            huffman_success = huffman.CompressPrimary(
                encoded_bitplane, compressed_bitplanes[bitplane_idx],
                cr_threshold, freq, queue_idx);
            if (huffman_success) {
              huffman.Serialize(compressed_bitplanes[bitplane_idx], queue_idx);
            }
          }
        }

        if (!huffman_success && !rle_success && !zstd_success) {
          // direct copy
          compressed_bitplanes[bitplane_idx].resize({merged_bitplane_size});
          MemoryManager<DeviceType>::Copy1D(
              compressed_bitplanes[bitplane_idx].data(), (Byte *)bitplane,
              merged_bitplane_size, queue_idx);
        }

        log::level = old_log_level;
        cr.push_back((float)merged_bitplane_size /
                     compressed_bitplanes[bitplane_idx].shape(0));

        timer.end();
        time.push_back(timer.get());
        timer.clear();
        // timer.print("Compressing bitplane", merged_bitplane_size);
        // timer.clear();
      } else {
        compressed_bitplanes[bitplane_idx].resize({0}, queue_idx);
      }
    }
    // std::string cr_string = "";
    // for (auto x : cr) {
    //   cr_string += std::to_string(x) + ", ";
    // }
    // log::info("CR: " + cr_string);

    // std::string time_string = "";
    // for (auto x : time) {
    //   time_string += std::to_string(x) + " ";
    // }
    // log::info("Time: " + time_string);
  }

  // LevelCompressorInterface: encoded bitplanes without sign rows
  void
  compress_level(SubArray<2, T_bitplane, DeviceType> &encoded_bitplanes,
                 std::vector<Array<1, Byte, DeviceType>> &compressed_bitplanes,
                 int level_idx, int queue_idx) {
    compress_level(encoded_bitplanes, compressed_bitplanes, level_idx,
                   queue_idx, 0);
  }
  void decompress_level(
      std::vector<Array<1, Byte, DeviceType>> &compressed_bitplanes,
      SubArray<2, T_bitplane, DeviceType> &encoded_bitplanes,
      uint8_t starting_bitplane, uint8_t num_bitplanes, int level_idx,
      int queue_idx) {
    decompress_level(compressed_bitplanes, encoded_bitplanes, starting_bitplane,
                     num_bitplanes, level_idx, queue_idx, 0);
  }

  // Zero elimination of all groups of a level: one counting pass over all
  // rows, one transfer of the row totals (to size the groups and choose raw
  // and sparse rows), and one writing pass. bits: the chunks' sparse-word
  // payload bits given with the bitmaps by the encoder. With sign_coding,
  // sign_counts and sign_segment_bits are the encoder's
  // (SignificanceCoding.hpp).
  void compress_level_zero_elimination(
      SubArray<2, T_bitplane, DeviceType> &encoded_bitplanes,
      std::vector<Array<1, Byte, DeviceType>> &compressed_bitplanes,
      int queue_idx, int sign_rows, SubArray<1, uint32_t, DeviceType> bitmaps,
      SubArray<1, uint32_t, DeviceType> bits = {},
      SubArray<1, uint32_t, DeviceType> sign_counts = {},
      SubArray<1, uint32_t, DeviceType> sign_segment_bits = {}) {
    using namespace zero_elimination;
    static_assert(sizeof(T_bitplane) == sizeof(uint32_t));
    SIZE num_rows = encoded_bitplanes.shape(0);
    SIZE num_words = encoded_bitplanes.shape(1);
    SIZE num_bitplanes = num_rows - sign_rows;
    SIZE nchunks = num_chunks(num_words), nsuper = num_super(num_words);
    // Totals: nonzero words, nonzero chunks and region words of each row,
    // then the sign words of each group. Super-chunk counts: the same for
    // each super-chunk.
    SIZE G = sign_coding ? num_groups(num_bitplanes) : 0;
    ze_totals.resize({3 * num_rows + G}, queue_idx);
    ze_super_counts.resize({3 * num_rows * nsuper}, queue_idx);
    SubArray<1, uint32_t, DeviceType> super_counts(ze_super_counts);
    // ze_chunk_counts holds the chunk bitmaps with the warp kernels (CUDA),
    // and the chunk counts with the portable ones.
    constexpr bool cuda = std::is_same<DeviceType, CUDA>::value;
    const bool warp_kernels = cuda && !portable_kernels();
    // Chunk bitmaps given by the encoder (warp kernels only) need only the
    // super-chunk counts.
    const bool given = warp_kernels && bitmaps.data() != nullptr &&
                       (!sparse_words || bits.data() != nullptr);
    if (!given) {
      ze_chunk_counts.resize({num_rows * nchunks}, queue_idx);
      bits = {};
      if (sparse_words) {
        ze_chunk_bits.resize({num_rows * nchunks}, queue_idx);
        bits = SubArray(ze_chunk_bits);
      }
    } else if (!sparse_words) {
      bits = {};
    }
    if constexpr (cuda) {
      if (given) {
        DeviceLauncher<DeviceType>::Execute(
            SuperCountKernel<DeviceType>(num_rows, nchunks, bitmaps, bits,
                                         super_counts),
            queue_idx);
      } else if (warp_kernels) {
        bitmaps = SubArray(ze_chunk_counts);
        DeviceLauncher<DeviceType>::Execute(
            BitmapKernel<T_bitplane, DeviceType>(encoded_bitplanes, num_rows,
                                                 bitmaps, bits, super_counts),
            queue_idx);
      }
    }
    if (!warp_kernels) {
      bitmaps = SubArray(ze_chunk_counts);
      DeviceLauncher<DeviceType>::Execute(
          CountKernel<T_bitplane, DeviceType>(encoded_bitplanes, num_rows,
                                              bitmaps, bits, super_counts),
          queue_idx);
    }
    DeviceLauncher<DeviceType>::Execute(
        ScanKernel<DeviceType>(3 * num_rows, nsuper, super_counts,
                               SubArray(ze_totals)),
        queue_idx);
    SIZE nseg = significance::num_segments(num_words);
    if (sign_coding) {
      sign_offsets.resize({G * nseg}, queue_idx);
      DeviceLauncher<DeviceType>::Execute(
          significance::SignScanKernel<DeviceType>(
              G, nseg, sign_segment_bits, SubArray(sign_offsets),
              SubArray<1, uint32_t, DeviceType>(
                  {G}, ze_totals.data() + 3 * num_rows)),
          queue_idx);
    }
    std::vector<uint32_t> totals(3 * num_rows + G);
    MemoryManager<DeviceType>::Copy1D(totals.data(), ze_totals.data(),
                                      totals.size(), queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    ze_tasks_host.clear();
    sign_tasks_host.clear();
    add_zero_elimination_write_tasks(
        totals.data(), totals.data() + num_rows, totals.data() + 2 * num_rows,
        num_bitplanes, num_words, sign_rows, compressed_bitplanes, queue_idx,
        sign_coding ? totals.data() + 3 * num_rows : nullptr);
    // With the shared region the sparse rows need (none without them).
    SIZE capacity = region_capacity(ze_tasks_host, 0, ze_tasks_host.size());
    ze_tasks.resize({(SIZE)ze_tasks_host.size()}, queue_idx);
    MemoryManager<DeviceType>::Copy1D(ze_tasks.data(), ze_tasks_host.data(),
                                      ze_tasks_host.size(), queue_idx);
    if constexpr (cuda) {
      if (warp_kernels) {
        DeviceLauncher<DeviceType>::Execute(
            WriteWarpKernel<T_bitplane, DeviceType>(
                num_rows, encoded_bitplanes, SubArray(ze_tasks),
                ze_tasks_host.size(), bitmaps, bits, super_counts, capacity),
            queue_idx);
      }
    }
    if (!warp_kernels) {
      DeviceLauncher<DeviceType>::Execute(
          WriteKernel<T_bitplane, DeviceType>(
              num_rows, encoded_bitplanes, SubArray(ze_tasks),
              ze_tasks_host.size(), bitmaps, bits, super_counts, capacity),
          queue_idx);
    }
    if (sign_coding) {
      upload_sign_tasks(queue_idx);
      write_signs(encoded_bitplanes, sign_counts, SubArray(sign_offsets), 0, G,
                  queue_idx);
    }
  }

  void upload_sign_tasks(int queue_idx) {
    sign_tasks.resize({(SIZE)sign_tasks_host.size()}, queue_idx);
    MemoryManager<DeviceType>::Copy1D(sign_tasks.data(), sign_tasks_host.data(),
                                      sign_tasks_host.size(), queue_idx);
  }

  // Writes the sign sections of a level's groups, whose tasks are
  // sign_tasks[first_task, first_task + num_groups).
  void write_signs(SubArray<2, T_bitplane, DeviceType> &encoded_bitplanes,
                   SubArray<1, uint32_t, DeviceType> sign_counts,
                   SubArray<1, uint32_t, DeviceType> offsets, SIZE first_task,
                   SIZE num_groups, int queue_idx) {
    SubArray<1, significance::SignTask, DeviceType> tasks(
        {num_groups}, sign_tasks.data() + first_task);
    DeviceLauncher<DeviceType>::Execute(
        significance::SignWriteKernel<T_bitplane, DeviceType>(
            num_groups, encoded_bitplanes, sign_counts, offsets, tasks),
        queue_idx);
  }

  // Sizes the groups of a level (two-level or sparse rows) from its row
  // totals of nonzero words, chunks and region words (choosing raw and sparse
  // rows) and sign words (sign_coding), resizes compressed_bitplanes and
  // appends the level's write tasks.
  void add_zero_elimination_write_tasks(
      const uint32_t *totals, const uint32_t *chunk_totals,
      const uint32_t *region_totals, SIZE num_bitplanes, SIZE num_words,
      int sign_rows,
      std::vector<Array<1, Byte, DeviceType>> &compressed_bitplanes,
      int queue_idx, const uint32_t *sign_words = nullptr) {
    using namespace zero_elimination;
    const bool signs = sign_words != nullptr;
    const uint32_t kind =
        ZE | TWO_LEVEL | (signs ? SIGNS : 0) | (sparse_words ? SPARSE : 0);
    for (SIZE b = 0; b < num_bitplanes; b++) {
      if (b % num_merged_bitplanes != 0) {
        compressed_bitplanes[b].resize({0}, queue_idx);
        continue;
      }
      SIZE first = ze_first_row(b, sign_rows);
      SIZE rows = ze_num_rows(b, num_bitplanes, sign_rows);
      std::vector<uint32_t> counts(rows), chunks(rows), payload(rows, 0);
      SIZE size = header_bytes(rows, kind);
      for (SIZE r = 0; r < rows; r++) {
        counts[r] = totals[first + r];
        chunks[r] = chunk_totals[first + r];
        uint32_t region = sparse_words ? region_totals[first + r] : 0;
        if (region > 0 && row_bytes(counts[r], chunks[r], num_words, true,
                                    region) <
                              row_bytes(counts[r], chunks[r], num_words, true)) {
          payload[r] = region;
        }
        if (row_bytes(counts[r], chunks[r], num_words, true, payload[r]) >=
            row_bytes(RAW_ROW, 0, num_words, true)) {
          counts[r] = RAW_ROW;
          payload[r] = 0;
        }
        size += row_bytes(counts[r], chunks[r], num_words, true, payload[r]);
      }
      SIZE sign_section = size;
      uint32_t words = signs ? sign_words[b / num_merged_bitplanes] : 0;
      if (signs) {
        size += (significance::num_segments(num_words) + words) *
                sizeof(uint32_t);
      }
      compressed_bitplanes[b].resize({size}, queue_idx);
      SIZE section = header_bytes(rows, kind);
      for (SIZE r = 0; r < rows; r++) {
        ze_tasks_host.push_back({(uint64_t)compressed_bitplanes[b].data(),
                                 (uint64_t)section, (uint32_t)(first + r),
                                 counts[r], chunks[r], (uint32_t)r,
                                 (uint32_t)rows, kind, payload[r]});
        section +=
            row_bytes(counts[r], chunks[r], num_words, true, payload[r]);
      }
      if (signs) {
        // The sign words are the last header word.
        sign_tasks_host.push_back(
            {(uint64_t)compressed_bitplanes[b].data(), (uint64_t)sign_section,
             (uint32_t)(header_words(rows, kind) - 1), words});
      }
    }
  }

  // Zero elimination of all levels at once, with the chunk bitmaps given by
  // the encoder (warp kernels): one transfer of every level's row totals
  // instead of one round trip per level.
  void compress_levels_zero_elimination(
      std::vector<SubArray<2, T_bitplane, DeviceType>> &encoded_bitplanes,
      std::vector<std::vector<Array<1, Byte, DeviceType>>>
          &compressed_bitplanes,
      std::vector<SubArray<1, uint32_t, DeviceType>> &bitmaps,
      std::vector<SubArray<1, uint32_t, DeviceType>> &bits, int queue_idx,
      int sign_rows,
      std::vector<SubArray<1, uint32_t, DeviceType>> *sign_counts = nullptr,
      std::vector<SubArray<1, uint32_t, DeviceType>> *sign_segment_bits =
          nullptr) {
    using namespace zero_elimination;
    static_assert(sizeof(T_bitplane) == sizeof(uint32_t));
    SIZE num_levels = encoded_bitplanes.size();
    SIZE num_rows = encoded_bitplanes[0].shape(0);
    SIZE num_bitplanes = num_rows - sign_rows;
    // Per level: totals of nonzero words, nonzero chunks and region words of
    // each row, then the sign words of each group.
    SIZE G = sign_coding ? num_groups(num_bitplanes) : 0;
    SIZE stride = 3 * num_rows + G;
    auto level_bits = [&](SIZE l) {
      return sparse_words ? bits[l] : SubArray<1, uint32_t, DeviceType>();
    };
    std::vector<SIZE> super_offset(num_levels + 1, 0);
    std::vector<SIZE> sign_offset(num_levels + 1, 0);
    for (SIZE l = 0; l < num_levels; l++) {
      super_offset[l + 1] =
          super_offset[l] +
          3 * num_rows * num_super(encoded_bitplanes[l].shape(1));
      sign_offset[l + 1] =
          sign_offset[l] +
          G * significance::num_segments(encoded_bitplanes[l].shape(1));
    }
    ze_super_counts.resize({super_offset[num_levels]}, queue_idx);
    ze_totals.resize({num_levels * stride}, queue_idx);
    sign_offsets.resize({std::max(sign_offset[num_levels], (SIZE)1)},
                        queue_idx);
    auto super_counts = [&](SIZE l) {
      return SubArray<1, uint32_t, DeviceType>(
          {super_offset[l + 1] - super_offset[l]},
          ze_super_counts.data() + super_offset[l]);
    };
    auto offsets = [&](SIZE l) {
      return SubArray<1, uint32_t, DeviceType>(
          {sign_offset[l + 1] - sign_offset[l]},
          sign_offsets.data() + sign_offset[l]);
    };
    for (SIZE l = 0; l < num_levels; l++) {
      SIZE num_words = encoded_bitplanes[l].shape(1);
      DeviceLauncher<DeviceType>::Execute(
          SuperCountKernel<DeviceType>(num_rows, num_chunks(num_words),
                                       bitmaps[l], level_bits(l),
                                       super_counts(l)),
          queue_idx);
      DeviceLauncher<DeviceType>::Execute(
          ScanKernel<DeviceType>(3 * num_rows, num_super(num_words),
                                 super_counts(l),
                                 SubArray<1, uint32_t, DeviceType>(
                                     {3 * num_rows},
                                     ze_totals.data() + l * stride)),
          queue_idx);
      if (sign_coding) {
        DeviceLauncher<DeviceType>::Execute(
            significance::SignScanKernel<DeviceType>(
                G, significance::num_segments(num_words),
                (*sign_segment_bits)[l], offsets(l),
                SubArray<1, uint32_t, DeviceType>(
                    {G}, ze_totals.data() + l * stride + 3 * num_rows)),
            queue_idx);
      }
    }
    std::vector<uint32_t> totals(num_levels * stride);
    MemoryManager<DeviceType>::Copy1D(totals.data(), ze_totals.data(),
                                      totals.size(), queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    ze_tasks_host.clear();
    sign_tasks_host.clear();
    std::vector<SIZE> task_offset(num_levels + 1, 0), capacity(num_levels);
    for (SIZE l = 0; l < num_levels; l++) {
      add_zero_elimination_write_tasks(
          totals.data() + l * stride, totals.data() + l * stride + num_rows,
          totals.data() + l * stride + 2 * num_rows, num_bitplanes,
          encoded_bitplanes[l].shape(1), sign_rows, compressed_bitplanes[l],
          queue_idx,
          sign_coding ? totals.data() + l * stride + 3 * num_rows : nullptr);
      task_offset[l + 1] = ze_tasks_host.size();
      capacity[l] = region_capacity(ze_tasks_host, task_offset[l],
                                    task_offset[l + 1]);
    }
    ze_tasks.resize({(SIZE)ze_tasks_host.size()}, queue_idx);
    MemoryManager<DeviceType>::Copy1D(ze_tasks.data(), ze_tasks_host.data(),
                                      ze_tasks_host.size(), queue_idx);
    if (sign_coding) {
      upload_sign_tasks(queue_idx);
    }
    for (SIZE l = 0; l < num_levels; l++) {
      SIZE n = task_offset[l + 1] - task_offset[l];
      DeviceLauncher<DeviceType>::Execute(
          WriteWarpKernel<T_bitplane, DeviceType>(
              num_rows, encoded_bitplanes[l],
              SubArray<1, RowTask, DeviceType>(
                  {n}, ze_tasks.data() + task_offset[l]),
              n, bitmaps[l], level_bits(l), super_counts(l), capacity[l]),
          queue_idx);
      if (sign_coding) {
        write_signs(encoded_bitplanes[l], (*sign_counts)[l], offsets(l), l * G,
                    G, queue_idx);
      }
    }
  }

  // Decoding tasks of a zero-elimination group, from its header on the host.
  // Header of a zero-elimination group read from the device; empty if the
  // group is not zero-eliminated.
  std::vector<Byte> read_zero_elimination_header(
      Array<1, Byte, DeviceType> &compressed, int queue_idx) {
    using namespace zero_elimination;
    std::vector<Byte> header(header_bytes(0, ZE));
    if (compressed.shape(0) < header.size()) {
      return {};
    }
    MemoryManager<DeviceType>::Copy1D(header.data(), compressed.data(),
                                      header.size(), queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    uint32_t kind = group_kind(header.data());
    if (kind == 0) {
      return {};
    }
    uint32_t rows;
    std::memcpy(&rows, header.data() + SIGNATURE_BYTES, sizeof(uint32_t));
    header.resize(header_bytes(rows, kind));
    MemoryManager<DeviceType>::Copy1D(header.data(), compressed.data(),
                                      header.size(), queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    return header;
  }

  // Decoding tasks of a zero-elimination group from its header; returns the
  // device address of its sign section (0 if it has none).
  uint64_t add_zero_elimination_tasks(Array<1, Byte, DeviceType> &compressed,
                                      const Byte *host, SIZE first_row,
                                      SIZE num_words) {
    using namespace zero_elimination;
    const uint32_t kind = group_kind(host);
    const bool two_level = kind & TWO_LEVEL, sparse = kind & SPARSE;
    uint32_t rows, words;
    std::memcpy(&rows, host + SIGNATURE_BYTES, sizeof(uint32_t));
    std::memcpy(&words, host + SIGNATURE_BYTES + sizeof(uint32_t),
                sizeof(uint32_t));
    if (words != num_words) {
      throw std::runtime_error("MDR-X: zero elimination row length mismatch.");
    }
    const uint32_t *header = (const uint32_t *)(host + SIGNATURE_BYTES);
    SIZE section = header_bytes(rows, kind);
    for (uint32_t r = 0; r < rows; r++) {
      uint32_t count, chunks = 0, payload = 0;
      std::memcpy(&count, header + 2 + r, sizeof(uint32_t));
      if (two_level) {
        std::memcpy(&chunks, header + 2 + rows + r, sizeof(uint32_t));
      }
      if (sparse) {
        std::memcpy(&payload, header + 2 + 2 * rows + r, sizeof(uint32_t));
      }
      ze_tasks_host.push_back({(uint64_t)compressed.data(), (uint64_t)section,
                               (uint32_t)(first_row + r), count, chunks, r,
                               rows, kind, payload});
      section += row_bytes(count, chunks, num_words, two_level, payload);
    }
    return (kind & SIGNS) ? (uint64_t)(compressed.data() + section) : 0;
  }

  // Upper bound on the CR that Huffman::CompressPrimary estimates (and tests
  // against target_cr): with two or more symbols every codeword has >= 1 bit
  // and the total length is >= n * entropy, so the encoded bits are at least
  // n * max(H, 1).
  static bool huffman_may_reach(const unsigned int *freq, SIZE n,
                                float target_cr) {
    int num_symbols = 0;
    double entropy = 0;
    for (int s = 0; s < _huff_dict_size; s++) {
      if (freq[s] > 0) {
        num_symbols++;
        double p = (double)freq[s] / n;
        entropy -= p * std::log2(p);
      }
    }
    if (num_symbols < 2) {
      return true;
    }
    double min_bits = n * std::max(entropy, 1.0) * (1.0 - 1e-9);
    double max_cr = (double)(n * sizeof(T_compress)) / (min_bits / 8 + 2000);
    return max_cr >= target_cr;
  }

  // Run counts and byte histograms of all merged groups of a level, in one
  // kernel and one transfer (group_runs, group_freqs on the host).
  void group_statistics(SubArray<2, T_bitplane, DeviceType> &encoded_bitplanes,
                        SIZE num_bitplanes, int sign_rows, int queue_idx) {
    SIZE num_groups =
        (num_bitplanes + num_merged_bitplanes - 1) / num_merged_bitplanes;
    group_runs_array.resize({num_groups}, queue_idx);
    group_freqs_array.resize({num_groups * _huff_dict_size}, queue_idx);
    group_runs_array.memset(0, queue_idx);
    group_freqs_array.memset(0, queue_idx);
    SIZE max_run = (SIZE)1 << (sizeof(u_int32_t) * 8);
    DeviceLauncher<DeviceType>::Execute(
        GroupStatisticsKernel<T_bitplane, DeviceType>(
            encoded_bitplanes, num_bitplanes, num_merged_bitplanes, sign_rows,
            max_run, num_groups, SubArray(group_runs_array),
            SubArray(group_freqs_array)),
        queue_idx);
    group_runs.resize(num_groups);
    group_freqs.resize(num_groups * _huff_dict_size);
    MemoryManager<DeviceType>::Copy1D(
        group_runs.data(), group_runs_array.data(), num_groups, queue_idx);
    MemoryManager<DeviceType>::Copy1D(group_freqs.data(),
                                      group_freqs_array.data(),
                                      num_groups * _huff_dict_size, queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
  }

  // decompress level, create new buffer and overwrite original streams; will
  // not change stream sizes
  void decompress_level(
      std::vector<Array<1, Byte, DeviceType>> &compressed_bitplanes,
      SubArray<2, T_bitplane, DeviceType> &encoded_bitplanes,
      uint8_t starting_bitplane, uint8_t num_bitplanes, int level_idx,
      int queue_idx, int sign_rows,
      const std::vector<Byte *> *host_bitplanes = nullptr,
      std::vector<uint64_t> *sign_sections = nullptr) {
    // With host_bitplanes (host copies of compressed_bitplanes), each group's
    // format and header are read on the host and the decoding stays queued:
    // no transfer or synchronization per group. Huffman decoding of a group
    // is latency bound (one thread per 1024-symbol chunk) and groups write
    // disjoint rows, so the Huffman groups of the level are decoded
    // concurrently on DECODE_QUEUES queues of their own, which are joined
    // before returning. With sign_coding, sign_sections gets the device
    // address of each decoded group's sign section, in order.
    if (sign_sections) {
      sign_sections->clear();
    }
    std::vector<float> time;
    SIZE total_bitplanes = encoded_bitplanes.shape(0) - sign_rows;
    int huffman_groups = 0;
    ze_tasks_host.clear();
    for (SIZE bitplane_idx = starting_bitplane;
         bitplane_idx < starting_bitplane + num_bitplanes; bitplane_idx++) {
      if (bitplane_idx % num_merged_bitplanes == 0) {
        Timer timer;
        timer.start();
        T_compress *bitplane = (T_compress *)encoded_bitplanes(
            group_first_row(bitplane_idx, sign_rows), 0);
        SIZE merged_bitplane_size =
            encoded_bitplanes.shape(1) * byte_ratio *
            group_num_rows(bitplane_idx, total_bitplanes, sign_rows);

        Array<1, T_compress, DeviceType> encoded_bitplane(
            {merged_bitplane_size}, bitplane);
        int old_log_level = log::level;
        log::level = 0;

        Array<1, Byte, DeviceType> &compressed =
            compressed_bitplanes[bitplane_idx];
        const Byte *host =
            host_bitplanes ? (*host_bitplanes)[bitplane_idx] : nullptr;
        // Zero-elimination group header: from the host copy, or else read
        // from the device (empty if the group is in another format).
        std::vector<Byte> ze_header;
        const Byte *ze = host;
        if (!host) {
          ze_header = read_zero_elimination_header(compressed, queue_idx);
          ze = ze_header.empty() ? nullptr : ze_header.data();
        }
        uint32_t ze_kind = ze ? zero_elimination::group_kind(ze) : 0;
        if (sign_coding &&
            (!(ze_kind & zero_elimination::SIGNS) || !sign_sections)) {
          throw std::runtime_error(
              "MDR-X: bitplane group without significance-coded signs.");
        }
        if (ze_kind != 0) {
          // Zero elimination: decoded for the whole level below
          uint64_t section = add_zero_elimination_tasks(
              compressed, ze, ze_first_row(bitplane_idx, sign_rows),
              encoded_bitplanes.shape(1));
          if (sign_coding) {
            sign_sections->push_back(section);
          }
          // Huffman
        } else if (host ? huffman.Verify(host)
                        : huffman.Verify(compressed, queue_idx)) {
          if (host) {
            if (huffman_groups == 0) {
              // The compressed groups are uploaded on queue_idx.
              DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
            }
            huffman.Deserialize(compressed, host);
            huffman.DecompressPrimary(
                compressed, encoded_bitplane,
                FIRST_DECODE_QUEUE + huffman_groups % DECODE_QUEUES, false);
            huffman_groups++;
          } else {
            huffman.Deserialize(compressed, queue_idx);
            huffman.DecompressPrimary(compressed, encoded_bitplane, queue_idx);
          }
          // RLE
        } else if (host ? rle.Verify(host)
                        : rle.Verify(compressed, queue_idx)) {
          if (host) {
            rle.Deserialize(compressed, host);
          } else {
            rle.Deserialize(compressed, queue_idx);
          }
          rle.Decompress(compressed, encoded_bitplane, queue_idx);
        } else if (host
                       ? is_zstd(host, compressed.shape(0),
                                 merged_bitplane_size)
                       : is_zstd(compressed, merged_bitplane_size, queue_idx)) {
          decompress_zstd(compressed_bitplanes[bitplane_idx], (Byte *)bitplane,
                          merged_bitplane_size, queue_idx);
        } else {
          // Direct copy
          MemoryManager<DeviceType>::Copy1D(
              (uint8_t *)bitplane, compressed_bitplanes[bitplane_idx].data(),
              merged_bitplane_size, queue_idx);
        }
        log::level = old_log_level;
        timer.end();
        time.push_back(timer.get());
        timer.clear();
      }
    }
    if (!ze_tasks_host.empty()) {
      SIZE capacity = zero_elimination::region_capacity(
          ze_tasks_host, 0, ze_tasks_host.size());
      ze_tasks.resize({(SIZE)ze_tasks_host.size()}, queue_idx);
      MemoryManager<DeviceType>::Copy1D(ze_tasks.data(), ze_tasks_host.data(),
                                        ze_tasks_host.size(), queue_idx);
      bool warp_kernels = false;
      if constexpr (std::is_same<DeviceType, CUDA>::value) {
        warp_kernels = !portable_kernels();
        if (warp_kernels) {
          DeviceLauncher<DeviceType>::Execute(
              zero_elimination::DecodeWarpKernel<T_bitplane, DeviceType>(
                  encoded_bitplanes, SubArray(ze_tasks), ze_tasks_host.size(),
                  capacity),
              queue_idx);
        }
      }
      if (!warp_kernels) {
        DeviceLauncher<DeviceType>::Execute(
            zero_elimination::DecodeKernel<T_bitplane, DeviceType>(
                encoded_bitplanes, SubArray(ze_tasks), ze_tasks_host.size()),
            queue_idx);
      }
    }
    for (int q = 0; q < std::min(huffman_groups, DECODE_QUEUES); q++) {
      DeviceRuntime<DeviceType>::SyncQueue(FIRST_DECODE_QUEUE + q);
    }
    // std::string time_string = "";
    // for (auto x : time) {
    //   time_string += std::to_string(x) + " ";
    // }
    // log::info("Time: " + time_string);
  }

  // ZSTD stage (Config::lossless == Huffman_Zstd): replaces RLE/byte Huffman
  // for groups above size_threshold and is kept whenever it is smaller than
  // the raw group. Stored as [signature][Zstd stream].
  static constexpr Byte zstd_signature[7] = {'M', 'G', 'X', 'Z', 'S', 'T', 'D'};

  bool compress_zstd(Byte *group, SIZE n, Array<1, Byte, DeviceType> &out,
                     int queue_idx) {
    Array<1, Byte, DeviceType> buffer({n});
    MemoryManager<DeviceType>::Copy1D(buffer.data(), group, n, queue_idx);
    zstd.Compress(buffer, queue_idx);
    SIZE size = buffer.shape(0);
    if (size + sizeof(zstd_signature) >= n) {
      return false;
    }
    out.resize({(SIZE)(size + sizeof(zstd_signature))}, queue_idx);
    MemoryManager<DeviceType>::Copy1D(out.data(), (Byte *)zstd_signature,
                                      sizeof(zstd_signature), queue_idx);
    MemoryManager<DeviceType>::Copy1D(out.data() + sizeof(zstd_signature),
                                      buffer.data(), size, queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    return true;
  }

  // A raw group is exactly n bytes; a ZSTD group is smaller and signed.
  bool is_zstd(const Byte *host_data, SIZE size, SIZE n) {
    if (size >= n || size <= sizeof(zstd_signature)) {
      return false;
    }
    return std::memcmp(host_data, zstd_signature, sizeof(zstd_signature)) == 0;
  }

  bool is_zstd(Array<1, Byte, DeviceType> &data, SIZE n, int queue_idx) {
    if (data.shape(0) >= n || data.shape(0) <= sizeof(zstd_signature)) {
      return false;
    }
    Byte signature[sizeof(zstd_signature)];
    MemoryManager<DeviceType>::Copy1D(signature, data.data(),
                                      sizeof(zstd_signature), queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    return std::memcmp(signature, zstd_signature, sizeof(zstd_signature)) == 0;
  }

  void decompress_zstd(Array<1, Byte, DeviceType> &data, Byte *group, SIZE n,
                       int queue_idx) {
    SIZE size = data.shape(0) - sizeof(zstd_signature);
    Array<1, Byte, DeviceType> buffer({size});
    MemoryManager<DeviceType>::Copy1D(
        buffer.data(), data.data() + sizeof(zstd_signature), size, queue_idx);
    zstd.Decompress(buffer, queue_idx);
    MemoryManager<DeviceType>::Copy1D(group, buffer.data(), n, queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
  }

  // release the buffer created
  void decompress_release() {}

  void print() const {}
  bool initialized;
  Huffman<T_compress, T_compress, uint64_t, DeviceType> huffman;
  parallel_rle::RunLengthEncoding<T_compress, u_int32_t, u_int32_t, DeviceType>
      rle;
  Zstd<DeviceType> zstd;
  Config config;
  // Per-group run counts and byte histograms of the level being compressed
  // (group_statistics); Huffman::CompressPrimary builds its codebook from the
  // host histogram.
  Array<1, unsigned int, DeviceType> group_runs_array, group_freqs_array;
  std::vector<unsigned int> group_runs, group_freqs;
  // Zero elimination workspaces and per-row tasks of the current level.
  Array<1, uint32_t, DeviceType> ze_chunk_counts, ze_chunk_bits,
      ze_super_counts, ze_totals;
  Array<1, zero_elimination::RowTask, DeviceType> ze_tasks;
  std::vector<zero_elimination::RowTask> ze_tasks_host;
  // Significance-coded signs: segment offsets of each group and its writer
  // tasks.
  Array<1, uint32_t, DeviceType> sign_offsets;
  Array<1, significance::SignTask, DeviceType> sign_tasks;
  std::vector<significance::SignTask> sign_tasks_host;
};

} // namespace MDR
} // namespace mgard_x
#endif
