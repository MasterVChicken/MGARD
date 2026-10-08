#ifndef _MDR_HYBRID_LEVEL_COMPRESSOR_HPP
#define _MDR_HYBRID_LEVEL_COMPRESSOR_HPP

#include "../../Lossless/ParallelHuffman/Huffman.hpp"
#include "../../Lossless/ParallelRLE/RunLengthEncoding.hpp"
#include "../../Lossless/Zstd.hpp"
// #include "../RefactorUtils.hpp"
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
    ze_totals.resize({max_rows * ZE_MAX_LEVELS}, queue_idx);
    if (config.mdr_zero_elimination) {
      ze_chunk_counts.resize(
          {max_rows * zero_elimination::num_chunks(max_n)}, queue_idx);
      ze_super_counts.resize(
          {max_rows * (3 * zero_elimination::num_super(max_n) + ZE_MAX_LEVELS)},
          queue_idx);
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
                 SubArray<1, uint32_t, DeviceType> ze_bitmaps = {}) {
    if (config.mdr_zero_elimination) {
      compress_level_zero_elimination(encoded_bitplanes, compressed_bitplanes,
                                      queue_idx, sign_rows, ze_bitmaps);
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
  // rows), and one writing pass.
  void compress_level_zero_elimination(
      SubArray<2, T_bitplane, DeviceType> &encoded_bitplanes,
      std::vector<Array<1, Byte, DeviceType>> &compressed_bitplanes,
      int queue_idx, int sign_rows, SubArray<1, uint32_t, DeviceType> bitmaps) {
    using namespace zero_elimination;
    static_assert(sizeof(T_bitplane) == sizeof(uint32_t));
    SIZE num_rows = encoded_bitplanes.shape(0);
    SIZE num_words = encoded_bitplanes.shape(1);
    SIZE num_bitplanes = num_rows - sign_rows;
    SIZE nchunks = num_chunks(num_words), nsuper = num_super(num_words);
    ze_totals.resize({num_rows}, queue_idx);
    ze_super_counts.resize({num_rows * nsuper}, queue_idx);
    SubArray<1, uint32_t, DeviceType> super_counts(ze_super_counts);
    // ze_chunk_counts holds the chunk bitmaps with the warp kernels.
    constexpr bool warp_kernels = std::is_same<DeviceType, CUDA>::value;
    // Chunk bitmaps given by the encoder (warp kernels only) need only the
    // super-chunk counts.
    const bool given = warp_kernels && bitmaps.data() != nullptr;
    if (given) {
      if constexpr (warp_kernels) {
        DeviceLauncher<DeviceType>::Execute(
            SuperCountKernel<DeviceType>(num_rows, nchunks, bitmaps,
                                         super_counts),
            queue_idx);
      }
    } else {
      ze_chunk_counts.resize({num_rows * nchunks}, queue_idx);
      bitmaps = SubArray(ze_chunk_counts);
      if constexpr (warp_kernels) {
        DeviceLauncher<DeviceType>::Execute(
            BitmapKernel<T_bitplane, DeviceType>(encoded_bitplanes, num_rows,
                                                 bitmaps, super_counts),
            queue_idx);
      } else {
        DeviceLauncher<DeviceType>::Execute(
            CountKernel<T_bitplane, DeviceType>(encoded_bitplanes, num_rows,
                                                bitmaps, super_counts),
            queue_idx);
      }
    }
    DeviceLauncher<DeviceType>::Execute(
        ScanKernel<DeviceType>(num_rows, nsuper, super_counts,
                               SubArray(ze_totals)),
        queue_idx);
    std::vector<uint32_t> totals(num_rows);
    MemoryManager<DeviceType>::Copy1D(totals.data(), ze_totals.data(),
                                      num_rows, queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    ze_tasks_host.clear();
    add_zero_elimination_write_tasks(totals.data(), num_bitplanes, num_words,
                                     sign_rows, compressed_bitplanes,
                                     queue_idx);
    ze_tasks.resize({(SIZE)ze_tasks_host.size()}, queue_idx);
    MemoryManager<DeviceType>::Copy1D(ze_tasks.data(), ze_tasks_host.data(),
                                      ze_tasks_host.size(), queue_idx);
    if constexpr (warp_kernels) {
      DeviceLauncher<DeviceType>::Execute(
          WriteWarpKernel<T_bitplane, DeviceType>(
              encoded_bitplanes, SubArray(ze_tasks), ze_tasks_host.size(),
              bitmaps, super_counts),
          queue_idx);
    } else {
      DeviceLauncher<DeviceType>::Execute(
          WriteKernel<T_bitplane, DeviceType>(
              encoded_bitplanes, SubArray(ze_tasks), ze_tasks_host.size(),
              bitmaps, super_counts),
          queue_idx);
    }
  }

  // Sizes the groups of a level from its row totals (choosing raw rows),
  // resizes compressed_bitplanes and appends the level's write tasks.
  void add_zero_elimination_write_tasks(
      const uint32_t *totals, SIZE num_bitplanes, SIZE num_words,
      int sign_rows,
      std::vector<Array<1, Byte, DeviceType>> &compressed_bitplanes,
      int queue_idx) {
    using namespace zero_elimination;
    for (SIZE b = 0; b < num_bitplanes; b++) {
      if (b % num_merged_bitplanes != 0) {
        compressed_bitplanes[b].resize({0}, queue_idx);
        continue;
      }
      SIZE first = group_first_row(b, sign_rows);
      SIZE rows = group_num_rows(b, num_bitplanes, sign_rows);
      std::vector<uint32_t> counts(rows);
      SIZE size = header_bytes(rows);
      for (SIZE r = 0; r < rows; r++) {
        counts[r] = totals[first + r];
        if (row_bytes(counts[r], num_words) >= row_bytes(RAW_ROW, num_words)) {
          counts[r] = RAW_ROW;
        }
        size += row_bytes(counts[r], num_words);
      }
      compressed_bitplanes[b].resize({size}, queue_idx);
      SIZE section = header_bytes(rows);
      for (SIZE r = 0; r < rows; r++) {
        ze_tasks_host.push_back({(uint64_t)compressed_bitplanes[b].data(),
                                 (uint64_t)section, (uint32_t)(first + r),
                                 counts[r], (uint32_t)r, (uint32_t)rows});
        section += row_bytes(counts[r], num_words);
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
      std::vector<SubArray<1, uint32_t, DeviceType>> &bitmaps, int queue_idx,
      int sign_rows) {
    using namespace zero_elimination;
    static_assert(sizeof(T_bitplane) == sizeof(uint32_t));
    SIZE num_levels = encoded_bitplanes.size();
    SIZE num_rows = encoded_bitplanes[0].shape(0);
    SIZE num_bitplanes = num_rows - sign_rows;
    std::vector<SIZE> super_offset(num_levels + 1, 0);
    for (SIZE l = 0; l < num_levels; l++) {
      super_offset[l + 1] =
          super_offset[l] +
          num_rows * num_super(encoded_bitplanes[l].shape(1));
    }
    ze_super_counts.resize({super_offset[num_levels]}, queue_idx);
    ze_totals.resize({num_levels * num_rows}, queue_idx);
    auto super_counts = [&](SIZE l) {
      return SubArray<1, uint32_t, DeviceType>(
          {super_offset[l + 1] - super_offset[l]},
          ze_super_counts.data() + super_offset[l]);
    };
    for (SIZE l = 0; l < num_levels; l++) {
      SIZE num_words = encoded_bitplanes[l].shape(1);
      DeviceLauncher<DeviceType>::Execute(
          SuperCountKernel<DeviceType>(num_rows, num_chunks(num_words),
                                       bitmaps[l], super_counts(l)),
          queue_idx);
      DeviceLauncher<DeviceType>::Execute(
          ScanKernel<DeviceType>(
              num_rows, num_super(num_words), super_counts(l),
              SubArray<1, uint32_t, DeviceType>(
                  {num_rows}, ze_totals.data() + l * num_rows)),
          queue_idx);
    }
    std::vector<uint32_t> totals(num_levels * num_rows);
    MemoryManager<DeviceType>::Copy1D(totals.data(), ze_totals.data(),
                                      totals.size(), queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    ze_tasks_host.clear();
    std::vector<SIZE> task_offset(num_levels + 1, 0);
    for (SIZE l = 0; l < num_levels; l++) {
      add_zero_elimination_write_tasks(
          totals.data() + l * num_rows, num_bitplanes,
          encoded_bitplanes[l].shape(1), sign_rows, compressed_bitplanes[l],
          queue_idx);
      task_offset[l + 1] = ze_tasks_host.size();
    }
    ze_tasks.resize({(SIZE)ze_tasks_host.size()}, queue_idx);
    MemoryManager<DeviceType>::Copy1D(ze_tasks.data(), ze_tasks_host.data(),
                                      ze_tasks_host.size(), queue_idx);
    for (SIZE l = 0; l < num_levels; l++) {
      SIZE n = task_offset[l + 1] - task_offset[l];
      DeviceLauncher<DeviceType>::Execute(
          WriteWarpKernel<T_bitplane, DeviceType>(
              encoded_bitplanes[l],
              SubArray<1, RowTask, DeviceType>(
                  {n}, ze_tasks.data() + task_offset[l]),
              n, bitmaps[l], super_counts(l)),
          queue_idx);
    }
  }

  // Decoding tasks of a zero-elimination group, from its header on the host.
  // Header of a zero-elimination group read from the device; empty if the
  // group is not zero-eliminated.
  std::vector<Byte> read_zero_elimination_header(
      Array<1, Byte, DeviceType> &compressed, int queue_idx) {
    using namespace zero_elimination;
    std::vector<Byte> header(header_bytes(0));
    if (compressed.shape(0) < header.size()) {
      return {};
    }
    MemoryManager<DeviceType>::Copy1D(header.data(), compressed.data(),
                                      header.size(), queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    if (std::memcmp(header.data(), signature(), SIGNATURE_BYTES) != 0) {
      return {};
    }
    uint32_t rows;
    std::memcpy(&rows, header.data() + SIGNATURE_BYTES, sizeof(uint32_t));
    header.resize(header_bytes(rows));
    MemoryManager<DeviceType>::Copy1D(header.data(), compressed.data(),
                                      header.size(), queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    return header;
  }

  void add_zero_elimination_tasks(Array<1, Byte, DeviceType> &compressed,
                                  const Byte *host, SIZE first_row,
                                  SIZE num_words) {
    using namespace zero_elimination;
    uint32_t rows, words;
    std::memcpy(&rows, host + SIGNATURE_BYTES, sizeof(uint32_t));
    std::memcpy(&words, host + SIGNATURE_BYTES + sizeof(uint32_t),
                sizeof(uint32_t));
    if (words != num_words) {
      throw std::runtime_error("MDR-X: zero elimination row length mismatch.");
    }
    SIZE section = header_bytes(rows);
    for (uint32_t r = 0; r < rows; r++) {
      uint32_t count;
      std::memcpy(&count, host + SIGNATURE_BYTES + (2 + r) * sizeof(uint32_t),
                  sizeof(uint32_t));
      ze_tasks_host.push_back({(uint64_t)compressed.data(), (uint64_t)section,
                               (uint32_t)(first_row + r), count, r, rows});
      section += row_bytes(count, num_words);
    }
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
      const std::vector<Byte *> *host_bitplanes = nullptr) {
    // With host_bitplanes (host copies of compressed_bitplanes), each group's
    // format and header are read on the host and the decoding stays queued:
    // no transfer or synchronization per group. Huffman decoding of a group
    // is latency bound (one thread per 1024-symbol chunk) and groups write
    // disjoint rows, so the Huffman groups of the level are decoded
    // concurrently on DECODE_QUEUES queues of their own, which are joined
    // before returning.
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
        if (ze && std::memcmp(ze, zero_elimination::signature(),
                              zero_elimination::SIGNATURE_BYTES) == 0) {
          // Zero elimination: decoded for the whole level below
          add_zero_elimination_tasks(
              compressed, ze, group_first_row(bitplane_idx, sign_rows),
              encoded_bitplanes.shape(1));
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
      ze_tasks.resize({(SIZE)ze_tasks_host.size()}, queue_idx);
      MemoryManager<DeviceType>::Copy1D(ze_tasks.data(), ze_tasks_host.data(),
                                        ze_tasks_host.size(), queue_idx);
      if constexpr (std::is_same<DeviceType, CUDA>::value) {
        DeviceLauncher<DeviceType>::Execute(
            zero_elimination::DecodeWarpKernel<T_bitplane, DeviceType>(
                encoded_bitplanes, SubArray(ze_tasks), ze_tasks_host.size()),
            queue_idx);
      } else {
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
  Array<1, uint32_t, DeviceType> ze_chunk_counts, ze_super_counts, ze_totals;
  Array<1, zero_elimination::RowTask, DeviceType> ze_tasks;
  std::vector<zero_elimination::RowTask> ze_tasks_host;
};

} // namespace MDR
} // namespace mgard_x
#endif
