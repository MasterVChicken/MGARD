#ifndef _MDR_HYBRID_LEVEL_COMPRESSOR_HPP
#define _MDR_HYBRID_LEVEL_COMPRESSOR_HPP

#include "../../Lossless/ParallelHuffman/Huffman.hpp"
#include "../../Lossless/ParallelRLE/RunLengthEncoding.hpp"
#include "../../Lossless/Zstd.hpp"
// #include "../RefactorUtils.hpp"
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
  static constexpr int num_merged_bitplanes = 4;

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
  static constexpr int MAX_GROUP_ROWS = num_merged_bitplanes + 1;
  static SIZE group_first_row(SIZE bitplane_idx, int sign_rows) {
    return bitplane_idx == 0 ? 0 : bitplane_idx + sign_rows;
  }
  static SIZE group_num_rows(SIZE bitplane_idx, SIZE num_bitplanes,
                             int sign_rows) {
    return std::min((SIZE)num_merged_bitplanes, num_bitplanes - bitplane_idx) +
           (bitplane_idx == 0 ? sign_rows : 0);
  }

  void Adapt(SIZE max_n, SIZE max_bitplanes, Config config, int queue_idx) {
    this->initialized = true;
    this->config = config;
    huffman.Resize(max_n * byte_ratio * MAX_GROUP_ROWS, _huff_dict_size,
                   _huff_block_size, config.estimate_outlier_ratio, queue_idx);
    rle.Resize(max_n * byte_ratio * MAX_GROUP_ROWS, queue_idx);
    zstd.Resize(max_n * sizeof(T_bitplane), config.zstd_compress_level,
                queue_idx);
  }
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
                 int level_idx, int queue_idx, int sign_rows) {

    std::vector<float> cr, time;
    bool huffman_success, rle_success, zstd_success;
    SIZE num_bitplanes = encoded_bitplanes.shape(0) - sign_rows;
    for (SIZE bitplane_idx = 0; bitplane_idx < num_bitplanes; bitplane_idx++) {
      if (bitplane_idx % num_merged_bitplanes == 0) {
        SIZE merged_bitplane_size =
            encoded_bitplanes.shape(1) * byte_ratio *
            group_num_rows(bitplane_idx, num_bitplanes, sign_rows);
        // What size_threshold is compared with: a full group in the format
        // version 0 layout, where encoders with a sign row instead reserved a
        // sign slot in every row and so had rows twice as long. This keeps
        // compressing the same groups; with the halved rows, groups of 1-2M
        // coefficients (e.g. the finest level of a 128^3 subdomain) would be
        // stored raw.
        SIZE threshold_size = encoded_bitplanes.shape(1) * byte_ratio *
                              num_merged_bitplanes * (sign_rows > 0 ? 2 : 1);
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
        } else if (threshold_size > size_threshold) {
          // Decide from cheap statistics before compressing: RLE only when
          // its own estimate passes (same formula, one counting pass instead
          // of start marks + scan), Huffman only when the entropy bound can
          // reach the target. Skips exactly the attempts that would fail, so
          // the output is unchanged.
          SubArray<1, T_compress, DeviceType> group(encoded_bitplane);
          if (rle.EstimateCRFast(group, queue_idx) >= cr_threshold) {
            rle_success = rle.Compress(encoded_bitplane,
                                       compressed_bitplanes[bitplane_idx],
                                       cr_threshold, queue_idx);
          }
          if (rle_success) {
            rle.Serialize(compressed_bitplanes[bitplane_idx], queue_idx);
          } else if (huffman_may_reach(group, cr_threshold, queue_idx)) {
            ATOMIC_IDX zero = 0;
            MemoryManager<DeviceType>::Copy1D(
                huffman.workspace.outlier_count_subarray.data(), &zero, 1,
                queue_idx);
            MemoryManager<DeviceType>::Copy1D(
                &huffman.outlier_count,
                huffman.workspace.outlier_count_subarray.data(), 1, queue_idx);
            huffman_success = huffman.CompressPrimary(
                encoded_bitplane, compressed_bitplanes[bitplane_idx],
                cr_threshold, queue_idx);
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

  // Upper bound on the CR that Huffman::CompressPrimary estimates (and tests
  // against target_cr): with two or more symbols every codeword has >= 1 bit
  // and the total length is >= n * entropy, so the encoded bits are at least
  // n * max(H, 1).
  bool huffman_may_reach(SubArray<1, T_compress, DeviceType> group,
                         float target_cr, int queue_idx) {
    SIZE n = group.shape(0);
    freq_array.resize({(SIZE)_huff_dict_size}, queue_idx);
    freq_array.memset(0, queue_idx);
    Histogram<T_compress, unsigned int, DeviceType>(
        group, SubArray(freq_array), n, _huff_dict_size, queue_idx);
    std::vector<unsigned int> freq(_huff_dict_size);
    MemoryManager<DeviceType>::Copy1D(freq.data(), freq_array.data(),
                                      _huff_dict_size, queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    int num_symbols = 0;
    double entropy = 0;
    for (unsigned int f : freq) {
      if (f > 0) {
        num_symbols++;
        double p = (double)f / n;
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

  // decompress level, create new buffer and overwrite original streams; will
  // not change stream sizes
  void decompress_level(
      std::vector<Array<1, Byte, DeviceType>> &compressed_bitplanes,
      SubArray<2, T_bitplane, DeviceType> &encoded_bitplanes,
      uint8_t starting_bitplane, uint8_t num_bitplanes, int level_idx,
      int queue_idx, int sign_rows) {

    std::vector<float> time;
    SIZE total_bitplanes = encoded_bitplanes.shape(0) - sign_rows;
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

        // Huffman
        if (huffman.Verify(compressed_bitplanes[bitplane_idx], queue_idx)) {
          huffman.Deserialize(compressed_bitplanes[bitplane_idx], queue_idx);
          huffman.DecompressPrimary(compressed_bitplanes[bitplane_idx],
                                    encoded_bitplane, queue_idx);
          // RLE
        } else if (rle.Verify(compressed_bitplanes[bitplane_idx], queue_idx)) {
          rle.Deserialize(compressed_bitplanes[bitplane_idx], queue_idx);
          rle.Decompress(compressed_bitplanes[bitplane_idx], encoded_bitplane,
                         queue_idx);
        } else if (is_zstd(compressed_bitplanes[bitplane_idx],
                           merged_bitplane_size, queue_idx)) {
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
  Array<1, unsigned int, DeviceType> freq_array;
};

} // namespace MDR
} // namespace mgard_x
#endif
