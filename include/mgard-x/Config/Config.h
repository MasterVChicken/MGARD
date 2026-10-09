#ifndef MGARD_X_CONFIG_HPP
#define MGARD_X_CONFIG_HPP

#include "../RuntimeX/RuntimeXPublic.h"
#include "../RuntimeX/Utilities/Log.h"
#include "../Utilities/Types.h"

namespace mgard_x {

struct Config {
  device_type dev_type;
  int dev_id;
  enum compressor_type compressor;
  enum domain_decomposition_type domain_decomposition;
  enum decomposition_type decomposition;
  double estimate_outlier_ratio;
  SIZE huff_dict_size;
  SIZE huff_block_size;
  SIZE block_delta_block_size;
  enum block_delta_mode_type block_delta_mode;
  SIZE lz4_block_size;
  int zstd_compress_level;
  bool normalize_coordinates;
  enum lossless_type lossless;
  int log_level;
  bool auto_pin_host_buffers;
  SIZE max_larget_level;
  SIZE max_memory_footprint;
  SIZE total_num_bitplanes;
  SIZE block_size;
  SIZE domain_decomposition_dim;
  std::vector<SIZE> domain_decomposition_sizes;
  bool mdr_adaptive_resolution;
  // MDR-X: bitplanes per merged group (1-4), the unit of compression and of
  // retrieval. Smaller groups retrieve closer to a requested error.
  int mdr_bitplane_group_size;
  // MDR-X: store bitplane groups with zero elimination instead of
  // Huffman/RLE (experimental).
  bool mdr_zero_elimination;
  // MDR-X: bitplane words of 32 consecutive coefficients (better locality
  // for the lossless stage) instead of 32 coefficients strided over a level.
  bool mdr_contiguous_words;
  // MDR-X, with zero elimination and contiguous words: store the sign of a
  // coefficient with the bitplane group in which it becomes nonzero instead
  // of in a sign row of the first group, so that retrievals read only the
  // signs of the coefficients they reconstruct as nonzero.
  bool mdr_significance_signs;
  // MDR-X, with zero elimination: rows may store their nonzero words as a
  // 3-bit code and the positions of their one bits (when that is smaller).
  bool mdr_sparse_words;
  bool adjust_shape;
  bool compress_with_dryrun;
  int num_local_refactoring_level;
  int num_global_refactoring_level;
  bool auto_cache_release;
  cpu_parallelization_mode cpu_mode;
  bool mdr_qoi_mode;
  int mdr_qoi_num_variables;
  std::vector<double> roi_tolerance_map;
  bool enable_roi;
  // Transform basis policy, shared by the plain Compressor and the hybrid
  // (BlockMGARD) HybridHierarchyCompressor. Auto (default) picks Hierarchical
  // under an L-infinity bound and Orthogonal otherwise; Orthogonal/
  // Hierarchical force a specific basis (Hierarchical still requires an
  // L-infinity bound). See compression_projection_mode_type in Types.h.
  enum compression_projection_mode_type projection_mode;
  // The hybrid (BlockMGARD) local stage fuses its decompose/recompose kernels
  // with quantization/dequantization so coefficients never round-trip through
  // global memory as T. These select the older separate-pass implementation,
  // which the test suite still exercises. Purely a performance choice: both
  // paths reconstruct identically, so a file compressed either way decompresses
  // either way and nothing about the choice is recorded in the file header.
  bool fuse_decompose_quantize;   // compression
  bool fuse_dequantize_recompose; // decompression

  Config();
  void apply();
};

} // namespace mgard_x

#endif
