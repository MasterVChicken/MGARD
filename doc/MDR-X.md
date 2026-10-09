
# MDR/MDR-X

Multi-precision Data Refactoring designed on top of MGARD for enabling fine-grain progressive data reconstruction with error control. Currently, there are two designs available:

* ***MGARD-DR (MDR):*** Full-featured CPU serial implementaion of multi-precision data refactoring
* ***MGARD-XDR (MDR-X):*** GPU acceleratoed portable implementation of MDR. Key features are implemented with other features under development.

***Note: Both MDR and MDR-X are experimenal compoments of the MGARD software. Their internal algorithms and API designs are subject to change in future releases of MGARD.***

## Supporting features
* **Data type:** Double and single precision floating-point data
* **Dimensions:** 1D-5D
* **Error-bound type:** L\_Inf error and L\_2 error
* **Data structure:** Uniform spaced Cartisan gird
* **Portability:** Same as MGARD-X (MDR-X only)
* **Decomposition (MDR-X only):** global multigrid hierarchy (default), or the hybrid BlockMGARD hierarchy (1D-3D, see below)

### Hybrid decomposition (MDR-X)
With `config.decomposition = decomposition_type::Hybrid` (`mdr-x -z ... -hh`), MDR-X replaces the global multigrid decomposition with the BlockMGARD design: `num_local_refactoring_level` (`-ll`, default 1) levels of the block-local in-cache decomposition (each 8^D block coarsens to 5^D nodes), followed by `num_global_refactoring_level` (`-gl`, default -1 = as many as possible) global multigrid levels over the coarsest block-local region. Each block-local level becomes one MDR level and its coefficients are written directly into the level buffer, so no interleaving pass is needed for them. The level counts are stored in the header, so reconstruction needs no extra options. The hierarchical basis (no L2 projection) is used. Adaptive-resolution reconstruction is not supported in this mode; it falls back to full resolution.

### L2 error bound (MDR-X)
An L2 request (`-s 0`, any finite `s` is treated the same way) bounds the discrete norm `sqrt(sum_i (x_i - x'_i)^2)` of the reconstruction error, for both the global and the hybrid decomposition. The refactor records, for every level and every number of retrieved bitplanes, the exact squared coefficient error of the truncating bitplane decoder. Since every recomposition stage is convex multilinear interpolation, recomposing level `l` alone amplifies the L2 norm by at most `sqrt(2^(D * s_l))`, where `s_l` is the number of stages finer than level `l`. The retrieval plan keeps `sum_l sqrt(2^(D * s_l) * E_l) <= tol` (triangle inequality over levels), and splits the tolerance as `tol / sqrt(#subdomains)` under domain decomposition. The bound is rigorous and therefore conservative.

### L-inf error bound (MDR-X)
With the hierarchical basis, every recomposition stage (global multilinear or block-local) sets each new node to its coefficient plus a convex combination of coarser nodes, so the L-inf error of the reconstruction is at most the sum over levels of each level's largest coefficient error. The binary bitplane encoder truncates, so with `b_l` of its bitplanes retrieved that error is below `2^(exp_l - b_l)`, where `2^exp_l` bounds the level's coefficients. The retrieval plan keeps `sum_l 2^(exp_l - b_l) <= tol - rho`; with the NegaBinary encoder each term is multiplied by 4 (2 bits of range headroom). The bound holds in exact arithmetic; `rho` = 2 ulp of `sum_l max|level l coefficients|` leaves room for floating-point rounding, which only matters for single precision near the data's precision. A tolerance below the precision of the data type itself (e.g. below one ulp of the largest single-precision value) cannot be met.

### Bitplane groups (MDR-X)
Consecutive bitplanes of a level are compressed together in merged groups of `config.mdr_bitplane_group_size` bitplanes (`mdr-x -z ... -gs <1-4>`, default 2). A group is also the unit of retrieval: a request reads whole groups, so smaller groups retrieve closer to the requested error (each group of `g` bitplanes lowers a level's error bound by `2^g`) at the cost of more, smaller compressed streams. The group size is recorded in the metadata, so reconstruction needs no option.

### Bitplane word order (MDR-X)
The binary bitplane encoder packs each bitplane of a level into 32-bit words. By default (`config.mdr_contiguous_words`) word `w` holds the bits of the 32 consecutive coefficients `32w .. 32w+31` (coefficient `32w + d` at bit `d`), so that a word covers neighboring coefficients and zero words are common; this is what makes zero elimination effective, and it also helps Huffman. `mdr-x -z ... -sw` keeps the earlier layout (32 coefficients strided over the level), which the NegaBinary encoder always uses. On CUDA the contiguous layout is encoded and decoded by warp kernels (a 32-lane bit transpose per word). The layout is recorded in the metadata.

### Zero elimination (MDR-X)
With `config.mdr_zero_elimination` (`mdr-x -z ... -ze`, off by default) every bitplane group is stored with zero elimination instead of Huffman/RLE/zstd. Each row of the group is split into chunks of 32 words; a chunk has a 32-bit bitmap of its nonzero words, and only the nonzero words are stored. One more bit per chunk marks the chunks with a nonzero bitmap and only those bitmaps are stored, so a sparse row costs little more than its nonzero words (with one bitmap per chunk, the bitmaps were most of what loose requests read: 86% on NYX at a relative tolerance of 1e-2). The offsets of every 256 chunks are stored so that thread blocks decode independently. A row whose packed form would not be smaller is stored raw. The group stream starts with a signature (`MGXZ2LM`; `MGXZELM` for data with one bitmap per chunk, which is still read), its row count and row length, and the nonzero-word and nonzero-chunk counts of each row, so reconstruction detects the format by itself. Zero elimination is a single data-parallel pass in both directions; on CUDA the encoder writes the chunk bitmaps along with the bitplanes and all levels are sized with one device-to-host transfer. Other backends use simpler kernels.

By default (`config.mdr_sparse_words`) a row may instead store each nonzero word as a 3-bit code and a payload: the positions of its one bits (5 bits each) when it has at most 6 of them, else the 32 bits. The codes and payloads of every 256 chunks form a word-aligned region whose offset is stored with the chunk offsets, so thread blocks still decode independently (the decoder stages the region in shared memory). Each row takes the smaller of the two forms (or raw). Words with few one bits are what the bitplanes of loose requests are mostly made of: at a relative L-inf tolerance of 1e-2 this reads 25% less on NYX, 12% less on Miranda, 11% less on Hurricane and 21% less on XGC, and the full size is 1-2% smaller. Groups with sparse-word rows have the signature `MGXZ3LM` (`MGXZ3SG` with significance-coded signs) and the region words of each row in their header. On CUDA the encoder also writes the payload bits of every chunk. `mdr-x -z ... -ze -dw` stores whole words only.

### Significance-coded signs (MDR-X)
With zero elimination and contiguous words, the sign of a coefficient is by default (`config.mdr_significance_signs`) stored with the bitplane group in which the coefficient becomes nonzero, not in a sign row of the first group. A request then reads the signs of the coefficients that it reconstructs as nonzero and no others: at loose tolerances, where most coefficients are still zero, the sign row was a large part of the data read. Each group stream (signature `MGXZ2SG`) ends with a sign section holding one bit per coefficient that becomes nonzero in the group, in coefficient order, preceded by the word offset of every 65536 coefficients so that thread blocks decode independently; the full-precision size is about the same as with a sign row. The decoder finds which coefficients become nonzero from the bitplanes it decodes and from a per-coefficient state kept with the signs of the level. On CUDA the encoder packs the signs while encoding, and the decoder resolves them from the bitplane words it holds; other backends use simpler kernels (also on CUDA with `MGARDX_MDR_PORTABLE_KERNELS=1`, for testing; this also selects the portable zero-elimination kernels). `mdr-x -z ... -ze -rs` keeps the sign row.

### Data format version (MDR-X)
The header of refactored data records a format version (currently 5). Version 1 stores the sign bits of the binary bitplane encoder once per level, as an extra row in the first merged bitplane group, instead of reserving a sign slot in every bitplane; incompressible bitplane groups are no longer half zeros. Version 2 also records the bitplane group size; version 1 data (groups of 4 bitplanes) is still read. Version 3 also records the bitplane word order; version 1 and 2 data (strided words) is still read. Version 4 also records the sign coding (significance-coded signs); version 1-3 data is still read. Version 5 zero-elimination groups may have sparse-word rows (identified by their signature); version 1-4 data is still read. Data refactored by an earlier MDR-X (version 0) cannot be read by this version: requests and reconstruction stop with an error asking to refactor the data again.

## Configure and build

Both MDR and MDR-X are automatically built together with MGARD-X. Please follow the [instruction of MGARD-X][mgard-x-build] to build MDR and MDR-X.

[mgard-x-build]: MGARD-X.md

## Use
### Header files
Inculde the follow header files to use MDR or MDR-X:

* MDR: `mgard/mdr.hpp`
* MDR-X: `mgard/mdr-x.hpp`


### APIs

The APIs of MDR and MDR-X are designed in a way such that the refactoring the reconstruction process are highly customizable to satisify users needs. Here lists the key components of MDR and MDR-X.  

* **Decomposer:** Responsible for transforming original multidimensional data to multilevel coefficients (decompose) and the other way around (recompose). This could be any external decorrelation algorithm such as MGARD and wavelet transforms. Currently MDR supports MGARD decomposer (multilinear interpolation with L2 projection, see `MGARDOrthoganalDecomposer`) and hierarchical decomposer (multilinear interpolation, see `MGARDHierarchicalDecomposer`).
 
* **ErrorCollector:** Responsible for collecting error information that is required for error estimation during retrieval. Currently MDR implements a max error collector (collecting the maximum coefficients in each level, see `MaxErrorCollector`) and squared error collector (collecting the sum of squared error in each level, see `SquaredErrorCollector`).
 
* **ErrorEstimator:** Responsible for estimating the error for each precision fragment based on the collected error information. Currently MDR implements a max error estimator (see `MaxErrorEstimator`) and a L2 error estimator (see `SquaredErrorEstimator`) for the two decomposition methods supported.
 
* **Interleaver:** Responsible for linearizing the multidimensional level coefficients to 1D for precision encoding. Currently MDR supports a direct interleaver (linearizing coefficients one by one, see `DirectInterleaver`), a blocked based interleaver (linearizing coefficients in blocks, see `BlockedInterleaver`), and a space-filling-curve based one (linearizing coefficients using specific space filling curves, see `SFCInterleaver`).
 
* **LosslessCompressor:** Responsible for lossless compressing the encoding bit-planes (using ZSTD in the implementation). Currently MDR implements a null compressor (performing no lossless compression, see `NullLevelCompressor`), a default compressor (losslessly compressing each bit-plane, see `DefaultLevelCompressor`),  and an adaptive compressor (compressing the first a few bit-planes based on the characteristics, see `AdaptiveLevelCompressor`).

* **SizeInterpreter:** Responsible for interpreting which precision fragment to fetch upon retrieval. Currently MDR implements an in-order size interpreter (fetching from coarse level to fine level based on bit-plane order, see `InorderSizeInterpreter`), a round-robin size interpreter (fetching one bit-plane per level, see `RoundRobinSizeInterpreter`), and three greedy-based size interpreters (fetching based on the error impact, or efficiency defined in the paper, see `GreedyBasedSizeInterpreter`, `SignExcludeGreedyBasedSizeInterpreter`, `NegaBinaryGreedyBasedSizeInterpreter`).
 
* **Writer:** Responsible for writing precision fragments to files. Users are suggested to implement a derived class to write with their preferred formats and I/O libraries. Currently MDR implements a concatenated writer (writing each level in one file, see `ConcatLevelFileWriter`) and a fragment writer (writing aggregated precision fragments in one file, see `HPSSFileWriter`).
 
* **Retriever:**  Responsible for reading precision fragments (inverse operations of the writer). Will be merged into the writer class in future release.
 
* **Refactor:** Responsible for constructing a data refactor using the components above. Users are encouraged to pick any component according to their needs.
 
* **Reconstructor:** Responsible for constructing a data reconstructor (inverse operations of the Refactor). Will be merged into the Refactor class in future release.

## Example Code
* MDR example code can be found in [here][mdr-example].
* MDR-X example code can be found in [here][mdr-x-example].

[mdr-example]: ../examples/mgard-x/MDR
[mdr-x-example]: ../examples/mgard-x/MDR-X
