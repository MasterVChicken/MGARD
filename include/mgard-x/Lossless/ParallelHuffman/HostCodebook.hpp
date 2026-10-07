/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: October 7, 2026
 */

#ifndef MGARD_X_HOST_CODEBOOK_HPP
#define MGARD_X_HOST_CODEBOOK_HPP

#include "../../RuntimeX/Utilities/Exceptions.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace mgard_x {

// Builds on the host the codebook and the decode tables that GetCodebook
// builds on the device, in the same layout: symbols sorted by ascending
// frequency (stable, as the device radix sort), canonical codewords as
// GenerateCW assigns them (emulated step by step), the codebook reversed and
// reordered by symbol, and the decodebook [first | entry | qcode]. For small
// dictionaries the device kernels are latency bound (~0.3 ms per codebook)
// while this takes microseconds.
//
// Code lengths come from a sequential Huffman construction instead of
// GenerateCL. Every Huffman code has the same (optimal) total length, so the
// encoded size is the same; on ties the individual lengths may differ, and
// the decode tables describe the code actually used.
//
// Returns false (and builds nothing) when fewer than two symbols occur; the
// caller then uses the device path.
template <typename Q, typename H>
bool HostCodebook(int dict_size, const unsigned int *freq,
                  std::vector<H> &codebook, std::vector<uint8_t> &decodebook,
                  double &total_bits) {
  const int type_bw = sizeof(H) * 8;

  // SortByKey: ascending frequency, ties in symbol order.
  std::vector<int> order(dict_size);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(),
                   [&](int a, int b) { return freq[a] < freq[b]; });
  int first_nonzero = 0;
  while (first_nonzero < dict_size && freq[order[first_nonzero]] == 0) {
    first_nonzero++;
  }
  const int n = dict_size - first_nonzero;
  if (n < 2) {
    return false;
  }

  // Code lengths: two-queue Huffman construction over the sorted leaves.
  std::vector<uint64_t> weight(2 * n - 1);
  std::vector<int> parent(2 * n - 1, 0);
  for (int j = 0; j < n; j++) {
    weight[j] = freq[order[first_nonzero + j]];
  }
  int leaf = 0, internal = n, next = n;
  auto pick = [&]() {
    if (leaf < n && (internal >= next || weight[leaf] <= weight[internal])) {
      return leaf++;
    }
    return internal++;
  };
  for (; next < 2 * n - 1; next++) {
    int a = pick();
    int b = pick();
    weight[next] = weight[a] + weight[b];
    parent[a] = parent[b] = next;
  }
  std::vector<unsigned int> depth(2 * n - 1, 0);
  for (int i = 2 * n - 3; i >= 0; i--) {
    depth[i] = depth[parent[i]] + 1;
  }
  // GenerateCL order: ascending frequency, so non-increasing length.
  std::vector<unsigned int> CL(depth.begin(), depth.begin() + n);
  std::sort(CL.begin(), CL.end(), std::greater<unsigned int>());

  int max_CW_bits = type_bw - 8;
  if ((int)CL[0] > max_CW_bits) {
    throw ProcessingException(
        "Cannot store all Huffman codewords in " +
        std::to_string(max_CW_bits + 8) +
        "-bit representation; representation requires at least " +
        std::to_string(CL[0] + 8) +
        " bits (longest codeword: " + std::to_string(CL[0]) + " bits)");
  }

  // GenerateCW, one operation at a time. first/entry start as the device
  // decodebook does after HuffmanWorkspace::reset (all bytes 0xff).
  const H H_MAX = std::numeric_limits<H>::max();
  std::vector<H> first(type_bw, H_MAX), entry(type_bw, H_MAX);
  std::vector<H> CW(n, 0);
  std::reverse(CL.begin(), CL.end());
  int CCL = CL[0], CDPI = 0, newCDPI = n - 1;
  entry[CCL] = 0;
  CW[0] = 0;
  first[CCL] = CW[0] ^ (((H)1 << (H)CL[0]) - 1);
  entry[CCL + 1] = 1;
  for (int i = 0; i < CCL; i++) {
    first[i] = H_MAX;
    entry[i] = 0;
  }
  while (CDPI < n - 1) {
    for (int i = 0; i < n - 1; i++) {
      if ((int)CL[i + 1] > CCL) {
        newCDPI = std::min(newCDPI, i);
      }
    }
    int updateEnd = (newCDPI >= n - 1) ? type_bw : (int)CL[newCDPI + 1];
    H curEntryVal = entry[CCL];
    int numCCL = newCDPI - CDPI + 1;
    CW[newCDPI] = (CDPI == 0) ? 0 : CW[CDPI];
    for (int i = CDPI; i < newCDPI; i++) {
      CW[i] = CW[newCDPI] + (newCDPI - i);
    }
    for (int i = CCL + 1; i < updateEnd; i++) {
      entry[i] = curEntryVal + numCCL;
    }
    if (updateEnd < type_bw) {
      entry[updateEnd] = curEntryVal + numCCL;
    }
    first[CCL] = CW[CDPI] ^ (((H)1 << (H)CL[CDPI]) - 1);
    for (int i = CCL + 1; i < updateEnd; i++) {
      first[i] = H_MAX;
    }
    if (newCDPI < n - 1) {
      int CLDiff = CL[newCDPI + 1] - CL[newCDPI];
      CW[newCDPI + 1] = ((CW[CDPI] + 1) << CLDiff);
      CCL = CL[newCDPI + 1];
      ++newCDPI;
    }
    CDPI = newCDPI;
    newCDPI = n - 1;
  }
  for (int i = 0; i < n; i++) {
    CW[i] = (CW[i] | (((H)CL[i] & (H)0xffu) << (type_bw - 8))) ^
            (((H)1 << (H)CL[i]) - 1);
  }
  std::reverse(CW.begin(), CW.end());

  // Reverse codebook and qcode, then reorder the codebook by symbol.
  std::vector<H> sorted_codebook(dict_size, 0);
  std::copy(CW.begin(), CW.end(), sorted_codebook.begin() + first_nonzero);
  std::reverse(sorted_codebook.begin(), sorted_codebook.end());
  std::vector<Q> qcode(dict_size);
  for (int i = 0; i < dict_size; i++) {
    qcode[i] = (Q)order[dict_size - 1 - i];
  }
  codebook.assign(dict_size, 0);
  for (int i = 0; i < dict_size; i++) {
    codebook[qcode[i]] = sorted_codebook[i];
  }

  decodebook.resize(sizeof(H) * 2 * type_bw + sizeof(Q) * dict_size);
  std::memcpy(decodebook.data(), first.data(), sizeof(H) * type_bw);
  std::memcpy(decodebook.data() + sizeof(H) * type_bw, entry.data(),
              sizeof(H) * type_bw);
  std::memcpy(decodebook.data() + sizeof(H) * 2 * type_bw, qcode.data(),
              sizeof(Q) * dict_size);

  total_bits = 0;
  for (int s = 0; s < dict_size; s++) {
    total_bits += (double)freq[s] * (double)(codebook[s] >> (type_bw - 8));
  }
  return true;
}

} // namespace mgard_x

#endif
