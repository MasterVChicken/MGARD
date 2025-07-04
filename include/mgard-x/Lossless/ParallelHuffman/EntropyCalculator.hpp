/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_ENTROPY_CALCULAOR_HPP
#define MGARD_X_ENTROPY_CALCULAOR_HPP

namespace mgard_x {

template <typename DeviceType>
double CalculateLC(SIZE total_num_elems, SIZE dict_size,
                   SubArray<1, unsigned int, DeviceType> freq,
                   SubArray<1, unsigned int, DeviceType> CL, int queue_idx) {
  unsigned int *_freq = new unsigned int[dict_size];
  unsigned int *_cl = new unsigned int[dict_size];
  MemoryManager<DeviceType>::Copy1D(_freq, freq.data(), dict_size, queue_idx);
  MemoryManager<DeviceType>::Copy1D(_cl, CL.data(), dict_size, queue_idx);
  DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
  double LC = 0;
  // for (SIZE i = 0; i < dict_size; i++) {
  //   std::cout << _freq[i] << " ";
  // }
  // std::cout << "\n";
  // for (SIZE i = 0; i < dict_size; i++) {
  //   std::cout << _cl[i] << " ";
  // }
  // std::cout << "\n";
  for (SIZE i = 0; i < dict_size; i++) {
    LC += (double)_freq[i] / total_num_elems * _cl[i];
  }
  delete[] _freq;
  delete[] _cl;
  return LC;
}

template <typename DeviceType>
double CalculateEntropy(SIZE total_num_elems, SIZE dict_size,
                        SubArray<1, unsigned int, DeviceType> freq,
                        int queue_idx) {
  unsigned int *_freq = new unsigned int[dict_size];
  MemoryManager<DeviceType>::Copy1D(_freq, freq.data(), dict_size, queue_idx);
  DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
  double entropy = 0;
  // printf("dict size: %u\n", dict_size);
  for (SIZE i = 0; i < dict_size; i++) {
    double p = (double)_freq[i] / total_num_elems;
    // printf("%u ", _freq[i]);
    entropy += -1 * log2(p) * p;
  }
  // printf("\n");

  delete[] _freq;
  return entropy;
}

} // namespace mgard_x
#endif