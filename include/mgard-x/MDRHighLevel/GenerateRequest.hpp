/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: September 21, 2026
 */

#ifndef MGARD_X_MDR_GENERATE_PIPELINE_HPP
#define MGARD_X_MDR_GENERATE_PIPELINE_HPP

namespace mgard_x {
namespace MDR {

template <DIM D, typename T, typename DeviceType>
void generate_request(DomainDecomposer<D, T, ComposedRefactor<D, T, DeviceType>,
                                       DeviceType> &domain_decomposer,
                      Config config, RefactoredMetadata &refactored_metadata) {

  for (int subdomain_id = 0; subdomain_id < domain_decomposer.num_subdomains();
       subdomain_id++) {
    Hierarchy<D, T, DeviceType> hierarchy =
        domain_decomposer.subdomain_hierarchy(subdomain_id);
    ComposedReconstructor<D, T, DeviceType> reconstructor(hierarchy, config);
    MDRMetadata &metadata = refactored_metadata.metadata[subdomain_id];
    // L2 errors of disjoint subdomains add in quadrature: give each subdomain
    // tol / sqrt(#subdomains) so the whole domain meets tol. L-inf needs no
    // split.
    double requested_tol = metadata.requested_tol;
    if (metadata.requested_s != std::numeric_limits<double>::infinity()) {
      metadata.requested_tol =
          requested_tol / std::sqrt((double)domain_decomposer.num_subdomains());
    }
    reconstructor.GenerateRequest(metadata);
    metadata.requested_tol = requested_tol;
  }
}

} // namespace MDR
} // namespace mgard_x
#endif