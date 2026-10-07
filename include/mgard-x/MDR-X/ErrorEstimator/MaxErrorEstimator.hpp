#ifndef _MDR_MAX_ERROR_ESTIMATOR_HPP
#define _MDR_MAX_ERROR_ESTIMATOR_HPP

#include "ErrorEstimatorInterface.hpp"
namespace mgard_x {
namespace MDR {
template <class T>
class MaxErrorEstimator : public concepts::ErrorEstimatorInterface<T> {};
// max error estimator for orthogonal basis
template <class T> class MaxErrorEstimatorOB : public MaxErrorEstimator<T> {
public:
  MaxErrorEstimatorOB(int num_dims) {
    switch (num_dims) {
    case 1:
      c = 1.0 + sqrt(3) / 2;
      break;
    case 2:
      c = 1.0 + 9.0 / 4;
      break;
    case 3:
      c = 1.0 + 21.0 * sqrt(3) / 8;
      break;
    default:
      throw std::runtime_error(
          std::to_string(num_dims) +
          "-Dimentional error estimation not implemented.");
    }
    c *= 4; // 2 more bitplane for negabinary
  }
  MaxErrorEstimatorOB() : MaxErrorEstimatorOB(1) {}

  inline T estimate_error(T error, int level) const { return c * error; }
  inline T estimate_error(T data, T reconstructed_data, int level) const {
    return c * (data - reconstructed_data);
  }
  inline T estimate_error_gain(T base, T current_level_err, T next_level_err,
                               int level) const {
    return c * (current_level_err - next_level_err);
  }
  void print() const {
    std::cout << "Max absolute error estimator (up to 3 dimensions) for "
                 "orthogonal basis."
              << std::endl;
  }

private:
  // derived constant
  T c = 0;
};
// max error estimator for hierarchical basis
// Every recomposition stage of the hierarchical basis (the global multilinear
// one and the block-local one) sets each new node to its coefficient plus a
// convex combination of coarser nodes, so the L-inf error of the
// reconstruction is at most the sum over levels of each level's largest
// coefficient error (exact arithmetic). With the binary encoder that error is
// below 2^(exp - b) for b bitplanes (MaxErrorCollector), so c = 1.
// NegaBinary shifts the encoder's fixed-point exponent by 2 extra bits of
// range headroom (EncodeNegaBinary, `exp += 2`), so a given bitplane count
// buys 4x less precision: c = 4, as MaxErrorEstimatorOB also discounts.
template <class T> class MaxErrorEstimatorHB : public MaxErrorEstimator<T> {
public:
  explicit MaxErrorEstimatorHB(bool negabinary) : c(negabinary ? 4 : 1) {}
  inline T estimate_error(T error, int level) const { return c * error; }
  inline T estimate_error(T data, T reconstructed_data, int level) const {
    return c * (data - reconstructed_data);
  }
  inline T estimate_error_gain(T base, T current_level_err, T next_level_err,
                               int level) const {
    return c * (current_level_err - next_level_err);
  }
  void print() const {
    std::cout << "Max absolute error estimator for hierarchical basis."
              << std::endl;
  }

private:
  T c;
};
} // namespace MDR
} // namespace mgard_x
#endif
