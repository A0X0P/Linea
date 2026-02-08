// created by : A.N. Prosper
// date : febuary 8th 2026
// time : 17:27

#ifndef LINEA_LU_HPP
#define LINEA_LU_HPP

#include "../Core/Concepts.hpp"
#include "../Core/PlatformMacros.hpp"
#include "../Matrix/Matrix.hpp"
#include "../Vector/Vector.hpp"
#include <cmath>

namespace Linea::Decompositions {
struct LUInfo {
  Vector<std::size_t> permutation_vector{0};
  std::size_t rank;
  std::size_t swap_count;
};

template <RealType T> class LU {

private:
  Matrix<T> data;
  T tolerance;
  LUInfo info_;

public:
  LU(const Matrix<T> &matrix, T epsilon = T{0})
      : data(matrix), tolerance(epsilon) {
    compute();
  }

  const LUInfo &info() const noexcept { return info_; }
  std::size_t rank() const noexcept { return info_.rank; }
  std::size_t swap_count() const noexcept { return info_.swap_count; }

  const Vector<std::size_t> &permutation() const noexcept {
    return info_.permutation_vector;
  }

  Matrix<T> L() { return extract_L(); }
  Matrix<T> U() { return extract_U(); }
  Matrix<T> P() { return extract_P(); }

private:
  auto compute() -> void {
    const std::size_t m = data.nrows();
    const std::size_t n = data.ncols();
    const std::size_t k_max = std::min(m, n);

    T *RESTRICT lu = data.raw();

    info_.rank = 0;
    info_.swap_count = 0;
    info_.permutation_vector = Vector<std::size_t>(m);

    for (std::size_t i = 0; i < m; ++i)
      info_.permutation_vector[i] = i;

    if (tolerance == T{0}) {
      T max_elem = T{0};
      for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j)
          max_elem = std::max(max_elem, std::abs(lu[i * n + j]));

      tolerance = std::numeric_limits<T>::epsilon() * max_elem * std::max(m, n);
    }

    for (std::size_t k = 0; k < k_max; ++k) {

      std::size_t pivot = k;
      T max_val = std::abs(lu[k * n + k]);

      for (std::size_t i = k + 1; i < m; ++i) {
        T v = std::abs(lu[i * n + k]);
        if (v > max_val) {
          max_val = v;
          pivot = i;
        }
      }

      if (max_val < tolerance)
        continue;

      if (pivot != k) {
        data.swap_row(k, pivot);
        std::swap(info_.permutation_vector[k], info_.permutation_vector[pivot]);
        lu = data.raw(); // pointer may change
        ++info_.swap_count;
      }

      const T pivot_val = lu[k * n + k];

      for (std::size_t i = k + 1; i < m; ++i) {
        lu[i * n + k] /= pivot_val;

        const T lik = lu[i * n + k];
        for (std::size_t j = k + 1; j < n; ++j)
          lu[i * n + j] -= lik * lu[k * n + j];
      }

      ++info_.rank;
    }
  }

  // L: m × min(m,n), unit diagonal
  Matrix<T> extract_L() const {
    const std::size_t m = data.nrows();
    const std::size_t n = data.ncols();
    const std::size_t k = std::min(m, n);

    const T *RESTRICT lu = data.raw();
    Matrix<T> L(m, k, T{});
    T *RESTRICT l = L.raw();

    for (std::size_t i = 0; i < m; ++i) {
      if (i < k)
        l[i * k + i] = T{1};

      for (std::size_t j = 0; j < std::min(i, k); ++j)
        l[i * k + j] = lu[i * n + j];
    }
    return L;
  }

  // U: min(m,n) × n
  Matrix<T> extract_U() const {
    const std::size_t m = data.nrows();
    const std::size_t n = data.ncols();
    const std::size_t k = std::min(m, n);

    const T *RESTRICT lu = data.raw();
    Matrix<T> U(k, n, T{});
    T *RESTRICT u = U.raw();

    for (std::size_t i = 0; i < k; ++i)
      for (std::size_t j = i; j < n; ++j)
        u[i * n + j] = lu[i * n + j];

    return U;
  }

  // P such that P*A = L*U
  Matrix<T> extract_P() const {
    const std::size_t m = info_.permutation_vector.size();
    Matrix<T> P(m, m, T{});
    T *RESTRICT pmat = P.raw();

    for (std::size_t i = 0; i < m; ++i)
      pmat[i * m + info_.permutation_vector[i]] = T{1};

    return P;
  }
};

} // namespace Linea::Decompositions
#endif
