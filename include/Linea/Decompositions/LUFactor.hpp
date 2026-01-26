// created by : A.N. Prosper
// date : January 24th 2026
// time : 20:09

#ifndef LINEA_LUFACTOR_H
#define LINEA_LUFACTOR_H

#include "../Matrix/Matrix.hpp"
#include "../Vector/Vector.hpp"

namespace Linea {

template <NumericType M> class LUFactor {

private:
  Matrix<M> LU;
  LUResult<M> info;

public:
  // Constructor
  LUFactor(Matrix<M> lu, LUResult<M> result)
      : LU(std::move(lu)), info(std::move(result)) {}

  // Getters
  std::size_t get_rank() const { return info.rank; }

  std::size_t get_swap_count() const { return info.swap_count; }

  const Vector<M> &get_permutation_vector() const {
    return info.permutation_vector;
  }

  const LUResult<M> &get_Info() const { return info; }

  // Extract L as m × min(m,n) lower triangular matrix with unit diagonal
  Matrix<M> extract_L() const {
    const std::size_t m = LU.nrows();
    const std::size_t n = LU.ncols();
    const std::size_t k_max = std::min(m, n);

    Matrix<M> L(m, k_max, M{});

    for (std::size_t i{}; i < m; i++) {
      // unit diagonal
      if (i < k_max) {
        L(i, i) = M{1};
      }

      // subdiagonal entries
      for (std::size_t j{}; j < std::min(i, k_max); j++) {
        L(i, j) = LU(i, j);
      }
    }
    return L;
  }

  // Extract U as min(m,n) × n upper triangular matrix
  Matrix<M> extract_U() const {
    const std::size_t m = LU.nrows();
    const std::size_t n = LU.ncols();
    const std::size_t k_max = std::min(m, n);

    Matrix<M> U(k_max, n, M{});

    for (std::size_t i{}; i < k_max; i++) {
      for (std::size_t j = i; j < n; j++) {
        U(i, j) = LU(i, j);
      }
    }
    return U;
  }

  // Extract permutation matrix P (m × m) such that P*A = L*U
  Matrix<M> extract_P() const {
    const std::size_t m = LU.nrows();
    Matrix<M> P(m, m, M{});

    for (std::size_t i{}; i < m; i++) {
      P(i, info.permutation_vector[i]) = M{1};
    }
    return P;
  }
};

} // namespace Linea

#endif // LINEA_LUFACTOR_H