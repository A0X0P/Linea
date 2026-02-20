
/**
 * @file LU.hpp
 * @author A.N. Prosper
 * @date Febuary 8th 2026
 * @brief LU decomposition with partial pivoting.
 *
 * Computes:
 *
 *      P A = L U
 *
 * where:
 *      - P is a permutation matrix
 *      - L is unit lower triangular
 *      - U is upper triangular
 *
 * Uses partial pivoting for numerical stability.
 *
 * Time Complexity:
 *      O(2n³ / 3)
 *
 * Rank determination:
 *      Based on pivot tolerance.
 */

#ifndef LINEA_LU_HPP
#define LINEA_LU_HPP

#include "../Core/Concepts.hpp"
#include "../Core/PlatformMacros.hpp"
#include "../Matrix/Matrix.hpp"
#include "../Vector/Vector.hpp"
#include <cmath>

namespace Linea::Decompositions {

/**
 * @brief Metadata produced by LU factorization.
 *
 * Stores structural information computed during
 * LU decomposition with partial pivoting:
 *
 *      P A = L U
 *
 * Contains:
 *      - Row permutation encoding P
 *      - Numerical rank
 *      - Number of row swaps performed
 *
 * The permutation vector defines P implicitly:
 *
 *      P(i, permutation_vector[i]) = 1
 *
 * and zeros elsewhere.
 *
 * Rank is determined using pivot magnitude thresholding
 * against the configured tolerance.
 *
 * swap_count equals the number of row interchanges,
 * which is relevant for determinant sign computation:
 *
 *      det(A) = (-1)^{swap_count} · ∏ U(i,i)
 *
 * @note permutation_vector.size() == number of rows.
 */

struct LUInfo {
  Vector<std::size_t> permutation_vector{0};
  std::size_t rank;
  std::size_t swap_count;
};

/**
 * @tparam T Floating-point scalar type.
 *
 * Provides:
 *      - Rank computation
 *      - Permutation tracking
 *      - Row swap count
 *      - Extraction of L, U, and P
 *
 * Stability:
 *      Partial pivoting significantly improves robustness
 *      but does not guarantee full stability for pathological matrices.
 */

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

  /**
   * @brief Returns decomposition metadata.
   *
   * Provides access to:
   *      - Permutation vector
   *      - Numerical rank
   *      - Row swap count
   *
   * @return Const reference to LUInfo structure.
   *
   * @note Returned reference remains valid for the lifetime
   *       of the LU object.
   */

  const LUInfo &info() const noexcept { return info_; }

  /**
   * @brief Returns the numerical rank of the matrix.
   *
   * Rank is determined during factorization using pivot
   * magnitude thresholding against the computed tolerance.
   *
   * @return Numerical rank.
   *
   * @note Rank ≤ min(m, n).
   */

  std::size_t rank() const noexcept { return info_.rank; }

  /**
   * @brief Returns the number of row swaps performed.
   *
   * Each swap corresponds to a permutation step during
   * partial pivoting.
   *
   * @return Total row swap count.
   *
   * @note Useful for determinant sign computation.
   */

  std::size_t swap_count() const noexcept { return info_.swap_count; }

  /**
   * @brief Returns the row permutation vector.
   *
   * The vector encodes the permutation matrix P such that:
   *
   *      P A = L U
   *
   * Entry permutation()[i] gives the original row index
   * moved to position i.
   *
   * @return Const reference to permutation vector.
   */

  const Vector<std::size_t> &permutation() const noexcept {
    return info_.permutation_vector;
  }

  /**
   * @brief Returns the unit lower triangular factor L.
   *
   * Constructs:
   *
   *      L ∈ ℝ^{m × min(m,n)}
   *
   * with unit diagonal and strictly lower portion stored
   * in the internal LU data.
   *
   * @return Explicit L matrix.
   *
   * @note Diagonal entries are 1 by definition.
   */

  Matrix<T> L() { return extract_L(); }

  /**
   * @brief Returns the upper triangular factor U.
   *
   * Constructs:
   *
   *      U ∈ ℝ^{min(m,n) × n}
   *
   * from the upper triangular portion of the internal LU data.
   *
   * @return Explicit U matrix.
   */

  Matrix<T> U() { return extract_U(); }

  /**
   * @brief Returns the permutation matrix P.
   *
   * Constructs a permutation matrix such that:
   *
   *      P A = L U
   *
   * where P encodes the row swaps performed during
   * partial pivoting.
   *
   * @return Explicit permutation matrix.
   *
   * @note Construction cost: O(m²)
   */

  Matrix<T> P() { return extract_P(); }

private:
  /**
   * @brief Performs in-place LU factorization with partial pivoting.
   *
   * Pivot strategy:
   *      Select row with maximum |a_ik| for current column.
   *
   * Tolerance:
   *      If not provided, computed as:
   *
   *          ε * max_element * max(m, n)
   *
   * Used for rank detection.
   */

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
        lu = data.raw();
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

  /**
   * @brief Internal extraction of L factor.
   *
   * Builds the unit lower triangular matrix L: m × min(m,n), unit diagonal from
   * the compact LU storage.
   *
   * @return Explicit L matrix.
   *
   * @note Does not modify internal state.
   */
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

  /**
   * @brief Internal extraction of U factor.
   *
   * Builds the upper triangular matrix U: min(m,n) × n from
   * the compact LU storage.
   *
   * @return Explicit U matrix.
   *
   * @note Does not modify internal state.
   */

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

  /**
   * @brief Internal construction of permutation matrix.
   *
   * Converts the stored permutation vector into an
   * explicit permutation matrix P.
   *
   * @return Explicit permutation matrix.
   *
   * @note P satisfies:
   *          P A = L U
   */

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
