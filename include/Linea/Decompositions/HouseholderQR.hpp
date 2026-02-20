
/**
 * @file HouseholderQR.hpp
 * @author A.N. Prosper
 * @date January 24th 2026
 * @brief QR factorization using Householder reflections.
 *
 * Computes:
 *
 *      A = Q R
 *
 * where:
 *      - Q is orthogonal (QᵀQ = I)
 *      - R is upper triangular
 *
 * Uses in-place storage of reflectors.
 *
 * Time Complexity:
 *      O(2mn² − 2n³/3)
 *
 * Rank Detection:
 *      Determined via column norm thresholding.
 *
 * Numerical Stability:
 *      Backward stable.
 *      Preferred over normal equations for least squares.
 */

#ifndef LINEA_HOUSEHOLDER_QR_HPP
#define LINEA_HOUSEHOLDER_QR_HPP

#include "../Core/Concepts.hpp"
#include "../Core/PlatformMacros.hpp"
#include "../Core/Types.hpp"
#include "../Matrix/Matrix.hpp"
#include "../Vector/Vector.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace Linea::Decompositions {

/**
 * @tparam T Floating-point scalar type.
 *
 * Stores:
 *      - QR matrix with reflectors
 *      - Householder coefficients (betas)
 *      - Numerical rank
 *
 * Supports:
 *      - Thin and Full decompositions
 *      - Least squares solving
 *      - Explicit Q and R extraction
 */

template <RealType T> class HouseholderQR {
private:
  Matrix<T> QR_;
  Vector<T> betas_;
  std::size_t m_, n_;
  std::size_t rank_;
  T tol_;

public:
  explicit HouseholderQR(const Matrix<T> &A)
      : QR_(A), betas_(std::min(A.nrows(), A.ncols())), m_(A.nrows()),
        n_(A.ncols()), rank_(0) {
    tol_ = std::numeric_limits<T>::epsilon() * std::max(m_, n_);
    factorize();
  }

  /**
   * @brief Returns the numerical rank of the matrix.
   *
   * The rank is determined during factorization by thresholding
   * the column 2-norm against:
   *
   *      tol = ε · max(m, n)
   *
   * where ε is machine precision.
   *
   * A column is considered linearly independent if its norm
   * exceeds this threshold.
   *
   * @return Numerical rank of A.
   *
   * @note Rank is computed without column pivoting.
   */

  std::size_t rank() const noexcept { return rank_; }

  // --- Accessors ---

  /**
   * @brief Extracts the upper triangular matrix R.
   *
   * Constructs R from the upper triangular portion of the
   * internally stored QR matrix.
   *
   * Modes:
   *      - ComputeMode::Thin  → returns rank × n matrix
   *      - ComputeMode::Full  → returns m × n matrix
   *
   * @param mode Specifies thin or full decomposition.
   * @return Explicit R matrix.
   *
   * @note Entries below the diagonal are zero.
   */

  Matrix<T> R(ComputeMode mode = ComputeMode::Thin) const {
    std::size_t target_rows = (mode == ComputeMode::Full) ? m_ : rank_;
    Matrix<T> Rmat(target_rows, n_);

    const T *RESTRICT QRp = QR_.raw();
    T *RESTRICT Rp = Rmat.raw();

    for (std::size_t i = 0; i < target_rows; ++i) {
      for (std::size_t j = i; j < n_; ++j) {
        if (i < m_) {
          Rp[i * n_ + j] = QRp[i * n_ + j];
        }
      }
    }
    return Rmat;
  }

  /**
   * @brief Forms the explicit orthogonal matrix Q.
   *
   * Q is reconstructed from the stored Householder reflectors:
   *
   *      Q = H₀ H₁ ... Hₖ₋₁
   *
   * Modes:
   *      - ComputeMode::Thin  → returns m × rank matrix
   *      - ComputeMode::Full  → returns m × m matrix
   *
   * @param mode Specifies thin or full decomposition.
   * @return Explicit orthogonal matrix Q.
   *
   * @note Construction cost: O(mn²).
   * @note Internally applies reflectors to the identity matrix.
   */

  Matrix<T> Q(ComputeMode mode = ComputeMode::Thin) const {
    std::size_t target_cols = (mode == ComputeMode::Full) ? m_ : rank_;
    Matrix<T> Qmat(m_, target_cols);

    T *RESTRICT Qp = Qmat.raw();

    for (std::size_t i = 0; i < m_; ++i)
      for (std::size_t j = 0; j < target_cols; ++j)
        Qp[i * target_cols + j] = (i == j ? T(1) : T(0));

    applyQ(Qmat);
    return Qmat;
  }

  // --- Solvers ---

  /**
   * @brief Solves the linear system Ax = b for square full-rank matrices.
   *
   * Uses QR factorization:
   *
   *      A = QR
   *      Qᵀ b = y
   *      Rx = y
   *
   * @param b Right-hand side vector.
   * @return Solution vector x.
   *
   * @throws std::logic_error
   *         If the matrix is not square.
   *
   * @throws std::runtime_error
   *         If the matrix is rank-deficient.
   *
   * @note Time complexity: O(n²)
   */

  Vector<T> solve(const Vector<T> &b) const {
    if (m_ != n_)
      throw std::logic_error("Matrix not square.");
    if (rank_ < n_)
      throw std::runtime_error("Matrix is rank-deficient.");
    return solveLeastSquares(b);
  }

  /**
   * @brief Solves the least squares problem:
   *
   *      min ||Ax − b||₂
   *
   * using QR factorization.
   *
   * Algorithm:
   *      1. y = Qᵀ b
   *      2. Solve Rx = y (back substitution on leading rank block)
   *
   * Supports rectangular (m ≥ n) matrices.
   *
   * @param b Right-hand side vector (size m).
   * @return Least-squares solution vector x.
   *
   * @throws std::invalid_argument
   *         If dimension mismatch occurs.
   *
   * @note Time complexity: O(mn + n²)
   * @note Preferred over normal equations for numerical stability.
   */

  Vector<T> solveLeastSquares(const Vector<T> &b) const {
    if (b.size() != m_)
      throw std::invalid_argument("Dimension mismatch");

    Vector<T> y = b;
    applyQT(y);

    Vector<T> x(n_);
    const T *RESTRICT QRp = QR_.raw();

    for (std::size_t i = rank_; i-- > 0;) {
      T sum = y[i];
      for (std::size_t j = i + 1; j < rank_; ++j)
        sum -= QRp[i * n_ + j] * x[j];
      x[i] = sum / QRp[i * n_ + i];
    }
    return x;
  }

private:
  /**
   * @brief Performs in-place Householder QR factorization.
   *
   * Overwrites QR_ with:
   *
   *      - Upper triangle → R
   *      - Strict lower triangle → Householder vectors
   *
   * Stores scalar reflector coefficients in betas_.
   *
   * Each step constructs a reflector:
   *
   *      H = I − β v vᵀ
   *
   * such that entries below the diagonal are annihilated.
   *
   * Rank is updated based on column norm thresholding.
   *
   * @note Time complexity: O(2mn² − 2n³/3)
   * @note No column pivoting is performed.
   */

  void factorize() {
    const std::size_t K = std::min(m_, n_);
    T *RESTRICT QRp = QR_.raw();

    for (std::size_t k = 0; k < K; ++k) {
      T sigma = 0;
      for (std::size_t i = k; i < m_; ++i) {
        T v = QRp[i * n_ + k];
        sigma += v * v;
      }

      T normx = std::sqrt(sigma);
      if (normx <= tol_) {
        betas_[k] = 0;
        continue;
      }

      rank_++;
      T x0 = QRp[k * n_ + k];
      T alpha = (x0 >= 0) ? -normx : normx;
      T v0 = x0 - alpha;

      QRp[k * n_ + k] = alpha;
      for (std::size_t i = k + 1; i < m_; ++i)
        QRp[i * n_ + k] /= v0;

      betas_[k] = -v0 / alpha;

      for (std::size_t j = k + 1; j < n_; ++j) {
        T dot = QRp[k * n_ + j];
        for (std::size_t i = k + 1; i < m_; ++i)
          dot += QRp[i * n_ + k] * QRp[i * n_ + j];

        dot *= betas_[k];
        QRp[k * n_ + j] -= dot;

        for (std::size_t i = k + 1; i < m_; ++i)
          QRp[i * n_ + j] -= QRp[i * n_ + k] * dot;
      }
    }
  }

  /**
   * @brief Applies Qᵀ to a vector in-place.
   *
   * Computes:
   *
   *      vec ← Qᵀ vec
   *
   * without explicitly forming Q.
   *
   * Reflectors are applied in forward order.
   *
   * @param vec Vector to transform.
   *
   * @note Time complexity: O(mn)
   * @note Used internally for least squares solving.
   */

  void applyQT(Vector<T> &vec) const {
    const std::size_t K = std::min(m_, n_);
    const T *RESTRICT QRp = QR_.raw();

    for (std::size_t k = 0; k < K; ++k) {
      if (betas_[k] == T(0))
        continue;

      T dot = vec[k];
      for (std::size_t i = k + 1; i < m_; ++i)
        dot += QRp[i * n_ + k] * vec[i];

      dot *= betas_[k];
      vec[k] -= dot;

      for (std::size_t i = k + 1; i < m_; ++i)
        vec[i] -= QRp[i * n_ + k] * dot;
    }
  }

  /**
   * @brief Applies Q to a matrix in-place.
   *
   * Computes:
   *
   *      mat ← Q mat
   *
   * without explicitly forming Q.
   *
   * Reflectors are applied in reverse order.
   *
   * @param mat Matrix to transform.
   *
   * @note Time complexity: O(mn·p) where p = number of columns of mat.
   * @note Used internally to construct explicit Q.
   */

  void applyQ(Matrix<T> &mat) const {
    const std::size_t K = std::min(m_, n_);
    const T *RESTRICT QRp = QR_.raw();
    T *RESTRICT Mp = mat.raw();
    const std::size_t ldM = mat.ncols();

    for (int k = static_cast<int>(K) - 1; k >= 0; --k) {
      if (betas_[k] == T(0))
        continue;

      for (std::size_t j = 0; j < ldM; ++j) {
        T dot = Mp[k * ldM + j];
        for (std::size_t i = k + 1; i < m_; ++i)
          dot += QRp[i * n_ + k] * Mp[i * ldM + j];

        dot *= betas_[k];
        Mp[k * ldM + j] -= dot;

        for (std::size_t i = k + 1; i < m_; ++i)
          Mp[i * ldM + j] -= QRp[i * n_ + k] * dot;
      }
    }
  }
};

} // namespace Linea::Decompositions

#endif