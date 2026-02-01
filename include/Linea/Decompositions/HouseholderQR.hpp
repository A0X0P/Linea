// created by : A.N. Prosper
// date : January 24th 2026
// time : 20:10

#ifndef LINEA_HOUSEHOLDER_QR_HPP
#define LINEA_HOUSEHOLDER_QR_HPP

#include "../Core/Concepts.hpp"
#include "../Matrix/Matrix.hpp"
#include "../Vector/Vector.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace Linea::Decompositions {

enum class QRMode { Full, Thin };

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

  std::size_t rank() const noexcept { return rank_; }

  // --- Accessors ---

  Matrix<T> R(QRMode mode = QRMode::Thin) const {
    std::size_t target_rows = (mode == QRMode::Full) ? m_ : rank_;

    Matrix<T> Rmat(target_rows, n_);
    for (std::size_t i = 0; i < target_rows; ++i) {
      for (std::size_t j = i; j < n_; ++j) {
        if (i < m_) {
          Rmat(i, j) = QR_(i, j);
        }
      }
    }
    return Rmat;
  }

  Matrix<T> Q(QRMode mode = QRMode::Thin) const {
    std::size_t target_cols = (mode == QRMode::Full) ? m_ : rank_;

    Matrix<T> Qmat(m_, target_cols);
    for (std::size_t i = 0; i < m_; ++i) {
      for (std::size_t j = 0; j < target_cols; ++j) {
        Qmat(i, j) = (i == j ? T(1) : T(0));
      }
    }
    applyQ(Qmat);
    return Qmat;
  }

  // --- Solvers ---

  Vector<T> solve(const Vector<T> &b) const {
    if (m_ != n_)
      throw std::logic_error("Matrix not square.");
    if (rank_ < n_)
      throw std::runtime_error("Matrix is rank-deficient.");
    return solveLeastSquares(b);
  }

  Vector<T> solveLeastSquares(const Vector<T> &b) const {
    if (b.size() != m_)
      throw std::invalid_argument("Dimension mismatch");
    Vector<T> y = b;
    applyQT(y);
    Vector<T> x(n_);
    for (std::size_t i = rank_; i-- > 0;) {
      T sum = y[i];
      for (std::size_t j = i + 1; j < rank_; ++j)
        sum -= QR_(i, j) * x[j];
      x[i] = sum / QR_(i, i);
    }
    return x;
  }

private:
  void factorize() {
    std::size_t K = std::min(m_, n_);
    for (std::size_t k = 0; k < K; ++k) {
      T sigma = 0;
      for (std::size_t i = k; i < m_; ++i)
        sigma += QR_(i, k) * QR_(i, k);
      T normx = std::sqrt(sigma);

      if (normx <= tol_) {
        betas_[k] = 0;
        continue;
      }

      rank_++;
      T x0 = QR_(k, k);
      T alpha = (x0 >= 0) ? -normx : normx;
      T v0 = x0 - alpha;

      QR_(k, k) = alpha;
      for (std::size_t i = k + 1; i < m_; ++i)
        QR_(i, k) /= v0;
      betas_[k] = -v0 / alpha;

      for (std::size_t j = k + 1; j < n_; ++j) {
        T dot = QR_(k, j);
        for (std::size_t i = k + 1; i < m_; ++i)
          dot += QR_(i, k) * QR_(i, j);
        dot *= betas_[k];
        QR_(k, j) -= dot;
        for (std::size_t i = k + 1; i < m_; ++i)
          QR_(i, j) -= QR_(i, k) * dot;
      }
    }
  }

  void applyQT(Vector<T> &vec) const {
    std::size_t K = std::min(m_, n_);
    for (std::size_t k = 0; k < K; ++k) {
      if (betas_[k] == T(0))
        continue;
      T dot = vec[k];
      for (std::size_t i = k + 1; i < m_; ++i)
        dot += QR_(i, k) * vec[i];
      dot *= betas_[k];
      vec[k] -= dot;
      for (std::size_t i = k + 1; i < m_; ++i)
        vec[i] -= QR_(i, k) * dot;
    }
  }

  void applyQ(Matrix<T> &mat) const {
    std::size_t K = std::min(m_, n_);
    for (int k = static_cast<int>(K) - 1; k >= 0; --k) {
      if (betas_[k] == T(0))
        continue;
      for (std::size_t j = 0; j < mat.ncols(); ++j) {
        T dot = mat(k, j);
        for (std::size_t i = k + 1; i < m_; ++i)
          dot += QR_(i, k) * mat(i, j);
        dot *= betas_[k];
        mat(k, j) -= dot;
        for (std::size_t i = k + 1; i < m_; ++i)
          mat(i, j) -= QR_(i, k) * dot;
      }
    }
  }
};

} // namespace Linea::Decompositions

#endif