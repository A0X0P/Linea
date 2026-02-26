#ifndef LINEA_SVD_HPP
#define LINEA_SVD_HPP

#include "../Core/Concepts.hpp"
#include "../Core/PlatformMacros.hpp"
#include "../Core/Types.hpp"
#include "../Matrix/Matrix.hpp"
#include "../Vector/Vector.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace Linea::Decompositions {

template <RealType Tp> class SVD {
private:
  std::size_t m, n, k;
  Tp tol;
  std::size_t maxIter;
  ComputeMode mode;
  Matrix<Tp> U_;
  Matrix<Tp> V_;
  Vector<Tp> d;

public:
  SVD(const Matrix<Tp> &A, ComputeMode mode = ComputeMode::Thin,
      Tp eps = Tp(1e-12), std::size_t maxIters = 1000)
      : m(A.nrows()), n(A.ncols()), k(std::min(m, n)), tol(eps),
        maxIter(maxIters), mode(mode),
        U_(mode == ComputeMode::Full ? Matrix<Tp>(m, m, Tp(0))
                                     : Matrix<Tp>(m, k, Tp(0))),
        V_(mode == ComputeMode::Full ? Matrix<Tp>(n, n, Tp(0))
                                     : Matrix<Tp>(n, k, Tp(0))),
        d(k, Tp(0)) {
    compute_from_normal_equations(A);
  }

  const Vector<Tp> &singularValues() const noexcept { return d; }
  const Matrix<Tp> &U() const noexcept { return U_; }
  const Matrix<Tp> &V() const noexcept { return V_; }

  std::size_t rank(Tp eps = Tp(1e-12)) const {
    Tp smax = *std::max_element(d.raw(), d.raw() + k);
    std::size_t r = 0;
    const Tp *RESTRICT d_raw = d.raw();
    for (std::size_t i = 0; i < k; ++i)
      if (d_raw[i] > eps * smax)
        ++r;
    return r;
  }

  Tp condition_number(Tp eps = Tp(1e-12)) const {
    Tp smax = Tp(0);
    Tp smin = std::numeric_limits<Tp>::max();
    const Tp *RESTRICT d_raw = d.raw();
    for (std::size_t i = 0; i < k; ++i) {
      if (d_raw[i] > eps) {
        smax = std::max(smax, d_raw[i]);
        smin = std::min(smin, d_raw[i]);
      }
    }
    return (smin == Tp(0)) ? std::numeric_limits<Tp>::infinity() : smax / smin;
  }

  Matrix<Tp> pseudoinverse(Tp eps = Tp(1e-12)) const {
    Matrix<Tp> Splus(V_.ncols(), U_.ncols(), Tp(0));
    Tp smax = *std::max_element(d.raw(), d.raw() + k);
    Tp *RESTRICT s_raw = Splus.raw();
    const Tp *RESTRICT d_raw = d.raw();

    const std::size_t s_cols = Splus.ncols();
    for (std::size_t i = 0; i < k; ++i)
      if (d_raw[i] > eps * smax)
        s_raw[i * s_cols + i] = Tp(1) / d_raw[i];

    return V_ * Splus * U_.transpose();
  }

private:
  static Tp dot_col(const Matrix<Tp> &M, std::size_t c1, std::size_t c2,
                    std::size_t from = 0) {
    Tp sum = 0;
    for (std::size_t r = from; r < M.nrows(); ++r)
      sum += M(r, c1) * M(r, c2);
    return sum;
  }

  static Tp norm_col(const Matrix<Tp> &M, std::size_t col, std::size_t from = 0) {
    return std::sqrt(std::max(dot_col(M, col, col, from), Tp(0)));
  }

  static void normalize_col(Matrix<Tp> &M, std::size_t col, std::size_t from = 0) {
    Tp nrm = norm_col(M, col, from);
    if (nrm == Tp(0))
      return;
    for (std::size_t r = from; r < M.nrows(); ++r)
      M(r, col) /= nrm;
  }

  static void orthogonalize_against_previous(Matrix<Tp> &M, std::size_t col,
                                             std::size_t prevCount,
                                             std::size_t from = 0) {
    for (std::size_t p = 0; p < prevCount; ++p) {
      Tp proj = dot_col(M, p, col, from);
      for (std::size_t r = from; r < M.nrows(); ++r)
        M(r, col) -= proj * M(r, p);
    }
  }

  void jacobi_eigen_symmetric(Matrix<Tp> &S, Matrix<Tp> &Q) const {
    const std::size_t nDim = S.nrows();
    Q = Matrix<Tp>::identity(nDim);

    for (std::size_t iter = 0; iter < maxIter; ++iter) {
      std::size_t p = 0, q = 1;
      Tp maxOff = Tp(0);

      for (std::size_t i = 0; i < nDim; ++i) {
        for (std::size_t j = i + 1; j < nDim; ++j) {
          Tp v = std::abs(S(i, j));
          if (v > maxOff) {
            maxOff = v;
            p = i;
            q = j;
          }
        }
      }

      if (maxOff <= tol)
        return;

      Tp app = S(p, p);
      Tp aqq = S(q, q);
      Tp apq = S(p, q);

      Tp tau = (aqq - app) / (Tp(2) * apq);
      Tp t = (tau >= 0) ? Tp(1) / (tau + std::sqrt(Tp(1) + tau * tau))
                        : Tp(-1) / (-tau + std::sqrt(Tp(1) + tau * tau));
      Tp c = Tp(1) / std::sqrt(Tp(1) + t * t);
      Tp s = t * c;

      for (std::size_t r = 0; r < nDim; ++r) {
        if (r == p || r == q)
          continue;
        Tp srp = S(r, p);
        Tp srq = S(r, q);
        S(r, p) = c * srp - s * srq;
        S(p, r) = S(r, p);
        S(r, q) = s * srp + c * srq;
        S(q, r) = S(r, q);
      }

      Tp new_pp = c * c * app - Tp(2) * s * c * apq + s * s * aqq;
      Tp new_qq = s * s * app + Tp(2) * s * c * apq + c * c * aqq;
      S(p, p) = new_pp;
      S(q, q) = new_qq;
      S(p, q) = Tp(0);
      S(q, p) = Tp(0);

      for (std::size_t r = 0; r < nDim; ++r) {
        Tp qrp = Q(r, p);
        Tp qrq = Q(r, q);
        Q(r, p) = c * qrp - s * qrq;
        Q(r, q) = s * qrp + c * qrq;
      }
    }

    throw std::runtime_error("SVD failed to converge");
  }

  void complete_orthonormal_basis(Matrix<Tp> &M, std::size_t existingCols) const {
    for (std::size_t col = existingCols; col < M.ncols(); ++col) {
      for (std::size_t r = 0; r < M.nrows(); ++r)
        M(r, col) = (r == col ? Tp(1) : Tp(0));

      orthogonalize_against_previous(M, col, col);
      Tp nrm = norm_col(M, col);
      if (nrm <= tol) {
        for (std::size_t r = 0; r < M.nrows(); ++r)
          M(r, col) = (r == (col + 1) % M.nrows() ? Tp(1) : Tp(0));
        orthogonalize_against_previous(M, col, col);
      }
      normalize_col(M, col);
    }
  }

  void compute_from_normal_equations(const Matrix<Tp> &A) {
    Matrix<Tp> AtA = A.transpose() * A;
    Matrix<Tp> Vfull(n, n, Tp(0));
    jacobi_eigen_symmetric(AtA, Vfull);

    struct Pair {
      Tp sigma;
      std::size_t idx;
    };

    std::vector<Pair> ordering(n);
    for (std::size_t i = 0; i < n; ++i) {
      Tp eig = std::max(AtA(i, i), Tp(0));
      ordering[i] = Pair{std::sqrt(eig), i};
    }

    std::sort(ordering.begin(), ordering.end(),
              [](const Pair &a, const Pair &b) { return a.sigma > b.sigma; });

    Matrix<Tp> Vs(n, k, Tp(0));
    for (std::size_t j = 0; j < k; ++j) {
      d[j] = ordering[j].sigma;
      const std::size_t src = ordering[j].idx;
      for (std::size_t r = 0; r < n; ++r)
        Vs(r, j) = Vfull(r, src);
      normalize_col(Vs, j);
    }

    Matrix<Tp> Us(m, k, Tp(0));
    for (std::size_t j = 0; j < k; ++j) {
      if (d[j] > tol) {
        for (std::size_t r = 0; r < m; ++r) {
          Tp sum = 0;
          for (std::size_t c = 0; c < n; ++c)
            sum += A(r, c) * Vs(c, j);
          Us(r, j) = sum / d[j];
        }
      } else {
        for (std::size_t r = 0; r < m; ++r)
          Us(r, j) = (r == j ? Tp(1) : Tp(0));
      }

      orthogonalize_against_previous(Us, j, j);
      Tp nrm = norm_col(Us, j);
      if (nrm <= tol) {
        for (std::size_t r = 0; r < m; ++r)
          Us(r, j) = (r == (j + 1) % m ? Tp(1) : Tp(0));
        orthogonalize_against_previous(Us, j, j);
      }
      normalize_col(Us, j);
    }

    if (mode == ComputeMode::Thin) {
      U_ = Us;
      V_ = Vs;
      return;
    }

    U_ = Matrix<Tp>(m, m, Tp(0));
    V_ = Matrix<Tp>(n, n, Tp(0));

    for (std::size_t c = 0; c < k; ++c) {
      for (std::size_t r = 0; r < m; ++r)
        U_(r, c) = Us(r, c);
      for (std::size_t r = 0; r < n; ++r)
        V_(r, c) = Vs(r, c);
    }

    complete_orthonormal_basis(U_, k);
    complete_orthonormal_basis(V_, k);
  }
};

} // namespace Linea::Decompositions

#endif
