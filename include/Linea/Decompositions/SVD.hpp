// created by : A.N. Prosper
// date : January 24th 2026
// time : 20:09

#ifndef LINEA_SVD_HPP
#define LINEA_SVD_HPP

#include "../Core/Concepts.hpp"
#include "../Core/PlatformMacros.hpp"
#include "../Core/Types.hpp"
#include "../Matrix/Matrix.hpp"
#include "../Vector/Vector.hpp"
#include <cmath>

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
  Vector<Tp> e;

public:
  SVD(const Matrix<Tp> &A, ComputeMode mode = ComputeMode::Thin,
      Tp eps = Tp(1e-12), std::size_t maxIters = 1000)
      : m(A.nrows()), n(A.ncols()), k(std::min(m, n)), tol(eps),
        maxIter(maxIters), mode(mode), d(k), e(k > 1 ? k - 1 : 0),
        U_(mode == ComputeMode::Full ? Matrix<Tp>::identity(m)
                                     : Matrix<Tp>(m, k, Tp(0))),
        V_(mode == ComputeMode::Full ? Matrix<Tp>::identity(n)
                                     : Matrix<Tp>(n, k, Tp(0))) {
    if (mode == ComputeMode::Thin) {
      for (std::size_t i = 0; i < k; ++i) {
        U_(i, i) = Tp(1);
        V_(i, i) = Tp(1);
      }
    }

    bidiagonalize(A);
    iterate();
    enforce_positive();
    sort_descending();
  }

  const Vector<Tp> &singularValues() const noexcept { return d; }
  const Matrix<Tp> &U() const noexcept { return U_; }
  const Matrix<Tp> &V() const noexcept { return V_; }

  std::size_t rank(Tp eps = Tp(1e-12)) const {
    Tp smax = *std::max_element(d.begin(), d.end());
    std::size_t r = 0;
    for (auto s : d)
      if (s > eps * smax)
        ++r;
    return r;
  }

  Tp condition_number(Tp eps = Tp(1e-12)) const {
    Tp smax = Tp(0);
    Tp smin = std::numeric_limits<Tp>::max();
    for (auto s : d) {
      if (s > eps) {
        smax = std::max(smax, s);
        smin = std::min(smin, s);
      }
    }
    return (smin == Tp(0)) ? std::numeric_limits<Tp>::infinity() : smax / smin;
  }

  Matrix<Tp> pseudoinverse(Tp eps = Tp(1e-12)) const {
    Matrix<Tp> Splus(n, m, Tp(0));
    Tp smax = *std::max_element(d.begin(), d.end());

    for (std::size_t i = 0; i < k; ++i)
      if (d[i] > eps * smax)
        Splus(i, i) = Tp(1) / d[i];

    return V_ * Splus * U_.transpose();
  }

private:
  // ------------------------------------------------
  // Stable Householder Bidiagonalization
  // ------------------------------------------------

  void bidiagonalize(const Matrix<Tp> &A) {
    Matrix<Tp> B = A;

    for (std::size_t i = 0; i < k; ++i) {
      householder_left(B, i);
      d[i] = B(i, i);

      if (i + 1 < n) {
        householder_right(B, i);
        if (i + 1 < k)
          e[i] = B(i, i + 1);
      }
    }
  }

  void householder_left(Matrix<Tp> &B, std::size_t i) {
    Tp sigma = 0;
    for (std::size_t r = i + 1; r < m; ++r)
      sigma += B(r, i) * B(r, i);

    Tp alpha = B(i, i);
    if (sigma == Tp(0) && alpha >= Tp(0))
      return;

    Tp norm = std::sqrt(alpha * alpha + sigma);
    Tp beta = (alpha <= 0) ? norm : -norm;
    Tp tau = (beta - alpha) / beta;

    Vector<Tp> v(m, Tp(0));
    v[i] = 1;
    for (std::size_t r = i + 1; r < m; ++r)
      v[r] = B(r, i) / (alpha - beta);

    B(i, i) = beta;
    for (std::size_t r = i + 1; r < m; ++r)
      B(r, i) = 0;

    for (std::size_t j = i + 1; j < n; ++j) {
      Tp w = 0;
      for (std::size_t r = i; r < m; ++r)
        w += v[r] * B(r, j);
      w *= tau;
      for (std::size_t r = i; r < m; ++r)
        B(r, j) -= v[r] * w;
    }

    // accumulate U
    for (std::size_t j = 0; j < m; ++j) {
      Tp w = 0;
      for (std::size_t r = i; r < m; ++r)
        w += v[r] * U_(r, j);
      w *= tau;
      for (std::size_t r = i; r < m; ++r)
        U_(r, j) -= v[r] * w;
    }
  }

  void householder_right(Matrix<Tp> &B, std::size_t i) {
    Tp sigma = 0;
    for (std::size_t c = i + 2; c < n; ++c)
      sigma += B(i, c) * B(i, c);

    Tp alpha = B(i, i + 1);
    if (sigma == Tp(0) && alpha >= Tp(0))
      return;

    Tp norm = std::sqrt(alpha * alpha + sigma);
    Tp beta = (alpha <= 0) ? norm : -norm;
    Tp tau = (beta - alpha) / beta;

    Vector<Tp> v(n, Tp(0));
    v[i + 1] = 1;
    for (std::size_t c = i + 2; c < n; ++c)
      v[c] = B(i, c) / (alpha - beta);

    B(i, i + 1) = beta;
    for (std::size_t c = i + 2; c < n; ++c)
      B(i, c) = 0;

    for (std::size_t r = i + 1; r < m; ++r) {
      Tp w = 0;
      for (std::size_t c = i + 1; c < n; ++c)
        w += B(r, c) * v[c];
      w *= tau;
      for (std::size_t c = i + 1; c < n; ++c)
        B(r, c) -= w * v[c];
    }

    for (std::size_t r = 0; r < n; ++r) {
      Tp w = 0;
      for (std::size_t c = i + 1; c < n; ++c)
        w += V_(r, c) * v[c];
      w *= tau;
      for (std::size_t c = i + 1; c < n; ++c)
        V_(r, c) -= w * v[c];
    }
  }

  // ------------------------------------------------
  // Golub–Kahan Implicit QR Iteration
  // ------------------------------------------------

  void iterate() {
    for (std::size_t iter = 0; iter < maxIter; ++iter) {
      bool converged = true;

      for (std::size_t i = 0; i + 1 < k; ++i) {
        if (std::abs(e[i]) <= tol * (std::abs(d[i]) + std::abs(d[i + 1])))
          e[i] = Tp(0);
        if (e[i] != Tp(0))
          converged = false;
      }

      if (converged)
        return;

      std::size_t l = 0;
      while (l < k - 1 && e[l] == Tp(0))
        ++l;
      std::size_t end = l + 1;
      while (end < k - 1 && e[end] != Tp(0))
        ++end;
      ++end;

      Tp mu = wilkinson_shift(end);

      Tp x = d[l] * d[l] - mu;
      Tp z = d[l] * e[l];

      for (std::size_t i = l; i < end - 1; ++i) {
        Tp cR, sR;
        givens(x, z, cR, sR);

        Tp f = cR * d[i] + sR * e[i];
        Tp g = -sR * d[i] + cR * e[i];
        Tp h = sR * d[i + 1];

        d[i] = f;
        e[i] = g;
        d[i + 1] = cR * d[i + 1];

        if (i + 1 < end - 1) {
          z = -sR * e[i + 1];
          e[i + 1] = cR * e[i + 1];
        }

        apply_right_rotation(i, cR, sR);

        Tp cL, sL;
        givens(d[i], h, cL, sL);

        d[i] = cL * d[i] + sL * h;
        d[i + 1] = -sL * d[i + 1];
        e[i] = 0;

        if (i + 1 < end - 1) {
          x = cL * e[i + 1];
          e[i + 1] = sL * e[i + 1];
        }

        apply_left_rotation(i, cL, sL);
      }
    }

    throw std::runtime_error("SVD failed to converge");
  }

  Tp wilkinson_shift(std::size_t end) const {
    Tp dk = d[end - 1];
    Tp dkm1 = d[end - 2];
    Tp ekm1 = e[end - 2];

    Tp a = dkm1 * dkm1 + ekm1 * ekm1;
    Tp c = dk * dk;
    Tp b = dkm1 * ekm1;

    Tp tr = a + c;
    Tp det = a * c - b * b;

    Tp disc = std::sqrt(std::max(tr * tr - 4 * det, Tp(0)));
    return (tr - disc) / 2;
  }

  void givens(Tp x, Tp y, Tp &c, Tp &s) {
    Tp r = std::hypot(x, y);
    if (r == Tp(0)) {
      c = 1;
      s = 0;
    } else {
      c = x / r;
      s = y / r;
    }
  }

  void apply_right_rotation(std::size_t i, Tp c, Tp s) {
    std::size_t cols = (mode == ComputeMode::Full) ? n : k;
    for (std::size_t r = 0; r < cols; ++r) {
      Tp x = V_(r, i);
      Tp y = V_(r, i + 1);
      V_(r, i) = c * x + s * y;
      V_(r, i + 1) = -s * x + c * y;
    }
  }

  void apply_left_rotation(std::size_t i, Tp c, Tp s) {
    std::size_t cols = (mode == ComputeMode::Full) ? m : k;
    for (std::size_t r = 0; r < cols; ++r) {
      Tp x = U_(r, i);
      Tp y = U_(r, i + 1);
      U_(r, i) = c * x + s * y;
      U_(r, i + 1) = -s * x + c * y;
    }
  }

  void enforce_positive() {
    for (std::size_t i = 0; i < k; ++i) {
      if (d[i] < 0) {
        d[i] = -d[i];
        for (std::size_t r = 0; r < n; ++r)
          V_(r, i) = -V_(r, i);
      }
    }
  }

  void sort_descending() {
    for (std::size_t i = 0; i < k; ++i) {
      std::size_t maxIdx = i;
      for (std::size_t j = i + 1; j < k; ++j)
        if (d[j] > d[maxIdx])
          maxIdx = j;

      if (maxIdx != i) {
        std::swap(d[i], d[maxIdx]);
        U_.swap_column(i, maxIdx);
        V_.swap_column(i, maxIdx);
      }
    }
  }
};

}; // namespace Linea::Decompositions

#endif
