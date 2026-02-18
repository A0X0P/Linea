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
      Tp *RESTRICT u_raw = U_.raw();
      Tp *RESTRICT v_raw = V_.raw();
      for (std::size_t i = 0; i < k; ++i) {
        u_raw[i * m + i] = Tp(1);
        v_raw[i * n + i] = Tp(1);
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
    Matrix<Tp> Splus(n, m, Tp(0));
    Tp smax = *std::max_element(d.raw(), d.raw() + k);
    Tp *RESTRICT s_raw = Splus.raw();
    const Tp *RESTRICT d_raw = d.raw();

    for (std::size_t i = 0; i < k; ++i)
      if (d_raw[i] > eps * smax)
        s_raw[i * n + i] = Tp(1) / d_raw[i];

    return V_ * Splus * U_.transpose();
  }

private:
  //  Householder Bidiagonalization (

  void bidiagonalize(const Matrix<Tp> &A) {
    Matrix<Tp> B = A;
    Tp *RESTRICT B_raw = B.raw();

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

  // Householder Left

  void householder_left(Matrix<Tp> &B, std::size_t i) {
    Tp *RESTRICT B_raw = B.raw();
    Tp *RESTRICT U_raw = U_.raw();

    // ---- compute sigma (column i, rows i+1:m)
    Tp sigma = 0;
    for (std::size_t r = i + 1; r < m; ++r) {
      Tp val = B_raw[r * n + i];
      sigma += val * val;
    }

    Tp alpha = B_raw[i * n + i];
    if (sigma == Tp(0) && alpha >= Tp(0))
      return;

    Tp norm = std::sqrt(alpha * alpha + sigma);
    Tp beta = (alpha <= 0) ? norm : -norm;
    Tp tau = (beta - alpha) / beta;

    // ---- build v
    Vector<Tp> v(m, Tp(0));
    Tp *RESTRICT v_raw = v.raw();

    v_raw[i] = 1;
    Tp denom = alpha - beta;

    for (std::size_t r = i + 1; r < m; ++r)
      v_raw[r] = B_raw[r * n + i] / denom;

    B_raw[i * n + i] = beta;
    for (std::size_t r = i + 1; r < m; ++r)
      B_raw[r * n + i] = 0;

    // ---- Apply reflector to B (columns i+1:n)
    for (std::size_t j = i + 1; j < n; ++j) {
      Tp w = 0;

      // dot(v, column j)
      for (std::size_t r = i; r < m; ++r)
        w += v_raw[r] * B_raw[r * n + j];

      w *= tau;

      // rank-1 update
      for (std::size_t r = i; r < m; ++r)
        B_raw[r * n + j] -= v_raw[r] * w;
    }

    // ---- Accumulate into U
    std::size_t Ucols = (mode == ComputeMode::Full) ? m : k;

    for (std::size_t j = 0; j < Ucols; ++j) {
      Tp w = 0;

      for (std::size_t r = i; r < m; ++r)
        w += v_raw[r] * U_raw[r * m + j];

      w *= tau;

      for (std::size_t r = i; r < m; ++r)
        U_raw[r * m + j] -= v_raw[r] * w;
    }
  }

  // Householder Right

  void householder_right(Matrix<Tp> &B, std::size_t i) {
    Tp *RESTRICT B_raw = B.raw();
    Tp *RESTRICT V_raw = V_.raw();

    // ---- compute sigma (row i, cols i+2:n)
    Tp sigma = 0;
    for (std::size_t c = i + 2; c < n; ++c) {
      Tp val = B_raw[i * n + c];
      sigma += val * val;
    }

    Tp alpha = B_raw[i * n + i + 1];
    if (sigma == Tp(0) && alpha >= Tp(0))
      return;

    Tp norm = std::sqrt(alpha * alpha + sigma);
    Tp beta = (alpha <= 0) ? norm : -norm;
    Tp tau = (beta - alpha) / beta;

    Vector<Tp> v(n, Tp(0));
    Tp *RESTRICT v_raw = v.raw();

    v_raw[i + 1] = 1;
    Tp denom = alpha - beta;

    for (std::size_t c = i + 2; c < n; ++c)
      v_raw[c] = B_raw[i * n + c] / denom;

    B_raw[i * n + i + 1] = beta;
    for (std::size_t c = i + 2; c < n; ++c)
      B_raw[i * n + c] = 0;

    // ---- Apply reflector to B (rows i+1:m)
    for (std::size_t r = i + 1; r < m; ++r) {
      Tp w = 0;

      for (std::size_t c = i + 1; c < n; ++c)
        w += B_raw[r * n + c] * v_raw[c];

      w *= tau;

      for (std::size_t c = i + 1; c < n; ++c)
        B_raw[r * n + c] -= w * v_raw[c];
    }

    // ---- Accumulate into V
    std::size_t Vcols = (mode == ComputeMode::Full) ? n : k;

    for (std::size_t r = 0; r < Vcols; ++r) {
      Tp w = 0;

      for (std::size_t c = i + 1; c < n; ++c)
        w += V_raw[r * n + c] * v_raw[c];

      w *= tau;

      for (std::size_t c = i + 1; c < n; ++c)
        V_raw[r * n + c] -= w * v_raw[c];
    }
  }

  // Golub–Kahan Implicit QR Iteration

  void iterate() {
    for (std::size_t iter = 0; iter < maxIter; ++iter) {
      bool converged = true;
      Tp *RESTRICT e_raw = e.raw();
      Tp *RESTRICT d_raw = d.raw();

      for (std::size_t i = 0; i + 1 < k; ++i) {
        if (std::abs(e_raw[i]) <=
            tol * (std::abs(d_raw[i]) + std::abs(d_raw[i + 1])))
          e_raw[i] = Tp(0);
        if (e_raw[i] != Tp(0))
          converged = false;
      }

      if (converged)
        return;

      std::size_t l = 0;
      while (l < k - 1 && e_raw[l] == Tp(0))
        ++l;
      std::size_t end = l + 1;
      while (end < k - 1 && e_raw[end] != Tp(0))
        ++end;
      ++end;

      Tp mu = wilkinson_shift(end);

      Tp x = d_raw[l] * d_raw[l] - mu;
      Tp z = d_raw[l] * e_raw[l];

      for (std::size_t i = l; i < end - 1; ++i) {
        Tp cR, sR;
        givens(x, z, cR, sR);

        Tp f = cR * d_raw[i] + sR * e_raw[i];
        Tp g = -sR * d_raw[i] + cR * e_raw[i];
        Tp h = sR * d_raw[i + 1];

        d_raw[i] = f;
        e_raw[i] = g;
        d_raw[i + 1] = cR * d_raw[i + 1];

        if (i + 1 < end - 1) {
          z = -sR * e_raw[i + 1];
          e_raw[i + 1] = cR * e_raw[i + 1];
        }

        apply_right_rotation(i, cR, sR);
        Tp cL, sL;
        givens(d_raw[i], h, cL, sL);
        d_raw[i] = cL * d_raw[i] + sL * h;
        d_raw[i + 1] = -sL * d_raw[i + 1];
        e_raw[i] = 0;

        if (i + 1 < end - 1) {
          x = cL * e_raw[i + 1];
          e_raw[i + 1] = sL * e_raw[i + 1];
        }

        apply_left_rotation(i, cL, sL);
      }
    }
    throw std::runtime_error("SVD failed to converge");
  }

  Tp wilkinson_shift(std::size_t end) const {
    const Tp *RESTRICT d_raw = d.raw();
    const Tp *RESTRICT e_raw = e.raw();
    Tp dk = d_raw[end - 1];
    Tp dkm1 = d_raw[end - 2];
    Tp ekm1 = e_raw[end - 2];

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

  // QR Iteration Rotations

  void apply_left_rotation(std::size_t i, Tp c, Tp s) {
    Tp *RESTRICT U_raw = U_.raw();
    std::size_t cols = (mode == ComputeMode::Full) ? m : k;

    for (std::size_t col = 0; col < cols; ++col) {
      Tp x = U_raw[i * m + col];
      Tp y = U_raw[(i + 1) * m + col];

      U_raw[i * m + col] = c * x + s * y;
      U_raw[(i + 1) * m + col] = -s * x + c * y;
    }
  }

  void apply_right_rotation(std::size_t i, Tp c, Tp s) {
    Tp *RESTRICT V_raw = V_.raw();
    std::size_t cols = (mode == ComputeMode::Full) ? n : k;

    for (std::size_t col = 0; col < cols; ++col) {
      Tp x = V_raw[i * n + col];
      Tp y = V_raw[(i + 1) * n + col];

      V_raw[i * n + col] = c * x + s * y;
      V_raw[(i + 1) * n + col] = -s * x + c * y;
    }
  }

  void enforce_positive() {
    Tp *RESTRICT d_raw = d.raw();
    Tp *RESTRICT V_raw = V_.raw();
    for (std::size_t i = 0; i < k; ++i) {
      if (d_raw[i] < 0) {
        d_raw[i] = -d_raw[i];
        for (std::size_t r = 0; r < n; ++r)
          V_raw[r * n + i] = -V_raw[r * n + i];
      }
    }
  }

  void sort_descending() {
    Tp *RESTRICT d_raw = d.raw();
    Tp *RESTRICT U_raw = U_.raw();
    Tp *RESTRICT V_raw = V_.raw();
    for (std::size_t i = 0; i < k; ++i) {
      std::size_t maxIdx = i;
      for (std::size_t j = i + 1; j < k; ++j)
        if (d_raw[j] > d_raw[maxIdx])
          maxIdx = j;

      if (maxIdx != i) {
        std::swap(d_raw[i], d_raw[maxIdx]);
        U_.swap_column(i, maxIdx);
        V_.swap_column(i, maxIdx);
      }
    }
  }
};

} // namespace Linea::Decompositions

#endif
