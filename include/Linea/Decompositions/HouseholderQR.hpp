// created by : A.N. Prosper
// date : January 24th 2026
// time : 20:10

#ifndef LINEA_HOUSEHOLDER_QR_HPP
#define LINEA_HOUSEHOLDER_QR_HPP

#include "../Core/Concepts.hpp"
#include "../Matrix/Matrix.hpp"
#include "../Vector/Vector.hpp"
#include <cmath>
#include <stdexcept>

namespace Linea::Decompositions {

template <RealType T> struct QRStorage {
  Matrix<T> QR;      // Upper triangle = R, lower = Householder tails
  Vector<T> v_heads; // v_k
  Vector<T> betas;   // 2 / ||v||^2
};

//  Compact Householder QR factorization
//  Stores reflectors in-place
template <RealType T> QRStorage<T> factorizeQR(const Matrix<T> &A) {
  int m = A.nrows();
  int n = A.ncols();

  QRStorage<T> res{A, Vector<T>(n), Vector<T>(n)};

  for (int k = 0; k < n && k < m - 1; ++k) {
    T sigma = T(0);
    for (int i = k; i < m; ++i)
      sigma += res.QR(i, k) * res.QR(i, k);

    if (sigma == T(0)) {
      res.betas[k] = T(0);
      continue;
    }

    T x0 = res.QR(k, k);
    T normx = std::sqrt(sigma);

    T vk = (x0 >= 0) ? x0 + normx : x0 - normx;
    res.v_heads[k] = vk;

    res.betas[k] = T(2) / (sigma - x0 * x0 + vk * vk);

    for (int i = k + 1; i < m; ++i)
      res.QR(i, k) /= vk;

    res.QR(k, k) = (x0 >= 0) ? -normx : normx;

    for (int j = k + 1; j < n; ++j) {
      T dot = res.QR(k, j);
      for (int i = k + 1; i < m; ++i)
        dot += res.QR(i, k) * res.QR(i, j);

      dot *= res.betas[k];

      res.QR(k, j) -= dot;
      for (int i = k + 1; i < m; ++i)
        res.QR(i, j) -= res.QR(i, k) * dot;
    }
  }

  return res;
}

// Apply Q^T to vector b in-place

template <RealType T> void applyQT(const QRStorage<T> &qr, Vector<T> &b) {
  int m = qr.QR.nrows();
  int n = qr.QR.ncols();

  for (int k = 0; k < n && k < m - 1; ++k) {
    T dot = b[k];
    for (int i = k + 1; i < m; ++i)
      dot += qr.QR(i, k) * b[i];

    dot *= qr.betas[k];

    b[k] -= dot;
    for (int i = k + 1; i < m; ++i)
      b[i] -= qr.QR(i, k) * dot;
  }
}

// Explicitly build Q (full orthogonal matrix)

template <RealType T> Matrix<T> buildQ(const QRStorage<T> &qr) {
  int m = qr.QR.nrows();
  int n = qr.QR.ncols();

  Matrix<T> Q = Matrix<T>::Identity(m);

  for (int k = n - 1; k >= 0; --k) {
    if (qr.betas[k] == T(0))
      continue;

    for (int j = 0; j < m; ++j) {
      T dot = Q(k, j);
      for (int i = k + 1; i < m; ++i)
        dot += qr.QR(i, k) * Q(i, j);

      dot *= qr.betas[k];

      Q(k, j) -= dot;
      for (int i = k + 1; i < m; ++i)
        Q(i, j) -= qr.QR(i, k) * dot;
    }
  }

  return Q;
}

// HouseholderQR class for direct use

template <RealType T> class HouseholderQR {
private:
  Matrix<T> QR;
  Vector<T> betas;
  int m, n;

public:
  HouseholderQR(const Matrix<T> &A) {
    m = A.nrows();
    n = A.ncols();

    auto fact = factorizeQR(A);
    QR = fact.QR;
    betas = fact.betas;
  }

  // Solve Ax = b
  Vector<T> solve(const Vector<T> &b) const {
    if (b.size() != m)
      throw std::invalid_argument("Vector size mismatch");

    Vector<T> y = b;

    // Apply Q^T to b
    for (int k = 0; k < n && k < m - 1; ++k) {
      T dot = y[k];
      for (int i = k + 1; i < m; ++i)
        dot += QR(i, k) * y[i];
      dot *= betas[k];

      y[k] -= dot;
      for (int i = k + 1; i < m; ++i)
        y[i] -= QR(i, k) * dot;
    }

    // Back substitution Rx = y
    Vector<T> x(n, T(0));
    for (int i = n - 1; i >= 0; --i) {
      T sum = T(0);
      for (int j = i + 1; j < n; ++j)
        sum += QR(i, j) * x[j];
      x[i] = (y[i] - sum) / QR(i, i);
    }

    return x;
  }

  // Extract Q
  Matrix<T> Q() const {
    Matrix<T> Qmat = Matrix<T>::Identity(m);

    for (int k = n - 1; k >= 0; --k) {
      if (betas[k] == T(0))
        continue;

      for (int j = 0; j < m; ++j) {
        T dot = Qmat(k, j);
        for (int i = k + 1; i < m; ++i)
          dot += QR(i, k) * Qmat(i, j);
        dot *= betas[k];

        Qmat(k, j) -= dot;
        for (int i = k + 1; i < m; ++i)
          Qmat(i, j) -= QR(i, k) * dot;
      }
    }

    return Qmat;
  }

  // Extract R
  Matrix<T> R() const {
    Matrix<T> Rmat = QR;
    for (int j = 0; j < n; ++j)
      for (int i = j + 1; i < m; ++i)
        Rmat(i, j) = T(0);
    return Rmat;
  }
};

} // namespace Linea::Decompositions

#endif
