// created by : A.N. Prosper
// date : January 24th 2026
// time : 20:10

#ifndef LINEA_HOUSEHOLDER_QR_HPP
#define LINEA_HOUSEHOLDER_QR_HPP

#include "../Core/Concepts.hpp"
#include "../Matrix/Matrix.hpp"
#include <cmath>
#include <vector>

namespace Linea::Decompositions {

template <RealType T> struct QRStorage {
  Matrix<T> QR;           // Upper triangle = R, lower = Householder tails
  std::vector<T> v_heads; // v_k
  std::vector<T> betas;   // 2 / ||v||^2
};

// Compact Householder QR factorization
// Stores reflectors in-place

template <RealType T> QRStorage<T> factorizeQR(const Matrix<T> &A) {
  std::size_t m = A.nrows();
  std::size_t n = A.ncols();

  QRStorage<T> res{A, std::vector<T>(n), std::vector<T>(n)};

  for (std::size_t k = 0; k < n && k < m - 1; ++k) {
    T sigma = 0;
    for (std::size_t i = k; i < m; ++i)
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

    for (std::size_t i = k + 1; i < m; ++i)
      res.QR(i, k) /= vk;

    res.QR(k, k) = (x0 >= 0) ? -normx : normx;

    for (std::size_t j = k + 1; j < n; ++j) {
      T dot = res.QR(k, j);
      for (std::size_t i = k + 1; i < m; ++i)
        dot += res.QR(i, k) * res.QR(i, j);

      dot *= res.betas[k];

      res.QR(k, j) -= dot;
      for (std::size_t i = k + 1; i < m; ++i)
        res.QR(i, j) -= res.QR(i, k) * dot;
    }
  }

  return res;
}

// Apply Q^T to vector b in-place

template <RealType T> void applyQT(const QRStorage<T> &qr, std::vector<T> &b) {
  std::size_t m = qr.QR.nrows();
  std::size_t n = qr.QR.ncols();

  for (std::size_t k = 0; k < n && k < m - 1; ++k) {
    T dot = b[k];
    for (std::size_t i = k + 1; i < m; ++i)
      dot += qr.QR(i, k) * b[i];

    dot *= qr.betas[k];

    b[k] -= dot;
    for (std::size_t i = k + 1; i < m; ++i)
      b[i] -= qr.QR(i, k) * dot;
  }
}

// Explicitly build Q (full orthogonal matrix)
template <RealType T> Matrix<T> buildQ(const QRStorage<T> &qr) {
  std::size_t m = qr.QR.nrows();
  std::size_t n = qr.QR.ncols();

  Matrix<T> Q = Matrix<T>::Identity(m);

  for (std::size_t k = n - 1; k >= 0; --k) {
    if (qr.betas[k] == T(0))
      continue;

    for (std::size_t j = 0; j < m; ++j) {
      T dot = Q(k, j);
      for (std::size_t i = k + 1; i < m; ++i)
        dot += qr.QR(i, k) * Q(i, j);

      dot *= qr.betas[k];

      Q(k, j) -= dot;
      for (std::size_t i = k + 1; i < m; ++i)
        Q(i, j) -= qr.QR(i, k) * dot;
    }
  }

  return Q;
}

} // namespace Linea::Decompositions

#endif
