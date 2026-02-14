// created by : A.N. Prosper
// date : january 25th 2026
// time : 15:05

#ifndef LINEA_MATRIX_UTILITIES_H
#define LINEA_MATRIX_UTILITIES_H

#include "../Decompositions/LU.hpp"
#include "../Decompositions/SVD.hpp"
#include "Matrix.hpp"
#include <iostream>
#include <limits>

namespace Linea {

// Shape
template <NumericType M> void Matrix<M>::shape() const {
  std::cout << "(" << row << ", " << column << ")" << std::endl;
}

// Size
template <NumericType M> std::size_t Matrix<M>::size() const {
  return row * column;
}

// Empty
template <NumericType M> bool Matrix<M>::empty() const noexcept {
  return this->data.empty();
}

// Symmetric
template <NumericType M> bool Matrix<M>::symmetric() const noexcept {
  if (row != column)
    return false;

  const auto *a = data.data();
  const std::size_t n = row;

  for (std::size_t i = 0; i < n; ++i) {
    const auto *row_i = a + i * n;
    for (std::size_t j = i + 1; j < n; ++j) {
      if (row_i[j] != a[j * n + i]) {
        return false;
      }
    }
  }
  return true;
}

// Square
template <NumericType M> bool Matrix<M>::square() const noexcept {
  return this->row == this->column;
}

// Singular
template <NumericType M> bool Matrix<M>::singular() const noexcept {
  return rank() < row;
}

// Trace
template <NumericType M> M Matrix<M>::trace() const {
  if (row != column) {
    throw std::invalid_argument("Matrix trace requires row == column.");
  }

  const std::size_t n = row;
  const auto *RESTRICT base = this->raw();
  M trace_value = M{0};

  for (std::size_t i = 0; i < n; ++i) {
    trace_value += base[i * n + i];
  }

  return trace_value;
}

// Diagonal
template <NumericType M> Vector<M> Matrix<M>::diagonal(Diagonal type) const {
  if (row != column) {
    throw std::invalid_argument("Matrix diagonal requires row == column.");
  }

  const std::size_t n = row;
  const auto *RESTRICT base = this->raw();

  Vector<M> _diagonal(n);
  auto *RESTRICT diag = _diagonal.raw();

  const auto *p = (type == Diagonal::Major) ? base : base + (n - 1);

  const std::size_t stride = (type == Diagonal::Major) ? (n + 1) : (n - 1);

  for (std::size_t i = 0; i < n; ++i, p += stride) {
    diag[i] = *p;
  }

  return _diagonal;
}

// Transpose
template <NumericType M> Matrix<M> Matrix<M>::transpose() const {
  Matrix<M> result(this->column, this->row);

  const std::size_t m = row;
  const std::size_t n = column;

  auto *RESTRICT out = result.raw();
  const auto *RESTRICT a = this->raw();

  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      out[j * m + i] = a[i * n + j];
    }
  }

  return result;
}

// Rank
template <NumericType M>
std::size_t Matrix<M>::rank() const
  requires RealType<M>
{
  return Linea::Decompositions::LU<M>(*this).rank();
}

// Reshape
template <NumericType M>
Matrix<M> Matrix<M>::reshape(std::size_t nrow, std::size_t ncol) const {
  const std::size_t reshape_size = nrow * ncol;

  if (reshape_size != data.size()) {
    throw std::invalid_argument(
        "No possible reshape exist for this combination.");
  }
  Matrix<M> result(nrow, ncol);

  auto *RESTRICT out = result.raw();
  const auto *RESTRICT in = this->raw();

  for (std::size_t i = 0; i < reshape_size; ++i) {
    out[i] = in[i];
  }
  return result;
}

// Flatten
template <NumericType M> Matrix<M> Matrix<M>::flatten() const {
  Matrix<M> result(1, data.size());
  std::copy(data.begin(), data.end(), result.data.begin());
  return result;
}

// Submatrix
template <NumericType M>
Matrix<M> Matrix<M>::subMatrix(std::size_t row_idx, std::size_t col_idx) const {
  if (row_idx >= row || col_idx >= column) {
    throw std::out_of_range("Matrix index is out of range.");
  }

  Matrix<M> sub_matrix(row - 1, column - 1);

  const std::size_t m = row;
  const std::size_t n = column;

  auto *RESTRICT out = sub_matrix.raw();
  const auto *RESTRICT a = this->raw();

  for (std::size_t i = 0; i < m; ++i) {
    if (i == row_idx)
      continue;

    const M *RESTRICT row_ptr = a + i * column;

    for (std::size_t j = 0; j < n; ++j) {
      if (j == col_idx)
        continue;
      *out++ = row_ptr[j];
    }
  }

  return sub_matrix;
}

// Block
template <NumericType M>
Matrix<M> Matrix<M>::block(std::size_t row, std::size_t col, std::size_t nrows,
                           std::size_t ncols) const {
  if (row + nrows > this->nrows() || col + ncols > this->ncols()) {
    throw std::out_of_range("Matrix::block out of range");
  }

  Matrix<M> result(nrows, ncols);

  auto *RESTRICT out = result.raw();
  const auto *RESTRICT a = this->raw();

  for (std::size_t i = 0; i < nrows; ++i) {
    const auto R = row + i;
    for (std::size_t j = 0; j < ncols; ++j) {
      const auto C = col + j;
      out[i * ncols + j] = a[R * column + C];
    }
  }
  return result;
}

// Minor
template <NumericType M>
M Matrix<M>::minor(std::size_t row_idx, std::size_t col_idx) const {
  if (row != column) {
    throw std::invalid_argument("minor requires a square matrix.");
  }
  return subMatrix(row_idx, col_idx).determinant();
}

// Cofactor
template <NumericType M>
M Matrix<M>::cofactor(std::size_t row_index, std::size_t column_index) const {
  M sign = ((row_index + column_index) % 2 ? -M{1} : M{1});
  return sign * minor(row_index, column_index);
}

// Cofactor matrix
template <NumericType M> Matrix<M> Matrix<M>::cofactor_matrix() const {
  if (row != column) {
    throw std::invalid_argument("Cofactor matrix requires a square matrix.");
  }
  Matrix<M> result(row, column);
  auto *RESTRICT out = result.raw();

  for (std::size_t i = 0; i < row; ++i) {
    for (std::size_t j = 0; j < column; ++j) {
      out[i * column + j] = cofactor(i, j);
    }
  }
  return result;
}

// Adjoint
template <NumericType M> Matrix<M> Matrix<M>::adjoint() const {
  return cofactor_matrix().transpose();
}

// Determinant
template <NumericType M>
M Matrix<M>::determinant() const
  requires RealType<M>
{

  if (row != column)
    throw std::invalid_argument("Determinant requires square matrix");

  Linea::Decompositions::LU<M> lu(*this);
  Matrix<M> U = lu.U();
  const M *RESTRICT u_ptr = U.raw();
  const std::size_t R = row;

  if (lu.rank() < R)
    return M{0};

  M det = M{1};
  for (std::size_t i = 0; i < R; ++i)
    det *= u_ptr[i * R + i];

  return (lu.swap_count() % 2 == 0) ? det : -det;
}

// Inverse
template <NumericType M>
Matrix<M> Matrix<M>::inverse() const
  requires RealType<M>
{

  if (row != column)
    throw std::invalid_argument("Inverse requires square matrix");

  Linea::Decompositions::LU<M> lu(*this);
  const std::size_t n = row;

  if (lu.rank() < n)
    throw std::runtime_error("Matrix is singular");

  auto L = lu.L();
  auto U = lu.U();

  Matrix<M> inv(n, n);
  M *RESTRICT invr = inv.raw();

  for (std::size_t i = 0; i < n; ++i) {
    Vector<M> e(n, M{});
    e[i] = M{1};

    Vector<M> y = forward_substitution(L, e, lu.permutation());
    Vector<M> x = backward_substitution(U, y);

    const M *RESTRICT x_ptr = x.raw();

    for (std::size_t j = 0; j < n; ++j)
      invr[j * n + i] = x_ptr[j];
  }

  return inv;
}

// Solve (vector)
template <NumericType M>
Vector<M> Matrix<M>::solve(const Vector<M> &b) const
  requires RealType<M>
{

  Linea::Decompositions::LU<M> lu(*this);
  Matrix<M> L = lu.L();
  Matrix<M> U = lu.U();

  Vector<M> y = forward_substitution(L, b, lu.permutation());
  Vector<M> x = backward_substitution(U, y);

  return x;
}

// Solve (matrix)
template <NumericType M>
Matrix<M> Matrix<M>::solve(const Matrix<M> &B) const
  requires RealType<M>
{
  Linea::Decompositions::LU<M> lu(*this);
  Matrix<M> L = lu.L();
  Matrix<M> U = lu.U();

  const std::size_t n = row;
  const std::size_t m = B.ncols();

  Matrix<M> X(n, m);
  M *RESTRICT X_ptr = X.raw();
  const M *RESTRICT other_ptr = B.raw();

  Vector<M> b(n);
  M *RESTRICT b_ptr = b.raw();

  for (std::size_t j = 0; j < m; ++j) {

    for (std::size_t i = 0; i < n; ++i)
      b_ptr[i] = other_ptr[i * m + j];

    Vector<M> y = forward_substitution(L, b, lu.permutation());
    Vector<M> x = backward_sub(U, y);
    const M *x_ptr = x.raw();

    for (std::size_t i = 0; i < n; ++i)
      X_ptr[i * m + j] = x_ptr[i];
  }

  return X;
}

// Norms
template <NumericType M> double Matrix<M>::norm() const {
  double sum = 0;

  for (std::size_t i{}; i < this->row * this->column; i++) {
    sum += this->data[i] * this->data[i];
  }
  return std::sqrt(sum);
}

template <NumericType M>
double Matrix<M>::norm(MatrixNorm type, std::size_t max_iter) const {
  switch (type) {
  case MatrixNorm::Frobenius:
    return norm();
  case MatrixNorm::One:
    return norm_1();
  case MatrixNorm::Infinity:
    return norm_infinity();
  case MatrixNorm::Spectral:
    return norm_spectral(max_iter);
  default:
    return norm();
  }
}

template <NumericType M> double Matrix<M>::norm_1() const {
  double max_sum = 0;

  auto *RESTRICT a = this->raw();

  for (std::size_t j{}; j < this->column; j++) {
    double sum = 0;
    for (std::size_t i{}; i < this->row; i++) {
      sum += std::abs(a[i * column + j]);
    }
    if (sum > max_sum) {
      max_sum = sum;
    }
  }

  return max_sum;
}

template <NumericType M> double Matrix<M>::norm_infinity() const {
  double max_sum = 0;

  auto *RESTRICT a = this->raw();

  for (std::size_t i{}; i < this->row; i++) {
    double sum = 0;
    for (std::size_t j{}; j < this->column; j++) {
      sum += std::abs(a[i * column + j]);
    }
    if (sum > max_sum) {
      max_sum = sum;
    }
  }

  return max_sum;
}

template <NumericType M>
double Matrix<M>::norm_spectral(std::size_t max_iter) const {

  static_assert(std::is_floating_point_v<M>,
                "Spectral norm requires floating-point type");

  const std::size_t m = row;
  const std::size_t n = column;

  //  power iteration for Large matrices
  if (std::min(m, n) > 64) {

    const M *RESTRICT Ap = raw();

    std::vector<M> x(n), y(m);

    for (std::size_t j = 0; j < n; ++j)
      x[j] = M(1);

    // Normalize x
    M norm_x = 0;
    for (M v : x)
      norm_x += v * v;
    norm_x = std::sqrt(norm_x);

    for (M &v : x)
      v /= norm_x;

    M sigma = 0;
    const M tol = std::sqrt(std::numeric_limits<M>::epsilon());

    for (std::size_t iter = 0; iter < max_iter; ++iter) {

      // y = A * x
      for (std::size_t i = 0; i < m; ++i) {
        M sum = 0;
        const M *Ai = Ap + i * n;
        for (std::size_t j = 0; j < n; ++j)
          sum += Ai[j] * x[j];
        y[i] = sum;
      }

      // sigma = ||y||
      M sigma_new = 0;
      for (M v : y)
        sigma_new += v * v;
      sigma_new = std::sqrt(sigma_new);

      // Convergence check
      if (std::abs(sigma_new - sigma) / std::max(M(1), sigma_new) < tol) {
        sigma = sigma_new;
        break;
      }

      sigma = sigma_new;

      // x = Aᵀ * y
      for (std::size_t j = 0; j < n; ++j) {
        M sum = 0;
        for (std::size_t i = 0; i < m; ++i)
          sum += Ap[i * n + j] * y[i];
        x[j] = sum;
      }

      // Normalize x
      norm_x = 0;
      for (M v : x)
        norm_x += v * v;
      norm_x = std::sqrt(norm_x);

      const M inv_norm = M(1) / norm_x;
      for (M &v : x)
        v *= inv_norm;
    }

    return static_cast<double>(sigma);
  }

  // --- Small / medium matrices: exact SVD ---
  Decompositions::SVD<M> svd(*this);
  return *std::max_element(svd.singularValues().begin(),
                           svd.singularValues().end());
}

// Forward substitution
template <NumericType M>
Vector<M> Matrix<M>::forward_substitution(const Matrix<M> &L,
                                          const Vector<M> &b,
                                          const Vector<std::size_t> &piv) const
  requires RealType<M>
{
  std::size_t n = L.nrows();
  const std::size_t k = L.ncols();
  Vector<M> y(n);
  auto *RESTRICT y_ptr = y.raw();
  M const *RESTRICT Lp = L.raw();
  for (std::size_t i = 0; i < n; ++i) {
    M sum{};

    for (std::size_t j = 0; j < i; ++j) {
      sum += Lp[i * k + j] * y_ptr[j];
    }

    y_ptr[i] = (b[piv[i]] - sum) / Lp[i * k + i];
  }
  return y;
}

// Backward substitution
template <NumericType M>
Vector<M> Matrix<M>::backward_substitution(const Matrix<M> &U,
                                           const Vector<M> &y) const
  requires RealType<M>
{

  std::size_t n = U.row;
  std::size_t k = U.column;
  Vector<M> x(n);
  auto *RESTRICT x_ptr = x.raw();
  M const *RESTRICT U_ptr = U.raw();
  for (std::size_t i = n; i-- > 0;) {
    M sum{};

    for (std::size_t j = i + 1; j < n; ++j) {
      sum += U_ptr[i * k + j] * x_ptr[j];
    }

    x_ptr[i] = (y[i] - sum) / U_ptr[i * k + i];
  }
  return x;
}

} // namespace Linea

#endif // LINEA_MATRIX_UTILITIES_H