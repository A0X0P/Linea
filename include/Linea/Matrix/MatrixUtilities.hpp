// created by : A.N. Prosper
// date : january 25th 2026
// time : 15:05

#ifndef LINEA_MATRIX_UTILITIES_H
#define LINEA_MATRIX_UTILITIES_H

#include "Matrix.hpp"
#include <iostream>
#include <string>

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
  return Rank() < row;
}

// Trace
template <NumericType M> M Matrix<M>::trace() const {
  if (row != column) {
    throw std::invalid_argument("Matrix trace requires row == column.");
  }

  const std::size_t n = row;
  const auto *base = data.data();
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
  const auto *base = data.data();

  Vector<M> _diagonal(n);

  const auto *p = (type == Diagonal::Major) ? base : base + (n - 1);

  const std::size_t stride = (type == Diagonal::Major) ? (n + 1) : (n - 1);

  for (std::size_t i = 0; i < n; ++i, p += stride) {
    _diagonal[i] = *p;
  }

  return _diagonal;
}

// Transpose
template <NumericType M> Matrix<M> Matrix<M>::Transpose() const {
  Matrix<M> result(this->column, this->row);

  for (std::size_t i = 0; i < this->row; ++i) {
    for (std::size_t j = 0; j < this->column; ++j) {
      result.data[j * this->row + i] = data[i * this->column + j];
    }
  }

  return result;
}

// Rank
template <NumericType M> std::size_t Matrix<M>::Rank() const {
  return lu_decompose().get_rank();
}

// Reshape
template <NumericType M>
Matrix<M> Matrix<M>::Reshape(std::size_t nrow, std::size_t ncol) const {
  const std::size_t reshape_size = nrow * ncol;

  if (reshape_size != data.size()) {
    throw std::invalid_argument(
        "No possible reshape exist for this combination.");
  }
  Matrix<M> result(nrow, ncol);
  auto *out = result.data.data();
  auto *in = data.data();
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

  const M *a = data.data();
  M *out = sub_matrix.data.data();

  for (std::size_t i = 0; i < row; ++i) {
    if (i == row_idx)
      continue;

    const M *row_ptr = a + i * column;

    for (std::size_t j = 0; j < column; ++j) {
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
  for (std::size_t i = 0; i < nrows; ++i) {
    for (std::size_t j = 0; j < ncols; ++j) {
      result(i, j) = (*this)(row + i, col + j);
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
  for (std::size_t i = 0; i < row; ++i) {
    for (std::size_t j = 0; j < column; ++j) {
      result(i, j) = cofactor(i, j);
    }
  }
  return result;
}

// Adjoint
template <NumericType M> Matrix<M> Matrix<M>::adjoint() const {
  return cofactor_matrix().Transpose();
}

// Determinant
template <NumericType M> M Matrix<M>::determinant() const {
  if (row != column) {
    throw std::invalid_argument("Matrix determinant requires row == column.");
  }

  LUFactor<M> result = lu_decompose();
  Matrix<M> U_matrix = result.extract_U();

  if (result.get_rank() < U_matrix.nrows()) {
    return M{0};
  }

  M major_diagonal = M{1};
  for (std::size_t i{}; i < U_matrix.nrows(); i++) {
    major_diagonal *= U_matrix(i, i);
  }
  bool even_swaps = (result.get_swap_count() % 2 == 0);
  return even_swaps ? major_diagonal : -major_diagonal;
}

// Inverse
template <NumericType M> Matrix<M> Matrix<M>::Inverse() const {
  LUFactor<M> lu_result = lu_decompose();
  Matrix<M> L = lu_result.extract_L();
  Matrix<M> U = lu_result.extract_U();
  Vector<M> piv = lu_result.get_permutation_vector();
  std::size_t n = this->row;

  if (lu_result.get_rank() < n) {
    throw std::runtime_error("Matrix is singular; cannot compute inverse");
  }

  Matrix<M> inverse_matrix(n, n);

  for (std::size_t i = 0; i < n; i++) {
    Vector<M> e_i(n, M{});
    e_i[i] = M{1};

    Vector<M> y = forward_substitution(L, e_i, piv);
    Vector<M> x = backward_substitution(U, y);

    for (std::size_t j = 0; j < n; j++) {
      inverse_matrix(j, i) = x[j];
    }
  }
  return inverse_matrix;
}

// Solve (vector)
template <NumericType M> Vector<M> Matrix<M>::solve(const Vector<M> &b) const {
  LUFactor<M> lu_result = lu_decompose();
  Matrix<M> L = lu_result.extract_L();
  Matrix<M> U = lu_result.extract_U();
  Vector<M> piv = lu_result.get_permutation_vector();

  Vector<M> y = forward_substitution(L, b, piv);
  Vector<M> x = backward_substitution(U, y);

  return x;
}

// Solve (matrix)
template <NumericType M> Matrix<M> Matrix<M>::solve(const Matrix<M> &B) const {
  LUFactor<M> lu_result = lu_decompose();
  Matrix<M> L = lu_result.extract_L();
  Matrix<M> U = lu_result.extract_U();
  Vector<M> piv = lu_result.get_permutation_vector();

  std::size_t n = this->row;
  std::size_t m = B.column;

  Matrix<M> X(n, m);

  for (std::size_t i = 0; i < m; i++) {
    Vector<M> b_i(n);
    for (std::size_t j = 0; j < n; j++) {
      b_i[j] = B(j, i);
    }

    Vector<M> y = forward_substitution(L, b_i, piv);
    Vector<M> x = backward_substitution(U, y);

    for (std::size_t j = 0; j < n; j++) {
      X(j, i) = x[j];
    }
  }

  return X;
}

// LU Decomposition
template <NumericType M> LUFactor<M> Matrix<M>::lu_decompose(M epsilon) const {
  const std::size_t m = row;
  const std::size_t n = column;
  const std::size_t k_max = std::min(m, n);

  Matrix<M> LU = (*this);

  LUResult<M> result{};
  result.rank = 0;
  result.swap_count = 0;
  result.permutation_vector = Vector<M>(m);
  for (std::size_t i{}; i < m; ++i) {
    result.permutation_vector[i] = i;
  }
  Vector<M> row_index = result.permutation_vector;

  // Compute adaptive epsilon if not provided by the user
  M effective_epsilon = epsilon;
  if (effective_epsilon == M(0)) {
    M max_elem = M(0);
    for (std::size_t i = 0; i < m; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        max_elem = std::max(max_elem, std::abs((*this)(i, j)));
      }
    }
    effective_epsilon =
        std::numeric_limits<M>::epsilon() * max_elem * std::max(m, n);
  }

  for (std::size_t k{}; k < k_max; ++k) {
    // Pivot selection
    std::size_t pivot = k;
    M max_value = std::abs(LU(row_index[k], k));

    for (std::size_t i = k + 1; i < m; ++i) {
      M val = std::abs(LU(row_index[i], k));
      if (val > max_value) {
        max_value = val;
        pivot = i;
      }
    }

    if (max_value < effective_epsilon) {
      continue;
    }

    if (pivot != k) {
      std::swap(row_index[k], row_index[pivot]);
      std::swap(result.permutation_vector[k], result.permutation_vector[pivot]);
      ++result.swap_count;
    }

    const std::size_t physical_pivot_row = row_index[k];

    // Elimination
    for (std::size_t i = k + 1; i < m; ++i) {
      const std::size_t physical_row = row_index[i];
      LU(physical_row, k) /= LU(physical_pivot_row, k);

      for (std::size_t j = k + 1; j < n; ++j) {
        LU(physical_row, j) -= LU(physical_row, k) * LU(physical_pivot_row, j);
      }
    }

    ++result.rank;
  }

  // Physically permute LU to match permutation
  Matrix<M> LU_perm(m, n);
  for (std::size_t i{}; i < m; ++i) {
    for (std::size_t j{}; j < n; ++j) {
      LU_perm(i, j) = LU(row_index[i], j);
    }
  }

  return LUFactor<M>(std::move(LU_perm), std::move(result));
}

// Norms
template <NumericType M> double Matrix<M>::norm() const {
  double sum = 0;

  for (std::size_t i{}; i < this->row * this->column; i++) {
    sum += this->data[i] * this->data[i];
  }
  return std::sqrt(sum);
}

template <NumericType M> double Matrix<M>::norm(NormType type) const {
  switch (type) {
  case NormType::Frobenius:
    return norm();
  case NormType::One:
    return norm_1();
  case NormType::Infinity:
    return norm_infinity();
  case NormType::Spectral:
    return norm_spectral();
  default:
    return norm();
  }
}

template <NumericType M> double Matrix<M>::norm_1() const {
  double max_sum = 0;
  for (std::size_t j{}; j < this->column; j++) {
    double sum = 0;
    for (std::size_t i{}; i < this->row; i++) {
      sum += std::abs((*this)(i, j));
    }
    if (sum > max_sum) {
      max_sum = sum;
    }
  }

  return max_sum;
}

template <NumericType M> double Matrix<M>::norm_infinity() const {
  double max_sum = 0;

  for (std::size_t i{}; i < this->row; i++) {
    double sum = 0;
    for (std::size_t j{}; j < this->column; j++) {
      sum += std::abs((*this)(i, j));
    }
    if (sum > max_sum) {
      max_sum = sum;
    }
  }

  return max_sum;
}

template <NumericType M> double Matrix<M>::norm_spectral() const {
  // Spectral norm is the largest singular value
  // For now, not implemented
  return 0;
}

// Forward substitution
template <NumericType M>
Vector<M> Matrix<M>::forward_substitution(const Matrix<M> &L,
                                          const Vector<M> &b,
                                          const Vector<M> &piv) const {
  std::size_t n = L.row;
  Vector<M> y(n);

  for (std::size_t i = 0; i < n; ++i) {
    M sum{};

    for (std::size_t j = 0; j < i; ++j) {
      sum += L(i, j) * y[j];
    }

    y[i] = (b[piv[i]] - sum) / L(i, i);
  }
  return y;
}

// Backward substitution
template <NumericType M>
Vector<M> Matrix<M>::backward_substitution(const Matrix<M> &U,
                                           const Vector<M> &y) const {
  std::size_t n = U.row;
  Vector<M> x(n);

  for (std::size_t i = n; i-- > 0;) {
    M sum{};

    for (std::size_t j = i + 1; j < n; ++j) {
      sum += U(i, j) * x[j];
    }

    x[i] = (y[i] - sum) / U(i, i);
  }
  return x;
}

} // namespace Linea

#endif // LINEA_MATRIX_UTILITIES_H