// created by : A.N. Prosper
// date : December 18th 2025
// time : 20:20

#ifndef MATRIX_H
#define MATRIX_H

#include <cassert>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

template <typename M> struct LUResult {
  std::vector<M> permutation_vector;
  std::size_t rank;
  std::size_t swap_count;
};

template <typename M> class LUFactor;

enum class NormType { Frobenius, One, Infinity, Spectral };

template <typename M>
// requires std::is_integral_v<M> || std::is_floating_point_v<M>
class Matrix {

private:
  // Attributes
  std::size_t row;
  std::size_t column;
  std::vector<M> data;

public:
  // Constructors:

  // Matrix() = default;

  // shape-only initialization
  Matrix(std::size_t row, std::size_t column)
      : row(row), column(column), data(row * column, M{}) {}

  // value-filled
  Matrix(std::size_t row, std::size_t column, M value)
      : row(row), column(column), data(row * column, value) {}

  // data-only initialization
  Matrix<M>(std::initializer_list<std::initializer_list<M>> list) {
    this->row = list.size();
    this->column = list.begin()->size();
    // Matrix<int> mm = {};
    for (const auto &values : list) {
      assert(values.size() == this->column);
      this->data.insert(this->data.end(), values.begin(), values.end());
    }
  }

  // Destructor:

  ~Matrix() {
    // delete  T_matrix[column][row];
  }

  // Getters:

  std::size_t getRow() const { return row; }

  std::size_t getColumn() const { return column; }

  std::vector<M> getdata() { return data; }

  // Static Methods:

  // Indentity
  static Matrix<M> Indentity(std::size_t row_column) {
    Matrix<M> I(row_column, row_column);

    for (std::size_t i = 0; i < row_column; ++i) {
      I.data[i * row_column + i] = M{1};
    }

    return I;
  }

  // zeros
  static Matrix<M> Zeros(std::size_t row, std::size_t column) {
    return Matrix<M>(row, column, M{0});
  }

  // ones
  static Matrix<M> Ones(std::size_t row, std::size_t column) {
    return Matrix<M>(row, column, M{1});
  }

  // random matrix initialization
  static Matrix<M> rand_fill(std::size_t row, std::size_t column, M min_range,
                             M max_range) {
    auto low = std::min(min_range, max_range);
    auto high = std::max(min_range, max_range);

    Matrix<M> result(row, column);
    static thread_local std::mt19937 engine(std::random_device{}());

    if constexpr (std::is_integral_v<M>) {
      std::uniform_int_distribution<M> distribute(low, high);

      for (std::size_t i{}; i < row; ++i) {
        for (std::size_t j{}; j < column; ++j) {
          result(i, j) = distribute(engine);
        }
      }

    } else if constexpr (std::is_floating_point_v<M>) {
      std::uniform_real_distribution<M> distribute(low, high);
      for (std::size_t i{}; i < row; ++i) {
        for (std::size_t j{}; j < column; ++j) {
          result(i, j) = distribute(engine);
        }
      }
    }

    return result;
  }

public:
  // Methods

  //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>//

  // Operations:

  // equality
  bool operator==(Matrix<M> other) const noexcept {
    if ((this->row != other.row) || (this->column != other.column)) {
      return false;
    }
    for (std::size_t i{}; i < this->data.size(); i++) {
      if (this->data[i] != other.data[i]) {
        return false;
      }
    }
    return true;
  };

  // inequality
  bool operator!=(Matrix<M> other) noexcept { return !(*this == other); }

  // addition
  Matrix<M> operator+(Matrix<M> other) {

    if ((this->row != other.row) || (this->column != other.column)) {
      throw std::invalid_argument(
          "Matrix addition requires identical dimensions.");
    }
    Matrix<M> result(this->row, this->column);

    for (std::size_t i = 0; i < this->row; ++i) {
      for (std::size_t j = 0; j < this->column; ++j) {
        result.data[i * this->column + j] =
            this->data[i * this->column + j] + other.data[i * this->column + j];
      }
    }
    return result;
  }

  // subtraction
  Matrix<M> operator-(Matrix<M> other) {

    if ((this->row != other.row) || (this->column != other.column)) {
      throw std::invalid_argument(
          "Matrix subtraction requires identical dimensions.");
    }
    Matrix<M> result(this->row, this->column);

    for (std::size_t i = 0; i < this->row; ++i) {
      for (std::size_t j = 0; j < this->column; ++j) {
        result.data[i * this->column + j] =
            this->data[i * this->column + j] - other.data[i * this->column + j];
      }
    }
    return result;
  }

  // unary minus
  Matrix<M> operator-() {
    Matrix<M> result(row, column);
    for (auto &data_index : data) {
      for (auto &value : result.data) {
        value = (-M{1} * data_index);
      }
    }
    return result;
  }

  // scalar multiplication
  Matrix<M> operator*(M scalar) const {

    Matrix<M> result(row, column);

    for (std::size_t i{}; i < row; i++) {
      for (std::size_t j{}; j < column; j++) {
        result(i, j) = scalar * (*this)(i, j);
      }
    }
    return result;
  }

  // matrix multiplication
  Matrix<M> operator*(Matrix<M> &other) {
    if (this->column != other.row) {
      throw std::invalid_argument("A.column must equal B.row");
    }

    Matrix<M> result(row, other.column);
    M sum{};
    for (std::size_t i = 0; i < row; ++i) {
      for (std::size_t k = 0; k < other.row; ++k) {
        sum = (*this)(i, k);
        for (std::size_t j = 0; j < other.column; ++j) {
          result(i, j) += sum * other(k, j);
        }
      }
    }
    return result;
  }

  // index
  M &operator()(std::size_t i, std::size_t j) {
    if (i >= row || j >= column) {
      throw std::out_of_range("Matrix index is out of range.");
    }
    return data[i * column + j];
  }

  //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>//

  // Mathematical functions

  // logarithm
  Matrix<M> log() {

    Matrix<M> result(this->row, this->column);
    for (auto &value : this->data) {
      for (auto &log_value : result.data) {
        log_value = std::log(value);
      }
    }
    return result;
  }

  // power
  Matrix<M> pow(M k) {

    Matrix<M> result(this->row, this->column);
    for (auto &value : this->data) {
      for (auto &pow_value : result.data) {
        pow_value = std::pow(value, k);
      }
    }
    return result;
  }

  // exponent
  Matrix<M> exp() {

    Matrix<M> result(this->row, this->column);
    for (auto &value : this->data) {
      for (auto &exp_value : result.data) {
        exp_value = std::exp(value);
      }
    }
    return result;
  }

  //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>//

  // Special utility methods:

  // display
  void display() {
    std::cout << "displaying.." << std::endl;
    for (std::size_t i{}; i < row; i++) {
      for (std::size_t j{}; j < column; j++) {
        std::cout << data[i * column + j] << " ";
      }
      std::cout << std::endl;
    }
  }

  // insert
  void insert(std::size_t row_index, std::size_t col_index, M number) {
    if ((row_index >= 0 && row_index < row) &&
        (col_index >= 0 && col_index < column)) {
      data[row_index * column + col_index] = number;
    } else {
      if (row_index < row) {
        throw std::invalid_argument("Invalid column index");
      } else if (col_index < column) {
        throw std::invalid_argument("Invalid row index");
      }
    }
  }

  // shape
  void shape() {
    std::cout << "(" << row << ", " << column << ")" << std::endl;
  }

  // size
  int size() { return static_cast<int>(this->row * this->column); }

  // empty
  bool empty() noexcept { return this->data.empty(); }

  // symmetric
  bool symmetric() const noexcept {
    if ((this->row != this->Transpose().row) ||
        (this->column != this->Transpose().column)) {
      return false;
    }

    for (std::size_t i{}; i < this->data.size(); i++) {
      if (this->data[i] != this->Transpose().data[i]) {
        return false;
      }
    }
    return true;
  }

  // square
  bool square() noexcept { return this->row == this->column; }

  // singular
  bool singular() noexcept { return Rank() < row; }

  // diagonal
  M diagonal() {

    M major_diagonal = M{1};

    if (row != column) {
      throw std::invalid_argument("Matrix diagonal requires row == column.");
    }
    for (std::size_t i{}; i < this->row; i++) {
      major_diagonal *= (*this)(i, i);
    }
    return major_diagonal;
  }

  // transpose
  Matrix<M> Transpose() const {
    Matrix<M> result(this->column, this->row);

    for (std::size_t i = 0; i < this->row; ++i) {
      for (std::size_t j = 0; j < this->column; ++j) {
        result.data[j * this->row + i] = data[i * this->column + j];
      }
    }

    return result;
  }

  // Rank
  std::size_t Rank() { return lu_decompose().get_rank(); }

  // cofactor
  M cofactor(std::size_t row_index, std::size_t column_index) {
    if (row != column) {
      throw std::invalid_argument("Cofactor requires a square matrix.");
    }

    if (row_index >= row || column_index >= column) {
      throw std::out_of_range("Matrix index is out of range.");
    }

    Matrix<M> minor(row - 1, column - 1);
    std::size_t k = 0;

    for (std::size_t i = 0; i < row; ++i) {
      for (std::size_t j = 0; j < column; ++j) {
        if (i != row_index && j != column_index) {
          minor.data[k++] = (*this)(i, j);
        }
      }
    }

    M sign = ((row_index + column_index) % 2 ? -M{1} : M{1});
    return sign * minor.determinant();
  }

  // cofactor matrix
  Matrix<M> cofactor_matrix() {}

  // adjoint
  Matrix<M> adjoint() {}

  // determinant
  M determinant() {
    if (row != column) {
      throw std::invalid_argument("Matrix determinant requires row == column.");
    }

    LUFactor<M> result = lu_decompose();
    Matrix<M> U_matrix = result.extract_U();

    if (result.get_rank() < U_matrix.getRow()) {
      return M{0};
    }

    M major_diagonal = M{1};
    for (std::size_t i{}; i < U_matrix.getRow(); i++) {
      major_diagonal *= U_matrix(i, i);
    }
    bool even_swaps = (result.get_swap_count() % 2 == 0);
    return even_swaps ? major_diagonal : -major_diagonal;
  }

  // inverse
  Matrix<M> Inverse() {

    LUFactor<M> lu_result = lu_decompose();
    Matrix<M> L = lu_result.extract_L();
    Matrix<M> U = lu_result.extract_U();
    std::vector<M> piv = lu_result.get_permutation_vector();
    std::size_t n = this->row;

    if (lu_result.get_rank() < n) {
      throw std::runtime_error("Matrix is singular; cannot compute inverse");
    }

    Matrix<M> inverse_matrix(n, n);

    for (std::size_t i = 0; i < n; i++) {

      std::vector<M> e_i(n, M{});
      e_i[i] = M{1};

      std::vector<M> y = forward_substitution(L, e_i, piv);

      std::vector<M> x = backward_substitution(U, y);

      for (std::size_t j = 0; j < n; j++) {
        inverse_matrix(j, i) = x[j];
      }
    }
    return inverse_matrix;
  }

  // linear system solver
  std::vector<M> solve(std::vector<M> b) {

    LUFactor<M> lu_result = lu_decompose();
    Matrix<M> L = lu_result.extract_L();
    Matrix<M> U = lu_result.extract_U();
    std::vector<M> piv = lu_result.get_permutation_vector();

    std::vector<M> y = forward_substitution(L, b, piv);

    std::vector<M> x = backward_substitution(U, y);

    return x;
  }

  Matrix<M> solve(Matrix<M> B) {

    LUFactor<M> lu_result = lu_decompose();
    Matrix<M> L = lu_result.extract_L();
    Matrix<M> U = lu_result.extract_U();
    std::vector<M> piv = lu_result.get_permutation_vector();

    std::size_t n = this->row;
    std::size_t m = B.column;

    Matrix<M> X(n, m);

    for (std::size_t i = 0; i < m; i++) {

      std::vector<M> b_i(n);
      for (std::size_t j = 0; j < n; j++) {
        b_i[j] = B(j, i);
      }

      std::vector<M> y = forward_substitution(L, b_i, piv);
      std::vector<M> x = backward_substitution(U, y);

      for (std::size_t j = 0; j < n; j++) {
        X(j, i) = x[j];
      }
    }

    return X;
  }

  // LU Decomposition with partial pivoting.
  LUFactor<M> lu_decompose(M epsilon = M(0)) {

    const std::size_t m = row;
    const std::size_t n = column;
    const std::size_t k_max = std::min(m, n);

    Matrix<M> LU = (*this);

    LUResult<M> result;
    result.permutation_vector.resize(m);
    for (std::size_t i{}; i < m; ++i) {
      result.permutation_vector[i] = i;
    }
    std::vector<M> row_index = result.permutation_vector;

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
        std::swap(result.permutation_vector[k],
                  result.permutation_vector[pivot]);
        ++result.swap_count;
      }

      const std::size_t physical_pivot_row = row_index[k];

      // Elimination
      for (std::size_t i = k + 1; i < m; ++i) {
        const std::size_t physical_row = row_index[i];
        LU(physical_row, k) /= LU(physical_pivot_row, k);

        for (std::size_t j = k + 1; j < n; ++j) {
          LU(physical_row, j) -=
              LU(physical_row, k) * LU(physical_pivot_row, j);
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

  double norm() {
    double sum = 0;

    for (std::size_t i{}; i < this->row * this->column; i++) {
      sum += this->data[i] * this->data[i];
    }
    return std::sqrt(sum);
  }

  double norm(NormType type) {
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

private:
  // Norm implementation

  double norm_1() {
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
  };
  double norm_infinity() {
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
  };
  double norm_spectral() {
    // Spectral norm is the largest singular value
    // For now, i will not compute it
    return 0;
  };

  // forward and backward substitution utilities for triangular matrices

  std::vector<M> forward_substitution(Matrix<M> L, std::vector<M> b,
                                      std::vector<M> piv) {
    std::size_t n = L.row;
    std::vector<M> y(n);

    for (std::size_t i = 0; i < n; ++i) {
      M sum{};

      for (std::size_t j = 0; j < i; ++j) {
        sum += L(i, j) * y[j];
      }

      y[i] = (b[piv[i]] - sum) / L(i, i);
    }
    return y;
  }

  std::vector<M> backward_substitution(Matrix<M> U, std::vector<M> y) {
    std::size_t n = U.row;
    std::vector<M> x(n);

    for (std::size_t i = n; i-- > 0;) {
      M sum{};

      for (std::size_t j = i + 1; j < n; ++j) {
        sum += U(i, j) * x[j];
      }

      x[i] = (y[i] - sum) / U(i, i);
    }
    return x;
  }
};

// scalar multiplication
template <typename T, typename Tp>
// requires std::is_integral_v<T> || std::is_floating_point_v<T>
Matrix<Tp> operator*(T scalar, Matrix<Tp> &matrix) {

  Matrix<Tp> result(matrix.getRow(), matrix.getColumn());

  for (std::size_t i{}; i < result.getRow(); i++) {
    for (std::size_t j{}; j < result.getColumn(); j++) {
      result(i, j) = scalar * matrix(i, j);
    }
  }
  return result;
}

// LUFactor
template <typename M> class LUFactor {

private:
  // --- attributes ---

  Matrix<M> LU;
  LUResult<M> info;

public:
  // ---- methods ----

  // constructor
  LUFactor(Matrix<M> lu, LUResult<M> result)
      : LU(std::move(lu)), info(std::move(result)) {}

  // getter

  std::size_t get_rank() { return info.rank; }

  std::size_t get_swap_count() { return info.swap_count; }

  std::vector<M> get_permutation_vector() { return info.permutation_vector; }

  LUResult<M> get_Info() { return info; }

  // Extract L as m × min(m,n) lower triangular matrix with unit diagonal from
  // the LU matrix
  Matrix<M> extract_L() const {

    const std::size_t m = LU.getRow();
    const std::size_t n = LU.getColumn();
    const std::size_t k_max = std::min(m, n);

    Matrix<M> L(m, k_max, M{});

    for (std::size_t i{}; i < m; i++) {

      // unit diagonal
      if (i < k_max) {
        L(i, i) = M{1};
      }

      // subdiagonal entries
      for (std::size_t j{}; j < std::min(i, k_max); j++) {
        L(i, j) = LU(i, j);
      }
    }
    return L;
  }

  // Extract U as min(m,n) × n upper triangular matrix from LU matrix
  Matrix<M> extract_U() const {

    const std::size_t m = LU.getRow();
    const std::size_t n = LU.getColumn();
    const std::size_t k_max = std::min(m, n);

    Matrix<M> U(k_max, n, M{});

    for (std::size_t i{}; i < k_max; i++) {
      for (std::size_t j = i; j < n; j++) {
        U(i, j) = LU(i, j);
      }
    }
    return U;
  }

  // Extract permutation matrix P (m × m) such that P*A = L*U
  Matrix<M> extract_P() const {

    const std::size_t m = LU.getRow();
    Matrix<M> P(m, m, M{});

    for (std::size_t i{}; i < m; i++) {
      P(i, info.permutation_vector[i]) = M{1};
    }
    return P;
  }
};

#endif
