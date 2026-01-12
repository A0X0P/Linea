// created by : A.N. Prosper
// date : December 18th 2025
// time : 20:20

#ifndef MATRIX_H
#define MATRIX_H

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace Linea {

template <typename T, typename S>
using scalar_multiply_result_t =
    std::conditional_t<std::is_integral_v<S> && std::is_integral_v<T>, int,
                       double>;

template <typename M> struct LUResult {
  std::vector<M> permutation_vector;
  std::size_t rank;
  std::size_t swap_count;
};

template <typename M> class LUFactor;

enum class NormType { Frobenius, One, Infinity, Spectral };

enum class Diagonal { Major, Minor };

template <typename M>
// requires std::is_integral_v<M> || std::is_floating_point_v<M>
class Matrix {

private:
  // Attributes
  std::size_t row;
  std::size_t column;
  std::vector<M> data;

public:
  template <typename U> friend class Matrix;
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

  //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>//

  // Dimension

  std::size_t nrows() const { return row; }

  std::size_t ncols() const { return column; }

  // Element extraction

  std::vector<M> getRow(std::size_t row_index) const {
    if (row_index >= row) {
      throw std::out_of_range("Row index out of range");
    }

    std::vector<M> result(column);
    std::copy(data.begin() + row_index * column,
              data.begin() + (row_index + 1) * column, result.begin());
    return result;
  }

  std::vector<M> getColumn(std::size_t column_index) const {
    if (column_index >= column) {
      throw std::out_of_range("Column index out of range");
    }
    std::vector<M> result(row);

    for (std::size_t j = 0; j < row; ++j) {
      result[j] = data.at(j * column + column_index);
    }

    return result;
  }

  const std::vector<M> &getdata() const & { return data; }

  //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>//

  // Setters:

  void setRow(std::size_t row_index, const std::vector<M> &other) {
    if (row_index >= row) {
      throw std::out_of_range("Row index out of range");
    }
    if (other.size() != column) {
      throw std::invalid_argument("Row size mismatch");
    }

    for (std::size_t i = 0; i < column; ++i) {
      data[row_index * column + i] = other[i];
    }
  }

  void setColumn(std::size_t column_index, const std::vector<M> &other) {
    if (column_index >= column) {
      throw std::out_of_range("Column index out of range");
    }
    if (other.size() != row) {
      throw std::invalid_argument("Column size mismatch");
    }

    for (std::size_t i = 0; i < row; ++i) {
      data[i * column + column_index] = other[i];
    }
  }

  //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>//

  // Permutation operations

  void swap_row(std::size_t row1, std::size_t row2) {
    if (row1 >= row || row2 >= row) {
      throw std::out_of_range("Row index out of range");
    }
    if (row1 == row2) {
      return;
    }

    std::swap_ranges(data.begin() + row1 * column,
                     data.begin() + (row1 + 1) * column,
                     data.begin() + row2 * column);
  }

  void swap_column(std::size_t column1, std::size_t column2) {
    if (column1 >= column || column2 >= column) {
      throw std::out_of_range("Column index out of range");
    }
    if (column2 == column1) {
      return;
    }

    for (std::size_t i = 0; i < row; ++i) {
      std::swap(data[i * column + column1], data[i * column + column2]);
    }
  }

  //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>//

  // Static Methods:

  // Identity
  static Matrix<M> Identity(std::size_t row_column) {
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

  // addition (element wise)
  Matrix<M> operator+(Matrix<M> other) {

    if ((this->row != other.row) || (this->column != other.column)) {
      throw std::invalid_argument(
          "Matrix addition requires identical dimensions.");
    }
    Matrix<M> result(this->row, this->column);

    auto *out = result.data.data();
    const auto *a = data.data();
    const auto *in = other.data.data();
    for (std::size_t i = 0; i < data.size(); ++i) {
      out[i] = a[i] + in[i];
    }
    return result;
  }

  // subtraction (element wise)
  Matrix<M> operator-(Matrix<M> other) {

    if ((this->row != other.row) || (this->column != other.column)) {
      throw std::invalid_argument(
          "Matrix subtraction requires identical dimensions.");
    }
    Matrix<M> result(this->row, this->column);

    auto *out = result.data.data();
    const auto *a = data.data();
    const auto *in = other.data.data();
    for (std::size_t i = 0; i < data.size(); ++i) {
      out[i] = a[i] - in[i];
    }
    return result;
  }

  // unary minus
  Matrix<M> operator-() {
    Matrix<M> result(row, column);

    auto *out = result.data.data();
    const auto *a = data.data();

    for (std::size_t i = 0; i < data.size(); ++i) {
      out[i] = -a[i];
    }
    return result;
  }

  // scalar multiplication
  template <typename S>
  Matrix<scalar_multiply_result_t<M, S>> operator*(S scalar) {

    using ResultType = scalar_multiply_result_t<M, S>;

    Matrix<ResultType> result(this->row, this->column);

    auto *out = result.data.data();
    const auto *a = data.data();

    for (std::size_t i = 0; i < data.size(); ++i) {
      out[i] = scalar * a[i];
    }
    return result;
  }

  // scalar multiplication
  template <typename T, typename S>
  friend Matrix<scalar_multiply_result_t<T, S>> operator*(S scalar,
                                                          Matrix<T> &matrix);

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

  // Hadamard product
  Matrix<M> Hadamard_product(Matrix<M> other) {
    if ((this->row != other.row) || (this->column != other.column)) {
      throw std::invalid_argument(
          "Hadamard product requires identical dimensions.");
    }
    Matrix<M> result(this->row, this->column);

    auto *out = result.data.data();
    const auto *a = data.data();
    const auto *in = other.data.data();
    for (std::size_t i = 0; i < data.size(); ++i) {
      out[i] = a[i] * in[i];
    }
    return result;
  }

  // division (element wise)
  Matrix<M> operator/(Matrix<M> other) {

    if ((this->row != other.row) || (this->column != other.column)) {
      throw std::invalid_argument(
          "Matrix division requires identical dimensions.");
    }
    Matrix<M> result(this->row, this->column);

    auto *out = result.data.data();
    const auto *a = data.data();
    const auto *in = other.data.data();
    for (std::size_t i = 0; i < data.size(); ++i) {
      if (in[i] == M{0}) {
        throw std::domain_error("Division by zero.");
      }
      out[i] = a[i] / in[i];
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

  // sine
  Matrix<double> sin() const {
    static_assert(std::is_floating_point_v<M>,
                  "sin() requires floating-point matrix");

    Matrix<double> result(row, column);
    auto *out = result.data.data();
    const auto *a = data.data();
    const std::size_t n = data.size();

    for (std::size_t i = 0; i < n; ++i) {
      out[i] = std::sin(a[i]);
    }
    return result;
  }

  // cosine
  Matrix<double> cos() const {
    static_assert(std::is_floating_point_v<M>,
                  "cos() requires floating-point matrix");
    Matrix<double> result(row, column);
    auto *out = result.data.data();
    const auto *a = data.data();
    const std::size_t n = data.size();

    for (std::size_t i = 0; i < n; ++i) {
      out[i] = std::cos(a[i]);
    }
    return result;
  }

  // tangent
  Matrix<double> tan() const {
    static_assert(std::is_floating_point_v<M>,
                  "tan() requires floating-point matrix");
    Matrix<double> result(row, column);
    auto *out = result.data.data();
    const auto *a = data.data();
    const std::size_t n = data.size();

    for (std::size_t i = 0; i < n; ++i) {
      if (std::cos(a[i]) == 0.0) {
        throw std::domain_error("tan undefined for element " +
                                std::to_string(a[i]));
      }
      out[i] = std::tan(a[i]);
    }
    return result;
  }

  // square root
  Matrix<double> sqrt() const {
    static_assert(std::is_floating_point_v<M>,
                  "sqrt() requires floating-point matrix");
    Matrix<double> result(row, column);
    auto *out = result.data.data();
    const auto *a = data.data();
    const std::size_t n = data.size();

    for (std::size_t i = 0; i < n; ++i) {
      if (a[i] < 0.0) {
        throw std::domain_error("sqrt undefined for element " +
                                std::to_string(a[i]));
      }
      out[i] = std::sqrt(a[i]);
    }
    return result;
  }

  // logarithm
  Matrix<double> log() const {
    static_assert(std::is_floating_point_v<M>,
                  "log() requires floating-point matrix");

    Matrix<double> result(row, column);
    auto *out = result.data.data();
    const auto *a = data.data();
    const std::size_t n = data.size();

    for (std::size_t i = 0; i < n; ++i) {
      if (a[i] <= 0.0) {
        throw std::domain_error("log undefined for element " +
                                std::to_string(a[i]));
      }
      out[i] = std::log(a[i]);
    }

    return result;
  }

  // power
  Matrix<int> pow(unsigned int exponent) const {
    static_assert(std::is_integral_v<M>,
                  "pow(unsigned) requires integral matrix");

    Matrix<int> result(row, column);
    auto *out = result.data.data();
    const auto *a = data.data();
    const std::size_t n = data.size();

    for (std::size_t i = 0; i < n; ++i) {
      out[i] = integer_pow(a[i], exponent);
    }

    return result;
  }

  Matrix<double> pow(double exponent) const {
    static_assert(std::is_floating_point_v<M>,
                  "pow(double) requires floating-point matrix");

    Matrix<double> result(row, column);
    auto *out = result.data.data();
    const auto *a = data.data();
    const std::size_t n = data.size();

    for (std::size_t i = 0; i < n; ++i) {
      out[i] = std::pow(static_cast<double>(a[i]), exponent);
    }

    return result;
  }

  // exponent
  Matrix<double> exp() const {
    static_assert(std::is_floating_point_v<M>,
                  "exp() requires floating-point matrix");

    Matrix<double> result(row, column);
    auto *out = result.data.data();
    const auto *a = data.data();
    const std::size_t n = data.size();

    for (std::size_t i = 0; i < n; ++i) {
      out[i] = std::exp(a[i]);
    }

    return result;
  }

  //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>//

  // Iterators
  using iterator = typename std::vector<M>::iterator;
  using reverse_iterator = typename std::vector<M>::reverse_iterator;
  using const_iterator = typename std::vector<M>::const_iterator;
  using const_reverse_iterator =
      typename std::vector<M>::const_reverse_iterator;

  // begin
  iterator begin() { return data.begin(); }
  reverse_iterator rbegin() { return data.rbegin(); }
  const_iterator begin() const { return data.begin(); }
  const_reverse_iterator rbegin() const { return data.rbegin(); }
  const_iterator cbegin() const { return data.cbegin(); }
  const_reverse_iterator crbegin() const { return data.crbegin(); }

  // end
  iterator end() { return data.end(); }
  reverse_iterator rend() { return data.rend(); }
  const_iterator end() const { return data.end(); }
  const_reverse_iterator rend() const { return data.rend(); }
  const_iterator cend() const { return data.cend(); }
  const_reverse_iterator crend() const { return data.crend(); }

  //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>//

  // Statistical Operations

  M sum() const { return std::accumulate(data.begin(), data.end(), M{0}); }

  double mean() const {
    if (data.empty()) {
      throw std::domain_error("Cannot compute mean of empty matrix");
    }
    return static_cast<double>(sum()) / static_cast<double>(data.size());
  }

  M min() const {
    if (data.empty()) {
      throw std::runtime_error("Cannot compute minimum of empty matrix");
    }
    return std::min_element(data.begin(), data.end());
  }

  M max() const {
    if (data.empty()) {
      throw std::runtime_error("Cannot compute maximum of empty matrix");
    }
    return std::max_element(data.begin(), data.end());
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
    if ((row_index >= row) || (col_index >= column)) {
      throw std::out_of_range("Matrix index out of bounds: (" +
                              std::to_string(row_index) + ", " +
                              std::to_string(col_index) + ")");
    }
    data[row_index * column + col_index] = number;
  }

  // shape
  void shape() {
    std::cout << "(" << row << ", " << column << ")" << std::endl;
  }

  // size
  std::size_t size() { return row * column; }

  // empty
  bool empty() noexcept { return this->data.empty(); }

  // symmetric
  bool symmetric() noexcept {
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

  // square
  bool square() noexcept { return this->row == this->column; }

  // singular
  bool singular() noexcept { return Rank() < row; }

  // Trace
  M trace() {
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

  // diagonal
  std::vector<M> diagonal(Diagonal type = Diagonal::Major) {

    if (row != column) {
      throw std::invalid_argument("Matrix diagonal requires row == column.");
    }

    const std::size_t n = row;
    const auto *base = data.data();

    std::vector<M> _diagonal(n);

    const auto *p = (type == Diagonal::Major) ? base : base + (n - 1);

    const std::size_t stride = (type == Diagonal::Major) ? (n + 1) : (n - 1);

    for (std::size_t i = 0; i < n; ++i, p += stride) {
      _diagonal[i] = *p;
    }

    return _diagonal;
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

  Matrix<M> subMatrix(std::size_t row_idx, std::size_t col_idx) {

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

  // minor
  M minor(std::size_t row_idx, std::size_t col_idx) {

    if (row != column) {
      throw std::invalid_argument("minor requires a square matrix.");
    }
    return subMatrix(row_idx, col_idx).determinant();
  }

  // cofactor
  M cofactor(std::size_t row_index, std::size_t column_index) {
    M sign = ((row_index + column_index) % 2 ? -M{1} : M{1});
    return sign * minor(row_index, column_index);
  }

  // cofactor matrix
  Matrix<M> cofactor_matrix() {
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

  // adjoint
  Matrix<M> adjoint() { return cofactor_matrix().Transpose(); }

  // determinant
  M determinant() {
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

  // integer power
  constexpr int integer_pow(int base, unsigned int exp) noexcept {
    int result = 1;

    while (exp > 0) {
      if (exp & 1) {
        result *= base;
      }
      exp >>= 1;
      if (exp) {
        base *= base;
      }
    }
    return result;
  }
};

// scalar multiplication
template <typename T, typename S>
// requires std::is_integral_v<T> || std::is_floating_point_v<T> ||
// std::is_integral_v<S> || std::is_floating_point_v<S>
Matrix<scalar_multiply_result_t<T, S>> operator*(S scalar, Matrix<T> &matrix) {

  using ResultType = scalar_multiply_result_t<T, S>;

  const auto *a = matrix.data.data();
  const std::size_t n = matrix.data.size();

  Matrix<ResultType> result(matrix.row, matrix.column);
  auto *out = result.data.data();

  for (std::size_t i{}; i < n; i++) {
    out[i] = static_cast<ResultType>(scalar * a[i]);
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

    const std::size_t m = LU.nrows();
    const std::size_t n = LU.ncols();
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

    const std::size_t m = LU.nrows();
    const std::size_t n = LU.ncols();
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

    const std::size_t m = LU.nrows();
    Matrix<M> P(m, m, M{});

    for (std::size_t i{}; i < m; i++) {
      P(i, info.permutation_vector[i]) = M{1};
    }
    return P;
  }
};

} // namespace Linea
#endif
