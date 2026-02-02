// created by : A.N. Prosper
// date : December 18th 2025
// time : 20:20

#ifndef LINEA_MATRIX_H
#define LINEA_MATRIX_H

#include "../Core/Concepts.hpp"
#include "../Core/Types.hpp"
#include "../Core/Utilities.hpp"
#include "../Vector/Vector.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace Linea {

template <NumericType M> struct LUResult {
  Vector<M> permutation_vector;
  std::size_t rank;
  std::size_t swap_count;
};

template <NumericType M> class LUFactor;

template <NumericType M> class Matrix {

private:
  // Attributes
  std::size_t row;
  std::size_t column;
  std::vector<M> data;

public:
  template <NumericType U> friend class Matrix;

  // Constructors

  // shape-only initialization
  Matrix(std::size_t row, std::size_t column)
      : row(row), column(column), data(row * column, M{}) {}

  // value-filled
  Matrix(std::size_t row, std::size_t column, M value)
      : row(row), column(column), data(row * column, value) {}

  // data-only initialization
  explicit Matrix(std::initializer_list<std::initializer_list<M>> list)
      : row(list.size()), column(0) {
    if (row == 0) {
      throw std::invalid_argument("Matrix initializer_list cannot be empty");
    }

    column = list.begin()->size();
    if (column == 0) {
      throw std::invalid_argument(
          "Matrix initializer_list rows cannot be empty");
    }

    data.reserve(row * column);

    for (const auto &r : list) {
      if (r.size() != column) {
        throw std::invalid_argument(
            "All rows in Matrix initializer_list must have the same size");
      }
      data.insert(data.end(), r.begin(), r.end());
    }
  }

  // Destructor
  ~Matrix() {}

  // Getters
  std::size_t nrows() const { return row; }
  std::size_t ncols() const { return column; }

  Vector<M> getRow(std::size_t row_index) const {
    if (row_index >= row) {
      throw std::out_of_range("Row index out of range");
    }

    Vector<M> result(column);
    std::copy(data.begin() + row_index * column,
              data.begin() + (row_index + 1) * column, result.begin());
    return result;
  }

  Vector<M> getColumn(std::size_t column_index) const {
    if (column_index >= column) {
      throw std::out_of_range("Column index out of range");
    }
    Vector<M> result(row);

    for (std::size_t j = 0; j < row; ++j) {
      result[j] = data.at(j * column + column_index);
    }

    return result;
  }

  const std::vector<M> &getdata() const & { return data; }

  void setRow(std::size_t row_index, const Vector<M> &other) {
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

  void setColumn(std::size_t column_index, const Vector<M> &other) {
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

  // Concatenation
  Matrix<M> vstack(const Matrix<M> &other) const {
    if (column != other.column) {
      throw std::invalid_argument("vstack requires A.columns == B.columns");
    }

    Matrix<M> result(row + other.row, column);

    const std::size_t a_size = data.size();
    const std::size_t b_size = other.data.size();

    std::copy(data.data(), data.data() + a_size, result.data.data());

    std::copy(other.data.data(), other.data.data() + b_size,
              result.data.data() + a_size);

    return result;
  }

  Matrix<M> hstack(const Matrix<M> &other) const {
    if (row != other.row) {
      throw std::invalid_argument("hstack requires A.rows == B.rows");
    }

    Matrix<M> result(row, column + other.column);

    for (std::size_t i = 0; i < row; ++i) {
      const M *row_a = data.data() + i * column;
      const M *row_b = other.data.data() + i * other.column;
      M *row_out = result.data.data() + i * result.column;

      std::copy(row_a, row_a + column, row_out);
      std::copy(row_b, row_b + other.column, row_out + column);
    }
    return result;
  }

  // Static Methods
  static Matrix<M> Identity(std::size_t row_column) {
    Matrix<M> I(row_column, row_column);

    for (std::size_t i = 0; i < row_column; ++i) {
      I.data[i * row_column + i] = M{1};
    }

    return I;
  }

  static Matrix<M> Zeros(std::size_t row, std::size_t column) {
    return Matrix<M>(row, column, M{0});
  }

  static Matrix<M> Ones(std::size_t row, std::size_t column) {
    return Matrix<M>(row, column, M{1});
  }

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

  // Operators - declarations (implementations in MatrixOperations.hpp)
  bool operator==(const Matrix<M> &other) const noexcept;
  bool operator!=(const Matrix<M> &other) const noexcept;

  Matrix<M> operator+(const Matrix<M> &other) const;
  template <NumericType T>
  Matrix<Numeric<M, T>> operator+(const Matrix<T> &other) const;

  Matrix<M> operator-(const Matrix<M> &other) const;
  template <NumericType T>
  Matrix<Numeric<M, T>> operator-(const Matrix<T> &other) const;

  Matrix<M> operator-() const;

  template <NumericType S> Matrix<Numeric<M, S>> operator*(S scalar) const;

  Matrix<M> operator*(const Matrix<M> &other) const;
  template <NumericType T>
  Matrix<Numeric<M, T>> operator*(const Matrix<T> &other) const;

  Matrix<M> Hadamard_product(const Matrix<M> &other) const;
  template <NumericType T>
  Matrix<Numeric<T, M>> Hadamard_product(const Matrix<T> &other) const;

  Matrix<M> operator/(const Matrix<M> &other) const;
  template <NumericType T>
  Matrix<Numeric<T, M>> operator/(const Matrix<T> &other) const;

  M &operator()(std::size_t i, std::size_t j);
  const M &operator()(std::size_t i, std::size_t j) const;

  // Iterators
  using iterator = typename std::vector<M>::iterator;
  using reverse_iterator = typename std::vector<M>::reverse_iterator;
  using const_iterator = typename std::vector<M>::const_iterator;
  using const_reverse_iterator =
      typename std::vector<M>::const_reverse_iterator;

  iterator begin() { return data.begin(); }
  reverse_iterator rbegin() { return data.rbegin(); }
  const_iterator begin() const { return data.begin(); }
  const_reverse_iterator rbegin() const { return data.rbegin(); }
  const_iterator cbegin() const { return data.cbegin(); }
  const_reverse_iterator crbegin() const { return data.crbegin(); }

  iterator end() { return data.end(); }
  reverse_iterator rend() { return data.rend(); }
  const_iterator end() const { return data.end(); }
  const_reverse_iterator rend() const { return data.rend(); }
  const_iterator cend() const { return data.cend(); }
  const_reverse_iterator crend() const { return data.crend(); }

  // Statistical Operations
  template <IntegralType I = M> std::common_type_t<I, long long> sum() const {
    return std::accumulate(data.begin(), data.end(),
                           std::common_type_t<I, long long>{0});
  }

  template <RealType R = M> R sum() const {
    return std::accumulate(data.begin(), data.end(), R{0});
  }

  template <IntegralType I = M> std::common_type_t<I, double> mean() const {
    if (data.empty())
      throw std::domain_error("Cannot compute mean of empty matrix");
    return static_cast<std::common_type_t<I, double>>(sum()) / data.size();
  }

  template <RealType R = M> R mean() const {
    if (data.empty())
      throw std::domain_error("Cannot compute mean of empty matrix");
    return sum() / static_cast<R>(data.size());
  }

  M min() const {
    if (data.empty()) {
      throw std::runtime_error("Cannot compute maximum of empty matrix");
    }
    return *std::min_element(data.begin(), data.end());
  }

  M max() const {
    if (data.empty()) {
      throw std::runtime_error("Cannot compute maximum of empty matrix");
    }
    return *std::max_element(data.begin(), data.end());
  }

  // Utility methods - declarations (defined in MatrixUtilities.hpp)
  void insert(std::size_t row_index, std::size_t col_index, M number);
  void shape() const;
  std::size_t size() const;
  bool empty() const noexcept;
  bool symmetric() const noexcept;
  bool square() const noexcept;
  bool singular() const noexcept;

  M trace() const;
  Vector<M> diagonal(Diagonal type = Diagonal::Major) const;
  Matrix<M> Transpose() const;
  std::size_t Rank() const;
  Matrix<M> Reshape(std::size_t nrow, std::size_t ncol) const;
  Matrix<M> flatten() const;
  Matrix<M> subMatrix(std::size_t row_idx, std::size_t col_idx) const;
  Matrix<M> block(std::size_t row, std::size_t col, std::size_t nrows,
                  std::size_t ncols) const;

  M minor(std::size_t row_idx, std::size_t col_idx) const;
  M cofactor(std::size_t row_index, std::size_t column_index) const;
  Matrix<M> cofactor_matrix() const;
  Matrix<M> adjoint() const;
  M determinant() const;
  Matrix<M> Inverse() const;

  Vector<M> solve(const Vector<M> &b) const;
  Matrix<M> solve(const Matrix<M> &B) const;

  LUFactor<M> lu_decompose(M epsilon = M(0)) const;
  Matrix<M> cholesky() const;

  // Norms
  double norm() const;
  double norm(NormType type) const;

private:
  double norm_1() const;
  double norm_infinity() const;
  double norm_spectral() const;

  Vector<M> forward_substitution(const Matrix<M> &L, const Vector<M> &b,
                                 const Vector<M> &piv) const;
  Vector<M> backward_substitution(const Matrix<M> &U, const Vector<M> &y) const;

  // Friend declarations

  template <RealType N> friend Matrix<N> sin(const Matrix<N> &);
  template <RealType N> friend Matrix<N> cos(const Matrix<N> &);
  template <RealType N> friend Matrix<N> tan(const Matrix<N> &);
  template <RealType N> friend Matrix<N> sqrt(const Matrix<N> &);
  template <RealType N> friend Matrix<N> log(const Matrix<N> &);
  template <RealType N> friend Matrix<N> exp(const Matrix<N> &);
  template <RealType U>
  friend Matrix<U> pow(const Matrix<U> &matrix, U exponent);
  template <IntegralType U>
  friend Matrix<U> pow(const Matrix<U> &, int exponent);
};

} // namespace Linea

#endif // LINEA_MATRIX_H
