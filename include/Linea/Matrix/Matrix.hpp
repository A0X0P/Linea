
/**
 * @file Matrix.hpp
 * @author A.N. Prosper
 * @date December 18th 2025
 * @brief Generic dense matrix container with linear algebra and numerical
 * utilities.
 *
 * @details
 * This file defines the class template Linea::Matrix<M>, a row-major,
 * contiguous-storage dense matrix implementation constrained by
 * the NumericType concept.
 *
 * The matrix supports:
 * - Shape construction
 * - Type-safe casting
 * - Row and column extraction and mutation
 * - Structural permutation (row/column swaps)
 * - Vertical and horizontal concatenation
 * - Statistical reductions (sum, mean, min, max)
 * - Random matrix generation
 * - Identity, zero, and ones constructors
 * - Element-wise access
 * - Iterator support
 *
 * Mathematical Representation:
 *
 * For a matrix A ∈ 𝔽^{m×n}:
 *
 *      A = [ a_{ij} ]
 *
 * Storage layout is row-major:
 *
 *      index(i, j) = i · n + j
 *
 * so that:
 *
 *      a_{ij} ↦ data[i * column + j]
 *
 * Memory Model:
 * - Contiguous std::vector<M>
 * - Row-major layout
 * - Cache-friendly for row traversal
 * - raw() provides direct pointer access
 *
 * Type Requirements:
 * - M must satisfy NumericType
 * - Certain operations require:
 *      • RealType<M>
 *      • IntegralType<M>
 *
 * Exception Safety:
 * - Strong guarantee for constructors
 * - Bounds-checked accessors throw std::out_of_range
 * - Dimension mismatches throw std::invalid_argument
 *
 * Thread Safety:
 * - Not inherently thread-safe
 * - random() uses thread_local RNG
 *
 * Complexity Model:
 * - Construction: O(m·n)
 * - Row/column access: O(n) / O(m)
 * - Concatenation: O(m·n)
 * - Statistical reductions: O(m·n)
 *
 * Design Goals:
 * - Deterministic performance
 * - Zero hidden allocations beyond std::vector
 * - SIMD-friendly contiguous layout
 * - Clear mathematical semantics
 *
 * @note
 * This is a dense matrix implementation. Sparse optimizations are not included.
 *
 * @warning
 * The class does not enforce numerical stability in advanced operations
 * (e.g., determinant, inverse). Stability depends on algorithmic
 * implementations in associated utility headers.
 */

#ifndef LINEA_MATRIX_H
#define LINEA_MATRIX_H

#include "../Core/Concepts.hpp"
#include "../Core/PlatformMacros.hpp"
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

/**
 * @class Matrix
 * @brief Dense, row-major matrix over a numeric scalar field.
 *
 * @tparam M Scalar type satisfying NumericType
 *         (IntegralType ∪ RealType).
 *
 * @details
 *
 * Matrix<M> models a finite-dimensional rectangular array:
 *
 *      A ∈ 𝔽^{m×n}
 *
 * where:
 *
 *      𝔽 = ℤ  ∪  ℝ
 *
 * depending on the scalar type M.
 *
 * ---------------   Mathematical Model    ------------------
 *
 *
 * A matrix A is defined as:
 *
 *      A = [ a_{ij} ]
 *
 * with:
 *
 *      0 ≤ i < m
 *      0 ≤ j < n
 *
 * Element access obeys:
 *
 *      A(i, j) = a_{ij}
 *
 * Storage mapping:
 *
 *      a_{ij} ↦ data[i · n + j]
 *
 * ensuring a row-major contiguous layout.
 *
 * ---------------    Memory Representation    ------------------
 *
 * Internally:
 *
 *      std::vector<M> data;
 *
 * is used for storage.
 *
 * Properties:
 * - Contiguous memory
 * - No padding
 * - No hidden allocations beyond std::vector
 * - Stable address unless reallocated
 *
 * This layout provides:
 * - Cache locality for row traversal
 * - SIMD compatibility
 * - O(1) raw pointer access via raw()
 *
 *
 * ---------------    Dimensional Invariants    ------------------
 *
 * For all instances:
 *
 *      rows() ≥ 0
 *      cols() ≥ 0
 *      size() = rows() · cols()
 *
 * The following invariant always holds:
 *
 *      data.size() == rows() * cols()
 *
 * ---------------    Type Semantics    ------------------
 *
 *
 * M must satisfy:
 *
 *      NumericType<M>
 *
 * Additional operations may require:
 *
 *      RealType<M>     (e.g., statistical mean, sqrt, etc.)
 *      IntegralType<M> (e.g., integer exponentiation)
 *
 * No implicit narrowing conversions are performed.
 *
 * Casting between scalar types must be explicit via  the cast constructor
 * Matrix<M>(const Matrix<U> &matrix).
 *
 * --------------    Complexity Guarantees    ------------------
 *
 *
 * Construction:
 *      O(m · n)
 *
 * Element access:
 *      O(1)
 *
 * Row extraction:
 *      O(n)
 *
 * Column extraction:
 *      O(m)
 *
 * Concatenation:
 *      O(m · n)
 *
 * Statistical reductions:
 *      O(m · n)
 *
 *
 * Exception Safety
 *
 *
 * - Strong guarantee for constructors
 * - Bounds-checked accessors throw std::out_of_range
 * - Dimension mismatch throws std::invalid_argument
 *
 *
 * ---------------   Iterator Semantics    ------------------
 *
 * Iterators are contiguous and satisfy:
 *
 *      RandomAccessIterator
 *
 * Traversal order matches row-major memory order.
 *
 *  ---------------   Thread Safety    ------------------
 *
 *
 * - Distinct Matrix objects are thread-safe.
 * - random() uses thread_local RNG for isolation.
 *
 *
 *  ---------------   Algebraic Interpretation    ------------------
 *
 *
 * Matrix<M> models a finite-dimensional module over M.
 *
 * If M ∈ ℝ:
 *      Matrix<M> forms a vector space over ℝ.
 *
 * If M ∈ ℤ:
 *      Matrix<M> forms a free ℤ-module.
 *
 * This class does NOT enforce:
 * - Symmetry
 * - Orthogonality
 * - Triangular structure
 * - Positive definiteness
 *
 * Such properties are algorithm-level invariants.
 *
 * ---------------   Design Goals    ------------------
 *
 *
 * - Deterministic performance
 * - Mathematical transparency
 * - No hidden heap allocations
 * - Explicit domain control
 * - STL interoperability
 *
 *
 * @note
 * This is a dense matrix container.
 *
 * @warning
 * Numerical stability is NOT guaranteed for higher-level operations
 * such as determinant or inverse. Stability depends on algorithm
 * implementation in higher-level utilities.
 */

template <NumericType M> class Matrix {

private:
  // Attributes
  std::size_t row;
  std::size_t column;
  std::vector<M> data;

public:
  template <NumericType U> friend class Matrix;

  // Constructors

  /**
   * shape-only initialization
   * @brief Constructs a matrix with given dimensions.
   *
   * Allocates a matrix of size \f$ r \times c \f$ and initializes all elements
   * to zero.
   *
   * @param row Number of rows.
   * @param column Number of columns.
   *
   * @post All elements are value-initialized to \f$ M\{0\} \f$.
   *
   * @complexity O(r \cdot c)
   */

  Matrix(std::size_t row, std::size_t column)
      : row(row), column(column), data(row * column, M{}) {}

  /**
   * value-filled
   * @brief Constructs a matrix with given dimensions filled with a constant
   * value.
   *
   * Creates a matrix:
   * \f[
   * A_{ij} = value
   * \f]
   *
   * @param row Number of rows.
   * @param column Number of columns.
   * @param value Constant initialization value.
   *
   * @complexity O(r \cdot c)
   */
  Matrix(std::size_t row, std::size_t column, M value)
      : row(row), column(column), data(row * column, value) {}

  /**
   * data-only initialization
   * @brief Constructs a matrix from nested initializer lists.
   *
   * Example:
   * @code
   * Matrix<double> A{{1,2,3},{4,5,6}};
   * @endcode
   *
   * @throws std::invalid_argument If:
   * - The outer list is empty.
   * - Any row is empty.
   * - Rows have inconsistent sizes.
   *
   * @complexity O(r \cdot c)
   */
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

  /**
   * casting
   * @brief Explicit casting constructor between different numeric matrix types.
   *
   * Performs element-wise static_cast:
   * \f[
   * A_{ij}^{(M)} = \text{static_cast<M>}(B_{ij}^{(U)})
   * \f]
   *
   * @tparam U Source numeric type.
   * @param matrix Input matrix.
   *
   * @complexity O(r \cdot c)
   */
  template <NumericType U>
  explicit Matrix(const Matrix<U> &matrix)
      : row(matrix.row), column(matrix.column),
        data(matrix.row * matrix.column) {
    auto *RESTRICT out = this->raw();
    const auto *RESTRICT in = matrix.raw();
    for (std::size_t i = 0; i < data.size(); ++i) {
      out[i] = static_cast<M>(in[i]);
    }
  }

  // Dimension Accessors

  /**
   * @brief Returns number of rows.
   * @return Row count.
   */
  std::size_t nrows() const { return row; }
  /**
   * @brief Returns number of columns.
   * @return Column count.
   */
  std::size_t ncols() const { return column; }

  // Row / Column Access

  /**
   * @brief Returns a copy of a row vector.
   *
   * \f[
   * v_j = A_{row\_index,j}
   * \f]
   *
   * @param row_index Row index.
   * @throws std::out_of_range If index is invalid.
   *
   * @complexity O(c)
   */

  Vector<M> get_row(std::size_t row_index) const {
    if (row_index >= row) {
      throw std::out_of_range("Row index out of range");
    }

    Vector<M> result(column);
    const auto *RESTRICT in = raw() + row_index * column;
    auto *RESTRICT out = result.raw();
    std::copy(in, in + column, out);
    return result;
  }

  /**
   * @brief Returns a copy of a column vector.
   *
   * \f[
   * v_i = A_{i,column\_index}
   * \f]
   *
   * @param column_index Column index.
   * @throws std::out_of_range If index is invalid.
   *
   * @complexity O(r)
   */

  Vector<M> get_column(std::size_t column_index) const {
    if (column_index >= column) {
      throw std::out_of_range("Column index out of range");
    }
    Vector<M> result(row);

    auto *RESTRICT out = result.raw();
    const auto *RESTRICT in = this->raw();
    for (std::size_t j = 0; j < row; ++j) {
      out[j] = in[j * column + column_index];
    }

    return result;
  }

  /**
   * @brief Returns constant reference to underlying storage.
   *
   * Storage is contiguous in row-major order:
   * \f[
   * A_{ij} \mapsto data[i \cdot n_{cols} + j]
   * \f]
   *
   * @return const std::vector<M>&
   */
  const std::vector<M> &data_ref() const & { return data; }

  // Row / Column Modification

  /**
   * @brief Replaces a row with a vector.
   *
   * @param row_index Row to replace.
   * @param other Vector of matching column size.
   *
   * @throws std::out_of_range If index invalid.
   * @throws std::invalid_argument If dimension mismatch.
   *
   * @complexity O(c)
   */

  void set_row(std::size_t row_index, const Vector<M> &other) {
    if (row_index >= row) {
      throw std::out_of_range("Row index out of range");
    }
    if (other.size() != column) {
      throw std::invalid_argument("Row size mismatch");
    }

    auto *RESTRICT out = this->raw();
    const auto *RESTRICT in = other.raw();
    for (std::size_t i = 0; i < column; ++i) {
      out[row_index * column + i] = in[i];
    }
  }

  /**
   * @brief Replaces a column with a vector.
   *
   * @param column_index Column to replace.
   * @param other Vector of matching row size.
   *
   * @throws std::out_of_range If index invalid.
   * @throws std::invalid_argument If dimension mismatch.
   *
   * @complexity O(r)
   */

  void set_column(std::size_t column_index, const Vector<M> &other) {
    if (column_index >= column) {
      throw std::out_of_range("Column index out of range");
    }
    if (other.size() != row) {
      throw std::invalid_argument("Column size mismatch");
    }

    auto *RESTRICT out = this->raw();
    const auto *RESTRICT in = other.raw();
    for (std::size_t i = 0; i < row; ++i) {
      out[i * column + column_index] = in[i];
    }
  }

  // Permutation operations

  /**
   * @brief Swaps two rows.
   *
   * @param row1 First row index.
   * @param row2 Second row index.
   *
   * @throws std::out_of_range If index invalid.
   *
   * @complexity O(c)
   */

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

  /**
   * @brief Swaps two columns.
   *
   * @param column1 First column index.
   * @param column2 Second column index.
   *
   * @throws std::out_of_range If index invalid.
   *
   * @complexity O(r)
   */

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

  /**
   * @brief Vertical concatenation.
   *
   * If:
   * \f$ A \in \mathbb{R}^{m \times n} \f$
   * and
   * \f$ B \in \mathbb{R}^{k \times n} \f$
   *
   * then:
   * \f[
   * C =
   * \begin{bmatrix}
   * A \\
   * B
   * \end{bmatrix}
   * \in \mathbb{R}^{(m+k)\times n}
   * \f]
   *
   * @throws std::invalid_argument If column mismatch.
   *
   * @complexity O((m+k)n)
   */

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

  /**
   * @brief Horizontal concatenation.
   *
   * If:
   * \f$ A \in \mathbb{R}^{m \times n} \f$
   * and
   * \f$ B \in \mathbb{R}^{m \times k} \f$
   *
   * then:
   * \f[
   * C =
   * \begin{bmatrix}
   * A & B
   * \end{bmatrix}
   * \in \mathbb{R}^{m \times (n+k)}
   * \f]
   *
   * @throws std::invalid_argument If row mismatch.
   *
   * @complexity O(m(n+k))
   */

  Matrix<M> hstack(const Matrix<M> &other) const {
    if (row != other.row) {
      throw std::invalid_argument("hstack requires A.rows == B.rows");
    }

    Matrix<M> result(row, column + other.column);

    for (std::size_t i = 0; i < row; ++i) {
      const M *RESTRICT row_a = data.data() + i * column;
      const M *RESTRICT row_b = other.data.data() + i * other.column;
      M *RESTRICT row_out = result.data.data() + i * result.column;

      std::copy(row_a, row_a + column, row_out);
      std::copy(row_b, row_b + other.column, row_out + column);
    }
    return result;
  }

  // Static Constructors
  /**
   * @brief Constructs identity matrix.
   *
   * \f[
   * I_{ij} =
   * \begin{cases}
   * 1 & i=j \\
   * 0 & i \neq j
   * \end{cases}
   * \f]
   *
   * @param row_column Dimension.
   *
   * @return Identity matrix.
   */

  static Matrix<M> identity(std::size_t row_column) {
    Matrix<M> I(row_column, row_column);
    auto *RESTRICT out = I.raw();
    for (std::size_t i = 0; i < row_column; ++i) {
      out[i * row_column + i] = M{1};
    }

    return I;
  }

  /**
   * @brief Constructs zero matrix.
   */
  static Matrix<M> zeros(std::size_t row, std::size_t column) {
    return Matrix<M>(row, column, M{0});
  }

  /**
   * @brief Constructs matrix filled with ones.
   */

  static Matrix<M> ones(std::size_t row, std::size_t column) {
    return Matrix<M>(row, column, M{1});
  }

  /**
   * @brief Constructs matrix with random values in [min_range, max_range].
   *
   * - Uses uniform_int_distribution for integral types.
   * - Uses uniform_real_distribution for floating types.
   *
   * @param row Rows.
   * @param column Columns.
   * @param min_range Lower bound.
   * @param max_range Upper bound.
   *
   * @complexity O(r \cdot c)
   */

  static Matrix<M> random(std::size_t row, std::size_t column, M min_range,
                          M max_range) {
    auto low = std::min(min_range, max_range);
    auto high = std::max(min_range, max_range);

    Matrix<M> result(row, column);
    auto *RESTRICT out = result.raw();
    static thread_local std::mt19937 engine(std::random_device{}());

    if constexpr (std::is_integral_v<M>) {
      std::uniform_int_distribution<M> distribute(low, high);

      for (std::size_t i{}; i < row; ++i) {
        for (std::size_t j{}; j < column; ++j) {
          out[i * column + j] = distribute(engine);
        }
      }

    } else if constexpr (std::is_floating_point_v<M>) {
      std::uniform_real_distribution<M> distribute(low, high);
      for (std::size_t i{}; i < row; ++i) {
        for (std::size_t j{}; j < column; ++j) {
          out[i * column + j] = distribute(engine);
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

  Matrix<M> hadamard_product(const Matrix<M> &other) const;
  template <NumericType T>
  Matrix<Numeric<T, M>> hadamard_product(const Matrix<T> &other) const;

  Matrix<M> operator/(const Matrix<M> &other) const;
  template <NumericType T>
  Matrix<Numeric<T, M>> operator/(const Matrix<T> &other) const;

  // Element Access

  /**
   * @brief Element access (mutable).
   *
   * Accesses element:
   * \f[
   * A_{ij}
   * \f]
   *
   * @param i Row index.
   * @param j Column index.
   *
   * @return Reference to element.
   */

  M &operator()(std::size_t i, std::size_t j);
  /**
   * @brief Element access (const).
   *
   * @param i Row index.
   * @param j Column index.
   *
   * @return Const reference to element.
   */
  const M &operator()(std::size_t i, std::size_t j) const;

  // Raw Pointer Access

  /**
   * @brief Returns mutable raw pointer to contiguous storage.
   *
   * @return Pointer to first element.
   */

  [[nodiscard]] inline M *raw() noexcept { return data.data(); }
  /**
   * @brief Returns const raw pointer to contiguous storage.
   *
   * @return Pointer to first element.
   */
  [[nodiscard]] inline const M *raw() const noexcept { return data.data(); }

public:
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

  /**
   * @brief Computes sum of all elements.
   *
   * \f[
   * \sum_{i,j} A_{ij}
   * \f]
   *
   * @return Accumulated sum.
   */

  template <IntegralType I = M> std::common_type_t<I, long long> sum() const {
    return std::accumulate(data.begin(), data.end(),
                           std::common_type_t<I, long long>{0});
  }

  template <RealType R = M> R sum() const {
    return std::accumulate(data.begin(), data.end(), R{0});
  }

  /**
   * @brief Computes arithmetic mean.
   *
   * \f[
   * \mu = \frac{1}{N} \sum_{i,j} A_{ij}
   * \f]
   *
   * @throws std::domain_error If matrix empty.
   */

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

  /**
   * @brief Returns minimum element.
   *
   * @throws std::runtime_error If matrix empty.
   */
  M min() const {
    if (data.empty()) {
      throw std::runtime_error("Cannot compute maximum of empty matrix");
    }
    return *std::min_element(data.begin(), data.end());
  }

  /**
   * @brief Returns maximum element.
   *
   * @throws std::runtime_error If matrix empty.
   */
  M max() const {
    if (data.empty()) {
      throw std::runtime_error("Cannot compute maximum of empty matrix");
    }
    return *std::max_element(data.begin(), data.end());
  }

  // Utility methods - declarations (defined in MatrixUtilities.hpp)
  void shape() const;
  std::size_t size() const;
  bool empty() const noexcept;
  bool symmetric() const noexcept;
  bool square() const noexcept;
  bool singular() const noexcept;

  M trace() const;
  Vector<M> diagonal(Diagonal type = Diagonal::Major) const;
  Matrix<M> transpose() const;
  std::size_t rank() const
    requires RealType<M>;
  Matrix<M> reshape(std::size_t nrow, std::size_t ncol) const;
  Matrix<M> flatten() const;
  Matrix<M> subMatrix(std::size_t row_idx, std::size_t col_idx) const;
  Matrix<M> block(std::size_t row, std::size_t col, std::size_t nrows,
                  std::size_t ncols) const;

  M minor(std::size_t row_idx, std::size_t col_idx) const;
  M cofactor(std::size_t row_idx, std::size_t col_idx) const;
  Matrix<M> cofactor_matrix() const;
  Matrix<M> adjoint() const;

  M determinant() const
    requires RealType<M>;
  Matrix<M> inverse() const
    requires RealType<M>;

  Vector<M> solve(const Vector<M> &b) const
    requires RealType<M>;
  Matrix<M> solve(const Matrix<M> &B) const
    requires RealType<M>;

  // Norms
  double norm() const;
  double norm(MatrixNorm type, std::size_t max_iter = 100) const;

private:
  double norm_1() const;
  double norm_infinity() const;
  double norm_spectral(std::size_t max_iter) const;

  Vector<M> forward_substitution(const Matrix<M> &L, const Vector<M> &b,
                                 const Vector<std::size_t> &piv) const
    requires RealType<M>;
  Vector<M> backward_substitution(const Matrix<M> &U, const Vector<M> &y) const
    requires RealType<M>;

  // Free functions declarations

  template <RealType N> Matrix<N> sin(const Matrix<N> &);
  template <RealType N> Matrix<N> cos(const Matrix<N> &);
  template <DomainCheck Check = DomainCheck::Enable, RealType N>
  Matrix<N> tan(const Matrix<N> &);
  template <DomainCheck Check = DomainCheck::Enable, RealType N>
  Matrix<N> sqrt(const Matrix<N> &);
  template <DomainCheck Check = DomainCheck::Enable, RealType N>
  Matrix<N> log(const Matrix<N> &);
  template <RealType N> Matrix<N> exp(const Matrix<N> &);
  template <RealType U> Matrix<U> pow(const Matrix<U> &matrix, U exponent);
  template <IntegralType U> Matrix<U> pow(const Matrix<U> &, int exponent);
};

} // namespace Linea

#endif // LINEA_MATRIX_H
