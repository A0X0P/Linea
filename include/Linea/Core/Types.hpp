// created by : A.N. Prosper
// date : january 25th 2025
// time : 13:21

#ifndef LINEA_TYPES_H
#define LINEA_TYPES_H

#include "Concepts.hpp"
#include <type_traits>

namespace Linea {

template <NumericType T, NumericType S>
using Numeric = std::common_type_t<T, S>;

enum class NormType { Frobenius, One, Infinity, Spectral };
enum class MatrixNorm { Frobenius, One, Infinity, Spectral };
enum class VectorNorm { L1, L2, LInf };
enum class Diagonal { Major, Minor };
enum class ComputeMode { Thin, Full };

} // namespace Linea
#endif // LINEA_TYPES_H
