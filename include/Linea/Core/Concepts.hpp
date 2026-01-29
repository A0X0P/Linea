// created by : A.N. Prosper
// date : january 25th 2025
// time : 13:21

#ifndef LINEA_CONCEPTS_H
#define LINEA_CONCEPTS_H

#include <type_traits>

namespace Linea {

template <typename T>
concept RealType = std::is_floating_point_v<T>;

template <typename T>
concept IntegralType =
    std::is_integral_v<T> && !std::is_same_v<T, bool> &&
    !std::is_same_v<T, char> && !std::is_same_v<T, signed char> &&
    !std::is_same_v<T, unsigned char> && !std::is_same_v<T, wchar_t> &&
    !std::is_same_v<T, char8_t> && !std::is_same_v<T, char16_t> &&
    !std::is_same_v<T, char32_t> && (sizeof(T) >= sizeof(int));

template <typename T>
concept NumericType = IntegralType<T> || RealType<T>;

} // namespace Linea

#endif // LINEA_CONCEPTS_H
