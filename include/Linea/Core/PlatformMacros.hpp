// created by : A.N. Prosper
// date : febuary 3rd 2026
// time : 11:55

#ifndef PLATFORM_MACROS_H
#define PLATFORM_MACROS_H

#if defined(__GNUC__) || defined(__clang__)
#define RESTRICT __restrict__
#elif defined(_MSC_VER)
#define RESTRICT __restrict
#else
#define RESTRICT
#endif

#endif // PLATFORM_MACROS_H
