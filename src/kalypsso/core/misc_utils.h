// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file misc_utils.h
 */
#ifndef KALYPSSO_CORE_MISC_UTILS_H
#define KALYPSSO_CORE_MISC_UTILS_H

#include <math.h>
#include <string> // string
#include <kalypsso/core/real_type.h>

namespace kalypsso
{

KALYPSSO_MATH_CONSTANT(FUZZY_THRESHOLD, 1e-12);

template <typename T>
KOKKOS_INLINE_FUNCTION T
FUZZY_THRESHOLD()
{
  if constexpr (std::is_same<T, float>::value)
  {
    return 1e-12f;
  }
  else if constexpr (std::is_same<T, double>::value)
  {
    return 1e-12;
  }
}


template <typename T>
KOKKOS_INLINE_FUNCTION bool
ISFUZZYNULL(T a)
{
  static_assert(std::is_floating_point_v<T>, "Type T must be a floating-point type");

  return (fabs(a) < FUZZY_THRESHOLD<T>());
}

template <typename T>
KOKKOS_INLINE_FUNCTION bool
FUZZYCOMPARE(T a, T b)
{
  return ((ISFUZZYNULL(a) && ISFUZZYNULL(b)) ||
          (fabs((a) - (b)) * 1000000000000. <= fmin(fabs(a), fabs(b))));
}

template <typename T>
KOKKOS_INLINE_FUNCTION bool
FUZZYLIMITS(T x, T a, T b)
{
  return (((x) > ((a)-FUZZY_THRESHOLD<T>())) && ((x) < ((b) + FUZZY_THRESHOLD<T>())));
}

/**
 * Print current date.
 */
void
print_current_date(std::ostream & stream);

/**
 * Get current data as a string.
 */
std::string
get_current_date();

} // namespace kalypsso

#endif // KALYPSSO_CORE_MISC_UTILS_H
