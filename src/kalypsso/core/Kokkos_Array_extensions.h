// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file Kokkos_Array_extensions.h
 * Define some extensions to class Kokkos::Array.
 */
#ifndef KALYPSSO_CORE_KOKKOSARRAYEXTENSIONS_H_
#define KALYPSSO_CORE_KOKKOSARRAYEXTENSIONS_H_

#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp> // for KOKKOS_ENABLE_XXX

#include <type_traits> // for std::is_integral
#include <array>

namespace Kokkos
{

// ================================================================
// ================================================================
/**
 * element-wise addition operator for Kokkos::array
 */
template <class T, std::size_t N>
KOKKOS_FUNCTION constexpr Array<T, N>
operator+(const Array<T, N> & a1, const Array<T, N> & a2) noexcept
{
  Array<T, N> a3;
  for (size_t i = 0; i < N; ++i)
    a3[i] = a1[i] + a2[i];
  return a3;
}

// ================================================================
// ================================================================
/**
 * element-wise subtraction operator for Kokkos::array
 */
template <class T, std::size_t N>
KOKKOS_FUNCTION constexpr Array<T, N>
operator-(const Array<T, N> & a1, const Array<T, N> & a2) noexcept
{
  Array<T, N> a3;
  for (size_t i = 0; i < N; ++i)
    a3[i] = a1[i] - a2[i];
  return a3;
}

// ================================================================
// ================================================================
/**
 * element-wise multiplication operator for Kokkos::array
 */
template <class T, size_t N>
KOKKOS_INLINE_FUNCTION constexpr Array<T, N>
operator*(const Array<T, N> & a1, const Array<T, N> & a2)
{
  Kokkos::Array<T, N> a3;
  for (size_t i = 0; i < N; ++i)
    a3[i] = a1[i] * a2[i];
  return a3;
}

// ================================================================
// ================================================================
/**
 * element-wise division operator for Kokkos::array
 */
template <class T, size_t N>
KOKKOS_INLINE_FUNCTION constexpr Array<T, N>
operator/(const Array<T, N> & a1, const Array<T, N> & a2)
{
  Kokkos::Array<T, N> a3;
  for (size_t i = 0; i < N; ++i)
    a3[i] = a1[i] / a2[i];
  return a3;
}

// ================================================================
// ================================================================
/**
 * element-wise addition by a scalar on the left for Kokkos::array
 */
template <class T, class T2, size_t N>
KOKKOS_INLINE_FUNCTION constexpr Array<T, N>
operator+(const T2 & scalar, const Array<T, N> & a)
{
  Kokkos::Array<T, N> a2;
  for (size_t i = 0; i < N; ++i)
    a2[i] = a[i] + static_cast<T>(scalar);
  return a2;
}

// ================================================================
// ================================================================
/**
 * element-wise addition by a scalar on the right for Kokkos::array
 */
template <class T, class T2, size_t N>
KOKKOS_INLINE_FUNCTION constexpr Array<T, N>
operator+(const Array<T, N> & a, const T2 & scalar)
{
  Kokkos::Array<T, N> a2;
  for (size_t i = 0; i < N; ++i)
    a2[i] = a[i] + static_cast<T>(scalar);
  return a2;
}

// =================================================================
// =================================================================
template <class T, size_t N>
KOKKOS_INLINE_FUNCTION constexpr Array<T, N> &
operator+=(Array<T, N> & lhs, const Array<T, N> & rhs)
{

  for (size_t i = 0; i < N; ++i)
    lhs[i] += rhs[i];

  return lhs;

} // operator+=

// =================================================================
// =================================================================
template <class T, size_t N>
KOKKOS_INLINE_FUNCTION Array<T, N> &
                       operator-=(Array<T, N> & lhs, const Array<T, N> & rhs)
{

  for (size_t i = 0; i < N; ++i)
    lhs[i] -= rhs[i];

  return lhs;

} // operator-=

// ================================================================
// ================================================================
/**
 * element-wise subtraction by a scalar on the left for Kokkos::array
 */
template <class T, class T2, size_t N>
KOKKOS_INLINE_FUNCTION constexpr Array<T, N>
operator-(const T2 & scalar, const Array<T, N> & a)
{
  Kokkos::Array<T, N> a2;
  for (size_t i = 0; i < N; ++i)
    a2[i] = static_cast<T>(scalar) - a[i];
  return a2;
}

// ================================================================
// ================================================================
/**
 * element-wise subtraction by a scalar on the right for Kokkos::array
 */
template <class T, class T2, size_t N>
KOKKOS_INLINE_FUNCTION constexpr Array<T, N>
operator-(const Array<T, N> & a, const T2 & scalar)
{
  Kokkos::Array<T, N> a2;
  for (size_t i = 0; i < N; ++i)
    a2[i] = a[i] - static_cast<T>(scalar);
  return a2;
}

// ================================================================
// ================================================================
/**
 * element-wise multiplication by a scalar on the left for Kokkos::array
 */
// template <class T, class T2, size_t N>
// KOKKOS_INLINE_FUNCTION constexpr Array<T, N>
// operator*(const T2 & scalar, const Array<T, N> & a)
// {
//   Kokkos::Array<T, N> a2;
//   for (size_t i = 0; i < N; ++i)
//     a2[i] = a[i] * static_cast<T>(scalar);
//   return a2;
// }

// ================================================================
// ================================================================
/**
 * element-wise multiplication by a scalar on the left for Kokkos::array
 */
template <class T, class T2, size_t N>
KOKKOS_INLINE_FUNCTION constexpr Array<T, N>
operator*(const T2 scalar, const Array<T, N> & a)
{
  Kokkos::Array<T, N> a2;
  for (size_t i = 0; i < N; ++i)
    a2[i] = a[i] * static_cast<T>(scalar);
  return a2;
}

// ================================================================
// ================================================================
/**
 * element-wise multiplication by a scalar on the right for Kokkos::array
 */
template <class T, class T2, size_t N>
KOKKOS_INLINE_FUNCTION constexpr Array<T, N>
operator*(const Array<T, N> & a, const T2 & scalar)
{
  Kokkos::Array<T, N> a2;
  for (size_t i = 0; i < N; ++i)
    a2[i] = a[i] * static_cast<T>(scalar);
  return a2;
}

// =================================================================
// =================================================================
template <class T, class T2, size_t N>
KOKKOS_INLINE_FUNCTION constexpr Array<T, N> &
operator*=(Array<T, N> & lhs, const T2 & scalar)
{

  for (size_t i = 0; i < N; ++i)
    lhs[i] *= static_cast<T>(scalar);

  return lhs;

} // operator*=

// ================================================================
// ================================================================
/**
 * element-wise division by a scalar on the right for Kokkos::array
 */
template <class T, class T2, size_t N>
KOKKOS_INLINE_FUNCTION constexpr Array<T, N>
operator/(const Array<T, N> & a, const T2 & scalar)
{
  Kokkos::Array<T, N> a2;
  for (size_t i = 0; i < N; ++i)
    a2[i] = a[i] / static_cast<T>(scalar);
  return a2;
}

// ================================================================
// ================================================================
/**
 * Compute the product of all items in Kokkos::Array
 */
template <typename T, size_t N>
KOKKOS_INLINE_FUNCTION T
dim_prod(const Array<T, N> & a)
{
  T res = a[0];
  for (size_t i = 1; i < N; ++i)
    res *= a[i];
  return res;
}

// ================================================================
// ================================================================
/**
 * Check two Kokkos::Array are equal (integral type only).
 *
 * \return true if two input array are equal element-wise
 */
template <typename T, size_t N>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_integral_v<T>, bool>
                       operator==(const Array<T, N> & a1, const Array<T, N> & a2)
{
  if constexpr (N == 0)
    return true;
  else
  {
    bool res = a1[0] == a2[0];
    for (size_t i = 1; i < N; ++i)
    {
      res = res and (a1[i] == a2[i]);
    }
    return res;
  }
}

} // namespace Kokkos

namespace kalypsso
{
// ===========================================================
// ===========================================================
//! Convert a Kokkos::Array into a std::array
//!
//! \note this could be refactored using std::to_array, but currently on available in c++20
//! \todo refactor when kalypsso will use c++20
template <typename T, size_t dim>
std::array<T, dim>
to_std_array(Kokkos::Array<T, dim> array)
{
  std::array<T, dim> res;
  res[0] = array[0];
  res[1] = array[1];
  if constexpr (dim == 3)
    res[2] = array[2];

  return res;
}

template <class T, size_t N>
KOKKOS_INLINE_FUNCTION constexpr Kokkos::Array<T, N>
init_kokkos_array(const T scalar)
{
  Kokkos::Array<T, N> res;
  for (size_t i = 0; i < N; ++i)
  {
    res[i] = scalar;
  }
  return res;
}

} // namespace kalypsso

#endif // KALYPSSO_CORE_KOKKOSARRAYEXTENSIONS_H_
