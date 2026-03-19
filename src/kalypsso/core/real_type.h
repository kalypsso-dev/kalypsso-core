// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file real_type.h
 * \brief Define macros to switch single/double precision.
 *
 */
#ifndef KALYPSSO_CORE_REAL_TYPE_H_
#define KALYPSSO_CORE_REAL_TYPE_H_

#include <math.h>

#include <kalypsso/core/kalypsso_core_config.h>
#include <Kokkos_Macros.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <Kokkos_Clamp.hpp>
#include <kalypsso/core/kalypsso_macros.h>

KALYPSSO_DISABLE_NVCC_WARNINGS_PUSH()

namespace kalypsso
{

/**
 * \typedef real_t (alias to float or double)
 */
#ifdef KALYPSSO_CORE_USE_DOUBLE
using real_t = double;
#else
using real_t = float;
#endif // KALYPSSO_CORE_USE_DOUBLE

/**
 * Helper macro to define portable numeric values.
 */
#ifdef KALYPSSO_CORE_USE_DOUBLE
#  define KALYPSSO_NUM(x) x
#else
#  define KALYPSSO_NUM(x) x##f
#endif // KALYPSSO_DOUBLE

// math function
#if KOKKOS_VERSION_MAJOR > 3
using Kokkos::clamp;
using Kokkos::exp;
using Kokkos::fmax;
using Kokkos::fmin;
using Kokkos::sqrt;
using Kokkos::fabs;
using Kokkos::fmod;
using Kokkos::isnan;
using Kokkos::fmod;
using Kokkos::exp;
using Kokkos::log;
using Kokkos::pow;
// trigonometric functions
using Kokkos::sin;
using Kokkos::cos;
using Kokkos::tan;
using Kokkos::asin;
using Kokkos::acos;
using Kokkos::atan;
using Kokkos::atan2;
// hyperbolic functions
using Kokkos::sinh;
using Kokkos::cosh;
using Kokkos::tanh;
using Kokkos::asinh;
using Kokkos::acosh;
using Kokkos::atanh;
#else
using Kokkos::Experimental::clamp;
using Kokkos::Experimental::exp;
using Kokkos::Experimental::fmax;
using Kokkos::Experimental::fmin;
using Kokkos::Experimental::sqrt;
using Kokkos::Experimental::fabs;
using Kokkos::Experimental::fmod;
using Kokkos::Experimental::isnan;
using Kokkos::Experimental::fmod;
using Kokkos::Experimental::exp;
using Kokkos::Experimental::log;
using Kokkos::Experimental::pow;
// trigonometric functions
using Kokkos::Experimental::sin;
using Kokkos::Experimental::cos;
using Kokkos::Experimental::tan;
using Kokkos::Experimental::asin;
using Kokkos::Experimental::acos;
using Kokkos::Experimental::atan;
using Kokkos::Experimental::atan2;
// hyperbolic functions
using Kokkos::Experimental::sinh;
using Kokkos::Experimental::cosh;
using Kokkos::Experimental::tanh;
using Kokkos::Experimental::asinh;
using Kokkos::Experimental::acosh;
using Kokkos::Experimental::atanh;
#endif

// this macro is slightly adapted from Kokkos
#define KALYPSSO_STATIC_MATH_CONSTANT(TRAIT, VALUE)                                          \
  template <class T>                                                                         \
  static constexpr auto TRAIT##_v = std::enable_if_t<std::is_floating_point_v<T>, T>(VALUE); \
  static constexpr auto TRAIT##_F = TRAIT##_v<real_t>

#define KALYPSSO_MATH_CONSTANT(TRAIT, VALUE)                                                 \
  template <class T>                                                                         \
  inline constexpr auto TRAIT##_v = std::enable_if_t<std::is_floating_point_v<T>, T>(VALUE); \
  inline constexpr auto TRAIT##_F = TRAIT##_v<real_t>

// clang-format off
KALYPSSO_MATH_CONSTANT(ZERO,       0.000000000000000000000000000000000000L);
KALYPSSO_MATH_CONSTANT(HALF,       0.500000000000000000000000000000000000L);
KALYPSSO_MATH_CONSTANT(ONE,        1.000000000000000000000000000000000000L);
KALYPSSO_MATH_CONSTANT(TWO,        2.000000000000000000000000000000000000L);
KALYPSSO_MATH_CONSTANT(THREE,      3.000000000000000000000000000000000000L);
KALYPSSO_MATH_CONSTANT(FOUR,       4.000000000000000000000000000000000000L);
KALYPSSO_MATH_CONSTANT(ONE_THIRD,  0.333333333333333333333333333333333333L);
KALYPSSO_MATH_CONSTANT(ONE_FOURTH, 0.250000000000000000000000000000000000L);
KALYPSSO_MATH_CONSTANT(PI,         3.141592653589793238462643383279502884L);
KALYPSSO_MATH_CONSTANT(SQRT_2,     1.414213562373095048763788073031832937L);
KALYPSSO_MATH_CONSTANT(SQRT_3,     1.732050807568877293573725295594556428L);
KALYPSSO_MATH_CONSTANT(SQRT_5,     2.236067977499789696414420059333849622L);
// clang-format on

// math function
#if defined(KALYPSSO_CORE_USE_DOUBLE)
#  define COPYSIGN(x, y) copysign(x, y)
#else
#  define COPYSIGN(x, y) copysignf(x, y)
#endif // KALYPSSO_CORE_USE_DOUBLE

// other useful macros
#define SQR(x) ((x) * (x))

//! Compute the maximum of two values.
template <typename T>
KOKKOS_INLINE_FUNCTION constexpr T const &
max(T const & a, T const & b)
{
  return (a > b) ? a : b;
}

//! Compute the minimum of two values.
template <typename T>
KOKKOS_INLINE_FUNCTION constexpr T const &
min(T const & a, T const & b)
{
  return (a < b) ? a : b;
}

#if defined(KALYPSSO_CORE_USE_DOUBLE)
#  define STATIC_CAST_REAL_T(x) (x)
#else
#  define STATIC_CAST_REAL_T(x) (static_cast<real_t>(x))
#endif

} // namespace kalypsso

KALYPSSO_DISABLE_NVCC_WARNINGS_POP()

#endif // KALYPSSO_CORE_REAL_TYPE_H_
