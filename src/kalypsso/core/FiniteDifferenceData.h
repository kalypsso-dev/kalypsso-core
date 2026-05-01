// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file FiniteDifferenceData.h
 */
#ifndef KALYPSSO_CORE_FINITEDIFFERENCEDATA_H_
#define KALYPSSO_CORE_FINITEDIFFERENCEDATA_H_

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/real_type.h> // for math functions (max, min, ...)

#include <kalypsso/utils/config/ConfigMap.h>
#include <../../better-enums/enum.h>

namespace kalypsso
{

namespace core
{

// clang-format off
/**
 * Enumerate types of stencil used for finite difference estimation first derivative.
 */
BETTER_ENUM(FIRST_DERIVATIVE_STENCIL, int,
            THREE_POINTS,
            FIVE_POINTS,
            SEVEN_POINTS)
// clang-format on

/**
 * A companion data structure for every class that need to do finite difference computation.
 */
struct FiniteDifferenceData
{

  // clang-format off
  //! first order derivative coefs using three point stencil
  //! ./stencil_coefs_helper.py -o 1 -p  1.0  2.0
  //! ./stencil_coefs_helper.py -o 1 -p -1.0  1.0 # this the second order central difference scheme
  //! ./stencil_coefs_helper.py -o 1 -p -2.0 -1.0
#ifdef KALYPSSO_CORE_USE_DOUBLE
  const Kokkos::Array<real_t, 9> FIRST_ORDER_3{
    0.0,      2.0, -1.0 / 2,
   -1.0 / 2,  0.0,  1.0 / 2,
    1.0 / 2, -2.0,  0.0 };
#else
  const Kokkos::Array<real_t, 9> FIRST_ORDER_3{
    0.0f,      2.0f, -1.0f / 2,
   -1.0f / 2,  0.0f,  1.0f / 2,
    1.0f / 2, -2.0f,  0.0f };
#endif

  //! first order derivative coefs using five point stencil
  //! coefficient are obtained by:
  //! ./stencil_coefs_helper.py -o 1 -p  1.0 2.0 3.0 4.0
  //! ./stencil_coefs_helper.py -o 1 -p -1.0 1.0 2.0 3.0
  //! ./stencil_coefs_helper.py -o 1 -p -2.0 -1.0 1.0 2.0 # this is the famous fourth order central difference scheme
  //! ./stencil_coefs_helper.py -o 1 -p -3.0 -2.0 -1.0 1.0
  //! ./stencil_coefs_helper.py -o 1 -p -4.0 -3.0 -2.0 -1.0
#ifdef KALYPSSO_CORE_USE_DOUBLE
  const Kokkos::Array<real_t, 25> FIRST_ORDER_5{
    0.0     ,  4.0,     -3.0    ,  4.0 / 3, -1.0 / 4 ,
   -1.0 / 4 ,  0.0,      3.0 / 2, -1.0 / 2,  1.0 / 12,
    1.0 / 12, -2.0 / 3,  0.0    ,  2.0 / 3, -1.0 / 12,
   -1.0 / 12,  1.0 / 2, -3.0 / 2,  0.0    ,  1.0 / 4 ,
    1.0 / 4 , -4.0 / 3,  3.0    , -4.0    ,  0.0     };
#else
  const Kokkos::Array<real_t, 25> FIRST_ORDER_5{
    0.0f     ,  4.0f,     -3.0f    ,  4.0f / 3, -1.0f / 4 ,
   -1.0f / 4 ,  0.0f,      3.0f / 2, -1.0f / 2,  1.0f / 12,
    1.0f / 12, -2.0f / 3,  0.0f    ,  2.0f / 3, -1.0f / 12,
   -1.0f / 12,  1.0f / 2, -3.0f / 2,  0.0f    ,  1.0f / 4 ,
    1.0f / 4 , -4.0f / 3,  3.0f    , -4.0f    ,  0.0f     };
#endif

  //! first order derivative coefs using seven point stencil
  //! coefficient are obtained by:
  //! ./stencil_coefs_helper.py -o 1 -p  1.0  2.0  3.0  4.0  5.0  6.0
  //! ./stencil_coefs_helper.py -o 1 -p -1.0  1.0  2.0  3.0  4.0  5.0
  //! ./stencil_coefs_helper.py -o 1 -p -2.0 -1.0  1.0  2.0  3.0  4.0
  //! ./stencil_coefs_helper.py -o 1 -p -3.0 -2.0 -1.0  1.0  2.0  3.0
  //! ./stencil_coefs_helper.py -o 1 -p -4.0 -3.0 -2.0 -1.0  1.0  2.0
  //! ./stencil_coefs_helper.py -o 1 -p -5.0 -4.0 -3.0 -2.0 -1.0  1.0
  //! ./stencil_coefs_helper.py -o 1 -p -6.0 -5.0 -4.0 -3.0 -2.0 -1.0
#ifdef KALYPSSO_CORE_USE_DOUBLE
  const Kokkos::Array<real_t, 49> FIRST_ORDER_7{
    0.0     ,    6.0    , -15.0 / 2 , 20.0 / 3, -15.0 / 4,  6.0 /  5, -1.0 /  6,
   -1.0 /  6,    0.0    ,   5.0 / 2,  -5.0 / 3,   5.0 / 6, -1.0 /  4,  1.0 / 30,
    1.0 / 30,  -2.0 / 5 ,   0.0    ,   4.0 / 3,  -1.0 / 2,  2.0 / 15, -1.0 / 60,
   -1.0 / 60,   3.0 / 20,  -3.0 / 4,   0.0    ,   3.0 / 4, -3.0 / 20,  1.0 / 60,
    1.0 / 60,  -2.0 / 15,   1.0 / 2,  -4.0 / 3,   0.0    ,  2.0 /  5, -1.0 / 30,
   -1.0 / 30,   1.0 / 4 ,  -5.0 / 6,   5.0 / 3,  -5.0 / 2,  0.0     ,  1.0 / 6,
    1.0 /  6,  -6.0 / 5 ,  15.0 / 4, -20.0 / 3,  15.0 / 2, -6.0,       0.0
  };
#else
  const Kokkos::Array<real_t, 49> FIRST_ORDER_7{
    0.0f     ,    6.0f    , -15.0f / 2 , 20.0f / 3, -15.0f / 4,  6.0f /  5, -1.0f /  6,
   -1.0f /  6,    0.0f    ,   5.0f / 2,  -5.0f / 3,   5.0f / 6, -1.0f /  4,  1.0f / 30,
    1.0f / 30,  -2.0f / 5 ,   0.0f    ,   4.0f / 3,  -1.0f / 2,  2.0f / 15, -1.0f / 60,
   -1.0f / 60,   3.0f / 20,  -3.0f / 4,   0.0f    ,   3.0f / 4, -3.0f / 20,  1.0f / 60,
    1.0f / 60,  -2.0f / 15,   1.0f / 2,  -4.0f / 3,   0.0f    ,  2.0f /  5, -1.0f / 30,
   -1.0f / 30,   1.0f / 4 ,  -5.0f / 6,   5.0f / 3,  -5.0f / 2,  0.0f     ,  1.0f / 6,
    1.0f /  6,  -6.0f / 5 ,  15.0f / 4, -20.0f / 3,  15.0f / 2, -6.0f,       0.0f
  };
#endif
  // clang-format on

  // ==============================================================
  // ==============================================================
  /**
   * Get first order finite direction coefficient (3 points).
   *
   * \param[in] pos current cell position in the stencil.
   * \param[in] shift
   */
  KOKKOS_INLINE_FUNCTION real_t const &
  coef3(int32_t pos, int32_t shift) const
  {
    return FIRST_ORDER_3[static_cast<size_t>(pos * 3 + shift)];
  }

  // ==============================================================
  // ==============================================================
  /**
   * Get first order finite direction coefficient (5 points).
   *
   * \param[in] pos current cell position in the stencil.
   * \param[in] shift
   */
  KOKKOS_INLINE_FUNCTION real_t const &
  coef5(int32_t pos, int32_t shift) const
  {
    return FIRST_ORDER_5[static_cast<size_t>(pos * 5 + shift)];
  }

  // ==============================================================
  // ==============================================================
  /**
   * Get first order finite direction coefficient (7 points).
   *
   * \param[in] pos current cell position in the stencil.
   * \param[in] shift
   */
  KOKKOS_INLINE_FUNCTION real_t const &
  coef7(int32_t pos, int32_t shift) const
  {
    return FIRST_ORDER_7[static_cast<size_t>(pos * 7 + shift)];
  }

}; // struct FiniteDifferenceData

} // namespace core

} // namespace kalypsso

#endif // KALYPSSO_CORE_FINITEDIFFERENCEDATA_H_
