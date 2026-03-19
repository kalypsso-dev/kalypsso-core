// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file test_func.h
 */
#ifndef KALYPSSO_TEST_TESTCOMMON_TESTFUNC_H_
#define KALYPSSO_TEST_TESTCOMMON_TESTFUNC_H_

#include <kalypsso/core/kalypsso_core_base.h> // for KALYPSSO_ASSERT
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/enums.h>

namespace kalypsso
{

// =============================================================================
// =============================================================================
//! pointwise init functor to init data with Gaussian shape
struct InitGaussian
{
  static constexpr bool has_face_averaged_values = false;

  InitGaussian(real_t sigma_)
    : sigma(sigma_)
  {}

  real_t sigma;

  KOKKOS_FUNCTION auto
  operator()(real_t x, real_t y, int var) const
  {
    constexpr real_t x0 = KALYPSSO_NUM(0.5);
    constexpr real_t y0 = KALYPSSO_NUM(0.5);
    // constexpr real_t sigma = KALYPSSO_NUM(0.3);

    if (var == 0)
      return exp(-((x - x0) * (x - x0) + (y - y0) * (y - y0)) / 2 / sigma / sigma) / sigma /
             sqrt(2 * PI_F);
    else
      return KALYPSSO_NUM(0.0);
  }

  KOKKOS_FUNCTION auto
  faverage([[maybe_unused]] real_t x,
           [[maybe_unused]] real_t y,
           [[maybe_unused]] real_t delta_x,
           [[maybe_unused]] real_t delta_y,
           [[maybe_unused]] int    var) const
  {
    return ZERO_F;
  }

  KOKKOS_FUNCTION auto
  operator()(real_t x, real_t y, real_t z, int var) const
  {
    constexpr real_t x0 = KALYPSSO_NUM(0.4);
    constexpr real_t y0 = KALYPSSO_NUM(0.8);
    constexpr real_t z0 = KALYPSSO_NUM(0.9);
    // constexpr real_t sigma = KALYPSSO_NUM(0.3);
    if (var == 0)
      return exp(-((x - x0) * (x - x0) + (y - y0) * (y - y0) + (z - z0) * (z - z0)) / 2 / sigma /
                 sigma) /
             sigma / sqrt(2 * PI_F);
    else
      return KALYPSSO_NUM(0.0);
  }
}; // struct InitGaussian

// =============================================================================
// =============================================================================
//! pointwise init functor to init data with Hat shape
struct InitHat
{
  static constexpr bool has_face_averaged_values = false;

  InitHat(real_t radius_)
    : radius(radius_)
  {}

  real_t radius;

  KOKKOS_FUNCTION auto
  operator()(real_t x, real_t y, int var) const
  {
    constexpr real_t x0 = KALYPSSO_NUM(0.5);
    constexpr real_t y0 = KALYPSSO_NUM(0.5);
    // constexpr real_t radius = KALYPSSO_NUM(0.3);

    if (var == 0)
      return ((x - x0) * (x - x0) + (y - y0) * (y - y0)) < radius * radius ? KALYPSSO_NUM(1.0)
                                                                           : KALYPSSO_NUM(0.0);
    else
      return KALYPSSO_NUM(0.0);
  }

  KOKKOS_FUNCTION auto
  faverage([[maybe_unused]] real_t x,
           [[maybe_unused]] real_t y,
           [[maybe_unused]] real_t delta_x,
           [[maybe_unused]] real_t delta_y,
           [[maybe_unused]] int    var) const
  {
    return ZERO_F;
  }

  KOKKOS_FUNCTION auto
  operator()(real_t x, real_t y, real_t z, int var) const
  {
    constexpr real_t x0 = KALYPSSO_NUM(0.4);
    constexpr real_t y0 = KALYPSSO_NUM(0.8);
    constexpr real_t z0 = KALYPSSO_NUM(0.9);
    // constexpr real_t radius = KALYPSSO_NUM(0.3);
    if (var == 0)
      return ((x - x0) * (x - x0) + (y - y0) * (y - y0) + (z - z0) * (z - z0)) < radius * radius
               ? KALYPSSO_NUM(1.0)
               : KALYPSSO_NUM(0.0);
    else
      return KALYPSSO_NUM(0.0);
  }
}; // struct InitHat

// =============================================================================
// =============================================================================
//! pointwise init functor to init data with 2nd order polynomial
struct InitFuncParabola
{
  static constexpr bool has_face_averaged_values = false;

  KOKKOS_FUNCTION auto
  operator()(real_t x, real_t y, int var) const
  {
    if (var == 0 or var == 1)
      return 3 * x * x + y * y;
    else if (var == 2)
      return x;
    else if (var == 3)
      return KALYPSSO_NUM(8.0); // exact value of Laplacian(3*x*x+y*y)
    else
      return KALYPSSO_NUM(0.0);
  }

  KOKKOS_FUNCTION auto
  faverage([[maybe_unused]] real_t x,
           [[maybe_unused]] real_t y,
           [[maybe_unused]] real_t delta_x,
           [[maybe_unused]] real_t delta_y,
           [[maybe_unused]] int    var) const
  {
    return ZERO_F;
  }

  KOKKOS_FUNCTION auto
  operator()(real_t x, real_t y, real_t z, int var) const
  {
    if (var == 0 or var == 1)
      return static_cast<real_t>(3) * x * x + y * y + KALYPSSO_NUM(0.5) * z * z;
    else if (var == 2)
      return x;
    else if (var == 3)
      return y;
    else if (var == 4)
      return KALYPSSO_NUM(9.0); // exact value of Laplacian(3*x*x+y*y+0.5*z*z)
    else
      return KALYPSSO_NUM(0.0);
  }
}; // struct InitFuncParabola

// =============================================================================
// =============================================================================
//! pointwise init functor to init data with sine wave
struct InitFuncSineWave
{
  static constexpr bool has_face_averaged_values = false;

  static constexpr real_t k = KALYPSSO_NUM(1.0);

  KOKKOS_FUNCTION auto
  operator()(real_t x, real_t y, int var) const
  {
    if (var == 0 or var == 1)
      return Kokkos::sin(2 * PI_F * k * (x + y));
    else if (var == 2)
      return x;
    else if (var == 3)
      return KALYPSSO_NUM(-8.0) * PI_F * PI_F * k * k *
             Kokkos::sin(2 * PI_F * k * (x + y)); // exact value of laplacian(sin(x+y))
    else
      return KALYPSSO_NUM(0.0);
  }

  KOKKOS_FUNCTION auto
  faverage([[maybe_unused]] real_t x,
           [[maybe_unused]] real_t y,
           [[maybe_unused]] real_t delta_x,
           [[maybe_unused]] real_t delta_y,
           [[maybe_unused]] int    var) const
  {
    return ZERO_F;
  }

  KOKKOS_FUNCTION auto
  operator()(real_t x, real_t y, real_t z, int var) const
  {
    if (var == 0 or var == 1)
      return Kokkos::sin(2 * PI_F * k * (x + y + z));
    else if (var == 2)
      return x;
    else if (var == 3)
      return y;
    else if (var == 4)
      return static_cast<real_t>(-12) * PI_F * PI_F * k * k *
             Kokkos::sin(2 * PI_F * k * (x + y + z)); // exact value of laplacian(sin(x+y))
    else
      return KALYPSSO_NUM(0.0);
  }
}; // struct InitFuncSineWave

} // namespace kalypsso

#endif // KALYPSSO_TEST_TESTCOMMON_TESTFUNC_H_
