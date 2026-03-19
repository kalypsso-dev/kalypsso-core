// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * init_funct.h
 *
 * Some dummy init functor, mostly used for test purpose.
 */
#ifndef KALYPSSO_CORE_INIT_FUNC_H_
#define KALYPSSO_CORE_INIT_FUNC_H_

#include <kalypsso/core/kalypsso_core_base.h> // for KALYPSSO_ASSERT
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/enums.h>

namespace kalypsso
{

// =============================================================================
// =============================================================================
//! a dummy pointwise init functor
struct InitFunc1
{
  static constexpr bool has_face_averaged_values = false;

  KOKKOS_FUNCTION auto
  operator()(real_t x, real_t y, int var) const
  {
    return x + y + static_cast<real_t>(var);
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
    return x + y + z + static_cast<real_t>(var);
  }
};

// =============================================================================
// =============================================================================
//! another dummy pointwise init functor
//! should be periodic in all direction
struct InitFunc2
{
  static constexpr bool has_face_averaged_values = false;

  KOKKOS_FUNCTION auto
  operator()(real_t x, real_t y, int var) const
  {
    return sin(TWO_F * PI_F * x + TWO_F * PI_F * y) + static_cast<real_t>(var);
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
    return sin(TWO_F * PI_F * x + TWO_F * PI_F * y + TWO_F * PI_F * z) + static_cast<real_t>(var);
  }
};

// =============================================================================
// =============================================================================
//! another dummy pointwise init functor.
//!
//! it is periodic in all directions and vector field is divergence-free.
//!
//! Let's call Bx(x,y,z), By(x,y,z), Bz(x,y,z) the vector field
//!
struct InitFunc3
{
  static constexpr bool has_face_averaged_values = true;

  KOKKOS_FUNCTION real_t
  operator()(real_t x, real_t y, int var) const
  {
    if (var == IX)
      return cos(2 * PI_F * (x + y)) * cos(2 * PI_F * (x + y));
    else if (var == IY)
      return sin(2 * PI_F * (x + y)) * sin(2 * PI_F * (x + y));
    else if (var == IZ)
      return cos(2 * PI_F * (x + y)) + sin(2 * PI_F * (x + y));
    else
      return ZERO_F;
  }

  //! face average value
  //!
  //!
  KOKKOS_FUNCTION real_t
  faverage(real_t x, real_t y, real_t delta_x, real_t delta_y, int var) const
  {
    if (var == IX)
      return HALF_F + ONE_F / PI_F * cos(PI_F * (x + y)) * sin(PI_F * delta_y / 2) / delta_y;
    else if (var == IY)
      return HALF_F - ONE_F / PI_F * cos(PI_F * (x + y)) * sin(PI_F * delta_x / 2) / delta_x;
    else if (var == IZ)
      return ONE_F / PI_F / PI_F * sin(PI_F * delta_x) * sin(PI_F * delta_y) *
             (cos(TWO_F * PI_F * (x + y)) + sin(TWO_F * PI_F * (x + y)));
    else
      return ZERO_F;
  }

  KOKKOS_FUNCTION real_t
  operator()(real_t x, real_t y, real_t z, int var) const
  {
    if (var == IX)
      return sin(2 * PI_F * z) * cos(2 * PI_F * (x + y));
    else if (var == IY)
      return sin(2 * PI_F * z) * sin(2 * PI_F * (x + y));
    else if (var == IZ)
      return cos(2 * PI_F * z) * (cos(2 * PI_F * (x + y)) - sin(2 * PI_F * (x + y)));
    else
      return ZERO_F;
  }
};

// =============================================================================
// =============================================================================
//! a dummy pointwise init functor
//!
//! If used as a vector field, should be divergence free.
struct InitFunc4
{
  static constexpr bool has_face_averaged_values = false;

  KOKKOS_FUNCTION auto
  operator()(real_t x, real_t y, int var) const
  {
    return x - y + static_cast<real_t>(var);
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
    return x + y - 2 * z + static_cast<real_t>(var);
  }
};

} // namespace kalypsso

#endif // KALYPSSO_CORE_INIT_FUNC_H_
