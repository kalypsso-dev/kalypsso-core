// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file utils_hydro.h
 */
#ifndef KALYPSSO_CORE_UTILS_HYDRO_H_
#define KALYPSSO_CORE_UTILS_HYDRO_H_

#include <type_traits>

#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/HydroParams.h>
#include <kalypsso/core/models/Hydro.h>
#include <kalypsso/core/models/HydroSettings.h>
#include <kalypsso/core/models/HydroState.h>

namespace kalypsso
{

namespace core
{

namespace models
{

// ================================================================================
// ================================================================================
/**
 * Convert conservative variables (rho, rho*u, rho*v, e) to
 * primitive variables (rho,u,v,p)
 * \param[in]  u  conservative variables array
 * \param[in]  settings
 * \param[out] valid will be true is pressure and internal energy are positive
 * \return     q  primitive    variables array
 */
KOKKOS_INLINE_FUNCTION
HydroState2d
computePrimitives(const HydroState2d & u, const HydroSettings & settings, bool & valid)
{
  real_t gamma0 = settings.gamma0;
  real_t smallr = settings.smallr;
  real_t smallp = settings.smallp;

  const auto d = fmax(u[Hydro::ID], smallr);
  const auto ux = u[Hydro::IU] / d;
  const auto uy = u[Hydro::IV] / d;

  // kinetic energy
  const auto eken = HALF_F * (ux * ux + uy * uy);

  // internal energy
  const auto e = u[Hydro::IP] / d - eken;

  valid = e >= 0;

  if (settings.abort_when_negative_eint and !valid)
  {
    Kokkos::abort("Negative internal energy detected - can't proceed further");
  }

  // compute pressure
  const auto p = fmax((gamma0 - ONE_F) * d * e, d * smallp);

  return { d, p, ux, uy };

} // computePrimitives - 2d

// ================================================================================
// ================================================================================
/**
 * Convert conservative variables (rho, rho*u, rho*v, e) to
 * primitive variables (rho,u,v,p)
 * \param[in]  u  conservative variables array
 * \param[in]  settings
 * \return     q  primitive    variables array
 *
 * \note discard argument "valid"
 */
KOKKOS_INLINE_FUNCTION
HydroState2d
computePrimitives(const HydroState2d & u, const HydroSettings & settings)
{

  [[maybe_unused]] bool valid = true;
  return computePrimitives(u, settings, valid);

} // computePrimitives - 2d

// ================================================================================
// ================================================================================
/**
 * Convert conservative variables (rho, rho*u, rho*v, e) to
 * primitive variables (rho,u,v,p)
 * \param[in]  u  conservative variables array
 * \return     q  primitive    variables array
 * nbvar)
 */
KOKKOS_INLINE_FUNCTION
HydroState3d
computePrimitives(const HydroState3d & u, const HydroSettings & settings, bool & valid)
{
  real_t gamma0 = settings.gamma0;
  real_t smallr = settings.smallr;
  real_t smallp = settings.smallp;

  const auto d = fmax(u[Hydro::ID], smallr);
  const auto ux = u[Hydro::IU] / d;
  const auto uy = u[Hydro::IV] / d;
  const auto uz = u[Hydro::IW] / d;

  // kinetic energy
  const auto eken = HALF_F * (ux * ux + uy * uy + uz * uz);

  // internal energy
  const auto e = u[Hydro::IP] / d - eken;

  valid = e >= 0;

  if (settings.abort_when_negative_eint and !valid)
  {
    Kokkos::abort("Negative internal energy detected - can't proceed further");
  }

  // compute pressure
  const auto p = fmax((gamma0 - ONE_F) * d * e, d * smallp);

  return { d, p, ux, uy, uz };

} // computePrimitives - 3d

// ================================================================================
// ================================================================================
/**
 * Convert conservative variables (rho, rho*u, rho*v, rho*w, e) to
 * primitive variables (rho,u,v,w,p)
 * \param[in]  u  conservative variables array
 * \param[in]  settings
 * \return     q  primitive    variables array
 *
 * \note discard argument "valid"
 */
KOKKOS_INLINE_FUNCTION
HydroState3d
computePrimitives(const HydroState3d & u, const HydroSettings & settings)
{

  [[maybe_unused]] bool valid = true;
  return computePrimitives(u, settings, valid);

} // computePrimitives - 3d

// ================================================================================
// ================================================================================
/**
 * Compute speed of sound from pressure and density.
 *
 * \param[in] p pressure
 * \param[in] d density
 * \param[in] gamma0 heat capacity ratio
 *
 * \return speed of sound
 */
KOKKOS_INLINE_FUNCTION
auto
compute_speed_of_sound(real_t const & p, real_t const & d, real_t const & gamma0)
{
  return sqrt(gamma0 * (p) / d);
}

} // namespace models

} // namespace core

} // namespace kalypsso

#endif // KALYPSSO_CORE_UTILS_HYDRO_H_
