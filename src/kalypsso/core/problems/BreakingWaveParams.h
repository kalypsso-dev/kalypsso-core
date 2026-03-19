// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file BreakingWaveParams.h
 */
#ifndef KALYPSSO_CORE_PROBLEMS_BREAKING_WAVE_PARAMS_H_
#define KALYPSSO_CORE_PROBLEMS_BREAKING_WAVE_PARAMS_H_

#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/core/real_type.h>

namespace kalypsso
{

/**
 * breaking waves test case parameters definition.
 */
struct BreakingWaveParams
{
  //! reference density
  const real_t rho0;

  //! reference pressure
  const real_t p0;

  //! relative amplitude of the sine wave
  const real_t alpha;

  //! specific heat ratio
  const real_t gamma;

  //! reference sound speed
  const real_t c0;

  //! compute density
  KOKKOS_INLINE_FUNCTION
  auto
  rho(real_t x) const
  {
    return rho0 * (1 + alpha * sin(2 * PI_F * x));
  }

  //! compute pressure from density
  KOKKOS_INLINE_FUNCTION
  auto
  p(real_t _rho) const
  {
    return p0 * pow(_rho / rho0, gamma);
  }

  //! compute sound speed
  KOKKOS_INLINE_FUNCTION
  real_t
  c(real_t _rho) const
  {
    return c0 * pow(_rho / rho0, (gamma - 1) / 2);
  }

  //! compute initial velocity
  KOKKOS_INLINE_FUNCTION
  real_t
  u(real_t _c) const
  {
    return 2 / (gamma - 1) * (c0 - _c);
  }

  //! compute time at which shock is formed
  KOKKOS_INLINE_FUNCTION real_t
  t_shock() const
  {
    return 1 / ((gamma + 1) * PI_F * alpha * c0);
  }

  //! For a given x,t find x0 such that u(x0,0) = u(x,t).
  //! x is the image of x0 by the advection at speed u-c as explained in article by Cook and Cabot.
  //! u-c is an invariant, so given an x, u-c can be computed at (x,t) or (x0,0).
  //!
  //! \param[in] x is the target point where we want to compute u(x,t) \param[in] t is time
  //! \param[in] uf velocity at final time
  //! \param[in] cf speed of sound at final time
  //!
  //! \return x0 the preimage of x
  KOKKOS_INLINE_FUNCTION real_t
  compute_x0(real_t x, real_t t, real_t uf, real_t cf) const
  {
    return x - (uf - cf) * t;
  }

  //! constructor
  BreakingWaveParams(ConfigMap const & config_map)
    : rho0(config_map.getReal("breaking_wave", "rho0", KALYPSSO_NUM(1e-3)))
    , p0(config_map.getReal("breaking_wave", "p0", KALYPSSO_NUM(1e6)))
    , alpha(config_map.getReal("breaking_wave", "alpha", KALYPSSO_NUM(0.1)))
    , gamma(config_map.getReal("hydro", "gamma0", KALYPSSO_NUM(5.0) / 3))
    , c0(sqrt(gamma * p0 / rho0))
  {}

}; // struct BreakingWaveParams

} // namespace kalypsso

#endif // KALYPSSO_CORE_PROBLEMS_BREAKING_WAVE_PARAMS_H_
