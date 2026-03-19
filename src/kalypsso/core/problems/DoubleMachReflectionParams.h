// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file DoubleMachReflectionParams.h
 */
#ifndef KALYPSSO_CORE_PROBLEMS_DOUBLE_MACH_REFLECTION_PARAMS_H_
#define KALYPSSO_CORE_PROBLEMS_DOUBLE_MACH_REFLECTION_PARAMS_H_

#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/core/real_type.h>

namespace kalypsso
{

/**
 * DoubleMachReflection test parameters.
 *
 * A shock wave making an angle with a reflecting wall
 *
 * see reference
 * Woodward, P. and Colella, P., "The Numerical Simulation of Two-Dimensional
 * Fluid Flow with Strong Shocks", J. Computational Physics, 54, 115-173 (1984).
 * https://doi.org/10.1016/0021-9991(84)90142-6
 *
 * This configuration is also some time called wedge; see e.g.
 * http://amroc.sourceforge.net/examples/euler/2d/html/ramp_n.htm
 *
 * This test is only available for monofluid, using ideal gas EOS.
 *
 */
struct DoubleMachReflectionParams
{

  // double Mach reflection problem parameters: left and right states
  real_t angle; // angle between the shock front and the vertical in degree
  real_t x0;    // initial position (lower left) of the shock

  real_t rhoL;
  real_t uL;
  real_t vL;
  real_t pL;

  real_t rhoR;
  real_t uR;
  real_t vR;
  real_t pR;

  real_t shock_speed;

  DoubleMachReflectionParams(ConfigMap const & config_map)
    : angle(config_map.getReal("double_mach_reflection", "angle", KALYPSSO_NUM(30.0)))
    , x0(config_map.getReal("double_mach_reflection", "x0", KALYPSSO_NUM(1.0) / 6))
    , rhoL(config_map.getReal("double_mach_reflection", "rhoL", KALYPSSO_NUM(8.0)))
    , uL(config_map.getReal("double_mach_reflection",
                            "uL",
                            KALYPSSO_NUM(8.25) * cos(PI_F * angle / 180)))
    , vL(config_map.getReal("double_mach_reflection",
                            "vL",
                            -KALYPSSO_NUM(8.25) * sin(PI_F * angle / 180)))
    , pL(config_map.getReal("double_mach_reflection", "pL", KALYPSSO_NUM(116.5)))
    , rhoR(config_map.getReal("double_mach_reflection", "rhoR", KALYPSSO_NUM(1.4)))
    , uR(config_map.getReal("double_mach_reflection", "uR", KALYPSSO_NUM(0.0)))
    , vR(config_map.getReal("double_mach_reflection", "vR", KALYPSSO_NUM(0.0)))
    , pR(config_map.getReal("double_mach_reflection", "pR", KALYPSSO_NUM(1.0)))
  {
    // TODO: replace the initialization of gamma0 when lag_remap_hydro is refactored
    const auto gamma0 = config_map.getReal("hydro", "gamma0", KALYPSSO_NUM(1.4));
    // const auto gamma0 = config_map.getReal("material0", "gamma", KALYPSSO_NUM(1.4));

    // soud speed
    const auto aR = sqrt(gamma0 * pR / rhoR);

    // shock speed (S3 in Toro, section 3.1.3, page 100)
    // should be 10 with default value (as in the original Woodward Collela article)
    shock_speed =
      uR + aR * sqrt((gamma0 + ONE_F) / (2 * gamma0) * pL / pR + (gamma0 - ONE_F) / (2 * gamma0));
  }

  KOKKOS_FUNCTION bool
  is_point_in_post_shock_region(real_t x, real_t y, real_t t) const
  {
    const real_t x1 = x - shock_speed / cos(PI_F * angle / 180) * t;
    return y >= (x1 - x0) / tan(PI_F * angle / 180);
  }

  KOKKOS_FUNCTION real_t
  initial_shock_position_x(real_t y) const
  {
    return x0 + y * tan(PI_F * angle / 180);
  }

}; // struct DoubleMachReflectionParams

} // namespace kalypsso

#endif // KALYPSSO_CORE_PROBLEMS_DOUBLE_MACH_REFLECTION_PARAMS_H_
