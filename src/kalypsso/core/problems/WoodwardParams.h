// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file WoodwardParams.h
 */
#ifndef KALYPSSO_CORE_PROBLEMS_WOODWARD_PARAMS_H_
#define KALYPSSO_CORE_PROBLEMS_WOODWARD_PARAMS_H_

#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/core/real_type.h>

namespace kalypsso
{

/**
 * Woodward test (two interacting blast waves) parameters.
 *
 * see reference
 * Woodward, P. and Colella, P., "The Numerical Simulation of Two-Dimensional
 * Fluid Flow with Strong Shocks", J. Computational Physics, 54, 115-173 (1984).
 * https://doi.org/10.1016/0021-9991(84)90142-6
 *
 */
struct WoodwardParams
{

  // woodward problem parameters: left, center and right states
  real_t rhoL;
  real_t uL;
  real_t pL;

  real_t rhoC;
  real_t uC;
  real_t pC;

  real_t rhoR;
  real_t uR;
  real_t pR;

  real_t xdL; // left discontinuity location
  real_t xdR; // right discontinuity location

  WoodwardParams(ConfigMap const & config_map)
    : rhoL(config_map.getReal("woodward", "rhoL", KALYPSSO_NUM(1.0)))
    , uL(config_map.getReal("woodward", "uL", KALYPSSO_NUM(0.0)))
    , pL(config_map.getReal("woodward", "pL", KALYPSSO_NUM(1000.0)))
    , rhoC(config_map.getReal("woodward", "rhoC", KALYPSSO_NUM(1.0)))
    , uC(config_map.getReal("woodward", "uC", KALYPSSO_NUM(0.0)))
    , pC(config_map.getReal("woodward", "pC", KALYPSSO_NUM(0.01)))
    , rhoR(config_map.getReal("woodward", "rhoR", KALYPSSO_NUM(1.0)))
    , uR(config_map.getReal("woodward", "uR", KALYPSSO_NUM(0.0)))
    , pR(config_map.getReal("woodward", "pR", KALYPSSO_NUM(100.0)))
  {

    const auto xmin = config_map.getReal("mesh", "xmin", KALYPSSO_NUM(0.0));

    const auto nbrick_x = config_map.getInteger("p4est_connectivity", "nbrick_x", 1);

    const auto scaling_factor = config_map.getReal("mesh", "scaling_factor", KALYPSSO_NUM(1.0));

    const auto xmax = xmin + static_cast<real_t>(nbrick_x) * scaling_factor;


    xdL = config_map.getReal("woodward", "xd_L", xmin + KALYPSSO_NUM(0.1) * (xmax - xmin));
    xdR = config_map.getReal("woodward", "xd_R", xmin + KALYPSSO_NUM(0.9) * (xmax - xmin));
  }

}; // struct WoodwardParams

} // namespace kalypsso

#endif // KALYPSSO_CORE_PROBLEMS_WOODWARD_PARAMS_H_
