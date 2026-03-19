// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file SodParams.h
 */
#ifndef KALYPSSO_CORE_PROBLEMS_SOD_PARAMS_H_
#define KALYPSSO_CORE_PROBLEMS_SOD_PARAMS_H_

#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/core/real_type.h>

namespace kalypsso
{

/**
 * Sod shock tube test parameters.
 *
 * see https://en.wikipedia.org/wiki/Sod_shock_tube
 */
struct SodParams
{

  // sod problem parameters: left and right state
  real_t rhoL;
  real_t uL;
  real_t pL;

  real_t rhoR;
  real_t uR;
  real_t pR;

  real_t xd; // discontinuity location

  SodParams(ConfigMap const & config_map)
    : rhoL(config_map.getReal("sod", "rhoL", KALYPSSO_NUM(1.0)))
    , uL(config_map.getReal("sod", "uL", KALYPSSO_NUM(0.0)))
    , pL(config_map.getReal("sod", "pL", KALYPSSO_NUM(1.0)))
    , rhoR(config_map.getReal("sod", "rhoR", KALYPSSO_NUM(0.125)))
    , uR(config_map.getReal("sod", "uR", KALYPSSO_NUM(0.0)))
    , pR(config_map.getReal("sod", "pR", KALYPSSO_NUM(0.1)))
  {

    const auto xmin = config_map.getReal("mesh", "xmin", KALYPSSO_NUM(0.0));

    const auto nbrick_x = config_map.getInteger("p4est_connectivity", "nbrick_x", 1);

    const auto scaling_factor = config_map.getReal("mesh", "scaling_factor", KALYPSSO_NUM(1.0));

    const auto xmax = xmin + static_cast<real_t>(nbrick_x) * scaling_factor;

    xd = config_map.getReal("sod", "xd", (xmin + xmax) / TWO_F);
  }

}; // struct SodParams

} // namespace kalypsso

#endif // KALYPSSO_CORE_PROBLEMS_SOD_PARAMS_H_
