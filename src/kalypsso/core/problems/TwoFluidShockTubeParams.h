// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file TwoFluidShockTubeParams.h
 *
 * All parameters for a two-fluid shock tube problem.
 *
 * Each fluid has its own equation of state (e.g. stiffened gas, ...)
 */
#ifndef KALYPSSO_CORE_PROBLEMS_TWOFLUIDSHOCKTUBE_PARAMS_H_
#define KALYPSSO_CORE_PROBLEMS_TWOFLUIDSHOCKTUBE_PARAMS_H_

#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/core/real_type.h>

namespace kalypsso
{

/**
 * TwoFluidShockTube test parameters.
 *
 */
struct TwoFluidShockTubeParams
{

  //! discontinuity location
  real_t xd;

  TwoFluidShockTubeParams(ConfigMap const & config_map)
  {

    const auto xmin = config_map.getReal("mesh", "xmin", KALYPSSO_NUM(0.0));
    // const auto ymin = config_map.getReal("mesh", "ymin", KALYPSSO_NUM(0.0));
    // const auto zmin = config_map.getReal("mesh", "zmin", KALYPSSO_NUM(0.0));

    const auto nbrick_x = config_map.getInteger("p4est_connectivity", "nbrick_x", 1);
    // const auto nbrick_y = config_map.getInteger("p4est_connectivity", "nbrick_y", 1);
    // const auto nbrick_z = config_map.getInteger("p4est_connectivity", "nbrick_z", 1);

    const auto scaling_factor = config_map.getReal("mesh", "scaling_factor", KALYPSSO_NUM(1.0));

    const auto xmax = xmin + static_cast<real_t>(nbrick_x) * scaling_factor;
    // const auto ymax = ymin + static_cast<real_t>(nbrick_y) * scaling_factor;
    // const auto zmax = zmin + static_cast<real_t>(nbrick_z) * scaling_factor;

    // discontinuity along x axis
    xd = config_map.getReal("two_fluid_shock_tube", "xd", (xmin + xmax) / KALYPSSO_NUM(2.0));
  }

}; // struct TwoFluidShockTubeParams

} // namespace kalypsso

#endif // KALYPSSO_CORE_PROBLEMS_TWOFLUIDSHOCKTUBE_PARAMS_H_
