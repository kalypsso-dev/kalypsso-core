// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file TriplePointParams.h
 *
 * A two-fluid flow parameter file, to be used with our Five-Equation solver.
 *
 * Each fluid has its own equation of state (e.g. stiffened gas, ...)
 */
#ifndef KALYPSSO_CORE_PROBLEMS_TRIPLEPOINT_PARAMS_H_
#define KALYPSSO_CORE_PROBLEMS_TRIPLEPOINT_PARAMS_H_

#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/core/real_type.h>

namespace kalypsso
{

/**
 * TwoFluidShockTube test parameters.
 *
 */
struct TriplePointParams
{

  // left state: material 0 is pure
  real_t rho0;
  real_t p0;
  real_t u0;

  // bottom-right state: material 1 is pure
  real_t rho1;
  real_t p1;
  real_t u1;

  // upper-right state: material 0 is pure
  real_t rho2;
  real_t p2;
  real_t u2;

  //! vertical discontinuity location
  real_t xd;

  //! horizontal discontinuity location
  real_t yd;

  TriplePointParams(ConfigMap const & config_map)
  {

    const auto xmin = config_map.getReal("mesh", "xmin", KALYPSSO_NUM(0.0));
    const auto ymin = config_map.getReal("mesh", "ymin", KALYPSSO_NUM(0.0));
    // const auto zmin = config_map.getReal("mesh", "zmin", KALYPSSO_NUM(0.0));

    const auto nbrick_x = config_map.getInteger("p4est_connectivity", "nbrick_x", 1);
    const auto nbrick_y = config_map.getInteger("p4est_connectivity", "nbrick_y", 1);
    // const auto nbrick_z = config_map.getInteger("p4est_connectivity", "nbrick_z", 1);

    const auto scaling_factor = config_map.getReal("mesh", "scaling_factor", KALYPSSO_NUM(1.0));

    const auto xmax = xmin + static_cast<real_t>(nbrick_x) * scaling_factor;
    const auto ymax = ymin + static_cast<real_t>(nbrick_y) * scaling_factor;
    // const auto zmax = zmin + static_cast<real_t>(nbrick_z) * scaling_factor;

    // left state: material 0 only
    rho0 = config_map.getReal("triple_point", "rho0", KALYPSSO_NUM(1.0));
    p0 = config_map.getReal("triple_point", "p0", KALYPSSO_NUM(100000.0));
    u0 = config_map.getReal("triple_point", "u0", KALYPSSO_NUM(0.0));

    // bottom-right state: material 1 only
    rho1 = config_map.getReal("triple_point", "rho1", KALYPSSO_NUM(0.125));
    p1 = config_map.getReal("triple_point", "p1", KALYPSSO_NUM(10000.0));
    u1 = config_map.getReal("triple_point", "u1", KALYPSSO_NUM(0.0));

    // upper-right state: material 0 only
    rho2 = config_map.getReal("triple_point", "rho2", KALYPSSO_NUM(0.125));
    p2 = config_map.getReal("triple_point", "p2", KALYPSSO_NUM(10000.0));
    u2 = config_map.getReal("triple_point", "u2", KALYPSSO_NUM(0.0));

    // discontinuity along vertical axis
    xd = config_map.getReal("triple_point", "xd", xmin + (xmax - xmin) / KALYPSSO_NUM(7.0));

    // discontinuity along horizontal axis
    yd = config_map.getReal("triple_point", "yd", (ymin + ymax) / KALYPSSO_NUM(2.0));
  }

}; // struct TriplePointParams

} // namespace kalypsso

#endif // KALYPSSO_CORE_PROBLEMS_TRIPLEPOINT_PARAMS_H_
