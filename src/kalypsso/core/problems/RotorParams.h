// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file RotorParams.h
 */
#ifndef KALYPSSO_CORE_PROBLEMS_ROTOR_PARAMS_H_
#define KALYPSSO_CORE_PROBLEMS_ROTOR_PARAMS_H_

#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/core/real_type.h>

namespace kalypsso
{

/**
 *
 * Rotor problem.
 *
 * Reference:
 *
 * - "The div(B)=0 constraint in shock-capturing MHD codes", G. Toth, JCP, 161, 605 (2000).
 *  https://doi.org/10.1006/jcph.2000.6519
 *
 */
struct RotorParams
{

  // rotor problem parameters

  //! inner radius
  real_t r0;

  //! outer radius
  real_t r1;

  //! velocity at radius r0
  real_t u0;

  //! density inside inner disk
  real_t rho0;

  //! density outside outer disk
  real_t rho1;

  //! pressure
  real_t p0;

  //! magnetic field components
  real_t bx, by, bz;

  //! center coordinates
  real_t xc, yc, zc;

  RotorParams(ConfigMap const & config_map)
  {

    r0 = config_map.getReal("rotor", "r0", KALYPSSO_NUM(0.1));
    r1 = config_map.getReal("rotor", "r1", KALYPSSO_NUM(0.115));
    u0 = config_map.getReal("rotor", "u0", KALYPSSO_NUM(2.0));
    rho0 = config_map.getReal("rotor", "rho0", KALYPSSO_NUM(10.0));
    rho1 = config_map.getReal("rotor", "rho1", KALYPSSO_NUM(1.0));
    p0 = config_map.getReal("rotor", "p0", KALYPSSO_NUM(1.0));
    bx = config_map.getReal("rotor", "bx", KALYPSSO_NUM(5.0) / sqrt(4 * PI_F));
    by = config_map.getReal("rotor", "by", KALYPSSO_NUM(0.0));
    bz = config_map.getReal("rotor", "bz", KALYPSSO_NUM(0.0));

    const auto xmin = config_map.getReal("mesh", "xmin", KALYPSSO_NUM(0.0));
    const auto ymin = config_map.getReal("mesh", "ymin", KALYPSSO_NUM(0.0));
    const auto zmin = config_map.getReal("mesh", "zmin", KALYPSSO_NUM(0.0));

    const auto nbrick_x = config_map.getInteger("p4est_connectivity", "nbrick_x", 1);
    const auto nbrick_y = config_map.getInteger("p4est_connectivity", "nbrick_y", 1);
    const auto nbrick_z = config_map.getInteger("p4est_connectivity", "nbrick_z", 1);

    const auto scaling_factor = config_map.getReal("mesh", "scaling_factor", KALYPSSO_NUM(1.0));

    const auto xmax = xmin + static_cast<real_t>(nbrick_x) * scaling_factor;
    const auto ymax = ymin + static_cast<real_t>(nbrick_y) * scaling_factor;
    const auto zmax = zmin + static_cast<real_t>(nbrick_z) * scaling_factor;

    // default value for rotor center coordinates are the box center coordinates
    xc = config_map.getReal("rotor", "xc", (xmin + xmax) / 2);
    yc = config_map.getReal("rotor", "yc", (ymin + ymax) / 2);
    zc = config_map.getReal("rotor", "zc", (zmin + zmax) / 2);
  }

}; // struct RotorParams

} // namespace kalypsso

#endif // KALYPSSO_CORE_PROBLEMS_ROTOR_PARAMS_H_
