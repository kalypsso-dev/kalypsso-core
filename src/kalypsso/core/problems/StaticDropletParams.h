// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file StaticDropletParams.h
 */
#ifndef KALYPSSO_CORE_PROBLEMS_STATIC_DROPLET_PARAMS_H_
#define KALYPSSO_CORE_PROBLEMS_STATIC_DROPLET_PARAMS_H_

#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/core/real_type.h>

namespace kalypsso
{

/**
 * Static droplet test parameters.
 *
 * A liquid droplet inside a gas at equilibrium (pressure difference between liquid and gas is
 * balanced by surface force).
 *
 * reference :
 * A finite-volume HLLC-based scheme for compressible interfacial flows with surface tension,
 * Garrick Owkes and Regele, Journal of Computational Physics Volume 339, 15 June 2017, Pages 46-67.
 * https://doi.org/10.1016/j.jcp.2017.03.007
 */
struct StaticDropletParams
{

  // static droplet problem parameters

  //! parameter used to control interface thickness
  real_t epsilon_coef;

  //! another parameter used to control interface thickness
  real_t Delta;

  //! droplet radius
  real_t radius;

  //! droplet center, x coordinate.
  real_t x;

  //! droplet center, y coordinate.
  real_t y;

  //! droplet center, z coordinate.
  real_t z;

  StaticDropletParams(ConfigMap const & config_map)
  {
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

    epsilon_coef = config_map.getReal("static_droplet", "epsilon_coef", KALYPSSO_NUM(0.72));
    Delta = config_map.getReal("static_droplet", "Delta", KALYPSSO_NUM(2.0));

    radius = config_map.getReal("static_droplet", "radius", KALYPSSO_NUM(1.0));

    x = config_map.getReal("static_droplet", "x", (xmin + xmax) / 2);
    y = config_map.getReal("static_droplet", "y", (ymin + ymax) / 2);
    z = config_map.getReal("static_droplet", "z", (zmin + zmax) / 2);
  }

}; // struct StaticDropletParams

} // namespace kalypsso

#endif // KALYPSSO_CORE_PROBLEMS_STATIC_DROPLET_PARAMS_H_
