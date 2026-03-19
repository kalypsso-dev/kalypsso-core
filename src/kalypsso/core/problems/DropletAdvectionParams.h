// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file DropletAdvectionParams.h
 */
#ifndef KALYPSSO_GODUNOV_FIVE_EQ_DROPLET_ADVECTION_PARAMS_H_
#define KALYPSSO_GODUNOV_FIVE_EQ_DROPLET_ADVECTION_PARAMS_H_

#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/core/real_type.h>

namespace kalypsso
{

/**
 * Droplet advection test parameters.
 *
 * A liquid droplet inside a gas advected with constant velocity.
 *
 * references :
 *
 * An interface capturing scheme for modeling atomization in compressible flows, Garrick et al.,
 * Journal of Computational Physics Volume 344, 1 September 2017, Pages 260-280.
 * https://doi.org/10.1016/j.jcp.2017.04.079
 *
 * A hybrid WENO5IS-THINC reconstruction scheme for compressible multiphase flows, Zhand et al.,
 * Journal of Computational Physics Volume 498, 1 February 2024, 112672.
 * https://doi.org/10.1016/j.jcp.2023.112672
 */
struct DropletAdvectionParams
{

  // droplet advection problem parameters

  //! droplet radius
  real_t radius;

  //! droplet center, x coordinate.
  real_t x;

  //! droplet center, y coordinate.
  real_t y;

  //! droplet center, z coordinate.
  real_t z;

  DropletAdvectionParams(ConfigMap const & config_map)
  {
    const auto xmin = config_map.getReal("mesh", "xmin", KALYPSSO_NUM(-1.0));
    const auto ymin = config_map.getReal("mesh", "ymin", KALYPSSO_NUM(-1.0));
    const auto zmin = config_map.getReal("mesh", "zmin", KALYPSSO_NUM(-1.0));

    const auto nbrick_x = config_map.getInteger("p4est_connectivity", "nbrick_x", 1);
    const auto nbrick_y = config_map.getInteger("p4est_connectivity", "nbrick_y", 1);
    const auto nbrick_z = config_map.getInteger("p4est_connectivity", "nbrick_z", 1);

    const auto scaling_factor = config_map.getReal("mesh", "scaling_factor", KALYPSSO_NUM(2.0));

    const auto xmax = xmin + static_cast<real_t>(nbrick_x) * scaling_factor;
    const auto ymax = ymin + static_cast<real_t>(nbrick_y) * scaling_factor;
    const auto zmax = zmin + static_cast<real_t>(nbrick_z) * scaling_factor;

    radius = config_map.getReal("droplet_advection", "radius", KALYPSSO_NUM(1.0));

    x = config_map.getReal("droplet_advection", "x", (xmin + xmax) / 2);
    y = config_map.getReal("droplet_advection", "y", (ymin + ymax) / 2);
    z = config_map.getReal("droplet_advection", "z", (zmin + zmax) / 2);
  }

}; // struct DropletAdvectionParams

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_FIVE_EQ_DROPLET_ADVECTION_PARAMS_H_
