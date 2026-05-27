// SPDX-FileCopyrightText: 2026 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file CircleAdvectionParams.h
 */
#ifndef KALYPSSO_CORE_PROBLEMS_CIRCLE_ADVECTION_PARAMS_H_
#define KALYPSSO_CORE_PROBLEMS_CIRCLE_ADVECTION_PARAMS_H_

#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/core/real_type.h>
#include <kalypsso/core/enums.h>

#include <../../better-enums/enum.h>

namespace kalypsso
{

/**
 * Circle advection test (mostly useful for multimaterial tests).
 *
 * The circle shape is divided in 4 quadrants, each of which filled with a different material.
 * There are actually two tests where the velocity field is imposed, either a pure constant
 * velocity advection or a vortex velocity field.
 *
 * The vortex case is taken from Rider and Kothe
 * Reconstructing Volume Tracking, William J. Rider, Douglas B. Kothe,
 * Journal of Computational Physics, Volume 141, Issue 2, 1998, Pages 112-152, ISSN 0021-9991,
 * https://doi.org/10.1006/jcph.1998.5906.
 *
 * Other references :
 *
 * Extension of the hybrid WENO5IS-THINC scheme to compressible multiphase flows with an
 * arbitrary number of components, Wenbin Zhang, Thomas Paula, Alexander Bußmann, Stefan Adami,
 * Nikolaus A. Adams, Journal of Computational Physics, Volume 524, 2025, 113702,
 * ISSN 0021-9991, https://doi.org/10.1016/j.jcp.2024.113702.
 * see section 4.1.2 and 4.2
 *
 * Multi-material hydrodynamics with algebraic sharp interface capturing, Aditya K. Pandare,
 * Jacob Waltz, Jozsef Bakosi, Computers & Fluids, Volume 215, 2021, 104804, ISSN 0045-7930,
 * https://doi.org/10.1016/j.compfluid.2020.104804.
 * see section 5.1
 *
 * Extension of generic two-component VOF interface advection schemes to an arbitrary number of
 * components, Matthieu Ancellin, Bruno Després, Stéphane Jaouen,
 * Journal of Computational Physics, Volume 473, 2023, 111721, ISSN 0021-9991,
 * https://doi.org/10.1016/j.jcp.2022.111721.
 * see section 3.5
 */
struct CircleAdvectionParams
{

  //! circle radius
  real_t r;

  //! circle center, x coordinate.
  real_t x;

  //! circle center, y coordinate.
  real_t y;

  //! circle center, z coordinate.
  real_t z;

  CircleAdvectionParams(ConfigMap const & config_map)
  {
    const auto xmin = config_map.getReal("mesh", "xmin", KALYPSSO_NUM(-25.0));
    const auto ymin = config_map.getReal("mesh", "ymin", KALYPSSO_NUM(-25.0));
    const auto zmin = config_map.getReal("mesh", "zmin", KALYPSSO_NUM(-25.0));

    const auto nbrick_x = config_map.getInteger("p4est_connectivity", "nbrick_x", 1);
    const auto nbrick_y = config_map.getInteger("p4est_connectivity", "nbrick_y", 1);
    const auto nbrick_z = config_map.getInteger("p4est_connectivity", "nbrick_z", 1);

    const auto scaling_factor = config_map.getReal("mesh", "scaling_factor", KALYPSSO_NUM(50.0));

    const auto xmax = xmin + static_cast<real_t>(nbrick_x) * scaling_factor;
    const auto ymax = ymin + static_cast<real_t>(nbrick_y) * scaling_factor;
    const auto zmax = zmin + static_cast<real_t>(nbrick_z) * scaling_factor;

    r = config_map.getReal("circle_advection", "r", KALYPSSO_NUM(0.25));

    x = config_map.getReal("circle_advection", "x", (xmin + xmax) / 2);
    y = config_map.getReal("circle_advection", "y", (ymin + ymax) / 2);
    z = config_map.getReal("circle_advection", "z", (zmin + zmax) / 2);
  }

}; // struct CircleAdvectionParams

} // namespace kalypsso

#endif // KALYPSSO_CORE_PROBLEMS_CIRCLE_ADVECTION_PARAMS_H_
