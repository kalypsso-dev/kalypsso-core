// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file FieldLoopParamsParams.h
 */
#ifndef KALYPSSO_CORE_PROBLEMS_FIELDLOOPADVECTION_PARAMS_H_
#define KALYPSSO_CORE_PROBLEMS_FIELDLOOPADVECTION_PARAMS_H_

#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/core/real_type.h>

namespace kalypsso
{

/**
 * Field loop advection problem.
 *
 *
 * Reference:
 * - T. Gardiner & J.M. Stone, "An unsplit Godunov method for ideal MHD
 *   via constrained transport", JCP, 205, 509 (2005)
 * - http://www.astro.princeton.edu/~jstone/Athena/tests/field-loop/Field-loop.html
 */
struct FieldLoopAdvectionParams
{

  real_t radius;
  real_t density;
  real_t pressure;
  real_t B0;
  real_t vflow;

  //! field loop center coordinates at t=0
  real_t xc, yc, zc;

  // ============================================================================
  // ============================================================================
  FieldLoopAdvectionParams(ConfigMap const & config_map)
    : radius(config_map.getReal("FieldLoop", "radius", KALYPSSO_NUM(1.0)))
    , density(config_map.getReal("FieldLoop", "density", KALYPSSO_NUM(1.0)))
    , pressure(config_map.getReal("FieldLoop", "pressure", KALYPSSO_NUM(1.0)))
    , B0(config_map.getReal("FieldLoop", "B0", KALYPSSO_NUM(0.001)))
    , vflow(config_map.getReal("FieldLoop", "vflow", SQRT_5_F))
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

    // default value for field loop center coordinates are the box center coordinates
    xc = config_map.getReal("FieldLoop", "xc", (xmin + xmax) / 2);
    yc = config_map.getReal("FieldLoop", "yc", (ymin + ymax) / 2);
    zc = config_map.getReal("FieldLoop", "zc", (zmin + zmax) / 2);
  }

}; // struct FieldLoopParams

} // namespace kalypsso

#endif // KALYPSSO_CORE_PROBLEMS_FIELDLOOPADVECTION_PARAMS_H_
