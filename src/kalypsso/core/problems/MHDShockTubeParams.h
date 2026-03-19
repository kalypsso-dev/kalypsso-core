// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file MHDShockTubeParams.h
 */
#ifndef KALYPSSO_CORE_PROBLEMS_MHDSHOCKTUBE_PARAMS_H_
#define KALYPSSO_CORE_PROBLEMS_MHDSHOCKTUBE_PARAMS_H_

#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/core/real_type.h>

namespace kalypsso
{

/**
 * MHD shock tube problem (Dai-Woodward, Brio-Wu, ...).
 *
 * Reference:
 * - https://flash.rochester.edu/site/flashcode/user_support/flash_ug_devel/node192.html
 * - https://www.astro.princeton.edu/~jstone/Athena/tests/brio-wu/Brio-Wu.html
 *
 * For reference tests input parameters, see table 4.1 in
 * Bouchut, F., Klingenberg, C. & Waagan, K. A multiwave approximate Riemann solver for
 * ideal MHD based on relaxation II: numerical implementation with 3 and 5 waves.
 * Numer. Math. 115, 647–679 (2010). https://doi.org/10.1007/s00211-010-0289-4
 */
struct MHDShockTubeParams
{

  enum Direction : int
  {
    X = 0,
    Y = 1,
    Z = 2
  };

  // Shock-Tube problem parameters (left and right state)
  real_t                   rhoL;
  real_t                   pL;
  real_t                   uL;
  real_t                   vL;
  real_t                   wL;
  Kokkos::Array<real_t, 3> BL;

  real_t                   rhoR;
  real_t                   pR;
  real_t                   uR;
  real_t                   vR;
  real_t                   wR;
  Kokkos::Array<real_t, 3> BR;

  Direction direction;

  real_t xd; // discontinuity location

  MHDShockTubeParams(ConfigMap const & config_map)
  {

    rhoL = config_map.getReal("shock-tube", "rhoL", KALYPSSO_NUM(1.0));
    pL = config_map.getReal("shock-tube", "pL", KALYPSSO_NUM(1.0));
    uL = config_map.getReal("shock-tube", "uL", KALYPSSO_NUM(0.0));
    vL = config_map.getReal("shock-tube", "vL", KALYPSSO_NUM(0.0));
    wL = config_map.getReal("shock-tube", "wL", KALYPSSO_NUM(0.0));
    BL[IX] = config_map.getReal("shock-tube", "BxL", KALYPSSO_NUM(0.75));
    BL[IY] = config_map.getReal("shock-tube", "ByL", KALYPSSO_NUM(1.0));
    BL[IZ] = config_map.getReal("shock-tube", "BzL", KALYPSSO_NUM(0.0));

    rhoR = config_map.getReal("shock-tube", "rhoR", KALYPSSO_NUM(0.125));
    pR = config_map.getReal("shock-tube", "pR", KALYPSSO_NUM(0.1));
    uR = config_map.getReal("shock-tube", "uR", KALYPSSO_NUM(0.0));
    vR = config_map.getReal("shock-tube", "vR", KALYPSSO_NUM(0.0));
    wR = config_map.getReal("shock-tube", "wR", KALYPSSO_NUM(0.0));
    BR[IX] = config_map.getReal("shock-tube", "BxR", KALYPSSO_NUM(0.75));
    BR[IY] = config_map.getReal("shock-tube", "ByR", KALYPSSO_NUM(-1.0));
    BR[IZ] = config_map.getReal("shock-tube", "BzR", KALYPSSO_NUM(0.0));

    direction =
      static_cast<Direction>(config_map.getInteger("shock-tube", "direction", Direction::X));

    if (direction == Direction::X)
    {
      const auto xmin = config_map.getReal("mesh", "xmin", 0.0);
      const auto nbrick_x = config_map.getInteger("p4est_connectivity", "nbrick_x", 1);
      const auto scaling_factor = config_map.getReal("mesh", "scaling_factor", 1.0);
      const auto xmax = xmin + static_cast<real_t>(nbrick_x) * scaling_factor;
      xd = config_map.getReal("shock-tube", "xd", (xmin + xmax) / 2);
    }
    else if (direction == Direction::Y)
    {
      const auto ymin = config_map.getReal("mesh", "ymin", 0.0);
      const auto nbrick_y = config_map.getInteger("p4est_connectivity", "nbrick_y", 1);
      const auto scaling_factor = config_map.getReal("mesh", "scaling_factor", 1.0);
      const auto ymax = ymin + static_cast<real_t>(nbrick_y) * scaling_factor;
      xd = config_map.getReal("shock-tube", "xd", (ymin + ymax) / 2);
    }
    else
    {
      const auto zmin = config_map.getReal("mesh", "zmin", 0.0);
      const auto nbrick_z = config_map.getInteger("p4est_connectivity", "nbrick_z", 1);
      const auto scaling_factor = config_map.getReal("mesh", "scaling_factor", 1.0);
      const auto zmax = zmin + static_cast<real_t>(nbrick_z) * scaling_factor;
      xd = config_map.getReal("shock-tube", "xd", (zmin + zmax) / 2);
    }
  }

}; // struct MHDShockTubeParams

} // namespace kalypsso

#endif // KALYPSSO_CORE_PROBLEMS_MHDSHOCKTUBE_PARAMS_H_
