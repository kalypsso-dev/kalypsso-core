// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file IsentropicVortexParams.h
 */
#ifndef KALYPSSO_CORE_PROBLEMS_ISENTROPIC_VORTEX_PARAMS_H_
#define KALYPSSO_CORE_PROBLEMS_ISENTROPIC_VORTEX_PARAMS_H_

#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/core/real_type.h>

namespace kalypsso
{

/**
 * isentropic vortex advection test parameters.
 */
struct IsentropicVortexParams
{

  //! isentropic vortex ambient flow hydrodynamic state variables
  real_t rho_a;
  real_t p_a;
  real_t T_a;
  real_t u_a;
  real_t v_a;
  real_t w_a;

  //! vortex center
  real_t vortex_x;
  real_t vortex_y;
  real_t vortex_z;

  //! vortex strength
  real_t beta;

  //! vortex scale factor
  real_t scale;

  //! number of quadrature points (used to compute initial cell-averaged values)
  int nQuadPts;

  //! useful to compute solution at final time
  bool   use_tEnd;
  real_t tEnd;

  IsentropicVortexParams(ConfigMap const & config_map)
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

    rho_a = config_map.getReal("isentropic_vortex", "density_ambient", KALYPSSO_NUM(1.0));
    p_a = config_map.getReal("isentropic_vortex", "pressure_ambient", KALYPSSO_NUM(1.0));
    T_a = config_map.getReal("isentropic_vortex", "temperature_ambient", KALYPSSO_NUM(1.0));
    u_a = config_map.getReal("isentropic_vortex", "vx_ambient", KALYPSSO_NUM(1.0));
    v_a = config_map.getReal("isentropic_vortex", "vy_ambient", KALYPSSO_NUM(1.0));
    w_a = config_map.getReal("isentropic_vortex", "vz_ambient", KALYPSSO_NUM(1.0));

    vortex_x = config_map.getReal("isentropic_vortex", "center_x", (xmin + xmax) / 2);
    vortex_y = config_map.getReal("isentropic_vortex", "center_y", (ymin + ymax) / 2);
    vortex_z = config_map.getReal("isentropic_vortex", "center_z", (zmin + zmax) / 2);

    beta = config_map.getReal("isentropic_vortex", "strength", KALYPSSO_NUM(5.0));
    scale = config_map.getReal("isentropic_vortex", "scale", KALYPSSO_NUM(1.0));

    nQuadPts = config_map.getInteger("isentropic_vortex", "num_quadrature_points", 4);

    // default value is false, meaning we compute the initial value (t=0)
    use_tEnd = config_map.getBool("isentropic_vortex", "use_tEnd", false);
    tEnd = config_map.getReal("run", "tEnd", KALYPSSO_NUM(1.0));
  }

}; // struct IsentropicVortexParams

} // namespace kalypsso

#endif // KALYPSSO_CORE_PROBLEMS_ISENTROPIC_VORTEX_PARAMS_H_
