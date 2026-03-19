// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file GreshoVortexParams.h
 */
#ifndef KALYPSSO_CORE_PROBLEMS_GRESHO_VORTEX_PARAMS_H_
#define KALYPSSO_CORE_PROBLEMS_GRESHO_VORTEX_PARAMS_H_

#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/core/real_type.h>

namespace kalypsso
{

/**
 * The Gresho problem is a rotating vortex problem independent of time
 * for the case of inviscid flow (Euler equations).
 *
 * reference : https://www.cfd-online.com/Wiki/Gresho_vortex
 */
struct GreshoVortexParams
{

  real_t rho0;
  real_t Ma;

  // advection velocity (optional)
  real_t u, v, w;

  GreshoVortexParams(ConfigMap & config_map)
  {

    rho0 = config_map.getReal("Gresho", "rho0", KALYPSSO_NUM(1.0));
    Ma = config_map.getReal("Gresho", "Ma", KALYPSSO_NUM(0.1));

    u = config_map.getReal("Gresho", "u", KALYPSSO_NUM(0.0));
    v = config_map.getReal("Gresho", "v", KALYPSSO_NUM(0.0));
    w = config_map.getReal("Gresho", "w", KALYPSSO_NUM(0.0));

  } // GreshoVortexParams

}; // struct GreshoVortexParams

} // namespace kalypsso

#endif // KALYPSSO_CORE_PROBLEMS_GRESHO_VORTEX_PARAMS_H_
