// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file AlfvenParams.h
 */
#ifndef KALYPSSO_CORE_PROBLEMS_ALFVEN_PARAMS_H_
#define KALYPSSO_CORE_PROBLEMS_ALFVEN_PARAMS_H_

#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/core/real_type.h>

namespace kalypsso
{

/*
*   Propagation of an Alfven wave test parameters.
*
*   see [Tóth2011] Reducing numerical diffusion in magnetospheric simulations.
                   Journal of Geophysical Research: Space Physics, 116(A7).
*/

struct AlfvenParams
{

  // Alfven wave parameters:

  // Density
  real_t rho;

  // Background magnetic field
  real_t B0x;

  // Pressure
  real_t p;

  // Intensity of the perturbation
  real_t Amp;

  // Range of the perturbation
  real_t xrange;

  AlfvenParams(ConfigMap const & config_map)
  {

    rho = config_map.getReal("alfven", "rho", KALYPSSO_NUM(1.0));
    p = config_map.getReal("alfven", "p", KALYPSSO_NUM(5.0));
    B0x = config_map.getReal("alfven", "B0x", KALYPSSO_NUM(3.0));
    Amp = config_map.getReal("alfven", "Amp", KALYPSSO_NUM(-0.1));
    xrange = config_map.getReal("alfven", "xrange", KALYPSSO_NUM(-1.5));
  }

}; // struct AlfvenParams

} // namespace kalypsso


#endif // KALYPSSO_CORE_PROBLEMS_ALFVEN_PARAMS_H_
