// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ViscosityParams.h
 */
#ifndef KALYPSSO_CORE_VISCOSITY_PARAMS_H_
#define KALYPSSO_CORE_VISCOSITY_PARAMS_H_

#include <kalypsso/core/kalypsso_core_base.h> // for assertm
#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/core/real_type.h>

namespace kalypsso
{

/**
 * Constant uniform viscosity parameters.
 */
struct ViscosityParams
{
  //! turn on/off viscosity at run run-time
  bool enabled;

  //! dynamic viscosity
  real_t mu;

  //! enable Hancock predictor
  bool hancock_predictor_enabled;

  ViscosityParams(ConfigMap const & config_map)
    : enabled(config_map.getBool("viscosity", "enabled", false))
    , mu(config_map.getReal("viscosity", "mu", KALYPSSO_NUM(1e-4)))
    , hancock_predictor_enabled(config_map.getBool("viscosity", "hancock_predictor_enabled", false))
  {
    assertm(mu >= 0, "[ViscosityParams] dynamic viscosity mu must be non-negative.");
  }

}; // struct ViscosityParams

} // namespace kalypsso

#endif // KALYPSSO_CORE_VISCOSITY_PARAMS_H_
