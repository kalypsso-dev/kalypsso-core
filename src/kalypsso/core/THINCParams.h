// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file THINCParams.h
 */
#ifndef KALYPSSO_CORE_THINC_PARAMS_H_
#define KALYPSSO_CORE_THINC_PARAMS_H_

#include <kalypsso/core/kalypsso_core_base.h> // for assertm
#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/core/real_type.h>

namespace kalypsso
{

/**
 * THINC (Tangent Hyperbolic INterface Capturing) parameters.
 */
struct THINCParams
{
  //! turn on/off THINC reconstruction at run run-time
  bool enabled;

  //! assumed interface width
  real_t beta;

  //! volume fraction threshold controlling when THINC reconstruction is used
  real_t epsilon;

  THINCParams(ConfigMap const & config_map)
    : enabled(config_map.getBool("THINC", "enabled", false))
    , beta(config_map.getReal("THINC", "beta", KALYPSSO_NUM(1.5)))
    , epsilon(config_map.getReal("THINC", "epsilon", KALYPSSO_NUM(1e-3)))
  {
    if (enabled)
    {
      assertm(beta >= 0, "[THINCParams] beta must be non-negative.");
      assertm(epsilon >= 0, "[THINCParams] epsilon must be non-negative.");
    }
  }

}; // struct THINCParams

} // namespace kalypsso

#endif // KALYPSSO_CORE_THINC_PARAMS_H_
