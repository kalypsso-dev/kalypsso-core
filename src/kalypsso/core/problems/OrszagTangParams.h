// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file OrszagTangParams.h
 */
#ifndef KALYPSSO_SAHRED_PROBLEMS_ORSZAG_TANG_PARAMS_H_
#define KALYPSSO_SAHRED_PROBLEMS_ORSZAG_TANG_PARAMS_H_

#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/core/real_type.h>

namespace kalypsso
{

/**
 * Orszag-Tang vortex parameters.
 */
struct OrszagTangParams
{
  enum VortexDir : int
  {
    X,
    Y,
    Z
  };

  //! transverse wave vector
  real_t kt;

  //! vortex direction (only use full for cross-checking and debug)
  int vortex_dir;

  OrszagTangParams(ConfigMap const & config_map)
  {
    kt = config_map.getReal("OrszagTang", "kt", KALYPSSO_NUM(0.0));
    vortex_dir = config_map.getInteger("OrszagTang", "vortex_dir", VortexDir::Z);
  }

}; // struct OrszagTangParams

} // namespace kalypsso

#endif // KALYPSSO_SAHRED_PROBLEMS_ORSZAG_TANG_PARAMS_H_
