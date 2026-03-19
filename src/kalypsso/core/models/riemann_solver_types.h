// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file riemann_solver_types.h
 */
#ifndef KALYPSSO_CORE_MODELS_RIEMANNSOLVERTYPES_H_
#define KALYPSSO_CORE_MODELS_RIEMANNSOLVERTYPES_H_

#include <../better-enums/enum.h>
#include <kalypsso/core/real_type.h>
#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/utils/log/kalypsso_log.h>

namespace kalypsso
{

// clang-format off
/**
 * Enumerate Riemann solver types.
 */
BETTER_ENUM(RiemannSolverType,
            int,
            UNKNOWN,             /*!< invalid value */
            APPROX,              /*!< quasi-exact Riemann solver (hydro-only) */
            LLF,                 /*!< LLF Local Lax-Friedrich */
            HLL,                 /*!< HLL hydro and MHD Riemann solver */
            HLLC,                /*!< HLLC hydro-only Riemann solver */
            HLLC_LM,             /*!< HLLC_LM hydro-only Riemann solver */
            HLLD,                /*!< HLLD MHD-only Riemann solver */
            HLLD_BORIS,          /*!< HLLD MHD-only Riemann solver with Boris Correction */
            FIVE_WAVES,          /*! 5+1 waves solver */
            FIVE_WAVES_ENTROPY   /*!< 5+1 waves solver with entropy correction */
)
// clang-format on

inline RiemannSolverType
get_riemann_solver_type(ConfigMap const & config_map)
{
  auto riemann_solver_name = config_map.getString("hydro", "riemann", "UNKNOWN");
  auto maybe_value = RiemannSolverType::_from_string_nothrow(riemann_solver_name.c_str());
  if (maybe_value)
  {
    return *maybe_value;
  }

  KALYPSSO_ERROR("Invalid riemann_solver_type : {}", riemann_solver_name);
  return RiemannSolverType::UNKNOWN;
}

} // namespace kalypsso

#endif // KALYPSSO_CORE_MODELS_RIEMANNSOLVERTYPES_H_
