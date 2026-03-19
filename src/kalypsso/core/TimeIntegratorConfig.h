// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file TimeIntegrationConfig.h
 *
 * Implement classic time integrator algorithm, e.g. :
 *
 * - Hancock (single state, second order)
 * - RK2-SSP (two stages, second order)
 * - RK3-SSP (three stages, third order)
 *
 * Reference on strong stability preserving Runge-Kutta methods:
 *
 * 1. http://epubs.siam.org/doi/pdf/10.1137/S0036142901389025
 * A NEW CLASS OF OPTIMAL HIGH-ORDER STRONG-STABILITY-PRESERVING
 * TIME DISCRETIZATION METHODS
 * RAYMOND J. SPITERI AND STEVEN J. RUUTH,
 * SIAM J. Numer. Anal, Vol 40, No 2, pp 469-491
 *
 * 2. https://link.springer.com/book/10.1007/978-1-4419-6412-0
 * Numerical Methods for Fluid Dynamics With Applications to Geophysics, Dale R. Durran
 * textbook, 2010, Springer. See section 2.3.3.
 */
#ifndef KALYPSSO_CORE_TIMEINTEGRATORCONFIG_H_
#define KALYPSSO_CORE_TIMEINTEGRATORCONFIG_H_

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/real_type.h> // for math functions (max, min, ...)

#include <kalypsso/utils/config/ConfigMap.h>
#include <../../better-enums/enum.h>

namespace kalypsso
{
// clang-format off
/**
 * An enum type to represent all implemented time integration methods.
 */
BETTER_ENUM(TimeIntegrator, uint8_t,
            HANCOCK = 0,
            RK2_SSP = 1,
            RK3_SSP = 2)
// clang-format on

/**
 * A companion data structure for class ComputeRefineFlags holding parameters used to
 * compute refine flags.
 */
struct TimeIntegratorConfig
{
  static TimeIntegrator
  get_time_integrator(ConfigMap const & config_map)
  {
    auto time_integrator_name = config_map.getString("amr", "time_integrator", "HANCOCK");
    auto maybe_value = TimeIntegrator::_from_string_nothrow(time_integrator_name.c_str());
    if (maybe_value)
    {
      return *maybe_value;
    }
    else
    {
      Kokkos::abort("Wrong time integrator parameter.");
    }
    return TimeIntegrator::HANCOCK;
  }

  using RKCoefs = Kokkos::Array<real_t, 2>;

  static constexpr RKCoefs RK2_SSP_STAGE1_COEFS{ KALYPSSO_NUM(0.0), KALYPSSO_NUM(1.0) };
  static constexpr RKCoefs RK2_SSP_STAGE2_COEFS{ KALYPSSO_NUM(0.5), KALYPSSO_NUM(0.5) };

  static constexpr RKCoefs RK3_SSP_STAGE1_COEFS{ KALYPSSO_NUM(0.0), KALYPSSO_NUM(1.0) };
  static constexpr RKCoefs RK3_SSP_STAGE2_COEFS{ KALYPSSO_NUM(0.75), KALYPSSO_NUM(0.25) };
  static constexpr RKCoefs RK3_SSP_STAGE3_COEFS{ KALYPSSO_NUM(1.0) / 3, KALYPSSO_NUM(2.0) / 3 };

}; // struct TimeIntegratorConfig

} // namespace kalypsso

#endif // KALYPSSO_CORE_TIMEINTEGRATORCONFIG_H_
