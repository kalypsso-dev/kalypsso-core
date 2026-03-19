// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file HydroSettings.h
 * \brief Hydrodynamics solver parameters.
 */
#ifndef KALYPSSO_CORE_MODELS_HYDROSETTINGS_H_
#define KALYPSSO_CORE_MODELS_HYDROSETTINGS_H_

#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/real_type.h>
#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/core/models/riemann_solver_types.h>

namespace kalypsso
{

// ===========================================================================
// ===========================================================================
/**
 * Parameters that can be passed by copy to a Kokkos device.
 */
struct HydroSettings
{

  // hydro (numerical scheme) parameters
  real_t gamma0; /*!< specific heat capacity ratio (adiabatic index) - DEPRECATED (use EosWrapper
                    instead) */
  real_t            cfl;               /*!< Courant-Friedrich-Lewy parameter.*/
  real_t            slope_type;        /*!< type of slope computation (2 for second order scheme).*/
  real_t            smallr;            /*!< small density cut-off*/
  real_t            smallc;            /*!< small speed of sound cut-off*/
  real_t            smallp;            /*!< small pressure cut-off*/
  real_t            smallpp;           /*!< smallp times smallr*/
  real_t            cIso;              /*!< if non zero, isothermal */
  int               niter_riemann;     /*!< number of iteration used in quasi-exact Riemann solver*/
  RiemannSolverType riemannSolverType; /*!< type of Riemann solver (LLF, HLLC, ...) */
  bool              abort_when_negative_eint; /*!< abort if negative internal energy */

  KALYPSSO_STATIC_MATH_CONSTANT(ONE_DOT_FOUR, 1.400000000000000000000000000000000000L);
  KALYPSSO_STATIC_MATH_CONSTANT(SMALLR, 1e-8);
  KALYPSSO_STATIC_MATH_CONSTANT(SMALLP, 1e-6);

  HydroSettings(ConfigMap const & config_map);
  ~HydroSettings() = default;

  void
  print() const;

}; // struct HydroSettings

} // namespace kalypsso

#endif // KALYPSSO_CORE_MODELS_HYDROSETTINGS_H_
