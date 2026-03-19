// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file IdealGasEos.h
 *
 * Ideal gas equation of state is defined by:
 *
 * \f$ p = (\gamma-1) rho e \f$
 *
 */
#ifndef KALYPSSO_CORE_EOS_IDEAL_GAS_EOS_H_
#define KALYPSSO_CORE_EOS_IDEAL_GAS_EOS_H_

#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/real_type.h>
#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/core/eos/eos_utils.h>

namespace kalypsso
{

namespace core
{

namespace eos
{

// ==============================================================================
// ==============================================================================
// ==============================================================================
// ==========================================================================
// ==========================================================================
/**
 * Ideal gas equation of state is defined by:
 *
 * \f$ p = (\gamma-1) rho e \f$
 *
 * This struct is a helper for converting internal energy to pressure and reverse.
 *
 */
struct IdealGasEos
{
  //! specific heat ratio
  real_t m_gamma;

  //! one over gamma minus one
  real_t m_one_over_gammam1;

  KOKKOS_DEFAULTED_FUNCTION
  IdealGasEos() = default;

  /**
   * Initialize Eos for a given material by reading property from ini file.
   *
   * \param[in] i_mat material id (between 0 and nmat-1)
   * \param[in] config_map input configuration map
   */
  IdealGasEos(const size_t i_mat, const ConfigMap & config_map)
    : m_gamma(get_gamma(i_mat, config_map))
    , m_one_over_gammam1(ONE_F / (m_gamma - ONE_F))
  {}

  /**
   * Initialize Eos .
   *
   * Useful for monofluid (only one material).
   *
   * \param[in] config_map input configuration map
   */
  IdealGasEos(const ConfigMap & config_map)
    : IdealGasEos(0, config_map)
  {}

  KOKKOS_INLINE_FUNCTION
  auto const &
  gamma() const
  {
    return m_gamma;
  }

  KOKKOS_INLINE_FUNCTION
  auto const &
  one_over_gammam1() const
  {
    return m_one_over_gammam1;
  }

  /**
   * Compute pressure from volumic internal energy
   */
  KOKKOS_INLINE_FUNCTION real_t
  pressure_from_volumic_eint(real_t eint_volumic, [[maybe_unused]] real_t rho) const
  {
    return (m_gamma - ONE_F) * eint_volumic;
  }

  /**
   * Compute pressure from specific internal energy
   */
  KOKKOS_INLINE_FUNCTION real_t
  pressure_from_specific_eint(real_t eint_specific, real_t rho) const
  {
    return pressure_from_volumic_eint(rho * eint_specific, rho);
  }

  /**
   * Compute volumic internal energy from pressure
   */
  KOKKOS_INLINE_FUNCTION real_t
  volumic_eint_from_pressure(real_t pressure) const
  {
    return pressure * m_one_over_gammam1;
  }

  /**
   * Compute specific internal energy from pressure and density
   */
  KOKKOS_INLINE_FUNCTION real_t
  specific_eint_from_pressure(real_t pressure, real_t rho) const
  {
    return volumic_eint_from_pressure(pressure) / rho;
  }

  /**
   * By definition isentropic bulk modulus is \f$\kappa = -V \frac{dP}{dV}\f$,
   * where the derivative is taken at constant entropy.
   *
   * In a fluid, one can show that sound speed is \f$c=\sqrt{\frac{\kappa}{\rho}}\f$,
   * thus \f$\kappa=\rho c^2\f$.
   */
  KOKKOS_INLINE_FUNCTION
  real_t
  sound_speed(real_t pressure, real_t rho) const
  {
    return sqrt(m_gamma * pressure / rho);
  }

  /**
   * By definition isentropic bulk modulus is \f$\kappa = -V \frac{dP}{dV}\f$,
   * where the derivative is taken at constant entropy.
   *
   * In a fluid, one can show that sound speed is \f$c=\sqrt{\frac{\kappa}{\rho}}\f$,
   * thus \f$\kappa=\rho c^2\f$.
   */
  KOKKOS_INLINE_FUNCTION
  real_t
  bulk_modulus(real_t pressure, real_t rho) const
  {
    const auto c = sound_speed(pressure, rho);
    return rho * c * c;
  }

}; // struct IdealGasEos

} // namespace eos

} // namespace core

} // namespace kalypsso

#endif // KALYPSSO_CORE_EOS_IDEAL_GAS_EOS_H_
