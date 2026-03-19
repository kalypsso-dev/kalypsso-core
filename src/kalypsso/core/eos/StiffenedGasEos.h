// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file StiffenedGasEos.h
 *
 * Stiffened gas equation of state is defined by:
 *
 * \f$ p = (\gamma-1) rho e - gamma p_\infty \f$
 *
 * Can be used for ideal gas by setting \f$ p_\infty=0\f$
 */
#ifndef KALYPSSO_CORE_EOS_STIFFENED_GAS_EOS_H_
#define KALYPSSO_CORE_EOS_STIFFENED_GAS_EOS_H_

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
 * Stiffened gas equation of state is defined by:
 *
 * \f$ p = (\gamma-1) rho e - gamma p_\infty \f$
 *
 * This struct is a helper for converting internal energy to pressure and reverse.
 *
 * The stiffened gas eos is useful for modeling liquids (or any weakly compressible medium).
 *
 * This equation of state is sometimes called the Tammann (or Tammann-Tait) EOS; e.g. see
 * https://arxiv.org/pdf/1604.08544
 */
struct StiffenedGasEos
{
  //! specific heat ratio
  real_t m_gamma;

  //! reference pressure (p infinity)
  real_t m_pinf;

  //! one over gamma minus one
  real_t m_one_over_gammam1;

  KOKKOS_DEFAULTED_FUNCTION
  StiffenedGasEos() = default;

  /**
   * Initialize Eos for a given material by reading property from ini file.
   *
   * \param[in] i_mat material id (between 0 and nmat-1)
   * \param[in] config_map input configuration map
   */
  StiffenedGasEos(const size_t i_mat, const ConfigMap & config_map)
    : m_gamma(get_gamma(i_mat, config_map))
    , m_pinf(get_pinf(i_mat, config_map))
    , m_one_over_gammam1(ONE_F / (m_gamma - ONE_F))
  {}

  /**
   * Initialize Eos when only one material is needed.
   *
   * Useful for monofluid (only one material).
   *
   * \param[in] config_map input configuration map
   */
  StiffenedGasEos(const ConfigMap & config_map)
    : StiffenedGasEos(0, config_map)
  {}

  KOKKOS_INLINE_FUNCTION
  auto const &
  gamma() const
  {
    return m_gamma;
  }

  KOKKOS_INLINE_FUNCTION
  auto const &
  pinf() const
  {
    return m_pinf;
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
  pressure_from_volumic_eint(real_t volumic_eint, [[maybe_unused]] real_t rho) const
  {
    return (m_gamma - ONE_F) * volumic_eint - m_gamma * m_pinf;
  }

  /**
   * Compute pressure from specific internal energy
   */
  KOKKOS_INLINE_FUNCTION real_t
  pressure_from_specific_eint(real_t specific_eint, real_t rho) const
  {
    return pressure_from_volumic_eint(rho * specific_eint, rho);
  }

  /**
   * Compute volumic internal energy from pressure
   */
  KOKKOS_INLINE_FUNCTION real_t
  volumic_eint_from_pressure(real_t pressure) const
  {
    return (pressure + m_gamma * m_pinf) * m_one_over_gammam1;
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
    return sqrt(m_gamma * (m_pinf + pressure) / rho);
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

}; // struct StiffenedGasEos

} // namespace eos

} // namespace core

} // namespace kalypsso

#endif // KALYPSSO_CORE_EOS_STIFFENED_GAS_EOS_H_
