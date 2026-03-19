// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file VanDerWaalsGasEos.h
 *
 * Van der Waals gas equation of state is defined by:
 *
 * \f$ p = \frac{\gamma-1}{1-b\rho} (rho e + a \rho^2) -a \rho^2 \f$
 *
 * some references:
 *
 * - Thermodynamic properties of the van der Waals Fluid, D.C. Johnston,
 *   https://arxiv.org/abs/1402.1205
 * - Thermokinetic model of compressible multiphase ﬂows, Ehsan Reyhanian, Benedikt Dorschner,
 *   Ilya Karlin. https://arxiv.org/abs/2002.09217; see equation A.17 for speed of sound
 * - An oscillation free shock-capturing method for compressible van der Waals supercritical
 *   fluid flows, C. Pantano , R. Saurel, T. Schmitt. Journal of Computational Physics,
 *   Volume 335, 2017, Pages 780-811, https://doi.org/10.1016/j.jcp.2017.01.057;
 *   see equation A.26 for speed of sound (after some rewriting it is exactly the same
 *   as previous reference)
 *
 */
#ifndef KALYPSSO_CORE_EOS_VANDERWAALSGAS_EOS_H_
#define KALYPSSO_CORE_EOS_VANDERWAALSGAS_EOS_H_

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
 * Van der Waals gas equation of state is defined by:
 *
 * \f$ p = \frac{\gamma-1}{1-b\rho} (rho e + a \rho^2) -a \rho^2 \f$
 *
 * This struct is a helper for converting internal energy to pressure and reverse.
 *
 */
struct VanDerWaalsGasEos
{
  //! specific heat ratio
  real_t m_gamma;

  //! one over gamma minus one
  real_t m_one_over_gammam1;

  //! cohesion parameter
  real_t m_a;

  //! covolume
  real_t m_b;

  /**
   * Read cohesion parameter.
   */
  static inline auto
  get_a(const size_t i_mat, const ConfigMap & config_map)
  {
    const auto material_id = "material" + std::to_string(i_mat);

    return config_map.getReal(material_id, "a", KALYPSSO_NUM(0.0));
  }

  /**
   * Read covolume parameter.
   */
  static inline auto
  get_b(const size_t i_mat, const ConfigMap & config_map)
  {
    const auto material_id = "material" + std::to_string(i_mat);

    return config_map.getReal(material_id, "b", KALYPSSO_NUM(0.0));
  }


  KOKKOS_DEFAULTED_FUNCTION
  VanDerWaalsGasEos() = default;

  /**
   * Initialize Eos for a given material by reading property from ini file.
   *
   * \param[in] i_mat material id (between 0 and nmat-1)
   * \param[in] config_map input configuration map
   */
  VanDerWaalsGasEos(const size_t i_mat, const ConfigMap & config_map)
    : m_gamma(get_gamma(i_mat, config_map))
    , m_one_over_gammam1(ONE_F / (m_gamma - ONE_F))
    , m_a(get_a(i_mat, config_map))
    , m_b(get_b(i_mat, config_map))
  {}

  /**
   * Initialize Van der Waals Eos.
   *
   * Useful for monofluid (only one material).
   *
   * \param[in] config_map input configuration map
   */
  VanDerWaalsGasEos(const ConfigMap & config_map)
    : VanDerWaalsGasEos(0, config_map)
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
  pressure_from_volumic_eint(real_t eint_volumic, real_t rho) const
  {
    const auto arho2 = m_a * rho * rho;
    return (m_gamma - ONE_F) / (1 - m_b * rho) * (eint_volumic + arho2) - arho2;
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
  volumic_eint_from_pressure(real_t pressure, real_t rho) const
  {
    const auto arho2 = m_a * rho * rho;
    return (pressure + arho2) * m_one_over_gammam1 * (1 - m_b * rho) - arho2;
  }

  /**
   * Compute specific internal energy from pressure and density
   */
  KOKKOS_INLINE_FUNCTION real_t
  specific_eint_from_pressure(real_t pressure, real_t rho) const
  {
    return volumic_eint_from_pressure(pressure, rho) / rho;
  }

  /**
   * By using the alternate definition of speed of sound
   *
   * \f$ \frac{\frac{P}{\rho^2}-\frac{\partial e}{\partial \rho}\bigg_P}{\frac{\partial e}{\partial
   * P}\bigg_\rho}\f$
   *
   * on obtains for the van der Waals EOS the following expression
   * \f$ c = \sqrt{ \frac{\gamma P + (\gamma-2) a  \rho^2 + 2 a b \rho^3}{\rho(1 -b \rho )} }\f$.
   */
  KOKKOS_INLINE_FUNCTION
  real_t
  sound_speed(real_t pressure, real_t rho) const
  {
    const auto arho2 = m_a * rho * rho;

    return sqrt((m_gamma * pressure + (m_gamma - 2) * arho2 + 2 * m_b * arho2 * rho) /
                (rho - m_b * rho * rho));
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

}; // struct VanDerWaalsGasEos

} // namespace eos

} // namespace core

} // namespace kalypsso

#endif // KALYPSSO_CORE_EOS_VANDERWAALSGAS_EOS_H_
