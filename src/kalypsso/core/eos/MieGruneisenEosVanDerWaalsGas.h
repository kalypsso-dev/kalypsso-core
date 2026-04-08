// SPDX-FileCopyrightText: 2026 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file MieGruneisenEosVanDerWaalsGas.h
 *
 * Mie-Gruneisen form of the van der Waals gas equation of state.
 *
 * A Mie-Gruneisen type Eos can be written as
 *
 * \f$ p(\rho,e_{\text{int}}) = \rho \Gamma(\rho) \left[ e_{\text{int}} - e_{\text{\ref}}\right] +
 * p_{\text{ref}} \f$
 *
 * \f$ e_{\text{int}}(\rho,p) = e_{\text{\ref}}(\rho) + \frac{1}{\rho \Gamma(\rho)}\left[
 * p-p_{\text{ref}}(\rho)\right] \f$
 *
 * where \f$ p_{\text{ref}}(\rho) = 0\f$ and \f$ e_{\text{\ref}}(\rho) = 0\f$
 *
 * - A method for compressible multimaterial flows with condensed phase explosive detonation and
 * airblast on unstructured grids, M.A. Price et al, Computers & Fluids, Volume 111, 2015, Pages
 * 76-90. https://doi.org/10.1016/j.compfluid.2015.01.006
 *
 * \sa VanDerWaalsGasEos.h
 *
 */
#ifndef KALYPSSO_CORE_EOS_MIE_GRUNEISEN_EOS_VANDERWAALS_GAS_H_
#define KALYPSSO_CORE_EOS_MIE_GRUNEISEN_EOS_VANDERWAALS_GAS_H_

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

/**
 * \struct MieGruneisenEosVanDerWaalsGasParam
 *
 * Define all configurable parameters used in Mie-Gruneisen for of van der Waals gas EOS.
 */
struct MieGruneisenEosVanDerWaalsGasParam
{
  //! specific heat ratio
  real_t gamma;

  //! one over gamma minus one
  real_t one_over_gammam1;

  //! cohesion parameter
  real_t a;

  //! covolume
  real_t b;

  static auto
  get_parameters(const size_t i_mat, const ConfigMap & config_map)
  {
    const auto material_id = "material" + std::to_string(i_mat) + "_mie_gruneisen_vdwg";

    MieGruneisenEosVanDerWaalsGasParam params;

    params.gamma = config_map.getReal(material_id, "gamma", KALYPSSO_NUM(1.0));
    params.one_over_gammam1 = ONE_F / (params.gamma - ONE_F);
    params.a = config_map.getReal(material_id, "a", KALYPSSO_NUM(0.0));
    params.b = config_map.getReal(material_id, "b", KALYPSSO_NUM(0.0));

    return params;
  } // get_parameters

}; // struct MieGruneisenEosVanDerWaalsGasParam

// ==============================================================================
// ==============================================================================
// ==============================================================================
/**
 * Van Der Waals gas EOS written in Mie-Gruneisen for.
 *
 * This struct is a helper for converting internal energy to pressure and reverse.
 */
struct MieGruneisenEosVanDerWaalsGas
{
  //! Van Der Waals gas Mie-Gruneisen parameters
  MieGruneisenEosVanDerWaalsGasParam m_params;

  KOKKOS_DEFAULTED_FUNCTION
  MieGruneisenEosVanDerWaalsGas() = default;

  /**
   * Initialize Eos for a given material by reading property from ini file.
   *
   * \param[in] i_mat material id (between 0 and nmat-1)
   * \param[in] config_map input configuration map
   */
  MieGruneisenEosVanDerWaalsGas(const size_t i_mat, const ConfigMap & config_map)
    : m_params(MieGruneisenEosVanDerWaalsGasParam::get_parameters(i_mat, config_map))
  {}

  /**
   * Initialize Mie-Gruneisen Eos.
   *
   * \param[in] config_map input configuration map
   */
  MieGruneisenEosVanDerWaalsGas(const ConfigMap & config_map)
    : MieGruneisenEosVanDerWaalsGas(0, config_map)
  {}

  /**
   * Compute Gruneisen parameter.
   *
   * Useful and needed for mixture computations.
   *
   * \return Gruneisen parameter.
   */
  KOKKOS_INLINE_FUNCTION real_t
  gamma(const real_t rho) const
  {
    return (m_params.gamma - ONE_F) / (1 - m_params.b * rho);
  }

  /**
   * Compute reference pressure.
   */
  KOKKOS_INLINE_FUNCTION real_t
  pressure_ref(const real_t rho) const
  {
    return -m_params.a * rho * rho;
  } // pressure_ref

  /**
   * Compute reference specific internal energy.
   */
  KOKKOS_INLINE_FUNCTION real_t
  eint_ref(const real_t rho) const
  {
    return -m_params.a * rho;
  } // eint_ref

  /**
   * Compute pressure from volumic internal energy.
   */
  KOKKOS_INLINE_FUNCTION real_t
  pressure_from_volumic_eint(real_t eint_volumic, [[maybe_unused]] real_t rho) const
  {
    return gamma(rho) * (eint_volumic - rho * eint_ref(rho)) + pressure_ref(rho);
  }

  /**
   * Compute pressure from specific internal energy.
   */
  KOKKOS_INLINE_FUNCTION real_t
  pressure_from_specific_eint(real_t eint_specific, real_t rho) const
  {
    return gamma(rho) * rho * (eint_specific - eint_ref(rho)) + pressure_ref(rho);
  } // pressure_from_specific_eint

  /**
   * Compute volumic internal energy from pressure.
   */
  KOKKOS_INLINE_FUNCTION real_t
  volumic_eint_from_pressure(real_t pressure, real_t rho) const
  {
    return (pressure - pressure_ref(rho)) / gamma(rho) + rho * eint_ref(rho);
  }

  /**
   * Compute specific internal energy from pressure and density.
   */
  KOKKOS_INLINE_FUNCTION real_t
  specific_eint_from_pressure(real_t pressure, real_t rho) const
  {
    return (pressure - pressure_ref(rho)) / (gamma(rho) * rho) + eint_ref(rho);
  } // specific_eint_from_pressure

  /**
   * Compute speed of sound square.
   *
   * By using the alternate definition of speed of sound
   *
   * \f$ \frac{\frac{P}{\rho^2}-\frac{\partial e}{\partial \rho}\bigg_P}{\frac{\partial e}{\partial
   * P}\bigg_\rho}\f$
   *
   * one obtains for the van der Waals EOS the following expression
   * \f$ c = \sqrt{ \frac{\gamma P + (\gamma-2) a  \rho^2 + 2 a b \rho^3}{\rho(1 -b \rho )} }\f$.
   */
  KOKKOS_INLINE_FUNCTION
  real_t
  sound_speed_square(real_t pressure, real_t rho) const
  {
    const auto arho2 = m_params.a * rho * rho;

    return (m_params.gamma * pressure + (m_params.gamma - 2) * arho2 +
            2 * m_params.b * arho2 * rho) /
           (rho - m_params.b * rho * rho);

  } // sound_speed_square

  /**
   * Speed of sound.
   */
  KOKKOS_INLINE_FUNCTION
  real_t
  sound_speed(real_t pressure, real_t rho) const
  {
    return sqrt(sound_speed_square(pressure, rho));
  } // sound_speed

  /**
   * Compute isentropic bulk modulus.
   *
   * By definition isentropic bulk modulus is \f$\kappa = \rho \frac{dP}{d\rho}|_S\f$,
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
  } // bulk_modulus

}; // struct MieGruneisenEosVanDerWaalsGas

} // namespace eos

} // namespace core

} // namespace kalypsso

#endif // KALYPSSO_CORE_EOS_MIE_GRUNEISEN_EOS_VANDERWAALS_GAS_H_
