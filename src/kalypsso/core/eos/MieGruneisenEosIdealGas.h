// SPDX-FileCopyrightText: 2026 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file MieGruneisenEosIdealGas.h
 *
 * Mie-Gruneisen form of the ideal gas equation of state.
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
 */
#ifndef KALYPSSO_CORE_EOS_MIE_GRUNEISEN_EOS_IDEAL_GAS_H_
#define KALYPSSO_CORE_EOS_MIE_GRUNEISEN_EOS_IDEAL_GAS_H_

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
 * \struct MieGruneisenEosIdealGasParam
 *
 * Define all configurable parameters used in Mie-Gruneisen form of ideal gas EOS.
 */
struct MieGruneisenEosIdealGasParam
{
  //! specific heat ratio
  real_t gamma;

  //! one over gamma minus one
  real_t one_over_gammam1;

  //! reference density.
  //! For multimaterial usage: when material volume fraction is too small, use reference density
  real_t rho0;

  //! retrieve parameters from input config
  static auto
  get_parameters(const size_t i_mat, const ConfigMap & config_map)
  {
    const auto material_id = "material" + std::to_string(i_mat) + "_mie_gruneisen_ig";

    MieGruneisenEosIdealGasParam params;

    params.gamma = config_map.getReal(material_id, "gamma", KALYPSSO_NUM(1.0));
    params.one_over_gammam1 = ONE_F / (params.gamma - ONE_F);
    params.rho0 = config_map.getReal(material_id, "rho0", KALYPSSO_NUM(1.0));

    return params;
  } // get_parameters

  //! print parameters
  void
  print()
  {
    KALYPSSO_INFO("Ideal gas: gamma={}", gamma);
  }

}; // struct MieGruneisenEosIdealGasParam

// ==============================================================================
// ==============================================================================
// ==============================================================================
/**
 * Ideal gas EOS written in Mie-Gruneisen for.
 *
 * This struct is a helper for converting internal energy to pressure and reverse.
 */
struct MieGruneisenEosIdealGas
{
  //! Ideal gas Mie-Gruneisen parameters
  MieGruneisenEosIdealGasParam m_params;

  KOKKOS_DEFAULTED_FUNCTION
  MieGruneisenEosIdealGas() = default;

  /**
   * Initialize Eos for a given material by reading property from ini file.
   *
   * \param[in] i_mat material id (between 0 and nmat-1)
   * \param[in] config_map input configuration map
   */
  MieGruneisenEosIdealGas(const size_t i_mat, const ConfigMap & config_map)
    : m_params(MieGruneisenEosIdealGasParam::get_parameters(i_mat, config_map))
  {}

  /**
   * Initialize ideal gas Mie-Gruneisen Eos.
   *
   * \param[in] config_map input configuration map
   */
  MieGruneisenEosIdealGas(const ConfigMap & config_map)
    : MieGruneisenEosIdealGas(0, config_map)
  {}

  //! print parameters
  void
  print()
  {
    m_params.print();
  }

  /**
   * Reference density.
   */
  KOKKOS_INLINE_FUNCTION real_t
  density_ref() const
  {
    return m_params.rho0;
  }

  /**
   * Compute Gruneisen parameter.
   *
   * Useful and needed for mixture computations.
   *
   * \return Gruneisen parameter.
   */
  KOKKOS_INLINE_FUNCTION real_t
  gamma([[maybe_unused]] const real_t rho) const
  {
    return m_params.gamma - ONE_F;
  }

  /**
   * Compute reference pressure.
   */
  KOKKOS_INLINE_FUNCTION real_t
  pressure_ref([[maybe_unused]] const real_t rho) const
  {
    return ZERO_F;
  } // pressure_ref

  /**
   * Compute reference specific internal energy.
   */
  KOKKOS_INLINE_FUNCTION real_t
  eint_ref([[maybe_unused]] const real_t rho) const
  {
    return ZERO_F;
  } // eint_ref

  /**
   * Compute pressure from volumic internal energy
   */
  KOKKOS_INLINE_FUNCTION real_t
  pressure_from_volumic_eint(real_t eint_volumic, [[maybe_unused]] real_t rho) const
  {
    return (m_params.gamma - ONE_F) * eint_volumic;
  }

  /**
   * Compute pressure from specific internal energy
   */
  KOKKOS_INLINE_FUNCTION real_t
  pressure_from_specific_eint(real_t eint_specific, real_t rho) const
  {
    return (m_params.gamma - ONE_F) * rho * eint_specific;
  } // pressure_from_specific_eint

  /**
   * Compute volumic internal energy from pressure
   */
  KOKKOS_INLINE_FUNCTION real_t
  volumic_eint_from_pressure(real_t pressure, [[maybe_unused]] real_t rho) const
  {
    return pressure * m_params.one_over_gammam1;
  }

  /**
   * Compute specific internal energy from pressure and density
   */
  KOKKOS_INLINE_FUNCTION real_t
  specific_eint_from_pressure(real_t pressure, real_t rho) const
  {
    return volumic_eint_from_pressure(pressure, rho) / rho;
  } // specific_eint_from_pressure

  /**
   * Speed of sound square.
   */
  KOKKOS_INLINE_FUNCTION
  real_t
  sound_speed_square(real_t pressure, real_t rho) const
  {
    return m_params.gamma * pressure / rho;
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
  bulk_modulus(real_t pressure, [[maybe_unused]] real_t rho) const
  {
    return m_params.gamma * pressure;
  } // bulk_modulus

}; // struct MieGruneisenEosIdealGas

} // namespace eos

} // namespace core

} // namespace kalypsso

#endif // KALYPSSO_CORE_EOS_MIE_GRUNEISEN_EOS_IDEAL_GAS_H_
