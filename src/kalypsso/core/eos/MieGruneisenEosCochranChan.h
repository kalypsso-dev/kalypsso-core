// SPDX-FileCopyrightText: 2026 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file MieGruneisenEosCochranChan.h
 *
 * Mie-Gruneisen form of the Cochran-Chan equation of state.
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
 * - A Fluid-Mixture Type Algorithm for Compressible Multicomponent Flow with Mie–Grüneisen Equation
 * of State, Keh-Ming Shyue, Journal of Computational Physics, Volume 171, Issue 2, 2001,Pages
 * 678-707. https://doi.org/10.1006/jcph.2001.6801
 *
 */
#ifndef KALYPSSO_CORE_EOS_MIE_GRUNEISEN_EOS_COCHRAN_CHAN_H_
#define KALYPSSO_CORE_EOS_MIE_GRUNEISEN_EOS_COCHRAN_CHAN_H_

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
 * \struct MieGruneisenEosCochranChanParam
 *
 * Define all configurable parameters used in Mie-Gruneisen for of Cochran-Chan EOS.
 */
struct MieGruneisenEosCochranChanParam
{
  //! reference Gruneisen parameter
  real_t Gamma0;

  //! reference volumic mass
  real_t rho0;

  //! E1
  real_t E1;

  //! E2
  real_t E2;

  //! scalar factor
  real_t A1;

  //! scalar factor
  real_t A2;

  //! reference specific internal energy
  real_t e0;

  //! retrieve parameters from input config
  static auto
  get_parameters(const size_t i_mat, const ConfigMap & config_map)
  {
    const auto material_id = "material" + std::to_string(i_mat) + "_mie_gruneisen_cc";

    MieGruneisenEosCochranChanParam params;

    params.Gamma0 = config_map.getReal(material_id, "Gamma0", KALYPSSO_NUM(1.0));
    params.rho0 = config_map.getReal(material_id, "rho0", KALYPSSO_NUM(1.0));
    params.E1 = config_map.getReal(material_id, "E1", KALYPSSO_NUM(1.0));
    params.E2 = config_map.getReal(material_id, "E2", KALYPSSO_NUM(1.0));
    params.A1 = config_map.getReal(material_id, "A1", KALYPSSO_NUM(1.0));
    params.A2 = config_map.getReal(material_id, "A2", KALYPSSO_NUM(1.0));
    params.e0 = config_map.getReal(material_id, "e0", KALYPSSO_NUM(0.0));

    return params;
  } // get_parameters

  //! print parameters
  void
  print()
  {
    KALYPSSO_INFO("Mie-Gruneisen Cochran-Chan: Gamma0={} rho0={} E1={} E2={} A1={} A2={} e0={}",
                  Gamma0,
                  rho0,
                  E1,
                  E2,
                  A1,
                  A2,
                  e0);
  }

}; // struct MieGruneisenEosCochranChanParam

// ==============================================================================
// ==============================================================================
// ==============================================================================
/**
 * Cochran-Chan EOS written in Mie-Gruneisen for.
 *
 * This struct is a helper for converting internal energy to pressure and reverse.
 */
struct MieGruneisenEosCochranChan
{
  //! Cochran-Chan Mie-Gruneisen parameters
  MieGruneisenEosCochranChanParam m_params;

  KOKKOS_DEFAULTED_FUNCTION
  MieGruneisenEosCochranChan() = default;

  /**
   * Initialize Eos for a given material by reading property from ini file.
   *
   * \param[in] i_mat material id (between 0 and nmat-1)
   * \param[in] config_map input configuration map
   */
  MieGruneisenEosCochranChan(const size_t i_mat, const ConfigMap & config_map)
    : m_params(MieGruneisenEosCochranChanParam::get_parameters(i_mat, config_map))
  {}

  /**
   * Initialize Mie-Gruneisen Eos.
   *
   * \param[in] config_map input configuration map
   */
  MieGruneisenEosCochranChan(const ConfigMap & config_map)
    : MieGruneisenEosCochranChan(0, config_map)
  {}

  //! print parameters
  void
  print()
  {
    m_params.print();
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
    return m_params.Gamma0;
  }

  /**
   * Compute reference pressure.
   */
  KOKKOS_INLINE_FUNCTION real_t
  pressure_ref(const real_t rho) const
  {
    return m_params.A1 * pow(rho / m_params.rho0, m_params.E1) -
           m_params.A2 * pow(rho / m_params.rho0, m_params.E2);
  } // pressure_ref

  /**
   * Compute derivative of reference pressure wrt density.
   */
  KOKKOS_INLINE_FUNCTION real_t
  dpressure_ref_drho(const real_t rho) const
  {
    return m_params.A1 * m_params.E1 / m_params.rho0 * pow(rho / m_params.rho0, m_params.E1 - 1) -
           m_params.A2 * m_params.E2 / m_params.rho0 * pow(rho / m_params.rho0, m_params.E2 - 1);

  } // dpressure_ref_drho

  /**
   * Compute reference specific internal energy.
   */
  KOKKOS_INLINE_FUNCTION real_t
  eint_ref(const real_t rho) const
  {
    return -m_params.A1 / m_params.rho0 / (1 - m_params.E1) *
             (pow(rho / m_params.rho0, m_params.E1 - 1) - 1) +
           m_params.A2 / m_params.rho0 / (1 - m_params.E2) *
             (pow(rho / m_params.rho0, m_params.E2 - 1) - 1) +
           m_params.e0;
  } // eint_ref

  /**
   * Compute derivative of reference specific internal energy wrt density.
   */
  KOKKOS_INLINE_FUNCTION real_t
  deint_ref_drho(const real_t rho) const
  {
    const auto inv_rho02 = ONE_F / (m_params.rho0 * m_params.rho0);
    return m_params.A1 * inv_rho02 * pow(rho / m_params.rho0, m_params.E1 - 2) -
           m_params.A2 * inv_rho02 * pow(rho / m_params.rho0, m_params.E2 - 2);
  } // deint_ref_drho

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
   * Compute square speed of sound.
   *
   * By using the alternate definition of speed of sound
   *
   * \f$ c^2 = \frac{\frac{P}{\rho^2}-\frac{\partial e}{\partial \rho}\bigg_P}{\frac{\partial
   * e}{\partial P}\bigg_\rho}\f$
   *
   * for a Mie-Gruneisen EOS, one obtains
   *
   * \f[
   * c^2 = \frac{d P_{\text{ref}}}{d\rho} -
   * \rho\Gamma \frac{d e_{\text{ref}}}{d\rho} \frac{\Gamma P}{\rho} +
   * (P-P_{\text{ref}})*\left[\frac{1}{\rho} +\frac{1}{\Gamma}\frac{d \Gamma}{d \rho} \right]
   * \f]
   *
   */
  KOKKOS_INLINE_FUNCTION
  real_t
  sound_speed_square(real_t pressure, real_t rho) const
  {
    const auto & rho0 = m_params.rho0;
    const auto & A1 = m_params.A1;
    const auto & A2 = m_params.A2;
    const auto & E1 = m_params.E1;
    const auto & E2 = m_params.E2;

    const auto Gamma = gamma(rho);

    const auto c2 = (Gamma + 1) * pressure / rho -
                    A1 / rho0 * (Gamma + 1 - E1) * pow(rho / rho0, E1 - 1) +
                    A2 / rho0 * (Gamma + 1 - E2) * pow(rho / rho0, E2 - 1);

    return c2;

  } // sound_speed_square

  /**
   * Compute speed of sound.
   */
  KOKKOS_INLINE_FUNCTION
  real_t
  sound_speed(real_t pressure, real_t rho) const
  {
    const auto c2 = sound_speed_square(pressure, rho);
    return sqrt(c2);
  }

  /**
   * Compute isentropic bulk modulus.
   *
   * By definition isentropic bulk modulus is \f$\kappa = \rho \frac{dP}{d\rho}|_S\f$,
   * where the derivative is taken at constant entropy.
   *
   */
  KOKKOS_INLINE_FUNCTION
  real_t
  bulk_modulus(real_t pressure, real_t rho) const
  {
    return rho * sound_speed_square(pressure, rho);

  } // bulk_modulus

}; // struct MieGruneisenEosCochranChan

} // namespace eos

} // namespace core

} // namespace kalypsso

#endif // KALYPSSO_CORE_EOS_MIE_GRUNEISEN_EOS_COCHRAN_CHAN_H_
