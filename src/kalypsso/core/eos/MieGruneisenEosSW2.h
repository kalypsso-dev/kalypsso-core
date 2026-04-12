// SPDX-FileCopyrightText: 2026 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file MieGruneisenEosSW2.h
 *
 * Mie-Gruneisen equation of state is a general EOS written as:
 *
 * \f$ p(\rho,e_{\text{int}}) = \rho \Gamma(\rho) \left[ e_{\text{int}} - e_{\text{\ref}}\right] +
 * p_{\text{ref}} \f$
 *
 * \f$ e_{\text{int}}(\rho,p) = e_{\text{\ref}}(\rho) + \frac{1}{\rho \Gamma(\rho)}\left[
 * p-p_{\text{ref}}(\rho)\right] \f$
 *
 * \sa MieGruneisenEosSW
 *
 * class MieGruneisenEosSW2 is a simplified version of MieGruneisenEosSW
 * We chose to implement it for comparison, e.g., with the following reference
 *
 * High-order finite volume method for solving compressible multicomponent flows with
 * Mie–Grüneisen equation of state. F. Zheng and J. Qiu, Computers & Fluids, Volume 284, 2024,
 * 106424, ISSN 0045-7930.
 * https://doi.org/10.1016/j.compfluid.2024.106424
 *
 */
#ifndef KALYPSSO_CORE_EOS_MIE_GRUNEISEN_EOS_SW2_H_
#define KALYPSSO_CORE_EOS_MIE_GRUNEISEN_EOS_SW2_H_

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
 * \struct MieGruneisenEosSW2Param
 *
 * Define all configurable parameters used in Mie-Gruneisen EOS of type shockwave (SW2).
 */
struct MieGruneisenEosSW2Param
{
  //! reference density
  real_t rho0;

  //! reference Mie-Gruneisen parameter
  real_t Gamma0;

  //! reference speed of sound
  real_t c0;

  //! reference speed of sound squared
  real_t c02;

  //! reference Hugoniot slope
  real_t s;

  //! alpha (power use to compute Gruneisen parameter)
  real_t alpha;

  //! reference internal energy
  real_t e0;

  //! reference pressure
  real_t p0;

  //! retrieve parameters from input config
  static auto
  get_parameters(const size_t i_mat, const ConfigMap & config_map)
  {
    const auto material_id = "material" + std::to_string(i_mat) + "_mie_gruneisen_sw2";

    MieGruneisenEosSW2Param params;

    params.rho0 = config_map.getReal(material_id, "rho0", KALYPSSO_NUM(1.0));
    params.Gamma0 = config_map.getReal(material_id, "Gamma0", KALYPSSO_NUM(1.0));
    params.c0 = config_map.getReal(material_id, "c0", KALYPSSO_NUM(1.0));
    params.c02 = params.c0 * params.c0;
    params.s = config_map.getReal(material_id, "s", KALYPSSO_NUM(1.0));
    params.alpha = config_map.getReal(material_id, "alpha", KALYPSSO_NUM(1.0));
    params.e0 = config_map.getReal(material_id, "e0", KALYPSSO_NUM(1.0));
    params.p0 = config_map.getReal(material_id, "p0", KALYPSSO_NUM(1.0));

    return params;
  } // get_parameters

  //! print parameters
  void
  print()
  {
    KALYPSSO_INFO("Mie-Gruneisen Shockwave (v2): rho0={} Gamma0={} c0={} s={} alpha={} e0={} p0={}",
                  rho0,
                  Gamma0,
                  c0,
                  s,
                  alpha,
                  e0,
                  p0);
  }

}; // struct MieGruneisenEosSW2Param

// ==============================================================================
// ==============================================================================
// ==============================================================================
/**
 * Mie-Gruneisen equation of state (shockwave v2).
 *
 * This struct is a helper for converting internal energy to pressure and reverse.
 */
struct MieGruneisenEosSW2
{
  //! Shockwave Mie-Gruneisen parameters
  MieGruneisenEosSW2Param m_params;

  KOKKOS_DEFAULTED_FUNCTION
  MieGruneisenEosSW2() = default;

  /**
   * Initialize Eos for a given material by reading property from ini file.
   *
   * \param[in] i_mat material id (between 0 and nmat-1)
   * \param[in] config_map input configuration map
   */
  MieGruneisenEosSW2(const size_t i_mat, const ConfigMap & config_map)
    : m_params(MieGruneisenEosSW2Param::get_parameters(i_mat, config_map))
  {}

  /**
   * Initialize Mie-Gruneisen Eos.
   *
   * \param[in] config_map input configuration map
   */
  MieGruneisenEosSW2(const ConfigMap & config_map)
    : MieGruneisenEosSW2(0, config_map)
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
   * \note \f$ e_v = 1-\frac{\rho_0}{\rho} = \frac{U_p}{U_s} \f$ relates particule velocity
   * to shock velocity.
   *
   * \return Gruneisen parameter.
   */
  KOKKOS_INLINE_FUNCTION real_t
  gamma(const real_t rho) const
  {
    return m_params.Gamma0 * pow(m_params.rho0 / rho, m_params.alpha);
  }

  /**
   * Compute reference pressure.
   */
  KOKKOS_INLINE_FUNCTION real_t
  pressure_ref(const real_t rho) const
  {
    const auto   ev = 1 - m_params.rho0 / rho;
    const auto & s = m_params.s;
    const auto   d = 1 - s * ev;

    return m_params.p0 + m_params.rho0 * m_params.c02 * ev / (d * d);

  } // pressure_ref

  /**
   * Compute derivative of reference pressure with respect to density.
   */
  KOKKOS_INLINE_FUNCTION real_t
  dpressure_ref_drho(const real_t rho) const
  {
    const auto   ev = 1 - m_params.rho0 / rho;
    const auto & s = m_params.s;
    const auto   d = 1 - s * ev;

    return (m_params.rho0 * m_params.rho0) / (rho * rho) * m_params.c02 / (d * d);

  } // dpressure_ref_drho

  /**
   * Compute reference internal energy.
   */
  KOKKOS_INLINE_FUNCTION real_t
  eint_ref(const real_t rho) const
  {
    const auto ev = 1 - m_params.rho0 / rho;

    return m_params.e0 + (m_params.p0 + pressure_ref(rho)) * ev / (2 * m_params.rho0);

  } // eint_ref

  /**
   * Compute derivative of reference internal energy with respect to density.
   */
  KOKKOS_INLINE_FUNCTION real_t
  deint_ref_drho(const real_t rho) const
  {
    const auto   ev = 1 - m_params.rho0 / rho;
    const auto & s = m_params.s;
    const auto   d = 1 - s * ev;

    return (m_params.p0 + m_params.rho0 * m_params.c02 * ev / (d * d)) / (rho * rho);

  } // deint_ref_drho

  /**
   * Compute pressure from volumic internal energy
   */
  KOKKOS_INLINE_FUNCTION real_t
  pressure_from_volumic_eint(real_t eint_volumic, real_t rho) const
  {
    return gamma(rho) * (eint_volumic - rho * eint_ref(rho)) + pressure_ref(rho);
  }

  /**
   * Compute pressure from specific internal energy
   */
  KOKKOS_INLINE_FUNCTION real_t
  pressure_from_specific_eint(real_t eint_specific, real_t rho) const
  {
    return gamma(rho) * rho * (eint_specific - eint_ref(rho)) + pressure_ref(rho);
  } // pressure_from_specific_eint

  /**
   * Compute specific internal energy from pressure and density
   */
  KOKKOS_INLINE_FUNCTION real_t
  specific_eint_from_pressure(real_t pressure, real_t rho) const
  {
    return (pressure - pressure_ref(rho)) / (gamma(rho) * rho) + eint_ref(rho);
  } // specific_eint_from_pressure

  /**
   * Compute volumic internal energy from pressure
   */
  KOKKOS_INLINE_FUNCTION real_t
  volumic_eint_from_pressure(real_t pressure, real_t rho) const
  {
    return (pressure - pressure_ref(rho)) / gamma(rho) + rho * eint_ref(rho);
  }

  /**
   * Compute \f$ \rho \frac{\Gamma'}{\Gamma}\f$.
   */
  KOKKOS_INLINE_FUNCTION real_t
  rho_dGammadrho_over_Gamma([[maybe_unused]] real_t rho) const
  {
    return -m_params.alpha;
  }

  /**
   * Compute square speed of sound.
   *
   * Speed of sound in the case of Mie-Gruneisen EOS can be obtained e.g., by using equation (12) of
   * M.A. Price et al, A method for compressible multimaterial flows with condensed phase explosive
   * detonation and airblast on unstructured grids (https://doi.org/10.1016/j.compfluid.2015.01.006)
   *
   */
  KOKKOS_INLINE_FUNCTION
  real_t
  sound_speed_square(real_t pressure, real_t rho) const
  {
    real_t c2 = ZERO_F;

    const auto Gamma = gamma(rho);
    const auto p_ref = pressure_ref(rho);

    c2 += (Gamma + 1 + rho_dGammadrho_over_Gamma(rho)) * (pressure - p_ref) / rho;
    c2 += Gamma * p_ref / rho + dpressure_ref_drho(rho) - Gamma * rho * deint_ref_drho(rho);

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
   */
  KOKKOS_INLINE_FUNCTION
  real_t
  bulk_modulus(real_t pressure, real_t rho) const
  {
    return rho * sound_speed_square(pressure, rho);
  } // bulk_modulus

}; // struct MieGruneisenEosSW2

} // namespace eos

} // namespace core

} // namespace kalypsso

#endif // KALYPSSO_CORE_EOS_MIE_GRUNEISEN_EOS_SW2_H_
