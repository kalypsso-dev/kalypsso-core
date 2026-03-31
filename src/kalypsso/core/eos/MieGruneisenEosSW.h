// SPDX-FileCopyrightText: 2026 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file MieGruneisenEosSW.h
 *
 * Mie-Gruneisen equation of state is a general EOS written as:
 *
 * \f$ p(\rho,e_{\text{int}}) = \rho \Gamma(\rho) \left[ e_{\text{int}} - e_{\text{\ref}}\right] +
 * p_{\text{ref}} \f$
 *
 * \f$ e_{\text{int}}(\rho,p) = e_{\text{\ref}}(\rho) + \frac{1}{\rho \Gamma(\rho)}\left[
 * p-p_{\text{ref}}(\rho)\right] \f$
 *
 * some references:
 *
 * - A method for compressible multimaterial flows with condensed phase explosive detonation and
 * airblast on unstructured grids, M.A. Price et al, Computers & Fluids, Volume 111, 2015, Pages
 * 76-90. https://doi.org/10.1016/j.compfluid.2015.01.006
 * - High-order methods for diffuse-interface models in compressible multi-medium flows: A review,
 * V. Maltsev et al, Physics of Fluids 1 February 2022; 34 (2): 021301.
 * https://doi.org/10.1063/5.0077314
 *
 */
#ifndef KALYPSSO_CORE_EOS_MIE_GRUNEISEN_EOS_SW_H_
#define KALYPSSO_CORE_EOS_MIE_GRUNEISEN_EOS_SW_H_

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
 * \struct MieGruneisenEosSWParam
 *
 * Define all configurable parameters used in Mie-Gruneisen EOS of type shockwave (SW).
 */
struct MieGruneisenEosSWParam
{
  //! reference density
  real_t rho0;

  //! reference Mie-Gruneisen parameter
  real_t Gamma0;

  //! reference speed of sound
  real_t c0;

  //! reference speed of sound squared
  real_t c02;

  //! reference pressure curve param s1
  real_t s1;

  //! reference pressure curve param s2
  real_t s2;

  //! reference pressure curve param s3
  real_t s3;

  //! Mie-Gruneisen parameter "b"
  real_t b;

  //! reference internal energy
  real_t e0;

  static auto
  get_parameters(const size_t i_mat, const ConfigMap & config_map)
  {
    const auto material_id = "material" + std::to_string(i_mat) + "_mie_gruneisen_sw";

    MieGruneisenEosSWParam params;

    params.rho0 = config_map.getReal(material_id, "rho0", KALYPSSO_NUM(1.0));
    params.Gamma0 = config_map.getReal(material_id, "Gamma0", KALYPSSO_NUM(1.0));
    params.c0 = config_map.getReal(material_id, "c0", KALYPSSO_NUM(1.0));
    params.c02 = params.c0 * params.c0;
    params.s1 = config_map.getReal(material_id, "s1", KALYPSSO_NUM(1.0));
    params.s2 = config_map.getReal(material_id, "s2", KALYPSSO_NUM(1.0));
    params.s3 = config_map.getReal(material_id, "s3", KALYPSSO_NUM(1.0));
    params.b = config_map.getReal(material_id, "b", KALYPSSO_NUM(1.0));
    params.e0 = config_map.getReal(material_id, "e0", KALYPSSO_NUM(1.0));

    return params;
  } // get_parameters

}; // struct MieGruneisenEosSWParam

// ==============================================================================
// ==============================================================================
// ==============================================================================
/**
 * Mie-Gruneisen equation of state (shockwave).
 *
 * This struct is a helper for converting internal energy to pressure and reverse.
 */
struct MieGruneisenEosSW
{
  //! Shockwave Mie-Gruneisen parameters
  const MieGruneisenEosSWParam m_params;

  KOKKOS_DEFAULTED_FUNCTION
  MieGruneisenEosSW() = default;

  /**
   * Initialize Eos for a given material by reading property from ini file.
   *
   * \param[in] i_mat material id (between 0 and nmat-1)
   * \param[in] config_map input configuration map
   */
  MieGruneisenEosSW(const size_t i_mat, const ConfigMap & config_map)
    : m_params(MieGruneisenEosSWParam::get_parameters(i_mat, config_map))
  {}

  /**
   * Initialize Mie-Gruneisen Eos.
   *
   * \param[in] config_map input configuration map
   */
  MieGruneisenEosSW(const ConfigMap & config_map)
    : MieGruneisenEosSW(0, config_map)
  {}

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
    const auto ev = 1 - m_params.rho0 / rho;

    return m_params.Gamma0 * (1 - ev) + m_params.b * ev;
  }

  /**
   * Compute Gruneisen parameter.
   *
   * Mostly for internal use.
   *
   * \return Gruneisen parameter.
   */
  KOKKOS_INLINE_FUNCTION real_t
  gamma_from_ev(const real_t ev) const
  {
    return m_params.Gamma0 * (1 - ev) + m_params.b * ev;
  }

  /**
   * Compute parameter "s" (Hugoniot slope)
   */
  KOKKOS_INLINE_FUNCTION real_t
  hugoniot_slope(const real_t ev) const
  {
    return m_params.s1 + m_params.s2 * ev + m_params.s3 * ev * ev;
  }

  /**
   * Compute derivative of parameter "s" (Hugoniot slope)
   */
  KOKKOS_INLINE_FUNCTION real_t
  hugoniot_slope_derivative(const real_t ev) const
  {
    return m_params.s1 + 2 * m_params.s2 * ev + 3 * m_params.s3 * ev * ev;
  }

  /**
   * Compute reference pressure.
   */
  KOKKOS_INLINE_FUNCTION real_t
  pressure_ref(const real_t rho) const
  {
    const auto ev = 1 - m_params.rho0 / rho;

    if (ev > 0) // COMPRESSION
    {
      const auto s = hugoniot_slope(ev);
      const auto d = 1 - s * ev;
      return m_params.rho0 * m_params.c02 * ev / (d * d);
    }
    else // RELEASE
    {
      return m_params.rho0 * m_params.c02 * ev / (1 - ev);
    }

    return ZERO_F;

  } // pressure_ref

  /**
   * Compute reference internal energy.
   */
  KOKKOS_INLINE_FUNCTION real_t
  eint_ref(const real_t rho) const
  {
    const auto ev = 1 - m_params.rho0 / rho;

    if (ev > 0) // COMPRESSION
    {
      const auto s = hugoniot_slope(ev);
      const auto d = 1 - s * ev;

      // reference pressure on Hugoniot
      const auto phi = m_params.rho0 * m_params.c02 * ev / (d * d);

      return m_params.e0 + phi * ev / (2 * m_params.rho0);
    }
    else // RELEASE
    {
      return m_params.e0;
    }

    return ZERO_F;

  } // eint_ref

  /**
   * Compute pressure from volumic internal energy
   */
  KOKKOS_INLINE_FUNCTION real_t
  pressure_from_volumic_eint(real_t eint_volumic, real_t rho) const
  {
    return pressure_from_specific_eint(eint_volumic / rho, rho);
  }

  /**
   * Compute pressure from specific internal energy
   */
  KOKKOS_INLINE_FUNCTION real_t
  pressure_from_specific_eint(real_t eint_specific, real_t rho) const
  {

    if (rho > 0)
    {
      const auto ev = 1 - m_params.rho0 / rho;
      const auto Gamma = gamma_from_ev(ev);
      const auto dpde = Gamma * rho;

      if (ev > 0) // COMPRESSION
      {
        const auto s = hugoniot_slope(ev);
        const auto d = 1 - s * ev;

        // reference pressure on Hugoniot
        const auto phi = m_params.rho0 * m_params.c02 * ev / (d * d);

        // reference internal energy
        const auto eh = m_params.e0 + phi * ev / (2 * m_params.rho0);

        // return pressure
        return phi + dpde * (eint_specific - eh);
      }
      else // RELEASE
      {
        const auto phi = m_params.rho0 * m_params.c02 * ev / (1 - ev);
        const auto eh = m_params.e0;

        // return pressure
        return phi + dpde * (eint_specific - eh);
      }
    }
    else
    {
      return KALYPSSO_NUM(0.0);
    }
  } // pressure_from_specific_eint

  /**
   * Compute volumic internal energy from pressure
   */
  KOKKOS_INLINE_FUNCTION real_t
  volumic_eint_from_pressure(real_t pressure, real_t rho) const
  {
    return specific_eint_from_pressure(pressure, rho) * rho;
  }

  /**
   * Compute specific internal energy from pressure and density
   */
  KOKKOS_INLINE_FUNCTION real_t
  specific_eint_from_pressure(real_t pressure, real_t rho) const
  {
    real_t eint_specific = KALYPSSO_NUM(0.0);

    if (rho > 0)
    {
      const auto ev = 1 - m_params.rho0 / rho;
      const auto Gamma = gamma_from_ev(ev);
      const auto dpde = Gamma * rho;

      if (ev > 0) // COMPRESSION
      {
        const auto s = hugoniot_slope(ev);
        const auto d = 1 - s * ev;

        // reference pressure on Hugoniot
        const auto phi = m_params.rho0 * m_params.c02 * ev / (d * d);

        // reference internal energy
        const auto eh = m_params.e0 + phi * ev / (2 * m_params.rho0);

        eint_specific = eh + (pressure - phi) / dpde;
      }
      else // RELEASE
      {
        const auto phi = m_params.rho0 * m_params.c02 * ev / (1 - ev);
        const auto eh = m_params.e0;

        eint_specific = eh + (pressure - phi) / dpde;
      }
    }

    return eint_specific;

  } // specific_eint_from_pressure

  /**
   * Speed of sound in the case of Mie-Gruneisen EOS can be obtained e.g., by using equation (12) of
   * M.A. Price et al, A method for compressible multimaterial flows with condensed phase explosive
   * detonation and airblast on unstructured grids (https://doi.org/10.1016/j.compfluid.2015.01.006)
   *
   */
  KOKKOS_INLINE_FUNCTION
  real_t
  sound_speed(real_t pressure, real_t rho) const
  {
    real_t c2 = ZERO_F;

    if (rho > 0)
    {
      const auto ev = 1 - m_params.rho0 / rho;

      const auto Gamma = m_params.Gamma0 * (1 - ev) + m_params.b * ev;
      const auto dpde = Gamma * rho;
      const auto dgam = m_params.rho0 * (m_params.Gamma0 - m_params.b);

      if (ev > 0) // COMPRESSION
      {
        const auto s = hugoniot_slope(ev);
        const auto d = 1 - s * ev;
        const auto phi = m_params.rho0 * m_params.c02 * ev / (d * d);
        const auto eh = m_params.e0 + phi * ev / (2 * m_params.rho0);

        const auto eint_specific = eh + (pressure - phi) / dpde;
        const auto s_der = hugoniot_slope_derivative(ev);

        constexpr auto min_real = Kokkos::Experimental::norm_min_v<real_t>;

        const auto dphi = phi * m_params.rho0 * (-1 / (ev + min_real) - 2 * s_der / d);
        const auto deh = phi * (-1 - ev * s_der / d);
        const auto dpdv =
          dphi + (dgam - Gamma * rho) * (eint_specific - eh) * rho - Gamma * rho * deh;

        // return sound speed
        c2 = (pressure * dpde - dpdv) / (rho * rho);
      }
      else // RELEASE
      {
        const auto phi = m_params.rho0 * m_params.c02 * ev / (1 - ev);
        const auto eh = m_params.e0;

        const auto eint_specific = eh + (pressure - phi) / dpde;

        const auto dphi = -m_params.c02 * rho * rho;
        const auto dpdv = dphi + (dgam - dpde) * (eint_specific - eh) * rho;

        // return sound speed
        c2 = (pressure * dpde - dpdv) / (rho * rho);
      }
    }

    return std::sqrt(c2);

  } // sound_speed

  /**
   * intermediate value useful for computing isentropic bulk modulus.
   */
  KOKKOS_INLINE_FUNCTION
  real_t
  dP_drho_e(real_t eint_specific, real_t rho) const
  {
    const auto ev = 1 - m_params.rho0 / rho;
    const auto Gamma = m_params.Gamma0 * (1 - ev) + m_params.b * ev;

    if (ev > 0) // COMPRESSION
    {
      const auto s = hugoniot_slope(ev) * ev;
      const auto ds = hugoniot_slope_derivative(ev);
      const auto dev = m_params.rho0 / (rho * rho);
      const auto dGamma = (m_params.b - m_params.Gamma0) * dev;
      const auto phi = m_params.rho0 * m_params.c02 * ev / (1 - ev);
      const auto dphi =
        m_params.c02 * m_params.rho0 / ((1 - s) * (1 - s)) * dev * (1 + 2 * ev * ds / (1 - s));
      const auto eh = m_params.e0 + phi * ev / (2 * m_params.rho0);
      const auto deh = dev * phi / m_params.rho0 / 2 + ev / m_params.rho0 / 2 * dphi;
      return dphi + Gamma * (eint_specific - eh) + rho * dGamma * (eint_specific - eh) -
             rho * Gamma * deh;
    }
    else // RELEASE
    {
      return m_params.c02 + Gamma * eint_specific;
    }

    return ZERO_F;

  } // dP_drho_e

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
    const auto eint_specific = specific_eint_from_pressure(pressure, rho);

    const auto ev = 1 - m_params.rho0 / rho;
    const auto Gamma = m_params.Gamma0 * (1 - ev) + m_params.b * ev;

    if (ev > 0) // COMPRESSION
    {
      const auto dPdr_e = dP_drho_e(eint_specific, rho);
      const auto dPde_r = rho * Gamma;

      // thermodynamic identity
      return rho * dPdr_e + pressure / rho * dPde_r;
    }
    else // RELEASE
    {
      return rho * m_params.c02 + m_params.Gamma0 * (rho * eint_specific + pressure);
    }

    return ZERO_F;

  } // bulk_modulus

}; // struct MieGruneisenEosSW

} // namespace eos

} // namespace core

} // namespace kalypsso

#endif // KALYPSSO_CORE_EOS_MIE_GRUNEISEN_EOS_SW_H_
