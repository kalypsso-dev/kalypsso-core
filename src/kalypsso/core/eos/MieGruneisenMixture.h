// SPDX-FileCopyrightText: 2026 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file MieGruneisenMixture.h
 *
 * This is only useful when using at least two material to define pressure and
 * internal energy in mixed cells.
 *
 * Reference for mixture rules (isobaric closure):
 *
 * Law and Barton, A cell centered volume-of-fluid method for compressible
 * multi-material flows. Journal of Computational Physics, 497 (2024) 112592.
 * https://doi.org/10.1016/j.jcp.2023.112592
 */
#ifndef KALYPSSO_CORE_EOS_MIE_GRUNEISEN_MIXTURE_H_
#define KALYPSSO_CORE_EOS_MIE_GRUNEISEN_MIXTURE_H_

#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/real_type.h>
#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/core/eos/eos_utils.h>
#include <kalypsso/core/DataArray.h>

#include <kalypsso/core/eos/MieGruneisenEosArray.h>

namespace kalypsso
{

namespace core
{

namespace eos
{

// ==========================================================================
// ==========================================================================
// ==========================================================================
/**
 * Mie-Gruneisen mixture (isobaric closure) helper class.
 *
 * In a multifluid simulation, when each fluid obeys an EOS of Mie-Gruneisen type,
 * one needs to define mixture rule to give an averaged (mixed) thermodynamical states
 * in mixed cells, i.e. where more than 1 materials are present.
 *
 * Mie-Gruneisen EOS for material \f$ i \f$
 *
 * \f$ p_i = p_{i,\text{ref}} + \rho_i \Gamma_i \left( \mathcal{E}_i - \mathcal{E}_{i,\text{ref}}
 * \right)\f$
 *
 * Main mixture relations are:
 *
 * - volumic fraction conservation : \f$ \sum_i \alpha_i = 1 \f$
 * - density : \f$ \rho = \sum_i \alpha_i \rho_i \f$
 * - volumic internal energy : \f$ \rho \mathcal{E} = \sum_i $\alpha_i \rho_i \mathcal{E}_i \f$
 * - pressure : \f$ \forall i, p_i = p \f$
 *
 * The mixture Gruneisen parameter is defined as :
 *
 * \f$ \Gamma_{\text{mix}} = \frac{1}{\sum_i \frac{\alpha_i}{\Gamma_i}} \f$
 *
 * where \f$ \alpha_i \f$ is the volume fraction of material \f$ i \f$ and \f$ \Gamma_i \f$
 * the Gruneisen parameter of material \f$ i \f$.
 *
 * The mixture isentropic bulk modulus is defined as :
 *
 * \f$ K_S = \frac{1}{\sum_i \frac{\alpha_i}{\K_{S,i}} } \f$
 *
 * The mixture volumic internal energy is given by:
 *
 * \f$ \rho \mathcal{E} = \sum_i $\alpha_i \rho_i eint_i \f$
 *
 * The mixture pressure (assuming isobaric closure) is given by:
 *
 * \f$ p = \Gamma_{\text{mix}} \left( \rho \mathcal{E} - \sum_i \alpha_i (\rho_i
 * \mathcal{E}_{i,\text{ref}} - \frac{p_{i,\text{ref}}}{\Gamma_i}) \right)\f$
 *
 */
template <typename device_t>
class MieGruneisenMixture
{

public:
  MieGruneisenMixture() = default;

  MieGruneisenMixture(ConfigMap const & config_map);

private:
  //! EOS array
  MieGruneisenEosArray<device_t> m_eos_array;

  //! small pressure guard
  const real_t m_small_p;

  //! low value for a volume fraction
  static constexpr real_t LOW_ALPHA = 1e-9;

public:
  /**
   * Compute mixture Gruneisen parameter (two materials).
   *
   * \param[in] alpha0 volume fraction of material 0
   * \param[in] alpha1 volume fraction of material 1
   * \param[in] alpha_rho0 partial density of material 0
   * \param[in] alpha_rho1 partial density of material 1
   */
  KOKKOS_FUNCTION
  real_t
  mixture_gruneisen_param(real_t alpha0, real_t alpha1, real_t alpha_rho0, real_t alpha_rho1) const;

  /**
   * Compute mixture pressure.
   *
   * To be used only when there are two materials
   *
   * \param[in] rho mixture density
   * \param[in] eint mixture specific internal energy
   * \param[in] alpha0 volume fraction of material 0
   * \param[in] alpha1 volume fraction of material 1
   * \param[in] alpha_rho0 partial density of material 0
   * \param[in] alpha_rho1 partial density of material 1
   */
  KOKKOS_FUNCTION
  real_t
  mixture_pressure(real_t rho,
                   real_t eint,
                   real_t alpha0,
                   real_t alpha1,
                   real_t alpha_rho0,
                   real_t alpha_rho1) const;

  /**
   * Compute mixture volumic internal energy.
   *
   * To be used only when there are two materials
   *
   * \param[in] rho mixture density
   * \param[in] pressure mixture pressure
   * \param[in] alpha0 volume fraction of material 0
   * \param[in] alpha1 volume fraction of material 1
   * \param[in] alpha_rho0 partial density of material 0
   * \param[in] alpha_rho1 partial density of material 1
   */
  KOKKOS_FUNCTION
  real_t
  mixture_volumic_eint(real_t rho,
                       real_t pressure,
                       real_t alpha0,
                       real_t alpha1,
                       real_t alpha_rho0,
                       real_t alpha_rho1) const;

  /**
   * Compute mixture specific internal energy.
   *
   * To be used only when there are two materials
   *
   * \param[in] rho mixture density
   * \param[in] pressure mixture pressure
   * \param[in] alpha0 volume fraction of material 0
   * \param[in] alpha1 volume fraction of material 1
   * \param[in] alpha_rho0 partial density of material 0
   * \param[in] alpha_rho1 partial density of material 1
   */
  KOKKOS_FUNCTION
  real_t
  mixture_specific_eint(real_t rho,
                        real_t pressure,
                        real_t alpha0,
                        real_t alpha1,
                        real_t alpha_rho0,
                        real_t alpha_rho1) const;


  /**
   * Compute mixture squared speed of sound.
   *
   * To be used only when there are two materials
   *
   * \param[in] rho mixture density
   * \param[in] eint mixture internal energy
   * \param[in] alpha0 volume fraction of material 0
   * \param[in] alpha1 volume fraction of material 1
   * \param[in] alpha_rho0 partial density of material 0
   * \param[in] alpha_rho1 partial density of material 1
   */
  KOKKOS_FUNCTION
  real_t
  mixture_sound_speed_square(real_t rho,
                             real_t pressure,
                             real_t alpha0,
                             real_t alpha1,
                             real_t alpha_rho0,
                             real_t alpha_rho1) const;

  /**
   * Compute mixture speed of sound.
   *
   * To be used only when there are two materials
   *
   * \param[in] rho mixture density
   * \param[in] eint mixture internal energy
   * \param[in] alpha0 volume fraction of material 0
   * \param[in] alpha1 volume fraction of material 1
   * \param[in] alpha_rho0 partial density of material 0
   * \param[in] alpha_rho1 partial density of material 1
   */
  KOKKOS_FUNCTION
  real_t
  mixture_sound_speed(real_t rho,
                      real_t pressure,
                      real_t alpha0,
                      real_t alpha1,
                      real_t alpha_rho0,
                      real_t alpha_rho1) const;

  /**
   * Compute mixture isentropic bulk modulus.
   */
  KOKKOS_FUNCTION
  real_t
  mixture_bulk_modulus(real_t rho,
                       real_t pressure,
                       real_t alpha0,
                       real_t alpha1,
                       real_t alpha_rho0,
                       real_t alpha_rho1) const;

}; // class MieGruneisenMixture

// explicit template instantiation
#ifdef KALYPSSO_CORE_INSTANTIATE_HOST_TEMPLATE
extern template class MieGruneisenMixture<HostDevice>;
#endif
extern template class MieGruneisenMixture<DefaultDevice>;

} // namespace eos
} // namespace core
} // namespace kalypsso

#endif // KALYPSSO_CORE_EOS_MIE_GRUNEISEN_MIXTURE_H_
