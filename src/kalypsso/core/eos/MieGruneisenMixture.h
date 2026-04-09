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

#include <kalypsso/core/eos/MieGruneisenEosCochranChan.h>
#include <kalypsso/core/eos/MieGruneisenEosJWL.h>
#include <kalypsso/core/eos/MieGruneisenEosSW.h>
#include <kalypsso/core/eos/MieGruneisenEosIdealGas.h>
#include <kalypsso/core/eos/MieGruneisenEosStiffenedGas.h>
#include <kalypsso/core/eos/MieGruneisenEosVanDerWaalsGas.h>


namespace kalypsso
{

namespace core
{

namespace eos
{

//! type alias for an array of Mie-Gruneisen EOS of type Cochran-Chan
template <typename device_t>
using mg_eos_cc_array_t = Kokkos::View<MieGruneisenEosCochranChan *, device_t>;

//! type alias for an array of Mie-Gruneisen EOS of type JWL
template <typename device_t>
using mg_eos_jwl_array_t = Kokkos::View<MieGruneisenEosJWL *, device_t>;

//! type alias for an array of Mie-Gruneisen EOS of type SW (Shockwave)
template <typename device_t>
using mg_eos_sw_array_t = Kokkos::View<MieGruneisenEosSW *, device_t>;

//! type alias for an array of Mie-Gruneisen EOS of type ideal gas
template <typename device_t>
using mg_eos_ig_array_t = Kokkos::View<MieGruneisenEosIdealGas *, device_t>;

//! type alias for an array of Mie-Gruneisen EOS of type stiffened gas
template <typename device_t>
using mg_eos_sg_array_t = Kokkos::View<MieGruneisenEosStiffenedGas *, device_t>;

//! type alias for an array of Mie-Gruneisen EOS of type van der Waals gas
template <typename device_t>
using mg_eos_vdw_array_t = Kokkos::View<MieGruneisenEosVanDerWaalsGas *, device_t>;

//! return the number of material of given Mie-Gruneisen type
size_t
get_number_of_eos(ConfigMap const & config_map, MG_EOS_TYPE mg_eos_type);


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
 * - volumic fraction conservation : \f$ \sum_i \phi_i = 1 \f$
 * - density : \f$ \rho = \sum_i \phi_i \rho_i \f$
 * - volumic internal energy : \f$ \rho \mathcal{E} = \sum_i $\phi_i \rho_i \mathcal{E}_i \f$
 * - pressure : \f$ \forall i, p_i = p \f$
 *
 * The mixture Gruneisen parameter is defined as :
 *
 * \f$ \Gamma_{\text{mix}} = \frac{1}{\sum_i \frac{\phi_i}{\Gamma_i}} \f$
 *
 * where \f$ \phi_i \f$ is the volume fraction of material \f$ i \f$ and \f$ \Gamma_i \f$
 * the Gruneisen parameter of material \f$ i \f$.
 *
 * The mixture isentropic bulk modulus is defined as :
 *
 * \f$ K_S = \frac{1}{\sum_i \frac{\phi_i}{\K_{S,i}} } \f$
 *
 * The mixture volumic internal energy is given by:
 *
 * \f$ \rho \mathcal{E} = \sum_i $\phi_i \rho_i eint_i \f$
 *
 * The mixture pressure (assuming isobaric closure) is given by:
 *
 * \f$ p = \Gamma_{\text{mix}} \left( \rho \mathcal{E} - \sum_i \phi_i (\rho_i
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
  //! number of materials
  size_t m_num_material;

  //! array of eos, one per material (Ideal gas)
  mg_eos_ig_array_t<device_t> m_mg_eos_ig;

  //! array of eos, one per material (stiffened gas)
  mg_eos_sg_array_t<device_t> m_mg_eos_sg;

  //! array of eos, one per material (van der waals gas)
  mg_eos_vdw_array_t<device_t> m_mg_eos_vdw;

  //! array of eos, one per material (shockwave)
  mg_eos_sw_array_t<device_t> m_mg_eos_sw;

  //! array of eos, one per material (Cochran-Chan)
  mg_eos_cc_array_t<device_t> m_mg_eos_cc;

  //! array of eos, one per material (JWL)
  mg_eos_jwl_array_t<device_t> m_mg_eos_jwl;

  //! type alias for material eos type array
  //! apparently one can't use a better-enum as a Kokkos::View data type on device (nvcc KO)
  // using material_eos_type_t = DataArray<MG_EOS_TYPE, device_t>; // KO
  using material_eos_type_t = DataArray<MG_EOS_TYPE::_integral, device_t>;

  //! type alias for material eos id array
  using material_eos_id_t = DataArray<int, device_t>;

  //! material eos type array
  material_eos_type_t m_material_eos_type;

  //! material eos id.
  //! material id among all material with the same eos
  material_eos_id_t m_material_eos_id;

  //! low value for a volume fraction
  static constexpr real_t LOW_PHI = 1e-5;

public:
  /**
   * Retrieve Gruneisen parameter of a given material.
   *
   * \param[in] i_mat material id
   * \param[in] rho material density (rho_phi/phi)
   *
   * \note the caller is responsible for computing material density and cross-checking
   * that the volume fraction is not too small
   */
  KOKKOS_FUNCTION
  real_t
  material_gruneisen_param(int i_mat, real_t rho) const;

  /**
   * Retrieve reference specific internal energy of a given material.
   *
   * \param[in] i_mat material id
   * \param[in] rho material density (rho_phi/phi)
   *
   * \note the caller is responsible for computing material density and cross-checking
   * that the volume fraction is not too small
   */
  KOKKOS_FUNCTION
  real_t
  material_specific_eint_ref(int i_mat, real_t rho) const;

  /**
   * Retrieve reference pressure of a given material.
   *
   * \param[in] i_mat material id
   * \param[in] rho material density (rho_phi/phi)
   *
   * \note the caller is responsible for computing material density and cross-checking
   * that the volume fraction is not too small
   */
  KOKKOS_FUNCTION
  real_t
  material_pressure_ref(int i_mat, real_t rho) const;

  /**
   * Retrieve sound speed square of a given material.
   *
   * \param[in] i_mat material id
   * \param[in] pressure (same for all materials, isobaric closure)
   * \param[in] rho material density (rho_phi/phi)
   *
   * \note the caller is responsible for computing material density and cross-checking
   * that the volume fraction is not too small
   */
  KOKKOS_FUNCTION
  real_t
  material_sound_speed_square(int i_mat, real_t pressure, real_t rho) const;

  /**
   * Retrieve isentropic bulk modulus of a given material.
   */
  KOKKOS_FUNCTION
  real_t
  material_bulk_modulus(int i_mat, real_t pressure, real_t rho) const;

  /**
   * Compute mixture Gruneisen parameter (two materials).
   *
   * \param[in] phi0 volume fraction of material 0
   * \param[in] phi1 volume fraction of material 1
   * \param[in] rho_phi0 partial density of material 0
   * \param[in] rho_phi1 partial density of material 1
   */
  KOKKOS_FUNCTION
  real_t
  mixture_gruneisen_param(real_t phi0, real_t phi1, real_t phi_rho0, real_t phi_rho1) const;

  /**
   * Compute mixture pressure.
   *
   * To be used only when there are two materials
   *
   * \param[in] rho mixture density
   * \param[in] eint mixture specific internal energy
   * \param[in] phi0 volume fraction of material 0
   * \param[in] phi1 volume fraction of material 1
   * \param[in] rho_phi0 partial density of material 0
   * \param[in] rho_phi1 partial density of material 1
   */
  KOKKOS_FUNCTION
  real_t
  mixture_pressure(real_t rho,
                   real_t eint,
                   real_t phi0,
                   real_t phi1,
                   real_t phi_rho0,
                   real_t phi_rho1) const;

  /**
   * Compute mixture volumic internal energy.
   *
   * To be used only when there are two materials
   *
   * \param[in] rho mixture density
   * \param[in] pressure mixture pressure
   * \param[in] phi0 volume fraction of material 0
   * \param[in] phi1 volume fraction of material 1
   * \param[in] rho_phi0 partial density of material 0
   * \param[in] rho_phi1 partial density of material 1
   */
  KOKKOS_FUNCTION
  real_t
  mixture_volumic_eint(real_t rho,
                       real_t pressure,
                       real_t phi0,
                       real_t phi1,
                       real_t phi_rho0,
                       real_t phi_rho1) const;

  /**
   * Compute mixture specific internal energy.
   *
   * To be used only when there are two materials
   *
   * \param[in] rho mixture density
   * \param[in] pressure mixture pressure
   * \param[in] phi0 volume fraction of material 0
   * \param[in] phi1 volume fraction of material 1
   * \param[in] rho_phi0 partial density of material 0
   * \param[in] rho_phi1 partial density of material 1
   */
  KOKKOS_FUNCTION
  real_t
  mixture_specific_eint(real_t rho,
                        real_t pressure,
                        real_t phi0,
                        real_t phi1,
                        real_t phi_rho0,
                        real_t phi_rho1) const;


  /**
   * Compute mixture squared speed of sound.
   *
   * To be used only when there are two materials
   *
   * \param[in] rho mixture density
   * \param[in] eint mixture internal energy
   * \param[in] phi0 volume fraction of material 0
   * \param[in] phi1 volume fraction of material 1
   * \param[in] rho_phi0 partial density of material 0
   * \param[in] rho_phi1 partial density of material 1
   */
  KOKKOS_FUNCTION
  real_t
  mixture_sound_speed_square(real_t rho,
                             real_t pressure,
                             real_t phi0,
                             real_t phi1,
                             real_t phi_rho0,
                             real_t phi_rho1) const;

  /**
   * Compute mixture speed of sound.
   *
   * To be used only when there are two materials
   *
   * \param[in] rho mixture density
   * \param[in] eint mixture internal energy
   * \param[in] phi0 volume fraction of material 0
   * \param[in] phi1 volume fraction of material 1
   * \param[in] rho_phi0 partial density of material 0
   * \param[in] rho_phi1 partial density of material 1
   */
  KOKKOS_FUNCTION
  real_t
  mixture_sound_speed(real_t rho,
                      real_t pressure,
                      real_t phi0,
                      real_t phi1,
                      real_t phi_rho0,
                      real_t phi_rho1) const;

  /**
   * Compute mixture isentropic bulk modulus.
   */
  KOKKOS_FUNCTION
  real_t
  mixture_bulk_modulus(real_t rho,
                       real_t pressure,
                       real_t phi0,
                       real_t phi1,
                       real_t phi_rho0,
                       real_t phi_rho1) const;

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
