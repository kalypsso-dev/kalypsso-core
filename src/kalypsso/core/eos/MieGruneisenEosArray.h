// SPDX-FileCopyrightText: 2026 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file MieGruneisenEosArray.h
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
#ifndef KALYPSSO_CORE_EOS_MIE_GRUNEISEN_EOS_ARRAY_H_
#define KALYPSSO_CORE_EOS_MIE_GRUNEISEN_EOS_ARRAY_H_

#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/real_type.h>
#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/core/eos/eos_utils.h>
#include <kalypsso/core/DataArray.h>

#include <kalypsso/core/eos/MieGruneisenEosCochranChan.h>
#include <kalypsso/core/eos/MieGruneisenEosJWL.h>
#include <kalypsso/core/eos/MieGruneisenEosSW.h>
#include <kalypsso/core/eos/MieGruneisenEosSW2.h>
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

//! type alias for an array of Mie-Gruneisen EOS of type SW2 (Shockwave v2)
template <typename device_t>
using mg_eos_sw2_array_t = Kokkos::View<MieGruneisenEosSW2 *, device_t>;

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
 * Mie-Gruneisen EOS array.
 *
 * Purpose: provide storage for all material EOS, a give access to each material
 * individual thermodynamic properties.
 */
template <typename device_t>
class MieGruneisenEosArray
{

public:
  MieGruneisenEosArray() = default;

  MieGruneisenEosArray(ConfigMap const & config_map);

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

  //! array of eos, one per material (shockwave v2)
  mg_eos_sw2_array_t<device_t> m_mg_eos_sw2;

  //! array of eos, one per material (Cochran-Chan)
  mg_eos_cc_array_t<device_t> m_mg_eos_cc;

  //! array of eos, one per material (JWL)
  mg_eos_jwl_array_t<device_t> m_mg_eos_jwl;

  //! type alias for material eos type array
  //! apparently we can't use a better-enum as a Kokkos::View data type on device (nvcc KO)
  // using material_eos_type_t = DataArray<MG_EOS_TYPE, device_t>; // KO
  using material_eos_type_t = DataArray<MG_EOS_TYPE::_integral, device_t>;

  //! type alias for material eos id array
  using material_eos_id_t = DataArray<int, device_t>;

  //! material eos type array
  material_eos_type_t m_material_eos_type;

  //! material eos id.
  //! material id among all material with the same eos
  material_eos_id_t m_material_eos_id;

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
  material_gruneisen_param(size_t i_mat, real_t rho) const;

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
  material_specific_eint_ref(size_t i_mat, real_t rho) const;

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
  material_pressure_ref(size_t i_mat, real_t rho) const;

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
  material_sound_speed_square(size_t i_mat, real_t pressure, real_t rho) const;

  /**
   * Retrieve isentropic bulk modulus of a given material.
   */
  KOKKOS_FUNCTION
  real_t
  material_bulk_modulus(size_t i_mat, real_t pressure, real_t rho) const;

  /**
   * Retrieve specific internal energy of a given material from pressure.
   */
  KOKKOS_FUNCTION
  real_t
  material_specific_eint_from_pressure(size_t i_mat, real_t pressure, real_t rho) const;

  /**
   * Retrieve pressure of a given material from specific internal energy.
   */
  KOKKOS_FUNCTION
  real_t
  material_pressure_from_specific_eint(size_t i_mat, real_t eint_specific, real_t rho) const;

}; // class MieGruneisenEosArray

// explicit template instantiation
#ifdef KALYPSSO_CORE_INSTANTIATE_HOST_TEMPLATE
extern template class MieGruneisenEosArray<HostDevice>;
#endif
extern template class MieGruneisenEosArray<DefaultDevice>;

} // namespace eos
} // namespace core
} // namespace kalypsso

#endif // KALYPSSO_CORE_EOS_MIE_GRUNEISEN_EOS_ARRAY_H_
