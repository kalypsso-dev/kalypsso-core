// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file UserDataRemapperImplMD.h
 *
 * \brief Implement actual user data remapping (after mesh change during AMR cycle) for material
 * data.
 */

#ifndef KALYPSSO_CORE_USERDATAREMAPPER_IMPL_MD_H_
#define KALYPSSO_CORE_USERDATAREMAPPER_IMPL_MD_H_

#include <kalypsso/core/MaterialPresence.h>
#include <kalypsso/core/prolongation.h>
#include <kalypsso/core/orchard_key_base.h>
#include <kalypsso/core/amr_hashmap.h>
#include <kalypsso/core/StencilHelper.h>

namespace kalypsso
{

template <size_t dim, typename device_t>
class UserDataRemapperImplMD
{
public:
  using AmrHashmap_t = typename hashmap_base_t<device_t>::map_t;
  using OrchardKeys_t = typename orchard_key_base_t<device_t>::view_t;
  using MaterialPresenceView_t = MaterialPresenceView<device_t>;
  using DataArrayBlockMultiVar_t = DataArrayBlockMultiVar<dim, real_t, device_t>;
  using StencilHelper_t = StencilHelper<dim, device_t>;

  UserDataRemapperImplMD(DataArrayBlockMultiVar_t old_data,
                         MaterialPresenceView_t   old_mat,
                         DataArrayBlockMultiVar_t new_data,
                         MaterialPresenceView_t   new_mat,
                         AmrHashmap_t             amr_hashmap_device_old,
                         OrchardKeys_t            orchard_keys_device_old,
                         OrchardKeys_t            orchard_keys_device_new,
                         const uint32_t           num_vars_per_mat,
                         const ConfigMap &        config_map);

  KOKKOS_FUNCTION void
  operator()(const int32_t i_new) const;

private:
  // ==============================================================================================
  // ==============================================================================================
  // INNER FUNCTIONS ADAPTED FROM UserDataRemapperImplBCC
  // ==============================================================================================
  // ==============================================================================================
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 2), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  linear_extrapolate_using_limited_slopes(CellLocation<2> const & cell_loc_old,
                                          coord_t<2> const &      coord_new,
                                          int32_t const &         i_new_cell,
                                          int32_t const &         i_new_oct) const;

  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 3), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  linear_extrapolate_using_limited_slopes(CellLocation<3> const & cell_loc_old,
                                          coord_t<3> const &      coord_new,
                                          int32_t const &         i_new_cell,
                                          int32_t const &         i_new_oct) const;

  //! Old data
  DataArrayBlockMultiVar_t m_old_data;

  //! Old material presence
  MaterialPresenceView_t m_old_mat;

  //! New data
  DataArrayBlockMultiVar_t m_new_data;

  //! New material presence
  MaterialPresenceView_t m_new_mat;

  //! AMR unordered map which map orchard keys to octant number for all key in the mesh
  //! before AMR cycle modification
  AmrHashmap_t m_amr_hashmap_device_old;

  //! list of orchard key of the old mesh (before AMR cycle modification)
  OrchardKeys_t m_orchard_keys_device_old;

  //! list of orchard key of the new mesh (after AMR cycle modification)
  OrchardKeys_t m_orchard_keys_device_new;

  //! Stencil helper
  StencilHelper_t m_stencil_helper;

  //! prolongation type (simple copy or linear extrapolation)
  CellCenteredProlongationType m_prolongation;

  //! Num vars per material
  int32_t m_num_vars_per_mat;

  //! block sizes
  block_size_t<dim> m_block_sizes;
};

extern template class UserDataRemapperImplMD<2, kalypsso::DefaultDevice>;
extern template class UserDataRemapperImplMD<3, kalypsso::DefaultDevice>;

#ifdef KALYPSSO_CORE_INSTANTIATE_HOST_TEMPLATE
extern template class UserDataRemapperImplMD<2, HostDevice>;
extern template class UserDataRemapperImplMD<3, HostDevice>;
#endif // KALYPSSO_CORE_INSTANTIATE_HOST_TEMPLATE

} // namespace kalypsso

#endif // KALYPSSO_CORE_USERDATAREMAPPER_IMPL_MD_H_
