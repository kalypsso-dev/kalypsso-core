// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file UserDataRemapperImplMP.h
 *
 * \brief Implement actual user data remapping (after mesh change during AMR cycle) for material
 * presence.
 */

#ifndef KALYPSSO_CORE_USERDATAREMAPPER_IMPL_MP_H_
#define KALYPSSO_CORE_USERDATAREMAPPER_IMPL_MP_H_

#include <kalypsso/core/MaterialPresence.h>
#include <kalypsso/core/prolongation.h>
#include <kalypsso/core/orchard_key_base.h>
#include <kalypsso/core/amr_hashmap.h>

namespace kalypsso
{

template <size_t dim, typename device_t>
class UserDataRemapperImplMP
{
public:
  using AmrHashmap_t = typename hashmap_base_t<device_t>::map_t;
  using OrchardKeys_t = typename orchard_key_base_t<device_t>::view_t;
  using MaterialPresenceView_t = MaterialPresenceView<device_t>;

  UserDataRemapperImplMP(const MaterialPresenceView_t old_mat,
                         MaterialPresenceView_t       new_mat,
                         AmrHashmap_t                 amr_hashmap_device_old,
                         OrchardKeys_t                orchard_keys_device_old,
                         OrchardKeys_t                orchard_keys_device_new,
                         const ConfigMap &            config_map);

  KOKKOS_FUNCTION void
  operator()(const int32_t i_new_oct) const;

private:
  //! Updates the material presence with the ones over the faces
  KOKKOS_FUNCTION void
  get_mat_over_face(const key_t key, const uint8_t face, const int32_t i_new_oct) const;

  //! Old material presence
  MaterialPresenceView_t m_old_mat;

  //! New material presence
  MaterialPresenceView_t m_new_mat;

  //! AMR unordered map which map orchard keys to octant number for all key in the mesh
  //! before AMR cycle modification
  AmrHashmap_t m_amr_hashmap_device_old;

  //! list of orchard key of the old mesh (before AMR cycle modification)
  OrchardKeys_t m_orchard_keys_device_old;

  //! list of orchard key of the new mesh (after AMR cycle modification)
  OrchardKeys_t m_orchard_keys_device_new;

  //! brick size
  brick_size_t<dim> m_brick_size;

  //! Brick periodicity
  Kokkos::Array<bool, dim> m_brick_periodicity;

  //! prolongation type (simple copy or linear extrapolation)
  CellCenteredProlongationType m_prolongation;
};

extern template class UserDataRemapperImplMP<2, kalypsso::DefaultDevice>;
extern template class UserDataRemapperImplMP<3, kalypsso::DefaultDevice>;

#ifdef KALYPSSO_CORE_INSTANTIATE_HOST_TEMPLATE
extern template class UserDataRemapperImplMP<2, HostDevice>;
extern template class UserDataRemapperImplMP<3, HostDevice>;
#endif // KALYPSSO_CORE_INSTANTIATE_HOST_TEMPLATE

} // namespace kalypsso

#endif // KALYPSSO_CORE_USERDATAREMAPPER_IMPL_MP_H_
