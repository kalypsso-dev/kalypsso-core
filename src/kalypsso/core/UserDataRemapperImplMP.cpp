// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file UserDataRemapperImplMP.cpp
 */

#include <kalypsso/core/UserDataRemapperImplMP.h>
#include <kalypsso/core/FillBlockGhosts_common.h> // for compute_face_neighbor_key_finer

namespace kalypsso
{

// ================================================================================================
// ================================================================================================
template <size_t dim, typename device_t>
UserDataRemapperImplMP<dim, device_t>::UserDataRemapperImplMP(const MaterialPresenceView_t old_mat,
                                                              MaterialPresenceView_t       new_mat,
                                                              AmrHashmap_t  amr_hashmap_device_old,
                                                              OrchardKeys_t orchard_keys_device_old,
                                                              OrchardKeys_t orchard_keys_device_new,
                                                              const ConfigMap & config_map)
  : m_old_mat(old_mat)
  , m_new_mat(new_mat)
  , m_amr_hashmap_device_old(amr_hashmap_device_old)
  , m_orchard_keys_device_old(orchard_keys_device_old)
  , m_orchard_keys_device_new(orchard_keys_device_new)
  , m_brick_size(get_brick_sizes<dim>(config_map))
  , m_brick_periodicity(get_brick_periodicity<dim>(config_map))
  , m_prolongation(get_cell_prolongation_type(config_map))
{}

// ================================================================================================
// ================================================================================================
template <size_t dim, typename device_t>
KOKKOS_FUNCTION void
UserDataRemapperImplMP<dim, device_t>::operator()(const int32_t i_new_oct) const
{
  const auto new_key = m_orchard_keys_device_new(i_new_oct);
  const auto new_key_index = m_amr_hashmap_device_old.find(new_key);

  // It hasn't changed
  if (m_amr_hashmap_device_old.valid_at(new_key_index))
  {
    const auto i_old_oct = static_cast<int32_t>(m_amr_hashmap_device_old.value_at(new_key_index));
    MaterialPresenceView_t::copy(m_new_mat, i_new_oct, m_old_mat, i_old_oct);
    return;
  }

  const auto new_father_key = orchard_key_t<dim>::father(new_key);
  const auto new_father_key_index = m_amr_hashmap_device_old.find(new_father_key);

  // It was refined
  if (m_amr_hashmap_device_old.valid_at(new_father_key_index))
  {
    const auto i_old_oct =
      static_cast<int32_t>(m_amr_hashmap_device_old.value_at(new_father_key_index));
    MaterialPresenceView_t::copy(m_new_mat, i_new_oct, m_old_mat, i_old_oct);

    // If the prolongation is to extrapolate using a linear min-mod, we must create space for the
    // new materials on the borders
    if (m_prolongation == +CellCenteredProlongationType::EXTRAPOLATE_LINEAR_MINMOD)
      for (uint8_t face = 0; face < Face::num_faces<dim>(); face++)
        get_mat_over_face(new_father_key, face, i_new_oct);

    return;
  }

  // This means it was coarsened so we have to regroup all of the material of its children
  constexpr auto NB_CHILDREN = orchard_key_t<dim>::NB_CHILDREN;

  const auto new_child_key = orchard_key_t<dim>::eldest_child(new_key);
  const auto new_child_key_index = m_amr_hashmap_device_old.find(new_child_key);

  // Must be valid
  if (m_amr_hashmap_device_old.valid_at(new_child_key_index))
  {
    const auto i_eldest_child_oct =
      static_cast<int32_t>(m_amr_hashmap_device_old.value_at(new_child_key_index));
    for (uint8_t i = 0; i < NB_CHILDREN; i++)
      MaterialPresenceView_t::update(m_new_mat, i_new_oct, m_old_mat, i_eldest_child_oct + i);
    return;
  }
}

// ================================================================================================
// ================================================================================================
template <size_t dim, typename device_t>
KOKKOS_FUNCTION void
UserDataRemapperImplMP<dim, device_t>::get_mat_over_face(const key_t   key,
                                                         const uint8_t face,
                                                         const int32_t i_new_oct) const
{
  coord_t<dim, int8_t> dir{};
  dir[face >> 1] = (face & 1) ? 1 : -1;

  const auto neighbor_key =
    orchard_key_t<dim>::get_neighbor_key_same_level(key, dir, m_brick_size, m_brick_periodicity);
  auto neighbor_key_hash = m_amr_hashmap_device_old.find(neighbor_key);
  auto is_key_valid = m_amr_hashmap_device_old.valid_at(neighbor_key_hash);

  // Neighbor is at same level
  if (is_key_valid)
  {
    const auto i_old_oct =
      static_cast<int32_t>(m_amr_hashmap_device_old.value_at(neighbor_key_hash));
    return MaterialPresenceView_t::update(m_new_mat, i_new_oct, m_old_mat, i_old_oct);
  }

  const auto neighbor_key_coarser = orchard_key_t<dim>::father(neighbor_key);
  neighbor_key_hash = m_amr_hashmap_device_old.find(neighbor_key_coarser);
  is_key_valid = m_amr_hashmap_device_old.valid_at(neighbor_key_hash);

  // Neighbor is at coarser level
  if (is_key_valid)
  {
    const auto i_old_oct =
      static_cast<int32_t>(m_amr_hashmap_device_old.value_at(neighbor_key_hash));
    return MaterialPresenceView_t::update(m_new_mat, i_new_oct, m_old_mat, i_old_oct);
  }

  const auto neighbor_keys_finer = compute_face_neighbor_key_finer<dim>(neighbor_key, face);
  neighbor_key_hash = m_amr_hashmap_device_old.find(neighbor_keys_finer[0]);
  is_key_valid = m_amr_hashmap_device_old.valid_at(neighbor_key_hash);

  // Neighbor is at finer level
  if (is_key_valid)
  {
    const auto i_old_oct =
      static_cast<int32_t>(m_amr_hashmap_device_old.value_at(neighbor_key_hash));
    MaterialPresenceView_t::update(m_new_mat, i_new_oct, m_old_mat, i_old_oct);
    for (uint8_t i = 1; i < neighbor_keys_finer.size(); i++)
    {
      const auto hash = m_amr_hashmap_device_old.find(neighbor_keys_finer[i]);
      const auto i_old_oct_bis = static_cast<int32_t>(m_amr_hashmap_device_old.value_at(hash));
      MaterialPresenceView_t::update(m_new_mat, i_new_oct, m_old_mat, i_old_oct_bis);
    }
  }
}

// ================================================================================================
// ================================================================================================
template class UserDataRemapperImplMP<2, kalypsso::DefaultDevice>;
template class UserDataRemapperImplMP<3, kalypsso::DefaultDevice>;

#ifdef KALYPSSO_CORE_INSTANTIATE_HOST_TEMPLATE
template class UserDataRemapperImplMP<2, HostDevice>;
template class UserDataRemapperImplMP<3, HostDevice>;
#endif // KALYPSSO_CORE_INSTANTIATE_HOST_TEMPLATE

} // namespace kalypsso
