// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file UserDataRemapper.cpp
 * \brief \copybrief UserDataRemapper.h
 */
#include <kalypsso/core/UserDataRemapper.h>
#include <kalypsso/core/FaceDataArrayBlock_utils.h>

#include <kalypsso/core/UserDataRemapperImplBCC.h>
#include <kalypsso/core/UserDataRemapperImplBFC.h>
#include <kalypsso/core/UserDataRemapperImplMP.h>
#include <kalypsso/core/UserDataRemapperImplMD.h>

namespace kalypsso
{

// ===========================================================================
// ===========================================================================
template <size_t dim, typename device_t>
void
UserDataRemapper<dim, device_t>::remap_leaf_data(
  typename UserDataRemapper<dim, device_t>::ExecutionSpace const & exec_space,
  DataArrayLeaf_t                                                  userdataLeaf_old,
  DataArrayLeaf_t                                                  userdataLeaf_new)
{
  // avoid warning "Implicit capture of 'this' in extended lambda expression"
  // by the cuda compiler
  auto amr_hashmap_device_old = m_amr_hashmap_device_old;
  auto orchard_keys_device_new = m_orchard_keys_device_new;

  auto num_vars = userdataLeaf_old.extent(1);

  // traverse the new list of keys, and for each key try to find it in the old hashmap
  Kokkos::parallel_for(
    "Userdata_remap_after_amr_cycle_hashmap - leaf",
    Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, m_local_num_octants_new),
    KOKKOS_LAMBDA(const int iOct) {
      auto key_new = orchard_keys_device_new(iOct);

      auto key_index = amr_hashmap_device_old.find(key_new);

      // printf("%d %d | %d\n", i, key_new, key_index);

      // first check if key exists in the old hashmap, if it exists, it means we have a
      // an octant that didn't change level
      if (amr_hashmap_device_old.valid_at(key_index))
      {
        auto iOct_old = amr_hashmap_device_old.value_at(key_index);
        for (uint32_t ivar = 0; ivar < num_vars; ++ivar)
          userdataLeaf_new(iOct, ivar) = userdataLeaf_old(iOct_old, ivar);
      }
      else
      {
        // we either have a refinement or coarsening

        // check if the key correspond to a refinement (increase of level)
        // to do that we search for father octant's key
        auto key_to_test = orchard_key_t<dim>::father(key_new);
        key_index = amr_hashmap_device_old.find(key_to_test);

        if (amr_hashmap_device_old.valid_at(key_index))
        {
          auto iOct_old = amr_hashmap_device_old.value_at(key_index);
          // copy the value from the old mesh
          for (uint32_t ivar = 0; ivar < num_vars; ++ivar)
            userdataLeaf_new(iOct, ivar) = userdataLeaf_old(iOct_old, ivar);
        }
        else
        {
          // check if the key correspond to a "coarsening" (decrease of level)
          // to do that we just need to search for the eldest child key in the old map,
          // and then average all the values from all the children (coarse graining values)

          /*
           * In principe, we should check that all the children are available in the map
           * but here, we just check for the eldest child key, and assume all the children
           * are also in the map (p4est ensure that).
           */

          // auto children_keys = orchard_key_t<dim>::all_children(key_new);
          // for (int ichild = 0; ichild < 0; ++ichild)
          // {
          // check that child key exist in old map
          // TODO
          // }

          constexpr auto NB_CHILDREN = orchard_key_t<dim>::NB_CHILDREN;

          auto eldest_child_key = orchard_key_t<dim>::eldest_child(key_new);
          key_index = amr_hashmap_device_old.find(eldest_child_key);
          if (amr_hashmap_device_old.valid_at(key_index))
          {
            auto iOct_old = amr_hashmap_device_old.value_at(key_index);

            for (uint32_t ivar = 0; ivar < num_vars; ++ivar)
            {
              auto coarsen_value = ZERO_F;
              // average value since we are coarsening
              // WARNING : here we assume all the children are available in the map
              // if this is not true, it means the map is corrupted => should abort
              // in principle, this is highly not probable because p4est already checked that
              // the whole family of octant are present
              for (int64_t ichild = 0; ichild < NB_CHILDREN; ++ichild)
              {
                coarsen_value += userdataLeaf_old(iOct_old + ichild, ivar);
              }
              userdataLeaf_new(iOct, ivar) = coarsen_value / NB_CHILDREN;
            } // end for ivar
          }
        } // end refine or coarsen
      }
    });

} // UserDataRemapper<dim, device_t>::remap_leaf_data

// ===========================================================================
// ===========================================================================
template <size_t dim, typename device_t>
void
UserDataRemapper<dim, device_t>::remap_block_data(
  [[maybe_unused]] typename UserDataRemapper<dim, device_t>::ExecutionSpace const & exec_space,
  const DataArrayBlock<dim, real_t, device_t> userdataBlock_old,
  DataArrayBlock<dim, real_t, device_t>       userdataBlock_new)
{

  const auto nbOcts = m_local_num_octants_new;
  const auto num_cells = userdataBlock_old.num_cells();

  UserDataRemapperImplBCC<dim, device_t> functor(userdataBlock_old,
                                                 userdataBlock_new,
                                                 m_amr_hashmap_device_old,
                                                 m_orchard_keys_device_new,
                                                 m_orchard_keys_device_old,
                                                 m_local_num_octants_new,
                                                 m_block_sizes,
                                                 m_config_map);

  Kokkos::parallel_for("Userdata_remap_after_amr_cycle_hashmap - cell",
                       Kokkos::RangePolicy<ExecutionSpace>(0, num_cells * nbOcts),
                       functor);

} // UserDataRemapper<dim, device_t>::remap_block_data - DataArrayBlock_t

// ===========================================================================
// ===========================================================================
template <size_t dim, typename device_t>
void
UserDataRemapper<dim, device_t>::remap_block_data(
  [[maybe_unused]] typename UserDataRemapper<dim, device_t>::ExecutionSpace const & exec_space,
  const FaceDataArrayBlock<dim, real_t, device_t>                                   facedata_old,
  FaceDataArrayBlock<dim, real_t, device_t>                                         facedata_new)
{

  const auto num_elements_per_octant = facedata_old.num_elements_per_octant();
  const auto nbOcts = m_local_num_octants_new;

  UserDataRemapperImplBFC<dim, device_t> functor(facedata_old,
                                                 facedata_new,
                                                 m_amr_hashmap_device_old,
                                                 m_orchard_keys_device_new,
                                                 m_orchard_keys_device_old,
                                                 m_local_num_octants_new,
                                                 m_block_sizes,
                                                 m_config_map);

  using TagComputeAllButInternalFaces =
    typename UserDataRemapperImplBFC<dim, device_t>::TagComputeAllButInternalFaces;

  // traverse the new list of keys, and for each key try to find it in the old hashmap
  Kokkos::parallel_for(
    "Userdata_remap_after_amr_cycle_hashmap - face data - all but internal face prolongation",
    Kokkos::RangePolicy<ExecutionSpace, TagComputeAllButInternalFaces>(
      0, num_elements_per_octant * nbOcts),
    functor);

  using TagComputeInternalFaces =
    typename UserDataRemapperImplBFC<dim, device_t>::TagComputeInternalFaces;

  Kokkos::parallel_for(
    "Userdata_remap_after_amr_cycle_hashmap - face data - all but internal face prolongation",
    Kokkos::RangePolicy<ExecutionSpace, TagComputeInternalFaces>(0,
                                                                 num_elements_per_octant * nbOcts),
    functor);

} // UserDataRemapper<dim, device_t>::remap_block_data - FaceDataArrayBlock_t

// ===========================================================================
// ===========================================================================
template <size_t dim, typename device_t>
void
UserDataRemapper<dim, device_t>::remap_material_presence(
  [[maybe_unused]] ExecutionSpace const & exec_space,
  const MaterialPresenceView_t            mat_pres_old,
  MaterialPresenceView_t                  mat_pres_new)
{
  UserDataRemapperImplMP<dim, device_t> functor(mat_pres_old,
                                                mat_pres_new,
                                                m_amr_hashmap_device_old,
                                                m_orchard_keys_device_old,
                                                m_orchard_keys_device_new,
                                                m_config_map);

  Kokkos::parallel_for("kalypsso::UserDataRemapperImplMP",
                       Kokkos::RangePolicy<ExecutionSpace>(0, m_local_num_octants_new),
                       functor);
}

// ===========================================================================
// ===========================================================================
template <size_t dim, typename device_t>
void
UserDataRemapper<dim, device_t>::remap_material_data(
  [[maybe_unused]] ExecutionSpace const & exec_space,
  const DataArrayBlockMultiVar_t          mat_data_old,
  const MaterialPresenceView_t            mat_pres_old,
  DataArrayBlockMultiVar_t                mat_data_new,
  const MaterialPresenceView_t            mat_pres_new,
  const uint32_t                          num_vars_per_mat)
{
  UserDataRemapperImplMD<dim, device_t> functor(mat_data_old,
                                                mat_pres_old,
                                                mat_data_new,
                                                mat_pres_new,
                                                m_amr_hashmap_device_old,
                                                m_orchard_keys_device_old,
                                                m_orchard_keys_device_new,
                                                num_vars_per_mat,
                                                m_config_map);

  Kokkos::parallel_for(
    "kalypsso::UserDataRemapperImplMD",
    Kokkos::RangePolicy<ExecutionSpace>(0, mat_data_new.num_cells() * m_local_num_octants_new),
    functor);
}

// explicit template instantiation
template class UserDataRemapper<2, DefaultDevice>;
template class UserDataRemapper<3, DefaultDevice>;

#ifdef KALYPSSO_CORE_INSTANTIATE_HOST_TEMPLATE
template class UserDataRemapper<2, HostDevice>;
template class UserDataRemapper<3, HostDevice>;
#endif // KALYPSSO_CORE_INSTANTIATE_HOST_TEMPLATE

} // namespace kalypsso
