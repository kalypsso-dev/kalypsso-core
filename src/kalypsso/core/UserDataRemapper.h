// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file UserDataRemapper.h
 * \brief Perform user data remapping after mesh change (AMR cycle)
 */
#ifndef KALYPSSO_CORE_USERDATAREMAPPER_H_
#define KALYPSSO_CORE_USERDATAREMAPPER_H_

#include <kalypsso/core/kalypsso_core_base.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/kalypsso_data_container.h> // for DataArray, DataArrayHost

#include <kalypsso/core/orchard_key_base.h>
#include <kalypsso/core/orchard_key_utils.h>

#include <kalypsso/core/amr_hashmap.h>

#include <kalypsso/core/prolongation.h>

#include <kalypsso/core/MaterialPresence.h>

// #include <kalypsso/utils/config/ConfigMap.h>
// #include <kalypsso/utils/mpi/ParallelEnv.h>

namespace kalypsso
{

/**
 * \class UserDataRemapper
 *
 * Remap user data from the old mesh to the new mesh (after AMR cycle modifications).
 * Currently user data are considered as conservative variables.
 *
 * There are 3 situations depending on source and destination octant levels:
 * - when level(dest) == level(src) : doing a simple copy
 * - when level(dest) <  level(src) : doing a restriction (i.e. just averaging values from source)
 * - when level(dest) >  level(src) : doing a prolongation (destination is at a finer AMR level)
 *
 * Prolongation is configurable, currently we support:
 *
 * - SIMPLE_COPY :
 *   values from the source (mother cell) is directly copied into the destination cells (child
 *   cells)
 * - EXTRAPOLATE_LINEAR_MINMOD :
 *   we first estimate limited slopes at source location using source
 *   neighbors cells, then do a linear extrapolation from the mother
 *   cells to the children cells.
 *
 * This class is able to remap
 * - cell-centered data stored in a DataArrayBlock_t
 * - face-centered data stored in a FaceDataArrayBlock_t
 *
 */
template <size_t dim, typename device_t>
class UserDataRemapper
{
public:
  using DataArrayLeaf_t = DataArrayLeaf<real_t, device_t>;
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;
  using FaceDataArrayBlock_t = FaceDataArrayBlock<dim, real_t, device_t>;
  using DataArrayBlockMultiVar_t = DataArrayBlockMultiVar<dim, real_t, device_t>;
  using MaterialPresenceView_t = MaterialPresenceView<device_t>;

  using amr_hashmap_t = typename hashmap_base_t<device_t>::map_t;
  using orchard_key_view_t = typename orchard_key_base_t<device_t>::view_t;

  using ExecutionSpace = typename device_t::execution_space;

  UserDataRemapper() = delete;
  UserDataRemapper(amr_hashmap_t      amr_hashmap_old,
                   orchard_key_view_t orchard_keys_new,
                   orchard_key_view_t orchard_keys_old,
                   int32_t            local_num_octants,
                   block_size_t<dim>  bSizes,
                   ConfigMap const &  config_map)
    : m_amr_hashmap_device_old(amr_hashmap_old)
    , m_orchard_keys_device_new(orchard_keys_new)
    , m_orchard_keys_device_old(orchard_keys_old)
    , m_local_num_octants_new(local_num_octants)
    , m_block_sizes(bSizes)
    , m_config_map(config_map)
  {}

  // ===========================================================================
  // ===========================================================================
  //! Apply data remapping (on device) using the hash map approach - leaf data
  void
  remap_leaf_data(ExecutionSpace const & exec_space,
                  DataArrayLeaf_t        userdataLeaf_old,
                  DataArrayLeaf_t        userdataLeaf_new);

  // ===========================================================================
  // ===========================================================================
  //! Apply data remapping (on device) using the hash map approach - block data - cell center
  //!
  //! this only work when we don't use ghost cells at the block level.
  void
  remap_block_data(ExecutionSpace const & exec_space,
                   const DataArrayBlock_t userdataBlock_old,
                   DataArrayBlock_t       userdataBlock_new);

  // ===========================================================================
  // ===========================================================================
  //! Apply data remapping (on device) using the hash map approach - block data - face-center
  //!
  //! this only work when we don't use ghost cells at the block level.
  void
  remap_block_data(ExecutionSpace const &     exec_space,
                   const FaceDataArrayBlock_t facedata_old,
                   FaceDataArrayBlock_t       facedata_new);

  // ===========================================================================
  // ===========================================================================
  //! Apply data remapping (on device) using the hash map approach - material presence
  //!
  //! Remaps the material presence from the old material presence to the new one
  void
  remap_material_presence(ExecutionSpace const &       exec_space,
                          const MaterialPresenceView_t mat_pres_old,
                          MaterialPresenceView_t       mat_pres_new);

  // ===========================================================================
  // ===========================================================================
  //! Apply data remapping (on device) using the hash map approach - material data
  //!
  //! Remaps the material data from the old material data to the new one
  void
  remap_material_data(ExecutionSpace const &         exec_space,
                      const DataArrayBlockMultiVar_t mat_data_old,
                      const MaterialPresenceView_t   mat_pres_old,
                      DataArrayBlockMultiVar_t       mat_data_new,
                      const MaterialPresenceView_t   mat_pres_new,
                      const uint32_t                 num_vars_per_mat);

private:
  //! AMR unordered map which map orchard keys to octant number for all key in the mesh
  //! before AMR cycle modification
  amr_hashmap_t m_amr_hashmap_device_old;

  //! list of orchard key of the new mesh (after AMR cycle modification)
  orchard_key_view_t m_orchard_keys_device_new;

  //! list of orchard key of the old mesh (before AMR cycle modification)
  orchard_key_view_t m_orchard_keys_device_old;

  //! number of octants in the new mesh
  int32_t m_local_num_octants_new;

  //! block sizes
  block_size_t<dim> m_block_sizes;

  //! config map
  const ConfigMap & m_config_map;

}; // class UserDataRemapper

// explicit template instantiation
extern template class UserDataRemapper<2, DefaultDevice>;
extern template class UserDataRemapper<3, DefaultDevice>;

#ifdef KALYPSSO_CORE_INSTANTIATE_HOST_TEMPLATE
extern template class UserDataRemapper<2, HostDevice>;
extern template class UserDataRemapper<3, HostDevice>;
#endif // KALYPSSO_CORE_INSTANTIATE_HOST_TEMPLATE

} // namespace kalypsso

#endif // KALYPSSO_CORE_USERDATAREMAPPER_H_
