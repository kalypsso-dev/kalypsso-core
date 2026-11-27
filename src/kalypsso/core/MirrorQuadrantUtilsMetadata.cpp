// SPDX-FileCopyrightText: 2026 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file MirrorQuadrantUtilsMetadata.cpp
 */
#include <kalypsso/core/MirrorQuadrantUtilsMetadata.h>

#include <kalypsso/core/StencilHelper.h>

namespace kalypsso
{

namespace core
{

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
MirrorQuadrantUtilsMetadata<dim, device_t>::update_ghost_mpi_ranks(AMRmesh<dim> & amr_mesh)
{

  m_ghost_quadrant_mpi_ranks.resize(static_cast<size_t>(amr_mesh.local_num_ghosts()));
  auto ghost_quadrant_mpi_ranks_host =
    DataArray<int32_t, device_t>::create_host_mirror_view(m_ghost_quadrant_mpi_ranks);

  for (int32_t iproc = 0; iproc < m_par_env.nRanks(); ++iproc)
  {
    for (int32_t ighost = amr_mesh.ghost()->proc_offsets[iproc];
         ighost < amr_mesh.ghost()->proc_offsets[iproc + 1];
         ++ighost)
    {
      ghost_quadrant_mpi_ranks_host(ighost) = iproc;
    }
  }

  Kokkos::deep_copy(m_ghost_quadrant_mpi_ranks.logical_view(),
                    ghost_quadrant_mpi_ranks_host.logical_view());

} // MirrorQuadrantUtilsMetadata<dim, device_t>::update_ghost_mpi_ranks

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
MirrorQuadrantUtilsMetadata<dim, device_t>::update_mirror_mpi_ranks(AMRmesh<dim> & amr_mesh)
{

  m_mirror_quadrant_mpi_ranks.resize(static_cast<size_t>(amr_mesh.local_num_mirrors()));
  auto mirror_quadrant_mpi_ranks_host =
    DataArray<int32_t, device_t>::create_host_mirror_view(m_mirror_quadrant_mpi_ranks);

  for (int32_t iproc = 0; iproc < m_par_env.nRanks(); ++iproc)
  {
    for (int32_t imirror = amr_mesh.ghost()->mirror_proc_offsets[iproc];
         imirror < amr_mesh.ghost()->mirror_proc_offsets[iproc + 1];
         ++imirror)
    {
      mirror_quadrant_mpi_ranks_host(imirror) = iproc;
    }
  }

  Kokkos::deep_copy(m_mirror_quadrant_mpi_ranks.logical_view(),
                    mirror_quadrant_mpi_ranks_host.logical_view());

} // MirrorQuadrantUtilsMetadata<dim, device_t>::update_mirror_mpi_ranks

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
MirrorQuadrantUtilsMetadata<dim, device_t>::update_mirror_global_indexes(
  MeshMap<dim, device_t> & mesh_map,
  AMRmesh<dim> &           amr_mesh)
{

  [[maybe_unused]] auto local_num_quadrants = amr_mesh.local_num_quadrants();
  const auto            local_num_mirrors = amr_mesh.local_num_mirrors();

  m_mirror_global_indexes.resize(static_cast<size_t>(local_num_mirrors));

  const auto mirror_keys = mesh_map.mirror_orchard_keys();
  const auto amr_hashmap = mesh_map.hashmap();
  const auto mirror_global_indexes = m_mirror_global_indexes;

  Kokkos::parallel_for(
    "Fill mirror global index",
    Kokkos::RangePolicy<exec_space>(0, local_num_mirrors),
    KOKKOS_LAMBDA(const int imirror) {
      const auto key = mirror_keys(imirror);

      // find key index in map
      const auto key_index_map = amr_hashmap.find(key);

      // first check if key exists in the hashmap, if it exists, it means we found a matching
      // owned mirror quadrant
      if (amr_hashmap.valid_at(key_index_map))
      {
        const auto global_index = amr_hashmap.value_at(key_index_map);

        KOKKOS_ASSERT(global_index < local_num_quadrants &&
                      "[MirrorQuadrantUtilsMetadata::update_mirror_global_indexes] index must "
                      "correspond to a locally "
                      "owned quadrant.");

        mirror_global_indexes(imirror) = global_index;
      }
      else
      {
        // printf("[MirrorQuadrantUtilsMetadata::update_mirror_global_indexes] Invalid mirror key
        // !\n");
      }
    });

} // MirrorQuadrantUtilsMetadata<dim, device_t>::update_mirror_global_indexes

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
MirrorQuadrantUtilsMetadata<dim, device_t>::update(MeshMap<dim, device_t> & mesh_map,
                                                   AMRmesh<dim> &           amr_mesh,
                                                   ConfigMap const &        config_map)
{
  // update mirror global indexes
  update_mirror_global_indexes(mesh_map, amr_mesh);

  // update ghost and mirror mpi ranks
  update_ghost_mpi_ranks(amr_mesh);
  update_mirror_mpi_ranks(amr_mesh);

  // TODO
  m_active_neighbors.resize(0);

  const auto mirror_keys = mesh_map.mirror_orchard_keys();
  const auto keys = mesh_map.orchard_keys();
  const auto amr_hashmap = mesh_map.hashmap();

  const auto local_num_quadrants = mesh_map.get_amr_mesh_info().local_num_quadrants();

  // Kokkos::parallel_for(
  //   "compute_mirror_index_to_neighbor_global",
  //   Kokkos::RangePolicy<exec_space>(0, m_mirror_index_to_neighbor_global_index.size()),
  //   KOKKOS_CLASS_LAMBDA(const size_t index) {
  //     // mapping is index = neighbor_index + NEIGHBOR_STENCIL_SIZE * mirror_id
  //     const auto mirror_id = index / NEIGHBOR_STENCIL_SIZE;
  //     const auto neighbor_index = index - mirror_id * NEIGHBOR_STENCIL_SIZE;


  //     // auto is_active = false;
  //     // // active neighbors are neighbors that are ghost, and the mpi rank is equal to the rank
  //     // // attached to current mirror.
  //     // if (cell_loc_neigh.iOct >= local_num_quadrants and !cell_loc_neigh.is_outside_domain)
  //     // {
  //     //   auto ighost = cell_loc_neigh.iOct - local_num_quadrants;
  //     //   if (m_ghost_quadrant_mpi_ranks(ighost) == m_mirror_quadrant_mpi_ranks(mirror_id))
  //     //   {
  //     //     is_active = true;
  //     //   }
  //     // }
  //     // m_active_neighbors(index) = is_active;
  //   });
}

// explicit template instantiation
template class MirrorQuadrantUtilsMetadata<2, kalypsso::DefaultDevice>;
template class MirrorQuadrantUtilsMetadata<3, kalypsso::DefaultDevice>;

#ifdef KALYPSSO_CORE_INSTANTIATE_HOST_TEMPLATE
template class MirrorQuadrantUtilsMetadata<2, kalypsso::HostDevice>;
template class MirrorQuadrantUtilsMetadata<3, kalypsso::HostDevice>;
#endif // KALYPSSO_CORE_INSTANTIATE_HOST_TEMPLATE

} // namespace core

} // namespace kalypsso
