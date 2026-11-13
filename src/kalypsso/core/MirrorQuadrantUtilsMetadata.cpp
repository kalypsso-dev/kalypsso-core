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
MirrorQuadrantUtilsMetadata<dim, device_t>::update(MeshMap<dim, device_t> & mesh_map,
                                                   AMRmesh<dim> &           amr_mesh,
                                                   ConfigMap const &        config_map)
{

  // update ghost and mirror mpi ranks
  update_ghost_mpi_ranks(amr_mesh);
  update_mirror_mpi_ranks(amr_mesh);

  // update m_mirror_index_to_neighbor_global_index
  m_mirror_index_to_neighbor_global_index.resize(
    NEIGHBOR_STENCIL_SIZE * static_cast<size_t>(mesh_map.get_amr_mesh_info().local_num_mirrors()));

  m_active_neighbors.resize(NEIGHBOR_STENCIL_SIZE *
                            static_cast<size_t>(mesh_map.get_amr_mesh_info().local_num_mirrors()));

  const auto mirror_keys = mesh_map.mirror_orchard_keys();
  const auto keys = mesh_map.orchard_keys();
  const auto amr_hashmap = mesh_map.hashmap();

  const auto block_size = []() {
    if constexpr (dim == 2)
      return block_size_t<2>{ 4, 4 };
    else if constexpr (dim == 3)
      return block_size_t<3>{ 4, 4, 4 };
  }();

  using CellLocation_t = CellLocation<dim>;

  using StencilHelper_t = StencilHelper<dim, device_t>;
  auto stencil_helper = StencilHelper_t(amr_hashmap,
                                        keys,
                                        block_size,
                                        get_brick_sizes<dim>(config_map),
                                        get_brick_periodicity<dim>(config_map));

  const auto local_num_quadrants = mesh_map.get_amr_mesh_info().local_num_quadrants();

  Kokkos::parallel_for(
    "compute_mirror_index_to_neighbor_global",
    Kokkos::RangePolicy<exec_space>(0, m_mirror_index_to_neighbor_global_index.size()),
    KOKKOS_CLASS_LAMBDA(const size_t index) {
      // mapping is index = neighbor_index + NEIGHBOR_STENCIL_SIZE * mirror_id
      const auto mirror_id = index / NEIGHBOR_STENCIL_SIZE;
      const auto neighbor_index = index - mirror_id * NEIGHBOR_STENCIL_SIZE;

      const auto direction = flat_index_to_direction(static_cast<int32_t>(neighbor_index));

      const auto coord = []() {
        if constexpr (dim == 2)
          return coord_t<dim>{ 0, 0 };
        else if constexpr (dim == 3)
          return coord_t<dim>{ 0, 0, 0 };
      }();

      // current key
      const auto key_cur = mirror_keys(mirror_id);

      // octant id in current mirror
      const auto key_status = key_to_value(key_cur, amr_hashmap);

      [[maybe_unused]] auto const & is_valid_key = key_status.first;

      // make sure key_inside actually exist in map
      // if it doesn't we have a serious logical problem
      KOKKOS_ASSERT(is_valid_key &&
                    "[MirrorQuadrantUtilsMetadata] key does not exist in hashmap !?");

      [[maybe_unused]] const auto i_oct_current = key_status.second;

      const auto is_stencil_center = [&]() {
        if constexpr (dim == 2)
        {
          return direction[IX] == 0 and direction[IY] == 0;
        }
        else if constexpr (dim == 3)
        {
          return direction[IX] == 0 and direction[IY] == 0 and direction[IZ] == 0;
        }
      }();

      if (is_stencil_center)
      {
        m_mirror_index_to_neighbor_global_index(index) = i_oct_current;
        m_active_neighbors(index) = false;
      }
      else
      {
        (void)block_size;
        const auto shift = [&]() {
          if constexpr (dim == 2)
          {
            return shift_t<2>{ block_size[IX] * direction[IX], block_size[IY] * direction[IY] };
          }
          else if constexpr (dim == 3)
          {
            return shift_t<3>{ block_size[IX] * direction[IX],
                               block_size[IY] * direction[IY],
                               block_size[IZ] * direction[IZ] };
          }
        }();

        const CellLocation_t cell_loc_cur{ coord, key_cur, i_oct_current, false };
        const auto           cell_loc_neigh = stencil_helper.getNeighLoc(cell_loc_cur, shift);
        m_mirror_index_to_neighbor_global_index(index) = cell_loc_neigh.iOct;

        auto is_active = false;
        // active neighbor are neighbor that are ghost, and the mpi rank is equal to the rank
        // attached to current mirror.
        if (cell_loc_neigh.iOct >= local_num_quadrants)
        {
          auto ighost = cell_loc_neigh.iOct - local_num_quadrants;
          if (m_ghost_quadrant_mpi_ranks(ighost) == m_mirror_quadrant_mpi_ranks(mirror_id))
          {
            is_active = true;
          }
        }
        m_active_neighbors(index) = is_active;
      }
    });
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
