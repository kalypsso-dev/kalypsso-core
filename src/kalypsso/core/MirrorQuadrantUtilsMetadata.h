// SPDX-FileCopyrightText: 2026 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file MirrorQuadrantUtilsMetadata.h
 *
 * This is a container for metadata related to mirror quadrants (used in ghost exchange).
 */
#ifndef KALYPSSO_CORE_MIRRORQUADRANTUTILSMETADATA_H_
#define KALYPSSO_CORE_MIRRORQUADRANTUTILSMETADATA_H_

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/BitFieldInteger.h>
#include <kalypsso/core/MeshMap.h>          // for orchard_key_view_t and amr_map_t
#include <kalypsso/core/AMRmesh.h>          // for p4est interface
#include <kalypsso/core/orchard_key_base.h> // for iOct_t
#include <kalypsso/core/real_type.h>        // for math functions (max, min, ...)
#include <kalypsso/core/kalypsso_data_container.h> // for DataArray, DataArrayHost, DataArrayGhostedBlock<dim, device_t>
#include <kalypsso/core/utils_block.h> // for definition of type block_size_t and shift_t
#include <kalypsso/utils/mpi/ParallelEnv.h>

#include <../better-enums/enum.h>

namespace kalypsso
{

namespace core
{

// ====================================================================
// ====================================================================
// ====================================================================
/**
 * \class MirrorQuadrantUtilsMetadata
 */
template <size_t dim, typename device_t>
class MirrorQuadrantUtilsMetadata
{
public:
  //! our kokkos execution space
  using exec_space = typename device_t::execution_space;

  //! Stencil size.
  //! In 2d, stencil is 3x3, in 3d it is 3x3x3
  constexpr static size_t NEIGHBOR_STENCIL_SIZE = dim == 2 ? 9 : 27;

private:
  //! parallel environment
  const ParallelEnv & m_par_env;

  //! array of global index (from 0 to local_num_quadrants_total) containing
  //! NEIGHBOR_STENCIL_SIZE values per mirror quadrants.
  //! So this array must be of size local_num_mirrors*NEIGHBOR_STENCIL_SIZE
  DataArray<iOct_t, device_t> m_mirror_index_to_neighbor_global_index;

  //! array with same layout as m_mirror_index_to_neighbor_global_index
  //! indicating which neighbor are actually ghost, which data can be send to
  DataArray<bool, device_t> m_active_neighbors;

  //! array of mpi ranks for all ghost quadrants.
  //! the mpi rank is the rank from which we must receive data
  //! its size should be local_num_ghosts
  DataArray<int32_t, device_t> m_ghost_quadrant_mpi_ranks;

  //! array of mpi ranks for all mirror quadrants.
  //! the mpi rank is the rank to which we must send data
  //! its size should be local_num_mirrors
  DataArray<int32_t, device_t> m_mirror_quadrant_mpi_ranks;

  //! convert direction to flat index (between 0 and NEIGHBOR_STENCIL_SIZE-1). 2d version
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 2), bool> = true>
  KOKKOS_INLINE_FUNCTION auto
  direction_to_flat_index(int32_t dir_x, int32_t dir_y) const -> size_t
  {
    KOKKOS_ASSERT(dir_x >= -1);
    KOKKOS_ASSERT(dir_x <= 1);
    KOKKOS_ASSERT(dir_y >= -1);
    KOKKOS_ASSERT(dir_y <= 1);

    return static_cast<size_t>((dir_x + 1) + 3 * (dir_y + 1));
  }

  //! convert direction to flat index (between 0 and NEIGHBOR_STENCIL_SIZE-1). 3d version
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 3), bool> = true>
  KOKKOS_INLINE_FUNCTION auto
  direction_to_flat_index(int32_t dir_x, int32_t dir_y, int32_t dir_z) const -> size_t
  {
    KOKKOS_ASSERT(dir_x >= -1);
    KOKKOS_ASSERT(dir_x <= 1);
    KOKKOS_ASSERT(dir_y >= -1);
    KOKKOS_ASSERT(dir_y <= 1);
    KOKKOS_ASSERT(dir_z >= -1);
    KOKKOS_ASSERT(dir_z <= 1);

    return static_cast<size_t>((dir_x + 1) + 3 * (dir_y + 1) + 9 * (dir_z + 1));
  }

  //! convert direction to flat index (between 0 and NEIGHBOR_STENCIL_SIZE-1)
  KOKKOS_INLINE_FUNCTION auto
  direction_to_flat_index(shift_t<dim> const & dir) const -> size_t
  {
    if constexpr (dim == 2)
    {
      return direction_to_flat_index(dir[IX], dir[IY]);
    }
    else if constexpr (dim == 3)
    {
      return direction_to_flat_index(dir[IX], dir[IY], dir[IZ]);
    }
  }

  //! convert flat index to direction. 2d version
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 2), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  flat_index_to_direction(int32_t flatindex, int32_t & dir_x, int32_t & dir_y) const
  {
    KOKKOS_ASSERT(flatindex >= 0);
    KOKKOS_ASSERT(flatindex < NEIGHBOR_STENCIL_SIZE);

    dir_y = flatindex / 3;
    dir_x = flatindex - 3 * dir_y;

    dir_x -= 1;
    dir_y -= 1;
  }

  //! convert flat index to direction. 2d version
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 3), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  flat_index_to_direction(int32_t   flatindex,
                          int32_t & dir_x,
                          int32_t & dir_y,
                          int32_t & dir_z) const
  {
    KOKKOS_ASSERT(flatindex >= 0);
    KOKKOS_ASSERT(flatindex < NEIGHBOR_STENCIL_SIZE);

    dir_z = flatindex / 9;
    int32_t tmp = flatindex - 9 * dir_z;

    dir_y = tmp / 3;
    dir_x = tmp - 3 * dir_y;

    dir_x -= 1;
    dir_y -= 1;
    dir_z -= 1;
  }

  KOKKOS_INLINE_FUNCTION auto
  flat_index_to_direction(int32_t flatindex) const -> shift_t<dim>
  {
    auto dir = shift_t<dim>{ 0, 0 };

    if constexpr (dim == 2)
    {
      flat_index_to_direction(flatindex, dir[IX], dir[IY]);
    }
    else if constexpr (dim == 3)
    {
      flat_index_to_direction(flatindex, dir[IX], dir[IY], dir[IZ]);
    }

    return dir;
  }

  //! After AMR cycle, mesh changed, i.e. ghosts changed, so here we update
  //! the array of MPI ranks associated
  void
  update_ghost_mpi_ranks(AMRmesh<dim> & amr_mesh);

  //! After AMR cycle, mesh changed, i.e. mirrors changed, so here we update
  //! the array of MPI ranks associated
  void
  update_mirror_mpi_ranks(AMRmesh<dim> & amr_mesh);

public:
  // ====================================================================
  // ====================================================================
  //! constructor.
  //!
  MirrorQuadrantUtilsMetadata(ParallelEnv const & par_env)
    : m_par_env(par_env)
    , m_mirror_index_to_neighbor_global_index("mirror_index_to_neighbor_global_index", 0)
    , m_active_neighbors("active_neighbor", 0)
    , m_ghost_quadrant_mpi_ranks("ghost_quadrant_mpi_ranks", 0)
    , m_mirror_quadrant_mpi_ranks("mirror_quadrant_mpi_ranks", 0)
  {}

  // ====================================================================
  // ====================================================================
  //! destructor.
  ~MirrorQuadrantUtilsMetadata() = default;

  DataArray<int32_t, device_t> const &
  ghost_quadrant_mpi_ranks()
  {
    return m_ghost_quadrant_mpi_ranks;
  }

  DataArray<int32_t, device_t> const &
  mirror_quadrant_mpi_ranks()
  {
    return m_mirror_quadrant_mpi_ranks;
  }

  // ====================================================================
  // ====================================================================
  DataArray<iOct_t, device_t> const &
  mirror_index_to_neighbor_global_index()
  {
    return m_mirror_index_to_neighbor_global_index;
  }

  // ====================================================================
  // ====================================================================
  //! \brief Update all mirror metadata.
  //!
  //! Must be called after AMR cycle to refresh mirror metadata
  //!
  //! \param[in] mesh_map Unordered map of AMR keys
  //! \param[in] config_map input parameter config
  void
  update(MeshMap<dim, device_t> & mesh_map, AMRmesh<dim> & amr_mesh, ConfigMap const & config_map);


  //! return index of neighbor quadrant to a given mirror in a given direction
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 2), bool> = true>
  KOKKOS_FUNCTION auto
  mirror_neighbor_index(iOct_t mirror_id, int32_t dir_x, int32_t dir_y) const -> iOct_t
  {
    return m_mirror_index_to_neighbor_global_index(mirror_id * NEIGHBOR_STENCIL_SIZE +
                                                   direction_to_flat_index(dir_x, dir_y));
  }

  //! return index of neighbor quadrant to a given mirror in a given direction
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 3), bool> = true>
  KOKKOS_FUNCTION auto
  mirror_neighbor_index(iOct_t mirror_id, int32_t dir_x, int32_t dir_y, int32_t dir_z) const
    -> iOct_t
  {
    return m_mirror_index_to_neighbor_global_index(mirror_id * NEIGHBOR_STENCIL_SIZE +
                                                   direction_to_flat_index(dir_x, dir_y, dir_z));
  }

  //! return index of neighbor quadrant to a given mirror in a given direction
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 2), bool> = true>
  KOKKOS_FUNCTION auto
  active_neigbor(iOct_t mirror_id, int32_t dir_x, int32_t dir_y) const -> bool
  {
    return m_active_neighbors(mirror_id * NEIGHBOR_STENCIL_SIZE +
                              direction_to_flat_index(dir_x, dir_y));
  }

  //! return index of neighbor quadrant to a given mirror in a given direction
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 3), bool> = true>
  KOKKOS_FUNCTION auto
  active_neighbor(iOct_t mirror_id, int32_t dir_x, int32_t dir_y, int32_t dir_z) const -> bool
  {
    return m_active_neighbors(mirror_id * NEIGHBOR_STENCIL_SIZE +
                              direction_to_flat_index(dir_x, dir_y, dir_z));
  }

}; // class MirrorQuadrantUtilsMetadata

// explicit template instantiation
extern template class MirrorQuadrantUtilsMetadata<2, kalypsso::DefaultDevice>;
extern template class MirrorQuadrantUtilsMetadata<3, kalypsso::DefaultDevice>;

#ifdef KALYPSSO_CORE_INSTANTIATE_HOST_TEMPLATE
extern template class MirrorQuadrantUtilsMetadata<2, kalypsso::HostDevice>;
extern template class MirrorQuadrantUtilsMetadata<3, kalypsso::HostDevice>;
#endif // KALYPSSO_CORE_INSTANTIATE_HOST_TEMPLATE

} // namespace core

} // namespace kalypsso

#endif // KALYPSSO_CORE_MIRRORQUADRANTUTILSMETADATA_H_
