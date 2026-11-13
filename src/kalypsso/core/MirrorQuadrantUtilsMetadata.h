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
#include <kalypsso/core/orchard_key_base.h> // for iOct_t
#include <kalypsso/core/real_type.h>        // for math functions (max, min, ...)
#include <kalypsso/core/kalypsso_data_container.h> // for DataArray, DataArrayHost, DataArrayGhostedBlock<dim, device_t>
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
  //! array of global index.
  //! this array maps each mirror (indexed from 0 to num_mirror-1) to corresponding index in locally
  //! owned quadrants. So this array must be of size local_num_mirrors
  DataArray<iOct_t, device_t> m_mirror_index_to_global_index;

  //! array of global index (from 0 to local_num_quadrants_total)  containing NEIGHBOR_STENCIL_SIZE
  //! values per mirror quadrants.
  //! So this array must be of size local_num_mirrors*NEIGHBOR_STENCIL_SIZE
  DataArray<iOct_t, device_t> m_mirror_index_to_neighbor_global_index;

  //! convert direction to flat index (between 0 and NEIGHBOR_STENCIL_SIZE-1). 2d version
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 2), bool> = true>
  KOKKOS_INLINE_FUNCTION auto
  flat_direction(int8_t dir_x, int8_t dir_y) const -> size_t
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
  flat_direction(int8_t dir_x, int8_t dir_y, int8_t dir_z) const -> size_t
  {
    KOKKOS_ASSERT(dir_x >= -1);
    KOKKOS_ASSERT(dir_x <= 1);
    KOKKOS_ASSERT(dir_y >= -1);
    KOKKOS_ASSERT(dir_y <= 1);
    KOKKOS_ASSERT(dir_z >= -1);
    KOKKOS_ASSERT(dir_z <= 1);

    return static_cast<size_t>((dir_x + 1) + 3 * (dir_y + 1) + 9 * (dir_y + 1));
  }

public:
  // ====================================================================
  // ====================================================================
  //! constructor.
  //!
  MirrorQuadrantUtilsMetadata()
    : m_mirror_index_to_global_index("mirror_index_to_global_index", 0)
    , m_mirror_index_to_neighbor_global_index("mirror_index_to_neighbor_global_index", 0)
  {}

  // ====================================================================
  // ====================================================================
  //! destructor.
  ~MirrorQuadrantUtilsMetadata() = default;

  // ====================================================================
  // ====================================================================
  //! \brief Update all mirror metadata.
  //!
  //! Must be called after AMR cycle to refresh mirror metadata
  //!
  //! \param[in] mesh_map Unordered map of AMR keys
  void
  update(MeshMap<dim, device_t> & mesh_map);

  //! return index of neighbor quadrant to a given mirror in a given direction
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 2), bool> = true>
  KOKKOS_FUNCTION auto
  mirror_neighbor_index(iOct_t mirror_id, int8_t dir_x, int8_t dir_y) const -> iOct_t;

  //! return index of neighbor quadrant to a given mirror in a given direction
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 3), bool> = true>
  KOKKOS_FUNCTION auto
  mirror_neighbor_index(iOct_t mirror_id, int8_t dir_x, int8_t dir_y, int8_t dir_z) const -> iOct_t;

}; // class MirrorQuadrantUtilsMetadata

// explicit template instantiation
extern template class MirrorQuadrantUtilsMetadata<2, kalypsso::DefaultDevice>;
extern template class MirrorQuadrantUtilsMetadata<3, kalypsso::DefaultDevice>;

} // namespace core

} // namespace kalypsso

#endif // KALYPSSO_CORE_MIRRORQUADRANTUTILSMETADATA_H_
