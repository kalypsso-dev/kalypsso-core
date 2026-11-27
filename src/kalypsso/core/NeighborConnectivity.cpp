// SPDX-FileCopyrightText: 2026 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file NeighborConnectivity.cpp
 */
#include <kalypsso/core/NeighborConnectivity.h>

namespace kalypsso
{

namespace core
{

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
NeighborConnectivity<dim, device_t>::refresh_neighbor_count(
  NeighborConnectivity & neighbor_connectivity)
{

  neighbor_connectivity.resize();

  Kokkos::parallel_for("kalypsso::core::NeighborConnectivity::refresh_neighbor_connectivity",
                       Kokkos::RangePolicy<exec_space>(0, neighbor_connectivity.num_quadrants()),
                       neighbor_connectivity);

  auto neighbor_count_view = neighbor_connectivity.neighbor_count();
  auto global_offsets_view = neighbor_connectivity.global_offsets();

  const auto num_quadrants = neighbor_connectivity.num_quadrants();

  size_t total_number_of_neighbors = 0;

  // refresh global offsets doing an exclusive scan
  Kokkos::parallel_scan(
    "Compute global offsets",
    num_quadrants,
    KOKKOS_LAMBDA(const size_t i, global_offsets_t & update, const bool final) {
      const auto value = neighbor_count_view(i);
      if (final)
      {
        global_offsets_view(i) = update;
        // Last iteration also write the total number of neighbors
        if (i == num_quadrants - 1)
        {
          global_offsets_view(num_quadrants) = update + value;
        }
      }
      update += value;
    },
    total_number_of_neighbors);

  // resize neighbor connectivity table
  neighbor_connectivity.resize_connectivity(total_number_of_neighbors);

  // fill neighbor index


} // NeighborConnectivity<dim, device_t>::count_neighbors

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
NeighborConnectivity<dim, device_t>::report(ParallelEnv const & par_env, std::string type)
{
#ifdef KALYPSSO_CORE_USE_MPI
  KALYPSSO_INFO("============================={:=^{}}", "=", type.size());
  KALYPSSO_INFO("Neighbor connectivity report {}", type);
  KALYPSSO_INFO("============================={:=^{}}", "=", type.size());

  const auto num_ranks = static_cast<size_t>(par_env.size());

  const auto num_quadrants = m_neighbor_count_view.size();
  size_t     num_quadrants_min = 0;
  size_t     num_quadrants_sum = 0;
  size_t     num_quadrants_max = 0;
  par_env.comm().MPI_Reduce<MpiComm::MIN>(&num_quadrants, &num_quadrants_min, 1, 0);
  par_env.comm().MPI_Reduce<MpiComm::SUM>(&num_quadrants, &num_quadrants_sum, 1, 0);
  par_env.comm().MPI_Reduce<MpiComm::MAX>(&num_quadrants, &num_quadrants_max, 1, 0);

  KALYPSSO_INFO("{:<30} ({:^8} / {:^8} / {:^8} of {} MPI ranks)",
                "Number of quadrants",
                "min",
                "mean",
                "max",
                num_ranks);
  KALYPSSO_INFO("{:<30}  {:^8} / {:^8} / {:^8}",
                "",
                num_quadrants_min,
                num_quadrants_sum / num_ranks,
                num_quadrants_max)

  const auto num_quadrants_neighbors = m_neighbor_octant_ids_view.size();
  size_t     num_quadrants_neighbors_min = 0;
  size_t     num_quadrants_neighbors_sum = 0;
  size_t     num_quadrants_neighbors_max = 0;
  par_env.comm().MPI_Reduce<MpiComm::MIN>(
    &num_quadrants_neighbors, &num_quadrants_neighbors_min, 1, 0);
  par_env.comm().MPI_Reduce<MpiComm::SUM>(
    &num_quadrants_neighbors, &num_quadrants_neighbors_sum, 1, 0);
  par_env.comm().MPI_Reduce<MpiComm::MAX>(
    &num_quadrants_neighbors, &num_quadrants_neighbors_max, 1, 0);

  KALYPSSO_INFO("{:<30} ({:^8} / {:^8} / {:^8} of {} MPI ranks)",
                "Number of quadrants neighbors",
                "min",
                "mean",
                "max",
                num_ranks);
  KALYPSSO_INFO("{:<30}  {:^8} / {:^8} / {:^8}",
                "",
                num_quadrants_neighbors_min,
                num_quadrants_neighbors_sum / num_ranks,
                num_quadrants_neighbors_max)

  const auto num_neigh_per_quad = static_cast<real_t>(m_neighbor_octant_ids_view.size()) /
                                  static_cast<real_t>(m_neighbor_count_view.size());
  real_t num_neigh_per_quad_min = ZERO_F;
  real_t num_neigh_per_quad_sum = ZERO_F;
  real_t num_neigh_per_quad_max = ZERO_F;
  par_env.comm().MPI_Reduce<MpiComm::MIN>(&num_neigh_per_quad, &num_neigh_per_quad_min, 1, 0);
  par_env.comm().MPI_Reduce<MpiComm::SUM>(&num_neigh_per_quad, &num_neigh_per_quad_sum, 1, 0);
  par_env.comm().MPI_Reduce<MpiComm::MAX>(&num_neigh_per_quad, &num_neigh_per_quad_max, 1, 0);

  KALYPSSO_INFO("{:<30} ({:^8} / {:^8} / {:^8} of {} MPI ranks)",
                "Number of neighbors per quad",
                "min",
                "mean",
                "max",
                num_ranks);
  KALYPSSO_INFO("{:<30}  {:^8.2f} / {:^8.2f} / {:^8.2f}",
                "",
                num_neigh_per_quad_min,
                num_neigh_per_quad_sum / static_cast<real_t>(num_ranks),
                num_neigh_per_quad_max)

  KALYPSSO_INFO("============================={:=^{}}", "=", type.size());

#endif
} // NeighborConnectivity<dim, device_t>::report

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION auto
NeighborConnectivity<dim, device_t>::count_face_neighbors(Face::face_t face, key_t key_cur) const
  -> neighbor_count_t
{
  const auto dir = core::get_direction<dim>(face);

  // get neighbor key at same level
  const auto key_neigh_same_level = orchard_key_t<dim>::get_neighbor_key_same_level(
    key_cur, dir, m_brick_sizes, m_is_brick_periodic);

  // check if neighbor key exists in hash map
  auto key_neigh_hashindex = m_amr_hashmap_device.find(key_neigh_same_level);

  const auto is_at_same_level = m_amr_hashmap_device.valid_at(key_neigh_hashindex);

  // if key is valid, it means neighbor is actually at the same level
  if (is_at_same_level)
  {
    return neighbor_count_t{ 1 };
  }
  else
  {
    // check if father exist (coarser level)
    const auto key_neigh_coarser = orchard_key_t<dim>::father(key_neigh_same_level);

    key_neigh_hashindex = m_amr_hashmap_device.find(key_neigh_coarser);

    const auto is_at_coarser_level = m_amr_hashmap_device.valid_at(key_neigh_hashindex);

    if (is_at_coarser_level)
    {
      return neighbor_count_t{ 1 };
    }
    else
    {
      // test if a neighbor at finer scale exists
      const auto child_id = orchard_key_t<dim>::get_face_neighbor_smallest_child_id(face);
      const auto key_neigh_finer = orchard_key_t<dim>::child(key_neigh_same_level, child_id);

      key_neigh_hashindex = m_amr_hashmap_device.find(key_neigh_finer);
      const auto is_at_finer_level = m_amr_hashmap_device.valid_at(key_neigh_hashindex);

      if (is_at_finer_level)
      {
        return dim == 2 ? neighbor_count_t{ 2 } : neighbor_count_t{ 4 };
      }
    }
  }

  return neighbor_count_t{ 0 };

} // NeighborConnectivity<dim, device_t>::count_face_neighbors

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION auto
NeighborConnectivity<dim, device_t>::count_edge_neighbors(Face::face_t face0,
                                                          Face::face_t face1,
                                                          key_t key_cur) const -> neighbor_count_t
{
  const auto dir = core::get_direction<dim>(face0, face1);

  // get neighbor key at same level
  const auto key_neigh_same_level = orchard_key_t<dim>::get_neighbor_key_same_level(
    key_cur, dir, m_brick_sizes, m_is_brick_periodic);

  // check if neighbor key exists in hash map
  auto key_neigh_hashindex = m_amr_hashmap_device.find(key_neigh_same_level);

  const auto is_at_same_level = m_amr_hashmap_device.valid_at(key_neigh_hashindex);

  // if key is valid, it means neighbor is actually at the same level
  if (is_at_same_level)
  {
    return neighbor_count_t{ 1 };
  }
  else
  {
    // check if father exist (coarser level)
    const auto key_neigh_coarser = orchard_key_t<dim>::father(key_neigh_same_level);

    key_neigh_hashindex = m_amr_hashmap_device.find(key_neigh_coarser);

    const auto is_at_coarser_level = m_amr_hashmap_device.valid_at(key_neigh_hashindex);

    if (is_at_coarser_level)
    {
      if (m_count_type == +NeighborConnectivityType::APPROXIMATED)
      {
        return neighbor_count_t{ 1 };
      }
      else if (m_count_type == +NeighborConnectivityType::EXACT)
      {
        const auto child_id = orchard_key_t<dim>::child_id(key_cur);
        if constexpr (dim == 2)
        {
          const auto corner_id = ((face0 & 0x1) << 0) + ((face1 & 0x1) << 1);
          return (child_id == corner_id) ? neighbor_count_t{ 1 } : neighbor_count_t{ 0 };
        }
        else if constexpr (dim == 3)
        {
          if ((face0 == Face::XMIN or face0 == Face::XMAX) and
              (face1 == Face::YMIN or face1 == Face::YMAX))
          {
            // face0 => X
            // face1 => Y
            const auto edge_id = ((face0 & 0x1) << 0) + ((face1 & 0x1) << 1);
            return ((child_id == edge_id) or (child_id == edge_id + 4)) ? neighbor_count_t{ 1 }
                                                                        : neighbor_count_t{ 0 };
          }
          else if ((face0 == Face::XMIN or face0 == Face::XMAX) and
                   (face1 == Face::ZMIN or face1 == Face::ZMAX))
          {
            // face0 => X
            // face1 => Z
            const auto edge_id = ((face0 & 0x1) << 0) + ((face1 & 0x1) << 2);
            return ((child_id == edge_id) or (child_id == edge_id + 2)) ? neighbor_count_t{ 1 }
                                                                        : neighbor_count_t{ 0 };
          }
          else if ((face0 == Face::YMIN or face1 == Face::YMAX) and
                   (face1 == Face::ZMIN or face1 == Face::ZMAX))
          {
            // face0 => Y
            // face1 => Z
            const auto edge_id = ((face0 & 0x1) << 1) + ((face1 & 0x1) << 2);
            return ((child_id == edge_id) or (child_id == edge_id + 1)) ? neighbor_count_t{ 1 }
                                                                        : neighbor_count_t{ 0 };
          }
        }
        return neighbor_count_t{ 0 };
      }
    }
    else
    {
      // test if a neighbor at finer scale exists
      const auto child_id = orchard_key_t<dim>::get_edge_neighbor_smallest_child_id(face0, face1);
      const auto key_neigh_finer = orchard_key_t<dim>::child(key_neigh_same_level, child_id);

      key_neigh_hashindex = m_amr_hashmap_device.find(key_neigh_finer);
      const auto is_at_finer_level = m_amr_hashmap_device.valid_at(key_neigh_hashindex);

      if (is_at_finer_level)
      {
        return dim == 2 ? neighbor_count_t{ 1 } : neighbor_count_t{ 2 };
      }
    }
  }

  return neighbor_count_t{ 0 };

} // NeighborConnectivity<dim, device_t>::count_edge_neighbors

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION auto
NeighborConnectivity<dim, device_t>::count_corner_neighbors(Face::face_t face0,
                                                            Face::face_t face1,
                                                            Face::face_t face2,
                                                            key_t key_cur) const -> neighbor_count_t
{
  if constexpr (dim == 3)
  {
    if (m_count_type == +NeighborConnectivityType::APPROXIMATED)
    {
      return neighbor_count_t{ 1 };
    }
    else if (m_count_type == +NeighborConnectivityType::EXACT)
    {

      const auto dir = core::get_direction<dim>(face0, face1, face2);

      // get neighbor key at same level
      const auto key_neigh_same_level = orchard_key_t<dim>::get_neighbor_key_same_level(
        key_cur, dir, m_brick_sizes, m_is_brick_periodic);

      // check if neighbor key exists in hash map
      auto key_neigh_hashindex = m_amr_hashmap_device.find(key_neigh_same_level);

      const auto is_at_same_level = m_amr_hashmap_device.valid_at(key_neigh_hashindex);

      // if key is valid, it means neighbor is actually at the same level
      if (is_at_same_level)
      {
        return neighbor_count_t{ 1 };
      }
      else
      {
        // check if father exist (coarser level)
        const auto key_neigh_coarser = orchard_key_t<dim>::father(key_neigh_same_level);

        key_neigh_hashindex = m_amr_hashmap_device.find(key_neigh_coarser);

        const auto is_at_coarser_level = m_amr_hashmap_device.valid_at(key_neigh_hashindex);

        if (is_at_coarser_level)
        {
          const auto child_id = orchard_key_t<dim>::child_id(key_cur);
          const auto corner_id = ((face0 & 0x1) << 0) + ((face1 & 0x1) << 1) + ((face2 & 0x1) << 2);
          return (child_id == corner_id) ? neighbor_count_t{ 1 } : neighbor_count_t{ 0 };
        }
        else
        {
          // test if a neighbor at finer scale exists
          const auto child_id =
            orchard_key_t<dim>::get_corner_neighbor_smallest_child_id(face0, face1, face2);
          const auto key_neigh_finer = orchard_key_t<dim>::child(key_neigh_same_level, child_id);

          key_neigh_hashindex = m_amr_hashmap_device.find(key_neigh_finer);
          const auto is_at_finer_level = m_amr_hashmap_device.valid_at(key_neigh_hashindex);

          if (is_at_finer_level)
          {
            return neighbor_count_t{ 1 };
          }
        }
      }
    }

  } // end if dim==3

  return neighbor_count_t{ 0 };

} // NeighborConnectivity<dim, device_t>::count_corner_neighbors

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION auto
NeighborConnectivity<dim, device_t>::count_sector_neighbors(uint8_t sector, key_t key_cur) const
  -> neighbor_count_t
{

  if constexpr (dim == 2)
  {
    // odd sector contain face neighbors
    if (is_face_sector_2d(sector))
    {
      const auto face = get_face_id_2d(sector);
      return count_face_neighbors(face, key_cur);
    }
    else if (is_corner_sector_2d(sector))
    {
      uint8_t face0, face1;
      get_corner_ids_2d(sector, face0, face1);
      return count_edge_neighbors(face0, face1, key_cur);
    }
    return 0;
  }
  else if constexpr (dim == 3)
  {
    if (is_face_sector_3d(sector))
    {
      const auto face = get_face_id_3d(sector);
      return count_face_neighbors(face, key_cur);
    }
    else if (is_edge_sector_3d(sector))
    {
      uint8_t face0, face1;
      get_edge_ids_3d(sector, face0, face1);
      return count_edge_neighbors(face0, face1, key_cur);
    }
    else if (is_corner_sector_3d(sector))
    {
      uint8_t face0, face1, face2;
      get_corner_ids_3d(sector, face0, face1, face2);
      return count_corner_neighbors(face0, face1, face2, key_cur);
    }
    return 0;
  }

} // NeighborConnectivity<dim, device_t>::count_sector_neighbors

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
NeighborConnectivity<dim, device_t>::operator()(const index_t & iOct) const
{

  // get orchard key of current octant
  auto key_cur = m_orchard_keys_device(iOct);

  neighbor_count_t count = 0;
  auto             local_offsets = local_offsets_t{}; // empty brace is zero initialization

  // compute number of neighbors, sector by sector (from lower left to upper right sector)

  const uint8_t num_sectors = (dim == 2) ? 9 : 27;

  NeighborLocalOffsets<dim>::encode_local_offset(local_offsets, 0, count);
  for (uint8_t sector = 0; sector < num_sectors; ++sector)
  {
    count += count_sector_neighbors(sector, key_cur);
    NeighborLocalOffsets<dim>::encode_local_offset(local_offsets, sector + 1, count);
  }

  // // clang-format off
  // KOKKOS_IF_ON_HOST(
  //   if ((m_par_env.rank() == 0 and (iOct >= 63 and iOct<67)) or
  //       (m_par_env.rank() == 1 and iOct == 440)) {
  //     for (uint8_t sector = 0; sector < num_sectors; ++sector)
  //     {
  //       KALYPSSO_INFO_ALL("mpi rank : {} iOct : {} sector: {} count : {} - mirror_size {}",
  //                         m_par_env.rank(),
  //                         iOct,
  //                         sector,
  //                         NeighborLocalOffsets<dim>::decode_local_offset(local_offsets, sector +
  //                         1) - NeighborLocalOffsets<dim>::decode_local_offset(local_offsets,
  //                         sector), m_orchard_keys_device.size());
  //     }
  //   })
  // // clang-format on

  // write count and local offsets
  m_neighbor_count_view(iOct) = count;
  m_local_offsets_view(iOct) = local_offsets;

} // NeighborConnectivity<dim, device_t>::operator()

// explicit template instantiation
template class NeighborConnectivity<2, kalypsso::DefaultDevice>;
template class NeighborConnectivity<3, kalypsso::DefaultDevice>;

} // namespace core

} // namespace kalypsso
