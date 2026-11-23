// SPDX-FileCopyrightText: 2026 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ComputeNumberOfNeighborsOfQuadrant.cpp
 */
#include <kalypsso/core/ComputeNumberOfNeighborsOfQuadrant.h>

namespace kalypsso
{

namespace core
{

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
auto
ComputeNumberOfNeighborsOfQuadrant<dim, device_t>::count_neighbors(
  amr_hashmap_t                          amr_hashmap,
  orchard_key_view_t                     orchard_keys,
  AMRMeshInfo                            amr_mesh_info,
  brick_size_t<dim>                      brick_sizes,
  Kokkos::Array<bool, dim>               is_brick_periodic,
  ComputeNumberOfNeighborsOfQuadrantType type) -> neighbor_count_view_t
{

  const auto num_quadrants = orchard_keys.extent(0);
  KOKKOS_ASSERT(orchard_keys.extent(0) <=
                  static_cast<size_t>((amr_mesh_info.local_num_quadrants())) &&
                "orchard keys has wrong size (should smaller that the number of owned quadrants)");

  const auto data_name = std::string("neighbor count view");

  auto neighbor_count_view = neighbor_count_view_t(data_name, num_quadrants);

  ComputeNumberOfNeighborsOfQuadrant<dim, device_t> functor(amr_hashmap,
                                                            orchard_keys,
                                                            amr_mesh_info,
                                                            brick_sizes,
                                                            is_brick_periodic,
                                                            neighbor_count_view,
                                                            type);

  Kokkos::parallel_for("kalypsso::core::ComputeNumberOfNeighborsOfQuadrant",
                       Kokkos::RangePolicy<exec_space>(0, num_quadrants),
                       functor);

  return neighbor_count_view;

} // ComputeNumberOfNeighborsOfQuadrant<dim, device_t>::count_neighbors

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION auto
ComputeNumberOfNeighborsOfQuadrant<dim, device_t>::count_face_neighbors(Face::face_t face,
                                                                        key_t        key_cur) const
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

} // ComputeNumberOfNeighborsOfQuadrant<dim, device_t>::count_face_neighbors

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION auto
ComputeNumberOfNeighborsOfQuadrant<dim, device_t>::count_edge_neighbors(Face::face_t face0,
                                                                        Face::face_t face1,
                                                                        key_t        key_cur) const
  -> neighbor_count_t
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
      if (m_count_type == +ComputeNumberOfNeighborsOfQuadrantType::APPROXIMATED)
      {
        return neighbor_count_t{ 1 };
      }
      else if (m_count_type == +ComputeNumberOfNeighborsOfQuadrantType::EXACT)
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

} // ComputeNumberOfNeighborsOfQuadrant<dim, device_t>::count_edge_neighbors

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION auto
ComputeNumberOfNeighborsOfQuadrant<dim, device_t>::count_corner_neighbors(Face::face_t face0,
                                                                          Face::face_t face1,
                                                                          Face::face_t face2,
                                                                          key_t key_cur) const
  -> neighbor_count_t
{
  if constexpr (dim == 3)
  {
    if (m_count_type == +ComputeNumberOfNeighborsOfQuadrantType::APPROXIMATED)
    {
      return neighbor_count_t{ 1 };
    }
    else if (m_count_type == +ComputeNumberOfNeighborsOfQuadrantType::EXACT)
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

} // ComputeNumberOfNeighborsOfQuadrant<dim, device_t>::count_corner_neighbors

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
ComputeNumberOfNeighborsOfQuadrant<dim, device_t>::operator()(const index_t & iOct) const
{

  // get orchard key of current octant
  auto key_cur = m_orchard_keys_device(iOct);

  neighbor_count_t count = 0;

  //
  // set face status
  //
  for (Face::face_t face = 0; face < Face::num_faces<dim>(); ++face)
  {
    count += count_face_neighbors(face, key_cur);
  } // end for face

  //
  // set edge status
  //
  count += count_edge_neighbors(Face::XMIN, Face::YMIN, key_cur);
  count += count_edge_neighbors(Face::XMAX, Face::YMIN, key_cur);
  count += count_edge_neighbors(Face::XMIN, Face::YMAX, key_cur);
  count += count_edge_neighbors(Face::XMAX, Face::YMAX, key_cur);

  if constexpr (dim == 3)
  {
    count += count_edge_neighbors(Face::YMIN, Face::ZMIN, key_cur);
    count += count_edge_neighbors(Face::YMAX, Face::ZMIN, key_cur);
    count += count_edge_neighbors(Face::YMIN, Face::ZMAX, key_cur);
    count += count_edge_neighbors(Face::YMAX, Face::ZMAX, key_cur);

    count += count_edge_neighbors(Face::XMIN, Face::ZMIN, key_cur);
    count += count_edge_neighbors(Face::XMAX, Face::ZMIN, key_cur);
    count += count_edge_neighbors(Face::XMIN, Face::ZMAX, key_cur);
    count += count_edge_neighbors(Face::XMAX, Face::ZMAX, key_cur);
  }

  if constexpr (dim == 3)
  {
    count += count_corner_neighbors(Face::XMIN, Face::YMIN, Face::ZMIN, key_cur);
    count += count_corner_neighbors(Face::XMAX, Face::YMIN, Face::ZMIN, key_cur);
    count += count_corner_neighbors(Face::XMIN, Face::YMAX, Face::ZMIN, key_cur);
    count += count_corner_neighbors(Face::XMAX, Face::YMAX, Face::ZMIN, key_cur);
    count += count_corner_neighbors(Face::XMIN, Face::YMIN, Face::ZMAX, key_cur);
    count += count_corner_neighbors(Face::XMAX, Face::YMIN, Face::ZMAX, key_cur);
    count += count_corner_neighbors(Face::XMIN, Face::YMAX, Face::ZMAX, key_cur);
    count += count_corner_neighbors(Face::XMAX, Face::YMAX, Face::ZMAX, key_cur);
  }

  // status is up to date, just write it now
  m_neighbor_count_view(iOct) = count;

} // ComputeNumberOfNeighborsOfQuadrant<dim, device_t>::operator()

// explicit template instantiation
template class ComputeNumberOfNeighborsOfQuadrant<2, kalypsso::DefaultDevice>;
template class ComputeNumberOfNeighborsOfQuadrant<3, kalypsso::DefaultDevice>;

} // namespace core

} // namespace kalypsso
