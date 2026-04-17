// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file StencilHelper.cpp
 */
#include <kalypsso/core/StencilHelper.h>
#include <kalypsso/core/prolongation.h>
#include <kalypsso/core/utils_block.h>

namespace kalypsso
{

// ====================================================================
// ====================================================================
template <size_t dim>
KOKKOS_FUNCTION CellEdgeLocation
get_CellEdgeLocation(EdgeLocation<dim> const & edge_loc, block_size_t<dim> const & block_size)
{

  if constexpr (dim == 2)
  {
    KOKKOS_ASSERT((edge_loc.ijk[dim] == IZ) && "Wrong edge direction (must be along Z in 2d)");

    int res = 0;

    if (edge_loc.ijk[IX] == block_size[IX])
      res += 1;
    if (edge_loc.ijk[IY] == block_size[IY])
      res += 2;
    return static_cast<CellEdgeLocation>(res);
  }
  else if constexpr (dim == 3)
  {
    if (edge_loc.ijk[dim] == IZ)
    {
      int res = 0;

      if (edge_loc.ijk[IX] == block_size[IX])
        res += 1;
      if (edge_loc.ijk[IY] == block_size[IY])
        res += 2;
      return static_cast<CellEdgeLocation>(res);
    }
    else if (edge_loc.ijk[dim] == IX)
    {
      int res = 0;

      if (edge_loc.ijk[IY] == block_size[IY])
        res += 1;
      if (edge_loc.ijk[IZ] == block_size[IZ])
        res += 2;
      return static_cast<CellEdgeLocation>(res);
    }
    else if (edge_loc.ijk[dim] == IY)
    {
      int res = 0;

      if (edge_loc.ijk[IX] == block_size[IX])
        res += 1;
      if (edge_loc.ijk[IZ] == block_size[IZ])
        res += 2;
      return static_cast<CellEdgeLocation>(res);
    }
    return EDGE_INVALID;
  } // dim == 3

} // get_CellEdgeLocation

template KOKKOS_FUNCTION CellEdgeLocation
get_CellEdgeLocation(EdgeLocation<2> const & edge_loc, block_size_t<2> const & block_size);
template KOKKOS_FUNCTION CellEdgeLocation
get_CellEdgeLocation(EdgeLocation<3> const & edge_loc, block_size_t<3> const & block_size);

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
StencilHelper<dim, device_t>::StencilHelper(amr_hashmap_t            amr_hashmap,
                                            orchard_key_view_t       orchard_keys,
                                            block_size_t<dim>        block_sizes,
                                            brick_size_t<dim>        brick_sizes,
                                            Kokkos::Array<bool, dim> is_brick_periodic)
  : m_amr_hashmap_device(amr_hashmap)
  , m_orchard_keys_device(orchard_keys)
  , m_block_sizes(block_sizes)
  , m_brick_sizes(brick_sizes)
  , m_is_brick_periodic(is_brick_periodic)
{} // StencilHelper<dim, device_t>::StencilHelper

// =========================================================================================
// =========================================================================================
template <size_t dim, typename device_t>
KOKKOS_FUNCTION bool
StencilHelper<dim, device_t>::is_cell_location_at_domain_border(
  CellLocation_t const & cell_loc) const
{
  const auto & cell_block_sizes = m_block_sizes;

  if (orchard_key_t<dim>::is_at_domain_border(cell_loc.key, Face::XMIN, m_brick_sizes) and
      (cell_loc.ijk[IX] == 0))
    return true;

  if (orchard_key_t<dim>::is_at_domain_border(cell_loc.key, Face::XMAX, m_brick_sizes) and
      cell_loc.ijk[IX] == (cell_block_sizes[IX] - 1))
    return true;

  if (orchard_key_t<dim>::is_at_domain_border(cell_loc.key, Face::YMIN, m_brick_sizes) and
      (cell_loc.ijk[IY] == 0))
    return true;

  if (orchard_key_t<dim>::is_at_domain_border(cell_loc.key, Face::YMAX, m_brick_sizes) and
      cell_loc.ijk[IY] == (cell_block_sizes[IY] - 1))
    return true;

  if constexpr (dim == 3)
  {
    if (orchard_key_t<dim>::is_at_domain_border(cell_loc.key, Face::ZMIN, m_brick_sizes) and
        (cell_loc.ijk[IZ] == 0))
      return true;

    if (orchard_key_t<dim>::is_at_domain_border(cell_loc.key, Face::ZMAX, m_brick_sizes) and
        cell_loc.ijk[IZ] == (cell_block_sizes[IZ] - 1))
      return true;
  }

  return false;

} // is_cell_location_at_domain_border

// =========================================================================================
// =========================================================================================
template <size_t dim, typename device_t>
KOKKOS_FUNCTION bool
StencilHelper<dim, device_t>::is_face_location_at_block_border(
  FaceLocation_t const & face_loc) const
{
  const auto & cell_block_sizes = m_block_sizes;
  auto const & ivar = face_loc.ijk[dim];

  if (ivar == IX and (face_loc.ijk[IX] == 0 or face_loc.ijk[IX] == cell_block_sizes[IX]))
    return true;

  if (ivar == IY and (face_loc.ijk[IY] == 0 or face_loc.ijk[IY] == cell_block_sizes[IY]))
    return true;

  if constexpr (dim == 3)
  {
    if (ivar == IZ and (face_loc.ijk[IZ] == 0 or face_loc.ijk[IZ] == cell_block_sizes[IZ]))
      return true;
  }

  return false;

} // is_face_location_at_block_border

// =========================================================================================
// =========================================================================================
template <size_t dim, typename device_t>
KOKKOS_FUNCTION bool
StencilHelper<dim, device_t>::is_edge_location_at_block_border(
  EdgeLocation_t const & edge_loc) const
{
  const auto & cell_block_sizes = m_block_sizes;
  auto const & edge_dir = edge_loc.ijk[dim];

  if constexpr (dim == 2)
  {
    KOKKOS_ASSERT((edge_dir == IZ) && "Wrong edge direction");
  }

  // get transverse directions
  const auto dir1 = static_cast<size_t>((edge_dir + 1) % 3);
  const auto dir2 = static_cast<size_t>((edge_dir + 2) % 3);

  if (edge_loc.ijk[dir1] == 0 or edge_loc.ijk[dir1] == cell_block_sizes[dir1] or
      edge_loc.ijk[dir2] == 0 or edge_loc.ijk[dir2] == cell_block_sizes[dir2])
    return true;

  return false;

} // is_edge_location_at_block_border

// =========================================================================================
// =========================================================================================
template <size_t dim, typename device_t>
KOKKOS_FUNCTION bool
StencilHelper<dim, device_t>::is_edge_location_at_domain_border(
  EdgeLocation_t const & edge_loc) const
{
  const auto & cell_block_sizes = m_block_sizes;

  if (orchard_key_t<dim>::is_at_domain_border(edge_loc.key, Face::XMIN, m_brick_sizes) and
      (edge_loc.ijk[dim] != IX) and (edge_loc.ijk[IX] == 0))
    return true;

  if (orchard_key_t<dim>::is_at_domain_border(edge_loc.key, Face::XMAX, m_brick_sizes) and
      (edge_loc.ijk[dim] != IX) and (edge_loc.ijk[IX] == (cell_block_sizes[IX])))
    return true;

  if (orchard_key_t<dim>::is_at_domain_border(edge_loc.key, Face::YMIN, m_brick_sizes) and
      (edge_loc.ijk[dim] != IY) and (edge_loc.ijk[IY] == 0))
    return true;

  if (orchard_key_t<dim>::is_at_domain_border(edge_loc.key, Face::YMAX, m_brick_sizes) and
      (edge_loc.ijk[dim] != IY) and (edge_loc.ijk[IY] == (cell_block_sizes[IY])))
    return true;

  if constexpr (dim == 3)
  {
    if (orchard_key_t<dim>::is_at_domain_border(edge_loc.key, Face::ZMIN, m_brick_sizes) and
        (edge_loc.ijk[dim] != IZ) and (edge_loc.ijk[IZ] == 0))
      return true;

    if (orchard_key_t<dim>::is_at_domain_border(edge_loc.key, Face::ZMAX, m_brick_sizes) and
        (edge_loc.ijk[dim] != IZ) and (edge_loc.ijk[IZ] == (cell_block_sizes[IZ])))
      return true;
  }

  return false;

} // is_edge_location_at_domain_border

// =========================================================================================
// =========================================================================================
template <size_t dim, typename device_t>
KOKKOS_FUNCTION auto
StencilHelper<dim, device_t>::getNeighLoc(CellLocation<dim> const & cell_loc,
                                          shift_t<dim>              shift) const -> CellLocation_t
{

  const auto & b = m_block_sizes;

  // beware signed values
  coord_t<dim, int32_t> coord_neigh;
  coord_neigh[IX] = static_cast<int32_t>(cell_loc.ijk[IX]) + shift[IX];
  coord_neigh[IY] = static_cast<int32_t>(cell_loc.ijk[IY]) + shift[IY];
  if constexpr (dim == 3)
    coord_neigh[IZ] = static_cast<int32_t>(cell_loc.ijk[IZ]) + shift[IZ];

  // compute direction to neighbor octant
  auto dir = []() {
    if constexpr (dim == 2)
      return Kokkos::Array<int8_t, 2>{ 0, 0 };
    else if constexpr (dim == 3)
      return Kokkos::Array<int8_t, 3>{ 0, 0, 0 };
  }();

  if (coord_neigh[IX] < 0)
    dir[IX] = -1;
  else if (coord_neigh[IX] >= static_cast<int32_t>(b[IX]))
    dir[IX] = 1;

  if (coord_neigh[IY] < 0)
    dir[IY] = -1;
  else if (coord_neigh[IY] >= static_cast<int32_t>(b[IY]))
    dir[IY] = 1;

  if constexpr (dim == 3)
  {
    if (coord_neigh[IZ] < 0)
      dir[IZ] = -1;
    else if (coord_neigh[IZ] >= static_cast<int32_t>(b[IZ]))
      dir[IZ] = 1;
  }

  auto dir_norm = dir[IX] * dir[IX] + dir[IY] * dir[IY];
  if constexpr (dim == 3)
    dir_norm += dir[IZ] * dir[IZ];

  // neighbor cell is inside current block (search is over)
  if (dir_norm == 0)
  {
    // unsigned neighbor coordinate
    coord_t<dim> coord_neigh_u;
    coord_neigh_u[IX] = static_cast<typename coord_t<dim>::value_type>(coord_neigh[IX]);
    coord_neigh_u[IY] = static_cast<typename coord_t<dim>::value_type>(coord_neigh[IY]);
    if constexpr (dim == 3)
      coord_neigh_u[IZ] = static_cast<typename coord_t<dim>::value_type>(coord_neigh[IZ]);

    CellLocation_t cell_loc_neigh{
      coord_neigh_u, cell_loc.key, cell_loc.iOct, cell_loc.is_outside_domain
    };

    return cell_loc_neigh;
  }

  // neighbor cell is outside current block

  // compute neighbor cell coord in the neighbor block
  coord_t<dim> coord_neigh_u;
  coord_neigh_u[IX] = static_cast<typename coord_t<dim>::value_type>(
    coord_neigh[IX] - b[IX] * (dir[IX] == 1) + b[IX] * (dir[IX] == -1));
  coord_neigh_u[IY] = static_cast<typename coord_t<dim>::value_type>(
    coord_neigh[IY] - b[IY] * (dir[IY] == 1) + b[IY] * (dir[IY] == -1));
  if constexpr (dim == 3)
  {
    coord_neigh_u[IZ] = static_cast<typename coord_t<dim>::value_type>(
      coord_neigh[IZ] - b[IZ] * (dir[IZ] == 1) + b[IZ] * (dir[IZ] == -1));
  }

  //
  // check if neighbor cell belongs to a block at same level
  //

  // get neighbor key at same level
  auto key_neigh_same_level = orchard_key_t<dim>::get_neighbor_key_same_level(
    cell_loc.key, dir, m_brick_sizes, m_is_brick_periodic);

  // check if neighbor key exists in hash map
  auto key_neigh_hashindex = m_amr_hashmap_device.find(key_neigh_same_level);
  auto is_at_same_level = m_amr_hashmap_device.valid_at(key_neigh_hashindex);
  if (is_at_same_level)
  {
    auto iOct_neigh = m_amr_hashmap_device.value_at(key_neigh_hashindex);

    CellLocation_t cell_loc_neigh{ coord_neigh_u, key_neigh_same_level, iOct_neigh, false };

    // check if neighbor is outside global domain
    if (orchard_key_t<dim>::is_at_any_domain_border(cell_loc.key, m_brick_sizes))
    {
      const auto normal =
        orchard_key_t<dim>::get_outside_normal(cell_loc.key, m_brick_sizes, m_is_brick_periodic);

      auto prod_scal = normal[IX] * shift[IX] + normal[IY] * shift[IY];
      if constexpr (dim == 3)
        prod_scal += normal[IZ] * shift[IZ];

      // if this scalar prod is greater than zero it means we are looking for a neighbor that is
      // really outside domain
      // this scalar prod will be zero if we are at a periodic border, and then the neighbor will be
      // found in the hashmap
      if (prod_scal > 0)
      {
        cell_loc_neigh.is_outside_domain = true;
      }
    }

    return cell_loc_neigh;
  }

  //
  // check if neighbor cell belongs to a block at coarser level
  //
  auto key_neigh_coarser = orchard_key_t<dim>::father(key_neigh_same_level);

  key_neigh_hashindex = m_amr_hashmap_device.find(key_neigh_coarser);

  auto is_at_coarser_level = m_amr_hashmap_device.valid_at(key_neigh_hashindex);

  if (is_at_coarser_level)
  {
    auto iOct_neigh = m_amr_hashmap_device.value_at(key_neigh_hashindex);

    // get child id of current block (needed to compute neighbor cell coordinates)
    const auto child_id = orchard_key_t<dim>::child_id(key_neigh_same_level);

    coord_neigh_u[IX] = (coord_neigh_u[IX] + ((child_id & 0x1) >> 0) * b[IX]) / 2;
    coord_neigh_u[IY] = (coord_neigh_u[IY] + ((child_id & 0x2) >> 1) * b[IY]) / 2;
    if constexpr (dim == 3)
    {
      coord_neigh_u[IZ] = (coord_neigh_u[IZ] + ((child_id & 0x4) >> 2) * b[IZ]) / 2;
    }

    CellLocation_t cell_loc_neigh{ coord_neigh_u, key_neigh_coarser, iOct_neigh, false };
    return cell_loc_neigh;
  }

  //
  // check if neighbor cell belongs to a block at finer level
  //
  {
    coord_neigh_u = coord_neigh_u * 2;

    // determine in which finer octant we need to use (i.e. determine child id)
    uint8_t child_id = 0;
    if (coord_neigh_u[IX] >= b[IX])
    {
      coord_neigh_u[IX] -= b[IX];
      child_id += 1;
    }
    if (coord_neigh_u[IY] >= b[IY])
    {
      coord_neigh_u[IY] -= b[IY];
      child_id += 2;
    }
    if constexpr (dim == 3)
      if (coord_neigh_u[IZ] >= b[IZ])
      {
        coord_neigh_u[IZ] -= b[IZ];
        child_id += 4;
      }

    // lookup for this child in the hashmap
    const auto key_neigh_fine = orchard_key_t<dim>::child(key_neigh_same_level, child_id);

    const auto key_neigh_fine_hashindex = m_amr_hashmap_device.find(key_neigh_fine);
    const auto valid = m_amr_hashmap_device.valid_at(key_neigh_fine_hashindex);

    if (valid)
    {
      auto iOct_neigh = m_amr_hashmap_device.value_at(key_neigh_fine_hashindex);

      CellLocation_t cell_loc_neigh{ coord_neigh_u, key_neigh_fine, iOct_neigh, false };
      return cell_loc_neigh;
    }
  } // end search for neighbor at finer level

  // if we are here, we have a genuine problem/bug !!!

  // default : return "self" as invalid value here
  return cell_loc;

} // StencilHelper<dim, device_t>::getNeighLoc - cell

// =========================================================================================
// =========================================================================================
template <size_t dim, typename device_t>
KOKKOS_FUNCTION auto
StencilHelper<dim, device_t>::getNeighLoc(FaceLocation<dim> const & face_loc,
                                          shift_t<dim>              shift) const -> FaceLocation_t
{

  const auto & cell_block_sizes = m_block_sizes;
  auto         face_block_sizes = cell_block_sizes;
  auto const & ivar = face_loc.ijk[dim];
  if constexpr (dim == 2)
  {
    if (ivar < 2)
      face_block_sizes[static_cast<size_t>(ivar)] += 1;
  }
  else if constexpr (dim == 3)
  {
    face_block_sizes[static_cast<size_t>(ivar)] += 1;
  }

  // beware signed values
  face_multiindex_t<dim, int32_t> coord_neigh;
  coord_neigh[IX] = static_cast<int32_t>(face_loc.ijk[IX]) + shift[IX];
  coord_neigh[IY] = static_cast<int32_t>(face_loc.ijk[IY]) + shift[IY];
  if constexpr (dim == 3)
  {
    coord_neigh[IZ] = static_cast<int32_t>(face_loc.ijk[IZ]) + shift[IZ];
  }
  coord_neigh[dim] = face_loc.ijk[dim];

  // compute direction to neighbor octant
  auto dir = []() {
    if constexpr (dim == 2)
      return Kokkos::Array<int8_t, 2>{ 0, 0 };
    else if constexpr (dim == 3)
      return Kokkos::Array<int8_t, 3>{ 0, 0, 0 };
  }();

  if (coord_neigh[IX] < 0)
    dir[IX] = -1;
  else if (coord_neigh[IX] >= static_cast<int32_t>(face_block_sizes[IX]))
    dir[IX] = 1;

  if (coord_neigh[IY] < 0)
    dir[IY] = -1;
  else if (coord_neigh[IY] >= static_cast<int32_t>(face_block_sizes[IY]))
    dir[IY] = 1;

  if constexpr (dim == 3)
  {
    if (coord_neigh[IZ] < 0)
      dir[IZ] = -1;
    else if (coord_neigh[IZ] >= static_cast<int32_t>(face_block_sizes[IZ]))
      dir[IZ] = 1;
  }

  auto dir_norm = dir[IX] * dir[IX] + dir[IY] * dir[IY];
  if constexpr (dim == 3)
    dir_norm += dir[IZ] * dir[IZ];

  // neighbor cell is inside current block (search is over)
  if (dir_norm == 0)
  {
    // unsigned neighbor coordinate
    face_multiindex_t<dim> coord_neigh_u;
    coord_neigh_u[IX] = static_cast<typename face_multiindex_t<dim>::value_type>(coord_neigh[IX]);
    coord_neigh_u[IY] = static_cast<typename face_multiindex_t<dim>::value_type>(coord_neigh[IY]);
    if constexpr (dim == 3)
    {
      coord_neigh_u[IZ] = static_cast<typename face_multiindex_t<dim>::value_type>(coord_neigh[IZ]);
    }
    coord_neigh_u[dim] = coord_neigh[dim];

    FaceLocation_t face_loc_neigh{
      coord_neigh_u, face_loc.key, face_loc.iOct, face_loc.is_outside_domain
    };

    return face_loc_neigh;
  }

  // neighbor cell is outside current block

  // compute neighbor cell coord in the neighbor block
  face_multiindex_t<dim> coord_neigh_u;
  coord_neigh_u[IX] = static_cast<typename face_multiindex_t<dim>::value_type>(
    coord_neigh[IX] - cell_block_sizes[IX] * (dir[IX] == 1) +
    cell_block_sizes[IX] * (dir[IX] == -1));
  coord_neigh_u[IY] = static_cast<typename face_multiindex_t<dim>::value_type>(
    coord_neigh[IY] - cell_block_sizes[IY] * (dir[IY] == 1) +
    cell_block_sizes[IY] * (dir[IY] == -1));
  if constexpr (dim == 3)
  {
    coord_neigh_u[IZ] = static_cast<typename face_multiindex_t<dim>::value_type>(
      coord_neigh[IZ] - cell_block_sizes[IZ] * (dir[IZ] == 1) +
      cell_block_sizes[IZ] * (dir[IZ] == -1));
  }
  coord_neigh_u[dim] = coord_neigh[dim];

  //
  // check if neighbor cell belongs to a block at same level
  //

  // get neighbor key at same level
  auto key_neigh_same_level = orchard_key_t<dim>::get_neighbor_key_same_level(
    face_loc.key, dir, m_brick_sizes, m_is_brick_periodic);

  // check if neighbor key exists in hash map
  auto key_neigh_hashindex = m_amr_hashmap_device.find(key_neigh_same_level);
  auto is_at_same_level = m_amr_hashmap_device.valid_at(key_neigh_hashindex);
  if (is_at_same_level)
  {
    auto iOct_neigh = m_amr_hashmap_device.value_at(key_neigh_hashindex);

    FaceLocation_t face_loc_neigh{ coord_neigh_u, key_neigh_same_level, iOct_neigh, false };

    // check if neighbor is outside global domain
    if (orchard_key_t<dim>::is_at_any_domain_border(face_loc.key, m_brick_sizes))
    {
      const auto normal =
        orchard_key_t<dim>::get_outside_normal(face_loc.key, m_brick_sizes, m_is_brick_periodic);

      auto prod_scal = normal[IX] * shift[IX] + normal[IY] * shift[IY];
      if constexpr (dim == 3)
        prod_scal += normal[IZ] * shift[IZ];

      // if this scalar prod is greater than zero it means we are looking for a neighbor that is
      // really outside domain
      // this scalar prod will be zero if we are at a periodic border, and then the neighbor will be
      // found in the hashmap
      if (prod_scal > 0)
      {
        face_loc_neigh.is_outside_domain = true;
      }
    }

    return face_loc_neigh;
  }

  //
  // check if neighbor cell belongs to a block at coarser level
  //
  auto key_neigh_coarser = orchard_key_t<dim>::father(key_neigh_same_level);

  key_neigh_hashindex = m_amr_hashmap_device.find(key_neigh_coarser);

  auto is_at_coarser_level = m_amr_hashmap_device.valid_at(key_neigh_hashindex);

  if (is_at_coarser_level)
  {
    auto iOct_neigh = m_amr_hashmap_device.value_at(key_neigh_hashindex);

    // get child id of current block (needed to compute neighbor face coordinates)
    const auto child_id = orchard_key_t<dim>::child_id(key_neigh_same_level);

    coord_neigh_u[IX] = (coord_neigh_u[IX] + ((child_id & 0x1) >> 0) * face_block_sizes[IX]) / 2;
    coord_neigh_u[IY] = (coord_neigh_u[IY] + ((child_id & 0x2) >> 1) * face_block_sizes[IY]) / 2;
    if constexpr (dim == 3)
    {
      coord_neigh_u[IZ] = (coord_neigh_u[IZ] + ((child_id & 0x4) >> 2) * face_block_sizes[IZ]) / 2;
    }

    FaceLocation_t face_loc_neigh{ coord_neigh_u, key_neigh_coarser, iOct_neigh, false };
    return face_loc_neigh;
  }

  //
  // check if neighbor cell belongs to a block at finer level
  //
  {
    coord_neigh_u[IX] = coord_neigh_u[IX] * 2;
    coord_neigh_u[IY] = coord_neigh_u[IY] * 2;
    if constexpr (dim == 3)
    {
      coord_neigh_u[IZ] = coord_neigh_u[IZ] * 2;
    }

    // determine in which finer octant we need to use (i.e. determine child id)
    uint8_t child_id = 0;
    if (coord_neigh_u[IX] >= cell_block_sizes[IX])
    {
      coord_neigh_u[IX] -= cell_block_sizes[IX];
      child_id += 1;
    }
    if (coord_neigh_u[IY] >= cell_block_sizes[IY])
    {
      coord_neigh_u[IY] -= cell_block_sizes[IY];
      child_id += 2;
    }
    if constexpr (dim == 3)
    {
      if (coord_neigh_u[IZ] >= cell_block_sizes[IZ])
      {
        coord_neigh_u[IZ] -= cell_block_sizes[IZ];
        child_id += 4;
      }
    }

    // lookup for this child in the hashmap
    const auto key_neigh_fine = orchard_key_t<dim>::child(key_neigh_same_level, child_id);

    const auto key_neigh_fine_hashindex = m_amr_hashmap_device.find(key_neigh_fine);
    const auto valid = m_amr_hashmap_device.valid_at(key_neigh_fine_hashindex);

    if (valid)
    {
      auto           iOct_neigh = m_amr_hashmap_device.value_at(key_neigh_fine_hashindex);
      FaceLocation_t face_loc_neigh{ coord_neigh_u, key_neigh_fine, iOct_neigh, false };
      return face_loc_neigh;
    }
  } // end search for neighbor at finer level

  // if we are here, we have a genuine problem/bug !!!

  // default : return "self" as invalid value here
  return face_loc;

} // StencilHelper<dim, device_t>::getNeighLoc - face

// =========================================================================================
// =========================================================================================
template <size_t dim, typename device_t>
KOKKOS_FUNCTION auto
StencilHelper<dim, device_t>::getNeighLocCoarser(CellLocation<dim> const & cell_loc,
                                                 shift_t<dim> shift) const -> CellLocation_t
{

  const auto & b = m_block_sizes;

  // beware signed values
  coord_t<dim, int32_t> coord_neigh;
  coord_neigh[IX] = static_cast<int32_t>(cell_loc.ijk[IX]) + shift[IX];
  coord_neigh[IY] = static_cast<int32_t>(cell_loc.ijk[IY]) + shift[IY];
  if constexpr (dim == 3)
    coord_neigh[IZ] = static_cast<int32_t>(cell_loc.ijk[IZ]) + shift[IZ];

  // compute direction to neighbor octant
  auto dir = []() {
    if constexpr (dim == 2)
      return Kokkos::Array<int8_t, 2>{ 0, 0 };
    else if constexpr (dim == 3)
      return Kokkos::Array<int8_t, 3>{ 0, 0, 0 };
  }();

  if (coord_neigh[IX] < 0)
    dir[IX] = -1;
  else if (coord_neigh[IX] >= static_cast<int32_t>(b[IX]))
    dir[IX] = 1;

  if (coord_neigh[IY] < 0)
    dir[IY] = -1;
  else if (coord_neigh[IY] >= static_cast<int32_t>(b[IY]))
    dir[IY] = 1;

  if constexpr (dim == 3)
  {
    if (coord_neigh[IZ] < 0)
      dir[IZ] = -1;
    else if (coord_neigh[IZ] >= static_cast<int32_t>(b[IZ]))
      dir[IZ] = 1;
  }

  [[maybe_unused]] const auto dir_norm = [&dir]() {
    if constexpr (dim == 2)
      return dir[IX] * dir[IX] + dir[IY] * dir[IY];
    else if constexpr (dim == 3)
      return dir[IX] * dir[IX] + dir[IY] * dir[IY] + dir[IZ] * dir[IZ];
  }();

  // dir_norm can't be 0, neighbor cell is necessarily outside current block
  KOKKOS_ASSERT(dir_norm > 0);

  // compute neighbor cell coord in the neighbor block
  coord_t<dim> coord_neigh_u;
  coord_neigh_u[IX] = static_cast<typename coord_t<dim>::value_type>(
    coord_neigh[IX] - b[IX] * (dir[IX] == 1) + b[IX] * (dir[IX] == -1));
  coord_neigh_u[IY] = static_cast<typename coord_t<dim>::value_type>(
    coord_neigh[IY] - b[IY] * (dir[IY] == 1) + b[IY] * (dir[IY] == -1));
  if constexpr (dim == 3)
  {
    coord_neigh_u[IZ] = static_cast<typename coord_t<dim>::value_type>(
      coord_neigh[IZ] - b[IZ] * (dir[IZ] == 1) + b[IZ] * (dir[IZ] == -1));
  }

  // note that neighbor cell can't be in an outside block (because face neighbor that are outside
  // are necessarily conform by design, so they can't be at coarser level)

  // get neighbor key at same level
  const auto key_neigh_same_level = orchard_key_t<dim>::get_neighbor_key_same_level(
    cell_loc.key, dir, m_brick_sizes, m_is_brick_periodic);

  const auto key_neigh_coarser = orchard_key_t<dim>::father(key_neigh_same_level);

  const auto key_neigh_hashindex = m_amr_hashmap_device.find(key_neigh_coarser);

  // neighbor MUST be at coarser level, if not it means
  // - either current function is called without checking neighbor is coarser
  // - either we have a genuine bug
  [[maybe_unused]] const auto is_at_coarser_level =
    m_amr_hashmap_device.valid_at(key_neigh_hashindex);
  KOKKOS_ASSERT(is_at_coarser_level);

  const auto iOct_neigh = m_amr_hashmap_device.value_at(key_neigh_hashindex);

  // get child id of current block (needed to compute neighbor cell coordinates)
  const auto child_id = orchard_key_t<dim>::child_id(key_neigh_same_level);

  coord_neigh_u[IX] = (coord_neigh_u[IX] + ((child_id & 0x1) >> 0) * b[IX]) / 2;
  coord_neigh_u[IY] = (coord_neigh_u[IY] + ((child_id & 0x2) >> 1) * b[IY]) / 2;
  if constexpr (dim == 3)
  {
    coord_neigh_u[IZ] = (coord_neigh_u[IZ] + ((child_id & 0x4) >> 2) * b[IZ]) / 2;
  }

  CellLocation_t cell_loc_neigh{ coord_neigh_u, key_neigh_coarser, iOct_neigh, false };

  return cell_loc_neigh;

} // StencilHelper<dim, device_t>::getNeighLocCoarser

// =========================================================================================
// =========================================================================================
template <size_t dim, typename device_t>
KOKKOS_FUNCTION auto
StencilHelper<dim, device_t>::getNeighLocFiner(CellLocation<dim> const & cell_loc,
                                               shift_t<dim> shift) const -> CellLocation_t
{

  const auto & b = m_block_sizes;

  // beware signed values
  coord_t<dim, int32_t> coord_neigh;
  coord_neigh[IX] = static_cast<int32_t>(cell_loc.ijk[IX]) + shift[IX];
  coord_neigh[IY] = static_cast<int32_t>(cell_loc.ijk[IY]) + shift[IY];
  if constexpr (dim == 3)
    coord_neigh[IZ] = static_cast<int32_t>(cell_loc.ijk[IZ]) + shift[IZ];

  // compute direction to neighbor octant
  auto dir = []() {
    if constexpr (dim == 2)
      return Kokkos::Array<int8_t, 2>{ 0, 0 };
    else if constexpr (dim == 3)
      return Kokkos::Array<int8_t, 3>{ 0, 0, 0 };
  }();

  if (coord_neigh[IX] < 0)
    dir[IX] = -1;
  else if (coord_neigh[IX] >= static_cast<int32_t>(b[IX]))
    dir[IX] = 1;

  if (coord_neigh[IY] < 0)
    dir[IY] = -1;
  else if (coord_neigh[IY] >= static_cast<int32_t>(b[IY]))
    dir[IY] = 1;

  if constexpr (dim == 3)
  {
    if (coord_neigh[IZ] < 0)
      dir[IZ] = -1;
    else if (coord_neigh[IZ] >= static_cast<int32_t>(b[IZ]))
      dir[IZ] = 1;
  }

  [[maybe_unused]] const auto dir_norm = [&dir]() {
    if constexpr (dim == 2)
      return dir[IX] * dir[IX] + dir[IY] * dir[IY];
    else if constexpr (dim == 3)
      return dir[IX] * dir[IX] + dir[IY] * dir[IY] + dir[IZ] * dir[IZ];
  }();

  // dir_norm can't be 0, neighbor cell is necessarily outside current block
  KOKKOS_ASSERT(dir_norm > 0);

  // compute neighbor cell coord in the neighbor block
  coord_t<dim> coord_neigh_u;
  coord_neigh_u[IX] = static_cast<typename coord_t<dim>::value_type>(
    coord_neigh[IX] - b[IX] * (dir[IX] == 1) + b[IX] * (dir[IX] == -1));
  coord_neigh_u[IY] = static_cast<typename coord_t<dim>::value_type>(
    coord_neigh[IY] - b[IY] * (dir[IY] == 1) + b[IY] * (dir[IY] == -1));
  if constexpr (dim == 3)
  {
    coord_neigh_u[IZ] = static_cast<typename coord_t<dim>::value_type>(
      coord_neigh[IZ] - b[IZ] * (dir[IZ] == 1) + b[IZ] * (dir[IZ] == -1));
  }

  // note that neighbor cell can't be in an outside block (because face neighbor that are outside
  // are necessarily conform by design, so they can't be at finer level)

  coord_neigh_u = coord_neigh_u * 2;

  // determine in which finer octant we need to use (i.e. determine child id)
  uint8_t child_id = 0;
  if (coord_neigh_u[IX] >= b[IX])
  {
    coord_neigh_u[IX] -= b[IX];
    child_id += 1;
  }
  if (coord_neigh_u[IY] >= b[IY])
  {
    coord_neigh_u[IY] -= b[IY];
    child_id += 2;
  }
  if constexpr (dim == 3)
  {
    if (coord_neigh_u[IZ] >= b[IZ])
    {
      coord_neigh_u[IZ] -= b[IZ];
      child_id += 4;
    }
  }

  // get neighbor key at same level
  const auto key_neigh_same_level = orchard_key_t<dim>::get_neighbor_key_same_level(
    cell_loc.key, dir, m_brick_sizes, m_is_brick_periodic);

  // lookup for this child in the hashmap
  const auto key_neigh_fine = orchard_key_t<dim>::child(key_neigh_same_level, child_id);

  const auto key_neigh_hashindex = m_amr_hashmap_device.find(key_neigh_fine);

  // neighbor MUST be at coarser level, if not it means
  // - either current function is called without checking neighbor is finer
  // - either we have a genuine bug
  [[maybe_unused]] const auto is_at_finer_level =
    m_amr_hashmap_device.valid_at(key_neigh_hashindex);
  KOKKOS_ASSERT(is_at_finer_level);

  const auto iOct_neigh = m_amr_hashmap_device.value_at(key_neigh_hashindex);

  CellLocation_t cell_loc_neigh{ coord_neigh_u, key_neigh_fine, iOct_neigh, false };

  return cell_loc_neigh;

} // StencilHelper<dim, device_t>::getNeighLocFiner

// =========================================================================================
// =========================================================================================
template <size_t dim, typename device_t>
KOKKOS_FUNCTION auto
StencilHelper<dim, device_t>::getNeighLocFinerNearer(CellLocation<dim> const & cell_loc,
                                                     shift_t<dim> shift) const -> CellLocation_t
{
  auto cell_loc_neigh = getNeighLocFiner(cell_loc, shift);

  if (shift[IX] < 0)
  {
    cell_loc_neigh.ijk[IX] += 1;
  };

  if (shift[IY] < 0)
  {
    cell_loc_neigh.ijk[IY] += 1;
  };

  if constexpr (dim == 3)
  {
    if (shift[IZ] < 0)
    {
      cell_loc_neigh.ijk[IZ] += 1;
    };
  }

  return cell_loc_neigh;

} // StencilHelper<dim, device_t>::getNeighLocFinerNearer

// =========================================================================================
// =========================================================================================
template <size_t dim, typename device_t>
KOKKOS_FUNCTION auto
StencilHelper<dim, device_t>::getBorderFaceLocSymmetric(FaceLocation<dim> const & face_loc) const
  -> FaceLocation_t
{

  const auto & cell_block_sizes = m_block_sizes;
  auto const & ivar = face_loc.ijk[dim];

  const auto           cell_coord = face_to_cell_coords<dim>(face_loc.ijk, cell_block_sizes);
  const CellLocation_t cell_loc_cur{
    cell_coord, face_loc.key, face_loc.iOct, face_loc.is_outside_domain
  };

  // check if face is at border
  if (ivar == IX and face_loc.ijk[IX] == 0)
  {
    const auto cell_loc_neigh = getNeighLoc(cell_loc_cur, unit_shift(-XDIR));
    auto       face_index_neigh = to_face_multiindex<dim>(cell_loc_neigh.ijk, ivar);
    face_index_neigh[IX] = cell_block_sizes[IX];
    const FaceLocation_t sym_face_loc{
      face_index_neigh, cell_loc_neigh.key, cell_loc_neigh.iOct, cell_loc_neigh.is_outside_domain
    };
    return sym_face_loc;
  }
  else if (ivar == IX and face_loc.ijk[IX] == cell_block_sizes[IX])
  {
    const auto cell_loc_neigh = getNeighLoc(cell_loc_cur, unit_shift(+XDIR));
    auto       face_index_neigh = to_face_multiindex<dim>(cell_loc_neigh.ijk, ivar);
    face_index_neigh[IX] = 0;
    const FaceLocation_t sym_face_loc{
      face_index_neigh, cell_loc_neigh.key, cell_loc_neigh.iOct, cell_loc_neigh.is_outside_domain
    };
    return sym_face_loc;
  }
  else if (ivar == IY and face_loc.ijk[IY] == 0)
  {
    const auto cell_loc_neigh = getNeighLoc(cell_loc_cur, unit_shift(-YDIR));
    auto       face_index_neigh = to_face_multiindex<dim>(cell_loc_neigh.ijk, ivar);
    face_index_neigh[IY] = cell_block_sizes[IY];
    const FaceLocation_t sym_face_loc{
      face_index_neigh, cell_loc_neigh.key, cell_loc_neigh.iOct, cell_loc_neigh.is_outside_domain
    };
    return sym_face_loc;
  }
  else if (ivar == IY and face_loc.ijk[IY] == cell_block_sizes[IY])
  {
    const auto cell_loc_neigh = getNeighLoc(cell_loc_cur, unit_shift(+YDIR));
    auto       face_index_neigh = to_face_multiindex<dim>(cell_loc_neigh.ijk, ivar);
    face_index_neigh[IY] = 0;
    const FaceLocation_t sym_face_loc{
      face_index_neigh, cell_loc_neigh.key, cell_loc_neigh.iOct, cell_loc_neigh.is_outside_domain
    };
    return sym_face_loc;
  }

  if constexpr (dim == 3)
  {
    if (ivar == IZ and face_loc.ijk[IZ] == 0)
    {
      const auto cell_loc_neigh = getNeighLoc(cell_loc_cur, unit_shift(-ZDIR));
      auto       face_index_neigh = to_face_multiindex<dim>(cell_loc_neigh.ijk, ivar);
      face_index_neigh[IZ] = cell_block_sizes[IZ];
      const FaceLocation_t sym_face_loc{
        face_index_neigh, cell_loc_neigh.key, cell_loc_neigh.iOct, cell_loc_neigh.is_outside_domain
      };
      return sym_face_loc;
    }
    else if (ivar == IZ and face_loc.ijk[IZ] == cell_block_sizes[IZ])
    {
      const auto cell_loc_neigh = getNeighLoc(cell_loc_cur, unit_shift(+ZDIR));
      auto       face_index_neigh = to_face_multiindex<dim>(cell_loc_neigh.ijk, ivar);
      face_index_neigh[IZ] = 0;
      const FaceLocation_t sym_face_loc{
        face_index_neigh, cell_loc_neigh.key, cell_loc_neigh.iOct, cell_loc_neigh.is_outside_domain
      };
      return sym_face_loc;
    }
  }

  FaceLocation_t default_face_loc = face_loc;

  return default_face_loc;

} // getBorderFaceLocSymmetric

// =========================================================================================
// =========================================================================================
template <size_t dim, typename device_t>
KOKKOS_FUNCTION auto
StencilHelper<dim, device_t>::edge_to_cell_location(EdgeLocation<dim> const & edge_loc) const
  -> CellLocation_t
{
  auto const & cell_block_sizes = m_block_sizes;

  const auto cell_coord = edge_to_cell_coords<dim>(edge_loc.ijk, cell_block_sizes);

  const CellLocation_t res{ cell_coord, edge_loc.key, edge_loc.iOct, edge_loc.is_outside_domain };

  return res;
}

// =========================================================================================
// =========================================================================================
template <size_t dim, typename device_t>
KOKKOS_FUNCTION auto
StencilHelper<dim, device_t>::getBorderEdgeLocSymmetric(EdgeLocation<dim> const & edge_loc,
                                                        EdgeNormalType edge_normal_type) const
  -> EdgeLocation_t
{

  const auto & cell_block_sizes = m_block_sizes;
  auto const & edge_dir = edge_loc.ijk[dim];

  const auto           cell_coord = edge_to_cell_coords<dim>(edge_loc.ijk, cell_block_sizes);
  const CellLocation_t cell_loc_cur{
    cell_coord, edge_loc.key, edge_loc.iOct, edge_loc.is_outside_domain
  };

  // shift vector to neighbor block where symmetric edge is located
  const auto out_shift =
    get_edge_outside_unit_vector<dim>(edge_loc.ijk, cell_block_sizes, edge_normal_type);

  const auto cell_loc_neigh = getNeighLoc(cell_loc_cur, out_shift);

  // if neighbor is coarser, just return edge_loc itself
  //
  // currently, the main use of this routine is when neighbor it at same or finer level
  // when neighbor is coarser, there may be no-colocated corresponding edge on the other side.
  //
  // \todo evaluate if something more useful could be returned here, if really needed
  if (cell_loc_neigh.level() < cell_loc_cur.level())
  {
    EdgeLocation_t sym_edge_loc = edge_loc;
    return sym_edge_loc;
  }

  auto edge_index_neigh = to_edge_multiindex<dim>(cell_loc_neigh.ijk, edge_dir);

  for (size_t idim = 0; idim < dim; ++idim)
  {
    if (out_shift[idim] < 0)
      edge_index_neigh[idim] = cell_block_sizes[idim];
    else if (out_shift[idim] > 0)
      edge_index_neigh[idim] = 0;
  }

  // watch out inversion dir1 <==> dir2
  if (edge_normal_type == +EdgeNormalType::DIR1)
  {
    const auto dir2 = static_cast<size_t>((edge_dir + 2) % 3);
    if (edge_loc.ijk[dir2] == cell_block_sizes[dir2])
      edge_index_neigh[dir2] = cell_block_sizes[dir2];
  }
  else if (edge_normal_type == +EdgeNormalType::DIR2)
  {
    const auto dir1 = static_cast<size_t>((edge_dir + 1) % 3);
    if (edge_loc.ijk[dir1] == cell_block_sizes[dir1])
      edge_index_neigh[dir1] = cell_block_sizes[dir1];
  }

  const EdgeLocation_t sym_edge_loc{ edge_index_neigh,
                                     cell_loc_neigh.key,
                                     cell_loc_neigh.iOct,
                                     cell_loc_neigh.is_outside_domain,
                                     true };

  return sym_edge_loc;

} // getBorderEdgeLocSymmetric

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_FUNCTION auto
StencilHelper<dim, device_t>::getEdgeSiblingLoc(EdgeLocation_t const & edge_loc_in,
                                                int nth_shift) const -> EdgeLocation_t
{
  const auto cell_edge_location = get_CellEdgeLocation(edge_loc_in, m_block_sizes);
  KOKKOS_ASSERT((cell_edge_location >= 0 and cell_edge_location < 4) &&
                "Wrong value for a CellEdgeLocation : must be integer in range [0,3]");

  auto const & edge_dir = edge_loc_in.ijk[dim];
  const auto   cell_shift = EdgeUtils::edge_neighbor_shift<dim>(
    cell_edge_location, nth_shift, static_cast<ComponentIndex3D>(edge_dir));

  // get current cell location
  const auto cell_loc = edge_to_cell_location(edge_loc_in);

  // get shifted cell location
  const auto cell_loc2 = getNeighLoc(cell_loc, cell_shift);

  // if shifted cell is inside current octant, just return edge_loc_in
  if (cell_loc2.iOct == edge_loc_in.iOct)
    return edge_loc_in;

  // shifted cell is outside current octant, we need to compute the edge coordinates as seen from
  // the shifted cell

  // get edge transverse directions for an edge along IZ
  size_t dir1 = IX;
  size_t dir2 = IY;
  if (edge_dir == IX)
  {
    dir1 = IY;
    dir2 = IZ;
  }
  else if (edge_dir == IY)
  {
    dir1 = IX;
    dir2 = IZ;
  }

  auto ijk_sibling = cell_loc2.ijk;

  // shift back to have "correct" edge index
  if (cell_loc2.level() <= cell_loc.level())
  {
    // neighbor cell is at same or coarser AMR level => shift by 1
    if (cell_shift[dir1] < 0 or (cell_shift[dir1] == 0 and
                                 (cell_edge_location == EDGE_01 or cell_edge_location == EDGE_11)))
      ijk_sibling[dir1]++;
    if (cell_shift[dir2] < 0 or (cell_shift[dir2] == 0 and
                                 (cell_edge_location == EDGE_10 or cell_edge_location == EDGE_11)))
      ijk_sibling[dir2]++;
  }
  else
  {
    // neighbor cell is at finer AMR level => shift by 2
    if (cell_shift[dir1] < 0 or (cell_shift[dir1] == 0 and
                                 (cell_edge_location == EDGE_01 or cell_edge_location == EDGE_11)))
      ijk_sibling[dir1] += 2;
    if (cell_shift[dir2] < 0 or (cell_shift[dir2] == 0 and
                                 (cell_edge_location == EDGE_10 or cell_edge_location == EDGE_11)))
      ijk_sibling[dir2] += 2;
  }

  // check edge location validity:
  // is edge touching a coarser neighbor through the middle of a face ? in that case edge location
  // is invalid
  auto is_edge_valid = true;
  if (cell_loc2.level() < cell_loc.level())
  {
    if ((edge_loc_in.ijk[dir1] == 0 or edge_loc_in.ijk[dir1] == m_block_sizes[dir1]) and
        ((edge_loc_in.ijk[dir2] & 0x1) == 1))
      is_edge_valid = false;

    if ((edge_loc_in.ijk[dir2] == 0 or edge_loc_in.ijk[dir2] == m_block_sizes[dir2]) and
        ((edge_loc_in.ijk[dir1] & 0x1) == 1))
      is_edge_valid = false;
  }

  const auto edge_index_out = to_edge_multiindex<dim>(ijk_sibling, edge_dir);

  auto edge_loc_out = EdgeLocation_t(
    edge_index_out, cell_loc2.key, cell_loc2.iOct, cell_loc2.is_outside_domain, is_edge_valid);

  return edge_loc_out;

} // StencilHelper<dim, device_t>::getEdgeSiblingLoc

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_FUNCTION void
StencilHelper<dim, device_t>::getAllEdgeSiblingLoc(EdgeLocation_t const & edge_loc,
                                                   EdgeLocation_t &       edge_loc0,
                                                   EdgeLocation_t &       edge_loc1,
                                                   EdgeLocation_t &       edge_loc2) const
{
  edge_loc0 = getEdgeSiblingLoc(edge_loc, 0);
  edge_loc1 = getEdgeSiblingLoc(edge_loc, 1);
  edge_loc2 = getEdgeSiblingLoc(edge_loc, 2);
}

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_FUNCTION real_t
StencilHelper<dim, device_t>::compute_siblings_average(
  CellLocation<dim> const &                     cell_loc,
  block_size_t<dim> const &                     block_sizes,
  int                                           ivar,
  DataArrayBlock<dim, real_t, device_t> const & userdata) const
{
  auto const & b = block_sizes;
  auto const & coord = cell_loc.ijk;

  KOKKOS_ASSERT(cell_loc.cellindex(userdata.block_size()) < userdata.num_cells() &&
                "compute_siblings_average: wrong cell_loc's cellindex");
  KOKKOS_ASSERT(cell_loc.iOct < userdata.num_quadrants() &&
                "compute_siblings_average: wrong cell_loc's octant id");
  KOKKOS_ASSERT(ivar < userdata.num_vars() && "compute_siblings_average: wrong var id");


  // get eldest siblings coordinates (all even coordinates)
  coord_t<dim> coord0;
  coord0[IX] = (coord[IX] / 2) * 2;
  coord0[IY] = (coord[IY] / 2) * 2;
  if constexpr (dim == 3)
    coord0[IZ] = (coord[IZ] / 2) * 2;

  auto cell_index0 = coord_to_cellindex<dim>(coord0, b);

  real_t   data = 0;
  uint16_t nbCells = 1 << dim;
  for (uint16_t i = 0; i < nbCells; ++i)
  {
    if constexpr (dim == 2)
    {
      // clang-format off
      data += userdata(cell_index0 +
                       ((i & 0x1) >> 0) +
                       ((i & 0x2) >> 1) * b[IX],
                       ivar, cell_loc.iOct);
      // clang-format on
    }
    else if constexpr (dim == 3)
    {
      // clang-format off
      data += userdata(cell_index0 +
                       ((i & 0x1) >> 0) +
                       ((i & 0x2) >> 1) * b[IX] +
                       ((i & 0x4) >> 2) * b[IX] * b[IY],
                       ivar,
                       cell_loc.iOct);
      // clang-format on
    }
  }

  return data / nbCells;

} // StencilHelper<dim, device_t>::compute_siblings_average

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_FUNCTION real_t
StencilHelper<dim, device_t>::compute_siblings_average(
  CellLocation<dim> const &                            cell_loc,
  int                                                  ivar,
  DataArrayGhostedBlock<dim, real_t, device_t> const & userdata) const
{
  // auto const & b = block_sizes;
  auto const & coord = cell_loc.ijk;

  KOKKOS_ASSERT(cell_loc.cellindex(userdata.block_size()) < userdata.num_cells_inner() &&
                "compute_siblings_average: wrong cell_loc's cellindex");
  KOKKOS_ASSERT(cell_loc.iOct < userdata.num_quadrants() &&
                "compute_siblings_average: wrong cell_loc's octant id");
  KOKKOS_ASSERT(ivar < userdata.num_vars() && "compute_siblings_average: wrong var id");


  // get eldest siblings coordinates (all even coordinates)
  coord_t<dim> coord0;
  coord0[IX] = (coord[IX] / 2) * 2;
  coord0[IY] = (coord[IY] / 2) * 2;
  if constexpr (dim == 3)
    coord0[IZ] = (coord[IZ] / 2) * 2;

  // auto cell_index0 = coord_to_cellindex<dim>(coord0, b);
  auto const & i0 = coord0[IX];
  auto const & j0 = coord0[IY];

  real_t   data = 0;
  uint16_t nbCells = 1 << dim;

  if constexpr (dim == 2)
  {

    for (int jj = 0; jj < 2; ++jj)
    {
      for (int ii = 0; ii < 2; ++ii)
      {
        data += userdata(i0 + ii, j0 + jj, ivar, cell_loc.iOct);
      }
    }
  } // dim == 2
  else if constexpr (dim == 3)
  {
    auto const & k0 = coord0[IZ];

    for (int kk = 0; kk < 2; ++kk)
    {
      for (int jj = 0; jj < 2; ++jj)
      {
        for (int ii = 0; ii < 2; ++ii)
        {
          data += userdata(i0 + ii, j0 + jj, k0 + kk, ivar, cell_loc.iOct);
        }
      }
    }
  } // dim==3

  return data / nbCells;

} // StencilHelper<dim, device_t>::compute_siblings_average

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_FUNCTION real_t
StencilHelper<dim, device_t>::compute_siblings_average(
  CellLocation<dim> const &        cell_loc,
  block_size_t<dim> const &        block_sizes,
  int                              ivar,
  DataArrayBlockMultiVar_t const & userdata) const
{
  auto const & b = block_sizes;
  auto const & coord = cell_loc.ijk;

  KOKKOS_ASSERT(cell_loc.cellindex(userdata.block_size()) < userdata.num_cells() &&
                "compute_siblings_average: wrong cell_loc's cellindex");
  KOKKOS_ASSERT(cell_loc.iOct < userdata.num_quadrants() &&
                "compute_siblings_average: wrong cell_loc's octant id");
  KOKKOS_ASSERT(ivar < userdata.num_vars(cell_loc.iOct) &&
                "compute_siblings_average: wrong var id");


  // get eldest siblings coordinates (all even coordinates)
  coord_t<dim> coord0;
  coord0[IX] = (coord[IX] / 2) * 2;
  coord0[IY] = (coord[IY] / 2) * 2;
  if constexpr (dim == 3)
    coord0[IZ] = (coord[IZ] / 2) * 2;

  auto cell_index0 = coord_to_cellindex<dim>(coord0, b);

  real_t   data = 0;
  uint16_t nbCells = 1 << dim;
  for (uint16_t i = 0; i < nbCells; ++i)
  {
    if constexpr (dim == 2)
    {
      // clang-format off
      data += userdata(cell_index0 +
                       ((i & 0x1) >> 0) +
                       ((i & 0x2) >> 1) * b[IX],
                       ivar, cell_loc.iOct);
      // clang-format on
    }
    else if constexpr (dim == 3)
    {
      // clang-format off
      data += userdata(cell_index0 +
                       ((i & 0x1) >> 0) +
                       ((i & 0x2) >> 1) * b[IX] +
                       ((i & 0x4) >> 2) * b[IX] * b[IY],
                       ivar,
                       cell_loc.iOct);
      // clang-format on
    }
  }

  return data / nbCells;

} // StencilHelper<dim, device_t>::compute_siblings_average

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_FUNCTION real_t
StencilHelper<dim, device_t>::compute_face_siblings_average(
  CellLocation<dim> const &    cell_loc,
  int                          ivar,
  face_type_t                  ivar_face,
  FaceDataArrayBlock_t const & Bface) const
{
  // only used in debug mode
  [[maybe_unused]] const auto & b = m_block_sizes;

  auto const & coord = cell_loc.ijk;

  // get eldest siblings coordinates (all even coordinates)
  const auto coord0 = [&coord]() {
    if constexpr (dim == 2)
    {
      coord_t<2> res{ (coord[IX] / 2) * 2, (coord[IY] / 2) * 2 };
      return res;
    }
    else if constexpr (dim == 3)
    {
      coord_t<3> res{ (coord[IX] / 2) * 2, (coord[IY] / 2) * 2, (coord[IZ] / 2) * 2 };
      return res;
    }
  }();

  const auto offset_x = (ivar == IX and ivar_face == face_type_t::RIGHT) ? 1 : 0;
  const auto offset_y = (ivar == IY and ivar_face == face_type_t::RIGHT) ? 1 : 0;
  const auto offset_z = (ivar == IZ and ivar_face == face_type_t::RIGHT) ? 1 : 0;

  real_t value = 0;

  // number of face siblings :
  // - 2 in 2d
  // - 4 in 3d
  // that is 2^(dim-1)
  // except along IZ direction, it is always 4 (2d and 3d) !!
  const int num_face_siblings = (ivar == IZ) ? 4 : 1 << (dim - 1);

  if constexpr (dim == 2)
  {
    for (auto jj = 0; jj < 2; ++jj)
    {
      for (auto ii = 0; ii < 2; ++ii)
      {

        KOKKOS_ASSERT((coord0[IX] + ii < b[IX]) && "Wrong values for coord0[IX]");
        KOKKOS_ASSERT((coord0[IY] + jj < b[IY]) && "Wrong values for coord0[IY]");

        // compute corresponding cell index in current block of cells
        const auto coord_sibling = coord_t<2>{ coord0[IX] + ii, coord0[IY] + jj };

        // average over face siblings
        // if ivar is IZ, we take them all
        if ((ivar == IX and coord_sibling[IX] == coord[IX]) or
            (ivar == IY and coord_sibling[IY] == coord[IY]) or (ivar == IZ))
        {
          // clang-format off
          value +=
            Bface(coord_sibling[IX] + offset_x,
                  coord_sibling[IY] + offset_y,
                  ivar,
                  cell_loc.iOct);
          // clang-format on
        }
      } // for ii
    } // for jj
  }
  else if constexpr (dim == 3)
  {
    for (auto kk = 0; kk < 2; ++kk)
    {
      for (auto jj = 0; jj < 2; ++jj)
      {
        for (auto ii = 0; ii < 2; ++ii)
        {

          KOKKOS_ASSERT((coord0[IX] + ii < b[IX]) && "Wrong values for coord0[IX]");
          KOKKOS_ASSERT((coord0[IY] + jj < b[IY]) && "Wrong values for coord0[IY]");
          KOKKOS_ASSERT((coord0[IZ] + kk < b[IZ]) && "Wrong values for coord0[IZ]");

          // compute corresponding cell index in current block of cells
          const auto coord_sibling =
            coord_t<3>{ coord0[IX] + ii, coord0[IY] + jj, coord0[IZ] + kk };

          if ((ivar == IX and coord_sibling[IX] == coord[IX]) or
              (ivar == IY and coord_sibling[IY] == coord[IY]) or
              (ivar == IZ and coord_sibling[IZ] == coord[IZ]))
          {
            // clang-format off
            value += Bface(coord_sibling[IX] + offset_x,
                           coord_sibling[IY] + offset_y,
                           coord_sibling[IZ] + offset_z,
                           ivar,
                           cell_loc.iOct);
            // clang-format on
          }
        } // for ii
      } // for jj
    } // for kk
  }

  return value / static_cast<real_t>(num_face_siblings);

} // StencilHelper<dim, device_t>::compute_face_siblings_average

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_FUNCTION real_t
StencilHelper<dim, device_t>::compute_face_siblings_average(
  FaceLocation<dim> const &                  face_loc,
  [[maybe_unused]] block_size_t<dim> const & cell_block_size,
  int32_t                                    ivar,
  DataArrayBlock_t const &                   fluxes) const
{
  // check that fluxes is a flux array along the given face location (normal)
  {
    [[maybe_unused]] const auto dir = face_loc.ijk[dim];
    KOKKOS_ASSERT((fluxes.shape()[dir] == cell_block_size[dir] + 1) && "fluxes has wrong size");
    KOKKOS_ASSERT((fluxes.shape()[static_cast<size_t>(dir + 1) % dim] ==
                   cell_block_size[static_cast<size_t>(dir + 1) % dim]) &&
                  "fluxes has wrong size");
    if constexpr (dim == 3)
    {
      KOKKOS_ASSERT((fluxes.shape()[static_cast<size_t>(dir + 2) % dim] ==
                     cell_block_size[static_cast<size_t>(dir + 2) % dim]) &&
                    "fluxes has wrong size");
    }
  }

  auto const & ijk = face_loc.ijk;

  auto const & dir = face_loc.ijk[dim];

  real_t value = 0;

  // number of face siblings :
  // - 2 in 2d
  // - 4 in 3d
  // that is 2^(dim-1)
  const int num_face_siblings = 1 << (dim - 1);

  if constexpr (dim == 2)
  {
    // clang-format off
    if (dir == IX)
    {
      const auto i0 =  ijk[IX];
      const auto j0 = (ijk[IY] / 2) * 2;

      value += fluxes(i0, j0    , ivar, face_loc.iOct);
      value += fluxes(i0, j0 + 1, ivar, face_loc.iOct);
    }
    else if (dir == IY)
    {
      const auto i0 = (ijk[IX] / 2) * 2;
      const auto j0 =  ijk[IY];

      value += fluxes(i0    , j0, ivar, face_loc.iOct);
      value += fluxes(i0 + 1, j0, ivar, face_loc.iOct);
    }
    // clang-format on
  }
  else if constexpr (dim == 3)
  {
    // clang-format off
    if (dir == IX)
    {
      const auto i0 =  ijk[IX];
      const auto j0 = (ijk[IY] / 2) * 2;
      const auto k0 = (ijk[IZ] / 2) * 2;
      value += fluxes(i0    , j0    , k0    , ivar, face_loc.iOct);
      value += fluxes(i0    , j0 + 1, k0    , ivar, face_loc.iOct);
      value += fluxes(i0    , j0    , k0 + 1, ivar, face_loc.iOct);
      value += fluxes(i0    , j0 + 1, k0 + 1, ivar, face_loc.iOct);
    }
    else if (dir == IY)
    {
      const auto i0 = (ijk[IX] / 2) * 2;
      const auto j0 =  ijk[IY];
      const auto k0 = (ijk[IZ] / 2) * 2;
      value += fluxes(i0    , j0    , k0    , ivar, face_loc.iOct);
      value += fluxes(i0 + 1, j0    , k0    , ivar, face_loc.iOct);
      value += fluxes(i0    , j0    , k0 + 1, ivar, face_loc.iOct);
      value += fluxes(i0 + 1, j0    , k0 + 1, ivar, face_loc.iOct);
    }
    else if (dir == IZ)
    {
      const auto i0 = (ijk[IX] / 2) * 2;
      const auto j0 = (ijk[IY] / 2) * 2;
      const auto k0 =  ijk[IZ];
      value += fluxes(i0    , j0    , k0    , ivar, face_loc.iOct);
      value += fluxes(i0 + 1, j0    , k0    , ivar, face_loc.iOct);
      value += fluxes(i0    , j0 + 1, k0    , ivar, face_loc.iOct);
      value += fluxes(i0 + 1, j0 + 1, k0    , ivar, face_loc.iOct);
    }
    // clang-format on
  }

  return value / static_cast<real_t>(num_face_siblings);

} // StencilHelper<dim, device_t>::compute_face_siblings_average

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_FUNCTION real_t
StencilHelper<dim, device_t>::compute_external_face_siblings_average(
  CellLocation<dim> const &    cell_loc,
  int                          ivar,
  face_type_t                  ivar_face,
  FaceDataArrayBlock_t const & Bface) const
{
  // only used in debug mode
  [[maybe_unused]] const auto & b = m_block_sizes;

  auto const & coord = cell_loc.ijk;

  KOKKOS_ASSERT(cell_loc.is_eldest_sibling() && "Wrong cell location; must be a eldest sibling.");

  const auto offset_x = (ivar == IX and ivar_face == face_type_t::RIGHT) ? 2 : 0;
  const auto offset_y = (ivar == IY and ivar_face == face_type_t::RIGHT) ? 2 : 0;
  const auto offset_z = (ivar == IZ and ivar_face == face_type_t::RIGHT) ? 2 : 0;

  real_t value = ZERO_F;

  // number of face siblings :
  // - 2 in 2d
  // - 4 in 3d
  // that is 2^(dim-1)
  // except along IZ direction, it is always 4 (2d and 3d) !!
  const int num_face_siblings = (ivar == IZ) ? 4 : 1 << (dim - 1);

  if constexpr (dim == 2)
  {
    for (auto jj = 0; jj < 2; ++jj)
    {
      for (auto ii = 0; ii < 2; ++ii)
      {

        KOKKOS_ASSERT((coord[IX] + ii < b[IX]) && "Wrong values for coord[IX]");
        KOKKOS_ASSERT((coord[IY] + jj < b[IY]) && "Wrong values for coord[IY]");

        // compute corresponding cell index in current block of cells
        const auto coord_sibling = coord_t<2>{ coord[IX] + ii, coord[IY] + jj };

        // average over face siblings
        // if ivar is IZ, we take them all
        if ((ivar == IX and ii == 0) or (ivar == IY and jj == 0) or (ivar == IZ))
        {
          // clang-format off
          value +=
            Bface(coord_sibling[IX] + offset_x,
                  coord_sibling[IY] + offset_y,
                  ivar,
                  cell_loc.iOct);
          // clang-format on
        }
      } // for ii
    } // for jj
  }
  else if constexpr (dim == 3)
  {
    for (auto kk = 0; kk < 2; ++kk)
    {
      for (auto jj = 0; jj < 2; ++jj)
      {
        for (auto ii = 0; ii < 2; ++ii)
        {

          KOKKOS_ASSERT((coord[IX] + ii < b[IX]) && "Wrong values for coord[IX]");
          KOKKOS_ASSERT((coord[IY] + jj < b[IY]) && "Wrong values for coord[IY]");
          KOKKOS_ASSERT((coord[IZ] + kk < b[IZ]) && "Wrong values for coord[IZ]");

          // compute corresponding cell index in current block of cells
          const auto coord_sibling = coord_t<3>{ coord[IX] + ii, coord[IY] + jj, coord[IZ] + kk };

          if ((ivar == IX and ii == 0) or (ivar == IY and jj == 0) or (ivar == IZ and kk == 0))
          {
            // clang-format off
            value += Bface(coord_sibling[IX] + offset_x,
                           coord_sibling[IY] + offset_y,
                           coord_sibling[IZ] + offset_z,
                           ivar,
                           cell_loc.iOct);
            // clang-format on
          }
        } // for ii
      } // for jj
    } // for kk
  }

  return value / static_cast<real_t>(num_face_siblings);

} // StencilHelper<dim, device_t>::compute_external_face_siblings_average

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_FUNCTION real_t
StencilHelper<dim, device_t>::compute_face_siblings_average(
  FaceLocation<dim> const &    face_loc,
  FaceDataArrayBlock_t const & Bface) const
{
  // only used in debug mode
  [[maybe_unused]] const auto & b = m_block_sizes;

  auto const & ivar = face_loc.ijk[dim];

  auto const & ijk = face_loc.ijk;

  real_t value = 0;

  // number of face siblings :
  // - 2 in 2d
  // - 4 in 3d
  // that is 2^(dim-1)
  // except along IZ direction, it is always 4 (2d and 3d) !!
  const int num_face_siblings = (ivar == IZ) ? 4 : 1 << (dim - 1);

  if constexpr (dim == 2)
  {
    // clang-format off
    if (ivar == IX)
    {
      const auto i0 =  ijk[IX];
      const auto j0 = (ijk[IY] / 2) * 2;

      value += Bface(i0, j0    , ivar, face_loc.iOct);
      value += Bface(i0, j0 + 1, ivar, face_loc.iOct);
    }
    else if (ivar == IY)
    {
      const auto i0 = (ijk[IX] / 2) * 2;
      const auto j0 =  ijk[IY];

      value += Bface(i0    , j0, ivar, face_loc.iOct);
      value += Bface(i0 + 1, j0, ivar, face_loc.iOct);
    }
    else if (ivar == IZ)
    {
      const auto i0 = (ijk[IX] / 2) * 2;
      const auto j0 = (ijk[IY] / 2) * 2;

      value += Bface(i0    , j0    , ivar, face_loc.iOct);
      value += Bface(i0 + 1, j0    , ivar, face_loc.iOct);
      value += Bface(i0    , j0 + 1, ivar, face_loc.iOct);
      value += Bface(i0 + 1, j0 + 1, ivar, face_loc.iOct);
    }
    // clang-format on
  }
  else if constexpr (dim == 3)
  {
    // clang-format off
    if (ivar == IX)
    {
      const auto i0 =  ijk[IX];
      const auto j0 = (ijk[IY] / 2) * 2;
      const auto k0 = (ijk[IZ] / 2) * 2;
      value += Bface(i0    , j0    , k0    , ivar, face_loc.iOct);
      value += Bface(i0    , j0 + 1, k0    , ivar, face_loc.iOct);
      value += Bface(i0    , j0    , k0 + 1, ivar, face_loc.iOct);
      value += Bface(i0    , j0 + 1, k0 + 1, ivar, face_loc.iOct);
    }
    else if (ivar == IY)
    {
      const auto i0 = (ijk[IX] / 2) * 2;
      const auto j0 =  ijk[IY];
      const auto k0 = (ijk[IZ] / 2) * 2;
      value += Bface(i0    , j0    , k0    , ivar, face_loc.iOct);
      value += Bface(i0 + 1, j0    , k0    , ivar, face_loc.iOct);
      value += Bface(i0    , j0    , k0 + 1, ivar, face_loc.iOct);
      value += Bface(i0 + 1, j0    , k0 + 1, ivar, face_loc.iOct);
    }
    else if (ivar == IZ)
    {
      const auto i0 = (ijk[IX] / 2) * 2;
      const auto j0 = (ijk[IY] / 2) * 2;
      const auto k0 =  ijk[IZ];
      value += Bface(i0    , j0    , k0    , ivar, face_loc.iOct);
      value += Bface(i0 + 1, j0    , k0    , ivar, face_loc.iOct);
      value += Bface(i0    , j0 + 1, k0    , ivar, face_loc.iOct);
      value += Bface(i0 + 1, j0 + 1, k0    , ivar, face_loc.iOct);
    }
    // clang-format on
  }

  return value / static_cast<real_t>(num_face_siblings);

} // StencilHelper<dim, device_t>::compute_face_siblings_average

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_FUNCTION real_t
StencilHelper<dim, device_t>::compute_face_siblings_sum(CellLocation_t const &   cell_loc,
                                                        int                      ivar,
                                                        DataArrayBlock_t const & userdata,
                                                        int                      dir) const
{
  // this version is only for regular DataArrayBlock (not flux)

  if constexpr (dim == 2)
  {
    KOKKOS_ASSERT((dir == IX or dir == IY) && "Wrong direction");
  }
  else if constexpr (dim == 3)
  {
    KOKKOS_ASSERT((dir == IX or dir == IY or dir == IZ) && "Wrong direction");
  }

  auto const & coord = cell_loc.ijk;

  // round down coordinates to a multiple of two (to get eldest sibling)
  // in principle, this is useless, if cell_loc was obtained  from a coarse quadrant, then
  // shifted
  // coord are a eldest sibling (even coordinates)
  coord_t<dim> coord0;
  coord0[IX] = (coord[IX] / 2) * 2;
  coord0[IY] = (coord[IY] / 2) * 2;
  if constexpr (dim == 3)
    coord0[IZ] = (coord[IZ] / 2) * 2;

  // auto cell_index0 = coord_to_cellindex<dim>(coord0, b);

  real_t data = 0;

  auto flux_offset = init_kokkos_array<int, dim>(0);
  // if (is_flux)
  // {
  //   flux_offset[dir] = 1;
  // }

  const uint16_t nbCells = (1 << (dim - 1));
  for (uint16_t i = 0; i < nbCells; ++i)
  {
    // index ii will span all siblings index matching a face orthogonal to direction "dir"
    // in other words, in the following we swap bit "dir" the last dimension (IY in 2D, IZ in 3D)
    uint16_t ii = swapBits(i, dim - 1, dir);

    // if coord[dir] is odd  we are in a right face
    // if coord[dir] is even we are in a left  face
    if (coord[static_cast<size_t>(dir)] % 2 == 1)
      ii += static_cast<uint16_t>(1 << dir);

    if constexpr (dim == 2)
    {
      // data += userdata(cell_index0 + (ii & 0x1) + ((ii & 0x2) >> 1) * b[IX], ivar,
      // cell_loc.iOct);
      data += userdata(coord0[IX] + ((ii & 0x1) << 0) + flux_offset[IX],
                       coord0[IY] + ((ii & 0x2) >> 1) + flux_offset[IY],
                       ivar,
                       cell_loc.iOct);
    }
    else if constexpr (dim == 3)
    {
      // data += userdata(cell_index0 + (ii & 0x1) + ((ii & 0x2) >> 1) * b[IX] +
      //                    ((ii & 0x4) >> 2) * b[IX] * b[IY],
      //                  ivar,
      //                  cell_loc.iOct);
      data += userdata(coord0[IX] + ((ii & 0x1) >> 0) + flux_offset[IX],
                       coord0[IY] + ((ii & 0x2) >> 1) + flux_offset[IY],
                       coord0[IZ] + ((ii & 0x4) >> 2) + flux_offset[IZ],
                       ivar,
                       cell_loc.iOct);
    }
  }

  return data;

} // StencilHelper<dim, device_t>::compute_face_siblings_sum

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_FUNCTION real_t
StencilHelper<dim, device_t>::compute_face_siblings_sum(CellLocation_t const &   cell_loc,
                                                        int                      ivar,
                                                        DataArrayBlock_t const & userdata,
                                                        bool use_right_flux) const
{
  // determine flux direction
  int dir = -1;
  if (userdata.is_flux_array(m_block_sizes, IX))
  {
    dir = IX;
  }
  if (userdata.is_flux_array(m_block_sizes, IY))
  {
    dir = IY;
  }
  if constexpr (dim == 3)
  {
    if (userdata.is_flux_array(m_block_sizes, IZ))
    {
      dir = IZ;
    }
  }

  // this version is only DataArrayBlock that are also flux array in direction dir
  KOKKOS_ASSERT(dir >= 0 && "userdata is not a flux array");
  if (dir < 0)
  {
    // we could also abort; we should not be here
    return 0;
  }

  auto const & coord = cell_loc.ijk;

  // round down coordinates to a multiple of two (to get eldest sibling)
  // in principle, this is useless, if cell_loc was obtained  from a coarse quadrant, then
  // shifted
  // coord are a eldest sibling (even coordinates)
  coord_t<dim> coord0;
  coord0[IX] = (coord[IX] / 2) * 2;
  coord0[IY] = (coord[IY] / 2) * 2;
  if constexpr (dim == 3)
    coord0[IZ] = (coord[IZ] / 2) * 2;

  // when using right flux (we must add an extra offset to shift index on the right)
  auto flux_offset = init_kokkos_array<int, dim>(0);
  if (use_right_flux)
  {
    flux_offset[static_cast<size_t>(dir)] = 1;
  }

  // return value
  real_t data = 0;

  // perform sum
  const uint16_t nbCells = (1 << (dim - 1));
  for (uint16_t i = 0; i < nbCells; ++i)
  {
    // index ii will span all siblings index matching a face orthogonal to direction "dir"
    // in other words, in the following we swap bit "dir" the last dimension (IY in 2D, IZ in 3D)
    uint16_t ii = swapBits(i, dim - 1, dir);

    // if coord[dir] is odd  we are in a right face
    // if coord[dir] is even we are in a left  face
    if (coord[static_cast<size_t>(dir)] % 2 == 1)
      ii += static_cast<uint16_t>(1 << dir);

    if constexpr (dim == 2)
    {
      data += userdata(coord0[IX] + ((ii & 0x1) << 0) + flux_offset[IX],
                       coord0[IY] + ((ii & 0x2) >> 1) + flux_offset[IY],
                       ivar,
                       cell_loc.iOct);
    }
    else if constexpr (dim == 3)
    {
      data += userdata(coord0[IX] + ((ii & 0x1) >> 0) + flux_offset[IX],
                       coord0[IY] + ((ii & 0x2) >> 1) + flux_offset[IY],
                       coord0[IZ] + ((ii & 0x4) >> 2) + flux_offset[IZ],
                       ivar,
                       cell_loc.iOct);
    }
  }

  return data;

} // StencilHelper<dim, device_t>::compute_face_siblings_sum

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_FUNCTION real_t
StencilHelper<dim, device_t>::compute_minmod_slopes(CellLocation_t const &   cell_loc_cur,
                                                    CellLocation_t const &   cell_loc_right,
                                                    CellLocation_t const &   cell_loc_left,
                                                    int                      ivar,
                                                    DataArrayBlock_t const & userdata,
                                                    real_t                   slope_type) const
{
  KOKKOS_ASSERT((cell_loc_right.level() == cell_loc_cur.level() or
                 cell_loc_right.level() == cell_loc_cur.level() + 1) &&
                "neighbor must be finer or at same level");

  KOKKOS_ASSERT((cell_loc_left.level() == cell_loc_cur.level() or
                 cell_loc_left.level() == cell_loc_cur.level() + 1) &&
                "neighbor must be finer or at same level");

  auto const & bSize = userdata.block_size();

  const real_t data_cur = userdata(cell_loc_cur.cellindex(bSize), ivar, cell_loc_cur.iOct);

  const real_t data_right = cell_loc_right.level() == cell_loc_cur.level()
                              ? userdata(cell_loc_right.cellindex(bSize), ivar, cell_loc_right.iOct)
                              : compute_siblings_average(cell_loc_right, bSize, ivar, userdata);

  const real_t data_left = cell_loc_left.level() == cell_loc_cur.level()
                             ? userdata(cell_loc_left.cellindex(bSize), ivar, cell_loc_left.iOct)
                             : compute_siblings_average(cell_loc_left, bSize, ivar, userdata);

  return minmod_scalar(data_cur, data_right, data_left, slope_type);

} // StencilHelper<dim, device_t>::compute_minmod_slopes

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_FUNCTION real_t
StencilHelper<dim, device_t>::compute_minmod_slopes(CellLocation_t const &          cell_loc_cur,
                                                    CellLocation_t const &          cell_loc_right,
                                                    CellLocation_t const &          cell_loc_left,
                                                    int                             ivar,
                                                    DataArrayGhostedBlock_t const & userdata,
                                                    real_t slope_type) const
{
  KOKKOS_ASSERT((cell_loc_right.level() == cell_loc_cur.level() or
                 cell_loc_right.level() == cell_loc_cur.level() + 1) &&
                "neighbor must be finer or at same level");

  KOKKOS_ASSERT((cell_loc_left.level() == cell_loc_cur.level() or
                 cell_loc_left.level() == cell_loc_cur.level() + 1) &&
                "neighbor must be finer or at same level");

  const real_t data_cur = userdata(cell_loc_cur.ijk, ivar, cell_loc_cur.iOct);

  const real_t data_right = cell_loc_right.level() == cell_loc_cur.level()
                              ? userdata(cell_loc_right.ijk, ivar, cell_loc_right.iOct)
                              : compute_siblings_average(cell_loc_right, ivar, userdata);

  const real_t data_left = cell_loc_left.level() == cell_loc_cur.level()
                             ? userdata(cell_loc_left.ijk, ivar, cell_loc_left.iOct)
                             : compute_siblings_average(cell_loc_left, ivar, userdata);

  return minmod_scalar(data_cur, data_right, data_left, slope_type);

} // StencilHelper<dim, device_t>::compute_minmod_slopes

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_FUNCTION real_t
StencilHelper<dim, device_t>::compute_minmod_slopes(CellLocation_t const &           cell_loc_cur,
                                                    int                              ivar_cur,
                                                    CellLocation_t const &           cell_loc_right,
                                                    int                              ivar_right,
                                                    CellLocation_t const &           cell_loc_left,
                                                    int                              ivar_left,
                                                    DataArrayBlockMultiVar_t const & userdata,
                                                    real_t slope_type) const
{
  KOKKOS_ASSERT((cell_loc_right.level() == cell_loc_cur.level() or
                 cell_loc_right.level() == cell_loc_cur.level() + 1) &&
                "neighbor must be finer or at same level");

  KOKKOS_ASSERT((cell_loc_left.level() == cell_loc_cur.level() or
                 cell_loc_left.level() == cell_loc_cur.level() + 1) &&
                "neighbor must be finer or at same level");

  auto const & bSize = userdata.block_size();

  const real_t data_cur = userdata(cell_loc_cur.cellindex(bSize), ivar_cur, cell_loc_cur.iOct);

  const real_t data_right =
    cell_loc_right.level() == cell_loc_cur.level()
      ? userdata(cell_loc_right.cellindex(bSize), ivar_right, cell_loc_right.iOct)
      : compute_siblings_average(cell_loc_right, bSize, ivar_right, userdata);

  const real_t data_left =
    cell_loc_left.level() == cell_loc_cur.level()
      ? userdata(cell_loc_left.cellindex(bSize), ivar_left, cell_loc_left.iOct)
      : compute_siblings_average(cell_loc_left, bSize, ivar_left, userdata);

  return minmod_scalar(data_cur, data_right, data_left, slope_type);
}

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_FUNCTION real_t
StencilHelper<dim, device_t>::compute_minmod_slopes_prolongation(
  CellLocation_t const &   cell_loc_cur,
  CellLocation_t const &   cell_loc_right,
  CellLocation_t const &   cell_loc_left,
  int                      ivar,
  DataArrayBlock_t const & userdata,
  real_t                   slope_type) const
{

  auto const & bSize = userdata.block_size();

  const real_t data_cur = userdata(cell_loc_cur.cellindex(bSize), ivar, cell_loc_cur.iOct);

  const real_t data_right = cell_loc_right.level() <= cell_loc_cur.level()
                              ? userdata(cell_loc_right.cellindex(bSize), ivar, cell_loc_right.iOct)
                              : compute_siblings_average(cell_loc_right, bSize, ivar, userdata);

  const real_t data_left = cell_loc_left.level() <= cell_loc_cur.level()
                             ? userdata(cell_loc_left.cellindex(bSize), ivar, cell_loc_left.iOct)
                             : compute_siblings_average(cell_loc_left, bSize, ivar, userdata);

  return minmod_scalar(data_cur, data_right, data_left, slope_type);
}

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_FUNCTION real_t
StencilHelper<dim, device_t>::compute_minmod_slopes_prolongation(
  CellLocation_t const &           cell_loc_cur,
  int                              ivar_cur,
  CellLocation_t const &           cell_loc_right,
  int                              ivar_right,
  CellLocation_t const &           cell_loc_left,
  int                              ivar_left,
  DataArrayBlockMultiVar_t const & userdata,
  real_t                           slope_type) const
{

  auto const & bSize = userdata.block_size();

  const real_t data_cur = userdata(cell_loc_cur.cellindex(bSize), ivar_cur, cell_loc_cur.iOct);

  const real_t data_right =
    cell_loc_right.level() <= cell_loc_cur.level()
      ? userdata(cell_loc_right.cellindex(bSize), ivar_right, cell_loc_right.iOct)
      : compute_siblings_average(cell_loc_right, bSize, ivar_right, userdata);

  const real_t data_left =
    cell_loc_left.level() <= cell_loc_cur.level()
      ? userdata(cell_loc_left.cellindex(bSize), ivar_left, cell_loc_left.iOct)
      : compute_siblings_average(cell_loc_left, bSize, ivar_left, userdata);

  return minmod_scalar(data_cur, data_right, data_left, slope_type);
}

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_FUNCTION real_t
StencilHelper<dim, device_t>::compute_minmod_slopes_prolongation(
  FaceLocation_t const &       face_loc_cur,
  FaceLocation_t const &       face_loc_right,
  FaceLocation_t const &       face_loc_left,
  FaceDataArrayBlock_t const & facedata) const
{
  auto const & ivar = face_loc_cur.ijk[dim];

  if constexpr (dim == 2)
  {
    const real_t data_cur =
      facedata(face_loc_cur.ijk[IX], face_loc_cur.ijk[IY], ivar, face_loc_cur.iOct);

    const real_t data_right =
      face_loc_right.level() <= face_loc_cur.level()
        ? facedata(face_loc_right.ijk[IX], face_loc_right.ijk[IY], ivar, face_loc_right.iOct)
        : compute_face_siblings_average(face_loc_right, facedata);

    const real_t data_left =
      face_loc_left.level() <= face_loc_cur.level()
        ? facedata(face_loc_left.ijk[IX], face_loc_left.ijk[IY], ivar, face_loc_left.iOct)
        : compute_face_siblings_average(face_loc_left, facedata);

    return minmod_scalar(data_cur, data_right, data_left, 1.0);
  }
  else if constexpr (dim == 3)
  {
    const real_t data_cur = facedata(
      face_loc_cur.ijk[IX], face_loc_cur.ijk[IY], face_loc_cur.ijk[IZ], ivar, face_loc_cur.iOct);

    const real_t data_right = face_loc_right.level() <= face_loc_cur.level()
                                ? facedata(face_loc_right.ijk[IX],
                                           face_loc_right.ijk[IY],
                                           face_loc_right.ijk[IZ],
                                           ivar,
                                           face_loc_right.iOct)
                                : compute_face_siblings_average(face_loc_right, facedata);

    const real_t data_left = face_loc_left.level() <= face_loc_cur.level()
                               ? facedata(face_loc_left.ijk[IX],
                                          face_loc_left.ijk[IY],
                                          face_loc_left.ijk[IZ],
                                          ivar,
                                          face_loc_left.iOct)
                               : compute_face_siblings_average(face_loc_left, facedata);

    return minmod_scalar(data_cur, data_right, data_left, 1.0);
  }
} // compute_minmod_slopes_prolongation

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_FUNCTION real_t
StencilHelper<dim, device_t>::compute_linear_combination(CellLocation_t const &   cell_loc_left,
                                                         CellLocation_t const &   cell_loc_cur,
                                                         CellLocation_t const &   cell_loc_right,
                                                         uint8_t                  target_level,
                                                         int const                ivar,
                                                         DataArrayBlock_t const & userdata,
                                                         Kokkos::Array<real_t, 3> const coef) const
{
  KOKKOS_ASSERT(
    (cell_loc_left.level() == target_level or cell_loc_left.level() == target_level + 1) &&
    "neighbor must be finer or at same level");

  KOKKOS_ASSERT(
    (cell_loc_cur.level() == target_level or cell_loc_cur.level() == target_level + 1) &&
    "neighbor must be finer or at same level");

  KOKKOS_ASSERT(
    (cell_loc_right.level() == target_level or cell_loc_right.level() == target_level + 1) &&
    "neighbor must be finer or at same level");


  auto const & bSize = userdata.block_size();

  const real_t data_cur = cell_loc_cur.level() == target_level
                            ? userdata(cell_loc_cur.cellindex(bSize), ivar, cell_loc_cur.iOct)
                            : compute_siblings_average(cell_loc_cur, bSize, ivar, userdata);

  const real_t data_right = cell_loc_right.level() == target_level
                              ? userdata(cell_loc_right.cellindex(bSize), ivar, cell_loc_right.iOct)
                              : compute_siblings_average(cell_loc_right, bSize, ivar, userdata);

  const real_t data_left = cell_loc_left.level() == target_level
                             ? userdata(cell_loc_left.cellindex(bSize), ivar, cell_loc_left.iOct)
                             : compute_siblings_average(cell_loc_left, bSize, ivar, userdata);

  return coef[0] * data_left + coef[1] * data_cur + coef[2] * data_right;

} // StencilHelper<dim, device_t>::compute_linear_combination

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_FUNCTION real_t
StencilHelper<dim, device_t>::compute_linear_combination(CellLocation_t const &   cell_loc_left2,
                                                         CellLocation_t const &   cell_loc_left,
                                                         CellLocation_t const &   cell_loc_cur,
                                                         CellLocation_t const &   cell_loc_right,
                                                         CellLocation_t const &   cell_loc_right2,
                                                         uint8_t                  target_level,
                                                         int const                ivar,
                                                         DataArrayBlock_t const & userdata,
                                                         Kokkos::Array<real_t, 5> const coef) const
{
  KOKKOS_ASSERT(
    (cell_loc_left2.level() == target_level or cell_loc_left2.level() == target_level + 1) &&
    "neighbor must be finer or at same level");

  KOKKOS_ASSERT(
    (cell_loc_left.level() == target_level or cell_loc_left.level() == target_level + 1) &&
    "neighbor must be finer or at same level");

  KOKKOS_ASSERT(
    (cell_loc_cur.level() == target_level or cell_loc_cur.level() == target_level + 1) &&
    "neighbor must be finer or at same level");

  KOKKOS_ASSERT(
    (cell_loc_right.level() == target_level or cell_loc_right.level() == target_level + 1) &&
    "neighbor must be finer or at same level");

  KOKKOS_ASSERT(
    (cell_loc_right2.level() == target_level or cell_loc_right2.level() == target_level + 1) &&
    "neighbor must be finer or at same level");

  auto const & bSize = userdata.block_size();

  const real_t data_left2 = cell_loc_left2.level() == target_level
                              ? userdata(cell_loc_left2.cellindex(bSize), ivar, cell_loc_left2.iOct)
                              : compute_siblings_average(cell_loc_left2, bSize, ivar, userdata);

  const real_t data_left = cell_loc_left.level() == target_level
                             ? userdata(cell_loc_left.cellindex(bSize), ivar, cell_loc_left.iOct)
                             : compute_siblings_average(cell_loc_left, bSize, ivar, userdata);

  const real_t data_cur = cell_loc_cur.level() == target_level
                            ? userdata(cell_loc_cur.cellindex(bSize), ivar, cell_loc_cur.iOct)
                            : compute_siblings_average(cell_loc_cur, bSize, ivar, userdata);

  const real_t data_right = cell_loc_right.level() == target_level
                              ? userdata(cell_loc_right.cellindex(bSize), ivar, cell_loc_right.iOct)
                              : compute_siblings_average(cell_loc_right, bSize, ivar, userdata);

  const real_t data_right2 =
    cell_loc_right2.level() == target_level
      ? userdata(cell_loc_right2.cellindex(bSize), ivar, cell_loc_right2.iOct)
      : compute_siblings_average(cell_loc_right2, bSize, ivar, userdata);

  return coef[0] * data_left2 + coef[1] * data_left + coef[2] * data_cur + coef[3] * data_right +
         coef[4] * data_right2;

} // StencilHelper<dim, device_t>::compute_linear_combination

// explicit template instantiation
template class StencilHelper<2, kalypsso::DefaultDevice>;
template class StencilHelper<3, kalypsso::DefaultDevice>;

#ifdef KALYPSSO_CORE_INSTANTIATE_HOST_TEMPLATE
template class StencilHelper<2, kalypsso::HostDevice>;
template class StencilHelper<3, kalypsso::HostDevice>;
#endif // KALYPSSO_CORE_INSTANTIATE_HOST_TEMPLATE

} // namespace kalypsso
