// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file FillBlockGhosts_common.h
 */
#ifndef KALYPSSO_CORE_FILLBLOCKGHOSTS_COMMON_H_
#define KALYPSSO_CORE_FILLBLOCKGHOSTS_COMMON_H_

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/utils/log/kalypsso_log.h>
#include <kalypsso/core/kokkos_shared.h>

#include <kalypsso/core/kalypsso_data_container.h> // for DataArray, DataArrayHost, DataArrayGhostedBlock<dim, device_t>
#include <kalypsso/core/mesh_utils.h>
#include <kalypsso/core/orchard_key.h>
#include <kalypsso/core/orchard_key_utils.h>
#include <kalypsso/core/Kokkos_extensions.h>

#include <kalypsso/utils/mpi/ParallelEnv.h>

#include <kalypsso/core/utils_block.h> // for definition of function cellindex_to_coord and coord_to_cellindex

namespace kalypsso
{

// =======================================================
// =======================================================
/**
 * return true if a face is a face on the left
 */
template <size_t dim>
KOKKOS_INLINE_FUNCTION bool
is_left_face(const Face::face_t & face)
{
  if constexpr (dim == 2)
    return (face == Face::XMIN) or (face == Face::YMIN);
  if constexpr (dim == 3)
    return (face == Face::XMIN) or (face == Face::YMIN) or (face == Face::ZMIN);
}

// =======================================================
// =======================================================
/**
 * return true if a face is a face on the right
 */
template <size_t dim>
KOKKOS_INLINE_FUNCTION bool
is_right_face(const Face::face_t & face)
{
  if constexpr (dim == 2)
    return (face == Face::XMAX) or (face == Face::YMAX);
  if constexpr (dim == 3)
    return (face == Face::XMAX) or (face == Face::YMAX) or (face == Face::ZMAX);
}

/**
 * input  : 2 faces identifying a corner
 * output : cornerId
 *
 * Corners are enumerated in Morton order
 *
 * 2 --- 3
 * |     |
 * |     |
 * 0 --- 1
 */
KOKKOS_INLINE_FUNCTION uint8_t
compute_corner_id(const Face::face_t & face_x, const Face::face_t & face_y)
{
  return (face_x & 0x1) + ((face_y & 0x1) << 1);
}

KOKKOS_INLINE_FUNCTION uint8_t
compute_corner_id(const Face::face_t & face_x,
                  const Face::face_t & face_y,
                  const Face::face_t & face_z)
{
  return static_cast<uint8_t>((face_x & 0x1) + ((face_y & 0x1) << 1) + ((face_z & 0x1) << 2));
}

template <size_t dim>
KOKKOS_INLINE_FUNCTION uint8_t
compute_corner_id(Kokkos::Array<Face::face_t, dim> const & faces)
{
  if constexpr (dim == 2)
  {
    return static_cast<uint8_t>((faces[IX] & 0x1) + ((faces[IY] & 0x1) << 1));
  }
  else if constexpr (dim == 3)
  {
    return static_cast<uint8_t>((faces[IX] & 0x1) + ((faces[IY] & 0x1) << 1) +
                                ((faces[IZ] & 0x1) << 2));
  }
}

// ===========================================================
// ===========================================================
template <Dir::dir_t dir>
KOKKOS_INLINE_FUNCTION bool
is_edge_along(Edge::edge_t edge)
{
  if constexpr (dir == Dir::Z)
    if (edge < 4)
      return true;

  if constexpr (dir == Dir::X)
    if (edge < 8 and edge >= 4)
      return true;

  if constexpr (dir == Dir::Y)
    if (edge < 12 and edge >= 8)
      return true;

  return false;
} // is_edge_along

// ==============================================================
// ==============================================================
KOKKOS_INLINE_FUNCTION Edge::edge_t
                       compute_edge_id(const Face::face_t & face0, const Face::face_t & face1)
{
  KOKKOS_ASSERT(face0 / 2 < face1 / 2 &&
                "WRONG VALUE, faces must be orthogonal and Morton ordered.");

  Edge::edge_t res = (face0 & 0x1) + ((face1 & 0x1) << 1);

  if (((face1 / 2) == Dir::Z) and ((face0 / 2) == Dir::Y))
    res += 4;

  if (((face1 / 2) == Dir::Z) and ((face0 / 2) == Dir::X))
    res += 8;

  return res;
}

// ==============================================================
// ==============================================================
/**
 * Shift (geometric translation) the coordinates of cell (inside a block), moving from right to left
 * face.
 *
 * \param[in,out] coord coordinates of a cell in a face, to be shifted to the left most face
 * \param[in] block sizes
 * \param[in] ghost sizes
 * \param[in] face id
 *
 */
template <size_t dim>
KOKKOS_INLINE_FUNCTION void
shift_coord_face(coord_t<dim> &            coord,
                 block_size_t<dim> const & b,
                 block_size_t<dim> const & g,
                 Face::face_t              face)
{

  const Dir::dir_t dir = face / 2;

  if constexpr (dim == 2)
  {
    if (dir == Dir::X)
    {
      if (is_right_face<dim>(face))
      {
        coord[IX] -= (b[IX] + g[IX]);
      }
      coord[IY] -= g[IY];
    }

    if (dir == Dir::Y)
    {
      if (is_right_face<dim>(face))
      {
        coord[IY] -= (b[IY] + g[IY]);
      }
      coord[IX] -= g[IX];
    }
  } // dim == 2

  else if constexpr (dim == 3)
  {
    if (dir == Dir::X)
    {
      if (is_right_face<dim>(face))
      {
        coord[IX] -= (b[IX] + g[IX]);
      }
      coord[IY] -= g[IY];
      coord[IZ] -= g[IZ];
    }

    if (dir == Dir::Y)
    {
      coord[IX] -= g[IX];
      if (is_right_face<dim>(face))
      {
        coord[IY] -= (b[IY] + g[IY]);
      }
      coord[IZ] -= g[IZ];
    }

    if (dir == Dir::Z)
    {
      coord[IX] -= g[IX];
      coord[IY] -= g[IY];
      if (is_right_face<dim>(face))
      {
        coord[IZ] -= (b[IZ] + g[IZ]);
      }
    }
  } // dim == 3

} // shift_coord_face

// ==============================================================
// ==============================================================
/**
 * Compute neighbor orchard key of neighbor octant across a given face, at same level.
 *
 * when neighbor is at finer level (i.e. larger level), there multiple neighbors, so we only
 * return the smallest key (the other neighbor keys can be deduced).
 *
 * \param[in] key is current octant key
 * \param[in] iface identifies a face (XMIN, XMAX, ...)
 */
template <size_t dim>
KOKKOS_INLINE_FUNCTION key_t
compute_face_neighbor_key(key_t key, Face::face_t iface, brick_size_t<dim> const & brick_sizes)
{

  // clang-format off
    KOKKOS_ASSERT(((iface == Face::XMIN) or
                   (iface == Face::XMAX) or
                   (iface == Face::YMIN) or
                   (iface == Face::YMAX) or
                   (iface == Face::ZMIN) or
                   (iface == Face::ZMAX)) &&
                  "Wrong Face Id");
  // clang-format on

  return orchard_key_t<dim>::get_face_neighbor_key(key, iface, brick_sizes);

} // compute_face_neighbor_key

// ==============================================================
// ==============================================================
/**
 * Compute neighbor orchard keys of all neighbor octants across a given face, at finer level.
 *
 * when neighbor is at finer level (i.e. larger level), there multiple neighbors, so we only
 * return the smallest key that touches the faces (the other neighbor keys can be deduced).
 *
 * \param[in] key is neigh octant key (at same level as current octant)
 * \param[in] iface identifies a face (XMIN, XMAX, ...)
 *
 * implementation note: in input we give the neighbor key at same level as current octant, because
 * we assume, we already check if neighbor at same level exists, so this key has already been
 * computed. The fine neighbors are children of the neighbor same level.
 *
 * In the drawing, we want to access the left neighbor at finer level.
 *
 *    left neighbors       current octant
 *    _______________     ______________
 *   |       |       |   |              |
 *   |       |   x   |   |              |
 *   |       |       |   |              |
 *   |_______|_______|   |              |
 *   |       |       |   |              |
 *   |       |   x   |   |              |
 *   |       |       |   |              |
 *   |_______|_______|   |______________|
 *
 *
 *
 * \return fine neighbor key
 */
template <size_t dim>
KOKKOS_INLINE_FUNCTION Kokkos::Array<key_t, orchard_key_t<dim>::NB_FACE_NEIGHBORS_FINE>
compute_face_neighbor_key_finer(key_t key_neigh_same_level, Face::face_t iface)
{

  // clang-format off
    KOKKOS_ASSERT(((iface == Face::XMIN) or
                   (iface == Face::XMAX) or
                   (iface == Face::YMIN) or
                   (iface == Face::YMAX) or
                   (iface == Face::ZMIN) or
                   (iface == Face::ZMAX)) &&
                  "Wrong Face Id");
  // clang-format on

  if constexpr (dim == 2)
  {
    if (iface == Face::XMIN)
    {
      return { orchard_key_t<dim>::child(key_neigh_same_level, 1),
               orchard_key_t<dim>::child(key_neigh_same_level, 3) };
    }
    else if (iface == Face::XMAX)
    {
      return { orchard_key_t<dim>::child(key_neigh_same_level, 0),
               orchard_key_t<dim>::child(key_neigh_same_level, 2) };
    }
    else if (iface == Face::YMIN)
    {
      return { orchard_key_t<dim>::child(key_neigh_same_level, 2),
               orchard_key_t<dim>::child(key_neigh_same_level, 3) };
    }
    else if (iface == Face::YMAX)
    {
      return { orchard_key_t<dim>::child(key_neigh_same_level, 0),
               orchard_key_t<dim>::child(key_neigh_same_level, 1) };
    }
    else
    {
      // we shouldn't be here, return invalid values
      return { key_neigh_same_level, key_neigh_same_level };
    }
  }
  else if constexpr (dim == 3)
  {
    if (iface == Face::XMIN)
    {
      return { orchard_key_t<dim>::child(key_neigh_same_level, 1),
               orchard_key_t<dim>::child(key_neigh_same_level, 3),
               orchard_key_t<dim>::child(key_neigh_same_level, 5),
               orchard_key_t<dim>::child(key_neigh_same_level, 7) };
    }
    else if (iface == Face::XMAX)
    {
      return { orchard_key_t<dim>::child(key_neigh_same_level, 0),
               orchard_key_t<dim>::child(key_neigh_same_level, 2),
               orchard_key_t<dim>::child(key_neigh_same_level, 4),
               orchard_key_t<dim>::child(key_neigh_same_level, 6) };
    }
    else if (iface == Face::YMIN)
    {
      return { orchard_key_t<dim>::child(key_neigh_same_level, 2),
               orchard_key_t<dim>::child(key_neigh_same_level, 3),
               orchard_key_t<dim>::child(key_neigh_same_level, 6),
               orchard_key_t<dim>::child(key_neigh_same_level, 7) };
    }
    else if (iface == Face::YMAX)
    {
      return { orchard_key_t<dim>::child(key_neigh_same_level, 0),
               orchard_key_t<dim>::child(key_neigh_same_level, 1),
               orchard_key_t<dim>::child(key_neigh_same_level, 4),
               orchard_key_t<dim>::child(key_neigh_same_level, 5) };
    }
    else if (iface == Face::ZMIN)
    {
      return { orchard_key_t<dim>::child(key_neigh_same_level, 4),
               orchard_key_t<dim>::child(key_neigh_same_level, 5),
               orchard_key_t<dim>::child(key_neigh_same_level, 6),
               orchard_key_t<dim>::child(key_neigh_same_level, 7) };
    }
    else if (iface == Face::ZMAX)
    {
      return { orchard_key_t<dim>::child(key_neigh_same_level, 0),
               orchard_key_t<dim>::child(key_neigh_same_level, 1),
               orchard_key_t<dim>::child(key_neigh_same_level, 2),
               orchard_key_t<dim>::child(key_neigh_same_level, 3) };
    }
    else
    {
      // we shouldn't be here, return invalid values
      return {
        key_neigh_same_level, key_neigh_same_level, key_neigh_same_level, key_neigh_same_level
      };
    }
  }

} // compute_face_neighbor_key_finer

// ==============================================================
// ==============================================================
/**
 * Compute neighbor orchard key of neighbor octant across a given corner, at same level - 2d.
 *
 * need 2 orthogonal faces to identify a corner.
 */
KOKKOS_INLINE_FUNCTION key_t
compute_corner_neighbor_key(key_t                   key,
                            Face::face_t            iface_x,
                            Face::face_t            iface_y,
                            brick_size_t<2> const & brick_sizes)
{

  // clang-format off
    KOKKOS_ASSERT( ( ( (iface_x == Face::XMIN) or (iface_x == Face::XMAX) ) and
                     ( (iface_y == Face::YMIN) or (iface_y == Face::YMAX) ) ) && "Wrong corner ids");
  // clang-format on

  return orchard_key_t<2>::get_corner_neighbor_key(key, iface_x, iface_y, brick_sizes);

} // compute_corner_neighbor_key

template <size_t dim>
KOKKOS_INLINE_FUNCTION key_t
compute_corner_neighbor_key(key_t                            key,
                            Kokkos::Array<Face::face_t, dim> faces,
                            brick_size_t<dim> const &        brick_sizes)
{

  // clang-format off
  if constexpr (dim==2) {
    KOKKOS_ASSERT( ( ( (faces[IX] == Face::XMIN) or (faces[IX] == Face::XMAX) ) and
                     ( (faces[IY] == Face::YMIN) or (faces[IY] == Face::YMAX) ) ) && "Wrong corner ids");
  }
  if constexpr (dim==3) {
    KOKKOS_ASSERT( ( ( (faces[IX] == Face::XMIN) or (faces[IX] == Face::XMAX) ) and
                     ( (faces[IY] == Face::YMIN) or (faces[IY] == Face::YMAX) ) and
                     ( (faces[IZ] == Face::ZMIN) or (faces[IZ] == Face::ZMAX) ) ) && "Wrong corner ids");
  }
  // clang-format on

  if constexpr (dim == 2)
    return orchard_key_t<2>::get_corner_neighbor_key(key, faces[IX], faces[IY], brick_sizes);

  if constexpr (dim == 3)
    return orchard_key_t<3>::get_corner_neighbor_key(
      key, faces[IX], faces[IY], faces[IZ], brick_sizes);

} // compute_corner_neighbor_key

// ==============================================================
// ==============================================================
/**
 * Compute neighbor orchard key of neighbor octant across a given corner, at same level - 3d.
 *
 * need 3 orthogonal faces to identify a corner.
 */
KOKKOS_INLINE_FUNCTION key_t
compute_corner_neighbor_key(key_t                   key,
                            Face::face_t            iface_x,
                            Face::face_t            iface_y,
                            Face::face_t            iface_z,
                            brick_size_t<3> const & brick_sizes)
{

  // clang-format off
    KOKKOS_ASSERT( ( ( (iface_x == Face::XMIN) or (iface_x == Face::XMAX) ) and
                     ( (iface_y == Face::YMIN) or (iface_y == Face::YMAX) ) and
                     ( (iface_z == Face::ZMIN) or (iface_z == Face::ZMAX) ) ) && "Wrong corner ids");
  // clang-format on

  return orchard_key_t<3>::get_corner_neighbor_key(key, iface_x, iface_y, iface_z, brick_sizes);

} // compute_corner_neighbor_key

// ==============================================================
// ==============================================================
/**
 * Compute neighbor orchard key of neighbor octant across a given corner, at finer level - 2d.
 *
 * Implementation note: key_neigh_same_level is the orchard key of the father octant of the octant
 * we are looking for, we just need to identify the child id.
 *
 * \param[in] key_neigh_same_level is the orchard key of the neighbor octant at same level
 * \param[in] iface_x is a face id
 * \param[in] iface_y is a face id
 *
 * Remember that 2 orthogonal faces are needed to specify a corner.
 */
KOKKOS_INLINE_FUNCTION
key_t
compute_corner_neighbor_finer_key(key_t        key_neigh_same_level,
                                  Face::face_t iface_x,
                                  Face::face_t iface_y)
{

  // clang-format off
  KOKKOS_ASSERT( ( ( (iface_x == Face::XMIN) or (iface_x == Face::XMAX) ) and
                   ( (iface_y == Face::YMIN) or (iface_y == Face::YMAX) ) ) && "Wrong corner ids");
  // clang-format on

  const auto corner = compute_corner_id(iface_x, iface_y);

  // if (corner == 0)
  //   return orchard_key_t<dim>::child(key_neigh_same_level, 3);
  // if (corner == 1)
  //   return orchard_key_t<dim>::child(key_neigh_same_level, 2);
  // if (corner == 2)
  //   return orchard_key_t<dim>::child(key_neigh_same_level, 1);
  // if (corner == 3)
  //   return orchard_key_t<dim>::child(key_neigh_same_level, 0);

  return orchard_key_t<2>::child(key_neigh_same_level, 3 - corner);

} // compute_corner_neighbor_finer_key

// ==============================================================
// ==============================================================
/**
 * Compute neighbor orchard key of neighbor octant across a given corner, at finer level - 2d.
 */
KOKKOS_INLINE_FUNCTION
key_t
compute_corner_neighbor_finer_key(key_t        key_neigh_same_level,
                                  Face::face_t iface_x,
                                  Face::face_t iface_y,
                                  Face::face_t iface_z)
{

  // clang-format off
  KOKKOS_ASSERT( ( ( (iface_x == Face::XMIN) or (iface_x == Face::XMAX) ) and
                   ( (iface_y == Face::YMIN) or (iface_y == Face::YMAX) ) and
                   ( (iface_z == Face::ZMIN) or (iface_z == Face::ZMAX) ) ) && "Wrong corner ids");
  // clang-format on

  const auto corner = compute_corner_id(iface_x, iface_y, iface_z);

  return orchard_key_t<3>::child(key_neigh_same_level, 7 - corner);

} // compute_corner_neighbor_finer_key

// ==============================================================
// ==============================================================
/**
 * Compute neighbor orchard key of neighbor octant across a given corner, at finer level - 2d.
 */
template <size_t dim>
KOKKOS_INLINE_FUNCTION key_t
compute_corner_neighbor_finer_key(key_t                                    key_neigh_same_level,
                                  Kokkos::Array<Face::face_t, dim> const & faces)
{

  // clang-format off
  if constexpr (dim == 2)
  {
    KOKKOS_ASSERT((((faces[IX] == Face::XMIN) or (faces[IX] == Face::XMAX)) and
                   ((faces[IY] == Face::YMIN) or (faces[IY] == Face::YMAX))) &&
                  "Wrong corner ids");
  }
  if constexpr (dim == 3)
  {
    KOKKOS_ASSERT((((faces[IX] == Face::XMIN) or (faces[IX] == Face::XMAX)) and
                   ((faces[IY] == Face::YMIN) or (faces[IY] == Face::YMAX)) and
                   ((faces[IZ] == Face::ZMIN) or (faces[IZ] == Face::ZMAX))) &&
                  "Wrong corner ids");
  }
  // clang-format on

  const auto corner = compute_corner_id<dim>(faces);

  return orchard_key_t<dim>::child(key_neigh_same_level, (1 << dim) - 1 - corner);

} // compute_corner_neighbor_finer_key

// ==============================================================
// ==============================================================
/**
 * Compute neighbor orchard key of neighbor octant across a given edge, at same level
 */
KOKKOS_INLINE_FUNCTION key_t
compute_edge_neighbor_key(key_t key, Edge::edge_t edge, brick_size_t<3> const & brick_sizes)
{
  Face::face_t face0, face1;
  edge_to_faces(edge, face0, face1);

  // edge along Z
  if (edge < 4)
  {
    // face0 is X
    // face1 is Y
    return orchard_key_t<3>::get_corner_neighbor_key(key, face0, face1, face0, brick_sizes);
  }
  // edge along X
  else if (edge >= 4 and edge < 8)
  {
    // face0 is Y
    // face1 is Z
    return orchard_key_t<3>::get_corner_neighbor_key(key, face0, face0, face1, brick_sizes);
  }
  // edge along Y
  else if (edge >= 8)
  {
    // face0 is X
    // face1 is Z
    return orchard_key_t<3>::get_corner_neighbor_key(key, face0, face0, face1, brick_sizes);
  }

  // return self as invalid value
  return key;

} // compute_edge_neighbor_key

// ==============================================================
// ==============================================================
/**
 * Compute neighbor orchard key of neighbor octant across a given edge, at same level.
 *
 * We will use the neighbor at same level, and look for a child id.
 */
KOKKOS_INLINE_FUNCTION Kokkos::Array<key_t, orchard_key_t<3>::NB_EDGE_NEIGHBORS_FINE>
compute_edge_neighbor_finer_key(key_t key_neigh_same_level, Edge::edge_t edge)
{
  Face::face_t face0, face1;
  edge_to_faces(edge, face0, face1);

  // edge along Z
  if (edge < 4)
  {
    // face0 is X
    // face1 is Y
    Edge::edge_t edge2 = 3 - edge;
    return { orchard_key_t<3>::child(key_neigh_same_level, edge2),
             orchard_key_t<3>::child(key_neigh_same_level, edge2 + 4) };
  }
  // edge along X
  else if (edge < 8)
  {
    // face0 is Y
    // face1 is Z
    Edge::edge_t edge2 = static_cast<Edge::edge_t>((3 - (edge - 4)) << 1);
    return { orchard_key_t<3>::child(key_neigh_same_level, edge2),
             orchard_key_t<3>::child(key_neigh_same_level, edge2 + 1) };
  }
  // edge along Y
  else
  {
    // face0 is Z
    // face1 is X
    Edge::edge_t tmp = (3 - (edge - 8));
    Edge::edge_t edge2 = static_cast<Edge::edge_t>(((tmp >> 1) << 2) + (tmp & 0x1));
    return { orchard_key_t<3>::child(key_neigh_same_level, edge2),
             orchard_key_t<3>::child(key_neigh_same_level, edge2 + 2) };
  }

} // compute_edge_neighbor_finer_key

} // namespace kalypsso

#endif // KALYPSSO_CORE_FILLBLOCKGHOSTS_COMMON_H_
