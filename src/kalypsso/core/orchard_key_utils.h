// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file orchard_key_utils.h
 *
 */
#ifndef KALYPSSO_CORE_ORCHARD_KEY_UTILS_H
#define KALYPSSO_CORE_ORCHARD_KEY_UTILS_H

#include <kalypsso/core/kalypsso_core_base.h> // for assertm
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/orchard_key.h>
#include <kalypsso/core/utils_block.h> // for block_size_t, face_multiindex_t
#include <kalypsso/core/brick_utils.h>

#include <type_traits>
#include <climits> // for CHAR_BIT

namespace kalypsso
{

// =============================================================================
// =============================================================================
/**
 * convert an orchard key to vertex absolutes coordinates (same units as in the p4est
 * connectivity) of the lower left corner of quadrant, assuming a brick connectivity.
 *
 * implementation: extract logical x,y,z and rescale them to unit cube [0,1]^dim; then used tree
 * coordinates, to have the final vertex coordinates.
 *
 * remember that logical values of x,y,z are integers in range [0, ROOT_LENGTH-1], where ROOT_LENGTH
 * is \f$2^{NUM_LEVELS - 1}\f$.
 *
 * \param[in] key is an orchard key identifying in a unique way a quadrant (over all MPI process)
 * \param[in] brick_sizes provides the sizes of the brick (p4est) connectivity
 * \param[in] centering if true, returns coordinates of the center of the quadrant
 *
 * \return array of absolute vertex coordinates of the lower left corner of quadrant/octant assuming
 * brick p4est connectivity.
 */
template <size_t dim>
KOKKOS_INLINE_FUNCTION Kokkos::Array<real_t, dim>
                       orchard_key_to_vertex_coord(uint64_t key, bool centering)
{

  Kokkos::Array<real_t, dim> XYZ;

  auto center_offset = centering ? orchard_key_t<dim>::octantLength(key) / 2 : 0;

  real_t x =
    static_cast<real_t>(orchard_key_t<dim>::template get_octant_coord<IX>(key) + center_offset) /
    orchard_key_t<dim>::ROOT_LENGTH;
  real_t y =
    static_cast<real_t>(orchard_key_t<dim>::template get_octant_coord<IY>(key) + center_offset) /
    orchard_key_t<dim>::ROOT_LENGTH;

  XYZ[IX] = x + static_cast<real_t>(orchard_key_t<dim>::template get_tree_coord<IX>(key));
  XYZ[IY] = y + static_cast<real_t>(orchard_key_t<dim>::template get_tree_coord<IY>(key));

  if constexpr (dim == 3)
  {
    real_t z =
      static_cast<real_t>(orchard_key_t<dim>::template get_octant_coord<IZ>(key) + center_offset) /
      orchard_key_t<dim>::ROOT_LENGTH;
    XYZ[IZ] = z + static_cast<real_t>(orchard_key_t<dim>::template get_tree_coord<IZ>(key));
  }

  return XYZ;

} // orchard_key_to_vertex_coord

// =====================================================================
// =====================================================================
//! Same as orchard_key_to_vertex_coord but for orchard key corresponding to
//! an outside quadrant.
//!
template <size_t dim>
KOKKOS_INLINE_FUNCTION Kokkos::Array<real_t, dim>
outside_key_to_vertex_coord(uint64_t key, bool centering, brick_size_t<dim> brick_sizes)
{

  // first check that input key is actually an "outside" key
  KOKKOS_ASSERT(orchard_key_t<dim>::is_outside(key) && "key is not an outside key !");

  KOKKOS_ASSERT(orchard_key_t<dim>::is_at_any_domain_border(key, brick_sizes) &&
                "orchard key is not associated to a quadrant touching external border.")

  // get coordinate of the "inside" quadrant
  auto coord = orchard_key_to_vertex_coord<dim>(key, centering);

  // when computing outside quadrant key, we always use a "virtual" key computed as the
  // periodic image of the outside quadrant; so here we need is_periodic to be array of
  // "true"
  constexpr auto is_periodic = get_bool_array<dim>(true);

  // quadrant is at domain border, compute outside normal
  const auto outside_normal = orchard_key_t<dim>::get_outside_normal(key, brick_sizes, is_periodic);

  if (orchard_key_t<dim>::is_touching_face_X(key))
    coord[IX] -= static_cast<real_t>(brick_sizes[IX] * outside_normal[IX]);
  if (orchard_key_t<dim>::is_touching_face_Y(key))
    coord[IY] -= static_cast<real_t>(brick_sizes[IY] * outside_normal[IY]);
  if constexpr (dim == 3)
  {
    if (orchard_key_t<dim>::is_touching_face_Z(key))
      coord[IZ] -= static_cast<real_t>(brick_sizes[IZ] * outside_normal[IZ]);
  }

  return coord;

} // outside_key_to_vertex_coord

// =============================================================================
// =============================================================================
template <size_t dim>
KOKKOS_INLINE_FUNCTION Kokkos::Array<real_t, dim>
orchard_key_to_cell_coord(uint64_t key, Kokkos::Array<int32_t, dim> cell_indexes, int block_size)
{

  // 1. compute lower left corner coordinates in real space
  constexpr bool centering = false;
  auto           XYZ_corner = orchard_key_to_vertex_coord<dim>(key, centering);

  // 2. compute cell length in real space
  real_t dx_cell = static_cast<real_t>(orchard_key_t<dim>::octantLength(key)) /
                   static_cast<real_t>(orchard_key_t<dim>::ROOT_LENGTH) /
                   static_cast<real_t>(block_size);

  // 3. compute cell center coordinates
  Kokkos::Array<real_t, dim> res;

  res[IX] = XYZ_corner[IX] + static_cast<real_t>(cell_indexes[IX]) * dx_cell + HALF_F * dx_cell;
  res[IY] = XYZ_corner[IY] + static_cast<real_t>(cell_indexes[IY]) * dx_cell + HALF_F * dx_cell;
  if constexpr (dim == 3)
  {
    res[IZ] = XYZ_corner[IZ] + static_cast<real_t>(cell_indexes[IZ]) * dx_cell + HALF_F * dx_cell;
  }

  return res;

} // orchard_key_to_cell_coord

// =============================================================================
// =============================================================================
/**
 * Get coordinates in vertex (p4est connectivity) space of the corner of a cell.
 *
 * Corners are identifid in Z-order.
 */
template <size_t dim>
KOKKOS_INLINE_FUNCTION Kokkos::Array<real_t, dim>
                       orchard_key_to_corner_coord(uint64_t                    key,
                                                   Kokkos::Array<int32_t, dim> cell_indexes,
                                                   int                         block_size,
                                                   uint8_t                     i_corner)
{

  // 1. compute lower left corner coordinates in real space
  constexpr bool centering = false;
  auto           XYZ_corner = orchard_key_to_vertex_coord<dim>(key, centering);

  // 2. compute cell length in real space
  real_t dx_cell = static_cast<real_t>(orchard_key_t<dim>::octantLength(key)) /
                   static_cast<real_t>(orchard_key_t<dim>::ROOT_LENGTH) /
                   static_cast<real_t>(block_size);

  // 3. compute corner coordinates
  Kokkos::Array<real_t, dim> res;

  res[IX] = XYZ_corner[IX] + static_cast<real_t>(cell_indexes[IX]) * dx_cell +
            ((i_corner >> 0) & 1) * dx_cell;
  res[IY] = XYZ_corner[IY] + static_cast<real_t>(cell_indexes[IY]) * dx_cell +
            ((i_corner >> 1) & 1) * dx_cell;
  if constexpr (dim == 3)
  {
    res[IZ] = XYZ_corner[IZ] + static_cast<real_t>(cell_indexes[IZ]) * dx_cell +
              ((i_corner >> 2) & 1) * dx_cell;
  }

  return res;

} // orchard_key_to_cell_coord

// =============================================================================
// =============================================================================
/**
 * This utility is mostly useful when dealing with a FaceDataArrayBlock.
 */
template <size_t dim>
KOKKOS_INLINE_FUNCTION Kokkos::Array<real_t, dim>
orchard_key_to_face_coord(uint64_t key, face_multiindex_t<dim> face_indexes, int block_size)
{

  // 1. compute lower left corner coordinates in real space
  constexpr bool centering = false;
  auto           XYZ_corner = orchard_key_to_vertex_coord<dim>(key, centering);

  // 2. compute cell length in real space
  real_t dx_cell = static_cast<real_t>(orchard_key_t<dim>::octantLength(key)) /
                   static_cast<real_t>(orchard_key_t<dim>::ROOT_LENGTH) /
                   static_cast<real_t>(block_size);

  // 3. compute cell center coordinates
  Kokkos::Array<real_t, dim> res;

  const auto & dir = face_indexes[dim];

  res[IX] = XYZ_corner[IX] + static_cast<real_t>(face_indexes[IX]) * dx_cell;
  res[IX] += (dir == IX) ? 0 : HALF_F * dx_cell;

  res[IY] = XYZ_corner[IY] + static_cast<real_t>(face_indexes[IY]) * dx_cell;
  res[IY] += (dir == IY) ? 0 : HALF_F * dx_cell;

  if constexpr (dim == 3)
  {
    res[IZ] = XYZ_corner[IZ] + static_cast<real_t>(face_indexes[IZ]) * dx_cell;
    res[IZ] += (dir == IZ) ? 0 : HALF_F * dx_cell;
  }

  return res;

} // orchard_key_to_face_coord

// =============================================================================
// =============================================================================
/**
 * This utility is mostly useful when dealing with an EdgeDataArrayBlock.
 */
template <size_t dim>
KOKKOS_INLINE_FUNCTION Kokkos::Array<real_t, dim>
orchard_key_to_edge_coord(uint64_t key, edge_multiindex_t<dim> edge_indexes, int block_size)
{

  // 1. compute lower left corner coordinates in real space
  constexpr bool centering = false;
  auto           XYZ_corner = orchard_key_to_vertex_coord<dim>(key, centering);

  // 2. compute cell length in real space
  real_t dx_cell = static_cast<real_t>(orchard_key_t<dim>::octantLength(key)) /
                   static_cast<real_t>(orchard_key_t<dim>::ROOT_LENGTH) /
                   static_cast<real_t>(block_size);

  // 3. compute cell center coordinates
  Kokkos::Array<real_t, dim> res;

  const auto & dir = edge_indexes[dim];

  res[IX] = XYZ_corner[IX] + static_cast<real_t>(edge_indexes[IX]) * dx_cell;
  res[IX] += (dir == IX) ? HALF_F * dx_cell : 0;

  res[IY] = XYZ_corner[IY] + static_cast<real_t>(edge_indexes[IY]) * dx_cell;
  res[IY] += (dir == IY) ? HALF_F * dx_cell : 0;

  if constexpr (dim == 3)
  {
    res[IZ] = XYZ_corner[IZ] + static_cast<real_t>(edge_indexes[IZ]) * dx_cell;
    res[IZ] += (dir == IZ) ? HALF_F * dx_cell : 0;
  }

  return res;

} // orchard_key_to_edge_coord

// =============================================================================
// =============================================================================
/**
 * This utility is mostly useful when dealing with a FaceDataArrayBlock.
 */
template <size_t dim>
KOKKOS_INLINE_FUNCTION Kokkos::Array<real_t, dim>
                       orchard_key_to_face_coord(uint64_t                  key,
                                                 face_multiindex_t<dim>    face_indexes,
                                                 block_size_t<dim> const & block_sizes)
{

  // 1. compute lower left corner coordinates in real space
  constexpr bool centering = false;
  auto           XYZ_corner = orchard_key_to_vertex_coord<dim>(key, centering);

  // 2. compute cell center coordinates
  Kokkos::Array<real_t, dim> res;

  const auto & dir = face_indexes[dim];

  {
    const real_t dx_cell = static_cast<real_t>(orchard_key_t<dim>::octantLength(key)) /
                           static_cast<real_t>(orchard_key_t<dim>::ROOT_LENGTH) /
                           static_cast<real_t>(block_sizes[IX]);

    res[IX] = XYZ_corner[IX] + static_cast<real_t>(face_indexes[IX]) * dx_cell;
    res[IX] += (dir == IX) ? KALYPSSO_NUM(0.0) : HALF_F * dx_cell;
  }

  {
    const real_t dy_cell = static_cast<real_t>(orchard_key_t<dim>::octantLength(key)) /
                           static_cast<real_t>(orchard_key_t<dim>::ROOT_LENGTH) /
                           static_cast<real_t>(block_sizes[IY]);

    res[IY] = XYZ_corner[IY] + static_cast<real_t>(face_indexes[IY]) * dy_cell;
    res[IY] += (dir == IY) ? KALYPSSO_NUM(0.0) : HALF_F * dy_cell;
  }

  if constexpr (dim == 3)
  {
    const real_t dz_cell = static_cast<real_t>(orchard_key_t<dim>::octantLength(key)) /
                           static_cast<real_t>(orchard_key_t<dim>::ROOT_LENGTH) /
                           static_cast<real_t>(block_sizes[IZ]);

    res[IZ] = XYZ_corner[IZ] + static_cast<real_t>(face_indexes[IZ]) * dz_cell;
    res[IZ] += (dir == IZ) ? KALYPSSO_NUM(0.0) : HALF_F * dz_cell;
  }

  return res;

} // orchard_key_to_face_coord

// =============================================================================
// =============================================================================
/**
 * This utility is mostly useful when dealing with a EdgeDataArrayBlock.
 */
template <size_t dim>
KOKKOS_INLINE_FUNCTION Kokkos::Array<real_t, dim>
                       orchard_key_to_edge_coord(uint64_t                  key,
                                                 edge_multiindex_t<dim>    edge_indexes,
                                                 block_size_t<dim> const & block_sizes)
{

  // 1. compute lower left corner coordinates in real space
  constexpr bool centering = false;
  auto           XYZ_corner = orchard_key_to_vertex_coord<dim>(key, centering);

  // 2. compute cell center coordinates
  Kokkos::Array<real_t, dim> res;

  const auto & dir = edge_indexes[dim];

  const auto octant_length = static_cast<real_t>(orchard_key_t<dim>::octantLength(key));

  {
    const real_t dx_cell = octant_length / static_cast<real_t>(orchard_key_t<dim>::ROOT_LENGTH) /
                           static_cast<real_t>(block_sizes[IX]);

    res[IX] = XYZ_corner[IX] + static_cast<real_t>(edge_indexes[IX]) * dx_cell;
    res[IX] += (dir == IX) ? HALF_F * dx_cell : ZERO_F;
  }

  {
    const real_t dy_cell = octant_length / static_cast<real_t>(orchard_key_t<dim>::ROOT_LENGTH) /
                           static_cast<real_t>(block_sizes[IY]);

    res[IY] = XYZ_corner[IY] + static_cast<real_t>(edge_indexes[IY]) * dy_cell;
    res[IY] += (dir == IY) ? HALF_F * dy_cell : ZERO_F;
  }

  if constexpr (dim == 3)
  {
    const real_t dz_cell = octant_length / static_cast<real_t>(orchard_key_t<dim>::ROOT_LENGTH) /
                           static_cast<real_t>(block_sizes[IZ]);

    res[IZ] = XYZ_corner[IZ] + static_cast<real_t>(edge_indexes[IZ]) * dz_cell;
    res[IZ] += (dir == IZ) ? HALF_F * dz_cell : ZERO_F;
  }

  return res;

} // orchard_key_to_edge_coord

/**
 * Transform from (p4est connectivity space) vertex coordinates to real space coordinates.
 *
 * Just apply a linear rescaling to domain [x_min, x_max] x [y_min, y_max] x [z_min, z_max]
 */
template <size_t dim>
KOKKOS_INLINE_FUNCTION auto
vertex_coord_to_real_space(Kokkos::Array<real_t, dim> const & vertex_coords,
                           real_t const &                     scaling_factor,
                           Kokkos::Array<real_t, dim> const & xyz_min)
{
  if constexpr (dim == 2)
  {
    Kokkos::Array<real_t, 2> real_coords;
    real_coords[IX] = xyz_min[IX] + scaling_factor * vertex_coords[IX];
    real_coords[IY] = xyz_min[IY] + scaling_factor * vertex_coords[IY];
    return real_coords;
  }
  else if constexpr (dim == 3)
  {
    Kokkos::Array<real_t, 3> real_coords;
    real_coords[IX] = xyz_min[IX] + scaling_factor * vertex_coords[IX];
    real_coords[IY] = xyz_min[IY] + scaling_factor * vertex_coords[IY];
    real_coords[IZ] = xyz_min[IZ] + scaling_factor * vertex_coords[IZ];
    return real_coords;
  }
} // vertex_coord_to_real_space

// =============================================================================
// =============================================================================
/**
 * Given an orchard key, given a cell indexes array (integers relative to a block), compute
 * cell-center coordinates in real-space.
 */
template <size_t dim>
KOKKOS_INLINE_FUNCTION Kokkos::Array<real_t, dim>
                       orchard_key_to_cellcenter_real_space(uint64_t                           key,
                                                            Kokkos::Array<int32_t, dim>        cell_indexes,
                                                            int                                block_size,
                                                            real_t const &                     scaling_factor,
                                                            Kokkos::Array<real_t, dim> const & xyz_min)
{
  const auto xyz_center_vertex = orchard_key_to_cell_coord<dim>(key, cell_indexes, block_size);

  return vertex_coord_to_real_space<dim>(xyz_center_vertex, scaling_factor, xyz_min);

} // orchard_key_to_cellcenter_real_space

// =============================================================================
// =============================================================================
/**
 * Given an orchard key, given a cell indexes array (integers relative to a block), compute
 * face-center coordinates in real-space.
 *
 * Block of cells is assumed to be a cube (same size along all direction).
 */
template <size_t dim>
KOKKOS_INLINE_FUNCTION Kokkos::Array<real_t, dim>
                       orchard_key_to_facecenter_real_space(uint64_t                           key,
                                                            face_multiindex_t<dim>             face_indexes,
                                                            int                                block_size,
                                                            real_t const &                     scaling_factor,
                                                            Kokkos::Array<real_t, dim> const & xyz_min)
{
  const auto xyz_face_vertex = orchard_key_to_face_coord<dim>(key, face_indexes, block_size);

  return vertex_coord_to_real_space<dim>(xyz_face_vertex, scaling_factor, xyz_min);

} // orchard_key_to_facecenter_real_space

// =============================================================================
// =============================================================================
/**
 * Given an orchard key, given a cell indexes array (integers relative to a block), compute
 * edge-center coordinates in real-space.
 *
 * Block of cells is assumed to be a cube (same size along all direction).
 */
template <size_t dim>
KOKKOS_INLINE_FUNCTION Kokkos::Array<real_t, dim>
                       orchard_key_to_edgecenter_real_space(uint64_t                           key,
                                                            edge_multiindex_t<dim>             edge_indexes,
                                                            int                                block_size,
                                                            real_t const &                     scaling_factor,
                                                            Kokkos::Array<real_t, dim> const & xyz_min)
{
  const auto xyz_edge_vertex = orchard_key_to_edge_coord<dim>(key, edge_indexes, block_size);

  return vertex_coord_to_real_space<dim>(xyz_edge_vertex, scaling_factor, xyz_min);

} // orchard_key_to_edgecenter_real_space

// =============================================================================
// =============================================================================
/**
 * Given an orchard key, given a cell indexes array (integers relative to a block), compute
 * face-center coordinates in real-space.
 */
template <size_t dim>
KOKKOS_INLINE_FUNCTION Kokkos::Array<real_t, dim>
                       orchard_key_to_facecenter_real_space(uint64_t                           key,
                                                            face_multiindex_t<dim>             face_indexes,
                                                            block_size_t<dim> const &          block_sizes,
                                                            real_t const &                     scaling_factor,
                                                            Kokkos::Array<real_t, dim> const & xyz_min)
{
  const auto xyz_face_vertex = orchard_key_to_face_coord<dim>(key, face_indexes, block_sizes);

  return vertex_coord_to_real_space<dim>(xyz_face_vertex, scaling_factor, xyz_min);

} // orchard_key_to_facecenter_real_space

// =============================================================================
// =============================================================================
/**
 * Given an orchard key, given a cell indexes array (integers relative to a block), compute
 * edge-center coordinates in real-space.
 */
template <size_t dim>
KOKKOS_INLINE_FUNCTION Kokkos::Array<real_t, dim>
                       orchard_key_to_edgecenter_real_space(uint64_t                           key,
                                                            edge_multiindex_t<dim>             edge_indexes,
                                                            block_size_t<dim> const &          block_sizes,
                                                            real_t const &                     scaling_factor,
                                                            Kokkos::Array<real_t, dim> const & xyz_min)
{
  const auto xyz_edge_vertex = orchard_key_to_edge_coord<dim>(key, edge_indexes, block_sizes);

  return vertex_coord_to_real_space<dim>(xyz_edge_vertex, scaling_factor, xyz_min);

} // orchard_key_to_edgecenter_real_space

// =============================================================================
// =============================================================================
/**
 * Compute cell length for a cell that belongs to a quadrant at a given level, given also the number
 * of cells per block direction.
 *
 * Remember we assume all block have the same number of cells per direction, hence the cell length
 * is same along all direction (X, Y or Z).
 */
template <size_t dim>
KOKKOS_INLINE_FUNCTION real_t
compute_cell_length(uint8_t level, int block_size)
{

  return static_cast<real_t>(orchard_key_t<dim>::octantLength_from_level(level)) /
         static_cast<real_t>(orchard_key_t<dim>::ROOT_LENGTH) / static_cast<real_t>(block_size);

} // compute_cell_length

// =============================================================================
// =============================================================================
/**
 * Compute cell length for a cell that belongs to a quadrant at a given level, given also the number
 * of cells per block direction.
 *
 * Remember we assume all block have the same number of cells per direction, hence the cell length
 * is same along all direction (X, Y or Z).
 */
template <size_t dim>
KOKKOS_INLINE_FUNCTION real_t
compute_cell_length(uint64_t key, int block_size)
{

  return static_cast<real_t>(orchard_key_t<dim>::octantLength(key)) /
         static_cast<real_t>(orchard_key_t<dim>::ROOT_LENGTH) / static_cast<real_t>(block_size);

} // compute_cell_length

// =============================================================================
// =============================================================================
/**
 * Compute block length for a quadrant at a given level.
 *
 * Here we assume all block have the same number of cells per direction, hence the cell length
 * is same along all direction (X, Y or Z).
 */
template <size_t dim>
KOKKOS_INLINE_FUNCTION real_t
compute_block_length(uint8_t level)
{

  return static_cast<real_t>(orchard_key_t<dim>::octantLength_from_level(level)) /
         static_cast<real_t>(orchard_key_t<dim>::ROOT_LENGTH);

} // compute_block_length

// =============================================================================
// =============================================================================
/**
 * Compute absolute real space coordinates of a cell center.
 *
 * \param[in] level is the AMR level of a quadrant /  block.
 *
 * \param[in] XYZ_corner are real space coordinates of lower left corner of the block.
 *            They can be obtained by a previous call to orchard_key_to_vertex
 * \param[in] cell_indexes is an array of local integer coordinates identifying a cell
 *            inside a block in range [0, bx-1] x [0, by-1] x [0, bz-1]
 *            where bx,by,bz are defined in the input ini parameter file
 * \param[in] block_size the number of cells per dimension in a block of cells
 *
 * \return array of physical coordinates of the cell center.
 */
template <size_t dim, typename index_t>
KOKKOS_INLINE_FUNCTION Kokkos::Array<real_t, dim>
                       compute_cell_coordinates(uint8_t                     level,
                                                Kokkos::Array<real_t, dim>  XYZ_corner,
                                                Kokkos::Array<index_t, dim> cell_indexes,
                                                block_size_t<dim>           block_sizes)
{

  // index_t is supposed to be either int32_t or uint32_t
  static_assert(std::is_integral<index_t>::value, "Integral type required.");

  // 1. compute cell length in real space
  real_t dx_cell = compute_cell_length<dim>(level, block_sizes[IX]);
  real_t dy_cell = compute_cell_length<dim>(level, block_sizes[IY]);

  // 2. compute cell center coordinates
  Kokkos::Array<real_t, dim> res;

  res[IX] = XYZ_corner[IX] + static_cast<real_t>(cell_indexes[IX]) * dx_cell + HALF_F * dx_cell;
  res[IY] = XYZ_corner[IY] + static_cast<real_t>(cell_indexes[IY]) * dy_cell + HALF_F * dy_cell;
  if constexpr (dim == 3)
  {
    real_t dz_cell = compute_cell_length<dim>(level, block_sizes[IZ]);
    res[IZ] = XYZ_corner[IZ] + static_cast<real_t>(cell_indexes[IZ]) * dz_cell + HALF_F * dz_cell;
  }

  return res;

} // compute_cell_coordinates

// =======================================================
// =======================================================
/**
 * input  : linear cell index inside a block of size (bSize,bSize,bSize)
 * output : extract integer coordinates
 */
template <size_t dim>
KOKKOS_INLINE_FUNCTION Kokkos::Array<int32_t, dim>
                       icell_to_icoord(int32_t icell, int bSize)
{

  Kokkos::Array<int32_t, dim> res;

  if constexpr (dim == 2)
  {
    res[IY] = (icell / bSize);
    res[IX] = (icell - bSize * res[IY]);
  }
  else
  {
    res[IZ] = (icell / (bSize * bSize));
    int32_t icell2 = icell - bSize * bSize * res[IZ];
    res[IY] = (icell2 / bSize);
    res[IX] = (icell2 - bSize * res[IY]);
  }

  return res;

} // icell_to_icoord

// =======================================================
// =======================================================
/**
 * input  : linear cell index inside a block of size (bSize[IX],bSize[IY],bSize[IZ])
 * output : extract integer coordinates
 */
template <size_t dim>
KOKKOS_INLINE_FUNCTION Kokkos::Array<uint32_t, dim>
                       icell_to_icoord(int32_t icell, Kokkos::Array<uint32_t, dim> bSize)
{

  Kokkos::Array<uint32_t, dim> res;

  if constexpr (dim == 2)
  {
    res[IY] = static_cast<uint32_t>(icell / bSize[IX]);
    res[IX] = static_cast<uint32_t>(icell - bSize[IX] * res[IY]);
  }
  else
  {
    res[IZ] = static_cast<uint32_t>(icell / (bSize[IX] * bSize[IY]));
    int32_t icell2 = icell - bSize[IX] * bSize[IY] * res[IZ];
    res[IY] = static_cast<uint32_t>(icell2 / bSize[IX]);
    res[IX] = static_cast<uint32_t>(icell2 - bSize[IX] * res[IY]);
  }

  return res;

} // icell_to_icoord

// =======================================================
// =======================================================
/**
 * input  : extract integer coordinates
 * output : linear cell index inside a block of size (bSize[IX],bSize[IY],bSize[IZ])
 */
template <size_t dim>
KOKKOS_INLINE_FUNCTION uint32_t
icoord_to_icell(Kokkos::Array<uint32_t, dim> icoord, int bSize)
{

  return dim == 2 ? icoord[IX] + bSize * icoord[IY]
                  : icoord[IX] + bSize * icoord[IY] + bSize * bSize * icoord[IZ];

} // icoord_to_icell

// =======================================================
// =======================================================
/**
 * input  : extract integer coordinates
 * output : linear cell index inside a block of size (bSize[IX],bSize[IY],bSize[IZ])
 */
template <size_t dim>
KOKKOS_INLINE_FUNCTION uint32_t
icoord_to_icell(Kokkos::Array<uint32_t, dim> icoord, Kokkos::Array<uint32_t, dim> bSize)
{

  return dim == 2 ? icoord[IX] + bSize[IX] * icoord[IY]
                  : icoord[IX] + bSize[IX] * icoord[IY] + bSize[IX] * bSize[IY] * icoord[IZ];

} // icoord_to_icell

// // =======================================================
// // =======================================================
// KOKKOS_INLINE_FUNCTION uint32_t
// icoord_to_icell(uint32_t ix, uint32_t iy, int bSize)
// {

//   return ix + bSize * iy;

// } // icoord_to_icell

// // =======================================================
// // =======================================================
// KOKKOS_INLINE_FUNCTION uint32_t
// icoord_to_icell(uint32_t ix, uint32_t iy, uint32_t iz, int bSize)
// {

//   return ix + bSize * iy + bSize * bSize * iz;

// } // icoord_to_icell

// // =======================================================
// // =======================================================
// KOKKOS_INLINE_FUNCTION uint32_t
// icoord_to_icell(uint32_t ix, uint32_t iy, uint32_t iz, int bx, int by)
// {

//   return ix + bx * iy + bx * by * iz;

// } // icoord_to_icell

// =======================================================
// =======================================================
//! \brief swap two bits in an integer.
//! see https://www.geeksforgeeks.org/how-to-swap-two-bits-in-a-given-integer/
//!
//! \param[in] n integer to consider
//! \param[in] p1 bit position 1
//! \param[in] p2 bit position 2
//! \return new integer computed as n where bit b1 (at position p1) and b2 (at position p2) are
//! swapped
template <typename T>
KOKKOS_INLINE_FUNCTION T
swapBits(T n, int p1, int p2)
{
  static_assert(std::is_integral<T>::value, "Integral required.");

  // make sure p1 and p2 are valid bit positions
  assertm(p1 >= 0 and p2 >= 0, "swapBits: invalid bit position.");
  [[maybe_unused]] constexpr auto bitwidth = sizeof(T) * CHAR_BIT;
  assertm(p1 < static_cast<int>(bitwidth) and p2 < static_cast<int>(bitwidth),
          "swapBits: invalid bit position.");

  // left-shift 1 p1 and p2 times
  // and using XOR
  if (((n & (1 << p1)) >> p1) ^ ((n & (1 << p2)) >> p2))
  {
    n ^= 1 << p1;
    n ^= 1 << p2;
  }
  return n;
} // swapBits


} // namespace kalypsso

#endif // KALYPSSO_CORE_ORCHARD_KEY_UTILS_H
