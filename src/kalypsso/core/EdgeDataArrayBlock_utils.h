// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file EdgeDataArrayBlock_utils.h
 *
 * This header is similar to FaceDataArrayBlock_utils.h but we do not completely define a new data
 * array type, just adapt the logic to enumerate edges in both 2d / 3d, and ease mapping
 * Kokkos parallel_for iterations to edges.
 *
 * \note in 2d, edges (along Z axis) are identified to cell nodes, but should think of infinitely
 * long edge in the Z direction.
 *
 * \note Just to be clear,
 * - in a 3d (bx,by,bz) block of cells, there are (bx+1)*(by+1)*bz edges along
 * - in 3d the total number of edges is the sum of edges along X, Y and Z
 */
#ifndef KALYPSSO_CORE_EDGEDATAARRAYBLOCK_UTILS_H_
#define KALYPSSO_CORE_EDGEDATAARRAYBLOCK_UTILS_H_

#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp> // for KOKKOS_ENABLE_XXX

#include <kalypsso/core/kalypsso_core_base.h> // for definition of assertm
#include <kalypsso/core/real_type.h>
#include <kalypsso/core/Kokkos_Array_extensions.h>

#include <kalypsso/core/Locations.h>   // for edge_multiindex_t definition
#include <kalypsso/core/utils_block.h> // for definition of type block_size_t
#include <kalypsso/core/enums.h>
#include <kalypsso/core/mesh_utils.h> // for Face::face_t definition

#include <../better-enums/enum.h>

namespace kalypsso
{

/**
 * A type alias used to store index offsets indicating where the different type of edge (edges along
 * X, Y or Z ) start in a flat edge index.
 */
using edge_flat_index_offset_t = Kokkos::Array<int32_t, 4>;

// clang-format off
/**
 * Enum used to distinguish the 3 possible neighbors of a block across an edge.
 *
 *   X ----> X1
 *   | \
 *   |  \
 *  \/   X diagonal
 *  X2
 *
 * if an edge is along direction DIR0, then DIR1 and DIR2 designated / can be used to enumerate the
 * two orthogonal directions computed as
 * DIR1 = (DIR0+1) % 3
 * DIR2 = (DIR1+2) % 3
 */
BETTER_ENUM(EdgeNormalType, int, DIR1, DIR2, DIAGONAL)
// clang-format on


// ============================================================================================
// ============================================================================================
/**
 * For a given edge direction, compute the logical edge array sizes associated
 * to a block of cells of size bSize.
 *
 * \param[in] cell block size (number of cells per direction)
 *
 * \return block size (number of edges in direction "dir") per direction
 *
 * \tparam dim is dimension (2 or 3)
 * \tparam dir is the edge direction
 *
 * \note in 2d, only edges along Z will be used in practice, but just for the sake of symmetry, we
 * allow edge along IX and IY, knowing that in practice, these are identical to "faces"
 */
template <size_t dim>
KOKKOS_FORCEINLINE_FUNCTION block_size_t<dim>
                            get_edge_block_size(block_size_t<dim> bSize, int dir)
{
  KOKKOS_ASSERT((dir == IX or dir == IY or dir == IZ) && "Wrong direction index.");

  block_size_t<dim> res = bSize;

  // enlarge sizes in edge transverse directions
  if constexpr (dim == 2)
  {
    res[IX] += (dir == IX) ? 0 : 1;
    res[IY] += (dir == IY) ? 0 : 1;
  }
  if constexpr (dim == 3)
  {
    res[IX] += (dir == IX) ? 0 : 1;
    res[IY] += (dir == IY) ? 0 : 1;
    res[IZ] += (dir == IZ) ? 0 : 1;
  }

  return res;
} // get_edge_block_size

// ============================================================================================
// ============================================================================================
//! compute the maximum number of edges (among all direction) in a block of cell of size bSize.
//!
//! - when the block is square (equal dimension in all direction), the result is exact
//! - when the block is rectangular, the value is rounded up
//!
//! Purpose: allocate an array that can be addressed with the same i,j,k bounds for x,y,z faces
//!
//! \param[in] bSize cell block size (number of cells per direction)
template <size_t dim>
KOKKOS_FORCEINLINE_FUNCTION auto
maximum_number_edges_per_leaf(block_size_t<dim> bSize)
{

  // optimization for the case block has equal sizes in all direction
  if constexpr (dim == 2)
  {
    if (bSize[IX] == bSize[IY])
    {
      return (bSize[IX] + 1) * bSize[IX];
    }
  }
  else if constexpr (dim == 3)
  {
    if ((bSize[IX] == bSize[IY]) and (bSize[IX] == bSize[IZ]))
    {
      return (bSize[IX] + 1) * (bSize[IX] + 1) * bSize[IX];
    }
  }

  // general case : a rectangular block
  return dim == 2 ? (bSize[IX] + 1) * (bSize[IY] + 1)
                  : (bSize[IX] + 1) * (bSize[IY] + 1) * (bSize[IZ] + 1);

} // maximum_number_faces_per_leaf

// ============================================================================================
// ============================================================================================
/**
 * accumulate over direction the number of edges in that direction per octant.
 *
 * \param[in] bSize cell block size (number of cells per direction)
 *
 * \todo evaluate if it wouldn't be better to use to co-dimension to enumerate edge directions
 * because as seen below it would better
 *
 * \note in 2d, this is only valid when one only care about edges along Z (for MHD emfs)
 */
template <size_t dim>
KOKKOS_FORCEINLINE_FUNCTION auto
compute_edge_flat_index_offsets_emf(block_size_t<dim> bSize) -> edge_flat_index_offset_t
{
  edge_flat_index_offset_t offset;

  if constexpr (dim == 2)
  {
    offset[0] = 0;
    offset[1] = offset[0]; // no edges along X
    offset[2] = offset[1]; // no edges along Y
    offset[3] = offset[2] + Kokkos::dim_prod(get_edge_block_size<dim>(bSize, IZ));
  }
  else if constexpr (dim == 3)
  {
    offset[0] = 0;
    offset[1] = offset[0] + Kokkos::dim_prod(get_edge_block_size<dim>(bSize, IX));
    offset[2] = offset[1] + Kokkos::dim_prod(get_edge_block_size<dim>(bSize, IY));
    offset[3] = offset[2] + Kokkos::dim_prod(get_edge_block_size<dim>(bSize, IZ));
  }

  return offset;
} // compute_edge_flat_index_offsets_emf

// ============================================================================================
// ============================================================================================
/**
 * accumulate over direction the number of edges in that direction per octant.
 *
 * \see compute_edge_flat_index_offsets
 */
template <size_t dim>
KOKKOS_FORCEINLINE_FUNCTION auto
compute_edge_flat_index_offsets(block_size_t<dim> bSize) -> edge_flat_index_offset_t
{
  edge_flat_index_offset_t offset;

  if constexpr (dim == 2)
  {
    offset[0] = 0;
    offset[1] = offset[0] + Kokkos::dim_prod(get_edge_block_size<dim>(bSize, IX));
    offset[2] = offset[1] + Kokkos::dim_prod(get_edge_block_size<dim>(bSize, IY));
    offset[3] = offset[2] + Kokkos::dim_prod(get_edge_block_size<dim>(bSize, IZ));
  }
  else if constexpr (dim == 3)
  {
    offset[0] = 0;
    offset[1] = offset[0] + Kokkos::dim_prod(get_edge_block_size<dim>(bSize, IX));
    offset[2] = offset[1] + Kokkos::dim_prod(get_edge_block_size<dim>(bSize, IY));
    offset[3] = offset[2] + Kokkos::dim_prod(get_edge_block_size<dim>(bSize, IZ));
  }

  return offset;
} // compute_edge_flat_index_offsets

// ============================================================================================
// ============================================================================================
/**
 * Decode i,j,k, edge_type from an edge flat index
 *
 * Only use this one when dealing with MHD emfs (electromotive forces).
 */
template <size_t dim>
KOKKOS_FORCEINLINE_FUNCTION auto
edge_flat_index_unravel_emf(int32_t                          flat_index,
                            block_size_t<dim> const &        block_size,
                            edge_flat_index_offset_t const & edge_flat_index_offsets)
{
  KOKKOS_ASSERT(flat_index >= 0 and "flat index must be positive !");
  KOKKOS_ASSERT(flat_index < edge_flat_index_offsets[3] and "flat index invalid (too large) !");

  edge_multiindex_t<dim> index;

  // determine edge type
  index[dim] = flat_index < edge_flat_index_offsets[1]   ? IX
               : flat_index < edge_flat_index_offsets[2] ? IY
                                                         : IZ;

  if constexpr (dim == 2)
  {
    KOKKOS_ASSERT(index[dim] == IZ and "Wrong edge index, should be IZ");
    // flat_index =  i + (block_size[IX]+1) * j
    index[IY] = flat_index / (block_size[IX] + 1);
    index[IX] = flat_index - index[IY] * (block_size[IX] + 1);
  }
  else if constexpr (dim == 3)
  {
    if (index[dim] == IX)
    {
      // flat_index = edge_flat_index_offsets[0] + i + (block_size[IX]) * j +
      // (block_size[IX])*(block_size[IY]+1) * k
      // remember that flat_index_offsets[0] = 0 by design
      index[IZ] = flat_index / ((block_size[IX]) * (block_size[IY] + 1));
      flat_index -= (block_size[IX]) * (block_size[IY] + 1) * index[IZ];
      index[IY] = flat_index / (block_size[IX]);
      index[IX] = flat_index - index[IY] * (block_size[IX]);
    }
    else if (index[dim] == IY)
    {
      // flat_index = flat_index_offsets[1] + i + (block_size[IX]+1)*j + (block_size[IX]+1) *
      // (block_size[IY]) * k
      flat_index -= edge_flat_index_offsets[1];
      index[IZ] = flat_index / ((block_size[IX] + 1) * (block_size[IY]));
      flat_index -= (block_size[IX] + 1) * (block_size[IY]) * index[IZ];
      index[IY] = flat_index / (block_size[IX] + 1);
      index[IX] = flat_index - index[IY] * (block_size[IX] + 1);
    }
    else if (index[dim] == IZ) // IZ
    {
      // flat_index = flat_index_offsets[2] + i + (block_size[IX]+1)*j + (block_size[IX]+1) *
      // (block_size[IY]+1) * k
      flat_index -= edge_flat_index_offsets[2];
      index[IZ] = flat_index / ((block_size[IX] + 1) * (block_size[IY] + 1));
      flat_index -= (block_size[IX] + 1) * (block_size[IY] + 1) * index[IZ];
      index[IY] = flat_index / (block_size[IX] + 1);
      index[IX] = flat_index - index[IY] * (block_size[IX] + 1);
    }
  }

  return index;
} // edge_flat_index_unravel_emf

// ============================================================================================
// ============================================================================================
/**
 * Decode i,j,k, edge_type from an edge flat index
 */
template <size_t dim>
KOKKOS_FORCEINLINE_FUNCTION auto
edge_flat_index_unravel(int32_t                          flat_index,
                        block_size_t<dim> const &        block_size,
                        edge_flat_index_offset_t const & edge_flat_index_offsets)
{
  KOKKOS_ASSERT(flat_index >= 0 and "flat index must be positive !");
  KOKKOS_ASSERT(flat_index < edge_flat_index_offsets[3] and "flat index invalid (too large) !");

  edge_multiindex_t<dim> index;

  // determine edge type
  index[dim] = flat_index < edge_flat_index_offsets[1]   ? IX
               : flat_index < edge_flat_index_offsets[2] ? IY
                                                         : IZ;

  if constexpr (dim == 2)
  {
    // flat_index =  i + (edge_bsize[IX]) * j
    const auto edge_bsize = get_edge_block_size<dim>(block_size, index[dim]);
    flat_index -= edge_flat_index_offsets[index[dim]];

    index[IY] = flat_index / edge_bsize[IX];
    index[IX] = flat_index - index[IY] * edge_bsize[IX];
  }
  else if constexpr (dim == 3)
  {
    // flat_index =  i + edge_bsize[IX] * j
    const auto edge_bsize = get_edge_block_size<dim>(block_size, index[dim]);
    flat_index -= edge_flat_index_offsets[index[dim]];

    index[IZ] = flat_index / (edge_bsize[IX] * edge_bsize[IY]);
    flat_index -= edge_bsize[IX] * edge_bsize[IY] * index[IZ];
    index[IY] = flat_index / edge_bsize[IX];
    index[IX] = flat_index - index[IY] * edge_bsize[IX];
  }

  return index;
} // edge_flat_index_unravel

// ============================================================================================
// ============================================================================================
/**
 * edge coordinates to cell coordinates.
 */
template <size_t dim>
KOKKOS_INLINE_FUNCTION coord_t<dim>
edge_to_cell_coords(edge_multiindex_t<dim> const & edge_indexes, block_size_t<dim> const & bSizes)
{
  if constexpr (dim == 2)
  {
    // clang-format off
   const coord_t<2> ijk{ edge_indexes[IX] == bSizes[IX] ? edge_indexes[IX] - 1 : edge_indexes[IX],
                         edge_indexes[IY] == bSizes[IY] ? edge_indexes[IY] - 1 : edge_indexes[IY] };
    // clang-format on
    return ijk;
  }
  else if constexpr (dim == 3)
  {
    // clang-format off
    const coord_t<3> ijk{ edge_indexes[IX] == bSizes[IX] ? edge_indexes[IX] - 1 : edge_indexes[IX],
                          edge_indexes[IY] == bSizes[IY] ? edge_indexes[IY] - 1 : edge_indexes[IY],
                          edge_indexes[IZ] == bSizes[IZ] ? edge_indexes[IZ] - 1 : edge_indexes[IZ] };
    // clang-format on
    return ijk;
  }
} // edge_to_cell_coords

// ============================================================================================
// ============================================================================================
/**
 * is edge on the far right ?
 */
template <size_t dim>
KOKKOS_INLINE_FUNCTION bool
is_edge_on_the_right(edge_multiindex_t<dim> const & edge_indexes, block_size_t<dim> const & bSizes)
{
  if constexpr (dim == 2)
  {
    return edge_indexes[IX] == bSizes[IX] or edge_indexes[IY] == bSizes[IY];
  }
  else if constexpr (dim == 3)
  {
    return edge_indexes[IX] == bSizes[IX] or edge_indexes[IY] == bSizes[IY] or
           edge_indexes[IZ] == bSizes[IZ];
  }
} // is_edge_on_the_right

// ============================================================================================
// ============================================================================================
/**
 * Returns true if edge "touches" block border.
 *
 * Said differently edge touches border when it belongs to a cell at border.
 */
template <size_t dim>
KOKKOS_INLINE_FUNCTION bool
is_edge_at_block_border(edge_multiindex_t<dim> const & ijk,
                        block_size_t<dim> const &      bSize,
                        Face::face_t                   border_loc)
{
  const int dir = border_loc / 2;
  KOKKOS_ASSERT((dir == IX or dir == IY or dir == IZ) && "Wrong direction index.");

  const auto edge_block_size = get_edge_block_size<dim>(bSize, dir);

  if constexpr (dim == 2)
  {
    return (border_loc == Face::XMIN and ijk[IX] == 0) or
           (border_loc == Face::XMAX and ijk[IX] == edge_block_size[IX] - 1) or
           (border_loc == Face::YMIN and ijk[IY] == 0) or
           (border_loc == Face::YMAX and ijk[IY] == edge_block_size[IY] - 1);
  }
  else if constexpr (dim == 3)
  {
    return (border_loc == Face::XMIN and ijk[IX] == 0) or
           (border_loc == Face::XMAX and ijk[IX] == edge_block_size[IX] - 1) or
           (border_loc == Face::YMIN and ijk[IY] == 0) or
           (border_loc == Face::YMAX and ijk[IY] == edge_block_size[IY] - 1) or
           (border_loc == Face::ZMIN and ijk[IZ] == 0) or
           (border_loc == Face::ZMAX and ijk[IZ] == edge_block_size[IZ] - 1);
  }

  return false;

} // is_edge_at_block_border

// ============================================================================================
// ============================================================================================
/**
 * Returns true if edge is contained in block surface.
 *
 * In 2d, this is equivalent to is_at_block_border
 */
template <size_t dim>
KOKKOS_INLINE_FUNCTION bool
is_edge_at_block_surface(edge_multiindex_t<dim> const & ijk,
                         block_size_t<dim> const &      bSize,
                         Face::face_t                   border_loc)
{
  if constexpr (dim == 2)
  {
    const int dir = border_loc / 2;
    KOKKOS_ASSERT((dir == IX or dir == IY or dir == IZ) && "Wrong direction index.");

    const auto   edge_block_size = get_edge_block_size<dim>(bSize, dir);
    auto const & edge_dir = ijk[dim];

    if (edge_dir == IZ)
      return (border_loc == Face::XMIN and ijk[IX] == 0) or
             (border_loc == Face::XMAX and ijk[IX] == edge_block_size[IX] - 1) or
             (border_loc == Face::YMIN and ijk[IY] == 0) or
             (border_loc == Face::YMAX and ijk[IY] == edge_block_size[IY] - 1);
    else if (edge_dir == IX)
    {
      return (border_loc == Face::YMIN and ijk[IY] == 0) or
             (border_loc == Face::YMAX and ijk[IY] == edge_block_size[IY] - 1);
    }
    else if (edge_dir == IY)
    {
      return (border_loc == Face::XMIN and ijk[IX] == 0) or
             (border_loc == Face::XMAX and ijk[IX] == edge_block_size[IX] - 1);
    }
  }
  else if constexpr (dim == 3)
  {
    const int dir = border_loc / 2;
    KOKKOS_ASSERT((dir == IX or dir == IY or dir == IZ) && "Wrong direction index.");

    const auto   edge_block_size = get_edge_block_size<dim>(bSize, dir);
    auto const & edge_dir = ijk[dim];

    if (edge_dir == IZ)
      return (border_loc == Face::XMIN and ijk[IX] == 0) or
             (border_loc == Face::XMAX and ijk[IX] == edge_block_size[IX] - 1) or
             (border_loc == Face::YMIN and ijk[IY] == 0) or
             (border_loc == Face::YMAX and ijk[IY] == edge_block_size[IY] - 1);
    else if (edge_dir == IX)
    {
      return (border_loc == Face::YMIN and ijk[IY] == 0) or
             (border_loc == Face::YMAX and ijk[IY] == edge_block_size[IY] - 1) or
             (border_loc == Face::ZMIN and ijk[IZ] == 0) or
             (border_loc == Face::ZMAX and ijk[IZ] == edge_block_size[IZ] - 1);
    }
    else if (edge_dir == IY)
    {
      return (border_loc == Face::ZMIN and ijk[IZ] == 0) or
             (border_loc == Face::ZMAX and ijk[IZ] == edge_block_size[IZ] - 1) or
             (border_loc == Face::XMIN and ijk[IX] == 0) or
             (border_loc == Face::XMAX and ijk[IX] == edge_block_size[IX] - 1);
    }
  }

  return false;

} // is_at_block_surface

// ============================================================================================
// ============================================================================================
template <size_t dim>
KOKKOS_INLINE_FUNCTION bool
is_edge_in_the_middle_of_a_face_impl(edge_multiindex_t<dim> const & ijk,
                                     block_size_t<dim> const &      bSize,
                                     ComponentIndex3D               dir1,
                                     ComponentIndex3D               dir2)
{

  if ((ijk[dir1] == 0 or ijk[dir1] == bSize[dir1]) and ijk[dir2] == bSize[dir2] / 2)
    return true;

  if ((ijk[dir2] == 0 or ijk[dir2] == bSize[dir2]) and ijk[dir1] == bSize[dir1] / 2)
    return true;

  return false;

} // is_edge_in_the_middle_of_a_face_impl

// ============================================================================================
// ============================================================================================
/**
 * Returns true if edge is in the middle of a face (see drawing).
 *
 * In 2d, this is simple, this corresponds exactly to four location marked with an "x"
 *
 *  _______x________
 * |   |   |   |   |
 * |___|___|___|___|
 * |   |   |   |   |
 * x___|___|___|___x
 * |   |   |   |   |
 * |___|___|___|___|
 * |   |   |   |   |
 * |___|___x___|___|
 *
 * In 3d, this a bit more complex, let look at a face in the (x,y) plane, it corresponds to all
 * the edges strictly contained in the face (edge along X or along Y axis not on the face border)
 * and aligned with one of the two lines passing through the face center that are axis aligned.
 *
 * In the drawing below:
 * - edges marked with an "x" are aligned with X axis and passing through the face center,
 * - edges marked with an "y" are aligned with Y axis and passing through the face center,
 *
 *  ________________
 * |   |   y   |   |
 * |___|___y___|___|
 * |   |   y   |   |
 * |xxx|xxx xxx|xxx|
 * |   |   y   |   |
 * |___|___y___|___|
 * |   |   y   |   |
 * |___|___y___|___|
 *
 * \note this function is useful when dealing with computation that need to access both a given
 * edge location in current current and corresponding edge location (same physical location) in
 * the face-neighbor octant.
 *
 * There are two situations:
 * - if the face is non-hanging, there isn't any problem, there is a 1-to-1 correspondence between
 * the edge in current octant and an edge in the neighbor;
 * - if the face is hanging (i.e. non-conform) then all edges at the middle of the large face have
 * 2 neighbor edges the fine neighbor octants.
 *
 * \note Important: this routine is only valid when cell block sizes are all even integers (which
 * we assume at multiple locations).
 */
template <size_t dim>
KOKKOS_INLINE_FUNCTION bool
is_edge_in_the_middle_of_a_face(edge_multiindex_t<dim> const & ijk, block_size_t<dim> const & bSize)
{
  auto const & edge_dir = ijk[dim];

  if constexpr (dim == 2)
  {
    if (edge_dir == IZ)
    {
      const auto dir1 = IX;
      const auto dir2 = IY;
      return is_edge_in_the_middle_of_a_face_impl(ijk, bSize, dir1, dir2);
    }
  }
  else if constexpr (dim == 3)
  {
    if (edge_dir == IZ)
    {
      const auto dir1 = IX;
      const auto dir2 = IY;
      return is_edge_in_the_middle_of_a_face_impl(ijk, bSize, dir1, dir2);
    }
    else if (edge_dir == IX)
    {
      const auto dir1 = IY;
      const auto dir2 = IZ;
      return is_edge_in_the_middle_of_a_face_impl(ijk, bSize, dir1, dir2);
    }
    else if (edge_dir == IY)
    {
      const auto dir1 = IZ;
      const auto dir2 = IX;
      return is_edge_in_the_middle_of_a_face_impl(ijk, bSize, dir1, dir2);
    }
  }

  return false;

} // is_edge_in_the_middle_of_a_face

// ============================================================================================
// ============================================================================================
/**
 * Returns true if cell edge is also an edge of the block itself.
 *
 * Edges marked with an "x" are also block edges.
 *
 *  x______________x
 * |   |   |   |   |
 * |___|___|___|___|
 * |   |   |   |   |
 * |___|___|___|___|
 * |   |   |   |   |
 * |___|___|___|___|
 * |   |   |   |   |
 * x___|___|___|___x
 *
 * \note a corner is identified by the intersection of two faces.
 *
 * \param[in] ijk identifies an edge location
 * \param[in] bSize cell block size
 * \param[in] border_loc0 identifies a border
 * \param[in] border_loc1 identifies another border that must be orthogonal to border_loc0
 */
template <size_t dim>
KOKKOS_INLINE_FUNCTION bool
is_edge_at_block_edge(edge_multiindex_t<dim> const & ijk,
                      block_size_t<dim> const &      bSize,
                      Face::face_t                   border_loc0,
                      Face::face_t                   border_loc1)
{
  [[maybe_unused]] const int dir0 = border_loc0 / 2;
  KOKKOS_ASSERT((dir0 == IX or dir0 == IY or dir0 == IZ) && "Wrong direction index.");

  [[maybe_unused]] const int dir1 = border_loc1 / 2;
  KOKKOS_ASSERT((dir1 == IX or dir1 == IY or dir1 == IZ) && "Wrong direction index.");

  // border_loc0 and border_loc1 must be orthogonal
  KOKKOS_ASSERT((dir0 != dir1) && "dir0 and dir1 must be orthogonal");

  return is_edge_at_block_surface<dim>(ijk, bSize, border_loc0) and
         is_edge_at_block_surface<dim>(ijk, bSize, border_loc1);

} // is_edge_at_block_edge

// ============================================================================================
// ============================================================================================
template <size_t dim>
KOKKOS_INLINE_FUNCTION auto
get_edge_outside_unit_vector(Face::face_t border_loc0,
                             Face::face_t border_loc1) -> Kokkos::Array<int, dim>
{
  Kokkos::Array<int, dim> v;
  for (int i = 0; i < dim; ++i)
    v[i] = 0;

  const auto dir0 = border_loc0 / 2;
  const auto dir1 = border_loc1 / 2;

  {
    const auto faces = Face::get_pair_of_faces<dim>(dir0);
    v[dir0] = border_loc0 == faces[0] ? -1 : border_loc0 == faces[1] ? 1 : 0;
  }
  {
    const auto faces = Face::get_pair_of_faces<dim>(dir1);
    v[dir1] = border_loc1 == faces[0] ? -1 : border_loc1 == faces[1] ? 1 : 0;
  }

  return v;

} // get_unit_outside_vector

// ============================================================================================
// ============================================================================================
/**
 * if edge is strictly inside block, this vector is zero (square norm is 0).
 * if edge is a border on a face, return the face outside normal (square norm is 1)
 * if edge is also a block edge return outside normal (square norm is 2)
 *
 * in all other cases, return null vector
 */
template <size_t dim>
KOKKOS_INLINE_FUNCTION auto
get_edge_outside_unit_vector(edge_multiindex_t<dim> const & ijk,
                             block_size_t<dim> const &      cell_block_size,
                             EdgeNormalType                 edge_normal_type) -> shift_t<dim>
{
  shift_t<dim> v;
  for (size_t i = 0; i < dim; ++i)
    v[i] = 0;

  auto const & edge_dir = ijk[dim];

  // compute normal directions to current edge
  const auto dir1 = (edge_dir + 1) % 3;
  const auto dir2 = (edge_dir + 2) % 3;

  if (edge_normal_type == +EdgeNormalType::DIAGONAL or edge_normal_type == +EdgeNormalType::DIR1)
  {
    if (ijk[dir1] == 0)
      v[dir1] = -1;
    else if (ijk[dir1] == cell_block_size[dir1])
      v[dir1] = 1;
  }

  if (edge_normal_type == +EdgeNormalType::DIAGONAL or edge_normal_type == +EdgeNormalType::DIR2)
  {
    if (ijk[dir2] == 0)
      v[dir2] = -1;
    else if (ijk[dir2] == cell_block_size[dir2])
      v[dir2] = 1;
  }

  return v;

} // get_unit_outside_vector

} // namespace kalypsso

#endif // KALYPSSO_CORE_EDGEDATAARRAYBLOCK_UTILS_H_
