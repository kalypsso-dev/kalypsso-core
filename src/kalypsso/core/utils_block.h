// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file utils_block.h
 */
#ifndef KALYPSSO_CORE_UTILS_BLOCK_H_
#define KALYPSSO_CORE_UTILS_BLOCK_H_

#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/enums.h>
#include <kalypsso/core/kalypsso_core_config.h> // for KALYPSSO_CORE_ENABLE_DEBUG
#include <kalypsso/core/Kokkos_Array_extensions.h>

namespace kalypsso
{

//! A type alias to use uniformly in kalypsso for holding block size, ghost sizes, etc...
template <size_t dim>
using block_size_t = Kokkos::Array<int32_t, dim>;

/**
 * Type alias to hold
 * - (i, j, ivar, iOct) in 2d,
 * - (i, j, k, ivar, iOct) in 3d,
 *
 * i.e. a multi-index locating a cell in a DataArrayBlock.
 *
 */
template <size_t dim>
using block_multiindex_t = Kokkos::Array<int32_t, dim + 2>;

/**
 * Type alias to hold (i,j,ivar) in 2d or (i,j,k,ivar) in 3d, i.e. a multi-index locating a face in
 * a FaceDataArrayBlock.
 *
 * ivar can only take 3 values: IX, IY and IZ representing respectively Y-face, Y-face or Z-face
 * data.
 *
 * \note note about valid range of values for i,j,k
 * In 2d, there are
 * - (block_size[IX]+1)*(block_sizes[IY]) vertical faces, aka X-faces (normal to X direction)
 * - (block_size[IX])*(block_sizes[IY]+1) horizontal faces, aka Y-faces (normal to Y direction)
 *
 */
template <size_t dim, typename T = int32_t>
using face_multiindex_t = Kokkos::Array<T, dim + 1>;

/**
 * Type alias to hold (i,j,dir) in 2d or (i,j,k,dir) in 3d, i.e. a multi-index locating an edge.
 *
 * dir can only take
 * - 1 value  in 2d : IZ
 * - 3 values in 3d : IX, IY or IZ representing edge direction
 *
 * This is class is designed to help manipulating EMF (line-integrated electric field in MHD).
 *
 * In 2d, it only makes sense to have edge along Z axis, because edges along X or Y are considered
 * as faces.
 *
 */
template <size_t dim, typename T = int32_t>
using edge_multiindex_t = Kokkos::Array<T, dim + 1>;

/**
 * return true if input face multiindex corresponds to an external face (i.e. has even index
 * along given direction).
 *
 * \note ivar=IZ is always an external face in 2d.
 *
 */
template <size_t dim>
KOKKOS_INLINE_FUNCTION bool
is_external_face(face_multiindex_t<dim> const & face_indexes)
{
  if constexpr (dim == 2)
  {
    auto const & i = face_indexes[IX];
    auto const & j = face_indexes[IY];
    const auto & ivar = face_indexes[dim];

    return (ivar == IX and (i & 0x1) == 0) or (ivar == IY and (j & 0x1) == 0) or (ivar == IZ);
  }
  else if constexpr (dim == 3)
  {
    auto const & i = face_indexes[IX];
    auto const & j = face_indexes[IY];
    auto const & k = face_indexes[IZ];
    const auto & ivar = face_indexes[dim];

    return (ivar == IX and (i & 0x1) == 0) or (ivar == IY and (j & 0x1) == 0) or
           (ivar == IZ and (k & 0x1) == 0);
  }
} // is_external_face

/**
 * return true if input face multiindex corresponds to an internal face (i.e. has odd index
 * along given direction).
 *
 * \note ivar=IZ is never an internal face in 2d.
 */
template <size_t dim>
KOKKOS_INLINE_FUNCTION bool
is_internal_face(face_multiindex_t<dim> const & face_indexes)
{
  if constexpr (dim == 2)
  {
    auto const & i = face_indexes[IX];
    auto const & j = face_indexes[IY];
    const auto & ivar = face_indexes[dim];

    return (ivar == IX and (i & 0x1) == 1) or (ivar == IY and (j & 0x1) == 1);
  }
  else if constexpr (dim == 3)
  {
    auto const & i = face_indexes[IX];
    auto const & j = face_indexes[IY];
    auto const & k = face_indexes[IZ];
    const auto & ivar = face_indexes[dim];

    return (ivar == IX and (i & 0x1) == 1) or (ivar == IY and (j & 0x1) == 1) or
           (ivar == IZ and (k & 0x1) == 1);
  }

} // is_internal_face

/**
 * create a face_multiindex_t from i,j,k and direction
 */
template <size_t dim>
KOKKOS_INLINE_FUNCTION face_multiindex_t<dim>
to_face_multiindex(Kokkos::Array<int32_t, dim> const & ijk, int32_t direction)
{
  face_multiindex_t<dim> res;
  res[IX] = ijk[IX];
  res[IY] = ijk[IY];
  if constexpr (dim == 3)
  {
    res[IZ] = ijk[IZ];
  }
  res[dim] = direction;

  return res;
} // to_face_multiindex

//! create uniform block size
template <size_t dim>
auto
get_block_size(int32_t value)
{
  if constexpr (dim == 2)
  {
    return block_size_t<2>{ value, value };
  }
  else if constexpr (dim == 3)
  {
    return block_size_t<3>{ value, value, value };
  }
}


//! A type alias for representing coordinates of a cell inside a block of cells (on the tree
//! leaves).
template <size_t dim, typename T = int32_t>
using coord_t = Kokkos::Array<T, dim>;

//! a type alias used in StencilHelper interface
template <size_t dim>
using shift_t = coord_t<dim, int32_t>;

//! create uniform shift
template <size_t dim>
KOKKOS_INLINE_FUNCTION auto
get_shift(int32_t value)
{
  if constexpr (dim == 2)
  {
    return shift_t<2>{ value, value };
  }
  else if constexpr (dim == 3)
  {
    return shift_t<3>{ value, value, value };
  }
}

// =======================================================
// =======================================================
/**
 * input  : index to a given cell inside a block of data of size (bx,by,bz)
 * output : extract integer coordinates
 */
template <size_t dim>
KOKKOS_INLINE_FUNCTION coord_t<dim>
                       cellindex_to_coord(int32_t index, block_size_t<dim> const & bSizes)
{
  KOKKOS_ASSERT(index >= 0);

  coord_t<dim> res;

  if constexpr (dim == 2)
  {
    const auto & bx = bSizes[IX];
    res[IY] = (index / bx);
    res[IX] = (index - bx * res[IY]);
  }
  else if constexpr (dim == 3)
  {
    const auto & bx = bSizes[IX];
    const auto & by = bSizes[IY];

    res[IZ] = (index / (bx * by));
    int32_t index2 = index - bx * by * res[IZ];
    res[IY] = (index2 / bx);
    res[IX] = (index2 - bx * res[IY]);
  }

  return res;

} // cellindex_to_coord

// =======================================================
// =======================================================
/**
 * input  : flat index to a given cell inside a block of data of size (bx,by,bz)
 * output : extract integer coordinates
 *
 * \note this variant is only useful for computing coordinates of a DataArrayGhostedBlock cell.
 */
template <size_t dim>
KOKKOS_INLINE_FUNCTION coord_t<dim>
                       cellindex_to_coord(int32_t                     flatindex,
                                          block_size_t<dim> const &   bSizes,
                                          coord_t<dim, int32_t> const shift)
{
  KOKKOS_ASSERT(flatindex >= 0);
  KOKKOS_ASSERT(flatindex < Kokkos::dim_prod(bSizes));

  coord_t<dim> res;

  if constexpr (dim == 2)
  {
    const auto & bx = bSizes[IX];
    res[IY] = (flatindex / bx);
    res[IX] = (flatindex - bx * res[IY]);

    res[IY] += shift[IY];
    res[IX] += shift[IX];
  }
  else if constexpr (dim == 3)
  {
    const auto & bx = bSizes[IX];
    const auto & by = bSizes[IY];

    res[IZ] = (flatindex / (bx * by));
    int32_t index2 = flatindex - bx * by * res[IZ];
    res[IY] = (index2 / bx);
    res[IX] = (index2 - bx * res[IY]);

    res[IZ] += shift[IZ];
    res[IY] += shift[IY];
    res[IX] += shift[IX];
  }

  return res;

} // cellindex_to_coord

// =======================================================
// =======================================================
/**
 * Compute cell coordinates modulo inner block size.
 *
 * Consider a block of cells with a ghost layer of width 1, the lower left cells has coordinates
 * (-1, -1), so it belongs to a neighbor block of cell. He were want to retrieve the coordinates
 * from the neighbor point of view (without ghost cells), that is we just want to compute
 * coordinates modulo the inner block size.
 *
 * Convert cell coordinates (ghosted) into cell coordinates (non ghost) using modulo operation.
 *
 * \param[out] cell_coords_inner coordinates of the source cell (non-ghosted coordinates)
 * \param[in] cell_coords_ghosted coordinates of the destination cell (ghosted coordinates)
 * \param[in] b block sizes
 *
 * \return relative direction to neighbor block where to look for data to copy (array of integer,
 * one by direction).
 *
 * - when cell_coords_out corresponds to a cell inside current, then direction is (0,0) in 2d
 * - when cell_coords_out corresponds to a cell outside current, then direction correspond to one
 * of the 8 sectors around current block.
 *
 *
 *               |                     |
 *  dir=(-1,1)   |      dir=(0,1)      |  dir=(1,1)
 *          _____|_____________________|_______
 *               |                     |
 *               |                     |
 *               |                     |
 *               |   Current block     |
 *  dir=(-1,0)   |      dir=(0,0)      |  dir=(1,0)
 *               |                     |
 *               |                     |
 *               |                     |
 *               |                     |
 *          _____|_____________________|_______
 *               |                     |
 *  dir=(-1,-1)  |     dir=(0,-1)      | dir=(1,-1)
 *               |                     |
 *
 */
template <size_t dim>
KOKKOS_INLINE_FUNCTION Kokkos::Array<int8_t, dim>
                       ghosted_coords_to_inner_coords(coord_t<dim> &            cell_coords_inner,
                                                      coord_t<dim> const &      cell_coords_ghosted,
                                                      block_size_t<dim> const & b)
{
  auto dir = []() {
    if constexpr (dim == 2)
      return Kokkos::Array<int8_t, 2>{ 0, 0 };
    else if constexpr (dim == 3)
      return Kokkos::Array<int8_t, 3>{ 0, 0, 0 };
  }();

  // check if source cell is inside current block, or in neighbor block
  if (cell_coords_ghosted[IX] < 0)
  {
    dir[IX] = -1;
    cell_coords_inner[IX] = cell_coords_ghosted[IX] + b[IX];
  }
  else if (cell_coords_ghosted[IX] >= b[IX])
  {
    dir[IX] = 1;
    cell_coords_inner[IX] = cell_coords_ghosted[IX] - b[IX];
  }
  else
  {
    cell_coords_inner[IX] = cell_coords_ghosted[IX];
  }

  if (cell_coords_ghosted[IY] < 0)
  {
    dir[IY] = -1;
    cell_coords_inner[IY] = cell_coords_ghosted[IY] + b[IY];
  }
  else if (cell_coords_ghosted[IY] >= b[IY])
  {
    dir[IY] = 1;
    cell_coords_inner[IY] = cell_coords_ghosted[IY] - b[IY];
  }
  else
  {
    cell_coords_inner[IY] = cell_coords_ghosted[IY];
  }

  if constexpr (dim == 3)
  {
    if (cell_coords_ghosted[IZ] < 0)
    {
      dir[IZ] = -1;
      cell_coords_inner[IZ] = cell_coords_ghosted[IZ] + b[IZ];
    }
    else if (cell_coords_ghosted[IZ] >= b[IZ])
    {
      dir[IZ] = 1;
      cell_coords_inner[IZ] = cell_coords_ghosted[IZ] - b[IZ];
    }
    else
    {
      cell_coords_inner[IZ] = cell_coords_ghosted[IZ];
    }
  }

  return dir;

} // ghosted_coords_to_inner_coords

// =======================================================
// =======================================================
/**
 * input  : index to a given cell inside a block of data of size (bx,by,bz)
 * output : extract integer coordinates, add shift
 */
template <size_t dim>
KOKKOS_INLINE_FUNCTION coord_t<dim>
                       cellindex_to_shifted_coord(uint32_t                  index,
                                                  block_size_t<dim> const & bSizes,
                                                  shift_t<dim> const &      shift)
{
  coord_t<dim> res;

  if constexpr (dim == 2)
  {
    const auto & bx = bSizes[IX];
    res[IY] = (index / bx);
    res[IX] = (index - bx * res[IY]);

    res[IX] += shift[IX];
    res[IY] += shift[IY];
  }
  else if constexpr (dim == 3)
  {
    const auto & bx = bSizes[IX];
    const auto & by = bSizes[IY];

    res[IZ] = (index / (bx * by));
    int32_t index2 = index - bx * by * res[IZ];
    res[IY] = (index2 / bx);
    res[IX] = (index2 - bx * res[IY]);

    res[IX] += shift[IX];
    res[IY] += shift[IY];
    res[IZ] += shift[IZ];
  }

  return res;

} // cellindex_to_shifted_coord

// =======================================================
// =======================================================
/**
 * output : extract integer coordinates
 * input  : index to a given cell inside a block of data of size (bx,by,bz)
 */
template <size_t dim>
KOKKOS_INLINE_FUNCTION int32_t
coord_to_cellindex(coord_t<dim> const & coord, block_size_t<dim> const & bSizes)
{
  int32_t index = 0;

  if constexpr (dim == 2)
  {
    return coord[IX] + bSizes[IX] * coord[IY];
  }
  else if constexpr (dim == 3)
  {
    return coord[IX] + bSizes[IX] * coord[IY] + bSizes[IX] * bSizes[IY] * coord[IZ];
  }

  return index;

} // coord_to_cellindex

// =======================================================
// =======================================================
/**
 * output : extract integer coordinates
 * input  : index to a given cell inside a block of data of size (bx,by,bz)
 */
template <size_t dim>
KOKKOS_INLINE_FUNCTION int32_t
coord_to_cellindex(coord_t<dim> const &         coord,
                   coord_t<dim, int8_t> const & shift,
                   block_size_t<dim> const &    bSizes)
{
  int32_t index = 0;

  if constexpr (dim == 2)
  {
    return (coord[IX] + shift[IX]) + bSizes[IX] * (coord[IY] + shift[IY]);
  }
  else if constexpr (dim == 3)
  {
    return (coord[IX] + shift[IX]) + bSizes[IX] * (coord[IY] + shift[IY]) +
           bSizes[IX] * bSizes[IY] * (coord[IZ] + shift[IZ]);
  }

  return index;

} // coord_to_cellindex

// =======================================================
// =======================================================
/**
 * \param[in] cellindex is index to a given cell inside a block of data of size (bx,by,bz)
 * \param[in] shift specify a displacement inside current block
 * \param[in] bSizes specify the block sizes
 *
 * \return the shifted cellindex
 *
 * \note in release build, we do not check that the return cellindex is valid, it would require to
 * convert cellindex into cell coordinates. We only do it in debug build
 */
template <size_t dim>
KOKKOS_INLINE_FUNCTION int32_t
shifted_cellindex(int32_t const &             cellindex,
                  coord_t<dim, int32_t> const shift,
                  block_size_t<dim> const &   bSizes)
{
  KOKKOS_ASSERT(cellindex < dim_prod(bSizes));

#ifdef KALYPSSO_CORE_ENABLE_DEBUG
  // check shifted cellindex is valid
  coord_t<dim> coord = cellindex_to_coord<dim>(cellindex, bSizes);
  for (size_t dir = 0; dir < dim; ++dir)
  {
    int32_t coord_dir = coord[dir] + shift[dir];
    KOKKOS_ASSERT(coord_dir >= 0 && coord_dir < static_cast<int32_t>(bSizes[dir]) &&
                  "Wrong shift, new cell is outside block");
  }
#endif

  int32_t index = 0;

  if constexpr (dim == 2)
  {
    return cellindex + shift[IX] + bSizes[IX] * shift[IY];
  }
  else if constexpr (dim == 3)
  {
    return cellindex + shift[IX] + bSizes[IX] * shift[IY] + bSizes[IX] * bSizes[IY] * shift[IZ];
  }

  return index;

} // shifted_cellindex

// =======================================================
// =======================================================
/**
 * \param[in] coords coordinates of a face
 * \param[in] fbSizes is a face block sizes (block size plus 1)
 *
 * \return true if multi-index coords correspond to an actual face multi-index
 */
template <size_t dim, int dir>
KOKKOS_INLINE_FUNCTION bool
are_face_coords_valid(coord_t<dim> const & coords, block_size_t<dim> const & fbSizes)
{

  if constexpr (dim == 2)
  {
    if constexpr (dir == IX)
    {
      return (coords[IY] < (fbSizes[IY] - 1));
    }
    else if constexpr (dir == IY)
    {
      return (coords[IX] < (fbSizes[IX] - 1));
    }
    else if constexpr (dir == IZ)
    {
      return (coords[IX] < (fbSizes[IX] - 1)) and (coords[IY] < (fbSizes[IY] - 1));
    }
  }
  else if constexpr (dim == 3)
  {
    if constexpr (dir == IX)
    {
      return (coords[IY] < (fbSizes[IY] - 1)) and (coords[IZ] < (fbSizes[IZ] - 1));
    }
    else if constexpr (dir == IY)
    {
      return (coords[IZ] < (fbSizes[IZ] - 1)) and (coords[IX] < (fbSizes[IX] - 1));
    }
    else if constexpr (dir == IZ)
    {
      return (coords[IX] < (fbSizes[IX] - 1)) and (coords[IY] < (fbSizes[IY] - 1));
    }
  }

  return false;

} // are_face_coords_valid

// =======================================================
// =======================================================
/**
 * \param[in] coords coordinates of a face
 * \param[in] fbSizes is a face block sizes (block size plus 1)
 *
 * \return a multi-index coords corresponding to a cell center to which current face is a left face,
 * except for the last face index which is a right face to last cell in block.
 */
template <size_t dim>
KOKKOS_INLINE_FUNCTION auto
face_to_center(coord_t<dim> const & face_coords, block_size_t<dim> const & fbSizes)
{

  coord_t<dim> center_coords = face_coords;

  center_coords[IX] =
    (face_coords[IX] == (fbSizes[IX] - 1)) ? face_coords[IX] - 1 : face_coords[IX];

  center_coords[IY] =
    (face_coords[IY] == (fbSizes[IY] - 1)) ? face_coords[IY] - 1 : face_coords[IY];

  if constexpr (dim == 3)
  {
    center_coords[IZ] =
      (face_coords[IZ] == (fbSizes[IZ] - 1)) ? face_coords[IZ] - 1 : face_coords[IZ];
  }

  return center_coords;

} // face_to_center

// =======================================================
// =======================================================
/**
 * create an edge_multiindex_t from i,j,k and direction
 */
template <size_t dim>
KOKKOS_INLINE_FUNCTION edge_multiindex_t<dim>
to_edge_multiindex(Kokkos::Array<int32_t, dim> const & ijk, int32_t direction)
{
  edge_multiindex_t<dim> res;
  res[IX] = ijk[IX];
  res[IY] = ijk[IY];
  if constexpr (dim == 3)
  {
    res[IZ] = ijk[IZ];
  }
  res[dim] = direction;

  return res;
} // to_edge_multiindex

/**
 * Create a shift to move along dir from one cell to another on the left.
 */
template <size_t dim, int dir>
KOKKOS_INLINE_FUNCTION constexpr decltype(auto)
get_shift_left()
{
  if constexpr (dim == 2)
  {
    if constexpr (dir == IX)
      return shift_t<2>{ -1, 0 };
    else
      return shift_t<2>{ 0, -1 };
  }
  else
  {
    if constexpr (dir == IX)
      return shift_t<3>{ -1, 0, 0 };
    else if constexpr (dir == IY)
      return shift_t<3>{ 0, -1, 0 };
    else
      return shift_t<3>{ 0, 0, -1 };
  }
};

/**
 * Create a shift to move along dir from one cell to another on the right.
 */
template <size_t dim, int dir>
KOKKOS_INLINE_FUNCTION constexpr decltype(auto)
get_shift_right()
{
  if constexpr (dim == 2)
  {
    if constexpr (dir == IX)
      return shift_t<2>{ 1, 0 };
    else
      return shift_t<2>{ 0, 1 };
  }
  else
  {
    if constexpr (dir == IX)
      return shift_t<3>{ 1, 0, 0 };
    else if constexpr (dir == IY)
      return shift_t<3>{ 0, 1, 0 };
    else
      return shift_t<3>{ 0, 0, 1 };
  }
};

// =======================================================
// =======================================================
/**
 * return arg of min of 3 values.
 */
template <typename T>
KOKKOS_INLINE_FUNCTION int
ARG_MIN2(T a0, T a1)
{

  T   minval = a0;
  int arg_min = IX;

  if (a1 < minval)
  {
    minval = a1;
    arg_min = IY;
  }

  return arg_min;

} // ARG_MIN2

// =======================================================
// =======================================================
/**
 * return arg of min of 3 values.
 */
template <typename T>
KOKKOS_INLINE_FUNCTION int
ARG_MIN3(T a0, T a1, T a2)
{

  T   minval = a0;
  int arg_min = IX;

  if (a1 < minval)
  {
    minval = a1;
    arg_min = IY;
  }

  if (a2 < minval)
  {
    minval = a2;
    arg_min = IZ;
  }

  return arg_min;

} // ARG_MIN3

// =======================================================
// =======================================================
template <size_t dim>
KOKKOS_INLINE_FUNCTION auto
max_flux_block_sizes(block_size_t<dim> const & bSizes)
{
  block_size_t<dim> res = bSizes;


  if constexpr (dim == 2)
  {
    const auto dir = ARG_MIN2(res[IX], res[IY]);
    res[dir]++;
  }
  else if constexpr (dim == 3)
  {
    const auto dir = ARG_MIN3(res[IX], res[IY], res[IZ]);
    res[dir]++;
  }

  return res;

} // max_flux_block_sizes

// =======================================================
// =======================================================
// determine local position of current cell inside virtual parent cell using integer
//  coordinates in {0, 1}
//
// --------------------------
// |           |            |
// |           |            |
// |   0,1     |    1,1     |
// |           |            |
// |___________|____________|
// |           |            |
// |           |            |
// |   0,0     |    1,0     |
// |           |            |
// |___________|____________|
template <size_t dim>
KOKKOS_INLINE_FUNCTION auto
get_local_position(coord_t<dim> const & coordinates)
{
  if constexpr (dim == 2)
  {
    return Kokkos::Array<int, 2>{
      // clang-format off
      coordinates[IX] - 2 * (coordinates[IX] / 2),
      coordinates[IY] - 2 * (coordinates[IY] / 2)
      // clang-format on
    };
  }
  else if constexpr (dim == 3)
  {
    return Kokkos::Array<int, 3>{
      // clang-format off
      coordinates[IX] - 2 * (coordinates[IX] / 2),
      coordinates[IY] - 2 * (coordinates[IY] / 2),
      coordinates[IZ] - 2 * (coordinates[IZ] / 2)
      // clang-format on
    };
  }
} // get_local_position

// =======================================================
// =======================================================
// determine local position of current cell inside virtual parent cell using integer
//  coordinates in {-1, 1}
//
// --------------------------
// |           |            |
// |           |            |
// |   -1,1    |    1,1     |
// |           |            |
// |___________|____________|
// |           |            |
// |           |            |
// |   -1,-1   |    1,-1    |
// |           |            |
// |___________|____________|
template <size_t dim>
KOKKOS_INLINE_FUNCTION auto
get_local_position2(coord_t<dim> const & coordinates)
{
  if constexpr (dim == 2)
  {
    return Kokkos::Array<int, 2>{
      // clang-format off
      2 * (coordinates[IX] - 2 * (coordinates[IX] / 2)) - 1,
      2 * (coordinates[IY] - 2 * (coordinates[IY] / 2)) - 1
      // clang-format on
    };
  }
  else if constexpr (dim == 3)
  {
    return Kokkos::Array<int, 3>{
      // clang-format off
      2 * (coordinates[IX] - 2 * (coordinates[IX] / 2)) - 1,
      2 * (coordinates[IY] - 2 * (coordinates[IY] / 2)) - 1,
      2 * (coordinates[IZ] - 2 * (coordinates[IZ] / 2)) - 1
      // clang-format on
    };
  }
} // get_local_position2

} // namespace kalypsso

#endif // KALYPSSO_CORE_UTILS_BLOCK_H_
