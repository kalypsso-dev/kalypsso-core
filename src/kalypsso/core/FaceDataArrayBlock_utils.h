// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file FaceDataArrayBlock_utils.h
 *
 * Some utility functions used by FaceDataArrayBlock
 */
#ifndef KALYPSSO_CORE_FACEDATAARRAYBLOCK_UTILS_H_
#define KALYPSSO_CORE_FACEDATAARRAYBLOCK_UTILS_H_

#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp> // for KOKKOS_ENABLE_XXX

#include <kalypsso/core/kalypsso_core_base.h> // for definition of assertm
#include <kalypsso/core/real_type.h>
#include <kalypsso/core/Kokkos_Array_extensions.h>

#include <kalypsso/core/utils_block.h> // for definition of type block_size_t
#include <kalypsso/core/enums.h>

#include <kalypsso/core/orchard_key.h>
#include <kalypsso/core/orchard_key_utils.h>

namespace kalypsso
{

/**
 * A type alias used to store index offset indicating where the different type of face (X, Y or Z
 * faces) start in a FaceDataArrayBlock.
 */
using face_flat_index_offset_t = Kokkos::Array<int32_t, 4>;

// ============================================================================================
// ============================================================================================
/**
 * For a given face direction, compute the logical face array sizes associated
 * to a block of cells of size bSize.
 *
 * \param[in] cell block size (number of cells per direction)
 *
 * \return block size (number of faces in direction "dir") per direction
 *
 * \tparam dim is dimension (2 or 3)
 * \tparam dir is the face direction
 */
template <size_t dim, int dir>
KOKKOS_FORCEINLINE_FUNCTION block_size_t<dim>
                            get_face_block_size(block_size_t<dim> bSize)
{
  block_size_t<dim> res = bSize;

  res[IX] += (dir == IX) ? 1 : 0;
  res[IY] += (dir == IY) ? 1 : 0;

  if constexpr (dim == 3)
  {
    res[IZ] += (dir == IZ) ? 1 : 0;
  }

  return res;
} // get_face_block_size

// ============================================================================================
// ============================================================================================
//! compute the maximum number of faces (among all direction) in a block of cell of size bSize.
//!
//! - when the block is square (equal dimension in all direction), the result is exact
//! - when the block is rectangular, the value is rounded up
//!
//! Purpose: allocate an array that can be addressed with the same i,j,k bounds for x,y,z faces
//!
//! \param[in] bSize cell block size (number of cells per direction)
template <size_t dim>
KOKKOS_FORCEINLINE_FUNCTION auto
maximum_number_faces_per_leaf(block_size_t<dim> bSize)
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
      return (bSize[IX] + 1) * bSize[IX] * bSize[IX];
    }
  }

  // general case : a rectangular block
  return dim == 2 ? (bSize[IX] + 1) * (bSize[IY] + 1)
                  : (bSize[IX] + 1) * (bSize[IY] + 1) * (bSize[IZ] + 1);

} // maximum_number_faces_per_leaf

// ============================================================================================
// ============================================================================================
/**
 * accumulate over direction the number of face elements in that direction per octant.
 *
 * \param[in] bSize cell block size (number of cells per direction)
 *
 */
template <size_t dim>
KOKKOS_FORCEINLINE_FUNCTION auto
compute_face_flat_index_offsets(block_size_t<dim> bSize) -> face_flat_index_offset_t
{
  face_flat_index_offset_t offset;

  offset[0] = 0;
  offset[1] = offset[0] + Kokkos::dim_prod(get_face_block_size<dim, IX>(bSize));
  offset[2] = offset[1] + Kokkos::dim_prod(get_face_block_size<dim, IY>(bSize));
  offset[3] = offset[2] + Kokkos::dim_prod(get_face_block_size<dim, IZ>(bSize));

  return offset;
} // compute_face_flat_index_offsets

// ============================================================================================
// ============================================================================================
/**
 * Compute total number of faces over all directions.
 *
 * \param[in] bSize cell block size (number of cells per direction)
 *
 */
template <size_t dim>
KOKKOS_FORCEINLINE_FUNCTION auto
compute_total_number_of_faces(block_size_t<dim> bSize)
{
  int32_t res = 0;

  res += Kokkos::dim_prod(get_face_block_size<dim, IX>(bSize));
  res += Kokkos::dim_prod(get_face_block_size<dim, IY>(bSize));
  res += Kokkos::dim_prod(get_face_block_size<dim, IZ>(bSize));

  return res;
} // compute_face_flat_index_offsets

// ============================================================================================
// ============================================================================================
/**
 * Decode i,j,k, face_type from a flat index (result are positive integers).
 *
 * \param[in] flat_index is a face index in range [0, face_flat_index_offset[3])
 * \param[in] block_size is cell block size
 * \param[in] face_flat_index_offsets array of cumulative number of faces over direction
 *
 * \return face_multiindex (i,j,k,face_type) corresponding to a given flat index
 */
template <size_t dim>
KOKKOS_FORCEINLINE_FUNCTION auto
face_flat_index_unravel_no_shift(int32_t                          flat_index,
                                 block_size_t<dim> const &        block_size,
                                 face_flat_index_offset_t const & face_flat_index_offsets)
  -> face_multiindex_t<dim>
{
  KOKKOS_ASSERT(flat_index >= 0 and "flat_index must be positive");
  KOKKOS_ASSERT(flat_index < face_flat_index_offsets[3] and "flat index invalid (too large) !");

  face_multiindex_t<dim> index;

  // determine face type
  index[dim] = flat_index < face_flat_index_offsets[1]   ? IX
               : flat_index < face_flat_index_offsets[2] ? IY
                                                         : IZ;

  if constexpr (dim == 2)
  {
    if (index[dim] == IX)
    {
      // flat_index = flat_index_offsets[0] + i + (block_size[IX]+1)*j
      // remember that flat_index_offsets[0] = 0 by design
      index[IY] = flat_index / (block_size[IX] + 1);
      index[IX] = flat_index - index[IY] * (block_size[IX] + 1);
    }
    else if (index[dim] == IY)
    {
      // flat_index = flat_index_offsets[1] + i + block_size[IX]*j
      flat_index -= face_flat_index_offsets[1];
      index[IY] = flat_index / (block_size[IX]);
      index[IX] = flat_index - index[IY] * (block_size[IX]);
    }
    else
    {
      // flat_index = flat_index_offsets[2] + i + block_size[IX] * j
      flat_index -= face_flat_index_offsets[2];
      index[IY] = flat_index / (block_size[IX]);
      index[IX] = flat_index - index[IY] * (block_size[IX]);
    }
  }
  else if constexpr (dim == 3)
  {
    if (index[dim] == IX)
    {
      // flat_index = face_flat_index_offsets[0] + i + (block_size[IX]+1) * j +
      // (block_size[IX]+1)*block_size[IY] * k
      // remember that flat_index_offsets[0] = 0 by design
      index[IZ] = flat_index / ((block_size[IX] + 1) * block_size[IY]);
      flat_index -= (block_size[IX] + 1) * block_size[IY] * index[IZ];
      index[IY] = flat_index / (block_size[IX] + 1);
      index[IX] = flat_index - index[IY] * (block_size[IX] + 1);
    }
    else if (index[dim] == IY)
    {
      // flat_index = flat_index_offsets[1] + i + block_size[IX]*j + block_size[IX] *
      // (block_size[IY]+1) * k
      flat_index -= face_flat_index_offsets[1];
      index[IZ] = flat_index / (block_size[IX] * (block_size[IY] + 1));
      flat_index -= block_size[IX] * (block_size[IY] + 1) * index[IZ];
      index[IY] = flat_index / block_size[IX];
      index[IX] = flat_index - index[IY] * (block_size[IX]);
    }
    else
    {
      // flat_index = flat_index_offsets[2] + i + block_size[IX]*j + block_size[IX] *
      // block_size[IY] * k
      flat_index -= face_flat_index_offsets[2];
      index[IZ] = flat_index / (block_size[IX] * block_size[IY]);
      flat_index -= block_size[IX] * block_size[IY] * index[IZ];
      index[IY] = flat_index / block_size[IX];
      index[IX] = flat_index - index[IY] * block_size[IX];
    }
  }

  return index;
} // face_flat_index_unravel_no_shift

// ============================================================================================
// ============================================================================================
/**
 * Decode i,j,k, face_type from a flat index (results are shifted).
 *
 * \param[in] flat_index is a face index in range [0, face_flat_index_offset[3])
 * \param[in] block_size is cell block size
 * \param[in] face_flat_index_offsets array of cumulative number of faces over direction
 * \param[in] shift
 *
 * \return face_multiindex (i,j,k,face_type) corresponding to a given flat index
 */
template <size_t dim>
KOKKOS_FORCEINLINE_FUNCTION auto
face_flat_index_unravel(int32_t                          flat_index,
                        block_size_t<dim> const &        block_size,
                        face_flat_index_offset_t const & face_flat_index_offsets,
                        coord_t<dim, int32_t> const      shift) -> face_multiindex_t<dim>
{
  auto res = face_flat_index_unravel_no_shift(flat_index, block_size, face_flat_index_offsets);

  res[IX] += shift[IX];
  res[IY] += shift[IY];
  if constexpr (dim == 3)
  {
    res[IZ] += shift[IZ];
  }
  return res;
}

// ==============================================================
// ==============================================================
/**
 * face coordinates to cell coordinates.
 *
 * \param[in] face_indexes a face multi-index (i,j,k,face_type)
 * \param[in] block_size is cell block size
 *
 * \return cell multi-index (i,j,k)
 */
template <size_t dim>
KOKKOS_INLINE_FUNCTION coord_t<dim>
face_to_cell_coords(face_multiindex_t<dim> const & face_indexes, block_size_t<dim> const & bSizes)
{
  if constexpr (dim == 2)
  {
    // clang-format off
    const coord_t<2> ijk{ face_indexes[IX] == bSizes[IX] ? face_indexes[IX] - 1 : face_indexes[IX],
                          face_indexes[IY] == bSizes[IY] ? face_indexes[IY] - 1 : face_indexes[IY] };
    // clang-format on
    return ijk;
  }
  else if constexpr (dim == 3)
  {
    // clang-format off
    const coord_t<3> ijk{ face_indexes[IX] == bSizes[IX] ? face_indexes[IX] - 1 : face_indexes[IX],
                          face_indexes[IY] == bSizes[IY] ? face_indexes[IY] - 1 : face_indexes[IY],
                          face_indexes[IZ] == bSizes[IZ] ? face_indexes[IZ] - 1 : face_indexes[IZ] };
    // clang-format on
    return ijk;
  }
} // face_to_cell_coords

} // namespace kalypsso

#endif // KALYPSSO_CORE_FACEDATAARRAYBLOCK_UTILS_H_
