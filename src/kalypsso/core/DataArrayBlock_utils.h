// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file DataArrayBlock_utils.h
 */
#ifndef KALYPSSO_CORE_DATAARRAYBLOCK_UTILS_H_
#define KALYPSSO_CORE_DATAARRAYBLOCK_UTILS_H_

#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp> // for KOKKOS_ENABLE_XXX

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/real_type.h>
#include <kalypsso/core/Kokkos_Array_extensions.h>

#include <kalypsso/core/utils_block.h> // for definition of type block_size_t and shift_t
#include <kalypsso/core/enums.h>

namespace kalypsso
{

// ============================================================================================
// ============================================================================================
/**
 * Compute the number of cells that are at surface of a block of cells in a given face.
 *
 * \param[in] bSize Sizes of a block of cells.
 *
 * \tparam[in] dim Dimension (2 or 3).
 * \tparam[in] dir Normal direction (identifying a face).
 *
 * \return the number of surface cells.
 */
template <size_t dim, ComponentIndex3D dir>
KOKKOS_FORCEINLINE_FUNCTION int32_t
get_number_of_cells_on_face(block_size_t<dim> const & bSize)
{
  if constexpr (dim == 2)
  {
    if constexpr (dir == IX)
      return bSize[IY];
    else if constexpr (dir == IY)
      return bSize[IX];
  }
  else if constexpr (dim == 3)
  {
    if constexpr (dir == IX)
    {
      return bSize[IY] * bSize[IZ];
    }
    else if constexpr (dir == IY)
    {
      return bSize[IZ] * bSize[IX];
    }
    else if constexpr (dir == IZ)
    {
      return bSize[IX] * bSize[IY];
    }
  }
} // get_number_of_cells_on_face

// ============================================================================================
// ============================================================================================
/**
 * Compute the number of cells that are at surface of a block of cells.
 *
 * \param[in] bSize Sizes of a block of cells.
 *
 * \tparam[in] dim Dimension (2 or 3).
 *
 * \return the number of surface cells.
 */
template <size_t dim>
KOKKOS_FORCEINLINE_FUNCTION int32_t
get_number_of_surface_cells(block_size_t<dim> const & bSize)
{
  if constexpr (dim == 2)
  {
    return 2 * (get_number_of_cells_on_face<dim, IX>(bSize) +
                get_number_of_cells_on_face<dim, IY>(bSize));
  }
  else if constexpr (dim == 3)
  {
    return 2 * (get_number_of_cells_on_face<dim, IX>(bSize) +
                get_number_of_cells_on_face<dim, IY>(bSize) +
                get_number_of_cells_on_face<dim, IZ>(bSize));
  }
} // get_number_of_surface_cells

// ============================================================================================
// ============================================================================================
/**
 * Unravel cell indexes from a flat index corresponding to a "surface" cell (i.e. a cell at the
 * border of a block of cells).
 *
 * \param[in] flat_index A surface cell flat index.
 * \param[in] bSize Sizes of a block of cells.
 *
 * \tparam[in] dim Dimension (2 or 3).
 *
 * \return coordinates of the surface cell identified by its flat index.
 */
template <size_t dim>
KOKKOS_FORCEINLINE_FUNCTION coord_t<dim, int32_t>
surface_flatindex_unravel_to_cell_ijk(int32_t flat_index, block_size_t<dim> const & bSize)
{
  KOKKOS_ASSERT(flat_index >= 0 and flat_index < get_number_of_surface_cells(bSize));

  coord_t<dim> res;

  if constexpr (dim == 2)
  {
    if (flat_index < 2 * get_number_of_cells_on_face<2, IX>(bSize))
    {
      bool is_left = flat_index < get_number_of_cells_on_face<2, IX>(bSize);

      res[IX] = is_left ? 0 : bSize[IX] - 1;
      res[IY] = is_left ? flat_index : flat_index - get_number_of_cells_on_face<2, IX>(bSize);
    }
    else
    {
      flat_index -= 2 * get_number_of_cells_on_face<2, IX>(bSize);

      bool is_left = flat_index < get_number_of_cells_on_face<2, IY>(bSize);

      res[IX] = is_left ? flat_index : flat_index - get_number_of_cells_on_face<2, IY>(bSize);
      res[IY] = is_left ? 0 : bSize[IY] - 1;
    }
  } // dim == 2
  else if constexpr (dim == 3)
  {
    if (flat_index < 2 * get_number_of_cells_on_face<3, IX>(bSize))
    {
      bool is_left = flat_index < get_number_of_cells_on_face<3, IX>(bSize);

      if (!is_left)
        flat_index -= get_number_of_cells_on_face<3, IX>(bSize);

      res[IZ] = flat_index / bSize[IY];
      res[IY] = flat_index - bSize[IY] * res[IZ];
      res[IX] = is_left ? 0 : bSize[IX] - 1;
    }
    else if (flat_index < 2 * (get_number_of_cells_on_face<3, IX>(bSize) +
                               get_number_of_cells_on_face<3, IY>(bSize)))
    {
      flat_index -= 2 * get_number_of_cells_on_face<3, IX>(bSize);

      bool is_left = flat_index < get_number_of_cells_on_face<3, IY>(bSize);

      if (!is_left)
        flat_index -= get_number_of_cells_on_face<3, IY>(bSize);

      res[IZ] = flat_index / bSize[IX];
      res[IX] = flat_index - bSize[IX] * res[IZ];
      res[IY] = is_left ? 0 : bSize[IY] - 1;
    }
    else
    {
      flat_index -=
        2 * (get_number_of_cells_on_face<3, IX>(bSize) + get_number_of_cells_on_face<3, IY>(bSize));

      bool is_left = flat_index < get_number_of_cells_on_face<3, IZ>(bSize);

      if (!is_left)
        flat_index -= get_number_of_cells_on_face<3, IZ>(bSize);

      res[IY] = flat_index / bSize[IX];
      res[IX] = flat_index - bSize[IX] * res[IY];
      res[IZ] = is_left ? 0 : bSize[IZ] - 1;
    }
  } // dim == 3

  return res;

} // surface_flatindex_unravel_to_cell_ijk

// ============================================================================================
// ============================================================================================
/**
 * Unravel cell indexes from a flat index corresponding to a "surface" cell (i.e. a cell at the
 * border of a block of cells).
 *
 * \param[in] flat_index A surface cell flat index.
 * \param[in] bSize Sizes of a block of cells.
 *
 * \tparam[in] dim Dimension (2 or 3).
 *
 * \return face multi-index identified by a given flat index.
 */
template <size_t dim>
KOKKOS_FORCEINLINE_FUNCTION face_multiindex_t<dim, int32_t>
surface_flatindex_unravel_to_face_ijk(int32_t flat_index, block_size_t<dim> const & bSize)
{
  KOKKOS_ASSERT(flat_index >= 0 and flat_index < get_number_of_surface_cells(bSize));

  face_multiindex_t<dim> res;

  if constexpr (dim == 2)
  {
    if (flat_index < 2 * get_number_of_cells_on_face<2, IX>(bSize))
    {
      bool is_left = flat_index < get_number_of_cells_on_face<2, IX>(bSize);

      res[IX] = is_left ? 0 : bSize[IX];
      res[IY] = is_left ? flat_index : flat_index - get_number_of_cells_on_face<2, IX>(bSize);
      res[dim] = IX;
    }
    else
    {
      flat_index -= 2 * get_number_of_cells_on_face<2, IX>(bSize);

      bool is_left = flat_index < get_number_of_cells_on_face<2, IY>(bSize);

      res[IX] = is_left ? flat_index : flat_index - get_number_of_cells_on_face<2, IY>(bSize);
      res[IY] = is_left ? 0 : bSize[IY];
      res[dim] = IY;
    }
  } // dim == 2
  else if constexpr (dim == 3)
  {
    if (flat_index < 2 * get_number_of_cells_on_face<3, IX>(bSize))
    {
      bool is_left = flat_index < get_number_of_cells_on_face<3, IX>(bSize);

      if (!is_left)
        flat_index -= get_number_of_cells_on_face<3, IX>(bSize);

      res[IZ] = flat_index / bSize[IY];
      res[IY] = flat_index - bSize[IY] * res[IZ];
      res[IX] = is_left ? 0 : bSize[IX];
      res[dim] = IX;
    }
    else if (flat_index < 2 * (get_number_of_cells_on_face<3, IX>(bSize) +
                               get_number_of_cells_on_face<3, IY>(bSize)))
    {
      flat_index -= 2 * get_number_of_cells_on_face<3, IX>(bSize);

      bool is_left = flat_index < get_number_of_cells_on_face<3, IY>(bSize);

      if (!is_left)
        flat_index -= get_number_of_cells_on_face<3, IY>(bSize);

      res[IZ] = flat_index / bSize[IX];
      res[IX] = flat_index - bSize[IX] * res[IZ];
      res[IY] = is_left ? 0 : bSize[IY];
      res[dim] = IY;
    }
    else
    {
      flat_index -=
        2 * (get_number_of_cells_on_face<3, IX>(bSize) + get_number_of_cells_on_face<3, IY>(bSize));

      bool is_left = flat_index < get_number_of_cells_on_face<3, IZ>(bSize);

      if (!is_left)
        flat_index -= get_number_of_cells_on_face<3, IZ>(bSize);

      res[IY] = flat_index / bSize[IX];
      res[IX] = flat_index - bSize[IX] * res[IY];
      res[IZ] = is_left ? 0 : bSize[IZ];
      res[dim] = IZ;
    }
  } // dim == 3

  return res;

} // surface_flatindex_unravel_to_face_ijk

// ============================================================================================
// ============================================================================================
/**
 * Compute face normal vector from a surface cell indexes.
 *
 * \param[in] flat_index A surface cell flat index.
 * \param[in] bSize Sizes of a block of cells.
 *
 * \tparam[in] dim Dimension (2 or 3).
 *
 * \return coordinates of the normal vector to a block of cells face  at a given cell identified by
 * its flat index.
 */
template <size_t dim>
KOKKOS_FORCEINLINE_FUNCTION coord_t<dim, int32_t>
surface_flatindex_to_normal_vector(int32_t flat_index, block_size_t<dim> const & bSize)
{
  KOKKOS_ASSERT(flat_index >= 0 and flat_index < get_number_of_surface_cells(bSize));

  coord_t<dim> res;
  res[IX] = 0;
  res[IY] = 0;
  if constexpr (dim == 3)
    res[IZ] = 0;

  if (flat_index < get_number_of_cells_on_face<dim, IX>(bSize))
  {
    res[IX] = -1;
  }
  else if (flat_index < 2 * get_number_of_cells_on_face<dim, IX>(bSize))
  {
    res[IX] = 1;
  }
  else if (flat_index < 2 * get_number_of_cells_on_face<dim, IX>(bSize) +
                          get_number_of_cells_on_face<dim, IY>(bSize))
  {
    res[IY] = -1;
  }
  else if (flat_index < 2 * get_number_of_cells_on_face<dim, IX>(bSize) +
                          2 * get_number_of_cells_on_face<dim, IY>(bSize))
  {
    res[IY] = 1;
  }
  else
  {
    if constexpr (dim == 3)
    {
      flat_index -= 2 * get_number_of_cells_on_face<dim, IX>(bSize) +
                    2 * get_number_of_cells_on_face<dim, IY>(bSize);

      if (flat_index < get_number_of_cells_on_face<dim, IZ>(bSize))
      {
        res[IZ] = -1;
      }
      else
      {
        res[IZ] = 1;
      }
    }
  }

  return res;

} // surface_flatindex_to_normal_vector

// ============================================================================================
// ============================================================================================
/**
 * Compute face normal vector from a surface cell indexes.
 *
 * \param[in] flat_index A surface cell flat index.
 * \param[in] bSize Sizes of a block of cells.
 *
 * \tparam[in] dim Dimension (2 or 3).
 *
 * \return coordinates of the normal vector to a block of cells face  at a given cell identified by
 * its flat index.
 */
template <size_t dim>
KOKKOS_FORCEINLINE_FUNCTION face_multiindex_t<dim, int32_t>
surface_flatindex_to_face_multiindex(int32_t flat_index, block_size_t<dim> const & bSize)
{
  KOKKOS_ASSERT(flat_index >= 0 and flat_index < get_number_of_surface_cells(bSize));

  const auto ijk = surface_flatindex_unravel_to_cell_ijk(flat_index, bSize);

  face_multiindex_t<dim> res;
  res[IX] = ijk[IX];
  res[IY] = ijk[IY];
  if constexpr (dim == 3)
    res[IZ] = ijk[IZ];
  res[dim] = IX; // default value to avoid uninitialized value warning (in 2d)

  if (flat_index < get_number_of_cells_on_face<dim, IX>(bSize))
  {
    res[dim] = IX;
  }
  else if (flat_index < 2 * get_number_of_cells_on_face<dim, IX>(bSize))
  {
    res[IX] += 1;
    res[dim] = IX;
  }
  else if (flat_index < 2 * get_number_of_cells_on_face<dim, IX>(bSize) +
                          get_number_of_cells_on_face<dim, IY>(bSize))
  {
    res[dim] = IY;
  }
  else if (flat_index < 2 * get_number_of_cells_on_face<dim, IX>(bSize) +
                          2 * get_number_of_cells_on_face<dim, IY>(bSize))
  {
    res[IY] += 1;
    res[dim] = IY;
  }
  else
  {
    if constexpr (dim == 3)
    {
      flat_index -= 2 * get_number_of_cells_on_face<dim, IX>(bSize) +
                    2 * get_number_of_cells_on_face<dim, IY>(bSize);

      if (flat_index < get_number_of_cells_on_face<dim, IZ>(bSize))
      {
        res[dim] = IZ;
      }
      else
      {
        res[IZ] += 1;
        res[dim] = IZ;
      }
    }
  }

  return res;

} // surface_flatindex_to_face_multiindex

} // namespace kalypsso

#endif // KALYPSSO_CORE_DATAARRAYBLOCK_UTILS_H_
