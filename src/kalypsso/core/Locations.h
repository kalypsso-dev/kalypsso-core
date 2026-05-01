// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file Locations.h
 *
 * Define some base structures used in StencilHelper.
 */
#ifndef KALYPSSO_CORE_LOCATIONS_H_
#define KALYPSSO_CORE_LOCATIONS_H_

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/kokkos_shared.h>

#include <kalypsso/core/orchard_key.h>
#include <kalypsso/core/orchard_key_utils.h>
#include <kalypsso/core/orchard_key_base.h> // for definition of iOct_t
#include <kalypsso/core/utils_block.h> // for definition of function cellindex_to_coord and coord_to_cellindex
#include <kalypsso/core/enums.h>

namespace kalypsso
{

// ====================================================================================
// ====================================================================================
// ====================================================================================
/**
 * \class CellLocation
 *
 * \tparam dim dimension
 *
 * This class is designed to hold all geometrical information about a cell inside a block (aka
 * octant).
 */
template <size_t dim>
struct CellLocation
{
  //! cell coord inside block
  coord_t<dim> ijk;

  //! orchard key of block
  uint64_t key;

  //! octant id (so that we avoid an extra hashmap lookup when calling StencilHelper::getNeighLoc
  //! in case the neighbor is in the same octant as current cell).
  iOct_t iOct;

  //! when computing location of a cell outside domain, this is true.
  bool is_outside_domain;

  KOKKOS_INLINE_FUNCTION
  decltype(auto)
  level() const
  {
    return orchard_key_t<dim>::level(key);
  } // level

  KOKKOS_INLINE_FUNCTION auto
  cellindex(block_size_t<dim> const & bSize) const
  {
    KOKKOS_ASSERT((ijk[IX] < bSize[IX]) &&
                  "StencilHelper: coordinate and block size are incompatible");
    KOKKOS_ASSERT((ijk[IY] < bSize[IY]) &&
                  "StencilHelper: coordinate and block size are incompatible");

    if constexpr (dim == 3)
    {
      KOKKOS_ASSERT((ijk[IZ] < bSize[IZ]) &&
                    "StencilHelper: coordinate and block size are incompatible");
    }


    if constexpr (dim == 2)
    {
      return ijk[IX] + bSize[IX] * ijk[IY];
    }
    else if constexpr (dim == 3)
    {
      return ijk[IX] + bSize[IX] * (ijk[IY] + bSize[IY] * ijk[IZ]);
    }
  } // cellindex

  /**
   * return true when current location correspond to the lower left corner of a group of siblings.
   *
   * This is true if all coordinates are even.
   */
  KOKKOS_INLINE_FUNCTION bool
  is_eldest_sibling() const
  {
    bool res = true;
    for (uint8_t dir = 0; dir < dim; ++dir)
    {
      res = res and (ijk[dir] & 0x1) == 0;
    }
    return res;
  } // is_eldest_sibling

}; // struct CellLocation

// ====================================================================================
// ====================================================================================
// ====================================================================================
/**
 * \class FaceLocation
 *
 * \tparam dim dimension
 *
 * This class is designed to hold all geometrical information about a face inside a block (aka
 * octant).
 */
template <size_t dim>
struct FaceLocation
{
  //! face coordinate inside block
  //! the last index (i.e. ijk[dim]) is face normal direction
  face_multiindex_t<dim> ijk;

  //! orchard key of block
  uint64_t key;

  //! octant id (so that we avoid an extra hashmap lookup when calling StencilHelper::getNeighLoc
  //! in case the neighbor face is in the same octant as current face).
  iOct_t iOct;

  //! when computing location of a cell outside domain, this is true.
  bool is_outside_domain;

  KOKKOS_DEFAULTED_FUNCTION
  FaceLocation() = default;

  KOKKOS_DEFAULTED_FUNCTION
  FaceLocation(FaceLocation const & other) = default;

  KOKKOS_INLINE_FUNCTION
  FaceLocation(face_multiindex_t<dim> _ijk, uint64_t _key, iOct_t _iOct, bool _is_outside_domain)
    : ijk(_ijk)
    , key(_key)
    , iOct(_iOct)
    , is_outside_domain(_is_outside_domain)
  {}

  KOKKOS_DEFAULTED_FUNCTION
  FaceLocation &
  operator=(FaceLocation const & other) = default;

  KOKKOS_INLINE_FUNCTION
  decltype(auto)
  level() const
  {
    return orchard_key_t<dim>::level(key);
  } // level

  KOKKOS_INLINE_FUNCTION
  bool
  is_valid(block_size_t<dim> const & bSize) const
  {
    auto const & face_dir = ijk[dim];

    if constexpr (dim == 2)
    {
      auto const & i = ijk[IX];
      auto const & j = ijk[IY];

      if (face_dir == IX)
      {
        return (i >= 0 and i <= bSize[IX]) and (j >= 0 and j < bSize[IY]);
      }
      else if (face_dir == IY)
      {
        return (j >= 0 and j <= bSize[IY]) and (i >= 0 and i < bSize[IX]);
      }
    }
    else if constexpr (dim == 3)
    {
      auto const & i = ijk[IX];
      auto const & j = ijk[IY];
      auto const & k = ijk[IZ];

      // clang-format off
      if (face_dir == IX)
      {
        return (i >= 0 and i <= bSize[IX]) and (j >= 0 and j < bSize[IY]) and (k >= 0 and k < bSize[IZ]);
      }
      else if (face_dir == IY)
      {
        return (j >= 0 and j <= bSize[IY]) and (k >= 0 and k < bSize[IZ]) and (i >= 0 and i < bSize[IX]);
      }
      else if (face_dir == IZ)
      {
        return (k >= 0 and k <= bSize[IZ]) and (i >= 0 and i < bSize[IX]) and (j >= 0 and j < bSize[IY]);
      }
      // clang-format on
    }
  } // is_valid

}; // struct FaceLocation

// ====================================================================================
// ====================================================================================
// ====================================================================================
/**
 * \class EdgeLocation
 *
 * \tparam dim dimension
 *
 * This class is designed to hold all geometrical information about an edge inside a block (aka
 * octant).
 */
template <size_t dim>
struct EdgeLocation
{
  //! edge coordinate inside block
  //! the last index (i.e. ijk[dim]) is edge direction
  edge_multiindex_t<dim> ijk;

  //! orchard key of block
  uint64_t key;

  //! octant id (so that we avoid an extra hashmap lookup when calling StencilHelper::getNeighLoc
  //! in case the neighbor face is in the same octant as current face).
  iOct_t iOct;

  //! when computing location of a cell outside domain, this is true.
  bool is_outside_domain;

  //! it is always true, except when the edge location is obtained from getEdgeSibling at a hanging
  //! edge (touching a coarser neighbor through the middle of a face)
  bool is_valid = true;

  KOKKOS_DEFAULTED_FUNCTION
  EdgeLocation() = default;

  KOKKOS_DEFAULTED_FUNCTION
  EdgeLocation(EdgeLocation const & other) = default;

  KOKKOS_INLINE_FUNCTION
  EdgeLocation(edge_multiindex_t<dim> _ijk,
               uint64_t               _key,
               iOct_t                 _iOct,
               bool                   _is_outside_domain,
               bool                   _is_valid)
    : ijk(_ijk)
    , key(_key)
    , iOct(_iOct)
    , is_outside_domain(_is_outside_domain)
    , is_valid(_is_valid)
  {}

  KOKKOS_INLINE_FUNCTION
  EdgeLocation(edge_multiindex_t<dim> _ijk, uint64_t _key, iOct_t _iOct, bool _is_outside_domain)
    : ijk(_ijk)
    , key(_key)
    , iOct(_iOct)
    , is_outside_domain(_is_outside_domain)
    , is_valid(true)
  {}

  KOKKOS_DEFAULTED_FUNCTION
  EdgeLocation &
  operator=(EdgeLocation const & other) = default;

  KOKKOS_INLINE_FUNCTION
  decltype(auto)
  level() const
  {
    return orchard_key_t<dim>::level(key);
  } // level

}; // struct EdgeLocation

} // namespace kalypsso

#endif // KALYPSSO_CORE_LOCATIONS_H_
