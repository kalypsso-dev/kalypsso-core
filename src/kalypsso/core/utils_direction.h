// SPDX-FileCopyrightText: 2026 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file utils_direction.h
 * \brief Define some useful utilities for computing conformal status.
 */
#ifndef KALYPSSO_CORE_UTILS_DIRECTION_H_
#define KALYPSSO_CORE_UTILS_DIRECTION_H_

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/mesh_utils.h> // for definition of Face::XMIN, etc...

namespace kalypsso
{

namespace core
{

// ==============================================================
// ==============================================================
//! \brief get outside normal direction to a face
//!
//! \param[in] face
template <size_t dim>
KOKKOS_INLINE_FUNCTION auto
get_direction(Face::face_t face)
{
  if constexpr (dim == 2)
  {
    if (face == Face::XMIN)
    {
      return Kokkos::Array<int8_t, 2>{ -1, 0 };
    }
    else if (face == Face::XMAX)
    {
      return Kokkos::Array<int8_t, 2>{ 1, 0 };
    }
    else if (face == Face::YMIN)
    {
      return Kokkos::Array<int8_t, 2>{ 0, -1 };
    }
    else if (face == Face::YMAX)
    {
      return Kokkos::Array<int8_t, 2>{ 0, 1 };
    }
    else
    {
      return Kokkos::Array<int8_t, 2>{ 0, 0 };
    }
  }
  else if constexpr (dim == 3)
  {
    if (face == Face::XMIN)
    {
      return Kokkos::Array<int8_t, 3>{ -1, 0, 0 };
    }
    else if (face == Face::XMAX)
    {
      return Kokkos::Array<int8_t, 3>{ 1, 0, 0 };
    }
    else if (face == Face::YMIN)
    {
      return Kokkos::Array<int8_t, 3>{ 0, -1, 0 };
    }
    else if (face == Face::YMAX)
    {
      return Kokkos::Array<int8_t, 3>{ 0, 1, 0 };
    }
    else if (face == Face::ZMIN)
    {
      return Kokkos::Array<int8_t, 3>{ 0, 0, -1 };
    }
    else if (face == Face::ZMAX)
    {
      return Kokkos::Array<int8_t, 3>{ 0, 0, 1 };
    }
    else
    {
      return Kokkos::Array<int8_t, 3>{ 0, 0, 0 };
    }
  }
} // get_direction - face normal

// ==============================================================
// ==============================================================
//! \brief get outside normal direction to an edge.
//!
//! An edge is identified by the intersection of two faces.
//!
//! \param[in] face0
//! \param[in] face1
template <size_t dim>
KOKKOS_INLINE_FUNCTION auto
get_direction(Face::face_t face0, Face::face_t face1)
{
  if constexpr (dim == 2)
  {
    if (face0 == Face::XMIN and face1 == Face::YMIN)
    {
      return Kokkos::Array<int8_t, 2>{ -1, -1 };
    }
    else if (face0 == Face::XMAX and face1 == Face::YMIN)
    {
      return Kokkos::Array<int8_t, 2>{ 1, -1 };
    }
    else if (face0 == Face::XMIN and face1 == Face::YMAX)
    {
      return Kokkos::Array<int8_t, 2>{ -1, 1 };
    }
    else if (face0 == Face::XMAX and face1 == Face::YMAX)
    {
      return Kokkos::Array<int8_t, 2>{ 1, 1 };
    }
    else
    {
      return Kokkos::Array<int8_t, 2>{ 0, 0 };
    }
  }
  else if constexpr (dim == 3)
  {
    if (face0 == Face::XMIN and face1 == Face::YMIN)
    {
      return Kokkos::Array<int8_t, 3>{ -1, -1, 0 };
    }
    else if (face0 == Face::XMAX and face1 == Face::YMIN)
    {
      return Kokkos::Array<int8_t, 3>{ 1, -1, 0 };
    }
    else if (face0 == Face::XMIN and face1 == Face::YMAX)
    {
      return Kokkos::Array<int8_t, 3>{ -1, 1, 0 };
    }
    else if (face0 == Face::XMAX and face1 == Face::YMAX)
    {
      return Kokkos::Array<int8_t, 3>{ 1, 1, 0 };
    }
    else if (face0 == Face::YMIN and face1 == Face::ZMIN)
    {
      return Kokkos::Array<int8_t, 3>{ 0, -1, -1 };
    }
    else if (face0 == Face::YMAX and face1 == Face::ZMIN)
    {
      return Kokkos::Array<int8_t, 3>{ 0, 1, -1 };
    }
    else if (face0 == Face::YMIN and face1 == Face::ZMAX)
    {
      return Kokkos::Array<int8_t, 3>{ 0, -1, 1 };
    }
    else if (face0 == Face::YMAX and face1 == Face::ZMAX)
    {
      return Kokkos::Array<int8_t, 3>{ 0, 1, 1 };
    }
    else if (face0 == Face::XMIN and face1 == Face::ZMIN)
    {
      return Kokkos::Array<int8_t, 3>{ -1, 0, -1 };
    }
    else if (face0 == Face::XMAX and face1 == Face::ZMIN)
    {
      return Kokkos::Array<int8_t, 3>{ 1, 0, -1 };
    }
    else if (face0 == Face::XMIN and face1 == Face::ZMAX)
    {
      return Kokkos::Array<int8_t, 3>{ -1, 0, 1 };
    }
    else if (face0 == Face::XMAX and face1 == Face::ZMAX)
    {
      return Kokkos::Array<int8_t, 3>{ 1, 0, 1 };
    }
    else
    {
      return Kokkos::Array<int8_t, 3>{ 0, 0, 0 };
    }
  }
} // get_direction - edge normal

// ==============================================================
// ==============================================================
//! \brief get outside normal direction to a corner (3d).
//!
//! A corner in 3d is identified by the intersection of three faces.
//!
//! \param[in] face0
//! \param[in] face1
//! \param[in] face2
template <size_t dim>
KOKKOS_INLINE_FUNCTION auto
get_direction(Face::face_t face0, Face::face_t face1, Face::face_t face2)
{
  if constexpr (dim == 3)
  {
    if (face0 == Face::XMIN and face1 == Face::YMIN and face2 == Face::ZMIN)
    {
      return Kokkos::Array<int8_t, 3>{ -1, -1, -1 };
    }
    else if (face0 == Face::XMAX and face1 == Face::YMIN and face2 == Face::ZMIN)
    {
      return Kokkos::Array<int8_t, 3>{ 1, -1, -1 };
    }
    else if (face0 == Face::XMIN and face1 == Face::YMAX and face2 == Face::ZMIN)
    {
      return Kokkos::Array<int8_t, 3>{ -1, 1, -1 };
    }
    else if (face0 == Face::XMAX and face1 == Face::YMAX and face2 == Face::ZMIN)
    {
      return Kokkos::Array<int8_t, 3>{ 1, 1, -1 };
    }
    else if (face0 == Face::XMIN and face1 == Face::YMIN and face2 == Face::ZMAX)
    {
      return Kokkos::Array<int8_t, 3>{ -1, -1, 1 };
    }
    else if (face0 == Face::XMAX and face1 == Face::YMIN and face2 == Face::ZMAX)
    {
      return Kokkos::Array<int8_t, 3>{ 1, -1, 1 };
    }
    else if (face0 == Face::XMIN and face1 == Face::YMAX and face2 == Face::ZMAX)
    {
      return Kokkos::Array<int8_t, 3>{ -1, 1, 1 };
    }
    else if (face0 == Face::XMAX and face1 == Face::YMAX and face2 == Face::ZMAX)
    {
      return Kokkos::Array<int8_t, 3>{ 1, 1, 1 };
    }
    else
    {
      return Kokkos::Array<int8_t, 3>{ 0, 0, 0 };
    }
  }
} // get_direction - corner normal (3d only)

} // namespace core

} // namespace kalypsso

#endif // KALYPSSO_CORE_UTILS_DIRECTION_H_
