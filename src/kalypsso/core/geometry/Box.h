// SPDX-FileCopyrightText: 2026 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/*
 * The code below is based on ArborX : https://github.com/arborx/ArborX
 *
 * Copyright (c) 2025, ArborX authors
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * \file Box.h
 */
#ifndef KALYPSSO_CORE_GEOMETRY_BOX_H_
#define KALYPSSO_CORE_GEOMETRY_BOX_H_

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/real_type.h> // for math functions (max, min, ...)

#include <kalypsso/core/geometry/GeometryTraits.h>
#include <kalypsso/core/geometry/Point.h>
#include <kalypsso/core/geometry/kokkos_ext/KokkosExtArithmeticTraits.hpp>

#include <Kokkos_Macros.hpp>
#include <Kokkos_ReductionIdentity.hpp>

namespace kalypsso
{

namespace core
{

namespace geometry
{

/**
 * Axis-Aligned Bounding Box. This is just a thin wrapper around an array of
 * size 2x spatial dimension with a default constructor to initialize
 * properly an "empty" box.
 */
template <int DIM, class Coordinate = real_t>
struct Box
{
  KOKKOS_FUNCTION
  constexpr Box() { reset(); }

  KOKKOS_FUNCTION
  constexpr Box(Point<DIM, Coordinate> const & min_corner,
                Point<DIM, Coordinate> const & max_corner)
    : _min_corner(min_corner)
    , _max_corner(max_corner)
  {}

  KOKKOS_FUNCTION
  constexpr void
  reset()
  {
    for (int d = 0; d < DIM; ++d)
    {
      _min_corner[d] = Details::KokkosExt::ArithmeticTraits::finite_max<Coordinate>::value;
      _max_corner[d] = Details::KokkosExt::ArithmeticTraits::finite_min<Coordinate>::value;
    }
  }

  KOKKOS_FUNCTION
  constexpr auto &
  minCorner()
  {
    return _min_corner;
  }

  KOKKOS_FUNCTION
  constexpr auto const &
  minCorner() const
  {
    return _min_corner;
  }

  KOKKOS_FUNCTION
  constexpr auto &
  maxCorner()
  {
    return _max_corner;
  }

  KOKKOS_FUNCTION
  constexpr auto const &
  maxCorner() const
  {
    return _max_corner;
  }

  //! Update/extend bounding-box to include a given point
  KOKKOS_FUNCTION
  constexpr void
  addPoint(Point<DIM, Coordinate> const & point)
  {
    for (int d = 0; d < DIM; ++d)
    {
      if (point[d] < _min_corner[d])
      {
        _min_corner[d] = point[d];
      }

      if (point[d] > _max_corner[d])
      {
        _max_corner[d] = point[d];
      }
    }
  } // addPoint

  friend KOKKOS_FUNCTION constexpr bool
  operator==(Point<DIM, Coordinate> const & a, Point<DIM, Coordinate> const & b)
  {
    bool match = true;
    for (int d = 0; d < DIM; ++d)
      match &= (a[d] == b[d]);
    return match;
  }

  Point<DIM, Coordinate> _min_corner;
  Point<DIM, Coordinate> _max_corner;
}; // struct Box

template <typename T, std::size_t N>
KOKKOS_DEDUCTION_GUIDE
Box(T const (&)[N], T const (&)[N]) -> Box<N, T>;

} // namespace geometry

template <int DIM, class Coordinate>
struct GeometryTraits::dimension<geometry::Box<DIM, Coordinate>>
{
  static constexpr int value = DIM;
};
template <int DIM, class Coordinate>
struct GeometryTraits::tag<geometry::Box<DIM, Coordinate>>
{
  using type = BoxTag;
};
template <int DIM, class Coordinate>
struct GeometryTraits::coordinate_type<geometry::Box<DIM, Coordinate>>
{
  using type = Coordinate;
};

} // namespace core

} // namespace kalypsso

#endif // KALYPSSO_CORE_GEOMETRY_BOX_H_
