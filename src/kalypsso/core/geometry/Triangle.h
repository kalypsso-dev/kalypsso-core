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
 * \file Triangle.h
 */
#ifndef KALYPSSO_CORE_GEOMETRY_TRIANGLE_H_
#define KALYPSSO_CORE_GEOMETRY_TRIANGLE_H_

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/real_type.h> // for math functions (max, min, ...)

#include <kalypsso/core/geometry/GeometryTraits.h>
#include <kalypsso/core/geometry/Vector.h>

namespace kalypsso
{

namespace core
{

namespace geometry
{

// need to add a protection that
// the points are not on the same line.
template <int DIM, class Coordinate = real_t>
struct Triangle
{
  Point<DIM, Coordinate> a;
  Point<DIM, Coordinate> b;
  Point<DIM, Coordinate> c;

  /**
   * \brief Returns the normal of the triangle (not normalized)
   *
   * \return n triangle normal when DIM=3, zero vector otherwise
   */
  KOKKOS_FUNCTION
  Vector<DIM, Coordinate>
  normal() const
  {
    if constexpr (DIM == 3)
    {
      const auto v1 = Vector{ b[0] - a[0], b[1] - a[1], b[2] - a[2] };
      const auto v2 = Vector{ c[0] - a[0], c[1] - a[1], c[2] - a[2] };
      return v2.cross(v1);
    }
  }

  /**
   * \return Signed area of a 2d triangle.
   */
  template <int DIM_ = DIM>
  typename std::enable_if<DIM_ == 2, Coordinate>::type
  signed_area() const
  {
    return HALF_F * ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]));
  }

  /**
   * \return Area of triangle
   */
  KOKKOS_FUNCTION Coordinate
  area() const
  {
    if constexpr (DIM == 2)
    {
      return fabs(signed_area());
    }
    else if constexpr (DIM == 3)
    {
      return HALF_F * normal().norm();
    }
  }

}; // struct Triangle

// CTAD
template <int DIM, class Coordinate>
KOKKOS_DEDUCTION_GUIDE Triangle(Point<DIM, Coordinate>,
                                Point<DIM, Coordinate>,
                                Point<DIM, Coordinate>) -> Triangle<DIM, Coordinate>;

// CTAD
template <typename T, std::size_t N>
KOKKOS_DEDUCTION_GUIDE
Triangle(T const (&)[N], T const (&)[N], T const (&)[N]) -> Triangle<N, T>;

} // namespace geometry

template <int DIM, class Coordinate>
struct GeometryTraits::dimension<geometry::Triangle<DIM, Coordinate>>
{
  static constexpr int value = DIM;
};
template <int DIM, class Coordinate>
struct GeometryTraits::tag<geometry::Triangle<DIM, Coordinate>>
{
  using type = TriangleTag;
};
template <int DIM, class Coordinate>
struct GeometryTraits::coordinate_type<geometry::Triangle<DIM, Coordinate>>
{
  using type = Coordinate;
};

} // namespace core

} // namespace kalypsso

#endif // KALYPSSO_CORE_GEOMETRY_TRIANGLE_H_
