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
 * \file Point.h
 */
#ifndef KALYPSSO_CORE_GEOMETRY_POINT_H_
#define KALYPSSO_CORE_GEOMETRY_POINT_H_

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/real_type.h> // for math functions (max, min, ...)

#include <kalypsso/core/geometry/GeometryTraits.h>

namespace kalypsso
{

namespace core
{

namespace geometry
{

/**
 * \struct Point
 *
 * \brief The Point class represents a point, \f$ P \in \mathcal{R}^d \f$ .
 *
 * \tparam dim Dimension (2 or 3)
 * \tparam value The point coordinates type (should be a floating point type: float, double, ...).
 */
template <int DIM, typename Coordinate = real_t>
struct Point
{

  // a valid point must have at least one dimension
  static_assert(DIM > 0);

  // should we enforce floating point value ?
  // static_assert(!std::is_floating_point_v<Coordinate>);

  KOKKOS_FUNCTION
  constexpr auto &
  operator[](int i)
  {
    KOKKOS_ASSERT(i >= 0 and i < DIM);
    return _coords[i];
  }

  KOKKOS_FUNCTION
  constexpr auto const &
  operator[](int i) const
  {
    KOKKOS_ASSERT(i >= 0 and i < DIM);
    return _coords[i];
  }

  KOKKOS_INLINE_FUNCTION
  static constexpr size_t
  dimension()
  {
    return DIM;
  };

  /**
   * \brief Returns the midpoint between point A and point B.
   *
   * \param [in] A user-supplied point
   * \param [in] B user-supplied point
   * \return p point at the midpoint A and B.
   */
  KOKKOS_FUNCTION
  static Point
  midpoint(const Point & A, const Point & B);

  /**
   * \brief Perform linear interpolation between point A and point B.
   *
   * \param [in] A user-supplied point
   * \param [in] B user-supplied point
   * \param [in] alpha user-supplied scalar weight \f$ \alpha\f$
   * \return p interpolated point.
   */
  KOKKOS_FUNCTION
  static Point
  lerp(const Point & A, const Point & B, Coordinate alpha);

  //! Point coordinates
  Coordinate _coords[DIM] = {};

}; // struct Point

// CTAD so that dimension can be deduced from the number of arguments passed to constructor
// example use: auto point = Point{1.0, 2.0};
template <typename T, typename... Ts>
KOKKOS_DEDUCTION_GUIDE
Point(T, Ts...)
  -> Point<sizeof...(Ts) + 1, std::enable_if_t<std::conjunction_v<std::is_same<T, Ts>...>, T>>;

// CTAD to initialize using a std::initializer_list
// example use: auto point = Point{{1.0, 2.0}};
template <typename T, std::size_t N>
KOKKOS_DEDUCTION_GUIDE
Point(T const (&)[N]) -> Point<N, std::enable_if_t<std::is_floating_point_v<T>, T>>;

// =======================================================================================
// =======================================================================================
template <int DIM, typename Coordinate>
KOKKOS_FUNCTION Point<DIM, Coordinate>
Point<DIM, Coordinate>::midpoint(const Point<DIM, Coordinate> & A, const Point<DIM, Coordinate> & B)
{
  Point<DIM, Coordinate> mid_point;

  for (size_t i = 0; i < DIM; ++i)
  {
    mid_point[i] = static_cast<Coordinate>(0.5 * (A[i] + B[i]));
  }

  return mid_point;

} // Point::midpoint

// =======================================================================================
// =======================================================================================
template <int DIM, typename Coordinate>
KOKKOS_FUNCTION Point<DIM, Coordinate>
                Point<DIM, Coordinate>::lerp(const Point<DIM, Coordinate> & A,
                             const Point<DIM, Coordinate> & B,
                             Coordinate                     alpha)
{
  Point<DIM, Coordinate> res;

  const Coordinate beta = 1. - alpha;
  for (int i = 0; i < DIM; ++i)
  {
    res[i] = beta * A[i] + alpha * B[i];
  }

  return res;

} // Point::lerp

} // namespace geometry

template <int DIM, class Coordinate>
struct GeometryTraits::dimension<geometry::Point<DIM, Coordinate>>
{
  static constexpr int value = DIM;
};
template <int DIM, class Coordinate>
struct GeometryTraits::tag<geometry::Point<DIM, Coordinate>>
{
  using type = PointTag;
};
template <int DIM, class Coordinate>
struct GeometryTraits::coordinate_type<geometry::Point<DIM, Coordinate>>
{
  using type = Coordinate;
};

} // namespace core

} // namespace kalypsso

#endif // KALYPSSO_CORE_GEOMETRY_POINT_H_
