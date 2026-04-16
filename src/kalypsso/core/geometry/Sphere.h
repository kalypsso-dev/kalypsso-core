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
 * \file Sphere.h
 */
#ifndef KALYPSSO_CORE_GEOMETRY_SPHERE_H_
#define KALYPSSO_CORE_GEOMETRY_SPHERE_H_

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/real_type.h> // for math functions (max, min, ...)

#include <kalypsso/core/geometry/GeometryTraits.h>
#include <kalypsso/core/geometry/Point.h>

#include <type_traits>

namespace kalypsso
{

namespace core
{

namespace geometry
{

template <int DIM, class Coordinate = real_t>
struct Sphere
{
  KOKKOS_DEFAULTED_FUNCTION
  Sphere() = default;

  KOKKOS_FUNCTION
  constexpr Sphere(Point<DIM, Coordinate> const & centroid, Coordinate radius)
    : _centroid(centroid)
    , _radius(radius)
  {}

  KOKKOS_FUNCTION
  constexpr auto &
  centroid()
  {
    return _centroid;
  }

  KOKKOS_FUNCTION
  constexpr auto const &
  centroid() const
  {
    return _centroid;
  }

  KOKKOS_FUNCTION
  constexpr auto
  radius() const
  {
    return _radius;
  }

  Point<DIM, Coordinate> _centroid = {};
  Coordinate             _radius = 0;
}; // struct Sphere

template <typename T, std::size_t N>
KOKKOS_DEDUCTION_GUIDE
Sphere(T const (&)[N], T) -> Sphere<N, T>;

} // namespace geometry

template <int DIM, class Coordinate>
struct GeometryTraits::dimension<geometry::Sphere<DIM, Coordinate>>
{
  static constexpr int value = DIM;
};
template <int DIM, class Coordinate>
struct GeometryTraits::tag<geometry::Sphere<DIM, Coordinate>>
{
  using type = SphereTag;
};
template <int DIM, class Coordinate>
struct GeometryTraits::coordinate_type<geometry::Sphere<DIM, Coordinate>>
{
  using type = Coordinate;
};

} // namespace core

} // namespace kalypsso

#endif // KALYPSSO_CORE_GEOMETRY_SPHERE_H_
