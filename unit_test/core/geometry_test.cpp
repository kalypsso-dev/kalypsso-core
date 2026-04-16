// SPDX-FileCopyrightText: 2025 kalypsso authors
// © Commissariat a l'Energie Atomique et aux Energies Alternatives (CEA)
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <kalypsso/core/geometry/Box.h>
#include <kalypsso/core/geometry/Point.h>
#include <kalypsso/core/geometry/Sphere.h>
#include <kalypsso/core/geometry/Triangle.h>

#include <gtest/gtest.h>

namespace kalypsso
{

namespace core
{

namespace geometry
{

TEST(kalypsso_core_geometry_test, box)
{
  static_assert(
    std::is_same_v<decltype(Box{ { 1.f, 3.f, 2.f }, { 5.f, 7.f, 9.f } }), Box<3, float>>);
  static_assert(std::is_same_v<decltype(Box{ { 1., 3., 2. }, { 5., 7., 9. } }), Box<3, double>>);
  static_assert(
    std::is_same_v<decltype(Box{ Point{ 1.f, 2.f }, Point{ 3.f, 4.f } }), Box<2, float>>);

  {
    const auto a = Point{ 1.0, 2.0 };
    const auto b = Point{ 2.0, 3.0 };
    auto       box = Box{ a, b };
    const auto v = Vector{ 0.0, 0.0 };
    EXPECT_EQ(box.minCorner() - a == v, true);
    const auto c = Point{ 3.0, 4.0 };
    box.addPoint(c);
    EXPECT_EQ(box.minCorner() - a == v, true);
    EXPECT_EQ(box.maxCorner() - c == v, true);
    const auto d = Point{ 2.4, 4.0 };
    box.addPoint(d);
    EXPECT_EQ(box.maxCorner() - c == v, true);
    const auto e = Point{ 2.4, 6.0 };
    const auto e2 = Point{ 3.0, 6.0 };
    box.addPoint(e);
    EXPECT_EQ(box.maxCorner() - e2 == v, true);
  }

} // Box

TEST(kalypsso_core_geometry_test, point)
{

  // testing 1st CTAD
  {
    static_assert(std::is_same_v<decltype(Point{ 1 }), Point<1, int>>);
    static_assert(std::is_same_v<decltype(Point{ 1. }), Point<1, double>>);

    auto point2d = Point{ 2.0, 3.0 };
    EXPECT_EQ(point2d.dimension(), 2);
    EXPECT_NEAR(point2d[1], 3.0, 1e-14);

    auto point3d = Point{ 2.0, 3.0, 4.0 };
    EXPECT_EQ(point3d.dimension(), 3);
    EXPECT_NEAR(point3d[2], 4.0, 1e-14);
  }

  // testing 2nd CTAD
  {
    static_assert(std::is_same_v<decltype(Point{ { 2.3, 3.3 } }), Point<2, double>>);
    static_assert(std::is_same_v<decltype(Point{ { 1.f } }), Point<1, float>>);

    auto point2d = Point{ { 2.0, 3.0 } };
    EXPECT_EQ(point2d.dimension(), 2);
    EXPECT_NEAR(point2d[1], 3.0, 1e-14);

    auto point3d = Point{ { 2.0, 3.0, 4.0 } };
    EXPECT_EQ(point3d.dimension(), 3);
    EXPECT_NEAR(point3d[2], 4.0, 1e-14);
  }
} // Point

TEST(kalypsso_shared_geometry_test, sphere)
{
  static_assert(std::is_same_v<decltype(Sphere{ { 0.f, 2.f, 5.f }, 2.f }), Sphere<3, float>>);
  static_assert(std::is_same_v<decltype(Sphere{ Point{ 3., 4., 2. }, 6. }), Sphere<3, double>>);
} // Sphere

TEST(kalypsso_shared_geometry_test, triangle)
{
  static_assert(
    std::is_same_v<decltype(Triangle{ { 0, 2 }, { 3, 1 }, { 2, 5 } }), Triangle<2, int>>);
  static_assert(std::is_same_v<decltype(Triangle{ { 0.f, 2.f }, { 3.f, 1.f }, { 2.f, 5.f } }),
                               Triangle<2, float>>);
  static_assert(std::is_same_v<decltype(Triangle{
                                 Point{ 3., 4., 2. }, Point{ 2., 2., 2. }, Point{ 6., 3., 5. } }),
                               Triangle<3, double>>);

  {
    const auto a = Point{ { 0.0, 0.0 } };
    const auto b = Point{ { 1.0, 0.0 } };
    const auto c = Point{ { 0.0, 1.0 } };
    const auto t = Triangle{ a, b, c };
    EXPECT_NEAR(t.area(), 0.5, 1e-14);
  }

  {
    const auto a = Point{ { 1.0, 0.0, 0.0 } };
    const auto b = Point{ { 0.0, 1.0, 0.0 } };
    const auto c = Point{ { 0.0, 0.0, 1.0 } };
    const auto t = Triangle{ a, b, c };
    EXPECT_NEAR(t.area(), sqrt(3.0) / 2, 1e-14);
  }
} // Triangle

} // namespace geometry

} // namespace core

} // namespace kalypsso
