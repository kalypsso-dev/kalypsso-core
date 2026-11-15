// SPDX-FileCopyrightText: 2026 kalypsso-core authors
// © Commissariat a l'Energie Atomique et aux Energies Alternatives (CEA)
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <kalypsso/core/Box.h>

#include <cstdint>

#include <gtest/gtest.h>

#include "kalypsso_unittest_utils.h"

namespace kalypsso
{

namespace core
{

TEST(kalypsso_core_box, test_constructor)
{

  auto origin = coord_t<2, real_t>{ -1.0, -1.0 };
  auto shape = coord_t<2, real_t>{ 8.0, 8.0 };

  auto box = Box(origin, shape);

  auto shift = coord_t<2, real_t>{ 5.0, -3.0 };

  // translate and scale
  auto box2 = box;
  box2.translate(shift);

  EXPECT_REAL_EQ(box2.origin()[IX], 4.0);
  EXPECT_REAL_EQ(box2.origin()[IY], -4.0);

  box2.scale(2.0);

  EXPECT_REAL_EQ(box2.origin()[IX], 4.0);
  EXPECT_REAL_EQ(box2.origin()[IY], -4.0);
  EXPECT_REAL_EQ(box2.shape()[IX], 16.0);
  EXPECT_REAL_EQ(box2.shape()[IY], 16.0);

  // scale and translate
  auto box3 = box;
  box3.scale(2.0);

  EXPECT_REAL_EQ(box3.origin()[IX], -1.0);
  EXPECT_REAL_EQ(box3.origin()[IY], -1.0);
  EXPECT_REAL_EQ(box2.shape()[IX], 16.0);
  EXPECT_REAL_EQ(box2.shape()[IY], 16.0);

  box3.translate(shift);
  EXPECT_REAL_EQ(box3.origin()[IX], 4.0);
  EXPECT_REAL_EQ(box3.origin()[IY], -4.0);
}

TEST(kalypsso_core_box, test_homthety)
{

  {
    auto origin = coord_t<2, real_t>{ -1.0, -1.0 };
    auto shape = coord_t<2, real_t>{ 8.0, 8.0 };

    auto box = Box(origin, shape);

    box.homothety(2.0);
    EXPECT_REAL_EQ(box.origin()[IX], -2.0);
    EXPECT_REAL_EQ(box.origin()[IY], -2.0);
    EXPECT_REAL_EQ(box.shape()[IX], 16.0);
    EXPECT_REAL_EQ(box.shape()[IY], 16.0);
  }

  {
    auto origin = coord_t<2, real_t>{ 0.0, -2.0 };
    auto shape = coord_t<2, real_t>{ 5.0, 3.0 };

    auto box = Box(origin, shape);

    box.homothety(2.0);
    EXPECT_REAL_EQ(box.origin()[IX], 0.0);
    EXPECT_REAL_EQ(box.origin()[IY], -4.0);
    EXPECT_REAL_EQ(box.shape()[IX], 10.0);
    EXPECT_REAL_EQ(box.shape()[IY], 6.0);
  }
}

TEST(kalypsso_core_box, test_intersection)
{

  auto origin0 = coord_t<2, real_t>{ -1.0, -1.0 };
  auto shape0 = coord_t<2, real_t>{ 8.0, 8.0 };

  auto b0 = Box(origin0, shape0);

  {
    auto origin1 = coord_t<2, real_t>{ -3.0, -1.0 };
    auto shape1 = coord_t<2, real_t>{ 6.0, 8.0 };

    auto b1 = Box(origin1, shape1);

    auto b2 = b0.intersect(b1);

    EXPECT_EQ(b2.area() > 0, true);
    EXPECT_REAL_EQ(b2.area(), 32.0);
  }

  {
    auto origin1 = coord_t<2, real_t>{ -3.0, -1.0 };
    auto shape1 = coord_t<2, real_t>{ 1.0, 1.0 };

    auto b1 = Box(origin1, shape1);

    auto b2 = b0.intersect(b1);

    EXPECT_EQ(b2.area() > 0, false);
  }

  {
    auto origin1 = coord_t<2, real_t>{ 0.0, 0.0 };
    auto shape1 = coord_t<2, real_t>{ 4.0, 5.0 };

    auto b1 = Box(origin1, shape1);

    auto b2 = b0.intersect(b1);

    EXPECT_EQ(b2.area() > 0, true);
    EXPECT_REAL_EQ(b2.area(), 20.0);
  }

  {
    auto origin1 = coord_t<2, real_t>{ -5.0, -5.0 };
    auto shape1 = coord_t<2, real_t>{ 20.0, 20.0 };

    auto b1 = Box(origin1, shape1);

    auto b2 = b0.intersect(b1);

    EXPECT_EQ(b2.area() > 0, true);
    EXPECT_REAL_EQ(b2.area(), 64.0);
  }

  {
    auto origin1 = coord_t<2, real_t>{ 7.0, 0.0 };
    auto shape1 = coord_t<2, real_t>{ 20.0, 20.0 };

    auto b1 = Box(origin1, shape1);

    auto b2 = b0.intersect(b1);

    EXPECT_EQ(b2.area() > 0, false);
    EXPECT_REAL_EQ(b2.area(), 0.0);
  }

  {
    auto origin1 = coord_t<2, real_t>{ 5.0, 1.0 };
    auto shape1 = coord_t<2, real_t>{ 3.0, 2.0 };

    auto b1 = Box(origin1, shape1);

    auto b2 = b0.intersect(b1);

    EXPECT_EQ(b2.area() > 0, true);
    EXPECT_REAL_EQ(b2.area(), 4.0);
  }
}

} // namespace core

} // namespace kalypsso
