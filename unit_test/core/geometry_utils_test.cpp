// SPDX-FileCopyrightText: 2025 kalypsso-core authors
// © Commissariat a l'Energie Atomique et aux Energies Alternatives (CEA)
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <kalypsso/core/geometry_utils.h>

#include <cstdint>

#include <gtest/gtest.h>

namespace kalypsso
{

//=================================================================
//
// basic test
//
//=================================================================

TEST(kalypsso_get_tangent_to_sphere_test, test_2d)
{

  Kokkos::Array<real_t, 2> O{ 0.0, 0.0 };
  Kokkos::Array<real_t, 2> M{ 0.5, 0.5 };
  real_t                   radius = 1.0;

  Kokkos::Array<real_t, 2> normal;
  real_t                   alpha;

  get_tangent_to_sphere(O, radius, M, normal, alpha);

  EXPECT_NEAR(normal[IX], sqrt(2.0) / 2, 1e-14);
  EXPECT_NEAR(normal[IY], sqrt(2.0) / 2, 1e-14);
  EXPECT_NEAR(alpha, 1.0, 1e-14);
}

TEST(kalypsso_get_tangent_to_sphere_test, test_2d_2)
{

  Kokkos::Array<real_t, 2> O{ 0.25, -0.25 };
  Kokkos::Array<real_t, 2> M{ 0.75, 0.25 };
  real_t                   radius = 1.0;

  Kokkos::Array<real_t, 2> normal;
  real_t                   alpha;

  get_tangent_to_sphere(O, radius, M, normal, alpha);

  EXPECT_NEAR(normal[IX], sqrt(2.0) / 2, 1e-14);
  EXPECT_NEAR(normal[IY], sqrt(2.0) / 2, 1e-14);
  EXPECT_NEAR(alpha, 1.0, 1e-14);
}

TEST(kalypsso_get_tangent_to_sphere_test, test_2d_3)
{

  Kokkos::Array<real_t, 2> O{ 0.0, 0.0 };
  Kokkos::Array<real_t, 2> M{ 0.5, 0.0 };
  real_t                   radius = 1.0;

  Kokkos::Array<real_t, 2> normal;
  real_t                   alpha;

  get_tangent_to_sphere(O, radius, M, normal, alpha);

  EXPECT_NEAR(normal[IX], 1.0, 1e-14);
  EXPECT_NEAR(normal[IY], 0.0, 1e-14);
  EXPECT_NEAR(alpha, 1.0, 1e-14);
}

TEST(kalypsso_get_tangent_to_sphere_test, test_3d)
{

  Kokkos::Array<real_t, 3> O{ 0.0, 0.0, 0.0 };
  Kokkos::Array<real_t, 3> M{ 0.5, 0.5, 0.5 };
  real_t                   radius = 1.0;

  Kokkos::Array<real_t, 3> normal;
  real_t                   alpha;

  get_tangent_to_sphere(O, radius, M, normal, alpha);

  EXPECT_NEAR(normal[IX], sqrt(3.0) / 3, 1e-14);
  EXPECT_NEAR(normal[IY], sqrt(3.0) / 3, 1e-14);
  EXPECT_NEAR(normal[IZ], sqrt(3.0) / 3, 1e-14);
  EXPECT_NEAR(alpha, 1.0, 1e-14);
}

} // namespace kalypsso
